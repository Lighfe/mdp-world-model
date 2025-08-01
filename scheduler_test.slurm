#!/bin/bash
#SBATCH --job-name=scheduler_par
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=log/scheduler_par_%A.out
#SBATCH --error=log/scheduler_par_%A.err


echo "========================================================"
echo "SCHEDULER TEST - Single Array Job"
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "GPU Info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "========================================================"

# Create log directory
mkdir -p log

# Load modules and activate environment
module load anaconda
source activate mdp-world-model

# Set environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# Test configuration
SCHEDULER="cosine_aggressive"
OPTIMIZER="adam"


echo "Test Configuration:"
echo "  Dataset: multi_saddle_${DATASET_NUM}.db"
echo "  Scheduler: $SCHEDULER"
echo "  Optimizer: $OPTIMIZER"
echo "  Seeds: 11-20 (10 seeds in parallel)"
echo "  Epochs: 100 only (for testing)"

# Create output directory
OUTPUT_DIR="neural_networks/output/scheduler_par"
mkdir -p "$OUTPUT_DIR"

# Start GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv --loop=30 > "log/gpu_usage_test_${SLURM_JOBID}.log" &
GPU_MONITOR_PID=$!

# Create training script for each parallel task
TRAINING_SCRIPT="test_scheduler_${SLURM_JOBID}.sh"
cat > "$TRAINING_SCRIPT" << 'EOF'
#!/bin/bash

# Get task ID (0-9) and calculate seed (11-20)
TASK_ID=$SLURM_PROCID
SEED=$((TASK_ID + 11))

DATASET_NUM=$((TASK_ID + 1))  # ← Move it here

# Set dataset path
DB_PATH="datasets/results/multi_saddle_${DATASET_NUM}.db"

export PYTHONUNBUFFERED=1

echo "[Task $TASK_ID] Starting test with dataset $DATASET_NUM, seed $SEED at $(date)"

# Base command arguments
BASE_CMD="python -u neural_networks/train_drm.py $DB_PATH \
    --system_type=saddle_system \
    --batch_size=64 \
    --num_states=4 \
    --hidden_dim=32 \
    --state_loss_weight=1.0 \
    --value_loss_weight=1.5 \
    --predictor_type=standard \
    --val_size=1000 \
    --test_size=1000 \
    --probing_size=2000 \
    --entropy_decay_proportion=0.5 \
    --use_warmup \
    --warmup_epochs=10 \
    --use_target_encoder \
    --ema_decay=0.9 \
    --use_entropy_reg \
    --entropy_weight=1.0 \
    --state_loss_type=kl_div \
    --use_gumbel \
    --min_temp=0.5 \
    --initial_temp=3.0 \
    --encoder_init_method=he \
    --seed=$SEED \
    --output_dir=$OUTPUT_DIR"

# Scheduler and optimizer configuration (cosine aggressive + adam)
SCHEDULER_ARGS="--scheduler_type=cosine --lr=5e-5 --min_lr=1e-6"
OPTIMIZER_ARGS="--optimizer=adam"

# Run epoch 100 only (test)
RUN_ID="test_ms${DATASET_NUM}_${SCHEDULER}_${OPTIMIZER}_epochs100_seed${SEED}"
echo "[Task $TASK_ID] Starting epochs=100 for seed $SEED"
$BASE_CMD $SCHEDULER_ARGS $OPTIMIZER_ARGS --epochs=100 --run_id="$RUN_ID" 2>&1

if [ $? -eq 0 ]; then
    echo "[Task $TASK_ID] ✓ Completed epochs=100 for seed $SEED"
else
    echo "[Task $TASK_ID] ✗ Failed epochs=100 for seed $SEED"
fi

echo "[Task $TASK_ID] Finished test training for seed $SEED at $(date)"
EOF

chmod +x "$TRAINING_SCRIPT"

# Export variables for the training script
export DB_PATH
export OUTPUT_DIR
export DATASET_NUM
export SCHEDULER
export OPTIMIZER

echo "Launching test with $SLURM_NTASKS parallel processes at $(date)..."

# Launch parallel test
srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=$SLURM_NTASKS \
     --output="log/test_task_%A_%t.out" \
     --error="log/test_task_%A_%t.err" \
     --unbuffered \
     ./"$TRAINING_SCRIPT"

SRUN_EXIT_CODE=$?

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null

# Clean up
rm -f "$TRAINING_SCRIPT"

echo "========================================================"
echo "SCHEDULER TEST completed at $(date)"
echo "srun exit code: $SRUN_EXIT_CODE"
echo "Check individual task logs: log/test_task_${SLURM_JOBID}_*.out"
echo "GPU usage log: log/gpu_usage_test_${SLURM_JOBID}.log"
echo "========================================================"

exit $SRUN_EXIT_CODE