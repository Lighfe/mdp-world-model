#!/bin/bash
#SBATCH --job-name=scheduler_grid
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=log/scheduler_grid_%A_%a.out
#SBATCH --error=log/scheduler_grid_%A_%a.err
#SBATCH --array=1-80  # 10 datasets × 4 schedulers × 2 optimizers

echo "========================================================"
echo "SCHEDULER GRID SEARCH - Array $SLURM_ARRAY_TASK_ID"
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
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

# Calculate dataset and configuration from array ID
# 80 arrays total = 10 datasets × 8 configs per dataset
DATASET_NUM=$(( ((SLURM_ARRAY_TASK_ID - 1) / 8) + 1 ))
CONFIG_ID=$(( ((SLURM_ARRAY_TASK_ID - 1) % 8) + 1 ))

# Define scheduler configurations
case $CONFIG_ID in
    1) SCHEDULER="fixed"; OPTIMIZER="adam" ;;
    2) SCHEDULER="fixed"; OPTIMIZER="adamw" ;;
    3) SCHEDULER="cosine_conservative"; OPTIMIZER="adam" ;;
    4) SCHEDULER="cosine_conservative"; OPTIMIZER="adamw" ;;
    5) SCHEDULER="cosine_aggressive"; OPTIMIZER="adam" ;;
    6) SCHEDULER="cosine_aggressive"; OPTIMIZER="adamw" ;;
    7) SCHEDULER="warm_restarts"; OPTIMIZER="adam" ;;
    8) SCHEDULER="warm_restarts"; OPTIMIZER="adamw" ;;
esac

# Dataset path
DB_PATH="datasets/results/multi_saddle_${DATASET_NUM}.db"

echo "Configuration for Array $SLURM_ARRAY_TASK_ID:"
echo "  Dataset: multi_saddle_${DATASET_NUM}.db"
echo "  Scheduler: $SCHEDULER"
echo "  Optimizer: $OPTIMIZER"
echo "  Seeds: 11-20 (10 seeds in parallel)"
echo "  Epochs: 100 then 200 (sequential)"

# Verify dataset exists
if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Dataset not found: $DB_PATH"
    exit 1
fi

# Create output directory
OUTPUT_DIR="neural_networks/output/scheduler_grid"
mkdir -p "$OUTPUT_DIR"

# Start GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv --loop=30 > "log/gpu_usage_${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}.log" &
GPU_MONITOR_PID=$!

# Create training script for parallel execution
TRAINING_SCRIPT="grid_train_${SLURM_ARRAY_TASK_ID}.sh"
cat > "$TRAINING_SCRIPT" << 'EOF'
#!/bin/bash

# Get task ID (0-9) and calculate seed (11-20)
TASK_ID=$SLURM_PROCID
SEED=$((TASK_ID + 11))

export PYTHONUNBUFFERED=1

echo "[Task $TASK_ID] Starting Array $SLURM_ARRAY_TASK_ID with seed $SEED at $(date)"

# Base command arguments - all fixed settings
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

# Configure scheduler and optimizer based on configuration
case $SCHEDULER in
    "fixed")
        SCHEDULER_ARGS="--scheduler_type=fixed --lr=1e-5"
        ;;
    "cosine_conservative")
        SCHEDULER_ARGS="--scheduler_type=cosine --lr=5e-5 --min_lr=5e-6"
        ;;
    "cosine_aggressive")
        SCHEDULER_ARGS="--scheduler_type=cosine --lr=5e-5 --min_lr=1e-6"
        ;;
    "warm_restarts")
        SCHEDULER_ARGS="--scheduler_type=warm_restarts --lr=5e-5 --min_lr=5e-6"
        ;;
esac

# Configure optimizer
case $OPTIMIZER in
    "adam")
        OPTIMIZER_ARGS="--optimizer=adam"
        ;;
    "adamw")
        OPTIMIZER_ARGS="--optimizer=adamw"
        ;;
esac

echo "[Task $TASK_ID] Config: $SCHEDULER + $OPTIMIZER"

# Run epochs=100
RUN_ID="ms${DATASET_NUM}_${SCHEDULER}_${OPTIMIZER}_epochs100_seed${SEED}"
echo "[Task $TASK_ID] Starting epochs=100 at $(date)"
$BASE_CMD $SCHEDULER_ARGS $OPTIMIZER_ARGS --epochs=100 --run_id="$RUN_ID" 2>&1

if [ $? -eq 0 ]; then
    echo "[Task $TASK_ID] ✓ Completed epochs=100 at $(date)"
else
    echo "[Task $TASK_ID] ✗ Failed epochs=100 at $(date)"
    exit 1
fi

# Brief pause between epoch configurations
sleep 10

# Run epochs=200
RUN_ID="ms${DATASET_NUM}_${SCHEDULER}_${OPTIMIZER}_epochs200_seed${SEED}"
echo "[Task $TASK_ID] Starting epochs=200 at $(date)"
$BASE_CMD $SCHEDULER_ARGS $OPTIMIZER_ARGS --epochs=200 --run_id="$RUN_ID" 2>&1

if [ $? -eq 0 ]; then
    echo "[Task $TASK_ID] ✓ Completed epochs=200 at $(date)"
else
    echo "[Task $TASK_ID] ✗ Failed epochs=200 at $(date)"
    exit 1
fi

echo "[Task $TASK_ID] Finished all training for seed $SEED at $(date)"
echo "[Task $TASK_ID] Final GPU memory: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
EOF

chmod +x "$TRAINING_SCRIPT"

# Export variables for the training script
export DB_PATH
export OUTPUT_DIR
export DATASET_NUM
export SCHEDULER
export OPTIMIZER

echo "Launching Array $SLURM_ARRAY_TASK_ID with $SLURM_NTASKS parallel processes at $(date)..."

# Launch parallel training
srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=$SLURM_NTASKS \
     --output="log/grid_task_%A_%a_%t.out" \
     --error="log/grid_task_%A_%a_%t.err" \
     --unbuffered \
     ./"$TRAINING_SCRIPT"

SRUN_EXIT_CODE=$?

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null

# Clean up
rm -f "$TRAINING_SCRIPT"

echo "========================================================"
echo "Array $SLURM_ARRAY_TASK_ID completed at $(date)"
echo "Dataset: multi_saddle_${DATASET_NUM}.db"
echo "Config: $SCHEDULER + $OPTIMIZER"  
echo "srun exit code: $SRUN_EXIT_CODE"
echo "Individual task logs: log/grid_task_${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}_*.out"
echo "GPU usage log: log/gpu_usage_${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}.log"
echo "========================================================"

exit $SRUN_EXIT_CODE