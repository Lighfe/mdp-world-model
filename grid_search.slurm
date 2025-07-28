#!/bin/bash
#SBATCH --job-name=social_tipping_test
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=10  # Total tasks we want
#SBATCH --ntasks-per-node=10  # All on same node
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=log/social_tipping_%A_%a.out
#SBATCH --error=log/social_tipping_%A_%a.err
#SBATCH --array=1-8  # 4 seeds × 2 num_states = 8 arrays

# Print job info (PIK style)
echo "========================================================"
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "========================================================"

# Create log directory
mkdir -p log

# Load modules and activate environment
module load anaconda
source activate mdp-world-model

# Check GPU
echo "GPU Info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Set environment (PIK recommended)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# Dataset mapping
DATASETS=(
    "social_tipping_default_t7.db"
    "social_tipping_default_subsidy_t7.db"
    "social_tipping_default_t58.db"
    "social_tipping_default_tmedium.db"
    "social_tipping_default_t21.db"
    "social_tipping_default_tsmallhigh.db"
    "social_tipping_default_t2.db"
    "social_tipping_default_subsidy_tall.db"
    "social_tipping_default_subsidy_t58.db"
    "social_tipping_default_tall.db"
)

# Calculate parameters from array ID
SEED=$(( ((SLURM_ARRAY_TASK_ID - 1) / 2) + 1 ))
if [ $((SLURM_ARRAY_TASK_ID % 2)) -eq 1 ]; then
    NUM_STATES=4
else
    NUM_STATES=8
fi

echo "Array $SLURM_ARRAY_TASK_ID: Seed=$SEED, NumStates=$NUM_STATES"
echo "Will use srun to launch ${#DATASETS[@]} parallel training processes"

# Create output directory
OUTPUT_DIR="neural_networks/output/social_tipping_test"
mkdir -p "$OUTPUT_DIR"

# Start GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv --loop=30 > "log/gpu_usage_${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}.log" &
GPU_MONITOR_PID=$!

# Create a script that will be executed by each srun process
TRAINING_SCRIPT="train_single_${SLURM_ARRAY_TASK_ID}.sh"
cat > "$TRAINING_SCRIPT" << 'EOF'
#!/bin/bash

# Get task ID from srun (0-9)
TASK_ID=$SLURM_PROCID

# Dataset mapping (recreate array in script)
DATASETS=(
    "social_tipping_default_t7.db"
    "social_tipping_default_subsidy_t7.db"
    "social_tipping_default_t58.db"
    "social_tipping_default_tmedium.db"
    "social_tipping_default_t21.db"
    "social_tipping_default_tsmallhigh.db"
    "social_tipping_default_t2.db"
    "social_tipping_default_subsidy_tall.db"
    "social_tipping_default_subsidy_t58.db"
    "social_tipping_default_tall.db"
)

# Get dataset for this task
DATASET="${DATASETS[$TASK_ID]}"
DB_PATH="datasets/results/$DATASET"

# Parameters passed as environment variables
DATASET_NAME=$(basename "$DATASET" .db)
RUN_ID="${DATASET_NAME}_states${NUM_STATES}_seed${SEED}_task${TASK_ID}"

echo "[Task $TASK_ID] Starting training: $DATASET ($(date))"

# Run training
python neural_networks/train_drm.py "$DB_PATH" \
    --system-type=social_tipping \
    --epochs=120 \
    --lr=1e-5 \
    --batch_size=64 \
    --num_states="$NUM_STATES" \
    --hidden_dim=32 \
    --state_loss_weight=1.0 \
    --value_loss_weight=1.5 \
    --predictor_type=standard \
    --val_size=1000 \
    --test_size=1000 \
    --entropy_decay_proportion=0.5 \
    --use_warmup \
    --warmup_epochs=10 \
    --use_target_encoder \
    --ema_decay=0.9 \
    --use_entropy_reg \
    --state_loss_type=kl_div \
    --use_gumbel \
    --min_temp=0.5 \
    --initial_temp=3.0 \
    --seed="$SEED" \
    --encoder_init_method=he \
    --output_dir="$OUTPUT_DIR" \
    --run_id="$RUN_ID"

EXIT_CODE=$?
echo "[Task $TASK_ID] Finished training: $DATASET (exit: $EXIT_CODE, $(date))"
exit $EXIT_CODE
EOF

chmod +x "$TRAINING_SCRIPT"

# Export variables that the training script needs
export NUM_STATES
export SEED
export OUTPUT_DIR

echo "Launching srun with $SLURM_NTASKS parallel processes at $(date)..."

# Use srun to launch parallel processes (PIK style)
srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=$SLURM_NTASKS \
     --cpus-per-task=1 \
     --output="log/srun_%A_%a_%t.out" \
     --error="log/srun_%A_%a_%t.err" \
     ./"$TRAINING_SCRIPT"

SRUN_EXIT_CODE=$?

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null

# Clean up
rm -f "$TRAINING_SCRIPT"

echo "========================================================"
echo "Array $SLURM_ARRAY_TASK_ID completed at $(date)"
echo "srun exit code: $SRUN_EXIT_CODE"
echo "Check individual task logs: log/srun_${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}_*.out"
echo "========================================================"

exit $SRUN_EXIT_CODE