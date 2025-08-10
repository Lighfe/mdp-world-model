#!/bin/bash
#SBATCH --job-name=lr_grid
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=log/lr_grid_%A_%a.out
#SBATCH --error=log/lr_grid_%A_%a.err
#SBATCH --array=1-8  # 8 array jobs (each handles 1 config × 16 seeds = 16 parallel)

echo "========================================================"
echo "LR GRID SEARCH - Array Job $SLURM_ARRAY_TASK_ID/8"
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS (16 parallel)"
echo "GPU Info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "========================================================"

# Create log directory
mkdir -p log

# Load modules and activate environment
module load anaconda
source activate mdp-world-model

# Set standard environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# ============= H100-SPECIFIC OPTIMIZATIONS =============
# Enable asynchronous GPU operations
export CUDA_LAUNCH_BLOCKING=0

# Start NVIDIA MPS for better GPU sharing...
echo "Starting NVIDIA MPS for better GPU sharing..."
mkdir -p /tmp/nvidia-mps
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps
nvidia-cuda-mps-control -d

echo "  Total parallel tasks: 16 (1 config × 16 seeds)"

# Create output directory
OUTPUT_DIR="neural_networks/output/lr_grid_search"
mkdir -p "$OUTPUT_DIR"

# Start GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 5 > "log/gpu_usage_lr_${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}.log" &
GPU_MONITOR_PID=$!

# Create training script for this array job
TRAINING_SCRIPT="lr_grid_${SLURM_ARRAY_TASK_ID}_${SLURM_JOBID}.sh"
cat > "$TRAINING_SCRIPT" << 'EOF'
#!/bin/bash

# Get task ID (0-15) and determine config and seed
TASK_ID=$SLURM_PROCID

# CONFIG mapping: array job index (1-8) selects the CONFIG_ID
CONFIG_ID=$SLURM_ARRAY_TASK_ID
# TASK_ID is 0-(SLURM_NTASKS-1); map task id to a seed value
SEED=$((TASK_ID + 11))  # Seeds 11..26 for 16 tasks

export PYTHONUNBUFFERED=1

echo "[Task $TASK_ID] Starting Config $CONFIG_ID, Seed $SEED at $(date)"

# Define configuration parameters based on CONFIG_ID
case $CONFIG_ID in
    1)  # adam fixed lr=1e-5
        OPTIMIZER="adam"
        SCHEDULER="fixed"
        LR="1e-5"
        MIN_LR=""
        WEIGHT_DECAY=""
        HIDDEN_DIM="32"
        CONFIG_NAME="adam_fixed_1e-5"
        ;;
    2)  # adam fixed lr=5e-5
        OPTIMIZER="adam"
        SCHEDULER="fixed"
        LR="5e-5"
        MIN_LR=""
        WEIGHT_DECAY=""
        HIDDEN_DIM="32"
        CONFIG_NAME="adam_fixed_5e-5"
        ;;
    3)  # adam cosine lr=1e-5->1e-6
        OPTIMIZER="adam"
        SCHEDULER="cosine"
        LR="1e-5"
        MIN_LR="1e-6"
        WEIGHT_DECAY=""
        HIDDEN_DIM="32"
        CONFIG_NAME="adam_cosine_1e-5"
        ;;
    4)  # adam cosine lr=5e-5->5e-6
        OPTIMIZER="adam"
        SCHEDULER="cosine"
        LR="5e-5"
        MIN_LR="5e-6"
        WEIGHT_DECAY=""
        HIDDEN_DIM="32"
        CONFIG_NAME="adam_cosine_5e-5"
        ;;
    5)  # adamw lr=1e-5
        OPTIMIZER="adamw"
        SCHEDULER="fixed"
        LR="1e-5"
        MIN_LR=""
        WEIGHT_DECAY="0.01"
        HIDDEN_DIM="32"
        CONFIG_NAME="adamw_1e-5"
        ;;
    6)  # adamw lr=5e-5
        OPTIMIZER="adamw"
        SCHEDULER="fixed"
        LR="5e-5"
        MIN_LR=""
        WEIGHT_DECAY="0.01"
        HIDDEN_DIM="32"
        CONFIG_NAME="adamw_5e-5"
        ;;
    7)  # adam hidden_dim=16 lr=1e-5
        OPTIMIZER="adam"
        SCHEDULER="fixed"
        LR="1e-5"
        MIN_LR=""
        WEIGHT_DECAY=""
        HIDDEN_DIM="16"
        CONFIG_NAME="adam_hd16_1e-5"
        ;;
    8)  # adam hidden_dim=16 lr=5e-5
        OPTIMIZER="adam"
        SCHEDULER="fixed"
        LR="5e-5"
        MIN_LR=""
        WEIGHT_DECAY=""
        HIDDEN_DIM="16"
        CONFIG_NAME="adam_hd16_5e-5"
        ;;
    *)
        echo "[Task $TASK_ID] ERROR: Unknown CONFIG_ID: $CONFIG_ID"
        exit 2
        ;;
esac

# For each dataset (run sequentially)
for DATASET_NUM in {1..10}; do
    DB_PATH="datasets/results/multi_saddle_${DATASET_NUM}.db"
    if [ ! -f "$DB_PATH" ]; then
        echo "[Task $TASK_ID] ERROR: Dataset not found: $DB_PATH"
        continue
    fi

    echo "[Task $TASK_ID] Starting dataset $DATASET_NUM at $(date)"

    # Build the command
    CMD="python -u neural_networks/train_drm.py $DB_PATH \
        --system_type=saddle_system \
        --batch_size=128 \
        --num_states=4 \
        --hidden_dim=$HIDDEN_DIM \
        --state_loss_weight=1.0 \
        --value_loss_weight=0.3 \
        --optimizer=$OPTIMIZER \
        --scheduler=$SCHEDULER \
        --lr=$LR \
        --seed=$SEED \
        --output_dir=$OUTPUT_DIR \
        --run_id=grid_ds${DATASET_NUM}_${CONFIG_NAME}_seed${SEED}"

    # Add optional parameters
    if [ ! -z "$MIN_LR" ]; then
        CMD="$CMD --min_lr=$MIN_LR"
    fi

    if [ ! -z "$WEIGHT_DECAY" ]; then
        CMD="$CMD --weight_decay=$WEIGHT_DECAY"
    fi

    # Execute the command
    $CMD 2>&1

done

EOF

chmod +x "$TRAINING_SCRIPT"

# Export variables for the training script
export OUTPUT_DIR

echo "========================================================"
echo "Launching 16 parallel processes (1 config × 16 seeds)..."
echo "Each will run 10 datasets sequentially"
echo "========================================================"

# Launch parallel training
srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=$SLURM_NTASKS \
     --output="log/lr_grid_task_%A_%a_%t.out" \
     --error="log/lr_grid_task_%A_%a_%t.err" \
     --unbuffered \
     ./"$TRAINING_SCRIPT"

SRUN_EXIT_CODE=$?

# Clean up
rm -f "$TRAINING_SCRIPT"

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null

# Stop MPS daemon
echo "Stopping NVIDIA MPS..."
echo quit | nvidia-cuda-mps-control

echo "========================================================"
echo "Array Job $SLURM_ARRAY_TASK_ID completed at $(date)"
echo "Configuration processed (config id): $SLURM_ARRAY_TASK_ID"
echo "srun exit code: $SRUN_EXIT_CODE"
echo "Results saved to: $OUTPUT_DIR"
echo "GPU usage log: log/gpu_usage_lr_${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}.log"
echo "Task logs: log/lr_grid_task_${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}_*.out"
echo "========================================================"

# Quick summary - try to match actual output filenames
echo ""
echo "Quick Summary for Array Job $SLURM_ARRAY_TASK_ID:"
echo "------------------------------------------------"

# Count completed runs
total_expected=$((1 * 16 * 10))  # 1 config × 16 seeds × 10 datasets
completed=0

for config_id in $SLURM_ARRAY_TASK_ID; do
    case $config_id in
        1) config_name="adam_fixed_1e-5" ;;
        2) config_name="adam_fixed_5e-5" ;;
        3) config_name="adam_cosine_1e-5" ;;
        4) config_name="adam_cosine_5e-5" ;;
        5) config_name="adamw_1e-5" ;;
        6) config_name="adamw_5e-5" ;;
        7) config_name="adam_hd16_1e-5" ;;
        8) config_name="adam_hd16_5e-5" ;;
    esac

    echo "Config $config_id ($config_name):"
    for seed in {11..26}; do
        seed_count=0
        for dataset in {1..10}; do
            # Try multiple possible filename patterns
            if [ -f "${OUTPUT_DIR}/drm_final_grid_ds${dataset}_${config_name}_seed${seed}.pt" ] || \
               [ -f "${OUTPUT_DIR}/grid_ds${dataset}_${config_name}_seed${seed}_final.pt" ] || \
               [ -f "${OUTPUT_DIR}/drm_grid_ds${dataset}_${config_name}_seed${seed}.pt" ]; then
                seed_count=$((seed_count + 1))
                completed=$((completed + 1))
            fi
        done
        echo "  Seed $seed: $seed_count/10 datasets"
    done
done

echo ""
echo "Total completed: $completed/$total_expected experiments"

exit $SRUN_EXIT_CODE
