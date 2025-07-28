#!/bin/bash
#SBATCH --job-name=social_tipping_test
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3  # 3 seeds in parallel
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --output=log/social_tipping_%A_%a_%t.out
#SBATCH --error=log/social_tipping_%A_%a_%t.err
#SBATCH --array=1-10

# Print job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_PROCID: $SLURM_PROCID"

# Create log directory if it doesn't exist
mkdir -p log

# Load modules and activate environment
module load anaconda
source activate mdp-world-model

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Set PyTorch threads
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

# Get dataset for this array task
DATASET=${DATASETS[$((SLURM_ARRAY_TASK_ID-1))]}
DB_PATH="datasets/results/$DATASET"

# Get seed for this parallel task (1, 2, 3)
SEED=$((SLURM_PROCID + 1))

# Fixed parameters
SYSTEM_TYPE="social_tipping"
OUTPUT_DIR="neural_networks/output/social_tipping_test"
EPOCHS=120
LR=1e-5
BATCH_SIZE=64
HIDDEN_DIM=32
STATE_LOSS_WEIGHT=1.0
VALUE_LOSS_WEIGHT=1.5
PREDICTOR_TYPE="standard"
VAL_SIZE=1000
TEST_SIZE=1000
ENTROPY_DECAY_PROPORTION=0.5
ENCODER_INIT_METHOD="he"

echo "========================================================"
echo "Social Tipping Test - Array $SLURM_ARRAY_TASK_ID, Task $SLURM_PROCID"
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "Will run 2 num_states values (4, 8) sequentially"
echo "========================================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start GPU monitoring in background
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv --loop=30 > "log/gpu_usage_${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}_${SLURM_PROCID}.log" &
GPU_MONITOR_PID=$!

# Sequential loop over num_states values
for NUM_STATES in 4 8; do
    echo "------------------------------------------------------------"
    echo "Array $SLURM_ARRAY_TASK_ID, Task $SLURM_PROCID - Running num_states=$NUM_STATES"
    echo "Dataset: $DATASET"
    echo "Seed: $SEED"
    echo "Started at: $(date)"
    
    # Create descriptive run name
    DATASET_NAME=$(basename "$DATASET" .db)
    RUN_ID="${DATASET_NAME}_states${NUM_STATES}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
    
    # Construct and run the command
    CMD="python neural_networks/train_drm.py $DB_PATH \
        --system-type=$SYSTEM_TYPE \
        --epochs=$EPOCHS \
        --lr=$LR \
        --batch_size=$BATCH_SIZE \
        --num_states=$NUM_STATES \
        --hidden_dim=$HIDDEN_DIM \
        --state_loss_weight=$STATE_LOSS_WEIGHT \
        --value_loss_weight=$VALUE_LOSS_WEIGHT \
        --predictor_type=$PREDICTOR_TYPE \
        --val_size=$VAL_SIZE \
        --test_size=$TEST_SIZE \
        --entropy_decay_proportion=$ENTROPY_DECAY_PROPORTION \
        --use_warmup \
        --warmup_epochs=10 \
        --use_target_encoder \
        --ema_decay=0.9 \
        --use_entropy_reg \
        --state_loss_type=kl_div \
        --use_gumbel \
        --min_temp=0.5 \
        --initial_temp=3.0 \
        --seed=$SEED \
        --encoder_init_method=$ENCODER_INIT_METHOD \
        --output_dir=$OUTPUT_DIR \
        --run_id=$RUN_ID"
    
    echo "Running command: $CMD"
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed num_states=$NUM_STATES at $(date)"
    else
        echo "✗ Failed num_states=$NUM_STATES at $(date)"
    fi
    
    # Small delay between num_states
    sleep 5
done

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null

echo "Array $SLURM_ARRAY_TASK_ID, Task $SLURM_PROCID completed all configurations at $(date)"