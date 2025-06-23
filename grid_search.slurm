#!/bin/bash
#SBATCH --job-name=multi_saddle_grid
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --output=log/multi_saddle_grid_%A_%a.out
#SBATCH --error=log/multi_saddle_grid_%A_%a.err
#SBATCH --array=1-40

# Print job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

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

# Fixed parameters for all runs
SYSTEM_TYPE="saddle_system"
BASE_OUTPUT_DIR="neural_networks/output/multi_saddle_datasets"
EPOCHS=100
VAL_SIZE=2000
TEST_SIZE=2000
BATCH_SIZE=64
PATIENCE=100
HIDDEN_DIM=64
LR=1e-5
USE_WARMUP="--use_warmup --warmup_epochs=10"
USE_TARGET_ENCODER="--use_target_encoder --ema_decay=0.9"
USE_ENTROPY_REG="--use_entropy_reg --entropy_weight=1.0"
STATE_LOSS_TYPE="kl_div"
PREDICTOR_TYPE="standard"

# FIXED PARAMETERS
GUMBEL_TEMP=3.0
GUMBEL_MIN=0.5
SORT_STATES="--sort_states"

# Fixed loss weights (no VICReg)
STATE_LOSS_WEIGHT=1.0
VALUE_LOSS_WEIGHT=1.5

# Seeds to run for each dataset
SEEDS=(7 11 42)

# Value methods to test
VALUE_METHODS=("angle" "identity")

# Number of states to test
NUM_STATES_OPTIONS=(4 6)

# Calculate configuration for this array task
# Array tasks 1-10: dataset 1-10 with "angle" and num_states=4
# Array tasks 11-20: dataset 1-10 with "identity" and num_states=4
# Array tasks 21-30: dataset 1-10 with "angle" and num_states=6
# Array tasks 31-40: dataset 1-10 with "identity" and num_states=6

if [ $SLURM_ARRAY_TASK_ID -le 10 ]; then
    DATASET_NUM=$SLURM_ARRAY_TASK_ID
    VALUE_METHOD="angle"
    NUM_STATES=4
elif [ $SLURM_ARRAY_TASK_ID -le 20 ]; then
    DATASET_NUM=$((SLURM_ARRAY_TASK_ID - 10))
    VALUE_METHOD="identity"
    NUM_STATES=4
elif [ $SLURM_ARRAY_TASK_ID -le 30 ]; then
    DATASET_NUM=$((SLURM_ARRAY_TASK_ID - 20))
    VALUE_METHOD="angle"
    NUM_STATES=6
else
    DATASET_NUM=$((SLURM_ARRAY_TASK_ID - 30))
    VALUE_METHOD="identity"
    NUM_STATES=6
fi

# Get database path for this dataset
DB_PATH="datasets/results/multi_saddle_${DATASET_NUM}.db"

echo "========================================================"
echo "Array task $SLURM_ARRAY_TASK_ID:"
echo "  Dataset: $DATASET_NUM"
echo "  Value method: $VALUE_METHOD"
echo "  Num states: $NUM_STATES"
echo "  Database: $DB_PATH"
echo "Will run ${#SEEDS[@]} seeds sequentially: ${SEEDS[*]}"
echo "========================================================"

# Run all seeds for this dataset and value method combination
for seed in "${SEEDS[@]}"; do
    
    # Create descriptive run name
    CONFIG_NAME="multi_saddle_${DATASET_NUM}_${VALUE_METHOD}_${PREDICTOR_TYPE}_states${NUM_STATES}_sl${STATE_LOSS_WEIGHT}_vl${VALUE_LOSS_WEIGHT}_novicreg_seed${seed}"
    
    # Create output directory
    OUTPUT_DIR="${BASE_OUTPUT_DIR}"
    mkdir -p "$OUTPUT_DIR"
    
    # Create unique run ID
    RUN_ID="${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"
    
    echo "--------------------------------------------------------"
    echo "Starting run with seed $seed:"
    echo "  database: $DB_PATH"
    echo "  system: $SYSTEM_TYPE"
    echo "  value_method: $VALUE_METHOD"
    echo "  num_states: $NUM_STATES"
    echo "  predictor_type: $PREDICTOR_TYPE"
    echo "  state_loss_weight: $STATE_LOSS_WEIGHT"
    echo "  value_loss_weight: $VALUE_LOSS_WEIGHT"
    echo "  seed: $seed"
    echo "  run_id: $RUN_ID"
    echo "  output_dir: $OUTPUT_DIR"
    
    # Construct and run the command
    CMD="python -m neural_networks.train_drm $DB_PATH \
      --system_type=$SYSTEM_TYPE \
      --value_method=$VALUE_METHOD \
      --epochs=$EPOCHS \
      --val_size=$VAL_SIZE \
      --test_size=$TEST_SIZE \
      --batch_size=$BATCH_SIZE \
      --patience=$PATIENCE \
      --hidden_dim=$HIDDEN_DIM \
      --state_loss_weight=$STATE_LOSS_WEIGHT \
      --value_loss_weight=$VALUE_LOSS_WEIGHT \
      --num_states=$NUM_STATES \
      $USE_WARMUP \
      $USE_TARGET_ENCODER \
      $USE_ENTROPY_REG \
      --state_loss_type=$STATE_LOSS_TYPE \
      --use_gumbel \
      --min_temp=$GUMBEL_MIN \
      --initial_temp=$GUMBEL_TEMP \
      --predictor_type=$PREDICTOR_TYPE \
      $SORT_STATES \
      --lr=$LR \
      --seed=$seed \
      --output_dir=$OUTPUT_DIR \
      --run_id=$RUN_ID"
    
    echo "Running command: $CMD"
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed seed $seed at $(date)"
    else
        echo "✗ Failed seed $seed at $(date)"
    fi
    
    # Small delay between runs
    sleep 2
done

echo "All seeds for array task $SLURM_ARRAY_TASK_ID (dataset $DATASET_NUM, value method $VALUE_METHOD, num_states $NUM_STATES) completed at $(date)"