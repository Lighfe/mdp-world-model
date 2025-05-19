#!/bin/bash
#SBATCH --job-name=drm_grid
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --output=log/drm_grid_%A_%a.out
#SBATCH --error=log/drm_grid_%A_%a.err
#SBATCH --array=1-32

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
source activate mdp-world-model  # Replace with your actual environment name

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Set PyTorch threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# Define databases
DB_PATHS=("datasets/results/single_saddle_system.db" "datasets/results/single_saddle_system_horizontal.db")
DB_NAMES=("regular" "horizontal")

# Fixed parameters for all runs
BASE_OUTPUT_DIR="neural_networks/output/saddle_grid_search"
EPOCHS=100
VAL_SIZE=2000
TEST_SIZE=2000
BATCH_SIZE=64
PATIENCE=100
HIDDEN_DIM=64
LR=1e-5
ENTROPY_WEIGHT=1.0
USE_WARMUP="--use_warmup --warmup_epochs=10"
USE_TARGET_ENCODER="--use_target_encoder --ema_decay=0.9"
USE_ENTROPY_REG="--use_entropy_reg --entropy_weight=$ENTROPY_WEIGHT"
STATE_LOSS_TYPE="kl_div"
VALUE_METHOD="angle"
VALUE_LOSS_WEIGHT=2.0

# Define parameter values for grid search
NUM_STATES_VALUES=(2 4)
GUMBEL_TEMP_VALUES=(2.0 5.0)
STATE_LOSS_WEIGHT_VALUES=(0.5 1.0)
STATE_DIVERSITY_OPTIONS=("--use_state_diversity --diversity_weight=0.01" "")
PREDICTOR_TYPE_VALUES=("bilinear" "standard")
SEEDS=(42 11)

# Each array task will handle 4 runs (2 configurations x 2 seeds)
# Calculate the base configurations for this array task
config_start=$((2 * (SLURM_ARRAY_TASK_ID - 1)))
config_count=2  # Run 2 configurations per array task

# Loop through the assigned configurations
for config_offset in $(seq 0 $((config_count - 1))); do
    config_idx=$((config_start + config_offset))
    
    # Break down config_idx into parameter indices
    # First determine database index (config_idx / 32 will be 0 for first 32 configs, 1 for next 32)
    db_idx=$((config_idx / 32))
    # Adjust config_idx to be relative within each database's parameter space
    local_config_idx=$((config_idx % 32))
    
    # Now extract parameter indices from local_config_idx
    num_states_idx=$((local_config_idx / 16))
    remainder=$((local_config_idx % 16))
    
    gumbel_temp_idx=$((remainder / 8))
    remainder=$((remainder % 8))
    
    state_loss_weight_idx=$((remainder / 4))
    remainder=$((remainder % 4))
    
    state_diversity_idx=$((remainder / 2))
    predictor_type_idx=$((remainder % 2))
    
    # Get actual parameter values
    DB_PATH=${DB_PATHS[$db_idx]}
    DB_NAME=${DB_NAMES[$db_idx]}
    NUM_STATES=${NUM_STATES_VALUES[$num_states_idx]}
    GUMBEL_TEMP=${GUMBEL_TEMP_VALUES[$gumbel_temp_idx]}
    STATE_LOSS_WEIGHT=${STATE_LOSS_WEIGHT_VALUES[$state_loss_weight_idx]}
    STATE_DIVERSITY=${STATE_DIVERSITY_OPTIONS[$state_diversity_idx]}
    PREDICTOR_TYPE=${PREDICTOR_TYPE_VALUES[$predictor_type_idx]}
    
    # Create descriptive directory name (short but readable)
    STATE_DIV_TAG="sd1"
    if [ -z "$STATE_DIVERSITY" ]; then
        STATE_DIV_TAG="sd0"
    fi
    
    CONFIG_NAME="ns${NUM_STATES}_gt${GUMBEL_TEMP}_slw${STATE_LOSS_WEIGHT}_${STATE_DIV_TAG}_${PREDICTOR_TYPE}"
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${DB_NAME}/${CONFIG_NAME}"
    mkdir -p "$OUTPUT_DIR"
    
    # Run with each seed
    for SEED in "${SEEDS[@]}"; do
        # Create a unique run ID with timestamp
        RUN_ID="${CONFIG_NAME}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
        
        echo "========================================================"
        echo "Running configuration: $CONFIG_NAME with seed $SEED"
        echo "Database: $DB_NAME"
        echo "Config index: $config_idx (local index: $local_config_idx)"
        echo "Full run ID: $RUN_ID"
        echo "Output directory: $OUTPUT_DIR"
        
        # Construct the command
        CMD="python -m neural_networks.train_drm $DB_PATH \
          --system_type=saddle_system \
          --epochs=$EPOCHS \
          --val_size=$VAL_SIZE \
          --test_size=$TEST_SIZE \
          --batch_size=$BATCH_SIZE \
          --patience=$PATIENCE \
          --hidden_dim=$HIDDEN_DIM \
          --state_loss_weight=$STATE_LOSS_WEIGHT \
          $USE_WARMUP \
          $USE_TARGET_ENCODER \
          $USE_ENTROPY_REG \
          --state_loss_type=$STATE_LOSS_TYPE \
          --use_gumbel \
          --min_temp=0.5 \
          --initial_temp=$GUMBEL_TEMP \
          --value_loss_weight=$VALUE_LOSS_WEIGHT \
          --value_method=$VALUE_METHOD \
          --predictor_type=$PREDICTOR_TYPE \
          $STATE_DIVERSITY \
          --lr=$LR \
          --seed=$SEED \
          --output_dir=$OUTPUT_DIR \
          --run_id=$RUN_ID"
        
        echo "Running command: $CMD"
        eval $CMD
        
        echo "Completed run with seed $SEED at $(date)"
        echo "------------------------------------------------------"
        
        # Small delay between runs
        sleep 5
    done
done

echo "All jobs for task $SLURM_ARRAY_TASK_ID completed at $(date)"