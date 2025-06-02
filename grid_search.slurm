#!/bin/bash
#SBATCH --job-name=drm_grid_predictor
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
#SBATCH --array=1-4

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

# Define databases
DB_PATHS=("datasets/results/single_saddle_system.db" "datasets/results/single_saddle_system_horizontal.db")
DB_NAMES=("regular" "horizontal")

# Fixed parameters for all runs
BASE_OUTPUT_DIR="neural_networks/output/saddle_grid_search_predictor"
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
VALUE_METHOD="angle"
STATE_LOSS_WEIGHT=1.0

# FIXED PARAMETERS as requested
NUM_STATES=4
GUMBEL_TEMP=3.0
GUMBEL_MIN=0.5
# No state diversity (removed from command)
SORT_STATES="--sort_states"

# Define parameter values for grid search
PREDICTOR_TYPE_VALUES=("bilinear" "standard")
VALUE_LOSS_WEIGHT_VALUES=(1.0 2.0)
SEEDS=(42 11)

# Total combinations: 2 predictors × 2 value_loss_weight × 2 datasets × 2 seeds = 16
# Running 4 combinations per array job = 4 array jobs total

# Calculate which 4 runs this array task should handle
runs_per_task=4
start_run=$(((SLURM_ARRAY_TASK_ID - 1) * runs_per_task))

echo "Array task $SLURM_ARRAY_TASK_ID will handle runs $start_run to $((start_run + runs_per_task - 1))"

# Loop through the 4 runs assigned to this array task
for run_offset in $(seq 0 $((runs_per_task - 1))); do
    global_run_idx=$((start_run + run_offset))
    
    # Break down global_run_idx into parameter indices
    # Order: predictor (2) × value_loss_weight (2) × dataset (2) × seed (2)
    
    # Extract indices using modular arithmetic
    seed_idx=$((global_run_idx % 2))
    temp1=$((global_run_idx / 2))
    
    db_idx=$((temp1 % 2))
    temp2=$((temp1 / 2))
    
    value_loss_weight_idx=$((temp2 % 2))
    predictor_idx=$((temp2 / 2))
    
    # Get actual parameter values
    PREDICTOR_TYPE=${PREDICTOR_TYPE_VALUES[$predictor_idx]}
    VALUE_LOSS_WEIGHT=${VALUE_LOSS_WEIGHT_VALUES[$value_loss_weight_idx]}
    DB_PATH=${DB_PATHS[$db_idx]}
    DB_NAME=${DB_NAMES[$db_idx]}
    SEED=${SEEDS[$seed_idx]}
    
    # Create descriptive run name
    CONFIG_NAME="ns${NUM_STATES}_${PREDICTOR_TYPE}_vlw${VALUE_LOSS_WEIGHT}_${DB_NAME}_seed${SEED}"
    
    # All runs go in the same base directory (no separate dataset folders)
    OUTPUT_DIR="${BASE_OUTPUT_DIR}"
    mkdir -p "$OUTPUT_DIR"
    
    # Create unique run ID
    RUN_ID="${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)_${global_run_idx}"
    
    echo "========================================================"
    echo "Running global run $global_run_idx:"
    echo "  num_states: $NUM_STATES"
    echo "  predictor_type: $PREDICTOR_TYPE"
    echo "  value_loss_weight: $VALUE_LOSS_WEIGHT"
    echo "  database: $DB_NAME ($DB_PATH)"
    echo "  seed: $SEED"
    echo "  run_id: $RUN_ID"
    echo "  output_dir: $OUTPUT_DIR"
    
    # Construct and run the command
    CMD="python -m neural_networks.train_drm $DB_PATH \
      --system_type=saddle_system \
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
      --value_method=$VALUE_METHOD \
      --predictor_type=$PREDICTOR_TYPE \
      $SORT_STATES \
      --lr=$LR \
      --seed=$SEED \
      --output_dir=$OUTPUT_DIR \
      --run_id=$RUN_ID"
    
    echo "Running command: $CMD"
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed run $global_run_idx at $(date)"
    else
        echo "✗ Failed run $global_run_idx at $(date)"
    fi
    echo "------------------------------------------------------"
    
    # Small delay between runs
    sleep 2
done

echo "All runs for array task $SLURM_ARRAY_TASK_ID completed at $(date)"