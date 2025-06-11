#!/bin/bash
#SBATCH --job-name=drm_vicreg_small
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --output=log/drm_vicreg_%A_%a.out
#SBATCH --error=log/drm_vicreg_%A_%a.err
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

# Define systems and their databases
SYSTEMS=("saddle_system" "tech_substitution")
VALUE_METHODS=("angle" "market_share")
DB_PATHS=("datasets/results/single_saddle_system_horizontal.db" "datasets/results/tech_sim.db")

# Fixed parameters for all runs
BASE_OUTPUT_DIR="neural_networks/output/vicreg_transformed_input"
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
PREDICTOR_TYPE="standard"  # Fixed to standard only

# FIXED PARAMETERS
NUM_STATES=4
GUMBEL_TEMP=3.0
GUMBEL_MIN=0.5
SORT_STATES="--sort_states"

# Define parameter values for grid search
SEEDS=(11 42)
STATE_LOSS_WEIGHTS=(0.5 1.0)                  # 2 options
VALUE_LOSS_WEIGHTS=(1.0 1.5)                  # 2 options

# VICReg configurations
VICREG_CONFIGS=(
    "NONE|novicreg"
    "--use_vicreg --vicreg_variance_target=0.5 --vicreg_weight=0.001 --vicreg_lambda=5 --vicreg_nu=2 --vicreg_invariance_schedule|vicreg_vt0.5_vw0.001_vl5_vn2"
)

# Total combinations: 2 systems × 2 seeds × 2 state_loss × 2 value_loss × 2 vicreg = 32
# Running 8 combinations per array job = 4 array jobs total

# Calculate which 8 runs this array task should handle
runs_per_task=8
start_run=$(((SLURM_ARRAY_TASK_ID - 1) * runs_per_task))

echo "Array task $SLURM_ARRAY_TASK_ID will handle runs $start_run to $((start_run + runs_per_task - 1))"

# Loop through the 8 runs assigned to this array task
for run_offset in $(seq 0 $((runs_per_task - 1))); do
    global_run_idx=$((start_run + run_offset))
    
    # Check if we've exceeded total runs
    if [ $global_run_idx -ge 32 ]; then
        echo "Reached end of parameter space at run $global_run_idx"
        break
    fi
    
    # Break down global_run_idx into parameter indices
    # Order: system (2) × seed (2) × state_loss (2) × value_loss (2) × vicreg (2)
    
    # Extract indices using modular arithmetic
    vicreg_idx=$((global_run_idx % 2))
    temp1=$((global_run_idx / 2))
    
    value_loss_idx=$((temp1 % 2))
    temp2=$((temp1 / 2))
    
    state_loss_idx=$((temp2 % 2))
    temp3=$((temp2 / 2))
    
    seed_idx=$((temp3 % 2))
    system_idx=$((temp3 / 2))
    
    # Get actual parameter values
    SYSTEM_TYPE=${SYSTEMS[$system_idx]}
    VALUE_METHOD=${VALUE_METHODS[$system_idx]}
    DB_PATH=${DB_PATHS[$system_idx]}
    SEED=${SEEDS[$seed_idx]}
    STATE_LOSS_WEIGHT=${STATE_LOSS_WEIGHTS[$state_loss_idx]}
    VALUE_LOSS_WEIGHT=${VALUE_LOSS_WEIGHTS[$value_loss_idx]}
    
    # Get VICReg settings using pipe delimiter
    IFS='|' read VICREG_SETTING VICREG_NAME <<< "${VICREG_CONFIGS[$vicreg_idx]}"

    # Handle the NONE case for VICReg
    if [ "$VICREG_SETTING" != "NONE" ]; then
        VICREG_CMD="$VICREG_SETTING"
    else
        VICREG_CMD=""
    fi
    
    # Create descriptive run name
    CONFIG_NAME="${SYSTEM_TYPE}_${PREDICTOR_TYPE}_sl${STATE_LOSS_WEIGHT}_vl${VALUE_LOSS_WEIGHT}_${VICREG_NAME}_seed${SEED}"
    
    # All runs go in the same base directory
    OUTPUT_DIR="${BASE_OUTPUT_DIR}"
    mkdir -p "$OUTPUT_DIR"
    
    # Create unique run ID
    RUN_ID="${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)_${global_run_idx}"
    
    echo "========================================================"
    echo "Running global run $global_run_idx:"
    echo "  system: $SYSTEM_TYPE"
    echo "  value_method: $VALUE_METHOD"
    echo "  predictor_type: $PREDICTOR_TYPE"
    echo "  state_loss_weight: $STATE_LOSS_WEIGHT"
    echo "  value_loss_weight: $VALUE_LOSS_WEIGHT"
    echo "  vicreg: $VICREG_NAME"
    echo "  database: $DB_PATH"
    echo "  seed: $SEED"
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
      $VICREG_CMD \
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