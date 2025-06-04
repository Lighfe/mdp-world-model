#!/bin/bash
#SBATCH --job-name=drm_vicreg_search
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=3:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --output=log/drm_vicreg_%A_%a.out
#SBATCH --error=log/drm_vicreg_%A_%a.err
#SBATCH --array=1-19

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
BASE_OUTPUT_DIR="neural_networks/output/vicreg_param_search"
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
STATE_LOSS_WEIGHT=1.0  # Fixed as requested
VALUE_LOSS_WEIGHT=1.0  # Fixed as requested

# FIXED PARAMETERS
NUM_STATES=4
GUMBEL_TEMP=3.0
GUMBEL_MIN=0.5
SORT_STATES="--sort_states"

# Define parameter values for grid search
PREDICTOR_TYPE_VALUES=("standard" "bilinear")
SEEDS=(11 42)

# VICReg parameter search space
VARIANCE_TARGETS=(0.4 0.3)                    # 2 options
VICREG_WEIGHTS=(0.1 0.3 1.0)                 # 3 options  
COVARIANCE_WEIGHTS=(1.0 3.0 10.0)            # 3 options
# Total vicreg combinations: 1 (no vicreg) + 2×3×3 (vicreg params) = 19

# Total combinations: 2 systems × 2 predictors × 2 seeds × 19 vicreg = 152
# Running 8 combinations per array job = 19 array jobs total

# Calculate which 8 runs this array task should handle
runs_per_task=8
start_run=$(((SLURM_ARRAY_TASK_ID - 1) * runs_per_task))

echo "Array task $SLURM_ARRAY_TASK_ID will handle runs $start_run to $((start_run + runs_per_task - 1))"

# Function to get vicreg settings for a given vicreg_idx
get_vicreg_settings() {
    local vicreg_idx=$1
    
    if [ $vicreg_idx -eq 0 ]; then
        # No VICReg
        echo "NONE" "novicreg"
    else
        # VICReg with parameters (indices 1-18 map to param combinations 0-17)
        local param_idx=$((vicreg_idx - 1))
        
        # Break down param_idx into individual parameter indices
        local cov_idx=$((param_idx % 3))
        local temp1=$((param_idx / 3))
        local weight_idx=$((temp1 % 3))
        local var_idx=$((temp1 / 3))
        
        # Get actual parameter values
        local variance_target=${VARIANCE_TARGETS[$var_idx]}
        local vicreg_weight=${VICREG_WEIGHTS[$weight_idx]}
        local covariance_weight=${COVARIANCE_WEIGHTS[$cov_idx]}
        
        # Return vicreg command and descriptive name
        local vicreg_cmd="--use_vicreg --vicreg_variance_target=$variance_target --vicreg_weight=$vicreg_weight --vicreg_nu=$covariance_weight"
        local vicreg_name="vicreg_vt${variance_target}_vw${vicreg_weight}_cw${covariance_weight}"
        
        echo "$vicreg_cmd" "$vicreg_name"
    fi
}

# Loop through the 8 runs assigned to this array task
for run_offset in $(seq 0 $((runs_per_task - 1))); do
    global_run_idx=$((start_run + run_offset))
    
    # Check if we've exceeded total runs
    if [ $global_run_idx -ge 152 ]; then
        echo "Reached end of parameter space at run $global_run_idx"
        break
    fi
    
    # Break down global_run_idx into parameter indices
    # Order: system (2) × predictor (2) × seed (2) × vicreg (19)
    
    # Extract indices using modular arithmetic
    vicreg_idx=$((global_run_idx % 19))
    temp1=$((global_run_idx / 19))
    
    seed_idx=$((temp1 % 2))
    temp2=$((temp1 / 2))
    
    predictor_idx=$((temp2 % 2))
    system_idx=$((temp2 / 2))
    
    # Get actual parameter values
    SYSTEM_TYPE=${SYSTEMS[$system_idx]}
    VALUE_METHOD=${VALUE_METHODS[$system_idx]}
    DB_PATH=${DB_PATHS[$system_idx]}
    PREDICTOR_TYPE=${PREDICTOR_TYPE_VALUES[$predictor_idx]}
    SEED=${SEEDS[$seed_idx]}
    
    # Get VICReg settings
    read VICREG_SETTING VICREG_NAME <<< $(get_vicreg_settings $vicreg_idx)

    # Handle the NONE case for VICReg
    if [ "$VICREG_SETTING" != "NONE" ]; then
        VICREG_CMD="$VICREG_SETTING"
    else
        VICREG_CMD=""
    fi
    
    # Create descriptive run name
    CONFIG_NAME="${SYSTEM_TYPE}_${PREDICTOR_TYPE}_${VICREG_NAME}_seed${SEED}"
    
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