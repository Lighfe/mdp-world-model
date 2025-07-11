#!/bin/bash
#SBATCH --job-name=collapse_analysis
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=log/collapse_analysis_%A_%a.out
#SBATCH --error=log/collapse_analysis_%A_%a.err
#SBATCH --array=1-2

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
echo "Activating conda environment..."

# Try different activation methods
if command -v conda &> /dev/null; then
    echo "Using conda activate..."
    conda activate mdp-world-model
else
    echo "Using source activate..."
    source activate mdp-world-model
fi

# Verify environment activation
echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Set PyTorch threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# Fixed parameters for all runs
SYSTEM_TYPE="saddle_system"
BASE_OUTPUT_DIR="neural_networks/output/collapse_analysis"
EPOCHS=100
VAL_SIZE=2000
TEST_SIZE=2000
BATCH_SIZE=64
HIDDEN_DIM=32
LR=1e-5
USE_WARMUP="--use_warmup --warmup_epochs=10"
USE_TARGET_ENCODER="--use_target_encoder --ema_decay=0.9"
USE_ENTROPY_REG="--use_entropy_reg --entropy_weight=1.0"
STATE_LOSS_TYPE="kl_div"
PREDICTOR_TYPE="standard"
PROBING_SIZE=5000

# Fixed parameters
GUMBEL_TEMP=3.0
GUMBEL_MIN=0.5
SORT_STATES="--sort_states"
STATE_LOSS_WEIGHT=1.0

# Define configurations for each array
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    # Array 1: Non-collapsed + some collapsed
    echo "=== ARRAY 1: Non-collapsed + Collapsed cases ==="
    CONFIGS=(
        "angle 2.0 0.2 4 5 non-collapsed"       # ms1_angle_vw2.0_ed0.2_s4_seed5
        "identity 1.5 0.2 4 6 non-collapsed"   # ms1_identity_vw1.5_ed0.2_s4_seed6
        "angle 2.0 0.5 4 3 non-collapsed"      # ms1_angle_vw2.0_ed0.5_s4_seed3
        "angle 2.0 0.2 4 6 collapsed"          # ms1_angle_vw2.0_ed0.2_s4_seed6
        "angle 2.0 0.5 4 10 collapsed"         # ms1_angle_vw2.0_ed0.5_s4_seed10
    )
else
    # Array 2: Remaining collapsed + collapsed different
    echo "=== ARRAY 2: Collapsed + Collapsed Different cases ==="
    CONFIGS=(
        "identity 1.5 0.2 4 11 collapsed"         # ms1_identity_vw1.5_ed0.2_s4_seed11
        "identity 1.5 0.5 4 6 collapsed-different" # ms1_identity_vw1.5_ed0.5_s4_seed6
        "angle 1.5 0.2 4 8 collapsed-different"   # ms1_angle_vw1.5_ed0.2_s4_seed8
        "identity 1.5 0.5 4 7 collapsed-different" # ms1_identity_vw1.5_ed0.5_s4_seed7
        "identity 1.5 0.2 4 3 collapsed-different" # ms1_identity_vw1.5_ed0.2_s4_seed3
    )
fi

# Fixed dataset
DATASET_NUM=1
DB_PATH="datasets/results/multi_saddle_${DATASET_NUM}.db"

echo "========================================================"
echo "Array $SLURM_ARRAY_TASK_ID will run ${#CONFIGS[@]} configurations sequentially"
echo "Database: $DB_PATH"
echo "========================================================"

# Run all configurations for this array
for i in "${!CONFIGS[@]}"; do
    CONFIG=(${CONFIGS[$i]})
    VALUE_METHOD=${CONFIG[0]}
    VALUE_LOSS_WEIGHT=${CONFIG[1]}
    ENTROPY_DECAY_PROPORTION=${CONFIG[2]}
    NUM_STATES=${CONFIG[3]}
    SEED=${CONFIG[4]}
    COLLAPSE_TYPE=${CONFIG[5]}
    
    # Create descriptive run name
    CONFIG_NAME="ms${DATASET_NUM}_${VALUE_METHOD}_vw${VALUE_LOSS_WEIGHT}_ed${ENTROPY_DECAY_PROPORTION}_s${NUM_STATES}_seed${SEED}"
    
    # Create output directory
    OUTPUT_DIR="${BASE_OUTPUT_DIR}"
    mkdir -p "$OUTPUT_DIR"
    
    # Create unique run ID with collapse type annotation
    RUN_ID="${CONFIG_NAME}_${COLLAPSE_TYPE}_$(date +%Y%m%d_%H%M%S)"
    
    echo "--------------------------------------------------------"
    echo "Array $SLURM_ARRAY_TASK_ID - Config $((i+1))/${#CONFIGS[@]} (${COLLAPSE_TYPE}):"
    echo "  database: $DB_PATH"
    echo "  value_method: $VALUE_METHOD"
    echo "  value_loss_weight: $VALUE_LOSS_WEIGHT"
    echo "  entropy_decay_proportion: $ENTROPY_DECAY_PROPORTION"
    echo "  num_states: $NUM_STATES"
    echo "  seed: $SEED"
    echo "  run_id: $RUN_ID"
    
    # Construct and run the command
    CMD="python -m neural_networks.train_drm $DB_PATH \
      --system_type=$SYSTEM_TYPE \
      --value_method=$VALUE_METHOD \
      --epochs=$EPOCHS \
      --val_size=$VAL_SIZE \
      --test_size=$TEST_SIZE \
      --probing-size=$PROBING_SIZE \
      --batch_size=$BATCH_SIZE \
      --hidden_dim=$HIDDEN_DIM \
      --state_loss_weight=$STATE_LOSS_WEIGHT \
      --value_loss_weight=$VALUE_LOSS_WEIGHT \
      --entropy_decay_proportion=$ENTROPY_DECAY_PROPORTION \
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
      --seed=$SEED \
      --output_dir=$OUTPUT_DIR \
      --run_id=$RUN_ID"
    
    echo "Running command: $CMD"
    eval $CMD
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed config $((i+1)) (${COLLAPSE_TYPE}) at $(date)"
    else
        echo "✗ Failed config $((i+1)) (${COLLAPSE_TYPE}) at $(date)"
    fi
    
    # Small delay between runs
    sleep 5
done

echo "Array $SLURM_ARRAY_TASK_ID completed all configurations at $(date)"