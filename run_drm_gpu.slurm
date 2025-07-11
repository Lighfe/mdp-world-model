#!/bin/bash
#SBATCH --job-name=test_single_collapse
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=log/test_single_collapse_%j.out
#SBATCH --error=log/test_single_collapse_%j.err

# Print job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# Create log directory if it doesn't exist
mkdir -p log

# Load modules and initialize conda
module load anaconda
echo "Initializing conda..."

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Alternative: Use direct path if conda command still not available
if ! command -v conda &> /dev/null; then
    echo "Conda command not found, trying direct initialization..."
    # Use the base conda path from your environment list
    source /p/system/packages/tools/miniforge/25.3.0-3/etc/profile.d/conda.sh
fi

echo "Activating conda environment..."
conda activate mdp-world-model

# Verify environment activation
echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Set PyTorch threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# Fixed parameters for all runs
SYSTEM_TYPE="saddle_system"
BASE_OUTPUT_DIR="neural_networks/output/collapse_analysis_test"
EPOCHS=50
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

# Test configuration: ms1_angle_vw2.0_ed0.5_s4_seed3
DATASET_NUM=1
VALUE_METHOD="angle"
VALUE_LOSS_WEIGHT=2.0
ENTROPY_DECAY_PROPORTION=0.5
NUM_STATES=4
SEED=3

# Get database path
DB_PATH="datasets/results/multi_saddle_${DATASET_NUM}.db"

# Create descriptive run name
CONFIG_NAME="ms${DATASET_NUM}_${VALUE_METHOD}_vw${VALUE_LOSS_WEIGHT}_ed${ENTROPY_DECAY_PROPORTION}_s${NUM_STATES}_seed${SEED}_test"

# Create output directory
OUTPUT_DIR="${BASE_OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

# Create unique run ID
RUN_ID="${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "========================================================"
echo "Test run configuration:"
echo "  Dataset: $DATASET_NUM"
echo "  Value method: $VALUE_METHOD"
echo "  Value weight: $VALUE_LOSS_WEIGHT"
echo "  Entropy decay proportion: $ENTROPY_DECAY_PROPORTION"
echo "  Num states: $NUM_STATES"
echo "  Seed: $SEED"
echo "  Epochs: $EPOCHS (reduced for testing)"
echo "  Database: $DB_PATH"
echo "  Run ID: $RUN_ID"
echo "========================================================"

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
    echo "✓ Successfully completed test run at $(date)"
else
    echo "✗ Failed test run at $(date)"
fi

echo "Test run completed at $(date)"