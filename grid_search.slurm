#!/bin/bash
#SBATCH --job-name=encoder_init_grid_search
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=log/encoder_init_%A_%a.out
#SBATCH --error=log/encoder_init_%A_%a.err
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

# Fixed parameters (from grid_search.slurm + user specifications)
SYSTEM_TYPE="saddle_system"
BASE_OUTPUT_DIR="neural_networks/output/encoder_init_test"
DB_PATH="datasets/results/multi_saddle_1.db"
VALUE_METHOD="angle"
VALUE_LOSS_WEIGHT=1.5
ENTROPY_DECAY_PROPORTION=0.3
SEED=17

# Fixed training parameters from grid_search.slurm
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
GUMBEL_TEMP=3.0
GUMBEL_MIN=0.5
STATE_LOSS_WEIGHT=1.0

# Grid search configurations: encoder initialization methods
ENCODER_INIT_METHODS=(
    "xavier_uniform"    # Array ID 1
    "xavier_normal"     # Array ID 2
    "chaos"             # Array ID 3
    "he"                # Array ID 4
)

# Select encoder initialization method based on array task ID
ENCODER_INIT_METHOD=${ENCODER_INIT_METHODS[$((SLURM_ARRAY_TASK_ID-1))]}

echo "========================================================"
echo "Encoder Initialization Grid Search - Array $SLURM_ARRAY_TASK_ID"
echo "Database: $DB_PATH"
echo "Encoder Init Method: $ENCODER_INIT_METHOD"
echo "Value Method: $VALUE_METHOD"
echo "Value Loss Weight: $VALUE_LOSS_WEIGHT"
echo "Entropy Decay Proportion: $ENTROPY_DECAY_PROPORTION"
echo "Seed: $SEED"
echo "========================================================"

# Create output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Create descriptive run name
CONFIG_NAME="ms1_${VALUE_METHOD}_vw${VALUE_LOSS_WEIGHT}_ed${ENTROPY_DECAY_PROPORTION}_encoder${ENCODER_INIT_METHOD}_seed${SEED}"

# Create unique run ID
RUN_ID="${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "Running configuration:"
echo "  database: $DB_PATH"
echo "  encoder_init_method: $ENCODER_INIT_METHOD"
echo "  value_method: $VALUE_METHOD"
echo "  value_loss_weight: $VALUE_LOSS_WEIGHT"
echo "  entropy_decay_proportion: $ENTROPY_DECAY_PROPORTION"
echo "  seed: $SEED"
echo "  run_id: $RUN_ID"

# Construct and run the command (note: no --sort_states flag as requested)
CMD="python -m neural_networks.train_drm $DB_PATH \
  --system_type=$SYSTEM_TYPE \
  --value_method=$VALUE_METHOD \
  --encoder_init_method=$ENCODER_INIT_METHOD \
  --epochs=$EPOCHS \
  --val_size=$VAL_SIZE \
  --test_size=$TEST_SIZE \
  --probing-size=$PROBING_SIZE \
  --batch_size=$BATCH_SIZE \
  --hidden_dim=$HIDDEN_DIM \
  --state_loss_weight=$STATE_LOSS_WEIGHT \
  --value_loss_weight=$VALUE_LOSS_WEIGHT \
  --entropy_decay_proportion=$ENTROPY_DECAY_PROPORTION \
  $USE_WARMUP \
  $USE_TARGET_ENCODER \
  $USE_ENTROPY_REG \
  --state_loss_type=$STATE_LOSS_TYPE \
  --use_gumbel \
  --min_temp=$GUMBEL_MIN \
  --initial_temp=$GUMBEL_TEMP \
  --predictor_type=$PREDICTOR_TYPE \
  --lr=$LR \
  --seed=$SEED \
  --output_dir=$BASE_OUTPUT_DIR \
  --run_id=$RUN_ID"

echo "Running command: $CMD"
eval $CMD

if [ $? -eq 0 ]; then
    echo "✓ Successfully completed $ENCODER_INIT_METHOD at $(date)"
else
    echo "✗ Failed $ENCODER_INIT_METHOD at $(date)"
fi

echo "Array $SLURM_ARRAY_TASK_ID completed at $(date)"