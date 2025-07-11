#!/bin/bash
#SBATCH --job-name=multi_saddle_probing
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=log/multi_saddle_test_%A_%a.out
#SBATCH --error=log/multi_saddle_test_%A_%a.err

# Print job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

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
BASE_OUTPUT_DIR="neural_networks/output/multi_saddle_test"
EPOCHS=50
VAL_SIZE=2000
TEST_SIZE=2000
BATCH_SIZE=64
HIDDEN_DIM=32
LR=1e-5
USE_WARMUP="--use_warmup --warmup_epochs=5"
USE_TARGET_ENCODER="--use_target_encoder --ema_decay=0.9"
USE_ENTROPY_REG="--use_entropy_reg --entropy_weight=1.0"
STATE_LOSS_TYPE="kl_div"
PREDICTOR_TYPE="standard"
PROBING_SIZE=2000
GUMBEL_TEMP=3.0
GUMBEL_MIN=0.5
STATE_LOSS_WEIGHT=1.0
SEED=3

# Parameter arrays
DATASET_NUM=1
VALUE_METHOD="angle"
VALUE_LOSS_WEIGHT=2.0
ENTROPY_DECAY_PROPORTION=0.5
NUM_STATES=4

# Get database path for this dataset
DB_PATH="datasets/results/multi_saddle_${DATASET_NUM}.db"

echo "========================================================"
echo "  Dataset: $DATASET_NUM"
echo "  Value method: $VALUE_METHOD"
echo "  Value weight: $VALUE_LOSS_WEIGHT"
echo "  Entropy decay proportion: $ENTROPY_DECAY_PROPORTION"
echo "  Num states: $NUM_STATES"
echo "  Database: $DB_PATH"
echo "  Probing size: $PROBING_SIZE"
echo "Will run ${#SEEDS[@]} seeds sequentially: ${SEEDS[*]}"
echo "========================================================"


# Create descriptive run name
CONFIG_NAME="ms${DATASET_NUM}_${VALUE_METHOD}_vw${VALUE_LOSS_WEIGHT}_ed${ENTROPY_DECAY_PROPORTION}_s${NUM_STATES}_seed${seed}"

# Create output directory
OUTPUT_DIR="${BASE_OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

# Create unique run ID
RUN_ID="${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"

echo "--------------------------------------------------------"
echo "Starting run with seed $seed:"
echo "  database: $DB_PATH"
echo "  value_method: $VALUE_METHOD"
echo "  value_loss_weight: $VALUE_LOSS_WEIGHT"
echo "  entropy_decay_proportion: $ENTROPY_DECAY_PROPORTION"
echo "  num_states: $NUM_STATES"
echo "  probing_size: $PROBING_SIZE"
echo "  seed: $seed"
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
    --patience=$PATIENCE \
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
    --seed=$seed \
    --output_dir=$OUTPUT_DIR \
    --run_id=$RUN_ID"

echo "Running command: $CMD"
eval $CMD 


echo "All seeds for array task $SLURM_ARRAY_TASK_ID completed at $(date)"