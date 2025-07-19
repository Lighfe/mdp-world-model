#!/bin/bash
#SBATCH --job-name=init_experiment
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=log/init_experiment_%A_%a.out
#SBATCH --error=log/init_experiment_%A_%a.err
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

# Calculate seed and dataset from array ID
# Array 1-10: seed 1-10, dataset 1
# Array 11-20: seed 1-10, dataset 3  
# Array 21-30: seed 1-10, dataset 5
# Array 31-40: seed 1-10, dataset 7
SEED=$(( ((SLURM_ARRAY_TASK_ID - 1) % 10) + 1 ))
DATASET_INDEX=$(( (SLURM_ARRAY_TASK_ID - 1) / 10 ))
DATASETS=(1 3 5 7)
DATASET_NUM=${DATASETS[$DATASET_INDEX]}

# Fixed parameters
SYSTEM_TYPE="saddle_system"
BASE_OUTPUT_DIR="neural_networks/output/init_experiment"
DB_PATH="datasets/results/multi_saddle_${DATASET_NUM}.db"
VALUE_METHOD="angle"
VALUE_LOSS_WEIGHT=1.5
ENTROPY_DECAY_PROPORTION=0.5

# Fixed training parameters
EPOCHS=120
VAL_SIZE=1000
TEST_SIZE=1000
BATCH_SIZE=64
HIDDEN_DIM=32
LR=1e-5
USE_WARMUP="--use_warmup --warmup_epochs=10"
USE_TARGET_ENCODER="--use_target_encoder --ema_decay=0.9"
USE_ENTROPY_REG="--use_entropy_reg"
STATE_LOSS_TYPE="kl_div"
PREDICTOR_TYPE="standard"
PROBING_SIZE=2000
GUMBEL_TEMP=3.0
GUMBEL_MIN=0.5
STATE_LOSS_WEIGHT=1.0

# Grid search configurations: encoder_init_method × entropy_weight
ENCODER_INIT_METHODS=(
    "xavier_uniform"
    "xavier_normal" 
    "chaos"
    "he"
)

ENTROPY_WEIGHTS=(
    "1.0"
    "0.5"
)

echo "========================================================"
echo "Init Experiment - Array $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED"
echo "Dataset: multi_saddle_${DATASET_NUM}.db"
echo "Will run ${#ENCODER_INIT_METHODS[@]} encoder initialization methods × ${#ENTROPY_WEIGHTS[@]} entropy weights = $((${#ENCODER_INIT_METHODS[@]} * ${#ENTROPY_WEIGHTS[@]})) total configurations"
echo "========================================================"

# Create output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Track configuration number
CONFIG_NUM=0

# Run all combinations
for ENCODER_INIT_METHOD in "${ENCODER_INIT_METHODS[@]}"; do
    for ENTROPY_WEIGHT in "${ENTROPY_WEIGHTS[@]}"; do
        CONFIG_NUM=$((CONFIG_NUM + 1))
        
        # Create descriptive run name
        CONFIG_NAME="ms${DATASET_NUM}_${VALUE_METHOD}_vw${VALUE_LOSS_WEIGHT}_ed${ENTROPY_DECAY_PROPORTION}_encoder${ENCODER_INIT_METHOD}_ew${ENTROPY_WEIGHT}_seed${SEED}"
        
        # Create unique run ID
        RUN_ID="${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"
        
        echo "--------------------------------------------------------"
        echo "Array $SLURM_ARRAY_TASK_ID - Config ${CONFIG_NUM}/8:"
        echo "  database: $DB_PATH"
        echo "  seed: $SEED"
        echo "  encoder_init_method: $ENCODER_INIT_METHOD"
        echo "  entropy_weight: $ENTROPY_WEIGHT"
        echo "  value_method: $VALUE_METHOD"
        echo "  value_loss_weight: $VALUE_LOSS_WEIGHT"
        echo "  entropy_decay_proportion: $ENTROPY_DECAY_PROPORTION"
        echo "  run_id: $RUN_ID"
        
        # Construct and run the command (note: no --sort_states flag)
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
          --entropy_weight=$ENTROPY_WEIGHT \
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
            echo "✓ Successfully completed config ${CONFIG_NUM}/8 (${ENCODER_INIT_METHOD}, ew=${ENTROPY_WEIGHT}) at $(date)"
        else
            echo "✗ Failed config ${CONFIG_NUM}/8 (${ENCODER_INIT_METHOD}, ew=${ENTROPY_WEIGHT}) at $(date)"
        fi
        
        # Small delay between configurations
        sleep 5
    done
done

echo "Array $SLURM_ARRAY_TASK_ID (seed=${SEED}, dataset=${DATASET_NUM}) completed all 8 configurations at $(date)"