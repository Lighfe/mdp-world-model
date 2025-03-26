#!/bin/bash
#SBATCH --job-name=drm_gumbel
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=log/drm_gumbel_%A_%a.out
#SBATCH --error=log/drm_gumbel_%A_%a.err
#SBATCH --array=1-16

# Print job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Load modules and activate environment
module load anaconda
source activate mdp-world-model

# Create log directory if it doesn't exist
mkdir -p log

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Set PyTorch threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# Get configuration based on array task ID
task_id=$SLURM_ARRAY_TASK_ID

# Fixed parameters
db_path="datasets/results/tech_angular_lt.db"
ema_decay="0.9"
loss_weights="--state_loss_weight=0.5 --value_loss_weight=2.0"
hidden_dim="64"
lr="0.00005"
warmup="--use_warmup --warmup_epochs=5"
value_method="market_share"
epochs="100"

# Define exploration parameters (2 options each)
predictor_options=(
    "standard"
    "bilinear"
)

temp_options=(
    "--initial_temp=5.0 --min_temp=0.5"
    "--initial_temp=10.0 --min_temp=1.0"
)

entropy_options=(
    ""
    "--use_entropy_reg --entropy_weight=5.0"
)

batch_size_options=(
    "64"
    "128"
)

# Calculate indices based on task_id using binary representation
# This ensures we get exactly one of each possible combination
# With 4 binary parameters, we have 2^4 = 16 combinations

# task_id is 1-based, convert to 0-based for calculations
adjusted_id=$((task_id - 1))

# Extract bits for each parameter (each bit represents a binary choice)
predictor_idx=$((adjusted_id & 1))         # 1st bit
temp_idx=$(((adjusted_id >> 1) & 1))       # 2nd bit
entropy_idx=$(((adjusted_id >> 2) & 1))    # 3rd bit
batch_size_idx=$(((adjusted_id >> 3) & 1)) # 4th bit

# Select parameters
predictor="${predictor_options[$predictor_idx]}"
temp="${temp_options[$temp_idx]}"
entropy="${entropy_options[$entropy_idx]}"
batch_size="${batch_size_options[$batch_size_idx]}"

# Debug information
echo "Parameter selection:"
echo "Predictor: $predictor_idx (${predictor})"
echo "Temperature: $temp_idx (${temp})"
echo "Entropy: $entropy_idx (${entropy})"
echo "Batch size: $batch_size_idx (${batch_size})"

# Function to run a single experiment
run_experiment() {
    local repeat_id=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    
    # Create a descriptive run ID
    local run_id="gumbel_p${predictor_idx}_t${temp_idx}_e${entropy_idx}_b${batch_size_idx}_r${repeat_id}_${timestamp}"
    
    echo "Running configuration (repeat ${repeat_id}): $run_id"
    echo "Database: $db_path"
    echo "Predictor: $predictor"
    echo "Temperature: $temp"
    echo "Entropy: $entropy"
    echo "Batch size: $batch_size"
    echo "Fixed parameters:"
    echo "  Hidden dim: $hidden_dim"
    echo "  EMA decay: $ema_decay"
    echo "  Loss weights: $loss_weights"
    echo "  Learning rate: $lr"
    echo "  Warmup: $warmup"
    echo "  Epochs: $epochs"
    
    # Run the training command
    python -m neural_networks.train_drm $db_path \
      --output_dir=neural_networks/output \
      --run_id="$run_id" \
      --val_size=2000 \
      --test_size=2000 \
      --batch_size=$batch_size \
      --epochs=$epochs \
      --lr=$lr \
      --patience=15 \
      --num_states=4 \
      --hidden_dim=$hidden_dim \
      --predictor_type=$predictor \
      --value_method="$value_method" \
      --use_target_encoder \
      --ema_decay=$ema_decay \
      --use_gumbel \
      $temp \
      $loss_weights \
      $warmup \
      $entropy
    
    echo "Experiment ${repeat_id} finished at $(date)"
}

# Run two identical experiments sequentially
run_experiment 0
run_experiment 1

echo "Job finished at $(date)"