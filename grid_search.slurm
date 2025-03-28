#!/bin/bash
#SBATCH --job-name=drm_state_div
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=log/drm_state_div_%A_%a.out
#SBATCH --error=log/drm_state_div_%A_%a.err
#SBATCH --array=1-6

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
predictor="bilinear"
hidden_dim="64"
batch_size="128"
value_method="market_share"
db_path="datasets/results/tech_angular_lt.db"
warmup="--use_warmup --warmup_epochs=5"
lr="0.00005"
loss_weights="--state_loss_weight=0.5 --value_loss_weight=2.0"
gumbel="--use_gumbel"
temp="--initial_temp=5.0 --min_temp=0.5"
target_encoder="--use_target_encoder"
patience="20"
state_div="--use_state_diversity"

# Exploring options
ema_decay_options=(
    "0.9"
    "0.996"
)

entropy_options=(
    ""
    "--use_entropy_reg --entropy_weight=5.0"
)

div_weight_options=(
    "--diversity_weight=0.001"
    "--diversity_weight=0.01"
    "--diversity_weight=0.1"
)

# Calculate total number of combinations
# 2 ema_decay options × 2 entropy options × 3 div_weight options = 12 combinations
total_combinations=$((2 * 2 * 3))

# Calculate which combinations to run based on task_id
# Each task will run 2 combinations (each combination twice)
# Task 1 runs combinations 0,1; Task 2 runs combinations 2,3; etc.
start_combo_idx=$(( 2 * (task_id - 1) ))
second_combo_idx=$(( start_combo_idx + 1 ))

# Function to extract parameter choices based on combination index
get_params() {
    local combo_idx=$1
    
    # Calculate indices for each parameter set
    local ema_decay_idx=$(( combo_idx / 6 ))         # 0-1 (2 options)
    local entropy_idx=$(( (combo_idx % 6) / 3 ))     # 0-1 (2 options)
    local div_weight_idx=$(( combo_idx % 3 ))        # 0-2 (3 options)
    
    # Select parameters
    local ema_decay="${ema_decay_options[$ema_decay_idx]}"
    local entropy="${entropy_options[$entropy_idx]}"
    local div_weight="${div_weight_options[$div_weight_idx]}"
    
    echo "$ema_decay" "$entropy" "$div_weight" "$ema_decay_idx" "$entropy_idx" "$div_weight_idx"
}

# Function to run a single experiment
run_experiment() {
    local combo_idx=$1
    local repeat_id=$2
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    
    # Get parameters for this combination
    read ema_decay entropy div_weight ema_decay_idx entropy_idx div_weight_idx <<< $(get_params $combo_idx)
    
    # Create a descriptive run ID
    local run_id="sd_ema${ema_decay_idx}_e${entropy_idx}_dw${div_weight_idx}_r${repeat_id}_${timestamp}"
    
    echo "Running configuration (repeat ${repeat_id}): $run_id"
    echo "Database: $db_path"
    echo "Predictor (fixed): $predictor"
    echo "Hidden dim (fixed): $hidden_dim"
    echo "Batch size (fixed): $batch_size"
    echo "Learning rate (fixed): $lr"
    echo "EMA decay: $ema_decay"
    echo "Entropy: $entropy"
    echo "Loss weights: $loss_weights"
    echo "State diversity: $state_div"
    echo "Diversity weight: $div_weight"
    echo "Gumbel: $gumbel"
    echo "Target encoder: $target_encoder"
    echo "Warmup: $warmup"
    
    # Run the training command
    python -m neural_networks.train_drm $db_path \
      --output_dir=neural_networks/output \
      --run_id="$run_id" \
      --val_size=2000 \
      --test_size=2000 \
      --batch_size=$batch_size \
      --epochs=100 \
      --lr=$lr \
      --patience=$patience \
      --num_states=4 \
      --hidden_dim=$hidden_dim \
      --predictor_type=$predictor \
      --value_method="$value_method" \
      --ema_decay=$ema_decay \
      $loss_weights \
      $warmup \
      $gumbel \
      $temp \
      $target_encoder \
      $state_div \
      $div_weight \
      $entropy
    
    echo "Experiment ${repeat_id} finished at $(date)"
}

# Run first combination
echo "Running combination $start_combo_idx (of $total_combinations total)"
run_experiment $start_combo_idx 0
run_experiment $start_combo_idx 1

# Run second combination
echo "Running combination $second_combo_idx (of $total_combinations total)"
run_experiment $second_combo_idx 0
run_experiment $second_combo_idx 1

echo "Job finished at $(date)"