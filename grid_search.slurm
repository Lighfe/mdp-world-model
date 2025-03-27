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
#SBATCH --array=1-32

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
gumbel="--use_gumbel"
target_encoder="--use_target_encoder"
patience="20"

# Define exploration parameters (2 options each)
ema_decay_options=(
    "0.9"
    "0.5"
)

temp_options=(
    "--initial_temp=5.0 --min_temp=0.5"
    "--initial_temp=10.0 --min_temp=0.5"
)

entropy_options=(
    ""
    "--use_entropy_reg --entropy_weight=3.0"
)

loss_weights_options=(
    "--state_loss_weight=0.5 --value_loss_weight=2.0"
    "--state_loss_weight=1.0 --value_loss_weight=1.0"
)

state_div_options=(
    "--use_state_diversity"
    ""
)

div_weight_options=(
    "--diversity_weight=0.5"
    "--diversity_weight=1.0"
)

# Calculate indices based on task_id
# With 32 jobs and 64 combinations, each job handles 2 combinations
# First combination: 2*(task_id-1)
# Second combination: 2*(task_id-1)+1
base_idx=$((2 * (task_id - 1)))

# Function to get parameter index from combination index
get_param_index() {
    local combo_idx=$1
    local param_pos=$2
    echo $(( (combo_idx >> param_pos) & 1 ))
}

# Function to run a single experiment
run_experiment() {
    local combo_idx=$1
    local repeat_id=$2
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    
    # Extract parameter indices for this combination
    ema_decay_idx=$(get_param_index $combo_idx 0)
    temp_idx=$(get_param_index $combo_idx 1)
    entropy_idx=$(get_param_index $combo_idx 2)
    loss_weights_idx=$(get_param_index $combo_idx 3)
    state_div_idx=$(get_param_index $combo_idx 4)
    div_weight_idx=$(get_param_index $combo_idx 5)
    
    # Select parameters
    ema_decay="${ema_decay_options[$ema_decay_idx]}"
    temp="${temp_options[$temp_idx]}"
    entropy="${entropy_options[$entropy_idx]}"
    loss_weights="${loss_weights_options[$loss_weights_idx]}"
    state_div="${state_div_options[$state_div_idx]}"
    div_weight="${div_weight_options[$div_weight_idx]}"
    
    # Only use div_weight if state_div is enabled
    div_weight_param=""
    if [ ! -z "$state_div" ]; then
        div_weight_param="$div_weight"
    fi
    
    # Create a descriptive run ID
    local run_id="sd_ema${ema_decay_idx}_t${temp_idx}_e${entropy_idx}_l${loss_weights_idx}_sd${state_div_idx}_dw${div_weight_idx}_r${repeat_id}_${timestamp}"
    
    echo "Running configuration (repeat ${repeat_id}): $run_id"
    echo "Database: $db_path"
    echo "Predictor (fixed): $predictor"
    echo "Hidden dim (fixed): $hidden_dim"
    echo "Batch size (fixed): $batch_size"
    echo "Learning rate (fixed): $lr"
    echo "EMA decay: $ema_decay"
    echo "Temperature: $temp"
    echo "Entropy: $entropy"
    echo "Loss weights: $loss_weights"
    echo "State diversity: $state_div"
    echo "Diversity weight: $div_weight_param"
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
      $temp \
      $loss_weights \
      $warmup \
      $gumbel \
      $target_encoder \
      $state_div \
      $div_weight_param \
      $entropy
    
    echo "Experiment ${repeat_id} finished at $(date)"
}

# Run first combination
run_experiment $base_idx 0
run_experiment $base_idx 1

# Run second combination
second_idx=$((base_idx + 1))
run_experiment $second_idx 0
run_experiment $second_idx 1

echo "Job finished at $(date)"