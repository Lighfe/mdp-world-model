#!/bin/bash
#SBATCH --job-name=drm_cosine
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=log/drm_cosine_%A_%a.out
#SBATCH --error=log/drm_cosine_%A_%a.err
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

# Base configuration parameters that are constant for all runs
base_output_dir="neural_networks/output/cosine_decay_study"
val_size=2000
test_size=2000
batch_size=64
epochs=200
patience=200
num_states=4
hidden_dim=64
predictor_type="bilinear"
use_target_encoder="--use_target_encoder --ema_decay=0.9"

# Learning rate parameters (same for all runs)
lr=5e-5
min_lr=5e-6
use_lr_scheduler="--use_lr_scheduler --scheduler_type=cosine --min_lr=$min_lr"
use_warmup="--use_warmup --warmup_epochs=10"

# Determine which db, entropy, gumbel, state_loss, and value_loss to use based on task ID
# Task ID from 1-32 will cover all combinations

# Determine combination indices using binary mapping
# Task_ID - 1 = 0-31, which can be represented as 5 binary digits
adjusted_id=$((SLURM_ARRAY_TASK_ID - 1))

# Extract individual feature selections from the binary representation
entropy_idx=$((adjusted_id & 1))           # Bit 0: 0=entropy_3, 1=entropy_5
gumbel_idx=$(((adjusted_id >> 1) & 1))     # Bit 1: 0=min_temp_0.1, 1=min_temp_0.01
db_idx=$(((adjusted_id >> 2) & 1))         # Bit 2: 0=tech_sim.db, 1=tech_angular_lt.db
state_loss_idx=$(((adjusted_id >> 3) & 1)) # Bit 3: 0=state_loss_0.5, 1=state_loss_1.0
value_loss_idx=$(((adjusted_id >> 4) & 1)) # Bit 4: 0=value_loss_1.0, 1=value_loss_2.0

# Define options based on indices
entropy_options=(
    "--use_entropy_reg --entropy_weight=3.0 --use_entropy_decay --entropy_decay_proportion=0.1"
    "--use_entropy_reg --entropy_weight=5.0 --use_entropy_decay --entropy_decay_proportion=0.1"
)
gumbel_options=(
    "--use_gumbel --initial_temp=5.0 --min_temp=0.1"
    "--use_gumbel --initial_temp=5.0 --min_temp=0.01"
)
db_options=(
    "datasets/results/tech_sim.db"
    "datasets/results/tech_angular_lt.db"
)
state_loss_options=(
    "0.5"
    "1.0"
)
value_loss_options=(
    "1.0"
    "2.0"
)

# Set parameters based on task ID
use_entropy="${entropy_options[$entropy_idx]}"
use_gumbel="${gumbel_options[$gumbel_idx]}"
db_path="${db_options[$db_idx]}"
state_loss_weight="${state_loss_options[$state_loss_idx]}"
value_loss_weight="${value_loss_options[$value_loss_idx]}"

# Create descriptive subfolder and experiment name
entropy_name="ent${entropy_idx}"
gumbel_name="gmbl${gumbel_idx}"
db_name="db${db_idx}"
state_name="st${state_loss_idx}"
value_name="val${value_loss_idx}"

# Create folder structure
experiment_name="${entropy_name}_${gumbel_name}_${db_name}_${state_name}_${value_name}"
output_subfolder="${db_name}/${entropy_name}_${gumbel_name}"

# Function to run experiment twice
run_experiment() {
    local experiment_name=$1
    local output_subfolder=$2
    local config=$3
    
    # Create output directory with subfolder
    output_dir="${base_output_dir}/${output_subfolder}"
    mkdir -p "$output_dir"
    
    # Generate timestamp
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Run the experiment twice with the same parameters
    for repeat_id in 0 1; do
        local run_id="${experiment_name}_r${repeat_id}_${timestamp}"
        
        echo "Running experiment: $experiment_name (repeat $repeat_id)"
        echo "Configuration: $config"
        echo "Database: $db_path"
        echo "State loss weight: $state_loss_weight"
        echo "Value loss weight: $value_loss_weight"
        echo "Gumbel settings: $use_gumbel"
        echo "Entropy settings: $use_entropy"
        echo "Learning rate: $lr with cosine decay to $min_lr"
        echo "Warmup: 10 epochs"
        
        # Construct the command
        cmd="python -m neural_networks.train_drm $db_path \
          --output_dir=$output_dir \
          --run_id=$run_id \
          --val_size=$val_size \
          --test_size=$test_size \
          --batch_size=$batch_size \
          --epochs=$epochs \
          --patience=$patience \
          --num_states=$num_states \
          --hidden_dim=$hidden_dim \
          --state_loss_weight=$state_loss_weight \
          --value_loss_weight=$value_loss_weight \
          --predictor_type=$predictor_type \
          --value_method=market_share \
          --lr=$lr \
          $use_lr_scheduler \
          $use_warmup \
          $use_gumbel \
          $use_target_encoder \
          $use_entropy"
        
        echo "Running command: $cmd"
        eval $cmd
        
        echo "Completed repeat $repeat_id at $(date)"
    done
}

# Get text descriptions of parameter combinations for logging
entropy_desc=("entropy_weight=3.0" "entropy_weight=5.0")
gumbel_desc=("min_temp=0.1" "min_temp=0.01")
db_desc=("tech_sim.db" "tech_angular_lt.db")
state_loss_desc=("state_loss=0.5" "state_loss=1.0")
value_loss_desc=("value_loss=1.0" "value_loss=2.0")

# Combine descriptions
config_description="${db_desc[$db_idx]}, ${entropy_desc[$entropy_idx]}, ${gumbel_desc[$gumbel_idx]}, ${state_loss_desc[$state_loss_idx]}, ${value_loss_desc[$value_loss_idx]}"

# Run the experiment
run_experiment "$experiment_name" "$output_subfolder" "$config_description"

echo "Job finished at $(date)"