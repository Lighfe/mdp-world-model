#!/bin/bash
#SBATCH --job-name=drm_test
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=log/drm_test_%j.out
#SBATCH --error=log/drm_test_%j.err

# Print job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

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

# Base configuration
db_path="datasets/results/tech_angular_lt.db"
base_output_dir="neural_networks/output/test_run"
val_size=2000
test_size=2000
batch_size=64
epochs=100
patience=100
num_states=4
hidden_dim=64
predictor_type="bilinear"
value_method="market_share"

# Default parameters
state_loss_weight=0.5
value_loss_weight=2.0
lr=1e-5
use_warmup="--use_warmup --warmup_epochs=5"
use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
use_target_encoder="--use_target_encoder --ema_decay=0.9"

# Set up state diversity and entropy for our test
use_diversity="--use_state_diversity --diversity_weight=0.5"
use_entropy="--use_entropy_reg --entropy_weight=3.0 --use_entropy_decay --entropy_decay_proportion=0.2"

# Function to run experiment three times
run_experiment() {
    local experiment_name=$1
    local output_subfolder=$2
    
    # Create output directory with subfolder
    output_dir="${base_output_dir}/${output_subfolder}"
    mkdir -p "$output_dir"
    
    # Generate timestamp
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Run the experiment three times with the same parameters
    for repeat_id in 0 1 2; do
        local run_id="${experiment_name}_r${repeat_id}_${timestamp}"
        
        echo "Running experiment: $experiment_name (repeat $repeat_id)"
        echo "Configuration:"
        echo "- State loss weight: $state_loss_weight"
        echo "- Value loss weight: $value_loss_weight"
        echo "- Learning rate: $lr"
        echo "- Warmup: $use_warmup"
        echo "- Gumbel: $use_gumbel"
        echo "- Target encoder: $use_target_encoder"
        echo "- Diversity: $use_diversity"
        echo "- Entropy: $use_entropy"
        
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
          --value_method=$value_method \
          --lr=$lr \
          $use_warmup \
          $use_gumbel \
          $use_target_encoder \
          $use_diversity \
          $use_entropy"
        
        echo "Running command: $cmd"
        eval $cmd
        
        echo "Completed repeat $repeat_id at $(date)"
    done
}

# Run a single test with state diversity and entropy
run_experiment "test_div_ent" "single_test" 

echo "Job finished at $(date)"