#!/bin/bash
#SBATCH --job-name=drm_grid2
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=log/drm_grid2_%A_%a.out
#SBATCH --error=log/drm_grid2_%A_%a.err
#SBATCH --array=1-54

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

# Define predictor options (2 options)
predictor_options=(
    "standard"
    "bilinear"
)

# 2 options
batch_size_options=(
    "64"
    "128"
)

# 2 options
hidden_dim_options=(
    "64"
    "128"
)

# 3 options
loss_weights_options=(
    "--state_loss_weight=1.0 --value_loss_weight=1.0"
    "--state_loss_weight=1.0 --value_loss_weight=3.0"
    "--state_loss_weight=3.0 --value_loss_weight=1.0"
)

# Learning rate strategies (3 options)
lr_strategies=(
    ""
    "--use_warmup --warmup_epochs=10"
    "--use_warmup --warmup_epochs=10 --use_lr_scheduler --scheduler_type=cosine --min_lr=1e-5"
)

# Entropy loss options (3 options)
entropy_options=(
    ""
    "--use_entropy_reg --entropy_weight=3.0"
    "--use_entropy_reg --entropy_weight=10.0"
)

# Fix
value_method="market_share"
db_path="datasets/results/tech_angular_lt.db"

# Count options
n_predictors=${#predictor_options[@]}
n_batch_sizes=${#batch_size_options[@]}
n_hidden_dims=${#hidden_dim_options[@]}
n_loss_weights=${#loss_weights_options[@]}
n_lr_strategies=${#lr_strategies[@]}
n_entropy_options=${#entropy_options[@]}

# Structured random parameter selection
# Use a hash of the task ID to deterministically select "random" parameters
# This ensures reproducibility while providing good coverage

# Prime numbers for hashing (different primes to avoid correlation)
p1=11
p2=17
p3=23
p4=31
p5=43

# Use the task ID to generate pseudo-random hash
hash_val=$((task_id * p1))

# Primary parameters (more variety)
predictor_idx=$((hash_val % n_predictors))
lr_strategy_idx=$(((hash_val * p2) % n_lr_strategies))
entropy_idx=$(((hash_val * p3) % n_entropy_options))
loss_weights_idx=$(((hash_val * p4) % n_loss_weights))

# Secondary parameters (less variety, change less frequently)
batch_size_idx=$(((hash_val * p5) % n_batch_sizes))
hidden_dim_idx=$(((hash_val * p5 + 7) % n_hidden_dims))

# Select parameters
predictor="${predictor_options[$predictor_idx]}"
batch_size="${batch_size_options[$batch_size_idx]}"
hidden_dim="${hidden_dim_options[$hidden_dim_idx]}"
loss_weights="${loss_weights_options[$loss_weights_idx]}"
lr_strategy="${lr_strategies[$lr_strategy_idx]}"
entropy="${entropy_options[$entropy_idx]}"

# Function to run a single experiment
run_experiment() {
    local repeat_id=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    
    # Create a descriptive run ID
    local run_id="newval_p${predictor_idx}_lr${lr_strategy_idx}_e${entropy_idx}_l${loss_weights_idx}_b${batch_size_idx}_h${hidden_dim_idx}_r${repeat_id}_${timestamp}"
    
    echo "Running configuration (repeat ${repeat_id}): $run_id"
    echo "Database: $db_path"
    echo "Predictor: $predictor"
    echo "Batch size: $batch_size"
    echo "Hidden dim: $hidden_dim"
    echo "Loss weights: $loss_weights"
    echo "Learning rate strategy: $lr_strategy"
    echo "Entropy: $entropy"
    
    # Run the training command
    python -m neural_networks.train_drm $db_path \
      --output_dir=neural_networks/output \
      --run_id="$run_id" \
      --val_size=2000 \
      --test_size=2000 \
      --batch_size=$batch_size \
      --epochs=100 \
      --lr=0.0001 \
      --patience=15 \
      --num_states=4 \
      --hidden_dim=$hidden_dim \
      --predictor_type=$predictor \
      --value_method="$value_method" \
      $loss_weights \
      $lr_strategy \
      $entropy
    
    echo "Experiment ${repeat_id} finished at $(date)"
}

# Run two identical experiments sequentially
run_experiment 0
run_experiment 1

echo "Job finished at $(date)"