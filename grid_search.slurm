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
#SBATCH --array=1-50

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

# Define database (fixed)
db_path="datasets/results/tech_angular_lt.db"

# Define predictor options (2 options)
predictor_options=(
    "bilinear"
    "standard"
)

# Define value method options (2 options)
value_method_options=(
    "market_share"
    "90% market share"
)

# Learning rate strategies (3 options)
lr_strategies=(
    ""
    "--use_warmup --warmup_epochs=5"
    "--use_warmup --warmup_epochs=5 --use_lr_scheduler --scheduler_type=cosine --min_lr=1e-5"
)

# Gumbel softmax options (2 options)
gumbel_options=(
    ""
    "--use_gumbel --initial_temp=5.0 --min_temp=0.5"
)

# Entropy loss options (3 options)
entropy_options=(
    ""
    "--use_entropy_reg --entropy_weight=3.0"
    "--use_entropy_reg --entropy_weight=10.0"
)

# Fixed parameters
batch_size="64"
hidden_dim="128"
loss_weights="--state_loss_weight=1.0 --value_loss_weight=1.0"

# REVISED: Calculate indices - prioritizing value_method and entropy
# Cycle order from fastest to slowest: value_method -> entropy -> predictor -> lr_strategy -> gumbel
value_method_idx=$(( (task_id-1) % 2 ))         # Changes every 1 job (fastest)
entropy_idx=$(( (task_id-1) / 2 % 3 ))          # Changes every 2 jobs
gumbel_idx=$(( (task_id-1) / 6 % 2 ))          # Changes every 6 jobs
lr_strategy_idx=$(( (task_id-1) / 12 % 3 ))     # Changes every 12 jobs
predictor_idx=$(( (task_id-1) / 36 % 2 ))        # Changes every 6 jobs

# Select parameters
predictor="${predictor_options[$predictor_idx]}"
value_method="${value_method_options[$value_method_idx]}"
lr_strategy="${lr_strategies[$lr_strategy_idx]}"
gumbel="${gumbel_options[$gumbel_idx]}"
entropy="${entropy_options[$entropy_idx]}"

# Function to run a single experiment
run_experiment() {
    local repeat_id=$1
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    
    # Create a descriptive run ID
    local run_id="grid2_v${value_method_idx}_e${entropy_idx}_p${predictor_idx}_lr${lr_strategy_idx}_g${gumbel_idx}_r${repeat_id}_${timestamp}"
    
    echo "Running configuration (repeat ${repeat_id}): $run_id"
    echo "Database: $db_path"
    echo "Value method: $value_method"
    echo "Entropy: $entropy"
    echo "Predictor: $predictor"
    echo "Learning rate strategy: $lr_strategy"
    echo "Gumbel: $gumbel"
    
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
      $gumbel \
      $entropy
    
    echo "Experiment ${repeat_id} finished at $(date)"
}

# Run two identical experiments sequentially
run_experiment 0
run_experiment 1

echo "Job finished at $(date)"