#!/bin/bash
#SBATCH --job-name=drm_grid
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=log/drm_grid_%A_%a.out
#SBATCH --error=log/drm_grid_%A_%a.err
#SBATCH --array=1-200

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

# Define database options (2 options)
db_options=(
    "tech_angular_xtr.db"
    "tech_angular_lt.db"
)

# Define predictor options (3 options)
predictor_options=(
    "standard"
    "bilinear"
    "control_gate"
)

# Define value method options (2 options)
value_method_options=(
    "market_share"
    "90% market_share"
)

# Loss weight combinations (2 options)
loss_weights=(
    "--state_loss_weight=1.0 --value_loss_weight=3.0"
    "--state_loss_weight=3.0 --value_loss_weight=1.0"
)

# Learning rate strategies (2 options)
lr_strategies=(
    ""
    "--use_warmup --warmup_epochs=5 --use_lr_scheduler --scheduler_type=cosine --min_lr=1e-5"
)

# Gumbel softmax options (2 options)
gumbel_options=(
    ""
    "--use_gumbel --initial_temp=5.0 --min_temp=0.5"
)

# Batch size options (2 options)
batch_size_options=(
    "64"
    "128"
)

# Hidden dim options (2 options)
hidden_dim_options=(
    "32"
    "64"
)

# Entropy loss options (3 options)
entropy_options=(
    ""
    "--use_entropy_reg --entropy_weight=3.0"
    "--use_entropy_reg --entropy_weight=10.0"
)

# Reduce array task ID to base configuration ID (1-50) and database selection (0-1)
base_id=$(( (task_id-1) % 50 + 1 ))  # Cycles through 1-50
db_idx=$(( (task_id-1) / 100 ))       # Swoitchs at 100
repeat_id=$(( (task_id-1) / 50 % 2 )) # 0 for first run, 1 for second run
db_path="datasets/results/${db_options[$db_idx]}"

# Map base_id to parameter selections using modular arithmetic to ensure good coverage
predictor_idx=$(( (base_id-1) % 3 ))
value_method_idx=$(( (base_id-1) / 3 % 2 ))
loss_weight_idx=$(( (base_id-1) / 6 % 2 ))
lr_strategy_idx=$(( (base_id-1) / 12 % 2 ))
gumbel_idx=$(( (base_id-1) / 24 % 2 ))
entropy_idx=$(( (base_id-1) / 48 % 3 ))

# Secondary parameters - cycle more frequently for better distribution
batch_idx=$(( (base_id-1) % 2 ))
hidden_idx=$(( (base_id-1) / 2 % 2 ))

# Prioritize promising combinations - adjust predictor distribution
# Bilinear and Standard are favored over ControlGate
if [ $predictor_idx -eq 2 ] && [ $(( base_id % 5 )) -ne 0 ]; then
    # Replace ControlGate with Bilinear 80% of the time
    predictor_idx=1
fi

# Select parameters
predictor="${predictor_options[$predictor_idx]}"
value_method="${value_method_options[$value_method_idx]}"
loss_weight="${loss_weights[$loss_weight_idx]}"
lr_strategy="${lr_strategies[$lr_strategy_idx]}"
gumbel="${gumbel_options[$gumbel_idx]}"
batch_size="${batch_size_options[$batch_idx]}"
hidden_dim="${hidden_dim_options[$hidden_idx]}"
entropy="${entropy_options[$entropy_idx]}"

# Create a descriptive run ID
run_id="grid_${task_id}_p${predictor_idx}_v${value_method_idx}_l${loss_weight_idx}_lr${lr_strategy_idx}_g${gumbel_idx}_b${batch_idx}_h${hidden_idx}_e${entropy_idx}_db${db_idx}_r${repeat_id}"

echo "Running configuration: $run_id"
echo "Database: $db_path"
echo "Predictor: $predictor"
echo "Value method: $value_method"
echo "Loss weights: $loss_weight"
echo "Learning rate strategy: $lr_strategy"
echo "Gumbel: $gumbel"
echo "Batch size: $batch_size"
echo "Hidden dim: $hidden_dim"
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
  $loss_weight \
  $lr_strategy \
  $gumbel \
  $entropy

echo "Job finished at $(date)"