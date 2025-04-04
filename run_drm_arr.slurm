#!/bin/bash
#SBATCH --job-name=drm_500ep
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=log/drm_500ep_%A_%a.out
#SBATCH --error=log/drm_500ep_%A_%a.err
#SBATCH --array=1-9

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

# Fixed configuration parameters for all runs
db_path="datasets/results/tech_sim.db"
base_output_dir="neural_networks/output/500epochs"
val_size=2000
test_size=2000
batch_size=64
epochs=500
patience=500
num_states=4
hidden_dim=64
predictor_type="bilinear"
value_method="market_share"

# Fixed parameters
state_loss_weight=1.0
value_loss_weight=2.0
lr=5e-5
min_lr=5e-6
use_lr_scheduler="--use_lr_scheduler --scheduler_type=cosine --min_lr=$min_lr"
use_warmup="--use_warmup --warmup_epochs=20"
use_target_encoder="--use_target_encoder --ema_decay=0.9"
use_entropy="--use_entropy_reg --entropy_weight=3.0 --use_entropy_decay --entropy_decay_proportion=0.05"

# Calculate experiment type and repetition from array task ID
# Tasks 1-3: min_temp=0.1 (repeats 0,1,2)
# Tasks 4-6: min_temp=0.01 (repeats 0,1,2)
# Tasks 7-9: min_temp=0.001 (repeats 0,1,2)
exp_type=$(( (SLURM_ARRAY_TASK_ID-1) / 3 ))
repeat_id=$(( (SLURM_ARRAY_TASK_ID-1) % 3 ))

# Define Gumbel temperature options
gumbel_options=(
    "--use_gumbel --initial_temp=5.0 --min_temp=0.1"
    "--use_gumbel --initial_temp=5.0 --min_temp=0.01"
    "--use_gumbel --initial_temp=5.0 --min_temp=0.001"
)

# Set Gumbel option based on experiment type
use_gumbel="${gumbel_options[$exp_type]}"

# Create descriptive experiment name
gumbel_names=("temp0.1" "temp0.01" "temp0.001")
experiment_name="gumbel_${gumbel_names[$exp_type]}"

# Generate timestamp for this run
timestamp=$(date +%Y%m%d_%H%M%S)
run_id="${experiment_name}_r${repeat_id}_${timestamp}"

# Set output directory
output_dir="${base_output_dir}/${gumbel_names[$exp_type]}"
mkdir -p "$output_dir"

echo "Running experiment: $experiment_name (repeat $repeat_id)"
echo "Configuration:"
echo "- Database: $db_path"
echo "- State loss weight: $state_loss_weight"
echo "- Value loss weight: $value_loss_weight"
echo "- Learning rate: $lr with cosine decay to $min_lr"
echo "- Gumbel settings: $use_gumbel"
echo "- Entropy settings: $use_entropy"
echo "- Warmup: 20 epochs"
echo "- Training: 500 epochs, patience 500"

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
  $use_lr_scheduler \
  $use_warmup \
  $use_gumbel \
  $use_target_encoder \
  $use_entropy"

echo "Running command: $cmd"
eval $cmd

echo "Job finished at $(date)"