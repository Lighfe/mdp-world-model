#!/bin/bash
#SBATCH --job-name=drm_grid
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --output=log/drm_grid_%A_%a.out
#SBATCH --error=log/drm_grid_%A_%a.err
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

# Fixed configuration parameters for all runs
db_path="datasets/results/tech_sim.db"
base_output_dir="neural_networks/output/target_deterministic"
val_size=2000
test_size=2000
batch_size=64
patience=200
num_states=4
hidden_dim=64
predictor_type="bilinear"

# Fixed parameters
state_loss_weight=0.5
lr=5e-5
min_lr=5e-6
use_lr_scheduler="--use_lr_scheduler --scheduler_type=cosine --min_lr=$min_lr"
use_warmup="--use_warmup --warmup_epochs=10"
use_target_encoder="--use_target_encoder --ema_decay=0.9"
use_entropy="--use_entropy_reg --entropy_weight=3.0 --use_entropy_decay --entropy_decay_proportion=0.1"

# Grid search parameters
state_loss_types=("kl_div" "js_div")
gumbel_min_temps=(0.1 0.5)
value_loss_weights=(1.0 2.0)
value_methods=("market_share" "identity")
epoch_values=(100 200)

# Calculate which combination to use based on array task ID
# Array task ID ranges from 1 to 32
# We have 2 state loss types × 2 gumbel min temps × 2 value loss weights × 2 value methods × 2 epoch values = 32 combinations

# Subtract 1 to make it zero-indexed
idx=$((SLURM_ARRAY_TASK_ID - 1))

# Calculate indices for each parameter using integer division and modulo
state_loss_type_idx=$((idx / 16))
remainder=$((idx % 16))
gumbel_min_temp_idx=$((remainder / 8))
remainder=$((remainder % 8))
value_loss_weight_idx=$((remainder / 4))
remainder=$((remainder % 4))
value_method_idx=$((remainder / 2))
epoch_idx=$((remainder % 2))

# Get parameter values
state_loss_type="${state_loss_types[$state_loss_type_idx]}"
gumbel_min_temp="${gumbel_min_temps[$gumbel_min_temp_idx]}"
value_loss_weight="${value_loss_weights[$value_loss_weight_idx]}"
value_method="${value_methods[$value_method_idx]}"
epochs="${epoch_values[$epoch_idx]}"

# Set up gumbel parameters
gumbel_params="--use_gumbel --initial_temp=5.0 --min_temp=$gumbel_min_temp"

# Create descriptive experiment name
experiment_name="slt_${state_loss_type}_gmt_${gumbel_min_temp}_vlw_${value_loss_weight}_vm_${value_method//\ /_}_ep_${epochs}"

# Set output directory 
output_dir="${base_output_dir}/${experiment_name}"
mkdir -p "$output_dir"

# Run the same parameter combination twice sequentially
for repeat in {1..2}; do
    echo "Starting run $repeat of 2 for experiment: $experiment_name"
    
    # Generate timestamp for this run
    timestamp=$(date +%Y%m%d_%H%M%S)
    run_id="${experiment_name}_rep${repeat}_${timestamp}"
    
    echo "Configuration (repetition $repeat):"
    echo "- Database: $db_path"
    echo "- State loss type: $state_loss_type"
    echo "- Gumbel min temp: $gumbel_min_temp"
    echo "- Value loss weight: $value_loss_weight"
    echo "- Value method: $value_method"
    echo "- Epochs: $epochs"
    echo "- Output directory: $output_dir"
    echo "- Run ID: $run_id"
    
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
      --value_method='$value_method' \
      --state_loss_type=$state_loss_type \
      --lr=$lr \
      $use_lr_scheduler \
      $use_warmup \
      $use_target_encoder \
      $use_entropy \
      $gumbel_params"
    
    echo "Running command for repetition $repeat: $cmd"
    eval $cmd
    
    echo "Completed run $repeat of 2 at $(date)"
    
    # Small delay between runs
    sleep 5
done

echo "Job finished at $(date)"