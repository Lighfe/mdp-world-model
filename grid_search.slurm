#!/bin/bash
#SBATCH --job-name=drm_grid
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --output=log/drm_grid_%A_%a.out
#SBATCH --error=log/drm_grid_%A_%a.err
#SBATCH --array=1-36

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
base_output_dir="neural_networks/output/value_function_exp"
val_size=2000
test_size=2000
batch_size=64
epochs=100
patience=100
num_states=4
hidden_dim=64
predictor_type="bilinear"

# Fixed parameters
state_loss_weight=0.5
lr=5e-5
min_lr=5e-6
use_lr_scheduler="--use_lr_scheduler --scheduler_type=cosine --min_lr=$min_lr"
use_warmup="--use_warmup --warmup_epochs=10"  # Changed warmup from 5 to 10
use_target_encoder="--use_target_encoder --ema_decay=0.9"
use_entropy="--use_entropy_reg --entropy_weight=3.0 --use_entropy_decay --entropy_decay_proportion=0.1"

# Grid search parameters
value_methods=("market_share" "90% market share" "identity")
value_loss_weights=(0.5 1.0 2.0)
state_loss_types=("kl_div" "js_div")
gumbel_temps=(5.0 2.0)  # Added new dimension for Gumbel initial temp

# Calculate which combination to use based on array task ID
# Array task ID ranges from 1 to 36
# We have 3 value methods × 3 value loss weights × 2 state loss types × 2 gumbel temps = 36 combinations

# Subtract 1 to make it zero-indexed
idx=$((SLURM_ARRAY_TASK_ID - 1))

# Calculate indices for each parameter
gumbel_temp_idx=$((idx % 2))
idx=$((idx / 2))
state_loss_type_idx=$((idx % 2))
idx=$((idx / 2))
value_loss_weight_idx=$((idx % 3))
value_method_idx=$((idx / 3))

# Get parameter values
value_method="${value_methods[$value_method_idx]}"
value_loss_weight="${value_loss_weights[$value_loss_weight_idx]}"
state_loss_type="${state_loss_types[$state_loss_type_idx]}"
gumbel_temp="${gumbel_temps[$gumbel_temp_idx]}"

# Create descriptive experiment name
experiment_name="vm_${value_method// /_}_vlw_${value_loss_weight}_slt_${state_loss_type}_gumbel_${gumbel_temp}"

# Create one output directory for both runs
output_dir="${base_output_dir}/${experiment_name}"
mkdir -p "$output_dir"

# Run the same parameter combination twice sequentially
for repeat in {1..2}; do
    echo "Starting run $repeat of 2 for experiment: $experiment_name"
    
    # Generate timestamp for this run
    timestamp=$(date +%Y%m%d_%H%M%S)
    run_id="${experiment_name}_rep${repeat}_${timestamp}"
    
    echo "Run $repeat configuration:"
    echo "- Database: $db_path"
    echo "- Value method: $value_method"
    echo "- Value loss weight: $value_loss_weight"
    echo "- State loss type: $state_loss_type"
    echo "- State loss weight: $state_loss_weight"
    echo "- Gumbel initial temp: $gumbel_temp"
    echo "- Learning rate: $lr with cosine decay to $min_lr"
    echo "- Output directory: $output_dir"
    
    # Construct the command with Gumbel softmax
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
      --use_gumbel --initial_temp=${gumbel_temp} --min_temp=0.1 \
      $use_lr_scheduler \
      $use_warmup \
      $use_target_encoder \
      $use_entropy"
    
    echo "Running command for repetition $repeat: $cmd"
    eval $cmd
    
    echo "Completed run $repeat of 2 at $(date)"
    
    # Small delay between runs
    sleep 5
done

echo "Job finished at $(date)"