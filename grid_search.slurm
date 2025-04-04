#!/bin/bash
#SBATCH --job-name=drm_state_loss
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=log/drm_state_loss_%A_%a.out
#SBATCH --error=log/drm_state_loss_%A_%a.err
#SBATCH --array=1-16

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
base_output_dir="neural_networks/output/state_loss_exp"
val_size=2000
test_size=2000
batch_size=64
epochs=100
patience=100
num_states=4
hidden_dim=64
predictor_type="bilinear"
value_method="market_share"

# Fixed parameters
value_loss_weight=2.0
lr=5e-5
min_lr=5e-6
use_lr_scheduler="--use_lr_scheduler --scheduler_type=cosine --min_lr=$min_lr"
use_warmup="--use_warmup --warmup_epochs=5"
use_target_encoder="--use_target_encoder --ema_decay=0.9"
use_entropy="--use_entropy_reg --entropy_weight=3.0 --use_entropy_decay --entropy_decay_proportion=0.1"
use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.1"

# Define state loss types and weights
state_loss_types=("kl_div" "mse" "js_div" "cross_entropy")
state_weights=(0.5 1.0 2.0 3.0)

# Map array task ID to configuration
# We have 16 unique configurations (4 loss types × 4 weights)
loss_idx=$(( (SLURM_ARRAY_TASK_ID-1) / 4 ))
weight_idx=$(( (SLURM_ARRAY_TASK_ID-1) % 4 ))

loss_type=${state_loss_types[$loss_idx]}
weight=${state_weights[$weight_idx]}

echo "This job will run configuration: loss_type=$loss_type, weight=$weight"

# Create output directory
output_dir="${base_output_dir}/${loss_type}/weight${weight}"
mkdir -p "$output_dir"

# Run the job twice sequentially
for run in {1..2}; do
    # Generate timestamp for this run
    timestamp=$(date +%Y%m%d_%H%M%S)
    run_id="${loss_type}_weight${weight}_run${run}_${timestamp}"
    
    echo "-------------------------------------------"
    echo "Starting job: $run_id at $(date)"
    echo "Configuration:"
    echo "- State loss type: $loss_type"
    echo "- State weight: $weight"
    echo "- Run: $run of 2"
    
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
      --state_loss_weight=$weight \
      --value_loss_weight=$value_loss_weight \
      --predictor_type=$predictor_type \
      --value_method=$value_method \
      --state_loss_type=$loss_type \
      --lr=$lr \
      $use_lr_scheduler \
      $use_warmup \
      $use_gumbel \
      $use_target_encoder \
      $use_entropy"
    
    echo "Running command: $cmd"
    eval $cmd
    
    echo "Completed run at $(date)"
    echo "-------------------------------------------"
done

echo "Job finished at $(date)"