#!/bin/bash
#SBATCH --job-name=drm_entropy
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=log/drm_entropy_%A_%a.out
#SBATCH --error=log/drm_entropy_%A_%a.err
#SBATCH --array=1-28

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

# Base configuration
db_path="datasets/results/tech_angular_lt.db"
base_output_dir="neural_networks/output/entropy_study"
val_size=2000
test_size=2000
batch_size=64
epochs=100
patience=100
num_states=4
hidden_dim=64
predictor_type="bilinear"
value_method="market_share"

# Default parameters for baseline
state_loss_weight=0.5
value_loss_weight=2.0
lr=1e-5
use_warmup="--use_warmup --warmup_epochs=5"
use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
use_target_encoder="--use_target_encoder --ema_decay=0.9"
use_diversity=""

# Define entropy options
entropy_none=""
entropy_decay_0_2="--use_entropy_reg --entropy_weight=3.0 --use_entropy_decay --entropy_decay_proportion=0.2"
entropy_decay_0_5="--use_entropy_reg --entropy_weight=10.0 --use_entropy_decay --entropy_decay_proportion=0.5"
entropy_no_decay="--use_entropy_reg --entropy_weight=3.0 --no_entropy_decay"

# Default to no entropy
use_entropy="$entropy_none"

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

# Map task ID to experiment combination
case $SLURM_ARRAY_TASK_ID in
    # BASELINE WITH DIFFERENT ENTROPY OPTIONS
    1)  # BASELINE - NO ENTROPY
        state_loss_weight=0.5
        value_loss_weight=2.0
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_diversity=""
        use_entropy="$entropy_none"
        run_experiment "baseline_no_ent" "baseline" "Baseline, no entropy"
        ;;
    2)  # BASELINE - ENTROPY WITH DECAY 0.2
        state_loss_weight=0.5
        value_loss_weight=2.0
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_diversity=""
        use_entropy="$entropy_decay_0_2"
        run_experiment "baseline_ent_0_2" "baseline" "Baseline, entropy weight 3, decay 0.2"
        ;;
    3)  # BASELINE - ENTROPY WITH DECAY 0.5
        state_loss_weight=0.5
        value_loss_weight=2.0
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_diversity=""
        use_entropy="$entropy_decay_0_5"
        run_experiment "baseline_ent_0_5" "baseline" "Baseline, entropy weight 10, decay 0.5"
        ;;
    4)  # BASELINE - ENTROPY NO DECAY
        state_loss_weight=0.5
        value_loss_weight=2.0
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_diversity=""
        use_entropy="$entropy_no_decay"
        run_experiment "baseline_ent_no_decay" "baseline" "Baseline, entropy weight 3, no decay"
        ;;
        
    # NO GUMBEL + DIV 0.001 WITH DIFFERENT ENTROPY OPTIONS
    5)  # NO GUMBEL + DIV 0.001 - NO ENTROPY
        use_gumbel=""
        use_diversity="--use_state_diversity --diversity_weight=0.001"
        use_entropy="$entropy_none"
        run_experiment "no_gumbel_div001_no_ent" "gumbel_div" "No gumbel, div 0.001, no entropy"
        ;;
    6)  # NO GUMBEL + DIV 0.001 - ENTROPY DECAY 0.2
        use_gumbel=""
        use_diversity="--use_state_diversity --diversity_weight=0.001"
        use_entropy="$entropy_decay_0_2"
        run_experiment "no_gumbel_div001_ent_0_2" "gumbel_div" "No gumbel, div 0.001, entropy decay 0.2"
        ;;
    7)  # NO GUMBEL + DIV 0.001 - ENTROPY DECAY 0.5
        use_gumbel=""
        use_diversity="--use_state_diversity --diversity_weight=0.001"
        use_entropy="$entropy_decay_0_5"
        run_experiment "no_gumbel_div001_ent_0_5" "gumbel_div" "No gumbel, div 0.001, entropy decay 0.5"
        ;;
    8)  # NO GUMBEL + DIV 0.001 - ENTROPY NO DECAY
        use_gumbel=""
        use_diversity="--use_state_diversity --diversity_weight=0.001"
        use_entropy="$entropy_no_decay"
        run_experiment "no_gumbel_div001_ent_no_decay" "gumbel_div" "No gumbel, div 0.001, entropy no decay"
        ;;
        
    # NO GUMBEL + DIV 0.01 WITH DIFFERENT ENTROPY OPTIONS
    9)  # NO GUMBEL + DIV 0.01 - NO ENTROPY
        use_gumbel=""
        use_diversity="--use_state_diversity --diversity_weight=0.01"
        use_entropy="$entropy_none"
        run_experiment "no_gumbel_div01_no_ent" "gumbel_div" "No gumbel, div 0.01, no entropy"
        ;;
    10) # NO GUMBEL + DIV 0.01 - ENTROPY DECAY 0.2
        use_gumbel=""
        use_diversity="--use_state_diversity --diversity_weight=0.01"
        use_entropy="$entropy_decay_0_2"
        run_experiment "no_gumbel_div01_ent_0_2" "gumbel_div" "No gumbel, div 0.01, entropy decay 0.2"
        ;;
    11) # NO GUMBEL + DIV 0.01 - ENTROPY DECAY 0.5
        use_gumbel=""
        use_diversity="--use_state_diversity --diversity_weight=0.01"
        use_entropy="$entropy_decay_0_5"
        run_experiment "no_gumbel_div01_ent_0_5" "gumbel_div" "No gumbel, div 0.01, entropy decay 0.5"
        ;;
    12) # NO GUMBEL + DIV 0.01 - ENTROPY NO DECAY
        use_gumbel=""
        use_diversity="--use_state_diversity --diversity_weight=0.01"
        use_entropy="$entropy_no_decay"
        run_experiment "no_gumbel_div01_ent_no_decay" "gumbel_div" "No gumbel, div 0.01, entropy no decay"
        ;;
        
    # GUMBEL + STATE LOSS 3, VALUE LOSS 1 WITH DIFFERENT ENTROPY OPTIONS
    13) # GUMBEL + STATE 3 VALUE 1 - NO ENTROPY
        state_loss_weight=3.0
        value_loss_weight=1.0
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_diversity=""
        use_entropy="$entropy_none"
        run_experiment "gumbel_state3_no_ent" "state_loss" "Gumbel, state 3 value 1, no entropy"
        ;;
    14) # GUMBEL + STATE 3 VALUE 1 - ENTROPY DECAY 0.2
        state_loss_weight=3.0
        value_loss_weight=1.0
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_diversity=""
        use_entropy="$entropy_decay_0_2"
        run_experiment "gumbel_state3_ent_0_2" "state_loss" "Gumbel, state 3 value 1, entropy decay 0.2"
        ;;
    15) # GUMBEL + STATE 3 VALUE 1 - ENTROPY DECAY 0.5
        state_loss_weight=3.0
        value_loss_weight=1.0
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_diversity=""
        use_entropy="$entropy_decay_0_5"
        run_experiment "gumbel_state3_ent_0_5" "state_loss" "Gumbel, state 3 value 1, entropy decay 0.5"
        ;;
    16) # GUMBEL + STATE 3 VALUE 1 - ENTROPY NO DECAY
        state_loss_weight=3.0
        value_loss_weight=1.0
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_diversity=""
        use_entropy="$entropy_no_decay"
        run_experiment "gumbel_state3_ent_no_decay" "state_loss" "Gumbel, state 3 value 1, entropy no decay"
        ;;
        
    # NO GUMBEL + STATE LOSS 3, VALUE LOSS 1 WITH DIFFERENT ENTROPY OPTIONS
    17) # NO GUMBEL + STATE 3 VALUE 1 - NO ENTROPY
        state_loss_weight=3.0
        value_loss_weight=1.0
        use_gumbel=""
        use_diversity=""
        use_entropy="$entropy_none"
        run_experiment "no_gumbel_state3_no_ent" "state_loss" "No gumbel, state 3 value 1, no entropy"
        ;;
    18) # NO GUMBEL + STATE 3 VALUE 1 - ENTROPY DECAY 0.2
        state_loss_weight=3.0
        value_loss_weight=1.0
        use_gumbel=""
        use_diversity=""
        use_entropy="$entropy_decay_0_2"
        run_experiment "no_gumbel_state3_ent_0_2" "state_loss" "No gumbel, state 3 value 1, entropy decay 0.2"
        ;;
    19) # NO GUMBEL + STATE 3 VALUE 1 - ENTROPY DECAY 0.5
        state_loss_weight=3.0
        value_loss_weight=1.0
        use_gumbel=""
        use_diversity=""
        use_entropy="$entropy_decay_0_5"
        run_experiment "no_gumbel_state3_ent_0_5" "state_loss" "No gumbel, state 3 value 1, entropy decay 0.5"
        ;;
    20) # NO GUMBEL + STATE 3 VALUE 1 - ENTROPY NO DECAY
        state_loss_weight=3.0
        value_loss_weight=1.0
        use_gumbel=""
        use_diversity=""
        use_entropy="$entropy_no_decay"
        run_experiment "no_gumbel_state3_ent_no_decay" "state_loss" "No gumbel, state 3 value 1, entropy no decay"
        ;;
        
    # ADDITIONAL EXPERIMENTS WITH STATE LOSS 4.0
    21) # GUMBEL + STATE 4 VALUE 1 - NO ENTROPY
        state_loss_weight=4.0
        value_loss_weight=1.0
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_diversity=""
        use_entropy="$entropy_none"
        run_experiment "gumbel_state4_no_ent" "state_loss_high" "Gumbel, state 4 value 1, no entropy"
        ;;
    22) # GUMBEL + STATE 4 VALUE 1 - ENTROPY DECAY 0.2
        state_loss_weight=4.0
        value_loss_weight=1.0
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_diversity=""
        use_entropy="$entropy_decay_0_2"
        run_experiment "gumbel_state4_ent_0_2" "state_loss_high" "Gumbel, state 4 value 1, entropy decay 0.2"
        ;;
    23) # GUMBEL + STATE 4 VALUE 1 - ENTROPY DECAY 0.5
        state_loss_weight=4.0
        value_loss_weight=1.0
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_diversity=""
        use_entropy="$entropy_decay_0_5"
        run_experiment "gumbel_state4_ent_0_5" "state_loss_high" "Gumbel, state 4 value 1, entropy decay 0.5"
        ;;
    24) # GUMBEL + STATE 4 VALUE 1 - ENTROPY NO DECAY
        state_loss_weight=4.0
        value_loss_weight=1.0
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_diversity=""
        use_entropy="$entropy_no_decay"
        run_experiment "gumbel_state4_ent_no_decay" "state_loss_high" "Gumbel, state 4 value 1, entropy no decay"
        ;;
    25) # NO GUMBEL + STATE 4 VALUE 1 - NO ENTROPY
        state_loss_weight=4.0
        value_loss_weight=1.0
        use_gumbel=""
        use_diversity=""
        use_entropy="$entropy_none"
        run_experiment "no_gumbel_state4_no_ent" "state_loss_high" "No gumbel, state 4 value 1, no entropy"
        ;;
    26) # NO GUMBEL + STATE 4 VALUE 1 - ENTROPY DECAY 0.2
        state_loss_weight=4.0
        value_loss_weight=1.0
        use_gumbel=""
        use_diversity=""
        use_entropy="$entropy_decay_0_2"
        run_experiment "no_gumbel_state4_ent_0_2" "state_loss_high" "No gumbel, state 4 value 1, entropy decay 0.2"
        ;;
    27) # NO GUMBEL + STATE 4 VALUE 1 - ENTROPY DECAY 0.5
        state_loss_weight=4.0
        value_loss_weight=1.0
        use_gumbel=""
        use_diversity=""
        use_entropy="$entropy_decay_0_5"
        run_experiment "no_gumbel_state4_ent_0_5" "state_loss_high" "No gumbel, state 4 value 1, entropy decay 0.5"
        ;;
    28) # NO GUMBEL + STATE 4 VALUE 1 - ENTROPY NO DECAY
        state_loss_weight=4.0
        value_loss_weight=1.0
        use_gumbel=""
        use_diversity=""
        use_entropy="$entropy_no_decay"
        run_experiment "no_gumbel_state4_ent_no_decay" "state_loss_high" "No gumbel, state 4 value 1, entropy no decay"
        ;;
        
    *)
        echo "Invalid task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "Job finished at $(date)"