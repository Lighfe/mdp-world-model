#!/bin/bash
#SBATCH --job-name=drm_collapse
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=log/drm_collapse_%A_%a.out
#SBATCH --error=log/drm_collapse_%A_%a.err
#SBATCH --array=1-55

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

# Base configuration (the one that typically works)
db_path="datasets/results/tech_angular_lt.db"
base_output_dir="neural_networks/output/collapse_study"
val_size=2000
test_size=2000
batch_size=64
epochs=100
patience=100
num_states=4
hidden_dim=64
predictor_type="bilinear"
value_method="market_share"

# Default parameters that work
state_loss_weight=0.5
value_loss_weight=2.0
lr=1e-5
use_warmup="--use_warmup --warmup_epochs=5"
use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
use_target_encoder="--use_target_encoder --ema_decay=0.9"
use_diversity=""  # No diversity by default

# Define two good entropy options
entropy_none=""
entropy_good="--use_entropy_reg --entropy_weight=3.0 --use_entropy_decay --entropy_decay_proportion=0.2"
entropy_bad="--use_entropy_reg --entropy_weight=5.0 --no_entropy_decay"

# Default to no entropy
use_entropy="$entropy_none"

# Function to run experiment three times
run_experiment() {
    local experiment_name=$1
    local output_subfolder=$2
    local config=$3
    
    # Create output directory with subfolder
    output_dir="${base_output_dir}/${output_subfolder}"
    mkdir -p "$output_dir"
    
    # Generate timestamp once for all runs
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Run the experiment three times with the same parameters
    for repeat_id in 0 1 2; do
        local run_id="${experiment_name}_r${repeat_id}_${timestamp}"
        
        echo "Running experiment: $experiment_name (repeat $repeat_id)"
        echo "Configuration: $config"
        
        # Construct the complete command
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

# Map task ID to experiment
case $SLURM_ARRAY_TASK_ID in
    # BASELINES
    1)  # BASELINE 1 - All good params, no entropy
        # Reset all parameters to default working values
        state_loss_weight=0.5
        value_loss_weight=2.0
        lr=1e-5
        use_warmup="--use_warmup --warmup_epochs=5"
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_target_encoder="--use_target_encoder --ema_decay=0.9"
        use_diversity=""
        use_entropy="$entropy_none"
        run_experiment "baseline_no_ent" "baseline" "All parameters good, no entropy"
        ;;
    2)  # BASELINE 2 - All good params, with good entropy
        # Reset all parameters to default working values
        state_loss_weight=0.5
        value_loss_weight=2.0
        lr=1e-5
        use_warmup="--use_warmup --warmup_epochs=5"
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_target_encoder="--use_target_encoder --ema_decay=0.9"
        use_diversity=""
        use_entropy="$entropy_good"
        run_experiment "baseline_good_ent" "baseline" "All parameters good, with good entropy"
        ;;
        
    # STATE LOSS EXPERIMENTS - NO ENTROPY
    3)  # STATE GOOD - NO ENTROPY
        state_loss_weight=0.5
        value_loss_weight=2.0
        use_entropy="$entropy_none"
        run_experiment "s_good_no_ent" "state" "state_loss=0.5, value_loss=2.0, no entropy"
        ;;
    4)  # STATE BAD - NO ENTROPY
        state_loss_weight=2.0
        value_loss_weight=1.0
        use_entropy="$entropy_none"
        run_experiment "s_bad_no_ent" "state" "state_loss=2.0, value_loss=1.0, no entropy"
        ;;
        
    # STATE LOSS EXPERIMENTS - GOOD ENTROPY
    5)  # STATE GOOD - GOOD ENTROPY
        state_loss_weight=0.5
        value_loss_weight=2.0
        use_entropy="$entropy_good"
        run_experiment "s_good_good_ent" "state" "state_loss=0.5, value_loss=2.0, good entropy"
        ;;
    6)  # STATE BAD - GOOD ENTROPY
        state_loss_weight=2.0
        value_loss_weight=1.0
        use_entropy="$entropy_good"
        run_experiment "s_bad_good_ent" "state" "state_loss=2.0, value_loss=1.0, good entropy"
        ;;
        
    # DIVERSITY EXPERIMENTS - NO ENTROPY  
    7)  # DIV GOOD (No diversity) - NO ENTROPY
        use_diversity=""
        use_entropy="$entropy_none"
        run_experiment "d_good_no_ent" "diversity" "No diversity, no entropy"
        ;;
    8)  # DIV BAD 1 - NO ENTROPY
        use_diversity="--use_state_diversity --diversity_weight=0.5"
        use_entropy="$entropy_none"
        run_experiment "d_bad_high_no_ent" "diversity" "diversity_weight=0.5, no entropy"
        ;;
    9)  # DIV BAD 2 - NO ENTROPY
        use_diversity="--use_state_diversity --diversity_weight=0.001"
        use_entropy="$entropy_none"
        run_experiment "d_bad_low_no_ent" "diversity" "diversity_weight=0.001, no entropy"
        ;;
        
    # DIVERSITY EXPERIMENTS - GOOD ENTROPY
    10)  # DIV GOOD (No diversity) - GOOD ENTROPY
        use_diversity=""
        use_entropy="$entropy_good"
        run_experiment "d_good_good_ent" "diversity" "No diversity, good entropy"
        ;;
    11)  # DIV BAD 1 - GOOD ENTROPY
        use_diversity="--use_state_diversity --diversity_weight=0.5"
        use_entropy="$entropy_good"
        run_experiment "d_bad_high_good_ent" "diversity" "diversity_weight=0.5, good entropy"
        ;;
    12)  # DIV BAD 2 - GOOD ENTROPY
        use_diversity="--use_state_diversity --diversity_weight=0.001"
        use_entropy="$entropy_good"
        run_experiment "d_bad_low_good_ent" "diversity" "diversity_weight=0.001, good entropy"
        ;;
        
    # GUMBEL EXPERIMENTS - NO ENTROPY
    13)  # GUMBEL GOOD - NO ENTROPY
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_entropy="$entropy_none"
        run_experiment "g_good_no_ent" "gumbel" "With gumbel, no entropy"
        ;;
    14)  # GUMBEL BAD - NO ENTROPY
        use_gumbel=""  # No gumbel
        use_entropy="$entropy_none"
        run_experiment "g_bad_no_ent" "gumbel" "No gumbel, no entropy"
        ;;
        
    # GUMBEL EXPERIMENTS - GOOD ENTROPY
    15)  # GUMBEL GOOD - GOOD ENTROPY
        use_gumbel="--use_gumbel --initial_temp=5.0 --min_temp=0.5"
        use_entropy="$entropy_good"
        run_experiment "g_good_good_ent" "gumbel" "With gumbel, good entropy"
        ;;
    16)  # GUMBEL BAD - GOOD ENTROPY
        use_gumbel=""  # No gumbel
        use_entropy="$entropy_good"
        run_experiment "g_bad_good_ent" "gumbel" "No gumbel, good entropy"
        ;;
        
    # EMA DECAY EXPERIMENTS - NO ENTROPY
    17)  # EMA GOOD - NO ENTROPY
        use_target_encoder="--use_target_encoder --ema_decay=0.9"
        use_entropy="$entropy_none"
        run_experiment "e_good_no_ent" "ema" "ema_decay=0.9, no entropy"
        ;;
    18)  # EMA BAD 1 - NO ENTROPY
        use_target_encoder="--use_target_encoder --ema_decay=0.996"
        use_entropy="$entropy_none"
        run_experiment "e_bad_high_no_ent" "ema" "ema_decay=0.996, no entropy"
        ;;
    19) # EMA BAD 2 - NO ENTROPY
        use_target_encoder="--use_target_encoder --ema_decay=0.3"
        use_entropy="$entropy_none"
        run_experiment "e_bad_low_no_ent" "ema" "ema_decay=0.3, no entropy"
        ;;
        
    # EMA DECAY EXPERIMENTS - GOOD ENTROPY
    20)  # EMA GOOD - GOOD ENTROPY
        use_target_encoder="--use_target_encoder --ema_decay=0.9"
        use_entropy="$entropy_good"
        run_experiment "e_good_good_ent" "ema" "ema_decay=0.9, good entropy"
        ;;
    21)  # EMA BAD 1 - GOOD ENTROPY
        use_target_encoder="--use_target_encoder --ema_decay=0.996"
        use_entropy="$entropy_good"
        run_experiment "e_bad_high_good_ent" "ema" "ema_decay=0.996, good entropy"
        ;;
    22) # EMA BAD 2 - GOOD ENTROPY
        use_target_encoder="--use_target_encoder --ema_decay=0.3"
        use_entropy="$entropy_good"
        run_experiment "e_bad_low_good_ent" "ema" "ema_decay=0.3, good entropy"
        ;;
        
    # LEARNING RATE EXPERIMENTS - NO ENTROPY
    23) # LR GOOD - NO ENTROPY
        lr=1e-5
        use_warmup="--use_warmup --warmup_epochs=5"
        use_entropy="$entropy_none"
        run_experiment "lr_good_no_ent" "learning_rate" "lr=1e-5 with warmup, no entropy"
        ;;
    24) # LR BAD - NO ENTROPY
        lr=1e-4
        use_warmup=""  # No warmup
        use_entropy="$entropy_none"
        run_experiment "lr_bad_no_ent" "learning_rate" "lr=1e-4 no warmup, no entropy"
        ;;
        
    # LEARNING RATE EXPERIMENTS - GOOD ENTROPY
    25) # LR GOOD - GOOD ENTROPY
        lr=1e-5
        use_warmup="--use_warmup --warmup_epochs=5"
        use_entropy="$entropy_good"
        run_experiment "lr_good_good_ent" "learning_rate" "lr=1e-5 with warmup, good entropy"
        ;;
    26) # LR BAD - GOOD ENTROPY
        lr=1e-4
        use_warmup=""  # No warmup
        use_entropy="$entropy_good"
        run_experiment "lr_bad_good_ent" "learning_rate" "lr=1e-4 no warmup, good entropy"
        ;;
        
    # ENTROPY EXPERIMENTS - BAD ENTROPY
    27) # ENTROPY BAD
        use_entropy="$entropy_bad"
        run_experiment "ent_bad" "entropy" "entropy_weight=5.0 no decay"
        ;;
        
    # COMBINATIONS - STATE FAILURE PLUS OTHERS
    28) # STATE + DIV 1
        state_loss_weight=2.0
        value_loss_weight=1.0
        use_diversity="--use_state_diversity --diversity_weight=0.5"
        use_entropy="$entropy_none"
        run_experiment "s_d_high" "combinations" "state_bad + div_weight=0.5"
        ;;
    29) # STATE + DIV 2
        state_loss_weight=2.0
        value_loss_weight=1.0
        use_diversity="--use_state_diversity --diversity_weight=0.001"
        use_entropy="$entropy_none"
        run_experiment "s_d_low" "combinations" "state_bad + div_weight=0.001"
        ;;
    30) # STATE + GUMBEL
        state_loss_weight=2.0
        value_loss_weight=1.0
        use_gumbel=""
        use_entropy="$entropy_none"
        run_experiment "s_g" "combinations" "state_bad + no_gumbel"
        ;;
    31) # STATE + EMA 1
        state_loss_weight=2.0
        value_loss_weight=1.0
        use_target_encoder="--use_target_encoder --ema_decay=0.996"
        use_entropy="$entropy_none"
        run_experiment "s_e_high" "combinations" "state_bad + ema=0.996"
        ;;
    32) # STATE + EMA 2
        state_loss_weight=2.0
        value_loss_weight=1.0
        use_target_encoder="--use_target_encoder --ema_decay=0.3"
        use_entropy="$entropy_none"
        run_experiment "s_e_low" "combinations" "state_bad + ema=0.3"
        ;;
    33) # STATE + LR
        state_loss_weight=2.0
        value_loss_weight=1.0
        lr=1e-4
        use_warmup=""
        use_entropy="$entropy_none"
        run_experiment "s_lr" "combinations" "state_bad + lr_bad"
        ;;
    34) # STATE + ENTROPY
        state_loss_weight=2.0
        value_loss_weight=1.0
        use_entropy="$entropy_bad"
        run_experiment "s_ent" "combinations" "state_bad + entropy_bad"
        ;;
        
    # COMBINATIONS - DIVERSITY FAILURE PLUS OTHERS
    35) # DIV + GUMBEL
        use_diversity="--use_state_diversity --diversity_weight=0.5"
        use_gumbel=""
        use_entropy="$entropy_none"
        run_experiment "d_g" "combinations" "div_bad + no_gumbel"
        ;;
    36) # DIV + EMA 1
        use_diversity="--use_state_diversity --diversity_weight=0.5"
        use_target_encoder="--use_target_encoder --ema_decay=0.996"
        use_entropy="$entropy_none"
        run_experiment "d_e_high" "combinations" "div_bad + ema=0.996"
        ;;
    37) # DIV + EMA 2
        use_diversity="--use_state_diversity --diversity_weight=0.5"
        use_target_encoder="--use_target_encoder --ema_decay=0.3"
        use_entropy="$entropy_none"
        run_experiment "d_e_low" "combinations" "div_bad + ema=0.3"
        ;;
    38) # DIV + LR
        use_diversity="--use_state_diversity --diversity_weight=0.5"
        lr=1e-4
        use_warmup=""
        use_entropy="$entropy_none"
        run_experiment "d_lr" "combinations" "div_bad + lr_bad"
        ;;
    39) # DIV + ENTROPY
        use_diversity="--use_state_diversity --diversity_weight=0.5"
        use_entropy="$entropy_bad"
        run_experiment "d_ent" "combinations" "div_bad + entropy_bad"
        ;;
        
    # COMBINATIONS - GUMBEL FAILURE PLUS OTHERS
    40) # GUMBEL + EMA 1
        use_gumbel=""
        use_target_encoder="--use_target_encoder --ema_decay=0.996"
        use_entropy="$entropy_none"
        run_experiment "g_e_high" "combinations" "no_gumbel + ema=0.996"
        ;;
    41) # GUMBEL + EMA 2
        use_gumbel=""
        use_target_encoder="--use_target_encoder --ema_decay=0.3"
        use_entropy="$entropy_none"
        run_experiment "g_e_low" "combinations" "no_gumbel + ema=0.3"
        ;;
    42) # GUMBEL + LR
        use_gumbel=""
        lr=1e-4
        use_warmup=""
        use_entropy="$entropy_none"
        run_experiment "g_lr" "combinations" "no_gumbel + lr_bad"
        ;;
    43) # GUMBEL + ENTROPY
        use_gumbel=""
        use_entropy="$entropy_bad"
        run_experiment "g_ent" "combinations" "no_gumbel + entropy_bad"
        ;;
        
    # COMBINATIONS - EMA FAILURE PLUS OTHERS
    44) # EMA 1 + LR
        use_target_encoder="--use_target_encoder --ema_decay=0.996"
        lr=1e-4
        use_warmup=""
        use_entropy="$entropy_none"
        run_experiment "e_high_lr" "combinations" "ema=0.996 + lr_bad"
        ;;
    45) # EMA 2 + LR
        use_target_encoder="--use_target_encoder --ema_decay=0.3"
        lr=1e-4
        use_warmup=""
        use_entropy="$entropy_none"
        run_experiment "e_low_lr" "combinations" "ema=0.3 + lr_bad"
        ;;
    46) # EMA 1 + ENTROPY
        use_target_encoder="--use_target_encoder --ema_decay=0.996"
        use_entropy="$entropy_bad"
        run_experiment "e_high_ent" "combinations" "ema=0.996 + entropy_bad"
        ;;
    47) # EMA 2 + ENTROPY
        use_target_encoder="--use_target_encoder --ema_decay=0.3"
        use_entropy="$entropy_bad"
        run_experiment "e_low_ent" "combinations" "ema=0.3 + entropy_bad"
        ;;
        
    # COMBINATIONS - LR FAILURE PLUS OTHERS
    48) # LR + ENTROPY
        lr=1e-4
        use_warmup=""
        use_entropy="$entropy_bad"
        run_experiment "lr_ent" "combinations" "lr_bad + entropy_bad"
        ;;
        
    # ADDITIONAL LOW DIVERSITY WEIGHT COMBINATIONS
    49) # DIV LOW + GUMBEL
        use_diversity="--use_state_diversity --diversity_weight=0.001"
        use_gumbel=""
        use_entropy="$entropy_none"
        run_experiment "d_low_g" "combinations" "div_weight=0.001 + no_gumbel"
        ;;
    50) # DIV LOW + LR
        use_diversity="--use_state_diversity --diversity_weight=0.001"
        lr=1e-4
        use_warmup=""
        use_entropy="$entropy_none"
        run_experiment "d_low_lr" "combinations" "div_weight=0.001 + lr_bad"
        ;;
    51) # DIV LOW + ENTROPY
        use_diversity="--use_state_diversity --diversity_weight=0.001"
        use_entropy="$entropy_bad"
        run_experiment "d_low_ent" "combinations" "div_weight=0.001 + entropy_bad"
        ;;
    
    # EXTRA COMBINATIONS WITH GOOD ENTROPY
    52) # STATE + GUMBEL + GOOD ENTROPY
        state_loss_weight=2.0
        value_loss_weight=1.0
        use_gumbel=""
        use_entropy="$entropy_good"
        run_experiment "s_g_good_ent" "combinations" "state_bad + no_gumbel + good_ent"
        ;;
    53) # DIV + GUMBEL + GOOD ENTROPY
        use_diversity="--use_state_diversity --diversity_weight=0.5"
        use_gumbel=""
        use_entropy="$entropy_good"
        run_experiment "d_g_good_ent" "combinations" "div_bad + no_gumbel + good_ent"
        ;;
    54) # GUMBEL + EMA 1 + GOOD ENTROPY
        use_gumbel=""
        use_target_encoder="--use_target_encoder --ema_decay=0.996"
        use_entropy="$entropy_good"
        run_experiment "g_e_good_ent" "combinations" "no_gumbel + ema_bad + good_ent"
        ;;
    55) # LR + GUMBEL + GOOD ENTROPY
        lr=1e-4
        use_warmup=""
        use_gumbel=""
        use_entropy="$entropy_good"
        run_experiment "lr_g_good_ent" "combinations" "lr_bad + no_gumbel + good_ent"
        ;;
        
    *)
        echo "Invalid task ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac

echo "Job finished at $(date)"