#!/bin/bash
#SBATCH --job-name=drm_training
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=log/drm_train_%A_%a.out
#SBATCH --error=log/drm_train_%A_%a.err
#SBATCH --array=1-2 # This runs the job twice
# Print job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Load modules and activate environment
module load anaconda
source activate mdp-world-model

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Set PyTorch threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# Add delay based on array task ID (in seconds)
# First job (ID=1) starts immediately, second job (ID=2) waits 5 seconds
sleep $((($SLURM_ARRAY_TASK_ID - 1) * 5))

echo "Starting training job ${SLURM_ARRAY_TASK_ID}"
python -m neural_networks.train_drm datasets/results/tech_angular_lt.db \
--output_dir=neural_networks/output \
--run_id="$(date +%Y%m%d_%H%M%S)_task${SLURM_ARRAY_TASK_ID}_e10_v2_ema9" \
--val_size=2000 \
--test_size=2000 \
--batch_size=64 \
--epochs=100 \
--lr=1e-5 \
--use_warmup \
--warmup_epochs=5 \
--patience=10 \
--num_states=4 \
--hidden_dim=64 \
--state_loss_weight=0.5 \
--value_loss_weight=2.0 \
--predictor_type=bilinear \
--value_method="market_share" \
--use_entropy_reg \
--entropy_weight=10.0 \
--use_target_encoder \
--ema_decay=0.996
echo "Job1 finished at $(date)"

echo "Starting training job ${SLURM_ARRAY_TASK_ID}"
python -m neural_networks.train_drm datasets/results/tech_angular_lt.db \
--output_dir=neural_networks/output \
--run_id="$(date +%Y%m%d_%H%M%S)_task${SLURM_ARRAY_TASK_ID}_e5_v2_ema9" \
--val_size=2000 \
--test_size=2000 \
--batch_size=64 \
--epochs=100 \
--lr=1e-5 \
--use_warmup \
--warmup_epochs=5 \
--patience=10 \
--num_states=4 \
--hidden_dim=64 \
--state_loss_weight=0.5 \
--value_loss_weight=2.0 \
--predictor_type=bilinear \
--value_method="market_share" \
--use_entropy_reg \
--entropy_weight=5.0 \
--use_target_encoder \
--ema_decay=0.996
echo "Job2 finished at $(date)"

echo "Starting training job ${SLURM_ARRAY_TASK_ID}"
python -m neural_networks.train_drm datasets/results/tech_angular_lt.db \
--output_dir=neural_networks/output \
--run_id="$(date +%Y%m%d_%H%M%S)_task${SLURM_ARRAY_TASK_ID}_e10_v2_ema3" \
--val_size=2000 \
--test_size=2000 \
--batch_size=64 \
--epochs=100 \
--lr=1e-5 \
--use_warmup \
--warmup_epochs=5 \
--patience=10 \
--num_states=4 \
--hidden_dim=64 \
--state_loss_weight=0.5 \
--value_loss_weight=2.0 \
--predictor_type=bilinear \
--value_method="market_share" \
--use_entropy_reg \
--entropy_weight=10.0 \
--use_target_encoder \
--ema_decay=0.3
echo "Job3 finished at $(date)"

echo "Starting training job ${SLURM_ARRAY_TASK_ID}"
python -m neural_networks.train_drm datasets/results/tech_angular_lt.db \
--output_dir=neural_networks/output \
--run_id="$(date +%Y%m%d_%H%M%S)_task${SLURM_ARRAY_TASK_ID}_e5_v2_ema3" \
--val_size=2000 \
--test_size=2000 \
--batch_size=64 \
--epochs=100 \
--lr=1e-5 \
--use_warmup \
--warmup_epochs=5 \
--patience=10 \
--num_states=4 \
--hidden_dim=64 \
--state_loss_weight=0.5 \
--value_loss_weight=2.0 \
--predictor_type=bilinear \
--value_method="market_share" \
--use_entropy_reg \
--entropy_weight=5.0 \
--use_target_encoder \
--ema_decay=0.3
echo "Job4 finished at $(date)"
