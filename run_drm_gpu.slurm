#!/bin/bash
#SBATCH --job-name=drm_training
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=02:00:00          # 24h limit for gpushort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # Request 8 CPU cores
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=16G                # Request 16GB memory
#SBATCH --output=log/drm_train_%j.out
#SBATCH --error=log/drm_train_%j.err

# Print job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# Load Anaconda module
module load anaconda
# Activate the environment
source activate mdp-world-model

# Check if we have GPU
echo "Checking GPU availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Set PyTorch to use available CPU cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# Run training script
echo "Starting training..."
python -m neural_networks.train_drm datasets/results/tech_sim.db \
    --output_dir=neural_networks/output \
    --val_size=2000 \
    --test_size=2000 \
    --batch_size=128 \
    --epochs=100 \
    --lr=0.0001 \
    --patience=10 \
    --num_states=3 \
    --hidden_dim=128

# Print completion information
echo "Job finished at $(date)"