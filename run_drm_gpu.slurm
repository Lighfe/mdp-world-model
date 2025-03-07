#!/bin/bash
#SBATCH --job-name=drm_training
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=02:00:00          # Just under 2h limit for gpushort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # Request 8 CPU cores
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=16G                # Request 16GB memory
#SBATCH --output=drm_train_%j.out
#SBATCH --error=drm_train_%j.err

# Print job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# Load Anaconda module
module load anaconda/2024.10

# Navigate to your project directory
cd $SLURM_SUBMIT_DIR

# Create and set up the environment from existing environment.yml
echo "Setting up conda environment from existing environment.yml..."

# Check if environment exists, create if it doesn't
if ! conda env list | grep -q "mdp-world-model"; then
    echo "Creating environment: mdp-world-model from environment.yml"
    conda env create -f environment.yml
else
    echo "Environment mdp-world-model already exists"
    echo "Updating environment from environment.yml"
    conda env update -f environment.yml
fi

# Activate the environment
source activate mdp-world-model

# Check if we have GPU
echo "Checking GPU availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Set PyTorch to use available CPU cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# Run your training script
echo "Starting training..."
python -m neural_networks.train_drm datasets/results/tech_toy2.db \
    --output_dir=neural_networks/output \
    --val_size=2000 \
    --test_size=2000 \
    --batch_size=128 \
    --epochs=100 \
    --lr=0.0001 \
    --patience=10 \
    --num_states=4 \
    --hidden_dim=256

# Print completion information
echo "Job finished at $(date)"