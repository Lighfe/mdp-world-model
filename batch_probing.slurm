#!/bin/bash
#SBATCH --job-name=batch_layer_probing
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=log/batch_probing_%j.out
#SBATCH --error=log/batch_probing_%j.err

# Print job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# Create log directory if it doesn't exist
mkdir -p log

# Load modules and activate environment
module load anaconda
source activate mdp-world-model

# Check GPU
echo "Checking GPU availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Set PyTorch threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# Configuration
BASE_OUTPUT_DIR="neural_networks/output/multi_saddle_probing"
PROBING_SIZE=5000
DEVICE="cuda"
OVERWRITE_EXISTING="true"

echo "========================================================"
echo "Batch Layer Probing Configuration:"
echo "  Base output directory: $BASE_OUTPUT_DIR"
echo "  Probing size: $PROBING_SIZE"
echo "  Device: $DEVICE"
echo "  Overwrite existing: $OVERWRITE_EXISTING"
echo "========================================================"

# Run batch layer probing
python run_batch_probing.py "$BASE_OUTPUT_DIR" \
    --probing_size $PROBING_SIZE \
    --device "$DEVICE" \
    $(if [ "$OVERWRITE_EXISTING" = "true" ]; then echo "--overwrite_existing"; fi)

echo "Job completed at $(date)"