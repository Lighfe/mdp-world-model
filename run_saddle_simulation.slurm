#!/bin/bash
#SBATCH --job-name=saddle_sim
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --time=01:00:00          # 1 hour should be sufficient
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # Request 8 CPU cores
#SBATCH --mem=8G                # Request 8GB memory
#SBATCH --output=log/saddle_sim_%j.out
#SBATCH --error=log/saddle_sim_%j.err

# Print job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# Load Anaconda module
module load anaconda
# Activate the environment
source activate mdp-world-model

# Set number of threads for parallel processing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# Create log directory if it doesn't exist
mkdir -p log

# Run the saddle simulation script with single saddle point at (1.0, 0.0) and 90° angle
echo "Starting saddle system simulation..."
python run_saddle_simulation.py \
    --resolution 20 20 \
    --avg-samples-per-cell 50 \
    --num-steps 1 \
    --delta-t 1.0 \
    --logistic-k 0.5 \
    --logistic-x0 0.0 \
    --saddle-points '[[1.0, 0.0]]' \
    --angles '[90]' \
    --lambda1 1.0 \
    --lambda2=-1.0 \
    --control 0 \
    --db-name single_saddle_system.db

# Print completion information
echo "Job finished at $(date)"