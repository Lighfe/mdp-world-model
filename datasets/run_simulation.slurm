#!/bin/bash
#SBATCH --job-name=run_simulations
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --time=02:00:00          # 1 hour should be sufficient for simulations
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # Request 8 CPU cores
#SBATCH --mem=16G                # Request 16GB memory
#SBATCH --output=log/sim_run_%j.out
#SBATCH --error=log/sim_run_%j.err

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

# Run the simulation script with control=0.5
echo "Starting simulation with control=0.5..."
python run_simulation.py \
    --resolution 30 30 \
    --use-importance \
    --importance-method angular \
    --min-samples-per-cell 1 \
    --importance-alpha 1.0 \
    --possible-controls 0.5 1.0 \
    --total-samples 10000 \
    --num-steps 1 \
    --delta-t 2.65 \
    --control 0.5 \
    --db-name tech_angular_xtr.db

# Run the simulation script with control=1.0
echo "Starting simulation with control=1.0..."
python run_simulation.py \
    --resolution 30 30 \
    --use-importance \
    --importance-method angular \
    --min-samples-per-cell 1 \
    --importance-alpha 1.0 \
    --possible-controls 0.5 1.0 \
    --total-samples 10000 \
    --num-steps 1 \
    --delta-t 2.65 \
    --control 1.0 \
    --db-name tech_angular_xtr.db

# Print completion information
echo "Job finished at $(date)"

# Run the simulation script with control=0.5
echo "Starting simulation with control=0.5..."
python run_simulation.py \
    --resolution 30 30 \
    --use-importance \
    --importance-method angular \
    --min-samples-per-cell 2 \
    --importance-alpha 0.5 \
    --possible-controls 0.5 1.0 \
    --total-samples 10000 \
    --num-steps 1 \
    --delta-t 2.65 \
    --control 0.5 \
    --db-name tech_angular_med.db

# Run the simulation script with control=1.0
echo "Starting simulation with control=1.0..."
python run_simulation.py \
    --resolution 30 30 \
    --use-importance \
    --importance-method angular \
    --min-samples-per-cell 2 \
    --importance-alpha 0.5 \
    --possible-controls 0.5 1.0 \
    --total-samples 10000 \
    --num-steps 1 \
    --delta-t 2.65 \
    --control 1.0 \
    --db-name tech_angular_med.db

# Print completion information
echo "Job finished at $(date)"

# Run the simulation script with control=0.5
echo "Starting simulation with control=0.5..."
python run_simulation.py \
    --resolution 30 30 \
    --use-importance \
    --importance-method angular \
    --min-samples-per-cell 5 \
    --importance-alpha 0.3 \
    --possible-controls 0.5 1.0 \
    --total-samples 10000 \
    --num-steps 1 \
    --delta-t 2.65 \
    --control 0.5 \
    --db-name tech_angular_lt.db

# Run the simulation script with control=1.0
echo "Starting simulation with control=1.0..."
python run_simulation.py \
    --resolution 30 30 \
    --use-importance \
    --importance-method angular \
    --min-samples-per-cell 5 \
    --importance-alpha 0.3 \
    --possible-controls 0.5 1.0 \
    --total-samples 10000 \
    --num-steps 1 \
    --delta-t 2.65 \
    --control 1.0 \
    --db-name tech_angular_lt.db

# Print completion information
echo "Job finished at $(date)"