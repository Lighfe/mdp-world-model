#!/bin/bash
#SBATCH --job-name=saddle_datasets
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=log/saddle_datasets_%j.out
#SBATCH --error=log/saddle_datasets_%j.err

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

echo "Starting sequential saddle system simulations..."

# 1) saddle at [0.0, 0.0] and 180 angle
echo "=== Running simulation 1/10: saddle at [0.0, 0.0], 180° ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0]]' \
--angles '[180]' \
--lambda1 1.0 \
--lambda2=-1.0 \
--control 0 \
--db-name single_saddle_1.db

# 2) saddle at [0.0, 2.0] and 180 angle (vertical shift)
echo "=== Running simulation 2/10: saddle at [0.0, 2.0], 180° ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 2.0]]' \
--angles '[180]' \
--lambda1 1.0 \
--lambda2=-1.0 \
--control 0 \
--db-name single_saddle_2.db

# 3) saddle at [0.0, -2.0] and 180 angle (vertical shift)
echo "=== Running simulation 3/10: saddle at [0.0, -2.0], 180° ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, -2.0]]' \
--angles '[180]' \
--lambda1 1.0 \
--lambda2=-1.0 \
--control 0 \
--db-name single_saddle_3.db

# 4) saddle at [0.0, 0.0] and 135 angle (diagonal)
echo "=== Running simulation 4/10: saddle at [0.0, 0.0], 135° ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0]]' \
--angles '[135]' \
--lambda1 1.0 \
--lambda2=-1.0 \
--control 0 \
--db-name single_saddle_4.db

# 5) saddle at [2.0, 0.0] and 180 angle (horizontal shift)
echo "=== Running simulation 5/10: saddle at [2.0, 0.0], 180° ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[2.0, 0.0]]' \
--angles '[180]' \
--lambda1 1.0 \
--lambda2=-1.0 \
--control 0 \
--db-name single_saddle_5.db

# 6) saddle at [0.0, 0.0] and 180 angle, lambda1 = 0.5
echo "=== Running simulation 6/10: saddle at [0.0, 0.0], 180°, lambda1=0.5 ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0]]' \
--angles '[180]' \
--lambda1 0.5 \
--lambda2=-1.0 \
--control 0 \
--db-name single_saddle_6.db

# 7) saddle at [2.0, 0.0] and 180 angle, lambda1 = 0.5 (horizontal shift)
echo "=== Running simulation 7/10: saddle at [2.0, 0.0], 180°, lambda1=0.5 ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[2.0, 0.0]]' \
--angles '[180]' \
--lambda1 0.5 \
--lambda2=-1.0 \
--control 0 \
--db-name single_saddle_7.db

# 8) saddle at [0.0, 0.0] and 180 angle, lambda1 = 1.5
echo "=== Running simulation 8/10: saddle at [0.0, 0.0], 180°, lambda1=1.5 ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0]]' \
--angles '[180]' \
--lambda1 1.5 \
--lambda2=-1.0 \
--control 0 \
--db-name single_saddle_8.db

# 9) saddle at [0.0, 0.0] and 180 angle, delta_t = 3.0
echo "=== Running simulation 9/10: saddle at [0.0, 0.0], 180°, delta_t=3.0 ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 3.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0]]' \
--angles '[180]' \
--lambda1 1.0 \
--lambda2=-1.0 \
--control 0 \
--db-name single_saddle_9.db

# 10) saddle at [0.0, 0.0] and 180 angle, resolution 30 30
echo "=== Running simulation 10/10: saddle at [0.0, 0.0], 180°, resolution 30x30 ==="
python run_saddle_simulation.py \
--resolution 30 30 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0]]' \
--angles '[180]' \
--lambda1 1.0 \
--lambda2=-1.0 \
--control 0 \
--db-name single_saddle_10.db

# Print completion information
echo "All simulations completed at $(date)"