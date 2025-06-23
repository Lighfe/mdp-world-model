#!/bin/bash
#SBATCH --job-name=multi_saddle_datasets
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=log/multi_saddle_datasets_%j.out
#SBATCH --error=log/multi_saddle_datasets_%j.err

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

echo "Starting sequential multi-saddle system simulations..."

# 1) Both saddles at (0,0), angles 180°, 90°
echo "=== Running simulation 1/12: both saddles at [0,0], angles 180°, 90° ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0], [0.0, 0.0]]' \
--angles '[180, 90]' \
--lambda1 1.0 \
--lambda2 -1.0 \
--control 0 \
--db-name multi_saddle_1.db \
--create-streamplot \
--streamplot-original \
--streamplot-range -5.0 5.0

# 2) Saddles at (0,0) and (0,2), angles 180°, 90°
echo "=== Running simulation 2/12: saddles at [0,0] and [0,2], angles 180°, 90° ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0], [0.0, 2.0]]' \
--angles '[180, 90]' \
--lambda1 1.0 \
--lambda2 -1.0 \
--control 0 \
--db-name multi_saddle_2.db \
--create-streamplot \
--streamplot-original \
--streamplot-range -5.0 5.0

# 3) Saddles at (0,0) and (2,0), angles 180°, 90°
echo "=== Running simulation 3/12: saddles at [0,0] and [2,0], angles 180°, 90° ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0], [2.0, 0.0]]' \
--angles '[180, 90]' \
--lambda1 1.0 \
--lambda2 -1.0 \
--control 0 \
--db-name multi_saddle_3.db \
--create-streamplot \
--streamplot-original \
--streamplot-range -5.0 5.0

# 4) Saddles at (2,0) and (0,2), angles 180°, 90°
echo "=== Running simulation 4/12: saddles at [2,0] and [0,2], angles 180°, 90° ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[2.0, 0.0], [0.0, 2.0]]' \
--angles '[180, 90]' \
--lambda1 1.0 \
--lambda2 -1.0 \
--control 0 \
--db-name multi_saddle_4.db \
--create-streamplot \
--streamplot-original \
--streamplot-range -5.0 5.0

# 5) Both saddles at (0,0), angles 180°, 45°
echo "=== Running simulation 5/12: both saddles at [0,0], angles 180°, 45° ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0], [0.0, 0.0]]' \
--angles '[180, 45]' \
--lambda1 1.0 \
--lambda2 -1.0 \
--control 0 \
--db-name multi_saddle_5.db \
--create-streamplot \
--streamplot-original \
--streamplot-range -5.0 5.0

# 6) Saddles at (0,0) and (0,2), angles 180°, 45°
echo "=== Running simulation 6/12: saddles at [0,0] and [0,2], angles 180°, 45° ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0], [0.0, 2.0]]' \
--angles '[180, 45]' \
--lambda1 1.0 \
--lambda2 -1.0 \
--control 0 \
--db-name multi_saddle_6.db \
--create-streamplot \
--streamplot-original \
--streamplot-range -5.0 5.0

# 7) Saddles at (0,0) and (2,0), angles 180°, 45°
echo "=== Running simulation 7/12: saddles at [0,0] and [2,0], angles 180°, 45° ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0], [2.0, 0.0]]' \
--angles '[180, 45]' \
--lambda1 1.0 \
--lambda2 -1.0 \
--control 0 \
--db-name multi_saddle_7.db \
--create-streamplot \
--streamplot-original \
--streamplot-range -5.0 5.0

# 8) Saddles at (2,0) and (0,2), angles 180°, 45°
echo "=== Running simulation 8/12: saddles at [2,0] and [0,2], angles 180°, 45° ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[2.0, 0.0], [0.0, 2.0]]' \
--angles '[180, 45]' \
--lambda1 1.0 \
--lambda2 -1.0 \
--control 0 \
--db-name multi_saddle_8.db \
--create-streamplot \
--streamplot-original \
--streamplot-range -5.0 5.0

# 9) Both saddles at (0,0), angles 180°, 90°, lambda1=0.5
echo "=== Running simulation 9/12: both saddles at [0,0], angles 180°, 90°, lambda1=0.5 ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 50 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0], [0.0, 0.0]]' \
--angles '[180, 90]' \
--lambda1 0.5 \
--lambda2 -1.0 \
--control 0 \
--db-name multi_saddle_9.db \
--create-streamplot \
--streamplot-original \
--streamplot-range -5.0 5.0

# 10a) Both saddles at (0,0), angles 180°, 90°, delta_t=0.1
echo "=== Running simulation 10a/12: both saddles at [0,0], angles 180°, 90°, delta_t=0.1 ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 20 \
--num-steps 1 \
--delta-t 0.1 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0], [0.0, 0.0]]' \
--angles '[180, 90]' \
--lambda1 1.0 \
--lambda2 -1.0 \
--control 0 \
--db-name multi_saddle_10.db \
--create-streamplot \
--streamplot-original \
--streamplot-range -5.0 5.0

# 10b) Both saddles at (0,0), angles 180°, 90°, delta_t=1.0
echo "=== Running simulation 10b/12: both saddles at [0,0], angles 180°, 90°, delta_t=1.0 ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 20 \
--num-steps 1 \
--delta-t 1.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0], [0.0, 0.0]]' \
--angles '[180, 90]' \
--lambda1 1.0 \
--lambda2 -1.0 \
--control 0 \
--db-name multi_saddle_10.db \
--create-streamplot \
--streamplot-original \
--streamplot-range -5.0 5.0

# 10c) Both saddles at (0,0), angles 180°, 90°, delta_t=5.0
echo "=== Running simulation 10c/12: both saddles at [0,0], angles 180°, 90°, delta_t=5.0 ==="
python run_saddle_simulation.py \
--resolution 25 25 \
--avg-samples-per-cell 20 \
--num-steps 1 \
--delta-t 5.0 \
--logistic-k 0.5 \
--logistic-x0 0.0 \
--saddle-points '[[0.0, 0.0], [0.0, 0.0]]' \
--angles '[180, 90]' \
--lambda1 1.0 \
--lambda2 -1.0 \
--control 0 \
--db-name multi_saddle_10.db \
--create-streamplot \
--streamplot-original \
--streamplot-range -5.0 5.0

# Print completion information
echo "All simulations completed at $(date)"