#!/bin/bash
#SBATCH --job-name=social_tipping_datasets
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=social_tipping_datasets_%j.out
#SBATCH --error=social_tipping_datasets_%j.err

# Print job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"

# Create log directory if it doesn't exist
mkdir -p log

# Load modules and activate environment
module load anaconda
source activate mdp-world-model

echo "=========================================="
echo "Generating 10 social tipping datasets..."
echo "=========================================="

# Dataset 1: Single action (default), single delta_t (small)
echo "[1/10] Generating social_tipping_default_t2.db..."
python run_social_tipping_simulation.py --db-name "social_tipping_default_t2.db" --actions default --delta-t 2.181

# Dataset 2: Single action (default), single delta_t (medium-low)
echo "[2/10] Generating social_tipping_default_t7.db..."
python run_social_tipping_simulation.py --db-name "social_tipping_default_t7.db" --actions default --delta-t 7.27

# Dataset 3: Single action (default), single delta_t (medium-high)
echo "[3/10] Generating social_tipping_default_t21.db..."
python run_social_tipping_simulation.py --db-name "social_tipping_default_t21.db" --actions default --delta-t 21.81

# Dataset 4: Single action (default), single delta_t (high)
echo "[4/10] Generating social_tipping_default_t58.db..."
python run_social_tipping_simulation.py --db-name "social_tipping_default_t58.db" --actions default --delta-t 58.887

# Dataset 5: Single action (default), two medium delta_t values
echo "[5/10] Generating social_tipping_default_tmedium.db..."
python run_social_tipping_simulation.py --db-name "social_tipping_default_tmedium.db" --actions default --delta-t 7.27 21.81

# Dataset 6: Single action (default), small + high delta_t
echo "[6/10] Generating social_tipping_default_tsmallhigh.db..."
python run_social_tipping_simulation.py --db-name "social_tipping_default_tsmallhigh.db" --actions default --delta-t 2.181 58.887

# Dataset 7: Single action (default), all four delta_t values
echo "[7/10] Generating social_tipping_default_tall.db..."
python run_social_tipping_simulation.py --db-name "social_tipping_default_tall.db" --actions default --delta-t 2.181 7.27 21.81 58.887

# Dataset 8: Two actions (default + subsidy), single medium-low delta_t
echo "[8/10] Generating social_tipping_default_subsidy_t7.db..."
python run_social_tipping_simulation.py --db-name "social_tipping_default_subsidy_t7.db" --actions default subsidy --delta-t 7.27

# Dataset 9: Two actions (default + subsidy), single high delta_t
echo "[9/10] Generating social_tipping_default_subsidy_t58.db..."
python run_social_tipping_simulation.py --db-name "social_tipping_default_subsidy_t58.db" --actions default subsidy --delta-t 58.887

# Dataset 10: Two actions (default + subsidy), all four delta_t values
echo "[10/10] Generating social_tipping_default_subsidy_tall.db..."
python run_social_tipping_simulation.py --db-name "social_tipping_default_subsidy_tall.db" --actions default subsidy --delta-t 2.181 7.27 21.81 58.887

echo "=========================================="
echo "All datasets generated successfully!"
echo "Completed at $(date)"

# List generated databases
echo ""
echo "Generated database files:"
ls -lh datasets/results/social_tipping_*.db

# Summary statistics
echo ""
echo "Dataset generation summary:"
echo "- Total datasets: 10"
echo "- Single action datasets: 7 (datasets 1-7)"
echo "- Multi-action datasets: 3 (datasets 8-10)"
echo "- Single delta_t datasets: 6 (datasets 1-4, 8-9)"
echo "- Multi delta_t datasets: 4 (datasets 5-7, 10)"

echo ""
echo "Job completed successfully at $(date)"