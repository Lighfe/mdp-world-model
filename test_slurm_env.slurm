#!/bin/bash
#SBATCH --job-name=test_env
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=log/test_env_%j.out
#SBATCH --error=log/test_env_%j.err

echo "========================================================"
echo "ENVIRONMENT TEST"
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "========================================================"

# Create log directory
mkdir -p log

echo "BEFORE environment setup:"
echo "which python: $(which python 2>/dev/null || echo 'python not found')"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-'NOT SET'}"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-'NOT SET'}"

echo ""
echo "Loading anaconda module..."
module load anaconda

echo "AFTER module load:"
echo "which python: $(which python 2>/dev/null || echo 'python not found')"
echo "which conda: $(which conda 2>/dev/null || echo 'conda not found')"

echo ""
echo "Activating conda environment..."
source activate mdp-world-model

echo "AFTER conda activation:"
echo "which python: $(which python 2>/dev/null || echo 'python not found')"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-'NOT SET'}"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-'NOT SET'}"

echo ""
echo "Testing Python imports..."
python -c "
import sys
print(f'Python executable: {sys.executable}')
try:
    import torch
    print(f'✓ Torch import successful: {torch.__version__}')
except ImportError as e:
    print(f'✗ Torch import failed: {e}')

try:
    import sys
    sys.path.insert(0, '/p/projects/ou/rd4/bega/mdp/dralle/mdp-world-model')
    from neural_networks.utils import load_config
    print('✓ Utils import successful')
except ImportError as e:
    print(f'✗ Utils import failed: {e}')
"

echo ""
echo "Testing subprocess execution..."
python /p/projects/ou/rd4/bega/mdp/dralle/mdp-world-model/debug_python_paths.py

echo "========================================================"
echo "Environment test completed at $(date)"
echo "========================================================"