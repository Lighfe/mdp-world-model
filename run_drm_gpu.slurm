#!/bin/bash
#SBATCH --job-name=single_run_baseline
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=0:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=log/single_run_baseline_%j.out
#SBATCH --error=log/single_run_baseline_%j.err

echo "========================================================"
echo "SINGLE RUN BASELINE TEST"
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "========================================================"

# Create log directory
mkdir -p log

# Load modules and activate environment
module load anaconda
source activate mdp-world-model

# Set standard environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# ============= H100-SPECIFIC OPTIMIZATIONS =============
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_USE_CUDA_DSA=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# ============= END H100 OPTIMIZATIONS =============

# Start GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv --loop=5 > "log/gpu_single_run_${SLURM_JOBID}.log" &
GPU_MONITOR_PID=$!

# Record start time
START_TIME=$(date +%s)

echo "Starting single training run with multi_run=True..."

# Run single training (using base config with multi_run flag)
python neural_networks/train_drm.py configs/base.yaml --multi_run

PYTHON_EXIT_CODE=$?

# Calculate runtime
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null

echo "========================================================"
echo "Single run completed at $(date)"
echo "Runtime: $RUNTIME seconds ($((RUNTIME/60)) minutes)"
echo "Python exit code: $PYTHON_EXIT_CODE"
echo "GPU usage log: log/gpu_single_run_${SLURM_JOBID}.log"
echo "========================================================"

exit $PYTHON_EXIT_CODE