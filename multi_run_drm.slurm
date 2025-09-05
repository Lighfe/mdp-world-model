#!/bin/bash
#SBATCH --job-name=multi_run_drm
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=log/multi_run_drm_%j.out
#SBATCH --error=log/multi_run_drm_%j.err

echo "========================================================"
echo "MULTI-RUN DRM TRAINING - PARALLEL EXECUTION"
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "GPU Info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "========================================================"

# Create log directory
mkdir -p log

# Load modules and activate environment
module load anaconda
source activate mdp-world-model

# Set standard environment
export OMP_NUM_THREADS=2
export OMP_PROC_BIND=true

# ============= H100-SPECIFIC OPTIMIZATIONS =============
# Enable asynchronous GPU operations
export CUDA_LAUNCH_BLOCKING=0

# Set architecture for H100 (compute capability 9.0)
export TORCH_CUDA_ARCH_LIST="9.0"

# Better error messages for debugging
export TORCH_USE_CUDA_DSA=1

# Enable TF32 for tensor operations
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Fix CuBLAS deterministic warning
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Start MPS daemon for better GPU sharing among parallel tasks
echo "Starting NVIDIA MPS for better GPU sharing..."
nvidia-cuda-mps-control -d
sleep 2  # Give MPS time to start

# Verify MPS is running
if nvidia-smi | grep -q "MPS"; then
    echo "✓ MPS daemon started successfully"
else
    echo "⚠ MPS may not be running, continuing anyway"
fi
# ============= END H100 OPTIMIZATIONS =============

# Start GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv --loop=30 > "log/gpu_multi_run_${SLURM_JOBID}.log" &
GPU_MONITOR_PID=$!

# Define parameters for the job (example - modify as needed)
CONFIG_PATH="configs/base.yaml"
OUTPUT_DIR="neural_networks/output"
CONFIG_ID="test_parallel7"
SEEDS="15,16,17"
DB_PATHS="datasets/results/multi_saddle_7.db,datasets/results/multi_saddle_9.db"
MAX_PARALLEL="6"

echo "========================================================"
echo "Starting Python multiprocessing training..."
echo "Config: $CONFIG_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Config ID: $CONFIG_ID"
echo "Seeds: $SEEDS"
echo "Databases: $DB_PATHS"
echo "Max parallel: $MAX_PARALLEL"
echo "========================================================"

# Execute the multi-run training (no srun needed - multiprocessing handles parallelization)
python neural_networks/train_drm_multi_run.py "$CONFIG_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --config-id "$CONFIG_ID" \
    --seeds "$SEEDS" \
    --db-paths "$DB_PATHS" \
    --max-parallel "$MAX_PARALLEL"

PYTHON_EXIT_CODE=$?

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null

# Stop MPS daemon
echo "Stopping NVIDIA MPS..."
echo quit | nvidia-cuda-mps-control

echo "========================================================"
echo "Multi-run training completed at $(date)"
echo "Python exit code: $PYTHON_EXIT_CODE"
echo "GPU usage log: log/gpu_multi_run_${SLURM_JOBID}.log"
echo "========================================================"

exit $PYTHON_EXIT_CODE