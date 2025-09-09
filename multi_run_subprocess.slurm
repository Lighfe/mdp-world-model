#!/bin/bash
#SBATCH --job-name=multi_run_mps
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --output=log/multi_run_mps_%j.out
#SBATCH --error=log/multi_run_mps_%j.err

echo "========================================================"
echo "MULTI-RUN DRM TRAINING - WITH CUDA MPS"
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "GPU Info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "========================================================"

# Create log directory
mkdir -p log

# Load modules and activate environment
module load anaconda
source activate mdp-world-model

# Explicitly set PATH to use conda environment
export PATH="$CONDA_PREFIX/bin:$PATH"

# Verify environment
echo "Python executable: $(which python)"
python -c "import torch; print(f'Torch {torch.__version__} available')"

# ============= CUDA MPS SETUP =============
echo "Setting up CUDA MPS..."

# Set GPU for MPS
export CUDA_VISIBLE_DEVICES=0

# Create MPS directories (fix for HPC systems)
MPS_LOG_DIR="log/mps_${SLURM_JOBID}"
MPS_PIPE_DIR="log/mps_pipes_${SLURM_JOBID}"
mkdir -p "$MPS_LOG_DIR" "$MPS_PIPE_DIR"

# Start CUDA MPS control daemon
echo "Starting CUDA MPS control daemon..."
export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR"
export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR"

# Kill any existing MPS daemon
echo quit | nvidia-cuda-mps-control 2>/dev/null || true

# Start new MPS daemon
nvidia-cuda-mps-control -d

# Verify MPS is running
if pgrep nvidia-cuda-mps > /dev/null; then
    echo "✓ CUDA MPS daemon started successfully"
else
    echo "✗ Failed to start CUDA MPS daemon"
    exit 1
fi
# ============= END MPS SETUP =============

# ============= THREAD CONTROL =============
# Prevent thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1 
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# H100 optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_USE_CUDA_DSA=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# ============= END THREAD CONTROL =============

# Job parameters
CONFIG_PATH="configs/test.yaml"
OUTPUT_DIR="neural_networks/output"
CONFIG_ID="test_aggregate4"
SEEDS="11,12,13,14,15,16,17,18,19,20"
DB_PATHS="datasets/results/multi_saddle_5.db,datasets/results/multi_saddle_7.db,datasets/results/multi_saddle_9.db"
MAX_PARALLEL="15"

echo "========================================================"
echo "Starting MPS-enabled multiprocessing training..."
echo "Config: $CONFIG_PATH"
echo "Config ID: $CONFIG_ID"
echo "Seeds: $SEEDS"
echo "Databases: $DB_PATHS"
echo "Max parallel: $MAX_PARALLEL"
echo "========================================================"

# Start GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv --loop=30 > "log/gpu_mps_${SLURM_JOBID}.log" &
GPU_MONITOR_PID=$!

# Change to project directory
cd /p/projects/ou/rd4/bega/mdp/dralle/mdp-world-model

# Execute training with MPS
python neural_networks/train_drm_multi.py "$CONFIG_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --config-id "$CONFIG_ID" \
    --seeds "$SEEDS" \
    --db-paths "$DB_PATHS" \
    --max-parallel "$MAX_PARALLEL"

PYTHON_EXIT_CODE=$?

# ============= MPS CLEANUP =============
echo "Cleaning up CUDA MPS..."

# First try graceful shutdown
echo quit | nvidia-cuda-mps-control 2>/dev/null || true

# Wait for graceful shutdown
sleep 3

# Check if MPS processes are still running (only our own processes)
if pgrep -u $USER nvidia-cuda-mps > /dev/null; then
    echo "Warning: MPS processes still running, attempting cleanup..."
    
    # Try to kill only our own MPS processes
    pkill -u $USER nvidia-cuda-mps 2>/dev/null || true
    
    # Wait a bit more
    sleep 2
    
    # Final check
    if pgrep -u $USER nvidia-cuda-mps > /dev/null; then
        echo "Some MPS processes may still be running (not owned by us or protected)"
        pgrep -u $USER nvidia-cuda-mps || echo "No processes found"
    else
        echo "✓ All MPS processes cleaned up"
    fi
else
    echo "✓ MPS processes already terminated"
fi
# ============= END MPS CLEANUP =============

echo "========================================================"
echo "MPS-enabled training completed at $(date)"
echo "Python exit code: $PYTHON_EXIT_CODE"
echo "Results saved to: $OUTPUT_DIR/$CONFIG_ID"
echo "GPU usage log: log/gpu_mps_${SLURM_JOBID}.log"
echo "MPS logs: $MPS_LOG_DIR"
echo "========================================================"

# GPU utilization summary
if [ -f "log/gpu_mps_${SLURM_JOBID}.log" ]; then
    avg_util=$(tail -n +2 "log/gpu_mps_${SLURM_JOBID}.log" | awk -F',' '{sum+=$3; count++} END {if(count>0) print sum/count; else print 0}')
    max_memory=$(tail -n +2 "log/gpu_mps_${SLURM_JOBID}.log" | awk -F',' '{if($1>max) max=$1} END {print max}')
    
    echo "GPU Utilization with MPS:"
    echo "  Average GPU utilization: ${avg_util}%"
    echo "  Peak memory usage: ${max_memory} MiB"
    
    # Expected improvement with MPS
    expected_util=$(echo "$avg_util > 50" | bc -l)
    if [ "$expected_util" -eq 1 ]; then
        echo "  ✓ Significant improvement with MPS!"
    else
        echo "  ⚠ GPU utilization still low - check kernel size/batch size"
    fi
fi

exit $PYTHON_EXIT_CODE