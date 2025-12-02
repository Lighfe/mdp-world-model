#!/bin/bash
#SBATCH --job-name=probing_test
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --output=log/probing_test_%j.out
#SBATCH --error=log/probing_test_%j.err

# ============================================================================
# PROBING MULTI-RUN TEST
# Tests: epoch tracking, probing aggregation, visualization
# 16 seeds × 2 databases = 32 runs → aggregated results
# ============================================================================

echo "========================================================"
echo "PROBING MULTI-RUN TEST"
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
echo ""
echo "Environment verification:"
echo "  Python: $(which python)"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
echo ""

# ============= CUDA MPS SETUP =============
echo "Setting up CUDA MPS for better GPU sharing..."

# Set GPU for MPS
export CUDA_VISIBLE_DEVICES=0

# Create MPS directories
MPS_LOG_DIR="log/mps_${SLURM_JOBID}"
MPS_PIPE_DIR="log/mps_pipes_${SLURM_JOBID}"
mkdir -p "$MPS_LOG_DIR" "$MPS_PIPE_DIR"

# Configure MPS
export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR"
export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR"

# Kill any existing MPS daemon
echo quit | nvidia-cuda-mps-control 2>/dev/null || true

# Start new MPS daemon
nvidia-cuda-mps-control -d
sleep 2  # Give MPS time to start

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

# Configuration
CONFIG_FILE="scripts/configs/saddle_test.yaml"
CONFIG_ID="saddle_probing_v0"
SEEDS="141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156"
DB_PATHS="datasets/results/multi_saddle_3.db,datasets/results/multi_saddle_8.db"
MAX_PARALLEL=16

echo ""
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Config ID: $CONFIG_ID"
echo "  Seeds: $SEEDS (16 total)"
echo "  Databases: $DB_PATHS (2 total)"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Total runs: 16 seeds × 2 databases = 32 runs"
echo ""

# Verify config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Verify databases exist
for DB in ${DB_PATHS//,/ }; do
    if [ ! -f "$DB" ]; then
        echo "WARNING: Database not found: $DB"
    else
        echo "✓ Found database: $DB"
    fi
done

# Start GPU monitoring
echo ""
echo "Starting GPU monitoring..."
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv --loop=10 > "log/gpu_probing_test_${SLURM_JOBID}.log" &
GPU_MONITOR_PID=$!
echo "GPU monitoring started (PID: $GPU_MONITOR_PID)"

# Run multi-run training
echo ""
echo "========================================================"
echo "Starting training at $(date)"
echo "========================================================"

python neural_networks/train_drm_multi.py "$CONFIG_FILE" \
    --config-id "$CONFIG_ID" \
    --seeds "$SEEDS" \
    --db-paths "$DB_PATHS" \
    --max-parallel "$MAX_PARALLEL"

EXIT_CODE=$?

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null
echo "GPU monitoring stopped"

# Stop MPS daemon
echo ""
echo "Stopping CUDA MPS..."
echo quit | nvidia-cuda-mps-control

echo ""
echo "========================================================"
echo "Job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ SUCCESS! Results saved to:"
    echo "  - neural_networks/output/$CONFIG_ID/aggregated_results.json"
    echo "  - neural_networks/output/$CONFIG_ID/probing_aggregated.png"
    echo "  - neural_networks/output/$CONFIG_ID/softmax_rank_aggregated.png"
    echo "  - neural_networks/output/$CONFIG_ID/state_metrics_aggregated.png"
    echo "  - neural_networks/output/$CONFIG_ID/training_curves_aggregated.png"
    echo "  - neural_networks/output/$CONFIG_ID/test_metrics_summary.png"
    echo ""
    echo "GPU usage log: log/gpu_probing_test_${SLURM_JOBID}.log"
    echo ""
    
    # Quick verification of aggregated results
    if [ -f "neural_networks/output/$CONFIG_ID/aggregated_results.json" ]; then
        echo "Verification:"
        echo "  - Checking epoch tracking..."
        if grep -q '"epochs".*\[5' "neural_networks/output/$CONFIG_ID/aggregated_results.json"; then
            echo "    ✓ Epochs preserved correctly (starts with 5)"
        else
            echo "    ✗ WARNING: Epoch tracking may be broken"
        fi
        
        if grep -q 'probing_metrics' "neural_networks/output/$CONFIG_ID/aggregated_results.json"; then
            echo "    ✓ Probing metrics found in aggregated results"
        else
            echo "    ✗ WARNING: No probing metrics in aggregated results"
        fi
    fi
else
    echo ""
    echo "✗ FAILED with exit code: $EXIT_CODE"
    echo "Check logs for errors:"
    echo "  - log/probing_test_${SLURM_JOBID}.err"
    echo "  - Individual run logs in neural_networks/output/$CONFIG_ID/individual_runs/*/logs/"
fi

exit $EXIT_CODE