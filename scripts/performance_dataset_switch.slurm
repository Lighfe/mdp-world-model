#!/bin/bash
#SBATCH --job-name=performance_test_run
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --output=log/performance_test_run_%j.out
#SBATCH --error=log/performance_test_run_%j.err

echo "========================================================"
echo "performance_test RUN - 3 CONFIGS WITH SEEDS 41-55"
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
    echo "✅ CUDA MPS daemon started successfully"
else
    echo "❌ Failed to start CUDA MPS daemon"
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

# ============= MULTI-RUN CONFIGURATION =============
OUTPUT_DIR="neural_networks/output/test_performance_postsweep"
DB_PATHS="datasets/results/multi_saddle_3.db,datasets/results/multi_saddle_8.db"
SEEDS="41,42,43,44,45,46,47,48,49,50,51,52,53,54,55"
MAX_PARALLEL="15"

# Configuration files and their corresponding trial names
declare -a CONFIGS=(
    "scripts/configs/trial_041_ce.yaml"
    "scripts/configs/trial_045_mse.yaml" 
    "scripts/configs/trial_049_kl.yaml"
)
declare -a TRIAL_NAMES=("trial_041_ce" "trial_045_mse" "trial_049_kl")

echo "Multi-Run Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Databases: $DB_PATHS"
echo "  Seeds: $SEEDS"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Configs: ${#CONFIGS[@]}"
echo "  Total seeds per config: 15"
echo "========================================================"

# Start GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv --loop=30 > "log/gpu_multi_config_${SLURM_JOBID}.log" &
GPU_MONITOR_PID=$!

# Change to project directory
cd /p/projects/ou/rd4/bega/mdp/dralle/mdp-world-model

# ============= EXECUTE MULTI-RUN =============
TOTAL_CONFIGS=${#CONFIGS[@]}

echo "Starting multi-run at $(date)..."

# Loop through each configuration sequentially
for i in "${!CONFIGS[@]}"; do
    CONFIG_FILE="${CONFIGS[$i]}"
    TRIAL_NAME="${TRIAL_NAMES[$i]}"
    CURRENT_CONFIG=$((i + 1))
    
    echo ""
    echo "=================================================="
    echo "[$CURRENT_CONFIG/$TOTAL_CONFIGS] Processing configuration: $TRIAL_NAME"
    echo "=================================================="
    echo "  Config file: $CONFIG_FILE"
    echo "  Seeds: $SEEDS (15 seeds)"
    echo "  Databases: $DB_PATHS"
    echo "  Max parallel: $MAX_PARALLEL"
    echo "  Start time: $(date)"
    
    # Execute training with MPS
    python neural_networks/train_drm_multi.py "$CONFIG_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --config-id "$TRIAL_NAME" \
        --seeds "$SEEDS" \
        --db-paths "$DB_PATHS" \
        --max-parallel "$MAX_PARALLEL"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ [$CURRENT_CONFIG/$TOTAL_CONFIGS] Configuration completed successfully: $TRIAL_NAME"
    else
        echo "❌ [$CURRENT_CONFIG/$TOTAL_CONFIGS] Configuration failed: $TRIAL_NAME (exit code: $EXIT_CODE)"
        echo "Continuing with remaining configurations..."
    fi
    
    echo "  End time: $(date)"
    echo "  Results saved to: $OUTPUT_DIR/$TRIAL_NAME/"
    
    # Brief pause between configurations to allow system cleanup
    if [ $CURRENT_CONFIG -lt $TOTAL_CONFIGS ]; then
        echo "  Pausing 10 seconds before next configuration..."
        sleep 10
    fi
done

# ============= MPS CLEANUP =============
echo ""
echo "========================================================"
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
        echo "✅ All MPS processes cleaned up"
    fi
else
    echo "✅ MPS processes already terminated"
fi

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true

echo "========================================================"
echo "MULTI-CONFIG RUN COMPLETED"
echo "========================================================"
echo "Job ID: $SLURM_JOBID"
echo "End time: $(date)"
echo "Total configurations executed: $TOTAL_CONFIGS"
echo ""
echo "Results structure:"
echo "  neural_networks/output/multi_config_run/"
echo "  ├── trial_041_ce/"
echo "  │   └── [15 seed results]"
echo "  ├── trial_045_mse/"
echo "  │   └── [15 seed results]"
echo "  └── trial_049_kl/"
echo "      └── [15 seed results]"
echo ""
echo "Logs:"
echo "  - GPU monitoring: log/gpu_multi_config_${SLURM_JOBID}.log"
echo "  - MPS logs: $MPS_LOG_DIR"
echo "========================================================"