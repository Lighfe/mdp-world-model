#!/bin/bash
#SBATCH --job-name=perf_validation
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --output=log/perf_validation_%j.out
#SBATCH --error=log/perf_validation_%j.err

echo "========================================================"
echo "PERFORMANCE VALIDATION - 3 CONFIGS × 3 SEED RANGES"
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

# ============= VALIDATION CONFIGURATION =============
OUTPUT_DIR="neural_networks/output/performance_validation"
DB_PATHS="datasets/results/multi_saddle_5.db,datasets/results/multi_saddle_7.db,datasets/results/multi_saddle_9.db"
MAX_PARALLEL="15"

# Configuration files and their corresponding trial names
declare -a CONFIGS=("configs/trial_182.yaml" "configs/trial_298.yaml" "configs/trial_346.yaml")
declare -a TRIAL_NAMES=("trial_182" "trial_298" "trial_346")

# Seed ranges to test
declare -a SEED_RANGES=("21,22,23,24,25,26,27,28,29,30" "31,32,33,34,35,36,37,38,39,40" "41,42,43,44,45,46,47,48,49,50")
declare -a RANGE_NAMES=("seeds_21_30" "seeds_31_40" "seeds_41_50")

echo "Validation Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Databases: $DB_PATHS"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Configs: ${#CONFIGS[@]}"
echo "  Seed ranges: ${#SEED_RANGES[@]}"
echo "  Total multi_runs: $((${#CONFIGS[@]} * ${#SEED_RANGES[@]}))"
echo "========================================================"

# Start GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv --loop=30 > "log/gpu_validation_${SLURM_JOBID}.log" &
GPU_MONITOR_PID=$!

# Change to project directory
cd /p/projects/ou/rd4/bega/mdp/dralle/mdp-world-model

# ============= EXECUTE VALIDATION RUNS =============
TOTAL_RUNS=$((${#CONFIGS[@]} * ${#SEED_RANGES[@]}))
CURRENT_RUN=0

echo "Starting validation runs at $(date)..."

# Loop through each configuration
for i in "${!CONFIGS[@]}"; do
    CONFIG_FILE="${CONFIGS[$i]}"
    TRIAL_NAME="${TRIAL_NAMES[$i]}"
    
    echo ""
    echo "=================================================="
    echo "Processing configuration: $TRIAL_NAME ($CONFIG_FILE)"
    echo "=================================================="
    
    # Loop through each seed range for this configuration
    for j in "${!SEED_RANGES[@]}"; do
        SEEDS="${SEED_RANGES[$j]}"
        RANGE_NAME="${RANGE_NAMES[$j]}"
        CURRENT_RUN=$((CURRENT_RUN + 1))
        
        CONFIG_ID="${TRIAL_NAME}_${RANGE_NAME}"
        
        echo ""
        echo "[$CURRENT_RUN/$TOTAL_RUNS] Running multi_run: $CONFIG_ID"
        echo "  Config: $CONFIG_FILE"
        echo "  Seeds: $SEEDS"
        echo "  Range: $RANGE_NAME"
        echo "  Start time: $(date)"
        
        # Execute training with MPS
        python neural_networks/train_drm_multi.py "$CONFIG_FILE" \
            --output-dir "$OUTPUT_DIR/$TRIAL_NAME" \
            --config-id "$RANGE_NAME" \
            --seeds "$SEEDS" \
            --db-paths "$DB_PATHS" \
            --max-parallel "$MAX_PARALLEL"
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ [$CURRENT_RUN/$TOTAL_RUNS] Multi_run completed successfully: $CONFIG_ID"
        else
            echo "❌ [$CURRENT_RUN/$TOTAL_RUNS] Multi_run failed: $CONFIG_ID (exit code: $EXIT_CODE)"
            echo "Continuing with remaining runs..."
        fi
        
        echo "  End time: $(date)"
        echo "  Results saved to: $OUTPUT_DIR/$TRIAL_NAME/$RANGE_NAME/"
        
        # Brief pause between runs to allow system cleanup
        sleep 5
    done
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
echo "PERFORMANCE VALIDATION COMPLETED"
echo "========================================================"
echo "Job ID: $SLURM_JOBID"
echo "End time: $(date)"
echo "Total multi_runs executed: $TOTAL_RUNS"
echo ""
echo "Results structure:"
echo "  neural_networks/output/performance_validation/"
echo "  ├── trial_182/"
echo "  │   ├── seeds_21_30/"
echo "  │   ├── seeds_31_40/"
echo "  │   └── seeds_41_50/"
echo "  ├── trial_298/"
echo "  │   ├── seeds_21_30/"
echo "  │   ├── seeds_31_40/"
echo "  │   └── seeds_41_50/"
echo "  └── trial_346/"
echo "      ├── seeds_21_30/"
echo "      ├── seeds_31_40/"
echo "      └── seeds_41_50/"
echo ""
echo "Logs:"
echo "  - GPU monitoring: log/gpu_validation_${SLURM_JOBID}.log"
echo "  - MPS logs: $MPS_LOG_DIR"
echo "========================================================"