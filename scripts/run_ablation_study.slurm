#!/bin/bash
#SBATCH --job-name=ablation_study
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --output=log/ablation_study_%j.out
#SBATCH --error=log/ablation_study_%j.err

# ============================================================================
# ABLATION STUDY - COMPLETE RUN
# 60 configs × 64 seeds = 3,840 total runs
# train_drm_multi.py handles parallelization (16 at a time) and aggregation
# Expected runtime: ~20 hours
# ============================================================================

echo "========================================================"
echo "ABLATION STUDY - COMPLETE EXECUTION"
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
CONFIG_DIR="scripts/configs/ablation_study"
MANIFEST_FILE="${CONFIG_DIR}/manifest.txt"
MAX_PARALLEL=16

# All 64 seeds as comma-separated string (train_drm_multi.py handles batching)
ALL_SEEDS="700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763"

echo ""
echo "Configuration:"
echo "  Config directory: $CONFIG_DIR"
echo "  Manifest file: $MANIFEST_FILE"
echo "  Seeds: 700-763 (64 total)"
echo "  Max parallel: $MAX_PARALLEL (handled by train_drm_multi.py)"
echo ""

# Verify config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "ERROR: Config directory not found: $CONFIG_DIR"
    exit 1
fi

# Verify manifest exists
if [ ! -f "$MANIFEST_FILE" ]; then
    echo "ERROR: Manifest file not found: $MANIFEST_FILE"
    exit 1
fi

# Extract config files from manifest (skip comments and empty lines)
CONFIG_FILES=($(grep -v '^#' "$MANIFEST_FILE" | grep -v '^##' | grep -v '^$' | grep '\.yaml$'))

echo "Found ${#CONFIG_FILES[@]} config files in manifest"
echo ""

# Verify all config files exist
missing_configs=0
for config_file in "${CONFIG_FILES[@]}"; do
    config_path="${CONFIG_DIR}/${config_file}"
    if [ ! -f "$config_path" ]; then
        echo "WARNING: Config file not found: $config_path"
        missing_configs=$((missing_configs + 1))
    fi
done

if [ $missing_configs -gt 0 ]; then
    echo "ERROR: $missing_configs config files missing"
    exit 1
fi

echo "✓ All ${#CONFIG_FILES[@]} config files verified"
echo ""

# Start GPU monitoring
echo "Starting GPU monitoring..."
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv --loop=10 > "log/gpu_ablation_${SLURM_JOBID}.log" &
GPU_MONITOR_PID=$!
echo "GPU monitoring started (PID: $GPU_MONITOR_PID)"
echo ""

# ============================================================================
# MAIN EXECUTION LOOP
# ============================================================================

TOTAL_CONFIGS=${#CONFIG_FILES[@]}
COMPLETED_CONFIGS=0
FAILED_CONFIGS=0

echo "========================================================"
echo "Starting ablation study at $(date)"
echo "Total configs: $TOTAL_CONFIGS"
echo "Total runs per config: 64 seeds"
echo "Total runs: $((TOTAL_CONFIGS * 64))"
echo "========================================================"
echo ""

for config_file in "${CONFIG_FILES[@]}"; do
    CONFIG_PATH="${CONFIG_DIR}/${config_file}"
    CONFIG_NAME="${config_file%.yaml}"
    
    echo "========================================================"
    echo "[$((COMPLETED_CONFIGS + FAILED_CONFIGS + 1))/$TOTAL_CONFIGS] Processing: $CONFIG_NAME"
    echo "Config: $CONFIG_PATH"
    echo "Started at: $(date)"
    echo "========================================================"
    
    # Extract db_path from config using Python
    DB_PATH=$(python -c "
import yaml
with open('$CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)
print(config['meta']['db_path'])
")
    
    echo "Database: $DB_PATH"
    echo "Seeds: 700-763 (64 total, running 16 in parallel)"
    echo ""
    
    # Verify database exists
    if [ ! -f "$DB_PATH" ]; then
        echo "ERROR: Database not found: $DB_PATH"
        FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
        echo ""
        continue
    fi
    
    # Run training with ALL 64 seeds (train_drm_multi.py handles parallelization)
    python neural_networks/train_drm_multi.py "$CONFIG_PATH" \
        --config-id "ablation_${CONFIG_NAME}" \
        --seeds "$ALL_SEEDS" \
        --db-paths "$DB_PATH" \
        --max-parallel "$MAX_PARALLEL"
    
    EXIT_CODE=$?
    
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Config $CONFIG_NAME completed successfully"
        echo "  - All 64 seeds completed"
        echo "  - Aggregated results saved"
        COMPLETED_CONFIGS=$((COMPLETED_CONFIGS + 1))
    else
        echo "✗ Config $CONFIG_NAME FAILED with exit code: $EXIT_CODE"
        FAILED_CONFIGS=$((FAILED_CONFIGS + 1))
    fi
    
    echo ""
    echo "Progress: $COMPLETED_CONFIGS completed, $FAILED_CONFIGS failed, $((TOTAL_CONFIGS - COMPLETED_CONFIGS - FAILED_CONFIGS)) remaining"
    
    # Estimate remaining time (assume 20 min per config)
    remaining_configs=$((TOTAL_CONFIGS - COMPLETED_CONFIGS - FAILED_CONFIGS))
    if [ $COMPLETED_CONFIGS -gt 0 ]; then
        # Use actual average time if we have data
        echo "Estimated time remaining: ~$((20 * remaining_configs)) minutes"
    fi
    echo ""
done

# ============================================================================
# CLEANUP AND SUMMARY
# ============================================================================

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null
echo "GPU monitoring stopped"

# Stop MPS daemon
echo ""
echo "Stopping CUDA MPS..."
echo quit | nvidia-cuda-mps-control

echo ""
echo "========================================================"
echo "ABLATION STUDY COMPLETED"
echo "Job finished at $(date)"
echo "========================================================"
echo ""
echo "Summary:"
echo "  Total configs: $TOTAL_CONFIGS"
echo "  Completed: $COMPLETED_CONFIGS"
echo "  Failed: $FAILED_CONFIGS"
echo "  Success rate: $(echo "scale=1; $COMPLETED_CONFIGS * 100 / $TOTAL_CONFIGS" | bc)%"
echo ""
echo "Total successful runs: $((COMPLETED_CONFIGS * 64))"
echo ""
echo "Logs:"
echo "  Main log: log/ablation_study_${SLURM_JOBID}.out"
echo "  Error log: log/ablation_study_${SLURM_JOBID}.err"
echo "  GPU usage: log/gpu_ablation_${SLURM_JOBID}.log"
echo "  MPS logs: ${MPS_LOG_DIR}/"
echo ""
echo "Results structure:"
echo "  Each config output: neural_networks/output/ablation_<CONFIG_NAME>/"
echo "    - aggregated_results.json (stats across all 64 seeds)"
echo "    - *_aggregated.png (visualization plots)"
echo "    - individual_runs/ (individual seed results)"
echo "    - run_summary.json (execution details)"
echo "========================================================"

if [ $FAILED_CONFIGS -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! All configs completed successfully!"
    exit 0
else
    echo ""
    echo "⚠️  WARNING: $FAILED_CONFIGS configs failed"
    exit 1
fi