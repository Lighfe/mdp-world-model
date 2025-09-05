#!/bin/bash
#SBATCH --job-name=multi_run_subprocess
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=log/multi_run_subprocess_%j.out
#SBATCH --error=log/multi_run_subprocess_%j.err

echo "========================================================"
echo "MULTI-RUN DRM TRAINING - SUBPROCESS EXECUTION"
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

# Set environment for subprocess approach
export OMP_NUM_THREADS=2  # Lower per-process threading since we have multiple subprocesses
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

# NOTE: No MPS daemon needed for subprocess approach
# Each subprocess gets its own independent GPU context
echo "Using subprocess approach - no MPS daemon needed"
# ============= END H100 OPTIMIZATIONS =============

# Define parameters for the job
CONFIG_PATH="configs/base.yaml"
OUTPUT_DIR="neural_networks/output"
CONFIG_ID="subprocess_test1"
SEEDS="11,12,13,14,15"
DB_PATHS="datasets/results/multi_saddle_7.db,datasets/results/multi_saddle_9.db"
MAX_PARALLEL="10"

echo "========================================================"
echo "Starting subprocess-based multiprocessing training..."
echo "Config: $CONFIG_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "Config ID: $CONFIG_ID"
echo "Seeds: $SEEDS"
echo "Databases: $DB_PATHS"
echo "Max parallel: $MAX_PARALLEL"
echo "========================================================"

# Start GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv --loop=30 > "log/gpu_subprocess_${SLURM_JOBID}.log" &
GPU_MONITOR_PID=$!

# Execute the subprocess-based multi-run training
python neural_networks/train_drm_multi_sub.py "$CONFIG_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --config-id "$CONFIG_ID" \
    --seeds "$SEEDS" \
    --db-paths "$DB_PATHS" \
    --max-parallel "$MAX_PARALLEL"

PYTHON_EXIT_CODE=$?

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null

echo "========================================================"
echo "Subprocess-based training completed at $(date)"
echo "Python exit code: $PYTHON_EXIT_CODE"
echo "Results saved to: $OUTPUT_DIR/$CONFIG_ID"
echo "GPU usage log: log/gpu_subprocess_${SLURM_JOBID}.log"
echo "========================================================"

# Summary information
echo ""
echo "Summary for Configuration: $CONFIG_ID"
echo "================================================"

# Check completion status
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully"
    
    # Count completed runs by checking for final output files
    if [ -d "$OUTPUT_DIR/$CONFIG_ID/individual_runs" ]; then
        completed_count=$(find "$OUTPUT_DIR/$CONFIG_ID/individual_runs" -name "*.png" -name "*final_state_assignment*" | wc -l)
        total_expected=$(echo "$SEEDS" | tr ',' '\n' | wc -l)
        total_expected=$((total_expected * $(echo "$DB_PATHS" | tr ',' '\n' | wc -l)))
        
        echo "  Completed visualizations: $completed_count/$total_expected"
        
        # Check for run summary
        if [ -f "$OUTPUT_DIR/$CONFIG_ID/run_summary.json" ]; then
            echo "  Summary file created: ✓"
        fi
    fi
    
else
    echo "✗ Training failed with exit code: $PYTHON_EXIT_CODE"
fi

# Check average runtime from GPU log
echo ""
echo "GPU Utilization Summary:"
if [ -f "log/gpu_subprocess_${SLURM_JOBID}.log" ]; then
    # Get GPU utilization stats (skip header line)
    avg_util=$(tail -n +2 "log/gpu_subprocess_${SLURM_JOBID}.log" | awk -F',' '{sum+=$3; count++} END {if(count>0) print sum/count; else print 0}')
    max_memory=$(tail -n +2 "log/gpu_subprocess_${SLURM_JOBID}.log" | awk -F',' '{if($1>max) max=$1} END {print max}')
    
    echo "  Average GPU utilization: ${avg_util}%"
    echo "  Peak memory usage: ${max_memory} MiB"
    
    # Compare with expected single-run utilization (~11%)
    if (( $(echo "$avg_util > 20" | bc -l) )); then
        echo "  ✓ GPU utilization significantly improved over single-run baseline"
    else
        echo "  ⚠ GPU utilization similar to single-run baseline (~11%)"
    fi
fi

echo "================================================"

exit $PYTHON_EXIT_CODE