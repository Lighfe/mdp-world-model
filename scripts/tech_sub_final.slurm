#!/bin/bash
#SBATCH --job-name=tech_sub_final
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10           # Number of parallel tasks (adjust based on your needs)
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=log/sweep_%A_%a.out
#SBATCH --error=log/sweep_%A_%a.err

# ============================================================================
# PARAMETER SWEEP SLURM TEMPLATE
# 
# This script runs multiple training jobs in parallel using configs generated
# by create_sweep_configs.py
#
# Usage:
#   1. Generate configs: python create_sweep_configs.py --experiment <name> ...
#   2. Edit EXPERIMENT_NAME below to match your experiment
#   3. Adjust --ntasks based on desired parallelism
#   4. Submit: sbatch run_param_sweep.slurm
# ============================================================================

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================
EXPERIMENT_NAME="tech_sub_vlw_sweep"  # Must match the experiment name used in create_sweep_configs.py
OUTPUT_BASE="neural_networks/output"

# ============================================================================
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
# ============================================================================

echo "========================================================"
echo "PARAMETER SWEEP: $EXPERIMENT_NAME"
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "GPU Info: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "========================================================"

# Create log directory
mkdir -p log

# Load modules and activate environment
module load anaconda
source activate mdp-world-model

# Set standard environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# H100-specific optimizations (adjust if using different GPU)
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_USE_CUDA_DSA=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Start MPS daemon for better GPU sharing
echo "Starting NVIDIA MPS for better GPU sharing..."
nvidia-cuda-mps-control -d
sleep 2

if nvidia-smi | grep -q "MPS"; then
    echo "✓ MPS daemon started successfully"
else
    echo "⚠ MPS may not be running, continuing anyway"
fi

# Read metadata
METADATA_FILE="${OUTPUT_BASE}/${EXPERIMENT_NAME}/sweep_metadata.json"

if [ ! -f "$METADATA_FILE" ]; then
    echo "ERROR: Metadata file not found: $METADATA_FILE"
    echo "Did you run create_sweep_configs.py first?"
    exit 1
fi

# Get number of configs
NUM_CONFIGS=$(python3 -c "import json; print(json.load(open('$METADATA_FILE'))['num_configs'])")

echo "Found $NUM_CONFIGS configurations to run"
echo "Running with $SLURM_NTASKS parallel tasks"

# Start GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv --loop=10 > "log/gpu_${SLURM_JOBID}.log" &
GPU_MONITOR_PID=$!

# Create training script
TRAINING_SCRIPT="sweep_${SLURM_JOBID}.sh"
cat > "$TRAINING_SCRIPT" << 'EOFSCRIPT'
#!/bin/bash

# Get task ID (0 to SLURM_NTASKS-1)
TASK_ID=$SLURM_PROCID

export PYTHONUNBUFFERED=1

# Read metadata
METADATA_FILE="${OUTPUT_BASE}/${EXPERIMENT_NAME}/sweep_metadata.json"

# Calculate which configs this task should run
# Distribute configs evenly across tasks
CONFIGS_PER_TASK=$(( (NUM_CONFIGS + SLURM_NTASKS - 1) / SLURM_NTASKS ))
START_CONFIG=$(( TASK_ID * CONFIGS_PER_TASK + 1 ))
END_CONFIG=$(( START_CONFIG + CONFIGS_PER_TASK - 1 ))

# Don't exceed total number of configs
if [ $END_CONFIG -gt $NUM_CONFIGS ]; then
    END_CONFIG=$NUM_CONFIGS
fi

echo "[Task $TASK_ID] Assigned configs $START_CONFIG to $END_CONFIG"

# Run assigned configs sequentially
for CONFIG_NUM in $(seq $START_CONFIG $END_CONFIG); do
    # Get config file path from metadata (arrays are 0-indexed in JSON)
    CONFIG_INDEX=$(( CONFIG_NUM - 1 ))
    CONFIG_FILE=$(python3 -c "import json; print(json.load(open('$METADATA_FILE'))['config_files'][$CONFIG_INDEX])")
    RUN_NAME=$(python3 -c "import json; print(json.load(open('$METADATA_FILE'))['run_names'][$CONFIG_INDEX])")
    
    echo "[Task $TASK_ID] Starting config $CONFIG_NUM/$NUM_CONFIGS: $RUN_NAME"
    echo "[Task $TASK_ID] Config file: $CONFIG_FILE"
    
    START_TIME=$(date +%s)
    
    # Run training
    python neural_networks/train_drm.py "$CONFIG_FILE" 2>&1 | sed "s/^/[Task $TASK_ID] /"
    
    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[Task $TASK_ID] ✓ Config $CONFIG_NUM completed in ${DURATION}s"
    else
        echo "[Task $TASK_ID] ✗ Config $CONFIG_NUM failed with exit code $EXIT_CODE"
    fi
done

echo "[Task $TASK_ID] All assigned configs completed"
EOFSCRIPT

chmod +x "$TRAINING_SCRIPT"

# Export variables for training script
export EXPERIMENT_NAME
export OUTPUT_BASE
export NUM_CONFIGS

# Run parallel training
echo "Launching $SLURM_NTASKS parallel tasks..."
srun --output=log/sweep_task_${SLURM_JOBID}_%t.out \
     --error=log/sweep_task_${SLURM_JOBID}_%t.err \
     bash "$TRAINING_SCRIPT"

SRUN_EXIT_CODE=$?

# Clean up
rm -f "$TRAINING_SCRIPT"

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null

# Stop MPS daemon
echo "Stopping NVIDIA MPS..."
echo quit | nvidia-cuda-mps-control

echo "========================================================"
echo "Parameter sweep completed at $(date)"
echo "Experiment: $EXPERIMENT_NAME"
echo "srun exit code: $SRUN_EXIT_CODE"
echo "Results saved to: ${OUTPUT_BASE}/${EXPERIMENT_NAME}/runs/"
echo "GPU usage log: log/gpu_${SLURM_JOBID}.log"
echo "Task logs: log/sweep_task_${SLURM_JOBID}_*.out"
echo "========================================================"

# Print summary
echo ""
echo "Summary for Experiment: $EXPERIMENT_NAME"
echo "================================================"

# Count completed runs
RUNS_DIR="${OUTPUT_BASE}/${EXPERIMENT_NAME}/runs"
if [ -d "$RUNS_DIR" ]; then
    COMPLETED_RUNS=$(find "$RUNS_DIR" -name "*.pt" -o -name "final_model.pt" | wc -l)
    echo "Completed runs with saved models: $COMPLETED_RUNS"
    echo "Expected runs: $NUM_CONFIGS"
    
    if [ $COMPLETED_RUNS -eq $NUM_CONFIGS ]; then
        echo "Status: ✓ All runs completed successfully"
    else
        echo "Status: ⚠ Some runs may have failed"
        echo "Check individual task logs for details"
    fi
else
    echo "Warning: Runs directory not found: $RUNS_DIR"
fi

echo "================================================"

exit $SRUN_EXIT_CODE