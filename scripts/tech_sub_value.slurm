#!/bin/bash
#SBATCH --job-name=tech_sub_value
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=16           # 16 parallel tasks (each runs 5 configs)
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=log/tech_sub_value_%j.out
#SBATCH --error=log/tech_sub_value_%j.err

# ============================================================================
# PARAMETER SWEEP: tech_sub_value
# 
# Experiment: entropy_reg × value_loss_weight × value_method × seed
# Total configs: 80 (2 × 4 × 2 × 5)
# Parallelization: 16 tasks, each running 5 configs sequentially
# ============================================================================

EXPERIMENT_NAME="tech_sub_value"
OUTPUT_BASE="neural_networks/output"

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

# CRITICAL: Fix PATH to ensure correct Python is used
export PATH="$CONDA_PREFIX/bin:$PATH"

# Verify environment
echo "Python executable: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Set standard environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# H100-specific optimizations (adjust if using different GPU)
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_USE_CUDA_DSA=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Create MPS directories with proper permissions
MPS_LOG_DIR="log/mps_${SLURM_JOBID}"
MPS_PIPE_DIR="log/mps_pipes_${SLURM_JOBID}"
mkdir -p "$MPS_LOG_DIR" "$MPS_PIPE_DIR"

# Set MPS environment variables BEFORE starting daemon
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR"
export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR"

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
NUM_CONFIGS=$("$CONDA_PREFIX/bin/python" -c "import json; print(json.load(open('$METADATA_FILE'))['num_configs'])")

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
    CONFIG_FILE=$("$CONDA_PREFIX/bin/python" -c "import json; print(json.load(open('$METADATA_FILE'))['config_files'][$CONFIG_INDEX])")
    RUN_NAME=$("$CONDA_PREFIX/bin/python" -c "import json; print(json.load(open('$METADATA_FILE'))['run_names'][$CONFIG_INDEX])")
    
    echo "[Task $TASK_ID] Starting config $CONFIG_NUM/$NUM_CONFIGS: $RUN_NAME"
    echo "[Task $TASK_ID] Config file: $CONFIG_FILE"
    
    START_TIME=$(date +%s)
    
    # Run training with explicit conda python
    "$CONDA_PREFIX/bin/python" neural_networks/train_drm.py "$CONFIG_FILE" 2>&1 | sed "s/^/[Task $TASK_ID] /"
    
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
srun --output=log/tech_sub_value_task_${SLURM_JOBID}_%t.out \
     --error=log/tech_sub_value_task_${SLURM_JOBID}_%t.err \
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
echo "Task logs: log/tech_sub_value_task_${SLURM_JOBID}_*.out"
echo "========================================================"

# Print summary
echo ""
echo "Summary for Experiment: $EXPERIMENT_NAME"
echo "================================================"

# Check for output in multiple possible locations
RUNS_DIR="${OUTPUT_BASE}/${EXPERIMENT_NAME}/runs"
EXPERIMENT_DIR="${OUTPUT_BASE}/${EXPERIMENT_NAME}"

if [ -d "$RUNS_DIR" ]; then
    # Count completed runs by looking for model files
    COMPLETED_RUNS=$(find "$RUNS_DIR" -type f -name "*.pt" 2>/dev/null | wc -l)
    NUM_RUN_DIRS=$(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    
    echo "Run directories found: $NUM_RUN_DIRS"
    echo "Model files (.pt) found: $COMPLETED_RUNS"
    echo "Expected runs: $NUM_CONFIGS"
    
    if [ $NUM_RUN_DIRS -eq $NUM_CONFIGS ]; then
        echo "Status: ✓ All run directories created"
    else
        echo "Status: ⚠ Only $NUM_RUN_DIRS/$NUM_CONFIGS run directories found"
    fi
elif [ -d "$EXPERIMENT_DIR" ]; then
    echo "Experiment directory exists: $EXPERIMENT_DIR"
    echo "But runs subdirectory not found: $RUNS_DIR"
    echo ""
    echo "This could mean:"
    echo "  - Training hasn't started yet"
    echo "  - Training failed early"
    echo "  - Output directory structure is different than expected"
    echo ""
    echo "Check task logs for details: log/tech_sub_value_task_${SLURM_JOBID}_*.out"
else
    echo "Experiment directory not found: $EXPERIMENT_DIR"
    echo "Training may have failed to start."
    echo "Check main log: log/tech_sub_value_${SLURM_JOBID}.out"
fi

echo "================================================"

exit $SRUN_EXIT_CODE