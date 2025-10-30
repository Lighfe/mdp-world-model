#!/bin/bash
#SBATCH --job-name=tech_sub
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=34G
#SBATCH --output=log/tech_sub_%j.out
#SBATCH --error=log/tech_sub_%j.err

echo "========================================================"
echo "TECH SUBSTITUTION EXPERIMENTS"
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

# Set standard environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# H100 optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_USE_CUDA_DSA=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Starting NVIDIA MPS..."
nvidia-cuda-mps-control -d
sleep 2

# Configuration
BASE_CONFIG="scripts/configs/tech_sub_base.yaml"
OUTPUT_DIR="tech_substitution_results"
CONFIG_ID="tech_sub_configs"

echo ""
echo "Experiment Configuration:"
echo "  Base config: $BASE_CONFIG"
echo "  Output dir: $OUTPUT_DIR"
echo "  Seeds: 101-110"
echo "  Value loss weights: 0.5, 1.0, 2.0, 3.0, 4.0"
echo "  Total runs: 50"
echo ""

# Verify base config exists
if [ ! -f "$BASE_CONFIG" ]; then
    echo "ERROR: Base config not found: $BASE_CONFIG"
    exit 1
fi

# Generate all config files using the existing utility function
echo "Generating config files..."
python3 << EOFPYTHON
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neural_networks.utils import generate_config_combinations

# Define parameter grid
override_params = {
    "meta.seed": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "loss.value_loss_weight": [0.5, 1.0, 2.0, 3.0, 4.0]
}

# Generate configs
configs = generate_config_combinations(
    base_config_path="${BASE_CONFIG}",
    config_id="${CONFIG_ID}",
    override_params=override_params,
    output_configs_dir="${OUTPUT_DIR}"
)

print(f"Successfully generated {len(configs)} config files")
EOFPYTHON

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to generate config files"
    exit 1
fi

echo "Config generation complete!"
echo ""

# Start GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv --loop=10 > "log/gpu_tech_sub_${SLURM_JOBID}.log" &
GPU_MONITOR_PID=$!

# Create training script for parallel execution
TRAINING_SCRIPT="run_tech_sub_${SLURM_JOBID}.sh"
cat > "$TRAINING_SCRIPT" << 'EOFSCRIPT'
#!/bin/bash

# Get task ID (0-9) which maps to seed (101-110)
TASK_ID=$SLURM_PROCID
SEED=$((TASK_ID + 101))

export PYTHONUNBUFFERED=1

echo "[Task $TASK_ID] Starting Seed $SEED at $(date)"

CONFIG_DIR="tech_substitution_results/tech_sub_configs"

# Find all config files for this seed
# The generate_config_combinations creates files like: config_seed101_vlw0.5.yaml
CONFIGS=($(ls ${CONFIG_DIR}/config_*seed${SEED}_*.yaml 2>/dev/null | sort))

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "[Task $TASK_ID] ERROR: No config files found for seed $SEED in $CONFIG_DIR"
    exit 1
fi

echo "[Task $TASK_ID] Found ${#CONFIGS[@]} configs for seed $SEED"

task_completed=0
task_failed=0

# Run each config
for CONFIG_FILE in "${CONFIGS[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
    
    echo "[Task $TASK_ID] ----------------------------------------"
    echo "[Task $TASK_ID] Starting: $CONFIG_NAME"
    echo "[Task $TASK_ID] Config: $CONFIG_FILE"
    echo "[Task $TASK_ID] Time: $(date)"
    
    # Run training
    python -u neural_networks/train_drm.py "$CONFIG_FILE" 2>&1
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[Task $TASK_ID] ✓ SUCCESS: $CONFIG_NAME"
        task_completed=$((task_completed + 1))
    else
        echo "[Task $TASK_ID] ✗ FAILED: $CONFIG_NAME (exit: $EXIT_CODE)"
        task_failed=$((task_failed + 1))
    fi
done

echo ""
echo "[Task $TASK_ID] ========================================"
echo "[Task $TASK_ID] Task Summary for Seed $SEED"
echo "[Task $TASK_ID] Completed: $task_completed/${#CONFIGS[@]}"
echo "[Task $TASK_ID] Failed: $task_failed/${#CONFIGS[@]}"
echo "[Task $TASK_ID] Finished at $(date)"
echo "[Task $TASK_ID] ========================================"
EOFSCRIPT

chmod +x "$TRAINING_SCRIPT"

echo "========================================================"
echo "Launching 10 parallel tasks..."
echo "========================================================"

# Launch parallel training
srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=$SLURM_NTASKS \
     --output="log/tech_sub_task_%j_%t.out" \
     --error="log/tech_sub_task_%j_%t.err" \
     --unbuffered \
     ./"$TRAINING_SCRIPT"

SRUN_EXIT_CODE=$?

# Cleanup
rm -f "$TRAINING_SCRIPT"
kill $GPU_MONITOR_PID 2>/dev/null
echo quit | nvidia-cuda-mps-control

echo ""
echo "========================================================"
echo "Job completed at $(date)"
echo "Exit code: $SRUN_EXIT_CODE"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================================"

exit $SRUN_EXIT_CODE