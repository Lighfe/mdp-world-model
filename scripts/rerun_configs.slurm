#!/bin/bash
#SBATCH --job-name=rerun_configs
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=log/rerun_configs_%j.out
#SBATCH --error=log/rerun_configs_%j.err

echo "========================================================"
echo "RE-RUNNING CONFIGS SEQUENTIALLY"
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "========================================================"

# ============= USER CONFIGURATION =============

# EDIT THIS: List of config paths to re-run
CONFIG_PATHS=(
    "neural_networks/output/ablation/ablation_baseline_ds3/configs/config_seed_715_multi_saddle_3.yaml"
    "neural_networks/output/ablation/ablation_baseline_ds8/configs/config_seed_713_multi_saddle_8.yaml"
    "neural_networks/output/ablation/ablation_baseline_ds5/configs/config_seed_713_multi_saddle_5.yaml"
    "neural_networks/output/ablation/ablation_baseline_ds9/configs/config_seed_713_multi_saddle_9.yaml"
    "neural_networks/output/ablation/ablation_temp10.0_ds8/configs/config_seed_713_multi_saddle_8.yaml"
    "neural_networks/output/ablation/ablation_temp10.0_ds3/configs/config_seed_715_multi_saddle_3.yaml"
    "neural_networks/output/ablation/ablation_baseline_ds9/configs/config_seed_714_multi_saddle_9.yaml"
    "neural_networks/output/ablation/ablation_gumbel_false_ds9/configs/config_seed_714_multi_saddle_9.yaml"
    "neural_networks/output/ablation/ablation_gumbel_false_ds9/configs/config_seed_715_multi_saddle_9.yaml"
    "neural_networks/output/ablation/ablation_gumbel_false_ds9/configs/config_seed_717_multi_saddle_9.yaml"
)

# EDIT THIS: New output directory
NEW_OUTPUT_DIR="neural_networks/output/rerun"

# ============= END USER CONFIGURATION =============

echo "Total configs to process: ${#CONFIG_PATHS[@]}"
echo "New output directory: $NEW_OUTPUT_DIR"
echo ""

# Create log directory
mkdir -p log

# Load modules and activate environment
module load anaconda
source activate mdp-world-model

# Explicitly set PATH to use conda environment
export PATH="$CONDA_PREFIX/bin:$PATH"

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# ============= THREAD CONTROL =============
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

# Create new configs directory
mkdir -p "${NEW_OUTPUT_DIR}/configs"

# Change to project directory
cd /p/projects/ou/rd4/bega/mdp/dralle/mdp-world-model

# ============= LOOP THROUGH CONFIGS =============

CURRENT=0
TOTAL=${#CONFIG_PATHS[@]}

for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    echo "========================================================"
    echo "PROCESSING CONFIG $CURRENT/$TOTAL"
    echo "========================================================"
    echo "Config: $CONFIG_PATH"
    
    # Check if config exists
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "ERROR: Config file not found: $CONFIG_PATH"
        echo "Skipping..."
        echo ""
        continue
    fi
    
    # Extract config filename
    CONFIG_FILENAME=$(basename "$CONFIG_PATH")
    NEW_CONFIG_PATH="${NEW_OUTPUT_DIR}/configs/${CONFIG_FILENAME}"
    
    echo "Creating modified config: $NEW_CONFIG_PATH"
    
    # Modify config using Python
    python - <<EOF
import yaml
from pathlib import Path
import re

# Load original config
with open('$CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)

# Extract prefix from original output_dir
# e.g., "neural_networks/output/ablation_vlw10.0_ds8/individual_runs" -> "ablation_vlw10.0_ds8"
original_output_dir = config['meta']['output_dir']
original_run_id = config['meta']['run_id']

# Extract the folder name before /individual_runs
# Pattern: anything between /output/ and /individual_runs (or last folder if no individual_runs)
if '/individual_runs' in original_output_dir:
    # Extract folder between /output/ and /individual_runs
    match = re.search(r'/output/([^/]+)/individual_runs', original_output_dir)
    if match:
        prefix = match.group(1)
    else:
        # Fallback: use last folder before /individual_runs
        prefix = Path(original_output_dir).parent.name
else:
    # Fallback: use last folder in path
    prefix = Path(original_output_dir).name

# Create new run_id with prefix
new_run_id = f"{prefix}_{original_run_id}"

# Modify config
config['meta']['output_dir'] = '$NEW_OUTPUT_DIR'
config['meta']['run_id'] = new_run_id

# Save modified config
Path('$NEW_CONFIG_PATH').parent.mkdir(parents=True, exist_ok=True)
with open('$NEW_CONFIG_PATH', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"  Original run_id: {original_run_id}")
print(f"  Prefix extracted: {prefix}")
print(f"  New run_id: {new_run_id}")
print(f"  Seed: {config['meta']['seed']}")
print(f"  Database: {config['meta']['db_path']}")
print(f"  New output: {config['meta']['output_dir']}")
EOF
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to modify config"
        echo "Skipping..."
        echo ""
        continue
    fi
    
    # Run training
    echo ""
    echo "Starting training..."
    START_TIME=$(date +%s)
    
    python train_drm.py "$NEW_CONFIG_PATH"
    
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo "Training completed in ${DURATION} seconds"
    echo "Exit code: $EXIT_CODE"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Config $CURRENT/$TOTAL completed successfully"
    else
        echo "✗ Config $CURRENT/$TOTAL failed"
    fi
    
    echo ""
done

# ============= FINAL SUMMARY =============

echo "========================================================"
echo "ALL CONFIGS COMPLETED"
echo "========================================================"
echo "Total configs processed: $TOTAL"
echo "Output directory: $NEW_OUTPUT_DIR"
echo "Modified configs saved in: ${NEW_OUTPUT_DIR}/configs/"
echo "End time: $(date)"
echo "========================================================"