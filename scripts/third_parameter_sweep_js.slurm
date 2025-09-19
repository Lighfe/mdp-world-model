#!/bin/bash
#SBATCH --job-name=grid_sweep_js
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --output=log/grid_sweep_js_%j.out
#SBATCH --error=log/grid_sweep_js_%j.err

echo "========================================================"
echo "GRID + BAYESIAN PARAMETER SWEEP - JS DIVERGENCE"
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
python -c "import optuna; print(f'Optuna {optuna.__version__} available')"

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

# ============= PARAMETER SWEEP CONFIGURATION =============
# MODIFY THESE PARAMETERS AS NEEDED
SWEEP_CONFIG="scripts/configs/sweeps/third_sweep_js.yaml"
SWEEP_ID="third_sweep_js"
N_TRIALS="100"
SAMPLER_TYPE="explorative"  # Options: explorative, balanced, exploitative

echo "========================================================"
echo "Grid + Bayesian Optimization Parameter Sweep Configuration:"
echo "  Sweep config: $SWEEP_CONFIG"
echo "  Sweep ID: $SWEEP_ID"
echo "  Number of trials: $N_TRIALS"
echo "  Sampler type: $SAMPLER_TYPE (with grid + Bayesian optimization)"
echo "  Working directory: $(pwd)"
echo "  Loss function: JS Divergence"
echo "  Value method: angle (fixed)"
echo "  Encoder methods: 2 (he vs xavier_uniform)"
echo "  Trials per encoder: 50"
echo "  Learning rate range: 1e-4 to 1e-3"
echo "  Num states: [4, 6, 8] (step: 2)"
echo "========================================================"

# Validate sweep config exists
if [ ! -f "$SWEEP_CONFIG" ]; then
    echo "ERROR: Sweep config file not found: $SWEEP_CONFIG"
    exit 1
fi

echo "Sweep output will be organized in: neural_networks/output/$SWEEP_ID/"

# ============= MONITORING SETUP =============
# Start GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv --loop=60 > "log/gpu_sweep_${SLURM_JOBID}.log" &
GPU_MONITOR_PID=$!

# Start system monitoring
(while true; do
    echo "$(date): CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}'), Memory: $(free -h | grep Mem | awk '{print $3"/"$2}')"
    sleep 300  # Every 5 minutes
done) > "log/system_sweep_${SLURM_JOBID}.log" &
SYSTEM_MONITOR_PID=$!

# ============= RUN PARAMETER SWEEP =============
echo "Starting grid + Bayesian optimization parameter sweep at $(date)..."

# Change to project directory
cd /p/projects/ou/rd4/bega/mdp/dralle/mdp-world-model

# Run the parameter sweep with grid + Bayesian optimization
python scripts/parameter_sweep.py "$SWEEP_CONFIG" \
    --sweep-id "$SWEEP_ID" \
    --n-trials "$N_TRIALS" \
    --sampler-type "$SAMPLER_TYPE"

SWEEP_EXIT_CODE=$?

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

# ============= CLEANUP MONITORING =============
echo "Stopping monitoring processes..."

# Stop monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true
kill $SYSTEM_MONITOR_PID 2>/dev/null || true

# ============= FINAL SUMMARY =============
echo "========================================================"
echo "GRID + BAYESIAN PARAMETER SWEEP COMPLETED"
echo "========================================================"
echo "Job ID: $SLURM_JOBID"
echo "End time: $(date)"
echo "Exit code: $SWEEP_EXIT_CODE"
echo "Sweep ID: $SWEEP_ID"
echo "Results directory: neural_networks/output/$SWEEP_ID/"
echo "  - Trial outputs: neural_networks/output/$SWEEP_ID/trial_*/"
echo "  - Config files: neural_networks/output/$SWEEP_ID/configs/"
echo "  - Optuna database: neural_networks/output/$SWEEP_ID/optuna_study.db"
echo "  - Results summary: neural_networks/output/$SWEEP_ID/sweep_results.json"
echo "Logs:"
echo "  - GPU monitoring: log/gpu_sweep_${SLURM_JOBID}.log"
echo "  - System monitoring: log/system_sweep_${SLURM_JOBID}.log"
echo "  - MPS logs: $MPS_LOG_DIR"
echo "========================================================"

# GPU utilization summary
if [ -f "log/gpu_sweep_${SLURM_JOBID}.log" ]; then
    avg_util=$(tail -n +2 "log/gpu_sweep_${SLURM_JOBID}.log" | awk -F',' '{sum+=$3; count++} END {if(count>0) print sum/count; else print 0}')
    max_memory=$(tail -n +2 "log/gpu_sweep_${SLURM_JOBID}.log" | awk -F',' '{if($1>max) max=$1} END {print max}')
    
    echo "GPU Utilization Summary:"
    echo "  Average GPU utilization: ${avg_util}%"
    echo "  Peak memory usage: ${max_memory} MiB"
    
    # Expected improvement with MPS and grid+Bayesian optimization
    if (( $(echo "$avg_util > 40" | bc -l) )); then
        echo "  ✓ Good GPU utilization with MPS and grid+Bayesian optimization!"
    else
        echo "  ⚠ GPU utilization lower than expected - check trial settings"
    fi
fi

# Grid + Bayesian optimization analysis
echo ""
echo "Grid + Bayesian Optimization Summary:"
echo "  Encoder initialization methods explored: 2"
echo "  Trials per encoder method: 50"
echo "  Fixed parameters:"
echo "    - Loss function: JS divergence"
echo "    - Value method: angle"
echo "    - Learning rate range: 1e-4 to 1e-3"
echo "    - Num states: [4, 6, 8] (step: 2)"
echo "  Encoder methods:"
echo "    1. he initialization (trials 0-49)"
echo "    2. xavier_uniform initialization (trials 50-99)"
echo ""
echo "Next Steps:"
echo "  1. Review results in neural_networks/output/$SWEEP_ID/"
echo "  2. Compare TPE-optimized continuous parameters for each encoder method"
echo "  3. Determine which encoder initialization + optimal hyperparams works best"
echo "  4. Consider focused follow-up sweeps on the winning combination"
echo ""
echo "Analysis Commands:"
echo "  # Load and analyze Optuna study:"
echo "  python -c \"import optuna; study = optuna.load_study(study_name='third_sweep_js', storage='sqlite:///neural_networks/output/$SWEEP_ID/optuna_study.db'); print('Best trial:', study.best_trial)\""

exit $SWEEP_EXIT_CODE