#!/bin/bash
#SBATCH --job-name=param_sweep
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=11
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=log/sweep_%j.out
#SBATCH --error=log/sweep_%j.err

# ============= ENVIRONMENT SETUP =============
echo "Job ID: $SLURM_JOBID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================================"

# Load modules and activate environment
module load anaconda
source activate neural_dralle
export PATH="$CONDA_PREFIX/bin:$PATH"

# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import optuna; print(f'Optuna: {optuna.__version__}')"

# ============= GPU SETUP (MPS) =============
echo "Setting up CUDA MPS..."

# Clean up any existing MPS
echo quit | nvidia-cuda-mps-control 2>/dev/null || true

# Create MPS directories
MPS_LOG_DIR="log/mps_${SLURM_JOBID}"
MPS_PIPE_DIR="log/mps_pipes_${SLURM_JOBID}"
mkdir -p "$MPS_LOG_DIR" "$MPS_PIPE_DIR"

# Set MPS environment
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE_DIR"
export CUDA_MPS_LOG_DIRECTORY="$MPS_LOG_DIR"

# Start MPS daemon
nvidia-cuda-mps-control -d
echo "MPS daemon started"

# Verify GPU access
nvidia-smi

# ============= PARAMETER SWEEP SETUP =============
# Configuration - MODIFY THESE PARAMETERS AS NEEDED
SWEEP_CONFIG="configs/sweeps/first_sweep.yaml"
SWEEP_ID="first_sweep_$(date +%Y%m%d_%H%M%S)"
N_TRIALS="100"

echo "========================================================"
echo "Parameter Sweep Configuration:"
echo "  Sweep config: $SWEEP_CONFIG"
echo "  Sweep ID: $SWEEP_ID"
echo "  Number of trials: $N_TRIALS"
echo "  Working directory: $(pwd)"
echo "========================================================"

# Validate sweep config exists
if [ ! -f "$SWEEP_CONFIG" ]; then
    echo "ERROR: Sweep config file not found: $SWEEP_CONFIG"
    exit 1
fi

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
echo "Starting parameter sweep at $(date)..."

# Change to project directory
cd /p/projects/ou/rd4/bega/mdp/dralle/mdp-world-model

# Run the parameter sweep
python scripts/parameter_sweep.py "$SWEEP_CONFIG" \
    --sweep-id "$SWEEP_ID" \
    --n-trials "$N_TRIALS"

SWEEP_EXIT_CODE=$?

# ============= CLEANUP =============
echo "Parameter sweep completed with exit code: $SWEEP_EXIT_CODE"
echo "End time: $(date)"

# Stop monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true
kill $SYSTEM_MONITOR_PID 2>/dev/null || true

# Stop MPS daemon
echo quit | nvidia-cuda-mps-control
echo "MPS daemon stopped"

# ============= SUMMARY =============
echo "========================================================"
echo "PARAMETER SWEEP SUMMARY"
echo "========================================================"
echo "Job ID: $SLURM_JOBID"
echo "Exit code: $SWEEP_EXIT_CODE"
echo "Sweep ID: $SWEEP_ID"
echo "Results location: configs/sweeps/$SWEEP_ID/"
echo "GPU monitoring log: log/gpu_sweep_${SLURM_JOBID}.log"
echo "System monitoring log: log/system_sweep_${SLURM_JOBID}.log"
echo "========================================================"

exit $SWEEP_EXIT_CODE