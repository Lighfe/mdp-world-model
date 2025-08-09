#!/bin/bash
#SBATCH --job-name=h100_parallel_test
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --output=log/h100_test_%A.out
#SBATCH --error=log/h100_test_%A.err

echo "========================================================"
echo "H100 PARALLEL TEST - Dataset 8, 10 Seeds"
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

# ============= H100-SPECIFIC OPTIMIZATIONS =============
# Enable asynchronous GPU operations
export CUDA_LAUNCH_BLOCKING=0

# Set architecture for H100 (compute capability 9.0)
export TORCH_CUDA_ARCH_LIST="9.0"

# Better error messages for debugging
export TORCH_USE_CUDA_DSA=1

# Enable TF32 for tensor operations
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Start MPS daemon for better GPU sharing among parallel tasks
echo "Starting NVIDIA MPS for better GPU sharing..."
nvidia-cuda-mps-control -d
sleep 2  # Give MPS time to start

# Verify MPS is running
if nvidia-smi | grep -q "MPS"; then
    echo "✓ MPS daemon started successfully"
else
    echo "⚠ MPS may not be running, continuing anyway"
fi
# ============= END H100 OPTIMIZATIONS =============

# Fixed configuration - Dataset 8 for all tasks
DATASET_NUM=8
DB_PATH="datasets/results/multi_saddle_${DATASET_NUM}.db"

echo "Test Configuration:"
echo "  Dataset: multi_saddle_${DATASET_NUM}.db (fixed for all tasks)"
echo "  Learning rates: 1e-5 and 5e-5 (sequential)"
echo "  Optimizer: adam"
echo "  Value loss weight: 0.3"
echo "  Seeds: 11-20 (10 seeds in parallel)"
echo "  Epochs: 100"
echo "  H100 optimizations: TF32 enabled, MPS active"

# Verify dataset exists
if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Dataset not found: $DB_PATH"
    exit 1
fi

# Create output directory
OUTPUT_DIR="neural_networks/output/h100_parallel_test"
mkdir -p "$OUTPUT_DIR"

# Start GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv --loop=10 > "log/gpu_usage_h100_${SLURM_JOBID}.log" &
GPU_MONITOR_PID=$!

# Function to run experiments with a specific learning rate
run_lr_experiment() {
    local LR=$1
    local LR_LABEL=$2
    
    echo "========================================================"
    echo "Starting experiments with LR=${LR} (${LR_LABEL}) at $(date)"
    echo "========================================================"
    
    # Create training script for this LR
    TRAINING_SCRIPT="h100_train_${LR_LABEL}_${SLURM_JOBID}.sh"
    cat > "$TRAINING_SCRIPT" << EOF
#!/bin/bash

# Get task ID (0-9) and calculate seed (11-20)
TASK_ID=\$SLURM_PROCID
SEED=\$((TASK_ID + 11))

export PYTHONUNBUFFERED=1

echo "[Task \$TASK_ID] Starting with LR=${LR}, seed \$SEED at \$(date)"

# Base command with H100 optimizations built in
CMD="python -u neural_networks/train_drm.py ${DB_PATH} \\
    --system_type=saddle_system \\
    --batch_size=256 \\
    --num_states=4 \\
    --hidden_dim=32 \\
    --state_loss_weight=1.0 \\
    --value_loss_weight=0.3 \\
    --predictor_type=standard \\
    --val_size=1000 \\
    --test_size=1000 \\
    --probing_size=2000 \\
    --entropy_decay_proportion=0.5 \\
    --use_warmup \\
    --warmup_epochs=10 \\
    --use_target_encoder \\
    --ema_decay=0.9 \\
    --use_entropy_reg \\
    --entropy_weight=1.0 \\
    --state_loss_type=kl_div \\
    --use_gumbel \\
    --min_temp=0.5 \\
    --initial_temp=3.0 \\
    --encoder_init_method=he \\
    --optimizer=adam \\
    --scheduler_type=fixed \\
    --lr=${LR} \\
    --epochs=100 \\
    --seed=\$SEED \\
    --output_dir=${OUTPUT_DIR} \\
    --run_id=h100_ds${DATASET_NUM}_lr${LR_LABEL}_seed\${SEED}"

echo "[Task \$TASK_ID] Executing command..."
\$CMD 2>&1

if [ \$? -eq 0 ]; then
    echo "[Task \$TASK_ID] ✓ Completed successfully (seed \$SEED)"
else
    echo "[Task \$TASK_ID] ✗ Failed (seed \$SEED)"
fi

echo "[Task \$TASK_ID] Finished at \$(date)"
EOF

    chmod +x "$TRAINING_SCRIPT"
    
    # Launch parallel training for this LR
    echo "Launching 10 parallel processes for LR=${LR}..."
    srun --ntasks=$SLURM_NTASKS \
         --ntasks-per-node=$SLURM_NTASKS \
         --output="log/h100_task_${LR_LABEL}_%A_%t.out" \
         --error="log/h100_task_${LR_LABEL}_%A_%t.err" \
         --unbuffered \
         ./"$TRAINING_SCRIPT"
    
    SRUN_EXIT_CODE=$?
    
    # Clean up
    rm -f "$TRAINING_SCRIPT"
    
    echo "Completed LR=${LR} experiments with exit code: $SRUN_EXIT_CODE"
    echo ""
    
    # Brief pause between learning rates
    sleep 10
}

# Run experiments sequentially for each learning rate
run_lr_experiment "1e-5" "1e-5"
run_lr_experiment "5e-5" "5e-5"

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null

# Stop MPS daemon
echo "Stopping NVIDIA MPS..."
echo quit | nvidia-cuda-mps-control

echo "========================================================"
echo "H100 PARALLEL TEST completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "GPU usage log: log/gpu_usage_h100_${SLURM_JOBID}.log"
echo "Task logs: log/h100_task_*_${SLURM_JOBID}_*.out"
echo "========================================================"

# Quick summary of results
echo ""
echo "Quick Summary:"
echo "--------------"
for lr in "1e-5" "5e-5"; do
    echo "LR=$lr:"
    completed=$(ls -1 ${OUTPUT_DIR}/drm_final_h100_ds${DATASET_NUM}_lr${lr}_seed*.pt 2>/dev/null | wc -l)
    echo "  Completed runs: $completed/10"
done

exit 0