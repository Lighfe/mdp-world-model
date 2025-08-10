#!/bin/bash
#SBATCH --job-name=h100_parallel_test
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=15
#SBATCH --ntasks-per-node=15
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=28G
#SBATCH --output=log/h100_test_%A.out
#SBATCH --error=log/h100_test_%A.err

echo "========================================================"
echo "H100 PARALLEL TEST - Dataset 8, 15 Seeds"
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

# Fix CuBLAS deterministic warning
export CUBLAS_WORKSPACE_CONFIG=:4096:8

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
echo "  Learning rate: 5e-5 (cosine schedule, min_lr=5e-6)"
echo "  Optimizer: adam"
echo "  Scheduler: cosine"
echo "  Batch size: 128"
echo "  Value loss weight: 0.3"
echo "  Seeds: 11-25 (15 seeds in parallel)"
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

echo "========================================================"
echo "Starting parallel experiments at $(date)"
echo "========================================================"

# Create training script for parallel execution
TRAINING_SCRIPT="h100_train_${SLURM_JOBID}.sh"
cat > "$TRAINING_SCRIPT" << 'EOF'
#!/bin/bash

# Get task ID (0-14) and calculate seed (11-25)
TASK_ID=$SLURM_PROCID
SEED=$((TASK_ID + 11))

export PYTHONUNBUFFERED=1

echo "[Task $TASK_ID] Starting with seed $SEED at $(date)"

# Fixed parameters for all runs
DATASET_NUM=8
DB_PATH="datasets/results/multi_saddle_${DATASET_NUM}.db"
OUTPUT_DIR="neural_networks/output/h100_parallel_test"

# Base command with H100 optimizations built in
CMD="python -u neural_networks/train_drm.py ${DB_PATH} \
    --system_type=saddle_system \
    --batch_size=128 \
    --num_states=4 \
    --hidden_dim=32 \
    --state_loss_weight=1.0 \
    --value_loss_weight=0.3 \
    --predictor_type=standard \
    --val_size=1000 \
    --test_size=1000 \
    --probing_size=2000 \
    --entropy_decay_proportion=0.5 \
    --use_warmup \
    --warmup_epochs=10 \
    --use_target_encoder \
    --ema_decay=0.9 \
    --use_entropy_reg \
    --entropy_weight=1.0 \
    --state_loss_type=kl_div \
    --use_gumbel \
    --min_temp=0.5 \
    --initial_temp=3.0 \
    --encoder_init_method=he \
    --optimizer=adam \
    --scheduler_type=cosine \
    --lr=5e-5 \
    --min_lr=5e-6 \
    --epochs=100 \
    --seed=$SEED \
    --output_dir=${OUTPUT_DIR} \
    --run_id=h100_ds${DATASET_NUM}_cosine_seed${SEED}"

echo "[Task $TASK_ID] Executing command..."
$CMD 2>&1

if [ $? -eq 0 ]; then
    echo "[Task $TASK_ID] ✓ Completed successfully (seed $SEED)"
else
    echo "[Task $TASK_ID] ✗ Failed (seed $SEED)"
fi

echo "[Task $TASK_ID] Finished at $(date)"
EOF

chmod +x "$TRAINING_SCRIPT"

# Launch parallel training
echo "Launching 15 parallel processes..."
srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=$SLURM_NTASKS \
     --output="log/h100_task_%A_%t.out" \
     --error="log/h100_task_%A_%t.err" \
     --unbuffered \
     ./"$TRAINING_SCRIPT"

SRUN_EXIT_CODE=$?

# Clean up
rm -f "$TRAINING_SCRIPT"

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null

# Stop MPS daemon
echo "Stopping NVIDIA MPS..."
echo quit | nvidia-cuda-mps-control

echo "========================================================"
echo "H100 PARALLEL TEST completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "GPU usage log: log/gpu_usage_h100_${SLURM_JOBID}.log"
echo "Task logs: log/h100_task_${SLURM_JOBID}_*.out"
echo "========================================================"

# Quick summary of results - fixed path to match actual output
echo ""
echo "Quick Summary:"
echo "--------------"
# Count completed runs by checking for final model files
completed=$(ls -1 ${OUTPUT_DIR}/drm_final_h100_ds${DATASET_NUM}_cosine_seed*.pt 2>/dev/null | wc -l)
echo "Completed runs: $completed/15"

# Show which seeds completed
echo ""
echo "Completed seeds:"
for seed in {11..25}; do
    if [ -f "${OUTPUT_DIR}/drm_final_h100_ds${DATASET_NUM}_cosine_seed${seed}.pt" ]; then
        echo "  Seed $seed: ✓"
    else
        echo "  Seed $seed: ✗"
    fi
done

# Calculate average runtime if log files exist
echo ""
if ls log/h100_task_${SLURM_JOBID}_*.out 1> /dev/null 2>&1; then
    echo "Checking individual task runtimes..."
    total_time=0
    count=0
    for logfile in log/h100_task_${SLURM_JOBID}_*.out; do
        if grep -q "Training completed in" "$logfile" 2>/dev/null; then
            runtime=$(grep "Training completed in" "$logfile" | tail -1 | awk '{print $4}')
            if [ ! -z "$runtime" ]; then
                total_time=$(echo "$total_time + $runtime" | bc)
                count=$((count + 1))
            fi
        fi
    done
    
    if [ $count -gt 0 ]; then
        avg_time=$(echo "scale=2; $total_time / $count" | bc)
        echo "Average training time: ${avg_time} seconds"
    fi
fi

exit $SRUN_EXIT_CODE