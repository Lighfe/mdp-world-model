#!/bin/bash
#SBATCH --job-name=adamw_cosine
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=34G
#SBATCH --output=log/adamw_cosine_%A_%a.out
#SBATCH --error=log/adamw_cosine_%A_%a.err
#SBATCH --array=1-2  # 2 array jobs (one per configuration)

echo "========================================================"
echo "ADAMW COSINE TEST - Array Job $SLURM_ARRAY_TASK_ID/2"
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS (16 parallel)"
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

# Define configuration for this array job
CONFIG_ID=$SLURM_ARRAY_TASK_ID

case $CONFIG_ID in
    1)
        export OPTIMIZER="adamw"
        export SCHEDULER="cosine"
        export LR="1e-5"
        export MIN_LR="1e-6"
        export WEIGHT_DECAY="0.01"
        export HIDDEN_DIM="32"
        export CONFIG_NAME="adamw_cosine_1e-5"
        CONFIG_DESC="AdamW with cosine LR=1e-5→1e-6, WD=0.01"
        ;;
    2)
        export OPTIMIZER="adamw"
        export SCHEDULER="cosine"
        export LR="5e-5"
        export MIN_LR="5e-6"
        export WEIGHT_DECAY="0.01"
        export HIDDEN_DIM="32"
        export CONFIG_NAME="adamw_cosine_5e-5"
        CONFIG_DESC="AdamW with cosine LR=5e-5→5e-6, WD=0.01"
        ;;
esac

echo "Array Job $SLURM_ARRAY_TASK_ID Configuration:"
echo "  Config ID: $CONFIG_ID"
echo "  Description: $CONFIG_DESC"
echo "  Seeds: 11-26 (16 seeds)"
echo "  Datasets: 1-10 (sequential per task)"
echo "  Total parallel tasks: 16 seeds"

# Use same output directory as main grid search
export OUTPUT_DIR="neural_networks/output/lr_grid_search"
mkdir -p "$OUTPUT_DIR"

# Start GPU monitoring
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv --loop=10 > "log/gpu_adamw_cosine_${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}.log" &
GPU_MONITOR_PID=$!

# Create training script for this array job
TRAINING_SCRIPT="adamw_cosine_${SLURM_ARRAY_TASK_ID}_${SLURM_JOBID}.sh"
cat > "$TRAINING_SCRIPT" << 'EOF'
#!/bin/bash

# Get task ID (0-15) which maps to seed (11-26)
TASK_ID=$SLURM_PROCID
SEED=$((TASK_ID + 11))  # Seeds 11-26

export PYTHONUNBUFFERED=1

echo "[Task $TASK_ID] Starting Config $CONFIG_ID ($CONFIG_NAME), Seed $SEED at $(date)"
echo "[Task $TASK_ID] Config details: optimizer=$OPTIMIZER, scheduler=$SCHEDULER, lr=$LR→$MIN_LR, wd=$WEIGHT_DECAY"

# Run 10 datasets sequentially
for DATASET_NUM in {1..10}; do
    DB_PATH="datasets/results/multi_saddle_${DATASET_NUM}.db"
    
    # Verify dataset exists
    if [ ! -f "$DB_PATH" ]; then
        echo "[Task $TASK_ID] ERROR: Dataset not found: $DB_PATH"
        continue
    fi
    
    echo "[Task $TASK_ID] Starting dataset $DATASET_NUM at $(date)"
    
    # Build the command
    CMD="python -u neural_networks/train_drm.py $DB_PATH \
        --system_type=saddle_system \
        --batch_size=128 \
        --num_states=4 \
        --hidden_dim=$HIDDEN_DIM \
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
        --optimizer=$OPTIMIZER \
        --scheduler_type=$SCHEDULER \
        --lr=$LR \
        --min_lr=$MIN_LR \
        --weight_decay=$WEIGHT_DECAY \
        --epochs=100 \
        --seed=$SEED \
        --output_dir=$OUTPUT_DIR \
        --run_id=lr_grid_ds${DATASET_NUM}_${CONFIG_NAME}_seed${SEED}"
    
    # Execute the command
    $CMD 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[Task $TASK_ID] ✓ Dataset $DATASET_NUM completed"
    else
        echo "[Task $TASK_ID] ✗ Dataset $DATASET_NUM failed"
    fi
done

echo "[Task $TASK_ID] Finished all datasets at $(date)"
EOF

chmod +x "$TRAINING_SCRIPT"

echo "========================================================"
echo "Launching 16 parallel processes (16 seeds)..."
echo "Each will run 10 datasets sequentially"
echo "========================================================"

# Launch parallel training
srun --ntasks=$SLURM_NTASKS \
     --ntasks-per-node=$SLURM_NTASKS \
     --output="log/adamw_cosine_task_%A_%a_%t.out" \
     --error="log/adamw_cosine_task_%A_%a_%t.err" \
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
echo "Array Job $SLURM_ARRAY_TASK_ID completed at $(date)"
echo "Configuration: $CONFIG_NAME"
echo "srun exit code: $SRUN_EXIT_CODE"
echo "Results saved to: $OUTPUT_DIR"
echo "GPU usage log: log/gpu_adamw_cosine_${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}.log"
echo "Task logs: log/adamw_cosine_task_${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}_*.out"
echo "========================================================"

# Comprehensive summary
echo ""
echo "Summary for Configuration: $CONFIG_NAME"
echo "================================================"

# Count completed runs
total_expected=$((16 * 10))  # 16 seeds × 10 datasets
completed=0

# Check completion for each seed
for seed in {11..26}; do
    seed_count=0
    missing_datasets=""
    
    for dataset in {1..10}; do
        # Check for the expected output file
        if [ -f "${OUTPUT_DIR}/drm_final_lr_grid_ds${dataset}_${CONFIG_NAME}_seed${seed}.pt" ]; then
            seed_count=$((seed_count + 1))
            completed=$((completed + 1))
        else
            missing_datasets="$missing_datasets $dataset"
        fi
    done
    
    if [ $seed_count -eq 10 ]; then
        echo "  Seed $seed: ✓ All 10 datasets completed"
    elif [ $seed_count -eq 0 ]; then
        echo "  Seed $seed: ✗ No datasets completed"
    else
        echo "  Seed $seed: $seed_count/10 datasets (missing:$missing_datasets)"
    fi
done

echo ""
echo "Total completed: $completed/$total_expected experiments"
echo "Success rate: $(echo "scale=1; $completed * 100 / $total_expected" | bc)%"

# Check average runtime from logs
echo ""
if ls log/adamw_cosine_task_${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}_*.out 1> /dev/null 2>&1; then
    echo "Runtime Statistics:"
    total_time=0
    count=0
    min_time=999999
    max_time=0
    
    for logfile in log/adamw_cosine_task_${SLURM_JOBID}_${SLURM_ARRAY_TASK_ID}_*.out; do
        if grep -q "Training completed in" "$logfile" 2>/dev/null; then
            # Extract all training times from this task's log
            while IFS= read -r line; do
                runtime=$(echo "$line" | awk '{print $4}')
                if [ ! -z "$runtime" ]; then
                    # Check if runtime is a valid number
                    if [[ "$runtime" =~ ^[0-9]+\.?[0-9]*$ ]]; then
                        total_time=$(echo "$total_time + $runtime" | bc)
                        count=$((count + 1))
                        
                        # Update min/max
                        if (( $(echo "$runtime < $min_time" | bc -l) )); then
                            min_time=$runtime
                        fi
                        if (( $(echo "$runtime > $max_time" | bc -l) )); then
                            max_time=$runtime
                        fi
                    fi
                fi
            done <<< "$(grep "Training completed in" "$logfile")"
        fi
    done
    
    if [ $count -gt 0 ]; then
        avg_time=$(echo "scale=2; $total_time / $count" | bc)
        echo "  Average training time: ${avg_time} seconds per run"
        echo "  Min time: ${min_time} seconds"  
        echo "  Max time: ${max_time} seconds"
        echo "  Total runtime samples: $count"
    fi
fi

echo "================================================"

exit $SRUN_EXIT_CODE