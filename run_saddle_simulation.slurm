#!/bin/bash
#SBATCH --job-name=multi_saddle_datasets
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=log/multi_saddle_datasets_%j.out
#SBATCH --error=log/multi_saddle_datasets_%j.err

# Print job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

# Load Anaconda module
module load anaconda

# Activate the environment
source activate mdp-world-model

# Set number of threads for parallel processing
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# Create log directory if it doesn't exist
mkdir -p log

echo "Starting sequential multi-saddle system simulations..."

# Function to run simulation for both controls
run_multi_saddle_simulation() {
    local saddle_points=$1
    local angles=$2
    local lambda1=$3
    local lambda2=$4
    local delta_t=$5
    local samples_per_cell=$6
    local db_name=$7
    local description=$8
    
    echo "=== Running simulation: $description ==="
    echo "  Saddle points: $saddle_points"
    echo "  Angles: $angles"
    echo "  Lambda values: λ₁=$lambda1, λ₂=$lambda2"
    echo "  Delta t: $delta_t"
    echo "  Samples per cell: $samples_per_cell"
    echo "  Database: $db_name"
    
    # Count number of saddles to determine how many controls to run
    # Count commas between saddle points and add 1
    local num_saddles=$(echo $saddle_points | grep -o '], \[' | wc -l)
    num_saddles=$((num_saddles + 1))
    echo "  Detected $num_saddles saddles - will run controls 0 to $((num_saddles-1))"
    
    # Calculate samples per control to maintain total data amount
    local samples_per_control=$((samples_per_cell / num_saddles))
    echo "  Total samples per cell: $samples_per_cell, split into $samples_per_control per control"
    
    # Run simulation for each control (saddle)
    for control in $(seq 0 $((num_saddles-1))); do
        echo "  → Running control $control (saddle $((control+1))/$num_saddles) with $samples_per_control samples per cell..."
        
        python run_saddle_simulation.py \
        --resolution 25 25 \
        --avg-samples-per-cell $samples_per_control \
        --num-steps 1 \
        --delta-t $delta_t \
        --logistic-k 0.5 \
        --logistic-x0 0.0 \
        --saddle-points "$saddle_points" \
        --angles "$angles" \
        --lambda1 $lambda1 \
        --lambda2 $lambda2 \
        --control $control \
        --db-name $db_name \
        --create-streamplot \
        --streamplot-original \
        --streamplot-range -5.0 5.0
        
        if [ $? -eq 0 ]; then
            echo "    ✓ Control $control completed successfully"
        else
            echo "    ✗ Control $control failed"
            return 1
        fi
    done
    
    echo "  ✓ All controls completed for: $description"
    echo ""
}

# 1) Both saddles at (0,0), angles 180°, 90°
run_multi_saddle_simulation \
    '[[0.0, 0.0], [0.0, 0.0]]' '[180, 90]' 1.0 -1.0 1.0 50 \
    'multi_saddle_1.db' 'both saddles at [0,0], angles 180°, 90°'

# 2) Saddles at (0,0) and (0,2), angles 180°, 90°
run_multi_saddle_simulation \
    '[[0.0, 0.0], [0.0, 2.0]]' '[180, 90]' 1.0 -1.0 1.0 50 \
    'multi_saddle_2.db' 'saddles at [0,0] and [0,2], angles 180°, 90°'

# 3) Saddles at (0,0) and (2,0), angles 180°, 90°
run_multi_saddle_simulation \
    '[[0.0, 0.0], [2.0, 0.0]]' '[180, 90]' 1.0 -1.0 1.0 50 \
    'multi_saddle_3.db' 'saddles at [0,0] and [2,0], angles 180°, 90°'

# 4) Saddles at (2,0) and (0,2), angles 180°, 90°
run_multi_saddle_simulation \
    '[[2.0, 0.0], [0.0, 2.0]]' '[180, 90]' 1.0 -1.0 1.0 50 \
    'multi_saddle_4.db' 'saddles at [2,0] and [0,2], angles 180°, 90°'

# 5) Both saddles at (0,0), angles 180°, 45°
run_multi_saddle_simulation \
    '[[0.0, 0.0], [0.0, 0.0]]' '[180, 45]' 1.0 -1.0 1.0 50 \
    'multi_saddle_5.db' 'both saddles at [0,0], angles 180°, 45°'

# 6) Saddles at (0,0) and (0,2), angles 180°, 45°
run_multi_saddle_simulation \
    '[[0.0, 0.0], [0.0, 2.0]]' '[180, 45]' 1.0 -1.0 1.0 50 \
    'multi_saddle_6.db' 'saddles at [0,0] and [0,2], angles 180°, 45°'

# 7) Saddles at (0,0) and (2,0), angles 180°, 45°
run_multi_saddle_simulation \
    '[[0.0, 0.0], [2.0, 0.0]]' '[180, 45]' 1.0 -1.0 1.0 50 \
    'multi_saddle_7.db' 'saddles at [0,0] and [2,0], angles 180°, 45°'

# 8) Saddles at (2,0) and (0,2), angles 180°, 45°
run_multi_saddle_simulation \
    '[[2.0, 0.0], [0.0, 2.0]]' '[180, 45]' 1.0 -1.0 1.0 50 \
    'multi_saddle_8.db' 'saddles at [2,0] and [0,2], angles 180°, 45°'

# 9) Both saddles at (0,0), angles 180°, 90°, lambda1=0.5
run_multi_saddle_simulation \
    '[[0.0, 0.0], [0.0, 0.0]]' '[180, 90]' 0.5 -1.0 1.0 50 \
    'multi_saddle_9.db' 'both saddles at [0,0], angles 180°, 90°, lambda1=0.5'

# 10) Both saddles at (0,0), angles 180°, 90°, different delta_t values
echo "=== Running simulation 10: three different delta_t values ==="
run_multi_saddle_simulation \
    '[[0.0, 0.0], [0.0, 0.0]]' '[180, 90]' 1.0 -1.0 0.1 20 \
    'multi_saddle_10.db' 'both saddles at [0,0], angles 180°, 90°, delta_t=0.1'

run_multi_saddle_simulation \
    '[[0.0, 0.0], [0.0, 0.0]]' '[180, 90]' 1.0 -1.0 1.0 20 \
    'multi_saddle_10.db' 'both saddles at [0,0], angles 180°, 90°, delta_t=1.0'

run_multi_saddle_simulation \
    '[[0.0, 0.0], [0.0, 0.0]]' '[180, 90]' 1.0 -1.0 5.0 20 \
    'multi_saddle_10.db' 'both saddles at [0,0], angles 180°, 90°, delta_t=5.0'

# Print completion information
echo "All simulations completed at $(date)"