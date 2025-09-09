# neural_networks/train_drm_multi_sub.py
import os
import sys
import argparse
import time
import subprocess
import json
import numpy as np
from pathlib import Path
import psutil

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neural_networks.utils import (
    load_config, 
    parse_comma_separated, 
    generate_config_combinations,
    setup_output_structure,
    safe_json_dump
)

from neural_networks.drm_viz import (
    plot_training_curves_aggregated,
    plot_softmax_rank_aggregated, 
    plot_state_metrics_aggregated,
    plot_test_metrics_summary
)

def _calculate_descriptive_stats(values):
    """Calculate descriptive statistics for a list of values."""
    if not values:
        return None
    
    values = np.array(values)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'count': len(values)
    }

def _aggregate_curve_data(all_curves):
    """Aggregate training curve data (lists of values per epoch)."""
    if not all_curves:
        return None
    
    # Convert to numpy array: (num_runs, num_epochs)
    curves_array = np.array(all_curves)
    
    return {
        'mean': curves_array.mean(axis=0).tolist(),
        'std': curves_array.std(axis=0).tolist(), 
        'median': np.median(curves_array, axis=0).tolist(),
        'min': curves_array.min(axis=0).tolist(),
        'max': curves_array.max(axis=0).tolist(),
        'count': curves_array.shape[0]
    }

def _aggregate_epoch_dict_data(all_epoch_dicts):
    """Aggregate epoch-based dict data (like softmax_rank_metrics, state_metrics)."""
    if not all_epoch_dicts or not any(all_epoch_dicts):
        return {}
    
    # Find all unique metric keys across all runs
    all_keys = set()
    for epoch_list in all_epoch_dicts:
        for epoch_dict in epoch_list:
            all_keys.update(epoch_dict.keys())
    
    # Remove 'epoch' key as it's just indexing
    metric_keys = [key for key in all_keys if key != 'epoch']
    
    aggregated_metrics = {}
    
    for key in metric_keys:
        # Collect values for this metric across all runs and epochs
        all_values_for_key = []
        
        for epoch_list in all_epoch_dicts:
            run_values = [epoch_dict.get(key) for epoch_dict in epoch_list if key in epoch_dict]
            # Filter out None values
            run_values = [v for v in run_values if v is not None]
            all_values_for_key.append(run_values)
        
        # Convert to curve format if we have consistent epoch data
        if all_values_for_key and all(len(vals) == len(all_values_for_key[0]) for vals in all_values_for_key):
            aggregated_metrics[key] = _aggregate_curve_data(all_values_for_key)
        else:
            # Fallback: flatten all values and get overall stats
            flattened = [val for run_vals in all_values_for_key for val in run_vals]
            if flattened:
                aggregated_metrics[key] = _calculate_descriptive_stats(flattened)
    
    return aggregated_metrics

def aggregate_results(config_output_dir):
    """
    Aggregate results from all successful runs (any history_*.json file).
    
    Returns:
        float: mean_test_prob_discrete_accuracy for sweeper
    """
    individual_runs_dir = config_output_dir / "individual_runs"
    history_files = list(individual_runs_dir.glob("history_*.json"))
    
    print(f"Found {len(history_files)} successful runs to aggregate")
    
    if len(history_files) == 0:
        raise ValueError("No successful runs found for aggregation")
    
    # Load all histories
    histories = []
    for history_file in history_files:
        with open(history_file, 'r') as f:
            history = json.load(f)
            histories.append(history)
    
    aggregated = {
        'num_successful_runs': len(histories),
        'training_curves': {},
        'test_metrics': {},
        'softmax_rank_metrics': {},
        'state_metrics': {}
    }
    
    # 1. Aggregate training curves (lists per epoch)
    training_curve_keys = [
        'train_loss', 'train_state_loss', 'train_value_loss', 'train_entropy_loss',
        'val_loss', 'val_state_loss', 'val_value_loss', 'val_entropy_loss'
    ]
    
    for key in training_curve_keys:
        if key in histories[0]:  # Check if key exists
            # Collect all curves for this metric
            all_curves = [hist[key] for hist in histories if key in hist]
            aggregated['training_curves'][key] = _aggregate_curve_data(all_curves)
    
    # 2. Aggregate test metrics (single values)
    if 'test_metrics' in histories[0]:
        test_keys = histories[0]['test_metrics'].keys()
        for key in test_keys:
            values = [hist['test_metrics'][key] for hist in histories 
                     if 'test_metrics' in hist and key in hist['test_metrics'] and hist['test_metrics'][key] is not None]
            if values:
                aggregated['test_metrics'][key] = _calculate_descriptive_stats(values)
    
    # 3. Aggregate softmax rank metrics (lists of dicts per epoch)
    if 'softmax_rank_metrics' in histories[0] and histories[0]['softmax_rank_metrics']:
        aggregated['softmax_rank_metrics'] = _aggregate_epoch_dict_data(
            [hist.get('softmax_rank_metrics', []) for hist in histories]
        )
    
    # 4. Aggregate state metrics (lists of dicts per epoch)  
    if 'state_metrics' in histories[0] and histories[0]['state_metrics']:
        aggregated['state_metrics'] = _aggregate_epoch_dict_data(
            [hist.get('state_metrics', []) for hist in histories]
        )
    
    # Save aggregated results
    output_path = config_output_dir / "aggregated_results.json"
    with open(output_path, 'w') as f:
        safe_json_dump(aggregated, f, indent=2)
    
    print(f"Saved aggregated results to {output_path}")
    
    # Create all 4 visualizations
    print("Creating aggregated visualizations...")
    
    # 1. Training curves
    training_curves_path = config_output_dir / "training_curves_aggregated.png"
    plot_training_curves_aggregated(aggregated, training_curves_path)
    
    # 2. Softmax rank evolution
    softmax_rank_path = config_output_dir / "softmax_rank_aggregated.png"
    plot_softmax_rank_aggregated(aggregated, softmax_rank_path)
    
    # 3. State metrics
    state_metrics_path = config_output_dir / "state_metrics_aggregated.png"
    plot_state_metrics_aggregated(aggregated, state_metrics_path)
    
    # 4. Test metrics summary
    test_summary_path = config_output_dir / "test_metrics_summary.png"
    plot_test_metrics_summary(aggregated, test_summary_path)
    
    print("All visualizations created successfully!")
    
    # Return primary metric for sweeper
    if 'prob_discrete_accuracy' in aggregated['test_metrics']:
        mean_test_prob_discrete_accuracy = aggregated['test_metrics']['prob_discrete_accuracy']['mean']
        print(f"Primary metric (test_prob_discrete_accuracy): {mean_test_prob_discrete_accuracy:.4f}")
        return mean_test_prob_discrete_accuracy
    else:
        print("WARNING: prob_discrete_accuracy not found in test metrics")
        return 0.0


def run_subprocess_training(config_file_path, run_name, run_index, total_runs, log_dir):
    """
    Launch a single training run as a subprocess.
    
    CHANGES: 
    1. Use file output instead of pipes to avoid deadlock
    2. Use explicit conda Python path to ensure correct environment
    """
    # Use explicit conda environment Python instead of sys.executable
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix and os.path.exists(f"{conda_prefix}/bin/python"):
        python_executable = f"{conda_prefix}/bin/python"
        print(f"Using conda Python: {python_executable}")
    else:
        python_executable = sys.executable
        print(f"Using sys.executable (fallback): {python_executable}")
    
    cmd = [
        python_executable,
        "neural_networks/train_drm.py", 
        config_file_path,
        "--multi_run"
    ]
    
    print(f"Starting subprocess {run_index}/{total_runs}: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    
    # Launch subprocess with proper environment
    env = os.environ.copy()
    # Ensure each subprocess gets proper CUDA environment
    env['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    
    # Create log file for this run
    log_file = log_dir / f"{run_name}.log"
    
    # PIPE FIX: Write to file instead of pipe
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                env=env,
                cwd=str(PROJECT_ROOT)  # Set working directory to project root
            )
    except Exception as e:
        print(f"ERROR starting subprocess: {e}")
        raise
    
    return process, log_file

def monitor_process(process, log_file, run_name, run_index, total_runs):
    """
    Monitor a subprocess and collect its output.
    
    ONLY CHANGE: Read from log file instead of process.communicate()
    """
    start_time = time.time()
    
    try:
        # Wait for process to complete
        process.wait()
        runtime = time.time() - start_time
        
        # Read output from log file
        try:
            with open(log_file, 'r') as f:
                output = f.read()
        except:
            output = "Could not read log file"
        
        success = (process.returncode == 0)
        
        if success:
            print(f"✓ RUN {run_index}/{total_runs}: {run_name} completed in {runtime:.1f}s")
        else:
            print(f"✗ RUN {run_index}/{total_runs}: {run_name} failed in {runtime:.1f}s (exit code: {process.returncode})")
            # Print last few lines of output for debugging
            output_lines = output.split('\n')[-10:]
            print("Last 10 lines of output:")
            for line in output_lines:
                if line.strip():
                    print(f"  {line}")
        
        return {
            'run_name': run_name,
            'success': success,
            'runtime': runtime,
            'exit_code': process.returncode,
            'output_lines': len(output.split('\n')) if output else 0
        }
        
    except Exception as e:
        runtime = time.time() - start_time
        print(f"✗ RUN {run_index}/{total_runs}: {run_name} failed with exception in {runtime:.1f}s: {e}")
        
        return {
            'run_name': run_name,
            'success': False,
            'runtime': runtime,
            'error': str(e)
        }

def multi_train_drm_subprocess(config_path, output_dir, config_id, seeds, db_paths, max_parallel=None):
    """
    Main function to orchestrate multi-run training using subprocesses.
    
    Returns:
        tuple: (successful_runs, failed_runs, output_dir_path, primary_metric)
            - successful_runs: List of successful run results
            - failed_runs: List of failed run results  
            - output_dir_path: Path to output directory
            - primary_metric: Mean test_prob_discrete_accuracy for sweeper
    """
    if max_parallel is None:
        max_parallel = len(seeds)
    
    print("="*60)
    print(f"MULTI-RUN DRM TRAINING (SUBPROCESS)")
    print(f"Config ID: {config_id}")
    print(f"Seeds: {seeds}")
    print(f"Databases: {db_paths}")
    print(f"Total runs: {len(seeds) * len(db_paths)}")
    print(f"Max parallel processes: {max_parallel}")
    print("="*60)
    
    # Step 1: Generate individual run configs
    override_params = {
        "meta.seed": seeds,
        "meta.db_path": db_paths,
        "meta.output_dir": [f"{output_dir}/{config_id}/individual_runs"]
    }
    
    run_configs = generate_config_combinations(
        base_config_path=config_path,
        config_id=config_id,
        override_params=override_params
    )
    
    # Step 2: Set up output directory structure
    config_output_dir = setup_output_structure(output_dir, config_id, config_path)
    
    # PIPE FIX: Create log directory for subprocess outputs
    log_dir = config_output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Resource monitoring
    print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    print(f"CPU count: {psutil.cpu_count()}")
    
    # Step 3: Execute subprocess-based parallel training
    print(f"Starting subprocess-based parallel execution...")
    
    total_start_time = time.time()
    
    # Track active processes and results
    active_processes = {}  # {process: (log_file, run_name, run_index, total_runs)}
    completed_results = []
    pending_configs = list(enumerate(run_configs, 1))  # [(run_index, config_info)]
    
    # Start initial batch of processes
    while len(active_processes) < max_parallel and pending_configs:
        run_index, (config_file_path, run_name, override_values) = pending_configs.pop(0)
        
        # PIPE FIX: Pass log_dir to subprocess function
        process, log_file = run_subprocess_training(
            config_file_path, run_name, run_index, len(run_configs), log_dir
        )
        active_processes[process] = (log_file, run_name, run_index, len(run_configs))
        
        # Small delay to stagger process starts
        time.sleep(0.2)
    
    print(f"Started initial batch of {len(active_processes)} processes")
    
    # Monitor processes and start new ones as they complete
    monitor_count = 0
    while active_processes or pending_configs:
        
        # Check for completed processes
        completed_processes = []
        for process in list(active_processes.keys()):
            if process.poll() is not None:  # Process has finished
                completed_processes.append(process)
        
        # Debug output every 10 iterations to avoid spam
        monitor_count += 1
        if monitor_count % 10 == 0:
            print(f"[Monitor] Active: {len(active_processes)}, Pending: {len(pending_configs)}, Completed this check: {len(completed_processes)}")
            
            # If no progress after many iterations, check process status
            if monitor_count > 50 and len(completed_processes) == 0:
                print("[DEBUG] No processes completing - checking first 3 process statuses:")
                for i, (process, (log_file, run_name, run_index, total_runs)) in enumerate(list(active_processes.items())[:3]):
                    try:
                        poll_result = process.poll()
                        print(f"  Process {i+1} ({run_name}): poll()={poll_result}, pid={process.pid}")
                        
                        # Check if process exists on system
                        if psutil.pid_exists(process.pid):
                            proc_info = psutil.Process(process.pid)
                            print(f"    Status: {proc_info.status()}, CPU%: {proc_info.cpu_percent()}")
                        else:
                            print(f"    PID {process.pid} does not exist!")
                    except Exception as e:
                        print(f"    Error checking process: {e}")
        
        # Handle completed processes
        for process in completed_processes:
            log_file, run_name, run_index, total_runs = active_processes[process]
            
            # Monitor and collect results
            result = monitor_process(process, log_file, run_name, run_index, total_runs)
            completed_results.append(result)
            
            # Remove from active processes
            del active_processes[process]
            
            # Start next pending process if available
            if pending_configs:
                run_index, (config_file_path, run_name, override_values) = pending_configs.pop(0)
                
                new_process, new_log_file = run_subprocess_training(
                    config_file_path, run_name, run_index, len(run_configs), log_dir
                )
                active_processes[new_process] = (new_log_file, run_name, run_index, len(run_configs))
                
                time.sleep(0.2)  # Small delay
        
        # Brief sleep to avoid busy waiting
        if active_processes:
            time.sleep(5.0)
    
    # Final summary
    total_time = time.time() - total_start_time
    
    # Separate successful and failed runs
    successful_runs = [r for r in completed_results if r['success']]
    failed_runs = [r for r in completed_results if not r['success']]
    
    print(f"\n{'='*60}")
    print(f"SUBPROCESS MULTI-RUN TRAINING COMPLETED")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Successful runs: {len(successful_runs)}/{len(completed_results)}")
    print(f"Failed runs: {len(failed_runs)}")
    print("="*60)
    
    # Save summary
    summary = {
        'config_id': config_id,
        'total_runs': len(run_configs),
        'successful_runs': len(successful_runs),
        'failed_runs': len(failed_runs),
        'total_time': total_time,
        'seeds': seeds,
        'db_paths': db_paths,
        'results': completed_results
    }
    
    summary_path = config_output_dir / "run_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to {summary_path}")

    # Step 4: Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print("="*60)

    try:
        primary_metric = aggregate_results(config_output_dir)
        print(f"Aggregation completed. Primary metric: {primary_metric:.4f}")
    except Exception as e:
        print(f"Error during aggregation: {e}")
        import traceback
        traceback.print_exc()
        primary_metric = 0.0

    return successful_runs, failed_runs, str(config_output_dir), primary_metric

def main():
    parser = argparse.ArgumentParser(description="Multi-run DRM training with subprocess execution")
    parser.add_argument(
        "config_path", 
        help="Path to base YAML configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        default="neural_networks/output",
        help="Output directory for results (default: neural_networks/output)"
    )
    parser.add_argument(
        "--config-id", 
        required=True,
        help="Configuration identifier (used as subfolder name)"
    )
    parser.add_argument(
        "--seeds", 
        required=True,
        help="Comma-separated list of random seeds (e.g., '11,12,13,14,15')"
    )
    parser.add_argument(
        "--db-paths", 
        required=True,
        help="Comma-separated list of database paths"
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum number of parallel processes (default: number of seeds)"
    )
    
    args = parser.parse_args()
    
    # Parse comma-separated arguments
    seeds = [int(s) for s in parse_comma_separated(args.seeds)]
    db_paths = parse_comma_separated(args.db_paths)
    
    # Validate inputs
    if not seeds:
        raise ValueError("At least one seed must be provided")
    if not db_paths:
        raise ValueError("At least one database path must be provided")
    
    # Validate config file exists
    if not Path(args.config_path).exists():
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
    
    # Validate database files exist
    for db_path in db_paths:
        if not Path(db_path).exists():
            print(f"WARNING: Database file not found: {db_path}")
    
    # Execute multi-run training
    successful_runs, failed_runs, output_dir, primary_metric = multi_train_drm_subprocess(
        config_path=args.config_path,
        output_dir=args.output_dir,
        config_id=args.config_id,
        seeds=seeds,
        db_paths=db_paths,
        max_parallel=args.max_parallel
    )
    
    # Print final summary with primary metric
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Failed runs: {len(failed_runs)}")
    print(f"Primary metric (test_prob_discrete_accuracy): {primary_metric:.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()