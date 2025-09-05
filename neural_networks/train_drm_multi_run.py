# neural_networks/train_drm_multi_run.py
import os
import sys
import argparse
import time
import numpy as np
import json
import yaml
import torch.multiprocessing as mp
from pathlib import Path
import psutil
import subprocess

# Add project root to path (go up two levels from neural_networks/)
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
from neural_networks.train_drm import train_drm_model

def worker_wrapper(args, result_queue):
    """Wrapper that puts results in queue"""
    result = run_single_training_wrapper(args)
    result_queue.put(result)

def run_single_training_wrapper(args):
    """
    Minimal wrapper for multiprocessing. Just handles config path setup and calls train_drm_model.
    
    Args:
        args: Tuple of (config_file_path, run_name, individual_runs_dir, run_index, total_runs)
    
    Returns:
        dict: Run result dictionary
    """

    config_file_path, run_name, individual_runs_dir, run_index, total_runs = args
    
    print(f"RUN {run_index}/{total_runs}: {run_name} [PID: {os.getpid()}] - Starting...")
    
    run_start_time = time.time()
    
    try:
        # Update the config to set output directory for this specific run
        config = load_config(config_file_path)
        run_output_dir = individual_runs_dir / run_name
        config['meta']['output_dir'] = str(run_output_dir)
        
        # Save the updated config in the individual run directory
        run_output_dir.mkdir(parents=True, exist_ok=True)
        updated_config_path = run_output_dir / f"config_{run_name}.yaml"
        with open(updated_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        # Execute training with multi_run=True (minimal output)
        model, history = train_drm_model(str(updated_config_path), multi_run=True)
        
        run_time = time.time() - run_start_time
        
        # Extract key metric for immediate feedback
        accuracy = None
        if 'test_metrics' in history and 'prob_discrete_accuracy' in history['test_metrics']:
            accuracy = history['test_metrics']['prob_discrete_accuracy']
        
        print(f"✓ RUN {run_index}/{total_runs}: {run_name} [PID: {os.getpid()}] - COMPLETED in {run_time:.1f}s"
              + (f" (acc: {accuracy:.4f})" if accuracy else ""))
        
        # Return simplified result for aggregation
        return {
            'run_name': run_name,
            'success': True,
            'run_time': run_time,
            'history': history  # Let aggregation function extract what it needs
        }
        
    except Exception as e:
        run_time = time.time() - run_start_time
        print(f"✗ RUN {run_index}/{total_runs}: {run_name} [PID: {os.getpid()}] - FAILED in {run_time:.1f}s: {e}")
        
        return {
            'run_name': run_name,
            'success': False,
            'run_time': run_time,
            'error': str(e)
        }

def multi_train_drm_model(config_path, output_dir, config_id, seeds, db_paths, max_parallel=None):
    """
    Main function to orchestrate multi-run training with parallelization.
    
    Args:
        config_path: Path to base YAML config
        output_dir: Output directory for results  
        config_id: Identifier for this configuration
        seeds: List of random seeds
        db_paths: List of database paths
        max_parallel: Maximum number of parallel processes (default: len(seeds))
    """
    if max_parallel is None:
        max_parallel = len(seeds)
    
    print("="*60)
    print(f"MULTI-RUN DRM TRAINING (PARALLEL)")
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
    individual_runs_dir = config_output_dir / "individual_runs"

    # Debugging
    print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    print(f"CPU count: {psutil.cpu_count()}")

    # Check current process count
    result = subprocess.run(['ps', '-u', 'judralle'], capture_output=True, text=True)
    current_processes = len(result.stdout.split('\n')) - 1
    print(f"Current processes for user: {current_processes}")
    
    # Step 3: Execute training runs in parallel
    print(f"Starting parallel execution with {max_parallel} processes...")
    
    # Prepare arguments for parallel execution (simplified)
    worker_args = []
    for i, (config_file_path, run_name, override_values) in enumerate(run_configs):
        worker_args.append((
            config_file_path, 
            run_name, 
            individual_runs_dir,
            i + 1,  # run_index 
            len(run_configs)  # total_runs
        ))
    
    total_start_time = time.time()

    # Instead of Pool, use explicit torch multiprocessing processes (non-daemonic)
    processes = []
    ctx = mp.get_context('spawn') 
    results_queue = ctx.SimpleQueue()  # No feeder thread issues

    print(f"Starting {max_parallel} non-daemonic processes...")
    # Start all processes
    for i, worker_arg in enumerate(worker_args):
        p = ctx.Process(target=worker_wrapper, args=(worker_arg, results_queue))
        p.start()
        time.sleep(0.1) # Small stagger for CUDA initialization
        processes.append(p)
        print(f"Started process {i+1}/{len(worker_args)}", flush=True)
        
    # Add this check
    time.sleep(5)  # Wait a moment
    print("Checking if processes are still alive:")
    for i, p in enumerate(processes):
        print(f"Process {i+1}: PID {p.pid}, alive: {p.is_alive()}")

    # Wait for all to complete
    for p in processes:
        p.join()

    # Collect results with timeout
    results = []
    for i in range(len(worker_args)):
        try:
            result = results_queue.get(timeout=30)
            results.append(result)
        except:
            print(f"Warning: Could not get result {i+1}")
            break
    
    # Separate successful and failed runs
    completed_runs = [r for r in results if r['success']]
    failed_runs = [r for r in results if not r['success']]
    
    # Final summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*60}")
    print(f"PARALLEL MULTI-RUN TRAINING COMPLETED")
    print(f"Total time: {total_time:.1f}s")
    print(f"Successful runs: {len(completed_runs)}/{len(results)}")
    print(f"Failed runs: {len(failed_runs)}/{len(results)}")
    
    if failed_runs:
        print(f"Failed runs:")
        for failed in failed_runs:
            print(f"  - {failed['run_name']}: {failed['error']}")
    
    print(f"{'='*60}")
    
    return completed_runs, failed_runs, config_output_dir

def main():
    mp.set_start_method("spawn", force=True)
    print("Set torch multiprocessing to spawn method")

    parser = argparse.ArgumentParser(
        description="Multi-run DRM training with different seeds and datasets"
    )
    parser.add_argument(
        "config_path", 
        help="Path to base YAML configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        required=True,
        help="Base output directory for results"
    )
    parser.add_argument(
        "--config-id", 
        required=True,
        help="Identifier for this configuration (used as subfolder name)"
    )
    parser.add_argument(
        "--seeds", 
        required=True,
        help="Comma-separated list of random seeds (e.g., '11,12,13,14,15')"
    )
    parser.add_argument(
        "--db-paths", 
        required=True,
        help="Comma-separated list of database paths (e.g., 'data1.db,data2.db,data3.db')"
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
    completed_runs, failed_runs, output_dir = multi_train_drm_model(
        config_path=args.config_path,
        output_dir=args.output_dir,
        config_id=args.config_id,
        seeds=seeds,
        db_paths=db_paths,
        max_parallel=args.max_parallel
    )

if __name__ == "__main__":
    main()