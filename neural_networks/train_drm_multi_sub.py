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

def run_subprocess_training(config_file_path, run_name, run_index, total_runs):
    """
    Launch a single training run as a subprocess.
    
    Args:
        config_file_path: Path to config file
        run_name: Name for this run
        run_index: Current run number (1-based)
        total_runs: Total number of runs
    
    Returns:
        subprocess.Popen object
    """
    
    cmd = [
        sys.executable,  # Current Python interpreter
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
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr with stdout
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT)  # Set working directory to project root
    )
    
    return process

def monitor_process(process, run_name, run_index, total_runs):
    """
    Monitor a subprocess and collect its output.
    
    Args:
        process: subprocess.Popen object
        run_name: Name of the run
        run_index: Run index
        total_runs: Total runs
    
    Returns:
        dict: Result dictionary with success status and metrics
    """
    start_time = time.time()
    
    try:
        # Wait for process to complete and capture output
        stdout, _ = process.communicate()
        runtime = time.time() - start_time
        
        success = (process.returncode == 0)
        
        if success:
            print(f"✓ RUN {run_index}/{total_runs}: {run_name} completed in {runtime:.1f}s")
        else:
            print(f"✗ RUN {run_index}/{total_runs}: {run_name} failed in {runtime:.1f}s (exit code: {process.returncode})")
            # Print last few lines of output for debugging
            output_lines = stdout.split('\n')[-10:]
            print("Last 10 lines of output:")
            for line in output_lines:
                if line.strip():
                    print(f"  {line}")
        
        return {
            'run_name': run_name,
            'success': success,
            'runtime': runtime,
            'exit_code': process.returncode,
            'output_lines': len(stdout.split('\n')) if stdout else 0
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
    
    Args:
        config_path: Path to base YAML config
        output_dir: Output directory for results  
        config_id: Identifier for this configuration
        seeds: List of random seeds
        db_paths: List of database paths
        max_parallel: Maximum number of parallel processes
    """
    print("[DEBUG] Starting multi_train_drm_subprocess function")
    sys.stdout.flush()
    
    if max_parallel is None:
        max_parallel = len(seeds)
    
    print("="*60)
    print(f"MULTI-RUN DRM TRAINING (SUBPROCESS)")
    
    print("="*60)
    print(f"MULTI-RUN DRM TRAINING (SUBPROCESS)")
    print(f"Config ID: {config_id}")
    print(f"Seeds: {seeds}")
    print(f"Databases: {db_paths}")
    print(f"Total runs: {len(seeds) * len(db_paths)}")
    print(f"Max parallel processes: {max_parallel}")
    print("="*60)

    print("[DEBUG] About to generate config combinations")
    sys.stdout.flush()
    
    # Step 1: Generate individual run configs
    override_params = {
        "meta.seed": seeds,
        "meta.db_path": db_paths,
        "meta.output_dir": [f"{output_dir}/{config_id}/individual_runs"]
    }

    print("[DEBUG] About to call generate_config_combinations")
    sys.stdout.flush()
    
    run_configs = generate_config_combinations(
        base_config_path=config_path,
        config_id=config_id,
        override_params=override_params
    )

    print(f"[DEBUG] Generated {len(run_configs)} configs")
    sys.stdout.flush()
    
    print("[DEBUG] About to setup output structure")
    sys.stdout.flush()
    
    # Step 2: Set up output directory structure
    config_output_dir = setup_output_structure(output_dir, config_id, config_path)

    print("[DEBUG] Output structure setup complete")
    sys.stdout.flush()

    # Resource monitoring
    print("[DEBUG] About to check memory")
    sys.stdout.flush()
    print(f"Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")

    print("[DEBUG] About to check CPU count")
    sys.stdout.flush()
    print(f"CPU count: {psutil.cpu_count()}")

    print("[DEBUG] About to start subprocess execution")
    sys.stdout.flush()

    # Step 3: Execute subprocess-based parallel training
    print(f"Starting subprocess-based parallel execution...")

    print("[DEBUG] About to initialize variables")
    sys.stdout.flush()

    total_start_time = time.time()

    # Track active processes and results
    active_processes = {}  # {process: (run_name, run_index, total_runs)}
    completed_results = []
    pending_configs = list(enumerate(run_configs, 1))  # [(run_index, config_info)]

    print(f"[DEBUG] About to start initial batch, pending_configs length: {len(pending_configs)}")
    sys.stdout.flush()

    # Start initial batch of processes
    while len(active_processes) < max_parallel and pending_configs:
        print(f"[DEBUG] Starting process {len(active_processes) + 1}")
        sys.stdout.flush()
        
        run_index, (config_file_path, run_name, override_values) = pending_configs.pop(0)
        
        print(f"[DEBUG] About to call run_subprocess_training for {run_name}")
        sys.stdout.flush()
        
        process = run_subprocess_training(
            config_file_path, run_name, run_index, len(run_configs)
        )
        active_processes[process] = (run_name, run_index, len(run_configs))
        
        # Small delay to stagger process starts
        time.sleep(0.2)
    
    print(f"Started initial batch of {len(active_processes)} processes")
    
    # Monitor processes and start new ones as they complete
    while active_processes or pending_configs:
        
        # Check for completed processes
        completed_processes = []
        for process in list(active_processes.keys()):
            if process.poll() is not None:  # Process has finished
                completed_processes.append(process)
        
        # Handle completed processes
        for process in completed_processes:
            run_name, run_index, total_runs = active_processes[process]
            
            # Monitor and collect results
            result = monitor_process(process, run_name, run_index, total_runs)
            completed_results.append(result)
            
            # Remove from active processes
            del active_processes[process]
            
            # Start next pending process if available
            if pending_configs:
                run_index, (config_file_path, run_name, override_values) = pending_configs.pop(0)
                
                new_process = run_subprocess_training(
                    config_file_path, run_name, run_index, len(run_configs)
                )
                active_processes[new_process] = (run_name, run_index, len(run_configs))
                
                time.sleep(0.2)  # Small delay
        
        # Brief sleep to avoid busy waiting
        if active_processes:
            time.sleep(1.0)
    
    # Final summary
    total_time = time.time() - total_start_time
    
    # Separate successful and failed runs
    successful_runs = [r for r in completed_results if r['success']]
    failed_runs = [r for r in completed_results if not r['success']]
    
    print(f"\n{'='*60}")
    print(f"SUBPROCESS MULTI-RUN TRAINING COMPLETED")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Successful runs: {len(successful_runs)}/{len(completed_results)}")
    print(f"Failed runs: {len(failed_runs)}/{len(completed_results)}")
    
    if successful_runs:
        runtimes = [r['runtime'] for r in successful_runs]
        print(f"Runtime stats: avg={np.mean(runtimes):.1f}s, min={np.min(runtimes):.1f}s, max={np.max(runtimes):.1f}s")
    
    if failed_runs:
        print(f"Failed runs:")
        for failed in failed_runs:
            print(f"  - {failed['run_name']}: {failed.get('error', 'Unknown error')}")
    
    print(f"{'='*60}")
    
    # Save summary results
    summary = {
        'config_id': config_id,
        'total_time': total_time,
        'successful_runs': len(successful_runs),
        'failed_runs': len(failed_runs),
        'results': completed_results
    }
    
    summary_path = config_output_dir / "run_summary.json"
    with open(summary_path, 'w') as f:
        safe_json_dump(summary, f, indent=2)
    
    print(f"Saved run summary to: {summary_path}")
    
    return successful_runs, failed_runs, config_output_dir

def main():
    parser = argparse.ArgumentParser(
        description="Multi-run DRM training using subprocess approach"
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
    successful_runs, failed_runs, output_dir = multi_train_drm_subprocess(
        config_path=args.config_path,
        output_dir=args.output_dir,
        config_id=args.config_id,
        seeds=seeds,
        db_paths=db_paths,
        max_parallel=args.max_parallel
    )

if __name__ == "__main__":
    main()