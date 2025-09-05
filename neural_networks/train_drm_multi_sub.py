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

def run_subprocess_training(config_file_path, run_name, run_index, total_runs, log_dir):
    """
    Launch a single training run as a subprocess.
    
    ONLY CHANGE: Use file output instead of pipes to avoid deadlock.
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
    
    ONLY CHANGE: Create log directory and pass it to subprocess functions.
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
    safe_json_dump(summary, summary_path)
    
    print(f"Summary saved to {summary_path}")
    
    return successful_runs, failed_runs, str(config_output_dir)

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