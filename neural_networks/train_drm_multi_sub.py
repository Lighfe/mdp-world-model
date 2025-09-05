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
    print(f"[DEBUG] Entering run_subprocess_training for {run_name}")
    sys.stdout.flush()
    
    cmd = [
        sys.executable,  # Current Python interpreter
        "neural_networks/train_drm.py", 
        config_file_path,
        "--multi_run"
    ]
    
    print(f"[DEBUG] Command prepared: {' '.join(cmd)}")
    print(f"[DEBUG] sys.executable: {sys.executable}")
    print(f"[DEBUG] config_file_path exists: {Path(config_file_path).exists()}")
    print(f"[DEBUG] train_drm.py exists: {Path('neural_networks/train_drm.py').exists()}")
    print(f"[DEBUG] Current working directory: {os.getcwd()}")
    sys.stdout.flush()
    
    print(f"Starting subprocess {run_index}/{total_runs}: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    
    # Launch subprocess with proper environment
    env = os.environ.copy()
    # Ensure each subprocess gets proper CUDA environment
    env['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    
    print(f"[DEBUG] Environment prepared, CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[DEBUG] About to call subprocess.Popen")
    sys.stdout.flush()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            env=env,
            cwd=str(PROJECT_ROOT)  # Set working directory to project root
        )
        
        print(f"[DEBUG] subprocess.Popen succeeded, PID: {process.pid}")
        sys.stdout.flush()
        
        return process
        
    except Exception as e:
        print(f"[DEBUG] ERROR in subprocess.Popen: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise

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
    print("[DEBUG] Entering multi_train_drm_subprocess function")
    sys.stdout.flush()
    
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
    
    print("[DEBUG] About to call generate_config_combinations")
    sys.stdout.flush()
    
    # Step 1: Generate individual run configs
    override_params = {
        "meta.seed": seeds,
        "meta.db_path": db_paths,
        "meta.output_dir": [f"{output_dir}/{config_id}/individual_runs"]
    }
    
    print("[DEBUG] About to call generate_config_combinations with params:", override_params)
    sys.stdout.flush()
    
    try:
        run_configs = generate_config_combinations(
            base_config_path=config_path,
            config_id=config_id,
            override_params=override_params
        )
        print(f"[DEBUG] generate_config_combinations completed, got {len(run_configs)} configs")
        sys.stdout.flush()
    except Exception as e:
        print(f"[DEBUG] ERROR in generate_config_combinations: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise
    
    print("[DEBUG] About to call setup_output_structure")
    sys.stdout.flush()
    
    # Step 2: Set up output directory structure
    try:
        config_output_dir = setup_output_structure(output_dir, config_id, config_path)
        print(f"[DEBUG] setup_output_structure completed: {config_output_dir}")
        sys.stdout.flush()
    except Exception as e:
        print(f"[DEBUG] ERROR in setup_output_structure: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise
    
    print("[DEBUG] About to check resources")
    sys.stdout.flush()
    
    # Resource monitoring
    try:
        available_mem = psutil.virtual_memory().available / 1024**3
        cpu_count = psutil.cpu_count()
        print(f"Available memory: {available_mem:.1f} GB")
        print(f"CPU count: {cpu_count}")
        print("[DEBUG] Resource check completed")
        sys.stdout.flush()
    except Exception as e:
        print(f"[DEBUG] ERROR in resource check: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise
    
    # Step 3: Execute subprocess-based parallel training
    print(f"Starting subprocess-based parallel execution...")
    print("[DEBUG] About to start subprocess monitoring")
    sys.stdout.flush()
    
    total_start_time = time.time()
    
    # Track active processes and results
    active_processes = {}  # {process: (run_name, run_index, total_runs)}
    completed_results = []
    pending_configs = list(enumerate(run_configs, 1))  # [(run_index, config_info)]
    
    print(f"[DEBUG] Created pending_configs list with {len(pending_configs)} items")
    sys.stdout.flush()
    
    # Start initial batch of processes
    print(f"[DEBUG] About to start process creation loop")
    print(f"[DEBUG] max_parallel: {max_parallel}, pending_configs length: {len(pending_configs)}")
    sys.stdout.flush()
    
    process_creation_count = 0
    while len(active_processes) < max_parallel and pending_configs:
        process_creation_count += 1
        print(f"[DEBUG] Process creation iteration {process_creation_count}")
        sys.stdout.flush()
        
        try:
            run_index, (config_file_path, run_name, override_values) = pending_configs.pop(0)
            print(f"[DEBUG] Processing config {run_index}: {run_name}, path: {config_file_path}")
            sys.stdout.flush()
            
            print(f"[DEBUG] About to call run_subprocess_training")
            sys.stdout.flush()
            
            process = run_subprocess_training(
                config_file_path, run_name, run_index, len(run_configs)
            )
            
            print(f"[DEBUG] run_subprocess_training returned, process PID: {process.pid}")
            sys.stdout.flush()
            
            active_processes[process] = (run_name, run_index, len(run_configs))
            
            print(f"[DEBUG] Added process to active_processes, total active: {len(active_processes)}")
            sys.stdout.flush()
            
            # Small delay to stagger process starts
            time.sleep(0.2)
            
        except Exception as e:
            print(f"[DEBUG] ERROR in process creation iteration {process_creation_count}: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            break
    
    print(f"[DEBUG] Exited process creation loop")
    print(f"Started initial batch of {len(active_processes)} processes")
    sys.stdout.flush()
    
    # Monitor processes and start new ones as they complete
    print(f"[DEBUG] About to enter monitoring loop")
    sys.stdout.flush()
    
    monitor_count = 0
    while active_processes or pending_configs:
        monitor_count += 1
        
        if monitor_count <= 5 or monitor_count % 10 == 0:  # Print first 5 iterations, then every 10th
            print(f"[DEBUG] Monitoring iteration {monitor_count}: active={len(active_processes)}, pending={len(pending_configs)}")
            sys.stdout.flush()
        
        # Check for completed processes
        completed_processes = []
        for i, process in enumerate(list(active_processes.keys())):
            poll_result = process.poll()
            if poll_result is not None:  # Process has finished
                completed_processes.append(process)
                if monitor_count <= 5:
                    print(f"[DEBUG] Process {i} completed with exit code: {poll_result}")
                    sys.stdout.flush()
        
        if monitor_count <= 5 or monitor_count % 10 == 0:
            print(f"[DEBUG] Found {len(completed_processes)} completed processes")
            sys.stdout.flush()
        
        # Debug output every 10 iterations to avoid spam, but with more detail early on
        if monitor_count <= 10 or monitor_count % 20 == 0:
            print(f"[Monitor] Active: {len(active_processes)}, Pending: {len(pending_configs)}, Completed this check: {len(completed_processes)}")
            
            # If no progress after many iterations, check process status
            if monitor_count > 50 and len(completed_processes) == 0:
                print("[DEBUG] No processes completing - checking first 3 process statuses:")
                for i, (process, (run_name, run_index, total_runs)) in enumerate(list(active_processes.items())[:3]):
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
                sys.stdout.flush()
        
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
        
        # Safety exit after too many iterations
        if monitor_count > 1000:
            print(f"[DEBUG] ERROR: Monitoring loop exceeded 1000 iterations, breaking")
            sys.stdout.flush()
            break
    
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