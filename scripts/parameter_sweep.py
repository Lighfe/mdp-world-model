#!/usr/bin/env python3
"""
Main Parameter Sweep Script
Orchestrates the complete parameter optimization using Optuna.
"""

import os
import sys
import argparse
import subprocess
import optuna
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.parameter_sampler import SweepParameterSampler
from scripts.config_generator import create_sweep_folder, create_trial_config, copy_sweep_config_to_folder, cleanup_trial_config


class ParameterSweep:
    """
    Main parameter sweep orchestrator using Optuna.
    """
    
    def __init__(self, sweep_config_path: str, sweep_id: Optional[str] = None, 
                 study_name: Optional[str] = None, storage: Optional[str] = None):
        """
        Initialize parameter sweep.
        
        Args:
            sweep_config_path: Path to sweep configuration file
            sweep_id: Custom sweep identifier (default: datetime-based)
            study_name: Optuna study name (default: from sweep config)
            storage: Optuna storage URL (default: SQLite in sweep folder)
        """
        self.sweep_config_path = sweep_config_path
        self.sampler = SweepParameterSampler(sweep_config_path)
        
        # Create sweep folder
        self.sweep_folder = create_sweep_folder(sweep_id)
        print(f"Sweep folder: {self.sweep_folder}")
        
        # Copy sweep config to folder for reference
        copy_sweep_config_to_folder(sweep_config_path, self.sweep_folder)
        
        # Get sweep info
        self.sweep_info = self.sampler.get_sweep_info()
        self.multi_run_config = self.sampler.get_multi_run_config()
        
        # Set up Optuna study
        self.study_name = study_name or self.sweep_info.get('name', 'parameter_sweep')
        self.storage = storage or f"sqlite:///{self.sweep_folder}/optuna_study.db"
        
        # Results tracking
        self.results_file = self.sweep_folder / "sweep_results.json"
        self.trial_results = []
        
        print(f"Study name: {self.study_name}")
        print(f"Storage: {self.storage}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function - runs one complete trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Performance metric to optimize (prob_discrete_accuracy)
        """
        trial_start_time = time.time()
        trial_number = trial.number
        
        print(f"\n{'='*60}")
        print(f"STARTING TRIAL {trial_number}")
        print(f"{'='*60}")
        
        try:
            # 1. Sample trial parameters
            trial_params = self.sampler.sample_trial_params(trial)
            print("Trial parameters:")
            for key, value in trial_params.items():
                if value is not None:
                    print(f"  {key}: {value}")
            
            # 2. Generate trial config file
            trial_config_path = create_trial_config(
                trial_params=trial_params,
                trial_number=trial_number,
                base_config_path=self.sampler.get_base_config_path(),
                sweep_folder=self.sweep_folder,
                fixed_params=self.sampler.get_fixed_params()
            )
            print(f"Generated config: {trial_config_path}")
            
            # 3. Run train_drm_multi.py
            performance_metric = self._run_training(trial_config_path, trial_number)
            
            # 4. Record results
            trial_result = {
                'trial_number': trial_number,
                'performance_metric': performance_metric,
                'parameters': {k: v for k, v in trial_params.items() if v is not None},
                'config_path': str(trial_config_path),
                'runtime_seconds': time.time() - trial_start_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            
            self.trial_results.append(trial_result)
            self._save_results()
            
            print(f"{'='*60}")
            print(f"TRIAL {trial_number} COMPLETED")
            print(f"Performance metric: {performance_metric:.4f}")
            print(f"Runtime: {trial_result['runtime_seconds']:.1f} seconds")
            print(f"{'='*60}")
            
            return performance_metric
            
        except Exception as e:
            # Record failed trial
            trial_result = {
                'trial_number': trial_number,
                'performance_metric': 0.0,
                'parameters': {k: v for k, v in trial_params.items() if v is not None} if 'trial_params' in locals() else {},
                'error': str(e),
                'runtime_seconds': time.time() - trial_start_time,
                'timestamp': datetime.now().isoformat(),
                'status': 'failed'
            }
            
            self.trial_results.append(trial_result)
            self._save_results()
            
            print(f"{'='*60}")
            print(f"TRIAL {trial_number} FAILED")
            print(f"Error: {e}")
            print(f"{'='*60}")
            
            # Re-raise for Optuna to handle
            raise
    
    def _run_training(self, config_path: str, trial_number: int) -> float:
        """
        Run train_drm_multi.py and extract performance metric.
        
        Args:
            config_path: Path to trial config file
            trial_number: Current trial number
            
        Returns:
            Performance metric (prob_discrete_accuracy)
        """
        # Prepare command arguments
        seeds_str = ','.join(map(str, self.multi_run_config['seeds']))
        db_paths_str = ','.join(self.multi_run_config['db_paths'])
        max_parallel = self.multi_run_config.get('max_parallel', 15)
        
        # Construct command
        cmd = [
            sys.executable,  # Use same Python interpreter
            'neural_networks/train_drm_multi.py',
            config_path,
            '--output-dir', 'neural_networks/output',
            '--config-id', f'trial_{trial_number:03d}',
            '--seeds', seeds_str,
            '--db-paths', db_paths_str,
            '--max-parallel', str(max_parallel)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"Working directory: {os.getcwd()}")
        
        # Run the command
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=project_root,  # Ensure we run from project root
                timeout=3600  # 1 hour timeout per trial
            )
            
            if result.returncode != 0:
                print(f"Command failed with return code {result.returncode}")
                print(f"STDERR: {result.stderr}")
                print(f"STDOUT: {result.stdout}")
                raise RuntimeError(f"train_drm_multi.py failed with return code {result.returncode}")
            
            # Extract performance metric from stdout
            performance_metric = self._extract_performance_metric(result.stdout)
            
            print(f"Training completed successfully")
            print(f"Performance metric: {performance_metric:.4f}")
            
            return performance_metric
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Training timeout (1 hour exceeded)")
        except Exception as e:
            print(f"Error running training: {e}")
            raise
    
    def _extract_performance_metric(self, stdout: str) -> float:
        """
        Extract the primary performance metric from train_drm_multi.py output.
        
        Args:
            stdout: Standard output from train_drm_multi.py
            
        Returns:
            Primary metric (prob_discrete_accuracy)
        """
        # Look for the final summary line
        lines = stdout.strip().split('\n')
        
        for line in reversed(lines):  # Search from end
            if 'Primary metric (test_prob_discrete_accuracy):' in line:
                try:
                    # Extract number after the colon
                    metric_str = line.split(':')[-1].strip()
                    return float(metric_str)
                except (ValueError, IndexError) as e:
                    print(f"Error parsing metric from line: {line}")
                    print(f"Parse error: {e}")
        
        # Fallback: look for other patterns
        for line in reversed(lines):
            if 'test_prob_discrete_accuracy' in line and ':' in line:
                try:
                    metric_str = line.split(':')[-1].strip()
                    return float(metric_str)
                except (ValueError, IndexError):
                    continue
        
        print("Could not extract performance metric from output:")
        print("="*50)
        print(stdout)
        print("="*50)
        raise ValueError("Could not extract performance metric from train_drm_multi.py output")
    
    def _save_results(self):
        """Save current results to JSON file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.trial_results, f, indent=2)
    
    def run_sweep(self, n_trials: Optional[int] = None) -> None:
        """
        Run the complete parameter sweep.
        
        Args:
            n_trials: Number of trials to run (default: from sweep config)
        """
        if n_trials is None:
            n_trials = self.sweep_info.get('n_trials', 100)
        
        print(f"\n{'='*60}")
        print(f"STARTING PARAMETER SWEEP")
        print(f"Sweep: {self.sweep_info.get('name', 'Unknown')}")
        print(f"Description: {self.sweep_info.get('description', 'No description')}")
        print(f"Number of trials: {n_trials}")
        print(f"Seeds per trial: {len(self.multi_run_config['seeds'])}")
        print(f"Datasets per trial: {len(self.multi_run_config['db_paths'])}")
        print(f"Total runs: {n_trials * len(self.multi_run_config['seeds']) * len(self.multi_run_config['db_paths'])}")
        print(f"{'='*60}")
        
        # Create or load Optuna study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction='maximize',  # We want to maximize prob_discrete_accuracy
            load_if_exists=True
        )
        
        print(f"Study created/loaded: {study.study_name}")
        print(f"Previous trials: {len(study.trials)}")
        
        # Run optimization
        sweep_start_time = time.time()
        
        try:
            study.optimize(self.objective, n_trials=n_trials)
            
        except KeyboardInterrupt:
            print("\nSweep interrupted by user")
        except Exception as e:
            print(f"\nSweep failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        # Final summary
        sweep_runtime = time.time() - sweep_start_time
        self._print_final_summary(study, sweep_runtime)
    
    def _print_final_summary(self, study: optuna.Study, sweep_runtime: float):
        """Print final sweep summary."""
        print(f"\n{'='*60}")
        print(f"SWEEP COMPLETED")
        print(f"{'='*60}")
        print(f"Total runtime: {sweep_runtime:.1f} seconds ({sweep_runtime/3600:.1f} hours)")
        print(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        
        if study.best_trial:
            print(f"\nBest trial: {study.best_trial.number}")
            print(f"Best performance: {study.best_value:.4f}")
            print("Best parameters:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")
        
        print(f"\nResults saved to: {self.sweep_folder}")
        print(f"Optuna database: {self.storage}")
        print(f"Trial configs: {self.sweep_folder}/trial_*_config.yaml")
        print(f"Results summary: {self.results_file}")
        print(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run parameter sweep optimization")
    parser.add_argument(
        "sweep_config", 
        help="Path to sweep configuration YAML file"
    )
    parser.add_argument(
        "--sweep-id", 
        help="Custom sweep identifier (default: datetime-based)"
    )
    parser.add_argument(
        "--n-trials", 
        type=int,
        help="Number of trials to run (default: from sweep config)"
    )
    parser.add_argument(
        "--study-name", 
        help="Optuna study name (default: from sweep config)"
    )
    parser.add_argument(
        "--storage", 
        help="Optuna storage URL (default: SQLite in sweep folder)"
    )
    
    args = parser.parse_args()
    
    # Validate sweep config exists
    if not Path(args.sweep_config).exists():
        print(f"Error: Sweep config file not found: {args.sweep_config}")
        sys.exit(1)
    
    # Create and run sweep
    try:
        sweep = ParameterSweep(
            sweep_config_path=args.sweep_config,
            sweep_id=args.sweep_id,
            study_name=args.study_name,
            storage=args.storage
        )
        
        sweep.run_sweep(n_trials=args.n_trials)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()