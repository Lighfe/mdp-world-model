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
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import itertools
from collections import defaultdict
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.parameter_sampler import SweepParameterSampler
from scripts.config_generator import create_trial_config

class CyclicCategoricalSampler(SweepParameterSampler):
    """
    Ensures each categorical combination gets equal optimization attention.
    Cycles through categorical combinations while using TPE for continuous parameters.
    """
    
    def __init__(self, sweep_config_path: str, min_trials_per_combo: int = 10):
        super().__init__(sweep_config_path)
        self.min_trials_per_combo = min_trials_per_combo
        
        # Identify categorical parameters that should be cycled
        self.cycle_categoricals = {}
        self.other_categoricals = {}
        
        # Read which parameters should be cycled from config
        cycle_params = self.sweep_config.get('cycle_categoricals', [])
        
        for param_path, param_config in self.parameters.items():
            if param_config.get('type') == 'categorical':
                if param_path in cycle_params:
                    self.cycle_categoricals[param_path] = param_config['choices']
                else:
                    self.other_categoricals[param_path] = param_config['choices']
        
        # Generate all combinations of cycled categoricals
        if self.cycle_categoricals:
            param_names = list(self.cycle_categoricals.keys())
            param_choices = [self.cycle_categoricals[name] for name in param_names]
            self.categorical_combos = list(itertools.product(*param_choices))
            self.combo_param_names = param_names
        else:
            self.categorical_combos = [()]
            self.combo_param_names = []
        
        print(f"Cycling through {len(self.categorical_combos)} categorical combinations:")
        for i, combo in enumerate(self.categorical_combos):
            combo_dict = dict(zip(self.combo_param_names, combo))
            print(f"  Combo {i}: {combo_dict}")
        
        # Track trials per combination
        self.combo_trial_counts = defaultdict(int)
        
    def sample_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Sample parameters with cyclic categorical selection."""
        
        # Determine which categorical combination to use
        combo_idx = self._select_categorical_combination(trial.number)
        current_combo = self.categorical_combos[combo_idx]
        
        print(f"Trial {trial.number}: Using categorical combo {combo_idx}: {dict(zip(self.combo_param_names, current_combo))}")
        
        sampled_params = {}
        
        # Fix the cycled categorical parameters
        for param_name, value in zip(self.combo_param_names, current_combo):
            sampled_params[param_name] = value
        
        # Sample all other parameters normally (including non-cycled categoricals)
        for param_path, param_config in self.parameters.items():
            if param_path not in sampled_params:  # Skip already fixed categoricals
                # Check conditions first
                if not self._evaluate_parameter_condition(param_config, sampled_params):
                    sampled_params[param_path] = None
                    continue
                
                sampled_params[param_path] = self._sample_single_parameter(
                    trial, param_path, param_config, sampled_params
                )
        
        # Track this trial for the current combo
        self.combo_trial_counts[combo_idx] += 1
        
        return {k: v for k, v in sampled_params.items() if v is not None}
    
    def _select_categorical_combination(self, trial_number: int) -> int:
        """
        Select which categorical combination to use for this trial.
        
        Strategy: 
        - Phase 1 (Strictly Balanced): Ensure exactly min_trials_per_combo for each
        - Phase 2 (Flexibly Balanced): Continue cycling but could adapt based on performance
        """
        n_combos = len(self.categorical_combos)
        strict_phase_trials = self.min_trials_per_combo * n_combos
        
        if trial_number < strict_phase_trials:
            # PHASE 1: Strictly balanced - exact round-robin
            combo_idx = trial_number % n_combos
            print(f"  [STRICT PHASE] Trial {trial_number}/{strict_phase_trials-1}")
            return combo_idx
        else:
            # PHASE 2: Flexibly balanced - still cycle but could be enhanced
            # For now: continue round-robin (could add performance-based weighting later)
            combo_idx = trial_number % n_combos
            print(f"  [FLEXIBLE PHASE] Trial {trial_number}")
            return combo_idx
    
    def _evaluate_parameter_condition(self, param_config: Dict[str, Any], 
                                     current_params: Dict[str, Any]) -> bool:
        """Check if parameter should be sampled based on conditions."""
        condition = param_config.get('condition')
        if not condition:
            return True
        return self._evaluate_condition(condition, current_params)

class ParameterSweep:
    """
    Main parameter sweep orchestrator using Optuna.
    """
    
    def __init__(self, sweep_config_path: str, sweep_id: Optional[str] = None, 
                 study_name: Optional[str] = None, storage: Optional[str] = None,
                 sampler_type: str = 'balanced'):
        """
        Initialize parameter sweep.
        
        Args:
            sweep_config_path: Path to sweep configuration file
            sweep_id: Custom sweep identifier (default: datetime-based)
            study_name: Optuna study name (default: from sweep config)
            storage: Optuna storage URL (default: SQLite in sweep folder)
            sampler_type: TPE sampler type ('explorative', 'balanced', 'exploitative')
        """
        self.sweep_config_path = sweep_config_path
    
        # Check if this sweep uses cyclic categoricals
        with open(sweep_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'cycle_categoricals' in config:
            print("Using Cyclic Categorical Sampler")
            min_trials = config.get('min_trials_per_combo', 10)
            self.sampler = CyclicCategoricalSampler(sweep_config_path, min_trials)
        else:
            print("Using Standard Parameter Sampler")
            self.sampler = SweepParameterSampler(sweep_config_path)
        
        # Get sweep info early
        self.sweep_info = self.sampler.get_sweep_info()
        self.multi_run_config = self.sampler.get_multi_run_config()
        
        # Generate sweep_id if not provided
        if sweep_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sweep_id = f"sweep_{timestamp}"
        
        self.sweep_id = sweep_id
        
        # === OUTPUT DIRECTORY ORGANIZATION ===
        # Create main sweep directory in neural_networks/output/{sweep_id}/
        self.sweep_output_dir = Path("neural_networks/output") / sweep_id
        self.sweep_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        self.sweep_configs_dir = self.sweep_output_dir / "configs"
        self.sweep_configs_dir.mkdir(exist_ok=True)
        
        print(f"Sweep output directory: {self.sweep_output_dir}")
        
        # Copy sweep config to sweep directory for reference
        dest_config = self.sweep_configs_dir / "sweep_config.yaml"
        shutil.copy2(sweep_config_path, dest_config)
        
        # Set up Optuna study with custom sampler
        self.study_name = study_name or self.sweep_info.get('name', 'parameter_sweep')
        self.storage = storage or f"sqlite:///{self.sweep_output_dir.absolute()}/optuna_study.db"
        self.sampler_type = sampler_type
        
        # Results tracking
        self.results_file = self.sweep_output_dir / "sweep_results.json"
        self.trial_results = []
        
        print(f"Study name: {self.study_name}")
        print(f"Storage: {self.storage}")
        print(f"Sampler type: {sampler_type}")
        
        # Create custom TPE sampler based on type
        self.tpe_sampler = self._create_tpe_sampler(sampler_type)
        print(f"TPE sampler configured: {type(self.tpe_sampler).__name__}")
    
    def _create_tpe_sampler(self, sampler_type: str) -> optuna.samplers.TPESampler:
        """Create TPE sampler - now works with cyclic categorical sampler."""
        # TPE will handle continuous parameters within each categorical combination
        if sampler_type == 'explorative':
            return optuna.samplers.TPESampler(
                n_startup_trials=20,  # Reduced since we're cycling categoricals
                gamma=lambda n: max(1, int(0.35 * n)),
                multivariate=True,
                group=True,
                seed=42
            )
        elif sampler_type == 'exploitative':
            return optuna.samplers.TPESampler(
                n_startup_trials=15,
                gamma=lambda n: max(1, int(0.15 * n)),
                multivariate=True,
                group=True,
                seed=42
            )
        else:  # 'balanced'
            return optuna.samplers.TPESampler(
                n_startup_trials=20,
                gamma=lambda n: max(1, int(0.25 * n)),
                multivariate=True,
                group=True,
                seed=42
            )
    
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
        print(f"Sampler type: {self.sampler_type}")
        print(f"{'='*60}")
        
        try:
            # 1. Sample trial parameters
            trial_params = self.sampler.sample_trial_params(trial)
            print("Trial parameters:")
            for key, value in trial_params.items():
                if value is not None:
                    print(f"  {key}: {value}")
            
            # 2. Generate trial config file - store in sweep configs directory
            trial_config_path = create_trial_config(
                trial_params=trial_params,
                trial_number=trial_number,
                base_config_path=self.sampler.get_base_config_path(),
                sweep_folder=self.sweep_configs_dir,  # Store in {sweep_id}/configs/
                fixed_params=self.sampler.get_fixed_params()
            )
            print(f"Generated config: {trial_config_path}")
            
            # 3. Run train_drm_multi.py with organized output directory structure
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
        
        # === ORGANIZED OUTPUT DIRECTORY STRUCTURE ===
        # Each trial gets its own subdirectory: {sweep_id}/trial_000/, trial_001/, etc.
        trial_config_id = f"trial_{trial_number:03d}"
        
        # Construct command - output will go to neural_networks/output/{sweep_id}/trial_xxx/
        cmd = [
            sys.executable,  # Use same Python interpreter
            'neural_networks/train_drm_multi.py',
            config_path,
            '--output-dir', str(self.sweep_output_dir),  # neural_networks/output/{sweep_id}
            '--config-id', trial_config_id,              # Creates trial_000/, trial_001/, etc.
            '--seeds', seeds_str,
            '--db-paths', db_paths_str,
            '--max-parallel', str(max_parallel)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"Trial output will be in: {self.sweep_output_dir}/{trial_config_id}/")
        
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
        print(f"Output directory: {self.sweep_output_dir}")
        print(f"Sampler: {self.sampler_type} TPE")
        print(f"{'='*60}")
        
        # === DEBUG DATABASE CREATION ===
        print(f"Creating Optuna study...")
        print(f"  Study name: {self.study_name}")
        print(f"  Storage: {self.storage}")
        print(f"  Storage path exists: {Path(self.storage.replace('sqlite:///', '')).parent.exists()}")
        
        try:
            # Create or load Optuna study
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                direction='maximize',  # We want to maximize prob_discrete_accuracy
                sampler=self.tpe_sampler,  # Use custom TPE sampler
                load_if_exists=True
            )
            
            print(f"âœ“ Study created/loaded successfully: {study.study_name}")
            print(f"Previous trials: {len(study.trials)}")
            
            # Verify database file was created
            db_path = Path(self.storage.replace('sqlite:///', ''))
            if db_path.exists():
                print(f"âœ“ Database file created: {db_path}")
                print(f"  Database size: {db_path.stat().st_size} bytes")
            else:
                print(f"âš  Database file not found at: {db_path}")
        
        except Exception as e:
            print(f"âœ— Error creating study: {e}")
            import traceback
            traceback.print_exc()
            raise
        
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
        
        # === FINAL DATABASE CHECK ===
        db_path = Path(self.storage.replace('sqlite:///', ''))
        if db_path.exists():
            print(f"\nâœ“ Final database check: {db_path} ({db_path.stat().st_size} bytes)")
        else:
            print(f"\nâœ— Final database check: Database missing at {db_path}")
        
        # Final summary
        sweep_runtime = time.time() - sweep_start_time
        self._print_final_summary(study, sweep_runtime)
    
    def _print_final_summary(self, study, sweep_runtime):
        """Print final sweep summary with safe error handling."""
        
        print(f"\n{'='*60}")
        print(f"SWEEP COMPLETED")
        print(f"{'='*60}")
        print(f"Total runtime: {sweep_runtime:.1f} seconds ({sweep_runtime/3600:.1f} hours)")
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        
        print(f"Completed trials: {len(completed_trials)}")
        print(f"Failed trials: {len(failed_trials)}")
        
        # Safe best trial access
        try:
            if len(completed_trials) > 0 and study.best_trial:
                print(f"\nBest trial: {study.best_trial.number}")
                print(f"Best performance: {study.best_value:.4f}")
                print("Best parameters:")
                for key, value in study.best_params.items():
                    print(f"  {key}: {value}")
            else:
                print(f"\nNo completed trials - cannot determine best configuration.")
                if len(failed_trials) > 0:
                    print("Check the error messages above to diagnose issues.")
        except Exception as e:
            print(f"\nCould not access best trial (no successful completions): {e}")
        
        print(f"\nResults saved to: {self.sweep_output_dir}")
        print(f"Directory structure:")
        print(f"  - Trial outputs: {self.sweep_output_dir}/trial_*/")
        print(f"  - Trial configs: {self.sweep_configs_dir}/trial_*_config.yaml")
        print(f"  - Optuna database: {self.sweep_output_dir}/optuna_study.db")
        print(f"  - Results summary: {self.results_file}")
        
        # TPE Sampler info
        print(f"\nTPE Sampler Summary ({self.sampler_type}):")
        if self.sampler_type == 'explorative':
            print(f"  - Used 20 random startup trials")
            print(f"  - Considered top 35% as 'good' trials")  
            print(f"  - Good for unknown parameter spaces")
        elif self.sampler_type == 'exploitative':
            print(f"  - Used 15 random startup trials")
            print(f"  - Considered top 15% as 'good' trials")
            print(f"  - Good for refining known good regions")
        else:
            print(f"  - Used 20 random startup trials")
            print(f"  - Considered top 25% as 'good' trials")
            print(f"  - Balanced exploration/exploitation")
        
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
    parser.add_argument(
        "--sampler-type",
        choices=['explorative', 'balanced', 'exploitative'],
        default='balanced',
        help="TPE sampler strategy (default: balanced)"
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
            storage=args.storage,
            sampler_type=args.sampler_type
        )
        
        sweep.run_sweep(n_trials=args.n_trials)
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()