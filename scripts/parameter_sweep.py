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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.parameter_sampler import SweepParameterSampler
from scripts.config_generator import create_trial_config


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
        """
        Create TPE sampler based on exploration/exploitation strategy.
        
        Args:
            sampler_type: 'explorative', 'balanced', or 'exploitative'
            
        Returns:
            Configured TPESampler with multivariate=True and group=True for conditional parameters
        """
        if sampler_type == 'explorative':
            # MORE EXPLORATIVE: Good for unknown parameter spaces
            return optuna.samplers.TPESampler(
                n_startup_trials=20,                    # More random exploration (20% of 100 trials)
                gamma=lambda n: max(1, int(0.35 * n)),  # Top 35% considered "good" (less selective)
                multivariate=True,                      # Model parameter interactions
                group=True,                             # Handle conditional parameters properly
                seed=42                                 # Reproducible results
            )
        
        elif sampler_type == 'exploitative':
            # MORE EXPLOITATIVE: Good when you have prior knowledge
            return optuna.samplers.TPESampler(
                n_startup_trials=8,                     # Less random exploration (8% of 100 trials)
                gamma=lambda n: max(1, int(0.15 * n)),  # Top 15% considered "good" (very selective)
                multivariate=True,                      # Model parameter interactions
                group=True,                             # Handle conditional parameters properly
                seed=42
            )
        
        else:  # 'balanced' (default)
            # BALANCED: Recommended for most cases
            return optuna.samplers.TPESampler(
                n_startup_trials=12,                    # Standard exploration (12% of 100 trials)
                gamma=lambda n: max(1, int(0.25 * n)),  # Top 25% considered "good" (standard)
                multivariate=True,                      # Model parameter interactions
                group=True,                             # Handle conditional parameters properly
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
            
            # 2. Generate trial config
            trial_config_path = self.sweep_configs_dir / f"trial_{trial_number:03d}_config.yaml"
            create_trial_config(
                base_config_path=self.sampler.get_base_config_path(),
                trial_params=trial_params,
                fixed_params=self.sampler.get_fixed_params(),
                output_config_path=trial_config_path,
                trial_output_dir=self.sweep_output_dir / f"trial_{trial_number:03d}"
            )
            
            print(f"Generated config: {trial_config_path}")
            
            # 3. Run training with train_drm_multi.py
            multi_run_config = self.multi_run_config
            n_seeds = len(multi_run_config.get('seeds', [42]))
            n_datasets = len(multi_run_config.get('datasets', [1]))
            aggregation = multi_run_config.get('aggregation', 'mean')
            
            print(f"Running multi-run: {n_seeds} seeds × {n_datasets} datasets")
            print(f"Aggregation method: {aggregation}")
            
            # Build command for train_drm_multi.py
            cmd = [
                'python', 'train_drm_multi.py',
                '--config', str(trial_config_path),
                '--seeds'] + [str(s) for s in multi_run_config.get('seeds', [42])]
            
            if 'datasets' in multi_run_config:
                cmd.extend(['--datasets'] + [str(d) for d in multi_run_config['datasets']])
            
            cmd.extend([
                '--aggregation', aggregation,
                '--save_configs'  # Save individual run configs for debugging
            ])
            
            print(f"Command: {' '.join(cmd)}")
            
            # Execute training
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
            
            if result.returncode != 0:
                print(f"❌ Training failed for trial {trial_number}")
                print("STDERR:", result.stderr)
                print("STDOUT:", result.stdout)
                raise optuna.TrialPruned()
            
            # 4. Parse results from train_drm_multi.py output
            trial_output_dir = self.sweep_output_dir / f"trial_{trial_number:03d}"
            results_file = trial_output_dir / "aggregated_results.json"
            
            if not results_file.exists():
                print(f"❌ Results file not found: {results_file}")
                raise optuna.TrialPruned()
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Extract primary metric for optimization
            prob_discrete_accuracy = results.get('prob_discrete_accuracy', {}).get(aggregation, 0.0)
            
            trial_runtime = time.time() - trial_start_time
            
            print(f"✅ Trial {trial_number} completed in {trial_runtime:.1f}s")
            print(f"Primary metric (prob_discrete_accuracy): {prob_discrete_accuracy:.4f}")
            
            # Store trial results for later analysis
            trial_result = {
                'trial_number': trial_number,
                'parameters': trial_params,
                'prob_discrete_accuracy': prob_discrete_accuracy,
                'all_metrics': results,
                'runtime_seconds': trial_runtime,
                'timestamp': datetime.now().isoformat()
            }
            
            self.trial_results.append(trial_result)
            
            # Save incremental results
            with open(self.results_file, 'w') as f:
                json.dump(self.trial_results, f, indent=2)
            
            return prob_discrete_accuracy
            
        except optuna.TrialPruned:
            print(f"🚫 Trial {trial_number} was pruned")
            raise
        except Exception as e:
            print(f"❌ Trial {trial_number} failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise optuna.TrialPruned()
    
    def run_sweep(self, n_trials: int = 100) -> None:
        """
        Execute the parameter sweep.
        
        Args:
            n_trials: Number of trials to run
        """
        sweep_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"STARTING PARAMETER SWEEP")
        print(f"{'='*60}")
        print(f"Sweep ID: {self.sweep_id}")
        print(f"Number of trials: {n_trials}")
        print(f"Study name: {self.study_name}")
        print(f"Storage: {self.storage}")
        print(f"Output directory: {self.sweep_output_dir}")
        print(f"Sampler type: {self.sampler_type}")
        print(f"TPE sampler: {type(self.tpe_sampler).__name__}")
        
        # Print sampler details
        if self.sampler_type == 'explorative':
            print(f"Strategy: 20 startup trials, top 35% as 'good' (more exploration)")
        elif self.sampler_type == 'exploitative':
            print(f"Strategy: 8 startup trials, top 15% as 'good' (focused exploitation)")
        else:  # balanced
            print(f"Strategy: 12 startup trials, top 25% as 'good' (balanced)")
        
        print(f"All strategies use: multivariate=True, group=True (handles conditional parameters)")
        print(f"{'='*60}")
        
        # Create or load study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction='maximize',  # We want to maximize prob_discrete_accuracy
            sampler=self.tpe_sampler,
            load_if_exists=True
        )
        
        print(f"Study created/loaded. Existing trials: {len(study.trials)}")
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)
        
        sweep_runtime = time.time() - sweep_start_time
        
        # Print final summary
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
            print(f"  - Used 8 random startup trials")
            print(f"  - Considered top 15% as 'good' trials")
            print(f"  - Good for refining known good regions")
        else:
            print(f"  - Used 12 random startup trials")
            print(f"  - Considered top 25% as 'good' trials")
            print(f"  - Balanced exploration/exploitation")
        print(f"  - All strategies use multivariate=True + group=True for optimal conditional parameter handling")
        
        print(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run parameter sweep with Optuna")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to sweep configuration file')
    parser.add_argument('--sweep-id', type=str, default=None,
                       help='Custom sweep identifier (default: auto-generated)')
    parser.add_argument('--study-name', type=str, default=None,
                       help='Optuna study name (default: from config)')
    parser.add_argument('--storage', type=str, default=None,
                       help='Optuna storage URL (default: SQLite in output dir)')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of trials to run (default: 100)')
    parser.add_argument('--sampler-type', type=str, default='balanced',
                       choices=['explorative', 'balanced', 'exploitative'],
                       help='TPE sampler strategy (default: balanced)')
    
    args = parser.parse_args()
    
    # Validate sweep config exists
    if not Path(args.config).exists():
        print(f"Error: Sweep config not found: {args.config}")
        sys.exit(1)
    
    # Create and run sweep
    sweep = ParameterSweep(
        sweep_config_path=args.config,
        sweep_id=args.sweep_id,
        study_name=args.study_name,
        storage=args.storage,
        sampler_type=args.sampler_type
    )
    
    sweep.run_sweep(n_trials=args.n_trials)


if __name__ == "__main__":
    main()