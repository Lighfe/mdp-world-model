#!/usr/bin/env python3
"""
Parameter Sampler - Data-Driven Approach
Reads sweep configuration files and samples parameters accordingly.
"""

import yaml
import optuna
from pathlib import Path
from typing import Dict, Any, Optional, List
import re
import itertools
from collections import defaultdict


class SweepParameterSampler:
    """
    Data-driven parameter sampler that reads sweep configuration files.
    """
    
    def __init__(self, sweep_config_path: str):
        """
        Initialize sampler with sweep configuration.
        
        Args:
            sweep_config_path: Path to sweep configuration YAML file
        """
        self.sweep_config_path = sweep_config_path
        self.sweep_config = self._load_sweep_config()
        self.parameters = self.sweep_config.get('parameters', {})
        
    def _load_sweep_config(self) -> Dict[str, Any]:
        """Load and validate sweep configuration."""
        with open(self.sweep_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['sweep', 'base_config', 'parameters']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in sweep config")
        
        return config
    
    def _evaluate_condition(self, condition: str, current_params: Dict[str, Any]) -> bool:
        """
        Evaluate a conditional parameter expression.
        
        Args:
            condition: String condition like "model.use_gumbel == true"
            current_params: Currently sampled parameters
            
        Returns:
            True if condition is met, False otherwise
        """
        if not condition:
            return True
            
        # Simple condition parsing: "param_path == value" or "param_path != value"
        if "==" in condition:
            param_path, expected_value = condition.split("==")
            param_path = param_path.strip()
            expected_value = expected_value.strip()
            
            # Convert string values to appropriate types
            if expected_value.lower() == 'true':
                expected_value = True
            elif expected_value.lower() == 'false':
                expected_value = False
            elif expected_value.replace('.', '').isdigit():
                expected_value = float(expected_value) if '.' in expected_value else int(expected_value)
            else:
                expected_value = expected_value.strip('"\'')  # Remove quotes
            
            return current_params.get(param_path) == expected_value
            
        elif "!=" in condition:
            param_path, expected_value = condition.split("!=")
            param_path = param_path.strip()
            expected_value = expected_value.strip()
            
            # Same type conversion as above
            if expected_value.lower() == 'true':
                expected_value = True
            elif expected_value.lower() == 'false':
                expected_value = False
            elif expected_value.replace('.', '').isdigit():
                expected_value = float(expected_value) if '.' in expected_value else int(expected_value)
            else:
                expected_value = expected_value.strip('"\'')
            
            return current_params.get(param_path) != expected_value
        
        # Handle 'in' conditions for multiple values
        elif " in " in condition:
            param_path, values_str = condition.split(" in ")
            param_path = param_path.strip()
            values_str = values_str.strip()
            
            # Parse list of values: ["value1", "value2"]
            if values_str.startswith('[') and values_str.endswith(']'):
                values_str = values_str[1:-1]  # Remove brackets
                expected_values = [v.strip().strip('"\'') for v in values_str.split(',')]
                return current_params.get(param_path) in expected_values
        
        # If we can't parse the condition, assume it's true (safe fallback)
        print(f"Warning: Could not parse condition '{condition}', assuming True")
        return True
    
    def _compute_derived_parameter(self, formula: str, current_params: Dict[str, Any]) -> Any:
        """
        Compute a derived parameter from a formula.
        
        Args:
            formula: Formula string like "training.lr / 10"
            current_params: Currently sampled parameters
            
        Returns:
            Computed value
        """
        # Simple formula evaluation - for now just handle basic arithmetic
        # Replace parameter paths with actual values
        eval_formula = formula
        for param_path, value in current_params.items():
            if param_path in formula:
                eval_formula = eval_formula.replace(param_path, str(value))
        
        try:
            # Use eval for simple arithmetic (be careful in production!)
            result = eval(eval_formula)
            return result
        except Exception as e:
            print(f"Error evaluating formula '{formula}': {e}")
            return None
    
    def sample_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample trial parameters based on sweep configuration.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of sampled parameters
        """
        sampled_params = {}
        
        # First pass: sample non-conditional and non-derived parameters
        for param_path, param_config in self.parameters.items():
            param_type = param_config.get('type')
            condition = param_config.get('condition')
            
            # Skip conditional and derived parameters for now
            if condition or param_type == 'derived':
                continue
                
            value = self._sample_single_parameter(trial, param_path, param_config, sampled_params)
            sampled_params[param_path] = value
        
        # Second pass: sample conditional parameters
        for param_path, param_config in self.parameters.items():
            param_type = param_config.get('type')
            condition = param_config.get('condition')
            
            if condition and param_type != 'derived':
                if self._evaluate_condition(condition, sampled_params):
                    value = self._sample_single_parameter(trial, param_path, param_config, sampled_params)
                    sampled_params[param_path] = value
                else:
                    sampled_params[param_path] = None  # Will be skipped in config generation
        
        # Third pass: compute derived parameters
        for param_path, param_config in self.parameters.items():
            param_type = param_config.get('type')
            
            if param_type == 'derived':
                formula = param_config.get('formula')
                if formula:
                    value = self._compute_derived_parameter(formula, sampled_params)
                    sampled_params[param_path] = value
        
        return sampled_params
    
    def _sample_single_parameter(self, trial: optuna.Trial, param_path: str, param_config: Dict[str, Any], current_params: Dict[str, Any]) -> Any:
        """
        Sample a single parameter based on its configuration.
        
        Args:
            trial: Optuna trial object
            param_path: Parameter path (e.g., "model.num_states")
            param_config: Parameter configuration dict
            current_params: Currently sampled parameters (for conditional ranges)
            
        Returns:
            Sampled parameter value
        """
        param_type = param_config.get('type')
        
        if param_type == 'categorical':
            choices = param_config.get('choices', [])
            return trial.suggest_categorical(param_path, choices)
        
        elif param_type == 'float':
            # Check for conditional ranges
            if 'conditional_ranges' in param_config:
                ranges = param_config['conditional_ranges']
                for range_config in ranges:
                    condition = range_config.get('condition')
                    if self._evaluate_condition(condition, current_params):
                        min_val = range_config.get('min')
                        max_val = range_config.get('max')
                        log_scale = range_config.get('log', False)
                        break
                else:
                    # Fallback to default range
                    min_val = param_config.get('min')
                    max_val = param_config.get('max')
                    log_scale = param_config.get('log', False)
            else:
                # Standard range
                min_val = param_config.get('min')
                max_val = param_config.get('max')
                log_scale = param_config.get('log', False)
            
            if log_scale:
                return trial.suggest_float(param_path, min_val, max_val, log=True)
            else:
                return trial.suggest_float(param_path, min_val, max_val)
        
        elif param_type == 'int':
            min_val = param_config.get('min')
            max_val = param_config.get('max')
            step = param_config.get('step', 1)
            return trial.suggest_int(param_path, min_val, max_val, step=step)
        
        else:
            raise ValueError(f"Unknown parameter type '{param_type}' for parameter '{param_path}'")
    
    def get_sweep_info(self) -> Dict[str, Any]:
        """Get sweep metadata."""
        return self.sweep_config.get('sweep', {})
    
    def get_multi_run_config(self) -> Dict[str, Any]:
        """Get multi-run configuration for train_drm_multi.py."""
        return self.sweep_config.get('multi_run', {})
    
    def get_base_config_path(self) -> str:
        """Get base configuration file path with proper folder structure."""
        base_config = self.sweep_config.get('base_config', 'base.yaml')
        
        # If it's a relative path without folder, assume it's in scripts/configs/
        if not ('/' in base_config or '\\' in base_config):
            # Check if we're running from project root or scripts directory
            if Path('scripts/configs').exists():
                base_config = f'scripts/configs/{base_config}'
            elif Path('configs').exists():
                base_config = f'configs/{base_config}'
            else:
                # Try to find base.yaml in common locations
                for candidate in ['scripts/configs/base.yaml', 'configs/base.yaml', 'base.yaml']:
                    if Path(candidate).exists():
                        base_config = candidate
                        break
                else:
                    raise FileNotFoundError(f"Could not find base.yaml in scripts/configs/, configs/, or current directory")
        
        return base_config
    
    def get_fixed_params(self) -> Dict[str, Any]:
        """Get fixed parameters for this sweep."""
        return self.sweep_config.get('fixed_params', {})

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
        derived_params = {}
        
        # Fix the cycled categorical parameters
        for param_name, value in zip(self.combo_param_names, current_combo):
            sampled_params[param_name] = value
        
        # Sample all non-derived parameters first
        for param_path, param_config in self.parameters.items():
            if param_path not in sampled_params:  # Skip already fixed categoricals
                
                param_type = param_config.get('type')
                
                if param_type == 'derived':
                    # Store derived parameters for later calculation
                    derived_params[param_path] = param_config
                    continue
                
                # Check conditions first
                if not self._evaluate_parameter_condition(param_config, sampled_params):
                    sampled_params[param_path] = None
                    continue
                
                sampled_params[param_path] = self._sample_single_parameter(
                    trial, param_path, param_config, sampled_params
                )
        
        # Now compute derived parameters
        for param_path, param_config in derived_params.items():
            sampled_params[param_path] = self._compute_derived_parameter(
                param_path, param_config, sampled_params
            )
        
        # Track this trial for the current combo
        self.combo_trial_counts[combo_idx] += 1
        
        return {k: v for k, v in sampled_params.items() if v is not None}
    
    def _compute_derived_parameter(self, param_path: str, param_config: Dict[str, Any], 
                                  current_params: Dict[str, Any]) -> Any:
        """Compute a derived parameter using its formula."""
        
        formula = param_config.get('formula')
        if not formula:
            raise ValueError(f"Derived parameter '{param_path}' missing formula")
        
        try:
            # Simple formula evaluation - replace parameter references
            # This handles formulas like "training.lr / 10"
            eval_formula = formula
            
            # Replace parameter references with their values
            for other_param, value in current_params.items():
                if value is not None and other_param in eval_formula:
                    eval_formula = eval_formula.replace(other_param, str(value))
            
            # Evaluate the formula safely
            result = eval(eval_formula, {"__builtins__": {}})
            
            print(f"  Derived {param_path}: {formula} = {result}")
            return result
            
        except Exception as e:
            raise ValueError(f"Could not compute derived parameter '{param_path}' with formula '{formula}': {e}")
    
    
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

def test_parameter_sampler():
    """Test the parameter sampler with the actual sweep config."""
    
    print("Testing parameter sampler with actual sweep config...")
    
    # Updated path for new folder structure
    sweep_config_path = "scripts/configs/sweeps/second_sweep_kl.yaml"
    if not Path(sweep_config_path).exists():
        print(f"Error: {sweep_config_path} not found!")
        print("Please make sure your sweep config is in scripts/configs/sweeps/")
        return
    
    try:
        sampler = CyclicCategoricalSampler(sweep_config_path)
        
        print(f"Loaded sweep: {sampler.get_sweep_info()}")
        print(f"Base config: {sampler.get_base_config_path()}")
        print(f"Multi-run config: {sampler.get_multi_run_config()}")
        
        # Generate test samples
        for i in range(5):
            study = optuna.create_study()
            trial = study.ask()
            
            params = sampler.sample_trial_params(trial)
            
            print(f"\n=== Trial {i+1} ===")
            for key, value in params.items():
                if value is not None:
                    print(f"  {key}: {value}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parameter_sampler()