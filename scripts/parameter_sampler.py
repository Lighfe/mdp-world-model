#!/usr/bin/env python3
"""
Generic Parameter Sampler - Data-Driven Approach
Reads sweep configuration files and samples parameters accordingly.
"""

import yaml
import optuna
from pathlib import Path
from typing import Dict, Any, Optional, List
import re


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
                
            value = self._sample_single_parameter(trial, param_path, param_config)
            sampled_params[param_path] = value
        
        # Second pass: sample conditional parameters
        for param_path, param_config in self.parameters.items():
            param_type = param_config.get('type')
            condition = param_config.get('condition')
            
            if condition and param_type != 'derived':
                if self._evaluate_condition(condition, sampled_params):
                    value = self._sample_single_parameter(trial, param_path, param_config)
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
    
    def _sample_single_parameter(self, trial: optuna.Trial, param_path: str, param_config: Dict[str, Any]) -> Any:
        """
        Sample a single parameter based on its configuration.
        
        Args:
            trial: Optuna trial object
            param_path: Parameter path (e.g., "model.num_states")
            param_config: Parameter configuration dict
            
        Returns:
            Sampled parameter value
        """
        param_type = param_config.get('type')
        
        if param_type == 'categorical':
            choices = param_config.get('choices', [])
            return trial.suggest_categorical(param_path, choices)
        
        elif param_type == 'float':
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
        """Get base configuration file path."""
        return self.sweep_config.get('base_config', 'configs/base.yaml')
    
    def get_fixed_params(self) -> Dict[str, Any]:
        """Get fixed parameters for this sweep."""
        return self.sweep_config.get('fixed_params', {})


def test_generic_sampler():
    """Test the generic parameter sampler."""
    # For testing, create a minimal sweep config
    test_config = {
        'sweep': {'name': 'test_sweep', 'n_trials': 10},
        'base_config': 'configs/base.yaml',
        'parameters': {
            'model.num_states': {
                'type': 'categorical',
                'choices': [4, 6, 8]
            },
            'model.use_gumbel': {
                'type': 'categorical',
                'choices': [True, False]
            },
            'model.initial_temp': {
                'type': 'float',
                'min': 0.5,
                'max': 5.0,
                'condition': 'model.use_gumbel == true'
            },
            'training.lr': {
                'type': 'float',
                'min': 1e-5,
                'max': 1e-3,
                'log': True
            },
            'training.min_lr': {
                'type': 'derived',
                'formula': 'training.lr / 10'
            }
        }
    }
    
    # Save test config
    test_config_path = "/tmp/test_sweep_config.yaml"
    with open(test_config_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False, indent=2)
    
    # Test the sampler
    print("Testing generic parameter sampler...")
    
    try:
        sampler = SweepParameterSampler(test_config_path)
        
        # Generate a few test samples
        for i in range(3):
            study = optuna.create_study()
            trial = study.ask()
            
            params = sampler.sample_trial_params(trial)
            
            print(f"\n=== Test Sample {i+1} ===")
            for key, value in params.items():
                print(f"  {key}: {value}")
            
            # Verify conditional logic
            if params.get('model.use_gumbel') is True:
                assert params.get('model.initial_temp') is not None, "initial_temp should be sampled when use_gumbel=True"
            else:
                assert params.get('model.initial_temp') is None, "initial_temp should be None when use_gumbel=False"
            
            # Verify derived parameter
            expected_min_lr = params.get('training.lr', 0) / 10
            actual_min_lr = params.get('training.min_lr')
            assert abs(actual_min_lr - expected_min_lr) < 1e-10, f"Derived parameter incorrect: {actual_min_lr} != {expected_min_lr}"
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            import os
            os.remove(test_config_path)
        except:
            pass


if __name__ == "__main__":
    test_generic_sampler()