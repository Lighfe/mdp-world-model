#!/usr/bin/env python3
"""
Step 1: Parameter Space Definition and Sampling
Handles conditional parameter sampling for Optuna-based hyperparameter optimization.
"""

import optuna
from typing import Dict, Any, Optional


def sample_trial_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sample trial parameters with proper conditional logic.
    
    Args:
        trial: Optuna trial object for parameter sampling
    
    Returns:
        Dictionary of sampled parameters ready for config generation
    """
    params = {}
    
    # === MODEL ARCHITECTURE ===
    params['num_states'] = trial.suggest_categorical('num_states', [4, 6, 8])
    params['value_method'] = trial.suggest_categorical('value_method', ['angular', 'identity'])
    
    # Target encoder settings (conditional)
    params['use_target_encoder'] = trial.suggest_categorical('use_target_encoder', [True, False])
    if params['use_target_encoder']:
        params['ema_decay'] = trial.suggest_float('ema_decay', 0.5, 0.996)
    else:
        params['ema_decay'] = None  # Will use base.yaml default
    
    # Gumbel softmax settings (conditional) 
    params['use_gumbel'] = trial.suggest_categorical('use_gumbel', [True, False])
    if params['use_gumbel']:
        params['initial_temp'] = trial.suggest_float('initial_temp', 0.5, 5.0)
        params['min_temp'] = trial.suggest_float('min_temp', 0.1, 0.5)
    else:
        params['initial_temp'] = None
        params['min_temp'] = None
    
    # Weight initialization
    params['encoder_init_method'] = trial.suggest_categorical('encoder_init_method', ['he', 'xavier_uniform'])
    
    # === TRAINING PARAMETERS ===
    params['epochs'] = trial.suggest_int('epochs', 50, 100, step=25)  # [50, 75, 100]
    
    # Optimizer settings
    params['lr'] = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    params['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 0.04)
    
    # Derived parameter: min_lr = lr / 10
    params['min_lr'] = params['lr'] / 10
    
    # === LOSS FUNCTION CONFIGURATION ===
    params['state_loss_type'] = trial.suggest_categorical('state_loss_type', 
                                                         ['kl_div', 'cross_entropy', 'mse', 'js_div'])
    
    # Loss weights - keep wide ranges for now, but we can add conditional logic later
    params['state_loss_weight'] = trial.suggest_float('state_loss_weight', 0.5, 5.0)
    params['value_loss_weight'] = trial.suggest_float('value_loss_weight', 0.1, 5.0)
    
    # Entropy regularization (conditional)
    params['use_entropy_reg'] = trial.suggest_categorical('use_entropy_reg', [True, False])
    # Note: entropy_weight and related params are not in the reduced parameter set
    
    return params


def validate_trial_params(params: Dict[str, Any]) -> bool:
    """
    Validate that sampled parameters are consistent.
    
    Args:
        params: Dictionary of trial parameters
    
    Returns:
        True if parameters are valid, False otherwise
    """
    # Check conditional dependencies
    if params['use_target_encoder'] and params['ema_decay'] is None:
        return False
    
    if params['use_gumbel'] and (params['initial_temp'] is None or params['min_temp'] is None):
        return False
    
    # Check value constraints
    if params['use_gumbel'] and params['min_temp'] > params['initial_temp']:
        return False
    
    if params['min_lr'] > params['lr']:
        return False
    
    return True


def print_trial_summary(trial_number: int, params: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of trial parameters.
    
    Args:
        trial_number: Current trial number
        params: Dictionary of trial parameters
    """
    print(f"\n=== TRIAL {trial_number} PARAMETERS ===")
    
    # Model architecture
    print("Model:")
    print(f"  num_states: {params['num_states']}")
    print(f"  value_method: {params['value_method']}")
    print(f"  encoder_init_method: {params['encoder_init_method']}")
    
    if params['use_target_encoder']:
        print(f"  use_target_encoder: True (ema_decay: {params['ema_decay']:.3f})")
    else:
        print(f"  use_target_encoder: False")
    
    if params['use_gumbel']:
        print(f"  use_gumbel: True (initial_temp: {params['initial_temp']:.2f}, min_temp: {params['min_temp']:.2f})")
    else:
        print(f"  use_gumbel: False")
    
    # Training
    print("Training:")
    print(f"  epochs: {params['epochs']}")
    print(f"  lr: {params['lr']:.2e} (min_lr: {params['min_lr']:.2e})")
    print(f"  weight_decay: {params['weight_decay']:.3f}")
    
    # Loss
    print("Loss:")
    print(f"  state_loss_type: {params['state_loss_type']}")
    print(f"  state_loss_weight: {params['state_loss_weight']:.2f}")
    print(f"  value_loss_weight: {params['value_loss_weight']:.2f}")
    print(f"  use_entropy_reg: {params['use_entropy_reg']}")
    
    print("="*40)


def test_parameter_sampling(n_samples: int = 5) -> None:
    """
    Test the parameter sampling function with dummy trials.
    
    Args:
        n_samples: Number of test samples to generate
    """
    print("Testing parameter sampling function...")
    
    for i in range(n_samples):
        # Create a dummy study and trial for testing
        study = optuna.create_study()
        trial = study.ask()
        
        # Sample parameters
        params = sample_trial_params(trial)
        
        # Validate
        is_valid = validate_trial_params(params)
        
        # Print summary
        print_trial_summary(i+1, params)
        print(f"Validation: {'✓ VALID' if is_valid else '✗ INVALID'}")
        
        if not is_valid:
            print("ERROR: Invalid parameter combination detected!")
            break
    
    print(f"\nTesting completed. Generated {n_samples} parameter sets.")


if __name__ == "__main__":
    # Test the parameter sampling
    test_parameter_sampling(n_samples=3)