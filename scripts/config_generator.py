#!/usr/bin/env python3
"""
Step 2: Config File Generation (Data-Driven)
Creates trial config files in organized sweep folders.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import tempfile
import shutil
import traceback

# Add the project root to Python path so we can import utils
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from neural_networks.utils import load_config, set_nested_dict_value
except ImportError as e:
    print(f"Error importing utils: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def create_sweep_folder(sweep_id: Optional[str] = None) -> Path:
    """
    Create a sweep folder in configs/sweeps/{sweep_id}.
    
    Args:
        sweep_id: Custom sweep identifier, or None for datetime-based ID
        
    Returns:
        Path to the created sweep folder
    """
    if sweep_id is None:
        # Generate datetime-based ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_id = f"sweep_{timestamp}"
    
    sweep_folder = Path("configs/sweeps") / sweep_id
    sweep_folder.mkdir(parents=True, exist_ok=True)
    
    return sweep_folder


def create_trial_config(trial_params: Dict[str, Any], 
                       trial_number: int,
                       base_config_path: str = "configs/base.yaml",
                       sweep_folder: Optional[Path] = None,
                       fixed_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Create a trial config file by merging trial parameters with base config.
    
    Args:
        trial_params: Dictionary of trial parameters (using dot notation keys)
        trial_number: Current trial number for naming
        base_config_path: Path to base configuration file
        sweep_folder: Path to sweep folder (if None, creates temp folder)
        fixed_params: Additional fixed parameters to apply
    
    Returns:
        Path to the generated trial config file
    """
    # Load base configuration
    base_config = load_config(base_config_path)
    
    # Create a copy to modify
    trial_config = base_config.copy()
    
    # Apply fixed parameters first
    if fixed_params:
        for param_path, value in fixed_params.items():
            if value is not None:
                set_nested_dict_value(trial_config, param_path, value)
    
    # Apply trial parameters (skip None values)
    for param_path, value in trial_params.items():
        if value is not None:
            set_nested_dict_value(trial_config, param_path, value)
    
    # Determine output path
    if sweep_folder:
        config_file_path = sweep_folder / f"trial_{trial_number:03d}_config.yaml"
    else:
        # Fallback to temp file
        fd, config_file_path = tempfile.mkstemp(suffix=f"_trial_{trial_number:03d}.yaml", prefix="sweep_")
        os.close(fd)
        config_file_path = Path(config_file_path)
    
    # Save the trial config
    with open(config_file_path, 'w') as f:
        yaml.dump(trial_config, f, default_flow_style=False, indent=2)
    
    return str(config_file_path)


def copy_sweep_config_to_folder(sweep_config_path: str, sweep_folder: Path) -> None:
    """
    Copy the sweep configuration file to the sweep folder for reference.
    
    Args:
        sweep_config_path: Path to original sweep config
        sweep_folder: Path to sweep folder
    """
    dest_path = sweep_folder / "sweep_config.yaml"
    shutil.copy2(sweep_config_path, dest_path)


def verify_trial_config(config_path: str, expected_params: Dict[str, Any]) -> bool:
    """
    Verify that a generated config file contains the expected trial parameters.
    
    Args:
        config_path: Path to the trial config file
        expected_params: Expected parameter values (using dot notation)
    
    Returns:
        True if config is correct, False otherwise
    """
    try:
        config = load_config(config_path)
        
        # Check each expected parameter
        for param_path, expected_value in expected_params.items():
            if expected_value is None:
                continue  # Skip None values
                
            # Navigate through nested config to find actual value
            keys = param_path.split('.')
            actual_value = config
            
            try:
                for key in keys:
                    actual_value = actual_value[key]
            except (KeyError, TypeError):
                print(f"Config verification failed: parameter '{param_path}' not found")
                return False
            
            if actual_value != expected_value:
                print(f"Config verification failed for '{param_path}': expected {expected_value}, got {actual_value}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Error verifying config: {e}")
        return False


def cleanup_trial_config(config_path: str) -> None:
    """
    Remove a temporary trial config file.
    
    Args:
        config_path: Path to the config file to remove
    """
    try:
        os.remove(config_path)
    except OSError as e:
        print(f"Warning: Could not remove trial config {config_path}: {e}")


def test_config_generation() -> None:
    """
    Test config file generation with sample parameters.
    """
    print("Testing data-driven config file generation...")
    
    # Test sweep folder creation
    test_sweep_folder = create_sweep_folder("test_sweep_123")
    print(f"Created test sweep folder: {test_sweep_folder}")
    
    # Test parameters (using dot notation)
    test_params = {
        'model.num_states': 6,
        'model.use_gumbel': True,
        'model.initial_temp': 2.5,
        'training.lr': 5e-4,
        'training.min_lr': 5e-5,
        'loss.state_loss_type': 'kl_div',
        'loss.state_loss_weight': 1.5,
    }
    
    test_fixed_params = {
        'training.epochs': 75,
        'data.val_size': 1000,
    }
    
    print("Test parameters:")
    for key, value in test_params.items():
        print(f"  {key}: {value}")
    
    print("Fixed parameters:")
    for key, value in test_fixed_params.items():
        print(f"  {key}: {value}")
    
    # Generate config file
    try:
        config_path = create_trial_config(
            trial_params=test_params,
            trial_number=1,
            sweep_folder=test_sweep_folder,
            fixed_params=test_fixed_params
        )
        print(f"\nGenerated config: {config_path}")
        
        # Verify the config
        all_params = {**test_params, **test_fixed_params}
        is_valid = verify_trial_config(config_path, all_params)
        print(f"Config verification: {'✓ VALID' if is_valid else '✗ INVALID'}")
        
        # Show a snippet of the generated config
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        print(f"\nConfig file content (first 25 lines):")
        lines = config_content.split('\n')[:25]
        for line in lines:
            print(f"  {line}")
        
        print(f"\nTest files created in: {test_sweep_folder}")
        print("To clean up: rm -rf configs/sweeps/test_sweep_123")
        
    except Exception as e:
        print(f"Error during config generation test: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_config_generation()