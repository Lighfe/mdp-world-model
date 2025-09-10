#!/usr/bin/env python3
"""
Quick test of the complete parameter sweep pipeline.
Runs a few trials with reduced settings to verify everything works.
"""

import sys
import tempfile
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.parameter_sweep import ParameterSweep


def create_test_sweep_config() -> str:
    """Create a minimal sweep config for testing."""
    test_config = {
        'sweep': {
            'name': 'quick_test',
            'description': 'Quick test with minimal settings',
            'n_trials': 3
        },
        'base_config': 'configs/base.yaml',
        'fixed_params': {
            'training.epochs': 50,  # Reduced for testing
            'data.val_size': 500,   # Reduced for testing
            'data.test_size': 500   # Reduced for testing
        },
        'parameters': {
            # Just test a few key parameters
            'model.use_gumbel': {
                'type': 'categorical',
                'choices': [True, False]
            },
            'model.initial_temp': {
                'type': 'float',
                'min': 1.0,
                'max': 3.0,
                'condition': 'model.use_gumbel == true'
            },
            'training.lr': {
                'type': 'float',
                'min': 1e-4,
                'max': 1e-3,
                'log': True
            },
            'training.min_lr': {
                'type': 'derived',
                'formula': 'training.lr / 10'
            },
            'loss.state_loss_type': {
                'type': 'categorical',
                'choices': ['kl_div', 'cross_entropy']
            }
        },
        'multi_run': {
            'seeds': [11, 12, 13],  # Reduced for testing
            'db_paths': ['datasets/results/multi_saddle_7.db'],  # Single dataset
            'max_parallel': 3
        }
    }
    
    # Save to temporary file
    fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix='test_sweep_')
    with open(temp_path, 'w') as f:
        yaml.dump(test_config, f, default_flow_style=False, indent=2)
    
    return temp_path


def main():
    """Run quick test."""
    print("=" * 60)
    print("QUICK PARAMETER SWEEP TEST")
    print("=" * 60)
    print("This test runs 3 trials with reduced settings to verify the pipeline works.")
    print("Each trial should take ~3-5 minutes with the reduced settings.")
    print("=" * 60)
    
    # Create test config
    test_config_path = create_test_sweep_config()
    print(f"Created test config: {test_config_path}")
    
    try:
        # Check base config exists
        if not Path('configs/base.yaml').exists():
            print("ERROR: configs/base.yaml not found!")
            print("Make sure you're running from the project root directory.")
            return False
        
        # Check database exists
        test_db_path = Path('datasets/results/multi_saddle_7.db')
        if not test_db_path.exists():
            print(f"ERROR: Test database not found: {test_db_path}")
            print("Make sure the database file exists or update the path in the test config.")
            return False
        
        # Run test sweep
        sweep = ParameterSweep(
            sweep_config_path=test_config_path,
            sweep_id="quick_test"
        )
        
        print("\\nStarting test sweep...")
        sweep.run_sweep(n_trials=3)
        
        print("\\n" + "=" * 60)
        print("QUICK TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("If this test passed, your parameter sweep pipeline is working correctly.")
        print("You can now run the full sweep with your actual sweep configuration.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\\nQUICK TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            import os
            os.remove(test_config_path)
        except:
            pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)