#!/usr/bin/env python3
"""
Integration Test: Parameter Sampler + Config Generator
Tests the complete workflow with actual first_sweep.yaml config.
"""

import sys
import optuna
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.parameter_sampler import SweepParameterSampler
from scripts.config_generator import create_sweep_folder, create_trial_config, copy_sweep_config_to_folder, verify_trial_config


def test_complete_sweep_workflow():
    """
    Test the complete workflow: sweep config → parameter sampling → config generation → verification.
    """
    print("=== TESTING COMPLETE SWEEP WORKFLOW ===")
    
    # 1. Check if first_sweep.yaml exists
    sweep_config_path = "configs/sweeps/first_sweep.yaml"
    if not Path(sweep_config_path).exists():
        print(f"❌ Error: {sweep_config_path} not found!")
        print("Please save the 'Sweep Configuration Example' artifact as this file.")
        return False
    
    try:
        # 2. Initialize parameter sampler
        print(f"Loading sweep config: {sweep_config_path}")
        sampler = SweepParameterSampler(sweep_config_path)
        
        sweep_info = sampler.get_sweep_info()
        print(f"Sweep: {sweep_info.get('name')} - {sweep_info.get('description')}")
        
        # 3. Create sweep folder
        sweep_folder = create_sweep_folder("test_integration")
        print(f"Created sweep folder: {sweep_folder}")
        
        # 4. Copy sweep config to folder
        copy_sweep_config_to_folder(sweep_config_path, sweep_folder)
        print("✓ Copied sweep config to folder")
        
        # 5. Generate and test multiple trials
        print(f"\\nGenerating {3} test trials...")
        
        for trial_num in range(1, 4):
            print(f"\\n--- Trial {trial_num} ---")
            
            # Sample parameters
            study = optuna.create_study()
            trial = study.ask()
            trial_params = sampler.sample_trial_params(trial)
            
            print("Sampled parameters:")
            for key, value in trial_params.items():
                if value is not None:
                    print(f"  {key}: {value}")
            
            # Generate config file
            config_path = create_trial_config(
                trial_params=trial_params,
                trial_number=trial_num,
                base_config_path=sampler.get_base_config_path(),
                sweep_folder=sweep_folder,
                fixed_params=sampler.get_fixed_params()
            )
            
            print(f"Generated config: {config_path}")
            
            # Verify config
            all_params = {**sampler.get_fixed_params(), **trial_params}
            is_valid = verify_trial_config(config_path, all_params)
            print(f"Verification: {'✓ VALID' if is_valid else '❌ INVALID'}")
            
            if not is_valid:
                print("❌ Config verification failed!")
                return False
            
            # Test specific conditional logic
            success = test_conditional_logic(trial_params)
            if not success:
                print("❌ Conditional logic test failed!")
                return False
        
        # 6. Summary
        print(f"\\n=== SUMMARY ===")
        print(f"✓ Successfully generated {3} trial configs")
        print(f"✓ All configs verified correctly")
        print(f"✓ Conditional parameter logic working")
        print(f"✓ Files created in: {sweep_folder}")
        
        # List generated files
        generated_files = list(sweep_folder.glob("*"))
        print(f"\\nGenerated files:")
        for file_path in generated_files:
            print(f"  {file_path.name}")
        
        print(f"\\n🎉 Integration test PASSED!")
        print(f"\\nTo clean up: rm -rf {sweep_folder}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conditional_logic(trial_params):
    """
    Test that conditional parameter logic is working correctly.
    
    Args:
        trial_params: Dictionary of trial parameters
        
    Returns:
        True if all conditional logic tests pass
    """
    print("Testing conditional logic...")
    
    # Test 1: Gumbel temperature parameters
    if trial_params.get('model.use_gumbel') is True:
        if trial_params.get('model.initial_temp') is None or trial_params.get('model.min_temp') is None:
            print("❌ Gumbel temperature params should be set when use_gumbel=True")
            return False
        print("✓ Gumbel temperature logic correct")
    else:
        if trial_params.get('model.initial_temp') is not None or trial_params.get('model.min_temp') is not None:
            print("❌ Gumbel temperature params should be None when use_gumbel=False")
            return False
        print("✓ Gumbel temperature logic correct")
    
    # Test 2: Target encoder EMA decay
    if trial_params.get('model.use_target_encoder') is True:
        if trial_params.get('model.ema_decay') is None:
            print("❌ EMA decay should be set when use_target_encoder=True")
            return False
        print("✓ Target encoder logic correct")
    else:
        if trial_params.get('model.ema_decay') is not None:
            print("❌ EMA decay should be None when use_target_encoder=False")
            return False
        print("✓ Target encoder logic correct")
    
    # Test 3: Entropy regularization
    if trial_params.get('loss.use_entropy_reg') is True:
        if trial_params.get('loss.entropy_weight') is None:
            print("❌ Entropy weight should be set when use_entropy_reg=True")
            return False
        print("✓ Entropy regularization logic correct")
    else:
        if trial_params.get('loss.entropy_weight') is not None:
            print("❌ Entropy weight should be None when use_entropy_reg=False")
            return False
        print("✓ Entropy regularization logic correct")
    
    # Test 4: State loss weight conditional ranges
    state_loss_type = trial_params.get('loss.state_loss_type')
    state_loss_weight = trial_params.get('loss.state_loss_weight')
    
    if state_loss_type and state_loss_weight is not None:
        if state_loss_type in ['cross_entropy', 'kl_div']:
            if not (0.5 <= state_loss_weight <= 2.0):
                print(f"❌ state_loss_weight {state_loss_weight} outside expected range [0.5, 2.0] for {state_loss_type}")
                return False
        elif state_loss_type == 'js_div':
            if not (1.0 <= state_loss_weight <= 3.0):
                print(f"❌ state_loss_weight {state_loss_weight} outside expected range [1.0, 3.0] for {state_loss_type}")
                return False
        elif state_loss_type == 'mse':
            if not (1.0 <= state_loss_weight <= 5.0):
                print(f"❌ state_loss_weight {state_loss_weight} outside expected range [1.0, 5.0] for {state_loss_type}")
                return False
        
        print(f"✓ State loss weight range correct for {state_loss_type}: {state_loss_weight}")
    
    # Test 5: Derived parameters
    lr = trial_params.get('training.lr')
    min_lr = trial_params.get('training.min_lr')
    
    if lr is not None and min_lr is not None:
        expected_min_lr = lr / 10
        if abs(min_lr - expected_min_lr) > 1e-10:
            print(f"❌ Derived min_lr incorrect: {min_lr} != {expected_min_lr}")
            return False
        print(f"✓ Derived parameter correct: min_lr = lr/10 = {min_lr}")
    
    return True


if __name__ == "__main__":
    success = test_complete_sweep_workflow()
    if success:
        print("\\n🎉 All tests passed! Ready for Step 3.")
    else:
        print("\\n❌ Tests failed. Please fix issues before proceeding.")