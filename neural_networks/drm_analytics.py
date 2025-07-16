import json
import pandas as pd
from pathlib import Path
import re
import os
import sys

import torch
import torch.nn as nn


# Define project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neural_networks.drm import DiscreteRepresentationsModel, LinearProbe

from neural_networks.train_drm import run_layer_probing 


def analyze_experiment_results(base_dir, group_by=None, metrics=None, aggregate_seeds=False):
    """
    Generic function to aggregate experiment results by config parameters.
    
    Args:
        base_dir: Path to directory containing run folders
        group_by: Config parameter(s) to group by (string or list)
        metrics: Metric(s) to analyze from results (string or list)
        aggregate_seeds: If True, groups runs by config (ignoring seed differences)
    
    Returns:
        pandas DataFrame with aggregated results
    """
    base_path = Path(base_dir)
    
    # Convert single strings to lists for uniform handling
    if group_by is not None and isinstance(group_by, str):
        group_by = [group_by]
    if metrics is not None and isinstance(metrics, str):
        metrics = [metrics]
    
    results = []
    
    print(f"Scanning directory: {base_path}")
    
    # Scan all subdirectories for run folders
    for run_dir in base_path.iterdir():
        if not run_dir.is_dir():
            continue
            
        # Look for config and results files
        config_files = list(run_dir.glob("config_*.json"))
        result_files = list(run_dir.glob("drm_test_results_*.json"))
        
        if not config_files or not result_files:
            continue
            
        try:
            # Load config
            with open(config_files[0], 'r') as f:
                config = json.load(f)
            
            # Load results
            with open(result_files[0], 'r') as f:
                test_results = json.load(f)
            
            # Combine config and results
            row = config.copy()
            row.update(test_results)
            results.append(row)
            
        except Exception as e:
            print(f"Error processing {run_dir}: {e}")
            continue
    
    if not results:
        print("No valid results found")
        return pd.DataFrame()
    
    print(f"Found {len(results)} valid results")
    df = pd.DataFrame(results)
    
    # Handle seed aggregation
    if aggregate_seeds:
        def extract_base_config(run_id):
            # Pattern: {config}_seed{number}_{timestamp}
            match = re.match(r'(.+)_seed\d+_\d{8}_\d{6}', run_id)
            return match.group(1) if match else run_id
        
        df['base_config'] = df['run_id'].apply(extract_base_config)
        
        # Add base_config to grouping
        if group_by is not None:
            group_by = group_by + ['base_config']
        else:
            group_by = ['base_config']
    
    # Group and aggregate if requested
    if group_by is not None and metrics is not None:
        grouped = df.groupby(group_by)[metrics]
        return grouped.agg(['mean', 'count'])
    elif group_by is not None:
        # Group by specified columns, aggregate all numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            grouped = df.groupby(group_by)[numeric_cols]
            return grouped.agg(['mean', 'count'])
        else:
            return df.groupby(group_by).size().to_frame('count')
    elif metrics is not None:
        # Just return descriptive stats for specified metrics
        return df[metrics].describe()
    else:
        # Return full dataframe
        return df

def print_state_values(model_path, device='cpu'):
    """
    Load a trained model and print the value for each one-hot encoded state.
    
    Args:
        model_path: Path to the saved model checkpoint (.pt file)
        device: Device to load model on ('cpu' or 'cuda')
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Recreate the model
    model = DiscreteRepresentationsModel(
        obs_dim=config['obs_dim'],
        control_dim=config['control_dim'], 
        value_dim=config['value_dim'],
        num_states=config['num_states'],
        hidden_dim=config['hidden_dim'],
        # Add other config parameters that might be needed
        value_activation=config.get('value_activation', 'sigmoid')
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Compute values for all one-hot encoded states
    with torch.no_grad():
        values = model.compute_values_for_all_states()
    
    # Print results
    print(f"Values for each one-hot encoded state (model: {model_path}):")
    print("-" * 50)
    
    for state_idx in range(config['num_states']):
        if config['value_dim'] == 1:
            # Single value output
            value = values[state_idx, 0].item()
            print(f"State {state_idx}: {value:.4f}")
        else:
            # Multi-dimensional value output
            value_vec = values[state_idx].cpu().numpy()
            print(f"State {state_idx}: {value_vec}")
    
    return values.cpu().numpy()


def load_trained_model(model_path, device='cuda'):
    """
    Load a trained DRM model from checkpoint
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load model on
        
    Returns:
        model: Loaded and initialized model
        config: Model configuration dict
    """
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    print(f"Loading model from: {model_path}")
    print(f"Model config: {config}")
    
    # Check what keys exist in the state dict to infer missing config
    state_dict_keys = set(checkpoint['model_state_dict'].keys())
    print(f"State dict keys (first 10): {list(state_dict_keys)[:10]}")

    
    # Recreate the model with corrected config
    model = DiscreteRepresentationsModel(
        obs_dim=config['obs_dim'],
        control_dim=config['control_dim'],
        value_dim=config['value_dim'],
        num_states=config['num_states'],
        hidden_dim=config['hidden_dim'],
        # Add other config parameters that might be needed
        predictor_type=config.get('predictor_type', 'standard'),
        use_gumbel=config.get('use_gumbel', False),
        initial_temp=config.get('initial_temp', 5.0),
        min_temp=config.get('min_temp', 0.5),
        use_target_encoder=config.get('use_target_encoder', False),  # Now corrected above
        ema_decay=config.get('ema_decay', 0.996),
        value_activation=config.get('value_activation', 'sigmoid')
    )
    
    # Load the trained weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model weights loaded successfully")
    except RuntimeError as e:
        print(f"❌ Error loading model weights: {e}")
        print("Attempting to load with strict=False...")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        print("⚠️  Model loaded with mismatched keys - results may be unreliable")
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} total parameters")
    
    return model, config

def create_probing_data_loader(db_path, system_type, value_method, probing_size=5000, batch_size=64, seed=42):
    """
    Create just the probing data loader
    """
    from neural_networks.drm_dataset import create_data_loaders
    
    # Use a large val/test size, but we only care about the probing loader
    _, _, _, probing_loader = create_data_loaders(
        system_type=system_type,
        db_path=db_path,
        value_method=value_method,
        batch_size=batch_size,
        val_size=100,  # Dummy values
        test_size=100,  # Dummy values
        probing_size=probing_size,
        seed=seed
    )
    
    return probing_loader

def batch_layer_probing(base_output_dir, probing_size=5000, device='cuda', overwrite_existing=True):
    """
    Run layer probing on all training runs in a directory structure.
    
    Args:
        base_output_dir: Base directory containing run folders (e.g., "neural_networks/output/multi_saddle_probing")
        probing_size: Number of samples for probing
        device: Device to run probing on
        overwrite_existing: Whether to overwrite existing probing results
        
    Returns:
        pandas.DataFrame: Results dataframe similar to analyze_experiment_results
    """
    
    base_path = Path(base_output_dir)
    results = []
    
    print(f"Scanning directory: {base_path}")
    print(f"Looking for training runs to probe...")
    
    # Find all run directories
    run_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('run_')]
    print(f"Found {len(run_dirs)} potential run directories")
    
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for run_dir in run_dirs:
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {run_dir.name}")
            print(f"{'='*60}")
            
            # Look for config file
            config_files = list(run_dir.glob("config_*.json"))
            if not config_files:
                print(f"  ❌ No config file found in {run_dir}")
                failed_count += 1
                continue
                
            # Load config
            config_path = config_files[0]
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract run_id from config
            run_id = config.get('run_id', run_dir.name.replace('run_', ''))
            
            # Check if probing already exists and should skip
            if not overwrite_existing:
                if 'probing_discrete_accuracy' in config and 'probing_embedding_accuracy' in config:
                    print(f"  ⏭️  Probing results already exist, skipping...")
                    skipped_count += 1
                    continue
            
            # Look for model files (prioritize sorted model)
            model_files = list(run_dir.glob(f"drm_final_sorted_{run_id}.pt"))
            if not model_files:
                model_files = list(run_dir.glob(f"drm_final_{run_id}.pt"))
            if not model_files:
                print(f"  ❌ No model file found in {run_dir}")
                failed_count += 1
                continue
                
            model_path = model_files[0]
            print(f"  📁 Using model: {model_path.name}")
            
            # Extract database path from config
            db_path = config.get('db_path')
            if not db_path:
                print(f"  ❌ No db_path in config")
                failed_count += 1
                continue
                
            # Ensure absolute path for database
            db_path = Path(db_path)
            if not db_path.is_absolute():
                # Try relative to project root
                from neural_networks.train_drm import PROJECT_ROOT
                db_path = PROJECT_ROOT / db_path
            
            if not db_path.exists():
                print(f"  ❌ Database not found: {db_path}")
                failed_count += 1
                continue
                
            print(f"  🗄️  Using database: {db_path}")
            
            # Extract other needed parameters
            system_type = config.get('system_type', 'saddle_system')
            value_method = config.get('value_method', 'angle')
            
            print(f"  🔧 System: {system_type}, Value method: {value_method}")
            
            # Load model
            print(f"  🤖 Loading model...")
            model, model_config = load_trained_model(str(model_path), device)
            
            # Create probing data loader
            print(f"  📊 Creating probing data loader (size: {probing_size})...")
            probing_loader = create_probing_data_loader(
                db_path=str(db_path),
                system_type=system_type,
                value_method=value_method,
                probing_size=probing_size,
                batch_size=64,
                seed=42
            )
            
            # Run layer probing
            print(f"  🔍 Running layer probing...")
            probing_results = run_layer_probing(
                model=model,
                probing_loader=probing_loader,
                device=device,
                system_type=system_type,
                db_path=str(db_path)
            )
            
            # Update config with probing results
            config['probing_discrete_accuracy'] = probing_results['discrete_accuracy']
            config['probing_embedding_accuracy'] = probing_results['embedding_accuracy']
            config['probing_size_used'] = probing_size
            config['probing_device'] = device
            
            # Save updated config
            print(f"  💾 Saving updated config...")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            # Also update test results file if it exists
            test_results_files = list(run_dir.glob(f"drm_test_results_{run_id}.json"))
            if test_results_files:
                test_results_path = test_results_files[0]
                with open(test_results_path, 'r') as f:
                    test_results = json.load(f)
                
                test_results['probing_discrete_accuracy'] = probing_results['discrete_accuracy']
                test_results['probing_embedding_accuracy'] = probing_results['embedding_accuracy']
                
                with open(test_results_path, 'w') as f:
                    json.dump(test_results, f, indent=4)
                print(f"  💾 Updated test results file")
            
            print(f"  ✅ Completed! Discrete: {probing_results['discrete_accuracy']:.4f}, Embedding: {probing_results['embedding_accuracy']:.4f}")
            processed_count += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {run_dir}: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"BATCH PROBING SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Total: {len(run_dirs)}")
    
    # Create results dataframe using existing function
    print(f"\n📊 Creating results dataframe...")
    results_df = analyze_experiment_results(base_output_dir)
    
    if not results_df.empty:
        # Focus on probing columns
        probing_cols = ['probing_discrete_accuracy', 'probing_embedding_accuracy']
        available_probing_cols = [col for col in probing_cols if col in results_df.columns]
        
        if available_probing_cols:
            print(f"\nProbing Results Summary:")
            print(f"Rows with probing data: {results_df[available_probing_cols].notna().all(axis=1).sum()}")
            print(f"Mean discrete accuracy: {results_df['probing_discrete_accuracy'].mean():.4f}")
            print(f"Mean embedding accuracy: {results_df['probing_embedding_accuracy'].mean():.4f}")
            
        # Save results dataframe
        results_path = base_path / "probing_results_summary.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Saved results to: {results_path}")
    
    return results_df

# Usage example:
# results_df = batch_layer_probing("neural_networks/output/multi_saddle_probing", probing_size=5000)


# Usage examples:
if __name__ == "__main__":
    # Your specific question - average test_state_loss by value_method
    results = analyze_experiment_results(
        "neural_networks/output/saddle_datasets_double", 
        group_by="value_method", 
        metrics="test_state_loss"
    )
    print("Average test_state_loss by value_method:")
    print(results)
    
    # Aggregate over seeds and group by value_method
    results_seeds = analyze_experiment_results(
        "neural_networks/output/saddle_datasets_double", 
        group_by="value_method", 
        metrics="test_state_loss",
        aggregate_seeds=True
    )
    print("\nAverage test_state_loss by value_method (aggregated over seeds):")
    print(results_seeds)
    
    # Get full dataframe to explore
    full_df = analyze_experiment_results(
        "neural_networks/output/saddle_datasets_double"
    )
    print(f"\nFull dataset shape: {full_df.shape}")
    print(f"Available columns: {full_df.columns.tolist()}")


