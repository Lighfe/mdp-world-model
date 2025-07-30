import json
import pandas as pd
from pathlib import Path
import re
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn


# Define project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neural_networks.drm import DiscreteRepresentationsModel, LinearProbe

from neural_networks.train_drm import run_layer_probing
from neural_networks.system_registry import SystemType, get_visualization_bounds


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
    """
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    print(f"Loading model from: {model_path}")
    
    # Infer use_target_encoder from state dict keys
    state_dict_keys = set(checkpoint['model_state_dict'].keys())
    has_target_encoder = any(key.startswith('target_encoder.') for key in state_dict_keys)
    
    print(f"Detected target encoder in model: {has_target_encoder}")
    
    # Recreate the model with inferred config
    model = DiscreteRepresentationsModel(
        obs_dim=config['obs_dim'],
        control_dim=config['control_dim'],
        value_dim=config['value_dim'],
        num_states=config['num_states'],
        hidden_dim=config['hidden_dim'],
        predictor_type=config.get('predictor_type', 'standard'),
        use_gumbel=config.get('use_gumbel', False),
        initial_temp=config.get('initial_temp', 5.0),
        min_temp=config.get('min_temp', 0.5),
        use_target_encoder=has_target_encoder,  # Infer from state dict
        ema_decay=config.get('ema_decay', 0.996),
        value_activation=config.get('value_activation', 'sigmoid')
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model weights loaded successfully")
    
    model.to(device)
    model.eval()
    
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

def calculate_epoch_metrics(grid_points, state_probs, epoch, num_states, 
                          prev_dominant_states=None, grid_size=100):
    """
    Calculate all state assignment metrics for a single epoch.
    
    Args:
        grid_points: np.array of shape (grid_size^2, 2) - the x1, x2 coordinates
        state_probs: np.array of shape (grid_size^2, num_states) - state probabilities
        epoch: current epoch number
        num_states: number of discrete states
        prev_dominant_states: np.array from previous epoch for stability calculation
        grid_size: grid size for reshaping
    
    Returns:
        dict: All metrics for this epoch
    """
    
    # Basic info
    metrics = {'epoch': epoch}
    
    # 1. State usage (how often each state is dominant)
    dominant_states = np.argmax(state_probs, axis=1)
    
    for state_idx in range(num_states):
        usage = np.mean(dominant_states == state_idx)
        metrics[f'state_{state_idx}_usage'] = usage
    
    # 2. Mean probabilities per state
    for state_idx in range(num_states):
        mean_prob = np.mean(state_probs[:, state_idx])
        metrics[f'state_{state_idx}_mean'] = mean_prob
    
    # 3. Sharpness: per-grid-point entropy (discreteness measure)
    # Calculate entropy per grid point: -sum(p * log(p)) for each row
    per_point_entropy = -np.sum(state_probs * np.log(state_probs + 1e-8), axis=1)
    metrics['sharpness_mean'] = np.mean(per_point_entropy)
    metrics['sharpness_std'] = np.std(per_point_entropy)
    
    # 4. Stability metrics (if we have previous epoch data)
    if prev_dominant_states is not None:
        # Dominant state stability: % of points keeping same dominant state
        same_dominant = np.mean(dominant_states == prev_dominant_states) * 100
        metrics['dominant_stability'] = same_dominant
        
        # Note: For probability stability, we'd need previous state_probs
        # For now, we'll skip this or store prev_state_probs too if needed
    
    return metrics, dominant_states


def save_state_visualization_frame(grid_points, state_probs, epoch, num_states, 
                                 output_dir, run_id, bounds, grid_size=100):
    """
    Save a single visualization frame for this epoch.
    
    Args:
        grid_points: np.array of shape (grid_size^2, 2)
        state_probs: np.array of shape (grid_size^2, num_states)
        epoch: current epoch number
        num_states: number of states
        output_dir: directory to save frame (unique per run)
        run_id: run identifier  
        bounds: [(x1_min, x1_max), (x2_min, x2_max)]
        grid_size: grid size for reshaping
    
    Returns:
        str: path to saved frame
    """
    import tempfile
    
    # Calculate grid layout
    cols_per_row = 2
    rows = (num_states + cols_per_row - 1) // cols_per_row
    
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(12, 6*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for state_idx in range(num_states):
        row_idx = state_idx // cols_per_row
        col_idx = state_idx % cols_per_row
        ax = axes[row_idx, col_idx]
        
        # Reshape probabilities to grid
        state_grid = state_probs[:, state_idx].reshape(grid_size, grid_size)
        
        # Create heatmap
        im = ax.imshow(state_grid, extent=[bounds[0][0], bounds[0][1], 
                                          bounds[1][0], bounds[1][1]], 
                      origin='lower', cmap='viridis', vmin=0, vmax=1, 
                      aspect='auto', interpolation='bilinear')
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(f'State {state_idx + 1}')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('State Assignment Strength')
    
    # Hide unused subplots
    for idx in range(num_states, rows * cols_per_row):
        row_idx = idx // cols_per_row
        col_idx = idx % cols_per_row
        axes[row_idx, col_idx].set_visible(False)
    
    # Overall title
    fig.suptitle(f'State Space Visualization - Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    
    # SAFE FILE SAVING
    # Temp files are created in each run's unique output_dir (no conflicts!)
    final_filename = f"state_frame_epoch_{epoch}.png"
    output_dir = Path(output_dir)
    final_path = output_dir / final_filename
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use atomic write with temp file in the SAME directory
    # Since output_dir is unique per run (from SLURM), no collision risk
    with tempfile.NamedTemporaryFile(
        dir=output_dir,  # Safe: each run has unique output_dir
        suffix='.tmp', 
        delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)
    
    try:
        # Save to temporary file first
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Atomically move to final location
        temp_path.rename(final_path)
        
    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise e
    
    return str(final_path)


def create_gif_from_frames(frame_paths, output_path, gif_duration=250):
    """
    Create GIF from saved frame paths.
    
    Args:
        frame_paths: list of paths to frame images
        output_path: path for output GIF (without extension)
        gif_duration: duration per frame in milliseconds
    
    Returns:
        str: path to created GIF
    """
    try:
        import imageio
        
        gif_path = f"{output_path}_animation.gif"
        
        with imageio.get_writer(gif_path, mode='I', duration=gif_duration/1000.0) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)
        
        print(f"Created GIF with {len(frame_paths)} frames: {gif_path}")
        return gif_path
        
    except ImportError:
        print("Warning: imageio not available for GIF creation")
        return None


def create_state_evolution_analysis(all_metrics, output_path, num_states):
    """
    Create the analysis plot from collected metrics.
    
    Args:
        all_metrics: list of metric dicts from all epochs
        output_path: path to save analysis plot
        num_states: number of states
    
    Returns:
        dict: analysis results
    """
    if not all_metrics:
        return {}
    
    # Paul Tol's muted color scheme
    tol_muted = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933']
    
    # Convert to arrays for plotting
    epochs = [m['epoch'] for m in all_metrics]
    
    # Create analysis plots (2x2 layout)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. State usage over time
    ax = axes[0, 0]
    for state_idx in range(num_states):
        usage_values = [m.get(f'state_{state_idx}_usage', 0) * 100 for m in all_metrics]
        color = tol_muted[state_idx % len(tol_muted)]
        ax.plot(epochs, usage_values, label=f'State {state_idx + 1}', 
               marker='o', linewidth=2, color=color)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dominant State (%)')
    ax.set_title('Dominant State Assignments Over Training')
    ax.legend()
    ax.grid(True, alpha=0.4)
    
    # 2. Assignment sharpness over time
    ax = axes[0, 1]
    sharpness_mean = [m.get('sharpness_mean', 0) for m in all_metrics]
    sharpness_std = [m.get('sharpness_std', 0) for m in all_metrics]
    
    ax.plot(epochs, sharpness_mean, color='blue', linewidth=2, label='Mean')
    ax.fill_between(epochs, 
                   np.array(sharpness_mean) - np.array(sharpness_std),
                   np.array(sharpness_mean) + np.array(sharpness_std),
                   alpha=0.3, color='blue', label='± 1 std')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Per-Point Entropy (Sharpness)')
    ax.set_title('Assignment Sharpness Over Training\n(Lower = More Discrete)')
    ax.legend()
    ax.grid(True, alpha=0.4)
    
    # 3. Mean probabilities per state
    ax = axes[1, 0]
    for state_idx in range(num_states):
        mean_values = [m.get(f'state_{state_idx}_mean', 0) for m in all_metrics]
        color = tol_muted[state_idx % len(tol_muted)]
        ax.plot(epochs, mean_values, label=f'State {state_idx + 1}', 
               marker='o', linewidth=2, color=color)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Probability')
    ax.set_title('Average State Assignment Strength')
    ax.grid(True, alpha=0.4)
    
    # 4. Stability metrics
    ax = axes[1, 1]
    stability_epochs = []
    dominant_stability = []
    
    for m in all_metrics:
        if 'dominant_stability' in m:
            stability_epochs.append(m['epoch'])
            dominant_stability.append(m['dominant_stability'])
    
    if stability_epochs:
        ax.plot(stability_epochs, dominant_stability, 
               color='#AA4499', marker='o', linewidth=2, label='Dominant State Stability')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Stability (%)')
    ax.set_title('Assignment Stability Between Epochs')
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_ylim(50, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'epochs_analyzed': len(all_metrics),
        'num_states': num_states
    }


def extract_and_calculate_metrics(model, device, num_states, system_type, 
                                bounds=None, grid_size=100, epoch=None,
                                prev_dominant_states=None):
    """
    Extract state assignments and immediately calculate metrics.
    
    This replaces the extract_state_assignment_data + metric calculation pipeline.
    
    Args:
        model: DRM model
        device: PyTorch device
        num_states: number of discrete states
        system_type: system type string for bounds lookup
        bounds: visualization bounds (if None, will get from system registry)
        grid_size: grid size for visualization
        epoch: current epoch
        prev_dominant_states: previous epoch's dominant states for stability
    
    Returns:
        tuple: (metrics_dict, dominant_states, grid_points, state_probs)
    """
    
    # Get proper bounds if not provided
    if bounds is None:
        bounds = get_visualization_bounds(SystemType[system_type.upper()])
    
    # Generate grid points
    x_range = np.linspace(bounds[0][0], bounds[0][1], grid_size)
    y_range = np.linspace(bounds[1][0], bounds[1][1], grid_size)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(grid_points).to(device)
        state_probs = model.get_state_probs(obs_tensor, training=False, soft=True)
        state_probs = state_probs.cpu().numpy()
    
    # Calculate metrics immediately
    metrics, dominant_states = calculate_epoch_metrics(
        grid_points, state_probs, epoch, num_states, prev_dominant_states, grid_size
    )
    
    return metrics, dominant_states, grid_points, state_probs
