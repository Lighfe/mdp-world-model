import os
import sys
from pathlib import Path
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import math
import torch
import torch.nn as nn
import torch.optim as optim
import psutil
import copy

import pandas as pd
import pickle
import torch.nn.functional as F

from torch.utils.data import WeightedRandomSampler
from collections import Counter

# Define project root at the module level
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"Added {PROJECT_ROOT} to Python path")

# NOTE: absolute imports from project root
# Import application-specific modules
from neural_networks.system_registry import SystemType, get_system_config, get_transformation, get_visualization_bounds
from neural_networks.drm_dataset import create_data_loaders, TechSubstitutionDataset, SaddleSystemDataset, SocialTippingDataset, get_saddle_configuration
from neural_networks.drm_loss import StableDRMLoss
from neural_networks.drm import DiscreteRepresentationsModel, LinearProbe, initialize_model_weights
from neural_networks.drm_viz import (
    visualize_state_space, analyze_discrete_state_transitions, 
    visualize_transition_matrices, visualize_model_architecture, 
    plot_training_curves, analyze_mdp_from_model, visualize_mdp, plot_vicreg_metrics,
    create_state_viz_from_data, analyze_state_assignment_evolution,
    create_gif_from_frames, create_state_evolution_analysis,
    create_png_from_frame_data, create_gif_from_data_frames
)

def set_all_seeds(seed):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Also set deterministic behavior for CUDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_state_weights(targets):
    """
    Calculate sample-level weights based on ground truth state frequency
    
    Args:
        targets: [num_samples, num_halfspaces] binary targets
    
    Returns:
        sample_weights: [num_samples] weight for each sample
    """
    # Convert each sample's halfspace pattern to a state ID
    state_ids = []
    for i in range(targets.shape[0]):
        # Convert binary pattern to unique state ID
        state_pattern = tuple(targets[i].numpy())
        state_ids.append(state_pattern)
    
    # Count frequency of each state
    state_counts = Counter(state_ids)
    total_samples = len(state_ids)
    
    print("State distribution:")
    for state, count in state_counts.items():
        print(f"  State {state}: {count} samples ({count/total_samples*100:.1f}%)")
    
    # Calculate inverse frequency weights
    sample_weights = []
    for state_id in state_ids:
        # Weight = 1 / (num_states * frequency)
        weight = total_samples / (len(state_counts) * state_counts[state_id])
        sample_weights.append(weight)
    
    return torch.tensor(sample_weights, dtype=torch.float32)

def train_probe(features, targets, probe_name, num_epochs=50, lr=0.025, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = features.shape[1]
    num_saddles = targets.shape[1]
    probe = LinearProbe(input_dim, num_saddles).to(device)
    
    # Calculate weights from the extracted targets
    sample_weights = calculate_state_weights(targets)
    
    # Create dataset and weighted sampler
    dataset = torch.utils.data.TensorDataset(features, targets)
    weighted_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=weighted_sampler)
    
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for epoch in range(num_epochs):
        for batch_features, batch_targets in dataloader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            predictions = probe(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
        
        # Evaluate on full dataset
        if (epoch==0) or (epoch==10) or (epoch==40):
            with torch.no_grad():
                full_predictions = probe(features.to(device))
                binary_preds = (full_predictions > 0.5).float()
                targets_device = targets.to(device)
                
                # Balanced accuracy - calculate accuracy per state, then average
                unique_states = []
                state_accuracies = []
                
                # Get unique state patterns
                for i in range(targets_device.shape[0]):
                    state_pattern = tuple(targets_device[i].cpu().numpy())
                    if state_pattern not in unique_states:
                        unique_states.append(state_pattern)
                
                # Calculate accuracy for each state
                for state_pattern in unique_states:
                    # Find samples with this state
                    state_mask = torch.zeros(targets_device.shape[0], dtype=torch.bool)
                    for i in range(targets_device.shape[0]):
                        if tuple(targets_device[i].cpu().numpy()) == state_pattern:
                            state_mask[i] = True
                    
                    if state_mask.sum() > 0:
                        state_preds = binary_preds[state_mask]
                        state_targets = targets_device[state_mask]
                        # All halfspace value match = state correctly learned
                        state_acc = (state_preds == state_targets).all(dim=1).float().mean()
                        state_accuracies.append(state_acc.item())
                
                # Balanced accuracy = mean of per-state accuracies
                state_accuracy = np.mean(state_accuracies)
                # Standard (unweighted) accuracy - all samples treated equally
                unweighted_accuracy = (binary_preds == targets_device).all(dim=1).float().mean().item()
                print(f" Epoch {epoch+1}/{num_epochs} - Balanced State Acc: {state_accuracy:.4f}, Unweighted Acc: {unweighted_accuracy:.4f}")
                    
    # Final evaluation with same balanced accuracy calculation
    with torch.no_grad():
        full_predictions = probe(features.to(device))
        binary_preds = (full_predictions > 0.5).float()
        targets_device = targets.to(device)
        
        # Balanced accuracy calculation (same as above)
        unique_states = []
        state_accuracies = []
        
        for i in range(targets_device.shape[0]):
            state_pattern = tuple(targets_device[i].cpu().numpy())
            if state_pattern not in unique_states:
                unique_states.append(state_pattern)
        
        for state_pattern in unique_states:
            state_mask = torch.zeros(targets_device.shape[0], dtype=torch.bool)
            for i in range(targets_device.shape[0]):
                if tuple(targets_device[i].cpu().numpy()) == state_pattern:
                    state_mask[i] = True
            
            if state_mask.sum() > 0:
                state_preds = binary_preds[state_mask]
                state_targets = targets_device[state_mask]
                state_acc = (state_preds == state_targets).all(dim=1).float().mean()
                state_accuracies.append(state_acc.item())
        
        state_accuracy = np.mean(state_accuracies)
        unweighted_accuracy = (binary_preds == targets_device).all(dim=1).float().mean().item()
    
    return probe, state_accuracy, unweighted_accuracy

def extract_features_and_targets(model, probing_loader, device, system_type):
    """
    Extract features and targets for layer probing
    """
    model.eval()
    
    dataset = probing_loader.dataset
    probing_indices = list(probing_loader.sampler.indices)
    
    discrete_features = []
    targets = []
    
    with torch.no_grad():
        batch_size = probing_loader.batch_size
        for i in range(0, len(probing_indices), batch_size):
            batch_indices = probing_indices[i:i+batch_size]
            
            batch_x = []
            for idx in batch_indices:
                x, _, _, _ = dataset[idx]
                batch_x.append(x)
            
            batch_x = torch.stack(batch_x)
            
            # Calculate targets for entire batch at once - EFFICIENT!
            if system_type == 'saddle_system':
                batch_targets = dataset._halfspace_for_batch(batch_x)
                targets.append(batch_targets)
            else:
                raise NotImplementedError(f"Layer probing not implemented for system_type: {system_type}")
            
            # Process batch through model
            batch_x_gpu = batch_x.to(device)
            discrete_x = model.get_state_probs(batch_x_gpu, training=False, hard=True, use_target=False)
            
            discrete_features.append(discrete_x.cpu())
    
    return (torch.cat(discrete_features), 
            torch.cat(targets))


def run_layer_probing(model, probing_loader, device, system_type, db_path):
    """Simplified layer probing with guaranteed data alignment"""
    
    print("\n" + "="*50)
    print("STARTING LAYER PROBING")
    print("="*50)

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    # Extract everything together - guaranteed alignment
    discrete_features, probe_targets = extract_features_and_targets(
        model, probing_loader, device, system_type
    )
    
    print(f"Extracted paired data - Features: {discrete_features.shape}, Targets: {probe_targets.shape}")
    
    # Train probes
    discrete_probe, discrete_acc, discrete_acc_unweighted = train_probe(discrete_features, probe_targets, "discrete")
    
    return {
        'discrete_accuracy': discrete_acc,
        'discrete_accuracy_unweighted': discrete_acc_unweighted
    }

def extract_state_assignment_data(model, device, num_states, 
                                system_type, bounds=None, grid_size=100, 
                                epoch=None, softmax_temp=1.0):
    """
    Extract state assignment data for visualization without creating plots.
    
    Args:
        model: Trained DRM model
        transformations: List of transformation functions for each dimension
        device: PyTorch device
        num_states: Number of discrete states
        system_type: Type of system ('tech_substitution', 'saddle_system')
        bounds: [(x1_min, x1_max), (x2_min, x2_max)] or None for defaults
        grid_size: Number of points per dimension (grid_size x grid_size total)
        epoch: Current epoch number (for tracking)
        softmax_temp: Temperature for softmax (default 1.0 for visualization)
    
    Returns:
        pd.DataFrame: Contains grid points, transformations, and state probabilities
    """
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = [(-5, 5), (-5, 5)]
    
    # Generate grid points in original space
    x_range = np.linspace(bounds[0][0], bounds[0][1], grid_size)
    y_range = np.linspace(bounds[1][0], bounds[1][1], grid_size)
    xx, yy = np.meshgrid(x_range, y_range)
    
    # Flatten grid points
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(grid_points).to(device)
        
        state_probs = model.get_state_probs(obs_tensor, training=False, soft=True)
        
        state_probs = state_probs.cpu().numpy()
    
    # Create DataFrame
    df_data = []
    for i, orig_point in enumerate(grid_points):
        row = {
            'x1': orig_point[0],
            'x2': orig_point[1], 
            'grid_idx': i
        }
        
        # Add state probabilities
        for state_idx in range(num_states):
            row[f'state_{state_idx}_prob'] = state_probs[i, state_idx]
        
        # Add dominant state and its probability
        dominant_state = np.argmax(state_probs[i])
        row['dominant_state'] = dominant_state
        row['dominant_prob'] = state_probs[i, dominant_state]
        
        # Add metadata
        if epoch is not None:
            row['epoch'] = epoch
        row['softmax_temp'] = softmax_temp
        row['system_type'] = system_type
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Add grid metadata as attributes
    df.attrs = {
        'grid_size': grid_size,
        'bounds': bounds,
        'num_states': num_states,
        'x_range': x_range.tolist(),
        'y_range': y_range.tolist()
    }
    
    return df

def create_optimizer_with_bias_exclusion(model, optimizer_type, learning_rate, weight_decay):
    """
    Create optimizer with weight decay and bias exclusion (following I-JEPA approach)
    """
    if optimizer_type == 'adamw' and weight_decay > 0:
        # Separate parameter groups: regular weights vs bias/layernorm
        param_groups = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if ('bias' not in n) and (len(p.shape) != 1)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if ('bias' in n) or (len(p.shape) == 1)],
                'weight_decay': 0.0  # No weight decay for bias/layernorm
            }
        ]
        optimizer = optim.AdamW(param_groups, lr=learning_rate)
        print(f"Using AdamW optimizer with weight_decay={weight_decay} (bias excluded)")
    
    elif optimizer_type == 'adamw':
        # AdamW without weight decay
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
        print(f"Using AdamW optimizer without weight decay")
    
    else:  # adam
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print(f"Using Adam optimizer")
    
    return optimizer

def create_scheduler(optimizer, scheduler_type, epochs, warmup_epochs, use_warmup, min_lr, restart_period, restart_mult):
    """
    Create learning rate scheduler
    """
    if scheduler_type == 'fixed':
        # No scheduler - constant learning rate
        print("Using fixed learning rate (no scheduling)")
        return None

    
    elif scheduler_type == 'warm_restarts':
        # WarmRestarts - starts after warmup period
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=restart_period,
            T_mult=restart_mult,
            eta_min=min_lr
        )
        print(f"Using CosineAnnealingWarmRestarts scheduler (T_0: {restart_period}, T_mult: {restart_mult}, min_lr: {min_lr:.2e})")
    
    else:  # cosine
        T_max = epochs - warmup_epochs if use_warmup else epochs
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, T_max),
            eta_min=min_lr
        )
        print(f"Using CosineAnnealingLR scheduler (T_max: {T_max}, min_lr: {min_lr:.2e})")
    
    return lr_scheduler

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

def save_state_data_frame(grid_points, state_probs, epoch, num_states, 
                         output_dir, run_id, bounds, grid_size=100):
    """
    Save lightweight state data instead of PNG frame.
    """
    import pickle
    
    # Create lightweight data structure
    frame_data = {
        'grid_points': grid_points,
        'state_probs': state_probs,
        'epoch': epoch,
        'num_states': num_states,
        'bounds': bounds,
        'grid_size': grid_size,
        'run_id': run_id
    }
    
    # Save as pickle file (much faster than PNG)
    data_filename = f"state_data_epoch_{epoch}_{run_id}.pkl"
    output_dir = Path(output_dir)
    data_path = output_dir / data_filename
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(data_path, 'wb') as f:
        pickle.dump(frame_data, f)
    
    return str(data_path)

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


def train_drm_model(db_path, 
                    system_type,
                    output_dir="./neural_networks/output",
                    run_id=None,
                    seed=42,
                    val_size=1000,
                    test_size=1000, 
                    probing_size=None,
                    batch_size=64, 
                    epochs=100,
                    learning_rate=5e-5, 
                    num_states=4,
                    hidden_dim=32,
                    checkpoint_every=25,
                    state_loss_weight=1.0, 
                    value_loss_weight=1.5,
                    predictor_type='standard',
                    value_method=None,
                    value_loss_type=None,
                    use_warmup=False,
                    warmup_epochs=10,
                    min_lr=5e-6,
                    use_gumbel=False,
                    initial_temp=5.0,
                    min_temp=0.1,
                    use_entropy_reg=False,
                    entropy_weight=5.0,
                    use_entropy_decay=True,
                    entropy_decay_proportion=0.2,
                    use_target_encoder=False,
                    ema_decay=0.9,
                    state_loss_type="kl_div",
                    sort_states=False,
                    use_vicreg=True,
                    #TODO: adjust vicreg params
                    vicreg_weight=0.1,
                    vicreg_lambda=25.0,
                    vicreg_mu=25.0,
                    vicreg_nu=1.0,
                    vicreg_variance_target=0.4,
                    vicreg_invariance_schedule=True,
                    encoder_init_method="he",
                    optimizer_type='adam',
                    scheduler_type='cosine',
                    weight_decay=0.0,
                    restart_period=20,
                    restart_mult=1
                    ):
    """
    Full training function for the Discrete Representations Model with stability improvements
    
    Args:
        db_path: Path to the SQLite database
        output_dir: Directory to save model checkpoints and logs
        val_size: Number of samples to use for validation
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate for the optimizer
        num_states: Number of discrete states in the model
        hidden_dim: Hidden dimension size in the model
        checkpoint_every: Save checkpoint every N epochs
    
    Returns:
        model: Trained model
        history: Dictionary containing training history
    """
    # Set seeds for reproducibility at the very beginning
    set_all_seeds(seed)

    # Get system configuration
    system_config = get_system_config(SystemType[system_type.upper()])
    # Get system-specific transformation
    transformation = get_transformation(SystemType[system_type.upper()])
    
    # Validate/set value_method
    if value_method is None:
        value_method = system_config['default_value_method']
        print(f"Using default value method: {value_method}")
    elif value_method not in system_config['value_methods']:
        raise ValueError(f"Invalid value method '{value_method}' for {system_type}. "
                        f"Available: {system_config['value_methods']}")
    
    # Set default loss types based on system if not specified
    if state_loss_type is None:
        state_loss_type = "kl_div"  # Default for all systems



    # Update default value_loss_type to use system registry
    if value_loss_type is None:
        # Get from system registry
        value_loss_type = system_config['default_value_loss'][value_method]
        print(f"Using default value loss type: {value_loss_type}")


    # Access the global project root
    project_root = PROJECT_ROOT
    
    # Ensure db_path is absolute
    db_path = Path(db_path)
    if not db_path.is_absolute():
        # If relative, assume it's relative to project root
        db_path = project_root / db_path
    
    print(f"Using database at: {db_path}")
    
    # Make output_dir absolute
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize time and run ID
    start_time = time.time()
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if probing_size is not None:
        train_loader, val_loader, test_loader, probing_loader = create_data_loaders(
            system_type=system_type,
            db_path=db_path,
            value_method=value_method,
            batch_size=batch_size,
            val_size=val_size, 
            test_size=test_size,
            probing_size=probing_size,
            seed=seed
        )
    else:
        train_loader, val_loader, test_loader = create_data_loaders(
            system_type=system_type,
            db_path=db_path,
            value_method=value_method,
            batch_size=batch_size,
            val_size=val_size, 
            test_size=test_size,
            seed=seed
        )
        probing_loader = None

    # Get dimensions from first batch
    for x, c, y, v_true in train_loader:
        obs_dim = x.shape[1]
        control_dim = c.shape[1]
        value_dim = v_true.shape[1]
        break

    print(f"Detected dimensions: obs_dim={obs_dim}, control_dim={control_dim}, value_dim={value_dim}")

    # For visualization later
    points_config = None
    if system_type == 'saddle_system':
        saddle_config = get_saddle_configuration(db_path)
        if saddle_config:
            points_config = {
                'points': saddle_config['saddle_points'],
                'angles_degrees': saddle_config['angles_degrees']
            }


    # Update checkpoint config to use actual dimensions
    checkpoint_config = {
        'obs_dim': obs_dim,
        'control_dim': control_dim,
        'value_dim': value_dim,
        'num_states': num_states,
        'hidden_dim': hidden_dim,
        'system_type': system_type,
        'value_method': value_method
    }    

    value_activation = system_config['value_activation'].get(value_method, None)
    if value_activation is None:
        value_activation = 'identity' 
    
    # Initialize model with system-specific settings
    model = DiscreteRepresentationsModel(
        obs_dim=obs_dim,
        control_dim=control_dim,
        value_dim=value_dim,
        num_states=num_states,
        hidden_dim=hidden_dim,
        predictor_type=predictor_type,
        use_gumbel=use_gumbel,
        initial_temp=initial_temp,
        min_temp=min_temp,
        use_target_encoder=use_target_encoder,
        ema_decay=ema_decay,
        value_activation=value_activation
    )
    print(f"Model created with control_dim={model.predictor.control_dim}")
    
    # initialize weights
    initialize_model_weights(model, encoder_init_method=encoder_init_method)

    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Initialize loss function with appropriate types
    loss_fn = StableDRMLoss(
        state_loss_weight=state_loss_weight, 
        value_loss_weight=value_loss_weight,
        use_entropy_reg=use_entropy_reg,
        entropy_weight=entropy_weight,
        use_entropy_decay=use_entropy_decay,
        entropy_decay_proportion=entropy_decay_proportion,
        state_loss_type=state_loss_type,
        value_loss_type=value_loss_type,
        value_method=value_method,
        use_vicreg=use_vicreg,
        vicreg_weight=vicreg_weight,
        vicreg_lambda=vicreg_lambda,
        vicreg_mu=vicreg_mu,
        vicreg_nu=vicreg_nu,
        vicreg_variance_target=vicreg_variance_target,
        vicreg_invariance_schedule=vicreg_invariance_schedule
    )

    optimizer = create_optimizer_with_bias_exclusion(
        model, optimizer_type, learning_rate, weight_decay)

    # Setup learning rate scheduling if enabled
    if use_warmup:
        print(f"Using warmup for first {warmup_epochs} epochs (starting LR: {learning_rate * 0.1:.2e})")

    # Create scheduler
    lr_scheduler = create_scheduler(
        optimizer, scheduler_type, epochs, warmup_epochs, 
        use_warmup, min_lr, restart_period, restart_mult)

    # Add gradient clipping
    max_grad_norm = 1.0
    
    # Prepare for training
    history = {
        "train_loss": [],
        "train_state_loss": [],
        "train_value_loss": [],
        "train_entropy_loss": [],
        "train_batch_entropy": [],
        "train_individual_entropy": [],
        "train_vicreg_total": [],
        "train_vicreg_invariance": [],
        "train_vicreg_variance": [],
        "train_vicreg_covariance": [],
        "train_entropy_weight": [],
        "val_loss": [],
        "val_state_loss": [],
        "val_value_loss": [],
        "val_entropy_loss": [],
        "val_batch_entropy": [],
        "val_individual_entropy": [],
        "val_vicreg_total": [],
        "val_vicreg_invariance": [],
        "val_vicreg_variance": [],
        "val_vicreg_covariance": [],
        "state_metrics": []
    }

    # Initialize state metrics collection (replace state_assignment_data)
    collect_every_n_epochs = 2
    collect_initial = True  
    collect_early_batches = 20
    visualization_frames = []  # Store frame paths for GIF
    prev_dominant_states = None  # For stability calculation

    # COLLECTION RIGHT AFTER INITIALIZATION (EPOCH 0)
    if collect_initial:
        print("Collecting initial state assignments (epoch 0)...")
        bounds = get_visualization_bounds(SystemType[system_type.upper()])
        
        metrics, dominant_states, grid_points, state_probs = extract_and_calculate_metrics(
            model=model,
            device=device,
            num_states=num_states,
            system_type=system_type,
            bounds=bounds,
            epoch=0,
            prev_dominant_states=None
        )
        
        # Store metrics
        history["state_metrics"].append(metrics)
        
        # Save lightweight data frame
        data_frame_path = save_state_data_frame(
            grid_points, state_probs, epoch=0, num_states=num_states,
            output_dir=output_dir, run_id=run_id, bounds=bounds
        )
        visualization_frames.append(data_frame_path)
        
        # Keep dominant states for next epoch's stability calculation
        prev_dominant_states = dominant_states

    
    # Start training loop
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Update Gumbel temperature
        if use_gumbel:
            current_temp = model.update_temperature(epoch, epochs)
            print(f"Epoch {epoch+1}/{epochs} - Gumbel temperature: {current_temp:.4f}")

        # Always print current LR for debugging
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.2e}")

        # Handle warmup if enabled
        if use_warmup and epoch < warmup_epochs:
            # Linearly increase learning rate during warmup period
            progress = (epoch + 1) / warmup_epochs
            new_lr = learning_rate * (0.1 + 0.9 * progress)  # 10% to 100% of base LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f"Warmup epoch {epoch+1}/{warmup_epochs}, LR set to {new_lr:.2e}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_state_loss = 0.0
        train_value_loss = 0.0
        train_entropy_loss = 0.0
        train_batch_entropy = 0.0 
        train_individual_entropy = 0.0
        train_vicreg_total = 0.0
        train_vicreg_invariance = 0.0
        train_vicreg_variance = 0.0
        train_vicreg_covariance = 0.0

        # TODO: maybe move this to the loss function
        if loss_fn.use_entropy_reg:
            entropy_weight = loss_fn.update_entropy_weight(epoch, epochs)
            history["train_entropy_weight"].append(entropy_weight)
            decay_status = "decaying" if loss_fn.use_entropy_decay else "constant"
            print(f"Epoch {epoch+1}/{epochs} - Entropy weight: {entropy_weight:.4f} ({decay_status})")
        
        for batch_idx, (x, c, y, v_true) in enumerate(train_loader):

            # Move data to device
            x, c, y, v_true = x.to(device), c.to(device), y.to(device), v_true.to(device)
            
            # Check for NaN values in the batch
            if (torch.isnan(x).any() or torch.isnan(c).any() or 
                torch.isnan(y).any() or torch.isnan(v_true).any()):
                print(f"WARNING: NaN values in batch {batch_idx}, skipping")
                continue
            
            # Zero gradients
            optimizer.zero_grad()
            
            try:
                # Forward pass
                s_x, s_y, s_y_pred, v_pred_for_all_states, embed_x, embed_y = model(x, c, y, v_true, training=True)
                
                # Check for NaN values in model outputs
                if torch.isnan(s_y).any() or torch.isnan(s_y_pred).any() or torch.isnan(v_pred_for_all_states).any():
                    print(f"WARNING: NaN values in model outputs for batch {batch_idx}, skipping")
                    continue
                
                # Calculate loss
                (total_loss, state_loss, value_loss, entropy_loss, batch_entropy, individual_entropy,
                    vicreg_total, vicreg_invariance, vicreg_variance, vicreg_covariance) = loss_fn(
                        s_y, s_y_pred, v_true, v_pred_for_all_states, s_x, 
                        embed_x, embed_y, epoch=epoch, max_epochs=epochs) 
                
                # Check if loss is NaN
                if torch.isnan(total_loss):
                    print(f"WARNING: NaN loss in batch {batch_idx}, skipping")
                    continue
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Update weights
                optimizer.step()

                if model.use_target_encoder:
                    model.update_target_encoder()
                
                # Print batch info occasionally
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}: Loss={total_loss.item():.4f}")
                
                # Accumulate losses
                train_loss += total_loss.item()
                train_state_loss += state_loss.item()
                train_value_loss += value_loss.item()
                train_entropy_loss += entropy_loss.item()
                train_batch_entropy += batch_entropy.item()
                train_individual_entropy += individual_entropy.item()
                train_vicreg_total += vicreg_total.item()
                train_vicreg_invariance += vicreg_invariance.item()
                train_vicreg_variance += vicreg_variance.item()
                train_vicreg_covariance += vicreg_covariance.item()
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Average training losses
        num_batches = len(train_loader)
        if num_batches > 0:  # Avoid division by zero
            train_loss /= num_batches
            train_state_loss /= num_batches
            train_value_loss /= num_batches
            train_entropy_loss /= num_batches
            train_batch_entropy /= num_batches
            train_individual_entropy /= num_batches
            train_vicreg_total /= num_batches
            train_vicreg_invariance /= num_batches
            train_vicreg_variance /= num_batches
            train_vicreg_covariance /= num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_state_loss = 0.0
        val_value_loss = 0.0
        val_entropy_loss = 0.0
        val_batch_entropy = 0.0
        val_individual_entropy = 0.0
        val_vicreg_total = 0.0
        val_vicreg_invariance = 0.0
        val_vicreg_variance = 0.0
        val_vicreg_covariance = 0.0
        valid_batches = 0
        
        with torch.no_grad():
            for x, c, y, v_true in val_loader:
                # Move data to device
                x, c, y, v_true = x.to(device), c.to(device), y.to(device), v_true.to(device)
                
                # Check for NaN values in the batch
                if (torch.isnan(x).any() or torch.isnan(c).any() or 
                    torch.isnan(y).any() or torch.isnan(v_true).any()):
                    continue
                
                try:
                    # Forward pass
                    s_x, s_y, s_y_pred, v_pred_for_all_states, embed_x, embed_y = model(x, c, y, v_true, training=False)
                    
                    # Check for NaN values in model outputs
                    if torch.isnan(s_y).any() or torch.isnan(s_y_pred).any() or torch.isnan(v_pred_for_all_states).any():
                        continue
                    
                    # Calculate loss
                    (total_loss, state_loss, value_loss, entropy_loss, batch_entropy, individual_entropy,
                        vicreg_total, vicreg_invariance, vicreg_variance, vicreg_covariance) = loss_fn(
                            s_y, s_y_pred, v_true, v_pred_for_all_states, s_x, 
                            embed_x, embed_y, epoch=epoch, max_epochs=epochs)
                    
                    # Check if loss is NaN
                    if torch.isnan(total_loss):
                        continue
                    
                    # Accumulate losses
                    val_loss += total_loss.item()
                    val_state_loss += state_loss.item()
                    val_value_loss += value_loss.item()
                    val_entropy_loss += entropy_loss.item()
                    val_batch_entropy += batch_entropy.item()
                    val_individual_entropy += individual_entropy.item()
                    val_vicreg_total += vicreg_total.item()
                    val_vicreg_invariance += vicreg_invariance.item()
                    val_vicreg_variance += vicreg_variance.item()
                    val_vicreg_covariance += vicreg_covariance.item()
                    valid_batches += 1
                    
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue
        
        # Average validation losses
        if valid_batches > 0:  # Avoid division by zero
            val_loss /= valid_batches
            val_state_loss /= valid_batches
            val_value_loss /= valid_batches
            val_entropy_loss /= valid_batches
            val_batch_entropy /= valid_batches
            val_individual_entropy /= valid_batches
            val_vicreg_total /= valid_batches
            val_vicreg_invariance /= valid_batches
            val_vicreg_variance /= valid_batches
            val_vicreg_covariance /= valid_batches
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_state_loss"].append(train_state_loss)
        history["train_value_loss"].append(train_value_loss)
        history["train_entropy_loss"].append(train_entropy_loss)
        history["train_batch_entropy"].append(train_batch_entropy)
        history["train_individual_entropy"].append(train_individual_entropy)
        history["train_vicreg_total"].append(train_vicreg_total)
        history["train_vicreg_invariance"].append(train_vicreg_invariance)
        history["train_vicreg_variance"].append(train_vicreg_variance)
        history["train_vicreg_covariance"].append(train_vicreg_covariance)
        history["val_loss"].append(val_loss)
        history["val_state_loss"].append(val_state_loss)
        history["val_value_loss"].append(val_value_loss)
        history["val_entropy_loss"].append(val_entropy_loss)
        history["val_batch_entropy"].append(val_batch_entropy)
        history["val_individual_entropy"].append(val_individual_entropy)
        history["val_vicreg_total"].append(val_vicreg_total)
        history["val_vicreg_invariance"].append(val_vicreg_invariance)
        history["val_vicreg_variance"].append(val_vicreg_variance)
        history["val_vicreg_covariance"].append(val_vicreg_covariance)
        
        # Print progress
        # Print progress with all metrics
        print(f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f} (State: {train_state_loss:.4f}, Value: {train_value_loss:.4f} "
            )
        print(f"  VICReg: {train_vicreg_total/num_batches:.4f} (Inv: {train_vicreg_invariance/num_batches:.4f}, "
            f"Var: {train_vicreg_variance/num_batches:.4f}, Cov: {train_vicreg_covariance/num_batches:.4f})")
        print(f"  Batch Entropy: {train_batch_entropy:.4f}, Individual Entropy: {train_individual_entropy:.4f}")
        print(f"Val Loss: {val_loss:.4f} (State: {val_state_loss:.4f}, Value: {val_value_loss:.4f}")
        print(f"  Batch Entropy: {val_batch_entropy:.4f}, Individual Entropy: {val_individual_entropy:.4f}")


        if epoch == 10 or epoch == 50:

            # Visualize the state space
            state_vis_path = os.path.join(output_dir, f"states_after{epoch}_{run_id}.png")
            visualize_state_space(
                model=model,
                output_path=state_vis_path,
                transformations=[
                    transformation,  # For x1 dimension
                    transformation   # For x2 dimension
                ],
                device=device,
                num_states=num_states,
                system_type=system_type,
                # TODO add point and line?
                # TODO: move bounds to argparser
                bounds=get_visualization_bounds(SystemType[system_type.upper()])
            )
        
        # Collect state assignment data
        if (epoch % collect_every_n_epochs == 0) or (epoch in (0, 1, 2, 3)):
            print(f"Collecting state assignment metrics for epoch {epoch+1}...")
            
            bounds = get_visualization_bounds(SystemType[system_type.upper()])
            
            metrics, dominant_states, grid_points, state_probs = extract_and_calculate_metrics(
                model=model,
                device=device,
                num_states=num_states,
                system_type=system_type,
                bounds=bounds,
                epoch=epoch+1,
                prev_dominant_states=prev_dominant_states
            )
            
            # Store metrics
            history["state_metrics"].append(metrics)
            
            # Save lightweight data frame (not PNG!)
            data_frame_path = save_state_data_frame(
                grid_points, state_probs, epoch=epoch+1, num_states=num_states,
                output_dir=output_dir, run_id=run_id, bounds=bounds
            )
            visualization_frames.append(data_frame_path)
            
            # Update for next epoch's stability calculation
            prev_dominant_states = dominant_states


        # Apply learning rate scheduler after validation
        if lr_scheduler is not None and (not use_warmup or epoch >= warmup_epochs):
            lr_scheduler.step()
        
            
        # Save checkpoint every N epochs
        if (epoch + 1) % checkpoint_every == 0 or (epoch+1) == 10:
            checkpoint_path = os.path.join(output_dir, f"drm_checkpoint_epoch{epoch+1}_{run_id}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': checkpoint_config
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # After training, create visualizations and analysis from collected metrics
    if history["state_metrics"]:
        print("Creating state evolution analysis and GIF...")
        
        # Create state evolution analysis plot
        analysis_path = os.path.join(output_dir, f"state_evolution_analysis_{run_id}.png")
        analysis_result = create_state_evolution_analysis(
            all_metrics=history["state_metrics"],
            output_path=analysis_path,
            num_states=num_states
        )
        print(f"Created state evolution analysis: {analysis_path}")
        
    # Create GIF from collected data frames
    if len(visualization_frames) > 1:
        gif_path = create_gif_from_data_frames(
            data_frame_paths=visualization_frames,
            output_path=os.path.join(output_dir, f"state_evolution_{run_id}"),
            gif_duration=250
        )
        if gif_path:
            print(f"Created state evolution GIF: {gif_path}")

    # Clean up individual data frame files to save space
    for data_frame_path in visualization_frames:
        try:
            os.remove(data_frame_path)
            print(f"Cleaned up data frame: {os.path.basename(data_frame_path)}")
        except Exception as e:
            print(f"Warning: Could not clean up {data_frame_path}: {e}")
            pass   


    # Final evaluation on test set
    print("Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_state_loss = 0.0
    test_value_loss = 0.0
    test_entropy_loss = 0.0
    test_batch_entropy = 0.0
    test_individual_entropy = 0.0
    test_vicreg_total = 0.0
    test_vicreg_invariance = 0.0
    test_vicreg_variance = 0.0
    test_vicreg_covariance = 0.0
    test_samples = 0
    
    with torch.no_grad():
        for x, c, y, v_true in test_loader:
            # Move data to device
            x, c, y, v_true = x.to(device), c.to(device), y.to(device), v_true.to(device)
            
            # Skip batches with NaN values
            if (torch.isnan(x).any() or torch.isnan(c).any() or 
                torch.isnan(y).any() or torch.isnan(v_true).any()):
                continue
                
            try:
                # Forward pass
                s_x, s_y, s_y_pred, v_pred_for_all_states, embed_x, embed_y = model(x, c, y, v_true, training=False)
                
                # Skip if model outputs have NaN values
                if torch.isnan(s_y).any() or torch.isnan(s_y_pred).any() or torch.isnan(v_pred_for_all_states).any():
                    continue
                
                # TODO: entropy

                # Calculate loss
                (total_loss, state_loss, value_loss, entropy_loss, batch_entropy, individual_entropy,
                    vicreg_total, vicreg_invariance, vicreg_variance, vicreg_covariance) = loss_fn(
                        s_y, s_y_pred, v_true, v_pred_for_all_states, s_x, 
                        embed_x, embed_y, epoch=epoch, max_epochs=epochs)
                
                # Skip if loss is NaN
                if torch.isnan(total_loss):
                    continue
                
                # Accumulate losses
                test_loss += total_loss.item() * len(x)
                test_state_loss += state_loss.item() * len(x)
                test_value_loss += value_loss.item() * len(x)
                test_entropy_loss += entropy_loss.item() * len(x)
                test_batch_entropy += batch_entropy.item() * len(x)
                test_individual_entropy += individual_entropy.item() * len(x)
                test_vicreg_total += vicreg_total.item() * len(x)
                test_vicreg_invariance += vicreg_invariance.item() * len(x)
                test_vicreg_variance += vicreg_variance.item() * len(x)
                test_vicreg_covariance += vicreg_covariance.item() * len(x)
                test_samples += len(x)
                
            except Exception as e:
                print(f"Error in test evaluation: {e}")
                continue
    
    # Calculate average test losses
    if test_samples > 0:
        test_loss /= test_samples
        test_state_loss /= test_samples
        test_value_loss /= test_samples
        test_entropy_loss /= test_samples
        test_batch_entropy /= test_samples
        test_individual_entropy /= test_samples
        test_vicreg_total /= test_samples
        test_vicreg_invariance /= test_samples
        test_vicreg_variance /= test_samples
        test_vicreg_covariance /= test_samples
    
    # Layer probing
    probing_results = None
    if probing_loader is not None and system_type == 'saddle_system':
        probing_results = run_layer_probing(model, probing_loader, device, system_type, db_path)

    

    # Create test metrics dictionary
    test_metrics = {
        "test_loss": float(test_loss),
        "test_state_loss": float(test_state_loss),
        "test_value_loss": float(test_value_loss),
        "test_entropy_loss": float(test_entropy_loss),
        "test_batch_entropy": float(test_batch_entropy),
        "test_individual_entropy": float(test_individual_entropy),
        "test_vicreg_total": float(test_vicreg_total),
        "test_vicreg_invariance": float(test_vicreg_invariance),
        "test_vicreg_variance": float(test_vicreg_variance),
        "test_vicreg_covariance": float(test_vicreg_covariance),
        "test_samples": test_samples,
        "prob_discrete_accuracy": probing_results['discrete_accuracy'] if probing_results else None,
        "prob_discrete_accuracy_unweighted": probing_results['discrete_accuracy_unweighted'] if probing_results else None
    }
    
    # Print test results
    print(f"Test Results:")
    print(f"  Loss: {test_loss:.4f} (State: {test_state_loss:.4f}, Value: {test_value_loss:.4f})")
    print(f"  Entropy Loss: {test_entropy_loss:.4f}")
    print(f"  Batch Entropy: {test_batch_entropy:.4f}, Individual Entropy: {test_individual_entropy:.4f}")
    if probing_results:
        print(f"  Linear Probing - Discrete Accuracy: {probing_results['discrete_accuracy']:.4f}, Unweighted Accuracy: {probing_results['discrete_accuracy_unweighted']:.4f}")

    # Save test results
    test_results_path = os.path.join(output_dir, f"drm_test_results_{run_id}.json")
    with open(test_results_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"Saved test results to {test_results_path}")

    
    # Save final model
    final_model_path = os.path.join(output_dir, f"drm_final_{run_id}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': checkpoint_config
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")

    # Sort states if requested and save sorted model separately
    if sort_states:
        print("Sorting states by value...")
        try:
            # Make a copy for sorting (don't modify the original model)
            sorted_model = copy.deepcopy(model)
            sorted_model, sorted_indices = sorted_model.sort_states_by_value(
                system_type=args.system_type,
                value_method=args.value_method
            )
            
            # Save sorted model
            sorted_model_path = os.path.join(output_dir, f"drm_final_sorted_{run_id}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': sorted_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': checkpoint_config,
                'sorted_indices': sorted_indices.tolist(),
                'system_type': args.system_type,
                'value_method': args.value_method
            }, sorted_model_path)
            print(f"Saved sorted model to {sorted_model_path}")
            
            # Use sorted model for visualizations
            model = sorted_model
            print("State sorting completed successfully. Using sorted model for visualizations.")
            
        except Exception as e:
            print(f"Warning: State sorting failed: {e}")
            print("Continuing with original model for visualizations...")
    
    # Save training history
    history_path = os.path.join(output_dir, f"drm_history_{run_id}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"Saved training history to {history_path}")
    
    # Plot training curves
    plot_training_curves(history, os.path.join(output_dir, f"training_curves_{run_id}.png"), 
                         state_loss_type=loss_fn.state_loss_type)
    # Plot VICReg metrics
    vicreg_weights = {
        'lambda': vicreg_lambda, 
        'mu': vicreg_mu, 
        'nu': vicreg_nu
    }
    # plot_vicreg_metrics(history, test_metrics, vicreg_weights, os.path.join(output_dir, f"vicreg_metrics_{run_id}.png"))

    # Visualize Modl Architecture
    #arch_vis_path = os.path.join(output_dir, f"model_architecture_{run_id}")
    #visualize_model_architecture(model, arch_vis_path)

    # Visualize the state space
    # state_vis_path = os.path.join(output_dir, f"states_{run_id}.png")
    """visualize_state_space(
        model=model,
        output_path=state_vis_path,
        transformations=[
            transformation,  # For x1 dimension
            transformation   # For x2 dimension
        ],
        device=device,
        num_states=num_states,
        system_type=system_type,
        points=points_config['points'] if points_config else None,
        angles_degrees=points_config['angles_degrees'] if points_config else None,
        # TODO: move bounds to argparser
        bounds=[(-5, 5), (-5, 5)] if points_config else None
    )"""

    state_soft_vis_path = os.path.join(output_dir, f"states_soft_{run_id}.png")
    visualize_state_space(
        model=model,
        output_path=state_soft_vis_path,
        transformations=[
            transformation,  # For x1 dimension
            transformation   # For x2 dimension
        ],
        device=device,
        num_states=num_states,
        soft=True,
        system_type=system_type,
        points=points_config['points'] if points_config else None,
        angles_degrees=points_config['angles_degrees'] if points_config else None,
        # TODO: move bounds to argparser
        bounds=get_visualization_bounds(SystemType[system_type.upper()])
    )

    # Analyze and visualize state transitions with argmax assignment
    
    # Get control values based on system type
    if system_type == 'tech_substitution':
        # For continuous controls, use sensible defaults
        control_values = [0.5, 1.0]
    elif system_type == 'saddle_system':
        # For categorical controls (derived from data loader)
        control_values = list(range(control_dim)) #all categorical options
    elif system_type == 'social_tipping':
        # Use all possible control combinations for analysis
        control_values = [
            [0.6, 0.65, 0.65, 0.65],   # default
            [0.65, 0.65, 0.65, 0.65],  # subsidy  
            [0.6, 0.6, 0.65, 0.65],    # tax
            [0.6, 0.65, 0.65, 0.6]     # campaign
        ]
        
        print(f"Using all possible control combinations for social_tipping analysis")
        for i, ctrl in enumerate(control_values):
            print(f"  Control {i+1}: {ctrl}")

    else:
        raise ValueError(f"Unknown system type: {system_type}")

    transition_matrices = analyze_discrete_state_transitions(
        model=model,
        control_values=control_values,
        device=device,
        system_type=system_type
    )
    transitions_vis_path = os.path.join(output_dir, f"transitions_{run_id}.png")
    visualize_transition_matrices(transition_matrices, control_values, transitions_vis_path)
    #NOTE: visualization was too chaotic
    #mdp_vis_path = os.path.join(output_dir, f"mdp_visualization_{run_id}.png")
    #mdp_data = analyze_mdp_from_model(model, control_values=control_values, device='cuda' if torch.cuda.is_available() else 'cpu')
    #graphs, paths = visualize_mdp(mdp_data, output_path=mdp_vis_path, min_prob_to_show=0.02)


    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    return model, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Discrete Representations Model')
    parser.add_argument('db_path', type=str, help='Path to the SQLite database (can be relative to project root)')
    parser.add_argument('--output_dir', type=str, default='neural_networks/output', help='Directory to save outputs')
    parser.add_argument('--run_id', type=str, default=None, 
                    help='Custom run ID for this training run (default: timestamp)')
    parser.add_argument('--val_size', type=int, default=1000, help='Number of validation samples')
    parser.add_argument('--test_size', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--num_states', type=int, default=4, help='Number of discrete states')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension size')
    parser.add_argument('--state_loss_weight', type=float, default=1.0, help='Stateloss weight')
    parser.add_argument('--value_loss_weight', type=float, default=2.0, help='Value loss weight')

    parser.add_argument('--predictor_type', type=str, default='standard', 
                        choices=['standard', 'control_gate', 'bilinear'],
                        help='Type of predictor to use (standard, control_gate or bilinear)')

    parser.add_argument('--use_warmup', action='store_true', 
                        help='Use learning rate warmup')
    parser.add_argument('--warmup_epochs', type=int, default=10, 
                        help='Number of epochs for warmup')
    parser.add_argument('--min_lr', type=float, default=5e-6, 
                        help='Minimum learning rate')
    
    parser.add_argument('--use_gumbel', action='store_true', 
                        help='Use Gumbel softmax for state encoding')
    parser.add_argument('--initial_temp', type=float, default=3.0,
                        help='Initial temperature for Gumbel softmax')
    parser.add_argument('--min_temp', type=float, default=0.5,
                        help='Minimum temperature for Gumbel softmax')
    
    parser.add_argument('--use_entropy_reg', action='store_true', 
                    help='Use entropy regularization to prevent state collapse')
    parser.add_argument('--entropy_weight', type=float, default=1.0,
                        help='Weight for entropy regularization loss')
    parser.add_argument('--use_entropy_decay', action='store_true', 
                        help='Decay entropy regularization weight during training')
    parser.add_argument('--no_entropy_decay', dest='use_entropy_decay', action='store_false',
                        help='Keep entropy regularization weight constant')
    parser.set_defaults(use_entropy_decay=True)  # Default is to use entropy decay
    parser.add_argument('--entropy_decay_proportion', type=float, default=0.5,
                    help='Proportion of training after which entropy weight reaches minimum (0.2 = 20%%)')
    
    parser.add_argument('--use_target_encoder', action='store_true',
                    help='Use target encoder with EMA updates')
    parser.add_argument('--ema_decay', type=float, default=0.9,
                    help='EMA decay rate for target encoder (higher = slower updates)')
    
    
    parser.add_argument('--state_loss_type', type=str, default=None,
                    choices=['kl_div', 'cross_entropy', 'mse', 'js_div'],
                    help='Type of state loss function to use')
    
    parser.add_argument('--value_loss_type', type=str, default=None,
                choices=['mse', 'angular', 'binary_cross_entropy'],
                help='Type of value loss function to use')

    # System-specific parameters
    parser.add_argument('--value_method', type=str, default=None, help='Which value function should be used (system-specific)')
    parser.add_argument('--system_type', type=str, required=True,
                        choices=['tech_substitution', 'saddle_system', 'social_tipping'],
                        help='Type of dynamical system to train on')

    # Randomness control
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    parser.add_argument('--sort_states', action='store_true',
                    help='Sort states by value after training for consistent visualizations')
    
    # VICReg arguments
    # TODO: adjust default
    parser.add_argument('--use_vicreg', action='store_true', 
                    help='Use VICReg-style state regularization')
    parser.add_argument('--vicreg_weight', type=float, default=0.1,
                help='Overall weight for VICReg loss components')
    parser.add_argument('--vicreg_lambda', type=float, default=25.0,
                    help='Weight for VICReg invariance term')
    parser.add_argument('--vicreg_mu', type=float, default=25.0,
                    help='Weight for VICReg variance term')
    parser.add_argument('--vicreg_nu', type=float, default=1.0,
                    help='Weight for VICReg covariance term')
    parser.add_argument('--vicreg_variance_target', type=float, default=0.4,
                    help='Target standard deviation for VICReg variance term')
    parser.add_argument('--vicreg_invariance_schedule', action='store_true',
                    help='Gradually reduce VICReg invariance weight over training')
    parser.add_argument('--probing_size', type=int, default=None,
                    help='Number of samples to reserve for layer probing (default: None)')
    
    parser.add_argument('--encoder_init_method', type=str, default='xavier_uniform',
                    choices=['xavier_uniform', 'xavier_normal', 'chaos', 'chaos2', 'he'],
                    help='Initialization method for encoder layers')
    
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw'], 
                    default='adam', help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help='Weight decay (only used with AdamW)')
    parser.add_argument('--restart_period', type=int, default=20, 
                        help='T_0 for warm restarts scheduler')
    parser.add_argument('--restart_mult', type=int, default=1, 
                        help='T_mult for warm restarts (1=fixed period, 2=doubling)')
    parser.add_argument('--scheduler_type', type=str, 
                    choices=['cosine', 'warm_restarts', 'fixed'], 
                    default='cosine', help='Type of learning rate scheduler to use')
    
    args = parser.parse_args()

    # Create the run_id and complete output directory
    #run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_dir)
    if not base_output_dir.is_absolute():
        base_output_dir = Path.cwd() / base_output_dir

    run_id = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the final output directory with run_id
    output_dir = base_output_dir / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the config BEFORE training
    args_dict = vars(args)
    args_dict['run_id'] = run_id  # Add run_id to the saved config
    with open(output_dir / f"config_{run_id}.json", 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    # Run parsed training
    model, history = train_drm_model(
        db_path=args.db_path,
        system_type=args.system_type,
        output_dir=str(output_dir),
        run_id=run_id,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
        probing_size=args.probing_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_states=args.num_states,
        hidden_dim=args.hidden_dim,
        checkpoint_every=25,
        state_loss_weight=args.state_loss_weight,
        value_loss_weight=args.value_loss_weight,
        predictor_type=args.predictor_type,
        value_method=args.value_method,
        use_warmup=args.use_warmup,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        use_gumbel=args.use_gumbel,
        initial_temp=args.initial_temp,
        min_temp=args.min_temp,
        use_entropy_reg=args.use_entropy_reg,
        entropy_weight=args.entropy_weight,
        use_entropy_decay=args.use_entropy_decay,
        entropy_decay_proportion=args.entropy_decay_proportion,
        use_target_encoder=args.use_target_encoder,
        ema_decay=args.ema_decay,
        state_loss_type=args.state_loss_type,
        value_loss_type=args.value_loss_type,
        sort_states = args.sort_states,
        use_vicreg=args.use_vicreg,
        vicreg_weight=args.vicreg_weight,
        vicreg_lambda=args.vicreg_lambda,
        vicreg_mu=args.vicreg_mu,
        vicreg_nu=args.vicreg_nu,
        vicreg_variance_target=args.vicreg_variance_target,
        vicreg_invariance_schedule=args.vicreg_invariance_schedule,
        encoder_init_method=args.encoder_init_method,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler_type,
        weight_decay=args.weight_decay,
        restart_period=args.restart_period,
        restart_mult=args.restart_mult, 
    )