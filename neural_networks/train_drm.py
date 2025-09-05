import os
import sys
from pathlib import Path
import time
import json
import yaml
import shutil
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
import pickle

import pandas as pd
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
    plot_training_curves, analyze_mdp_from_model, visualize_mdp,
    create_state_viz_from_data, analyze_state_assignment_evolution,
    create_gif_from_frames, create_state_evolution_analysis,
    create_png_from_frame_data, create_gif_from_data_frames,
    plot_softmax_rank_evolution
)

from neural_networks.utils import *

def train_drm_model(config_path, multi_run=False):
    """
    Train DRM model - now with config support and batch mode.
    
    New parameters:
        config: Configuration dictionary or path to config file
        dataset_id: Dataset ID for database path formatting  
        multi_run: If True, minimizes outputs for batch processing
        
    """
    # Load config
    config = load_and_validate_config(config_path)
    verbose = not multi_run
    
    # Extract output directory (already complete)
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config file
    config_copy_path = output_dir / "config.yaml"
    shutil.copy2(config_path, config_copy_path)

    
    # === META PARAMETERS ===
    db_path = config['db_path']
    seed = config['seed']
    run_id = config['run_id']
    
    # === DATA PARAMETERS ===
    val_size = config['val_size']
    test_size = config['test_size']
    
    # === MODEL PARAMETERS ===
    system_type = config['system_type']
    num_states = config['num_states']
    hidden_dim = config['hidden_dim']
    predictor_type = config['predictor_type']
    value_method = config['value_method']
    use_target_encoder = config['use_target_encoder']
    ema_decay = config['ema_decay']
    use_gumbel = config['use_gumbel']
    initial_temp = config['initial_temp']
    min_temp = config['min_temp']
    encoder_init_method = config['encoder_init_method']
    
    # === TRAINING PARAMETERS ===
    epochs = config['epochs']
    batch_size = config['batch_size']
    checkpoint_every = config['checkpoint_every']
    optimizer_type = config['optimizer_type']
    lr = config['lr']
    weight_decay = config['weight_decay']
    scheduler_type = config['scheduler_type']
    min_lr = config['min_lr']
    use_warmup = config['use_warmup']
    warmup_epochs = config['warmup_epochs']
    restart_period = config['restart_period']
    restart_mult = config['restart_mult']
    
    # === LOSS PARAMETERS ===
    state_loss_weight = config['state_loss_weight']
    value_loss_weight = config['value_loss_weight']
    state_loss_type = config['state_loss_type']
    value_loss_type = config['value_loss_type']
    use_entropy_reg = config['use_entropy_reg']
    entropy_weight = config['entropy_weight']
    use_entropy_decay = config['use_entropy_decay']
    entropy_decay_proportion = config['entropy_decay_proportion']
    
    
    # Set seeds
    set_all_seeds(seed)
    
    # System configuration
    system_config = get_system_config(SystemType[system_type.upper()])
    transformation = get_transformation(SystemType[system_type.upper()])

    if value_method is None:
        value_method = system_config['default_value_method']
    
    # Value loss type - Legacy code
    if value_loss_type is None:
        value_loss_type = system_config['default_value_loss'][value_method]
        if not multi_run:
            print(f"Using default value loss type: {value_loss_type}")
    
    # Setup paths
    db_path = Path(db_path)
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    
    # Make output_dir absolute
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize time and run ID
    start_time = time.time()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        system_type=system_type,
        db_path=db_path,
        value_method=value_method,
        batch_size=batch_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        num_workers=1,
        verboise=verbose
    )
    
    # Get dimensions from first batch
    for x, c, y, v_true in train_loader:
        obs_dim = x.shape[1]
        control_dim = c.shape[1]
        value_dim = v_true.shape[1]
        break
    
    if not multi_run:
        print(f"Detected dimensions: obs_dim={obs_dim}, control_dim={control_dim}, value_dim={value_dim}")
    
    # For plotting the saddle nodes, extract config
    points_config = None
    if system_type == 'saddle_system':
        saddle_config = get_saddle_configuration(db_path, verbose=verbose)
        if saddle_config:
            points_config = {
                'points': saddle_config['saddle_points'],
                'angles_degrees': saddle_config['angles_degrees']
            }
    
    # Checkpint config
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
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not multi_run:
        print(f"Using device: {device}")
    
    # Create model (keeping original)
    model = DiscreteRepresentationsModel(
        obs_dim=obs_dim,
        control_dim=control_dim,
        value_dim=value_dim,
        num_states=num_states,
        hidden_dim=hidden_dim,
        predictor_type=predictor_type,
        use_target_encoder=use_target_encoder,
        ema_decay=ema_decay,
        use_gumbel=use_gumbel,
        initial_temp=initial_temp,
        min_temp=min_temp,
        value_activation=value_activation
    ).to(device)

    # NOTE: Running without gumbel will also use initial_temp value, might want to change this
    current_temp=initial_temp
    
    # Initialize model weights
    initialize_model_weights(model, encoder_init_method=encoder_init_method)
    
    # Create optimizer (keeping original function)
    optimizer = create_optimizer_with_bias_exclusion(
        model, optimizer_type, lr, weight_decay)
    
    # Setup warmup if enabled
    if use_warmup and not multi_run:
        print(f"Using warmup for first {warmup_epochs} epochs (starting LR: {lr * 0.1:.2e})")
    
    # Create scheduler (keeping original function)
    lr_scheduler = create_scheduler(
        optimizer, scheduler_type, epochs, warmup_epochs,
        use_warmup, min_lr, restart_period, restart_mult)
    
    # Gradient clipping
    max_grad_norm = 1.0
    
    # Create loss function (keeping original)
    loss_fn = StableDRMLoss(
        state_loss_weight=state_loss_weight,
        value_loss_weight=value_loss_weight,
        value_method=value_method,
        use_entropy_reg=use_entropy_reg,
        entropy_weight=entropy_weight,
        use_entropy_decay=use_entropy_decay,
        state_loss_type=state_loss_type,
        value_loss_type=value_loss_type,
        entropy_decay_proportion=entropy_decay_proportion
    )
    
    # Initialize history (keeping all original fields)
    history = {
        "train_loss": [],
        "train_state_loss": [],
        "train_value_loss": [],
        "train_entropy_loss": [],
        "train_batch_entropy": [],
        "train_individual_entropy": [],
        "train_entropy_weight": [],
        "train_softmax_temp": [],
        "val_loss": [],
        "val_state_loss": [],
        "val_value_loss": [],
        "val_entropy_loss": [],
        "val_batch_entropy": [],
        "val_individual_entropy": [],
        "state_metrics": [],
        "softmax_rank_metrics": []
    }
    
    # State metrics collection (only if not multi_run)
    collect_every_n_epochs = 2
    collect_initial = True
    visualization_frames = []
    prev_dominant_states = None

    bounds = get_visualization_bounds(SystemType[system_type.upper()])
    
    # INITIAL STATE COLLECTION (EPOCH 0) - only if not multi_run
    if collect_initial and not multi_run:
        print("Collecting initial state assignments (epoch 0)...")
        
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
        collect_softmax_rank_metrics(model, val_loader, device, epoch=0, history=history, verbose=verbose)
        
        # Save lightweight data frame
        data_frame_path = save_state_data_frame(
            grid_points, state_probs, epoch=0, num_states=num_states,
            output_dir=output_dir, run_id=run_id, bounds=bounds
        )
        visualization_frames.append(data_frame_path)
        prev_dominant_states = dominant_states
    
    # TRAINING LOOP
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_state_loss = 0.0
        train_value_loss = 0.0
        train_entropy_loss = 0.0
        train_batch_entropy = 0.0
        train_individual_entropy = 0.0
        
        # Update temperature if using gumbel
        if use_gumbel:
            current_temp = model.update_temperature(epoch, epochs)
        
        history["train_softmax_temp"].append(current_temp)
        
        # Update entropy weight
        if loss_fn.use_entropy_reg:
            current_entropy_weight = loss_fn.update_entropy_weight(epoch, epochs)
            history["train_entropy_weight"].append(current_entropy_weight)
        # Handle warmup if enabled
        if use_warmup and epoch < warmup_epochs:
            # Linearly increase learning rate during warmup period
            progress = (epoch) / warmup_epochs
            new_lr = lr * (0.1 + 0.9 * progress)  # 10% to 100% of base LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            if not multi_run:
                print(f"Warmup epoch {epoch+1}/{warmup_epochs}, LR set to {new_lr:.2e}")
        
        for batch_idx, (x, c, y, v_true) in enumerate(train_loader):
            # Move to device
            x, c, y, v_true = x.to(device), c.to(device), y.to(device), v_true.to(device)
            
            # Check for NaN
            if (torch.isnan(x).any() or torch.isnan(c).any() or 
                torch.isnan(y).any() or torch.isnan(v_true).any()):
                if not multi_run:
                    print(f"WARNING: NaN values in batch {batch_idx}, skipping")
                continue
            
            try:
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass
                s_x, s_y, s_y_pred, v_pred_for_all_states = model(
                    x, c, y, v_true, training=True
                )
                
                # Calculate loss
                (total_loss, state_loss, value_loss, entropy_loss, batch_entropy,
                 individual_entropy) = loss_fn(
                    s_y, s_y_pred, v_true, v_pred_for_all_states, s_x,
                    epoch=epoch, max_epochs=epochs
                )
                
                # Check for NaN
                if torch.isnan(total_loss):
                    print(f"WARNING: NaN loss in batch {batch_idx}, skipping")
                    continue
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                
                # Update target encoder if used
                if use_target_encoder:
                    model.update_target_encoder()
                
                # Accumulate losses
                train_loss += total_loss.item()
                train_state_loss += state_loss.item()
                train_value_loss += value_loss.item()
                train_entropy_loss += entropy_loss.item()
                train_batch_entropy += batch_entropy.item()
                train_individual_entropy += individual_entropy.item()
                
            except Exception as e:
                if not multi_run:
                    print(f"Error in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                continue
        
        # Average training losses
        num_batches = len(train_loader)
        if num_batches > 0:
            train_loss /= num_batches
            train_state_loss /= num_batches
            train_value_loss /= num_batches
            train_entropy_loss /= num_batches
            train_batch_entropy /= num_batches
            train_individual_entropy /= num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_state_loss = 0.0
        val_value_loss = 0.0
        val_entropy_loss = 0.0
        val_batch_entropy = 0.0
        val_individual_entropy = 0.0
        valid_batches = 0
        
        with torch.no_grad():
            for x, c, y, v_true in val_loader:
                x, c, y, v_true = x.to(device), c.to(device), y.to(device), v_true.to(device)
                
                if (torch.isnan(x).any() or torch.isnan(c).any() or 
                    torch.isnan(y).any() or torch.isnan(v_true).any()):
                    continue
                
                try:
                    # Forward pass
                    s_x, s_y, s_y_pred, v_pred_for_all_states = model(
                        x, c, y, v_true, training=False
                    )
                    
                    # Calculate loss
                    (total_loss, state_loss, value_loss, entropy_loss, batch_entropy,
                     individual_entropy) = loss_fn(
                        s_y, s_y_pred, v_true, v_pred_for_all_states, s_x,
                        epoch=epoch, max_epochs=epochs
                    )
                    
                    if torch.isnan(total_loss):
                        continue
                    
                    # Accumulate
                    val_loss += total_loss.item()
                    val_state_loss += state_loss.item()
                    val_value_loss += value_loss.item()
                    val_entropy_loss += entropy_loss.item()
                    val_batch_entropy += batch_entropy.item()
                    val_individual_entropy += individual_entropy.item()
                    valid_batches += 1
                    
                except Exception as e:
                    if not multi_run:
                        print(f"Error in validation: {e}")
                    continue
        
        # Average validation losses
        if valid_batches > 0:
            val_loss /= valid_batches
            val_state_loss /= valid_batches
            val_value_loss /= valid_batches
            val_entropy_loss /= valid_batches
            val_batch_entropy /= valid_batches
            val_individual_entropy /= valid_batches
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_state_loss"].append(train_state_loss)
        history["train_value_loss"].append(train_value_loss)
        history["train_entropy_loss"].append(train_entropy_loss)
        history["train_batch_entropy"].append(train_batch_entropy)
        history["train_individual_entropy"].append(train_individual_entropy)
        history["val_loss"].append(val_loss)
        history["val_state_loss"].append(val_state_loss)
        history["val_value_loss"].append(val_value_loss)
        history["val_entropy_loss"].append(val_entropy_loss)
        history["val_batch_entropy"].append(val_batch_entropy)
        history["val_individual_entropy"].append(val_individual_entropy)
        
        # Print progress (only if not multi_run)
        if not multi_run:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f} (State: {train_state_loss:.4f}, Value: {train_value_loss:.4f})")
            print(f"  Batch Entropy: {train_batch_entropy:.4f}, Individual Entropy: {train_individual_entropy:.4f}")
            print(f"Val Loss: {val_loss:.4f} (State: {val_state_loss:.4f}, Value: {val_value_loss:.4f})")
            print(f"  Batch Entropy: {val_batch_entropy:.4f}, Individual Entropy: {val_individual_entropy:.4f}")
        
        # INTERMEDIATE LAYER PROBING - every 10 epochs
        if (epoch + 1) % 10 == 0 and system_type == 'saddle_system':
            if not multi_run:
                print(f"\n--- Intermediate Layer Probing at Epoch {epoch+1} ---")
            intermediate_results = run_layer_probing(model, val_loader, device, system_type, db_path)
            if not multi_run:
                print(f"Validation Probing - Discrete Accuracy: {intermediate_results['discrete_accuracy']:.4f}")
                print(f"Validation Probing - Unweighted Accuracy: {intermediate_results['discrete_accuracy_unweighted']:.4f}")
            
            # Store in history
            if 'intermediate_probing' not in history:
                history['intermediate_probing'] = []
            history['intermediate_probing'].append({
                'epoch': epoch + 1,
                'discrete_accuracy': intermediate_results['discrete_accuracy'],
                'discrete_accuracy_unweighted': intermediate_results['discrete_accuracy_unweighted']
            })
        
        # Visualizations at specific epochs (only if not multi_run)
        if not multi_run and (epoch == 10 or epoch == 50):
            state_vis_path = os.path.join(output_dir, f"states_after{epoch}_{run_id}.png")
            visualize_state_space(
                model=model,
                output_path=state_vis_path,
                transformations=[transformation, transformation],
                device=device,
                num_states=num_states,
                system_type=system_type,
                points=points_config['points'] if points_config else None,
                angles_degrees=points_config['angles_degrees'] if points_config else None,
                bounds=bounds
            )
        
        # Collect softmax rank metrics
        if (epoch + 1) % collect_every_n_epochs == 0:
            collect_softmax_rank_metrics(model, val_loader, device, epoch=epoch+1, history=history)
        
        # Collect state metrics periodically (only if not multi_run)
        if not multi_run and ((epoch + 1) % collect_every_n_epochs == 0):
            
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
            
            # Save lightweight data frame
            data_frame_path = save_state_data_frame(
                grid_points, state_probs, epoch=epoch+1, num_states=num_states,
                output_dir=output_dir, run_id=run_id, bounds=bounds
            )
            visualization_frames.append(data_frame_path)
            # Update for next epoch's stability calculation
            prev_dominant_states = dominant_states
        elif multi_run and ((epoch + 1) % collect_every_n_epochs == 0):
            # Basic assignment metrics only (no visualization)
            metrics, dominant_states, _, _ = extract_and_calculate_metrics(
                model=model, device=device, num_states=num_states,
                system_type=system_type, bounds=bounds, epoch=epoch+1,
                prev_dominant_states=prev_dominant_states
            )
            history["state_metrics"].append(metrics)
            prev_dominant_states = dominant_states
        
        # Apply scheduler
        if lr_scheduler is not None and (not use_warmup or epoch >= warmup_epochs):
            lr_scheduler.step()
        
        # Save checkpoint (only if not multi_run)
        if not multi_run and ((epoch + 1) % checkpoint_every == 0 or (epoch+1) == 10):
            checkpoint_path = os.path.join(output_dir, f"drm_checkpoint_epoch{epoch+1}_{run_id}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': checkpoint_config
            }, checkpoint_path)
            if not multi_run:
                print(f"Saved checkpoint to {checkpoint_path}")
    
    # POST-TRAINING: Create visualizations (only if not multi_run)
    if not multi_run and history["state_metrics"]:
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
            
            # Clean up frames
            for data_frame_path in visualization_frames:
                try:
                    os.remove(data_frame_path)
                except:
                    pass
    
    # FINAL TEST EVALUATION (keeping exact original structure)
    if not multi_run:
        print("Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_state_loss = 0.0
    test_value_loss = 0.0
    test_entropy_loss = 0.0
    test_batch_entropy = 0.0
    test_individual_entropy = 0.0
    test_samples = 0
    
    with torch.no_grad():
        for x, c, y, v_true in test_loader:
            x, c, y, v_true = x.to(device), c.to(device), y.to(device), v_true.to(device)
            
            if (torch.isnan(x).any() or torch.isnan(c).any() or 
                torch.isnan(y).any() or torch.isnan(v_true).any()):
                continue
            
            try:
                # Forward pass
                s_x, s_y, s_y_pred, v_pred_for_all_states = model(x, c, y, v_true, training=False)
                
                if torch.isnan(s_y).any() or torch.isnan(s_y_pred).any() or torch.isnan(v_pred_for_all_states).any():
                    continue
                
                # Calculate loss:
                (total_loss, state_loss, value_loss, entropy_loss, batch_entropy, individual_entropy) = loss_fn(
                    s_y, s_y_pred, v_true, v_pred_for_all_states, s_x,
                    epoch=epoch, max_epochs=epochs)
                
                if torch.isnan(total_loss):
                    continue
                
                # Accumulate
                test_loss += total_loss.item() * len(x)
                test_state_loss += state_loss.item() * len(x)
                test_value_loss += value_loss.item() * len(x)
                test_entropy_loss += entropy_loss.item() * len(x)
                test_batch_entropy += batch_entropy.item() * len(x)
                test_individual_entropy += individual_entropy.item() * len(x)
                test_samples += len(x)
                
            except Exception as e:
                if not multi_run:
                    print(f"Error in test evaluation: {e}")
                continue
    
    # Average test losses
    if test_samples > 0:
        test_loss /= test_samples
        test_state_loss /= test_samples
        test_value_loss /= test_samples
        test_entropy_loss /= test_samples
        test_batch_entropy /= test_samples
        test_individual_entropy /= test_samples
    
    # FINAL LAYER PROBING on test set
    probing_results = None
    if system_type == 'saddle_system':
        if not multi_run:
            print("\n--- Final Layer Probing on Test Set ---")
        probing_results = run_layer_probing(model, test_loader, device, system_type, db_path)
    
    # Create test metrics dictionary
    test_metrics = {
        "test_loss": float(test_loss),
        "test_state_loss": float(test_state_loss),
        "test_value_loss": float(test_value_loss),
        "test_entropy_loss": float(test_entropy_loss),
        "test_batch_entropy": float(test_batch_entropy),
        "test_individual_entropy": float(test_individual_entropy),
        "test_samples": test_samples,
        "prob_discrete_accuracy": probing_results['discrete_accuracy'] if probing_results else None,
        "prob_discrete_accuracy_unweighted": probing_results['discrete_accuracy_unweighted'] if probing_results else None
    }
    
    # Print test results (only if not multi_run)
    if not multi_run:
        print(f"Test Results:")
        print(f"  Loss: {test_loss:.4f} (State: {test_state_loss:.4f}, Value: {test_value_loss:.4f})")
        print(f"  Entropy Loss: {test_entropy_loss:.4f}")
        print(f"  Batch Entropy: {test_batch_entropy:.4f}, Individual Entropy: {test_individual_entropy:.4f}")
        if probing_results:
            print(f"  Linear Probing - Discrete Accuracy: {probing_results['discrete_accuracy']:.4f}")
            print(f"  Linear Probing - Unweighted Accuracy: {probing_results['discrete_accuracy_unweighted']:.4f}")
    
    # Save test metrics
    history['test_metrics'] = test_metrics
    
    # Save history
    history_path = os.path.join(output_dir, f"history_{run_id}.json")
    with open(history_path, 'w') as f:
        safe_json_dump(history, f, indent=4)
    
    # SOFTMAX RANK POST-PROCESSING AND PLOTTING:
    if 'softmax_rank_metrics' in history and history['softmax_rank_metrics']:
        # Post-process to add globally normalized singular values
        if not multi_run:
            print("Post-processing: Computing global normalization for singular values...")
        add_global_normalized_singular_values(history)
        
        # Re-save history with globally normalized values
        with open(history_path, 'w') as f:
            safe_json_dump(history, f)
        print(f"Updated training history with global normalization")
        
        if not multi_run:
            # Create plots
            rank_plot_path = os.path.join(output_dir, f"softmax_rank_evolution_{run_id}.png")
            plot_softmax_rank_evolution(history, rank_plot_path)
            
            # Print final summary
            final_metrics = history['softmax_rank_metrics'][-1]
            print("\n" + "="*60)
            print("SOFTMAX RANK SUMMARY")
            print("="*60)
            print(f"Final hidden rank: {final_metrics['hidden_rank']}")
            print(f"Final logit rank: {final_metrics['logit_rank']}")
            print(f"Final σ₂/σ₁ ratio: {final_metrics.get('logit_sv_ratio_2nd_to_1st', 0):.4f}")
            print(f"Hidden ||A₃||_F: {final_metrics['hidden_frobenius_norm']:.3f}")
            print(f"Logit ||M₄||_F: {final_metrics['logit_frobenius_norm']:.3f}")
            print("="*60)

    # Save state assignments (for batch aggregation)
    with torch.no_grad():
        all_states = []
        all_obs = []
        for x, _, _, _ in val_loader:
            x = x.to(device)
            state_probs = model.encoder(x)
            states = state_probs.argmax(dim=1)
            all_states.append(states.cpu().numpy())
            all_obs.append(x.cpu().numpy())
        
        state_assignments = {
            'states': np.concatenate(all_states),
            'observations': np.concatenate(all_obs),
            'num_states': num_states
        }
        
        assignments_path = os.path.join(output_dir, f"state_assignments_{run_id}.pkl")
        with open(assignments_path, 'wb') as f:
            pickle.dump(state_assignments, f)
    
    # Create final state assignment visualization (even in batch mode)
    final_state_path = os.path.join(output_dir, f"final_state_assignment_{run_id}.png")
    visualize_state_space(
        model=model,
        output_path=final_state_path,
        transformations=[transformation, transformation],
        device=device,
        num_states=num_states,
        system_type=system_type,
        points=points_config['points'] if points_config else None,
        angles_degrees=points_config['angles_degrees'] if points_config else None,
        bounds=bounds
    )
    
    # Create other visualizations (only if not multi_run)
    if not multi_run:
        # Training curves
        plot_path = os.path.join(output_dir, f"training_curves_{run_id}.png")
        plot_training_curves(history, plot_path)
        
        # Transition matrices
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
        transition_matrices = analyze_discrete_state_transitions(
            model=model,
            control_values=control_values,
            device=device,
            system_type=system_type
        )
        transitions_vis_path = os.path.join(output_dir, f"transitions_{run_id}.png")
        visualize_transition_matrices(transition_matrices, control_values, transitions_vis_path)
    
    # Calculate training time
    training_time = time.time() - start_time
    if not multi_run:
        print(f"Training completed in {training_time:.2f} seconds")
    
    return model, history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help='Path to YAML config file')
    parser.add_argument('--multi_run', action='store_true')
    
    args = parser.parse_args()
    model, history = train_drm_model(args.config_path, args.multi_run)