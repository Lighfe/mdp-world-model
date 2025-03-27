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

# Define project root at the module level
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"Added {PROJECT_ROOT} to Python path")

# NOTE: absolute imports from project root
# Import application-specific modules
from neural_networks.drm_dataset import create_data_loaders
from neural_networks.drm_loss import StableDRMLoss
from neural_networks.drm import DiscreteRepresentationsModel
from data_generation.models.tech_substitution import TechnologySubstitution, TechSubNumericalSolver
from data_generation.simulations.grid import tangent_transformation
from neural_networks.drm_viz import (
    visualize_state_space, analyze_state_transitions, 
    visualize_transition_matrices, visualize_model_architecture, plot_training_curves
)


def train_drm_model(db_path, 
                    output_dir="./neural_networks/output",
                    run_id=None,
                    val_size=2000,
                    test_size=2000, 
                    batch_size=64, 
                    epochs=100,
                    learning_rate=1e-4, 
                    early_stopping_patience=10,
                    num_states=4,
                    hidden_dim=128,
                    checkpoint_every=10,
                    state_loss_weight=1.0, 
                    value_loss_weight=3.0,
                    predictor_type='control_gate',
                    value_method=None,
                    use_lr_scheduler=False,
                    scheduler_type='cosine',
                    use_warmup=False,
                    warmup_epochs=5,
                    min_lr=1e-5,
                    use_gumbel=False,
                    initial_temp=1.0,
                    min_temp=0.1,
                    use_entropy_reg=False,
                    entropy_weight=5.0,
                    use_target_encoder=False,
                    ema_decay=0.996,
                    use_state_diversity=False,
                    diversity_weight=1.0
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
        early_stopping_patience: Number of epochs to wait before early stopping
        num_states: Number of discrete states in the model
        hidden_dim: Hidden dimension size in the model
        checkpoint_every: Save checkpoint every N epochs
    
    Returns:
        model: Trained model
        history: Dictionary containing training history
    """

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
    
    # Fix Python path to import TechnologySubstitution
    # Get the current file's directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Find the project root (assuming it's named "mdp-world-model")
    project_root = current_dir
    while project_root.name != "mdp-world-model" and project_root.parent != project_root:
        project_root = project_root.parent
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added {project_root} to Python path")
    
    # Import the TechnologySubstitution model and NumericalSolver
    tech_sub_model = TechnologySubstitution()
    tech_sub_solver = TechSubNumericalSolver(tech_sub_model)

    transformation = tangent_transformation(3.0, 0.5)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        db_path=db_path,
        tech_sub_solver=tech_sub_solver,
        value_method=value_method,
        batch_size=batch_size,
        val_size=val_size, test_size=test_size
    )

        # Get a batch from the train loader to determine dimensions
    for x, c, y, v_true in train_loader:
        obs_dim = x.shape[1]  # Dimension of observation
        control_dim = c.shape[1]  # Dimension of control
        value_dim = v_true.shape[1]  # Dimension of value
        break  # We just need one batch to determine dimensions
    
    # Initialize model
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
        ema_decay=ema_decay
    )
    
    # Initialize model weights properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    model.apply(init_weights)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # init loss function
    loss_fn = StableDRMLoss(
        state_loss_weight=state_loss_weight, 
        value_loss_weight=value_loss_weight,
        use_state_diversity=use_state_diversity,
        diversity_weight=diversity_weight,
        use_entropy_reg=use_entropy_reg,
        entropy_weight=entropy_weight
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
        # Setup learning rate scheduling if enabled
    if use_warmup:
        # If warmup is enabled, start with a lower learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * 0.1  # Start at 10% of base learning rate
        print(f"Using warmup for first {warmup_epochs} epochs (starting LR: {learning_rate * 0.1:.2e})")
    
    # Create scheduler if enabled
    lr_scheduler = None
    if use_lr_scheduler:
        if scheduler_type == 'plateau':
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5,
                patience=5,
                min_lr=min_lr
            )
            print(f"Using ReduceLROnPlateau scheduler (min_lr: {min_lr:.2e})")
        else:  # cosine
            T_max = epochs - warmup_epochs if use_warmup else epochs
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, T_max),  # Ensure T_max is at least 1
                eta_min=min_lr
            )
            print(f"Using CosineAnnealingLR scheduler (T_max: {T_max}, min_lr: {min_lr:.2e})")

    # Add gradient clipping
    max_grad_norm = 1.0
    
    # Prepare for training
    history = {
        "train_loss": [],
        "train_state_loss": [],
        "train_value_loss": [],
        "train_div_loss": [],
        "train_entropy_loss": [],
        "val_loss": [],
        "val_state_loss": [],
        "val_value_loss": [],
        "val_div_loss": [],
        "val_entropy_loss": []
    }
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    # Start training loop
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Update Gumbel temperature
        if use_gumbel:
            current_temp = model.update_temperature(epoch, epochs)
            print(f"Epoch {epoch+1}/{epochs} - Gumbel temperature: {current_temp:.4f}")

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
        train_div_loss = 0.0
        train_entropy_loss = 0.0


        if loss_fn.use_entropy_decay:
            # Update entropy weight
            entropy_weight = loss_fn.update_entropy_weight(epoch, epochs)
            print(f"Epoch {epoch+1}/{epochs} - Entropy weight: {entropy_weight:.4f}")
        
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
                s_x, s_y, s_y_pred, v_pred_for_all_states = model(x, c, y, v_true, training=True)
                
                # Check for NaN values in model outputs
                if torch.isnan(s_y).any() or torch.isnan(s_y_pred).any() or torch.isnan(v_pred_for_all_states).any():
                    print(f"WARNING: NaN values in model outputs for batch {batch_idx}, skipping")
                    continue
                
                # Calculate loss
                total_loss, state_loss, value_loss, div_loss, entropy_loss = loss_fn(
                        s_y, s_y_pred, v_true, v_pred_for_all_states, s_x)
                
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
                train_div_loss += div_loss.item()
                train_entropy_loss += entropy_loss.item()
                
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
            train_div_loss /= num_batches
            train_entropy_loss /= num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_state_loss = 0.0
        val_value_loss = 0.0
        val_div_loss = 0.0
        val_entropy_loss = 0.0
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
                    s_x, s_y, s_y_pred, v_pred_for_all_states = model(x, c, y, v_true, training=False)
                    
                    # Check for NaN values in model outputs
                    if torch.isnan(s_y).any() or torch.isnan(s_y_pred).any() or torch.isnan(v_pred_for_all_states).any():
                        continue
                    
                    # Calculate loss
                    total_loss, state_loss, value_loss, div_loss, entropy_loss = loss_fn(
                            s_y, s_y_pred, v_true, v_pred_for_all_states, s_x)
                    
                    # Check if loss is NaN
                    if torch.isnan(total_loss):
                        continue
                    
                    # Accumulate losses
                    val_loss += total_loss.item()
                    val_state_loss += state_loss.item()
                    val_value_loss += value_loss.item()
                    val_div_loss += div_loss.item()
                    val_entropy_loss += entropy_loss.item()
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
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_state_loss"].append(train_state_loss)
        history["train_value_loss"].append(train_value_loss)
        history["train_div_loss"].append(train_div_loss)
        history["train_entropy_loss"].append(train_entropy_loss)
        history["val_loss"].append(val_loss)
        history["val_state_loss"].append(val_state_loss)
        history["val_value_loss"].append(val_value_loss)
        history["val_div_loss"].append(val_div_loss)
        history["val_entropy_loss"].append(val_entropy_loss)
        
        # Print progress
        if loss_fn.use_entropy_reg:
            print(f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} (State: {train_state_loss:.4f}, Value: {train_value_loss:.4f}, "
                f"Entropy: {train_entropy_loss:.4f}) - "
                f"Val Loss: {val_loss:.4f} (State: {val_state_loss:.4f}, Value: {val_value_loss:.4f}, "
                f"Entropy: {val_entropy_loss:.4f})")
        # lazy
        if loss_fn.use_diversity_loss:
            print(f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} (State: {train_state_loss:.4f}, Value: {train_value_loss:.4f}, Div: {train_div_loss:.4f}) - "
                f"Val Loss: {val_loss:.4f} (State: {val_state_loss:.4f}, Value: {val_value_loss:.4f}, Div: {val_div_loss:.4f})")
        else:
            print(f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} (State: {train_state_loss:.4f}, Value: {train_value_loss:.4f}) - "
                f"Val Loss: {val_loss:.4f} (State: {val_state_loss:.4f}, Value: {val_value_loss:.4f})")

        if epoch == 5:

            # Visualize the state space
            state_vis_path = os.path.join(output_dir, f"states_after5_{run_id}.png")
            visualize_state_space(
                model=model,
                output_path=state_vis_path,
                transformations=[
                    transformation,  # For x1 dimension
                    transformation   # For x2 dimension
                ],
                device=device,
                num_states=num_states
            )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(output_dir, f"drm_best_{run_id}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'config': {
                    'obs_dim': 2,
                    'control_dim': 1,
                    'num_states': num_states,
                    'hidden_dim': hidden_dim
                }
            }, best_model_path)
            print(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1

        # Apply learning rate scheduler after validation
        if use_lr_scheduler and (not use_warmup or epoch >= warmup_epochs):
            if scheduler_type == 'plateau':
                old_lr = optimizer.param_groups[0]['lr']
                lr_scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    print(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
            else:  # cosine
                        lr_scheduler.step()
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.2e}")
            
        # Save checkpoint every N epochs
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(output_dir, f"drm_checkpoint_epoch{epoch+1}_{run_id}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'obs_dim': 2,
                    'control_dim': 1,
                    'num_states': num_states,
                    'hidden_dim': hidden_dim
                }
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Final evaluation on test set
    print("Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_state_loss = 0.0
    test_value_loss = 0.0
    test_div_loss = 0.0
    test_entropy_loss = 0.0
    test_samples = 0
    test_value_predictions = []
    test_value_targets = []
    
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
                s_x, s_y, s_y_pred, v_pred_for_all_states = model(x, c, y, v_true, training=False)
                
                # Skip if model outputs have NaN values
                if torch.isnan(s_y).any() or torch.isnan(s_y_pred).any() or torch.isnan(v_pred_for_all_states).any():
                    continue
                
                # Calculate loss
                total_loss, state_loss, value_loss, div_loss, entropy_loss = loss_fn(
                        s_y, s_y_pred, v_true, v_pred_for_all_states, s_x)
                
                # Skip if loss is NaN
                if torch.isnan(total_loss):
                    continue
                
                # Calculate expected value prediction based on predicted state distribution
                expected_v_pred = (s_y_pred @ v_pred_for_all_states).detach()  # Matrix multiply: [batch, states] @ [states, value_dim]

                # Store predictions and targets for analysis
                test_value_predictions.append(expected_v_pred.cpu().numpy())
                test_value_targets.append(v_true.cpu().numpy())
                
                # Accumulate losses
                test_loss += total_loss.item() * len(x)
                test_state_loss += state_loss.item() * len(x)
                test_value_loss += value_loss.item() * len(x)
                test_div_loss += div_loss.item() * len(x)
                test_entropy_loss += entropy_loss.item() * len(x)
                test_samples += len(x)
                
            except Exception as e:
                print(f"Error in test evaluation: {e}")
                continue
    
    # Calculate average test losses
    if test_samples > 0:
        test_loss /= test_samples
        test_state_loss /= test_samples
        test_value_loss /= test_samples
        test_div_loss /= test_samples
        test_entropy_loss /= test_samples
    
    # Calculate additional metrics if needed
    test_value_predictions = np.concatenate(test_value_predictions) if test_value_predictions else np.array([])
    test_value_targets = np.concatenate(test_value_targets) if test_value_targets else np.array([])
    
    # Calculate R^2 for value predictions if we have enough data
    r2_score = 0
    if len(test_value_predictions) > 0:
        from sklearn.metrics import r2_score as sklearn_r2_score
        r2_score = sklearn_r2_score(test_value_targets, test_value_predictions)
    
    # Create test metrics dictionary
    test_metrics = {
        "test_loss": float(test_loss),
        "test_state_loss": float(test_state_loss),
        "test_div_loss": float(test_div_loss),
        "test_value_loss": float(test_value_loss),
        "test_entropy_loss": float(test_entropy_loss),
        "test_r2_score": float(r2_score),
        "test_samples": test_samples
    }
    
    # Print test results
    print(f"Test Results - Loss: {test_loss:.4f} (State: {test_state_loss:.4f}, Div: {test_div_loss:.4f}, Value: {test_value_loss:.4f}), Entropy: {test_entropy_loss:.4f}, R²: {r2_score:.4f}")
    
    # Save test results
    test_results_path = os.path.join(output_dir, f"drm_test_results_{run_id}.json")
    with open(test_results_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"Saved test results to {test_results_path}")
    
    # Print test errors
    print(f"\nTest Set Evaluation Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test State Loss: {test_state_loss:.4f}")
    print(f"  Test Value Loss: {test_value_loss:.4f}")
    if test_div_loss > 0:
        print(f"  Test Diversity Loss: {test_div_loss:.4f}")

    # For multi-dimensional outputs, report per-dimension errors if applicable
    if test_value_predictions.ndim > 1 and test_value_predictions.shape[1] > 1:
        print(f"\nPer-dimension value metrics:")
        for dim in range(test_value_predictions.shape[1]):
            dim_mse = np.mean((test_value_targets[:, dim] - test_value_predictions[:, dim])**2)
            dim_mae = np.mean(np.abs(test_value_targets[:, dim] - test_value_predictions[:, dim]))
            print(f"  Dimension {dim}: MSE={dim_mse:.4f}, MAE={dim_mae:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, f"drm_final_{run_id}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': {
            'obs_dim': 2,
            'control_dim': 1,
            'num_states': num_states,
            'hidden_dim': hidden_dim
        }
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, f"drm_history_{run_id}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    print(f"Saved training history to {history_path}")
    
    # Plot training curves
    plot_training_curves(history, os.path.join(output_dir, f"training_curves_{run_id}.png"))
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")


    # Visualize Modl Architecture
    arch_vis_path = os.path.join(output_dir, f"model_architecture_{run_id}")
    visualize_model_architecture(model, arch_vis_path)

    # Visualize the state space
    state_vis_path = os.path.join(output_dir, f"states_{run_id}.png")
    visualize_state_space(
        model=model,
        output_path=state_vis_path,
        transformations=[
            transformation,  # For x1 dimension
            transformation   # For x2 dimension
        ],
        device=device,
        num_states=num_states
    )

    # Analyze and visualize state transitions with argmax assignment
    control_values = [0.5, 1.0]  # Example control values, adjust as needed
    argmax_transitions = analyze_state_transitions(
        model=model,
        transformations=[
            transformation,  # For x1 dimension
            transformation   # For x2 dimension
        ],
        control_values=control_values,
        assignment_method='argmax',
        device=device
    )
    argmax_vis_path = os.path.join(output_dir, f"argmax_transitions_{run_id}.png")
    visualize_transition_matrices(argmax_transitions, control_values, argmax_vis_path)

    # 3. Analyze and visualize state transitions with soft assignment
    soft_transitions = analyze_state_transitions(
        model=model,
        transformations=[
            transformation,  # For x1 dimension
            transformation   # For x2 dimension
        ],
        control_values=control_values,
        assignment_method='soft',
        device=device
    )
    soft_vis_path = os.path.join(output_dir, f"soft_transitions_{run_id}.png")
    visualize_transition_matrices(soft_transitions, control_values, soft_vis_path)

    return model, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Discrete Representations Model')
    parser.add_argument('db_path', type=str, help='Path to the SQLite database (can be relative to project root)')
    parser.add_argument('--output_dir', type=str, default='neural_networks/output', help='Directory to save outputs')
    parser.add_argument('--run_id', type=str, default=None, 
                    help='Custom run ID for this training run (default: timestamp)')
    parser.add_argument('--val_size', type=int, default=2000, help='Number of validation samples')
    parser.add_argument('--test_size', type=int, default=2000, help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--num_states', type=int, default=4, help='Number of discrete states')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--state_loss_weight', type=float, default=1.0, help='Stateloss weight')
    parser.add_argument('--value_loss_weight', type=float, default=3.0, help='Value loss weight')

    parser.add_argument('--predictor_type', type=str, default='control_gate', 
                        choices=['standard', 'control_gate', 'bilinear'],
                        help='Type of predictor to use (standard, control_gate or bilinear)')
    parser.add_argument('--value_method', type=str, default='None', help='Which value function should be used')

    parser.add_argument('--use_lr_scheduler', action='store_true', 
                        help='Use learning rate scheduler')
    parser.add_argument('--scheduler_type', type=str, choices=['plateau', 'cosine'], 
                        default='cosine', help='Type of learning rate scheduler to use')
    parser.add_argument('--use_warmup', action='store_true', 
                        help='Use learning rate warmup')
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                        help='Number of epochs for warmup')
    parser.add_argument('--min_lr', type=float, default=1e-5, 
                        help='Minimum learning rate')
    
    parser.add_argument('--use_gumbel', action='store_true', 
                        help='Use Gumbel softmax for state encoding')
    parser.add_argument('--initial_temp', type=float, default=1.0,
                        help='Initial temperature for Gumbel softmax')
    parser.add_argument('--min_temp', type=float, default=0.1,
                        help='Minimum temperature for Gumbel softmax')
    
    parser.add_argument('--use_entropy_reg', action='store_true', 
                    help='Use entropy regularization to prevent state collapse')
    parser.add_argument('--entropy_weight', type=float, default=5.0,
                        help='Weight for entropy regularization loss')
    
    parser.add_argument('--use_target_encoder', action='store_true',
                    help='Use target encoder with EMA updates')
    parser.add_argument('--ema_decay', type=float, default=0.996,
                    help='EMA decay rate for target encoder (higher = slower updates)')
    
    parser.add_argument('--use_state_diversity', action='store_true',
                    help='Whether to use correlation-based state diversity regularization')
    parser.add_argument('--no_state_diversity', dest='use_state_diversity',
                        action='store_false', help='Disable state diversity regularization')
    parser.set_defaults(use_state_diversity=False)
    parser.add_argument('--diversity_weight', type=float, default=1.0,
                        help='Weight for state diversity regularization')
    
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
        output_dir=str(output_dir),
        run_id=run_id,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
        num_states=args.num_states,
        hidden_dim=args.hidden_dim,
        checkpoint_every=10,
        state_loss_weight=args.state_loss_weight,
        value_loss_weight=args.value_loss_weight,
        predictor_type=args.predictor_type,
        value_method=args.value_method,
        use_lr_scheduler=args.use_lr_scheduler,
        scheduler_type=args.scheduler_type,
        use_warmup=args.use_warmup,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        use_gumbel=args.use_gumbel,
        initial_temp=args.initial_temp,
        min_temp=args.min_temp,
        use_entropy_reg=args.use_entropy_reg,
        entropy_weight=args.entropy_weight,
        use_target_encoder=args.use_target_encoder,
        ema_decay=args.ema_decay,
        use_state_diversity=args.use_state_diversity,
        diversity_weight=args.diversity_weight
    )

# python -m neural_networks.train_drm datasets/results/tech_toy.db --num_states 4