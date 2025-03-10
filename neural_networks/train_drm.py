import os
import sys
from pathlib import Path
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
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
from data_generation.models.tech_substitution import TechnologySubstitution, NumericalSolver


def plot_training_curves(history, save_path=None):
    """
    Plot training and validation loss curves
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot total loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot state loss
    plt.subplot(2, 2, 2)
    plt.plot(history['train_state_loss'], label='Train State Loss')
    plt.plot(history['val_state_loss'], label='Validation State Loss')
    plt.title('State Loss (KL Divergence)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot value loss
    plt.subplot(2, 2, 3)
    plt.plot(history['train_value_loss'], label='Train Value Loss')
    plt.plot(history['val_value_loss'], label='Validation Value Loss')
    plt.title('Value Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Add plot for diversity loss
    plt.subplot(2, 2, 4)
    plt.plot(history['train_div_loss'], label='Train Diversity Loss')
    plt.plot(history['val_div_loss'], label='Validation Diversity Loss')
    plt.title('Diversity Loss (KL Divergence)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved loss curves to {save_path}")
    
    plt.show()

def train_drm_model(db_path, 
                    output_dir="./neural_networks/output", 
                    val_size=2000,
                    test_size=2000, 
                    batch_size=64, 
                    epochs=100,
                    learning_rate=0.0001, 
                    early_stopping_patience=10,
                    num_states=4,
                    hidden_dim=128,
                    checkpoint_every=10,
                    initial_div_weight=0.5,
                    min_div_weight=0.05,
                    use_diversity_loss=False):
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
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    
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
    tech_sub_solver = NumericalSolver(tech_sub_model)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        db_path=db_path,
        tech_sub_solver=tech_sub_solver,
        batch_size=batch_size,
        val_size=val_size, test_size=test_size
    )
    
    # Initialize model
    model = DiscreteRepresentationsModel(
        obs_dim=2,
        control_dim=1,
        num_states=num_states,
        hidden_dim=hidden_dim
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
    
    # use loss function
    loss_fn = StableDRMLoss(
        state_loss_weight=1.0, 
        value_loss_weight=10.0,
        initial_diversity_weight=initial_div_weight,
        min_diversity_weight=min_div_weight,
        use_diversity_loss=use_diversity_loss 
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add gradient clipping
    max_grad_norm = 1.0
    
    # Prepare for training
    history = {
        "train_loss": [],
        "train_state_loss": [],
        "train_value_loss": [],
        "train_div_loss": [],
        "val_loss": [],
        "val_state_loss": [],
        "val_value_loss": [],
        "val_div_loss": []
    }
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    # Start training loop
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_state_loss = 0.0
        train_value_loss = 0.0
        train_div_loss = 0.0
        if loss_fn.use_diversity_loss:
            # Update diversity weight
            current_div_weight = loss_fn.update_diversity_weight(epoch, epochs)
            print(f"Epoch {epoch+1}/{epochs} - Diversity weight: {current_div_weight:.4f}")
        
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
                s_x, s_y, s_y_pred, v_pred = model(x, c, y, v_true)
                
                # Check for NaN values in model outputs
                if torch.isnan(s_y).any() or torch.isnan(s_y_pred).any() or torch.isnan(v_pred).any():
                    print(f"WARNING: NaN values in model outputs for batch {batch_idx}, skipping")
                    continue
                
                # Calculate loss
                total_loss, state_loss, value_loss, div_loss = loss_fn(s_y, s_y_pred, v_true, v_pred, s_x)
                
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
                
                # Print batch info occasionally
                if batch_idx % 50 == 0:
                    print(f"  Batch {batch_idx}: Loss={total_loss.item():.4f}")
                
                # Accumulate losses
                train_loss += total_loss.item()
                train_state_loss += state_loss.item()
                train_value_loss += value_loss.item()
                train_div_loss += div_loss.item()
                
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
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_state_loss = 0.0
        val_value_loss = 0.0
        val_div_loss = 0.0
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
                    s_x, s_y, s_y_pred, v_pred = model(x, c, y, v_true)
                    
                    # Check for NaN values in model outputs
                    if torch.isnan(s_y).any() or torch.isnan(s_y_pred).any() or torch.isnan(v_pred).any():
                        continue
                    
                    # Calculate loss
                    total_loss, state_loss, value_loss, div_loss = loss_fn(s_y, s_y_pred, v_true, v_pred, s_x)
                    
                    # Check if loss is NaN
                    if torch.isnan(total_loss):
                        continue
                    
                    # Accumulate losses
                    val_loss += total_loss.item()
                    val_state_loss += state_loss.item()
                    val_value_loss += value_loss.item()
                    val_div_loss += div_loss.item()
                    valid_batches += 1
                    
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue
        
        # Average validation losses
        if valid_batches > 0:  # Avoid division by zero
            val_loss /= valid_batches
            val_state_loss /= valid_batches
            val_value_loss /= valid_batches
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_state_loss"].append(train_state_loss)
        history["train_value_loss"].append(train_value_loss)
        history["train_div_loss"].append(train_div_loss)
        history["val_loss"].append(val_loss)
        history["val_state_loss"].append(val_state_loss)
        history["val_value_loss"].append(val_value_loss)
        history["val_div_loss"].append(val_div_loss)
        
        # Print progress
        if loss_fn.use_diversity_loss:
            print(f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} (State: {train_state_loss:.4f}, Value: {train_value_loss:.4f}, Div: {train_div_loss:.4f}) - "
                f"Val Loss: {val_loss:.4f} (State: {val_state_loss:.4f}, Value: {val_value_loss:.4f}, Div: {val_div_loss:.4f})")
        else:
            print(f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f} (State: {train_state_loss:.4f}, Value: {train_value_loss:.4f}) - "
                f"Val Loss: {val_loss:.4f} (State: {val_state_loss:.4f}, Value: {val_value_loss:.4f})")

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
                s_x, s_y, s_y_pred, v_pred = model(x, c, y, v_true)
                
                # Skip if model outputs have NaN values
                if torch.isnan(s_y).any() or torch.isnan(s_y_pred).any() or torch.isnan(v_pred).any():
                    continue
                
                # Calculate loss
                total_loss, state_loss, value_loss, div_loss = loss_fn(s_y, s_y_pred, v_true, v_pred)
                
                # Skip if loss is NaN
                if torch.isnan(total_loss):
                    continue
                
                # Store predictions and targets for analysis
                test_value_predictions.append(v_pred.cpu().numpy())
                test_value_targets.append(v_true.cpu().numpy())
                
                # Accumulate losses
                test_loss += total_loss.item() * len(x)
                test_state_loss += state_loss.item() * len(x)
                test_value_loss += value_loss.item() * len(x)
                test_div_loss += div_loss.item() * len(x)
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
        "test_r2_score": float(r2_score),
        "test_samples": test_samples
    }
    
    # Print test results
    print(f"Test Results - Loss: {test_loss:.4f} (State: {test_state_loss:.4f}, Div: {test_div_loss:.4f}, Value: {test_value_loss:.4f}), R²: {r2_score:.4f}")
    
    # Save test results
    test_results_path = os.path.join(output_dir, f"drm_test_results_{run_id}.json")
    with open(test_results_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"Saved test results to {test_results_path}")
    
    # Optional: Plot test predictions vs targets
    plt.figure(figsize=(10, 6))
    plt.scatter(test_value_targets, test_value_predictions, alpha=0.5)
    plt.plot([min(test_value_targets), max(test_value_targets)], 
             [min(test_value_targets), max(test_value_targets)], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Test Set Predictions vs Targets (R² = {r2_score:.4f})')
    plt.savefig(os.path.join(output_dir, f"test_predictions_{run_id}.png"))
    plt.close()
    
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
    
    return model, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Discrete Representations Model')
    parser.add_argument('db_path', type=str, help='Path to the SQLite database (can be relative to project root)')
    parser.add_argument('--output_dir', type=str, default='neural_networks/output', help='Directory to save outputs')
    
    parser.add_argument('--val_size', type=int, default=2000, help='Number of validation samples')
    parser.add_argument('--test_size', type=int, default=2000, help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--num_states', type=int, default=4, help='Number of discrete states')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')

    parser.add_argument('--use_diversity_loss', action='store_true', 
                        help='Whether to use diversity regularization')
    parser.add_argument('--no_diversity_loss', dest='use_diversity_loss', 
                        action='store_false', help='Disable diversity regularization')
    parser.set_defaults(use_diversity_loss=False)

    parser.add_argument('--initial_div_weight', type=float, default=1.0, help='Initial diversity loss weight')
    parser.add_argument('--min_div_weight', type=float, default=0.05, help='Minimum diversity loss weight')
    
    args = parser.parse_args()
    
    # Train the model
    model, history = train_drm_model(
        db_path=args.db_path,
        output_dir=args.output_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
        num_states=args.num_states,
        hidden_dim=args.hidden_dim,
        initial_div_weight=args.initial_div_weight,
        min_div_weight=args.min_div_weight
    )

# python -m neural_networks.train_drm datasets/results/tech_toy.db --num_states 4