import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math
import pandas as pd
import seaborn as sns

import torch
#from torchviz import make_dot
#from torchinfo import summary

import torch
from torch.utils.tensorboard import SummaryWriter
import os
import subprocess
from pathlib import Path

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

""" 
NOTE: Important, don't get confused with the layout of the grid. It is in a coordinate system. 
[0, 0] is bottom left, not like with typical numpy array top left! Same applies in higher dimensions.
"""
def visualize_state_space(model, output_path, transformations=None, device='cpu',
                         num_points=1000, num_states=None):
    """
    Visualize the state probabilities in z-space (transformed space) with x-space coordinate labels.
    
    Args:
        model: Trained DRM model
        output_path: Path to save the visualization
        transformations: List of transformation functions
        device: Device to run the model on
        num_points: Number of points in each dimension of the mesh
        num_states: Number of states in the model (if None, will be inferred)
    """
    if transformations is None:
        # Default to tangent transformation if none provided
        transformations = [
            tangent_transformation(1.0, 0.5),  # For x1 dimension
            tangent_transformation(1.0, 0.5)   # For x2 dimension
        ]
    
    # Unpack the transformation functions
    _, inverse_transforms, _ = zip(*transformations)
    
    # Z-space bounds (transformed space)
    z_bounds = [(0, 1), (0, 1)]
    
    # Create a grid of points in z-space
    z1_values = np.linspace(z_bounds[0][0], z_bounds[0][1], num_points)
    z2_values = np.linspace(z_bounds[1][0], z_bounds[1][1], num_points)
    
    z1_grid, z2_grid = np.meshgrid(z1_values, z2_values)
    z1_flat, z2_flat = z1_grid.flatten(), z2_grid.flatten()
    
    # Transform z-space coordinates to x-space for the input to the model
    x1_flat = np.array([inverse_transforms[0](z) for z in z1_flat])
    x2_flat = np.array([inverse_transforms[1](z) for z in z2_flat])
    
    # Create input tensor for model
    x_test = torch.tensor(np.column_stack((x1_flat, x2_flat)), dtype=torch.float32).to(device)
    
    # Get state probabilities
    model.to(device)
    model.eval()
    with torch.no_grad():
        state_probs = model.get_state_probs(x_test, training=False)
    
    # Infer number of states if not provided
    if num_states is None:
        num_states = state_probs.shape[1]
    
    # Define grid layout for the plots
    cols_per_row = min(2, num_states)
    rows = math.ceil(num_states / cols_per_row)
    
    # Create figure
    figsize = (12, 5 * rows)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=figsize, squeeze=False)
    
    # Generate tick positions for z-space
    num_ticks = 5
    z1_ticks = np.linspace(z_bounds[0][0], z_bounds[0][1], num_ticks)
    z2_ticks = np.linspace(z_bounds[1][0], z_bounds[1][1], num_ticks)
    
    # Format functions for ticks (z to x transformation)
    def format_x1_ticks(z, pos):
        x = inverse_transforms[0](z)
        if np.isinf(x) or x > 1000:
            return "∞"
        elif x < 0.1:
            return f"{x:.2e}"
        else:
            return f"{x:.1f}"
    
    def format_x2_ticks(z, pos):
        x = inverse_transforms[1](z)
        if np.isinf(x) or x > 1000:
            return "∞"
        elif x < 0.1:
            return f"{x:.2e}"
        else:
            return f"{x:.1f}"
    
    # Plot each state
    for state in range(num_states):
        row_idx = state // cols_per_row
        col_idx = state % cols_per_row
        ax = axes[row_idx, col_idx]
        
        probs = state_probs[:, state].cpu().detach().numpy().reshape(z1_grid.shape)
        im = ax.pcolormesh(z1_grid, z2_grid, probs, vmin=0, vmax=1, cmap='viridis')
        ax.set_title(f'State {state+1} Probability')
        
        # Set ticks
        ax.set_xticks(z1_ticks)
        ax.set_yticks(z2_ticks)
        
        # Set custom tick formatters
        ax.xaxis.set_major_formatter(FuncFormatter(format_x1_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(format_x2_ticks))
        
        # Set labels
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('Probability')
    
    # Hide unused subplots
    for i in range(num_states, rows * cols_per_row):
        row_idx = i // cols_per_row
        col_idx = i % cols_per_row
        axes[row_idx, col_idx].set_visible(False)
    
    # Adjust layout
    plt.subplots_adjust(wspace=0.6, hspace=0.5)
    
    # Save the figure
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved state visualization to {output_path}")
    
    plt.close(fig)
    return fig, axes

def analyze_state_transitions(model, 
                              transformations,
                              control_values=[0.5, 1.0],
                              z_bounds=[(0, 1), (0, 1)],
                              num_points=1000,
                              assignment_method='argmax',
                              device='cpu'):
    """
    Analyze state transitions for different control values.
    
    Args:
        model: Trained DRM model
        transformations: List of transformation functions
        control_values: List of control values to analyze
        z_bounds: Bounds for z-space
        num_points: Number of points in each dimension
        assignment_method: 'argmax' or 'soft'
        device: Device for computation
    
    Returns:
        Dictionary of transition matrices for each control value
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Unpack transformations
    _, inverse_transforms, _ = zip(*transformations)
    
    # Create grid in z-space avoiding the z=1 boundary point
    epsilon = 1e-5  # Small adjustment to avoid exactly z=1
    z1 = np.linspace(z_bounds[0][0], z_bounds[0][1] - epsilon, num_points)
    z2 = np.linspace(z_bounds[1][0], z_bounds[1][1] - epsilon, num_points)
    Z1, Z2 = np.meshgrid(z1, z2)
    
    # Flatten grid
    z1_flat = Z1.flatten()
    z2_flat = Z2.flatten()
    
    # Transform to x-space (original space)
    x1_flat = np.array([inverse_transforms[0](z) for z in z1_flat])
    x2_flat = np.array([inverse_transforms[1](z) for z in z2_flat])
    
    # Prepare input for model
    x_batch = torch.tensor(np.column_stack((x1_flat, x2_flat)), dtype=torch.float32).to(device)
    
    # Get current state probabilities 
    with torch.no_grad():
        current_state_probs = model.get_state_probs(x_batch, training=False)
        
        # Check for NaN and stabilize if needed
        if torch.isnan(current_state_probs).any():
            print("WARNING: NaN values detected in current state probabilities!")
            current_state_probs = torch.nan_to_num(current_state_probs, nan=0.0)
            # Renormalize
            row_sums = current_state_probs.sum(dim=1, keepdim=True)
            current_state_probs = current_state_probs / torch.clamp(row_sums, min=1e-10)
    
    num_states = current_state_probs.shape[1]
    results = {}
    
    # Debug info
    print(f"Min/max state probabilities: {torch.min(current_state_probs).item()}, {torch.max(current_state_probs).item()}")
    
    # Clamp values for stability
    current_state_probs = torch.clamp(current_state_probs, min=1e-10, max=1.0)
    
    # Determine state assignments based on chosen method
    if assignment_method == 'argmax':
        # Assign each point to its most likely state
        state_assignments = torch.argmax(current_state_probs, dim=1).cpu().numpy()
    elif assignment_method == 'soft':
        # For soft, we'll handle differently
        state_assignments = None
    else:
        raise ValueError(f"Unknown assignment method: {assignment_method}")
    
    # Debug assignments
    if state_assignments is not None:
        unique_assignments = np.unique(state_assignments)
        print(f"Unique state assignments: {unique_assignments}")
        assignment_counts = np.bincount(state_assignments, minlength=num_states)
        print(f"Assignment counts: {assignment_counts}")
    
    for control_value in control_values:
        # Prepare control tensor
        control_batch = torch.full((x_batch.shape[0], 1), control_value, dtype=torch.float32).to(device)
        
        # Predict next state with error checking
        with torch.no_grad():
            # FIXED: Pass state and control separately to predict_next_state
            next_state_probs = model.predict_next_state(current_state_probs, control_batch)
            
            # Check for NaN
            if torch.isnan(next_state_probs).any():
                print("WARNING: NaN detected in next_state_probs!")
                next_state_probs = torch.nan_to_num(next_state_probs, nan=0.0)
                # Renormalize
                row_sums = next_state_probs.sum(dim=1, keepdim=True)
                next_state_probs = next_state_probs / torch.clamp(row_sums, min=1e-10)
        
        # Move to CPU for analysis
        next_state_probs = next_state_probs.cpu()
        current_state_probs_cpu = current_state_probs.cpu()
        
        # Initialize transition matrix
        transition_matrix = np.zeros((num_states, num_states))
        
        try:
            if assignment_method == 'soft':
                # Soft assignment - use outer products weighted by current state probability
                for i in range(x_batch.shape[0]):
                    current_probs = current_state_probs_cpu[i].numpy()
                    next_probs = next_state_probs[i].numpy()
                    
                    # For each possible current state, distribute its probability to next states
                    for current_state in range(num_states):
                        transition_matrix[current_state] += current_probs[current_state] * next_probs
                
                # Normalize rows
                row_sums = transition_matrix.sum(axis=1, keepdims=True)
                transition_matrix = np.divide(transition_matrix, row_sums, 
                                           out=np.zeros_like(transition_matrix), 
                                           where=row_sums!=0)
            else:
                # Hard assignment - group points by their assigned state
                for state in range(num_states):
                    # Get indices of points assigned to this state
                    state_indices = np.where(state_assignments == state)[0]
                    
                    if len(state_indices) > 0:
                        # Get the average predicted next state distribution for these points
                        avg_next_state = torch.mean(next_state_probs[state_indices], dim=0).numpy()
                        transition_matrix[state] = avg_next_state
                    else:
                        print(f"Warning: No points assigned to state {state+1}")
                        # Fill with uniform distribution for stability
                        transition_matrix[state] = np.ones(num_states) / num_states
            
            # Ensure rows sum to 1
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix = np.divide(transition_matrix, row_sums, 
                                       out=np.zeros_like(transition_matrix), 
                                       where=row_sums!=0)
            
        except Exception as e:
            print(f"Error computing transition matrix: {e}")
            # Provide a fallback uniform transition matrix
            transition_matrix = np.ones((num_states, num_states)) / num_states
        
        results[control_value] = transition_matrix
    
    return results

def visualize_transition_matrices(transition_matrices, control_values, output_path=None):

    """
    Visualize transition matrices as heatmaps.
    
    Args:
        transition_matrices: Dictionary of transition matrices
        control_values: List of control values used
        output_path: Path to save the visualization
    
    Returns:
        Figure object
    """
    num_states = transition_matrices[control_values[0]].shape[0]
    
    # Create heatmaps
    fig, axes = plt.subplots(1, len(control_values), figsize=(len(control_values)*6, 5))
    if len(control_values) == 1:
        axes = [axes]
    
    for i, control in enumerate(control_values):
        ax = axes[i]
        matrix = transition_matrices[control]
        
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", 
                  vmin=0, vmax=1, ax=ax, cbar=True)
        
        ax.set_title(f"Transition Probabilities (c={control})")
        ax.set_xlabel("Next State")
        ax.set_ylabel("Current State")
        ax.set_xticklabels([f"S{j+1}" for j in range(num_states)])
        ax.set_yticklabels([f"S{j+1}" for j in range(num_states)])
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved transition matrix visualization to {output_path}")
        plt.close(fig)
    else:
        plt.show()
    
    # Print transition matrices as tables
    for control in control_values:
        matrix = transition_matrices[control]
        df = pd.DataFrame(matrix, 
                       index=[f"State {i+1}" for i in range(num_states)],
                       columns=[f"State {i+1}" for i in range(num_states)])
        
        print(f"\nTransition Probabilities for c={control}:")
        print(df.round(3))
    
    return fig

def visualize_model_architecture(model, output_path, input_shape=None, detailed=True):
    """
    Visualize the architecture of a PyTorch model and save it to a file.
    
    Args:
        model: The PyTorch model to visualize
        output_path: Path to save the visualization image
        input_shape: Tuple with shape of input for torchinfo (e.g. (batch_size, obs_dim))
                     Default is None, in which case a dummy input is created based on model parameters
        detailed: Whether to include detailed parameter counts (True) or just structure (False)
    
    Returns:
        Path to the saved visualization
    """
    import os
    
    # Try to use torchviz first for graph visualization
    try:
 
        # Create dummy inputs for the model if not provided
        if input_shape is None:
            # Extract dimensions from model
            obs_dim = model.encoder[0].in_features
            control_dim = 1  # Default, can be refined based on model inspection
            
            # Create dummy inputs
            x = torch.randn(1, obs_dim)
            c = torch.randn(1, control_dim)
            y = torch.randn(1, obs_dim)
            v_true = torch.randn(1, 1)  # Assume scalar value for simplicity
        else:
            # Use provided shape
            batch_size, obs_dim = input_shape
            control_dim = 1  # Default
            x = torch.randn(batch_size, obs_dim)
            c = torch.randn(batch_size, control_dim)
            y = torch.randn(batch_size, obs_dim)
            v_true = torch.randn(batch_size, 1)
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        x, c, y, v_true = x.to(device), c.to(device), y.to(device), v_true.to(device)
        
        # Forward pass to build computation graph
        s_x, s_y, s_y_pred, v_pred = model(x, c, y, v_true)
        
        # Create the dot graph
        dot = make_dot(v_pred, params=dict(model.named_parameters()))
        
        # Set graph attributes for better readability
        dot.attr('graph', rankdir='TB', ratio='fill', size='7.5,10')
        dot.attr('node', shape='box', style='filled', fillcolor='lightyellow')
        
        # Save the graph
        graph_path = output_path.replace('.png', '_graph.png')
        dot.render(graph_path.replace('.png', ''), format='png', cleanup=True)
        print(f"Saved model graph to {graph_path}")
        
    except ImportError:
        print("torchviz not found. Install with 'pip install torchviz'")
    except Exception as e:
        print(f"Error generating graph visualization: {e}")
    
    # Also use torchinfo for a detailed text summary
    try:
        # Create figure to hold the summary text
        fig, ax = plt.subplots(figsize=(10, 14))
        
        # Hide axes
        ax.axis('off')
        
        # Generate model summary
        if input_shape is None:
            # Use reasonable defaults
            batch_size = 1
            obs_dim = model.encoder[0].in_features
            control_dim = 1
            s = summary(model, 
                      input_size=[(batch_size, obs_dim), (batch_size, control_dim), 
                                 (batch_size, obs_dim), (batch_size, 1)],
                      depth=5 if detailed else 3,
                      col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"] if detailed else ["input_size", "output_size"],
                      verbose=0)
        else:
            # Use provided shape
            batch_size, obs_dim = input_shape
            control_dim = 1
            s = summary(model, 
                      input_size=[(batch_size, obs_dim), (batch_size, control_dim), 
                                 (batch_size, obs_dim), (batch_size, 1)],
                      depth=5 if detailed else 3,
                      col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"] if detailed else ["input_size", "output_size"],
                      verbose=0)
        
        # Convert summary to string and add as text to figure
        summary_str = str(s)
        ax.text(0.05, 0.95, summary_str, fontsize=9, verticalalignment='top', 
               family='monospace', wrap=True)
        
        # Add title with basic model info
        title = f"Discrete Representations Model\n"
        title += f"States: {model.num_states}, Hidden Dim: {model.encoder[2].out_features}\n"
        title += f"Predictor Type: {model.predictor.__class__.__name__}"
        ax.text(0.5, 0.98, title, fontsize=14, horizontalalignment='center', verticalalignment='top')
        
        # Save the figure
        summary_path = output_path.replace('.png', '_summary.png')
        plt.savefig(summary_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved model summary to {summary_path}")
        
        # If graph visualization failed, use the summary as the main output
        if not os.path.exists(graph_path):
            import shutil
            shutil.copy(summary_path, output_path)
        
    except ImportError:
        print("torchinfo not found. Install with 'pip install torchinfo'")
    except Exception as e:
        print(f"Error generating summary visualization: {e}")
        
    return output_path



def visualize_model_with_tensorboard(model, output_dir, run_id=None):
    """
    Visualize the model architecture using TensorBoard and export as an image.
    
    Args:
        model: The PyTorch model to visualize
        output_dir: Directory to save the visualization
        run_id: Unique identifier for this run (default: None)
        
    Returns:
        Path to the saved visualization image
    """
    
    # Create a unique subdirectory for this visualization
    if run_id is None:
        from datetime import datetime
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_dir = os.path.join(output_dir, f"tb_logs_{run_id}")
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    # Extract dimensions from model to create dummy inputs
    obs_dim = model.encoder[0].in_features
    control_dim = 1  # Default, can be adjusted based on inspection
    
    # Create dummy inputs
    device = next(model.parameters()).device
    x = torch.randn(1, obs_dim, device=device)
    c = torch.randn(1, control_dim, device=device)
    y = torch.randn(1, obs_dim, device=device)
    v_true = torch.randn(1, 1, device=device)
    
    # Add the model graph to TensorBoard
    try:
        # Try to capture the complete model with multiple outputs
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x, c, y, v_true):
                s_x, s_y, s_y_pred, v_pred = self.model(x, c, y, v_true)
                # Return all important outputs as a dictionary
                return {
                    's_x': s_x,
                    's_y': s_y,
                    's_y_pred': s_y_pred,
                    'v_pred': v_pred
                }
        
        wrapped_model = ModelWrapper(model)
        writer.add_graph(wrapped_model, (x, c, y, v_true))
    except Exception as e:
        print(f"Error adding full model graph: {e}")
        # Fall back to simpler approach with just the main prediction path
        try:
            def forward_simple(x, c):
                s_x = model.get_state_probs(x)
                s_y_pred = model.predict_next_state(s_x, c)
                v_pred = model.compute_value(s_y_pred)
                return v_pred
            
            writer.add_graph(model, (x, c, y, v_true))
        except Exception as e2:
            print(f"Error adding simplified model graph: {e2}")
    
    # Add model summary as text
    model_info = f"""
    # Discrete Representations Model Architecture
    
    ## Overview
    - Number of states: {model.num_states}
    - Hidden dimension: {model.encoder[2].out_features}
    - Predictor type: {model.predictor.__class__.__name__}
    - Observation dim: {obs_dim}
    - Control dim: {control_dim}
    
    ## Component Structure
    
    ### Encoder
    ```
    {model.encoder}
    ```
    
    ### Predictor
    ```
    {model.predictor}
    ```
    
    ### Value Network
    ```
    {model.value_net}
    ```
    
    ## Parameter Count
    Total params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}
    """
    
    writer.add_text("Model Architecture", model_info)
    
    # Add model hyperparameters
    hparams = {
        "num_states": model.num_states,
        "hidden_dim": model.encoder[2].out_features,
        "predictor_type": model.predictor.__class__.__name__,
        "obs_dim": obs_dim,
        "control_dim": control_dim
    }
    
    # Add hyperparameters to TensorBoard
    writer.add_hparams(hparams, {"hparam/placeholder": 0})
    
    # Close the writer to flush all events to disk
    writer.close()
    
    # Path to image output
    image_path = os.path.join(output_dir, f"model_architecture_{run_id}.png")
    
    # Add a note about viewing in TensorBoard
    print(f"Model graph added to TensorBoard: {log_dir}")
    print(f"To view, run: tensorboard --logdir={log_dir}")
    
    # Save a text file with the command to view the graph
    with open(os.path.join(output_dir, f"tensorboard_command_{run_id}.txt"), 'w') as f:
        f.write(f"tensorboard --logdir={log_dir}")
    
    # Attempt to export the graph to an image using TensorBoard's export feature
    # Note: This won't work perfectly in all environments but provides a starting point
    try:
        # Create basic HTML pointing to TensorBoard
        html_path = os.path.join(output_dir, f"model_architecture_{run_id}.html")
        with open(html_path, 'w') as f:
            f.write(f"""
            <html>
            <head>
                <meta http-equiv="refresh" content="0;URL=http://localhost:6006/#graphs">
            </head>
            <body>
                <p>Please run: <code>tensorboard --logdir={log_dir}</code> and open <a href="http://localhost:6006/#graphs">http://localhost:6006/#graphs</a></p>
            </body>
            </html>
            """)
        
        print(f"Created HTML redirect at {html_path}")
        
        # Use tensorboard's built-in export features if TensorBoard is running
        # Note: This is a placeholder - integrating with a running TensorBoard
        # instance programmatically requires more complex implementation
        
    except Exception as e:
        print(f"Unable to export graph image: {e}")
    
    return log_dir

def get_model_summary(model):
    """
    Generate a text summary of the model architecture.
    
    Args:
        model: The PyTorch model to summarize
        
    Returns:
        str: Summary text of the model
    """
    try:
        from torchinfo import summary
        
        # Extract dimensions
        obs_dim = model.encoder[0].in_features
        control_dim = 1  # Default
        
        # Generate summary using torchinfo
        s = summary(model, 
                  input_size=[(1, obs_dim), (1, control_dim), 
                             (1, obs_dim), (1, 1)],
                  depth=5,
                  verbose=0,
                  col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
        
        return str(s)
    except ImportError:
        # Fall back to basic string representation if torchinfo isn't available
        return f"""
        Model: {model.__class__.__name__}
        
        Encoder: {model.encoder}
        
        Predictor: {model.predictor}
        
        Value Network: {model.value_net}
        
        Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}
        """
    except Exception as e:
        return f"Error generating model summary: {e}"       