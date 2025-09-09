import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math
import pandas as pd

import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import graphviz
import torch.nn as nn
import re
from collections import defaultdict

import tempfile
import shutil
import imageio
import pickle
import io



# Define project root at the module level
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"Added {PROJECT_ROOT} to Python path")

# NOTE: absolute imports from project root
# Import application-specific modules
from neural_networks.drm_dataset import create_data_loaders
from neural_networks.drm_loss import StableDRMLoss
from neural_networks.drm import DiscreteRepresentationsModel, BilinearPredictor, StandardPredictor
from data_generation.models.tech_substitution import TechnologySubstitution, TechSubNumericalSolver

from neural_networks.system_registry import get_transformation, SystemType
from data_generation.simulations.grid import tangent_transformation

""" 
NOTE: Important, don't get confused with the layout of the grid. It is in a coordinate system. 
[0, 0] is bottom left, not like with typical numpy array top left! Same applies in higher dimensions.
"""

def plot_training_curves(history, save_path=None, state_loss_type=None):
    """Plot training and validation loss curves"""

    # Paul Tol's muted color scheme for colorblind accessibility
    tol_blue = '#332288'    # Primary blue for training
    tol_yellow = '#DDCC77'  # Yellow for validation
    tol_red = '#CC6677'     # Red for entropy weight

    plt.figure(figsize=(15, 10))
    
    # Plot total loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], color=tol_blue, label='Train Loss')
    plt.plot(history['val_loss'], color=tol_yellow, label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot state loss
    plt.subplot(2, 2, 2)
    plt.plot(history['train_state_loss'], color=tol_blue, label='Train State Loss')
    plt.plot(history['val_state_loss'], color=tol_yellow, label='Validation State Loss')
    if state_loss_type is None:
        plt.title('State Loss')
    else:
        plt.title(f'State Loss ({state_loss_type})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot value loss
    plt.subplot(2, 2, 3)
    plt.plot(history['train_value_loss'], color=tol_blue, label='Train Value Loss')
    plt.plot(history['val_value_loss'], color=tol_yellow, label='Validation Value Loss')
    plt.title('Value Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot batch entropy and entropy weight with dual y-axes in fourth panel
    plt.subplot(2, 2, 4)
    
    # Create first y-axis for batch entropy
    ax1 = plt.gca()

    # Plot batch entropy if it exists
    if 'train_batch_entropy' in history and len(history['train_batch_entropy']) > 0:
        ax1.plot(history['train_batch_entropy'], label='Train Batch Entropy')
        if 'val_batch_entropy' in history and len(history['val_batch_entropy']) > 0:
            ax1.plot(history['val_batch_entropy'], label='Val Batch Entropy')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Batch Entropy (Higher = More Uniform)')  # Remove color='b'
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for entropy weight
    ax2 = ax1.twinx()
    # Plot entropy weight if it exists
    if 'train_entropy_weight' in history and len(history['train_entropy_weight']) > 0:
        ax2.plot(history['train_entropy_weight'], color=tol_red, label='Entropy Weight')
    ax2.set_ylabel('Entropy Loss Weight')
    
    # Combine legends properly
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Batch Entropy & Entropy Weight')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved loss curves to {save_path}")

def visualize_state_space(model, output_path=None, transformations=None, device='cpu',
                         num_points=1000, num_states=None, soft=False, system_type=None,
                         points=None, angles_degrees=None, bounds=None):
   """
   Visualize the state probabilities in either transformed z-space or original x-space.
   
   Args:
       model: Trained DRM model.
       output_path: Optional path to save the visualization. If None, the plot will be displayed.
       transformations: List of transformation functions.
       device: Device to run the model on.
       num_points: Number of points in each dimension of the mesh.
       num_states: Number of states in the model (if None, will be inferred).
       soft: Whether to use soft assignment for state probabilities.
       system_type: Type of system (for default transformations).
       points: List of point coordinates (e.g., [[1.0, 0.0], [2.0, 1.0]])
       angles_degrees: List of angles in degrees corresponding to points (e.g., [90, 45])
       bounds: If None, plot in z-space (0,1) with x-space tick labels. 
              If provided as [(x1_min, x1_max), (x2_min, x2_max)], plot directly in x-space.
   """
   if transformations is None:
       if system_type is None:
           raise ValueError("Either transformations or system_type must be provided")
       # Get default transformation for the system
       transformation = get_transformation(SystemType[system_type.upper()])
       transformations = [transformation, transformation]  # Same for both dimensions
   
   # Unpack the transformation functions
   forward_transforms, inverse_transforms, _ = zip(*transformations)
   
   if bounds is None:
       # Original behavior: plot in z-space (0,1) with x-space tick labels
       plot_bounds = [(0, 1), (0, 1)]
       use_x_space = False
       
       # Create a grid of points in z-space
       z1_values = np.linspace(plot_bounds[0][0], plot_bounds[0][1], num_points)
       z2_values = np.linspace(plot_bounds[1][0], plot_bounds[1][1], num_points)
       
       z1_grid, z2_grid = np.meshgrid(z1_values, z2_values)
       z1_flat, z2_flat = z1_grid.flatten(), z2_grid.flatten()
       
       # Transform z-space coordinates to x-space for the model input
       x1_flat = np.array([inverse_transforms[0](z) for z in z1_flat])
       x2_flat = np.array([inverse_transforms[1](z) for z in z2_flat])
       
       # Transform points from x-space to z-space for plotting
       points_plot = None
       if points is not None:
           points_plot = []
           for point in points:
               z1 = forward_transforms[0](point[0])
               z2 = forward_transforms[1](point[1])
               points_plot.append([z1, z2])
   else:
       # New behavior: plot directly in x-space
       plot_bounds = bounds
       use_x_space = True
       
       # Create a grid of points directly in x-space
       x1_values = np.linspace(plot_bounds[0][0], plot_bounds[0][1], num_points)
       x2_values = np.linspace(plot_bounds[1][0], plot_bounds[1][1], num_points)
       
       x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
       x1_flat, x2_flat = x1_grid.flatten(), x2_grid.flatten()
       
       # Points are already in x-space, no transformation needed for plotting
       points_plot = points
   
   # Create input tensor for model (always in x-space)
   x_test = torch.tensor(np.column_stack((x1_flat, x2_flat)), dtype=torch.float32).to(device)
   
   # Get state probabilities
   model.to(device)
   model.eval()
   with torch.no_grad():
       state_probs = model.get_state_probs(x_test, training=False, soft=soft)
   
   # Infer number of states if not provided
   if num_states is None:
       num_states = state_probs.shape[1]
   
   # Define grid layout for the plot
   cols_per_row = min(4, num_states)
   rows = (num_states + cols_per_row - 1) // cols_per_row
   
   # Create subplots
   fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 5, rows * 4))
   if rows == 1 and cols_per_row == 1:
       axes = np.array([[axes]])
   elif rows == 1 or cols_per_row == 1:
       axes = axes.reshape(rows, cols_per_row)
   
   # Plot each state
   for state in range(num_states):
       row_idx = state // cols_per_row
       col_idx = state % cols_per_row
       ax = axes[row_idx, col_idx]
       
       # Reshape probabilities for this state
       state_prob_grid = state_probs[:, state].cpu().numpy().reshape(num_points, num_points)
       
       # Create the plot (using extent to set coordinate system)
       im = ax.imshow(state_prob_grid, 
                     extent=[plot_bounds[0][0], plot_bounds[0][1], plot_bounds[1][0], plot_bounds[1][1]],
                     origin='lower', 
                     cmap='viridis', 
                     vmin=0, vmax=1)
       
       # Set ticks and labels based on coordinate system
       if use_x_space:
           # Direct x-space plotting: use actual coordinate values
           num_ticks = 6
           ax.set_xticks(np.linspace(plot_bounds[0][0], plot_bounds[0][1], num_ticks))
           ax.set_yticks(np.linspace(plot_bounds[1][0], plot_bounds[1][1], num_ticks))
       else:
           # Z-space plotting with x-space labels: transform tick values
           ax.set_xticks(np.linspace(0, 1, 6))
           ax.set_yticks(np.linspace(0, 1, 6))
           
           # Create custom tick formatter functions for x-space labels
           def format_x1_ticks(z_val, pos):
               if z_val < 0 or z_val > 1:
                   return ''
               x_val = inverse_transforms[0](z_val)
               return f'{x_val:.1f}'
           
           def format_x2_ticks(z_val, pos):
               if z_val < 0 or z_val > 1:
                   return ''
               x_val = inverse_transforms[1](z_val)
               return f'{x_val:.1f}'
           
           # Apply formatters
           ax.xaxis.set_major_formatter(FuncFormatter(format_x1_ticks))
           ax.yaxis.set_major_formatter(FuncFormatter(format_x2_ticks))
       
       # Overlay points and angles if provided
       if points_plot is not None and angles_degrees is not None:
           for i, (point_plot, angle_deg) in enumerate(zip(points_plot, angles_degrees)):
               # Draw white point
               ax.plot(point_plot[0], point_plot[1], 'wx', markersize=6, markeredgecolor='black', markeredgewidth=0.5)
               
               # Draw angle line from edge to edge
               angle_rad = np.radians(angle_deg)
               px, py = point_plot[0], point_plot[1]

               # Line equation: x = px + t*cos(θ), y = py + t*sin(θ)
               # Find t values where line hits boundaries
               cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

               t_values = []
               # Left/right boundaries
               if abs(cos_a) > 1e-10:  # Avoid division by zero
                   t_values.extend([(plot_bounds[0][0]-px) / cos_a, (plot_bounds[0][1]-px) / cos_a])
               # Top/bottom boundaries
               if abs(sin_a) > 1e-10:  # Avoid division by zero
                   t_values.extend([(plot_bounds[1][0]-py) / sin_a, (plot_bounds[1][1]-py) / sin_a])

               # Get min/max t to find intersection points
               t_min, t_max = min(t_values), max(t_values)

               x_start = px + t_min * cos_a
               x_end = px + t_max * cos_a
               y_start = py + t_min * sin_a
               y_end = py + t_max * sin_a

               ax.plot([x_start, x_end], [y_start, y_end], 'w--', linewidth=0.5)
       
       # Set title
       ax.set_title(f'State {state + 1}', fontsize=14, fontweight='bold')
       
       # Set labels
       ax.set_xlabel('x1')
       ax.set_ylabel('x2')
       
       # Add grid
       ax.grid(True, linestyle='--', alpha=0.6)
       
       # Add colorbar
       divider = make_axes_locatable(ax)
       cax = divider.append_axes("right", size="5%", pad=0.1)
       cbar = fig.colorbar(im, cax=cax)
       cbar.set_label('State Assignment Strength')
   
   # Hide unused subplots
   for i in range(num_states, rows * cols_per_row):
       row_idx = i // cols_per_row
       col_idx = i % cols_per_row
       axes[row_idx, col_idx].set_visible(False)
   
   # Adjust layout
   plt.subplots_adjust(wspace=0.6, hspace=0.5)
   
   # Save the figure if an output path is provided; otherwise, display it
   if output_path:
       plt.savefig(output_path, dpi=100, bbox_inches='tight')
       print(f"Saved state visualization to {output_path}")
   else:
       plt.show()
   
   plt.close(fig)
   return fig, axes

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


def create_gif_from_data_frames(data_frame_paths, output_path, gif_duration=250):
    """
    Create GIF from saved data frame paths instead of PNG paths.
    """
    try:
        
        gif_path = f"{output_path}_animation.gif"
        
        with imageio.get_writer(gif_path, mode='I', duration=gif_duration) as writer:
            for data_path in data_frame_paths:
                # Load data frame
                with open(data_path, 'rb') as f:
                    frame_data = pickle.load(f)
                
                # Create visualization from data
                png_data = create_png_from_frame_data(frame_data)
                writer.append_data(png_data)
        
        print(f"Created GIF with {len(data_frame_paths)} frames: {gif_path}")
        return gif_path
        
    except ImportError:
        print("Warning: imageio not available for GIF creation")
        return None
    except Exception as e:
        print(f"Warning: Failed to create GIF: {e}")
        return None

def create_png_from_frame_data(frame_data):
    """
    Create PNG image data from frame data dict.
    """
    grid_points = frame_data['grid_points']
    state_probs = frame_data['state_probs']
    epoch = frame_data['epoch']
    num_states = frame_data['num_states']
    bounds = frame_data['bounds']
    grid_size = frame_data['grid_size']
    
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
    
    # Convert to image data instead of saving to file
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    
    # Read image data
    image_data = imageio.imread(buf)
    
    plt.close()
    buf.close()
    
    return image_data

def analyze_discrete_state_transitions(model, control_values, device='cpu', system_type=None):
    """
    Analyze state transitions for different control values using discrete state assignments.
    
    Args:
        model: Trained DRM model
        control_values: List of control values to analyze
        device: Device for computation
        system_type: Type of system for control formatting
    
    Returns:
        Dictionary of transition matrices for each control value
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    num_states = model.num_states
    control_dim = model.predictor.control_dim
    
    # Create one-hot encoded states
    one_hot_states = torch.eye(num_states, device=device)
    
    results = {}
    
    for control_value in control_values:
        # Handle different control formats based on system type
        if system_type == 'saddle_system':
            # For categorical controls, create one-hot encoding
            control_batch = torch.zeros((num_states, control_dim), dtype=torch.float32, device=device)
            control_batch[:, control_value] = 1.0
        else:
            # For continuous controls, handle both scalar and multi-dimensional cases
            if isinstance(control_value, (list, tuple, np.ndarray)):
                # Multi-dimensional control (e.g., social_tipping with [b, c, f, g])
                control_tensor = torch.tensor(control_value, dtype=torch.float32, device=device)
                # Expand to batch size: (num_states, control_dim)
                control_batch = control_tensor.unsqueeze(0).expand(num_states, -1)
            else:
                # Scalar control (e.g., tech_substitution)
                control_batch = torch.full((num_states, control_dim), control_value, dtype=torch.float32, device=device)
        
        # Predict next state with error checking
        with torch.no_grad():
            next_state_probs = model.predict_next_state(one_hot_states, control_batch)
            
            # Check for NaN
            if torch.isnan(next_state_probs).any():
                print("WARNING: NaN detected in next_state_probs!")
                next_state_probs = torch.nan_to_num(next_state_probs, nan=0.0)
                # Renormalize
                row_sums = next_state_probs.sum(dim=1, keepdim=True)
                next_state_probs = next_state_probs / torch.clamp(row_sums, min=1e-10)
        
        # Convert to numpy for analysis
        transition_matrix = next_state_probs.cpu().numpy()
        
        # Use a string representation of the control for the key
        if isinstance(control_value, (list, tuple, np.ndarray)):
            control_key = tuple(control_value)  # Use tuple instead of string
        else:
            control_key = control_value
            
        results[control_key] = transition_matrix
    
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
    
    # Helper function to convert control values to dictionary keys
    def get_control_key(control_value):
        if isinstance(control_value, (list, tuple, np.ndarray)):
            return tuple(control_value)
        else:
            return control_value
    
    # Get number of states from first control
    first_control_key = get_control_key(control_values[0])
    num_states = transition_matrices[first_control_key].shape[0]
    
    # Create heatmaps
    fig, axes = plt.subplots(1, len(control_values), figsize=(len(control_values)*6, 5))
    if len(control_values) == 1:
        axes = [axes]
        
    for i, control in enumerate(control_values):
        ax = axes[i]
        control_key = get_control_key(control)  # Convert to proper key format
        matrix = transition_matrices[control_key]
        
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues",
                   vmin=0, vmax=1, ax=ax, cbar=True)
        # Format control value for display
        if isinstance(control, (list, tuple, np.ndarray)):
            # Convert to regular Python list and round for readability
            control_display = [round(float(x), 3) for x in control]
        else:
            control_display = control
            
        ax.set_title(f"Transition Probabilities (c={control_display})")
        ax.set_xlabel("Next State")
        ax.set_ylabel("Current State")
        ax.set_xticklabels([f"S{j+1}" for j in range(num_states)])
        ax.set_yticklabels([f"S{j+1}" for j in range(num_states)])
    
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"Saved transition matrix visualization to {output_path}")
        plt.close(fig)
    else:
        plt.show()
    
    # Print transition matrices as tables
    for control in control_values:
        control_key = get_control_key(control)  # Convert to proper key format
        matrix = transition_matrices[control_key]
        
        # Format control value for display
        if isinstance(control, (list, tuple, np.ndarray)):
            control_display = [round(float(x), 3) for x in control]
        else:
            control_display = control
            
        df = pd.DataFrame(matrix,
                         index=[f"State {i+1}" for i in range(num_states)],
                         columns=[f"State {i+1}" for i in range(num_states)])
        print(f"\nTransition Probabilities for c={control_display}:")
        print(df.round(3))
    
    return fig

def visualize_model_architecture(model, output_path):
    """
    Create a visual representation of the model architecture by analyzing the actual PyTorch model structure.
    Enhanced to better represent non-sequential flows in both BilinearPredictor and StandardPredictor.
    
    Args:
        model: The PyTorch model to visualize
        output_path: Path to save the visualization image
    """
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Model Architecture', 
                           format='png',
                           engine='dot')
    
    # Set graph attributes for better appearance
    dot.attr('graph', rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.7')
    dot.attr('node', shape='box', style='filled', fontname='Arial', fontsize='12')
    dot.attr('edge', fontname='Arial', fontsize='10')
    
    # Helper functions to extract info from model
    def get_module_name(module_path):
        """Convert module path to a readable name"""
        return module_path.replace('.', '_')
    
    def get_module_type(module):
        """Get module type in a readable format"""
        class_name = module.__class__.__name__
        return class_name
    
    def get_layer_shape(module):
        """Try to extract input/output dimensions from module"""
        shape_info = ""
        if isinstance(module, nn.Linear):
            shape_info = f"{module.in_features}→{module.out_features}"
        return shape_info
    
    def get_color(module_type):
        """Assign colors based on module type"""
        colors = {
            'Linear': 'skyblue',
            'ReLU': 'lightgrey',
            'Softmax': 'lightgrey',
            'Sequential': 'lightblue',
            'ModuleList': 'lightblue',
            'ControlGate': 'palegreen',
            'StandardPredictor': 'palegreen',
            'BilinearPredictor': 'palegreen',
            'ControlGatePredictor': 'palegreen',
            'Bilinear': 'lightyellow',  # Special color for bilinear
            'DiscreteRepresentationsModel': 'lightgrey',
        }
        # Check if any substring matches
        for key in colors:
            if key in module_type:
                return colors[key]
        return 'white'  # Default color
    
    # Track modules to create clusters
    clusters = {
        'encoder': [],
        'predictor': [],
        'value_net': []
    }
    
    all_modules = {}
    connections = defaultdict(list)
    custom_connections = []  # For special non-sequential connections
    
    # Add main model node
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    main_node_id = 'model'
    dot.node(main_node_id, 
             f'Discrete Representations Model\n{model.num_states} states\n{total_params:,} parameters', 
             fillcolor='lightgrey')
    
    # Extract model structure
    def extract_model_structure(model, prefix=''):
        for name, child in model.named_children():
            module_path = f"{prefix}.{name}" if prefix else name
            module_id = get_module_name(module_path)
            
            # Determine cluster
            cluster_key = None
            for key in clusters:
                if key == name or module_path.startswith(key):
                    cluster_key = key
                    break
            
            if cluster_key:
                clusters[cluster_key].append(module_id)
            
            # Get module details
            module_type = get_module_type(child)
            shape_info = get_layer_shape(child)
            
            # Store module info
            label = f"{name}\n{module_type}"
            if shape_info:
                label += f"\n{shape_info}"
            
            all_modules[module_id] = {
                'name': name,
                'path': module_path,
                'type': module_type,
                'label': label,
                'color': get_color(module_type),
                'module': child  # Store reference to actual module
            }
            
            # Check if module is a container
            if list(child.named_children()):
                # Module has children, add connections to them
                child_ids = []
                extract_model_structure(child, module_path)
                
                # Connect container to its first child
                for child_name, _ in child.named_children():
                    child_path = f"{module_path}.{child_name}"
                    child_id = get_module_name(child_path)
                    child_ids.append(child_id)
                
                # Link first and last element only if this is a Sequential container
                if isinstance(child, nn.Sequential) and child_ids:
                    # If sequential, connect first and last elements
                    first_id = child_ids[0]
                    for i in range(len(child_ids) - 1):
                        connections[child_ids[i]].append(child_ids[i+1])
            else:
                # Leaf module, add it to graph
                if isinstance(child, nn.Linear):
                    # Calculate params for this layer
                    params = child.in_features * child.out_features
                    if child.bias is not None:
                        params += child.out_features
                    label += f"\n{params:,} params"
    
    # Extract the structure
    extract_model_structure(model)
    
    # Add special input/output nodes for forward path
    special_nodes = {
        'x_input': {'label': 'x (observation)', 'shape': 'ellipse', 'color': 'lightgrey'},
        'c_input': {'label': 'c (control)', 'shape': 'ellipse', 'color': 'lightgrey'},
        'y_input': {'label': 'y (next obs)', 'shape': 'ellipse', 'color': 'lightgrey'},
        'v_true_input': {'label': 'v_true', 'shape': 'ellipse', 'color': 'lightgrey'},
        's_x_output': {'label': 's_x (state probs)', 'shape': 'ellipse', 'color': 'lightgrey'},
        's_y_output': {'label': 's_y (true next state)', 'shape': 'ellipse', 'color': 'lightgrey'},
        's_y_pred_output': {'label': 's_y_pred (predicted)', 'shape': 'ellipse', 'color': 'lightgrey'},
        'v_pred_output': {'label': 'v_pred (value)', 'shape': 'ellipse', 'color': 'lightgrey'},
        'concat_node': {'label': 'concatenated features', 'shape': 'ellipse', 'color': 'lightyellow'}
    }
    
    # Add special nodes
    for node_id, node_info in special_nodes.items():
        if node_id == 'concat_node' and not isinstance(model.predictor, StandardPredictor):
            # Only add concat node for StandardPredictor
            continue
        dot.node(node_id, node_info['label'], shape=node_info['shape'], fillcolor=node_info['color'])
    
    # Analyze special data flows for different predictor types
    if hasattr(model, 'predictor'):
        if isinstance(model.predictor, BilinearPredictor):
            # Add custom data flow connections for BilinearPredictor
            predictor = model.predictor
            control_encoder_id = get_module_name('predictor.control_encoder')
            interaction_id = get_module_name('predictor.interaction')
            hidden_id = get_module_name('predictor.hidden')
            output_id = get_module_name('predictor.output')
            
            # Create special path descriptions
            custom_connections.extend([
                ('c_input', control_encoder_id, {'label': 'control input', 'color': 'blue'}),
                (control_encoder_id, interaction_id, {'label': 'control features', 'color': 'blue'}),
                ('s_x_output', interaction_id, {'label': 'state input', 'color': 'green'}),
                (interaction_id, hidden_id, {'label': 'interaction', 'color': 'red'}),
                (hidden_id, output_id, {'label': 'hidden features', 'color': 'purple'}),
                (output_id, 's_y_pred_output', {'label': 'logits->softmax', 'color': 'orange'})
            ])
            
        elif isinstance(model.predictor, StandardPredictor):
            # Add custom data flow connections for StandardPredictor
            control_encoder_id = get_module_name('predictor.control_encoder')
            predictor_id = get_module_name('predictor.predictor')
            
            # Get the first layer of the sequential predictor
            first_layer_id = None
            for module_id in all_modules:
                if module_id.startswith('predictor_predictor_0'):
                    first_layer_id = module_id
                    break
            
            # Create special path descriptions for StandardPredictor
            custom_connections.extend([
                ('c_input', control_encoder_id, {'label': 'control input', 'color': 'blue'}),
                (control_encoder_id, 'concat_node', {'label': 'encoded control', 'color': 'blue'}),
                ('s_x_output', 'concat_node', {'label': 'state probs', 'color': 'green'}),
                ('concat_node', first_layer_id if first_layer_id else predictor_id, {'label': 'concatenated', 'color': 'red'}),
                (get_module_name('predictor.predictor'), 's_y_pred_output', {'label': 'output', 'color': 'orange'})
            ])
    
    # Create encoder cluster
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='Encoder', style='filled', color='lightblue', fillcolor='azure')
        
        # Add encoder modules
        for module_id in clusters['encoder']:
            if module_id in all_modules:
                module = all_modules[module_id]
                c.node(module_id, module['label'], fillcolor=module['color'])
        
        # Add connections between encoder modules
        for src, dests in connections.items():
            if src in clusters['encoder']:
                for dest in dests:
                    if dest in clusters['encoder']:
                        c.edge(src, dest)
        
        # Connect inputs/outputs
        c.edge('x_input', clusters['encoder'][0] if clusters['encoder'] else 'encoder')
        c.edge(clusters['encoder'][-1] if clusters['encoder'] else 'encoder', 's_x_output')
        
        # Add y path for training (dashed)
        c.edge('y_input', clusters['encoder'][0] if clusters['encoder'] else 'encoder', 
              style='dashed', label='shared weights')
        c.edge(clusters['encoder'][-1] if clusters['encoder'] else 'encoder', 's_y_output', 
              style='dashed')
    
    # Create predictor cluster with special handling for different predictor types
    with dot.subgraph(name='cluster_predictor') as c:
        predictor_type = model.predictor.__class__.__name__
        c.attr(label=f'{predictor_type}', style='filled', color='lightgreen', fillcolor='mintcream')
        
        # Add predictor modules
        for module_id in clusters['predictor']:
            if module_id in all_modules:
                module = all_modules[module_id]
                c.node(module_id, module['label'], fillcolor=module['color'])
        
        # For BilinearPredictor, create a custom layout
        if isinstance(model.predictor, BilinearPredictor):
            # Highlight the interaction module
            interaction_id = get_module_name('predictor.interaction')
            if interaction_id in all_modules:
                c.node(interaction_id, all_modules[interaction_id]['label'], 
                      shape='Mrecord', fillcolor='gold')
                      
        # For StandardPredictor, add the concatenation node
        elif isinstance(model.predictor, StandardPredictor):
            # Concatenation happens outside any specific module
            c.node('concat_node', 'Concatenate\ns_x + encoded c', 
                  shape='Mrecord', fillcolor='gold')
        
        # Only add standard connections for the remaining modules
        # (not for BilinearPredictor or StandardPredictor where we use custom connections)
        if not isinstance(model.predictor, (BilinearPredictor, StandardPredictor)):
            # Add connections between predictor modules
            for src, dests in connections.items():
                if src in clusters['predictor']:
                    for dest in dests:
                        if dest in clusters['predictor']:
                            c.edge(src, dest)
            
            # Connect inputs/outputs
            c.edge('s_x_output', clusters['predictor'][0] if clusters['predictor'] else 'predictor')
            c.edge('c_input', clusters['predictor'][0] if clusters['predictor'] else 'predictor')
            c.edge(clusters['predictor'][-1] if clusters['predictor'] else 'predictor', 's_y_pred_output')
    
    # Create value network cluster
    with dot.subgraph(name='cluster_value') as c:
        c.attr(label='Value Network', style='filled', color='salmon', fillcolor='seashell')
        
        # Add value network modules
        for module_id in clusters['value_net']:
            if module_id in all_modules:
                module = all_modules[module_id]
                c.node(module_id, module['label'], fillcolor=module['color'])
        
        # Add connections between value network modules
        for src, dests in connections.items():
            if src in clusters['value_net']:
                for dest in dests:
                    if dest in clusters['value_net']:
                        c.edge(src, dest)
        
        # Connect inputs/outputs
        c.edge('s_y_pred_output', clusters['value_net'][0] if clusters['value_net'] else 'value_net')
        c.edge(clusters['value_net'][-1] if clusters['value_net'] else 'value_net', 'v_pred_output')
    
    # Add custom connections for non-sequential flows
    for src, dest, attrs in custom_connections:
        dot.edge(src, dest, 
                color=attrs.get('color', 'blue'), 
                label=attrs.get('label', ''), 
                style=attrs.get('style', 'dashed'))
    
    # Add loss nodes
    dot.node('state_loss', 'State Loss\nKL Divergence', shape='diamond', fillcolor='lightpink')
    dot.node('value_loss', 'Value Loss\nMSE', shape='diamond', fillcolor='lightpink')
    
    # Connect for training path
    dot.edge('s_y_output', 'state_loss')
    dot.edge('s_y_pred_output', 'state_loss')
    dot.edge('v_pred_output', 'value_loss')
    dot.edge('v_true_input', 'value_loss')
    
    # Render the graph
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    try:
        dot.render(output_path.replace('.png', ''), format='png', cleanup=True)
        print(f"Model architecture visualization saved to {output_path}.png")
        return output_path + '.png'
    except Exception as e:
        print(f"Error rendering graph: {e}")
        # Try to save in current directory as fallback
        fallback_path = f"model_arch_{model.num_states}_states"
        dot.render(fallback_path, format='png', cleanup=True)
        print(f"Fallback visualization saved to {fallback_path}.png")
        return fallback_path + '.png'

def plot_softmax_rank_evolution(history, save_path):
    """
    Plot evolution of softmax rank metrics over training.
    Uses Paul Tol's muted color scheme for colorblind accessibility.
    Uses global normalization for singular values (like the paper).
    
    Args:
        history: Training history containing softmax_rank_metrics
        save_path: Path to save the plot
    """
    
    if 'softmax_rank_metrics' not in history or not history['softmax_rank_metrics']:
        print("No softmax rank metrics found in history")
        return
    
    # Paul Tol's muted color scheme for colorblind accessibility
    tol_muted = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933']
    
    metrics_list = history['softmax_rank_metrics']
    epochs = [m['epoch'] for m in metrics_list]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # ========================================================================
    # Plot 1: Rank Evolution During Training
    # ========================================================================
    ax = axes[0, 0]
    hidden_ranks = [m['hidden_rank'] for m in metrics_list]
    logit_ranks = [m['logit_rank'] for m in metrics_list]
    softmax_ranks = [m['softmax_rank'] for m in metrics_list]
    
    ax.plot(epochs, hidden_ranks, 'o-', label='Hidden Layer (32-dim)', 
           linewidth=2, markersize=4, color=tol_muted[0])
    ax.plot(epochs, logit_ranks, 's-', label='Logit Layer (4-dim)', 
           linewidth=2, markersize=4, color=tol_muted[1]) 
    ax.plot(epochs, softmax_ranks, '^-', label='Post-Softmax (4-dim)', 
           linewidth=2, markersize=4, color=tol_muted[2])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Numerical Rank')
    ax.set_title('Rank Evolution During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)  # Start y-axis at 0 as requested
    
    # ========================================================================
    # Plot 2: Norm Evolution (Hidden A, Logit M, Softmax A)
    # ========================================================================
    ax = axes[0, 1]
    hidden_norms = [m['hidden_frobenius_norm'] for m in metrics_list]      # ||A₃||_F
    logit_norms = [m['logit_frobenius_norm'] for m in metrics_list]        # ||M₄||_F  
    softmax_norms = [m['softmax_frobenius_norm'] for m in metrics_list]    # ||A₄||_F
    
    ax.plot(epochs, hidden_norms, 'o-', label='Hidden ||A₃||_F', 
           linewidth=2, markersize=4, color=tol_muted[0])
    ax.plot(epochs, logit_norms, 's-', label='Logit ||M₄||_F', 
           linewidth=2, markersize=4, color=tol_muted[1])
    ax.plot(epochs, softmax_norms, '^-', label='Softmax ||A₄||_F', 
           linewidth=2, markersize=4, color=tol_muted[2])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Frobenius Norm')
    ax.set_title('Norm Evolution During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Keep log scale for norms (makes sense here)
    
    # ========================================================================
    # Plot 3: Logit Singular Values Evolution (GLOBAL NORMALIZATION, LINEAR SCALE)
    # ========================================================================
    ax = axes[1, 0]
    
    # Plot first 4 globally normalized singular values for logits
    colors_sv = [tol_muted[0], tol_muted[1], tol_muted[2], tol_muted[3]]
    for i in range(4):
        sv_global_key = f'logit_sv_global_norm_{i}'
        if sv_global_key in metrics_list[0]:  # Check if globally normalized values exist
            sv_values = [m.get(sv_global_key, 0) for m in metrics_list]
            ax.plot(epochs, sv_values, 'o-', label=f'σ_{i+1}', 
                   color=colors_sv[i], linewidth=2, markersize=3)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Globally Normalized Singular Value')
    ax.set_title('Logit Singular Values Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)  # Linear scale from 0 to 1
    
    # ========================================================================
    # Plot 4: Hidden Singular Values Evolution (GLOBAL NORMALIZATION, LINEAR SCALE)
    # ========================================================================
    ax = axes[1, 1]
    
    # Plot first 4 globally normalized singular values for hidden layer
    colors_sv = [tol_muted[0], tol_muted[1], tol_muted[2], tol_muted[3]]
    for i in range(4):
        sv_global_key = f'hidden_sv_global_norm_{i}'
        if sv_global_key in metrics_list[0]:  # Check if globally normalized values exist
            sv_values = [m.get(sv_global_key, 0) for m in metrics_list]
            ax.plot(epochs, sv_values, 'o-', label=f'σ_{i+1}', 
                   color=colors_sv[i], linewidth=2, markersize=3)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Globally Normalized Singular Value')
    ax.set_title('Hidden Layer Singular Values Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)  # Linear scale from 0 to 1
    
    # ========================================================================
    # Final styling and save
    # ========================================================================
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved softmax rank evolution plot to {save_path}")

def plot_training_curves_aggregated(aggregated_data, save_path):
    """
    Plot aggregated training curves with soft std visualization for train curves only.
    Uses Paul Tol's muted color scheme consistent with other visualizations.
    """
    # Paul Tol's muted color scheme
    tol_muted = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933']
    
    training_curves = aggregated_data.get('training_curves', {})
    if not training_curves:
        print("No training curves found for aggregation plot")
        return
    
    # Create subplots for different loss types
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Define plot configurations - align colors with existing train/val pattern
    plot_configs = [
        {
            'ax': axes[0, 0],
            'title': 'Total Loss',
            'train_key': 'train_loss',
            'val_key': 'val_loss'
        },
        {
            'ax': axes[0, 1], 
            'title': 'State Loss',
            'train_key': 'train_state_loss',
            'val_key': 'val_state_loss'
        },
        {
            'ax': axes[1, 0],
            'title': 'Value Loss', 
            'train_key': 'train_value_loss',
            'val_key': 'val_value_loss'
        },
        {
            'ax': axes[1, 1],
            'title': 'Entropy Loss',
            'train_key': 'train_entropy_loss',
            'val_key': 'val_entropy_loss'
        }
    ]
    
    # Consistent train/val colors
    train_color = tol_muted[1]  # '#332288' - blue (to match existing)
    val_color = tol_muted[2]    # '#DDCC77' - yellow (to match existing)
    
    for config in plot_configs:
        ax = config['ax']
        
        # Plot training curve with std band
        if config['train_key'] in training_curves and training_curves[config['train_key']] is not None:
            curve_data = training_curves[config['train_key']]
            
            mean_values = np.array(curve_data['mean'])
            std_values = np.array(curve_data['std'])
            epochs = np.arange(len(mean_values))
            
            # Plot mean line
            ax.plot(epochs, mean_values, label='Train', color=train_color, linewidth=2)
            
            # Plot std band (higher alpha as requested)
            ax.fill_between(epochs, 
                           mean_values - std_values, 
                           mean_values + std_values,
                           color=train_color, alpha=0.1)
        
        # Plot validation curve (NO std band)
        if config['val_key'] in training_curves and training_curves[config['val_key']] is not None:
            curve_data = training_curves[config['val_key']]
            
            mean_values = np.array(curve_data['mean'])
            epochs = np.arange(len(mean_values))
            
            # Plot mean line only
            ax.plot(epochs, mean_values, label='Val', color=val_color, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(config['title'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)  # Start from 0 for loss curves
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves aggregated plot to {save_path}")

def plot_softmax_rank_aggregated(aggregated_data, save_path):
    """
    Plot aggregated softmax rank evolution with std bands.
    Based on existing plot_softmax_rank_evolution structure.
    """
    # Paul Tol's muted color scheme
    tol_muted = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933']
    
    softmax_metrics = aggregated_data.get('softmax_rank_metrics', {})
    if not softmax_metrics:
        print("No softmax rank metrics found for aggregation plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Helper function to plot metric with std
    def plot_metric_with_std(ax, metric_key, label, color):
        if metric_key in softmax_metrics and softmax_metrics[metric_key] is not None:
            data = softmax_metrics[metric_key]
            mean_values = np.array(data['mean'])
            std_values = np.array(data['std'])
            epochs = np.arange(len(mean_values))  # Individual epochs per metric
            
            ax.plot(epochs, mean_values, 'o-', label=label, 
                   linewidth=2, markersize=4, color=color)
            ax.fill_between(epochs, 
                           mean_values - std_values, 
                           mean_values + std_values,
                           color=color, alpha=0.1)
    
    # Plot 1: Rank Evolution
    ax = axes[0, 0]
    plot_metric_with_std(ax, 'hidden_rank', 'Hidden Layer (32-dim)', tol_muted[0])
    plot_metric_with_std(ax, 'logit_rank', 'Logit Layer (4-dim)', tol_muted[1])
    plot_metric_with_std(ax, 'softmax_rank', 'Post-Softmax (4-dim)', tol_muted[2])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Numerical Rank')
    ax.set_title('Rank Evolution During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 2: Norm Evolution
    ax = axes[0, 1]
    plot_metric_with_std(ax, 'hidden_frobenius_norm', 'Hidden ||A₃||_F', tol_muted[0])
    plot_metric_with_std(ax, 'logit_frobenius_norm', 'Logit ||M₄||_F', tol_muted[1])
    plot_metric_with_std(ax, 'softmax_frobenius_norm', 'Softmax ||A₄||_F', tol_muted[2])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Frobenius Norm')
    ax.set_title('Norm Evolution During Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Logit Singular Values (if globally normalized data exists)
    ax = axes[1, 0]
    colors_sv = [tol_muted[0], tol_muted[1], tol_muted[2], tol_muted[3]]
    for i in range(4):
        sv_key = f'logit_sv_global_norm_{i}'
        plot_metric_with_std(ax, sv_key, f'σ_{i+1}', colors_sv[i])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Globally Normalized Singular Value')
    ax.set_title('Logit Singular Values Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 4: Hidden Singular Values
    ax = axes[1, 1]
    colors_sv = [tol_muted[0], tol_muted[1], tol_muted[2], tol_muted[3]]
    for i in range(4):
        sv_key = f'hidden_sv_global_norm_{i}'
        plot_metric_with_std(ax, sv_key, f'σ_{i+1}', colors_sv[i])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Globally Normalized Singular Value')
    ax.set_title('Hidden Layer Singular Values Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved softmax rank aggregated plot to {save_path}")

def plot_state_metrics_aggregated(aggregated_data, save_path):
    """
    Plot aggregated state assignment quality metrics (stability and sharpness).
    """
    # Paul Tol's muted color scheme
    tol_muted = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933']
    
    state_metrics = aggregated_data.get('state_metrics', {})
    if not state_metrics:
        print("No state metrics found for aggregation plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Helper function to plot metric with std - FIXED to handle epochs per metric
    def plot_metric_with_std(ax, metric_key, label, color):
        if metric_key in state_metrics and state_metrics[metric_key] is not None:
            data = state_metrics[metric_key]
            mean_values = np.array(data['mean'])
            std_values = np.array(data['std'])
            epochs = np.arange(len(mean_values))  # Create epochs for THIS metric specifically
            
            ax.plot(epochs, mean_values, 'o-', label=label, 
                   linewidth=2, markersize=4, color=color)
            ax.fill_between(epochs, 
                           mean_values - std_values, 
                           mean_values + std_values,
                           color=color, alpha=0.1)  # Lower alpha for softer appearance
    
    # Plot 1: Sharpness metrics
    ax = axes[0]
    plot_metric_with_std(ax, 'sharpness_mean', 'Mean Sharpness', tol_muted[1])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Entropy (Sharpness)')
    ax.set_title('State Assignment Sharpness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Stability metrics
    ax = axes[1]
    plot_metric_with_std(ax, 'dominant_stability', 'Dominant State Stability', tol_muted[1])
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Stability (%)')
    ax.set_title('State Assignment Stability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)  # Stability is in percentage
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved state metrics aggregated plot to {save_path}")

def plot_test_metrics_summary(aggregated_data, save_path):
    """
    Plot summary of final test metrics with error bars.
    """
    # Paul Tol's muted color scheme
    tol_muted = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933']
    
    test_metrics = aggregated_data.get('test_metrics', {})
    if not test_metrics:
        print("No test metrics found for summary plot")
        return
    
    # Define important metrics and their display names
    metric_configs = [
        ('test_loss', 'Total Loss'),
        ('test_state_loss', 'State Loss'), 
        ('test_value_loss', 'Value Loss'),
        ('prob_discrete_accuracy', 'Discrete Accuracy'),
        ('test_batch_entropy', 'Batch Entropy'),
        ('test_individual_entropy', 'Individual Entropy')
    ]
    
    # Filter to available metrics
    available_metrics = [(key, name) for key, name in metric_configs 
                        if key in test_metrics and test_metrics[key] is not None]
    
    if not available_metrics:
        print("No available test metrics for summary plot")
        return
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metric_names = [name for _, name in available_metrics]
    means = [test_metrics[key]['mean'] for key, _ in available_metrics]
    stds = [test_metrics[key]['std'] for key, _ in available_metrics]
    
    x_pos = np.arange(len(metric_names))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                  color=[tol_muted[i % len(tol_muted)] for i in range(len(metric_names))],
                  alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Test Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Final Test Metrics Summary')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01*max(means),
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved test metrics summary plot to {save_path}")


### NOTE: NO LONGER IN USE

def analyze_mdp_from_model(model, control_values=None, device='cpu'):
    """
    Analyze MDP representation from a trained DRM model using one-hot encoding
    
    Args:
        model: Trained DRM model
        control_values: List of control values to analyze (default: [0.5, 1.0])
        device: Device to run computation on
        
    Returns:
        Dictionary containing:
        - transition_matrices: Dict of transition probability matrices for each control
        - state_values: Values predicted for each state
        - control_values: List of control values analyzed
    """
    if control_values is None:
        control_values = [0.5, 1.0]
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    num_states = model.num_states
    
    # Create one-hot encoded states
    one_hot_states = torch.eye(num_states, device=device)
    
    # Get state values directly from the value network
    with torch.no_grad():
        state_values = model.compute_value(one_hot_states).cpu().numpy()
    
    # Initialize results dictionary
    transition_matrices = {}
    
    # Calculate transition probabilities for each control value
    for control_value in control_values:
        # Prepare control tensor (same control for all states)
        control_batch = torch.full((num_states, 1), control_value, dtype=torch.float32, device=device)
        
        # Get next state predictions for each one-hot state
        with torch.no_grad():
            next_state_probs = model.predict_next_state(one_hot_states, control_batch)
            
        # Convert to numpy for analysis
        transition_matrix = next_state_probs.cpu().numpy()
        transition_matrices[control_value] = transition_matrix
    
    return {
        'transition_matrices': transition_matrices,
        'state_values': state_values,
        'control_values': control_values
    }

def visualize_mdp(mdp_data, output_path=None, min_prob_to_show=0.02):
    """
    Create a unified graphviz visualization of the MDP with all control values
    
    Args:
        mdp_data: Output from analyze_mdp_from_model
        output_path: Path to save the visualization
        min_prob_to_show: Minimum probability to display (for cleaner graphs)
        
    Returns:
        Rendered graphviz graph and output path
    """
    transition_matrices = mdp_data['transition_matrices']
    state_values = mdp_data['state_values']
    control_values = mdp_data['control_values']
    
    num_states = state_values.shape[0]
    
    # Create a single digraph for all controls
    dot = graphviz.Digraph(comment='MDP with all controls')
    dot.attr('graph', rankdir='LR', splines='true', nodesep='0.8', ranksep='1.5')
    dot.attr('node', shape='box', style='filled', fontname='Arial', fontsize='12')
    dot.attr('edge', fontname='Arial', fontsize='10')
    
    # Add state nodes
    for state in range(num_states):
        value_str = ", ".join([f"{val:.3f}" for val in state_values[state]])
        dot.node(f's{state+1}', 
                 f's{state+1}\nValue: {value_str}', 
                 shape='circle', 
                 fillcolor='#f8d7e0',  # Light pink
                 style='filled')
    
    # Add action nodes and transitions for all control values
    for control_value in control_values:
        for from_state in range(num_states):
            # Create ONE action node per state-control pair
            action_id = f'a{control_value}_{from_state+1}'
            action_label = f'a{control_value}'
            dot.node(action_id, action_label, shape='diamond', fillcolor='#d7d7f8', style='filled')
            
            # Connect state to its action node
            dot.edge(f's{from_state+1}', action_id)
            
            # Process all possible transitions from this state-action
            transitions = transition_matrices[control_value][from_state]
            
            for to_state in range(num_states):
                prob = transitions[to_state]
                
                # Skip low probability transitions for clarity
                if prob < min_prob_to_show:
                    continue
                
                # Connect action to next state with probability as label
                # Adjust line weight (penwidth) based on probability 
                penwidth = 1.0 + 3.0 * prob  # Scale between 1 and 4 based on probability
                dot.edge(action_id, f's{to_state+1}', 
                         label=f'{prob:.3f}', 
                         color='green',
                         penwidth=str(penwidth))
    
    # Render the graph
    if output_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        rendered_path = dot.render(output_path.replace('.png', ''), format='png', cleanup=True)
        print(f"Rendered unified MDP visualization to {rendered_path}")
    else:
        dot.view()
    
    return dot, output_path

### NOTE: DEPRECATED

def create_state_viz_from_data(df, output_path=None, epochs=None, epoch_frequency=None, 
                               create_gif=False, gif_duration=0.5, figsize=(12, 10)):
    """
    NOTE: DEPRECATED
    Create state space visualizations from DataFrame data.
    
    Args:
        df: DataFrame with state assignment data (from extract_state_assignment_data)
        output_path: Base path for saving (without extension)
        epochs: List of specific epochs to plot, or None for all
        epoch_frequency: Plot every nth epoch, or None to use epochs parameter
        create_gif: If True, create animated GIF for multiple epochs
        gif_duration: Duration per frame in GIF (seconds)
        figsize: Figure size for plots
    
    Returns:
        dict: Dictionary with information about created files
    """
    
    # Get available epochs
    if 'epoch' in df.columns:
        available_epochs = sorted(df['epoch'].unique())
    else:
        available_epochs = [None]  # Single epoch case
    
    # Determine which epochs to plot
    if epochs is not None:
        epochs_to_plot = [e for e in epochs if e in available_epochs]
    elif epoch_frequency is not None:
        epochs_to_plot = available_epochs[::epoch_frequency]
    else:
        epochs_to_plot = available_epochs
    
    if not epochs_to_plot:
        raise ValueError("No valid epochs found to plot")
    
    # Get metadata from DataFrame attributes
    grid_size = df.attrs.get('grid_size', int(np.sqrt(len(df) // len(epochs_to_plot))))
    bounds = df.attrs.get('bounds', [(-5, 5), (-5, 5)])
    num_states = df.attrs.get('num_states', 4)
    
    created_files = []
    temp_files = []  # For GIF creation
    temp_dir = None
    
    # Create temp directory if creating GIF
    if create_gif and len(epochs_to_plot) > 1:
        temp_dir = tempfile.mkdtemp()
    
    try:
        for epoch in epochs_to_plot:
            # Filter data for this epoch
            if epoch is not None:
                epoch_df = df[df['epoch'] == epoch].copy()
                epoch_suffix = f"_epoch_{epoch}"
            else:
                epoch_df = df.copy()
                epoch_suffix = ""
            
            if len(epoch_df) == 0:
                continue
                
            # Determine subplot layout based on number of states
            if num_states <= 4:
                nrows, ncols = 2, 2
            elif num_states <= 6:
                nrows, ncols = 2, 3
            elif num_states <= 9:
                nrows, ncols = 3, 3
            elif num_states <= 12:
                nrows, ncols = 3, 4
            else:
                nrows, ncols = 4, 4  # Max 4x4 grid, limit to 16 states
            
            # Adjust figure size based on layout
            fig_width = ncols * (figsize[0] / 2)
            fig_height = nrows * (figsize[1] / 2)
            
            # Create the visualization
            fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
            if nrows * ncols == 1:
                axes = [axes]
            else:
                axes = axes.ravel()
            
            # Get grid coordinates
            x1_vals = epoch_df['x1'].values.reshape(grid_size, grid_size)
            x2_vals = epoch_df['x2'].values.reshape(grid_size, grid_size)
            
            # Plot each state
            for state_idx in range(min(num_states, nrows * ncols)):
                ax = axes[state_idx]
                
                # Get state probabilities and reshape to grid
                state_col = f'state_{state_idx}_prob'
                if state_col in epoch_df.columns:
                    state_probs = epoch_df[state_col].values.reshape(grid_size, grid_size)
                else:
                    # Fallback: create empty grid
                    state_probs = np.zeros((grid_size, grid_size))
                
                # Create heatmap
                im = ax.imshow(state_probs, extent=[bounds[0][0], bounds[0][1], 
                                                   bounds[1][0], bounds[1][1]], 
                              origin='lower', cmap='viridis', vmin=0, vmax=1, 
                              aspect='auto', interpolation='bilinear')
                
                # Styling
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                ax.set_title(f'State {state_idx + 1}')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('State Assignment Strength')
            
            # Hide unused subplots
            for idx in range(num_states, len(axes)):
                axes[idx].set_visible(False)
            
            # Overall title
            if epoch is not None:
                fig.suptitle(f'State Space Visualization - Epoch {epoch}', fontsize=14)
            else:
                fig.suptitle('State Space Visualization', fontsize=14)
            
            plt.tight_layout()
            
            # Save or store for GIF
            if create_gif and len(epochs_to_plot) > 1:
                temp_path = os.path.join(temp_dir, f"state_viz_epoch_{epoch}.png")
                plt.savefig(temp_path, dpi=150, bbox_inches='tight')
                temp_files.append(temp_path)
                plt.close()
            elif output_path:
                file_path = f"{output_path}{epoch_suffix}.png"
                plt.savefig(file_path, dpi=150, bbox_inches='tight')
                created_files.append(file_path)
                plt.close()
            else:
                plt.show()
        
        # Create GIF if requested
        if create_gif and len(temp_files) > 1 and output_path:
            try:
                gif_path = f"{output_path}_animation.gif"
                
                with imageio.get_writer(gif_path, mode='I', duration=gif_duration) as writer:
                    for temp_file in temp_files:
                        image = imageio.imread(temp_file)
                        writer.append_data(image)
                
                created_files.append(gif_path)
                print(f"Created GIF with {len(temp_files)} frames: {gif_path}")
                
            except ImportError:
                print("Warning: imageio not available for GIF creation")
                print("Individual PNG files saved instead")
                
                # If GIF creation fails, save individual files instead
                if output_path:
                    for i, temp_file in enumerate(temp_files):
                        final_path = f"{output_path}_frame_{i:03d}.png"
                        shutil.copy2(temp_file, final_path)
                        created_files.append(final_path)
    
    finally:
        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    return {
        'created_files': created_files,
        'epochs_plotted': epochs_to_plot,
        'num_files': len(created_files)
    }

def analyze_state_assignment_evolution(df, output_path=None):
    """
    NOTE: DEPRECATED!
    Analyze how state assignments change over training epochs.
    
    Args:
        df: DataFrame with state assignment data across epochs
        output_path: Path to save analysis plot
    
    Returns:
        dict: Analysis results
    """

    # Paul Tol's muted color scheme for colorblind accessibility
    tol_muted = ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933']
    
    if 'epoch' not in df.columns:
        print("No epoch information found in DataFrame")
        return {}
    
    epochs = sorted(df['epoch'].unique())
    num_states = df.attrs.get('num_states', 4)
    
    # Calculate metrics per epoch
    metrics = []
    for i, epoch in enumerate(epochs):
        epoch_df = df[df['epoch'] == epoch]
        
        epoch_metrics = {'epoch': epoch}
        
        # State usage (how often each state is dominant)
        state_usage = epoch_df['dominant_state'].value_counts(normalize=True)
        for state_idx in range(num_states):
            epoch_metrics[f'state_{state_idx}_usage'] = state_usage.get(state_idx, 0.0)
        
        # Mean probabilities per state
        for state_idx in range(num_states):
            state_col = f'state_{state_idx}_prob'
            if state_col in epoch_df.columns:
                mean_prob = epoch_df[state_col].mean()  
                epoch_metrics[f'state_{state_idx}_mean'] = mean_prob
        
        # Sharpness: per-grid-point entropy
        state_prob_cols = [f'state_{i}_prob' for i in range(num_states) if f'state_{i}_prob' in epoch_df.columns]
        if state_prob_cols:
            state_probs = epoch_df[state_prob_cols].values
            # Calculate entropy per grid point: -sum(p * log(p)) for each row
            per_point_entropy = -np.sum(state_probs * np.log(state_probs + 1e-8), axis=1)
            epoch_metrics['sharpness_mean'] = np.mean(per_point_entropy)
            epoch_metrics['sharpness_std'] = np.std(per_point_entropy)
        
        # Stability metrics (require previous epoch)
        if i > 0:
            prev_epoch = epochs[i-1]
            prev_epoch_df = df[df['epoch'] == prev_epoch]
            
            # Ensure same ordering
            epoch_df_sorted = epoch_df.sort_values('grid_idx')
            prev_epoch_df_sorted = prev_epoch_df.sort_values('grid_idx')
            
            if len(epoch_df_sorted) == len(prev_epoch_df_sorted):
                # Dominant state stability: % of points keeping same dominant state
                current_dominant = epoch_df_sorted['dominant_state'].values
                prev_dominant = prev_epoch_df_sorted['dominant_state'].values
                same_dominant = np.mean(current_dominant == prev_dominant) * 100
                epoch_metrics['dominant_stability'] = same_dominant
                
                # Probability change stability: average change converted to stability %
                current_probs = epoch_df_sorted[state_prob_cols].values
                prev_probs = prev_epoch_df_sorted[state_prob_cols].values
                avg_prob_change = np.mean(np.abs(current_probs - prev_probs))
                # Convert to stability percentage (lower change = higher stability)
                prob_stability = (1 - min(avg_prob_change * 2, 1.0)) * 100  # Scale factor 2 to make it more sensitive
                epoch_metrics['probability_stability'] = prob_stability
        
        metrics.append(epoch_metrics)
    
    metrics_df = pd.DataFrame(metrics)
    
    # Create analysis plots (2x2 layout)
    if output_path:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. State usage over time
        ax = axes[0, 0]
        for state_idx in range(num_states):
            usage_col = f'state_{state_idx}_usage'
            if usage_col in metrics_df.columns:
                color = tol_muted[state_idx % len(tol_muted)]
                ax.plot(metrics_df['epoch'], metrics_df[usage_col] * 100, 
                       label=f'State {state_idx + 1}', marker='o', 
                       markersize=3, linewidth=1.5, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dominant State (%)')
        ax.set_title('Dominant State Assignments Over Training')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.4)
        
        # 2. Assignment sharpness over time (with distribution)
        ax = axes[0, 1]
        if 'sharpness_mean' in metrics_df.columns:
            epochs_arr = metrics_df['epoch'].values
            mean_sharpness = metrics_df['sharpness_mean'].values
            std_sharpness = metrics_df['sharpness_std'].values
            
            # Plot mean line
            ax.plot(epochs_arr, mean_sharpness, color='blue', linewidth=2, label='Mean')
            # Plot standard deviation as shaded area
            ax.fill_between(epochs_arr, 
                           mean_sharpness - std_sharpness,
                           mean_sharpness + std_sharpness,
                           alpha=0.3, color='blue', label='± 1 std')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Per-Point Entropy (Sharpness)')
        ax.set_title('Assignment Sharpness Over Training\n(Lower = More Discrete)')
        ax.legend()
        ax.grid(True, alpha=0.4)
        
        # 3. Mean probabilities per state
        ax = axes[1, 0]
        for state_idx in range(num_states):
            mean_col = f'state_{state_idx}_mean' 
            if mean_col in metrics_df.columns:
                color = tol_muted[state_idx % len(tol_muted)]
                ax.plot(metrics_df['epoch'], metrics_df[mean_col], 
                       label=f'State {state_idx + 1}', marker='o', 
                       markersize=3, linewidth=1.5, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Probability')
        ax.set_title('Average State Assignment Strength')
        ax.grid(True, alpha=0.4)
        
        # 4. Stability metrics (both on same scale)
        ax = axes[1, 1]
        if 'dominant_stability' in metrics_df.columns:
            # Remove first epoch (no stability data)
            stability_df = metrics_df.dropna(subset=['dominant_stability'])
            ax.plot(stability_df['epoch'], stability_df['dominant_stability'], 
            color='#AA4499', marker='o', linewidth=2, label='Dominant State Stability')  # Purple
            ax.plot(stability_df['epoch'], stability_df['probability_stability'], 
            color='#DDDDDD', marker='^', linewidth=2, label='Probability Stability')    # Gray
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Stability (%)')
        ax.set_title('Assignment Stability Between Epochs')
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_ylim(50, 100)  # Both are percentages
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        'metrics_df': metrics_df,
        'epochs_analyzed': len(epochs),
        'num_states': num_states
    }

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
