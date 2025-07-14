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
    plt.plot(history['train_value_loss'], label='Train Value Loss')
    plt.plot(history['val_value_loss'], label='Validation Value Loss')
    plt.title('Value Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot combined regularization losses in fourth panel
    plt.subplot(2, 2, 4)
    
    # Plot diversity loss if it exists
    if 'train_div_loss' in history and any(v != 0 for v in history['train_div_loss']):
        plt.plot(history['train_div_loss'], label='Train Diversity Loss', linestyle='-')
        plt.plot(history['val_div_loss'], label='Val Diversity Loss', linestyle='-')
    
    # Plot entropy loss if it exists
    if 'train_entropy_loss' in history and any(v != 0 for v in history['train_entropy_loss']):
        plt.plot(history['train_entropy_loss'], label='Train Entropy Loss', linestyle='--')
        plt.plot(history['val_entropy_loss'], label='Val Entropy Loss', linestyle='--')
    
    # Set title based on what's being displayed
    if ('train_div_loss' in history and any(v != 0 for v in history['train_div_loss']) and
        'train_entropy_loss' in history and any(v != 0 for v in history['train_entropy_loss'])):
        plt.title('Regularization Losses')
    elif 'train_div_loss' in history and any(v != 0 for v in history['train_div_loss']):
        plt.title('Diversity Loss')
    elif 'train_entropy_loss' in history and any(v != 0 for v in history['train_entropy_loss']):
        plt.title('Entropy Loss')
    else:
        plt.title('Regularization Losses (None Active)')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
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

def create_state_viz_from_data(df, output_path=None, epochs=None, epoch_frequency=None, 
                               create_gif=False, gif_duration=0.5, figsize=(12, 10)):
    """
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
                       linewidth=2, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dominant State (%)')
        ax.set_title('Dominant State Assignments Over Training')
        ax.legend()
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
                       linewidth=2, color=color)
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

def analyze_state_transitions(model, 
                              transformations,
                              control_values=None,
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

def analyze_discrete_state_transitions(model, control_values, device='cpu', system_type=None):
    """
    Analyze state transitions for different control values using discrete state assignments.
    
    Args:
        model: Trained DRM model
        control_values: List of control values to analyze
        device: Device for computation
    
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
        # For saddle systems, always use one-hot encoding
        if system_type == 'saddle_system':
            # For categorical controls, create one-hot encoding
            control_batch = torch.zeros((num_states, control_dim), dtype=torch.float32, device=device)
            control_batch[:, control_value] = 1.0
        else:
            # For continuous controls, use scalar value
            control_batch = torch.full((num_states, 1), control_value, dtype=torch.float32, device=device)
        
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
        
        # Store transition matrix
        transition_matrix = next_state_probs.cpu().numpy()
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
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
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
    
def plot_regulization_metrics(history, test_metrics, save_path=None):
    """
    Plot state diversity, batch entropy, individual entropy, and entropy loss curves.
    
    Args:
        history: Dictionary containing training history with metrics
        test_metrics: Dictionary containing test metrics
        save_path: Path to save the visualization
    
    Returns:
        fig: The figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Define metrics to plot
    metrics = [
        {
            'name': 'State Diversity',
            'train_key': 'train_div_loss',
            'val_key': 'val_div_loss',
            'test_key': 'test_div_loss',
            'title': 'State Diversity Loss',
            'ylabel': 'Loss'
        },
        {
            'name': 'Batch Entropy',
            'train_key': 'train_batch_entropy',
            'val_key': 'val_batch_entropy',
            'test_key': 'test_batch_entropy',
            'title': 'Batch Entropy (Higher = More Uniform State Usage)',
            'ylabel': 'Normalized Entropy'
        },
        {
            'name': 'Individual Entropy',
            'train_key': 'train_individual_entropy',
            'val_key': 'val_individual_entropy',
            'test_key': 'test_individual_entropy',
            'title': 'Individual Entropy (Lower = More Discrete States)',
            'ylabel': 'Normalized Entropy'
        },
        {
            'name': 'Entropy Loss',
            'train_key': 'train_entropy_loss',
            'val_key': 'val_entropy_loss',
            'test_key': 'test_entropy_loss',
            'title': 'Entropy Loss',
            'ylabel': 'Loss'
        }
    ]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot training curve
        if metric['train_key'] in history and len(history[metric['train_key']]) > 0:
            ax.plot(history[metric['train_key']], label='Train', color='blue')
        
        # Plot validation curve
        if metric['val_key'] in history and len(history[metric['val_key']]) > 0:
            ax.plot(history[metric['val_key']], label='Validation', color='orange')
        
        # Add test result as horizontal line
        if metric['test_key'] in test_metrics:
            test_value = test_metrics[metric['test_key']]
            ax.axhline(y=test_value, color='red', linestyle='--', 
                      label=f'Test ({test_value:.4f})')
        
        # Add labels and grid
        ax.set_title(metric['title'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric['ylabel'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-limits for entropy metrics to [0, 1] range
        if 'entropy' in metric['name'].lower() and 'loss' not in metric['name'].lower():
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved entropy metrics visualization to {save_path}")
    
    return fig


    import torch
import numpy as np
import graphviz
import os
from pathlib import Path

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

def plot_vicreg_metrics(history, test_metrics, vicreg_weights, save_path=None):
    """
    Plot VICReg total, invariance, variance, and covariance loss curves.
    
    Args:
        history: Dictionary containing training history with VICReg metrics
        test_metrics: Dictionary containing test metrics
        vicreg_weights: Dictionary with VICReg weights {'lambda': X, 'mu': Y, 'nu': Z}
        save_path: Path to save the visualization
    
    Returns:
        fig: The figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Define metrics to plot
    metrics = [
        {
            'name': 'VICReg Total',
            'train_key': 'train_vicreg_total',
            'val_key': 'val_vicreg_total',
            'test_key': 'test_vicreg_total',
            'title': 'VICReg Total Loss',
            'ylabel': 'Total Loss',
            'weight': None  # Total already has weights applied
        },
        {
            'name': 'VICReg Invariance',
            'train_key': 'train_vicreg_invariance',
            'val_key': 'val_vicreg_invariance',
            'test_key': 'test_vicreg_invariance',
            'title': f'VICReg Invariance Loss (λ={vicreg_weights["lambda"]:.1f})',
            'ylabel': 'Raw Invariance Loss',
            'weight': vicreg_weights['lambda']
        },
        {
            'name': 'VICReg Variance',
            'train_key': 'train_vicreg_variance',
            'val_key': 'val_vicreg_variance',
            'test_key': 'test_vicreg_variance',
            'title': f'VICReg Variance Loss (μ={vicreg_weights["mu"]:.1f})',
            'ylabel': 'Raw Variance Loss',
            'weight': vicreg_weights['mu']
        },
        {
            'name': 'VICReg Covariance',
            'train_key': 'train_vicreg_covariance',
            'val_key': 'val_vicreg_covariance',
            'test_key': 'test_vicreg_covariance',
            'title': f'VICReg Covariance Loss (ν={vicreg_weights["nu"]:.1f})',
            'ylabel': 'Raw Covariance Loss',
            'weight': vicreg_weights['nu']
        }
    ]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot training curve
        if metric['train_key'] in history and len(history[metric['train_key']]) > 0:
            ax.plot(history[metric['train_key']], label='Train', color='blue')
        
        # Plot validation curve
        if metric['val_key'] in history and len(history[metric['val_key']]) > 0:
            ax.plot(history[metric['val_key']], label='Validation', color='orange')
        
        # Add test result as horizontal line
        if metric['test_key'] in test_metrics:
            test_value = test_metrics[metric['test_key']]
            ax.axhline(y=test_value, color='red', linestyle='--', 
                      label=f'Test ({test_value:.4f})')
        
        # Add labels and grid
        ax.set_title(metric['title'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric['ylabel'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add note about raw vs weighted values for individual components
        if metric['weight'] is not None:
            ax.text(0.02, 0.98, f'Raw values (multiply by {metric["weight"]:.1f} for contribution)', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Saved VICReg metrics visualization to {save_path}")
    
    return fig