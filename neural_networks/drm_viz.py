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
                          points=None, angles_degrees=None):
    """
    Visualize the state probabilities in z-space (transformed space) with x-space coordinate labels.
    
    Args:
        model: Trained DRM model.
        output_path: Optional path to save the visualization. If None, the plot will be displayed.
        transformations: List of transformation functions.
        device: Device to run the model on.
        num_points: Number of points in each dimension of the mesh.
        num_states: Number of states in the model (if None, will be inferred).
        soft: Whether to use soft assignment for state probabilities.
        system_type: Type of system (for default transformations).
        points: List of point coordinates in x-space (e.g., [[1.0, 0.0], [2.0, 1.0]])
        angles_degrees: List of angles in degrees corresponding to points (e.g., [90, 45])
    """
    if transformations is None:
        if system_type is None:
            raise ValueError("Either transformations or system_type must be provided")
        # Get default transformation for the system
        transformation = get_transformation(SystemType[system_type.upper()])
        transformations = [transformation, transformation]  # Same for both dimensions
    
    # Unpack the transformation functions
    forward_transforms, inverse_transforms, _ = zip(*transformations)
    
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
        state_probs = model.get_state_probs(x_test, training=False, soft=soft)
    
    # Infer number of states if not provided
    if num_states is None:
        num_states = state_probs.shape[1]
    
    # Transform points from x-space to z-space for plotting
    points_z = None
    if points is not None:
        points_z = []
        for point in points:
            z1 = forward_transforms[0](point[0])
            z2 = forward_transforms[1](point[1])
            points_z.append([z1, z2])
    
    # Define grid layout for the plot
    cols_per_row = min(4, num_states)
    #cols_per_row = 2 if num_states == 4 else min(4, num_states)
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
                      extent=[z_bounds[0][0], z_bounds[0][1], z_bounds[1][0], z_bounds[1][1]],
                      origin='lower', 
                      cmap='viridis', 
                      vmin=0, vmax=1)
        
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        
        # Overlay points and angles if provided
        if points_z is not None and angles_degrees is not None:
            for i, (point_z, angle_deg) in enumerate(zip(points_z, angles_degrees)):
                # Draw white point
                ax.plot(point_z[0], point_z[1], 'wx', markersize=6, markeredgecolor='black', markeredgewidth=0.5)
                
                # Draw angle line from edge to edge
                angle_rad = np.radians(angle_deg)
                px, py = point_z[0], point_z[1]

                # Line equation: x = px + t*cos(θ), y = py + t*sin(θ)
                # Find t values where line hits boundaries
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

                t_values = []
                # Left/right boundaries (x = 0, x = 1)
                if abs(cos_a) > 1e-10:  # Avoid division by zero
                    t_values.extend([(-px) / cos_a, (1-px) / cos_a])
                # Top/bottom boundaries (y = 0, y = 1)  
                if abs(sin_a) > 1e-10:  # Avoid division by zero
                    t_values.extend([(-py) / sin_a, (1-py) / sin_a])

                # Get min/max t to find intersection points
                t_min, t_max = min(t_values), max(t_values)

                x_start = px + t_min * cos_a
                x_end = px + t_max * cos_a
                y_start = py + t_min * sin_a
                y_end = py + t_max * sin_a

                ax.plot([x_start, x_end], [y_start, y_end], 'w--', linewidth=0.5)
        
        # Set title
        ax.set_title(f'State {state + 1}', fontsize=14, fontweight='bold')
        
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

def analyze_discrete_state_transitions(model, control_values, device='cpu'):
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
        # Prepare control tensor based on control format
        if hasattr(model, 'control_format') and model.control_format == 'categorical':
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