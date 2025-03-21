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