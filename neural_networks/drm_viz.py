import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch
import math
import pandas as pd
import seaborn as sns

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
from data_generation.simulations.grid import tangent_transformation


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
        from data_generation.simulations.grid import tangent_transformation
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
        state_probs = model.get_state_probs(x_test)
    
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
        current_state_probs = model.get_state_probs(x_batch)
        
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