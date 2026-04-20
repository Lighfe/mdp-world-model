import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
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
import matplotlib.patches as mpatches


# Define project root at the module level
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"Added {PROJECT_ROOT} to Python path")

# NOTE: absolute imports from project root
# Import application-specific modules
from neural_networks.drm_dataset import create_data_loaders, get_saddle_configuration
from neural_networks.drm_loss import StableDRMLoss
from neural_networks.drm import (
    DiscreteRepresentationsModel,
    BilinearPredictor,
    StandardPredictor,
)

from neural_networks.system_registry import get_transformation, SystemType, get_visualization_bounds
from data_generation.simulations.grid import tangent_transformation

""" 
NOTE: Important, don't get confused with the layout of the grid. It is in a coordinate system. 
[0, 0] is bottom left, not like with typical numpy array top left! Same applies in higher dimensions.
"""


def plot_training_curves(history, save_path=None, state_loss_type=None):
    """Plot training and validation loss curves"""

    # Paul Tol's muted color scheme for colorblind accessibility
    tol_blue = "#332288"  # Primary blue for training
    tol_yellow = "#DDCC77"  # Yellow for validation
    tol_red = "#CC6677"  # Red for entropy weight

    plt.figure(figsize=(15, 10))

    # Plot total loss
    plt.subplot(2, 2, 1)
    plt.plot(history["train_loss"], color=tol_blue, label="Train Loss")
    plt.plot(history["val_loss"], color=tol_yellow, label="Validation Loss")
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot state loss
    plt.subplot(2, 2, 2)
    plt.plot(history["train_state_loss"], color=tol_blue, label="Train State Loss")
    plt.plot(history["val_state_loss"], color=tol_yellow, label="Validation State Loss")
    if state_loss_type is None:
        plt.title("State Loss")
    else:
        plt.title(f"State Loss ({state_loss_type})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot value loss
    plt.subplot(2, 2, 3)
    plt.plot(history["train_value_loss"], color=tol_blue, label="Train Value Loss")
    plt.plot(history["val_value_loss"], color=tol_yellow, label="Validation Value Loss")
    plt.title("Value Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot batch entropy and entropy weight with dual y-axes in fourth panel
    plt.subplot(2, 2, 4)

    # Create first y-axis for batch entropy
    ax1 = plt.gca()

    # Plot batch entropy if it exists
    if "train_batch_entropy" in history and len(history["train_batch_entropy"]) > 0:
        ax1.plot(history["train_batch_entropy"], label="Train Batch Entropy")
        if "val_batch_entropy" in history and len(history["val_batch_entropy"]) > 0:
            ax1.plot(history["val_batch_entropy"], label="Val Batch Entropy")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Batch Entropy (Higher = More Uniform)")  # Remove color='b'
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for entropy weight
    ax2 = ax1.twinx()
    # Plot entropy weight if it exists
    if "train_entropy_weight" in history and len(history["train_entropy_weight"]) > 0:
        ax2.plot(history["train_entropy_weight"], color=tol_red, label="Entropy Weight")
    ax2.set_ylabel("Entropy Loss Weight")

    # Combine legends properly
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Batch Entropy & Entropy Weight")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved loss curves to {save_path}")


def visualize_state_space(
    model,
    output_path=None,
    transformations=None,
    device: str | torch.device = "cpu",
    num_points=1000,
    num_states=None,
    soft=False,
    system_type=None,
    points=None,
    angles_degrees=None,
    bounds=None,
):
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

    if bounds is None:  # TBD this will not trigger anymore, with system registry bounds
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
    x_test = torch.tensor(np.column_stack((x1_flat, x2_flat)), dtype=torch.float32).to(
        device
    )

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
        state_prob_grid = (
            state_probs[:, state].cpu().numpy().reshape(num_points, num_points)
        )

        # Create the plot (using extent to set coordinate system)
        im = ax.imshow(
            state_prob_grid,
            extent=(
                plot_bounds[0][0],
                plot_bounds[0][1],
                plot_bounds[1][0],
                plot_bounds[1][1],
            ),
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

        # Set ticks and labels based on coordinate system
        if use_x_space:
            # Direct x-space plotting: use actual coordinate values
            num_ticks = 5
            ax.set_xticks(np.linspace(plot_bounds[0][0], plot_bounds[0][1], num_ticks))
            ax.set_yticks(np.linspace(plot_bounds[1][0], plot_bounds[1][1], num_ticks))
        else:
            # Z-space plotting with x-space labels: transform tick values
            ax.set_xticks(np.linspace(0, 1, 5))
            ax.set_yticks(np.linspace(0, 1, 5))

            # Create custom tick formatter functions for x-space labels
            def format_x1_ticks(z_val, pos):
                if z_val < 0 or z_val > 1:
                    return ""
                x_val = inverse_transforms[0](z_val)
                return f"{x_val:.1f}"

            def format_x2_ticks(z_val, pos):
                if z_val < 0 or z_val > 1:
                    return ""
                x_val = inverse_transforms[1](z_val)
                return f"{x_val:.1f}"

            # Apply formatters
            ax.xaxis.set_major_formatter(FuncFormatter(format_x1_ticks))
            ax.yaxis.set_major_formatter(FuncFormatter(format_x2_ticks))

        if system_type == "tech_substitution":
            steigung = 0.125
            x_line = np.array([0, 40])
            y_line = steigung * x_line
            ax.plot(
                x_line, x_line, color="#D81B60", linestyle="--", linewidth=0.6
            )  # diagonal basin boundary
            ax.plot(
                x_line, y_line, color="#D81B60", linestyle="--", linewidth=0.6
            )  # lower basin boundary

        # Overlay points and angles if provided
        if points_plot is not None and angles_degrees is not None:
            for i, (point_plot, angle_deg) in enumerate(
                zip(points_plot, angles_degrees)
            ):
                # White outline
                ax.plot(point_plot[0], point_plot[1], 'x', color='white', markersize=6, markeredgewidth=1.8, zorder=10)
                # Magenta X on top
                ax.plot(point_plot[0], point_plot[1], 'x', color='#D81B60', markersize=6, markeredgewidth=1.0, zorder=11)

                # Draw angle line from edge to edge
                angle_rad = np.radians(angle_deg)
                px, py = point_plot[0], point_plot[1]

                # Line equation: x = px + t*cos(θ), y = py + t*sin(θ)
                # Find t values where line hits boundaries
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

                t_values = []
                # Left/right boundaries
                if abs(cos_a) > 1e-10:  # Avoid division by zero
                    t_values.extend(
                        [
                            (plot_bounds[0][0] - px) / cos_a,
                            (plot_bounds[0][1] - px) / cos_a,
                        ]
                    )
                # Top/bottom boundaries
                if abs(sin_a) > 1e-10:  # Avoid division by zero
                    t_values.extend(
                        [
                            (plot_bounds[1][0] - py) / sin_a,
                            (plot_bounds[1][1] - py) / sin_a,
                        ]
                    )

                # For each t, check if the resulting point is within bounds
                valid_intersections = []
                for t in t_values:
                    x_int = px + t * cos_a
                    y_int = py + t * sin_a
                    # Check if this intersection point is within the plot bounds
                    if (
                        plot_bounds[0][0] <= x_int <= plot_bounds[0][1]
                        and plot_bounds[1][0] <= y_int <= plot_bounds[1][1]
                    ):
                        valid_intersections.append((t, x_int, y_int))

                # Draw line between the two valid intersection points
                if len(valid_intersections) >= 2:
                    # Sort by t value to get the correct direction
                    valid_intersections.sort(key=lambda x: x[0])
                    t1, x1, y1 = valid_intersections[0]
                    t2, x2, y2 = valid_intersections[-1]
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color="#D81B60",
                        linestyle="--",
                        linewidth=0.6,
                    )

        # Set title
        ax.set_title(f"State {state + 1}", fontsize=14, fontweight="bold")

        # Set labels
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.2)

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("State Assignment Strength")

    # Hide unused subplots
    for i in range(num_states, rows * cols_per_row):
        row_idx = i // cols_per_row
        col_idx = i % cols_per_row
        axes[row_idx, col_idx].set_visible(False)

    # Adjust layout
    plt.subplots_adjust(wspace=0.6, hspace=0.5)

    # Save the figure if an output path is provided; otherwise, display it
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        print(f"Saved state visualization to {output_path}")
    else:
        plt.show()

    plt.close(fig)
    return fig, axes

def visualize_final_state_assignments(
    model,
    output_path,
    system_type,
    device: str | torch.device = "cpu",
    num_points=1000,
    num_states=None,
    visualization_style='scatter',
    point_size=10,
    jitter_scale=0.3,
    points=None,
    angles_degrees=None,
    bounds=None,
):
    """
    Visualize final state assignments by generating a grid of points.
    
    This function is designed to be called from the training loop.
    It generates a grid (num_points x num_points) and evaluates
    the model to get state assignments.
    
    For scatter plots, jitter is applied to make points look more natural.
    For region plots, a regular grid is used for smooth boundaries.
    
    Recommended settings:
    - Scatter: num_points=100 (10k points), jitter_scale=0.3, point_size=10
    - Regions: num_points=1000 (1M points), no jitter applied
    
    Args:
        model: Trained DRM model
        output_path: Path to save the visualization
        system_type: 'saddle_system' or 'tech_substitution' (string or enum)
        device: Device to run model on
        num_points: Number of points per dimension (creates num_points^2 total points)
        num_states: Number of discrete states (inferred from model if None)
        visualization_style: 'scatter' (plot points) or 'regions' (color regions)
        point_size: Size of scatter points (only used if style='scatter')
        jitter_scale: Amount of jitter for scatter plots (fraction of cell size, 0=no jitter)
        points: Saddle points for plotting (list of [x, y] coordinates)
        angles_degrees: Angles for saddle separatrices (list of degrees)
        bounds: Plotting bounds [(x1_min, x1_max), (x2_min, x2_max)]
    """
    
    # Paul Tol's muted color scheme (colorblind accessible, 8 colors for up to 8 states)
    tol_muted = ["#332288", "#DDCC77", "#117733", "#88CCEE", "#CC6677", "#44AA99", "#882255","#999933"]
    magenta = "#D81B60"

    # Convert system_type string to SystemType enum if needed
    if isinstance(system_type, str):
        system_type_enum = SystemType(system_type)
    else:
        system_type_enum = system_type
    
    # Get visualization bounds from system registry if not provided
    if bounds is None:
        bounds = get_visualization_bounds(system_type_enum)
    
    print(f"Generating grid with {num_points}x{num_points} = {num_points**2} points")
    print(f"Using bounds: {bounds}")
    
    # Generate grid of points in x-space
    x1_values = np.linspace(bounds[0][0], bounds[0][1], num_points)
    x2_values = np.linspace(bounds[1][0], bounds[1][1], num_points)
    x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
    x1_flat, x2_flat = x1_grid.flatten(), x2_grid.flatten()
    
    # Apply jitter for scatter plots to make them look more natural
    if visualization_style == 'scatter' and jitter_scale > 0:
        # Calculate cell sizes
        cell_size_x1 = (bounds[0][1] - bounds[0][0]) / num_points
        cell_size_x2 = (bounds[1][1] - bounds[1][0]) / num_points
        
        # Add random jitter (uniform within fraction of cell size)
        jitter_x1 = np.random.uniform(-jitter_scale * cell_size_x1 / 2, 
                                       jitter_scale * cell_size_x1 / 2, 
                                       size=x1_flat.shape)
        jitter_x2 = np.random.uniform(-jitter_scale * cell_size_x2 / 2, 
                                       jitter_scale * cell_size_x2 / 2, 
                                       size=x2_flat.shape)
        
        x1_flat = x1_flat + jitter_x1
        x2_flat = x2_flat + jitter_x2
        
        # Clip to bounds
        x1_flat = np.clip(x1_flat, bounds[0][0], bounds[0][1])
        x2_flat = np.clip(x2_flat, bounds[1][0], bounds[1][1])
        
        print(f"Applied jitter with scale {jitter_scale}")
    
    # Create input tensor
    grid_points = np.column_stack((x1_flat, x2_flat))
    x_test = torch.tensor(grid_points, dtype=torch.float32).to(device)
    
    # Get state assignments from model
    model.to(device)
    model.eval()
    with torch.no_grad():
        state_probs = model.get_state_probs(x_test, training=False, soft=False)
        states = state_probs.argmax(dim=1).cpu().numpy()
    
    # Infer number of states if not provided
    if num_states is None:
        num_states = state_probs.shape[1]
    
    print(f"Number of states: {num_states}")
    print(f"State distribution: {np.bincount(states)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot reference geometry based on system type
    system_type_str = system_type_enum.value if isinstance(system_type_enum, SystemType) else system_type_enum
    
    if system_type_str == "tech_substitution":
        # Plot nullclines for tech substitution
        steigung = 0.125
        x_line = np.array([bounds[0][0], bounds[0][1]])
        y_line = steigung * x_line
        ax.plot(
            x_line, x_line, 
            color=magenta, 
            linestyle="--", 
            linewidth=1.5,
            label="Basin boundaries",
            zorder=1
        )
        ax.plot(
            x_line, y_line, 
            color=magenta, 
            linestyle="--", 
            linewidth=1.5,
            zorder=1
        )
        
    elif system_type_str == "saddle_system":
        if points is not None and angles_degrees is not None:
            # Plot separatrices (lines through saddle points at given angles)
            for i, (point, angle_deg) in enumerate(zip(points, angles_degrees)):
                # Plot saddle point
                ax.plot(
                    point[0], point[1], 
                    'x', 
                    color='white', 
                    markersize=8, 
                    markeredgewidth=2.0, 
                    zorder=10
                )
                ax.plot(
                    point[0], point[1], 
                    'x', 
                    color=magenta, 
                    markersize=8, 
                    markeredgewidth=1.2, 
                    zorder=11
                )
                
                # Draw angle line from edge to edge
                angle_rad = np.radians(angle_deg)
                px, py = point[0], point[1]
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                
                # Find intersection with plot boundaries
                t_values = []
                # Left/right boundaries
                if abs(cos_a) > 1e-10:
                    t_values.extend([
                        (bounds[0][0] - px) / cos_a,
                        (bounds[0][1] - px) / cos_a,
                    ])
                # Top/bottom boundaries
                if abs(sin_a) > 1e-10:
                    t_values.extend([
                        (bounds[1][0] - py) / sin_a,
                        (bounds[1][1] - py) / sin_a,
                    ])
                
                # Find valid intersections within bounds
                valid_intersections = []
                for t in t_values:
                    x_int = px + t * cos_a
                    y_int = py + t * sin_a
                    if (bounds[0][0] <= x_int <= bounds[0][1] and 
                        bounds[1][0] <= y_int <= bounds[1][1]):
                        valid_intersections.append((x_int, y_int))
                
                # Draw line between the two intersection points
                if len(valid_intersections) >= 2:
                    x_coords = [valid_intersections[0][0], valid_intersections[1][0]]
                    y_coords = [valid_intersections[0][1], valid_intersections[1][1]]
                    ax.plot(
                        x_coords, y_coords,
                        color=magenta,
                        linestyle="--",
                        linewidth=1.5,
                        zorder=1
                    )
    
    # Visualize state assignments
    if visualization_style == 'scatter':
        # Scatter plot - each observation as a colored point
        for state in range(num_states):
            mask = states == state
            if np.sum(mask) > 0:
                ax.scatter(
                    grid_points[mask, 0],
                    grid_points[mask, 1],
                    c=tol_muted[state % len(tol_muted)],
                    s=point_size,
                    alpha=0.8,
                    label=f"State {state + 1}",
                    zorder=2
                )
    
    elif visualization_style == 'regions':
        # Region coloring - use the grid we already generated
        # Reshape states to match the grid
        grid_states = states.reshape(num_points, num_points)
        
        # Create custom colormap from tol_muted colors
        colors_used = [tol_muted[i % len(tol_muted)] for i in range(num_states)]
        cmap = ListedColormap(colors_used)
        
        # Plot colored regions
        im = ax.imshow(
            grid_states,
            extent=(bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]),
            origin='lower',
            cmap=cmap,
            alpha=0.9,
            vmin=0,
            vmax=num_states - 1,
            zorder=0
        )
        
        # Create legend manually for regions
        legend_elements = [
            Patch(facecolor=tol_muted[i % len(tol_muted)], 
                  label=f"State {i + 1}") 
            for i in range(num_states)
        ]
        ax.legend(handles=legend_elements, loc='best')
    
    else:
        raise ValueError(f"Unknown visualization_style: {visualization_style}")
    
    # Set labels and limits
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_aspect('equal', adjustable='box')
    
    # Add legend (only if scatter, regions already have manual legend)
    if visualization_style == 'scatter':
        ax.legend(loc='upper right', framealpha=0.9)
    
    # Add title
    #ax.set_title(f'Final State Assignments - {visualization_style.capitalize()} ({system_type_str})', fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    plt.close()

def visualize_final_state_assignments_from_pkl(
    pkl_path,
    output_path,
    system_type,
    db_path=None,
    visualization_style='scatter',
    point_size=10,
    grid_resolution=200,
):
    """
    Visualize final state assignments from saved pkl file.
    
    Args:
        pkl_path: Path to state_assignments_{run_id}.pkl file
        output_path: Path to save the visualization
        system_type: 'saddle_system' or 'tech_substitution'
        db_path: Path to database (required for saddle_system to get separatrices)
        visualization_style: 'scatter' (plot points) or 'regions' (color regions)
        point_size: Size of scatter points (only used if style='scatter')
        grid_resolution: Grid resolution for region coloring (only used if style='regions')
    """
    
    # Paul Tol's muted color scheme (colorblind accessible, 8 colors for up to 8 states)
    tol_muted = ["#332288", "#DDCC77", "#117733", "#88CCEE", "#CC6677", "#44AA99", "#882255", "#999933"]
    magenta = "#D81B60"

    # Load the pkl file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    observations = data['observations']  # Shape: (N, 2)
    states = data['states']  # Shape: (N,)
    num_states = data['num_states']
    
    print(f"Loaded {len(observations)} observations")
    print(f"Number of states: {num_states}")
    print(f"State distribution: {np.bincount(states)}")
    
    # Get visualization bounds from system registry
    # Convert system_type string to SystemType enum if needed
    if isinstance(system_type, str):
        system_type_enum = SystemType(system_type)
    else:
        system_type_enum = system_type

    # Get visualization bounds from system registry
    bounds = get_visualization_bounds(system_type_enum)
    print(f"Using bounds: {bounds}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot reference geometry based on system type
    if system_type == "tech_substitution":
        # Plot nullclines for tech substitution
        steigung = 0.125
        x_line = np.array([bounds[0][0], bounds[0][1]])
        y_line = steigung * x_line
        ax.plot(
            x_line, x_line, 
            color=magenta, 
            linestyle="--", 
            linewidth=1.5,
            label="Basin boundaries",
            zorder=1
        )
        ax.plot(
            x_line, y_line, 
            color=magenta, 
            linestyle="--", 
            linewidth=1.5,
            zorder=1
        )
        
    elif system_type == "saddle_system":
        if db_path is None:
            raise ValueError("db_path is required for saddle_system to plot separatrices")
        
        # Get saddle configuration
        saddle_config = get_saddle_configuration(db_path, verbose=True)
        if saddle_config:
            points = saddle_config["saddle_points"]
            angles_degrees = saddle_config["angles_degrees"]
            
            # Plot separatrices (lines through saddle points at given angles)
            for i, (point, angle_deg) in enumerate(zip(points, angles_degrees)):
                # Plot saddle point
                ax.plot(
                    point[0], point[1], 
                    'x', 
                    color='white', 
                    markersize=8, 
                    markeredgewidth=2.0, 
                    zorder=10
                )
                ax.plot(
                    point[0], point[1], 
                    'x', 
                    color=magenta, 
                    markersize=8, 
                    markeredgewidth=1.2, 
                    zorder=11
                )
                
                # Draw angle line from edge to edge
                angle_rad = np.radians(angle_deg)
                px, py = point[0], point[1]
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                
                # Find intersection with plot boundaries
                t_values = []
                # Left/right boundaries
                if abs(cos_a) > 1e-10:
                    t_values.extend([
                        (bounds[0][0] - px) / cos_a,
                        (bounds[0][1] - px) / cos_a,
                    ])
                # Top/bottom boundaries
                if abs(sin_a) > 1e-10:
                    t_values.extend([
                        (bounds[1][0] - py) / sin_a,
                        (bounds[1][1] - py) / sin_a,
                    ])
                
                # Find valid intersections within bounds
                valid_intersections = []
                for t in t_values:
                    x_int = px + t * cos_a
                    y_int = py + t * sin_a
                    if (bounds[0][0] <= x_int <= bounds[0][1] and 
                        bounds[1][0] <= y_int <= bounds[1][1]):
                        valid_intersections.append((x_int, y_int))
                
                # Draw line between the two intersection points
                if len(valid_intersections) >= 2:
                    x_coords = [valid_intersections[0][0], valid_intersections[1][0]]
                    y_coords = [valid_intersections[0][1], valid_intersections[1][1]]
                    label = "Basin boundaries" if i == 0 else None
                    ax.plot(
                        x_coords, y_coords,
                        color=magenta,
                        linestyle="--",
                        linewidth=1.5,
                        label=label,
                        zorder=1
                    )
    
    # Visualize state assignments
    if visualization_style == 'scatter':
        # Scatter plot - each observation as a colored point
        for state in range(num_states):
            mask = states == state
            if np.sum(mask) > 0:
                ax.scatter(
                    observations[mask, 0],
                    observations[mask, 1],
                    c=tol_muted[state % len(tol_muted)],
                    s=point_size,
                    alpha=0.8,
                    label=f"State {state + 1}",
                    zorder=2
                )
    
    elif visualization_style == 'regions':
        # Region coloring - create a grid and assign colors based on nearest observation
        from scipy.spatial import KDTree
        
        x1_grid = np.linspace(bounds[0][0], bounds[0][1], grid_resolution)
        x2_grid = np.linspace(bounds[1][0], bounds[1][1], grid_resolution)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        grid_points = np.column_stack([X1.ravel(), X2.ravel()])
        
        # For each grid point, find nearest observation and assign its state
        tree = KDTree(observations)
        _, nearest_indices = tree.query(grid_points)
        grid_states = states[nearest_indices].reshape(grid_resolution, grid_resolution)
        
        # Create custom colormap from tol_muted colors
        from matplotlib.colors import ListedColormap
        colors_used = [tol_muted[i % len(tol_muted)] for i in range(num_states)]
        cmap = ListedColormap(colors_used)
        
        # Plot colored regions
        im = ax.imshow(
            grid_states,
            extent=(bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]),
            origin='lower',
            cmap=cmap,
            alpha=0.9,
            vmin=0,
            vmax=num_states - 1,
            zorder=0
        )
        
        # Create legend manually for regions
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=tol_muted[i % len(tol_muted)], 
                  label=f"State {i + 1}") 
            for i in range(num_states)
        ]
        ax.legend(handles=legend_elements, loc='best')
    
    else:
        raise ValueError(f"Unknown visualization_style: {visualization_style}")
    
    # Set labels and limits
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_xlim(bounds[0][0], bounds[0][1])
    ax.set_ylim(bounds[1][0], bounds[1][1])
    ax.set_aspect('equal', adjustable='box')
    
    # Add legend (only if scatter, regions already have manual legend)
    if visualization_style == 'scatter':
        ax.legend(loc='best', framealpha=0.9)
    
    # Add title
    #ax.set_title(f'Final State Assignments ({system_type})', fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    plt.close()

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
    tol_muted = [
        "#CC6677",
        "#332288",
        "#DDCC77",
        "#117733",
        "#88CCEE",
        "#882255",
        "#44AA99",
        "#999933",
    ]

    # Convert to arrays for plotting
    epochs = [m["epoch"] for m in all_metrics]

    # Create analysis plots (2x2 layout)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. State usage over time
    ax = axes[0, 0]
    for state_idx in range(num_states):
        usage_values = [m.get(f"state_{state_idx}_usage", 0) * 100 for m in all_metrics]
        color = tol_muted[state_idx % len(tol_muted)]
        ax.plot(
            epochs,
            usage_values,
            label=f"State {state_idx + 1}",
            marker="o",
            linewidth=2,
            color=color,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dominant State (%)")
    ax.set_title("Dominant State Assignments Over Training")
    ax.legend()
    ax.grid(True, alpha=0.4)

    # 2. Assignment sharpness over time
    ax = axes[0, 1]
    sharpness_mean = [m.get("sharpness_mean", 0) for m in all_metrics]
    sharpness_std = [m.get("sharpness_std", 0) for m in all_metrics]

    ax.plot(epochs, sharpness_mean, color="blue", linewidth=2, label="Mean")
    ax.fill_between(
        epochs,
        np.array(sharpness_mean) - np.array(sharpness_std),
        np.array(sharpness_mean) + np.array(sharpness_std),
        alpha=0.3,
        color="blue",
        label="± 1 std",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Per-Point Entropy (Sharpness)")
    ax.set_title("Assignment Sharpness Over Training\n(Lower = More Discrete)")
    ax.legend()
    ax.grid(True, alpha=0.4)

    # 3. Mean probabilities per state
    ax = axes[1, 0]
    for state_idx in range(num_states):
        mean_values = [m.get(f"state_{state_idx}_mean", 0) for m in all_metrics]
        color = tol_muted[state_idx % len(tol_muted)]
        ax.plot(
            epochs,
            mean_values,
            label=f"State {state_idx + 1}",
            marker="o",
            linewidth=2,
            color=color,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Probability")
    ax.set_title("Average State Assignment Strength")
    ax.grid(True, alpha=0.4)

    # 4. Stability metrics
    ax = axes[1, 1]
    stability_epochs = []
    dominant_stability = []

    for m in all_metrics:
        if "dominant_stability" in m:
            stability_epochs.append(m["epoch"])
            dominant_stability.append(m["dominant_stability"])

    if stability_epochs:
        ax.plot(
            stability_epochs,
            dominant_stability,
            color="#AA4499",
            marker="o",
            linewidth=2,
            label="Dominant State Stability",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Stability (%)")
    ax.set_title("Assignment Stability Between Epochs")
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_ylim(50, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {"epochs_analyzed": len(all_metrics), "num_states": num_states}


def create_gif_from_data_frames(data_frame_paths, output_path, gif_duration=250):
    """
    Create GIF from saved data frame paths instead of PNG paths.
    """
    try:

        gif_path = f"{output_path}_animation.gif"

        with imageio.get_writer(gif_path, mode="I", duration=gif_duration) as writer:
            for data_path in data_frame_paths:
                # Load data frame
                with open(data_path, "rb") as f:
                    frame_data = pickle.load(f)

                # Create visualization from data
                png_data = create_png_from_frame_data(frame_data)
                writer.append_data(png_data)  # type: ignore[attr-defined]

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
    grid_points = frame_data["grid_points"]
    state_probs = frame_data["state_probs"]
    epoch = frame_data["epoch"]
    num_states = frame_data["num_states"]
    bounds = frame_data["bounds"]
    grid_size = frame_data["grid_size"]

    # Calculate grid layout
    cols_per_row = 2
    rows = (num_states + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(12, 6 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for state_idx in range(num_states):
        row_idx = state_idx // cols_per_row
        col_idx = state_idx % cols_per_row
        ax = axes[row_idx, col_idx]

        # Reshape probabilities to grid
        state_grid = state_probs[:, state_idx].reshape(grid_size, grid_size)

        # Create heatmap
        im = ax.imshow(
            state_grid,
            extent=(bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]),
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="bilinear",
        )

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"State {state_idx + 1}")
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("State Assignment Strength")

    # Hide unused subplots
    for idx in range(num_states, rows * cols_per_row):
        row_idx = idx // cols_per_row
        col_idx = idx % cols_per_row
        axes[row_idx, col_idx].set_visible(False)

    # Overall title
    fig.suptitle(f"State Space Visualization - Epoch {epoch}", fontsize=14)
    plt.tight_layout()

    # Convert to image data instead of saving to file
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)

    # Read image data
    image_data = imageio.imread(buf)

    plt.close()
    buf.close()

    return image_data


def analyze_discrete_state_transitions(
    model, control_values, device: str | torch.device = "cpu", system_type=None
):
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
        if system_type == "saddle_system":
            # For categorical controls, create one-hot encoding
            control_batch = torch.zeros(
                (num_states, control_dim), dtype=torch.float32, device=device
            )
            control_batch[:, control_value] = 1.0
        else:
            # For continuous controls, handle both scalar and multi-dimensional cases
            if isinstance(control_value, (list, tuple, np.ndarray)):
                # Multi-dimensional control (e.g., social_tipping with [b, c, f, g])
                control_tensor = torch.tensor(
                    control_value, dtype=torch.float32, device=device
                )
                # Expand to batch size: (num_states, control_dim)
                control_batch = control_tensor.unsqueeze(0).expand(num_states, -1)
            else:
                # Scalar control (e.g., tech_substitution)
                control_batch = torch.full(
                    (num_states, control_dim),
                    control_value,
                    dtype=torch.float32,
                    device=device,
                )

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


def visualize_transition_matrices(
    transition_matrices, control_values, output_path=None
):
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
    fig, axes = plt.subplots(
        1, len(control_values), figsize=(len(control_values) * 6, 5)
    )
    if len(control_values) == 1:
        axes = [axes]

    for i, control in enumerate(control_values):
        ax = axes[i]
        control_key = get_control_key(control)  # Convert to proper key format
        matrix = transition_matrices[control_key]

        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar=True,
        )
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
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
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

        df = pd.DataFrame(
            matrix,
            index=[f"State {i+1}" for i in range(num_states)],
            columns=[f"State {i+1}" for i in range(num_states)],
        )
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
    dot = graphviz.Digraph(comment="Model Architecture", format="png", engine="dot")

    # Set graph attributes for better appearance
    dot.attr("graph", rankdir="TB", splines="ortho", nodesep="0.5", ranksep="0.7")
    dot.attr("node", shape="box", style="filled", fontname="Arial", fontsize="12")
    dot.attr("edge", fontname="Arial", fontsize="10")

    # Helper functions to extract info from model
    def get_module_name(module_path):
        """Convert module path to a readable name"""
        return module_path.replace(".", "_")

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
            "Linear": "skyblue",
            "ReLU": "lightgrey",
            "Softmax": "lightgrey",
            "Sequential": "lightblue",
            "ModuleList": "lightblue",
            "ControlGate": "palegreen",
            "StandardPredictor": "palegreen",
            "BilinearPredictor": "palegreen",
            "ControlGatePredictor": "palegreen",
            "Bilinear": "lightyellow",  # Special color for bilinear
            "DiscreteRepresentationsModel": "lightgrey",
        }
        # Check if any substring matches
        for key in colors:
            if key in module_type:
                return colors[key]
        return "white"  # Default color

    # Track modules to create clusters
    clusters = {"encoder": [], "predictor": [], "value_net": []}

    all_modules = {}
    connections = defaultdict(list)
    custom_connections = []  # For special non-sequential connections

    # Add main model node
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    main_node_id = "model"
    dot.node(
        main_node_id,
        f"Discrete Representations Model\n{model.num_states} states\n{total_params:,} parameters",
        fillcolor="lightgrey",
    )

    # Extract model structure
    def extract_model_structure(model, prefix=""):
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
                "name": name,
                "path": module_path,
                "type": module_type,
                "label": label,
                "color": get_color(module_type),
                "module": child,  # Store reference to actual module
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
                        connections[child_ids[i]].append(child_ids[i + 1])
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
        "x_input": {
            "label": "x (observation)",
            "shape": "ellipse",
            "color": "lightgrey",
        },
        "c_input": {"label": "c (control)", "shape": "ellipse", "color": "lightgrey"},
        "y_input": {"label": "y (next obs)", "shape": "ellipse", "color": "lightgrey"},
        "v_true_input": {"label": "v_true", "shape": "ellipse", "color": "lightgrey"},
        "s_x_output": {
            "label": "s_x (state probs)",
            "shape": "ellipse",
            "color": "lightgrey",
        },
        "s_y_output": {
            "label": "s_y (true next state)",
            "shape": "ellipse",
            "color": "lightgrey",
        },
        "s_y_pred_output": {
            "label": "s_y_pred (predicted)",
            "shape": "ellipse",
            "color": "lightgrey",
        },
        "v_pred_output": {
            "label": "v_pred (value)",
            "shape": "ellipse",
            "color": "lightgrey",
        },
        "concat_node": {
            "label": "concatenated features",
            "shape": "ellipse",
            "color": "lightyellow",
        },
    }

    # Add special nodes
    for node_id, node_info in special_nodes.items():
        if node_id == "concat_node" and not isinstance(
            model.predictor, StandardPredictor
        ):
            # Only add concat node for StandardPredictor
            continue
        dot.node(
            node_id,
            node_info["label"],
            shape=node_info["shape"],
            fillcolor=node_info["color"],
        )

    # Analyze special data flows for different predictor types
    if hasattr(model, "predictor"):
        if isinstance(model.predictor, BilinearPredictor):
            # Add custom data flow connections for BilinearPredictor
            predictor = model.predictor
            control_encoder_id = get_module_name("predictor.control_encoder")
            interaction_id = get_module_name("predictor.interaction")
            hidden_id = get_module_name("predictor.hidden")
            output_id = get_module_name("predictor.output")

            # Create special path descriptions
            custom_connections.extend(
                [
                    (
                        "c_input",
                        control_encoder_id,
                        {"label": "control input", "color": "blue"},
                    ),
                    (
                        control_encoder_id,
                        interaction_id,
                        {"label": "control features", "color": "blue"},
                    ),
                    (
                        "s_x_output",
                        interaction_id,
                        {"label": "state input", "color": "green"},
                    ),
                    (
                        interaction_id,
                        hidden_id,
                        {"label": "interaction", "color": "red"},
                    ),
                    (
                        hidden_id,
                        output_id,
                        {"label": "hidden features", "color": "purple"},
                    ),
                    (
                        output_id,
                        "s_y_pred_output",
                        {"label": "logits->softmax", "color": "orange"},
                    ),
                ]
            )

        elif isinstance(model.predictor, StandardPredictor):
            # Add custom data flow connections for StandardPredictor
            control_encoder_id = get_module_name("predictor.control_encoder")
            predictor_id = get_module_name("predictor.predictor")

            # Get the first layer of the sequential predictor
            first_layer_id = None
            for module_id in all_modules:
                if module_id.startswith("predictor_predictor_0"):
                    first_layer_id = module_id
                    break

            # Create special path descriptions for StandardPredictor
            custom_connections.extend(
                [
                    (
                        "c_input",
                        control_encoder_id,
                        {"label": "control input", "color": "blue"},
                    ),
                    (
                        control_encoder_id,
                        "concat_node",
                        {"label": "encoded control", "color": "blue"},
                    ),
                    (
                        "s_x_output",
                        "concat_node",
                        {"label": "state probs", "color": "green"},
                    ),
                    (
                        "concat_node",
                        first_layer_id if first_layer_id else predictor_id,
                        {"label": "concatenated", "color": "red"},
                    ),
                    (
                        get_module_name("predictor.predictor"),
                        "s_y_pred_output",
                        {"label": "output", "color": "orange"},
                    ),
                ]
            )

    # Create encoder cluster
    subgraph = dot.subgraph(name="cluster_encoder")
    if subgraph is not None:
        with subgraph as c:
            c.attr(label="Encoder", style="filled", color="lightblue", fillcolor="azure")

            # Add encoder modules
            for module_id in clusters["encoder"]:
                if module_id in all_modules:
                    module = all_modules[module_id]
                    c.node(module_id, module["label"], fillcolor=module["color"])

            # Add connections between encoder modules
            for src, dests in connections.items():
                if src in clusters["encoder"]:
                    for dest in dests:
                        if dest in clusters["encoder"]:
                            c.edge(src, dest)

            # Connect inputs/outputs
            c.edge("x_input", clusters["encoder"][0] if clusters["encoder"] else "encoder")
            c.edge(
                clusters["encoder"][-1] if clusters["encoder"] else "encoder", "s_x_output"
            )

            # Add y path for training (dashed)
            c.edge(
                "y_input",
                clusters["encoder"][0] if clusters["encoder"] else "encoder",
                style="dashed",
                label="shared weights",
            )
            c.edge(
                clusters["encoder"][-1] if clusters["encoder"] else "encoder",
                "s_y_output",
                style="dashed",
            )

    # Create predictor cluster with special handling for different predictor types
    subgraph = dot.subgraph(name="cluster_predictor")
    if subgraph is not None:
        with subgraph as c:
            predictor_type = model.predictor.__class__.__name__
            c.attr(
                label=f"{predictor_type}",
                style="filled",
                color="lightgreen",
                fillcolor="mintcream",
            )

            # Add predictor modules
            for module_id in clusters["predictor"]:
                if module_id in all_modules:
                    module = all_modules[module_id]
                    c.node(module_id, module["label"], fillcolor=module["color"])

            # For BilinearPredictor, create a custom layout
            if isinstance(model.predictor, BilinearPredictor):
                # Highlight the interaction module
                interaction_id = get_module_name("predictor.interaction")
                if interaction_id in all_modules:
                    c.node(
                        interaction_id,
                        all_modules[interaction_id]["label"],
                        shape="Mrecord",
                        fillcolor="gold",
                    )

            # For StandardPredictor, add the concatenation node
            elif isinstance(model.predictor, StandardPredictor):
                # Concatenation happens outside any specific module
                c.node(
                    "concat_node",
                    "Concatenate\ns_x + encoded c",
                    shape="Mrecord",
                    fillcolor="gold",
                )

            # Only add standard connections for the remaining modules
            # (not for BilinearPredictor or StandardPredictor where we use custom connections)
            if not isinstance(model.predictor, (BilinearPredictor, StandardPredictor)):
                # Add connections between predictor modules
                for src, dests in connections.items():
                    if src in clusters["predictor"]:
                        for dest in dests:
                            if dest in clusters["predictor"]:
                                c.edge(src, dest)

                # Connect inputs/outputs
                c.edge(
                    "s_x_output",
                    clusters["predictor"][0] if clusters["predictor"] else "predictor",
                )
                c.edge(
                    "c_input",
                    clusters["predictor"][0] if clusters["predictor"] else "predictor",
                )
                c.edge(
                    clusters["predictor"][-1] if clusters["predictor"] else "predictor",
                    "s_y_pred_output",
                )

    # Create value network cluster
    subgraph = dot.subgraph(name="cluster_value")
    if subgraph is not None:
        with subgraph as c:
            c.attr(
                label="Value Network", style="filled", color="salmon", fillcolor="seashell"
            )

            # Add value network modules
            for module_id in clusters["value_net"]:
                if module_id in all_modules:
                    module = all_modules[module_id]
                    c.node(module_id, module["label"], fillcolor=module["color"])

            # Add connections between value network modules
            for src, dests in connections.items():
                if src in clusters["value_net"]:
                    for dest in dests:
                        if dest in clusters["value_net"]:
                            c.edge(src, dest)

            # Connect inputs/outputs
            c.edge(
                "s_y_pred_output",
                clusters["value_net"][0] if clusters["value_net"] else "value_net",
            )
            c.edge(
                clusters["value_net"][-1] if clusters["value_net"] else "value_net",
                "v_pred_output",
            )

    # Add custom connections for non-sequential flows
    for src, dest, attrs in custom_connections:
        dot.edge(
            src,
            dest,
            color=attrs.get("color", "blue"),
            label=attrs.get("label", ""),
            style=attrs.get("style", "dashed"),
        )

    # Add loss nodes
    dot.node(
        "state_loss",
        "State Loss\nKL Divergence",
        shape="diamond",
        fillcolor="lightpink",
    )
    dot.node("value_loss", "Value Loss\nMSE", shape="diamond", fillcolor="lightpink")

    # Connect for training path
    dot.edge("s_y_output", "state_loss")
    dot.edge("s_y_pred_output", "state_loss")
    dot.edge("v_pred_output", "value_loss")
    dot.edge("v_true_input", "value_loss")

    # Render the graph
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        dot.render(output_path.replace(".png", ""), format="png", cleanup=True)
        print(f"Model architecture visualization saved to {output_path}.png")
        return output_path + ".png"
    except Exception as e:
        print(f"Error rendering graph: {e}")
        # Try to save in current directory as fallback
        fallback_path = f"model_arch_{model.num_states}_states"
        dot.render(fallback_path, format="png", cleanup=True)
        print(f"Fallback visualization saved to {fallback_path}.png")
        return fallback_path + ".png"


def plot_softmax_rank_evolution(history, save_path):
    """
    Plot evolution of softmax rank metrics over training.
    Uses Paul Tol's muted color scheme for colorblind accessibility.
    Uses global normalization for singular values (like the paper).

    Args:
        history: Training history containing softmax_rank_metrics
        save_path: Path to save the plot
    """

    if "softmax_rank_metrics" not in history or not history["softmax_rank_metrics"]:
        print("No softmax rank metrics found in history")
        return

    # Paul Tol's muted color scheme for colorblind accessibility
    tol_muted = [
        "#CC6677",
        "#332288",
        "#DDCC77",
        "#117733",
        "#88CCEE",
        "#882255",
        "#44AA99",
        "#999933",
    ]

    metrics_list = history["softmax_rank_metrics"]
    epochs = [m["epoch"] for m in metrics_list]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ========================================================================
    # Plot 1: Rank Evolution During Training
    # ========================================================================
    ax = axes[0, 0]
    hidden_ranks = [m["hidden_rank"] for m in metrics_list]
    logit_ranks = [m["logit_rank"] for m in metrics_list]
    softmax_ranks = [m["softmax_rank"] for m in metrics_list]

    ax.plot(
        epochs,
        hidden_ranks,
        "o-",
        label="Hidden Layer (32-dim)",
        linewidth=2,
        markersize=4,
        color=tol_muted[0],
    )
    ax.plot(
        epochs,
        logit_ranks,
        "s-",
        label="Logit Layer (4-dim)",
        linewidth=2,
        markersize=4,
        color=tol_muted[1],
    )
    ax.plot(
        epochs,
        softmax_ranks,
        "^-",
        label="Post-Softmax (4-dim)",
        linewidth=2,
        markersize=4,
        color=tol_muted[2],
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Numerical Rank")
    ax.set_title("Rank Evolution During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)  # Start y-axis at 0 as requested

    # ========================================================================
    # Plot 2: Norm Evolution (Hidden A, Logit M, Softmax A)
    # ========================================================================
    ax = axes[0, 1]
    hidden_norms = [m["hidden_frobenius_norm"] for m in metrics_list]  # ||A₃||_F
    logit_norms = [m["logit_frobenius_norm"] for m in metrics_list]  # ||M₄||_F
    softmax_norms = [m["softmax_frobenius_norm"] for m in metrics_list]  # ||A₄||_F

    ax.plot(
        epochs,
        hidden_norms,
        "o-",
        label="Hidden ||A₃||_F",
        linewidth=2,
        markersize=4,
        color=tol_muted[0],
    )
    ax.plot(
        epochs,
        logit_norms,
        "s-",
        label="Logit ||M₄||_F",
        linewidth=2,
        markersize=4,
        color=tol_muted[1],
    )
    ax.plot(
        epochs,
        softmax_norms,
        "^-",
        label="Softmax ||A₄||_F",
        linewidth=2,
        markersize=4,
        color=tol_muted[2],
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Frobenius Norm")
    ax.set_title("Norm Evolution During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")  # Keep log scale for norms (makes sense here)

    # ========================================================================
    # Plot 3: Logit Singular Values Evolution (GLOBAL NORMALIZATION, LINEAR SCALE)
    # ========================================================================
    ax = axes[1, 0]

    # Plot first 4 globally normalized singular values for logits
    colors_sv = [tol_muted[0], tol_muted[1], tol_muted[2], tol_muted[3]]
    for i in range(4):
        sv_global_key = f"logit_sv_global_norm_{i}"
        if (
            sv_global_key in metrics_list[0]
        ):  # Check if globally normalized values exist
            sv_values = [m.get(sv_global_key, 0) for m in metrics_list]
            ax.plot(
                epochs,
                sv_values,
                "o-",
                label=f"σ_{i+1}",
                color=colors_sv[i],
                linewidth=2,
                markersize=3,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Globally Normalized Singular Value")
    ax.set_title("Logit Singular Values Evolution")
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
        sv_global_key = f"hidden_sv_global_norm_{i}"
        if (
            sv_global_key in metrics_list[0]
        ):  # Check if globally normalized values exist
            sv_values = [m.get(sv_global_key, 0) for m in metrics_list]
            ax.plot(
                epochs,
                sv_values,
                "o-",
                label=f"σ_{i+1}",
                color=colors_sv[i],
                linewidth=2,
                markersize=3,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Globally Normalized Singular Value")
    ax.set_title("Hidden Layer Singular Values Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)  # Linear scale from 0 to 1

    # ========================================================================
    # Final styling and save
    # ========================================================================
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved softmax rank evolution plot to {save_path}")


def plot_probing_evolution(history, save_path):
    """
    Plot evolution of layer probing discrete accuracy over training.
    Uses Paul Tol's muted color scheme for colorblind accessibility.
    
    Args:
        history: Training history containing intermediate_probing metrics
        save_path: Path to save the plot
    """
    
    if "intermediate_probing" not in history or not history["intermediate_probing"]:
        print("No intermediate probing metrics found in history")
        return
    
    # Paul Tol's muted color scheme
    tol_blue = "#332288"
    
    probing_list = history["intermediate_probing"]
    epochs = [p["epoch"] for p in probing_list]
    accuracies = [p["discrete_accuracy"] for p in probing_list]
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot discrete accuracy evolution
    ax.plot(
        epochs,
        accuracies,
        "o-",
        label="Discrete Accuracy (Validation)",
        linewidth=2,
        markersize=6,
        color=tol_blue,
    )
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Discrete Accuracy", fontsize=12)
    ax.set_title("Layer Probing: Discrete Accuracy Evolution", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)  # Accuracy is between 0 and 1
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved probing evolution plot to: {save_path}")

### MDP

def extract_mdp_transitions_fixed(model, num_actions=2, device: str | torch.device = "cpu"):
    """
    Extract MDP transition probabilities and state values from a trained DRM model.
    Fixed version that handles control dimensions correctly.
    
    Args:
        model: Trained DiscreteRepresentationsModel
        num_actions: Number of discrete actions
        device: Device to run computations on
    
    Returns:
        tuple: (transition_probs, state_values)
            - transition_probs: np.array of shape (num_states, num_actions, num_states)
            - state_values: np.array of shape (num_states,)
    """
    model.eval()
    num_states = model.num_states
    
    # Infer control_dim from the model's predictor
    # For categorical controls (like saddle_system), control_dim = num_actions
    if hasattr(model.predictor, 'control_encoder'):
        if isinstance(model.predictor.control_encoder, nn.Sequential):
            control_dim = model.predictor.control_encoder[0].in_features
        else:
            control_dim = model.predictor.control_encoder.in_features
    else:
        control_dim = 1
    
    print(f"[DEBUG] Model has control_dim={control_dim}, visualizing num_actions={num_actions}")
    
    # Create one-hot encoded states
    states = torch.eye(num_states, device=device)
    
    # Initialize transition probability matrix: P(s'|s,a)
    transition_probs = np.zeros((num_states, num_actions, num_states))
    
    with torch.no_grad():
        # For each state and action, get next state distribution
        for state_idx in range(num_states):
            state = states[state_idx:state_idx+1]  # Shape: (1, num_states)
            
            for action_idx in range(num_actions):
                # Create control input with CORRECT dimension
                # For categorical controls (saddle_system): one-hot encode the action
                # For continuous controls: use scalar value
                
                if control_dim == num_actions:
                    # Categorical control: one-hot encoding
                    control = torch.zeros(1, control_dim, device=device)
                    control[0, action_idx] = 1.0
                elif control_dim == 1:
                    # Single continuous control
                    control = torch.tensor([[float(action_idx)]], device=device)
                else:
                    # Multi-dimensional continuous control
                    # Use normalized action index
                    control = torch.zeros(1, control_dim, device=device)
                    control[0, 0] = action_idx / max(1, num_actions - 1)
                
                # Get predicted next state probabilities
                next_state_probs = model.predict_next_state(state, control)
                
                # Store in transition matrix
                transition_probs[state_idx, action_idx] = next_state_probs.cpu().numpy()[0]
        
        # Get state values
        state_values_tensor = model.compute_values_for_all_states()
        state_values = state_values_tensor.cpu().numpy()  # Don't squeeze yet
        
        print(f"[DEBUG] Raw state_values shape: {state_values.shape}")
        print(f"[DEBUG] Raw state_values:\n{state_values}")
        
        # Handle different value formats:
        # - 1D: direct values (e.g., market share)
        # - 2D: [sin(θ), cos(θ)] representation for angles
        if state_values.ndim > 1 and state_values.shape[1] == 2:
            # This is likely angle representation: [sin(θ), cos(θ)]
            # Reconstruct angle using atan2
            sin_vals = state_values[:, 0]
            cos_vals = state_values[:, 1]
            angles_rad = np.arctan2(sin_vals, cos_vals)
            # Convert to [0, 2π] range, then to [0, 1] for consistent scaling
            angles_rad = np.where(angles_rad < 0, angles_rad + 2*np.pi, angles_rad)
            state_values = angles_rad / (2 * np.pi)  # Normalize to [0, 1]
            print(f"[DEBUG] Converted 2D [sin, cos] to angles (normalized): {state_values}")
            print(f"[DEBUG] Angles in degrees: {state_values * 360}")
        else:
            # Single value per state or already 1D
            state_values = state_values.squeeze()
            if state_values.ndim > 1:
                # Multi-dimensional but not angle format - take first dimension
                state_values = state_values[:, 0]
            print(f"[DEBUG] Using values directly: {state_values}")
    
    return transition_probs, state_values


def get_mdp_text_color(hex_color):
    """Return 'white' for dark colors, 'black' for light colors"""
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return 'white' if luminance < 0.5 else 'black'


def visualize_mdp_matplotlib(
    model=None,
    transition_probs=None,
    state_values=None,
    output_path=None,
    device: str | torch.device = "cpu",
    num_actions=2,
    value_format='angle',
    title='MDP Visualization',
    threshold=0.02,
):
    """
    Visualize the learned MDP with matplotlib using curved arrows.
    This is the NEW matplotlib-based visualization (different from old graphviz one).
    
    Args:
        model: Trained DiscreteRepresentationsModel
        transition_probs: Pre-computed transition probabilities (optional)
        state_values: Pre-computed state values (optional)
        output_path: Path to save the visualization
        device: Device for model inference
        num_actions: Number of discrete actions
        value_format: 'angle' for degrees, 'float' for raw values
        title: Title for the plot
        threshold: Minimum probability to draw an arrow
    """
    # Extract MDP from model if not provided
    if transition_probs is None or state_values is None:
        if model is None:
            raise ValueError("Must provide either model or both transition_probs and state_values")
        transition_probs, state_values = extract_mdp_transitions_fixed(model, num_actions, device)
    
    num_states = len(state_values)
    
    # Validate num_states
    if num_states != 4:
        raise ValueError(f"Visualization currently only supports 4 states, got {num_states}")
    
    # Format state values
    if value_format == 'angle':
        formatted_values = state_values * 360
        value_suffix = '°'
    else:
        formatted_values = state_values
        value_suffix = ''
    
    # Colors
    state_colors = ["#332288", "#DDCC77", "#117733", "#88CCEE"]
    action_colors = ["#882255", "#44AA99"]
    
    # Positions
    state_positions = {
        0: (0.35, 0.6), 1: (0.65, 0.6),
        2: (0.35, 0.4), 3: (0.65, 0.4),
    }
    
    # Self-loop angles
    angle_offset = 0.2
    self_loop_specs = {
        0: {0: (np.pi/2 - angle_offset, np.pi + angle_offset),
            1: (np.pi/2 + angle_offset, np.pi - angle_offset)},
        1: {0: (np.pi/2 + angle_offset, -angle_offset),
            1: (np.pi/2 - angle_offset, angle_offset)},
        2: {0: (-np.pi/2 + angle_offset, np.pi + angle_offset),
            1: (-np.pi/2 - angle_offset, np.pi - angle_offset)},
        3: {0: (-np.pi/2 - angle_offset, -angle_offset),
            1: (-np.pi/2 + angle_offset, angle_offset)},
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.25, 0.75)
    ax.axis('off')
    
    min_linewidth, max_linewidth = 0.5, 12
    state_radius = 0.06
    
    # Draw transitions
    for state in range(num_states):
        for action in range(min(num_actions, 2)):  # Only visualize up to 2 actions
            action_color = action_colors[action]
            
            for next_state in range(num_states):
                prob = transition_probs[state, action, next_state]
                
                if prob > threshold:
                    state_pos = state_positions[state]
                    next_state_pos = state_positions[next_state]
                    linewidth = min_linewidth + (max_linewidth - min_linewidth) * prob
                    
                    if state == next_state:
                        # Self-loop
                        start_angle, end_angle = self_loop_specs[state][action]
                        start = (state_pos[0] + state_radius * np.cos(start_angle),
                                state_pos[1] + state_radius * np.sin(start_angle))
                        end = (state_pos[0] + state_radius * np.cos(end_angle),
                              state_pos[1] + state_radius * np.sin(end_angle))
                        
                        curve_rad = 1.2 if action == 0 else 1.4
                        if state in [1, 2]:
                            curve_rad = -curve_rad
                        
                        arrow = mpatches.FancyArrowPatch(
                            start, end,
                            connectionstyle=f"arc3,rad={curve_rad}",
                            arrowstyle='->,head_width=0.4,head_length=0.3',
                            color=action_color,
                            linewidth=linewidth,
                            alpha=0.7,
                            mutation_scale=15,
                            zorder=5
                        )
                    else:
                        # Regular transition
                        dx = next_state_pos[0] - state_pos[0]
                        dy = next_state_pos[1] - state_pos[1]
                        angle_to_next = np.arctan2(dy, dx)
                        
                        start = (state_pos[0] + state_radius * np.cos(angle_to_next),
                                state_pos[1] + state_radius * np.sin(angle_to_next))
                        end = (next_state_pos[0] - state_radius * np.cos(angle_to_next),
                              next_state_pos[1] - state_radius * np.sin(angle_to_next))
                        
                        curve_rad = 0.2 if action == 0 else -0.2
                        
                        arrow = mpatches.FancyArrowPatch(
                            start, end,
                            connectionstyle=f"arc3,rad={curve_rad}",
                            arrowstyle='->,head_width=0.4,head_length=0.3',
                            color=action_color,
                            linewidth=linewidth,
                            alpha=0.7,
                            mutation_scale=15,
                            zorder=5
                        )
                    
                    ax.add_patch(arrow)
    
    # Draw states
    for state, pos in state_positions.items():
        circle = mpatches.Circle(pos, state_radius, color=state_colors[state],
                                ec='black', linewidth=2, zorder=10)
        ax.add_patch(circle)
        
        text_color = get_mdp_text_color(state_colors[state])
        value = formatted_values[state]
        
        # Handle case where value might still be an array
        if isinstance(value, np.ndarray):
            value = value.item() if value.size == 1 else value[0]
        
        ax.text(pos[0], pos[1], f'State {state+1}:\n{value:.1f}{value_suffix}',
               ha='center', va='center', fontsize=9, color=text_color,
               weight='bold', zorder=11)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=action_colors[0], label='Action a0'),
        mpatches.Patch(color=action_colors[1], label='Action a1')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    #ax.set_title(title, fontsize=14, weight='bold', pad=20)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved MDP visualization to {output_path}")
        plt.close()
    else:
        plt.show()

### Aggregation functions
############################################################################

def plot_training_curves_aggregated(aggregated_data, save_path):
    """
    Plot aggregated training curves with soft std visualization for train curves only.
    Uses Paul Tol's muted color scheme consistent with other visualizations.
    """
    # Paul Tol's muted color scheme
    tol_muted = [
        "#CC6677",
        "#332288",
        "#DDCC77",
        "#117733",
        "#88CCEE",
        "#882255",
        "#44AA99",
        "#999933",
    ]

    training_curves = aggregated_data.get("training_curves", {})
    if not training_curves:
        print("No training curves found for aggregation plot")
        return

    # Create subplots for different loss types
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Define plot configurations - align colors with existing train/val pattern
    plot_configs = [
        {
            "ax": axes[0, 0],
            "title": "Total Loss",
            "train_key": "train_loss",
            "val_key": "val_loss",
        },
        {
            "ax": axes[0, 1],
            "title": "State Loss",
            "train_key": "train_state_loss",
            "val_key": "val_state_loss",
        },
        {
            "ax": axes[1, 0],
            "title": "Value Loss",
            "train_key": "train_value_loss",
            "val_key": "val_value_loss",
        },
        {
            "ax": axes[1, 1],
            "title": "Entropy Loss",
            "train_key": "train_entropy_loss",
            "val_key": "val_entropy_loss",
        },
    ]

    # Consistent train/val colors
    train_color = tol_muted[1]  # '#332288' - blue (to match existing)
    val_color = tol_muted[2]  # '#DDCC77' - yellow (to match existing)

    for config in plot_configs:
        ax = config["ax"]

        # Plot training curve with std band
        if (
            config["train_key"] in training_curves
            and training_curves[config["train_key"]] is not None
        ):
            curve_data = training_curves[config["train_key"]]

            mean_values = np.array(curve_data["mean"])
            std_values = np.array(curve_data["std"])
            epochs = np.arange(len(mean_values))

            # Plot mean line
            ax.plot(epochs, mean_values, label="Train", color=train_color, linewidth=2)

            # Plot std band (higher alpha as requested)
            ax.fill_between(
                epochs,
                mean_values - std_values,
                mean_values + std_values,
                color=train_color,
                alpha=0.1,
            )

        # Plot validation curve (NO std band)
        if (
            config["val_key"] in training_curves
            and training_curves[config["val_key"]] is not None
        ):
            curve_data = training_curves[config["val_key"]]

            mean_values = np.array(curve_data["mean"])
            epochs = np.arange(len(mean_values))

            # Plot mean line only
            ax.plot(epochs, mean_values, label="Val", color=val_color, linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(config["title"])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)  # Start from 0 for loss curves

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves aggregated plot to {save_path}")


def plot_softmax_rank_aggregated(aggregated_data, save_path):
    """
    Plot aggregated softmax rank evolution with std bands.
    Based on existing plot_softmax_rank_evolution structure.
    """
    # Paul Tol's muted color scheme
    tol_muted = [
        "#CC6677",
        "#332288",
        "#DDCC77",
        "#117733",
        "#88CCEE",
        "#882255",
        "#44AA99",
        "#999933",
    ]

    softmax_metrics = aggregated_data.get("softmax_rank_metrics", {})
    if not softmax_metrics:
        print("No softmax rank metrics found for aggregation plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Helper function to plot metric with std
    def plot_metric_with_std(ax, metric_key, label, color):
        if metric_key in softmax_metrics and softmax_metrics[metric_key] is not None:
            data = softmax_metrics[metric_key]
            mean_values = np.array(data["mean"])
            std_values = np.array(data["std"])
            epochs = np.arange(len(mean_values))  # Individual epochs per metric

            ax.plot(
                epochs,
                mean_values,
                "o-",
                label=label,
                linewidth=2,
                markersize=4,
                color=color,
            )
            ax.fill_between(
                epochs,
                mean_values - std_values,
                mean_values + std_values,
                color=color,
                alpha=0.1,
            )

    # Plot 1: Rank Evolution
    ax = axes[0, 0]
    plot_metric_with_std(ax, "hidden_rank", "Hidden Layer (32-dim)", tol_muted[0])
    plot_metric_with_std(ax, "logit_rank", "Logit Layer (4-dim)", tol_muted[1])
    plot_metric_with_std(ax, "softmax_rank", "Post-Softmax (4-dim)", tol_muted[2])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Numerical Rank")
    ax.set_title("Rank Evolution During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Plot 2: Norm Evolution
    ax = axes[0, 1]
    plot_metric_with_std(ax, "hidden_frobenius_norm", "Hidden ||A₃||_F", tol_muted[0])
    plot_metric_with_std(ax, "logit_frobenius_norm", "Logit ||M₄||_F", tol_muted[1])
    plot_metric_with_std(ax, "softmax_frobenius_norm", "Softmax ||A₄||_F", tol_muted[2])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Frobenius Norm")
    ax.set_title("Norm Evolution During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Plot 3: Logit Singular Values (if globally normalized data exists)
    ax = axes[1, 0]
    colors_sv = [tol_muted[0], tol_muted[1], tol_muted[2], tol_muted[3]]
    for i in range(4):
        sv_key = f"logit_sv_global_norm_{i}"
        plot_metric_with_std(ax, sv_key, f"σ_{i+1}", colors_sv[i])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Globally Normalized Singular Value")
    ax.set_title("Logit Singular Values Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Plot 4: Hidden Singular Values
    ax = axes[1, 1]
    colors_sv = [tol_muted[0], tol_muted[1], tol_muted[2], tol_muted[3]]
    for i in range(4):
        sv_key = f"hidden_sv_global_norm_{i}"
        plot_metric_with_std(ax, sv_key, f"σ_{i+1}", colors_sv[i])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Globally Normalized Singular Value")
    ax.set_title("Hidden Layer Singular Values Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved softmax rank aggregated plot to {save_path}")


def plot_state_metrics_aggregated(aggregated_data, save_path):
    """
    Plot aggregated state assignment quality metrics (stability and sharpness).
    """
    # Paul Tol's muted color scheme
    tol_muted = [
        "#CC6677",
        "#332288",
        "#DDCC77",
        "#117733",
        "#88CCEE",
        "#882255",
        "#44AA99",
        "#999933",
    ]

    state_metrics = aggregated_data.get("state_metrics", {})
    if not state_metrics:
        print("No state metrics found for aggregation plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Helper function to plot metric with std - FIXED to handle epochs per metric
    def plot_metric_with_std(ax, metric_key, label, color):
        if metric_key in state_metrics and state_metrics[metric_key] is not None:
            data = state_metrics[metric_key]
            mean_values = np.array(data["mean"])
            std_values = np.array(data["std"])
            epochs = np.arange(
                len(mean_values)
            )  # Create epochs for THIS metric specifically

            ax.plot(
                epochs,
                mean_values,
                "o-",
                label=label,
                linewidth=2,
                markersize=4,
                color=color,
            )
            ax.fill_between(
                epochs,
                mean_values - std_values,
                mean_values + std_values,
                color=color,
                alpha=0.1,
            )

    # Plot 1: Sharpness metrics
    ax = axes[0]
    plot_metric_with_std(ax, "sharpness_mean", None, tol_muted[1])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Entropy (Sharpness)")
    #ax.legend()
    ax.grid(True, alpha=0.2)

    # Plot 2: Stability metrics
    ax = axes[1]
    plot_metric_with_std(
        ax, "dominant_stability", None, tol_muted[1]
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("State Assignment Stability (%)")
    #ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 100)  # Stability is in percentage

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved state metrics aggregated plot to {save_path}")


def plot_test_metrics_summary(aggregated_data, save_path):
    """
    Plot summary of final test metrics with error bars.
    """
    # Paul Tol's muted color scheme
    tol_muted = [
        "#CC6677",
        "#332288",
        "#DDCC77",
        "#117733",
        "#88CCEE",
        "#882255",
        "#44AA99",
        "#999933",
    ]

    test_metrics = aggregated_data.get("test_metrics", {})
    if not test_metrics:
        print("No test metrics found for summary plot")
        return

    # Define important metrics and their display names
    metric_configs = [
        ("test_loss", "Total Loss"),
        ("test_state_loss", "State Loss"),
        ("test_value_loss", "Value Loss"),
        ("prob_discrete_accuracy", "Discrete Accuracy"),
        ("test_batch_entropy", "Batch Entropy"),
        ("test_individual_entropy", "Individual Entropy"),
    ]

    # Filter to available metrics
    available_metrics = [
        (key, name)
        for key, name in metric_configs
        if key in test_metrics and test_metrics[key] is not None
    ]

    if not available_metrics:
        print("No available test metrics for summary plot")
        return

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    metric_names = [name for _, name in available_metrics]
    means = [test_metrics[key]["mean"] for key, _ in available_metrics]
    stds = [test_metrics[key]["std"] for key, _ in available_metrics]

    x_pos = np.arange(len(metric_names))
    bars = ax.bar(
        x_pos,
        means,
        yerr=stds,
        capsize=5,
        color=[tol_muted[i % len(tol_muted)] for i in range(len(metric_names))],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Test Metrics")
    ax.set_ylabel("Value")
    ax.set_title("Final Test Metrics Summary")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std + 0.01 * max(means),
            f"{mean:.3f}±{std:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved test metrics summary plot to {save_path}")

def plot_probing_aggregated(aggregated_data, save_path):
    """
    Plot aggregated probing evolution with std bands.
    Uses Paul Tol's muted color scheme for colorblind accessibility.
    
    Args:
        aggregated_data: Aggregated results containing probing_metrics
        save_path: Path to save the plot
    """
    
    # Paul Tol's muted color scheme
    tol_blue = "#332288"
    
    probing_metrics = aggregated_data.get("probing_metrics", {})
    if not probing_metrics:
        print("No probing metrics found for aggregation plot")
        return
    
    # Extract epochs array (actual epoch numbers like 5, 10, 15, ...)
    epochs = probing_metrics.get("epochs", [])
    if not epochs:
        print("No epoch information found in probing metrics")
        return
    
    # Extract accuracy data
    accuracy_data = probing_metrics.get("discrete_accuracy")
    if not accuracy_data:
        print("No discrete_accuracy data found in probing metrics")
        return
    
    mean_values = np.array(accuracy_data["mean"])
    std_values = np.array(accuracy_data["std"])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot mean with std bands
    ax.plot(
        epochs,
        mean_values,
        "o-",
        linewidth=2,
        markersize=6,
        color=tol_blue,
    )
    
    # Add std band
    ax.fill_between(
        epochs,
        mean_values - std_values,
        mean_values + std_values,
        color=tol_blue,
        alpha=0.1,
    )
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Discrete Accuracy", fontsize=12)
    #ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1)  # Accuracy is between 0 and 1
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved probing aggregated plot to: {save_path}")

