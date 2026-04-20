import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import sys

import os
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np


def plot_2D_vector_field(vectorfield, bounds, resolution= 50, streamplot=True):
    """
    Plots a 2D vector field for a given ODE function.

    Args:
        ode_func (function): A function of the form ode_func(t, state).
        space (list of tuples): The space intervals for each dimension.
        streamplot (bool): If True, a streamplot is generated.

    Returns:
        None
    """
    X = np.linspace(bounds[0][0], bounds[0][1], resolution)
    Y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    X, Y = np.meshgrid(X, Y)

    U, V = vectorfield(X, Y)

    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, U, V, color='b', alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Vector Field')
    plt.grid(True)

    if streamplot:
        plt.streamplot(X, Y, U, V, color='b', linewidth=0.7)

    plt.show()


def create_2D_vectorfield(X1, X2, control, solver):
    """
    Computes the vector field components (U, V) on a meshgrid (X1, X2).
    #TODO fix this also for models with more than 2 dimensions (by projection or something).
    #TODO make this more efficient by using vectorized operations, doesn't work yet.

    Parameters:
    X1, X2 (2D arrays): Meshgrid coordinates, assumed to be of the same dimension.
    control: control parameter, expected as an np.array
    solver: Solver class connected to a model

    Return:
    U, V: (2D arrays): Vector components at each grid point.

    """
    #TODO Check vectorized version

    nrows, ncols = X1.shape

    U = np.zeros_like(X1)  # Placeholder for dx/dt
    V = np.zeros_like(X2)  # Placeholder for dy/dt

    #TRY TO VECTORIZE IT... but doesn't work yet for higher resolutions
    #X = np.column_stack((X1.ravel(), X2.ravel()))
    #control_stacked = np.repeat(control, X.shape[0]) #here also check if control has 1 entry or n_samples many
    #derivatives = solver.solve_equilibrium(X, control_stacked)
    #U = derivatives[:,0].reshape(nrows, ncols)
    #V = derivatives[:,1].reshape(nrows, ncols)
    
    
    if True:
        # Loop through each point in the meshgrid
        for i in range(X1.shape[0]):
            for j in range(X1.shape[1]):
                state = np.array([X1[i, j], X2[i, j]])  # Current state vector
                state = np.atleast_2d(state)
                derivatives = solver.get_derivative(state, control).ravel()    # Evaluate 
                U[i, j] = derivatives[0]            # dx/dt
                V[i, j] = derivatives[1]            # dy/dt

    return U, V



def plot_2D_vector_field_over_grid(grid, 
                                   solver, 
                                   control, 
                                   dim1 = 0, dim2 = 1, 
                                   resolution=51,
                                   streamplot_colour = 'darkblue',
                                   vectorfield_colour = 'mediumblue', 
                                   title='2D Vector Field',
                                   axis_names =['X1-Axis','X2-Axis'], 
                                   ax = None,
                                   save_to = None, 
                                   display_vectorfield=True, 
                                   display_grid=True, 
                                   display_streamplot=True,
                                   broken_streamlines = True,
                                   display_vectorfield_magnitude=False,
                                   display_nullclines=False,
                                   nullcline_colors=('red', 'green')):
    """
    Plots a 2D vector field over a specified grid.
    
    Parameters:
        grid (object): The grid object containing the bounds and transformation information.
        solver (object): The solver object containing the model. This will be used to compute the vector field.
        control (np.array): The control vector for the vectorfield. Has to be given in the form of np.array([control]). 
                            #TODO should this be adaptable for different controls corresponding to different points?

        dim1 (int, optional): The first dimension to plot. Defaults to 0.
        dim2 (int, optional): The second dimension to plot. Defaults to 1.
        resolution (int, optional): The resolution of the meshgrid for the vectorfield. Defaults to 51 (number_cells + 1).
        title (str, optional): The title of the plot. Defaults to '2D Vector Field'.
        save_to (str, optional): The path where to save the plot.
        display_vectorfield (bool, optional): Whether to display the vector field using quiver. Defaults to True.
        display_grid (bool, optional): Whether to display the grid. Defaults to True.
        display_vectorfield_magnitude (bool, optional): Whether to display the magnitude of the vector field. Defaults to False.
    
    Returns:
        fig, ax
    """
        
    bounds = [grid.tf_bounds[dim1], grid.tf_bounds[dim2]]
    X1lin = np.linspace(bounds[0][0], bounds[0][1], resolution)
    X2lin = np.linspace(bounds[1][0], bounds[1][1], resolution)
    
    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if grid.transformed_bool:
        
        #Remove boundaries for infinity case (transformation doesn't work herd)
        if np.isinf(grid.bounds[dim1][1]):
            X1lin = X1lin[:-1]
        if np.isinf(grid.bounds[dim1][0]):
            X1lin = X1lin[1:]
        if np.isinf(grid.bounds[dim2][1]):
            X2lin = X2lin[:-1]
        if np.isinf(grid.bounds[dim2][0]):
            X2lin = X2lin[1:]
        
        X1, X2 = np.meshgrid(X1lin, X2lin)
        X1_org = np.vectorize(grid.inverse_transformations[dim1])(X1)
        X2_org = np.vectorize(grid.inverse_transformations[dim2])(X2)
    
        U_org, V_org = create_2D_vectorfield(X1_org, X2_org, control=control, solver=solver)
        
        #Transform the vectorfield, multiply with Jacobian of transformation
        U = np.multiply(np.vectorize(grid.transformation_derivatives[dim1])(X1_org), U_org)
        V = np.multiply(np.vectorize(grid.transformation_derivatives[dim2])(X2_org), V_org)

    else:
        
        X1, X2 = np.meshgrid(X1lin, X2lin)
        U, V = create_2D_vectorfield(X1, X2, control=control, solver=solver)
    
    
    # Plot streamlines
    if display_streamplot:
        ax.streamplot(X1, X2, U, V, color=streamplot_colour, linewidth=0.7, density=1, arrowsize=0.8, broken_streamlines=broken_streamlines)

    # Plot nullclines if requested
    if display_nullclines:
        # Use contour to find nullclines (U=0, V=0 curves)
        ax.contour(X1, X2, U, levels=[0], colors=[nullcline_colors[0]], linewidths=1, alpha=0.7)
        ax.contour(X1, X2, V, levels=[0], colors=[nullcline_colors[1]], linewidths=1, alpha=0.7)
    

    if display_vectorfield:
        ax.quiver(X1, X2, U, V, color=vectorfield_colour, alpha=0.2)

    if display_vectorfield_magnitude:
        # Add a color mesh to visualize the magnitude of the vector field
        magnitude = np.sqrt(U**2 + V**2)
        contour = ax.contourf(X1, X2, magnitude, levels=50, cmap='viridis', alpha=0.6)
        fig.colorbar(contour, ax=ax, label='Vectorfield Magnitude')

    if grid.transformed_bool:
        #Set ticklabels according to original space
        xtick_pos = np.linspace(bounds[0][0], bounds[0][1], 6)
        xlabels = [round(grid.inverse_transformations[dim1](tick),1) for tick in xtick_pos]
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xlabels)

        ytick_pos = np.linspace(bounds[1][0], bounds[1][1], 6)
        ylabels = [round(grid.inverse_transformations[dim1](tick),2) for tick in ytick_pos]
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ylabels)

    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.set_title(title, pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position(('data', bounds[0][0]))
    ax.spines['left'].set_position(('data', bounds[1][0]))
    ax.set_xlim(left=bounds[0][0])
    ax.set_ylim(bottom=bounds[1][0])
    if display_grid:
        # Changed this to the actual grid lines
        for x in grid.tf_grid_lines[0]:
            ax.axvline(x, color='gray', linestyle='--', alpha=0.5, linewidth=0.7)
        for y in grid.tf_grid_lines[1]:
            ax.axhline(y, color='gray', linestyle='--', alpha=0.5, linewidth=0.7)

    if save_to != None:
        fig.savefig(save_to)

  
    return

def create_saddle_streamplot_visualization(
    grid, solver, model, saddle_points, angles_degrees,
    output_dir="data_generation/figs", 
    filename="saddle_streamplot.png",
    resolution=51,
    density=0.8,
    alpha=0.9,
    use_transformed_space=True,
    x_range=(-10, 10),
    y_range=(-10, 10)
):
    """
    Create overlaid streamplot visualization for saddle system.
    
    Args:
        grid: Grid object with transformation info
        solver: Solver object to compute derivatives  
        model: Model object containing saddle information
        saddle_points: List of saddle point coordinates
        angles_degrees: List of angles for stable manifolds
        output_dir: Directory to save visualization
        filename: Name of output file
        resolution: Grid resolution for streamplot
        density: Streamline density 
        alpha: Base transparency for streamlines
        use_transformed_space: If True, plot in transformed [0,1]x[0,1] space
        x_range: Range for x-axis when use_transformed_space=False
        y_range: Range for y-axis when use_transformed_space=False
    """
    import os
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    import numpy as np
    
    # Colorblind-friendly palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    if use_transformed_space:
        # Work in transformed space [0,1] x [0,1]
        bounds = [grid.tf_bounds[0], grid.tf_bounds[1]]
        X1lin = np.linspace(bounds[0][0], bounds[0][1], resolution)
        X2lin = np.linspace(bounds[1][0], bounds[1][1], resolution)
        
        # Handle infinity bounds if needed
        if grid.transformed_bool:
            if np.isinf(grid.bounds[0][1]):
                X1lin = X1lin[:-1]
            if np.isinf(grid.bounds[0][0]):
                X1lin = X1lin[1:]
            if np.isinf(grid.bounds[1][1]):
                X2lin = X2lin[:-1]
            if np.isinf(grid.bounds[1][0]):
                X2lin = X2lin[1:]
        
        X1, X2 = np.meshgrid(X1lin, X2lin)
        
        # Transform to original space for computation
        X1_org = np.vectorize(grid.inverse_transformations[0])(X1)
        X2_org = np.vectorize(grid.inverse_transformations[1])(X2)
        
        space_type = "Transformed Space [0,1]×[0,1]"
        
    else:
        # Work directly in original space
        X1lin = np.linspace(x_range[0], x_range[1], resolution)
        X2lin = np.linspace(y_range[0], y_range[1], resolution)
        X1, X2 = np.meshgrid(X1lin, X2lin)
        X1_org, X2_org = X1, X2  # No transformation needed
        
        space_type = f"Original Space [{x_range[0]},{x_range[1]}]×[{y_range[0]},{y_range[1]}]"
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot streamlines for each control value
    num_controls = len(saddle_points)

    # Create different starting point sets for each control
    start_points_list = []
    for i in range(num_controls):
        # Create slightly different starting grids
        offset_factor = i * 0.015  # Small offset for each control (1.5% of range)
        
        if use_transformed_space:
            bounds = [grid.tf_bounds[0], grid.tf_bounds[1]]
            # Create starting points with offset
            start_x = np.linspace(bounds[0][0] + offset_factor, bounds[0][1] - offset_factor, 8)
            start_y = np.linspace(bounds[1][0] + offset_factor, bounds[1][1] - offset_factor, 8)
        else:
            # Scale offset by range for original space
            x_range_size = x_range[1] - x_range[0]
            y_range_size = y_range[1] - y_range[0]
            x_offset = offset_factor * x_range_size
            y_offset = offset_factor * y_range_size
            
            start_x = np.linspace(x_range[0] + x_offset, x_range[1] - x_offset, 8)
            start_y = np.linspace(y_range[0] + y_offset, y_range[1] - y_offset, 8)
        
        # Create meshgrid and flatten to get starting points
        start_X, start_Y = np.meshgrid(start_x, start_y)
        # Stack as (2, N) array: first row = x coords, second row = y coords
        start_points = np.column_stack([start_X.ravel(), start_Y.ravel()])
        start_points_list.append(start_points)
    
    for i in range(num_controls):
        if i >= len(colors):
            break
            
        control_value = i
        
        # Compute vector field in original space
        U_org, V_org = create_2D_vectorfield(X1_org, X2_org, control=control_value, solver=solver)
        
        if use_transformed_space:
            # Transform the vector field to z-space (multiply by Jacobian)
            U = np.multiply(np.vectorize(grid.transformation_derivatives[0])(X1_org), U_org)
            V = np.multiply(np.vectorize(grid.transformation_derivatives[1])(X2_org), V_org)
        else:
            # Use original vector field
            U, V = U_org, V_org
        
        # Calculate magnitude for color intensity
        magnitude = np.sqrt(U**2 + V**2)

        # Log-based scaling with offset to handle zeros and compress dynamic range
        log_magnitude = np.log1p(magnitude)  # log(1 + magnitude) to handle zeros
        max_log = np.max(log_magnitude) if np.max(log_magnitude) > 0 else 1.0

        # Normalize to [0.5, 1.0] range for better visibility
        intensity = 0.5 + 0.5 * (log_magnitude / max_log)
        
        # Create colormap with alpha
        base_rgba = to_rgba(colors[i])
        transparent_base = (*base_rgba[:3], alpha)
        colors_with_alpha = [(1, 1, 1, 0.1), transparent_base]
        
        # Plot streamlines
        ax.streamplot(X1, X2, U, V, 
                     start_points=start_points_list[i],
                     color=intensity,
                     cmap=plt.cm.colors.LinearSegmentedColormap.from_list('', colors_with_alpha),
                     linewidth=1.2, 
                     density=density, 
                     arrowsize=1.2)
    
    # Plot saddle points and manifold lines
    for i, (point, angle_deg) in enumerate(zip(saddle_points, angles_degrees)):
        if use_transformed_space:
            # Transform saddle point to z-space
            z1 = grid.transformations[0](point[0])
            z2 = grid.transformations[1](point[1])
            point_plot = [z1, z2]
            plot_bounds = [(0, 1), (0, 1)]
        else:
            # Use original coordinates
            point_plot = point
            plot_bounds = [x_range, y_range]
        
        # Plot saddle point
        ax.plot(point_plot[0], point_plot[1], 'o', color=colors[i], markersize=10, 
               markeredgecolor='black', markeredgewidth=0.5, zorder=10)
        
        # Draw manifold line
        angle_rad = np.radians(angle_deg)
        px, py = point_plot[0], point_plot[1]
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Calculate intersection with plot boundaries
        t_values = []
        if abs(cos_a) > 1e-10:
            t_values.extend([(plot_bounds[0][0] - px) / cos_a, (plot_bounds[0][1] - px) / cos_a])
        if abs(sin_a) > 1e-10:
            t_values.extend([(plot_bounds[1][0] - py) / sin_a, (plot_bounds[1][1] - py) / sin_a])
        
        if t_values:
            t_min, t_max = min(t_values), max(t_values)
            x_start = px + t_min * cos_a
            x_end = px + t_max * cos_a
            y_start = py + t_min * sin_a
            y_end = py + t_max * sin_a
            
            # Plot manifold line
            ax.plot([x_start, x_end], [y_start, y_end], 
                   color=colors[i], linestyle='--', linewidth=1.5, 
                   alpha=0.6, zorder=5)
    
    # Set coordinate labels and plot properties
    if use_transformed_space:
        # Set tick labels to show original space values
        bounds = [grid.tf_bounds[0], grid.tf_bounds[1]]
        xtick_pos = np.linspace(bounds[0][0], bounds[0][1], 6)
        xlabels = [f"{grid.inverse_transformations[0](tick):.1f}" for tick in xtick_pos]
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xlabels)
        
        ytick_pos = np.linspace(bounds[1][0], bounds[1][1], 6)
        ylabels = [f"{grid.inverse_transformations[1](tick):.1f}" for tick in ytick_pos]
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ylabels)
        
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
    else:
        # Use original space coordinates directly
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_aspect('equal')  # Equal aspect ratio for proper angles in original space
        
        # Add explicit tick control for non-transformed space
        xtick_pos = np.linspace(x_range[0], x_range[1], 6)
        ax.set_xticks(xtick_pos)
        
        ytick_pos = np.linspace(y_range[0], y_range[1], 6)
        ax.set_yticks(ytick_pos)
    
    ax.set_title(f'Saddle System Dynamics')
    
    # Create legend
    legend_elements = []
    for i in range(min(num_controls, len(colors))):
        legend_elements.append(
            plt.Line2D([0], [0], color=colors[i], linewidth=2, 
                      label=f'Action {i} (Saddle {i+1})')
        )
    legend_elements.append(
        plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=3,
                  label='Stable Manifolds')
    )
    
    ax.legend(handles=legend_elements, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved saddle streamplot visualization to {output_path}")
    
    plt.close(fig)
    return output_path


if __name__ == '__main__':
    pass

