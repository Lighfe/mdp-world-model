import numpy as np
import matplotlib.pyplot as plt
from data_generation.visualization.create_plots import plot_2D_vector_field_over_grid

def plot_trajectories_in_transformed_space(results, grid, solver, control, patch_ids = None, dim1=0, dim2=1, 
                                           resolution=21, ax=None, max_trajectories=None, 
                                           save_to=None, plot_vector_field = True, title="Trajectories in Transformed Space"):
    """
    Plot trajectories in transformed space on top of a vector field.
    
    Args:
        results (DataFrame): Simulation results containing trajectories
        grid (Grid): The grid object containing transformation information
        solver (object): The solver object for the model
        control: Control parameter
        dim1, dim2: Dimensions to plot (default: 0, 1)
        resolution: Resolution of the vector field (default: 51)
        ax: Matplotlib axis (if None, a new figure is created)
        max_trajectories: Maximum number of trajectories to plot (None for all)
        save_to: Path to save the figure (default: None)
        title: Plot title (default: "Trajectories in Transformed Space")
    
    Returns:
        Matplotlib axis with the plot
    """
    # Create the vector field plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the vector field
    
    plot_2D_vector_field_over_grid(grid, solver, control, dim1, dim2, 
                                    resolution, ax=ax, display_grid=True,
                                    title=title, broken_streamlines=False, display_vectorfield=False, display_streamplot=plot_vector_field)
    if patch_ids != None:
        results = results[results['x_patch_id'].isin(patch_ids)]
    # Group by trajectory_id
    trajectory_groups = list(results.groupby(['trajectory_id']))

    # Limit the number of trajectories if specified
    if max_trajectories is not None and max_trajectories < len(trajectory_groups):
        import random
        trajectory_groups = random.sample(trajectory_groups, max_trajectories)
    color_count = 0
    color_map = plt.cm.get_cmap('tab10', 10)  # Use a colormap with 10 distinct colors
    # Plot each trajectory
    for traj_id, traj_data in trajectory_groups:
        # Assign a unique color for each trajectory
        # Generate a unique color for each trajectory using a colormap with many colors
        
        color = color_map(color_count % 10)  # Use modulo to cycle through 10 colors
        color_count = color_count + 1

        traj_by_control = traj_data.groupby(['run_id'])
        for run_id, traj_control_data in traj_by_control:
            # Sort by time
            traj_control_data  = traj_control_data.sort_values('t0')
            
            # Initialize trajectory points array
            traj_points = np.zeros((len(traj_control_data ) + 1, 2))
            
            # Fill in the points
            for i, (_, row) in enumerate(traj_control_data .iterrows()):
                if i == 0:
                    # For the first row, use the x values as the starting point
                    traj_points[i] = row[f'x']
                
                # For all rows, add the y values as the next point
                traj_points[i+1] = row[f'y']
            
            # Transform to z-space
            if grid.transformed_bool:
                z_points = np.zeros_like(traj_points)
                z_points[:, 0] = np.vectorize(grid.transformations[dim1])(traj_points[:, 0])
                z_points[:, 1] = np.vectorize(grid.transformations[dim2])(traj_points[:, 1])
            else:
                z_points = traj_points
            
            # Plot the trajectory
            ax.plot(z_points[:, 0], z_points[:, 1], 'o-', linewidth=1, markersize=3, alpha=0.7, color = color)
    
    if save_to is not None:
        plt.savefig(save_to)
    
    return ax



