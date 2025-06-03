import json
import time
from datetime import datetime
from pathlib import Path
import platform
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, differential_evolution
from data_generation.models import tech_substitution as ts
from data_generation.simulations.grid import Grid
from datasets.database import get_engine, init_db, create_results_table

#to run this file as a package: python -m data_generation.simulations.simulator while being in the mdp-world-model folder

class Simulator:
    """
    Class for simulating trajectories on a grid.
    Expects a grid object from the Grid class.
    """

    def __init__(self, grid, model, solver):
        self.grid = grid
        self.model = model
        self.solver = solver
        self.x_dim = model.x_dim
        self.control_dim = model.control_dim
        self.steady_control = False
        self.results = self._init_df()
        self.configs = pd.DataFrame(columns=['run_id', 'metadata'])

    def _init_df(self):
        columns = ['run_id', 'trajectory_id', 't0', 't1']
        columns.extend([f'x{i}' for i in range(self.x_dim)])
        columns.extend([f'c{i}' for i in range(self.control_dim)])
        columns.extend([f'y{i}' for i in range(self.x_dim)])
        return pd.DataFrame(columns=columns)

    def simulate(self, control, delta_t, num_samples_per_cell=10, num_steps=1, save_result=False):   
        """
        Simulates the differential equation for a given initial condition and time period.
        
        Args:
            control: control input in supported format
            delta_t: time step size
            num_samples_per_cell: Number of samples per grid cell
            num_steps: Number of simulation steps
            save_result: Whether to save results to internal storage

        Returns: df (DataFrame): simulation results
        """
        # Start timing
        start_time = time.time()

        X, trajectory_ids = self.grid.get_initial_conditions(num_samples_per_cell)
        n_samples = X.shape[0]
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


        # Convert control to full array format
        control = np.array(control)
        control = self.create_control_array(control, num_steps, n_samples, self.control_dim)

        # get full trajectory
        trajectory = self.solver.step(X, control, delta_t, num_steps, self.steady_control)

        # Create timestamps
        times = np.linspace(0, num_steps * delta_t, num_steps + 1) # arange had floating point errors
        
        # Initialize data dictionary
        data = {
            'run_id': np.full(n_samples * num_steps, run_id),
            'trajectory_id': np.repeat(trajectory_ids, num_steps),
            't0': np.tile(times[:-1], n_samples),  
            't1': np.tile(times[1:], n_samples),  
        }
        
        # Add start observation
        for i in range(self.x_dim):
            data[f'x{i}'] = trajectory[:-1, :, i].flatten(order='F')
        
        # Add control dimensions
        for i in range(self.control_dim):
            # data[f'c{i}'] = np.repeat(control[:, i], num_steps)
            data[f'c{i}'] = control[:,:,i].flatten(order='F')

        # add result observations
        for i in range(self.x_dim):
            data[f'y{i}'] = trajectory[1:, :, i].flatten(order='F')

        df = pd.DataFrame(data)

        print(f"Simulation complete:\n"
          f"- {n_samples} samples × {num_steps} timesteps = {n_samples * num_steps} total rows\n"
          f"- State dimensions: {self.x_dim}\n"
          f"- Control dimensions: {self.control_dim}")

        # Possibly store the resulting df in the self.result_df attribute (by adding it to the existing df)
        if save_result == True:
            
            # Initialize empty DataFrame with correct dtypes if needed
            if self.results.empty:
                self.results = pd.DataFrame(columns=df.columns).astype(df.dtypes)
            
            self.results = pd.concat([self.results, df], ignore_index=True)
            
                # Collect all metadata in a dictionary
                # TODO: add name of the classes
            metadata = {
                'configurations': {
                    'grid': self.grid.get_config(),
                    'solver': self.solver.get_config()
                },
                'simulation_params': {
                    'n_samples': n_samples,
                    'num_steps': num_steps
                },
                'performance': {
                    'execution_time': time.time() - start_time,
                    'peak_memory_mb': psutil.Process().memory_info().rss / (1024 * 1024)
                },
                'system': {
                    'os': platform.system(),
                    'cpu_count': psutil.cpu_count(logical=False)
                    # 'gpu': get_gpu_info(),
                    # 'ram': psutil.virtual_memory().total / (1024**3)
                }
            }    
            # Create config entry with single JSON field
            config_entry = pd.DataFrame({
                'run_id': [run_id],
                'metadata': [json.dumps(metadata)]
            })
            
            # Add to configs dataframe
            self.configs = pd.concat([self.configs, config_entry], ignore_index=True)

            print("Saved results and config.")

        # NOTE: We could also make this function not return the df object, 
        # since it is already saving it anyway. Could instead give a status update
        # or even return nothing

        return df
    

    def simulate_with_stopping_criteria(self, control, delta_t, stopping_criteria, num_samples_per_cell=10,  max_steps=1000, num_steps=1, save_result=False):   
        """
        Simulates the differential equation for a given initial condition and time period.
        
        Args:
            control: control input in supported format
            delta_t: time step size
            stopping_criteria: Function to determine if the simulation should stop
            num_samples_per_cell: Number of samples per grid cell
            max_steps: Maximum number of simulation steps
            num_steps: Number of simulation steps
            save_result: Whether to save results to internal storage

        Returns: df (DataFrame): simulation results
        """
        # Start timing
        start_time = time.time()

        X, trajectory_ids = self.grid.get_initial_conditions(num_samples_per_cell)
        n_samples = X.shape[0]
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert control to full array format
        direct_control = np.array(control)
        control = self.create_control_array(direct_control, num_steps, n_samples, self.control_dim)

        # Simulate until Stopping Criteria is met
        total_steps = 0
        all_trajectories = np.expand_dims(X, axis=0)
        X_cell_ids = np.apply_along_axis(self.grid.get_cell_index, 1, X)
        all_cell_indices = np.expand_dims(X_cell_ids, axis=0)

        while total_steps < max_steps:
            trajectory = self.solver.step(X, control, delta_t, num_steps, self.steady_control)
            #print(trajectory)
            all_trajectories = np.concatenate((all_trajectories,trajectory[1:]), axis=0)
            all_cell_indices = np.concatenate((all_cell_indices, np.apply_along_axis(self.grid.get_cell_index, 2, trajectory[1:])))
                        
            total_steps += num_steps
            
            
            # Check stopping criteria
            if stopping_criteria(all_cell_indices):
                break
            if total_steps >= max_steps:
                print(f"Caution: Maximum number of steps reached ({max_steps})")

            X = trajectory[-1]
            #print("X", X)
            
        # After stopping, process everything into a DataFrame
        
        # Create timestamps 
        times = np.linspace(0, total_steps * delta_t, total_steps + 1) 

        # Create total control array
        total_control = self.create_control_array(direct_control, total_steps, n_samples, self.control_dim)
        
        # Initialize data dictionary
        data = {
            'run_id': np.full(n_samples * total_steps, run_id),
            'trajectory_id': np.repeat(trajectory_ids, total_steps),
            't0': np.tile(times[:-1], n_samples),  
            't1': np.tile(times[1:], n_samples),  
        }
        
        # Add start observation
        for i in range(self.x_dim):
            data[f'x{i}'] = all_trajectories[:-1, :, i].flatten(order='F')
        
        # Add control dimensions
        for i in range(self.control_dim):
            # data[f'c{i}'] = np.repeat(control[:, i], num_steps)
            data[f'c{i}'] = total_control[:,:,i].flatten(order='F')

        # add result observations
        for i in range(self.x_dim):
            data[f'y{i}'] = all_trajectories[1:, :, i].flatten(order='F')
        
        df = pd.DataFrame(data)

        print(f"Simulation complete:\n"
          f"- {n_samples} samples × {total_steps} timesteps = {n_samples * total_steps} total rows\n"
          f"- State dimensions: {self.x_dim}\n"
          f"- Control dimensions: {self.control_dim}")

        # Possibly store the resulting df in the self.result_df attribute (by adding it to the existing df)
        if save_result == True:

            # Initialize empty DataFrame with correct dtypes if needed
            if self.results.empty:
                self.results = pd.DataFrame(columns=df.columns).astype(df.dtypes)

            self.results = pd.concat([self.results, df], ignore_index=True)

                # Collect all metadata in a dictionary
                # TODO: add name of the classes
            metadata = {
                'configurations': {
                    'grid': self.grid.get_config(),
                    'solver': self.solver.get_config()
                },
                'simulation_params': {
                    'n_samples': n_samples,
                    'num_steps': num_steps
                },
                'performance': {
                    'execution_time': time.time() - start_time,
                    'peak_memory_mb': psutil.Process().memory_info().rss / (1024 * 1024)
                },
                'system': {
                    'os': platform.system(),
                    'cpu_count': psutil.cpu_count(logical=False)
                    # 'gpu': get_gpu_info(),
                    # 'ram': psutil.virtual_memory().total / (1024**3)
                }
            }    

            # Create config entry with single JSON field
            config_entry = pd.DataFrame({
                'run_id': [run_id],
                'metadata': [json.dumps(metadata)]
            })
            
            # Add to configs dataframe
            self.configs = pd.concat([self.configs, config_entry], ignore_index=True)

            print("Saved results and config.")

        # NOTE: We could also make this function not return the df object, 
        # since it is already saving it anyway. Could instead give a status update
        # or even return nothing

        return df, all_cell_indices
    
    def create_control_array(self, control, num_steps, n_samples, control_dim):
        '''
        Args:
            control (np.ndarray): Input control array of scalar, list or (num_steps, n_samples, control_dim)
            num_steps (int): Number of simulation timesteps
            n_samples (int): Number of samples being simulated
            control_dim (int): Dimensionality of the control space
            
        Returns:
            np.ndarray: Control array of shape (num_steps, n_samples, control_dim)
        '''
        # scalar input
        if control.ndim == 0:
            if control_dim != 1:
                raise ValueError(f"Scalar control input requires control_dim=1, got {control_dim}")
            # Create full array with scalar value
            control = np.full((num_steps, n_samples, 1), control)
            self.steady_control = True
        # Case list input (as np.array)
        elif control.ndim == 1:
            if len(control) != control_dim:
                raise ValueError(f"1D control input length must match control_dim={control_dim}")
            # Repeat control values across all timesteps and samples
            control = control.reshape(1, 1, control_dim)
            control = np.broadcast_to(control, (num_steps, n_samples, control_dim)) 
            self.steady_control = True
        # check input
        else:
            required_shape = (num_steps, n_samples, control_dim)
            if control.shape != required_shape:
                raise ValueError(
                    f"Control array must have shape {required_shape}, got {control.shape}"
                )
            # Compare each timestep against the first timestep
            # If all True, control is steady
            self.steady_control = np.all(
                np.all(control == control[0:1], axis=2),  # Compare all control dimensions
                axis=0  # Compare all samples
            ).all()
        
        return control


    def get_gridcell_centers_and_derivatives(self, controls):
        """
        Calculates the grid cell centers and their derivatives for possibly multiple controls for a given grid and solver.
        
        Parameters:
        grid (Grid): The grid object containing the grid information and transformations.
        solver: The solver object used to compute the derivatives.
        controls (np.array): An array of control parameters for which the derivatives are computed.
        
        Returns:
        transformed_centers (np.array): The transformed centers of the grid cells.
        centers (np.array): The original centers of the grid cells.
        derivatives (np.array): The derivatives at the grid cell centers for each control parameter.
        transformed_derivates (np.array): The transformed derivatives at the grid cell centers for each control parameter.
        """
        
        grid = self.grid
        solver = self.solver


        transformed_centers = grid.get_cell_centers(transformed_space=True)
        centers = grid.get_cell_centers()
        num_centers = np.prod(grid.resolution)
        derivatives = np.zeros((controls.shape[0], num_centers, grid.dimension))
        transformed_derivates = np.zeros((controls.shape[0], num_centers, grid.dimension))
        print('got here')
        for k, c in enumerate(controls):
            derivatives[k] = solver.get_derivative(centers, controls[k])
            if grid.transformed_bool:
                #Transform the vectorfield, multiply with Jacobian of transformation
                for i in range(grid.dimension):
                    transformed_derivates[k, :, i] = np.multiply(np.vectorize(grid.transformation_derivatives[i])(centers[:,i]), derivatives[k,:,i])
            else:
                transformed_derivates[k] = derivatives[k]

        return centers, transformed_centers, derivatives, transformed_derivates
    


    def get_optimal_delta_t(self, controls, search_space=(0.001, 10), path_to_save_plot=None):
        """
        Calculate the optimal delta_t for the simulation based on the transformed derivates.
        The calculated delta_t is optimal such that the maximum number of transformed derivatives from the grid cell centers
         'lands' in a neighboring cell (also diagonal allowed).
        
        Args:
            controls (np.array): Array of control values for which to calculate the optimal delta_t
            search_space (tuple): Search space for delta_t, default is (0.001, 100)
            
        Returns:
            float: Optimal delta_t value
        """
        _ , _, _, transformed_derivatives = self.get_gridcell_centers_and_derivatives(controls)
        
        if not all(res == self.grid.resolution[0] for res in self.grid.resolution):
            raise ValueError("All entries of self.grid.resolution must be equal, otherwise the implementation has to be changed.")
        gridcellwidth = 1/self.grid.resolution[0]

        td_2d = transformed_derivatives.reshape(-1, transformed_derivatives.shape[-1])
        td_abs = np.abs(td_2d) 

        def neighbor_transition_fraction(t):
            if t <= 0:
                return 0  # Avoid division by zero or negative t
            stretched_td =  t * td_abs
            stretched_td_minus_w = stretched_td - gridcellwidth/2

            stretched_td_to_neighbor = stretched_td_minus_w[~np.all(stretched_td_minus_w < 0, axis=1)] - 2*gridcellwidth/2
            negative_rows_count_neighbor = np.sum(np.all(stretched_td_to_neighbor < 0, axis=1))

            return -negative_rows_count_neighbor/len(td_abs)


        # Step 1: Global search
        global_result = differential_evolution(lambda t: neighbor_transition_fraction(t), bounds=[search_space]) 
        global_t = global_result.x[0]
        print(f"Global search optimal delta_t: {global_t:.4f}, max_fraction: {-global_result.fun:.4f}")

        # Step 2: Local refinement
        refinement_radius = 0.2 * global_t
        local_result = minimize_scalar(lambda t: neighbor_transition_fraction(t), bounds=(global_t-refinement_radius, global_t+refinement_radius), method="bounded")
        print(f"Local search optimal delta_t: {local_result.x:.4f}, max_fraction: {-local_result.fun:.4f}")

        optimal_delta_t = local_result.x
        #max_fraction = -local_result.fun
    

        if path_to_save_plot is not None:

            fig, ax = plt.subplots()
        
            t_values = np.linspace(search_space[0], search_space[1], 100)  # Avoid t=0 to prevent division by zero
            coverage_values = [-neighbor_transition_fraction(t) for t in t_values]

            ax.plot(t_values, coverage_values)
            ax.axvline(x=optimal_delta_t, color='r', linestyle='--', label=f'Optimal $\\Delta t$ = {optimal_delta_t:.4f}')
            ax.set_xlabel(f'$\\Delta t$')
            ax.set_ylabel('Fraction of direct neighbor transitions')
            ax.set_title(f'Direct neighbor transitions of center gradients as a function of $\\Delta t$')
            ax.legend()
            ax.grid(True)
            fig.savefig(path_to_save_plot, bbox_inches='tight', dpi=300)
            
        return optimal_delta_t



    def calculate_importance_measure(self, controls, method='angular', alpha=1.0, debug=False):
        """
        Calculate a normalized importance measure for each grid cell based on the variance 
        of derivatives across different control values.
        
        Args:
            controls (np.array): Array of shape (n_controls, control_dim) with different control values
            method (str): Method to calculate variance: 'norm', 'component_wise', or 'angular'
            alpha (float): Power transformation parameter (0-1). Lower values make distribution more uniform.
            
        Returns:
            np.array: Normalized importance measure for each grid cell, shape matches grid resolution
        """
        # Get centers and derivatives for all controls
        centers, _, derivatives, _ = self.get_gridcell_centers_and_derivatives(controls)
        
        # Calculate variance based on selected method
        if method == 'component_wise':
            # Calculate variance for each component separately
            component_variance = np.var(derivatives, axis=0)  # shape: (n_cells, n_dims)
            
            # Sum variance across all dimensions
            variance = np.sum(component_variance, axis=1)  # shape: (n_cells)
        
        elif method == 'angular':
            # Initialize array for importance
            importance = np.zeros(self.grid.resolution)
            
            # Debug information
            if debug:
                debug_info = {}
            
            # Directly iterate over grid indices to avoid flattening issues
            for idx in self.grid.indices:
                # Convert grid index to flattened index for accessing centers/derivatives
                flat_idx = np.ravel_multi_index(idx, self.grid.resolution)
                
                # Get vectors for this cell
                cell_vectors = derivatives[:, flat_idx, :]
                cell_coord = centers[flat_idx]
                
                # Calculate angles between control vectors
                angles = []
                for j in range(len(controls)):
                    norm_j = np.linalg.norm(cell_vectors[j])
                    if norm_j < 1e-10:
                        continue
                    unit_j = cell_vectors[j] / norm_j
                    
                    for k in range(j+1, len(controls)):
                        norm_k = np.linalg.norm(cell_vectors[k])
                        if norm_k < 1e-10:
                            continue
                        unit_k = cell_vectors[k] / norm_k
                        
                        # Calculate angle (0 to π)
                        cos_angle = np.clip(np.dot(unit_j, unit_k), -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                        angles.append(angle)
                
                # Store mean angle as importance
                mean_angle = np.mean(angles) if angles else 0
                importance[idx] = mean_angle
                
                # Store debug info
                if debug:
                    debug_info[idx] = {
                        'coord': cell_coord,
                        'angles': angles,
                        'mean_angle': mean_angle,
                        # Clear definition of above/below diagonal
                        'is_above_diagonal': cell_coord[1] > cell_coord[0],  # x2 > x1
                        'is_below_diagonal': cell_coord[1] < cell_coord[0]   # x2 < x1
                    }
            
            # Additional debugging
            if debug:
                above_diagonal = [info['mean_angle'] for idx, info in debug_info.items() if info['is_above_diagonal']]
                below_diagonal = [info['mean_angle'] for idx, info in debug_info.items() if info['is_below_diagonal']]
                
                print(f"Average angle above diagonal (x2 > x1): {np.mean(above_diagonal) if above_diagonal else 'N/A'}")
                print(f"Average angle below diagonal (x2 < x1): {np.mean(below_diagonal) if below_diagonal else 'N/A'}")
                
                # Find cell with max angle
                if debug_info:
                    max_angle_cell = max(debug_info.items(), key=lambda x: x[1]['mean_angle'] if x[1]['angles'] else 0)
                    print(f"Cell with max angular difference: {max_angle_cell[0]}, value: {max_angle_cell[1]['mean_angle']}")
                    print(f"Located at coordinates: {max_angle_cell[1]['coord']}")
                    print(f"This cell is {'above' if max_angle_cell[1]['is_above_diagonal'] else 'below' if max_angle_cell[1]['is_below_diagonal'] else 'on'} diagonal")
            
            # Flatten importance array for the rest of the function
            variance = importance.flatten()
        
        else:
            raise ValueError(f"Unknown variance calculation method: {method}")
        
        # Reshape to match grid structure
        importance = variance.reshape(self.grid.resolution)
        
        # Ensure positive importance measure (add small epsilon to avoid zero variance)
        importance = importance + 1e-10
        
        # Apply power transformation to compress the range
        transformed_importance = importance ** alpha
        
        # Normalize to get weights that sum to 1.0
        normalized_importance = transformed_importance / np.sum(transformed_importance)
        
        return normalized_importance
    
    
    def visualize_derivatives_at_cell(self, grid_idx, controls):
        # NOTE: I just used this to bug fix
        """Visualize how derivatives change with different controls at a specific grid cell"""
        center = self.grid.choose_random_point_from_cell(grid_idx)
        center = np.array([center])  # Convert to appropriate shape
        
        # Calculate derivatives for each control
        all_derivs = []
        for c in controls:
            deriv = self.solver.get_derivative(center, c)
            all_derivs.append(deriv[0])  # First (only) point
            
        # Plot vectors
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(controls)))
        
        for i, deriv in enumerate(all_derivs):
            ax.arrow(center[0, 0], center[0, 1], deriv[0], deriv[1], 
                    head_width=0.05, head_length=0.1, fc=colors[i], ec=colors[i], 
                    label=f"Control {i}")
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(f'Derivative vectors at grid cell {grid_idx}')
        ax.legend()
        
        # Add the diagonal
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lim_max = max(xlim[1], ylim[1])
        ax.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.5)
        
        plt.grid(True)
        return fig
    
    def visualize_importance_measure(self, importance, title="Cell Importance Measure", 
                                    cmap="viridis", save_path=None, figsize=(10, 8)):
        """
        Visualize the importance measure for each grid cell as a heatmap with x-space axis labels.
        
        Args:
            importance (np.array): Importance measure array with shape matching grid resolution
            title (str): Title for the plot
            cmap (str): Colormap to use for the heatmap
            save_path (str): Optional path to save the figure
            figsize (tuple): Figure size (width, height) in inches
            
        Returns:
            fig, ax: The figure and axis objects
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get bounds in transformed space (z-space)
        z_bounds = self.grid.tf_bounds
        
        # Create the heatmap - use origin='lower' to ensure the origin is at bottom left
        im = ax.imshow(importance.T, extent=[z_bounds[0][0], z_bounds[0][1], 
                                            z_bounds[1][0], z_bounds[1][1]], 
                    origin='lower', aspect='auto', cmap=cmap)
        
        # Note: We transpose the importance array so that x1 is horizontal and x2 is vertical
        # The .T ensures that im[i,j] corresponds to x1=i, x2=j
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Normalized Importance')
        
        # Define formatters to convert z-space ticks to x-space
        def format_x1_tick(z, pos):
            x = self.grid.inverse_transformations[0](z)
            if np.isinf(x) or x > 1000:
                return "∞"
            elif x < 0.01:
                return f"{x:.2e}"
            else:
                return f"{x:.2f}"
                
        def format_x2_tick(z, pos):
            x = self.grid.inverse_transformations[1](z)
            if np.isinf(x) or x > 1000:
                return "∞"
            elif x < 0.01:
                return f"{x:.2e}"
            else:
                return f"{x:.2f}"
        
        # Set tick formatters to convert from z-space to x-space
        ax.xaxis.set_major_formatter(FuncFormatter(format_x1_tick))
        ax.yaxis.set_major_formatter(FuncFormatter(format_x2_tick))
        
        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Set labels and title
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(title)
        
        # Add grid lines at cell boundaries if needed
        if hasattr(self.grid, 'tf_grid_lines'):
            for x in self.grid.tf_grid_lines[0]:
                ax.axvline(x, color='gray', linestyle='--', alpha=0.3)
            for y in self.grid.tf_grid_lines[1]:
                ax.axhline(y, color='gray', linestyle='--', alpha=0.3)
        
        # Add diagonal line (x1 = x2)
        # First, create a series of points in original space along the diagonal
        if self.grid.transformed_bool:
            # Create points in original space
            x_bounds = self.grid.bounds
            x_min = max(x_bounds[0][0], x_bounds[1][0])  # Take the larger of the two minimums
            x_max = min(x_bounds[0][1], x_bounds[1][1])  # Take the smaller of the two maximums
            
            if np.isinf(x_max):
                # If upper bound is infinity, use a large finite value
                x_max = 100.0 if x_min < 100.0 else x_min * 10
            
            # Create diagonal points in original space
            diagonal_points = np.linspace(x_min, x_max, 100)
            diagonal_coords = np.column_stack((diagonal_points, diagonal_points))
            
            # Transform to z-space
            z_diagonal = self.grid.transform(diagonal_coords)
            
            # Plot the diagonal line
            ax.plot(z_diagonal[:, 0], z_diagonal[:, 1], 'k--', label='x1 = x2', linewidth=1.5)
        else:
            # For untransformed grids, diagonal is a simple line
            diagonal = np.array([
                [z_bounds[0][0], z_bounds[1][0]],
                [z_bounds[0][1], z_bounds[1][1]]
            ])
            ax.plot([diagonal[0, 0], diagonal[1, 0]], 
                    [diagonal[0, 1], diagonal[1, 1]], 
                    'k--', label='x1 = x2', linewidth=1.5)
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Figure saved to {save_path}")
        
        plt.tight_layout()
        return fig, ax
    
    def importance_to_samples(self, importance_measure, total_samples, min_samples_per_cell=1):
        """
        Convert an importance measure to number of samples per cell.
        
        Args:
            importance_measure: array with shape matching grid.resolution (assumed to sum to 1)
            total_samples: total number of samples to allocate
            min_samples_per_cell: minimum number of samples per cell (default: 1)
            
        Returns:
            samples_per_cell: array with shape matching grid.resolution, containing
                            the number of samples to generate for each cell
        """
        # Get number of cells
        num_cells = np.prod(self.grid.resolution)
        
        # Calculate minimum required samples
        min_required = num_cells * min_samples_per_cell
        
        # Make sure total_samples is enough to satisfy minimum samples per cell
        if total_samples < min_required:
            raise ValueError(f"total_samples must be at least {min_required} to ensure {min_samples_per_cell} sample(s) per cell")
        
        # Allocate minimum samples to each cell
        samples_per_cell = np.full_like(importance_measure, min_samples_per_cell, dtype=int)
        
        # Calculate remaining samples to distribute
        remaining_samples = total_samples - min_required
        
        if remaining_samples > 0:
            # Calculate continuous distribution of remaining samples
            continuous_distribution = importance_measure * remaining_samples
            
            # Initialize an array for additional samples (beyond the minimum per cell)
            additional_samples = np.floor(continuous_distribution).astype(int)
            
            # Calculate how many samples are still unassigned due to rounding down
            unassigned = remaining_samples - np.sum(additional_samples)
            
            # Distribute the remaining samples based on fractional parts
            if unassigned > 0:
                # Get fractional parts
                fractional_parts = continuous_distribution - additional_samples
                
                # Convert to flat array for easier handling
                flat_fractional = fractional_parts.flatten()
                
                # Get indices of cells with largest fractional parts
                flat_indices = np.argsort(flat_fractional)[-int(unassigned):]
                
                # Convert flat indices back to multi-dimensional indices
                multi_indices = np.unravel_index(flat_indices, importance_measure.shape)
                
                # Create a temporary array to hold the additions
                additions = np.zeros_like(additional_samples)
                additions[multi_indices] += 1
                
                # Add these to the additional_samples
                additional_samples += additions
            
            # Add additional samples to base samples
            samples_per_cell += additional_samples
        
        return samples_per_cell

    def get_importance_based_samples(self, samples_per_cell):
        """
        Generate samples from grid cells based on importance sampling.
        
        Args:
            samples_per_cell: Array with shape matching grid resolution containing
                            the number of samples to generate for each cell
        
        Returns:
            tuple: (X, trajectory_ids) where:
                - X is an array of points in original space (shape: (total_samples, dimension))
                - trajectory_ids is an array of IDs with format "i-j_k" where i-j is the cell
                index and k is the sample number within that cell
        """
        # Calculate total number of samples
        total_samples = np.sum(samples_per_cell)
        
        # Pre-allocate arrays
        all_points = np.zeros((total_samples, self.grid.dimension))
        all_ids = np.empty(total_samples, dtype=object)
        
        # Index to keep track of where we are in the output arrays
        current_idx = 0
        
        # Iterate through all grid cells
        for idx in self.grid.indices:
            # Get number of samples for this cell
            n_samples = samples_per_cell[idx]
            
            if n_samples > 0:
                # Generate points for this cell
                cell_points = self.grid.choose_multiple_random_points_from_cell(idx, n_samples)
                
                # Store points
                all_points[current_idx:current_idx + n_samples] = cell_points
                
                # Generate and store trajectory IDs
                cell_id = "-".join(map(str, idx))
                for j in range(n_samples):
                    all_ids[current_idx + j] = f"{cell_id}_{j}"
                
                # Update the index
                current_idx += n_samples
        
        return all_points, all_ids
    

    
    def store_results_to_sqlite(self, filename='simulation_results.db'):
        """Store simulation results in SQLite database using SQLAlchemy"""
        
        if self.results.empty:
            raise ValueError("No results to store.")

        # Extract model metadata
        metadata = json.loads(self.configs.iloc[0]['metadata'])
        model_config = metadata['configurations']['solver']['model']
        model_name = model_config['model']
        control_params = model_config.get('control_params', [])

        # Create results table name
        control_params_str = '_'.join(control_params) if control_params else 'no_control'
        results_table_name = f"{model_name}_{control_params_str}"
        
        # Setup database
        engine = get_engine(filename)
        
        # Create results table
        results_table = create_results_table(
            results_table_name, 
            x_dim=self.x_dim,
            control_dim=self.control_dim
        )
        
        # Create all tables
        init_db(engine)
        
        try:
            # Restructure configs data to match schema and convert dicts to JSON strings
            configs_data = pd.DataFrame({
                'run_id': self.configs['run_id'],
                'configurations': self.configs['metadata'].apply(
                    lambda x: json.dumps(json.loads(x)['configurations'])
                ),
                'simulation_params': self.configs['metadata'].apply(
                    lambda x: json.dumps(json.loads(x)['simulation_params'])
                ),
                'performance': self.configs['metadata'].apply(
                    lambda x: json.dumps(json.loads(x)['performance'])
                ),
                'system': self.configs['metadata'].apply(
                    lambda x: json.dumps(json.loads(x)['system'])
                )
            })
            
            # Store data
            configs_data.to_sql('configs', engine, index=False, if_exists='append')
            self.results.to_sql(results_table_name, engine, index=False, if_exists='append')
            
        except Exception as e:
            print(f"Error storing results: {e}")
            raise


def run_and_store_simulations(output_dir, 
                              bounds, 
                              transformations, 
                              model, 
                              solver, 
                              control, 
                              resolution, 
                              num_samples_per_cell, 
                              num_steps, 
                              delta_t):
    """Simulate and store simulation results in a SQLite database.
    
    Args:
        output_dir (str or Path): Absolute path to output directory
        bounds (list): List of tuples containing the lower and upper bounds of the grid
        transformations (list): List of transformation functions
        model : Model to simulate
        solver(model): Solver to use, already initialized with the model
        control (scalar or array): Control parameters for the simulation
        resolution (list of ints): Number of grid cells per dimension
        num_samples_per_cell (int): Number of samples per grid cell
        num_steps (int): Number of simulation steps
        delta_t (float): Time step size
    """
    # Setup paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create grid and simulator
    grid = Grid(bounds, resolution, transformations)
    simulator = Simulator(grid, model, solver)
    
    # Run simulation
    results = simulator.simulate(
        control=control,
        delta_t=delta_t,
        num_samples_per_cell=num_samples_per_cell,
        num_steps=num_steps,
        save_result=True
    )
    
    # Store results
    db_path = output_path / f'simulation_results.db'
    simulator.store_results_to_sqlite(db_path)
    
    print(f"Stored {len(results)} rows of simulation data in {db_path}")
    print("\nConfigs data:")
    print(simulator.configs.head())
    print("\nResults data:")
    print(simulator.results.head())

    return simulator
