import json
import time
import platform
import psutil
import numpy as np
import pandas as pd
from ..models import tech_substitution as ts
from . import grid as g
from datetime import datetime

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

        Returns: df (DataFrame): A DataFrame containing the simulation results.
        """
        # Start timing
        start_time = time.time()

        X, trajectory_ids = self.grid.get_initial_conditions(num_samples_per_cell)
        n_samples = X.shape[0]
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # NOTE: We should probably add hyper-parameter information somewhere and somehow, 
        # e.g., the transformation that was used to create the grid sample

        # Convert control to full array format
        control = np.array(control)
        control = self.create_control_array(control, num_steps, n_samples)

        # get full trajectory
        trajectory = self.solver.step(X, control, delta_t, num_steps)

        # Create timestamps
        times = np.arange(0, (num_steps + 1) * delta_t, delta_t)
        
        # Initialize data dictionary
        data = {
            'run_id': np.full(n_samples * num_steps, run_id),
            'trajectory_id': np.repeat(trajectory_ids, num_steps),
            't0': np.tile(times[:-1], n_samples),
            't1': np.tile(times[1:], n_samples)
        }
        
        # Add state and observation dimensions
        for i in range(self.x_dim):
            data[f'x{i}'] = trajectory[:-1, :, i].flatten(order='F')
            data[f'y{i}'] = trajectory[1:, :, i].flatten(order='F')
        
        # Add control dimensions
        for i in range(self.control_dim):
            # data[f'c{i}'] = np.repeat(control[:, i], num_steps)
            data[f'c{i}'] = control.flatten(order='F')
        
        # print(data) # debugging

        df = pd.DataFrame(data)

        print(f"Simulation complete:\n"
          f"- {n_samples} samples × {num_steps} timesteps = {n_samples * num_steps} total rows\n"
          f"- State dimensions: {self.x_dim}\n"
          f"- Control dimensions: {self.control_dim}")

        # Possibly store the resulting df in the self.result_df attribute (by adding it to the existing df)
        if save_result == True:
            self.results = pd.concat([self.results, df], ignore_index=True)

                # Collect all metadata in a dictionary
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
    
    def create_control_array(self, control, num_steps, n_samples):
        '''
        Takes as input a np.array of different shapes
        Returns array of shape (num_steps, n_samples)
        '''
        if control.ndim == 0:  # scalar
            control = np.full((num_steps, n_samples), control)
        elif control.ndim == 1:  # vector [0.5] or [0.5, 0.6, 0.7]
            if len(control) == 1:
                # Single value for all timesteps/samples
                control = np.full((num_steps, n_samples), control[0])
            elif len(control) == n_samples:
                # Different value for each sample, same across timesteps
                control = np.tile(control[np.newaxis, :], (num_steps, 1))
            elif len(control) == num_steps:
                # Different value for each timestep, same across samples
                control = np.tile(control[:, np.newaxis], (1, n_samples))
            else:
                raise ValueError(f"Control length {len(control)} doesn't match either num_steps={num_steps} or n_samples={n_samples}")
        elif control.ndim == 2 and control.shape == (1, 1): # edge case
            control = np.full((num_steps, n_samples), control[0,0])
            
        assert control.shape == (num_steps, n_samples), \
            f"Control shape {control.shape} doesn't match (num_steps={num_steps}, n_samples={n_samples})"
        
        return control

    
    def store_results_to_sqlite(self, filename='simulation_results.db'):
        
        '''
        Store the results of the simulation in a sqlite database, NOT YET IMPLEMENTED.
        Args:
            filename (str): The name of the sqlite database file.
        '''


        if self.results is None:
            raise ValueError("No results to store.")
        
        
        

        raise NotImplementedError("Storing results in sqlite database is not yet implemented.")
    