import numpy as np
import pandas as pd
from ..models import tech_substitution as ts
from datetime import datetime

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
        # do we need more?

    def _init_df(self):
        columns = ['run_id', 'trajectory_id', 't0', 't1']
        columns.extend([f'x{i}' for i in range(self.x_dim)])
        columns.extend([f'c{i}' for i in range(self.control_dim)])
        columns.extend([f'y{i}' for i in range(self.x_dim)])
        return pd.DataFrame(columns=columns)
                

    def simulate(self, control, delta_t, ids, num_samples_per_cell=10, save_result=False):   
        # TODO the solver step function will return a trajectory
        """
        Simulates the differential equation for a given initial condition and time period.
        
        Args:

        Returns: df (DataFrame): A DataFrame containing the simulation results.
        """

        X = self.grid.get_initial_conditions(num_samples_per_cell)
        n_samples = X.shape[0]
        
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # TODO trajectory id should give information about the cell and sample number
        trajectory_ids = np.arange(n_samples) 

        t0 = 0.0

        # two control cases
        if control.ndim == 1:
            # Same control for all -> expand to match n_samples
            control = np.tile(control, (n_samples, 1))
        elif control.shape[0] != n_samples:
            raise ValueError(f"Controls must have shape ({n_samples}, {self.control_dim}) or ({self.control_dim},)")

        # do the step
        Y = self.solver.step(X, control, delta_t)
        
        new_data = {
            'run_id': np.full(n_samples, run_id),
            'trajectory_id': trajectory_ids,
            't0': np.full(n_samples, t0),
            't1': np.full(n_samples, t0 + delta_t)
        }

        # Add observation dimensions
        for i in range(X.shape[1]):
            new_data[f'x{i}'] = X[:, i]
            new_data[f'y{i}'] = Y[:, i]
            
        # Add control dimensions
        for i in range(control.shape[1]):
            new_data[f'c{i}'] = control[:, i]
        
        df = pd.DataFrame(new_data)

        # Possibly store the resulting df in the self.result_df attribute (by adding it to the existing df)
        if save_result == True:
            self.results = pd.concat([self.results, df], ignore_index=True)

        # NOTE: We could also make this function not return the df object, 
        # since it is already saving it anyway. Could instead give a status update
        # or even return nothing

        return df

    
    def store_results_to_sqlite(self, filename='simulation_results.db'):
        
        '''
        Store the results of the simulation in a sqlite database, NOT YET IMPLEMENTED.
        Args:
            filename (str): The name of the sqlite database file.
        '''


        if self.result_df is None:
            raise ValueError("No results to store.")
        
        
        

        raise NotImplementedError("Storing results in sqlite database is not yet implemented.")