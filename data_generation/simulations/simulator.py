import json
import time
from datetime import datetime
import platform
import psutil
import numpy as np
import pandas as pd
from data_generation.models import tech_substitution as ts
from data_generation.simulations import grid as g
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



