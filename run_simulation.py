import os
import sys
import json
import time
from pathlib import Path
import argparse
import numpy as np
import logging
import time
import matplotlib.pyplot as plt
from math import floor, log10
from data_generation.models.tech_substitution import TechnologySubstitution, TechSubNumericalSolver
from data_generation.models.general_ode_solver import FitzHughNagumoModel, GeneralODENumericalSolver
from data_generation.models.simple_test_models import *
from data_generation.simulations.grid import Grid, VoronoiGrid, fractional_transformation, logistic_transformation
from data_generation.simulations.simulator import Simulator
from datasets.database import clear_data_by_run_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output logs to the console
        logging.FileHandler("data_generation/simulations/simulation.log")  # Save logs to a file
    ]
)


def load_config(config_path):
    """Load simulation configuration from a JSON file."""
   
    with open(config_path, "r") as file:
        return json.load(file)


def determine_delta_t(config, simulator, config_path, significant_figures=3):
    """
    Determine the optimal delta_t for the simulation, 
    round it to a specified number of significant figures, 
    and update the configuration file.
    """
    
    search_space = config['delta_t']
    delta_t_optimization_plot_path = config_path.replace(".json", "") + "_delta_t_optimization_plot.png" #TODO maybe improve this
    

    optimal_delta_t = simulator.get_optimal_delta_t(np.array(config['controls']), search_space, path_to_save_plot = delta_t_optimization_plot_path)
    
    while True:
        #Show the plot and ask the user for confirmation
        print(f"Plot of optimal delta_t search saved at {delta_t_optimization_plot_path}.")

        user_input = input(f"Optimal delta_t found at {optimal_delta_t}. Proceed with this value? (yes/no): ").strip().lower()
        if user_input in ['yes', 'y']:
            plt.close()
            break
        else:
            new_bounds = input("Enter new bounds for delta_t search as a comma-separated list (e.g., 0.01,0.1): ").strip()
            try:
                new_bounds = [float(bound) for bound in new_bounds.split(',')]
                optimal_delta_t = simulator.get_optimal_delta_t(np.array(config['controls']), new_bounds, path_to_save_plot = delta_t_optimization_plot_path)
            except ValueError:
                logging.error("Invalid bounds entered. Exiting.")
                sys.exit(1)
            plt.close()

    # Use a rounded value for delta_t
    config['delta_t'] =  round(optimal_delta_t, -int(floor(log10(abs(optimal_delta_t)))) + (significant_figures - 1)) #Rounds to number of significant figures
    config['comment'] += f"  Delta_t value was optimized at {time.strftime('%Y-%m-%d %H:%M:%S')}."
    print(config_path)
    # Save the updated configuration back to the JSON file
    with open(config_path, "r") as file:
        save_config = json.load(file)
    with open(config_path, "w") as file:
        save_config['delta_t'] = config['delta_t']
        save_config['comment'] = config['comment']
        json.dump(save_config, file, indent=4)


    logging.info(f"Delta_t was not provided. Using optimized value: {config['delta_t']}")
    
    return config

def replace_inf(value):
    """Recursively replace 'np.inf' strings with actual np.inf values."""
    if isinstance(value, list):
        return [replace_inf(v) for v in value]
    if value == "np.inf":
        return np.inf
    if value == "-np.inf":
        return -np.inf
    return value


def update_run_dict(config):
    """ Update the run dictionary with the current simulation parameters for overview"""
    
    run_dict_path = config['output_path'] / config['run_dict_name']
    if run_dict_path.exists():
        with open(run_dict_path, "r") as file:
            run_dict = json.load(file)
        if config['model'] not in run_dict.keys():
            run_dict[config['model']] = {}
            next_key = 1
        else:
            next_key = sorted([int(key) for key in run_dict[config['model']].keys()])[-1] +1 
    else:
        run_dict = {}
        run_dict[config['model']] = {}
        next_key = 1

    
    model_run_dict = run_dict[config['model']]   
    model_run_dict[next_key] = dict()
    model_run_dict[next_key]['run_ids'] = list()
    model_run_dict[next_key]['resolution'] = config["resolution"] if 'resolution' in config else None
    model_run_dict[next_key]['numbercells'] = config["numbercells"] if 'numbercells' in config else None
    model_run_dict[next_key]['num_samples_per_cell'] = config['num_samples_per_cell']
    model_run_dict[next_key]['controls'] = list(config['controls'])
    model_run_dict[next_key]['delta_t'] = config['delta_t']
    model_run_dict[next_key]['transformations'] = config["transformations"]
    model_run_dict[next_key]['comment'] = config['comment']

    return next_key, run_dict

def store_run_dict(run_dict, config):
    with open(config['output_path']/  config['run_dict_name'], "w") as file:
        json.dump(run_dict, file, indent=4)
    return



def initialize_simulation(config, config_path):
    """Initialize the grid, model, and solver based on the configuration."""
    
    if config["transformations"] != None: #check if it not nOe
        transformations = [globals()[transformation](config['trafo_params'][i]) for i, transformation in enumerate(config["transformations"])]
    else:
        transformations = None
    
    config["bounds"] = replace_inf(config["bounds"])

    if "grid_type" in config and config["grid_type"] == "Voronoi":
        logging.info("Voronoi grid type detected.")
        grid = VoronoiGrid(bounds=config["bounds"], 
                           numbercells=config.get("numbercells", 10), 
                           grid_transformations = transformations, 
                           seed=config.get("grid_seed", None),
                           cell_centers=config.get("cell_centers", None),
                           padding_margin= config.get("padding_margin", 'auto')
                           )
    else:
        grid = Grid(bounds=config["bounds"], resolution=config["resolution"], grid_transformations=transformations)

    if 'control_params' in config.keys():
        model = globals()[config["model"]](control_params=config["control_params"])
    else:
        model = globals()[config["model"]]()

    solver = globals()[config["solver"]](model)
    simulator = Simulator(grid, model, solver)
    
    # Check if delta_t is given, else determine optimal delta_t and save it to the config json file
    if type(config['delta_t']) == list:
        config = determine_delta_t(config, simulator, config_path)

    # Log simulation start
    logging.info(f"Initialized simulation of {config['model']} with the following parameters:")
    logging.info(f"Bounds: {config['bounds']}")
    logging.info(f"Transformations: {None if transformations == None else [tf[0].__name__ for tf in transformations]}")
    logging.info(f"Transformation Parameters: {config['trafo_params']}")

    logging.info(f"Resolution: {config['resolution']}") if 'resolution' in config else None
    logging.info(f"Grid Type: {config.get('grid_type', 'Regular')}")  # Default to 'Regular' if not specified
    logging.info(f"Numbercells: {config['numbercells']}") if 'numbercells' in config else None
    logging.info(f"Padding Margin: {config['padding_margin']}") if 'padding_margin' in config else None

    logging.info(f"Number of samples per cell: {config['num_samples_per_cell']}")
    logging.info(f"Delta t: {config['delta_t']}")

    return simulator


def run_and_store_simulations(config, config_path):
    """Run simulations based on the configuration, possibly for multiple controls."""
    
    run_ids = []
    simulator = initialize_simulation(config, config_path)

    for control in np.array(config['controls']):
        
        try:
            
            logging.info(f"Starting simulation with control: {control}")
            # Run simulation
            simulator.simulate(
                control=control,
                delta_t=config['delta_t'],
                num_samples_per_cell=config['num_samples_per_cell'],
                num_steps=config['num_steps'],
                save_result=True )

            logging.info(f"Simulation completed for control: {control}")         
            # Store results
            db_path = config['output_path'] / config['db-name']
            simulator.store_results_to_sqlite(db_path)
            
            run_ids.append(simulator.configs['run_id'][0])
            
            # Log success and details
            logging.info(f"Stored {len(simulator.results)} rows of simulation data in {db_path}")
            logging.info("\nConfigs data:")
            logging.info(simulator.configs.drop(columns=['cell_centers'], errors='ignore').head().to_string())  # Exclude 'cell centers' column for logging
            logging.info("\nResults data:")
            logging.info(simulator.results.head().to_string())  # Convert DataFrame to string for logging
            
            logging.info("Simulation completed successfully.")
        
        except Exception as e:
            # Remove stored data for the specific run_ids from the database in case of failure
            logging.error(f"Simulation failed: {e}")
            try:
                db_path = config['output_path'] / config['db-name']
                model_name = config['model']
                control_params = config.get('control_params', [])
                control_params_str = '_'.join(control_params) if control_params else 'no_control' 
                tablename = f"{model_name}_{control_params_str}"
                print(db_path, tablename, run_ids)
                clear_data_by_run_id(db_path, tablename, run_ids)
                logging.info(f"Cleared stored data for run_ids {run_ids} from table {tablename} in {db_path} due to simulation failure.")
            except Exception as cleanup_error:
                logging.error(f"Failed to clear stored data for specific run_ids: {cleanup_error}")
                          
            sys.exit(1)  # Exit the program with an error code
        
        time.sleep(1) #if the simulation is faster than 1 second, to keep the ids unique

    if config["run_dict_name"]:
        key_run_dict, run_dict = update_run_dict(config)
        run_dict[config['model']][key_run_dict]['run_ids'].extend(run_ids)
        store_run_dict(run_dict, config)
        


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run simulations.")
    parser.add_argument("--config", type=str, help="Path to the configuration JSON file.") 
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Load from JSON configuration file
    config = load_config(args.config)


     # Setup paths
    config['output_path'] = Path(config['output_path'])
    config['output_path'].mkdir(parents=True, exist_ok=True)

    # Run simulations
    start_time = time.time()
    run_and_store_simulations(config, args.config)
    end_time = time.time()

    logging.info(f"Simulations completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()



# Example usage:
# python run_simulation.py --config configdata.json

# Example configdata.json:
# {   
# "model": "FitzHughNagumoModel",
# "control_params": ["b", "I"], 
# "solver": "GeneralODENumericalSolver",
# "bounds": [[0, "np.inf"], [0, "np.inf"]],
# "resolution": [30,30],
# "trafo_params": [{"k": 1, "x_0": 0}, {"k": 1, "x_0": 0}],
# "transformations": ["logistic_transformation", "logistic_transformation"],
# "controls": [[0.5, 0.35], [2, 0], [2, 0.35]],
# "num_samples_per_cell": 200,
# "num_steps": 1,
# "delta_t": 0.11,
# "output_path": "datasets/results",
# "db-name": "fhn-model.db",
# "run_dict_name": "fhn-model-run-dict.json",
# "comment": "test"
# }"
