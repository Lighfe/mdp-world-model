import time
import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import sqlite3
import sys

# Import necessary modules
from data_generation.models.social_tipping import SocialTipping
from data_generation.simulations.grid import Grid
from data_generation.simulations.simulator import Simulator
from data_generation.models.general_ode_solver import GeneralODENumericalSolver

def generate_balanced_dataset(actions, delta_t_values, total_samples=20000, 
                            grid_resolution=[20, 20], output_dir='datasets/results',
                            db_name='balanced_dataset.db'):
    """
    Generate a balanced dataset across actions, delta_t values, and grid cells.
    """
    # Ensure delta_t_values is a list
    if not isinstance(delta_t_values, list):
        delta_t_values = [delta_t_values]
    
    # Setup grid - no transformations needed since your space is already [0,1]
    bounds = [(0, 1), (0, 1)]
    
    # Calculate total number of combinations
    n_actions = len(actions)
    n_delta_t = len(delta_t_values)
    n_grid_cells = grid_resolution[0] * grid_resolution[1]
    total_combinations = n_actions * n_delta_t * n_grid_cells
    
    # Calculate samples per combination
    samples_per_combination = max(1, total_samples // total_combinations)
    
    print(f"Generating balanced dataset:")
    print(f"  Actions: {actions}")
    print(f"  Delta_t values: {delta_t_values}")
    print(f"  Grid resolution: {grid_resolution}")
    print(f"  Total combinations: {total_combinations}")
    print(f"  Samples per combination: {samples_per_combination}")
    print(f"  Expected total samples: {samples_per_combination * total_combinations}")
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # Generate data for each combination - collect everything first
    all_results_data = []
    all_configs_data = []
    
    simulation_count = 0
    for action in actions:
        for delta_t in delta_t_values:
            simulation_count += 1
            print(f"Running simulation {simulation_count}/{len(actions) * len(delta_t_values)} for action='{action}', delta_t={delta_t}")
            
            # Create fresh instances for each simulation
            model = SocialTipping()
            solver = GeneralODENumericalSolver(model)
            grid = Grid(bounds, grid_resolution)
            simulator = Simulator(grid, model, solver)
            
            # Get control parameters
            control_params = model.get_control_params(action)
            control = [control_params['b'], control_params['c'], 
                      control_params['f'], control_params['g']]
            
            # Run simulation
            results = simulator.simulate(
                control=control,
                delta_t=delta_t,
                avg_samples_per_cell=samples_per_combination,
                num_steps=1,
                save_result=True
            )
            
            print(f"  Generated {len(results)} samples")
            
            # Create unique run_id to avoid conflicts
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_run_id = f"{timestamp}_{action}_{delta_t:.3f}_{simulation_count}"
            
            # Get the data and update run_ids
            sim_results = simulator.results.copy()
            sim_configs = simulator.configs.copy()
            
            # Update run_ids to be unique
            sim_results['run_id'] = unique_run_id
            if len(sim_configs) > 0:
                sim_configs['run_id'] = unique_run_id
            
            # Collect data
            all_results_data.append(sim_results)
            all_configs_data.append(sim_configs)
    
    end_time = time.time()
    print(f"All simulations completed in {end_time - start_time:.2f} seconds.")
    
    # Combine all data
    print("Combining all simulation data...")
    if all_results_data:
        combined_results = pd.concat(all_results_data, ignore_index=True)
    else:
        combined_results = pd.DataFrame()
    
    if all_configs_data:
        # Filter out empty configs
        non_empty_configs = [cfg for cfg in all_configs_data if len(cfg) > 0]
        if non_empty_configs:
            combined_configs = pd.concat(non_empty_configs, ignore_index=True)
        else:
            combined_configs = pd.DataFrame()
    else:
        combined_configs = pd.DataFrame()
    
    # Store to database using simple SQLite
    print(f"Storing {len(combined_results)} results and {len(combined_configs)} configs to {db_name}...")
    
    db_path = output_dir / db_name
    
    # Remove existing database to ensure clean table order (results first)
    if db_path.exists():
        db_path.unlink()
        print(f"✓ Removed existing database to ensure clean table order")
    
    conn = sqlite3.connect(str(db_path))
    
    try:
        # Store results FIRST (will create table first)
        if len(combined_results) > 0:
            combined_results.to_sql('results', conn, index=False, if_exists='replace')
            print(f"✓ Stored {len(combined_results)} result rows")
        
        # Store configs SECOND (will create table second)
        if len(combined_configs) > 0:
            combined_configs.to_sql('configs', conn, index=False, if_exists='replace')
            print(f"✓ Stored {len(combined_configs)} config rows")
        
        conn.close()
        
        print(f"✓ Dataset generation completed successfully!")
        print(f"✓ Results stored in {db_path}")
        
        # Display summary
        print(f"✓ Generated {len(combined_results)} total data points.")
        
        # Show unique run_ids to verify they're all different
        if len(combined_results) > 0:
            unique_run_ids = combined_results['run_id'].unique()
            print(f"✓ Created {len(unique_run_ids)} unique run_ids")
            for run_id in unique_run_ids[:5]:  # Show first 5
                print(f"  - {run_id}")
            if len(unique_run_ids) > 5:
                print(f"  ... and {len(unique_run_ids) - 5} more")
        
        return combined_results
        
    except Exception as e:
        conn.close()
        print(f"✗ Error storing to database: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Generate balanced social tipping dataset')
    
    # Dataset parameters
    parser.add_argument('--actions', type=str, nargs='+', 
                       default=['default', 'subsidy', 'tax', 'campaign'],
                       help='List of actions to include')
    parser.add_argument('--delta-t', type=float, nargs='+', default=[1.0],
                       help='Delta_t value(s) to use')
    parser.add_argument('--total-samples', type=int, default=20000,
                       help='Total number of samples to generate')
    
    # Grid parameters
    parser.add_argument('--resolution', type=int, nargs=2, default=[20, 20],
                       help='Grid resolution as two integers')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='datasets/results',
                       help='Output directory for results')
    parser.add_argument('--db-name', type=str, default='balanced_dataset.db',
                       help='Database filename')
    
    args = parser.parse_args()
    
    try:
        results = generate_balanced_dataset(
            actions=args.actions,
            delta_t_values=args.delta_t,
            total_samples=args.total_samples,
            grid_resolution=args.resolution,
            output_dir=args.output_dir,
            db_name=args.db_name
        )
        
        print(f"\n🎉 SUCCESS: Generated dataset with {len(results)} trajectories!")
        
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



if __name__ == "__main__":
    main()

# Use all actions with single delta_t
#python run_social_tipping_simulation.py --actions default subsidy tax campaign --delta-t 1.0

# Use subset of actions with multiple delta_t values  
#python run_social_tipping_simulation.py --actions default subsidy --delta-t 0.7 7.0 20.0

# Adjust total samples and grid resolution
#python run_social_tipping_simulation.py --total-samples 40000 --resolution 30 30


# Hypothesis is that small delta_t will learn something different than high delta_t
#python run_social_tipping_simulation.py --db-name "social_tipping_default_t2.db" --actions default --delta-t 2.181
#python run_social_tipping_simulation.py --db-name "social_tipping_default_t7.db" --actions default --delta-t 7.27
#python run_social_tipping_simulation.py --db-name "social_tipping_default_t21.db" --actions default --delta-t 21.81
#python run_social_tipping_simulation.py --db-name "social_tipping_default_t58.db" --actions default --delta-t 58.887
# Hypothesis is that this will better learn with mix of delta_t
#python run_social_tipping_simulation.py --db-name "social_tipping_default_tmedium.db" --actions default --delta-t 7.27 21.81
#python run_social_tipping_simulation.py --db-name "social_tipping_default_tsmallhigh.db" --actions default --delta-t 2.181 58.887
#python run_social_tipping_simulation.py --db-name "social_tipping_default_tsmallhigh.db" --actions default --delta-t 2.181 7.27 21.81 58.887


# test multi actions
#python run_social_tipping_simulation.py --db-name "social_tipping_default_subsidy_t21.db" --actions default subsidy --delta-t 7.27
#python run_social_tipping_simulation.py --db-name "social_tipping_default_subsidy_t58.db" --actions default subsidy --delta-t 58.887

# test multi actions multi delta_t
#python run_social_tipping_simulation.py --db-name "social_tipping_default_tsmallhigh.db" --actions default subsidy --delta-t 2.181 7.27 21.81 58.887