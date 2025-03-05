# run_simulation.py
import sys
import time
from pathlib import Path
import argparse
import numpy as np

# Import necessary modules
from data_generation.models.tech_substitution import TechnologySubstitution, NumericalSolver as TechNumericalSolver
from data_generation.simulations.grid import Grid, tangent_transformation
from data_generation.simulations.simulator import Simulator

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run technology substitution simulation')
    parser.add_argument('--output-dir', type=str, default='datasets/results', 
                        help='Output directory for results')
    parser.add_argument('--db-name', type=str, default='tech_toy.db',
                        help='Database filename')
    parser.add_argument('--control', type=float, default=0.5,
                        help='Control parameter value')
    parser.add_argument('--delta-t', type=float, default=3.0,
                        help='Time step size')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='Number of simulation steps')
    parser.add_argument('--resolution', type=int, nargs=2, default=[10, 10],
                        help='Grid resolution as two integers, e.g., 10 10')
    parser.add_argument('--samples-per-cell', type=int, default=10,
                        help='Number of samples per cell')
    
    args = parser.parse_args()
    
    # Setup parameters
    bounds = [(0, np.inf), (0, np.inf)]
    transformations = [tangent_transformation(3, alpha=0.5), tangent_transformation(3, alpha=0.5)]
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running simulation with parameters:")
    print(f"  Control: {args.control}")
    print(f"  Delta t: {args.delta_t}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Samples per cell: {args.samples_per_cell}")
    print(f"  Output: {output_dir / args.db_name}")
    
    try:
        # Create model, solver, grid and simulator
        model = TechnologySubstitution()
        num_solver = TechNumericalSolver(model)
        grid = Grid(bounds, args.resolution, transformations)
        simulator = Simulator(grid, model, num_solver)
        
        print("Running simulation...")
        # Run simulation
        start_time = time.time()
        results = simulator.simulate(
            control=args.control,
            delta_t=args.delta_t,
            num_samples_per_cell=args.samples_per_cell,
            num_steps=args.num_steps,
            save_result=True
        )
        print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")
        
        print(f"Storing results to {args.db_name}...")
        # Store results to database
        simulator.store_results_to_sqlite(filename=str(output_dir / args.db_name))
        
        print(f"Simulation completed successfully. Results stored in {output_dir / args.db_name}")
        
        # Display summary of results
        num_trajectories = len(results['trajectory_id'].unique())
        print(f"Generated {len(results)} data points across {num_trajectories} trajectories.")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

# python run_simulation.py --resolution 10 10 --samples-per-cell 20 --num-steps 5 --delta-t 15.0 --control 1.0 --db-name tech_toy2.db