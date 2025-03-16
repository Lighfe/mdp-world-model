# run_simulation.py
import sys
import time
from pathlib import Path
import argparse
import numpy as np

# Import necessary modules
from data_generation.models.tech_substitution import TechnologySubstitution, TechSubNumericalSolver
from data_generation.simulations.grid import Grid, tangent_transformation
from data_generation.simulations.simulator import Simulator

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run technology substitution simulation')
    parser.add_argument('--output-dir', type=str, default='datasets/results', 
                        help='Output directory for results')
    parser.add_argument('--db-name', type=str, default='tech_importance.db',
                        help='Database filename')
    parser.add_argument('--control', type=float, default=0.5,
                        help='Control parameter value')
    parser.add_argument('--delta-t', type=float, default=2.65,
                        help='Time step size')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='Number of simulation steps')
    parser.add_argument('--resolution', type=int, nargs=2, default=[20, 20],
                        help='Grid resolution as two integers, e.g., 10 10')
    parser.add_argument('--avg-samples-per-cell', type=int, default=10,
                        help='Average number of samples per cell')
    
    # New arguments for importance-based sampling
    parser.add_argument('--use-importance', action='store_true',
                        help='Use importance-based sampling')
    parser.add_argument('--importance-method', type=str, default='angular', 
                        choices=['angular', 'norm', 'component_wise'],
                        help='Method to calculate importance measure')
    parser.add_argument('--importance-alpha', type=float, default=1.0,
                        help='Alpha parameter for importance calculation (power transformation, 0-1)')
    parser.add_argument('--min-samples-per-cell', type=int, default=1,
                        help='Minimum samples per cell when using importance-based sampling')
    parser.add_argument('--total-samples', type=int, default=None,
                        help='Total samples to distribute for importance-based sampling')
    parser.add_argument('--possible-controls', type=float, nargs='+',
                        help='List of control values to use for importance calculation (required with --use-importance)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_importance and not args.possible_controls:
        parser.error("--possible-controls is required when using --use-importance")
    
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
    
    if args.use_importance:
        print(f"  Using importance-based sampling:")
        print(f"    Method: {args.importance_method}")
        print(f"    Alpha: {args.importance_alpha}")
        print(f"    Min samples per cell: {args.min_samples_per_cell}")
        total_samples = args.total_samples or (np.prod(args.resolution) * args.avg_samples_per_cell)
        print(f"    Total samples: {total_samples}")
        print(f"    Possible controls: {args.possible_controls}")
    else:
        print(f"  Average samples per cell: {args.avg_samples_per_cell}")
    
    print(f"  Output: {output_dir / args.db_name}")
    
    try:
        # Create model, solver, grid and simulator
        model = TechnologySubstitution()
        num_solver = TechSubNumericalSolver(model)
        grid = Grid(bounds, args.resolution, transformations)
        simulator = Simulator(grid, model, num_solver)
        
        print("Running simulation...")
        # Run simulation
        start_time = time.time()
        
        # Prepare simulation parameters
        sim_params = {
            'control': args.control,
            'delta_t': args.delta_t,
            'num_steps': args.num_steps,
            'save_result': True
        }
        
        if args.use_importance:
            sim_params.update({
                'use_importance': True,
                'importance_method': args.importance_method,
                'min_samples_per_cell': args.min_samples_per_cell,
                'possible_controls': args.possible_controls,
                'total_samples': args.total_samples
            })
        else:
            sim_params['avg_samples_per_cell'] = args.avg_samples_per_cell
        
        results = simulator.simulate(**sim_params)
        
        print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")
        
        print(f"Storing results to {args.db_name}...")
        # Store results to database
        simulator.store_results_to_sqlite(filename=str(output_dir / args.db_name))
        
        print(f"Simulation completed successfully. Results stored in {output_dir / args.db_name}")
        
        # Display summary of results
        num_trajectories = len(results['trajectory_id'].unique())
        print(f"Generated {len(results)} data points across {num_trajectories} trajectories.")
        
        # Visualize importance measure if using importance-based sampling
        if args.use_importance:
            try:
                import matplotlib.pyplot as plt
                importance = simulator.calculate_importance_measure(
                    np.array(args.possible_controls), 
                    method=args.importance_method,
                    alpha=args.importance_alpha
                )
                
                fig_path = output_dir / f"importance_{args.db_name.replace('.db', '.png')}"
                fig, ax = simulator.visualize_importance_measure(
                    importance,
                    title=f"Importance Measure ({args.importance_method}, α={args.importance_alpha})",
                    save_path=str(fig_path)
                )
                print(f"Saved importance visualization to {fig_path}")
            except Exception as viz_error:
                print(f"Warning: Could not generate importance visualization: {viz_error}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

# Example usage:
# Basic uniform sampling:
# python run_simulation.py --resolution 10 10 --avg-samples-per-cell 20 --num-steps 5 --delta-t 15.0 --control 1.0 --db-name tech_toy2.db
#
# With importance-based sampling:
# python run_simulation.py --use-importance --importance-method angular --importance-alpha 0.5 --possible-controls 0.5 1.0 --total-samples 5000 --num-steps 5 --delta-t 15.0 --control 1.0 --db-name tech_importance.db