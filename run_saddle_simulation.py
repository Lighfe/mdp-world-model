# run_saddle_simulation.py
import sys
import time
import json
from pathlib import Path
import argparse
import numpy as np

# Import necessary modules
from data_generation.models.saddle_system import MultiSaddleSystem
from data_generation.simulations.grid import Grid, logistic_transformation
from data_generation.simulations.simulator import Simulator
from data_generation.models.general_ode_solver import GeneralODENumericalSolver
from data_generation.visualization.create_plots import create_saddle_streamplot_visualization

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run saddle system simulation')
    parser.add_argument('--output-dir', type=str, default='datasets/results', 
                        help='Output directory for results')
    parser.add_argument('--db-name', type=str, default='saddle_system.db',
                        help='Database filename')
    parser.add_argument('--delta-t', type=float, default=1.0,
                        help='Time step size')
    parser.add_argument('--num-steps', type=int, default=1,
                        help='Number of simulation steps')
    parser.add_argument('--resolution', type=int, nargs=2, default=[25, 25],
                        help='Grid resolution as two integers, e.g., 20 20')
    parser.add_argument('--avg-samples-per-cell', type=int, default=50,
                        help='Average number of samples per cell')
    
    # Transformation parameters
    parser.add_argument('--logistic-k', type=float, default=0.5,
                        help='Logistic transformation k parameter')
    parser.add_argument('--logistic-x0', type=float, default=0.0,
                        help='Logistic transformation x0 parameter')
    
    # Saddle system parameters
    parser.add_argument('--saddle-points', type=str, default='[[0.0, 0.0]]',
                        help='JSON array of saddle points, e.g., [[1.0, 0.0], [2.0, 3.0]]')
    parser.add_argument('--angles', type=str, default='[90]',
                        help='JSON array of angles in degrees, e.g., [90, 180]')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='Positive Lyapunov exponent')
    parser.add_argument('--lambda2', type=float, default=-1.0,
                        help='Negative Lyapunov exponent')
    parser.add_argument('--control', type=int, default=0,
                        help='Control value for selecting saddle dynamics')
    
    # visualization
    parser.add_argument('--create-streamplot', action='store_true',
                    help='Create streamplot visualization after simulation')
    parser.add_argument('--streamplot-transformed', action='store_true', default=True,
                    help='Plot streamlines in transformed [0,1]x[0,1] space (default: True)')
    parser.add_argument('--streamplot-original', dest='streamplot_transformed', action='store_false',
                        help='Plot streamlines in original coordinate space')
    parser.add_argument('--streamplot-range', type=float, nargs=2, default=[-5.0, 5.0],
                        metavar=('MIN', 'MAX'),
                        help='Range for x and y axes when using original space (default: -5 5)')
    
    args = parser.parse_args()
    
    # Setup parameters
    bounds = [(float('-inf'), float('inf')), (float('-inf'), float('inf'))]  # Unbounded space
    transformations = [
        logistic_transformation({'k': args.logistic_k, 'x_0': args.logistic_x0}),  # For x1 dimension
        logistic_transformation({'k': args.logistic_k, 'x_0': args.logistic_x0})   # For x2 dimension
    ]
    
    # Parse saddle points and angles from JSON strings
    try:
        saddle_points = json.loads(args.saddle_points)
        saddle_points = [np.array(point) for point in saddle_points]
        angles = json.loads(args.angles)
        
        # Validate that we have angles for each saddle point
        if len(angles) != len(saddle_points):
            print(f"Warning: Number of angles ({len(angles)}) doesn't match number of saddle points ({len(saddle_points)})")
            print("Using default angle of 90° for missing entries")
            # Extend angles list if needed
            angles.extend([90] * (len(saddle_points) - len(angles)))
            
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"saddle-points format should be like '[[1.0, 0.0]]'")
        print(f"angles format should be like '[90]'")
        sys.exit(1)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running saddle system simulation with parameters:")
    print(f"  Saddle points: {saddle_points}")
    print(f"  Angles: {angles}")
    print(f"  Lyapunov exponents: λ₁={args.lambda1}, λ₂={args.lambda2}")
    print(f"  Control: {args.control}")
    print(f"  Delta t: {args.delta_t}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Average samples per cell: {args.avg_samples_per_cell}")
    print(f"  Logistic transformation: k={args.logistic_k}, x₀={args.logistic_x0}")
    print(f"  Output: {output_dir / args.db_name}")
    
    try:
        # Create model, solver, grid and simulator
        model = MultiSaddleSystem(
            k=len(saddle_points), 
            saddle_points=saddle_points, 
            angles=angles,
            lambda1=args.lambda1,
            lambda2=args.lambda2
        )

        if len(saddle_points) != len(angles):
            raise ValueError(f"Number of saddle points ({len(saddle_points)}) must match number of angles ({len(angles)})")

        print(f"Created MultiSaddleSystem with {len(saddle_points)} saddle points")
        print(f"Valid control values: 0 to {len(saddle_points)-1}")
        if args.control >= len(saddle_points):
            raise ValueError(f"Control value {args.control} is invalid. Must be 0 to {len(saddle_points)-1}")

        solver = GeneralODENumericalSolver(model)
        grid = Grid(bounds, args.resolution, transformations)
        simulator = Simulator(grid, model, solver)
        
        print("Running simulation...")
        # Run simulation
        start_time = time.time()
        
        # Run simulation with specified parameters
        results = simulator.simulate(
            control=args.control,  # Control selects the saddle
            delta_t=args.delta_t,
            avg_samples_per_cell=args.avg_samples_per_cell,
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

        if args.create_streamplot:
            print("Creating streamplot visualization...")
            
            # Use the same range for both x and y
            x_range = tuple(args.streamplot_range)
            y_range = tuple(args.streamplot_range)
            
            streamplot_path = create_saddle_streamplot_visualization(
                grid=grid,
                solver=solver, 
                model=model,
                saddle_points=saddle_points,
                angles_degrees=angles,
                output_dir="data_generation/figs",
                filename=f"{args.db_name.replace('.db', '')}_streamplot.png",
                use_transformed_space=args.streamplot_transformed,
                x_range=x_range,
                y_range=y_range
    )
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()