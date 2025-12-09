#!/usr/bin/env python3
"""
Standalone script to visualize final state assignments from pkl files.

Location: scripts/visualize_state_assignments.py

Usage:
    python scripts/visualize_state_assignments.py \
        --pkl_path outputs/run_XXX/state_assignments_XXX.pkl \
        --output_path outputs/visualization.png \
        --system_type saddle_system \
        --db_path datasets/results/database/saddle_system.db \
        --style scatter \
        --point_size 15
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the visualization function from neural_networks module
from neural_networks.drm_viz import visualize_final_state_assignments


def main():
    parser = argparse.ArgumentParser(
        description="Visualize final state assignments from pkl file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scatter plot for saddle system
  python scripts/visualize_state_assignments.py \\
      --pkl_path outputs/state_assignments_001.pkl \\
      --output_path outputs/states_scatter.png \\
      --system_type saddle_system \\
      --db_path datasets/results/database/saddle.db \\
      --style scatter

  # Region coloring for tech substitution
  python scripts/visualize_state_assignments.py \\
      --pkl_path outputs/state_assignments_002.pkl \\
      --output_path outputs/states_regions.png \\
      --system_type tech_substitution \\
      --style regions
        """
    )
    
    parser.add_argument(
        '--pkl_path',
        type=str,
        required=True,
        help='Path to state_assignments_{run_id}.pkl file (relative or absolute)'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save the output visualization (PNG)'
    )
    
    parser.add_argument(
        '--system_type',
        type=str,
        required=True,
        choices=['saddle_system', 'tech_substitution'],
        help='Type of dynamical system'
    )
    
    parser.add_argument(
        '--db_path',
        type=str,
        default=None,
        help='Path to database (required for saddle_system to get separatrices)'
    )
    
    parser.add_argument(
        '--style',
        type=str,
        default='scatter',
        choices=['scatter', 'regions'],
        help='Visualization style: scatter (plot points) or regions (color areas)'
    )
    
    parser.add_argument(
        '--point_size',
        type=int,
        default=10,
        help='Size of scatter points (only used with --style scatter)'
    )
    
    parser.add_argument(
        '--grid_resolution',
        type=int,
        default=200,
        help='Grid resolution for region coloring (only used with --style regions)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects and resolve
    pkl_path = Path(args.pkl_path)
    if not pkl_path.is_absolute():
        pkl_path = PROJECT_ROOT / pkl_path
    
    if not pkl_path.exists():
        print(f"Error: pkl file not found: {pkl_path}")
        return 1
    
    if args.system_type == 'saddle_system' and args.db_path is None:
        print("Error: --db_path is required for saddle_system")
        return 1
    
    if args.db_path is not None:
        db_path = Path(args.db_path)
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / db_path
        if not db_path.exists():
            print(f"Error: database file not found: {db_path}")
            return 1
    else:
        db_path = None
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run visualization
    print("=" * 60)
    print("VISUALIZING STATE ASSIGNMENTS")
    print("=" * 60)
    print(f"Input pkl:       {pkl_path}")
    print(f"Output:          {output_path}")
    print(f"System type:     {args.system_type}")
    print(f"Database:        {db_path}")
    print(f"Style:           {args.style}")
    if args.style == 'scatter':
        print(f"Point size:      {args.point_size}")
    else:
        print(f"Grid resolution: {args.grid_resolution}")
    print("=" * 60)
    
    visualize_final_state_assignments(
        pkl_path=str(pkl_path),
        output_path=str(output_path),
        system_type=args.system_type,
        db_path=str(db_path) if db_path else None,
        visualization_style=args.style,
        point_size=args.point_size,
        grid_resolution=args.grid_resolution,
    )
    
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())