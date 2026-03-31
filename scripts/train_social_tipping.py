#!/usr/bin/env python3
"""
Train DRM on Social Tipping System

Simple command-line script to train the Discrete Representation Model
on the social tipping dynamics dataset.

Usage:
    python train_social_tipping.py
    python train_social_tipping.py --config custom_config.yaml
    python train_social_tipping.py --db-path path/to/database.db
    python train_social_tipping.py --epochs 50 --num-states 6
"""

import argparse
import sys
from pathlib import Path

# Add project root to path (parent of scripts/ directory)
script_dir = Path(__file__).parent
project_root = script_dir.parent if script_dir.name == 'scripts' else script_dir

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import training function
try:
    # Try importing from neural_networks package first
    from neural_networks.train_drm import train_drm_model
except ModuleNotFoundError:
    # Fallback to direct import (if train_drm.py is at root)
    print("couldn't load")

def main():
    parser = argparse.ArgumentParser(
        description='Train DRM on Social Tipping System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python train_social_tipping.py
  
  # Train with custom config
  python train_social_tipping.py --config my_config.yaml
  
  # Override specific parameters
  python train_social_tipping.py --db-path datasets/results/my_data.db --epochs 50
  
  # Quick test run
  python train_social_tipping.py --epochs 10 --num-states 3
        """
    )
    
    # Config file option
    parser.add_argument(
        '--config',
        type=str,
        default='social_tipping.yaml',
        help='Path to YAML config file (default: social_tipping.yaml)'
    )
    
    # Common overrides
    parser.add_argument(
        '--db-path',
        type=str,
        help='Override database path'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of training epochs'
    )
    
    parser.add_argument(
        '--num-states',
        type=int,
        help='Override number of discrete states'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Override random seed'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )
    
    parser.add_argument(
        '--value-method',
        type=str,
        choices=['abs_distance', 'identity'],
        help='Override value method (abs_distance or identity)'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists (try multiple locations)
    config_path = Path(args.config)
    
    # Try in order: 1) as specified, 2) in scripts/configs/, 3) in scripts/, 4) in project root
    search_paths = [
        config_path,
        script_dir / 'configs' / args.config,
        script_dir / args.config,
        project_root / 'scripts' / 'configs' / args.config,
        project_root / args.config,
    ]
    
    config_path = None
    for path in search_paths:
        if path.exists():
            config_path = path
            break
    
    if config_path is None:
        print(f"Error: Config file not found: {args.config}")
        print(f"Searched in:")
        for path in search_paths:
            print(f"  - {path}")
        print(f"\nMake sure the config file exists in one of these locations")
        sys.exit(1)
    
    print("="*70)
    print("Training DRM on Social Tipping System")
    print("="*70)
    print(f"Config file: {config_path}")
    
    # Apply overrides if provided
    if any([args.db_path, args.epochs, args.num_states, args.batch_size, 
            args.lr, args.seed, args.output_dir, args.value_method]):
        print("\nParameter overrides:")
        
        # Load config to modify
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Apply overrides
        if args.db_path:
            config['meta']['db_path'] = args.db_path
            print(f"  db_path: {args.db_path}")
        
        if args.epochs:
            config['training']['epochs'] = args.epochs
            print(f"  epochs: {args.epochs}")
        
        if args.num_states:
            config['model']['num_states'] = args.num_states
            print(f"  num_states: {args.num_states}")
        
        if args.batch_size:
            config['training']['batch_size'] = args.batch_size
            print(f"  batch_size: {args.batch_size}")
        
        if args.lr:
            config['training']['lr'] = args.lr
            print(f"  lr: {args.lr}")
        
        if args.seed:
            config['meta']['seed'] = args.seed
            print(f"  seed: {args.seed}")
        
        if args.output_dir:
            config['meta']['output_dir'] = args.output_dir
            print(f"  output_dir: {args.output_dir}")
        
        if args.value_method:
            config['model']['value_method'] = args.value_method
            print(f"  value_method: {args.value_method}")
        
        # Save modified config to temporary file
        temp_config = config_path.parent / f"temp_{config_path.name}"
        with open(temp_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        config_to_use = str(temp_config)
    else:
        config_to_use = str(config_path)
    
    print("\nStarting training...")
    print("="*70)
    
    try:
        # Run training
        model, history = train_drm_model(config_to_use, multi_run=False)
        
        print("\n" + "="*70)
        print("Training completed successfully!")
        print("="*70)
        
        # Clean up temp config if created
        if config_to_use != str(config_path):
            Path(config_to_use).unlink()
        
        return 0
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: Training failed")
        print(f"{'='*70}")
        print(f"{e}")
        
        # Clean up temp config if created
        if config_to_use != str(config_path):
            temp_path = Path(config_to_use)
            if temp_path.exists():
                temp_path.unlink()
        
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())