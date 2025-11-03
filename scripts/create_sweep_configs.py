#!/usr/bin/env python3
"""
Create sweep configuration files for parameter grid search.

This script generates individual config files for each combination of parameters
and creates metadata for SLURM job array submission.
"""

import argparse
import json
import sys
from pathlib import Path
from itertools import product
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neural_networks.utils import load_config, set_nested_dict_value

def parse_parameter_spec(param_spec):
    """
    Parse parameter specification from command line.
    
    Format: "param.path=[val1,val2,val3]"
    
    Args:
        param_spec: String specification
        
    Returns:
        tuple: (param_path, list of values)
    """
    if '=' not in param_spec:
        raise ValueError(f"Invalid parameter spec: {param_spec}. Expected format: 'param.path=[val1,val2,...]'")
    
    param_path, values_str = param_spec.split('=', 1)
    param_path = param_path.strip()
    
    # Parse list of values
    if not (values_str.startswith('[') and values_str.endswith(']')):
        raise ValueError(f"Values must be in brackets: {values_str}")
    
    values_str = values_str[1:-1]  # Remove brackets
    values = [v.strip() for v in values_str.split(',')]
    
    # Try to convert to appropriate types
    converted_values = []
    for v in values:
        # Try int
        try:
            converted_values.append(int(v))
            continue
        except ValueError:
            pass
        
        # Try float
        try:
            converted_values.append(float(v))
            continue
        except ValueError:
            pass
        
        # Keep as string
        converted_values.append(v)
    
    return param_path, converted_values

def create_smart_run_name(override_values, experiment_name):
    """
    Create intelligent run name based on swept parameters.
    
    Args:
        override_values: Dictionary of parameter overrides
        experiment_name: Base experiment name
        
    Returns:
        String run name with abbreviated parameters
    """
    # Common abbreviations for parameters
    abbrev_map = {
        'meta.seed': 's',
        'meta.db_path': 'db',
        'loss.value_loss_weight': 'vlw',
        'loss.state_loss_weight': 'slw',
        'training.lr': 'lr',
        'training.epochs': 'ep',
        'model.hidden_dim': 'hd',
        'model.num_states': 'ns',
        'training.batch_size': 'bs',
        'training.weight_decay': 'wd',
    }
    
    parts = [experiment_name]
    
    # Sort for consistent ordering
    for param_path in sorted(override_values.keys()):
        value = override_values[param_path]
        
        # Get abbreviation or use last part of path
        if param_path in abbrev_map:
            abbrev = abbrev_map[param_path]
        else:
            abbrev = param_path.split('.')[-1][:3]  # First 3 chars of last part
        
        # Format value
        if param_path == 'meta.db_path':
            # Extract dataset number from path like "datasets/tech_sub_1.db"
            value_str = Path(str(value)).stem.split('_')[-1]
        elif isinstance(value, float):
            # Format floats nicely
            if value < 0.01:
                value_str = f"{value:.0e}".replace('e-0', 'e-')
            else:
                value_str = f"{value:.3f}".rstrip('0').rstrip('.')
        else:
            value_str = str(value)
        
        parts.append(f"{abbrev}{value_str}")
    
    return "_".join(parts)

def generate_sweep_configs(base_config_path, experiment_name, parameter_grid, 
                          output_base_dir="neural_networks/output"):
    """
    Generate all configuration files for parameter sweep.
    
    Args:
        base_config_path: Path to base YAML config
        experiment_name: Name of experiment
        parameter_grid: Dict mapping parameter paths to lists of values
        output_base_dir: Base output directory
        
    Returns:
        tuple: (configs_dir, metadata_dict)
    """
    # Load base config
    base_config = load_config(base_config_path)
    
    # Create output directory structure
    experiment_dir = Path(output_base_dir) / experiment_name
    configs_dir = experiment_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations
    param_names = list(parameter_grid.keys())
    param_value_lists = [parameter_grid[name] for name in param_names]
    
    config_files = []
    run_names = []
    
    config_num = 1
    for value_combination in product(*param_value_lists):
        # Create override dictionary
        override_values = dict(zip(param_names, value_combination))
        
        # Create intelligent run name
        run_name = create_smart_run_name(override_values, experiment_name)
        
        # Create modified config
        config_copy = base_config.copy()
        
        # Apply all overrides
        for param_path, value in override_values.items():
            set_nested_dict_value(config_copy, param_path, value)
        
        # Set run_id and output_dir in meta section
        if 'meta' in config_copy:
            config_copy['meta']['run_id'] = run_name
            config_copy['meta']['output_dir'] = str(experiment_dir / "runs" / run_name)
        
        # Save config with numbered filename for easy indexing
        config_filename = f"config_{config_num:03d}.yaml"
        config_path = configs_dir / config_filename
        
        with open(config_path, 'w') as f:
            yaml.dump(config_copy, f, default_flow_style=False, indent=2)
        
        # Store path as-is (already relative or make it relative if absolute)
        if config_path.is_absolute():
            try:
                config_files.append(str(config_path.relative_to(Path.cwd())))
            except ValueError:
                config_files.append(str(config_path))
        else:
            config_files.append(str(config_path))
        
        run_names.append(run_name)
        config_num += 1
    
    # Create metadata
    metadata = {
        "experiment_name": experiment_name,
        "num_configs": len(config_files),
        "config_files": config_files,
        "run_names": run_names,
        "parameter_grid": {k: [str(v) for v in vals] for k, vals in parameter_grid.items()},
        "base_config": str(base_config_path)
    }
    
    # Save metadata
    metadata_path = experiment_dir / "sweep_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return configs_dir, metadata

def main():
    parser = argparse.ArgumentParser(
        description="Generate configuration files for parameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python create_sweep_configs.py \\
      --base-config configs/base.yaml \\
      --experiment tech_sub_vlw_sweep \\
      --params "loss.value_loss_weight=[0.1,0.2,0.3,0.4,0.5]" \\
               "meta.seed=[11,12,13,14,15,16,17,18,19,20]" \\
               "meta.db_path=[datasets/tech_sub_1.db,datasets/tech_sub_2.db]"
        """
    )
    
    parser.add_argument(
        '--base-config',
        required=True,
        help='Path to base YAML configuration file'
    )
    
    parser.add_argument(
        '--experiment',
        required=True,
        help='Experiment name (used for output directory)'
    )
    
    parser.add_argument(
        '--params',
        nargs='+',
        required=True,
        help='Parameter specifications in format: param.path=[val1,val2,...]'
    )
    
    parser.add_argument(
        '--output-dir',
        default='neural_networks/output',
        help='Base output directory (default: neural_networks/output)'
    )
    
    args = parser.parse_args()
    
    # Verify base config exists
    if not Path(args.base_config).exists():
        print(f"Error: Base config not found: {args.base_config}")
        sys.exit(1)
    
    # Parse parameter specifications
    parameter_grid = {}
    for param_spec in args.params:
        param_path, values = parse_parameter_spec(param_spec)
        parameter_grid[param_path] = values
    
    print("="*60)
    print("SWEEP CONFIGURATION GENERATOR")
    print("="*60)
    print(f"Experiment: {args.experiment}")
    print(f"Base config: {args.base_config}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nParameter grid:")
    for param_path, values in parameter_grid.items():
        print(f"  {param_path}: {len(values)} values")
        print(f"    {values}")
    
    # Calculate total combinations
    total_configs = 1
    for values in parameter_grid.values():
        total_configs *= len(values)
    
    print(f"\nTotal configurations: {total_configs}")
    
    # Confirm if large number
    if total_configs > 100:
        response = input(f"\nThis will create {total_configs} config files. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Generate configs
    print("\nGenerating configuration files...")
    configs_dir, metadata = generate_sweep_configs(
        args.base_config,
        args.experiment,
        parameter_grid,
        args.output_dir
    )
    
    print(f"\n✓ Generated {metadata['num_configs']} configuration files")
    print(f"✓ Configs directory: {configs_dir}")
    print(f"✓ Metadata: {Path(args.output_dir) / args.experiment / 'sweep_metadata.json'}")
    print("\nNext steps:")
    print("  1. Review configs in:", configs_dir)
    print("  2. Submit SLURM job with the generated configs")
    print("="*60)

if __name__ == "__main__":
    main()