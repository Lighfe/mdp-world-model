#!/usr/bin/env python3
"""
Generate config files for tech substitution experiments.
Uses the existing generate_config_combinations utility.
"""

import sys
import yaml
from pathlib import Path
from itertools import product

def main():
    # Add project root to path
    PROJECT_ROOT = Path.cwd()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    print(f"Project root: {PROJECT_ROOT}")

    try:
        from neural_networks.utils import load_config
        print("✓ Successfully imported utilities")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        sys.exit(1)

    # Configuration
    BASE_CONFIG = "scripts/configs/tech_sub_base.yaml"
    OUTPUT_DIR = "neural_networks/output/tech_substitution_results/configs"
    
    print(f"\nConfiguration:")
    print(f"  Base config: {BASE_CONFIG}")
    print(f"  Output dir: {OUTPUT_DIR}")

    # Verify base config exists
    base_config_path = Path(BASE_CONFIG)
    if not base_config_path.exists():
        print(f"\n✗ ERROR: Base config not found at {base_config_path.absolute()}")
        sys.exit(1)
    else:
        print(f"  ✓ Base config found")

    # Load base config
    base_config = load_config(BASE_CONFIG)
    
    # Define parameter grid
    seeds = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    value_loss_weights = [0.5, 1.0, 2.0, 3.0, 4.0]

    print(f"\nParameter grid:")
    print(f"  Seeds: {seeds}")
    print(f"  Value loss weights: {value_loss_weights}")
    print(f"  Total combinations: {len(seeds) * len(value_loss_weights)}")

    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Created output directory: {output_dir.absolute()}")

    # Generate configs
    print(f"\nGenerating config files...")
    generated_count = 0
    
    for seed, vlw in product(seeds, value_loss_weights):
        # Create a copy of base config
        config = base_config.copy()
        
        # Deep copy nested dicts to avoid reference issues
        for key in ['meta', 'data', 'model', 'training', 'loss']:
            if key in base_config:
                config[key] = base_config[key].copy()
        
        # Update parameters
        config['meta']['seed'] = seed
        config['loss']['value_loss_weight'] = vlw
        
        # Create run name
        run_name = f"seed{seed}_vlw{vlw}"
        config['meta']['run_id'] = f"tech_sub_{run_name}"
        
        # Save config
        config_filename = f"config_{run_name}.yaml"
        config_path = output_dir / config_filename
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        generated_count += 1
        
        # Print first few
        if generated_count <= 5:
            print(f"  Created: {config_filename}")
    
    print(f"\n✓ Successfully generated {generated_count} config files")
    print(f"\nConfig directory: {output_dir.absolute()}")

    print("\n" + "="*60)
    print("Config generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()