#!/usr/bin/env python3
"""
Generate config files for tech substitution experiments.
Uses the existing generate_config_combinations utility.
"""

import sys
from pathlib import Path

def main():
    # Add project root to path
    PROJECT_ROOT = Path.cwd()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python path: {sys.path[:3]}")

    try:
        from neural_networks.utils import generate_config_combinations
        print("✓ Successfully imported generate_config_combinations")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        sys.exit(1)

    # Configuration
    BASE_CONFIG = "scripts/configs/tech_sub_base.yaml"
    OUTPUT_DIR = "tech_substitution_results"
    CONFIG_ID = "tech_sub_configs"

    print(f"\nConfiguration:")
    print(f"  Base config: {BASE_CONFIG}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Config ID: {CONFIG_ID}")

    # Verify base config exists
    base_config_path = Path(BASE_CONFIG)
    if not base_config_path.exists():
        print(f"\n✗ ERROR: Base config not found at {base_config_path.absolute()}")
        sys.exit(1)
    else:
        print(f"  ✓ Base config found")

    # Define parameter grid
    override_params = {
        "meta.seed": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "loss.value_loss_weight": [0.5, 1.0, 2.0, 3.0, 4.0]
    }

    print(f"\nParameter grid:")
    print(f"  Seeds: {override_params['meta.seed']}")
    print(f"  Value loss weights: {override_params['loss.value_loss_weight']}")
    print(f"  Total combinations: {len(override_params['meta.seed']) * len(override_params['loss.value_loss_weight'])}")

    # Generate configs
    print(f"\nGenerating config files...")
    try:
        configs = generate_config_combinations(
            base_config_path=BASE_CONFIG,
            config_id=CONFIG_ID,
            override_params=override_params,
            output_configs_dir=OUTPUT_DIR
        )
        
        print(f"\n✓ Successfully generated {len(configs)} config files")
        
        # Show where configs were created
        if configs:
            first_config_path = Path(configs[0][0])
            config_dir = first_config_path.parent
            print(f"\nConfig directory: {config_dir.absolute()}")
            print(f"\nFirst few configs:")
            for i, (path, name, overrides) in enumerate(configs[:5]):
                print(f"  {i+1}. {Path(path).name}")
        
    except Exception as e:
        print(f"\n✗ ERROR: Failed to generate configs: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*60)
    print("Config generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()