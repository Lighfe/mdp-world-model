#!/usr/bin/env python3
"""
Generate all config files for ablation study.

Creates configs for:
- Baseline (4 datasets)
- Value loss weight ablation (5 values × 4 datasets)
- Entropy weight ablation (4 values × 4 datasets)
- Gumbel softmax ablation (1 value × 4 datasets)
- Initial temperature ablation (4 values × 4 datasets)

Total: 60 config files

Usage:
    python scripts/create_ablation_configs.py
"""

import sys
from pathlib import Path
import yaml
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Ablation study parameters
DATASETS = {
    3: "datasets/results/multi_saddle_3.db",
    5: "datasets/results/multi_saddle_5.db", 
    8: "datasets/results/multi_saddle_8.db",
    9: "datasets/results/multi_saddle_9.db",
}

ABLATIONS = {
    "value_loss_weight": [0.0, 0.1, 1.0, 3.0, 10.0],
    "entropy_weight": [0.0, 0.3, 0.9, 1.5],
    "gumbel": [False],  # Baseline has True
    "initial_temp": [1.0, 3.0, 5.0, 10.0],
}


def load_base_config(base_config_path):
    """Load the base configuration."""
    with open(base_config_path, 'r') as f:
        return yaml.safe_load(f)


def create_config_variant(base_config, modifications):
    """
    Create a config variant with specified modifications.
    
    Args:
        base_config: Base configuration dict
        modifications: Dict of modifications to apply
        
    Returns:
        Modified config dict
    """
    import copy
    config = copy.deepcopy(base_config)
    
    for key, value in modifications.items():
        # Handle nested keys (e.g., "loss.value_loss_weight")
        if '.' in key:
            parts = key.split('.')
            current = config
            for part in parts[:-1]:
                current = current[part]
            current[parts[-1]] = value
        else:
            config[key] = value
    
    return config


def save_config(config, output_path):
    """Save config to YAML file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def generate_ablation_configs(base_config_path, output_dir):
    """
    Generate all ablation study config files.
    
    Args:
        base_config_path: Path to base config file
        output_dir: Output directory for generated configs
    """
    
    base_config = load_base_config(base_config_path)
    output_dir = Path(output_dir)
    
    # Clear output directory if it exists
    if output_dir.exists():
        print(f"Clearing existing configs in {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated_configs = []
    
    # ========================================================================
    # 1. BASELINE CONFIGS (1 per dataset)
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING BASELINE CONFIGS")
    print("="*70)
    
    for ds_id, db_path in DATASETS.items():
        config_name = f"baseline_ds{ds_id}.yaml"
        
        modifications = {
            "meta.db_path": db_path,
            "meta.output_dir": f"neural_networks/output/ablation_baseline_ds{ds_id}",
        }
        
        config = create_config_variant(base_config, modifications)
        output_path = output_dir / config_name
        save_config(config, output_path)
        
        generated_configs.append({
            "name": config_name,
            "type": "baseline",
            "dataset": ds_id,
            "path": str(output_path),
        })
        
        print(f"  ✓ {config_name}")
    
    # ========================================================================
    # 2. VALUE LOSS WEIGHT ABLATION
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING VALUE LOSS WEIGHT CONFIGS")
    print("="*70)
    
    for vlw in ABLATIONS["value_loss_weight"]:
        for ds_id, db_path in DATASETS.items():
            config_name = f"vlw{vlw}_ds{ds_id}.yaml"
            
            modifications = {
                "meta.db_path": db_path,
                "meta.output_dir": f"neural_networks/output/ablation_vlw{vlw}_ds{ds_id}",
                "loss.value_loss_weight": vlw,
            }
            
            config = create_config_variant(base_config, modifications)
            output_path = output_dir / config_name
            save_config(config, output_path)
            
            generated_configs.append({
                "name": config_name,
                "type": "value_loss_weight",
                "dataset": ds_id,
                "value": vlw,
                "path": str(output_path),
            })
            
            print(f"  ✓ {config_name}")
    
    # ========================================================================
    # 3. ENTROPY WEIGHT ABLATION
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING ENTROPY WEIGHT CONFIGS")
    print("="*70)
    
    for ew in ABLATIONS["entropy_weight"]:
        for ds_id, db_path in DATASETS.items():
            config_name = f"entropy{ew}_ds{ds_id}.yaml"
            
            modifications = {
                "meta.db_path": db_path,
                "meta.output_dir": f"neural_networks/output/ablation_entropy{ew}_ds{ds_id}",
                "loss.entropy_weight": ew,
            }
            
            config = create_config_variant(base_config, modifications)
            output_path = output_dir / config_name
            save_config(config, output_path)
            
            generated_configs.append({
                "name": config_name,
                "type": "entropy_weight",
                "dataset": ds_id,
                "value": ew,
                "path": str(output_path),
            })
            
            print(f"  ✓ {config_name}")
    
    # ========================================================================
    # 4. GUMBEL SOFTMAX ABLATION
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING GUMBEL SOFTMAX CONFIGS")
    print("="*70)
    
    for gumbel in ABLATIONS["gumbel"]:
        for ds_id, db_path in DATASETS.items():
            gumbel_str = "false" if not gumbel else "true"
            config_name = f"gumbel_{gumbel_str}_ds{ds_id}.yaml"
            
            modifications = {
                "meta.db_path": db_path,
                "meta.output_dir": f"neural_networks/output/ablation_gumbel_{gumbel_str}_ds{ds_id}",
                "model.use_gumbel": gumbel,
            }
            
            config = create_config_variant(base_config, modifications)
            output_path = output_dir / config_name
            save_config(config, output_path)
            
            generated_configs.append({
                "name": config_name,
                "type": "gumbel",
                "dataset": ds_id,
                "value": gumbel,
                "path": str(output_path),
            })
            
            print(f"  ✓ {config_name}")
    
    # ========================================================================
    # 5. INITIAL TEMPERATURE ABLATION
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING INITIAL TEMPERATURE CONFIGS")
    print("="*70)
    
    for temp in ABLATIONS["initial_temp"]:
        for ds_id, db_path in DATASETS.items():
            config_name = f"temp{temp}_ds{ds_id}.yaml"
            
            modifications = {
                "meta.db_path": db_path,
                "meta.output_dir": f"neural_networks/output/ablation_temp{temp}_ds{ds_id}",
                "model.initial_temp": temp,
            }
            
            config = create_config_variant(base_config, modifications)
            output_path = output_dir / config_name
            save_config(config, output_path)
            
            generated_configs.append({
                "name": config_name,
                "type": "initial_temp",
                "dataset": ds_id,
                "value": temp,
                "path": str(output_path),
            })
            
            print(f"  ✓ {config_name}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    
    # Count by type
    type_counts = {}
    for cfg in generated_configs:
        cfg_type = cfg["type"]
        type_counts[cfg_type] = type_counts.get(cfg_type, 0) + 1
    
    print(f"\nTotal configs generated: {len(generated_configs)}")
    print("\nBreakdown by ablation type:")
    for cfg_type, count in sorted(type_counts.items()):
        print(f"  {cfg_type}: {count} configs")
    
    print(f"\nConfigs saved to: {output_dir}")
    
    # Save manifest file
    manifest_path = output_dir / "manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write("# Ablation Study Config Manifest\n")
        f.write(f"# Total configs: {len(generated_configs)}\n\n")
        
        for cfg_type in sorted(type_counts.keys()):
            f.write(f"\n## {cfg_type.upper()}\n")
            for cfg in generated_configs:
                if cfg["type"] == cfg_type:
                    f.write(f"{cfg['name']}\n")
    
    print(f"Manifest saved to: {manifest_path}")
    
    return generated_configs


def main():
    """Main entry point."""
    
    # Paths
    base_config_path = project_root / "scripts" / "configs" / "base.yaml"
    output_dir = project_root / "scripts" / "configs" / "ablation_study"
    
    # Verify base config exists
    if not base_config_path.exists():
        print(f"ERROR: Base config not found at {base_config_path}")
        return 1
    
    print("="*70)
    print("ABLATION STUDY CONFIG GENERATOR")
    print("="*70)
    print(f"Base config: {base_config_path}")
    print(f"Output dir: {output_dir}")
    print(f"\nDatasets: {list(DATASETS.keys())}")
    print(f"Ablations:")
    for name, values in ABLATIONS.items():
        print(f"  {name}: {values}")
    
    # Generate configs
    configs = generate_ablation_configs(base_config_path, output_dir)
    
    print(f"\n✓ Successfully generated {len(configs)} config files!")
    print(f"\nNext steps:")
    print(f"1. Review configs in: {output_dir}")
    print(f"2. Run ablation study with SLURM script")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())