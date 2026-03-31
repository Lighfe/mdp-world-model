#!/usr/bin/env python3
"""
Aggregate sharpness metrics from ablation study runs.

Extracts sharpness_mean and sharpness_std from epoch 75 state_metrics,
aggregates separately for runs with/without Gumbel-Softmax.

Usage:
    python scripts/aggregate_sharpness.py \
        --runs-dir neural_networks/output/exp2/runs \
        --output sharpness_comparison.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_sharpness_from_history(history_path, target_epoch=75):
    """
    Extract sharpness_mean and sharpness_std from a history file.
    
    Args:
        history_path: Path to history JSON file
        target_epoch: Which epoch to extract (default: 75)
        
    Returns:
        dict: {"sharpness_mean": float, "sharpness_std": float} or None if not found
    """
    try:
        with open(history_path, 'r') as f:
            data = json.load(f)
        
        if "state_metrics" not in data:
            return None
        
        state_metrics = data["state_metrics"]
        
        # Find the target epoch
        for entry in state_metrics:
            if entry.get("epoch") == target_epoch:
                if "sharpness_mean" in entry and "sharpness_std" in entry:
                    return {
                        "sharpness_mean": entry["sharpness_mean"],
                        "sharpness_std": entry["sharpness_std"]
                    }
        
        return None
        
    except Exception as e:
        print(f"  Warning: Failed to load {history_path.name}: {e}")
        return None


def parse_run_name(run_name):
    """
    Parse run folder name to determine if Gumbel is enabled.
    
    Example names:
    - exp2_gumbel_ent0.6_vlw0_gumF_s700  -> gumbel=False
    - exp2_gumbel_ent0.6_vlw0_gumT_s700  -> gumbel=True
    
    Returns:
        bool: True if Gumbel enabled, False otherwise
    """
    # Look for "gumT" (True) or "gumF" (False) in the name
    if "_gumT_" in run_name:
        return True
    elif "_gumF_" in run_name:
        return False
    else:
        # Try to infer from other patterns
        print(f"  Warning: Could not determine Gumbel setting from name: {run_name}")
        return None


def aggregate_sharpness(runs_dir, target_epoch=75):
    """
    Aggregate sharpness metrics from all runs, grouped by Gumbel setting.
    
    Args:
        runs_dir: Directory containing run subfolders
        target_epoch: Which epoch to extract
        
    Returns:
        dict: {
            "gumbel_true": {"sharpness_means": [...], "sharpness_stds": [...]},
            "gumbel_false": {"sharpness_means": [...], "sharpness_stds": [...]}
        }
    """
    runs_dir = Path(runs_dir)
    
    if not runs_dir.exists():
        raise ValueError(f"Runs directory not found: {runs_dir}")
    
    # Group data by Gumbel setting
    data = {
        "gumbel_true": {"sharpness_means": [], "sharpness_stds": []},
        "gumbel_false": {"sharpness_means": [], "sharpness_stds": []}
    }
    
    skipped = []
    
    # Iterate through all run folders
    run_folders = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(run_folders)} run folders in {runs_dir}")
    print()
    
    for run_folder in run_folders:
        run_name = run_folder.name
        
        # Determine Gumbel setting
        has_gumbel = parse_run_name(run_name)
        
        if has_gumbel is None:
            skipped.append(run_name)
            continue
        
        # Look for history file
        # Try different possible names
        possible_names = [
            f"history_{run_name}.json",
            "history.json"
        ]
        
        history_path = None
        for name in possible_names:
            candidate = run_folder / name
            if candidate.exists():
                history_path = candidate
                break
        
        if history_path is None:
            print(f"  Warning: No history file found in {run_name}")
            skipped.append(run_name)
            continue
        
        # Extract sharpness metrics
        metrics = extract_sharpness_from_history(history_path, target_epoch)
        
        if metrics is None:
            print(f"  Warning: Could not extract sharpness from {run_name}")
            skipped.append(run_name)
            continue
        
        # Add to appropriate group
        group_key = "gumbel_true" if has_gumbel else "gumbel_false"
        data[group_key]["sharpness_means"].append(metrics["sharpness_mean"])
        data[group_key]["sharpness_stds"].append(metrics["sharpness_std"])
    
    # Print summary
    print("=" * 70)
    print("DATA COLLECTION SUMMARY")
    print("=" * 70)
    print(f"Gumbel=True:  {len(data['gumbel_true']['sharpness_means'])} runs")
    print(f"Gumbel=False: {len(data['gumbel_false']['sharpness_means'])} runs")
    print(f"Skipped:      {len(skipped)} runs")
    print()
    
    if skipped:
        print("Skipped runs:")
        for name in skipped[:10]:  # Show first 10
            print(f"  - {name}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
        print()
    
    return data


def compute_statistics(data):
    """
    Compute summary statistics for each group.
    
    Returns:
        dict: Statistics for each group
    """
    results = {}
    
    for group_name, group_data in data.items():
        means = np.array(group_data["sharpness_means"])
        stds = np.array(group_data["sharpness_stds"])
        
        if len(means) == 0:
            results[group_name] = {
                "n": 0,
                "sharpness_mean": {"mean": None, "std": None, "min": None, "max": None},
                "sharpness_std": {"mean": None, "std": None, "min": None, "max": None}
            }
            continue
        
        results[group_name] = {
            "n": len(means),
            "sharpness_mean": {
                "mean": float(np.mean(means)),
                "std": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
                "min": float(np.min(means)),
                "max": float(np.max(means))
            },
            "sharpness_std": {
                "mean": float(np.mean(stds)),
                "std": float(np.std(stds, ddof=1)) if len(stds) > 1 else 0.0,
                "min": float(np.min(stds)),
                "max": float(np.max(stds))
            }
        }
    
    return results


def print_results(results):
    """Print results in readable format."""
    print("=" * 70)
    print("SHARPNESS COMPARISON RESULTS")
    print("=" * 70)
    print()
    
    for group_name in ["gumbel_false", "gumbel_true"]:
        stats = results[group_name]
        
        print(f"{group_name.replace('_', ' ').title()}: (n={stats['n']})")
        print("-" * 70)
        
        if stats["n"] == 0:
            print("  No data available")
            print()
            continue
        
        print(f"  Sharpness Mean (entropy):")
        print(f"    Mean:  {stats['sharpness_mean']['mean']:.4f} ± {stats['sharpness_mean']['std']:.4f}")
        print(f"    Range: [{stats['sharpness_mean']['min']:.4f}, {stats['sharpness_mean']['max']:.4f}]")
        print()
        print(f"  Sharpness Std (entropy variability):")
        print(f"    Mean:  {stats['sharpness_std']['mean']:.4f} ± {stats['sharpness_std']['std']:.4f}")
        print(f"    Range: [{stats['sharpness_std']['min']:.4f}, {stats['sharpness_std']['max']:.4f}]")
        print()
    
    # Compare the two groups
    if results["gumbel_true"]["n"] > 0 and results["gumbel_false"]["n"] > 0:
        print("=" * 70)
        print("COMPARISON")
        print("=" * 70)
        
        diff_mean = results["gumbel_true"]["sharpness_mean"]["mean"] - results["gumbel_false"]["sharpness_mean"]["mean"]
        diff_std = results["gumbel_true"]["sharpness_std"]["mean"] - results["gumbel_false"]["sharpness_std"]["mean"]
        
        print(f"Sharpness Mean difference (True - False): {diff_mean:+.4f}")
        print(f"  → {'Lower' if diff_mean < 0 else 'Higher'} entropy with Gumbel")
        print(f"  → {'Sharper' if diff_mean < 0 else 'Less sharp'} assignments with Gumbel")
        print()
        print(f"Sharpness Std difference (True - False): {diff_std:+.4f}")
        print(f"  → {'More' if diff_std > 0 else 'Less'} variable sharpness with Gumbel")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate sharpness metrics by Gumbel setting"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        required=True,
        help="Directory containing run subfolders"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=75,
        help="Which epoch to extract (default: 75)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (optional)"
    )
    
    args = parser.parse_args()
    
    # Aggregate data
    print("Collecting sharpness metrics from all runs...")
    print()
    data = aggregate_sharpness(args.runs_dir, args.epoch)
    
    # Compute statistics
    results = compute_statistics(data)
    
    # Print results
    print_results(results)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Include raw data and statistics
        output_data = {
            "epoch": args.epoch,
            "raw_data": data,
            "statistics": results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved results to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())