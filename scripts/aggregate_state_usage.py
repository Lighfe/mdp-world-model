#!/usr/bin/env python3
"""
Analyze state usage from ablation study runs and count unused states.

Extracts state_X_usage metrics from epoch 75 state_metrics,
counts how many states are below 0.01 (unused), and compares
between runs with/without Gumbel-Softmax.

Usage:
    python scripts/aggregate_state_usage.py \
        --runs-dir neural_networks/output/exp2/runs \
        --threshold 0.01 \
        --output state_usage_comparison.json
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


def extract_state_usage_from_history(history_path, target_epoch=75, num_states=4):
    """
    Extract state usage metrics from a history file.
    
    Args:
        history_path: Path to history JSON file
        target_epoch: Which epoch to extract (default: 75)
        num_states: Number of discrete states (default: 4)
        
    Returns:
        list: [state_0_usage, state_1_usage, ...] or None if not found
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
                # Extract state usage for all states
                usages = []
                for state_idx in range(num_states):
                    key = f"state_{state_idx}_usage"
                    if key in entry:
                        usages.append(entry[key])
                    else:
                        return None  # Missing state usage
                
                return usages
        
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


def count_unused_states(state_usages, threshold=0.01):
    """
    Count how many states are below the usage threshold.
    
    Args:
        state_usages: List of state usage values
        threshold: Usage threshold (default: 0.01 = 1%)
        
    Returns:
        int: Number of states below threshold
    """
    return sum(1 for usage in state_usages if usage < threshold)


def aggregate_state_usage(runs_dir, target_epoch=75, num_states=4, threshold=0.01):
    """
    Aggregate state usage from all runs, grouped by Gumbel setting.
    
    Args:
        runs_dir: Directory containing run subfolders
        target_epoch: Which epoch to extract
        num_states: Number of discrete states
        threshold: Usage threshold for counting "unused" states
        
    Returns:
        dict: {
            "gumbel_true": {
                "unused_counts": [...],  # How many states below threshold per run
                "all_usages": [[...], ...]  # All state usages per run
            },
            "gumbel_false": {...}
        }
    """
    runs_dir = Path(runs_dir)
    
    if not runs_dir.exists():
        raise ValueError(f"Runs directory not found: {runs_dir}")
    
    # Group data by Gumbel setting
    data = {
        "gumbel_true": {"unused_counts": [], "all_usages": []},
        "gumbel_false": {"unused_counts": [], "all_usages": []}
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
        
        # Extract state usages
        state_usages = extract_state_usage_from_history(history_path, target_epoch, num_states)
        
        if state_usages is None:
            print(f"  Warning: Could not extract state usage from {run_name}")
            skipped.append(run_name)
            continue
        
        # Count unused states
        unused_count = count_unused_states(state_usages, threshold)
        
        # Add to appropriate group
        group_key = "gumbel_true" if has_gumbel else "gumbel_false"
        data[group_key]["unused_counts"].append(unused_count)
        data[group_key]["all_usages"].append(state_usages)
    
    # Print summary
    print("=" * 70)
    print("DATA COLLECTION SUMMARY")
    print("=" * 70)
    print(f"Gumbel=True:  {len(data['gumbel_true']['unused_counts'])} runs")
    print(f"Gumbel=False: {len(data['gumbel_false']['unused_counts'])} runs")
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


def compute_statistics(data, threshold, num_states=4):
    """
    Compute summary statistics for each group.
    
    Returns:
        dict: Statistics for each group
    """
    results = {}
    
    for group_name, group_data in data.items():
        unused_counts = np.array(group_data["unused_counts"])
        all_usages = np.array(group_data["all_usages"])  # Shape: (n_runs, num_states)
        
        if len(unused_counts) == 0:
            results[group_name] = {
                "n": 0,
                "unused_states": {"mean": None, "std": None, "min": None, "max": None},
                "distribution": {},
                "per_state_usage": {}
            }
            continue
        
        # Count distribution (how many runs have 0, 1, 2, 3, 4 unused states)
        distribution = {}
        for i in range(num_states + 1):
            count = np.sum(unused_counts == i)
            percentage = (count / len(unused_counts)) * 100
            distribution[f"{i}_unused"] = {
                "count": int(count),
                "percentage": float(percentage)
            }
        
        # Per-state usage statistics
        per_state_usage = {}
        for state_idx in range(num_states):
            state_usages = all_usages[:, state_idx]
            per_state_usage[f"state_{state_idx}"] = {
                "mean": float(np.mean(state_usages)),
                "std": float(np.std(state_usages, ddof=1)) if len(state_usages) > 1 else 0.0,
                "min": float(np.min(state_usages)),
                "max": float(np.max(state_usages)),
                "below_threshold": int(np.sum(state_usages < threshold))
            }
        
        results[group_name] = {
            "n": len(unused_counts),
            "unused_states": {
                "mean": float(np.mean(unused_counts)),
                "std": float(np.std(unused_counts, ddof=1)) if len(unused_counts) > 1 else 0.0,
                "min": int(np.min(unused_counts)),
                "max": int(np.max(unused_counts))
            },
            "distribution": distribution,
            "per_state_usage": per_state_usage
        }
    
    return results


def print_results(results, threshold):
    """Print results in readable format."""
    print("=" * 70)
    print("STATE USAGE COMPARISON RESULTS")
    print("=" * 70)
    print(f"Threshold for 'unused': {threshold} ({threshold*100:.1f}%)")
    print()
    
    for group_name in ["gumbel_false", "gumbel_true"]:
        stats = results[group_name]
        
        print(f"{group_name.replace('_', ' ').title()}: (n={stats['n']})")
        print("-" * 70)
        
        if stats["n"] == 0:
            print("  No data available")
            print()
            continue
        
        # Unused states statistics
        unused = stats["unused_states"]
        print(f"  Unused States per Run:")
        print(f"    Mean:  {unused['mean']:.2f} ± {unused['std']:.2f}")
        print(f"    Range: [{unused['min']}, {unused['max']}]")
        print()
        
        # Distribution
        print(f"  Distribution of Unused States:")
        dist = stats["distribution"]
        for key in sorted(dist.keys()):
            info = dist[key]
            print(f"    {key}: {info['count']} runs ({info['percentage']:.1f}%)")
        print()
        
        # Per-state usage
        print(f"  Per-State Usage Statistics:")
        for state_idx in range(len(stats["per_state_usage"])):
            state_key = f"state_{state_idx}"
            state_info = stats["per_state_usage"][state_key]
            print(f"    State {state_idx}:")
            print(f"      Mean usage: {state_info['mean']:.4f} ± {state_info['std']:.4f}")
            print(f"      Below threshold: {state_info['below_threshold']} / {stats['n']} runs")
        print()
    
    # Compare the two groups
    if results["gumbel_true"]["n"] > 0 and results["gumbel_false"]["n"] > 0:
        print("=" * 70)
        print("COMPARISON")
        print("=" * 70)
        
        diff_mean = results["gumbel_true"]["unused_states"]["mean"] - results["gumbel_false"]["unused_states"]["mean"]
        
        print(f"Average unused states difference (True - False): {diff_mean:+.2f}")
        print(f"  → {'More' if diff_mean > 0 else 'Fewer'} unused states with Gumbel")
        print()
        
        # Compare collapse rates
        gum_t_no_collapse = results["gumbel_true"]["distribution"]["0_unused"]["percentage"]
        gum_f_no_collapse = results["gumbel_false"]["distribution"]["0_unused"]["percentage"]
        
        print(f"No collapse (0 unused states):")
        print(f"  Gumbel=False: {gum_f_no_collapse:.1f}%")
        print(f"  Gumbel=True:  {gum_t_no_collapse:.1f}%")
        print(f"  Difference: {gum_t_no_collapse - gum_f_no_collapse:+.1f} percentage points")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate state usage metrics by Gumbel setting"
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
        "--num-states",
        type=int,
        default=4,
        help="Number of discrete states (default: 4)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Usage threshold for 'unused' states (default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (optional)"
    )
    
    args = parser.parse_args()
    
    # Aggregate data
    print("Collecting state usage metrics from all runs...")
    print()
    data = aggregate_state_usage(
        args.runs_dir, 
        args.epoch, 
        args.num_states,
        args.threshold
    )
    
    # Compute statistics
    results = compute_statistics(data, args.threshold, args.num_states)
    
    # Print results
    print_results(results, args.threshold)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Include raw data and statistics
        output_data = {
            "epoch": args.epoch,
            "num_states": args.num_states,
            "threshold": args.threshold,
            "raw_data": data,
            "statistics": results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Saved results to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())