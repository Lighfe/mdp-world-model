#!/usr/bin/env python3
"""
Aggregate ablation study results across datasets.

For each unique run_id (e.g., baseline, vlw0.0), aggregates results
across the 4 datasets (ds3, ds5, ds8, ds9).

Input:
    neural_networks/output/ablation_<RUN_ID>_ds<N>/aggregated_results.json
    
Output:
    neural_networks/output/ablation/summary/<RUN_ID>/cross_dataset_summary.json
    neural_networks/output/ablation/summary/<RUN_ID>/per_dataset_results.json

Usage:
    python scripts/aggregate_ablation_datasets.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_folder_name(folder_name):
    """
    Parse folder name to extract run_id and dataset number.
    
    Examples:
        ablation_baseline_ds3 -> (baseline, 3)
        ablation_vlw0.0_ds5 -> (vlw0.0, 5)
        ablation_entropy0.3_ds8 -> (entropy0.3, 8)
    
    Returns:
        tuple: (run_id, dataset_num) or None if parsing fails
    """
    if not folder_name.startswith("ablation_"):
        return None
    
    # Remove "ablation_" prefix
    remainder = folder_name[9:]
    
    # Find last occurrence of "_ds"
    ds_idx = remainder.rfind("_ds")
    if ds_idx == -1:
        return None
    
    run_id = remainder[:ds_idx]
    dataset_str = remainder[ds_idx + 3:]
    
    try:
        dataset_num = int(dataset_str)
        return (run_id, dataset_num)
    except ValueError:
        return None


def extract_probing_accuracy(aggregated_results):
    """
    Extract discrete_accuracy from probing_metrics.
    
    Returns:
        dict: {
            'epochs': list of epoch numbers,
            'mean': list of mean accuracies per epoch,
            'std': list of std per epoch,
            'final_accuracy': final epoch mean accuracy
        }
    """
    probing = aggregated_results.get("probing_metrics", {})
    discrete_acc = probing.get("discrete_accuracy", {})
    epochs = probing.get("epochs", [])
    
    if not discrete_acc or not epochs:
        raise ValueError("Missing probing_metrics.discrete_accuracy or epochs")
    
    mean_curve = discrete_acc.get("mean", [])
    std_curve = discrete_acc.get("std", [])
    
    if not mean_curve:
        raise ValueError("Missing discrete_accuracy.mean")
    
    return {
        "epochs": epochs,
        "mean": mean_curve,
        "std": std_curve,
        "median": discrete_acc.get("median", []),
        "min": discrete_acc.get("min", []),
        "max": discrete_acc.get("max", []),
        "final_accuracy": mean_curve[-1] if mean_curve else None
    }


def aggregate_across_datasets(dataset_results):
    """
    Aggregate results across multiple datasets.
    
    Args:
        dataset_results: dict mapping dataset_num to extracted metrics
        
    Returns:
        dict with cross-dataset statistics
    """
    datasets = sorted(dataset_results.keys())
    
    # Extract final accuracies
    final_accuracies = [dataset_results[ds]["final_accuracy"] for ds in datasets]
    
    # Extract per-epoch accuracies (mean curves)
    epoch_curves = [dataset_results[ds]["mean"] for ds in datasets]
    
    # Get epochs (should be same for all datasets)
    epochs = dataset_results[datasets[0]]["epochs"]
    
    # Per-dataset final accuracies
    per_dataset_accuracy = {
        f"ds{ds}": dataset_results[ds]["final_accuracy"]
        for ds in datasets
    }
    
    # Cross-dataset statistics on final accuracy
    final_acc_array = np.array(final_accuracies)
    cross_dataset_stats = {
        "mean": float(np.mean(final_acc_array)),
        "std": float(np.std(final_acc_array, ddof=1)),  # sample std
        "median": float(np.median(final_acc_array)),
        "min": float(np.min(final_acc_array)),
        "max": float(np.max(final_acc_array)),
        "cv": float(np.std(final_acc_array, ddof=1) / np.mean(final_acc_array))
    }
    
    # Aggregate epoch-by-epoch curves
    epoch_curves_array = np.array(epoch_curves)  # shape: (n_datasets, n_epochs)
    aggregated_curve = {
        "epochs": epochs,
        "mean_across_datasets": epoch_curves_array.mean(axis=0).tolist(),
        "std_across_datasets": epoch_curves_array.std(axis=0, ddof=1).tolist(),
        "min_across_datasets": epoch_curves_array.min(axis=0).tolist(),
        "max_across_datasets": epoch_curves_array.max(axis=0).tolist()
    }
    
    return {
        "datasets": datasets,
        "per_dataset_accuracy": per_dataset_accuracy,
        "cross_dataset_stats": cross_dataset_stats,
        "aggregated_curve": aggregated_curve
    }


def main():
    """Main aggregation workflow."""
    
    output_base = PROJECT_ROOT / "neural_networks" / "output" / "ablation"
    
    # Find all ablation folders
    ablation_folders = sorted([
        f for f in output_base.iterdir()
        if f.is_dir() and f.name.startswith("ablation_")
    ])
    
    if not ablation_folders:
        print("ERROR: No ablation folders found in neural_networks/output/")
        return 1
    
    print(f"Found {len(ablation_folders)} ablation folders")
    
    # Group folders by run_id
    run_id_groups = defaultdict(dict)  # {run_id: {dataset_num: folder_path}}
    
    for folder in ablation_folders:
        parsed = parse_folder_name(folder.name)
        if parsed is None:
            print(f"WARNING: Could not parse folder name: {folder.name}")
            continue
        
        run_id, dataset_num = parsed
        run_id_groups[run_id][dataset_num] = folder
    
    print(f"Found {len(run_id_groups)} unique run_ids")
    print(f"Run IDs: {sorted(run_id_groups.keys())}")
    print()
    
    # Create summary output directory
    summary_dir = output_base / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each run_id
    successful = 0
    failed = 0
    
    for run_id in sorted(run_id_groups.keys()):
        print(f"Processing run_id: {run_id}")
        
        datasets_dict = run_id_groups[run_id]
        expected_datasets = [3, 5, 8, 9]
        
        # Check if all expected datasets are present
        missing = [ds for ds in expected_datasets if ds not in datasets_dict]
        if missing:
            print(f"  WARNING: Missing datasets {missing}, skipping")
            failed += 1
            continue
        
        print(f"  Found all 4 datasets: {expected_datasets}")
        
        # Load and extract data from each dataset
        dataset_results = {}
        load_failed = False
        
        for ds_num in expected_datasets:
            folder = datasets_dict[ds_num]
            agg_file = folder / "aggregated_results.json"
            
            if not agg_file.exists():
                print(f"  ERROR: Missing {agg_file}")
                load_failed = True
                break
            
            try:
                with open(agg_file, 'r') as f:
                    agg_data = json.load(f)
                
                metrics = extract_probing_accuracy(agg_data)
                dataset_results[ds_num] = metrics
                
                print(f"    ds{ds_num}: final_accuracy = {metrics['final_accuracy']:.6f}")
                
            except Exception as e:
                print(f"  ERROR loading {agg_file}: {e}")
                load_failed = True
                break
        
        if load_failed:
            failed += 1
            continue
        
        # Aggregate across datasets
        try:
            aggregated = aggregate_across_datasets(dataset_results)
            
            # Create output directory for this run_id
            run_output_dir = summary_dir / run_id
            run_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save cross-dataset summary
            summary_output = {
                "run_id": run_id,
                "datasets": aggregated["datasets"],
                "per_dataset_accuracy": aggregated["per_dataset_accuracy"],
                "cross_dataset_stats": aggregated["cross_dataset_stats"],
                "aggregated_curve": aggregated["aggregated_curve"]
            }
            
            summary_path = run_output_dir / "cross_dataset_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary_output, f, indent=2)
            
            print(f"  Cross-dataset mean: {aggregated['cross_dataset_stats']['mean']:.6f} "
                  f"± {aggregated['cross_dataset_stats']['std']:.6f}")
            
            # Save per-dataset raw results
            per_dataset_output = {
                "run_id": run_id,
                "datasets": {
                    f"ds{ds_num}": {
                        "epochs": dataset_results[ds_num]["epochs"],
                        "mean": dataset_results[ds_num]["mean"],
                        "std": dataset_results[ds_num]["std"],
                        "final_accuracy": dataset_results[ds_num]["final_accuracy"]
                    }
                    for ds_num in expected_datasets
                }
            }
            
            per_dataset_path = run_output_dir / "per_dataset_results.json"
            with open(per_dataset_path, 'w') as f:
                json.dump(per_dataset_output, f, indent=2)
            
            print(f"  ✓ Saved to {run_output_dir}")
            successful += 1
            
        except Exception as e:
            print(f"  ERROR during aggregation: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        
        print()
    
    # Final summary
    print("=" * 70)
    print("AGGREGATION COMPLETE")
    print("=" * 70)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {summary_dir}")
    print()
    
    if successful > 0:
        print("Next step: Run analyze_ablation_study.py for statistical comparisons")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())