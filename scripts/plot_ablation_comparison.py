#!/usr/bin/env python3
"""
Plot accuracy over epochs for ablation study, comparing baseline vs ablations.

Aggregates across all runs (256 = 64 seeds × 4 datasets) to get proper statistics.

Usage:
    python scripts/plot_ablation_comparison.py \
        --base-dir neural_networks/output/ablation \
        --baseline baseline \
        --ablations gumbel_false \
        --datasets 3 5 8 9 \
        --output neural_networks/figs/ablation_comparison.png \
        --labels "Baseline" "No Gumbel"
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_history_file(history_path):
    """
    Load a history file and extract epoch-wise accuracy from intermediate_probing.
    
    Expected format:
    {
        "intermediate_probing": [
            {"epoch": 1, "discrete_accuracy": 0.678, ...},
            {"epoch": 5, "discrete_accuracy": 0.739, ...},
            ...
        ],
        "test_metrics": {
            "prob_discrete_accuracy": 0.737  # final test accuracy
        }
    }
    
    Returns:
        dict: {"epochs": [...], "accuracies": [...]}
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Primary format: intermediate_probing list
    if "intermediate_probing" in history:
        probing_data = history["intermediate_probing"]
        
        if isinstance(probing_data, list) and len(probing_data) > 0:
            epochs = []
            accuracies = []
            
            for entry in probing_data:
                if "epoch" in entry and "discrete_accuracy" in entry:
                    epochs.append(entry["epoch"])
                    accuracies.append(entry["discrete_accuracy"])
            
            if epochs:
                return {"epochs": epochs, "accuracies": accuracies}
    
    # Fallback: try test_metrics for final value only
    if "test_metrics" in history:
        test_metrics = history["test_metrics"]
        if "prob_discrete_accuracy" in test_metrics:
            acc = test_metrics["prob_discrete_accuracy"]
            
            if isinstance(acc, (int, float)):
                # Return as single epoch (epoch 75 assumed)
                return {"epochs": [75], "accuracies": [acc]}
    
    raise ValueError(
        f"Could not parse history file. Expected 'intermediate_probing' list or "
        f"'test_metrics.prob_discrete_accuracy'. Keys found: {list(history.keys())}"
    )


def load_experiment_epoch_data(base_dir, experiment_name, datasets):
    """
    Load epoch-wise accuracy for all runs of an experiment.
    
    Args:
        base_dir: Base directory containing ablation folders
        experiment_name: Name of experiment (e.g., "baseline", "gumbel_false")
        datasets: List of dataset IDs
        
    Returns:
        dict: {epoch: [list of accuracies across all runs]}
    """
    epoch_data = defaultdict(list)  # {epoch: [accuracies]}
    loaded_count = 0
    failed_files = []
    
    for ds in datasets:
        folder_name = f"ablation_{experiment_name}_ds{ds}"
        folder_path = base_dir / folder_name / "individual_runs"
        
        if not folder_path.exists():
            print(f"  WARNING: Folder not found: {folder_path}")
            continue
        
        # Find all history files
        history_files = list(folder_path.glob(f"history_seed_*_multi_saddle_{ds}.json"))
        
        for hist_file in history_files:
            try:
                data = load_history_file(hist_file)
                
                # Add each epoch's accuracy to the epoch_data dict
                for epoch, acc in zip(data["epochs"], data["accuracies"]):
                    epoch_data[epoch].append(acc)
                
                loaded_count += 1
                
            except Exception as e:
                failed_files.append((str(hist_file), str(e)))
    
    if failed_files:
        print(f"  WARNING: Failed to load {len(failed_files)} files")
        for f, e in failed_files[:3]:  # Show first 3 errors
            print(f"    {Path(f).name}: {e}")
    
    print(f"  ✓ Loaded {loaded_count} runs across {len(datasets)} datasets")
    
    # Convert to sorted arrays
    epochs = sorted(epoch_data.keys())
    means = []
    stds = []
    
    for epoch in epochs:
        values = np.array(epoch_data[epoch])
        means.append(np.mean(values))
        stds.append(np.std(values, ddof=1))  # Sample std
    
    return {
        "epochs": np.array(epochs),
        "mean": np.array(means),
        "std": np.array(stds),
        "n_runs": loaded_count,
        "n_values_per_epoch": [len(epoch_data[e]) for e in epochs]
    }


def plot_ablation_comparison(experiment_data, labels, output_path, title=None):
    """
    Create comparison plot matching the probing style.
    
    Args:
        experiment_data: List of dicts with {"epochs", "mean", "std", "n_runs"}
        labels: List of labels for each experiment
        output_path: Where to save the plot
        title: Optional title
    """
    # Paul Tol's muted color scheme
    tol_muted = [
        "#CC6677",  # Rose
        "#332288",  # Indigo
        "#DDCC77",  # Sand        
        "#117733",  # Green
        "#88CCEE",  # Cyan
        "#882255",  # Wine
        "#44AA99",  # Teal
        "#999933",  # Olive
    ]
    
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    hatches = ["///", "\\\\\\", "---", "***", "...", "xxx"]
    
    legend_handles = []
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot each experiment
    for i, (data, label) in enumerate(zip(experiment_data, labels)):
        color = tol_muted[i % len(tol_muted)]
        marker = markers[i % len(markers)]
        
        # Plot mean line with markers
        ax.plot(
            data["epochs"],
            data["mean"],
            marker=marker,
            linestyle="-",
            label=label,
            linewidth=2,
            markersize=4,
            color=color,
            alpha=0.9,
        )
        
        # Add std band
        ax.fill_between(
            data["epochs"],
            data["mean"] - data["std"],
            data["mean"] + data["std"],
            facecolor=color,
            alpha=0.12,
            hatch=hatches[i % len(hatches)],
            edgecolor=color,
            linewidth=0.8,
        )
        
        std_patch = Patch(
            facecolor="white",
            edgecolor=color,
            hatch=hatches[i % len(hatches)],
            linewidth=1.2,
            label=f"{label} ±1 std"
        )
        legend_handles.append(std_patch)
    
    # Styling
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("State Accuracy", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    
    # Get line handles from the axes
    line_handles, line_labels = ax.get_legend_handles_labels()
    
    # Combine: lines first, then std patches
    ax.legend(
        handles=line_handles + legend_handles,
        fontsize=10,
        loc='best'
    )
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-limits
    all_mins = [np.min(d["mean"] - d["std"]) for d in experiment_data]
    all_maxs = [np.max(d["mean"] + d["std"]) for d in experiment_data]
    y_min = max(0.0, min(all_mins) - 0.05)
    y_max = min(1.0, max(all_maxs) + 0.05)
    ax.set_ylim([y_min, y_max])
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved comparison plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot ablation study comparison over training epochs"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="neural_networks/output/ablation",
        help="Base directory containing ablation folders"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline experiment name (e.g., 'baseline')"
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        required=True,
        help="Ablation experiment names (e.g., 'gumbel_false' 'entropy0.0')"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=int,
        default=[3, 5, 8, 9],
        help="Dataset IDs to include (default: 3 5 8 9)"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Custom labels (default: use experiment names)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for plot (e.g., figs/ablation_comparison.png)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Plot title (optional)"
    )
    
    args = parser.parse_args()
    
    base_dir = PROJECT_ROOT / args.base_dir
    
    # All experiments to load (baseline + ablations)
    all_experiments = [args.baseline] + args.ablations
    
    # Default labels
    if args.labels:
        if len(args.labels) != len(all_experiments):
            raise ValueError(
                f"Number of labels ({len(args.labels)}) must match "
                f"number of experiments ({len(all_experiments)})"
            )
        labels = args.labels
    else:
        labels = all_experiments
    
    print("=" * 70)
    print("LOADING ABLATION COMPARISON DATA")
    print("=" * 70)
    print(f"Base directory: {base_dir}")
    print(f"Experiments: {all_experiments}")
    print(f"Datasets: {args.datasets}")
    print()
    
    # Load data for each experiment
    experiment_data = []
    
    for exp_name in all_experiments:
        print(f"Loading {exp_name}...")
        try:
            data = load_experiment_epoch_data(base_dir, exp_name, args.datasets)
            experiment_data.append(data)
            print(f"  Final epoch {data['epochs'][-1]}: "
                  f"{data['mean'][-1]:.4f} ± {data['std'][-1]:.4f} (n={data['n_runs']})")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
            return 1
    
    if not experiment_data:
        print("ERROR: No data loaded!")
        return 1
    
    # Create output directory
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create plot
    print("Creating comparison plot...")
    plot_ablation_comparison(experiment_data, labels, output_path, args.title)
    
    # Print summary
    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    for label, data in zip(labels, experiment_data):
        final_epoch = data["epochs"][-1]
        final_mean = data["mean"][-1]
        final_std = data["std"][-1]
        print(f"{label}:")
        print(f"  Epochs: {data['epochs'][0]} - {final_epoch}")
        print(f"  Final accuracy: {final_mean:.4f} ± {final_std:.4f}")
        print(f"  n = {data['n_runs']} runs")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())