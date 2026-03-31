#!/usr/bin/env python3
"""
Plot state loss curves over training for ablation study comparisons.

Aggregates across all runs (256 = 64 seeds × 4 datasets) to get proper statistics.
Uses validation state loss if available, falls back to training state loss.

Usage:
    python scripts/plot_ablation_loss.py \
        --base-dir neural_networks/output/ablation \
        --baseline baseline \
        --ablations gumbel_false \
        --datasets 3 5 8 9 \
        --output neural_networks/figs/loss_comparison.png \
        --labels "Baseline" "No Gumbel"
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_loss_from_history(history_path):
    """
    Load state loss values from history file.
    
    Tries in order:
    1. val_state_loss (preferred)
    2. train_state_loss (fallback)
    
    Returns:
        dict: {"epochs": [...], "losses": [...], "loss_type": "val" or "train"}
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Try validation state loss first
    if "val_state_loss" in history:
        losses = history["val_state_loss"]
        if isinstance(losses, list) and len(losses) > 0:
            epochs = list(range(1, len(losses) + 1))
            return {"epochs": epochs, "losses": losses, "loss_type": "val"}
    
    # Fallback to training state loss
    if "train_state_loss" in history:
        losses = history["train_state_loss"]
        if isinstance(losses, list) and len(losses) > 0:
            epochs = list(range(1, len(losses) + 1))
            return {"epochs": epochs, "losses": losses, "loss_type": "train"}
    
    raise ValueError(
        f"Could not find 'val_state_loss' or 'train_state_loss' in history file. "
        f"Keys found: {list(history.keys())}"
    )


def load_experiment_loss_data(base_dir, experiment_name, datasets):
    """
    Load epoch-wise loss for all runs of an experiment.
    
    Args:
        base_dir: Base directory containing ablation folders
        experiment_name: Name of experiment (e.g., "baseline", "gumbel_false")
        datasets: List of dataset IDs
        
    Returns:
        dict: {epoch: [list of losses across all runs]}
    """
    epoch_data = defaultdict(list)  # {epoch: [losses]}
    loaded_count = 0
    failed_files = []
    loss_type = None
    
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
                data = load_loss_from_history(hist_file)
                
                # Track which loss type we're using
                if loss_type is None:
                    loss_type = data["loss_type"]
                
                # Add each epoch's loss to the epoch_data dict
                for epoch, loss in zip(data["epochs"], data["losses"]):
                    epoch_data[epoch].append(loss)
                
                loaded_count += 1
                
            except Exception as e:
                failed_files.append((str(hist_file), str(e)))
    
    if failed_files:
        print(f"  WARNING: Failed to load {len(failed_files)} files")
        for f, e in failed_files[:3]:  # Show first 3 errors
            print(f"    {Path(f).name}: {e}")
    
    print(f"  ✓ Loaded {loaded_count} runs across {len(datasets)} datasets")
    print(f"  Using: {loss_type}_state_loss")
    
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
        "loss_type": loss_type
    }


def plot_loss_comparison(experiment_data, labels, output_path, title=None):
    """
    Create loss comparison plot with clean line style (no markers) and hatched std bands.
    
    Args:
        experiment_data: List of dicts with {"epochs", "mean", "std", "loss_type"}
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
    
    # Hatch patterns for std bands
    hatches = ["///", "\\\\\\", "---", "***", "...", "xxx", "+++", "|||"]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot each experiment
    for i, (data, label) in enumerate(zip(experiment_data, labels)):
        color = tol_muted[i % len(tol_muted)]
        hatch = hatches[i % len(hatches)]
        
        # Plot mean line (no markers, just smooth line)
        ax.plot(
            data["epochs"],
            data["mean"],
            linestyle="-",
            label=label,
            linewidth=2.5,
            color=color,
            alpha=0.9,
        )
        
        # Add std band with hatch
        ax.fill_between(
            data["epochs"],
            data["mean"] - data["std"],
            data["mean"] + data["std"],
            facecolor=color,
            alpha=0.12,
            hatch=hatch,
            edgecolor=color,
            linewidth=0.8,
        )
    
    # Styling
    ax.set_xlabel("Epoch", fontsize=12)
    
    # Label based on loss type
    loss_type = experiment_data[0]["loss_type"]
    loss_label = "Validation State Loss" if loss_type == "val" else "Training State Loss"
    ax.set_ylabel(loss_label, fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-limits
    all_mins = [np.min(d["mean"] - d["std"]) for d in experiment_data]
    all_maxs = [np.max(d["mean"] + d["std"]) for d in experiment_data]
    y_min = max(0.0, min(all_mins) - 0.02)
    y_max = max(all_maxs) + 0.05
    ax.set_ylim([y_min, y_max])
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Saved loss comparison plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss curves for ablation study comparison"
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
        help="Output path for plot (e.g., figs/loss_comparison.png)"
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
    print("LOADING LOSS COMPARISON DATA")
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
            data = load_experiment_loss_data(base_dir, exp_name, args.datasets)
            experiment_data.append(data)
            print(f"  Final epoch {data['epochs'][-1]}: "
                  f"loss = {data['mean'][-1]:.4f} ± {data['std'][-1]:.4f}")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            print()
            return 1
    
    if not experiment_data:
        print("ERROR: No data loaded!")
        return 1
    
    # Check if all use same loss type
    loss_types = set(d["loss_type"] for d in experiment_data)
    if len(loss_types) > 1:
        print(f"WARNING: Mixed loss types detected: {loss_types}")
        print("This may produce misleading comparisons!")
    
    # Create output directory
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create plot
    print("Creating loss comparison plot...")
    plot_loss_comparison(experiment_data, labels, output_path, args.title)
    
    # Print summary
    print()
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    for label, data in zip(labels, experiment_data):
        final_epoch = data["epochs"][-1]
        final_mean = data["mean"][-1]
        final_std = data["std"][-1]
        initial_mean = data["mean"][0]
        reduction = initial_mean - final_mean
        reduction_pct = (reduction / initial_mean) * 100
        
        print(f"{label}:")
        print(f"  Epochs: {data['epochs'][0]} - {final_epoch}")
        print(f"  Initial loss: {initial_mean:.4f}")
        print(f"  Final loss: {final_mean:.4f} ± {final_std:.4f}")
        print(f"  Reduction: {reduction:.4f} ({reduction_pct:.1f}%)")
        print(f"  n = {data['n_runs']} runs")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())