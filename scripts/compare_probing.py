#!/usr/bin/env python3
"""
Compare probing accuracy across multiple configs.

Usage:
    python scripts/compare_probing.py \
        neural_networks/output/config1 \
        neural_networks/output/config2 \
        --labels "Baseline" "Experimental" \
        --output neural_networks/figs/probing_comparison.png
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Add project root to path (scripts/ -> project root)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_probing_data(aggregated_results_path):
    """
    Load probing metrics from aggregated_results.json.

    Args:
        aggregated_results_path: Path to aggregated_results.json

    Returns:
        dict with epochs, mean, std, count
    """
    with open(aggregated_results_path) as f:
        data = json.load(f)

    probing = data.get("probing_metrics", {})
    if not probing:
        raise ValueError(f"No probing metrics found in {aggregated_results_path}")

    if "epochs" not in probing:
        raise ValueError(f"No epochs found in probing metrics")

    if "discrete_accuracy" not in probing:
        raise ValueError(f"No discrete_accuracy found in probing metrics")

    accuracy_data = probing["discrete_accuracy"]

    return {
        "epochs": probing["epochs"],
        "mean": np.array(accuracy_data["mean"]),
        "std": np.array(accuracy_data["std"]),
        "count": accuracy_data["count"],
    }


def plot_probing_comparison(config_data_dict, output_path):
    """
    Create comparison plot matching softmax rank aggregated style.

    Args:
        config_data_dict: {"config_label": probing_data, ...}
        output_path: Where to save the plot
    """
    # Paul Tol's muted color scheme (same as softmax rank plots)
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

    # Different marker symbols for each config
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    hatches = ["///", "\\\\\\", "---", "***", "...", "***"]

    legend_handles = []

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot each config
    for i, (label, data) in enumerate(config_data_dict.items()):
        color = tol_muted[i % len(tol_muted)]
        marker = markers[i % len(markers)]

        # Plot mean line with markers (same style as softmax rank)
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

        if True:
            # Add std band (same alpha as softmax rank)
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

    # Styling (match softmax rank plots)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("State Accuracy", fontsize=12)
    #ax.set_title("Ablation Results", fontsize=14)
    # Get line handles from the axes
    line_handles, line_labels = ax.get_legend_handles_labels()

    # Combine: lines first, then std patches
    ax.legend(
        handles=line_handles + legend_handles,
        fontsize=10
    )
    ax.grid(True, alpha=0.2)
    ax.set_ylim((0.4, 1.0))  # Accuracy is between 0 and 1

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare probing accuracy across multiple configs"
    )
    parser.add_argument(
        "folders",
        nargs="+",
        help="Paths to config output folders (e.g., neural_networks/output/config1 config2 ...)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels for each config (must match number of folders)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for comparison plot (e.g., neural_networks/figs/comparison.png)",
    )

    args = parser.parse_args()

    # Validate
    if args.labels and len(args.labels) != len(args.folders):
        raise ValueError(
            f"Number of labels ({len(args.labels)}) must match number of folders ({len(args.folders)})"
        )

    # Load data from each config
    print(f"Loading data from {len(args.folders)} configs...")
    config_data = {}

    for i, folder in enumerate(args.folders):
        folder_path = Path(folder)
        aggregated_path = folder_path / "aggregated_results.json"

        if not aggregated_path.exists():

            aggregated_path = folder_path / "per_dataset_results.json"

            if not aggregated_path.exists():

                print(f"ERROR: {aggregated_path} not found, skipping {folder}")
                continue

        # Determine label
        if args.labels:
            label = args.labels[i]
        else:
            label = folder_path.name  # Use folder name as label

        try:
            data = load_probing_data(aggregated_path)
            config_data[label] = data
            print(
                f"  ✓ Loaded {label}: {len(data['epochs'])} epochs, n={data['count']}"
            )
        except Exception as e:
            print(f"  ✗ Failed to load {label}: {e}")

    if not config_data:
        print("\nERROR: No valid configs found!")
        return 1

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create comparison plot
    print(f"\nCreating comparison plot...")
    plot_probing_comparison(config_data, output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for label, data in config_data.items():
        final_mean = data["mean"][-1]
        final_std = data["std"][-1]
        final_epoch = data["epochs"][-1]
        print(f"{label}:")
        print(
            f"  Final accuracy (epoch {final_epoch}): {final_mean:.4f} ± {final_std:.4f}"
        )
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
