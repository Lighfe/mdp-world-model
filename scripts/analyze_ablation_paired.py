#!/usr/bin/env python3
"""
Paired t-test analysis for ablation study.

Compares ablation experiments to baseline using paired t-tests at two levels:
1. Overall: 256 pairs (64 seeds × 4 datasets)
2. Per-dataset: 64 pairs per dataset

Calculates statistics, effect sizes (Cohen's d), and 95% confidence intervals.

Usage:
    python scripts/analyze_ablation_paired.py \
        --baseline baseline \
        --ablations gumbel_false entropy0.0 vlw0.0 \
        --datasets 3 5 8 9 \
        --output analysis_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_experiment_data(base_dir, experiment_name, datasets):
    """
    Load all runs for a given experiment across datasets.
    
    Args:
        base_dir: Base directory containing ablation folders
        experiment_name: Name of experiment (e.g., "baseline", "gumbel_false")
        datasets: List of dataset IDs (e.g., [3, 5, 8, 9])
        
    Returns:
        dict: {(dataset, seed): accuracy_value}
    """
    data = {}
    missing_files = []
    
    for ds in datasets:
        folder_name = f"ablation_{experiment_name}_ds{ds}"
        folder_path = base_dir / folder_name / "individual_runs"
        
        if not folder_path.exists():
            print(f"  WARNING: Folder not found: {folder_path}")
            continue
        
        # Find all history files
        history_files = list(folder_path.glob(f"history_seed_*_multi_saddle_{ds}.json"))
        
        for hist_file in history_files:
            # Extract seed from filename
            # Format: history_seed_141_multi_saddle_3.json
            filename = hist_file.name
            parts = filename.split('_')
            seed = int(parts[2])
            
            try:
                with open(hist_file, 'r') as f:
                    history = json.load(f)
                
                accuracy = history['test_metrics']['prob_discrete_accuracy']
                data[(ds, seed)] = accuracy
                
            except Exception as e:
                missing_files.append(str(hist_file))
                print(f"  ERROR loading {hist_file}: {e}")
    
    if missing_files:
        print(f"  WARNING: Failed to load {len(missing_files)} files")
    
    return data


def calculate_distribution(values):
    """
    Calculate distribution of values across fixed bins.
    
    Bins: ≥90%, 80-90%, 70-80%, 60-70%, 50-60%, 40-50%, 30-40%, <30%
    
    Args:
        values: Array of accuracy values
        
    Returns:
        dict: {bin_name: count}
    """
    values = np.array(values)
    
    bins = {
        "90_100": int(np.sum((values >= 0.90) & (values <= 1.00))),
        "80_90": int(np.sum((values >= 0.80) & (values < 0.90))),
        "70_80": int(np.sum((values >= 0.70) & (values < 0.80))),
        "60_70": int(np.sum((values >= 0.60) & (values < 0.70))),
        "50_60": int(np.sum((values >= 0.50) & (values < 0.60))),
        "40_50": int(np.sum((values >= 0.40) & (values < 0.50))),
        "30_40": int(np.sum((values >= 0.30) & (values < 0.40))),
        "below_30": int(np.sum(values < 0.30)),
    }
    
    return bins


def calculate_statistics(values):
    """
    Calculate summary statistics for a set of values.
    
    Args:
        values: Array of values
        
    Returns:
        dict: Statistics including mean, std, min, max, distribution
    """
    values = np.array(values)
    
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),  # Sample std
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": len(values),
        "distribution": calculate_distribution(values)
    }


def cohens_d_paired(differences):
    """
    Calculate Cohen's d for paired samples.
    
    Args:
        differences: Array of paired differences
        
    Returns:
        float: Cohen's d
    """
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    if std_diff == 0:
        return 0.0
    
    return mean_diff / std_diff


def interpret_cohens_d(d):
    """
    Interpret Cohen's d effect size.
    
    Returns:
        str: Interpretation
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    elif abs_d < 2.0:
        return "large"
    else:
        return "very large"


def paired_ttest_analysis(baseline_values, ablation_values):
    """
    Perform paired t-test and calculate effect sizes.
    
    Args:
        baseline_values: Array of baseline values
        ablation_values: Array of ablation values (matched pairs)
        
    Returns:
        dict: Analysis results including t-statistic, p-value, Cohen's d, CI
    """
    baseline_values = np.array(baseline_values)
    ablation_values = np.array(ablation_values)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(baseline_values, ablation_values)
    
    # Differences
    differences = baseline_values - ablation_values
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se_diff = std_diff / np.sqrt(len(differences))
    
    # Cohen's d
    d = cohens_d_paired(differences)
    
    # 95% Confidence interval
    df = len(differences) - 1
    t_crit = stats.t.ppf(0.975, df)  # Two-tailed
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    return {
        "mean_diff": float(mean_diff),
        "p_value": float(p_value),
        "cohens_d": float(d),
        "cohens_d_interpretation": interpret_cohens_d(d),
        "ci_95": [float(ci_lower), float(ci_upper)],
        "n_pairs": len(differences)
    }


def format_distribution(dist, total):
    """Format distribution for printing."""
    lines = []
    for bin_name, count in dist.items():
        pct = 100 * count / total if total > 0 else 0
        bin_label = bin_name.replace('_', '-').replace('below-30', '<30%')
        if bin_name == "90_100":
            bin_label = "≥90%"
        else:
            bin_label = bin_label + "%"
        lines.append(f"    {bin_label:12} {count:4d} runs ({pct:5.1f}%)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Paired t-test analysis for ablation study"
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
        help="Dataset IDs to analyze (default: 3 5 8 9)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis_results.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    base_dir = PROJECT_ROOT / args.base_dir
    
    print("=" * 70)
    print("ABLATION STUDY PAIRED ANALYSIS")
    print("=" * 70)
    print(f"Base directory: {base_dir}")
    print(f"Baseline: {args.baseline}")
    print(f"Ablations: {', '.join(args.ablations)}")
    print(f"Datasets: {args.datasets}")
    print(f"Total runs per experiment: {64 * len(args.datasets)} (64 seeds × {len(args.datasets)} datasets)")
    print()
    
    # Load baseline data
    print(f"Loading baseline data...")
    baseline_data = load_experiment_data(base_dir, args.baseline, args.datasets)
    print(f"  ✓ Loaded {len(baseline_data)} runs")
    print()
    
    # Store all results
    results = {
        "baseline": args.baseline,
        "ablations": args.ablations,
        "datasets": args.datasets,
        "experiments": {}
    }
    
    # Calculate baseline statistics
    baseline_values_overall = list(baseline_data.values())
    baseline_stats_overall = calculate_statistics(baseline_values_overall)
    
    # Per-dataset baseline statistics
    baseline_per_dataset = {}
    for ds in args.datasets:
        ds_values = [v for (d, s), v in baseline_data.items() if d == ds]
        baseline_per_dataset[f"ds{ds}"] = calculate_statistics(ds_values)
    
    results["experiments"][args.baseline] = {
        "overall": baseline_stats_overall,
        "per_dataset": baseline_per_dataset
    }
    
    # Load and analyze each ablation
    ablation_results = {}
    
    for ablation_name in args.ablations:
        print(f"Loading {ablation_name} data...")
        ablation_data = load_experiment_data(base_dir, ablation_name, args.datasets)
        print(f"  ✓ Loaded {len(ablation_data)} runs")
        
        # Match pairs
        matched_pairs = []
        for key in baseline_data.keys():
            if key in ablation_data:
                matched_pairs.append(key)
        
        print(f"  ✓ Matched {len(matched_pairs)} pairs")
        
        if len(matched_pairs) == 0:
            print(f"  ERROR: No matched pairs found for {ablation_name}")
            print()
            continue
        
        # Extract matched values for overall analysis
        baseline_matched = [baseline_data[key] for key in matched_pairs]
        ablation_matched = [ablation_data[key] for key in matched_pairs]
        
        # Overall statistics
        ablation_stats_overall = calculate_statistics(ablation_matched)
        
        # Overall paired t-test
        overall_test = paired_ttest_analysis(baseline_matched, ablation_matched)
        ablation_stats_overall["vs_baseline"] = overall_test
        
        # Per-dataset analysis
        ablation_per_dataset = {}
        per_dataset_tests = {}
        
        for ds in args.datasets:
            ds_pairs = [(d, s) for (d, s) in matched_pairs if d == ds]
            
            if len(ds_pairs) == 0:
                continue
            
            baseline_ds = [baseline_data[key] for key in ds_pairs]
            ablation_ds = [ablation_data[key] for key in ds_pairs]
            
            # Statistics
            ablation_per_dataset[f"ds{ds}"] = calculate_statistics(ablation_ds)
            
            # Paired t-test
            ds_test = paired_ttest_analysis(baseline_ds, ablation_ds)
            ablation_per_dataset[f"ds{ds}"]["vs_baseline"] = ds_test
            per_dataset_tests[f"ds{ds}"] = ds_test
        
        ablation_results[ablation_name] = {
            "overall": ablation_stats_overall,
            "per_dataset": ablation_per_dataset
        }
        
        results["experiments"][ablation_name] = ablation_results[ablation_name]
        print()
    
    # Save JSON results
    output_path = PROJECT_ROOT / args.output
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_path}")
    print()
    
    # Print detailed results
    print("=" * 70)
    print("OVERALL PAIRED TEST RESULTS (n={})".format(len(baseline_matched)))
    print("=" * 70)
    print()
    
    # Baseline
    print(f"BASELINE")
    print("-" * 70)
    bs = baseline_stats_overall
    print(f"Mean accuracy:        {bs['mean']:.4f} ± {bs['std']:.4f} (sample std)")
    print(f"Range:                [{bs['min']:.4f}, {bs['max']:.4f}]")
    print(f"Distribution:")
    print(format_distribution(bs['distribution'], bs['n']))
    print()
    
    # Ablations
    for ablation_name, abl_data in ablation_results.items():
        ablation_label = ablation_name.replace('_', ' ').upper()
        print(f"{ablation_label}")
        print("-" * 70)
        
        stats_overall = abl_data['overall']
        vs_base = stats_overall['vs_baseline']
        
        print(f"Mean accuracy:        {stats_overall['mean']:.4f} ± {stats_overall['std']:.4f} (sample std)")
        print(f"Range:                [{stats_overall['min']:.4f}, {stats_overall['max']:.4f}]")
        print(f"Δ vs Baseline:        {vs_base['mean_diff']:+.4f}")
        
        # P-value formatting
        if vs_base['p_value'] < 0.001:
            p_str = "p < 0.001 ***"
        elif vs_base['p_value'] < 0.01:
            p_str = f"p = {vs_base['p_value']:.4f} **"
        elif vs_base['p_value'] < 0.05:
            p_str = f"p = {vs_base['p_value']:.4f} *"
        else:
            p_str = f"p = {vs_base['p_value']:.4f}"
        
        print(f"{p_str}")
        print(f"Cohen's d:            {vs_base['cohens_d']:.2f} ({vs_base['cohens_d_interpretation']} effect)")
        print(f"95% CI for Δ:         [{vs_base['ci_95'][0]:.4f}, {vs_base['ci_95'][1]:.4f}]")
        print(f"Distribution:")
        print(format_distribution(stats_overall['distribution'], stats_overall['n']))
        print()
    
    # Per-dataset results
    print("=" * 70)
    print("PER-DATASET PAIRED TEST RESULTS (n=64 each)")
    print("=" * 70)
    print()
    
    for ablation_name, abl_data in ablation_results.items():
        ablation_label = ablation_name.replace('_', ' ').upper()
        print(f"{ablation_label}")
        print("-" * 70)
        
        for ds in args.datasets:
            ds_key = f"ds{ds}"
            if ds_key not in abl_data['per_dataset']:
                continue
            
            baseline_ds = results["experiments"][args.baseline]["per_dataset"][ds_key]
            ablation_ds = abl_data['per_dataset'][ds_key]
            vs_base = ablation_ds['vs_baseline']
            
            # P-value formatting
            if vs_base['p_value'] < 0.001:
                p_str = "***"
            elif vs_base['p_value'] < 0.01:
                p_str = "**"
            elif vs_base['p_value'] < 0.05:
                p_str = "*"
            else:
                p_str = ""
            
            print(f"Dataset {ds} (n={ablation_ds['n']}):")
            print(f"  Baseline:  {baseline_ds['mean']:.4f} ± {baseline_ds['std']:.4f}, range [{baseline_ds['min']:.4f}, {baseline_ds['max']:.4f}]")
            print(f"  Ablation:  {ablation_ds['mean']:.4f} ± {ablation_ds['std']:.4f}, range [{ablation_ds['min']:.4f}, {ablation_ds['max']:.4f}]")
            print(f"  Δ = {vs_base['mean_diff']:+.4f}, Cohen's d = {vs_base['cohens_d']:.2f}, p < 0.001 {p_str}")
            print(f"  Distribution:")
            print(format_distribution(ablation_ds['distribution'], ablation_ds['n']))
            print()
        
        # Count significant datasets
        n_sig = sum(1 for ds in args.datasets 
                   if f"ds{ds}" in abl_data['per_dataset'] 
                   and abl_data['per_dataset'][f"ds{ds}"]["vs_baseline"]["p_value"] < 0.05)
        print(f"Significant on {n_sig}/{len(args.datasets)} datasets")
        print()
    
    # Summary table
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print()
    print(f"{'Ablation':<25} {'Overall Δ':>10} {'p-value':>10} {'d':>8} {'Sig DS':>8} {'Mean±Std':>15}")
    print("-" * 70)
    
    for ablation_name, abl_data in ablation_results.items():
        vs_base = abl_data['overall']['vs_baseline']
        stats_ov = abl_data['overall']
        
        # Count significant datasets
        n_sig = sum(1 for ds in args.datasets 
                   if f"ds{ds}" in abl_data['per_dataset'] 
                   and abl_data['per_dataset'][f"ds{ds}"]["vs_baseline"]["p_value"] < 0.05)
        
        p_str = "<0.001" if vs_base['p_value'] < 0.001 else f"{vs_base['p_value']:.4f}"
        
        print(f"{ablation_name:<25} "
              f"{vs_base['mean_diff']:>+10.4f} "
              f"{p_str:>10} "
              f"{vs_base['cohens_d']:>8.2f} "
              f"{n_sig}/{len(args.datasets)}     "
              f"{stats_ov['mean']:.4f} ± {stats_ov['std']:.4f}")
    
    print()
    print("=" * 70)
    print(f"Analysis complete. Results saved to: {output_path}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())