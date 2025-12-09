#!/usr/bin/env python3
"""
Statistical analysis of ablation study results.

For each ablation type, compares all configurations against baseline
using paired t-tests and effect sizes.

Input:
    neural_networks/output/ablation/summary/<RUN_ID>/cross_dataset_summary.json
    
Output:
    neural_networks/output/ablation/analysis/<ABLATION_TYPE>_analysis.json

Usage:
    python scripts/analyze_ablation_study.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def infer_ablation_type(run_id):
    """
    Infer ablation type from run_id.
    
    Examples:
        baseline -> baseline (belongs to all)
        vlw0.0 -> value_loss_weight
        entropy0.3 -> entropy_weight
        gumbel_false -> gumbel
        temp3.0 -> initial_temp
    
    Returns:
        str: ablation type or None
    """
    if run_id == "baseline":
        return "baseline"
    elif run_id.startswith("vlw"):
        return "value_loss_weight"
    elif run_id.startswith("entropy"):
        return "entropy_weight"
    elif run_id.startswith("gumbel"):
        return "gumbel"
    elif run_id.startswith("temp"):
        return "initial_temp"
    else:
        return None


def extract_parameter_value(run_id, ablation_type):
    """
    Extract the parameter value from run_id.
    
    Examples:
        vlw0.0 -> 0.0
        entropy0.3 -> 0.3
        gumbel_false -> False
        temp3.0 -> 3.0
    
    Returns:
        parameter value (float, bool, or str)
    """
    if ablation_type == "baseline":
        return "baseline"
    elif ablation_type == "value_loss_weight":
        return float(run_id[3:])  # after "vlw"
    elif ablation_type == "entropy_weight":
        return float(run_id[7:])  # after "entropy"
    elif ablation_type == "gumbel":
        return run_id.split("_")[1] == "true"  # gumbel_false or gumbel_true
    elif ablation_type == "initial_temp":
        return float(run_id[4:])  # after "temp"
    else:
        return run_id


def compute_cohens_d(mean1, std1, mean2, std2, n1=4, n2=4):
    """
    Compute Cohen's d effect size for paired samples.
    
    For paired samples, we use the standard deviation of the differences.
    Since we don't have the raw differences, we approximate using the pooled std.
    """
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (mean1 - mean2) / pooled_std
    return d


def paired_t_test_from_summary(data1, data2):
    """
    Perform paired t-test using per-dataset values.
    
    Args:
        data1: dict with 'per_dataset_accuracy' for config 1
        data2: dict with 'per_dataset_accuracy' for config 2
        
    Returns:
        tuple: (t_statistic, p_value)
    """
    # Extract values in consistent order
    datasets = sorted(data1['per_dataset_accuracy'].keys())
    
    values1 = [data1['per_dataset_accuracy'][ds] for ds in datasets]
    values2 = [data2['per_dataset_accuracy'][ds] for ds in datasets]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(values1, values2)
    
    return t_stat, p_value


def compute_win_rate(data1, data2):
    """
    Compute how many times data1 beats data2 across datasets.
    
    Returns:
        tuple: (wins, total) e.g., (3, 4) means "3/4"
    """
    datasets = sorted(data1['per_dataset_accuracy'].keys())
    
    wins = 0
    for ds in datasets:
        if data1['per_dataset_accuracy'][ds] > data2['per_dataset_accuracy'][ds]:
            wins += 1
    
    return wins, len(datasets)


def analyze_ablation_type(ablation_type, configs_data, baseline_data):
    """
    Analyze all configs for a specific ablation type.
    
    Args:
        ablation_type: str, e.g., "value_loss_weight"
        configs_data: dict mapping run_id to cross_dataset_summary data
        baseline_data: cross_dataset_summary data for baseline
        
    Returns:
        dict with analysis results
    """
    results = {
        "ablation_type": ablation_type,
        "baseline": {
            "run_id": "baseline",
            "mean": float(baseline_data['cross_dataset_stats']['mean']),
            "std": float(baseline_data['cross_dataset_stats']['std']),
            "cv": float(baseline_data['cross_dataset_stats']['cv']),
            "per_dataset": baseline_data['per_dataset_accuracy']
        },
        "configs": []
    }
    
    # Collect all configs with their metrics
    config_metrics = []
    
    for run_id, data in configs_data.items():
        param_value = extract_parameter_value(run_id, ablation_type)
        
        # Convert param_value to JSON-serializable type
        if isinstance(param_value, bool):
            param_value_json = param_value  # Python bool is JSON serializable
        elif isinstance(param_value, (int, float)):
            param_value_json = float(param_value)  # Ensure it's a Python float
        else:
            param_value_json = str(param_value)  # Convert to string as fallback
        
        config_metrics.append({
            "run_id": run_id,
            "param_value": param_value_json,
            "mean": float(data['cross_dataset_stats']['mean']),
            "std": float(data['cross_dataset_stats']['std']),
            "cv": float(data['cross_dataset_stats']['cv']),
            "per_dataset": data['per_dataset_accuracy'],
            "data": data  # keep for t-test
        })
    
    # Sort by mean (descending)
    config_metrics.sort(key=lambda x: x['mean'], reverse=True)
    
    # Assign ranks
    for rank, config in enumerate(config_metrics, start=1):
        config['rank'] = rank
    
    # Number of comparisons for Bonferroni correction
    n_comparisons = len(config_metrics)
    
    # Compare each config to baseline
    for config in config_metrics:
        # Statistical comparison vs baseline
        mean_diff = config['mean'] - baseline_data['cross_dataset_stats']['mean']
        relative_diff = (mean_diff / baseline_data['cross_dataset_stats']['mean']) * 100
        
        # Cohen's d
        cohens_d = compute_cohens_d(
            config['mean'], config['std'],
            baseline_data['cross_dataset_stats']['mean'],
            baseline_data['cross_dataset_stats']['std']
        )
        
        # Paired t-test
        t_stat, p_value = paired_t_test_from_summary(config['data'], baseline_data)
        p_value_bonferroni = min(p_value * n_comparisons, 1.0)
        
        # Win rate
        wins, total = compute_win_rate(config['data'], baseline_data)
        
        # Determine significance (convert to Python bool for JSON serialization)
        significant_uncorrected = bool(p_value < 0.05)
        significant_bonferroni = bool(p_value_bonferroni < 0.05)
        
        config['vs_baseline'] = {
            "absolute_diff": round(float(mean_diff), 6),
            "relative_diff_pct": round(float(relative_diff), 2),
            "cohens_d": round(float(cohens_d), 3),
            "t_statistic": round(float(t_stat), 3),
            "p_value": round(float(p_value), 6),
            "p_value_bonferroni": round(float(p_value_bonferroni), 6),
            "significant_uncorrected": significant_uncorrected,
            "significant_bonferroni": significant_bonferroni,
            "win_rate": f"{wins}/{total}",
            "interpretation": interpret_effect_size(cohens_d)
        }
        
        # Clean up data field before output
        del config['data']
        
        results['configs'].append(config)
    
    # Add summary statistics
    results['summary'] = {
        "n_configs": len(config_metrics),
        "n_comparisons": n_comparisons,
        "bonferroni_threshold": round(0.05 / n_comparisons, 6),
        "best_config": config_metrics[0]['run_id'],
        "best_mean": float(config_metrics[0]['mean']),
        "worst_config": config_metrics[-1]['run_id'],
        "worst_mean": float(config_metrics[-1]['mean']),
        "range": round(float(config_metrics[0]['mean'] - config_metrics[-1]['mean']), 6)
    }
    
    return results


def interpret_effect_size(d):
    """
    Interpret Cohen's d effect size.
    
    Returns:
        str: interpretation
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def main():
    """Main analysis workflow."""
    
    summary_dir = PROJECT_ROOT / "neural_networks" / "output" / "ablation" / "summary"
    
    if not summary_dir.exists():
        print(f"ERROR: Summary directory not found: {summary_dir}")
        print("Run aggregate_ablation_datasets.py first")
        return 1
    
    # Load all cross_dataset_summary.json files
    print("Loading cross-dataset summaries...")
    
    summaries = {}  # {run_id: data}
    
    for run_dir in sorted(summary_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        summary_file = run_dir / "cross_dataset_summary.json"
        if not summary_file.exists():
            print(f"WARNING: Missing {summary_file}")
            continue
        
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        run_id = data['run_id']
        summaries[run_id] = data
        print(f"  Loaded: {run_id} (mean = {data['cross_dataset_stats']['mean']:.6f})")
    
    print(f"\nLoaded {len(summaries)} run_ids")
    print()
    
    # Check for baseline
    if 'baseline' not in summaries:
        print("ERROR: Baseline results not found")
        return 1
    
    baseline_data = summaries['baseline']
    print(f"Baseline mean accuracy: {baseline_data['cross_dataset_stats']['mean']:.6f}")
    print()
    
    # Group by ablation type
    ablation_groups = defaultdict(dict)  # {ablation_type: {run_id: data}}
    
    for run_id, data in summaries.items():
        if run_id == 'baseline':
            continue
        
        ablation_type = infer_ablation_type(run_id)
        if ablation_type is None:
            print(f"WARNING: Could not infer ablation type for {run_id}")
            continue
        
        ablation_groups[ablation_type][run_id] = data
    
    print(f"Found {len(ablation_groups)} ablation types:")
    for abl_type, configs in ablation_groups.items():
        print(f"  {abl_type}: {len(configs)} configs")
    print()
    
    # Create analysis output directory
    analysis_dir = PROJECT_ROOT / "neural_networks" / "output" / "ablation" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each ablation type
    print("=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    print()
    
    for ablation_type in sorted(ablation_groups.keys()):
        print(f"Analyzing: {ablation_type}")
        print("-" * 70)
        
        configs_data = ablation_groups[ablation_type]
        
        try:
            analysis = analyze_ablation_type(ablation_type, configs_data, baseline_data)
            
            # Save analysis
            output_file = analysis_dir / f"{ablation_type}_analysis.json"
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            print(f"Configs: {analysis['summary']['n_configs']}")
            print(f"Best: {analysis['summary']['best_config']} "
                  f"(mean = {analysis['summary']['best_mean']:.6f})")
            print(f"Worst: {analysis['summary']['worst_config']} "
                  f"(mean = {analysis['summary']['worst_mean']:.6f})")
            print(f"Range: {analysis['summary']['range']:.6f}")
            print(f"Bonferroni threshold: p < {analysis['summary']['bonferroni_threshold']:.6f}")
            print()
            
            # Print comparison table
            print(f"{'Config':<20} {'Mean':<10} {'Rank':<6} {'Δ vs Base':<12} "
                  f"{'d':<8} {'p-value':<10} {'Sig?':<5}")
            print("-" * 70)
            
            for cfg in analysis['configs']:
                vs_base = cfg['vs_baseline']
                sig_marker = "***" if vs_base['significant_bonferroni'] else \
                            ("*" if vs_base['significant_uncorrected'] else "")
                
                print(f"{cfg['run_id']:<20} "
                      f"{cfg['mean']:<10.6f} "
                      f"{cfg['rank']:<6} "
                      f"{vs_base['absolute_diff']:>+.6f} "
                      f"{vs_base['cohens_d']:>+8.3f} "
                      f"{vs_base['p_value']:<10.6f} "
                      f"{sig_marker:<5}")
            
            print()
            print(f"✓ Saved to {output_file}")
            print()
            
        except Exception as e:
            print(f"ERROR analyzing {ablation_type}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # Create overall summary
    print("=" * 70)
    print("Creating overall summary...")
    
    overall_summary = {
        "baseline": {
            "mean": float(baseline_data['cross_dataset_stats']['mean']),
            "std": float(baseline_data['cross_dataset_stats']['std']),
            "per_dataset": baseline_data['per_dataset_accuracy']
        },
        "ablation_types": {}
    }
    
    for ablation_type in sorted(ablation_groups.keys()):
        analysis_file = analysis_dir / f"{ablation_type}_analysis.json"
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            
            overall_summary['ablation_types'][ablation_type] = {
                "n_configs": analysis['summary']['n_configs'],
                "best_config": analysis['summary']['best_config'],
                "best_mean": float(analysis['summary']['best_mean']),
                "best_vs_baseline": float(next(
                    cfg['vs_baseline']['absolute_diff']
                    for cfg in analysis['configs']
                    if cfg['run_id'] == analysis['summary']['best_config']
                )),
                "worst_config": analysis['summary']['worst_config'],
                "worst_mean": float(analysis['summary']['worst_mean']),
                "range": float(analysis['summary']['range'])
            }
    
    overall_file = analysis_dir / "overall_summary.json"
    with open(overall_file, 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    print(f"✓ Saved overall summary to {overall_file}")
    print()
    
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {analysis_dir}")
    print(f"Files created:")
    for f in sorted(analysis_dir.glob("*.json")):
        print(f"  - {f.name}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())