#!/usr/bin/env python3
"""
Post-Sweep Analysis Script
Analyzes completed parameter sweep results and generates insights.
"""

import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json

def analyze_sweep_results(sweep_output_dir: str):
    """
    Comprehensive analysis of completed parameter sweep.
    
    Args:
        sweep_output_dir: Path to sweep output directory (e.g., neural_networks/output/sweep_123/)
    """
    sweep_dir = Path(sweep_output_dir)
    db_path = sweep_dir / "optuna_study.db"
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return
    
    # Load study from database
    print("Loading Optuna study from database...")
    studies = optuna.study.get_all_study_summaries(f"sqlite:///{db_path}")
    study_name = studies[0].study_name  # Get the first study
    
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}"
    )
    
    print(f"Loaded study '{study_name}' with {len(study.trials)} trials")
    
    # === BASIC SUMMARY ===
    print("\\n" + "="*60)
    print("PARAMETER SWEEP RESULTS SUMMARY")
    print("="*60)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    print(f"Total trials: {len(study.trials)}")
    print(f"Completed: {len(completed_trials)}")  
    print(f"Failed: {len(failed_trials)}")
    
    if study.best_trial:
        print(f"\\nBest performance: {study.best_value:.4f}")
        print(f"Best trial number: {study.best_trial.number}")
        print("\\nBest parameters:")
        for param, value in study.best_params.items():
            print(f"  {param}: {value}")
    
    # === PARAMETER IMPORTANCE ===
    print("\\n" + "="*60)
    print("PARAMETER IMPORTANCE ANALYSIS")
    print("="*60)
    
    try:
        importance = optuna.importance.get_param_importances(study)
        importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        print("Parameters ranked by importance:")
        for i, (param, score) in enumerate(importance_sorted, 1):
            print(f"{i:2d}. {param:25s}: {score:.4f}")
        
        # Save importance to file
        importance_file = sweep_dir / "parameter_importance.json"
        with open(importance_file, 'w') as f:
            json.dump(importance, f, indent=2)
        print(f"\\nParameter importance saved to: {importance_file}")
        
    except Exception as e:
        print(f"Could not calculate parameter importance: {e}")
    
    # === PERFORMANCE DISTRIBUTION ===
    print("\\n" + "="*60)
    print("PERFORMANCE DISTRIBUTION")
    print("="*60)
    
    if completed_trials:
        performances = [trial.value for trial in completed_trials]
        
        print(f"Performance statistics:")
        print(f"  Mean: {sum(performances)/len(performances):.4f}")
        print(f"  Std:  {pd.Series(performances).std():.4f}")
        print(f"  Min:  {min(performances):.4f}")
        print(f"  Max:  {max(performances):.4f}")
        
        # Top 10 trials
        top_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:10]
        print(f"\\nTop 10 trials:")
        for i, trial in enumerate(top_trials, 1):
            print(f"{i:2d}. Trial {trial.number}: {trial.value:.4f}")
    
    # === PARAMETER VALUE ANALYSIS ===
    print("\\n" + "="*60)
    print("PARAMETER VALUE ANALYSIS")  
    print("="*60)
    
    # Convert trials to DataFrame for analysis
    trial_data = []
    for trial in completed_trials:
        trial_info = {'trial_number': trial.number, 'performance': trial.value}
        trial_info.update(trial.params)
        trial_data.append(trial_info)
    
    if trial_data:
        df = pd.DataFrame(trial_data)
        
        # Analyze each parameter
        for param in df.columns:
            if param in ['trial_number', 'performance']:
                continue
                
            print(f"\\n{param}:")
            if df[param].dtype in ['object', 'bool']:  # Categorical
                param_performance = df.groupby(param)['performance'].agg(['mean', 'count', 'std']).round(4)
                print(param_performance)
            else:  # Numerical
                correlation = df[param].corr(df['performance'])
                print(f"  Correlation with performance: {correlation:.4f}")
        
        # Save detailed results
        results_file = sweep_dir / "detailed_results.csv"  
        df.to_csv(results_file, index=False)
        print(f"\\nDetailed results saved to: {results_file}")
    
    # === LEARNING CURVE ===
    print("\\n" + "="*60)
    print("OPTIMIZATION LEARNING CURVE")
    print("="*60)
    
    # Best performance over time
    best_so_far = []
    current_best = 0
    
    for trial in sorted(completed_trials, key=lambda t: t.number):
        if trial.value > current_best:
            current_best = trial.value
        best_so_far.append(current_best)
    
    if len(best_so_far) >= 10:
        improvement_10 = best_so_far[9] - best_so_far[0] if best_so_far[0] > 0 else 0
        improvement_total = best_so_far[-1] - best_so_far[0] if best_so_far[0] > 0 else 0
        
        print(f"Performance improvement:")
        print(f"  First 10 trials: +{improvement_10:.4f}")
        print(f"  Overall: +{improvement_total:.4f}")
        print(f"  Trials to 90% of best: {next((i for i, v in enumerate(best_so_far) if v >= 0.9 * best_so_far[-1]), len(best_so_far))}")
    
    # === RECOMMENDATIONS ===
    print("\\n" + "="*60)
    print("RECOMMENDATIONS FOR NEXT SWEEP")
    print("="*60)
    
    if len(completed_trials) >= 20:
        # Analyze convergence
        last_20_best = max([t.value for t in completed_trials[-20:]])
        overall_best = max([t.value for t in completed_trials])
        
        if abs(last_20_best - overall_best) < 0.001:
            print("✓ Optimization appears to have converged")
            print("→ Consider: More exploitative sampler for fine-tuning")
        else:
            print("⚠ Still improving in recent trials")  
            print("→ Consider: Running more trials or more explorative sampler")
        
        # Parameter-specific recommendations
        if 'importance' in locals():
            most_important = importance_sorted[0][0]
            least_important = importance_sorted[-1][0] 
            
            print(f"→ Focus next sweep on: {most_important} (most important)")
            print(f"→ Consider fixing: {least_important} (least important)")
    
    print("\\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved in: {sweep_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze parameter sweep results")
    parser.add_argument(
        "sweep_dir",
        help="Path to sweep output directory (e.g., neural_networks/output/sweep_123/)"
    )
    
    args = parser.parse_args()
    analyze_sweep_results(args.sweep_dir)


if __name__ == "__main__":
    main()