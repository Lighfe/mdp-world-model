#!/usr/bin/env python3
"""
Migration script to convert existing JSON results to Weights & Biases runs.

This script handles:
- Config parameters from config_*.json
- Training metrics from drm_history_*.json  
- Test results from drm_test_results_*.json
- State metrics with proper epoch alignment
- Offline W&B logging for HPC environments

Usage:
    python migrate_old_runs_to_wandb.py /path/to/output/directory --project my-project-name
"""

import wandb
import json
import argparse
from pathlib import Path
import sys
from datetime import datetime
import re


def extract_dataset_name(db_path):
    """Extract dataset name from db_path."""
    # Example: "datasets/results/multi_saddle_1.db" -> "multi_saddle_1"
    return Path(db_path).stem


def determine_config_to_log(config):
    """
    Determine which config parameters to log to W&B.
    
    Returns config dict with only relevant parameters for filtering/analysis.
    """
    # Core hyperparameters that are most useful for analysis
    relevant_keys = {
        # Model architecture
        'num_states', 'hidden_dim', 'predictor_type',
        
        # Training setup  
        'epochs', 'batch_size', 'lr', 'optimizer', 'scheduler_type',
        'use_warmup', 'warmup_epochs', 'min_lr', 'weight_decay',
        
        # Loss function configuration
        'state_loss_weight', 'value_loss_weight', 'state_loss_type', 'value_loss_type',
        'use_entropy_reg', 'entropy_weight', 'use_entropy_decay', 'entropy_decay_proportion',
        
        # Target encoder
        'use_target_encoder', 'ema_decay',
        
        # Gumbel softmax
        'use_gumbel', 'initial_temp', 'min_temp',
        
        # System/data configuration
        'system_type', 'val_size', 'test_size', 'probing_size',
        
        # Other important settings
        'encoder_init_method', 'seed', 'run_id'
    }
    
    # Filter config to only relevant keys
    filtered_config = {}
    for key in relevant_keys:
        if key in config:
            filtered_config[key] = config[key]
    
    # Add computed fields for easier filtering
    if 'db_path' in config:
        filtered_config['dataset_name'] = extract_dataset_name(config['db_path'])
    
    return filtered_config

def log_training_metrics(history, state_metrics_offset=1):
    """
    Log training metrics to W&B with proper epoch alignment.
    
    Args:
        history: Dictionary from drm_history_*.json
        state_metrics_offset: Offset between train metrics and state metrics epochs
    """
    
    # Get lengths for validation
    train_epochs = len(history.get('train_loss', []))
    state_metrics = history.get('state_metrics', [])
    softmax_rank_metrics = history.get('softmax_rank_metrics', [])
    
    print(f"    Logging {train_epochs} training epochs, {len(state_metrics)} state metric collections, {len(softmax_rank_metrics)} rank metric collections")
    
    # Log training/validation metrics (every epoch)
    for epoch in range(train_epochs):
        metrics_dict = {'epoch': epoch}
        
        # Basic training metrics
        metric_keys = [
            'train_loss', 'train_state_loss', 'train_value_loss', 'train_entropy_loss',
            'train_batch_entropy', 'train_individual_entropy',
            'val_loss', 'val_state_loss', 'val_value_loss', 'val_entropy_loss',
            'val_batch_entropy', 'val_individual_entropy'
        ]
        
        for key in metric_keys:
            if key in history and epoch < len(history[key]):
                # Use W&B's metric grouping (train/, val/, etc.)
                wandb_key = key.replace('_', '/', 1)  # train_loss -> train/loss
                metrics_dict[wandb_key] = history[key][epoch]
        
        # Entropy weight (if available)
        if 'train_entropy_weight' in history and epoch < len(history['train_entropy_weight']):
            metrics_dict['train/entropy_weight'] = history['train_entropy_weight'][epoch]
            
        wandb.log(metrics_dict)
    
    # Log state metrics (collected less frequently)
    for state_metric in state_metrics:
        if 'epoch' not in state_metric:
            continue
            
        state_epoch = state_metric['epoch']
        metrics_dict = {'epoch': state_epoch}
        
        # Extract all state-related metrics
        for key, value in state_metric.items():
            if key == 'epoch':
                continue
                
            # Convert state indices from 0-based to 1-based for W&B
            if key.startswith('state_') and ('_usage' in key or '_mean' in key):
                # Extract state index and increment by 1
                if '_usage' in key:
                    state_idx = int(key.split('_')[1])
                    wandb_key = f'states/state_{state_idx + 1}_usage'
                elif '_mean' in key:
                    state_idx = int(key.split('_')[1])
                    wandb_key = f'states/state_{state_idx + 1}_mean'
                else:
                    wandb_key = f'states/{key}'
            else:
                # Group other state metrics under 'states/' prefix
                wandb_key = f'states/{key}'
                
            metrics_dict[wandb_key] = value
        
        wandb.log(metrics_dict)
    
    # Log softmax rank metrics (collected less frequently)
    for rank_metric in softmax_rank_metrics:
        if 'epoch' not in rank_metric:
            continue
            
        rank_epoch = rank_metric['epoch']
        metrics_dict = {'epoch': rank_epoch}
        
        # Extract all rank-related metrics
        for key, value in rank_metric.items():
            if key == 'epoch':
                continue
                
            # Group all rank metrics under 'rank/' prefix
            wandb_key = f'rank/{key}'
            metrics_dict[wandb_key] = value
        
        wandb.log(metrics_dict)


def log_test_results(test_results):
    """Log final test/probing results as summary metrics."""

    print(f"    🧪 log_test_results called with: {test_results is not None}")
    
    if not test_results:
        print(f"    🧪 Early return - no test results")
        return
        
    summary_dict = {}
    
    # Test metrics to log as final results
    test_keys = [
        'test_loss', 'test_state_loss', 'test_value_loss', 'test_entropy_loss',
        'test_batch_entropy', 'test_individual_entropy',
        'prob_discrete_accuracy', 'prob_discrete_accuracy_unweighted'
    ]
    
    for key in test_keys:
        if key in test_results and test_results[key] is not None:
            # Use final/ prefix for summary metrics
            wandb_key = f'final/{key}'
            summary_dict[wandb_key] = test_results[key]
    
    # Log as summary (final values)
    for key, value in summary_dict.items():
        wandb.run.summary[key] = value
    
    print(f"    Logged {len(summary_dict)} final test metrics")


def migrate_single_run(run_dir, project_name, experiment_tag=None, dry_run=False):
    """
    Migrate a single run directory to W&B.
    
    Args:
        run_dir: Path to run directory
        project_name: W&B project name
        dry_run: If True, only print what would be done
    
    Returns:
        bool: Success status
    """
    
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {run_dir.name}")
        print(f"{'='*60}")
        
        # Find required files
        config_files = list(run_dir.glob("config_*.json"))
        history_files = list(run_dir.glob("drm_history_*.json"))
        
        if not config_files:
            print(f"  ❌ No config file found")
            return False
            
        if not history_files:
            print(f"  ❌ No history file found") 
            return False
        
        # Load config
        with open(config_files[0], 'r') as f:
            config = json.load(f)
            
        # Load history  
        with open(history_files[0], 'r') as f:
            history = json.load(f)
            
        # Load test results (optional)
        test_results = {}
        test_files = list(run_dir.glob("drm_test_results_*.json"))
        if test_files:
            with open(test_files[0], 'r') as f:
                test_results = json.load(f)
        
        run_id = config.get('run_id', run_dir.name.replace('run_', ''))
        dataset_name = extract_dataset_name(config.get('db_path', '')) if 'db_path' in config else 'unknown'
        system_type = config.get('system_type', 'unknown')
        
        print(f"  📊 Run ID: {run_id}")
        print(f"  🗂️  Dataset: {dataset_name}")
        print(f"  🔧 System: {system_type}")
        print(f"  📈 Train epochs: {len(history.get('train_loss', []))}")
        print(f"  🎯 State metrics: {len(history.get('state_metrics', []))}")
        print(f"  🧪 Test results: {'✓' if test_results else '❌'}")
        
        if dry_run:
            print(f"  🔍 DRY RUN - Would create W&B run")
            return True
            
        # Filter config for W&B
        wandb_config = determine_config_to_log(config)
        
        # Create W&B run
        tags = ["migrated"]
        if experiment_tag:
            tags.append(experiment_tag)
            
        run = wandb.init(
            project=project_name,
            name=run_id,
            group=system_type,  # Group by system_type only
            tags=tags,
            config=wandb_config,
            mode="offline"  # Critical for HPC
        )

        # Define custom metrics to allow logging out of order
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch") 
        wandb.define_metric("states/*", step_metric="epoch")
        wandb.define_metric("rank/*", step_metric="epoch")
        
        print(f"  🚀 Created W&B run: {run.name}")
        
        # Log all metrics
        log_training_metrics(history)
        print(f"  🧪 Test results keys: {list(test_results.keys()) if test_results else 'None'}")
        if test_results:
            print(f"  🧪 Sample test result: test_loss = {test_results.get('test_loss', 'MISSING')}")

        log_test_results(test_results)
        
        # Finish run
        wandb.finish()
        
        print(f"  ✅ Successfully migrated run")
        return True
        
    except Exception as e:
        print(f"  ❌ Error processing {run_dir.name}: {e}")
        return False


def migrate_experiment_directory(base_dir, project_name, experiment_tag=None, dry_run=False, filter_pattern=None):
    """
    Migrate all runs in an experiment directory.
    
    Args:
        base_dir: Base output directory containing run_* subdirectories
        project_name: W&B project name  
        dry_run: If True, only show what would be migrated
        filter_pattern: Optional regex to filter run directories
        
    Returns:
        dict: Migration statistics
    """
    
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ Directory does not exist: {base_path}")
        return {'success': 0, 'failed': 0, 'total': 0}
    
    print(f"🔍 Scanning directory: {base_path}")
    
    # Find all run directories
    run_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('run_')]
    
    # Apply filter if specified
    if filter_pattern:
        import re
        pattern = re.compile(filter_pattern)
        run_dirs = [d for d in run_dirs if pattern.search(d.name)]
        print(f"🔍 Applied filter '{filter_pattern}': {len(run_dirs)} runs match")
    
    if not run_dirs:
        print("❌ No run directories found")
        return {'success': 0, 'failed': 0, 'total': 0}
    
    print(f"📦 Found {len(run_dirs)} run directories")
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - No actual migration will occur")
    
    # Process each run
    success_count = 0
    failed_count = 0
    
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"\n[{i}/{len(run_dirs)}]", end=" ")
        
        if migrate_single_run(run_dir, project_name, experiment_tag, dry_run):
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"MIGRATION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📊 Total: {len(run_dirs)}")
    
    if not dry_run and success_count > 0:
        print(f"\n🚀 Next steps:")
        print(f"1. From HPC login node, run: wandb sync <wandb_offline_dir>")
        print(f"2. Check results at: https://wandb.ai/YOUR_USERNAME/{project_name}")
    
    return {
        'success': success_count,
        'failed': failed_count, 
        'total': len(run_dirs)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing JSON experiment results to Weights & Biases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Migrate all runs in a directory with experiment tag
    python migrate_old_runs_to_wandb.py neural_networks/output/grid_search_results --project social-tipping-grid --tag grid_search_v1
    
    # Dry run to see what would be migrated
    python migrate_old_runs_to_wandb.py neural_networks/output/experiments --project my-project --dry-run
    
    # Migrate only runs matching a pattern
    python migrate_old_runs_to_wandb.py neural_networks/output/experiments --project my-project --filter "social_tipping.*seed42" --tag pilot_study
        """
    )
    
    parser.add_argument('base_dir', 
                       help='Base directory containing run_* subdirectories')
    parser.add_argument('--project', '-p',
                       required=True,
                       help='W&B project name')
    parser.add_argument('--tag',
                       help='Experiment tag to add to all runs')
    parser.add_argument('--dry-run', 
                       action='store_true',
                       help='Show what would be migrated without actually doing it')
    parser.add_argument('--filter', '-f',
                       help='Regex pattern to filter run directories')
    
    args = parser.parse_args()
    
    # Run migration
    stats = migrate_experiment_directory(
        base_dir=args.base_dir,
        project_name=args.project,
        experiment_tag=args.tag,
        dry_run=args.dry_run,
        filter_pattern=args.filter
    )
    
    # Exit with appropriate code
    if stats['total'] == 0:
        sys.exit(1)
    elif stats['failed'] > 0:
        sys.exit(2)  
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()