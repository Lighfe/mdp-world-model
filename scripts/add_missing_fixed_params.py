#!/usr/bin/env python3
"""
Post-hoc Database Fix: Add Missing Fixed Parameters
Adds fixed parameters to specific trial ranges in existing Optuna database.
"""

import sqlite3
import json
import argparse
from pathlib import Path


def analyze_database(db_path: str, start_trial: int, end_trial: int):
    """Analyze database and validate trial range."""
    print("="*60)
    print("DATABASE ANALYSIS")
    print("="*60)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get trial range info
    cursor.execute("SELECT MIN(trial_id), MAX(trial_id), COUNT(*) FROM trials")
    min_trial, max_trial, total_trials = cursor.fetchone()
    print(f"Database trials: {min_trial} to {max_trial} ({total_trials} total)")
    
    # Validate requested range
    cursor.execute("SELECT COUNT(*) FROM trials WHERE trial_id BETWEEN ? AND ?", (start_trial, end_trial))
    range_count = cursor.fetchone()[0]
    print(f"Requested range: {start_trial} to {end_trial} ({range_count} trials found)")
    
    if range_count == 0:
        print("❌ No trials found in specified range!")
        conn.close()
        return None, None, None
    
    # Get actual trial IDs in range
    cursor.execute("SELECT trial_id FROM trials WHERE trial_id BETWEEN ? AND ? ORDER BY trial_id", 
                   (start_trial, end_trial))
    trial_ids = [row[0] for row in cursor.fetchall()]
    
    # Get max param_id for safe incrementation
    cursor.execute("SELECT MAX(param_id) FROM trial_params")
    max_param_id = cursor.fetchone()[0] or 0
    print(f"Current max param_id: {max_param_id}")
    
    # Show current parameters for one trial in range
    cursor.execute("""SELECT param_name FROM trial_params 
                     WHERE trial_id = ? ORDER BY param_name""", (trial_ids[0],))
    current_params = [row[0] for row in cursor.fetchall()]
    print(f"Current parameters in trial {trial_ids[0]} ({len(current_params)}):")
    for param in current_params:
        print(f"  - {param}")
    
    conn.close()
    return trial_ids, max_param_id, range_count


def add_fixed_parameter(conn, trial_ids, param_name, param_value_index, distribution_json, start_param_id):
    """Add a fixed parameter to all specified trials."""
    cursor = conn.cursor()
    
    print(f"\nAdding {param_name} to {len(trial_ids)} trials...")
    print(f"  Value index: {param_value_index}")
    print(f"  Distribution: {distribution_json}")
    
    # Prepare insert data
    insert_data = []
    for i, trial_id in enumerate(trial_ids):
        param_id = start_param_id + i
        insert_data.append((
            param_id,
            trial_id,
            param_name,
            param_value_index,
            distribution_json
        ))
    
    # Insert all rows
    cursor.executemany("""
        INSERT INTO trial_params (param_id, trial_id, param_name, param_value, distribution_json)
        VALUES (?, ?, ?, ?, ?)
    """, insert_data)
    
    print(f"  ✅ Inserted {len(insert_data)} rows (param_ids {start_param_id} to {start_param_id + len(insert_data) - 1})")
    return start_param_id + len(insert_data)


def main():
    parser = argparse.ArgumentParser(description="Add missing fixed parameters to specific trial range")
    parser.add_argument("db_path", help="Path to Optuna database file")
    parser.add_argument("--start-trial", type=int, required=True, 
                       help="First trial_id to modify")
    parser.add_argument("--end-trial", type=int, required=True,
                       help="Last trial_id to modify (inclusive)")
    
    # Optional fixed parameter values (only add if specified)
    parser.add_argument("--state-loss-type", 
                       choices=["kl_div", "cross_entropy", "mse", "js_div"],
                       help="Add loss.state_loss_type with this fixed value")
    parser.add_argument("--use-entropy-reg", choices=["true", "false"],
                       help="Add loss.use_entropy_reg with this fixed value")
    parser.add_argument("--use-gumbel", choices=["true", "false"],
                       help="Add model.use_gumbel with this fixed value") 
    parser.add_argument("--use-target-encoder", choices=["true", "false"],
                       help="Add model.use_target_encoder with this fixed value")
    
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.start_trial > args.end_trial:
        print("❌ start-trial must be <= end-trial")
        return
    
    if not Path(args.db_path).exists():
        print(f"❌ Database not found: {args.db_path}")
        return
    
    print(f"Database: {args.db_path}")
    print(f"Trial range: {args.start_trial} to {args.end_trial}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE MODIFICATION'}")
    
    # Analyze database and validate range
    result = analyze_database(args.db_path, args.start_trial, args.end_trial)
    if result[0] is None:
        return
    trial_ids, max_param_id, range_count = result
    
    print(f"\nTrials to modify: {trial_ids}")
    
    # Define only the parameters that were specified
    fixed_params = {}
    
    # Only add loss.state_loss_type if specified
    if args.state_loss_type:
        state_loss_choices = ["kl_div", "cross_entropy", "mse", "js_div"]
        fixed_params["loss.state_loss_type"] = {
            "value_index": state_loss_choices.index(args.state_loss_type),
            "distribution": {
                "name": "CategoricalDistribution",
                "attributes": {"choices": state_loss_choices}
            }
        }
    
    # Only add loss.use_entropy_reg if specified
    if args.use_entropy_reg:
        bool_choices = [False, True]
        fixed_params["loss.use_entropy_reg"] = {
            "value_index": bool_choices.index(args.use_entropy_reg.lower() == "true"),
            "distribution": {
                "name": "CategoricalDistribution", 
                "attributes": {"choices": bool_choices}
            }
        }
    
    # Only add model.use_gumbel if specified
    if args.use_gumbel:
        bool_choices = [False, True]
        fixed_params["model.use_gumbel"] = {
            "value_index": bool_choices.index(args.use_gumbel.lower() == "true"),
            "distribution": {
                "name": "CategoricalDistribution", 
                "attributes": {"choices": bool_choices}
            }
        }
    
    # Only add model.use_target_encoder if specified
    if args.use_target_encoder:
        bool_choices = [False, True]
        fixed_params["model.use_target_encoder"] = {
            "value_index": bool_choices.index(args.use_target_encoder.lower() == "true"),
            "distribution": {
                "name": "CategoricalDistribution", 
                "attributes": {"choices": bool_choices}
            }
        }
    
    # Check if any parameters were specified
    if not fixed_params:
        print("❌ No parameters specified! Use --state-loss-type, --use-entropy-reg, --use-gumbel, or --use-target-encoder")
        return
    
    # Show modification plan
    print("\n" + "="*60)
    print("MODIFICATION PLAN")
    print("="*60)
    
    next_param_id = max_param_id + 1
    for param_name, param_info in fixed_params.items():
        actual_value = param_info['distribution']['attributes']['choices'][param_info['value_index']]
        print(f"Parameter: {param_name}")
        print(f"  Value: {actual_value} (index {param_info['value_index']})")
        print(f"  Param IDs: {next_param_id} to {next_param_id + len(trial_ids) - 1}")
        next_param_id += len(trial_ids)
    
    print(f"\nParameters to add: {list(fixed_params.keys())}")
    
    total_new_rows = len(fixed_params) * len(trial_ids)
    print(f"Total new rows: {total_new_rows}")
    print(f"Trials affected: {len(trial_ids)}")
    print(f"Parameters per trial: {len(fixed_params)}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - No changes made to database")
        return
    
    # Confirm before proceeding
    print(f"\nThis will add {total_new_rows} rows to the database.")
    print("Note: Non-contiguous param_ids are normal and won't cause issues.")
    response = input(f"Proceed? [y/N]: ")
    if response.lower() != 'y':
        print("❌ Operation cancelled")
        return
    
    # Execute modifications
    print("\n" + "="*60)
    print("EXECUTING MODIFICATIONS")
    print("="*60)
    
    conn = sqlite3.connect(args.db_path)
    next_param_id = max_param_id + 1
    
    try:
        for param_name, param_info in fixed_params.items():
            distribution_json = json.dumps(param_info['distribution'])
            
            next_param_id = add_fixed_parameter(
                conn=conn,
                trial_ids=trial_ids, 
                param_name=param_name,
                param_value_index=param_info['value_index'],
                distribution_json=distribution_json,
                start_param_id=next_param_id
            )
        
        # Commit changes
        conn.commit()
        print(f"\n✅ Successfully added {total_new_rows} rows for {len(fixed_params)} parameters!")
        
        # Verify final counts
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trial_params")
        new_total = cursor.fetchone()[0]
        print(f"Database now has {new_total} parameter rows total")
        
        # Show sample of what was added
        cursor.execute("""
            SELECT trial_id, param_name, param_value 
            FROM trial_params 
            WHERE param_id >= ? 
            ORDER BY trial_id, param_name 
            LIMIT 12
        """, (max_param_id + 1,))
        
        print(f"\nSample of added rows:")
        for trial_id, param_name, param_value in cursor.fetchall():
            print(f"  Trial {trial_id}: {param_name} = {param_value}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        conn.rollback()
        raise
    
    finally:
        conn.close()
    
    print("\n🎉 Database modification completed!")
    print(f"Added parameters: {list(fixed_params.keys())}")
    print("These parameters are now visible in Optuna Dashboard.")


if __name__ == "__main__":
    main()