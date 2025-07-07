"""
Script to run batch layer probing on all training runs
"""
import sys
import argparse
from pathlib import Path

from neural_networks.drm_analytics import batch_layer_probing

def main():
    parser = argparse.ArgumentParser(description='Run batch layer probing on training runs')
    parser.add_argument('base_output_dir', type=str, 
                       help='Base directory containing run folders')
    parser.add_argument('--probing_size', type=int, default=5000,
                       help='Number of samples for probing (default: 5000)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (default: cuda)')
    parser.add_argument('--overwrite_existing', action='store_true',
                       help='Overwrite existing probing results')
    
    args = parser.parse_args()
    
    print("="*60)
    print("BATCH LAYER PROBING")
    print("="*60)
    print(f"Base directory: {args.base_output_dir}")
    print(f"Probing size: {args.probing_size}")
    print(f"Device: {args.device}")
    print(f"Overwrite existing: {args.overwrite_existing}")
    print("="*60)
    
    # Run batch probing
    results_df = batch_layer_probing(
        base_output_dir=args.base_output_dir,
        probing_size=args.probing_size,
        device=args.device,
        overwrite_existing=args.overwrite_existing
    )
    
    print(f"\nBatch probing completed!")
    print(f"Results dataframe shape: {results_df.shape}")
    
    if not results_df.empty:
        # Print summary statistics
        probing_cols = ['probing_discrete_accuracy', 'probing_embedding_accuracy']
        available_cols = [col for col in probing_cols if col in results_df.columns]
        
        if available_cols:
            print(f"\nSummary statistics:")
            for col in available_cols:
                if results_df[col].notna().sum() > 0:
                    print(f"  {col}: mean={results_df[col].mean():.4f}, "
                          f"std={results_df[col].std():.4f}, "
                          f"count={results_df[col].notna().sum()}")

if __name__ == "__main__":
    main()