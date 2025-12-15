#!/usr/bin/env python
"""Test script for the leakage probe functionality."""
import argparse
import sys
from pathlib import Path

# Import the module the same way run_gene_cv_sigmoid.py does
from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils


def main():
    """Run a test of the leakage probe functionality."""
    parser = argparse.ArgumentParser(description="Test the leakage probe functionality")
    parser.add_argument("dataset_path", help="Path to the dataset (directory or file)")
    parser.add_argument("run_dir", help="Directory containing model and for output")
    parser.add_argument("--threshold", type=float, default=0.95, 
                        help="Correlation threshold for leakage detection")
    parser.add_argument("--sample", type=int, default=10000,
                        help="Number of samples to use")
    args = parser.parse_args()
    
    print(f"Testing leakage probe with dataset: {args.dataset_path}")
    print(f"Output directory: {args.run_dir}")
    print(f"Threshold: {args.threshold}, Sample size: {args.sample}")
    
    try:
        # Run the leakage probe function
        df_hits = _cutils.leakage_probe(
            dataset_path=args.dataset_path,
            run_dir=args.run_dir,
            threshold=args.threshold,
            sample=args.sample
        )
        
        # Print summary
        print("\nLeakage probe test completed successfully!")
        print(f"Total features analyzed: {len(df_hits)}")
        leaky_features = df_hits[df_hits["is_leaky"]]
        print(f"Potentially leaky features: {len(leaky_features)}")
        
        if not leaky_features.empty:
            print("\nTop 5 highest correlation features:")
            for idx, row in leaky_features.head(5).iterrows():
                print(f"  - {row['feature']}: correlation = {row['correlation']:.4f}")
                
        # Show the full report path
        report_path = Path(args.run_dir) / "feature_correlations.csv"
        print(f"\nFull report available at: {report_path}")
        
        return 0
    except Exception as e:
        print(f"Error testing leakage probe: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
