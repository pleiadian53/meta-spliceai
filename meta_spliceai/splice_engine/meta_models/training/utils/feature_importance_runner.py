#!/usr/bin/env python3
"""
Feature importance analysis runner utility.

This script provides a command-line interface for running feature importance
analysis on trained meta-models.
"""

import argparse
import sys
from pathlib import Path


def run_feature_importance_analysis(dataset_path: str, run_dir: str, sample: int = 25000):
    """
    Run comprehensive feature importance analysis.
    
    Args:
        dataset_path: Path to the dataset
        run_dir: Directory containing training results
        sample: Sample size for analysis
        
    Returns:
        Path to feature importance analysis directory
    """
    try:
        from meta_spliceai.splice_engine.meta_models.evaluation.feature_importance_integration import run_gene_cv_feature_importance_analysis
        
        print("🔍 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        print(f"Dataset: {dataset_path}")
        print(f"Run directory: {run_dir}")
        print(f"Sample size: {sample:,}")
        
        # Run the analysis
        feature_importance_dir = run_gene_cv_feature_importance_analysis(
            dataset_path=dataset_path,
            run_dir=run_dir,
            sample=sample
        )
        
        print(f"\n✅ Feature importance analysis completed!")
        print(f"📁 Results saved to: {feature_importance_dir}")
        
        return feature_importance_dir
        
    except ImportError as e:
        print(f"❌ Failed to import feature importance module: {e}")
        print("Make sure the evaluation module is properly installed.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Feature importance analysis failed: {e}")
        sys.exit(1)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Feature importance analysis runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run feature importance analysis
  python feature_importance_runner.py \\
    --dataset train_pc_5000_3mers_diverse/master \\
    --run-dir results/gene_cv_pc_5000_3mers_diverse_run1 \\
    --sample 25000

  # Run with default sample size
  python feature_importance_runner.py \\
    --dataset train_pc_5000_3mers_diverse/master \\
    --run-dir results/gene_cv_pc_5000_3mers_diverse_run1
        """
    )
    
    parser.add_argument("--dataset", required=True,
                       help="Path to the dataset")
    parser.add_argument("--run-dir", required=True,
                       help="Directory containing training results")
    parser.add_argument("--sample", type=int, default=25000,
                       help="Sample size for analysis (default: 25000)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate inputs
    dataset_path = Path(args.dataset)
    run_dir = Path(args.run_dir)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        sys.exit(1)
    
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        sys.exit(1)
    
    # Run feature importance analysis
    feature_importance_dir = run_feature_importance_analysis(
        str(dataset_path), 
        str(run_dir), 
        args.sample
    )
    
    print(f"\n🎉 Analysis completed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()




