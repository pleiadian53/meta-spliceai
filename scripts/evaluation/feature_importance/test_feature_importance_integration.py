#!/usr/bin/env python3
"""
Test script for feature importance integration with gene-wise CV workflow.

This script runs a small-scale test to verify that the comprehensive feature
importance analysis is properly integrated into the CV workflow.

Usage Examples:
--------------
# Basic test with default settings
python scripts/test_feature_importance_integration.py train_pc_1000/master

# Test with custom output directory
python scripts/test_feature_importance_integration.py train_pc_1000/master --output-dir models/meta_model_debug_run

# Full custom test
python scripts/test_feature_importance_integration.py train_pc_1000/master \
    --output-dir models/meta_model_debug_run \
    --sample-genes 15 \
    --n-folds 3 \
    --diag-sample 5000

# Analyze existing results only
python scripts/test_feature_importance_integration.py train_pc_1000/master \
    --analyze-only \
    --output-dir models/meta_model_debug_run
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
import argparse


def run_test_integration(
    dataset_path: str,
    output_dir: str = None,
    sample_genes: int = 15,
    n_folds: int = 3,
    diag_sample: int = 5000,
    verbose: bool = True
):
    """
    Run a small test of the feature importance integration.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset for testing
    output_dir : str, optional
        Output directory (creates temp dir if None)
    sample_genes : int
        Number of genes to sample for testing
    n_folds : int
        Number of CV folds to run
    diag_sample : int
        Sample size for diagnostics
    verbose : bool
        Whether to show verbose output
    """
    
    # Create output directory
    if output_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="feature_importance_test_")
        output_dir = temp_dir
        print(f"Using temporary output directory: {output_dir}")
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FEATURE IMPORTANCE INTEGRATION TEST")
    print("="*60)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Sample genes: {sample_genes}")
    print(f"CV folds: {n_folds}")
    print(f"Diagnostic sample: {diag_sample}")
    print()
    
    # Validate dataset path
    dataset_path_obj = Path(dataset_path)
    if not dataset_path_obj.exists():
        print(f"❌ ERROR: Dataset path does not exist: {dataset_path}")
        print("Please provide a valid dataset path.")
        print("Common examples:")
        print("  - train_pc_1000/master")
        print("  - path/to/your/dataset.parquet")
        print("  - path/to/your/dataset/directory")
        return None
    
    # Construct the command
    cmd = [
        sys.executable, "-m", 
        "meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid",
        "--dataset", str(dataset_path),
        "--out-dir", str(output_dir),
        "--sample-genes", str(sample_genes),
        "--n-folds", str(n_folds),
        "--diag-sample", str(diag_sample),
        "--row-cap", "50000",  # Limit total rows
        "--seed", "42",  # For reproducibility
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Let output go to console
            text=True
        )
        
        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Check for expected outputs
        expected_files = [
            "model_multiclass.pkl",
            "gene_cv_metrics.csv",
            "feature_manifest.csv",
            "feature_importance_analysis/gene_cv_comprehensive_results.xlsx",
            "feature_importance_analysis/integrated_summary.json"
        ]
        
        print("\nChecking expected output files:")
        output_path = Path(output_dir)
        all_present = True
        
        for file_path in expected_files:
            full_path = output_path / file_path
            if full_path.exists():
                size = full_path.stat().st_size
                print(f"✓ {file_path} ({size:,} bytes)")
            else:
                print(f"✗ {file_path} (missing)")
                all_present = False
        
        # Check for SHAP results
        shap_files = [
            "shap_importance_incremental.csv",
            "shap_viz/feature_importance_aggregate.pdf"
        ]
        
        print("\nChecking SHAP integration:")
        for file_path in shap_files:
            full_path = output_path / file_path
            if full_path.exists():
                size = full_path.stat().st_size
                print(f"✓ {file_path} ({size:,} bytes)")
            else:
                print(f"? {file_path} (optional, may fail due to data issues)")
        
        # Check feature importance analysis specifically
        fi_dir = output_path / "feature_importance_analysis"
        if fi_dir.exists():
            print(f"\nFeature importance analysis directory contents:")
            for item in sorted(fi_dir.iterdir()):
                if item.is_file():
                    size = item.stat().st_size
                    print(f"  {item.name} ({size:,} bytes)")
        
        if all_present:
            print("\n🎉 All core files generated successfully!")
            print("The feature importance integration is working correctly.")
        else:
            print("\n⚠️  Some expected files are missing.")
            print("The integration may have partial issues.")
        
        return output_dir
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ TEST FAILED!")
        print(f"Command failed with return code: {e.returncode}")
        print("Check the output above for error details.")
        return None
    except Exception as e:
        print(f"\n❌ TEST FAILED!")
        print(f"Unexpected error: {e}")
        return None


def analyze_results(output_dir: str):
    """
    Analyze the test results to provide insights.
    """
    if not output_dir or not Path(output_dir).exists():
        print("No valid output directory to analyze.")
        return
    
    print("\n" + "="*60)
    print("RESULT ANALYSIS")
    print("="*60)
    
    output_path = Path(output_dir)
    
    # Check CV metrics
    cv_metrics_file = output_path / "gene_cv_metrics.csv"
    if cv_metrics_file.exists():
        import pandas as pd
        try:
            cv_metrics = pd.read_csv(cv_metrics_file)
            print(f"\nCV Performance Summary:")
            print(f"  Number of folds: {len(cv_metrics)}")
            if 'test_accuracy' in cv_metrics.columns:
                print(f"  Mean test accuracy: {cv_metrics['test_accuracy'].mean():.3f} ± {cv_metrics['test_accuracy'].std():.3f}")
            if 'test_macro_f1' in cv_metrics.columns:
                print(f"  Mean macro F1: {cv_metrics['test_macro_f1'].mean():.3f} ± {cv_metrics['test_macro_f1'].std():.3f}")
            if 'top_k_accuracy' in cv_metrics.columns:
                print(f"  Mean top-k accuracy: {cv_metrics['top_k_accuracy'].mean():.3f} ± {cv_metrics['top_k_accuracy'].std():.3f}")
        except Exception as e:
            print(f"Error reading CV metrics: {e}")
    
    # Check feature importance summary
    fi_summary_file = output_path / "feature_importance_analysis" / "integrated_summary.json"
    if fi_summary_file.exists():
        import json
        try:
            with open(fi_summary_file, 'r') as f:
                fi_summary = json.load(f)
            
            print(f"\nFeature Importance Analysis Summary:")
            
            # Show top consensus features
            if 'consensus_features' in fi_summary and fi_summary['consensus_features']:
                print(f"  Top consensus features: {fi_summary['consensus_features'][:5]}")
            
            # Show method agreement
            if 'method_agreement' in fi_summary:
                agreements = fi_summary['method_agreement']
                if agreements:
                    avg_jaccard = sum(pair['jaccard_similarity'] for pair in agreements) / len(agreements)
                    print(f"  Average method agreement (Jaccard): {avg_jaccard:.3f}")
            
            # Show methods used
            if 'top_features_by_method' in fi_summary:
                methods = list(fi_summary['top_features_by_method'].keys())
                print(f"  Methods analyzed: {', '.join(methods)}")
                
        except Exception as e:
            print(f"Error reading feature importance summary: {e}")
    
    # Check Excel file
    excel_file = output_path / "feature_importance_analysis" / "gene_cv_comprehensive_results.xlsx"
    if excel_file.exists():
        print(f"\nExcel results file: {excel_file.name}")
        print(f"  Size: {excel_file.stat().st_size:,} bytes")
        
        try:
            import pandas as pd
            # Try to read sheet names
            excel_sheets = pd.ExcelFile(excel_file).sheet_names
            print(f"  Sheets: {', '.join(excel_sheets)}")
        except Exception as e:
            print(f"  Could not read Excel structure: {e}")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Test feature importance integration with CV workflow",
        epilog="""
Examples:
  # Basic test with default settings
  python scripts/test_feature_importance_integration.py train_pc_1000/master

  # Test with custom output directory  
  python scripts/test_feature_importance_integration.py train_pc_1000/master --output-dir models/meta_model_debug_run

  # Full custom test
  python scripts/test_feature_importance_integration.py train_pc_1000/master \\
      --output-dir models/meta_model_debug_run \\
      --sample-genes 15 \\
      --n-folds 3

  # Analyze existing results only
  python scripts/test_feature_importance_integration.py train_pc_1000/master \\
      --analyze-only \\
      --output-dir models/meta_model_debug_run
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "dataset",
        help="Path to the dataset for testing (e.g., 'train_pc_1000/master')"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for test results (uses temp dir if not specified)"
    )
    
    parser.add_argument(
        "--sample-genes", "-g",
        type=int,
        default=15,
        help="Number of genes to sample for testing (default: 15)"
    )
    
    parser.add_argument(
        "--n-folds", "-f",
        type=int,
        default=3,
        help="Number of CV folds to run (default: 3)"
    )
    
    parser.add_argument(
        "--diag-sample", "-s",
        type=int,
        default=5000,
        help="Sample size for diagnostics (default: 5000)"
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip test run and only analyze existing results"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce verbosity"
    )
    
    args = parser.parse_args()
    
    if args.analyze_only:
        if args.output_dir:
            analyze_results(args.output_dir)
        else:
            print("Error: --output-dir required when using --analyze-only")
            return 1
    else:
        # Run the test
        output_dir = run_test_integration(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            sample_genes=args.sample_genes,
            n_folds=args.n_folds,
            diag_sample=args.diag_sample,
            verbose=not args.quiet
        )
        
        # Analyze results if test succeeded
        if output_dir:
            analyze_results(output_dir)
            
            if not args.output_dir:  # Using temp directory
                print(f"\nTest output saved in: {output_dir}")
                print("Directory will be cleaned up when the script exits.")
        else:
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 