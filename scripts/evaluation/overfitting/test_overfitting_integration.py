#!/usr/bin/env python3
"""
Test script for overfitting monitoring integration with CV scripts.

This script demonstrates how to use the integrated overfitting monitoring
with both gene-aware and chromosome-aware CV approaches.
"""

import os
import sys
import subprocess
from pathlib import Path
import json


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Command completed successfully")
        if result.stdout:
            print(f"Output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with return code {e.returncode}")
        if e.stdout:
            print(f"Stdout:\n{e.stdout}")
        if e.stderr:
            print(f"Stderr:\n{e.stderr}")
        return False


def check_output_files(output_dir: Path, cv_type: str) -> None:
    """Check that expected overfitting analysis files were created."""
    print(f"\n📁 Checking output files for {cv_type}:")
    
    # Main output directory files
    main_files = [
        f"{cv_type}_metrics.csv",
        "model_multiclass.pkl",
        "feature_manifest.csv"
    ]
    
    for file in main_files:
        file_path = output_dir / file
        status = "✅" if file_path.exists() else "❌"
        print(f"  {status} {file}")
    
    # Overfitting analysis subdirectory
    overfitting_dir = output_dir / "overfitting_analysis"
    print(f"\n📂 Overfitting analysis directory: {overfitting_dir}")
    
    if overfitting_dir.exists():
        print("  ✅ Overfitting analysis directory created")
        
        overfitting_files = [
            "overfitting_analysis.json",
            "overfitting_summary.txt",
            "learning_curves_by_fold.pdf",
            "aggregated_learning_curves.pdf",
            "overfitting_summary.pdf"
        ]
        
        for file in overfitting_files:
            file_path = overfitting_dir / file
            status = "✅" if file_path.exists() else "❌"
            print(f"    {status} {file}")
            
        # Read and display summary if available
        summary_file = overfitting_dir / "overfitting_analysis.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    print(f"\n📊 Overfitting Analysis Summary:")
                    print(f"    Total folds: {summary['summary']['total_folds']}")
                    print(f"    Folds with overfitting: {summary['summary']['folds_with_overfitting']}")
                    print(f"    Early stopped folds: {summary['summary']['early_stopped_folds']}")
                    print(f"    Mean performance gap: {summary['summary']['mean_performance_gap']:.4f}")
                    print(f"    Recommended n_estimators: {summary['summary']['recommended_n_estimators']}")
            except Exception as e:
                print(f"    ⚠️  Could not read summary: {e}")
    else:
        print("  ❌ Overfitting analysis directory not found")


def test_gene_cv_integration():
    """Test overfitting monitoring integration with gene-aware CV."""
    print("\n🧬 Testing Gene-Aware CV with Overfitting Monitoring")
    
    # Use small parameters for quick testing
    cmd = """
    python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
        --dataset train_pc_1000/master \
        --out-dir test_gene_cv_overfitting \
        --sample-genes 10 \
        --n-folds 3 \
        --n-estimators 50 \
        --monitor-overfitting \
        --overfitting-threshold 0.05 \
        --early-stopping-patience 10 \
        --convergence-improvement 0.001 \
        --plot-curves \
        --verbose \
        --seed 42
    """.strip().replace('\n    ', ' ')
    
    success = run_command(cmd, "Gene-aware CV with overfitting monitoring")
    
    if success:
        check_output_files(Path("test_gene_cv_overfitting"), "gene_cv")
    
    return success


def test_loco_cv_integration():
    """Test overfitting monitoring integration with chromosome-aware CV."""
    print("\n🧱 Testing Chromosome-Aware CV with Overfitting Monitoring")
    
    # Use small parameters for quick testing
    cmd = """
    python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
        --dataset train_pc_1000/master \
        --out-dir test_loco_cv_overfitting \
        --row-cap 5000 \
        --n-estimators 50 \
        --monitor-overfitting \
        --overfitting-threshold 0.05 \
        --early-stopping-patience 10 \
        --convergence-improvement 0.001 \
        --plot-curves \
        --verbose 2 \
        --seed 42
    """.strip().replace('\n    ', ' ')
    
    success = run_command(cmd, "Chromosome-aware CV with overfitting monitoring")
    
    if success:
        check_output_files(Path("test_loco_cv_overfitting"), "loco")
    
    return success


def compare_output_structures():
    """Compare output structures between the two approaches."""
    print("\n📋 Comparing Output Structures")
    
    gene_dir = Path("test_gene_cv_overfitting")
    loco_dir = Path("test_loco_cv_overfitting")
    
    for approach, output_dir in [("Gene CV", gene_dir), ("LOCO CV", loco_dir)]:
        print(f"\n{approach} Output Structure:")
        if output_dir.exists():
            # List all files and directories
            all_items = sorted(output_dir.rglob("*"))
            for item in all_items:
                if item.is_file():
                    rel_path = item.relative_to(output_dir)
                    size = item.stat().st_size
                    print(f"  📄 {rel_path} ({size:,} bytes)")
                elif item.is_dir() and item != output_dir:
                    rel_path = item.relative_to(output_dir)
                    print(f"  📁 {rel_path}/")
        else:
            print("  ❌ Output directory not found")


def generate_integration_report():
    """Generate a summary report of the integration test."""
    print("\n📊 Integration Test Report")
    print("=" * 60)
    
    gene_success = Path("test_gene_cv_overfitting/overfitting_analysis").exists()
    loco_success = Path("test_loco_cv_overfitting/overfitting_analysis").exists()
    
    print(f"Gene-aware CV integration:      {'✅ PASS' if gene_success else '❌ FAIL'}")
    print(f"Chromosome-aware CV integration: {'✅ PASS' if loco_success else '❌ FAIL'}")
    
    if gene_success and loco_success:
        print("\n🎉 All integrations successful!")
        print("\nNext steps:")
        print("1. Run with full datasets using the provided command examples")
        print("2. Analyze overfitting patterns in your specific data")
        print("3. Adjust n_estimators based on recommendations")
        print("4. Use the visualizations for presentations and publications")
    else:
        print("\n⚠️  Some integrations failed. Check the error messages above.")
        print("Common issues:")
        print("- Dataset path not found")
        print("- Missing dependencies")
        print("- Insufficient memory for larger datasets")
    
    # Cleanup test directories
    print(f"\n🧹 Cleaning up test directories...")
    import shutil
    for test_dir in ["test_gene_cv_overfitting", "test_loco_cv_overfitting"]:
        if Path(test_dir).exists():
            try:
                shutil.rmtree(test_dir)
                print(f"  Removed: {test_dir}")
            except Exception as e:
                print(f"  Could not remove {test_dir}: {e}")


def main():
    """Main test function."""
    print("🔬 Overfitting Monitoring Integration Test")
    print("=" * 60)
    print("This script tests the overfitting monitoring integration")
    print("with both gene-aware and chromosome-aware CV approaches.")
    
    # Check if we're in the right directory
    if not Path("meta_spliceai").exists():
        print("\n❌ Error: Please run this script from the project root directory")
        print("Expected to find 'meta_spliceai' directory")
        sys.exit(1)
    
    # Test both integrations
    try:
        gene_success = test_gene_cv_integration()
        loco_success = test_loco_cv_integration()
        
        if gene_success or loco_success:
            compare_output_structures()
        
        generate_integration_report()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 