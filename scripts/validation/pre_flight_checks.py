#!/usr/bin/env python3
"""
Comprehensive pre-flight checks for CV scripts.

This script validates dataset structure, module imports, system resources,
and provides detailed dataset analysis before running cross-validation workflows.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-flight checks for CV scripts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset", 
        default="train_pc_1000/master",
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--row-cap", 
        type=int, 
        default=1000,
        help="Maximum number of rows to load for testing"
    )
    
    parser.add_argument(
        "--gene-cv-dir", 
        default="results_gene_cv_1000",
        help="Output directory for gene-wise CV"
    )
    
    parser.add_argument(
        "--loco-cv-dir", 
        default="results_loco_cv_1000",
        help="Output directory for chromosome-wise CV"
    )
    
    parser.add_argument(
        "--skip-dataset-load", 
        action="store_true",
        help="Skip dataset loading test (faster, but less thorough)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def check_dataset_basic(dataset_path: str) -> bool:
    """Basic dataset existence and structure check."""
    print(f"=== Dataset Basic Checks ===")
    print(f"Dataset path: {dataset_path}")
    
    path = Path(dataset_path)
    
    # Check if dataset path exists
    if not path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False
    
    print(f"✅ Dataset path exists")
    
    # List contents
    if path.is_dir():
        contents = list(path.iterdir())
        print(f"📁 Dataset contents: {len(contents)} files/directories")
        
        # Show parquet files specifically
        parquet_files = [f for f in contents if f.suffix == '.parquet']
        if parquet_files:
            print(f"   Parquet files: {len(parquet_files)}")
            for file in parquet_files[:5]:
                print(f"     - {file.name}")
            if len(parquet_files) > 5:
                print(f"     ... and {len(parquet_files) - 5} more")
        
        # Show other files
        other_files = [f for f in contents if f.suffix != '.parquet']
        if other_files:
            print(f"   Other files: {len(other_files)}")
            for file in other_files[:3]:
                print(f"     - {file.name}")
    elif path.is_file():
        print(f"📄 Single file: {path.name}")
        if path.suffix == '.parquet':
            print(f"   ✅ Parquet format detected")
        else:
            print(f"   ⚠️  File format: {path.suffix}")
    
    return True


def check_dataset_loading(dataset_path: str, row_cap: int, verbose: bool = False) -> Dict[str, Any]:
    """Check dataset loading with detailed analysis."""
    print(f"\n=== Dataset Loading & Analysis ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Row cap: {row_cap:,}")
    
    result = {
        "success": False,
        "shape": None,
        "columns": [],
        "essential_cols": {},
        "target_cols": [],
        "label_distribution": {},
        "base_model_cols": [],
    }
    
    try:
        # Import the datasets module
        from meta_spliceai.splice_engine.meta_models.training import datasets
        print("✅ Successfully imported datasets module")
        
        # Set environment variable for row cap (gene CV style)
        original_max_rows = os.environ.get("SS_MAX_ROWS")
        os.environ["SS_MAX_ROWS"] = str(row_cap)
        
        try:
            # Load dataset
            print("Loading dataset...")
            df = datasets.load_dataset(dataset_path)
            print(f"✅ Dataset loaded successfully")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {len(df.columns)} total")
            
            result["success"] = True
            result["shape"] = df.shape
            result["columns"] = list(df.columns)
            
            # Check for target/label columns
            target_candidates = ['label', 'splice_type', 'class', 'target', 'y']
            present_targets = [col for col in target_candidates if col in df.columns]
            result["target_cols"] = present_targets
            
            if present_targets:
                print(f"✅ Target column candidates found: {present_targets}")
            else:
                print(f"⚠️  No standard target columns found from: {target_candidates}")
            
            # Check for essential columns needed by CV scripts
            essential_cols = ['splice_type', 'gene_id', 'chrom', 'donor_score', 'acceptor_score']
            for col in essential_cols:
                if col in df.columns:
                    print(f"✅ {col}: present")
                    result["essential_cols"][col] = True
                    
                    # Get label distribution for splice_type
                    if col == 'splice_type':
                        try:
                            label_counts = df[col].value_counts().to_dict()
                            result["label_distribution"] = label_counts
                            print(f"   Label distribution: {label_counts}")
                        except Exception as e:
                            print(f"   ⚠️  Could not get label distribution: {e}")
                else:
                    print(f"❌ {col}: missing")
                    result["essential_cols"][col] = False
            
            # Check for base model score columns
            base_cols = ['donor_score', 'acceptor_score', 'score']
            present_base_cols = [col for col in base_cols if col in df.columns]
            result["base_model_cols"] = present_base_cols
            
            if present_base_cols:
                print(f"✅ Base model columns found: {present_base_cols}")
            else:
                print(f"⚠️  No base model score columns found from: {base_cols}")
            
            # Show sample of column names if verbose
            if verbose:
                print(f"\nSample of column names:")
                for i, col in enumerate(df.columns[:20]):
                    print(f"  {i+1:2d}. {col}")
                if len(df.columns) > 20:
                    print(f"  ... and {len(df.columns) - 20} more columns")
                
                # Check for any column that might contain class information
                print(f"\nColumns containing 'splice', 'class', or 'label' in name:")
                relevant_cols = [col for col in df.columns if any(word in col.lower() for word in ['splice', 'class', 'label', 'type'])]
                for col in relevant_cols:
                    print(f"  - {col}")
                    if hasattr(df[col], 'value_counts'):
                        try:
                            counts = df[col].value_counts()
                            if len(counts) <= 10:  # Only show if not too many unique values
                                print(f"    Values: {counts.to_dict()}")
                        except:
                            pass
            
        finally:
            # Restore original environment variable
            if original_max_rows is not None:
                os.environ["SS_MAX_ROWS"] = original_max_rows
            elif "SS_MAX_ROWS" in os.environ:
                del os.environ["SS_MAX_ROWS"]
        
        return result
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        result["error"] = str(e)
        return result


def check_imports() -> bool:
    """Check all required module imports."""
    print("\n=== Module Import Checks ===")
    
    import_tests = [
        ("meta_spliceai.splice_engine.meta_models.training.datasets", "datasets"),
        ("meta_spliceai.splice_engine.meta_models.builder.preprocessing", "preprocessing"),
        ("meta_spliceai.splice_engine.meta_models.evaluation.viz_utils", "viz_utils"),
        ("meta_spliceai.splice_engine.meta_models.evaluation.multiclass_roc_pr", "multiclass_roc_pr"),
        ("meta_spliceai.splice_engine.meta_models.evaluation.shap_viz", "shap_viz"),
        ("meta_spliceai.splice_engine.meta_models.evaluation.feature_importance_integration", "feature_importance"),
        ("meta_spliceai.splice_engine.meta_models.evaluation.cv_metrics_viz", "cv_metrics_viz"),
        ("meta_spliceai.splice_engine.meta_models.evaluation.shap_incremental", "shap_incremental"),
        ("meta_spliceai.splice_engine.meta_models.training.classifier_utils", "classifier_utils"),
        ("xgboost", "XGBoost"),
        ("sklearn", "scikit-learn"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
    ]
    
    failed_imports = []
    
    for module_name, display_name in import_tests:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
        except ImportError as e:
            print(f"❌ {display_name}: {e}")
            failed_imports.append(display_name)
    
    if failed_imports:
        print(f"\n⚠️  Failed imports: {', '.join(failed_imports)}")
        return False
    else:
        print(f"\n✅ All imports successful")
        return True


def check_output_directories(gene_cv_dir: str, loco_cv_dir: str) -> bool:
    """Check that output directories can be created."""
    print(f"\n=== Output Directory Checks ===")
    
    test_dirs = [gene_cv_dir, loco_cv_dir]
    
    for dir_name in test_dirs:
        try:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            print(f"✅ Can create directory: {dir_name}")
            
            # Test write permission
            test_file = Path(dir_name) / "test_write.txt"
            test_file.write_text("test")
            test_file.unlink()
            print(f"✅ Write permission confirmed: {dir_name}")
            
        except Exception as e:
            print(f"❌ Directory/write issue for {dir_name}: {e}")
            return False
    
    return True


def check_system_resources() -> Dict[str, Any]:
    """Check system resources and provide estimates."""
    print(f"\n=== System Resource Analysis ===")
    
    resources = {
        "memory": {},
        "storage": {},
        "recommendations": []
    }
    
    # Get memory info
    try:
        import psutil
        total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        
        resources["memory"]["total"] = total_memory
        resources["memory"]["available"] = available_memory
        
        print(f"💾 Total system memory: {total_memory:.1f} GB")
        print(f"💾 Available memory: {available_memory:.1f} GB")
        
        # Memory recommendations
        if available_memory < 8:
            print("⚠️  Warning: Less than 8GB available memory")
            print("   Consider reducing --diag-sample or --neigh-sample")
            resources["recommendations"].append("Reduce diagnostic sample sizes")
        elif available_memory >= 16:
            print("✅ Sufficient memory for full analysis")
        else:
            print("✅ Adequate memory, monitor usage during run")
            
    except ImportError:
        print("📊 Install psutil for detailed memory information")
        resources["memory"]["error"] = "psutil not available"
    
    # Storage estimates  
    print(f"💿 Estimated storage per run: 2-5 GB")
    print(f"💿 Total storage needed: ~10-15 GB")
    
    resources["storage"]["per_run"] = "2-5 GB"
    resources["storage"]["total"] = "10-15 GB"
    
    # Check for potential schema issues with large datasets
    if "5000" in dataset_path or "5k" in dataset_path.lower():
        print(f"📊 Large dataset detected: {dataset_path}")
        print(f"   ⚠️  Note: Large datasets may have schema differences")
        print(f"   💡 If you encounter schema errors, try:")
        print(f"      export POLARS_EXTRA_COLUMNS_IGNORE=true")
        resources["recommendations"].append("Set POLARS_EXTRA_COLUMNS_IGNORE=true for large datasets")
    
    # Time estimates
    print(f"⏱️  Estimated runtime:")
    print(f"   - Gene-wise CV: 2-4 hours")
    print(f"   - Chromosome-wise CV: 1-3 hours")
    print(f"   - Total: 3-7 hours")
    
    resources["time"] = {
        "gene_cv": "2-4 hours",
        "loco_cv": "1-3 hours",
        "total": "3-7 hours"
    }
    
    return resources


def generate_ready_commands(dataset_path: str, gene_cv_dir: str, loco_cv_dir: str) -> Dict[str, str]:
    """Generate ready-to-run commands for CV scripts using module syntax."""
    
    commands = {
        "gene_cv": f"""python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \\
    --dataset {dataset_path} \\
    --out-dir {gene_cv_dir} \\
    --n-folds 5 \\
    --n-estimators 500 \\
    --diag-sample 15000 \\
    --top-k 5 \\
    --plot-curves \\
    --plot-format pdf \\
    --check-leakage \\
    --leakage-threshold 0.95 \\
    --calibrate \\
    --calib-method platt \\
    --neigh-sample 5000 \\
    --neigh-window 10 \\
    --verbose \\
    --seed 42""",
        
        "loco_cv": f"""python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \\
    --dataset {dataset_path} \\
    --out-dir {loco_cv_dir} \\
    --n-estimators 500 \\
    --diag-sample 15000 \\
    --plot-curves \\
    --plot-format pdf \\
    --check-leakage \\
    --leakage-threshold 0.95 \\
    --calibrate \\
    --calib-method platt \\
    --neigh-sample 5000 \\
    --neigh-window 10 \\
    --verbose 2 \\
    --seed 42"""
    }
    
    return commands


def main():
    """Main function to run all pre-flight checks."""
    args = parse_args()
    
    print("🚀 Comprehensive Pre-flight Checks for CV Scripts")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Row cap: {args.row_cap:,}")
    print(f"Gene CV output: {args.gene_cv_dir}")
    print(f"LOCO CV output: {args.loco_cv_dir}")
    
    # Define checks
    checks = [
        ("Dataset Basic", lambda: check_dataset_basic(args.dataset)),
        ("Module Imports", check_imports),
        ("Output Directories", lambda: check_output_directories(args.gene_cv_dir, args.loco_cv_dir)),
        ("System Resources", lambda: check_system_resources()),
    ]
    
    # Add dataset loading check if not skipped
    if not args.skip_dataset_load:
        checks.insert(1, ("Dataset Loading", lambda: check_dataset_loading(args.dataset, args.row_cap, args.verbose)))
    
    all_passed = True
    results = {}
    
    for check_name, check_func in checks:
        print(f"\n{'='*20} {check_name} {'='*20}")
        try:
            result = check_func()
            results[check_name] = result
            if result is False:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            results[check_name] = {"error": str(e)}
            all_passed = False
    
    # Summary
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 All pre-flight checks passed!")
        print("✅ Ready to run CV scripts")
        
        # Generate and display commands
        commands = generate_ready_commands(args.dataset, args.gene_cv_dir, args.loco_cv_dir)
        
        print(f"\n📋 Ready-to-run commands:")
        print(f"\n🧬 Gene-wise CV:")
        print(commands["gene_cv"])
        
        print(f"\n🧬 Chromosome-wise CV:")
        print(commands["loco_cv"])
        
        # Save commands to file for convenience
        commands_file = Path("cv_commands.sh")
        with open(commands_file, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Generated CV commands\n\n")
            f.write("# Gene-wise CV\n")
            f.write(commands["gene_cv"].replace(" \\", " \\") + "\n\n")
            f.write("# Chromosome-wise CV\n")
            f.write(commands["loco_cv"].replace(" \\", " \\") + "\n")
        
        print(f"\n💾 Commands saved to: {commands_file}")
        print(f"   Make executable with: chmod +x {commands_file}")
        
    else:
        print("❌ Some pre-flight checks failed")
        print("🔧 Please address the issues above before running CV scripts")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 