#!/usr/bin/env python
"""
Real data workflow test script for the error classifier.

This script tests the workflow_train_error_classifier function with real splicing datasets,
but applies stratified subsampling to make it run faster while maintaining class distributions.
It's designed for testing with realistic data without requiring a full-sized dataset.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import inspect
import matplotlib.pyplot as plt

# Ensure the meta_spliceai package is in the path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from meta_spliceai.splice_engine.model_training.error_classifier import (
    workflow_train_error_classifier,
    train_error_classifier
)
from meta_spliceai.splice_engine.splice_error_analyzer import ErrorAnalyzer
# from meta_spliceai.splice_engine.model_evaluation_file_handler import ModelEvaluationFileHandler
from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler

from meta_spliceai.splice_engine.utils_doc import print_emphasized, print_with_indent

from meta_spliceai.splice_engine.performance_analyzer import (
    plot_cv_roc_curve, 
    plot_cv_pr_curve
)


def load_and_subsample_dataset(
    error_label="FP", 
    correct_label="TP", 
    splice_type="donor",
    sampling_ratio=0.1,
    min_samples=100,
    max_samples=1000,
    random_state=42,
    verbose=1
):
    """
    Load a real dataset and apply stratified subsampling to reduce its size 
    while maintaining class distribution.
    
    Parameters
    ----------
    error_label : str, default="FP"
        Label for error class (e.g., "FP", "FN")
    correct_label : str, default="TP"
        Label for correct prediction class (e.g., "TP", "TN")
    splice_type : str, default="donor"
        Type of splice site ("donor", "acceptor", or "any")
    sampling_ratio : float, default=0.1
        Fraction of dataset to sample (0.0-1.0)
    min_samples : int, default=100
        Minimum number of samples to return, regardless of ratio
    max_samples : int, default=1000
        Maximum number of samples to return, regardless of ratio
    random_state : int, default=42
        Random seed for reproducible sampling
    verbose : int, default=1
        Verbosity level
        
    Returns
    -------
    pd.DataFrame
        Subsampled dataset with balanced class representation
    """
    if verbose > 0:
        print_emphasized(f"Loading real dataset: {error_label} vs {correct_label} ({splice_type})")
        
    # Load the real dataset using ModelEvaluationFileHandler
    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')
    
    try:
        df_full = mefd.load_featurized_dataset(
            aggregated=True, 
            error_label=error_label, 
            correct_label=correct_label,
            splice_type=splice_type
        )
    except Exception as e:
        if verbose > 0:
            print(f"Error loading dataset: {str(e)}")
            print("This may be due to missing data files or incorrect paths.")
            print("Returning None - workflow will need to handle this.")
        return None
    
    if df_full is None or len(df_full) == 0:
        if verbose > 0:
            print(f"No data found for {error_label} vs {correct_label} ({splice_type})")
        return None
    
    # Convert Polars DataFrame to pandas if needed
    if not isinstance(df_full, pd.DataFrame):
        if verbose > 0:
            print("Converting Polars DataFrame to pandas DataFrame")
        try:
            df_full = df_full.to_pandas()
        except Exception as e:
            if verbose > 0:
                print(f"Error converting to pandas: {str(e)}")
                print("Trying alternative conversion method...")
            try:
                # Alternative method
                import polars as pl
                if isinstance(df_full, pl.DataFrame):
                    df_full = df_full.to_pandas()
                else:
                    print("Unknown DataFrame type, cannot convert")
                    return None
            except Exception as e2:
                print(f"Failed to convert DataFrame: {str(e2)}")
                return None
    
    # Get original dataset info
    label_col = 'label'  # Default label column
    if label_col not in df_full.columns:
        # Try to identify the label column
        potential_label_cols = [col for col in df_full.columns if 'label' in col.lower()]
        if potential_label_cols:
            label_col = potential_label_cols[0]
        else:
            if verbose > 0:
                print("Label column not found. Cannot perform stratified sampling.")
            # Return a simple random sample as fallback
            n_samples = max(min(int(len(df_full) * sampling_ratio), max_samples), min_samples)
            return df_full.sample(n=n_samples, random_state=random_state)
    
    if verbose > 0:
        print(f"Original dataset size: {len(df_full)} samples")
        class_counts = df_full[label_col].value_counts()
        print(f"Class distribution: {dict(class_counts)}")
    
    # Calculate sampling size with bounds
    n_samples = int(len(df_full) * sampling_ratio)
    n_samples = max(min(n_samples, max_samples), min_samples)
    n_samples = min(n_samples, len(df_full))  # Can't sample more than we have
    
    # Perform stratified sampling
    df_sampled, _ = train_test_split(
        df_full,
        train_size=n_samples,
        stratify=df_full[label_col],
        random_state=random_state
    )
    
    if verbose > 0:
        print(f"Sampled dataset size: {len(df_sampled)} samples")
        sampled_class_counts = df_sampled[label_col].value_counts()
        print(f"Sampled class distribution: {dict(sampled_class_counts)}")
        
        # Calculate and display class ratios to verify stratification
        original_ratio = class_counts.iloc[0] / class_counts.iloc[1] if len(class_counts) > 1 else 0
        sampled_ratio = sampled_class_counts.iloc[0] / sampled_class_counts.iloc[1] if len(sampled_class_counts) > 1 else 0
        print(f"Original class ratio: {original_ratio:.2f}")
        print(f"Sampled class ratio: {sampled_ratio:.2f}")
    
    return df_sampled


def verify_outputs(experiment_dir, splice_types=None, error_models=None):
    """
    Verify that expected output files exist with correct naming conventions.
    
    Parameters
    ----------
    experiment_dir : str
        Full path to the experiment directory containing outputs
    splice_types : list, default=None
        List of splice types to check, if None checks all ["donor", "acceptor", "any"]
    error_models : list, default=None
        List of error models to check, if None checks all ["fp_vs_tp", "fn_vs_tp", "fn_vs_tn"]
        
    Returns
    -------
    bool
        True if any expected files exist, False otherwise
    """
    if splice_types is None:
        splice_types = ["donor", "acceptor", "any"]
    
    if error_models is None:
        error_models = ["fp_vs_tp", "fn_vs_tp", "fn_vs_tn"]
    
    # Build a list of expected status files
    expected_files = []
    for splice_type in splice_types:
        for error_model in error_models:
            # Skip FP vs TN as it's not biologically meaningful
            if error_model == "fp_vs_tn":
                continue
                
            # Convert model name format
            if error_model == "fp_vs_tp":
                error_label, correct_label = "FP", "TP"
            elif error_model == "fn_vs_tp":
                error_label, correct_label = "FN", "TP"
            elif error_model == "fn_vs_tn":
                error_label, correct_label = "FN", "TN"
            
            # Expected status file - use lowercase for directory names
            expected_file = os.path.join(
                experiment_dir, 
                splice_type,  # Keep original case for splice types
                error_model,  # This is already lowercase: fp_vs_tp, fn_vs_tp, fn_vs_tn
                "xgboost",
                f"{splice_type}_{error_label}_vs_{correct_label}_done.txt"
            )
            expected_files.append(expected_file)
    
    # Check if files exist
    found_files = [f for f in expected_files if os.path.exists(f)]
    
    if not found_files:
        print("\nNo expected output files were found.")
        return False
    else:
        print(f"\nFound {len(found_files)} out of {len(expected_files)} expected output files.")
        print("Files found:")
        for f in found_files:
            print(f"  • {os.path.basename(f)}")
        
        # Show missing files if any
        if len(found_files) < len(expected_files):
            missing_files = [f for f in expected_files if f not in found_files]
            print("\nFiles missing:")
            
            # Group missing files by splice type and error model for better readability
            missing_by_group = {}
            for f in missing_files:
                # Extract splice type and error model from path
                parts = f.split(os.sep)
                if len(parts) >= 4:
                    splice_type = parts[-4]  # Splice type directory
                    error_model = parts[-3]  # Error model directory (already lowercase)
                    
                    key = f"{splice_type}: {error_model}"
                    if key not in missing_by_group:
                        missing_by_group[key] = []
                    missing_by_group[key].append(os.path.basename(f))
            
            # Print missing files by group
            for group, files in missing_by_group.items():
                print(f"  Missing in {group}:")
                for f in files:
                    print(f"    - {f}")
            
            print("\nPossible reasons for missing files:")
            print("  • Dataset not found for this combination")
            print("  • Error during processing of this combination")
            print("  • Check the log output above for specific errors")
        
        return True


def main():
    """Main test function using real data with subsampling"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Test the error classifier workflow with real data and subsampling"
    )
    parser.add_argument(
        "--sampling-ratio", 
        type=float, 
        default=0.1, 
        help="Fraction of dataset to sample (0.0-1.0)"
    )
    parser.add_argument(
        "--min-samples", 
        type=int, 
        default=100, 
        help="Minimum number of samples"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=1000, 
        help="Maximum number of samples"
    )
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="real_data_test", 
        help="Experiment name for output directories"
    )
    parser.add_argument(
        "--splice-type", 
        type=str, 
        choices=["donor", "acceptor", "any", "all"], 
        default="all",
        help="Specific splice type to test, or 'all' for all types"
    )
    parser.add_argument(
        "--error-model", 
        type=str, 
        choices=["fp_vs_tp", "fn_vs_tp", "fn_vs_tn", "all"], 
        default="all",
        help="Specific error model to test, or 'all' for all models"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top features to display in visualizations"
    )
    parser.add_argument(
        "--importance-methods",
        type=str,
        default="shap,xgboost,mutual_info",
        help="Comma-separated list of feature importance methods"
    )
    parser.add_argument(
        "--shap-local-top-k",
        type=int,
        default=10,
        help="Number of top features to consider per sample in SHAP analysis"
    )
    parser.add_argument(
        "--shap-global-top-k",
        type=int,
        default=20,
        help="Number of top features to consider globally in SHAP analysis"
    )
    parser.add_argument(
        "--shap-plot-top-k",
        type=int,
        default=20,
        help="Number of top features to show in SHAP visualization plots"
    )
    
    args = parser.parse_args()
    
    # Add debug logging for args
    print(f"Debug - args.importance_methods: {args.importance_methods} (type: {type(args.importance_methods)})")
    
    # Make sure importance_methods is always a list
    importance_methods = args.importance_methods
    if isinstance(importance_methods, str):
        importance_methods = importance_methods.split(',')
    print(f"Debug - processed importance_methods: {importance_methods} (type: {type(importance_methods)})")
    
    # Define splice types and error models to test
    splice_types = ["donor", "acceptor", "any"] if args.splice_type == "all" else [args.splice_type]
    
    error_models = []
    if args.error_model == "all":
        error_models = ["fp_vs_tp", "fn_vs_tp", "fn_vs_tn"]
    else:
        error_models.append(args.error_model)
    
    # Create dictionary mapping from error model names to label pairs
    error_label_map = {
        "fp_vs_tp": ("FP", "TP"),
        "fn_vs_tp": ("FN", "TP"),
        "fn_vs_tn": ("FN", "TN")
    }

    model_type = "xgboost"
    
    # Create output directories for experiment
    experiment_name = args.experiment
    analyzer = ErrorAnalyzer(experiment=experiment_name)
    experiment_dir = os.path.join(analyzer.analysis_dir, experiment_name)
    
    print_emphasized(f"Starting real data workflow test")
    print(f"Experiment name: {experiment_name}")
    print(f"Output directory: {experiment_dir}")
    print(f"Sampling ratio: {args.sampling_ratio} (min: {args.min_samples}, max: {args.max_samples})")
    print(f"Splice types: {splice_types}")
    print(f"Error models: {error_models}")
    print(f"Feature importance methods: {importance_methods}")
    print(f"SHAP parameters: local_top_k={args.shap_local_top_k}, global_top_k={args.shap_global_top_k}, plot_top_k={args.shap_plot_top_k}")
    
    # Sample datasets for each model
    sampled_datasets = {}
    
    for splice_type in splice_types:
        for error_model in error_models:
            error_label, correct_label = error_label_map[error_model]
            
            # Load and subsample the dataset
            sampled_df = load_and_subsample_dataset(
                error_label=error_label,
                correct_label=correct_label,
                splice_type=splice_type,
                sampling_ratio=args.sampling_ratio,
                min_samples=args.min_samples,
                max_samples=args.max_samples,
                verbose=1
            )
            
            if sampled_df is not None:
                key = f"{splice_type}_{error_model}"
                sampled_datasets[key] = sampled_df
    
    # Proceed if we have at least one dataset
    if not sampled_datasets:
        print_emphasized("No datasets could be loaded. Please check your data paths.")
        return 1
    
    # Run workflow for each dataset
    for splice_type in splice_types:
        for error_model in error_models:
            key = f"{splice_type}_{error_model}"
            if key not in sampled_datasets:
                continue
                
            error_label, correct_label = error_label_map[error_model]
            
            print_emphasized(f"Running workflow for {splice_type}: {error_label} vs {correct_label}")
            
            # Set output directory specifically for this combination
            output_dir = os.path.join(
                experiment_dir, 
                splice_type,  # Keep original case for splice types 
                error_model,  # Use lowercase error_model (e.g., "fp_vs_tp")
                model_type   # Use lowercase model_type (e.g., "xgboost")
            )
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                # Make sure importance_methods stays a list here
                if importance_methods is None:
                    importance_methods = ["shap", "xgboost", "mutual_info"]
                elif isinstance(importance_methods, str):
                    importance_methods = importance_methods.split(',')
                
                # Configure parameters for train_error_classifier
                train_params = {
                    "output_dir": output_dir,
                    "top_k": args.top_k,
                    "importance_methods": importance_methods,  # Already ensured to be a list
                    "shap_local_top_k": args.shap_local_top_k,
                    "shap_global_top_k": args.shap_global_top_k,
                    "shap_plot_top_k": args.shap_plot_top_k,
                    "test_data": sampled_datasets[key],
                    "use_advanced_feature_plots": True,
                    "feature_plot_type": "box",  # Options: "box", "strip", "violin"
                    "feature_plot_top_k_motifs": args.top_k,
                    "verbose": 1
                }
                
                print(f"Debug - Before calling train_error_classifier: importance_methods={importance_methods}, type={type(importance_methods)}")
                
                # Train the classifier
                model, result_set = train_error_classifier(
                    error_label=error_label,
                    correct_label=correct_label,
                    splice_type=splice_type,
                    **train_params
                )
                
                # Run additional analyses that would be in the workflow
                
                # 1. Cross-validation performance          
                try:
                    # Define output paths for ROC and PR curves
                    roc_output_path = os.path.join(output_dir, f"{splice_type}_{error_label.lower()}_vs_{correct_label.lower()}_roc_curve.png")
                    pr_output_path = os.path.join(output_dir, f"{splice_type}_{error_label.lower()}_vs_{correct_label.lower()}_pr_curve.png")
                    
                    # Process data through to_xy to handle categorical features and metadata columns
                    from meta_spliceai.mllib.model_trainer import to_xy
                    
                    print("[info] Processing data for performance curves using to_xy...")
                    # Use to_xy to convert raw DataFrame into ML-ready X and y
                    X_processed, y_processed = to_xy(
                        sampled_datasets[key],
                        dummify=True,  # One-hot encode categorical variables
                        drop_first=False,
                        verbose=1
                    )
                    
                    # Generate ROC curve with properly processed data
                    plot_cv_roc_curve(
                        model, 
                        X_processed,  # Use properly processed features 
                        y_processed,  # Use properly processed labels
                        cv_splits=3,  # Use cv_splits instead of n_folds
                        random_state=42,
                        output_path=roc_output_path,
                        verbose=1
                    )
                    
                    # Generate PR curve with properly processed data
                    plot_cv_pr_curve(
                        model, 
                        X_processed,  # Use properly processed features
                        y_processed,  # Use properly processed labels
                        cv_splits=3,  # Use cv_splits instead of n_folds
                        random_state=42,
                        output_path=pr_output_path,
                        verbose=1
                    )
                    
                except Exception as e:
                    print(f"Warning: Could not generate CV performance curves: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # Extract feature importance results that have already been calculated by train_error_classifier
                # instead of recalculating them redundantly
                print_emphasized("[info] Processing feature importance results from train_error_classifier...")
                
                # Check if we have feature importance results in the result_set
                for method in importance_methods:
                    result_key = f'importance_{method}'
                    if result_key in result_set:
                        # Feature importance was already calculated by train_error_classifier
                        print(f"[info] Using pre-calculated {method} feature importance from result_set")
                        
                        # Create output paths consistent with the rest of the workflow
                        if method == "mutual_info":
                            output_file = os.path.join(output_dir, f"{splice_type}_{error_label.lower()}_vs_{correct_label.lower()}_mutual_info.csv")
                            if not os.path.exists(output_file):  # Only save if it doesn't already exist
                                result_set[result_key]['all'].to_csv(output_file, index=False)
                                print(f"[info] Saved mutual information results to {output_file}")
                        
                        elif method == "hypothesis":
                            output_file = os.path.join(output_dir, f"{splice_type}_{error_label.lower()}_vs_{correct_label.lower()}_hypothesis_test.csv")
                            if not os.path.exists(output_file):  # Only save if it doesn't already exist
                                result_set[result_key]['all'].to_csv(output_file, index=False)
                                print(f"[info] Saved hypothesis testing results to {output_file}")
                    else:
                        print(f"[warning] {method} feature importance not found in result_set")
                
                # Create status file
                status_file = os.path.join(
                    output_dir, 
                    f"{splice_type}_{error_label}_vs_{correct_label}_done.txt"
                )
                with open(status_file, 'w') as f:
                    f.write(f"Completed processing {splice_type}: {error_label} vs {correct_label}")
                    
                print(f"Successfully processed {splice_type}: {error_label} vs {correct_label}")
            
            except Exception as e:
                print(f"Error processing {splice_type}: {error_label} vs {correct_label}")
                print(f"Exception: {str(e)}")
                continue
    
    # Verify outputs
    success = verify_outputs(
        experiment_dir, 
        splice_types=splice_types,
        error_models=error_models
    )
    
    if success:
        print_emphasized("Test completed successfully!")
        return 0
    else:
        print_emphasized("Test completed with issues. Check the output logs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
