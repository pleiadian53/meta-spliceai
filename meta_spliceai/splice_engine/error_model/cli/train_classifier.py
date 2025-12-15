#!/usr/bin/env python
"""
Command-line interface for training error classifiers.

This is a wrapper around the error_model package that provides command-line functionality
for training error models using the workflow functions.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the root directory to the path
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from meta_spliceai.splice_engine.error_model import (
    process_error_model,
    process_all_error_models,
    verify_error_model_outputs,
    load_and_subsample_dataset
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train error classifiers to analyze splice site prediction errors."
    )
    
    # Define experiment group
    experiment_group = parser.add_argument_group("Experiment Configuration")
    experiment_group.add_argument(
        "--experiment",
        type=str,
        default="error_analysis",
        help="Experiment name for output directory organization"
    )
    
    # Define model selection group
    model_group = parser.add_argument_group("Model Selection")
    
    # Define modes for running the script
    model_selection = model_group.add_mutually_exclusive_group()
    model_selection.add_argument(
        "--all",
        action="store_true",
        help="Process all valid combinations of splice types and error models"
    )
    model_selection.add_argument(
        "--model",
        type=str,
        help="Shorthand for model specification (e.g., 'donor:fp_vs_tp')"
    )
    
    # Define individual model parameters
    model_group.add_argument(
        "--splice-type",
        type=str,
        choices=["donor", "acceptor", "any"],
        default="any",
        help="Splice type to process (default: any)"
    )
    model_group.add_argument(
        "--error-label",
        type=str,
        choices=["FP", "FN"],
        default="FP",
        help="Error class label (default: FP)"
    )
    model_group.add_argument(
        "--correct-label",
        type=str,
        choices=["TP", "TN"],
        default="TP",
        help="Correct class label (default: TP)"
    )
    
    # Define data sampling group
    data_group = parser.add_argument_group("Data Sampling")
    data_group.add_argument(
        "--sample-ratio",
        type=float,
        default=None,
        help="Fraction of dataset to sample (0.0-1.0) for faster processing"
    )
    data_group.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum number of samples to use"
    )
    data_group.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use"
    )
    
    # Define feature importance group
    feature_group = parser.add_argument_group("Feature Importance")
    feature_group.add_argument(
        "--importance-methods",
        type=str,
        default="shap,xgboost,mutual_info",
        help="Comma-separated list of feature importance methods"
    )
    feature_group.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top features to display in visualizations"
    )
    feature_group.add_argument(
        "--shap-local-top-k",
        type=int,
        default=20,
        help="Number of top features to consider per sample in SHAP analysis"
    )
    feature_group.add_argument(
        "--shap-global-top-k",
        type=int,
        default=50,
        help="Number of top features to consider globally in SHAP analysis"
    )
    feature_group.add_argument(
        "--shap-plot-top-k",
        type=int,
        default=25,
        help="Number of top features to show in SHAP visualization plots"
    )
    
    # Define visualization group
    visualization_group = parser.add_argument_group("Visualization")
    visualization_group.add_argument(
        "--feature-plot-type",
        type=str,
        choices=["box", "violin", "strip", "swarm"],
        default="violin",
        help="Type of plot for feature distributions"
    )
    visualization_group.add_argument(
        "--no-advanced-plots",
        action="store_true",
        help="Disable advanced feature distribution plots"
    )
    
    # Define model parameters group
    model_params_group = parser.add_argument_group("Model Parameters")
    model_params_group.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in XGBoost"
    )
    model_params_group.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth in XGBoost"
    )
    model_params_group.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for XGBoost"
    )
    model_params_group.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Subsample ratio for XGBoost"
    )
    model_params_group.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Column subsample ratio for XGBoost"
    )
    
    # Define general options
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (0=quiet, 1=normal, 2=detailed)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip combinations that have already been processed"
    )
    parser.add_argument(
        "--verify-outputs",
        action="store_true",
        help="Verify output files after processing"
    )
    
    return parser.parse_args()


def process_single_combination(args, experiment, splice_type, error_label, correct_label, sample_ratio, min_samples, max_samples, ml_params, viz_params, verbose, skip_existing):
    """
    Process a single combination of error model and splice type.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    experiment : str
        Experiment name
    splice_type : str
        Splice type
    error_label : str
        Error class label
    correct_label : str
        Correct class label
    sample_ratio : float
        Fraction of dataset to sample
    min_samples : int
        Minimum number of samples to use
    max_samples : int
        Maximum number of samples to use
    ml_params : dict
        Machine learning parameters
    viz_params : dict
        Visualization parameters
    verbose : int
        Verbosity level
    skip_existing : bool
        Skip existing combinations
    
    Returns
    -------
    dict or None
        Result set if successful, None otherwise
    """
    import inspect
    
    # Prepare model parameters
    model_params = {
        "n_estimators": ml_params.get("n_estimators", 100),
        "max_depth": ml_params.get("max_depth", 6),
        "learning_rate": ml_params.get("learning_rate", 0.1),
        "subsample": ml_params.get("subsample", 0.8),
        "colsample_bytree": ml_params.get("colsample_bytree", 0.8)
    }
    
    # Parse importance methods
    importance_methods = viz_params.get("importance_methods", "shap,xgboost,mutual_info").split(",")
    
    # Option 1: For faster iteration during development,
    # we can preload and subsample the dataset
    test_data = None
    if sample_ratio is not None and sample_ratio < 1.0:
        if verbose > 0:
            print(f"Preloading and subsampling data with ratio {sample_ratio}...")
        test_data = load_and_subsample_dataset(
            error_label=error_label,
            correct_label=correct_label,
            splice_type=splice_type,
            sampling_ratio=sample_ratio,
            min_samples=min_samples,
            max_samples=max_samples,
            verbose=verbose
        )
        if verbose > 0:
            print(f"Loaded subsampled dataset with {len(test_data)} samples")
    
    # Use the parameter namespace pattern to ensure flexibility and forward compatibility
    # Get signature of process_error_model to understand its parameters
    process_sig = inspect.signature(process_error_model)
    process_params = set(process_sig.parameters.keys())
    
    # Build parameter dictionary with all possible parameters
    params = {
        "error_label": error_label,
        "correct_label": correct_label,
        "splice_type": splice_type,
        "experiment": experiment,
        "importance_methods": importance_methods,
        "test_data": test_data,  # Pass pre-loaded data if available
        "sample_ratio": None if test_data is not None else sample_ratio,  # Don't double-sample
        "min_samples": min_samples,
        "max_samples": max_samples,
        "top_k": viz_params.get("top_k", 20),
        "use_advanced_feature_plots": not viz_params.get("skip_advanced_plots", False),
        "feature_plot_type": viz_params.get("feature_plot_type", "violin"),
        "shap_local_top_k": viz_params.get("shap_local_top_k", 20),
        "shap_global_top_k": viz_params.get("shap_global_top_k", 50),
        "shap_plot_top_k": viz_params.get("shap_plot_top_k", 25),
        "verbose": verbose,
        "model_params": model_params
    }
    
    # Filter params to only include those expected by process_error_model
    filtered_params = {k: v for k, v in params.items() if k in process_params}
    
    # Process using the workflow function with filtered parameters
    return process_error_model(**filtered_params)


def process_all_combinations(args, experiment, sample_ratio, min_samples, max_samples, ml_params, viz_params, verbose, skip_existing):
    """
    Process all combinations of splice types and error models.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    experiment : str
        Experiment name
    sample_ratio : float
        Fraction of dataset to sample
    min_samples : int
        Minimum number of samples to use
    max_samples : int
        Maximum number of samples to use
    ml_params : dict
        Machine learning parameters
    viz_params : dict
        Visualization parameters
    verbose : int
        Verbosity level
    skip_existing : bool
        Skip existing combinations
    
    Returns
    -------
    dict
        Results from processing all models
    """
    import inspect
    
    # Parse importance methods
    importance_methods = viz_params.get("importance_methods", "shap,xgboost,mutual_info").split(",")
    
    # Prepare model parameters
    model_params = {
        "n_estimators": ml_params.get("n_estimators", 100),
        "max_depth": ml_params.get("max_depth", 6),
        "learning_rate": ml_params.get("learning_rate", 0.1),
        "subsample": ml_params.get("subsample", 0.8),
        "colsample_bytree": ml_params.get("colsample_bytree", 0.8)
    }
    
    # Use the parameter namespace pattern for future-proof flexibility
    # Get signature of process_all_error_models
    process_sig = inspect.signature(process_all_error_models)
    process_params = set(process_sig.parameters.keys())
    
    # Build parameter dictionary with all possible parameters
    params = {
        "experiment": experiment,
        "splice_types": ["donor", "acceptor", "any"],
        "error_models": ["fp_vs_tp", "fn_vs_tp", "fn_vs_tn"],
        "importance_methods": importance_methods,
        "sample_ratio": sample_ratio,
        "min_samples": min_samples,
        "max_samples": max_samples,
        "top_k": viz_params.get("top_k", 20),
        "use_advanced_feature_plots": not viz_params.get("skip_advanced_plots", False),
        "feature_plot_type": viz_params.get("feature_plot_type", "violin"),
        "shap_local_top_k": viz_params.get("shap_local_top_k", 20),
        "shap_global_top_k": viz_params.get("shap_global_top_k", 50),
        "shap_plot_top_k": viz_params.get("shap_plot_top_k", 25),
        "verbose": verbose,
        "skip_biologically_meaningless": True,
        "verify_outputs": args.verify_outputs,
        "enable_check_existing": skip_existing,
        "model_params": model_params
    }
    
    # Filter params to only include those expected by process_all_error_models
    filtered_params = {k: v for k, v in params.items() if k in process_params}
    
    # Process all combinations
    results = process_all_error_models(**filtered_params)
    
    # Verify outputs if requested
    if args.verify_outputs and results:
        # Add experiment to path for verification
        verify_error_model_outputs(experiment)
    
    return results


def main():
    """Main entry point for the CLI."""
    args = parse_arguments()
    if args is None:
        return 2  # Error in argument parsing
    
    # Get experiment name and other common parameters
    experiment = args.experiment
    sample_ratio = args.sample_ratio
    min_samples = args.min_samples
    max_samples = args.max_samples
    verbose = args.verbose
    skip_existing = args.skip_existing
    verify_outputs = args.verify_outputs
    
    # Extract ML parameters from args to pass to process_error_model
    ml_params = {}
    for param in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']:
        if hasattr(args, param) and getattr(args, param) is not None:
            ml_params[param] = getattr(args, param)
    
    # Extract visualization parameters for feature importance
    viz_params = {}
    for param in ['importance_methods', 'top_k', 'shap_local_top_k', 
                  'shap_global_top_k', 'shap_plot_top_k', 'feature_plot_type']:
        if hasattr(args, param) and getattr(args, param) is not None:
            viz_params[param] = getattr(args, param)
    
    # Flag for generating advanced plots
    if hasattr(args, 'no_advanced_plots') and args.no_advanced_plots:
        viz_params['skip_advanced_plots'] = True
    
    # Default to single model mode using specified parameters or defaults
    if not args.all and not args.model:
        print(f"Processing single model with splice_type={args.splice_type}, "
              f"error_label={args.error_label}, correct_label={args.correct_label}")
        result = process_single_combination(
            args, experiment, 
            args.splice_type, args.error_label, args.correct_label, 
            sample_ratio, min_samples, max_samples, 
            ml_params, viz_params, verbose, skip_existing
        )
        if result is None:
            return 1  # Error in processing
        return 0  # Success
    
    # Parse model shorthand format if provided
    splice_type = args.splice_type
    error_label = args.error_label
    correct_label = args.correct_label
    
    # Handle model shorthand format if provided
    if args.model:
        try:
            splice_type, error_model = args.model.split(":")
            if error_model == "fp_vs_tp":
                error_label, correct_label = "FP", "TP"
            elif error_model == "fn_vs_tp":
                error_label, correct_label = "FN", "TP"
            elif error_model == "fp_vs_tn":
                error_label, correct_label = "FP", "TN"
            else:
                print(f"Unsupported error model: {error_model}")
                return 2
        except ValueError:
            print(f"Invalid format for --model: {args.model}")
            print("Expected format: 'splice_type:error_model' (e.g., 'donor:fp_vs_tp')")
            return 2
        
        # Process the specific combination
        result = process_single_combination(
            args, experiment, 
            splice_type, error_label, correct_label, 
            sample_ratio, min_samples, max_samples, 
            ml_params, viz_params, verbose, skip_existing
        )
        if result is None:
            return 1  # Error in processing
        return 0  # Success
    
    # Process all combinations
    if args.all:
        # Process all combinations of splice types and error models
        result = process_all_combinations(
            args, experiment, 
            sample_ratio, min_samples, max_samples, 
            ml_params, viz_params, verbose, skip_existing
        )
        
        if result is None:
            return 1  # Error in processing
        
        # Verify outputs if requested
        if verify_outputs:
            print("\nVerifying all outputs...")
            verify_error_model_outputs(experiment)
        
        return 0  # Success
    
    # Should never reach here
    return 2


if __name__ == "__main__":
    sys.exit(main())
