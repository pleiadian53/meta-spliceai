#!/usr/bin/env python
"""
New entry point for training error classification models using the refactored
modular components in the model_training package.

This script provides similar functionality to train_error_model.py but uses the new
modular design for better organization and maintainability. It can process all
combinations of splice types and error models at once, with optional subsampling.

Usage:
    # Process all combinations:
    python -m meta_spliceai.splice_engine.train_error_classifier --all

    # Process specific combination:
    python -m meta_spliceai.splice_engine.train_error_classifier --error-label FP --correct-label TP --splice-type donor
    
    # With subsampling for faster results:
    python -m meta_spliceai.splice_engine.train_error_classifier --all --sample-ratio 0.2
"""

import os
import sys
import argparse
from pathlib import Path
from itertools import product
from sklearn.model_selection import train_test_split

from .model_training.error_classifier import train_error_classifier, workflow_train_error_classifier
from .splice_error_analyzer import ErrorAnalyzer
from .model_evaluator import ModelEvaluationFileHandler
from .utils_doc import print_emphasized, print_section_separator, print_with_indent


def load_and_subsample_dataset(
    error_label,
    correct_label,
    splice_type=None,
    sampling_ratio=1.0,
    min_samples=100,
    max_samples=None,
    random_state=42,
    verbose=1
):
    """
    Load a dataset for error classification and optionally subsample it.
    
    Parameters
    ----------
    error_label : str
        Label for error class (e.g., "FP", "FN")
    correct_label : str
        Label for correct prediction class (e.g., "TP", "TN")
    splice_type : str, optional
        Type of splice site ("donor", "acceptor", or None for "any")
    sampling_ratio : float, default=1.0
        Fraction of dataset to sample (0.0-1.0)
    min_samples : int, default=100
        Minimum number of samples to return, regardless of ratio
    max_samples : int, default=None
        Maximum number of samples to return, regardless of ratio
        If None, no maximum limit is applied
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
        print_emphasized(f"Loading dataset: {error_label} vs {correct_label}" + 
                         (f" ({splice_type})" if splice_type else " (any)"))
        
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
        return None
    
    if df_full is None or len(df_full) == 0:
        if verbose > 0:
            print(f"No data found for {error_label} vs {correct_label}" + 
                 (f" ({splice_type})" if splice_type else " (any)"))
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
    
    # Get dataset info
    total_samples = len(df_full)
    label_col = 'label'  # Default label column
    
    if verbose > 0:
        print(f"Original dataset size: {total_samples:,} samples")
        
        # Show class distribution if possible
        if label_col in df_full.columns:
            class_counts = df_full[label_col].value_counts()
            print(f"Class distribution: {dict(class_counts)}")
    
    # If sampling ratio is 1.0 (or greater), return the full dataset
    if sampling_ratio >= 1.0 and max_samples is None:
        if verbose > 0:
            print("Using full dataset (no subsampling)")
        return df_full
    
    # Determine sample size based on ratio and bounds
    n_samples = int(total_samples * sampling_ratio)
    
    if max_samples is not None:
        n_samples = min(n_samples, max_samples)
    
    n_samples = max(min_samples, n_samples)
    n_samples = min(n_samples, total_samples)  # Can't sample more than we have
    
    if verbose > 0:
        print(f"Subsampling to {n_samples:,} samples ({n_samples/total_samples:.1%} of original)")
    
    # Perform stratified sampling if label column exists
    if label_col in df_full.columns:
        try:
            df_sampled, _ = train_test_split(
                df_full,
                train_size=n_samples,
                stratify=df_full[label_col],
                random_state=random_state
            )
            
            if verbose > 0:
                sampled_class_counts = df_sampled[label_col].value_counts()
                print(f"Sampled class distribution: {dict(sampled_class_counts)}")
        except Exception as e:
            print(f"Error in stratified sampling: {str(e)}")
            print("Falling back to random sampling")
            df_sampled = df_full.sample(n=n_samples, random_state=random_state)
    else:
        # Simple random sampling if no label column
        df_sampled = df_full.sample(n=n_samples, random_state=random_state)
    
    return df_sampled


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train error classifier models using the refactored modular components.'
    )
    
    # Required arguments group
    required_group = parser.add_argument_group('Required arguments (unless --all is specified)')
    
    # Top-level mode arguments as mutually exclusive
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    mode_group.add_argument(
        '--all',
        action='store_true',
        help='Process all combinations of splice types and error models'
    )
    
    mode_group.add_argument(
        '--workflow',
        action='store_true',
        help='Process all combinations using the built-in workflow function'
    )
    
    # Error type selection
    error_type_group = parser.add_mutually_exclusive_group()
    
    error_type_group.add_argument(
        '--pred-type', 
        type=str, 
        choices=['FP', 'FN'],  # Only error classes since this is being subsumed by --error-label
        help='Legacy parameter for backward compatibility. Used for specifying error class. Use --error-label instead.'
    )
    
    error_type_group.add_argument(
        '--error-label',
        type=str,
        choices=['FP', 'FN'],  # Only error classes
        help='Label for error cases (e.g., "FP" for False Positives, "FN" for False Negatives).'
    )
    
    error_type_group.add_argument(
        '--correct-label',
        type=str,
        choices=['TP', 'TN'],  # Only correct classes
        help='Label for correct prediction cases (e.g., "TP" for True Positives, "TN" for True Negatives).'
    )
    
    # Optional arguments
    parser.add_argument(
        '--splice-type',
        type=str,
        choices=['donor', 'acceptor', 'any'],
        help='Type of splice site to analyze. If not specified, "any" is used.'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results. If not specified, uses default location.'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default='hard_genes',
        help='Experiment name for workflow mode.'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        default='xgboost',
        choices=['xgboost', 'logistic', 'rf', 'svm'],
        help='Type of model to train.'
    )
    
    parser.add_argument(
        '--remove-strong-predictors',
        action='store_true',
        help='Remove strong predictor features (e.g., score) from training.'
    )
    
    parser.add_argument(
        '--strong-predictors',
        type=str,
        nargs='+',
        default=['score'],
        help='List of feature names to consider as strong predictors.'
    )
    
    # Subsampling options
    parser.add_argument(
        '--sample-ratio',
        type=float,
        default=1.0,
        help='Fraction of dataset to use (0.0-1.0). Default: 1.0 (full dataset)'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=100,
        help='Minimum number of samples to use, regardless of ratio.'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to use, regardless of ratio.'
    )
    
    # Feature importance options
    parser.add_argument(
        '--importance-methods',
        type=str,
        default='shap,xgboost,mutual_info',
        help='Comma-separated list of feature importance methods to use.'
    )
    
    parser.add_argument(
        '--shap-local-top-k',
        type=int,
        default=25,
        help='Number of top features to consider per sample in SHAP analysis.'
    )
    
    parser.add_argument(
        '--shap-global-top-k',
        type=int,
        default=50,
        help='Number of top features to consider globally in SHAP analysis.'
    )
    
    parser.add_argument(
        '--shap-plot-top-k',
        type=int,
        default=25,
        help='Number of top features to show in SHAP visualization plots.'
    )
    
    # Advanced feature plotting options
    parser.add_argument(
        '--use-advanced-feature-plots',
        action='store_false',  # Note: we use store_false to make the default True
        dest='use_advanced_feature_plots',
        default=True,  # Explicitly set default to True
        help='Generate advanced feature distribution plots (default: True). Use --no-advanced-feature-plots to disable.'
    )
    
    parser.add_argument(
        '--no-advanced-feature-plots',  # Add a complementary flag for clarity
        action='store_false',
        dest='use_advanced_feature_plots',
        help=argparse.SUPPRESS,  # Hide from help but provide the functionality
    )
    
    parser.add_argument(
        '--feature-plot-type',
        type=str,
        default='violin',
        choices=['violin', 'box', 'histplot'],
        help='Type of plot to use for advanced feature plots.'
    )
    
    # Model training parameters
    parser.add_argument(
        '--col-label',
        type=str,
        default='label',
        help='Name of the label column in the dataset.'
    )
    
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Number of cross-validation splits.'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='Number of top features to display in visualizations.'
    )
    
    parser.add_argument(
        '--top-k-motifs',
        type=int,
        default=20,
        help='Max number of motif features to use. If not specified, uses all.'
    )
    
    # XGBoost parameters
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help='Learning rate for XGBoost.'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Maximum depth of trees in XGBoost.'
    )
    
    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help='Number of trees (estimators) in XGBoost.'
    )
    
    parser.add_argument(
        '--subsample',
        type=float,
        default=0.8,
        help='Subsample ratio for XGBoost.'
    )
    
    parser.add_argument(
        '--colsample-bytree',
        type=float,
        default=0.8,
        help='Column subsample ratio for XGBoost.'
    )
    
    # Other parameters
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility.'
    )
    
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level.'
    )
    
    return parser.parse_args()


def process_single_combination(
    error_label,
    correct_label,
    splice_type,
    args,
    xgb_params
):
    """
    Process a single combination of error model and splice type.
    
    Parameters
    ----------
    error_label : str
        Label for error class (e.g., "FP", "FN")
    correct_label : str
        Label for correct prediction class (e.g., "TP", "TN")
    splice_type : str
        Type of splice site ("donor", "acceptor", or "any")
    args : argparse.Namespace
        Command-line arguments
    xgb_params : dict
        XGBoost parameters
        
    Returns
    -------
    dict or None
        Result set if successful, None otherwise
    """
    # Set up analyzer and output directory
    analyzer = ErrorAnalyzer(experiment=args.experiment)
    
    # Use custom output directory if specified, otherwise use analyzer's directory
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, splice_type, f"{error_label.lower()}_vs_{correct_label.lower()}", args.model_type)
    else:
        output_dir = analyzer.set_analysis_output_dir(
            error_label=error_label, 
            correct_label=correct_label, 
            splice_type=splice_type
        )
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset with optional subsampling
    dataset = load_and_subsample_dataset(
        error_label=error_label,
        correct_label=correct_label,
        splice_type=splice_type,
        sampling_ratio=args.sample_ratio,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        random_state=args.random_state,
        verbose=args.verbose
    )
    
    if dataset is None or len(dataset) == 0:
        print_emphasized(f"Skipping {splice_type}: {error_label} vs {correct_label} - No data available")
        return None
    
    # Parse importance methods
    importance_methods = args.importance_methods.split(',') if args.importance_methods else None
    
    print_emphasized(f"Training model for {splice_type}: {error_label} vs {correct_label}")
    print_with_indent(f"Dataset size: {len(dataset):,} samples", indent_level=1)
    print_with_indent(f"Output directory: {output_dir}", indent_level=1)
    
    try:
        # Run error classifier training
        model, result_set = train_error_classifier(
            error_label=error_label,
            correct_label=correct_label,
            splice_type=splice_type,
            remove_strong_predictors=args.remove_strong_predictors,
            strong_predictors=args.strong_predictors,
            model_type=args.model_type,
            test_data=dataset,
            col_label=args.col_label,
            n_splits=args.n_splits,
            top_k=args.top_k,
            top_k_motifs=args.top_k_motifs,
            output_dir=output_dir,
            random_state=args.random_state,
            importance_methods=importance_methods,
            shap_local_top_k=args.shap_local_top_k,
            shap_global_top_k=args.shap_global_top_k,
            shap_plot_top_k=args.shap_plot_top_k,
            use_advanced_feature_plots=args.use_advanced_feature_plots,
            feature_plot_type=args.feature_plot_type,
            verbose=args.verbose,
            **xgb_params
        )
        
        # Create status file
        status_file = os.path.join(output_dir, f"{splice_type}_{error_label}_vs_{correct_label}_done.txt")
        with open(status_file, 'w') as f:
            f.write(f"Completed processing {splice_type}: {error_label} vs {correct_label}")
        
        print_emphasized(f"Successfully processed {splice_type}: {error_label} vs {correct_label}")
        return result_set
    
    except Exception as e:
        print_emphasized(f"Error processing {splice_type}: {error_label} vs {correct_label}")
        print(f"Exception: {str(e)}")
        return None


def main():
    """Main entry point for training error classifier models."""
    args = parse_arguments()
    
    # Collect XGBoost parameters
    xgb_params = {
        'learning_rate': args.learning_rate,
        'max_depth': args.max_depth,
        'n_estimators': args.n_estimators,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
    }
    
    # Process based on mode
    if args.workflow:
        print_emphasized("Running full workflow with all combinations")
        
        # Convert XGBoost params to kwargs for workflow
        workflow_kwargs = {
            'model_type': args.model_type,
            'enable_check_existing': False,  # Always process all combinations
        }
        
        # Add feature importance parameters
        if args.importance_methods:
            workflow_kwargs['importance_methods'] = args.importance_methods.split(',')
        
        if args.shap_local_top_k:
            workflow_kwargs['shap_local_top_k'] = args.shap_local_top_k
            
        if args.shap_global_top_k:
            workflow_kwargs['shap_global_top_k'] = args.shap_global_top_k
            
        if args.shap_plot_top_k:
            workflow_kwargs['shap_plot_top_k'] = args.shap_plot_top_k
            
        if args.use_advanced_feature_plots:
            workflow_kwargs['use_advanced_feature_plots'] = True
            workflow_kwargs['feature_plot_type'] = args.feature_plot_type
        
        # Add XGBoost parameters
        for key, value in xgb_params.items():
            workflow_kwargs[key] = value
        
        # Run workflow
        workflow_train_error_classifier(
            experiment=args.experiment,
            test_mode=False,  # Use real data
            **workflow_kwargs
        )
        
    elif args.all:
        print_emphasized("Processing all valid combinations of splice types and error models")
        
        # Define possible labels and splice types
        error_labels = ["FP", "FN"]
        correct_labels = ["TP", "TN"]
        splice_types = ["donor", "acceptor", "any"]
        
        # Process all combinations, excluding FP vs TN
        results = {}
        for splice_type in splice_types:
            for error_label, correct_label in product(error_labels, correct_labels):
                if (error_label, correct_label) == ("FP", "TN"):
                    continue  # Skip unwanted combination - FP vs TN doesn't make sense
                
                key = f"{splice_type}_{error_label}_vs_{correct_label}"
                results[key] = process_single_combination(
                    error_label=error_label,
                    correct_label=correct_label,
                    splice_type=splice_type,
                    args=args,
                    xgb_params=xgb_params
                )
        
        # Print summary
        print_section_separator()
        print_emphasized("Summary of processing results:")
        for key, result in results.items():
            status = "✓ Success" if result is not None else "✗ Failed"
            print(f"{status}: {key}")
            
    else:
        # Process a single combination
        print_emphasized("Processing single combination")
        
        # Determine which error label to use (prefer error-label if specified)
        error_label = args.error_label if args.error_label is not None else args.pred_type
        
        # Check required parameters
        if not error_label:
            print("Error: Must specify --error-label or --pred-type")
            return 1
            
        if not args.correct_label:
            print("Error: Must specify --correct-label")
            return 1
        
        # Use 'any' for splice type if not specified
        splice_type = args.splice_type if args.splice_type else "any"
        
        # Process the combination
        result = process_single_combination(
            error_label=error_label,
            correct_label=args.correct_label,
            splice_type=splice_type,
            args=args,
            xgb_params=xgb_params
        )
        
        if result is None:
            print_emphasized("Processing failed")
            return 1
    
    print_section_separator()
    print_emphasized("All processing completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
