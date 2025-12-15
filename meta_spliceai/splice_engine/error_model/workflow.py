"""
Workflow module for error model training and evaluation.

This module provides high-level workflows for training, evaluating, and analyzing
error models across different splice types and error categories.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from sklearn.model_selection import train_test_split

# Import from existing functionality
from ..model_training.error_classifier import train_error_classifier
from ..splice_error_analyzer import ErrorAnalyzer
from ..model_evaluator import ModelEvaluationFileHandler


def load_and_subsample_dataset(
    error_label="FP", 
    correct_label="TP", 
    splice_type="donor",
    sampling_ratio=0.1,
    min_samples=100,
    max_samples=5000,
    input_dir=None,
    separator="\t",
    random_state=42,
    verbose=1
):
    """
    Load a dataset and apply stratified subsampling to reduce its size 
    while maintaining class distribution.
    
    Parameters
    ----------
    error_label : str, default="FP"
        Label for error class (e.g., "FP", "FN")
    correct_label : str, default="TP"
        Label for correct prediction class (e.g., "TP", "TN")
    splice_type : str, default="donor"
        Type of splice site ("donor", "acceptor", or None)
    sampling_ratio : float, default=0.1
        Fraction of dataset to sample (0.0-1.0)
    min_samples : int, default=100
        Minimum number of samples to keep per class
    max_samples : int, default=5000
        Maximum total samples to use
    input_dir : str, optional
        Custom input directory. If None, uses ErrorAnalyzer.eval_dir
    separator : str, default="\t"
        File separator for datasets
    random_state : int, default=42
        Random seed for reproducibility
    verbose : int, default=1
        Verbosity level
        
    Returns
    -------
    pd.DataFrame
        Subsampled dataset with stratified class distribution
    """
    if verbose > 0:
        print(f"Loading data for {error_label} vs {correct_label} ({splice_type})...")
    
    # Use ModelEvaluationFileHandler to load the dataset
    if input_dir is None:
        input_dir = ErrorAnalyzer.eval_dir
        
    mefd = ModelEvaluationFileHandler(input_dir, separator=separator)
    full_dataset = mefd.load_featurized_dataset(
        aggregated=True,
        error_label=error_label,
        correct_label=correct_label,
        splice_type=splice_type
    )
    
    if verbose > 0:
        print(f"Loaded full dataset: {full_dataset.shape}")
    
    # Apply stratified sampling
    subsampled_df = apply_stratified_sampling(
        full_dataset,
        sampling_ratio=sampling_ratio,
        min_samples=min_samples,
        max_samples=max_samples,
        random_state=random_state,
        verbose=verbose
    )
    
    return subsampled_df


def apply_stratified_sampling(
    df: pd.DataFrame,
    sampling_ratio: Optional[float] = 0.1,
    min_samples: int = 100,
    max_samples: int = 1000,
    label_col: str = 'label',
    random_state: int = 42,
    verbose: int = 1
) -> pd.DataFrame:
    """
    Apply stratified subsampling to reduce dataset size while maintaining class distribution.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset to sample from
    sampling_ratio : float, default=0.1
        Fraction of dataset to sample (0.0-1.0)
    min_samples : int, default=100
        Minimum number of samples to return, regardless of ratio
    max_samples : int, default=1000
        Maximum number of samples to return, regardless of ratio
    label_col : str, default='label'
        Column name for class labels
    random_state : int, default=42
        Random seed for reproducible sampling
    verbose : int, default=1
        Verbosity level
        
    Returns
    -------
    pd.DataFrame
        Subsampled dataset with balanced class representation
    """
    if df is None or len(df) == 0:
        if verbose > 0:
            print(f"No data provided for sampling")
        return df
    
    # Convert Polars DataFrame to pandas if needed
    if not isinstance(df, pd.DataFrame):
        if verbose > 0:
            print("Converting Polars DataFrame to pandas DataFrame")
        try:
            import polars as pl
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()
            else:
                print("Unknown DataFrame type, cannot convert")
                return df
        except Exception as e:
            print(f"Failed to convert DataFrame: {str(e)}")
            return df
    
    # Get original dataset info
    if label_col not in df.columns:
        # Try to identify the label column
        potential_label_cols = [col for col in df.columns if 'label' in col.lower()]
        if potential_label_cols:
            label_col = potential_label_cols[0]
        else:
            if verbose > 0:
                print("Label column not found. Cannot perform stratified sampling.")
            # Return a simple random sample as fallback
            n_samples = max(min(int(len(df) * sampling_ratio), max_samples), min_samples)
            return df.sample(n=n_samples, random_state=random_state)
    
    if verbose > 0:
        print(f"Original dataset size: {len(df)} samples")
        class_counts = df[label_col].value_counts()
        print(f"Class distribution: {dict(class_counts)}")
    
    # Calculate sampling size with bounds
    n_samples = int(len(df) * sampling_ratio) if sampling_ratio is not None else max_samples
    n_samples = max(min(n_samples, max_samples), min_samples)
    n_samples = min(n_samples, len(df))  # Can't sample more than we have
    
    # Short circuit: if we're sampling the entire dataset, return it directly
    if n_samples >= len(df):
        if verbose > 0:
            print(f"Requested sample size {n_samples} is >= dataset size {len(df)}. Returning full dataset.")
        return df
    
    # Check if we have a single class dataset
    unique_labels = df[label_col].unique()
    if len(unique_labels) <= 1:
        if verbose > 0:
            print(f"Dataset contains only one class ({unique_labels[0]}). Performing random sampling instead of stratified sampling.")
        return df.sample(n=n_samples, random_state=random_state)
    
    # Perform stratified sampling
    df_sampled, _ = train_test_split(
        df,
        train_size=n_samples,
        stratify=df[label_col],
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


def process_error_model(
    error_label: str,
    correct_label: str,
    splice_type: str,
    experiment: str = "error_model_analysis",
    output_dir: Optional[str] = None,
    test_data: Optional[pd.DataFrame] = None,
    sample_ratio: Optional[float] = None,
    min_samples: int = 100,
    max_samples: int = 1000,
    importance_methods: Optional[List[str]] = None,
    top_k: int = 20,
    use_advanced_feature_plots: bool = True,
    feature_plot_type: str = "box",
    plot_feature_distributions: bool = True,
    feature_plot_cols: int = 3,
    feature_plot_top_k_motifs: int = 20,
    plot_motif_distribution: bool = True,
    verbose: int = 1,
    model_type: str = "xgboost",
    **kwargs
) -> Dict:
    """
    Process a single error model for a specific splice type and error category.
    
    Parameters
    ----------
    error_label : str
        Label for error class (e.g., "FP", "FN")
    correct_label : str
        Label for correct prediction class (e.g., "TP", "TN")
    splice_type : str
        Type of splice site ("donor", "acceptor", or "any")
    experiment : str, default="error_model_analysis"
        Experiment name for organizing outputs
    output_dir : str, optional
        Custom output directory. If None, uses the analyzer's default directory.
    test_data : pd.DataFrame, optional
        Test dataset to use for evaluation
    sample_ratio : float, optional
        Ratio of samples to use (for faster development/testing)
    min_samples : int, default=100
        Minimum number of samples to use
    max_samples : int, default=1000
        Maximum number of samples to use
    importance_methods : list, default=None
        Feature importance methods to use
    top_k : int, default=20
        Number of top features to display in visualizations
    use_advanced_feature_plots : bool, default=True
        Whether to use advanced feature distribution plots
    feature_plot_type : str, default="box"
        Type of plot for feature distributions (e.g., "violin", "box", "swarm")
    plot_feature_distributions : bool, default=True
        Whether to plot feature distributions
    feature_plot_cols : int, default=3
        Number of columns for feature distribution plots
    feature_plot_top_k_motifs : int, default=20
        Number of top motifs to show in feature distribution plots
    plot_motif_distribution : bool, default=True
        Whether to plot motif distribution
    verbose : int, default=1
        Verbosity level
    model_type : str, default="xgboost"
        Type of model to use
    **kwargs : dict
        Additional keyword arguments passed to train_error_classifier
        
    Returns
    -------
    dict
        Results from the training and analysis process
    """
    # Initialize the classifier using ErrorClassifier class
    from .classifier import ErrorClassifier
    
    # Initialize with basic parameters
    classifier = ErrorClassifier(
        error_label=error_label,
        correct_label=correct_label,
        splice_type=splice_type,
        experiment=experiment,
        model_type=model_type,
        output_dir=output_dir,
        verbose=verbose
    )
    
    # Load data - either from the provided test data or from the analyzer
    # This handles sampling internally
    if test_data is not None:
        classifier.load_data(
            custom_data=test_data,
            sample_ratio=sample_ratio,
            max_samples=max_samples
        )
    else:
        # If no test_data provided, load_data will use the analyzer
        classifier.load_data(
            sample_ratio=sample_ratio,
            max_samples=max_samples
        )
    
    # Set up training parameters
    train_kwargs = {
        'importance_methods': importance_methods if importance_methods is not None else ["shap", "xgboost", "mutual_info"],
        'top_k': top_k,
        'use_advanced_feature_plots': use_advanced_feature_plots,
        'feature_plot_type': feature_plot_type,
        'plot_feature_distributions': plot_feature_distributions,
        'feature_plot_cols': feature_plot_cols,
        'feature_plot_top_k_motifs': feature_plot_top_k_motifs,
        'plot_motif_distribution': plot_motif_distribution
    }
    
    # Add additional kwargs
    train_kwargs.update(kwargs)
    
    # Train the classifier with all parameters
    model, results = classifier.train(**train_kwargs)
    
    # Create a status file
    status_file = os.path.join(
        classifier.output_dir, 
        f"{splice_type}_{error_label}_vs_{correct_label}_done.txt"
    )
    
    with open(status_file, 'w') as f:
        f.write(f"Completed processing {splice_type}: {error_label} vs {correct_label}")
    
    if verbose > 0:
        print(f"Successfully processed {splice_type}: {error_label} vs {correct_label}")
    
    return {
        "model": model,
        "results": results,
        "output_dir": classifier.output_dir,
        "status_file": status_file
    }


def process_all_error_models(
    experiment: str = "error_analysis",
    splice_types: List[str] = ["donor", "acceptor", "any"],
    error_models: List[str] = ["fp_vs_tp", "fn_vs_tp", "fn_vs_tn"],
    skip_biologically_meaningless: bool = True,
    sample_ratio: Optional[float] = None,
    min_samples: int = 100,
    max_samples: Optional[int] = None,
    importance_methods: List[str] = ["shap", "xgboost", "mutual_info"],
    top_k: int = 20,
    use_advanced_feature_plots: bool = True,
    feature_plot_type: str = "box",
    shap_local_top_k: int = 10,
    shap_global_top_k: int = 20,
    shap_plot_top_k: int = 20,
    verify_outputs: bool = True,
    verbose: int = 1,
    model_type: str = "xgboost",
    **kwargs
) -> Dict:
    """
    Process all error models across specified splice types.
    
    Parameters
    ----------
    experiment : str, default="error_analysis"
        Experiment name for organizing outputs
    splice_types : list, default=["donor", "acceptor", "any"]
        List of splice types to process
    error_models : list, default=["fp_vs_tp", "fn_vs_tp", "fn_vs_tn"]
        List of error models to train
    skip_biologically_meaningless : bool, default=True
        Whether to skip biologically meaningless combinations (e.g., fp_vs_tn)
    sample_ratio : float, optional
        Ratio of samples to use (for faster development/testing)
    min_samples : int, default=100
        Minimum number of samples to use
    max_samples : int, optional
        Maximum number of samples to use
    importance_methods : list, default=["shap", "xgboost", "mutual_info"]
        Feature importance methods to use
    top_k : int, default=20
        Number of top features to display in visualizations
    use_advanced_feature_plots : bool, default=True
        Whether to use advanced feature distribution plots
    feature_plot_type : str, default="box"
        Type of plot for feature distributions (e.g., "violin", "box", "swarm")
    shap_local_top_k : int, default=10
        Number of top features to consider per sample in SHAP analysis
    shap_global_top_k : int, default=20
        Number of top features to consider globally in SHAP analysis
    shap_plot_top_k : int, default=20
        Number of top features to show in SHAP visualization plots
    verify_outputs : bool, default=True
        Whether to verify outputs after processing all models
    verbose : int, default=1
        Verbosity level
    model_type : str, default="xgboost"
        Type of model to use
    **kwargs : dict
        Additional keyword arguments passed to train_error_classifier
        
    Returns
    -------
    dict
        Mapping of splice_type -> error_model -> results
    """
    # Set up analyzer for directory creation
    analyzer = ErrorAnalyzer(experiment=experiment, model_type=model_type)
    experiment_dir = os.path.join(analyzer.analysis_dir, experiment)
    
    # Error label mapping
    error_label_map = {
        "fp_vs_tp": ("FP", "TP"),
        "fn_vs_tp": ("FN", "TP"),
        "fn_vs_tn": ("FN", "TN"),
        # "fp_vs_tn": ("FP", "TN")  # Unused 
    }
    
    # Validate error models
    for model in error_models:
        if model not in error_label_map:
            raise ValueError(f"Unsupported error model: {model}. Must be one of: {list(error_label_map.keys())}")
    
    # Keep track of results
    results = {}
    succeeded = 0
    failed = 0
    skipped = 0
    
    # Process all combinations
    for splice_type in splice_types:
        results[splice_type] = {}
        
        for error_model in error_models:
            # Skip biologically meaningless combinations if requested
            if skip_biologically_meaningless and error_model == "fp_vs_tn":
                if verbose > 0:
                    print(f"Skipping biologically meaningless combination: {splice_type} - {error_model}")
                skipped += 1
                continue
            
            # Map lowercase model names to actual labels
            error_label, correct_label = error_label_map[error_model]
            
            # Set output directory specifically for this combination
            output_dir = os.path.join(
                experiment_dir, 
                splice_type,
                error_model,
                "xgboost"
            )
            
            # Check if this combination has already been processed
            status_file = os.path.join(
                output_dir, 
                f"{splice_type}_{error_label}_vs_{correct_label}_done.txt"
            )
            
            if os.path.exists(status_file) and kwargs.get('enable_check_existing', False):
                if verbose > 0:
                    print(f"Skipping already processed combination: {splice_type} - {error_model}")
                skipped += 1
                continue
            
            try:
                if verbose > 0:
                    print(f"Processing {splice_type}: {error_label} vs {correct_label}")
                
                # Process this specific error model
                model_result = process_error_model(
                    error_label=error_label,
                    correct_label=correct_label,
                    splice_type=splice_type,
                    output_dir=output_dir,
                    experiment=experiment,
                    importance_methods=importance_methods,
                    sample_ratio=sample_ratio,
                    min_samples=min_samples,
                    max_samples=max_samples,
                    top_k=top_k,
                    use_advanced_feature_plots=use_advanced_feature_plots,
                    feature_plot_type=feature_plot_type,
                    shap_local_top_k=shap_local_top_k,
                    shap_global_top_k=shap_global_top_k,
                    shap_plot_top_k=shap_plot_top_k,
                    verbose=verbose,
                    model_type=model_type,  
                    **kwargs
                )
                
                # Store results for this combination
                results[splice_type][error_model] = model_result
                succeeded += 1
                
            except Exception as e:
                if verbose > 0:
                    print(f"Error processing {splice_type}: {error_label} vs {correct_label}")
                    print(f"Exception: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                results[splice_type][error_model] = {"error": str(e)}
                failed += 1
    
    # Verify outputs if requested
    if verify_outputs:
        from .utils import verify_error_model_outputs
        
        success = verify_error_model_outputs(
            experiment_dir,
            splice_types=splice_types,
            error_models=error_models
        )
        
        if success and verbose > 0:
            print(f"Successfully verified {succeeded} processed models")
        elif verbose > 0:
            print(f"Verified with issues. Check the output logs.")
    
    # Summary
    if verbose > 0:
        print(f"\nProcessing Summary:")
        print(f"  - Succeeded: {succeeded}")
        print(f"  - Failed: {failed}")
        print(f"  - Skipped: {skipped}")
        print(f"  - Total: {succeeded + failed + skipped}")
    
    return {
        "results": results,
        "summary": {
            "succeeded": succeeded,
            "failed": failed,
            "skipped": skipped,
            "total": succeeded + failed + skipped
        }
    }
