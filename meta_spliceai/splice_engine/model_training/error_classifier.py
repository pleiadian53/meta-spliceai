"""
Error classifier module for training and analyzing splice site error prediction models.

This module provides functionality for training models that classify and analyze
splice site prediction errors (e.g., FP vs TP, FN vs TN), with a focus on
identifying sequence motifs and other features that contribute to errors.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from datetime import datetime
import inspect

from ..utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator, 
    display_dataframe, 
    display_dataframe_in_chunks
)
from ..analysis_utils import (
    analyze_data_labels,
    filter_kmer_features,
    plot_feature_distributions as plot_feature_distributions_advanced, 
    plot_feature_distributions_v1 as plot_feature_distributions_advanced_v1
)
from ..model_utils import ( 
    get_unique_labels,
    compare_feature_rankings,
    compare_feature_importance_ranks
)
from ..sequence_featurizer import display_feature_set
from meta_spliceai.mllib.model_trainer import to_xy
from ..model_evaluator import ModelEvaluationFileHandler
from ..splice_error_analyzer import ErrorAnalyzer
from .visualization import (
    plot_feature_distributions,
    shap_analysis_with_local_agg_plots,
    bar_chart_local_feature_importance,
    plot_feature_rankings_comparison
)

# Import the refactored xgboost pipeline
from .xgboost_trainer import xgboost_pipeline


def train_error_classifier(
    pred_type=None,  # Kept for backward compatibility
    remove_strong_predictors=False, 
    strong_predictors=None, 
    splice_type=None,              # "donor" or "acceptor" or None
    error_label=None,              # e.g., "FP", "FN"
    correct_label=None,            # e.g., "TP", "TN"
    model_type="xgboost",          # Type of model to train
    _test_df=None,                 # For testing: directly provide a DataFrame
    test_data=None,                # Added parameter for test data
    importance_methods=None,       # List of feature importance methods to include in comparison
    xgb_importance_type="total_gain",  # Type of XGBoost feature importance to use
    shap_local_top_k=20,           # Number of top features to consider per sample in SHAP analysis
    shap_global_top_k=50,        # Number of top features to consider globally in SHAP analysis
    shap_plot_top_k=25,          # Number of top features to show in SHAP plots
    use_advanced_feature_plots=True,  # New parameter to control feature plotting
    feature_plot_type="box",    # Default plot type for advanced feature plots
    feature_plot_cols=3,           # Default number of columns for advanced feature plots
    feature_plot_top_k_motifs=20,  # Default top k motifs for advanced feature plots
    feature_plot_use_swarm=False,  # Default swarm plot option for advanced feature plots
    **kwargs
): 
    """
    Train an error classifier for splice-site errors using a featurized dataset.
    
    This function trains a model to classify or analyze splice site prediction
    errors, with a focus on identifying features that contribute to errors.

    Parameters
    ----------
    pred_type : str, optional
        Legacy parameter for backward compatibility. If error_label is not provided,
        this will be used as the error_label.
        
    remove_strong_predictors : bool
        If True, we exclude known "strong" features (e.g. "score") that might overshadow
        the motif-based features. 
        
    strong_predictors : list of str or None
        List of column names to drop if remove_strong_predictors is True. 
        If None, defaults to ["score"].
        
    splice_type : str or None
        If "donor", subset data where df["splice_type"]=="donor".
        If "acceptor", subset data where df["splice_type"]=="acceptor".
        If None, use all data (both donor & acceptor).
        
    error_label : str, optional
        Label for error cases, e.g., "FP" for False Positives or "FN" for False Negatives.
        If not provided, will use pred_type.
        
    correct_label : str, optional
        Label for correct cases, e.g., "TP" for True Positives or "TN" for True Negatives.
        If not provided, defaults to "TP" if error_label is "FP", or "TN" if error_label is "FN".
        
    model_type : str
        Type of model to train. Currently supports "xgboost".
        
    _test_df : DataFrame, optional
        For testing: directly provide a DataFrame.
        
    test_data : DataFrame, optional
        For testing: directly provide a DataFrame.
        
    importance_methods : list, optional
        List of feature importance methods to include in comparison.
        Available options: "shap", "xgboost", "hypothesis", "mutual_info"
        If None, defaults to ["shap", "xgboost", "mutual_info"]
        
    xgb_importance_type : str, default="total_gain"
        Type of XGBoost feature importance to use (e.g., "weight", "total_gain")
        
    shap_local_top_k : int, default=20
        Number of top features to consider per individual sample in SHAP analysis.
        This controls how many features are examined for each data point's explanation.
        
    shap_global_top_k : int, default=50
        Number of top features to consider globally across the entire dataset.
        This affects global importance rankings and feature aggregation.
        If None, defaults to the value of top_k.
        
    shap_plot_top_k : int, default=25
        Number of top features to show in SHAP visualization plots.
        This is purely for display purposes and doesn't affect the underlying analysis.
        If None, automatically calculated based on available features.
        
    use_advanced_feature_plots : bool, default=True
        If True, generates advanced feature distribution plots using the analysis_utils module.
        
    feature_plot_type : str, default="box"
        Type of plot to use for advanced feature plots (e.g., "box", "violin", "swarm").
        
    feature_plot_cols : int, default=3
        Number of columns to use for advanced feature plots.
        
    feature_plot_top_k_motifs : int, default=20
        Number of top motifs to consider for advanced feature plots.
        
    feature_plot_use_swarm : bool, default=False
        If True, uses swarm plots for advanced feature plots.
        
    **kwargs : 
        - col_label : str, default 'label'
        - n_splits : int, default 5
        - top_k : int, default 20
        - output_dir : str, directory for saving results
        - Additional parameters for model training

    Returns
    -------
    model : The trained model
        For model_type="xgboost", returns an XGBoost model.
    result_set : dict
        Contains various outputs such as feature importance, evaluation metrics, etc.
    """
    verbose = kwargs.get('verbose', 1)
    col_label = kwargs.get('col_label', 'label')
    n_splits = kwargs.get('n_splits', 5)
    top_k = kwargs.get('top_k', 20)
    top_k_motifs = kwargs.get('top_k_motifs', 20)
    
    # Handle parameter compatibility
    if error_label is None:
        if pred_type is not None:
            error_label = pred_type
        else:
            error_label = "FP"  # Default
    
    if correct_label is None:
        if error_label.upper() == "FP":
            correct_label = "TP"
        elif error_label.upper() == "FN":
            correct_label = "TN"
        else:
            correct_label = "TP"  # Default
    
    if verbose > 0:
        print_emphasized(f"[action] Training error classifier that compares {error_label} vs. {correct_label} ...")
        if splice_type:
            print_with_indent(f"Splice type: {splice_type}", indent_level=1)
        print_with_indent(f"Model type: {model_type}", indent_level=1)

    # Load dataset from ModelEvaluationFileHandler
    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')

    # By default, assume "score" is the strong predictor we might want to exclude
    if strong_predictors is None:
        strong_predictors = ["score"]  # Could add others, e.g. "spliceai_prob"

    # 1) Load pre-computed featurized dataset
    if _test_df is not None:
        df_trainset = _test_df
    elif test_data is not None:
        df_trainset = test_data
    else:
        df_trainset = mefd.load_featurized_dataset(
            aggregated=True, 
            error_label=error_label, 
            correct_label=correct_label,
            splice_type=splice_type
        )

        # Alternatively, we could use the analyzer to load the dataset
        # analyzer = ErrorAnalyzer(experiment=experiment, model_type=model_type.lower())
        # df_trainset = analyzer.load_featurized_dataset(
        #     aggregated=True, 
        #     error_label=error_label, 
        #     correct_label=correct_label,
        #     splice_type=splice_type
        # )
    
    if verbose > 0: 
        print_with_indent(f"Training set: {df_trainset.shape}", indent_level=1)
        print_with_indent(f"Columns: {display_feature_set(df_trainset, max_kmers=100)}", indent_level=1)
        analysis_result = \
            analyze_data_labels(df_trainset, label_col=col_label, verbose=verbose, handle_missing=None)

    # Convert polars DF to pandas if needed
    if not isinstance(df_trainset, pd.DataFrame):
        try:
            import polars as pl
            if isinstance(df_trainset, pl.DataFrame):
                df_trainset = df_trainset.to_pandas()
        except ImportError:
            pass
            
        if not isinstance(df_trainset, pd.DataFrame):
            raise ValueError(f"Invalid dataframe type: {type(df_trainset)}")

    # 1a) Optionally remove strong predictors if requested
    if remove_strong_predictors:
        for col_to_drop in strong_predictors:
            if col_to_drop in df_trainset.columns:
                df_trainset.drop(columns=col_to_drop, inplace=True, errors='ignore')
                if verbose > 0:
                    print_with_indent(f"[info] Dropped strong predictor: {col_to_drop}", indent_level=1)
            else:
                if verbose > 0:
                    print_with_indent(f"[warning] Column '{col_to_drop}' not found in the dataset. Skipping.", indent_level=1)

    # 2) Convert DF to X, y
    X, y = to_xy(df_trainset, dummify=True, verbose=verbose)

    if verbose > 0:
        print_with_indent("\nFeatures (X):", indent_level=1)
        print_with_indent(f"Shape: {X.shape}", indent_level=1)
        print_with_indent(f"Columns: {display_feature_set(X, max_kmers=100)}", indent_level=1)
        print_with_indent("\nLabels (y):", indent_level=1)
        print_with_indent(f"Unique labels: {get_unique_labels(y)}", indent_level=1) 

    # 3) Model training pipeline
    output_dir = kwargs.get('output_dir', ErrorAnalyzer.analysis_dir)  # depends on `experiment`
    os.makedirs(output_dir, exist_ok=True)

    # Prepare output subject => incorporate splice_type if provided
    if splice_type is None:
        st_str = "any"
    else:
        st_str = splice_type.lower()

    if st_str in ("donor", "acceptor", "neither"):
        if verbose > 0:
            print_with_indent(f"Adjusting the subject name to include splice_type={splice_type} ...", indent_level=1)
        subject = kwargs.get('subject', f"{st_str}_{error_label.lower()}_vs_{correct_label.lower()}")
        if verbose > 0:
            print_with_indent(f"New subject name: {subject}", indent_level=2)
    else:
        subject = kwargs.get('subject', f"{error_label.lower()}_vs_{correct_label.lower()}")

    if remove_strong_predictors:
        subject = f"{subject}-strong-vars-filtered"

    if verbose > 0:
        print('[test] Shape of X, y prior to model training pipeline', X.shape, y.shape)
    
    # Select and run the appropriate model pipeline
    if model_type.lower() == "xgboost":
        # Use inspect.signature to filter parameters based on xgboost_pipeline's signature
        from .xgboost_trainer import xgboost_pipeline
        xgb_sig = inspect.signature(xgboost_pipeline)
        xgb_param_names = set(xgb_sig.parameters.keys())
        
        # Build parameters dictionary with required parameters
        xgb_params = {
            'X': X, 
            'y': y,
            'output_dir': output_dir, 
            'top_k': top_k, 
            'top_k_motifs': top_k_motifs,
            'n_splits': n_splits, 
            'subject': subject,
        }
        
        # Add any additional compatible parameters from kwargs
        for param, value in kwargs.items():
            if param in xgb_param_names:
                xgb_params[param] = value
                
        if verbose > 0:
            print_with_indent(f"Running XGBoost pipeline with parameters: {', '.join(xgb_params.keys())}", indent_level=1)
            
        # Call with filtered parameters
        model, result_set = xgboost_pipeline(**xgb_params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Currently only 'xgboost' is supported.")

    print_section_separator()
    print_emphasized(f"[result] Completed training error classifier for {error_label} vs {correct_label}.")


    ######################################################

    # 4) Additional follow-up analysis
    if verbose > 0:
        print("[action] Running follow-up analyses on feature importance  ...")

    # Define default importance methods if not specified
    if importance_methods is None:
        importance_methods = ["shap", "xgboost", "mutual_info"]
    
    # Validate specified methods
    valid_methods = ["shap", "xgboost", "hypothesis", "mutual_info"]
    for method in importance_methods:
        if method not in valid_methods:
            raise ValueError(f"Invalid importance method: {method}. Valid options are: {valid_methods}")
    
    # Helper function to get importance dataframe with legacy support
    def get_importance_df(key, dict_key=None, default_type='total_gain'):
        """
        Helper to retrieve importance dataframes with support for both old and new 
        result set structures.

        E.g., 
        
        Args:
            key: The legacy key in result_set
            dict_key: The new dictionary key for structured storage
            default_type: The default importance type to use if multiple are available
            
        Returns:
            DataFrame with feature importance, or empty DataFrame if not found
        """
        if key in result_set:
            return result_set[key]
        elif dict_key in result_set:
            # Check if the dict_key points to a dictionary or directly to a DataFrame
            if isinstance(result_set[dict_key], dict) and default_type is not None:
                # This is for XGBoost's multiple importance types structure
                importance_types = result_set.get('importance_types', [default_type])
                importance_type = default_type if default_type in importance_types else importance_types[0]
                if importance_type in result_set[dict_key]:
                    return result_set[dict_key][importance_type]
                else:
                    print(f"[warning] Importance type '{importance_type}' not found in {dict_key}")
                    # Try to use any available type
                    if len(result_set[dict_key]) > 0:
                        first_key = next(iter(result_set[dict_key]))
                        print(f"[info] Using '{first_key}' instead")
                        return result_set[dict_key][first_key]
            else:
                # For simple dataframes without the nested structure
                return result_set[dict_key]
        
        # Return empty DataFrame if nothing found
        print(f"[warning] No {key} feature importance found in result set")
        return pd.DataFrame(columns=['feature', 'importance_score'])

    if verbose > 1:
        print(f"[debug] Result set keys: {list(result_set.keys())}")
    
    # Get feature importance dataframes
    importance_df_shap = result_set['importance_df_shap']
    importance_df_xgboost = get_importance_df(
        'importance_df_xgboost', 
        'xgboost_importance_dfs', 
        xgb_importance_type  # Use the specified importance type
    )
    importance_df_hypotest = get_importance_df(
        'importance_df_hypotest', 
        'importance_df_hypo', 
        None
    )
    importance_df_mutual_info = get_importance_df(
        'importance_df_mutual',  # Fix key name to match what's used in xgboost_trainer.py
        'mutual_info_df',
        None
    )
    
    # Extract top features from each method
    shap_features = importance_df_shap.head(top_k)['feature'].tolist() if not importance_df_shap.empty else []
    xgb_features = importance_df_xgboost.head(top_k)['feature'].tolist() if not importance_df_xgboost.empty else []
    sig_features = importance_df_hypotest.head(top_k)['feature'].tolist() if not importance_df_hypotest.empty else []
    mutual_info_features = importance_df_mutual_info.head(top_k)['feature'].tolist() if not importance_df_mutual_info.empty else []
    
    # Create a set combining features from selected importance methods
    combined_features = set()
    if "shap" in importance_methods:
        combined_features.update(shap_features)
    if "xgboost" in importance_methods:
        combined_features.update(xgb_features)
    if "hypothesis" in importance_methods:
        combined_features.update(sig_features)
    if "mutual_info" in importance_methods:
        combined_features.update(mutual_info_features)
    
    # Keep features that appear in at least two rankings from the selected methods
    if len(importance_methods) >= 2:
        # Create sets for each selected method
        method_feature_sets = []
        if "shap" in importance_methods and shap_features:
            method_feature_sets.append(set(shap_features))
        if "xgboost" in importance_methods and xgb_features:
            method_feature_sets.append(set(xgb_features))
        if "hypothesis" in importance_methods and sig_features:
            method_feature_sets.append(set(sig_features))
        if "mutual_info" in importance_methods and mutual_info_features:
            method_feature_sets.append(set(mutual_info_features))
        
        # Find features that appear in at least two methods
        final_features = []
        for feature in combined_features:
            count = sum(1 for feature_set in method_feature_sets if feature in feature_set)
            if count >= 2:
                final_features.append(feature)
    else:
        # If only one method is selected, use all features from that method
        final_features = list(combined_features)
    
    if verbose > 0:
        print_emphasized(f"[info] Final consolidated feature set (n={len(final_features)}): {final_features}")
        if "xgboost" in importance_methods:
            print_with_indent(f"[test] Data shape XGBoost importance ({xgb_importance_type}): {importance_df_xgboost.shape}", indent_level=1)
        if "shap" in importance_methods:
            print_with_indent(f"[test] Data shape SHAP importance: {importance_df_shap.shape}", indent_level=1)
        if "hypothesis" in importance_methods:
            print_with_indent(f"[test] Data shape HypoTest importance: {importance_df_hypotest.shape}", indent_level=1)
        if "mutual_info" in importance_methods:
            print_with_indent(f"[test] Data shape Mutual Information importance: {importance_df_mutual_info.shape}", indent_level=1)
            
        # NOTE: Different feature importance methods can have different shapes, which is expected:
        # - Mutual Information calculates scores for all features in the dataset by directly measuring 
        #   statistical relationships between each feature and the target.
        # - XGBoost/SHAP only include features that the model actually uses in its decision trees, 
        #   which is typically a subset of all features.
        # - The compare_feature_importance_ranks function handles these differences properly by
        #   performing an outer join when combining the dataframes.

    # Compare feature rankings from different methods
    importance_dfs = {}
    
    if "xgboost" in importance_methods and not importance_df_xgboost.empty:
        importance_dfs[f"XGBoost ({xgb_importance_type})"] = importance_df_xgboost
    
    if "shap" in importance_methods and not importance_df_shap.empty:
        importance_dfs["SHAP"] = importance_df_shap
    
    if "hypothesis" in importance_methods and not importance_df_hypotest.empty:
        importance_dfs["HypoTest"] = importance_df_hypotest
    
    if "mutual_info" in importance_methods and not importance_df_mutual_info.empty:
        importance_dfs["MutualInfo"] = importance_df_mutual_info
    
    if len(importance_dfs) >= 2:  # Need at least 2 methods to compare
        # <<< ADD DEBUG PRINT HERE >>>
        print("[Compare Debug] Input to compare_feature_importance_ranks:")
        for name, df in importance_dfs.items():
            print(f"  - {name}: shape {df.shape}")
            
        # First calculate statistical metrics on the overlap and correlation
        importance_df_list = list(importance_dfs.values())
        method_names = list(importance_dfs.keys())
        
        overlap, correlation = compare_feature_rankings(
            importance_df_list,
            method_names=method_names,
            top_k=top_k
        )
    
        # Now create the visual comparison using the correct function
        if verbose > 0:
            print_emphasized("[action] Feature importance comparison #2: ")
            
        output_file = f"{subject}-feature-importance-comparison.pdf"
        output_path = os.path.join(output_dir, output_file) if output_dir else output_file
        
        # Use the correct function for visualization
        compare_feature_importance_ranks(
            importance_dfs,
            top_k=top_k,
            primary_method=method_names[0],  # Use the first method as primary
            # primary_method="MutualInfo" if "MutualInfo" in importance_dfs else method_names[0],  # Prefer MutualInfo as primary if available
            output_path=output_path,
            verbose=verbose,
            save=True,
            plot_style="percentile_rank",  # More intuitive visualization
            # plot_style="raw",
            ensure_method_representation=True  # Ensure all methods have some representation in the plot
        )
    
    if verbose > 0:
        print_emphasized("[action] Generating additional visualizations...")
    
    # Get the top features from SHAP importance for visualization
    # This requires having SHAP importance results available
    top_features = None
    if "shap" in importance_methods and "importance_df_shap" in result_set:
        importance_df_shap = result_set["importance_df_shap"]
        if not importance_df_shap.empty:
            # Get top features from SHAP importance
            top_features = importance_df_shap["feature"].head(top_k).tolist()
            if verbose > 0:
                print_with_indent(f"[info] Using top {len(top_features)} features from SHAP importance for distribution plots", indent_level=1)
    
    # If SHAP importance wasn't run or failed, fall back to XGBoost importance
    if top_features is None and model is not None:
        try:
            # Try to get feature importance directly from the model
            importance_dict = {feature: score for feature, score in 
                              zip(X.columns, model.feature_importances_)}
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in sorted_features[:top_k]]
            if verbose > 0:
                print_with_indent(f"[info] Using top {len(top_features)} features from model importance for distribution plots", indent_level=1)
        except:
            # If all else fails, we'll let the plot functions handle feature selection
            top_features = None
            if verbose > 0:
                print_with_indent("[warning] Could not determine top features from importance; will use all features", indent_level=1)
    
    ######################################################
    if verbose > 0:
        print_emphasized("[viz] Generating advanced feature distribution plots...")

    # Generate advanced feature distribution plots if requested
    if use_advanced_feature_plots:
        
        # Set up parameters for the advanced plotting function
        advanced_plot_params = {
            "output_path": os.path.join(output_dir, f"{subject}-feature-distributions-advanced.pdf"),
            "plot_type": feature_plot_type,
            "label_text_0": correct_label,
            "label_text_1": error_label,
            "title": f"Feature Distributions - {subject}",
            "show_plot": False,
            "top_k_motifs": feature_plot_top_k_motifs,
            "use_swarm_for_motifs": feature_plot_use_swarm
        }
        
        # Filter params that won't conflict with the function signature
        advanced_plot_sig = inspect.signature(plot_feature_distributions_advanced)
        advanced_valid_params = {k: v for k, v in advanced_plot_params.items() 
                                if k in advanced_plot_sig.parameters}
        
        # Call the advanced plotting function with top features
        plot_feature_distributions_advanced(
            X,
            y,
            top_features if top_features is not None else list(X.columns),
            verbose=verbose,
            **advanced_valid_params
        )

        # Use another version of the same function (v1) but with different output file
        advanced_plot_params_v1 = {
            "output_path": os.path.join(output_dir, f"{subject}-feature-distributions-advanced-v1.pdf"),
            "plot_type": feature_plot_type,
            "label_text_0": correct_label,
            "label_text_1": error_label,
            "title": f"Feature Distributions - {subject}",
            "show_plot": False,
            "top_k_motifs": feature_plot_top_k_motifs,
            "use_swarm_for_motifs": feature_plot_use_swarm
        }
        
        # Filter params that won't conflict with the function signature
        advanced_plot_sig_v1 = inspect.signature(plot_feature_distributions_advanced_v1)
        advanced_valid_params_v1 = {k: v for k, v in advanced_plot_params_v1.items() 
                                   if k in advanced_plot_sig_v1.parameters}
        
        # Call the advanced plotting function with top features
        plot_feature_distributions_advanced_v1(
            X,
            y,
            top_features if top_features is not None else list(X.columns),
            verbose=verbose,
            **advanced_valid_params_v1
        )
    
    # Always generate the basic plots for consistency
    if verbose > 0:
        print_with_indent("[viz] Generating basic feature distribution plots...", indent_level=1)
    
    plot_feature_distributions(
        X, 
        y,
        features=top_features if top_features is not None else list(X.columns),
        output_dir=output_dir,
        subject=subject,
        verbose=verbose
    )
    
    ######################################################
    print_emphasized("[viz] Generating SHAP analysis with local aggregation plots...")

    # Generate SHAP analysis with local aggregation plots
    shap_analysis_with_local_agg_plots(
        model,
        X,
        y,
        output_dir=output_dir,
        class_labels={
            error_label: 1,  # Map error class name (e.g., "FP") to positive class (1)
            correct_label: 0  # Map correct class name (e.g., "TP") to negative class (0)
        },
        error_class=error_label,     # The error class name (e.g., "FP", "FN")
        correct_class=correct_label, # The correct class name (e.g., "TP", "TN")
        local_top_k=shap_local_top_k,     # Number of top features to examine per individual sample
        global_top_k=shap_global_top_k,   # Number of top features to consider across all samples
        plot_top_k=shap_plot_top_k,       # Number of features to display in plots
        subject=subject,
        verbose=verbose
    )
    
    # Generate local feature importance bar charts for selected samples
    # Default: misclassified samples (see demo_local_feature_importance_strategies for other options)
    # NOTE: For examples of different sample selection strategies (high confidence, decision boundary,
    #       custom samples, etc.), see visualization.demo_local_feature_importance_strategies
    bar_chart_local_feature_importance(
        model,
        X,
        output_dir=output_dir,
        top_k=top_k,
        subject=subject,
        verbose=verbose,
        n_samples=min(5, len(X)),  # Default to 5 samples
        sample_selection="misclassified",  # Focus on misclassified samples by default
        y_test=y  # Pass in true labels for misclassification detection
    )
    
    # Process all importance types if available
    if 'xgboost_importance_dfs' in result_set and 'importance_types' in result_set:
        importance_types = result_set['importance_types']
        
        # Create separate visualizations and analyses for each importance type
        for imp_type in importance_types:
            if imp_type not in result_set['xgboost_importance_dfs']:
                continue
                
            current_imp_df = result_set['xgboost_importance_dfs'][imp_type]
            
            # Generate specialized comparison for this importance type
            imp_output_file = f"{subject}-xgboost-{imp_type}-comparison.pdf"
            try:
                # Store the top features from this importance type in result_set
                if not current_imp_df.empty:
                    # Get top features from this importance type
                    top_features = current_imp_df.head(top_k)['feature'].tolist()
                    result_set[f'top_features_{imp_type}'] = top_features
                    
                    # Create a standard feature importance comparison plot using the better function
                    # Build a dictionary of importance methods for comparison
                    imp_importance_dfs = {}
                    
                    # Always include the current importance type
                    imp_importance_dfs[f"XGBoost ({imp_type})"] = current_imp_df
                    
                    # Include SHAP values if available
                    if 'importance_df_shap' in result_set and not result_set['importance_df_shap'].empty:
                        imp_importance_dfs["SHAP"] = result_set['importance_df_shap']
                    
                    # Include mutual information if available
                    if 'importance_df_mutual' in result_set and not result_set['importance_df_mutual'].empty:
                        imp_importance_dfs["MutualInfo"] = result_set['importance_df_mutual']
                    
                    imp_output_path = os.path.join(output_dir, imp_output_file) if output_dir else imp_output_file
                    
                    compare_feature_importance_ranks(
                        imp_importance_dfs,
                        top_k=top_k,
                        primary_method=f"XGBoost ({imp_type})",  # Use current importance type as primary
                        # primary_method="MutualInfo" if "MutualInfo" in imp_importance_dfs else f"XGBoost ({imp_type})",  # Prefer MutualInfo if available
                        output_path=imp_output_path,
                        verbose=verbose,
                        save=True,
                        plot_style="percentile_rank",
                        # plot_style="raw",
                        ensure_method_representation=True  # Ensure all methods have some representation in the plot
                    )
            except Exception as e:
                print(f"[warning] Failed to create comparison for '{imp_type}': {str(e)}")
                
            # Include this importance type's results in the result_set for return
            result_set[f'importance_df_xgboost_{imp_type}'] = current_imp_df
    
    # Create done flag
    done_file = os.path.join(output_dir, f"{subject}_done.txt")
    with open(done_file, "w") as f:
        f.write(f"Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Features analyzed: {len(X.columns)}\n")
        f.write(f"Top features: {', '.join(final_features[:min(10, len(final_features))])}\n")
        f.write(f"Importance types analyzed: {', '.join(result_set.get('importance_types', ['weight']))}\n")
    
    if verbose > 0:
        print_emphasized(f"[info] Analysis complete. All outputs saved to: {output_dir}")
            
    # Return the results
    return model, result_set


def workflow_train_error_classifier(experiment='hard_genes', test_mode=False, test_data=None, **kwargs):
    """
    Execute a comprehensive error analysis workflow by training error classifiers
    for all valid combinations of splice types and error models.
    
    This function systematically iterates through 9 combinations:
    - Splice types: "donor", "acceptor", "any"
    - Error models: "FP vs TP", "FN vs TP", "FN vs TN"
    
    For each combination, it creates the appropriate output directory structure,
    trains the error classifier model, and marks completion with a status file.
    
    Parameters
    ----------
    experiment : str, default='hard_genes'
        Experiment name used for directory organization
    test_mode : bool, default=False
        If True, use synthetic data for testing instead of loading real datasets
    test_data : DataFrame, default=None
        Synthetic test data to use when test_mode is True
    **kwargs : dict
        Additional keyword arguments to pass to train_error_classifier
        
        enable_check_existing : bool, default=False
            If True, skips combinations that have already been processed
            (indicated by presence of a done.txt file)
        model_type : str, default='xgboost'
            Type of model to train
        
    Returns
    -------
    None
    """
    enable_check_existing = kwargs.get('enable_check_existing', False)

    model_type = kwargs.get('model_type', 'xgboost')
    analyzer = ErrorAnalyzer(experiment=experiment, model_type=model_type.lower())
    
    # Define possible labels
    error_labels = ["FP", "FN"]  
    correct_labels = ["TP", "TN"]  

    # Iterate through all combinations, excluding (FP, TN)
    for splice_type in ["any", "donor", "acceptor"]:
        
        for error_label, correct_label in product(error_labels, correct_labels):
            if (error_label, correct_label) == ("FP", "TN"):
                continue  # Skip the unwanted combination - FP vs TN doesn't make biological sense

            print_emphasized(f"[info] Training error classifier: {error_label} vs {correct_label} ...")
            print_with_indent(f"Experiment: {experiment}", indent_level=1)
            print_with_indent(f"Splice type: {splice_type}", indent_level=1)

            output_dir = analyzer.set_analysis_output_dir(
                error_label=error_label, 
                correct_label=correct_label, 
                splice_type=splice_type
            )
            os.makedirs(output_dir, exist_ok=True)
            print_with_indent(f"[demo] Output directory set to: {output_dir}", indent_level=1)

            # Define a unique marker file name for the combination
            dummy_file_name = f"{splice_type}_{error_label}_vs_{correct_label}_done.txt"
            dummy_file_path = os.path.join(output_dir, dummy_file_name)

            # Check if the combination has already been processed
            if enable_check_existing: 
                if os.path.exists(dummy_file_path):
                    print(f"[info] Skipping {splice_type}: {error_label} vs {correct_label} as it is already processed.")
                    continue
            else: 
                # Clean up existing status files
                for file in os.listdir(output_dir): 
                    if file.startswith(f"{splice_type}_") and file.endswith("_done.txt"):
                        os.remove(os.path.join(output_dir, file))
                        print(f"[info] Removed status file: {file}")

            # Process with default feature importance options
            for remove_strong_predictors in [False]:  # Optional: can be extended to [True, False]
                print_emphasized(f"[info] Training error classifier ...")
                print_with_indent(f"Experiment: {experiment}", indent_level=1)
                print_with_indent(f"Error label: {error_label}", indent_level=1)
                print_with_indent(f"Correct label: {correct_label}", indent_level=1)
                print_with_indent(f"[i/o] Output directory: {output_dir}", indent_level=1)
                print_with_indent(f"Remove strong predictors: {remove_strong_predictors}", indent_level=1)
                
                # Prepare kwargs for train_error_classifier
                train_kwargs = {
                    k: v for k, v in kwargs.items() 
                    if k not in ['output_dir', 'enable_check_existing']
                }
                
                # Add test_data if in test_mode
                if test_mode and test_data is not None:
                    train_kwargs['test_data'] = test_data

                model, result_set = train_error_classifier(
                    pred_type=error_label, 
                    remove_strong_predictors=remove_strong_predictors, 
                    output_dir=output_dir, 
                    error_label=error_label, 
                    correct_label=correct_label,
                    splice_type=splice_type,
                    model_type=model_type,
                    verbose=1,
                    **train_kwargs
                )

            # Create the marker file to indicate completion
            with open(dummy_file_path, 'w') as f:
                f.write(f"Completed processing {splice_type}: {error_label} vs {correct_label}")
    
    print_emphasized("[info] Workflow completed successfully!")
    return None
