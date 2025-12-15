"""
XGBoost trainer module for training and evaluating XGBoost models.

This module provides modular components for training, evaluating, and analyzing XGBoost models
for splice site error classification and feature importance analysis.
"""

import os
import re
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import inspect

from sklearn.model_selection import (
    train_test_split, 
    cross_val_predict, 
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
from sklearn.exceptions import NotFittedError
import scipy.stats as stats

from ..utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator, 
    display_dataframe
)
from ..analysis_utils import plot_feature_importance
from ..feature_importance.wrappers import (
    calculate_xgboost_importance,
    calculate_shap_importance,
    calculate_hypothesis_testing,
    calculate_effect_sizes,
    calculate_mutual_information
)
from ..performance_analyzer import plot_cv_roc_curve, plot_cv_pr_curve


def format_subject_for_title(subject):
    """
    Format the subject parameter value to be a valid part of the plot title.

    Parameters:
    - subject (str): The subject parameter value (e.g., "fp_vs_tp").

    Returns:
    - str: The formatted subject for the plot title (e.g., "FP vs TP").
    """
    return subject.replace("_", " ").title()


def train_xgboost_model(X_train, y_train, **params):
    """
    Train an XGBoost classifier on the given training data.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    **params : dict
        Parameters to pass to XGBClassifier
        
    Returns
    -------
    xgb.XGBClassifier
        Trained XGBoost model
    """
    # Set default parameters if not provided
    default_params = {
        'eval_metric': 'logloss',
        'random_state': 42,
        'verbosity': 1
    }
    
    # Use function signature inspection to filter parameters for XGBClassifier
    xgb_classifier_sig = inspect.signature(xgb.XGBClassifier)
    xgb_param_names = set(xgb_classifier_sig.parameters.keys())
    
    # Filter parameters based on XGBClassifier's signature
    valid_params = {}
    for param_name, param_value in params.items():
        if param_name in xgb_param_names:
            valid_params[param_name] = param_value
    
    # Update default params with filtered valid params
    model_params = {**default_params, **valid_params}
    
    # Convert string labels to integers if needed
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    if isinstance(y_train.iloc[0], (str, bool)):
        y_train_encoded = le.fit_transform(y_train)
    else:
        y_train_encoded = y_train
    
    # Initialize and train model
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train_encoded)
    
    # Store the label encoder with the model for later use
    model.label_encoder_ = le
    
    return model


def evaluate_xgboost_model(model, X_test, y_test, output_dir=None, subject="model_evaluation", verbose=1, model_name="xgboost", **kwargs):
    """
    Evaluate an XGBoost model and generate performance metrics and visualizations.
    
    Parameters
    ----------
    model : xgb.XGBClassifier
        Trained XGBoost model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    output_dir : str, optional
        Directory to save output files
    subject : str, optional
        Subject name for output files
    verbose : int, optional
        Verbosity level
    model_name : str, optional
        Name of the model, used in file naming patterns
    **kwargs : dict
        Additional parameters
        
    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """
    # Import metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        roc_curve, precision_recall_curve, auc
    )
    
    plot_format = kwargs.get('plot_format', 'pdf')
    
    # Create output directory if needed
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Ensure test labels are encoded properly for metric calculation
    # If y_test has string labels, we need to handle them properly
    if hasattr(y_test, 'dtype') and y_test.dtype == object:
        try:
            y_test_encoded = y_test.astype(int)
        except (ValueError, TypeError):
            # If conversion fails, assume we need to encode string labels
            if hasattr(model, 'label_encoder_') and hasattr(model.label_encoder_, 'transform'):
                y_test_encoded = model.label_encoder_.transform(y_test)
            else:
                # As a fallback, create a temporary encoder just for evaluation
                from sklearn.preprocessing import LabelEncoder
                temp_encoder = LabelEncoder().fit(y_test)
                y_test_encoded = temp_encoder.transform(y_test)
    else:
        # Numeric labels can be used directly
        y_test_encoded = y_test
    
    # Generate predictions
    y_pred_encoded = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Convert predictions back to original labels if needed
    if hasattr(model, 'label_encoder_') and hasattr(model.label_encoder_, 'inverse_transform'):
        try:
            # Try to use the model's label encoder if available
            y_pred = model.label_encoder_.inverse_transform(y_pred_encoded)
        except (AttributeError, ValueError, NotFittedError):
            # If the label encoder isn't properly fitted, use encoded predictions
            y_pred = y_pred_encoded
    else:
        # If we're using to_xy() conversion, encoded predictions are fine
        y_pred = y_pred_encoded
    
    # Calculate metrics (using encoded versions for numeric metrics)
    results = {
        'accuracy': accuracy_score(y_test_encoded, y_pred_encoded),
        'precision': precision_score(y_test_encoded, y_pred_encoded, average='binary'),
        'recall': recall_score(y_test_encoded, y_pred_encoded, average='binary'),
        'f1': f1_score(y_test_encoded, y_pred_encoded, average='binary'),
        'roc_auc': roc_auc_score(y_test_encoded, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Print results if verbose
    if verbose > 0:
        print_emphasized("[info] Model Evaluation Results:")
        print_with_indent(f"Accuracy: {results['accuracy']:.4f}", indent_level=1)
        print_with_indent(f"Precision: {results['precision']:.4f}", indent_level=1)
        print_with_indent(f"Recall: {results['recall']:.4f}", indent_level=1)
        print_with_indent(f"F1 Score: {results['f1']:.4f}", indent_level=1)
        print_with_indent(f"ROC AUC: {results['roc_auc']:.4f}", indent_level=1)
        print_with_indent("\nConfusion Matrix:", indent_level=1)
        print_with_indent(str(confusion_matrix(y_test, y_pred)), indent_level=2)
        print_with_indent("\nClassification Report:", indent_level=1)
        print_with_indent(classification_report(y_test, y_pred), indent_level=2)
    
    # Generate and save ROC curve
    if output_dir is not None:
        # ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test_encoded, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{subject}-{model_name}-roc.{plot_format}"), dpi=300)

        # PR curve
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_test_encoded, y_proba)
        plt.plot(recall, precision, color='green', lw=2)
        plt.axhline(y=sum(y_test_encoded)/len(y_test_encoded), color='r', linestyle='--', label='No Skill')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{subject}-{model_name}-prc.{plot_format}"), dpi=300)
        
        if verbose > 0:
            print_with_indent(f"[output] Saved ROC and PR curves to {output_dir}", indent_level=1)
    
    return results


def filter_features_by_pattern(importance_df, pattern=r'^\d+mer_.*', keep_matches=True):
    """
    Filter features in an importance DataFrame based on a regex pattern.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance_score' columns
    pattern : str, default=r'^\d+mer_.*'
        Regular expression pattern to match feature names against
    keep_matches : bool, default=True
        If True, keep features matching the pattern; if False, remove matching features
        
    Returns
    -------
    pd.DataFrame
        Filtered importance DataFrame
    """
    import re
    
    # Create mask for matching features
    mask = importance_df['feature'].apply(lambda x: bool(re.match(pattern, x)))
    
    # Keep or remove matching features based on keep_matches parameter
    if keep_matches:
        return importance_df[mask].copy()
    else:
        return importance_df[~mask].copy()


def normalize_feature_importance_df(df, all_features, default_score=0):
    """
    Ensures that a feature importance DataFrame includes all features.
    For features not present in the original DataFrame, adds rows with default importance scores.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature importance DataFrame with 'feature' and 'importance_score' columns
    all_features : list
        Complete list of all features that should be included
    default_score : float, default=0
        Default importance score for features not in the original DataFrame
        
    Returns
    -------
    pd.DataFrame
        Normalized DataFrame containing all features
    """
    # Create a set of features already in the DataFrame
    existing_features = set(df['feature'])
    
    # Identify missing features
    missing_features = [f for f in all_features if f not in existing_features]
    
    # If there are missing features, add them with default scores
    if missing_features:
        missing_df = pd.DataFrame({
            'feature': missing_features,
            'importance_score': [default_score] * len(missing_features)
        })
        
        # Combine with original DataFrame
        result_df = pd.concat([df, missing_df], ignore_index=True)
        
        # Sort by importance score (descending)
        result_df = result_df.sort_values('importance_score', ascending=False)
        
        return result_df
    else:
        # If no missing features, return the original DataFrame
        return df


def save_feature_importance_outputs(
    importance_df, subject, model_name, output_dir, feature_type, 
    top_k, motif_pattern=r'^\d+mer_.*', plot_format='pdf', 
    colormap='viridis', verbose=1
):
    """
    Create and save standardized outputs for feature importance analysis.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance_score' columns
    subject : str
        Subject name (e.g., "fp_vs_tp")
    model_name : str
        Model name (e.g., "xgboost")
    output_dir : str
        Directory to save outputs
    feature_type : str
        Type of feature importance (e.g., "shap", "xgboost", "mutual", "hypo")
    top_k : int
        Number of top features to include in subset outputs
    motif_pattern : str, default=r'^\d+mer_.*'
        Regular expression pattern to identify motif features
    plot_format : str, default='pdf'
        File format for plots
    colormap : str, default='viridis'
        Colormap to use for plotting
    verbose : int, default=1
        Verbosity level
        
    Returns
    -------
    dict
        Dictionary containing filtered DataFrames for all, motif, and non-motif features
    """
    import os
    import matplotlib.pyplot as plt
    
    # Filter motif and non-motif features
    motif_df = filter_features_by_pattern(importance_df, pattern=motif_pattern, keep_matches=True)
    nonmotif_df = filter_features_by_pattern(importance_df, pattern=motif_pattern, keep_matches=False)
    
    results = {
        'all': importance_df,
        'motif': motif_df,
        'nonmotif': nonmotif_df
    }
    
    format = 'tsv'  # Use TSV for data files
    sep = '\t' if format == 'tsv' else ','
    
    # Process all feature types
    for feature_subset, df in results.items():
        if df.empty:
            if verbose > 0:
                print_with_indent(f"[info] No {feature_subset} features found for {feature_type}", indent_level=1)
            continue
            
        # Create barplot
        subset_label = f"-{feature_subset}" if feature_subset != 'all' else ""
        title_prefix = {
            'all': 'All Features',
            'motif': 'Motif Features Only',
            'nonmotif': 'Non-Motif Features Only'
        }[feature_subset]
        
        # 1. Barplot
        output_path = os.path.join(output_dir, f"{subject}-{model_name}-{feature_type}{subset_label}-barplot.{plot_format}")
        
        # Always rank by absolute values to get the most important features by magnitude
        # This maintains consistency across importance types while preserving directionality in the visualization
        
        plot_feature_importance(
            df=df,
            title=f"Feature Importance - {title_prefix} ({feature_type.title()})",
            output_path=output_path,
            top_k=top_k,
            colormap=colormap,
            use_continuous_color=True,  # Better visualization with continuous color mapping
            verbose=verbose,
            rank_by_abs=True  # Always use absolute values for ranking, but preserve directionality
        )
        if verbose > 0:
            print_with_indent(f"[output] Saved {feature_subset} feature importance plot ({feature_type}) to: {output_path}", indent_level=1)
        
        # 2. Top-k dataset
        output_file = f"{subject}-{model_name}-importance-{feature_type}{subset_label}.{format}"
        output_path = os.path.join(output_dir, output_file)
        df.head(top_k).to_csv(output_path, sep=sep, index=False)
        if verbose > 0:
            print_with_indent(f"[i/o] Top {top_k} {feature_subset} feature importance ({feature_type}) saved to: {output_path}", indent_level=1)
        
        # 3. Full dataset
        output_file = f"{subject}-{model_name}-importance-{feature_type}{subset_label}-full.{format}"
        output_path = os.path.join(output_dir, output_file)
        df.to_csv(output_path, sep=sep, index=False)
        if verbose > 0:
            print_with_indent(f"[i/o] Full {feature_subset} feature importance ({feature_type}) saved to: {output_path}", indent_level=1)
    
    return results


def xgboost_pipeline(
    X, y, 
    model=None,
    test_size=0.2,
    n_splits=5,
    random_state=42,
    output_dir=None, 
    save=True, 
    subject="fp_vs_tp", 
    importance_types=["total_gain"],  
    model_name="xgboost",
    **kwargs
):
    """
    Run the full XGBoost pipeline including:
    - Data splitting
    - Model training (or use existing model)
    - Model evaluation
    - Feature importance analysis

    Parameters
    ----------
    X : pd.DataFrame
        Features. IMPORTANT: Categorical variables must be properly encoded (e.g., one-hot encoded)
        before passing to this function. This encoding is typically handled in train_error_classifier
        using the to_xy function with dummify=True, but must be done manually if calling this
        function directly. XGBoost does not handle categorical variables automatically.
    y : pd.Series or np.array
        Labels
    model : XGBoost model, default=None
        Pre-trained model (if None, a new model will be trained)
    test_size : float, default=0.2
        Test set fraction
    n_splits : int, default=5
        Number of cross-validation splits
    random_state : int, default=42
        Random seed for reproducibility
    output_dir : str, default=None
        Directory to save outputs
    save : bool, default=True
        Whether to save outputs
    subject : str, default="fp_vs_tp"
        Subject of the analysis (used for file naming)
    importance_types : list, default=["total_gain"]
        Types of feature importance to calculate (options: "weight", "gain", "total_gain", "cover", "total_cover")
    model_name : str, default="xgboost"
        Name of the model, used in file naming patterns
    **kwargs
        Additional parameters
        
    Returns
    -------
    tuple
        (model, result_set)
    """
    verbose = kwargs.get('verbose', 1)
    use_full_data_for_explanation = kwargs.get('use_full_data_for_explanation', True) 
    top_k = kwargs.get('top_k', 20) if 'top_k' in kwargs else 20
    top_k_motifs = kwargs.get('top_k_motifs', 20) if 'top_k_motifs' in kwargs else 15
    plot_format = kwargs.get('plot_format', 'pdf')
    
    result_set = {}  # A dictionary to store outputs from each step. 

    # Create output directory if needed
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), f"{model_name}_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Ensure X, y indices are reset to avoid indexing mismatches
    if isinstance(X, pd.DataFrame):
        X = X.reset_index(drop=True)
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.reset_index(drop=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Store all feature names to ensure consistent output across importance methods
    all_feature_names = list(X.columns)
    result_set['all_feature_names'] = all_feature_names

    # Initialize and train XGBoost classifier
    print_emphasized(f"[info] Training XGBoost model for subject={subject} ...")
    if model is None:
        model = train_xgboost_model(
            X_train, 
            y_train, 
            random_state=random_state, 
            verbosity=verbose,
            **kwargs
        )

    # Cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_proba_cv = cross_val_predict(model, X, y, cv=skf, method="predict_proba")
    y_pred_cv = np.argmax(y_proba_cv, axis=1)

    # Evaluate model on test set
    eval_results = evaluate_xgboost_model(
        model, X_test, y_test, 
        output_dir=output_dir if save else None, 
        subject=subject,
        verbose=verbose,
        model_name=model_name,
        plot_format=plot_format
    )
    result_set['evaluation'] = eval_results
    
    # Generate cross-validation ROC and PR curves
    if save:
        print_emphasized("[info] Generating cross-validation ROC and PR curves...")
        # ROC curve with cross-validation
        roc_output_path = os.path.join(output_dir, f"{subject}-{model_name}-ROC-CV-{n_splits}folds.{plot_format}") if save else None
        plot_cv_roc_curve(
            model, X, y, 
            cv_splits=n_splits, 
            random_state=random_state,
            title=f"Cross-Validated ROC Curve for {format_subject_for_title(subject)}",
            output_path=roc_output_path,
            show_std=True,
            plot_folds=True
        )
        if verbose > 0 and save:
            print_with_indent(f"[output] Saved CV ROC curve to: {roc_output_path}", indent_level=1)
        
        # PR curve with cross-validation
        pr_output_path = os.path.join(output_dir, f"{subject}-{model_name}-PRC-CV-{n_splits}folds.{plot_format}") if save else None
        plot_cv_pr_curve(
            model, X, y, 
            cv_splits=n_splits, 
            random_state=random_state,
            title=f"Cross-Validated PR Curve for {format_subject_for_title(subject)}",
            output_path=pr_output_path,
            show_std=True,
            plot_folds=True
        )
        if verbose > 0 and save:
            print_with_indent(f"[output] Saved CV PR curve to: {pr_output_path}", indent_level=1)

    ######################################################

    # Retrain the model on the entire datase
    # - A common practice to get the “best possible final model” now that we’ve validated it.
    print_emphasized("[info] Retraining model on the full dataset ...")
    
    # Save feature columns used for training
    feature_columns = X.columns.tolist()
    
    # Retrain the model
    model.fit(X, y)
    
    # Store feature names in the model for future reference
    if hasattr(model, 'feature_names_'):
        model.feature_names_ = feature_columns
    else:
        # XGBoost specific approach
        try:
            model.get_booster().feature_names = feature_columns
        except:
            print("[warning] Unable to store feature names in model")
    
    ######################################################

    # Determine whether to use full data or just test set for all subsequent analyses
    X_shap = X if use_full_data_for_explanation else X_test
    y_shap = y if use_full_data_for_explanation else y_test
    print(f"[info] Using {'full dataset' if use_full_data_for_explanation else 'test set only'} for feature importance analysis")

    # Define pattern for motif features (used throughout the function)
    kmer_pattern = r'^\d+mer_.*'

    # Feature importance analysis
    print_emphasized("[info] Calculating feature importance using XGBoost...")
    result_set['importance_types'] = importance_types
    result_set['xgboost_importance_dfs'] = {}
    
    # Process each importance type
    for importance_type in importance_types:
        print_emphasized(f"[info] Calculating XGBoost feature importance with '{importance_type}'...")
        importance_df = calculate_xgboost_importance(
            model, X_shap, y_shap, importance_type=importance_type
        )
        result_set['xgboost_importance_dfs'][importance_type] = importance_df
        
        # Create and save standardized outputs for XGBoost importance
        if save:
            # Normalize XGBoost importance to include all features
            normalized_xgb_df = normalize_feature_importance_df(
                importance_df, all_feature_names
            )
            
            xgb_results = save_feature_importance_outputs(
                importance_df=normalized_xgb_df,  # Use normalized version
                subject=subject,
                model_name=model_name,
                output_dir=output_dir,
                feature_type=f"xgboost-{importance_type}",
                top_k=top_k,
                motif_pattern=kmer_pattern,
                plot_format=plot_format,
                colormap="viridis",  # Professional colormap
                verbose=verbose
            )
            # Store filtered DataFrames in result_set
            for subset, df in xgb_results.items():
                result_set[f'xgboost_{importance_type}_{subset}'] = df
    
    # SHAP Analysis for all features
    print_emphasized("[info] Calculating feature importance using SHAP (all features)...")
    importance_df_shap = \
        calculate_shap_importance(
            model, 
            X_shap, y_shap, 
            # top_k=top_k,
            fallback_to_xgboost=False,
            model_feature_columns=feature_columns  # Pass the training feature columns
        )
    result_set['importance_df_shap'] = importance_df_shap
    
    # Create and save standardized outputs for SHAP importance
    if save:
        # Normalize SHAP importance to include all features
        normalized_shap_df = normalize_feature_importance_df(
            importance_df_shap, all_feature_names
        )
        
        shap_results = save_feature_importance_outputs(
            importance_df=normalized_shap_df,  # Use normalized version
            subject=subject,
            model_name=model_name,
            output_dir=output_dir,
            feature_type="shap",
            top_k=top_k,
            motif_pattern=kmer_pattern,
            plot_format=plot_format,
            colormap="plasma",  # Professional and distinct colormap
            verbose=verbose
        )
        # Store filtered DataFrames in result_set
        for subset, df in shap_results.items():
            result_set[f'shap_{subset}'] = df
    
    # Calculate other importance metrics
    if kwargs.get('calculate_hypothesis_testing', True):
        print_emphasized("[info] Calculating feature importance using hypothesis testing...")
        importance_df_hypo = calculate_hypothesis_testing(X_shap, y_shap)
        result_set['importance_df_hypo'] = importance_df_hypo
        
        # Create and save standardized outputs for hypothesis testing
        if save:
            hypo_results = save_feature_importance_outputs(
                importance_df=importance_df_hypo,
                subject=subject,
                model_name=model_name,
                output_dir=output_dir,
                feature_type="hypo",
                top_k=top_k,
                motif_pattern=kmer_pattern,
                plot_format=plot_format,
                colormap="coolwarm",  # Professional colormap with contrasting colors
                verbose=verbose
            )
            # Store filtered DataFrames in result_set
            for subset, df in hypo_results.items():
                result_set[f'hypo_{subset}'] = df
            
    if kwargs.get('calculate_mutual_info', True):
        print_emphasized("[info] Calculating feature importance using mutual information...")
        try:
            importance_df_mutual = calculate_mutual_information(X_shap, y_shap)
            result_set['importance_df_mutual'] = importance_df_mutual
            
            # Create and save standardized outputs for mutual information
            if save:
                # Mutual info already includes all features, so no normalization needed
                # (But we could normalize for consistency if needed)
                mutual_results = save_feature_importance_outputs(
                    importance_df=importance_df_mutual,
                    subject=subject,
                    model_name=model_name,
                    output_dir=output_dir,
                    feature_type="mutual",
                    top_k=top_k,
                    motif_pattern=kmer_pattern,
                    plot_format=plot_format,
                    colormap="YlGnBu",  # Professional blue-green colormap
                    verbose=verbose
                )
                # Store filtered DataFrames in result_set
                for subset, df in mutual_results.items():
                    result_set[f'mutual_{subset}'] = df
        except NameError:
            print_emphasized("[warning] calculate_mutual_info function not found, skipping mutual information analysis")
    
    # Calculate effect sizes if enabled
    if kwargs.get('calculate_effect_sizes', True):
        print_emphasized("[info] Calculating feature importance using effect sizes...")
        try:
            importance_df_effect = calculate_effect_sizes(X_shap, y_shap)
            result_set['importance_df_effect'] = importance_df_effect
            
            # Create and save standardized outputs for effect sizes
            if save:
                effect_results = save_feature_importance_outputs(
                    importance_df=importance_df_effect,
                    subject=subject,
                    model_name=model_name,
                    output_dir=output_dir,
                    feature_type="effect",
                    top_k=top_k,
                    motif_pattern=kmer_pattern,
                    plot_format=plot_format,
                    colormap="BuPu",  # Professional purple-blue colormap
                    verbose=verbose
                )
                # Store filtered DataFrames in result_set
                for subset, df in effect_results.items():
                    result_set[f'effect_{subset}'] = df
        except NameError:
            print_emphasized("[warning] calculate_effect_sizes function not found, skipping effect size analysis")
    
    # Motif analysis (if k-mer features present)
    # motif_features = [col for col in X.columns if re.match(kmer_pattern, col)]
    
    # if motif_features and len(motif_features) > 0:
    #     print_emphasized(f"[info] Analyzing {len(motif_features)} motif features...")
        
    #     # Extract feature importance specific to motifs
    #     motif_importance_xgb = importance_df_xgboost[
    #         importance_df_xgboost['feature'].isin(motif_features)
    #     ].head(top_k_motifs)
        
    #     motif_importance_shap = importance_df_shap[
    #         importance_df_shap['feature'].isin(motif_features)
    #     ].head(top_k_motifs)
        
    #     result_set['motif_importance_xgb'] = motif_importance_xgb
    #     result_set['motif_importance_shap'] = motif_importance_shap
        
    #     if save and len(motif_importance_xgb) > 0:
    #         output_path = os.path.join(output_dir, f"{subject}-{model_name}-motif-importance.{plot_format}")
    #         plt.figure(figsize=(12, 8))
            
    #         motif_df = motif_importance_xgb.head(top_k_motifs).sort_values('importance_score')
    #         plt.barh(motif_df['feature'], motif_df['importance_score'])
    #         plt.title(f"Top {top_k_motifs} Motif Features (XGBoost)")
    #         plt.xlabel("Importance Score")
    #         plt.ylabel("Motif Feature")
    #         plt.tight_layout()
    #         plt.savefig(output_path, dpi=300)
    #         if verbose > 0:
    #             print_with_indent(f"[output] Saved motif importance plot to: {output_path}", indent_level=1)

    return model, result_set
