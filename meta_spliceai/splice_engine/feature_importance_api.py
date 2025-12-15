"""
Feature Importance API Module.

This module provides a clean, unified interface to the feature importance functionality in 
the feature_importance package.

Usage:
------
```python
from meta_spliceai.splice_engine.feature_importance_api import quantify_feature_importance_via_mutual_info
results = quantify_feature_importance_via_mutual_info(X, y, top_features)
```
"""

import warnings
import os
import numpy as np
import pandas as pd


def quantify_feature_importance_via_mutual_info(X, y, top_features=None, *, feature_categories=None, 
                                              verbose=1, output_path=None, **kwargs):
    """
    Quantify feature importance using mutual information.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : array-like
        Target vector.
    top_features : list of str, optional
        List of feature names to focus on. If None, use all features.
    feature_categories : dict, optional
        Dictionary mapping feature categories to feature names.
    verbose : int, default=1
        Controls verbosity of output.
    output_path : str, optional
        Path to save output plots and files.
    **kwargs : dict
        Additional keyword arguments.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with feature importance scores.
    """
    from meta_spliceai.splice_engine.feature_importance.mutual_info import quantify_feature_importance_via_mutual_info as mutual_info_func
    
    return mutual_info_func(
        X, y, 
        top_features=top_features,
        feature_categories=feature_categories,
        verbose=verbose, 
        output_path=output_path,
        **kwargs
    )


def quantify_feature_importance_via_hypothesis_testing(X, y, top_features=None, *, feature_categories=None, 
                                                     verbose=1, output_path=None, **kwargs):
    """
    Quantify feature importance using statistical hypothesis testing.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : array-like
        Target vector.
    top_features : list of str, optional
        List of feature names to focus on. If None, use all features.
    feature_categories : dict, optional
        Dictionary mapping feature categories to feature names.
    verbose : int, default=1
        Controls verbosity of output.
    output_path : str, optional
        Path to save output plots and files.
    **kwargs : dict
        Additional keyword arguments.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with feature importance scores.
    """
    from meta_spliceai.splice_engine.feature_importance.hypothesis_testing import quantify_feature_importance_via_hypothesis_testing as hypo_test_func
    
    return hypo_test_func(
        X, y, 
        top_features=top_features,
        feature_categories=feature_categories,
        verbose=verbose, 
        output_path=output_path,
        **kwargs
    )


def quantify_feature_importance_via_measuring_effect_sizes(X, y, top_features=None, *, feature_categories=None, 
                                                         verbose=1, output_path=None, **kwargs):
    """
    Quantify feature importance via measuring effect sizes.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : array-like
        Target vector.
    top_features : list of str, optional
        List of feature names to focus on. If None, use all features.
    feature_categories : dict, optional
        Dictionary mapping feature categories to feature names.
    verbose : int, default=1
        Controls verbosity of output.
    output_path : str, optional
        Path to save output plots and files.
    **kwargs : dict
        Additional keyword arguments.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with feature importance scores.
    """
    from meta_spliceai.splice_engine.feature_importance.effect_sizes import quantify_feature_importance_via_measuring_effect_sizes as effect_sizes_func
    
    return effect_sizes_func(
        X, y, 
        top_features=top_features,
        feature_categories=feature_categories,
        verbose=verbose, 
        output_path=output_path,
        **kwargs
    )


def compute_shap_values(model, X, feature_names=None, model_type="xgboost", **kwargs):
    """
    Compute SHAP values for a given model and dataset.
    
    Parameters
    ----------
    model : object
        Trained model (e.g., XGBoost, sklearn, etc.)
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix to compute SHAP values for.
    feature_names : list of str, optional
        Names of features, required if X is a numpy array.
    model_type : str, default="xgboost"
        Type of model. Currently supports "xgboost", "sklearn".
    **kwargs : dict
        Additional keyword arguments.
    
    Returns
    -------
    numpy.ndarray
        Array of SHAP values with shape (n_samples, n_features).
    list of str
        List of feature names.
    """
    from meta_spliceai.splice_engine.feature_importance.shap_analysis import compute_shap_values as shap_func
    
    return shap_func(model, X, feature_names=feature_names, model_type=model_type, **kwargs)


def quantify_feature_importance_via_shap(model, X_train, X_test, y_test, top_features=None, 
                                        model_type="xgboost", global_top_k=20, plot_top_k=15,
                                        verbose=1, output_path=None, **kwargs):
    """
    Quantify feature importance using SHAP values.
    
    Parameters
    ----------
    model : object
        Trained model (e.g., XGBoost, sklearn, etc.)
    X_train : pandas.DataFrame
        Training feature matrix.
    X_test : pandas.DataFrame
        Test feature matrix.
    y_test : array-like
        Test target vector.
    top_features : list of str, optional
        List of feature names to focus on. If None, use all features.
    model_type : str, default="xgboost"
        Type of model. Currently supports "xgboost", "sklearn".
    global_top_k : int, default=20
        Number of top features to return.
    plot_top_k : int, default=15
        Number of top features to display in summary plots.
    verbose : int, default=1
        Controls verbosity of output.
    output_path : str, optional
        Path to save output plots and files.
    **kwargs : dict
        Additional keyword arguments.
    
    Returns
    -------
    list of str
        List of top feature names sorted by importance.
    """
    from meta_spliceai.splice_engine.feature_importance.shap_analysis import quantify_feature_importance_via_shap as shap_importance_func
    
    return shap_importance_func(
        model, X_train, X_test, y_test,
        top_features=top_features,
        model_type=model_type,
        global_top_k=global_top_k,
        plot_top_k=plot_top_k,
        verbose=verbose,
        output_path=output_path,
        **kwargs
    )


def analyze_feature_group_importance(shap_values, feature_names, feature_groups, top_k=20, 
                                    verbose=1, output_path=None, **kwargs):
    """
    Analyze importance of feature groups using SHAP values.
    
    Parameters
    ----------
    shap_values : numpy.ndarray
        SHAP values with shape (n_samples, n_features).
    feature_names : list of str
        Names of features.
    feature_groups : dict
        Dictionary mapping group names to lists of feature names.
    top_k : int, default=20
        Number of top groups to display.
    verbose : int, default=1
        Controls verbosity of output.
    output_path : str, optional
        Path to save output plots and files.
    **kwargs : dict
        Additional keyword arguments.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with group importance scores.
    """
    from meta_spliceai.splice_engine.feature_importance.shap_analysis import analyze_feature_group_importance as group_importance_func
    
    return group_importance_func(
        shap_values, feature_names, feature_groups,
        top_k=top_k,
        verbose=verbose,
        output_path=output_path,
        **kwargs
    )


def identify_important_motifs_from_shap(X_test, shap_values, motif_columns, top_n_motifs=20, 
                                       verbose=1, return_scores=True, return_full_scores=False):
    """
    Identify important sequence motifs from SHAP values.
    
    Parameters
    ----------
    X_test : pandas.DataFrame
        Test feature matrix.
    shap_values : numpy.ndarray
        SHAP values with shape (n_samples, n_features).
    motif_columns : list of str
        List of column names that represent motifs.
    top_n_motifs : int, default=20
        Number of top motifs to return.
    verbose : int, default=1
        Controls verbosity of output.
    return_scores : bool, default=True
        Whether to return importance scores.
    return_full_scores : bool, default=False
        Whether to return scores for all motifs.
    
    Returns
    -------
    pandas.DataFrame or list
        DataFrame with motif importance scores, or list of top motifs.
    """
    from meta_spliceai.splice_engine.feature_importance.shap_analysis import identify_important_motifs_from_shap as motifs_func
    
    return motifs_func(
        X_test, shap_values, motif_columns,
        top_n_motifs=top_n_motifs,
        verbose=verbose,
        return_scores=return_scores,
        return_full_scores=return_full_scores
    )


def compute_global_feature_importance_from_shap(shap_values, feature_names, top_k=20):
    """
    Compute global feature importance from SHAP values.
    
    Parameters
    ----------
    shap_values : numpy.ndarray
        SHAP values with shape (n_samples, n_features).
    feature_names : list of str
        Names of features.
    top_k : int, default=20
        Number of top features to return.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with feature importance scores.
    list of str
        List of top feature names sorted by importance.
    """
    from meta_spliceai.splice_engine.feature_importance.shap_analysis import compute_global_feature_importance_from_shap as shap_importance_func
    
    return shap_importance_func(shap_values, feature_names, top_k=top_k)


def get_xgboost_feature_importance(model, feature_names=None, importance_type='gain',
                                 top_k=20, verbose=1, output_path=None):
    """
    Extract feature importance directly from an XGBoost model.
    
    Parameters
    ----------
    model : xgboost.Booster or xgboost.XGBClassifier
        Trained XGBoost model.
    feature_names : list of str, optional
        Names of features. If None, will be extracted from model if possible.
    importance_type : str, default='gain'
        Type of importance metric. Options include 'gain', 'weight', 'cover', etc.
    top_k : int, default=20
        Number of top features to return.
    verbose : int, default=1
        Controls verbosity of output.
    output_path : str, optional
        Path to save output plots and files.
    
    Returns
    -------
    list of str
        List of top feature names sorted by importance.
    """
    from meta_spliceai.splice_engine.feature_importance.xgboost_importance import get_xgboost_feature_importance as xgb_importance_func
    
    return xgb_importance_func(
        model, 
        feature_names=feature_names,
        importance_type=importance_type,
        top_k=top_k,
        verbose=verbose,
        output_path=output_path
    )
