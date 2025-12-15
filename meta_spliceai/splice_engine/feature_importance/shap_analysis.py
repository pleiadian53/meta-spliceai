"""
SHAP-based Feature Importance Module.

This module provides functions to quantify feature importance using SHAP (SHapley Additive exPlanations)
values, which provide a unified measure of feature importance with game-theoretic foundations.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from collections import defaultdict, Counter


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
        Additional arguments to pass to the SHAP explainer.
    
    Returns
    -------
    numpy.ndarray
        Array of SHAP values with shape (n_samples, n_features).
    list of str
        List of feature names.
    """
    # Convert X to DataFrame if it's a numpy array
    if isinstance(X, np.ndarray) and feature_names is not None:
        X = pd.DataFrame(X, columns=feature_names)
    
    # Extract feature names
    if feature_names is None:
        feature_names = X.columns.tolist()
    
    # Select appropriate explainer based on model type
    if model_type.lower() == "xgboost":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # For binary classification, shap_values might be a list with one element
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
    elif model_type.lower() == "sklearn":
        # For sklearn models that are tree-based
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # For binary classification, shap_values might be a list with one element
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
        else:
            # For other sklearn models, use KernelExplainer
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))
            shap_values = explainer.shap_values(X)[1]  # Class 1 explanation
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    return shap_values, feature_names


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
    # Calculate mean absolute SHAP values for each feature
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame with results
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_score': feature_importance
    }).sort_values('importance_score', ascending=False)
    
    # Get top K features
    top_features = importance_df['feature'].tolist()[:top_k]
    
    return importance_df, top_features


def analyze_feature_group_importance(shap_values, feature_names, feature_groups, top_k=20, 
                                    verbose=1, output_path=None, fig_size=(12, 8), **kargs):
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
    fig_size : tuple, default=(12, 8)
        Figure size for plots.
    **kargs : dict
        Additional keyword arguments.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with group importance scores.
    """
    # Create feature to index mapping
    feature_to_idx = {feat: idx for idx, feat in enumerate(feature_names)}
    
    # Initialize group importance dictionary
    group_importance = defaultdict(float)
    
    # For each group, sum the absolute SHAP values of its features
    for group_name, group_features in feature_groups.items():
        group_indices = [feature_to_idx[feat] for feat in group_features if feat in feature_to_idx]
        if group_indices:
            # Sum absolute SHAP values for all features in the group
            group_shap = np.abs(shap_values[:, group_indices]).mean()
            group_importance[group_name] = group_shap
    
    # Create DataFrame with results
    group_imp_df = pd.DataFrame({
        'feature_group': list(group_importance.keys()),
        'importance_score': list(group_importance.values()),
        'n_features': [len(feature_groups[group]) for group in group_importance.keys()]
    }).sort_values('importance_score', ascending=False)
    
    # Normalize by number of features in each group (optional)
    group_imp_df['importance_score_per_feature'] = group_imp_df['importance_score'] / group_imp_df['n_features']
    
    # Print results if verbose
    if verbose >= 1:
        print("\nFeature Group Importance via SHAP:")
        print(group_imp_df.head(top_k))
    
    # Create and save plot if output_path is provided
    if output_path is not None:
        plt.figure(figsize=fig_size)
        sns.barplot(
            x='importance_score', 
            y='feature_group', 
            hue='feature_group',
            data=group_imp_df.head(top_k),
            legend=False)
        plt.title('Feature Group Importance via SHAP')
        plt.tight_layout()
        
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, 'feature_group_importance_shap.pdf'), dpi=300)
        plt.savefig(os.path.join(output_path, 'feature_group_importance_shap.png'), dpi=300)
        
        if verbose >= 1:
            plt.show()
        plt.close()
        
        # Save results to CSV
        group_imp_df.to_csv(os.path.join(output_path, 'feature_group_importance_shap.csv'), index=False)
    
    return group_imp_df


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
    # Extract SHAP values for motif-specific features
    motif_indices = [X_test.columns.get_loc(col) for col in motif_columns]
    motif_shap_values = shap_values[:, motif_indices]

    # Calculate mean absolute SHAP values for motif-specific features
    feature_importance = np.abs(motif_shap_values).mean(axis=0)

    # Create a full DataFrame with all motifs and their scores
    full_motifs_df = pd.DataFrame({
        'motif': motif_columns,
        'importance_score': feature_importance
    }).sort_values(by='importance_score', ascending=False)

    # Rank motifs by importance
    top_indices = np.argsort(feature_importance)[-top_n_motifs:]  # Indices of top N motifs
    top_motifs = [motif_columns[i] for i in top_indices]
    top_motif_scores = feature_importance[top_indices]

    if verbose > 0:
        print(f"[info] Top {top_n_motifs} important motif-specific features: {top_motifs}")

    if return_scores:
        # Create a DataFrame to return for top motifs
        top_motifs_df = pd.DataFrame({
            'motif': top_motifs,
            'importance_score': top_motif_scores
        }).sort_values(by='importance_score', ascending=False)

        if return_full_scores:
            return top_motifs_df, full_motifs_df
        return top_motifs_df
    
    return top_motifs


def quantify_feature_importance_via_shap(model, X_train, X_test, y_test, top_features=None,
                                        model_type="xgboost", global_top_k=20, plot_top_k=15,
                                        verbose=1, output_path=None, fig_size=(12, 8), **kargs):
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
    fig_size : tuple, default=(12, 8)
        Figure size for plots.
    **kargs : dict
        Additional keyword arguments.
    
    Returns
    -------
    list of str
        List of top feature names sorted by importance.
    """
    if verbose >= 1:
        print(f"Quantifying feature importance via SHAP...")
    
    # Subset features if top_features is provided
    if top_features is not None:
        X_test_subset = X_test[top_features].copy()
        feature_names = top_features
    else:
        X_test_subset = X_test.copy()
        feature_names = X_test.columns.tolist()
    
    # Compute SHAP values
    shap_values, _ = compute_shap_values(
        model=model,
        X=X_test_subset,
        feature_names=feature_names,
        model_type=model_type
    )
    
    # Compute global feature importance
    importance_df, top_features = compute_global_feature_importance_from_shap(
        shap_values=shap_values,
        feature_names=feature_names,
        top_k=global_top_k
    )
    
    # Print results if verbose
    if verbose >= 1:
        print("\nFeature Importance via SHAP:")
        print(importance_df.head(global_top_k))
    
    # Create and save plots if output_path is provided
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        
        # Bar plot of feature importance
        plt.figure(figsize=fig_size)
        sns.barplot(
            x='importance_score', 
            y='feature', 
            hue='feature',
            data=importance_df.head(plot_top_k),
            legend=False)
        plt.title('Feature Importance via SHAP')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'feature_importance_shap_bar.pdf'), dpi=300)
        plt.savefig(os.path.join(output_path, 'feature_importance_shap_bar.png'), dpi=300)
        if verbose >= 1:
            plt.show()
        plt.close()
        
        # SHAP summary plot (requires matplotlib to be non-interactive)
        plt.figure(figsize=fig_size)
        shap.summary_plot(
            shap_values=shap_values,
            features=X_test_subset,
            feature_names=feature_names,
            max_display=plot_top_k,
            show=False
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'feature_importance_shap_summary.pdf'), dpi=300)
        plt.savefig(os.path.join(output_path, 'feature_importance_shap_summary.png'), dpi=300)
        if verbose >= 1:
            plt.show()
        plt.close()
        
        # Dependency plots for top features
        for feature in top_features[:5]:  # Limit to top 5 to avoid too many plots
            plt.figure(figsize=fig_size)
            feature_idx = feature_names.index(feature)
            shap.dependence_plot(
                ind=feature_idx,
                shap_values=shap_values,
                features=X_test_subset,
                feature_names=feature_names,
                show=False
            )
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f'shap_dependence_{feature}.pdf'), dpi=300)
            plt.savefig(os.path.join(output_path, f'shap_dependence_{feature}.png'), dpi=300)
            if verbose >= 1:
                plt.show()
            plt.close()
        
        # Save results to CSV
        importance_df.to_csv(os.path.join(output_path, 'feature_importance_shap.csv'), index=False)
    
    return top_features
