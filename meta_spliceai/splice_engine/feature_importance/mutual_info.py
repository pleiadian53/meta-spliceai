"""
Mutual Information Feature Importance Module.

This module provides functions to quantify feature importance using mutual information,
which measures how much knowing the value of one variable reduces uncertainty about another.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import os
import matplotlib.pyplot as plt
import seaborn as sns

def compute_feature_mutual_info(X, y, top_features, *, feature_categories=None, verbose=1, output_path=None, fig_size=(12, 8), **kargs):
    """
    Compute mutual information scores between features and target variable.
    Properly handles both numerical and categorical features.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    top_features : list of str
        List of feature names to analyze.
    feature_categories : dict, optional
        Dictionary mapping feature types to lists of feature names.
        Keys should include 'numerical_features' and 'categorical_features'.
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
        List of feature names sorted by importance.
    pd.DataFrame
        DataFrame with feature importance scores.
    """
    # Filter X to only include top_features
    X_subset = X[top_features].copy()
    
    # Skip features with NaN values
    valid_features = [feature for feature in top_features if not X_subset[feature].isnull().any()]
    X_subset = X_subset[valid_features]
    
    if len(valid_features) == 0:
        if verbose >= 1:
            print("No valid features found after filtering NaN values.")
        return [], pd.DataFrame({'feature': [], 'importance_score': []})
    
    # Determine feature types if not provided
    if feature_categories is None:
        # Simple heuristic to identify categorical features
        categorical_features = []
        numerical_features = []
        
        for feature in valid_features:
            if X_subset[feature].dtype == 'object' or X_subset[feature].dtype.name == 'category' or X_subset[feature].nunique() < 10:
                categorical_features.append(feature)
            else:
                numerical_features.append(feature)
    else:
        # Use provided feature categories but filter for valid features only
        all_categorical_features = feature_categories.get('categorical_features', []) + \
                                  feature_categories.get('derived_categorical_features', [])
        categorical_features = [f for f in all_categorical_features if f in valid_features]
        numerical_features = [f for f in feature_categories.get('numerical_features', []) if f in valid_features]
    
    # Create discrete_mask for mutual_info_classif
    discrete_mask = np.zeros(len(valid_features), dtype=bool)
    
    # Convert categorical features to codes
    for i, feature in enumerate(valid_features):
        if feature in categorical_features:
            # Mark as discrete feature
            discrete_mask[i] = True
            # Convert categorical variables to numeric codes for mutual_info_classif
            if X_subset[feature].dtype == 'object' or X_subset[feature].dtype.name == 'category':
                X_subset[feature] = X_subset[feature].astype('category').cat.codes
    
    try:
        # Compute mutual information scores
        mi_scores = mutual_info_classif(X_subset, y, discrete_features=discrete_mask, random_state=42)
        
        # Create DataFrame with results
        importance_df = pd.DataFrame({
            'feature': valid_features,
            'importance_score': mi_scores,
            'feature_type': ['categorical' if f in categorical_features else 'numerical' for f in valid_features]
        }).sort_values('importance_score', ascending=False)
        
        # Print results if verbose
        if verbose >= 1:
            print("\nFeature Importance via Mutual Information:")
            print(importance_df.head(20))
        
        # Create and save plot if output_path is provided
        if output_path is not None:
            plt.figure(figsize=fig_size)
            bar_plot = sns.barplot(
                x='importance_score', 
                y='feature', 
                hue="feature",  
                data=importance_df.head(20), 
                legend=False)
            
            # Add feature type as text annotation
            for i, row in enumerate(importance_df.head(20).itertuples()):
                bar_plot.text(row.importance_score / 2, i, row.feature_type, ha='center', va='center', color='black')
                
            plt.title('Feature Importance via Mutual Information')
            plt.tight_layout()
            
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, 'feature_importance_mutual_info.pdf'), dpi=300)
            # plt.savefig(os.path.join(output_path, 'feature_importance_mutual_info.png'), dpi=300)
            
            if verbose >= 1:
                plt.show()
            plt.close()
            
            # Save results to CSV
            importance_df.to_csv(os.path.join(output_path, 'feature_importance_mutual_info.csv'), index=False)
        
        # Return list of feature names sorted by importance
        return importance_df['feature'].tolist(), importance_df
        
    except Exception as e:
        if verbose >= 1:
            print(f"Error computing mutual information: {e}")
            print("Attempting to compute MI for numerical features only...")
            
        # Fall back to numerical features only if there's an error
        numerical_subset = X_subset[numerical_features].copy()
        
        if numerical_subset.shape[1] == 0:
            # No numerical features to use
            if verbose >= 1:
                print("No numerical features available for fallback.")
            return [], pd.DataFrame({'feature': [], 'importance_score': []})
            
        # Compute mutual information scores for numerical features only
        mi_scores = mutual_info_classif(numerical_subset, y, discrete_features=False, random_state=42)
        
        # Create DataFrame with results
        importance_df = pd.DataFrame({
            'feature': numerical_features,
            'importance_score': mi_scores,
            'feature_type': ['numerical'] * len(numerical_features)
        }).sort_values('importance_score', ascending=False)
        
        # Print results if verbose
        if verbose >= 1:
            print("\nFeature Importance via Mutual Information (numerical features only):")
            print(importance_df.head(20))
            
        # Create and save plot if output_path is provided
        if output_path is not None:
            plt.figure(figsize=fig_size)
            sns.barplot(
                x='importance_score', 
                y='feature', 
                hue="feature",  
                data=importance_df.head(20), 
                legend=False)
            plt.title('Feature Importance via Mutual Information (numerical features only)')
            plt.tight_layout()
            
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, 'feature_importance_mutual_info_numerical.pdf'), dpi=300)
            plt.savefig(os.path.join(output_path, 'feature_importance_mutual_info_numerical.png'), dpi=300)
            
            if verbose >= 1:
                plt.show()
            plt.close()
            
            # Save results to CSV
            importance_df.to_csv(os.path.join(output_path, 'feature_importance_mutual_info_numerical.csv'), index=False)
        
        # Return list of feature names sorted by importance
        return importance_df['feature'].tolist(), importance_df


def quantify_feature_importance_via_mutual_info(X, y, top_features, *, feature_categories=None, verbose=1, output_path=None, **kargs):
    """
    Quantify feature importance using mutual information.
    
    This provides a unified approach to measure feature importance for both categorical and numerical features
    on the same scale. Mutual information measures how much knowing one variable reduces uncertainty about another.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    top_features : list of str
        List of feature names to analyze.
    feature_categories : dict, optional
        Dictionary mapping feature names to their categories.
    verbose : int, default=1
        Controls verbosity of output.
    output_path : str, optional
        Path to save output plots and files.
    **kargs : dict
        Additional keyword arguments.
        
    Returns
    -------
    list of str
        List of feature names sorted by importance.
    """
    # Call the compute_feature_mutual_info function with all parameters
    top_features_sorted, _ = compute_feature_mutual_info(
        X, y, top_features, 
        feature_categories=feature_categories,
        verbose=verbose, 
        output_path=output_path,
        **kargs
    )
    
    return top_features_sorted
