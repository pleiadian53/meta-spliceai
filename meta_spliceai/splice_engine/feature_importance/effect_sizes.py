"""
Effect Size Feature Importance Module.

This module provides functions to quantify feature importance by measuring
practical effect sizes between classes for each feature.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency


def compute_effect_sizes(X, y, top_features, *, feature_categories=None, verbose=1, output_path=None, fig_size=(12, 8), **kargs):
    """
    Compute effect sizes for features to quantify importance.
    Handles different feature types appropriately:
    - Numeric features: Cohen's d
    - Categorical features: Cramer's V
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector (binary).
    top_features : list of str
        List of feature names to analyze.
    feature_categories : dict, optional
        Dictionary mapping feature names to their categories.
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
        List of feature names sorted by effect size.
    pd.DataFrame
        DataFrame with feature effect sizes.
    """
    # Convert y to numpy array if it's not already
    y_np = np.array(y)
    
    # Initialize results dictionary
    results = {
        'feature': [],
        'effect_size': [],  # Keep the original domain-specific metric name
        'effect_type': [],
        'p_value': []
    }
    
    # Determine feature types if not provided
    if feature_categories is None:
        # Simple heuristic to identify categorical features
        categorical_features = []
        numerical_features = []
        
        for feature in top_features:
            if X[feature].dtype == 'object' or X[feature].dtype.name == 'category' or X[feature].nunique() < 10:
                categorical_features.append(feature)
            else:
                numerical_features.append(feature)
    else:
        # Use provided feature categories
        categorical_features = feature_categories.get('categorical_features', []) + \
                              feature_categories.get('derived_categorical_features', [])
        numerical_features = feature_categories.get('numerical_features', [])
    
    # Calculate effect size for each feature
    for feature in top_features:
        # Skip if feature not in DataFrame
        if feature not in X.columns:
            continue
            
        # Skip features with NaN values
        if X[feature].isnull().any():
            if verbose >= 2:
                print(f"Skipping {feature} due to NaN values")
            continue
            
        if feature in numerical_features:
            # Handle numerical features with Cohen's d
            try:
                # Extract feature values
                feature_values = X[feature].values
                
                # Split by class
                group1 = feature_values[y_np == 1]  # Error class
                group0 = feature_values[y_np == 0]  # Correct class
                
                # Skip if not enough samples
                if len(group1) < 2 or len(group0) < 2:
                    if verbose >= 2:
                        print(f"Skipping {feature} due to insufficient samples")
                    continue
                
                # Calculate effect size (Cohen's d)
                mean1, mean0 = np.mean(group1), np.mean(group0)
                std1, std0 = np.std(group1, ddof=1), np.std(group0, ddof=1)
                
                # Pooled standard deviation
                n1, n0 = len(group1), len(group0)
                pooled_std = np.sqrt(((n1-1) * std1**2 + (n0-1) * std0**2) / (n1 + n0 - 2))
                
                # Cohen's d
                if pooled_std == 0:
                    effect_size = 0  # Avoid division by zero
                else:
                    effect_size = np.abs((mean1 - mean0) / pooled_std)
                
                # Calculate p-value (t-test)
                _, p_value = stats.ttest_ind(group1, group0, equal_var=False)
                
                # Add to results
                results['feature'].append(feature)
                results['effect_size'].append(effect_size)
                results['effect_type'].append("Cohen's d")
                results['p_value'].append(p_value)
            except Exception as e:
                if verbose >= 2:
                    print(f"Error processing numerical feature {feature}: {e}")
                continue
                
        elif feature in categorical_features:
            # Handle categorical features with Cramer's V
            try:
                # Create contingency table
                contingency = pd.crosstab(X[feature], y)
                
                # Skip if contingency table is not valid
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    if verbose >= 2:
                        print(f"Skipping {feature}: insufficient categories or class representation")
                    continue
                
                # Calculate chi-square
                chi2, p_value, _, _ = chi2_contingency(contingency)
                
                # Calculate Cramer's V
                n = contingency.sum().sum()
                phi2 = chi2 / n
                r, k = contingency.shape
                cramer_v = np.sqrt(phi2 / (min(k-1, r-1)))
                
                # Add to results
                results['feature'].append(feature)
                results['effect_size'].append(cramer_v)
                results['effect_type'].append("Cramer's V")
                results['p_value'].append(p_value)
            except Exception as e:
                if verbose >= 2:
                    print(f"Error processing categorical feature {feature}: {e}")
                continue
    
    # Create a DataFrame from the results
    effect_sizes_df = pd.DataFrame(results)
    
    # Sort by effect size
    if not effect_sizes_df.empty:
        # Add importance_score column that mirrors effect_size for consistency with other modules
        effect_sizes_df['importance_score'] = effect_sizes_df['effect_size']
        effect_sizes_df = effect_sizes_df.sort_values('effect_size', ascending=False)
    
    # Print results if verbose
    if verbose >= 1 and not effect_sizes_df.empty:
        print("\nFeature Importance via Effect Sizes:")
        print(effect_sizes_df.head(20))
    
    # Create and save plot if output_path is provided
    if output_path is not None and not effect_sizes_df.empty:
        plt.figure(figsize=fig_size)
        sns.barplot(
            x='importance_score', 
            y='feature', 
            hue="feature",  
            data=effect_sizes_df.head(20), 
            legend=False)
        plt.title('Feature Importance via Effect Sizes')
        plt.tight_layout()
        
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, 'feature_importance_effect_sizes.pdf'), dpi=300)
        plt.savefig(os.path.join(output_path, 'feature_importance_effect_sizes.png'), dpi=300)
        
        if verbose >= 1:
            plt.show()
        plt.close()
        
        # Save results to CSV
        effect_sizes_df.to_csv(os.path.join(output_path, 'feature_importance_effect_sizes.csv'), index=False)
    
    # Return a list of features sorted by effect size and the DataFrame
    feature_list = effect_sizes_df['feature'].tolist() if not effect_sizes_df.empty else []
    return feature_list, effect_sizes_df


def quantify_feature_importance_via_measuring_effect_sizes(X, y, top_features, *, feature_categories=None, verbose=1, output_path=None, **kargs):
    """
    Quantify feature importance by measuring practical effect sizes.
    
    This method applies appropriate effect size measures based on feature types:
    - Cohen's d for numerical features
    - Cramer's V for categorical features
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector (binary).
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
        List of feature names sorted by effect size.
    """
    # Call the compute_effect_sizes function with all parameters
    top_features_sorted, _ = compute_effect_sizes(
        X, y, top_features, 
        feature_categories=feature_categories,
        verbose=verbose, 
        output_path=output_path,
        **kargs
    )
    
    return top_features_sorted
