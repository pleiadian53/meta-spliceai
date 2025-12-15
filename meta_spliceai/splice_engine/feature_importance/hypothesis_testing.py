"""
Hypothesis Testing Feature Importance Module.

This module provides functions to quantify feature importance using statistical
hypothesis testing approaches such as t-tests for continuous features and chi-squared
tests for categorical features.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency


def compute_hypothesis_tests(X, y, top_features, *, feature_categories=None, verbose=1, 
                             output_path=None, fig_size=(12, 8), **kargs):
    """
    Compute statistical significance of features using hypothesis tests.
    Uses appropriate tests based on feature type:
    - Numerical features: t-tests
    - Categorical features: chi-square tests
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector (binary).
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
        List of feature names sorted by statistical significance.
    pd.DataFrame
        DataFrame with feature test statistics and p-values.
    """
    # Convert y to numpy array if it's not already
    y_np = np.array(y)
    
    # Initialize results dictionary
    results = {
        'feature': [],
        'test_statistic': [],
        'p_value': [],
        'test_type': []
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
    
    # Run hypothesis tests for each feature
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
            # Use t-test for numerical features
            try:
                # Extract feature values for each class
                group1 = X.loc[y_np == 1, feature].values  # Error class
                group0 = X.loc[y_np == 0, feature].values  # Correct class
                
                # Skip if not enough samples
                if len(group1) < 2 or len(group0) < 2:
                    if verbose >= 2:
                        print(f"Skipping {feature} due to insufficient samples")
                    continue
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(group1, group0, equal_var=False)
                
                # Add to results
                results['feature'].append(feature)
                results['test_statistic'].append(np.abs(t_stat))  # Use absolute t-statistic
                results['p_value'].append(p_value)
                results['test_type'].append('t-test')
            except Exception as e:
                if verbose >= 2:
                    print(f"Error performing t-test for {feature}: {e}")
                continue
                
        elif feature in categorical_features:
            # Use chi-square test for categorical features
            try:
                # Create contingency table
                contingency = pd.crosstab(X[feature], y)
                
                # Skip if contingency table is not valid
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    if verbose >= 2:
                        print(f"Skipping {feature}: insufficient categories or class representation")
                    continue
                
                # Perform chi-square test
                chi2, p_value, _, _ = chi2_contingency(contingency)
                
                # Add to results
                results['feature'].append(feature)
                results['test_statistic'].append(chi2)
                results['p_value'].append(p_value)
                results['test_type'].append('chi-square')
            except Exception as e:
                if verbose >= 2:
                    print(f"Error performing chi-square test for {feature}: {e}")
                continue
    
    # Convert to DataFrame
    significance_df = pd.DataFrame(results)
    
    # Add -log10(p) column for better visualization of significance
    significance_df['-log10(p)'] = -np.log10(significance_df['p_value'].clip(1e-10))  # Clip for numerical stability
    
    # Add importance_score column for consistency with other feature importance functions
    # Use -log10(p) as the importance score while keeping the original statistics
    significance_df['importance_score'] = significance_df['-log10(p)']
    
    # Sort by statistical significance (ascending p-values)
    significance_df = significance_df.sort_values('p_value', ascending=True)
    
    # Print results if verbose
    if verbose >= 1 and not significance_df.empty:
        print("\nFeature Importance via Statistical Tests:")
        print(significance_df.head(20))
    
    # Create and save plot if output_path is provided
    if output_path is not None and not significance_df.empty:
        plot_df = significance_df.head(20).copy()
        
        plt.figure(figsize=fig_size)
        bar_plot = sns.barplot(
            x='importance_score', 
            y='feature', 
            hue="feature",  
            data=plot_df, 
            legend=False)
        
        # Add test type as text annotation
        for i, row in enumerate(plot_df.itertuples()):
            bar_plot.text(0.5, i, row.test_type, ha='left', va='center', color='black')
            
        plt.title('Feature Importance via Statistical Tests (-log10 p-value)')
        plt.tight_layout()
        
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, 'feature_importance_hypothesis_testing.pdf'), dpi=300)
        plt.savefig(os.path.join(output_path, 'feature_importance_hypothesis_testing.png'), dpi=300)
        
        if verbose >= 1:
            plt.show()
        plt.close()
        
        # Save results to CSV
        significance_df.to_csv(os.path.join(output_path, 'feature_importance_hypothesis_testing.csv'), index=False)
    
    # Return a list of features sorted by significance and the DataFrame
    feature_list = significance_df['feature'].tolist() if not significance_df.empty else []
    return feature_list, significance_df


def quantify_feature_importance_via_hypothesis_testing(X, y, top_features, *, feature_categories=None, 
                                                      verbose=1, output_path=None, **kargs):
    """
    Quantify feature importance using statistical hypothesis testing.
    
    This method applies appropriate statistical tests based on feature types:
    - T-tests for continuous features
    - Chi-squared tests for categorical features
    
    Features are ranked by statistical significance (p-value).
    
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
        List of feature names sorted by statistical significance.
    """
    # Call the compute_hypothesis_tests function with all parameters
    top_features_sorted, _ = compute_hypothesis_tests(
        X, y, top_features, 
        feature_categories=feature_categories,
        verbose=verbose, 
        output_path=output_path,
        **kargs
    )
    
    return top_features_sorted
