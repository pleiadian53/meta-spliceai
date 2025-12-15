"""
XGBoost-specific Feature Importance Module.

This module provides functions to extract native feature importance directly from
XGBoost models, which offer built-in importance metrics.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def get_xgboost_feature_importance(model, feature_names=None, importance_type='gain',
                                  top_k=20, verbose=1, output_path=None, fig_size=(12, 8), **kargs):
    """
    Extract feature importance directly from an XGBoost model.
    
    Parameters
    ----------
    model : xgboost.Booster or xgboost.XGBClassifier
        Trained XGBoost model.
    feature_names : list of str, optional
        Names of features. If None, will be extracted from model if possible.
    importance_type : str, default='gain'
        Type of importance metric. Options include:
        - 'gain': Average gain of splits that use the feature
        - 'weight': Number of times the feature is used in trees
        - 'cover': Average coverage of splits that use the feature
        - 'total_gain': Total gain of splits that use the feature
        - 'total_cover': Total coverage of splits that use the feature
    top_k : int, default=20
        Number of top features to return.
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
    pandas.DataFrame
        DataFrame with feature importance scores.
    """
    # Extract feature importance
    try:
        # For scikit-learn API (XGBClassifier, XGBRegressor)
        if hasattr(model, 'get_booster'):
            booster = model.get_booster()
            importance = booster.get_score(importance_type=importance_type)
            
            # If feature_names is None, try to get from model
            if feature_names is None and hasattr(model, 'feature_names_'):
                feature_names = model.feature_names_
        else:
            # For native API (Booster)
            importance = model.get_score(importance_type=importance_type)
    except Exception as e:
        if verbose >= 1:
            print(f"Error extracting XGBoost feature importance: {e}")
            print("Falling back to feature_importances_ attribute...")
        
        # Fall back to feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            if feature_names is None:
                if hasattr(model, 'feature_names_'):
                    feature_names = model.feature_names_
                else:
                    feature_names = [f"f{i}" for i in range(len(model.feature_importances_))]
            
            importance = {name: imp for name, imp in zip(feature_names, model.feature_importances_)}
        else:
            raise ValueError("Could not extract feature importance from model")
    
    # Create DataFrame with results
    if not importance:
        if verbose >= 1:
            print("No feature importance found in the model")
        return [], pd.DataFrame(columns=['feature', 'importance_score'])
    
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance_score': list(importance.values())
    }).sort_values('importance_score', ascending=False)
    
    # Print results if verbose
    if verbose >= 1:
        print(f"\nXGBoost Feature Importance (type={importance_type}):")
        print(importance_df.head(top_k))
    
    # Create and save plot if output_path is provided
    if output_path is not None:
        plt.figure(figsize=fig_size)
        sns.barplot(
            x='importance_score', 
            y='feature', 
            hue='feature',
            data=importance_df.head(top_k),
            legend=False)
        plt.title(f'XGBoost Feature Importance ({importance_type})')
        plt.tight_layout()
        
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f'feature_importance_xgboost_{importance_type}.pdf'), dpi=300)
        plt.savefig(os.path.join(output_path, f'feature_importance_xgboost_{importance_type}.png'), dpi=300)
        
        if verbose >= 1:
            plt.show()
        plt.close()
        
        # Save results to CSV
        importance_df.to_csv(os.path.join(output_path, f'feature_importance_xgboost_{importance_type}.csv'), index=False)
    
    return importance_df['feature'].tolist()[:top_k], importance_df
