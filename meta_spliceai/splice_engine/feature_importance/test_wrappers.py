#!/usr/bin/env python
"""
Test script for feature importance wrapper functions.

This script creates synthetic datasets with known feature importance relationships,
then tests all feature importance wrapper functions to verify they identify the
important features correctly and provide consistent rankings.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Import the wrapper functions we want to test
from meta_spliceai.splice_engine.feature_importance import (
    calculate_xgboost_importance,
    calculate_shap_importance,
    perform_hypothesis_testing,
    calculate_effect_sizes,
    calculate_mutual_information
)


def create_synthetic_dataset(n_samples=1000, n_features=20, n_informative=5, random_state=42):
    """
    Create a synthetic dataset with known feature importance.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Total number of features
    n_informative : int
        Number of informative features (these should be ranked highest)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    X : pandas.DataFrame
        Feature matrix
    y : numpy.ndarray
        Target vector
    informative_features : list
        List of informative feature names that should be ranked highest
    """
    # Create a synthetic classification dataset
    X_np, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        class_sep=1.0,
        random_state=random_state
    )
    
    # Convert to DataFrame with meaningful feature names
    feature_names = []
    informative_features = []
    
    # Create informative feature names
    for i in range(n_informative):
        name = f"informative_{i}"
        feature_names.append(name)
        informative_features.append(name)
    
    # Create noise feature names
    for i in range(n_features - n_informative):
        feature_names.append(f"noise_{i}")
    
    X = pd.DataFrame(X_np, columns=feature_names)
    
    # Make the importance relationship even more clear
    for i, feature in enumerate(informative_features):
        # Increase separation between classes for informative features
        # The lower the index, the more important the feature (informative_0 is most important)
        X[feature] = X[feature] * (5 - i * 0.5)  # Decreasing importance
    
    return X, y, informative_features


def train_xgboost_model(X, y, random_state=42):
    """
    Train an XGBoost model on the given data.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix
    y : numpy.ndarray
        Target vector
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    model : xgboost.XGBClassifier
        Trained XGBoost model
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Create and train model
    model = xgb.XGBClassifier(
        random_state=random_state,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)
    
    # Print basic model performance
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"XGBoost model accuracy - Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    
    return model


def test_xgboost_importance(model, X, informative_features):
    """Test XGBoost importance wrapper."""
    print("\n=== Testing calculate_xgboost_importance ===")
    
    # Test with 'weight' importance
    importance_df_weight = calculate_xgboost_importance(
        model, X.columns, importance_type='weight'
    )
    print("\nXGBoost Importance (weight):")
    print(importance_df_weight.head(10))
    
    # Test with 'gain' importance
    importance_df_gain = calculate_xgboost_importance(
        model, X.columns, importance_type='gain'
    )
    print("\nXGBoost Importance (gain):")
    print(importance_df_gain.head(10))
    
    # Check if top features match our informative features
    top_features = importance_df_gain.head(len(informative_features))['feature'].tolist()
    found_informative = [f for f in top_features if f in informative_features]
    
    print(f"\nInformative features identified: {len(found_informative)}/{len(informative_features)}")
    print(f"Top 5 features: {top_features[:5]}")
    
    return importance_df_gain


def test_shap_importance(model, X, informative_features):
    """Test SHAP importance wrapper."""
    print("\n=== Testing calculate_shap_importance ===")
    
    # Calculate SHAP importance
    importance_df = calculate_shap_importance(model, X)
    print("\nSHAP Importance:")
    print(importance_df.head(10))
    
    # Check if top features match our informative features
    top_features = importance_df.head(len(informative_features))['feature'].tolist()
    found_informative = [f for f in top_features if f in informative_features]
    
    print(f"\nInformative features identified: {len(found_informative)}/{len(informative_features)}")
    print(f"Top 5 features: {top_features[:5]}")
    
    return importance_df


def test_hypothesis_testing(X, y, informative_features):
    """Test hypothesis testing wrapper."""
    print("\n=== Testing perform_hypothesis_testing ===")
    
    # Perform hypothesis testing
    importance_df = perform_hypothesis_testing(X, y)
    print("\nHypothesis Testing Importance:")
    print(importance_df.head(10))
    
    # Check if top features match our informative features
    top_features = importance_df.head(len(informative_features))['feature'].tolist()
    found_informative = [f for f in top_features if f in informative_features]
    
    print(f"\nInformative features identified: {len(found_informative)}/{len(informative_features)}")
    print(f"Top 5 features: {top_features[:5]}")
    
    return importance_df


def test_effect_sizes(X, y, informative_features):
    """Test effect sizes wrapper."""
    print("\n=== Testing calculate_effect_sizes ===")
    
    # Calculate effect sizes
    importance_df = calculate_effect_sizes(X, y)
    print("\nEffect Sizes Importance:")
    print(importance_df.head(10))
    
    # Check if top features match our informative features
    top_features = importance_df.head(len(informative_features))['feature'].tolist()
    found_informative = [f for f in top_features if f in informative_features]
    
    print(f"\nInformative features identified: {len(found_informative)}/{len(informative_features)}")
    print(f"Top 5 features: {top_features[:5]}")
    
    return importance_df


def test_mutual_information(X, y, informative_features):
    """Test mutual information wrapper."""
    print("\n=== Testing calculate_mutual_information ===")
    
    # Calculate mutual information
    importance_df = calculate_mutual_information(X, y)
    print("\nMutual Information Importance:")
    print(importance_df.head(10))
    
    # Check if top features match our informative features
    top_features = importance_df.head(len(informative_features))['feature'].tolist()
    found_informative = [f for f in top_features if f in informative_features]
    
    print(f"\nInformative features identified: {len(found_informative)}/{len(informative_features)}")
    print(f"Top 5 features: {top_features[:5]}")
    
    return importance_df


def compare_rankings(results, informative_features):
    """
    Compare feature rankings from different methods.
    
    Parameters
    ----------
    results : dict
        Dictionary of dataframes with feature importance results
    informative_features : list
        List of informative feature names
        
    Returns
    -------
    rank_correlation : pandas.DataFrame
        Correlation matrix between different ranking methods
    """
    print("\n=== Comparing Feature Rankings ===")
    
    # Extract top 10 features from each method
    top_features = {}
    for method, df in results.items():
        top_features[method] = df.head(10)['feature'].tolist()
    
    # Calculate overlap with informative features for each method
    for method, features in top_features.items():
        overlap = [f for f in features[:len(informative_features)] if f in informative_features]
        print(f"{method}: {len(overlap)}/{len(informative_features)} informative features in top {len(informative_features)}")
    
    # Create rank dataframe
    rank_df = pd.DataFrame(index=list(set(X.columns)))
    
    # Add ranks for each method
    for method, df in results.items():
        # Convert to ranks (1 = highest importance)
        feature_ranks = pd.Series(
            index=df['feature'],
            data=range(1, len(df) + 1)
        )
        rank_df[method] = feature_ranks
    
    # Fill NaN with max rank + 1
    max_rank = rank_df.max().max() + 1
    rank_df = rank_df.fillna(max_rank)
    
    # Calculate rank correlation
    rank_correlation = rank_df.corr(method='spearman')
    print("\nRank Correlation (Spearman):")
    print(rank_correlation)
    
    return rank_correlation


def visualize_rankings(results, informative_features, output_dir=None):
    """
    Visualize feature importance rankings.
    
    Parameters
    ----------
    results : dict
        Dictionary of dataframes with feature importance results
    informative_features : list
        List of informative feature names
    output_dir : str, optional
        Directory to save plots
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract top 10 features from each method
    for method, df in results.items():
        plt.figure(figsize=(10, 6))
        
        # Get top 10 features
        plot_df = df.head(10).sort_values('importance_score')
        
        # Color bars based on whether they're informative
        colors = ['red' if feature in informative_features else 'blue' 
                  for feature in plot_df['feature']]
        
        # Plot horizontal bar chart
        ax = plt.barh(plot_df['feature'], plot_df['importance_score'], color=colors)
        
        plt.title(f'Top 10 Features by {method}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Informative Feature'),
            Patch(facecolor='blue', label='Noise Feature')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        # Save plot if output directory is provided
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{method}_importance.png"), dpi=300)
            plt.close()
        else:
            plt.show()
    
    # Visualize rank correlation
    rank_correlation = compare_rankings(results, informative_features)
    plt.figure(figsize=(8, 6))
    sns.heatmap(rank_correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Rank Correlation Between Feature Importance Methods')
    plt.tight_layout()
    
    # Save plot if output directory is provided
    if output_dir:
        plt.savefig(os.path.join(output_dir, "rank_correlation.png"), dpi=300)
        plt.close()
    else:
        plt.show()


def run_all_tests():
    """Run all tests and return results."""
    # Create output directory
    output_dir = os.path.join(os.getcwd(), 'feature_importance_test_results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Testing Feature Importance Wrapper Functions ===")
    
    # Create synthetic dataset
    global X, y, informative_features
    X, y, informative_features = create_synthetic_dataset(
        n_samples=1000,
        n_features=20, 
        n_informative=5
    )
    
    print(f"\nCreated synthetic dataset with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Informative features: {informative_features}")
    
    # Train XGBoost model
    model = train_xgboost_model(X, y)
    
    # Test all wrapper functions
    results = {
        'XGBoost': test_xgboost_importance(model, X, informative_features),
        'SHAP': test_shap_importance(model, X, informative_features),
        'Hypothesis Testing': test_hypothesis_testing(X, y, informative_features),
        'Effect Sizes': test_effect_sizes(X, y, informative_features),
        'Mutual Information': test_mutual_information(X, y, informative_features)
    }
    
    # Compare rankings across methods
    compare_rankings(results, informative_features)
    
    # Visualize results
    visualize_rankings(results, informative_features, output_dir)
    
    print(f"\nTest complete. Results saved to: {output_dir}")
    return results


if __name__ == "__main__":
    run_all_tests()
