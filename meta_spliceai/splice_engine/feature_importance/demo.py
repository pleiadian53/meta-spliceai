"""
Feature Importance Demonstration Script.

This script demonstrates how to use the feature importance modules with a synthetic dataset.
It shows a complete workflow from data generation to feature importance analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import utilities for mock data generation
from meta_spliceai.splice_engine.feature_importance.demo_utils import (
    generate_mock_classification_data,
    train_test_split_mock_data,
    train_xgboost_model,
    evaluate_model
)

# Import feature importance methods
from meta_spliceai.splice_engine.feature_importance import (
    quantify_feature_importance_via_mutual_info,
    quantify_feature_importance_via_measuring_effect_sizes,
    quantify_feature_importance_via_hypothesis_testing,
    quantify_feature_importance_via_shap,
    get_xgboost_feature_importance
)


def run_feature_importance_demo(output_dir='./feature_importance_demo_results'):
    """
    Run a complete feature importance analysis demo.
    
    Parameters
    ----------
    output_dir : str, default='./feature_importance_demo_results'
        Directory to save output files.
    """
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS DEMONSTRATION")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate synthetic data
    print("\nStep 1: Generating synthetic classification data...")
    X, y, feature_names, feature_categories = generate_mock_classification_data(
        n_samples=1000,
        n_features=20,
        n_informative=5,
        n_redundant=3,
        class_sep=1.5
    )
    
    print(f"Generated dataset with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Step 2: Split data into train and test sets
    print("\nStep 2: Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split_mock_data(X, y, test_size=0.3)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 3: Train a model
    print("\nStep 3: Training an XGBoost model...")
    model = train_xgboost_model(X_train, y_train)
    
    # Step 4: Evaluate the model
    print("\nStep 4: Evaluating the model...")
    metrics = evaluate_model(model, X_test, y_test)
    print("Model performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save metrics to file
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'model_metrics.csv'), index=False)
    
    # Step 5: Feature importance analysis
    print("\nStep 5: Feature importance analysis...")
    
    # Step 5.1: Mutual Information
    print("\n5.1 Mutual Information Analysis")
    mi_features, mi_df = quantify_feature_importance_via_mutual_info(
        X=X_test,
        y=y_test,
        top_features=feature_names,
        feature_categories=feature_categories,
        verbose=1,
        output_path=os.path.join(output_dir, 'mutual_info')
    )
    
    # Step 5.2: Effect Sizes
    print("\n5.2 Effect Size Analysis")
    es_features, es_df = quantify_feature_importance_via_measuring_effect_sizes(
        X=X_test,
        y=y_test,
        top_features=feature_names,
        verbose=1,
        output_path=os.path.join(output_dir, 'effect_sizes')
    )
    
    # Step 5.3: Hypothesis Testing
    print("\n5.3 Hypothesis Testing Analysis")
    ht_features, ht_df = quantify_feature_importance_via_hypothesis_testing(
        X=X_test,
        y=y_test,
        top_features=feature_names,
        feature_categories=feature_categories,
        verbose=1,
        output_path=os.path.join(output_dir, 'hypothesis_testing')
    )
    
    # Step 5.4: XGBoost Native Importance
    print("\n5.4 XGBoost Native Importance")
    xgb_features, xgb_df = get_xgboost_feature_importance(
        model=model,
        feature_names=feature_names,
        importance_type='gain',
        verbose=1,
        output_path=os.path.join(output_dir, 'xgboost_importance')
    )
    
    # Step 5.5: SHAP Analysis
    print("\n5.5 SHAP Analysis")
    shap_features = quantify_feature_importance_via_shap(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        top_features=feature_names,
        model_type="xgboost",
        verbose=1,
        output_path=os.path.join(output_dir, 'shap_analysis')
    )
    
    # Step 6: Compare feature importance rankings
    print("\nStep 6: Comparing feature importance rankings...")
    
    # Collect top 10 features from each method
    comparison = {
        'Mutual Information': mi_features[:10],
        'Effect Sizes': es_features[:10],
        'Hypothesis Testing': ht_features[:10],
        'XGBoost Native': xgb_features[:10],
        'SHAP': shap_features[:10]
    }
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison)
    print("\nTop 10 features from each method:")
    print(comparison_df)
    
    # Save comparison to file
    comparison_df.to_csv(os.path.join(output_dir, 'feature_importance_comparison.csv'), index=False)
    
    # Create a presence matrix to identify consistent important features
    print("\nConsistency analysis of top 5 features across methods:")
    all_top_features = set()
    for method, features in comparison.items():
        all_top_features.update(features[:5])  # Consider top 5 from each
    
    presence_matrix = pd.DataFrame(index=sorted(all_top_features), columns=comparison.keys())
    
    for feature in all_top_features:
        for method, features in comparison.items():
            presence_matrix.loc[feature, method] = feature in features[:5]
    
    # Count how many methods identify each feature as important
    presence_matrix['Count'] = presence_matrix.sum(axis=1)
    presence_matrix = presence_matrix.sort_values('Count', ascending=False)
    
    print(presence_matrix)
    
    # Save consistency analysis to file
    presence_matrix.to_csv(os.path.join(output_dir, 'feature_importance_consistency.csv'))
    
    print("\nFeature importance analysis complete. Results saved to:", output_dir)
    print("\n" + "="*80)


if __name__ == "__main__":
    run_feature_importance_demo()
