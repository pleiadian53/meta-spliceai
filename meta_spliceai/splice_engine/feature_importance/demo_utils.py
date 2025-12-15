"""
Mock Data Generator for Testing Feature Importance Methods.

This module provides utility functions to generate synthetic datasets with known patterns
for testing and demonstrating the feature importance modules.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import os


def generate_mock_classification_data(n_samples=1000, n_features=20, n_informative=5, 
                                     n_redundant=5, n_repeated=0, class_sep=1.0, 
                                     random_state=42):
    """
    Generate a synthetic classification dataset with known important features.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples.
    n_features : int, default=20
        Total number of features.
    n_informative : int, default=5
        Number of informative features.
    n_redundant : int, default=5
        Number of redundant features.
    n_repeated : int, default=0
        Number of repeated features.
    class_sep : float, default=1.0
        Class separation factor.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    pandas.DataFrame
        Feature matrix.
    numpy.ndarray
        Target vector.
    list of str
        List of feature names, with informative features prefixed with 'inf_'.
    """
    # Generate synthetic data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        class_sep=class_sep,
        random_state=random_state
    )
    
    # Create feature names
    feature_names = []
    for i in range(n_features):
        if i < n_informative:
            feature_names.append(f"inf_feature_{i}")  # Informative features
        elif i < n_informative + n_redundant:
            feature_names.append(f"red_feature_{i}")  # Redundant features
        elif i < n_informative + n_redundant + n_repeated:
            feature_names.append(f"rep_feature_{i}")  # Repeated features
        else:
            feature_names.append(f"noise_feature_{i}")  # Noise features
    
    # Create DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Add some categorical features
    X_df['cat_feature_1'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    X_df['cat_feature_2'] = np.random.choice(['X', 'Y', 'Z'], size=n_samples)
    
    # Make categorical features informative
    # Class 0 more likely to have 'A' and 'X'
    # Class 1 more likely to have 'C' and 'Z'
    for i, label in enumerate(y):
        if label == 0:
            if np.random.random() < 0.7:  # 70% chance for class 0 to have 'A'
                X_df.loc[i, 'cat_feature_1'] = 'A'
            if np.random.random() < 0.7:  # 70% chance for class 0 to have 'X'
                X_df.loc[i, 'cat_feature_2'] = 'X'
        else:
            if np.random.random() < 0.7:  # 70% chance for class 1 to have 'C'
                X_df.loc[i, 'cat_feature_1'] = 'C'
            if np.random.random() < 0.7:  # 70% chance for class 1 to have 'Z'
                X_df.loc[i, 'cat_feature_2'] = 'Z'
    
    # Add a few motif-like features (sequence patterns)
    motifs = ['ACGT', 'GATA', 'TATA', 'CGCG', 'ATGC']
    for i, motif in enumerate(motifs):
        # Motif presence is correlated with class label for first 3 motifs
        if i < 3:
            X_df[f'motif_{motif}'] = (np.random.random(n_samples) + 0.3 * y) > 0.5
        else:
            X_df[f'motif_{motif}'] = np.random.random(n_samples) > 0.5
    
    # Update feature names list with new features
    feature_names = X_df.columns.tolist()
    
    # Define feature categories
    feature_categories = {}
    for feature in feature_names:
        if feature.startswith('cat_'):
            feature_categories[feature] = 'categorical'
        elif feature.startswith('motif_'):
            feature_categories[feature] = 'binary'
        else:
            feature_categories[feature] = 'numeric'
    
    return X_df, y, feature_names, feature_categories


def train_test_split_mock_data(X, y, test_size=0.3, random_state=42):
    """
    Split mock data into training and test sets.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : numpy.ndarray
        Target vector.
    test_size : float, default=0.3
        Proportion of data to use for testing.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    pandas.DataFrame
        Training feature matrix.
    pandas.DataFrame
        Test feature matrix.
    numpy.ndarray
        Training target vector.
    numpy.ndarray
        Test target vector.
    """
    from sklearn.model_selection import train_test_split
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_xgboost_model(X_train, y_train, params=None):
    """
    Train an XGBoost model on the given data.
    
    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    y_train : numpy.ndarray
        Training target vector.
    params : dict, optional
        XGBoost parameters. If None, use default parameters.
    
    Returns
    -------
    xgboost.XGBClassifier
        Trained model.
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost is required for this function. Install with 'pip install xgboost'.")
    
    # Set default parameters if not provided
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42
        }
    
    # Create model
    model = xgb.XGBClassifier(**params)
    
    # Train model
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model on test data.
    
    Parameters
    ----------
    model : object
        Trained model with predict_proba method.
    X_test : pandas.DataFrame
        Test feature matrix.
    y_test : numpy.ndarray
        Test target vector.
    
    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    return metrics
