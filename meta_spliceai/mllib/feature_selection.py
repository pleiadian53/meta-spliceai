import os, sys
import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from tqdm import tqdm

from typing import Tuple, Any, Dict
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFE



def drop_constant_features(df, **kargs):

    verbose = kargs.get('verbose', 1)
    return_const_cols = kargs.get('return_constant_features', False)

    # Check for columns with constant values
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]

    # Print the constant columns
    if verbose: print(f"(drop_constant_features) List of constant columns (n={len(constant_columns)})")
    for i, col in enumerate(constant_columns): 
        print(f"[{i+1}] {col}")

    shape0 = df.shape
    # Drop the constant columns from the dataframe
    df = df.drop(constant_columns, axis=1)

    # Display the first few rows of the updated dataframe
    # print(df.head())

    if verbose: print(f"(drop_constant_features) shape(df): {shape0} -> {df.shape}")

    if return_const_cols: 
        return df, constant_columns

    return df

def apply_feature_selection_v0(data, model, n_features, **kargs): 
    """
    
    **kargs
        step
    """
    from sklearn.feature_selection import RFE

    # Optional parameters 
    verbose = kargs.get("verbose", 1)

    X, y, *rest = data # `data` has to be at least a 2-tuple

    step = kargs.get('step', 1)
    selector = RFE(model, n_features_to_select=n_features, step=step, verbose=verbose)
    # NOTE: 
    #   - step: if >= 1, number of features to remove at each iteration
    selector = selector.fit(X, y)
    assert sum(selector.support_) == n_features

    return selector.support_


def apply_feature_selection(X: np.ndarray, y: np.ndarray, model: BaseEstimator, n_features: int, **kwargs) -> np.ndarray:
    """
    Apply Recursive Feature Elimination (RFE) to select the best subset of features for the given model.

    Parameters:
    - X (np.ndarray): Feature dataset.
    - y (np.ndarray): Target variable.
    - model (BaseEstimator): A scikit-learn estimator that supports the fit method.
    - n_features (int): The number of features to select.
    
    **kwargs: Optional parameters such as 'step' and 'verbose'.

    Returns:
    - np.ndarray: Boolean array indicating selected features.

    Example usage:
        apply_feature_selection(X, y, model, 5, step=2, verbose=0)

    """
    # Optional parameters
    step = kwargs.get('step', 1)
    verbose = kwargs.get('verbose', 1)

    # Validate inputs
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X and y must be numpy arrays.")
    if not hasattr(model, 'fit'):
        raise ValueError("The model must support the fit method.")
    if not isinstance(n_features, int) or n_features <= 0:
        raise ValueError("n_features must be a positive integer.")

    # Apply Recursive Feature Elimination
    selector = RFE(model, n_features_to_select=n_features, step=step, verbose=verbose)
    selector = selector.fit(X, y)

    if sum(selector.support_) != n_features:
        raise ValueError("The number of selected features does not match n_features.")

    return selector.support_

def find_associated_features(df, correlation_threshold=0.95, **kargs):
    """
    Identifies features that are highly correlated, differ by constant values, or have identical values.

    Parameters:
    - df: DataFrame containing the feature set.
    - correlation_threshold: Threshold above which features are considered highly correlated.

    Returns:
    - correlated_features: Set of tuples where each tuple contains names of highly correlated features.
    - constant_difference_features: Set of tuples where each tuple contains names of features differing by constant values.
    - identical_features: Set of tuples where each tuple contains names of identical features.
    """
    from itertools import combinations

    correlated_features = set()
    constant_difference_features = set()
    identical_features = set()

    # Convert boolean columns to integers for subtraction operation
    df = df.apply(lambda col: col.astype(int) if col.dtype == bool else col)
    # NOTE: For converting the data types of columns, the default behavior (without specifying axis=1) 
    #       is sufficient since we are applying the function to each column as a whole.

    # Compute pairwise correlations between features
    numeric_only = kargs.get("numeric_only", True)
    corr_matrix = df.corr(numeric_only=numeric_only)

    # Iterate over combinations of features to check for high correlation and constant differences
    for (feature1, feature2) in combinations(df.columns, 2):
        if abs(corr_matrix.at[feature1, feature2]) > correlation_threshold:
            correlated_features.add((feature1, feature2))
        
        if df[feature1].equals(df[feature2]):
            identical_features.add((feature1, feature2))
        elif (df[feature1] - df[feature2]).nunique() == 1:
            constant_difference_features.add((feature1, feature2))
    
    return correlated_features, constant_difference_features, identical_features

def feature_analysis(df, max_cardinality=50):
    """
    Analyzes each feature in a dataframe, providing a 5 number summary for numeric features,
    or distinct values and their counts for categorical features. If a categorical feature
    has high cardinality, it provides example values and their counts.

    Parameters:
    - df: Pandas DataFrame containing the features.
    - max_cardinality: The maximum number of unique values a categorical feature
      can have before only examples are shown.

    Returns:
    - A dictionary containing the analysis of each feature.
    """
    analysis_dict = {}
    for feature in df.columns:
        if df[feature].dtype in ['int64', 'float64']:  # Numeric feature
            analysis_dict[feature] = df[feature].describe().to_dict()
        else:  # Categorical feature
            unique_values = df[feature].nunique()
            if unique_values <= max_cardinality:
                value_counts = df[feature].value_counts().to_dict()
                analysis_dict[feature] = value_counts
            else:
                examples = df[feature].value_counts().head(10).to_dict()  # Top 10 as examples
                analysis_dict[feature] = {'examples': examples, 'total_unique_values': unique_values}
    
    return analysis_dict

def display_feature_analysis(analysis_dict):
    """
    Displays the analysis of each feature in a readable format.

    Parameters:
    - analysis_dict: Dictionary containing the analysis of each feature.
    """
    i = 0
    for feature, analysis in analysis_dict.items():
        print(f"[{i+1}] {feature}")
        if isinstance(analysis, dict):
            if 'examples' in analysis:
                print("  High cardinality - showing examples:")
                for value, count in analysis['examples'].items():
                    print(f"    {value}: {count}")
                print(f"    Total unique values: {analysis['total_unique_values']}")
            else:
                print("  Value counts:")
                for value, count in analysis.items():
                    print(f"    {value}: {count}")
        else:
            print("  Summary statistics:")
            for stat, value in analysis.items():
                print(f"    {stat}: {value}")
        print()

        i += 1

def demo_feature_selection(): 
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression

    # Generate synthetic dataset with continuous features
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=2, random_state=42)

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(X, columns=[f'cont_{i}' for i in range(X.shape[1])])

    # Add categorical features
    df['cat_1'] = np.random.choice(['A', 'B', 'C'], size=df.shape[0])
    df['cat_2'] = np.random.choice(['X', 'Y', 'Z'], size=df.shape[0])

    # One-hot encode the categorical features
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_cats = encoder.fit_transform(df[['cat_1', 'cat_2']])

    # Combine continuous and encoded categorical features
    X_combined = np.hstack((X, encoded_cats))

    # Apply feature selection
    model = LogisticRegression(solver='liblinear', random_state=42)  # Simple classifier
    selected_features = apply_feature_selection(X_combined, y, model, 5, verbose=0)

    selected_features_indices = np.where(selected_features)[0]

    print(f"> Selected feature indices:\n{selected_features_indices}\n")

    return

def feature_selection_with_hyperparameter_tuning(X, y, model: BaseEstimator, param_grid: Dict[str, Any], **kwargs):
    """
    Perform feature selection combined with hyperparameter tuning using a pipeline approach.

    Parameters:
    - X (np.ndarray): Feature dataset.
    - y (np.ndarray): Target variable.
    - model (BaseEstimator): A scikit-learn estimator that supports the fit method.
    - param_grid (Dict[str, Any]): Dictionary with parameters names as keys and lists of parameter settings to try as values.

    Returns:
    - GridSearchCV: Fitted GridSearchCV object.

    Example usage: 

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    param_grid = {
        'feature_selection__n_features_to_select': [5, 10, 15],  # Number of features for RFE
        'model__C': [0.1, 1, 10]  # Regularization parameter for LogisticRegression
    }
    result = feature_selection_with_hyperparameter_tuning(X, y, model, param_grid, cv=3)

    Memo
    ----
    1. In this Pipeline usage, each step is named (in this case, the steps are named 'feature_selection' and 'model'). 
       When you want to specify hyperparameters for the estimators in these steps, you use the step name 
       followed by a double underscore (__), and then the hyperparameter name. 
       This syntax tells GridSearchCV which hyperparameter belongs to which step in the pipeline.

       E.g., 
            `feature_selection__n_features_to_select` sets the n_features_to_select parameter for the RFE instance 
            in the pipeline step named 'feature_selection'. 
            
            `model__C` sets the C parameter (regularization strength) for the LogisticRegression instance in 
            the pipeline step named 'model'.

    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    # Optional parameters
    step = kwargs.get('step', 1)
    cv = kwargs.get('cv', 5)  # Cross-validation splitting strategy

    # Create the RFE (feature selection) and model pipeline
    rfe = RFE(estimator=model, step=step)
    pipeline = Pipeline(steps=[('feature_selection', rfe), ('model', model)])
    
    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=1)

    # Fit the GridSearchCV object
    grid_search.fit(X, y)

    return grid_search


def demo_model_selection_pipeline_approach(): 
    from sklearn.datasets import make_classification
    # from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_validate
    from sklearn.metrics import roc_auc_score, f1_score, make_scorer

    # Create a synthetic dataset
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10, n_redundant=5, random_state=42)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Set up the model
    model = LogisticRegression(solver='liblinear', random_state=42)

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'feature_selection__n_features_to_select': [5, 10, 15],
        'model__C': [0.1, 1, 10]
    }

    # Use the feature_selection_with_hyperparameter_tuning function
    tuned_results = feature_selection_with_hyperparameter_tuning(X_train, y_train, model, param_grid, cv=3)

    # Best model from hyperparameter tuning and feature selection
    best_model = tuned_results.best_estimator_

    # Test the model
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)
    # accuracy_with_tuning = accuracy_score(y_test, predictions)

    ### Evaluation
    # Define additional scoring metrics
    scoring_metrics = {'roc_auc': make_scorer(roc_auc_score, needs_proba=True), 'f1': make_scorer(f1_score)}

    # Evaluate the best model with additional metrics
    evaluation_results_best = cross_validate(best_model, X_train, y_train, scoring=scoring_metrics, cv=3)

    ########

    # Feature selection without hyperparameter tuning for comparison
    selector = RFE(model, n_features_to_select=10, step=1)
    selector = selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    model.fit(X_train_selected, y_train)
    predictions = model.predict(X_test_selected)
    # accuracy_without_tuning = accuracy_score(y_test, predictions)

    ########

    # Evaluate the regular model with additional metrics
    evaluation_results_regular = cross_validate(model, X_train_selected, y_train, scoring=scoring_metrics, cv=3)

    # Extract and calculate average scores for the best model
    average_roc_auc_best = np.mean(evaluation_results_best['test_roc_auc'])
    average_f1_best = np.mean(evaluation_results_best['test_f1'])

    # Extract and calculate average scores for the regular model
    average_roc_auc_regular = np.mean(evaluation_results_regular['test_roc_auc'])
    average_f1_regular = np.mean(evaluation_results_regular['test_f1'])

    metrics_regular = (average_roc_auc_regular, average_f1_regular)
    metrics_best = (average_roc_auc_best, average_f1_best)

    print(f"> Model (FS only): {metrics_regular}")
    print(f"> Model (FS + MS): {metrics_best}")

    return

def test(): 

    # demo_feature_selection()

    demo_model_selection_pipeline_approach()


if __name__ == "__main__": 
    test()


