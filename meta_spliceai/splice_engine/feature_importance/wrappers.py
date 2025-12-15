"""
Feature Importance Wrapper Module.

This module provides standardized wrappers around the various feature importance
calculation methods with consistent naming conventions and interfaces.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

from .xgboost_importance import get_xgboost_feature_importance
from .shap_analysis import compute_shap_values, compute_global_feature_importance_from_shap
from .hypothesis_testing import quantify_feature_importance_via_hypothesis_testing
from .effect_sizes import quantify_feature_importance_via_measuring_effect_sizes
from .mutual_info import quantify_feature_importance_via_mutual_info
from ..analysis_utils import classify_features

def calculate_xgboost_importance(model, X, y, importance_type='weight'):
    """
    Calculate feature importance using XGBoost's built-in methods.
    Handles all feature types that XGBoost can process, including numerical and categorical
    features that have been properly encoded.
    
    Parameters
    ----------
    model : xgboost.Booster or xgboost.XGBClassifier
        Trained XGBoost model
    X : pd.DataFrame
        Feature matrix
    y : array-like
        Target vector
    importance_type : str, default='weight'
        Type of importance metric (weight, gain, cover, total_gain)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature names and importance scores
    """
    # Use classify_features to identify feature types
    feature_categories = classify_features((X, y))
    numerical_features = feature_categories['numerical_features']
    categorical_features = feature_categories['categorical_features'] + feature_categories['derived_categorical_features']
    
    # FIXED: Include motif features (k-mers) in usable features for XGBoost
    motif_features = feature_categories.get('motif_features', [])
    
    # XGBoost can handle both numeric and properly encoded categorical features, including k-mers
    usable_features = numerical_features + categorical_features + motif_features
    
    # Filter dataset to only include usable features
    X_filtered = X[usable_features].copy()
    
    # Ensure categorical features are properly encoded 
    for col in categorical_features:
        if col in X_filtered.columns:
            # Convert to category codes for XGBoost
            if X_filtered[col].dtype == 'object' or X_filtered[col].dtype.name == 'category':
                X_filtered[col] = X_filtered[col].astype('category').cat.codes.astype(float)
    
    # Get feature names
    feature_names = list(X_filtered.columns)
    
    # The get_xgboost_feature_importance function returns a tuple of (top_features, importance_df)
    _, importance_df = get_xgboost_feature_importance(
        model, 
        feature_names, 
        importance_type=importance_type, 
        full_importance=True
    )
    
    # Ensure the dataframe has the expected columns
    importance_df = importance_df.rename(
        columns={'importance': 'importance_score'}
    )[['feature', 'importance_score']]
    
    return importance_df


def calculate_shap_importance(
    model, 
    X, 
    y=None, 
    top_k=None, 
    max_samples=20000, 
    random_state=42,
    fallback_to_xgboost=False,  
    model_feature_columns=None  # <-- explicitly provided columns for safety
):
    """
    Calculate feature importance using SHAP values.

    Parameters
    ----------
    model : object
        Model to explain (XGBoost, sklearn, etc.)
    X : pd.DataFrame
        Feature matrix
    y : array-like, optional
        Target vector
    top_k : int, default=None
        Number of top features to return
    max_samples : int, default=20000
        Maximum number of samples to use for SHAP calculation to prevent memory issues
        Set to None to use all samples (may cause memory errors with large datasets)
    random_state : int, default=42
        Random seed for reproducible sampling
    fallback_to_xgboost : bool, default=False
        Whether to fall back to XGBoost feature importance if SHAP fails
    model_feature_columns : list, optional
        List of feature names expected by the model
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature names and importance scores
    """
    import numpy as np
    import pandas as pd
    import shap
    import sys

    # Verify feature consistency explicitly
    if model_feature_columns is not None:
        missing_cols = set(model_feature_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in X: {missing_cols}")

        # Explicitly reorder and subset columns to match model exactly
        X_filtered = X[model_feature_columns].copy()
    else:
        # (Your original feature classification logic, if you prefer)
        feature_categories = classify_features((X, y))
        numerical_features = feature_categories['numerical_features']
        categorical_features = (
            feature_categories['categorical_features'] + feature_categories['derived_categorical_features']
        )
        usable_features = numerical_features + categorical_features
        X_filtered = X[usable_features].copy()

        # Properly encode categorical features
        for col in categorical_features:
            if col in X_filtered.columns:
                if X_filtered[col].dtype == 'object' or X_filtered[col].dtype.name == 'category':
                    X_filtered[col] = X_filtered[col].astype('category').cat.codes.astype(float)

    # Subsample if needed
    if max_samples and len(X_filtered) > max_samples:
        print(f"[info] Limiting SHAP calculation to {max_samples} samples (from {len(X_filtered)})")
        X_sampled = X_filtered.sample(max_samples, random_state=random_state)
    else:
        X_sampled = X_filtered

    try:
        # SHAP computation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sampled)

        # Global feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': X_sampled.columns,
            'importance_score': feature_importance
        }).sort_values(by='importance_score', ascending=False).reset_index(drop=True)

        print("[SHAP Debug] SHAP values computed successfully!")

        if top_k is None:
            top_k = len(importance_df)

        return importance_df.head(top_k)

    except Exception as e:
        import traceback
        print(f"[error] SHAP calculation failed: {str(e)}")
        print(f"[error] Error type: {type(e).__name__}")
        traceback.print_exc()

        if fallback_to_xgboost:
            print("[info] Falling back to XGBoost feature importance")
            try:
                from .xgboost_importance import get_xgboost_feature_importance
                
                feature_names = list(X_sampled.columns)
                _, importance_df = get_xgboost_feature_importance(
                    model, 
                    feature_names, 
                    importance_type='total_gain', 
                    full_importance=True
                )
                importance_df = importance_df.rename(
                    columns={'importance': 'importance_score'}
                )[['feature', 'importance_score']]

                print("[info] Successfully fell back to XGBoost feature importance")
                if top_k is None:
                    top_k = len(importance_df)
                return importance_df.head(top_k)

            except Exception as fallback_error:
                print(f"[error] Fallback also failed: {str(fallback_error)}")

        # Return empty DataFrame if fallback disabled or failed
        return pd.DataFrame(columns=['feature', 'importance_score'])


def calculate_shap_importance_for_motifs(model, X, y=None, motif_pattern=r'^\d+mer_.*', top_k=15):
    """
    Calculate feature importance using SHAP values, but specifically for motif features.
    Filters features matching the provided pattern (typically k-mer motifs) before
    performing SHAP analysis.
    
    Parameters
    ----------
    model : object
        Model to explain (XGBoost, sklearn, etc.)
    X : pd.DataFrame
        Feature matrix
    y : array-like, optional
        Target vector (not used in this function but included for API consistency)
    motif_pattern : str, default=r'^\d+mer_.*'
        Regular expression pattern to identify motif features (default matches "1mer_", "2mer_", etc.)
    top_k : int, default=15
        Number of top features to return
        
    Returns
    -------
    pd.DataFrame
        DataFrame with motif feature names and importance scores
    """
    import re
    import numpy as np
    import shap
    import pandas as pd

    # Filter columns matching the motif pattern
    motif_columns = [col for col in X.columns if re.match(motif_pattern, col)]
    
    if not motif_columns:
        print("[warn] No motif-specific features found. Check your motif_pattern or input data.")
        return pd.DataFrame(columns=['feature', 'importance_score'])

    # Create a filtered DataFrame with only motif features
    X_motif = X[motif_columns].copy()
    
    # Compute SHAP values for motif features
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_motif)
    
    # For multi-class models, we need to select the class to explain
    if isinstance(shap_values, list):
        # For binary classification, class 1 (positive class)
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    
    # Calculate mean absolute SHAP values for each feature
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame with results
    importance_df = pd.DataFrame({
        'feature': motif_columns,
        'importance_score': feature_importance
    }).sort_values('importance_score', ascending=False)
    
    # Return full results - don't limit to top_k
    return importance_df


def calculate_hypothesis_testing(X, y):
    """
    Calculate feature importance using hypothesis testing.
    Uses appropriate tests for different feature types:
    - Numeric features: t-test
    - Categorical features: chi-square test
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : array-like
        Target vector (binary)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature names and importance scores for ALL features.
        Features that could not be calculated will have NaN as their importance score.
    """
    from scipy.stats import chi2_contingency
    
    # Use classify_features to identify appropriate features for hypothesis testing
    feature_categories = classify_features((X, y))
    numerical_features = feature_categories['numerical_features']
    categorical_features = feature_categories['categorical_features'] + feature_categories['derived_categorical_features']
    
    # FIXED: Include motif features (k-mers) in numerical features for t-tests
    motif_features = feature_categories.get('motif_features', [])
    numerical_features = numerical_features + motif_features
    
    # Get feature names
    feature_names = list(X.columns)
    
    # Initialize dictionaries to store results for ALL features
    importance_dict = {feature: np.nan for feature in feature_names}
    calculation_success = {feature: False for feature in feature_names}
    
    # Convert to numpy arrays for faster processing
    y_np = np.array(y)
    
    for feature in feature_names:
        try:
            # Skip calculations (but keep feature with NaN) if it contains NaN values
            if X[feature].isnull().any():
                continue
                
            if feature in numerical_features:
                # Calculate t-test for numerical features (including k-mers)
                X_feature = X[feature].values
                
                # Get positive and negative class samples
                X_pos = X_feature[y_np == 1]
                X_neg = X_feature[y_np == 0]
                
                # Calculate p-value with t-test
                _, p_value = stats.ttest_ind(X_pos, X_neg, equal_var=False)
                # Use -log10(p) as importance score (higher = more significant)
                importance_score = -np.log10(max(p_value, 1e-10))  # Clip p-value to avoid numerical issues
                
                importance_dict[feature] = importance_score
                calculation_success[feature] = True
                    
            elif feature in categorical_features:
                # Calculate chi-square test for categorical features
                contingency = pd.crosstab(X[feature], y)
                chi2, p_value, _, _ = chi2_contingency(contingency)
                
                # Use -log10(p) as importance score (higher = more significant)
                importance_score = -np.log10(max(p_value, 1e-10))
                
                importance_dict[feature] = importance_score
                calculation_success[feature] = True
                
            else:
                # For unknown feature types, default to numerical method
                try:
                    X_feature = X[feature].values
                    
                    # Get positive and negative class samples
                    X_pos = X_feature[y_np == 1]
                    X_neg = X_feature[y_np == 0]
                    
                    # Calculate p-value with t-test
                    _, p_value = stats.ttest_ind(X_pos, X_neg, equal_var=False)
                    
                    # Use -log10(p) as importance score
                    importance_score = -np.log10(max(p_value, 1e-10))
                    
                    importance_dict[feature] = importance_score
                    calculation_success[feature] = True
                except:
                    # Keep NaN value if calculation fails
                    pass
                    
        except Exception as e:
            # Keep the NaN value for this feature
            print(f"Warning: Could not calculate hypothesis test for feature '{feature}': {str(e)}")
    
    # Create DataFrame with ALL features, including those that couldn't be calculated
    importance_df = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance_score': list(importance_dict.values()),
        'calculation_success': [calculation_success[feature] for feature in importance_dict.keys()]
    })
    
    # Sort by importance (descending), with NaN values at the end
    importance_df = importance_df.sort_values('importance_score', ascending=False, na_position='last')
    
    return importance_df


def calculate_effect_sizes(X, y):
    """
    Calculate effect sizes for each feature to quantify importance.
    Handles different feature types appropriately:
    - Numeric features: Cohen's d
    - Categorical features: Cramer's V
    - Ordinal features: Rank-biserial correlation
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : array-like
        Target vector (binary)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature names and effect size scores for ALL features.
        Features that could not be calculated will have NaN as their importance score.
    """
    from scipy.stats import mannwhitneyu, chi2_contingency
    
    # Use classify_features to identify different feature types
    feature_categories = classify_features((X, y))
    numerical_features = feature_categories['numerical_features']
    categorical_features = feature_categories['categorical_features'] + feature_categories['derived_categorical_features']
    
    # Get feature names - we'll calculate or assign a value for EVERY feature
    feature_names = list(X.columns)
    
    # Initialize dictionaries to store results for ALL features
    effect_sizes_dict = {feature: np.nan for feature in feature_names}
    calculation_success = {feature: False for feature in feature_names}
    
    # Convert to numpy arrays for faster processing
    y_np = np.array(y)
    
    for feature in feature_names:
        try:
            # Get positive and negative class samples
            pos_class = X.loc[y_np == 1, feature]
            neg_class = X.loc[y_np == 0, feature]
            
            # Skip calculations (but keep feature with NaN) if it contains NaN values
            if X[feature].isnull().any():
                continue
                
            if feature in numerical_features:
                # Calculate Cohen's d for numerical features
                # Cohen's d = (mean1 - mean2) / pooled_std
                mean_diff = pos_class.mean() - neg_class.mean()
                pooled_std = np.sqrt(
                    ((len(pos_class)-1)*pos_class.var() + (len(neg_class)-1)*neg_class.var()) / 
                    (len(pos_class) + len(neg_class) - 2)
                )
                
                # Handle potential division by zero
                if pooled_std == 0:
                    effect_sizes_dict[feature] = 0
                else:
                    # Preserve directionality of Cohen's d by NOT taking absolute value
                    effect_sizes_dict[feature] = mean_diff / pooled_std
                    
                calculation_success[feature] = True
                    
            elif feature in categorical_features:
                # Calculate Cramer's V for categorical features
                contingency = pd.crosstab(X[feature], y)
                chi2, _, _, _ = chi2_contingency(contingency)
                n = contingency.sum().sum()
                phi2 = chi2 / n
                r, k = contingency.shape
                
                # Check for division by zero - occurs when either feature or target has only 1 unique value
                denominator = min(k - 1, r - 1)
                if denominator > 0:
                    cramer_v = np.sqrt(phi2 / denominator)
                else:
                    # If either variable has only one category, Cramer's V is undefined
                    # Set to 0 as this indicates no predictive power
                    cramer_v = 0.0
                
                effect_sizes_dict[feature] = cramer_v
                calculation_success[feature] = True
                
            else:
                # For unknown feature types, default to numerical method but mark as not successful
                # This ensures all features get a value
                try:
                    mean_diff = pos_class.mean() - neg_class.mean()
                    pooled_std = np.sqrt(
                        ((len(pos_class)-1)*pos_class.var() + (len(neg_class)-1)*neg_class.var()) / 
                        (len(pos_class) + len(neg_class) - 2)
                    )
                    
                    if pooled_std == 0:
                        effect_sizes_dict[feature] = 0
                    else:
                        effect_sizes_dict[feature] = mean_diff / pooled_std
                        
                    # Mark as successful since we could calculate it
                    calculation_success[feature] = True
                except:
                    # Keep NaN value if calculation fails
                    pass
                    
        except Exception as e:
            # Keep the NaN value for this feature
            print(f"Warning: Could not calculate effect size for feature '{feature}': {str(e)}")
    
    # Create DataFrame with ALL features, including those that couldn't be calculated
    effect_size_df = pd.DataFrame({
        'feature': list(effect_sizes_dict.keys()),
        'importance_score': list(effect_sizes_dict.values()),
        'calculation_success': [calculation_success[feature] for feature in effect_sizes_dict.keys()]
    })
    
    # Sort by absolute importance (descending), with NaN values at the end
    effect_size_df = effect_size_df.copy()
    effect_size_df['abs_importance'] = effect_size_df['importance_score'].abs()
    effect_size_df = effect_size_df.sort_values('abs_importance', ascending=False, na_position='last')
    
    # Remove the temporary column
    effect_size_df = effect_size_df.drop(columns=['abs_importance'])
    
    return effect_size_df


def calculate_mutual_information(X, y):
    """
    Calculate mutual information between features and target.
    For categorical features, uses special handling to ensure proper computation.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : array-like
        Target vector
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature names and mutual information scores for ALL features.
        Features that could not be calculated will have NaN as their importance score.
    """
    # Use classify_features to identify feature types
    feature_categories = classify_features((X, y))
    numerical_features = feature_categories['numerical_features']
    categorical_features = feature_categories['categorical_features'] + feature_categories['derived_categorical_features']
    
    # FIXED: Include motif features (k-mers) in numerical features for mutual information
    motif_features = feature_categories.get('motif_features', [])
    numerical_features = numerical_features + motif_features
    
    print(f"[MutualInfo Debug] Classified Features: {len(numerical_features)} numerical (including {len(motif_features)} k-mers), {len(categorical_features)} categorical.")
    
    # Get ALL feature names
    feature_names = list(X.columns)
    
    # Initialize dictionaries to store results for ALL features
    mutual_info_dict = {feature: np.nan for feature in feature_names}
    calculation_success = {feature: False for feature in feature_names}
    
    # Prepare X for mutual information calculation
    X_transformed = X.copy()
    
    # Create a list to track which features are discrete
    discrete_features = []
    feature_indices = {}
    valid_features = []
    
    # Process each feature
    for i, feature in enumerate(feature_names):
        try:
            feature_indices[feature] = i
            
            if feature in categorical_features:
                # For categorical features, ensure they're encoded properly
                X_transformed[feature] = X_transformed[feature].astype('category').cat.codes
                discrete_features.append(True)
                valid_features.append(i)
            elif feature in numerical_features:
                # For numerical features, convert booleans to floats
                if X_transformed[feature].dtype == bool or X_transformed[feature].dtype == np.bool_:
                    X_transformed[feature] = X_transformed[feature].astype(float)
                discrete_features.append(False)
                
                # Only include if no NaN values
                if not X_transformed[feature].isnull().any():
                    valid_features.append(i)
            else:
                # For other features, make a best guess
                if X_transformed[feature].dtype == 'object' or X_transformed[feature].nunique() < 10:
                    # Treat as categorical
                    X_transformed[feature] = X_transformed[feature].astype('category').cat.codes
                    discrete_features.append(True)
                else:
                    # Treat as numerical
                    discrete_features.append(False)
                
                # Only include if no NaN values
                if not X_transformed[feature].isnull().any():
                    valid_features.append(i)
        except Exception as e:
            print(f"Warning: Error preprocessing feature '{feature}' for mutual information: {str(e)}")
            discrete_features.append(False)  # Default to numerical
    
    # Create filtered versions for calculation
    try:
        # Calculate mutual information only for valid features
        if len(valid_features) > 0:
            X_filtered = X_transformed.iloc[:, valid_features]
            discrete_filtered = [discrete_features[i] for i in valid_features]
            
            print(f"[MutualInfo Debug] Attempting calculation with {X_filtered.shape[1]} valid features ({sum(discrete_filtered)} discrete).")
            
            # Calculate mutual information
            mi_scores = mutual_info_classif(
                X_filtered, y, 
                discrete_features=discrete_filtered,
                random_state=42
            )
            
            print("[MutualInfo Debug] Main calculation successful.")
            
            # Map scores back to original features
            for i, idx in enumerate(valid_features):
                feature = feature_names[idx]
                mutual_info_dict[feature] = mi_scores[i]
                calculation_success[feature] = True
        else:
             print("[MutualInfo Debug] No valid features found for main calculation.")
             
    except Exception as e:
        print(f"[MutualInfo Debug] Main calculation failed: {str(e)}")
        print("[MutualInfo Debug] Falling back to numerical features only...")
        
        try:
            # Try with just numerical features
            X_numeric = X[numerical_features].copy()
            for col in X_numeric.columns:
                if X_numeric[col].dtype == bool or X_numeric[col].dtype == np.bool_:
                    X_numeric[col] = X_numeric[col].astype(float)
            
            # Skip columns with NaN values
            numeric_valid_cols = X_numeric.columns[~X_numeric.isnull().any()]
            X_numeric = X_numeric[numeric_valid_cols]
            
            # Calculate mutual information with numeric features only
            if X_numeric.shape[1] > 0:
                print(f"[MutualInfo Debug] Attempting fallback calculation with {X_numeric.shape[1]} numerical features.")
                mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
                
                print("[MutualInfo Debug] Fallback calculation successful.")
                
                # Map scores back to original features
                for i, col in enumerate(numeric_valid_cols):
                    mutual_info_dict[col] = mi_scores[i]
                    calculation_success[col] = True
            else:
                print("[MutualInfo Debug] No valid numerical features found for fallback calculation.")
                
        except Exception as fallback_e:
            print(f"[MutualInfo Debug] Fallback calculation also failed: {str(fallback_e)}")

    # Create DataFrame with ALL features, including those that couldn't be calculated
    mutual_info_df = pd.DataFrame({
        'feature': list(mutual_info_dict.keys()),
        'importance_score': list(mutual_info_dict.values()),
        'calculation_success': [calculation_success[feature] for feature in mutual_info_dict.keys()]
    })
    
    # Add debug print for number of successful calculations
    num_success = mutual_info_df['calculation_success'].sum()
    print(f"[MutualInfo Debug] Total features calculated successfully: {num_success} out of {len(feature_names)}")
    
    # Sort by importance (descending), with NaN values at the end
    mutual_info_df = mutual_info_df.sort_values('importance_score', ascending=False, na_position='last')
    
    return mutual_info_df

# Add alias for backward compatibility
perform_hypothesis_testing = calculate_hypothesis_testing
