import numpy as np
import pandas as pd
import shap
import traceback


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
        try:
            print("[SHAP Debug] Starting SHAP calculation for XGBoost model")
            
            # First, check if the model has feature_names attribute
            if hasattr(model, 'feature_names') and model.feature_names:
                print(f"[SHAP Debug] Model has {len(model.feature_names)} feature_names")
                # Ensure we're only using the features the model knows about
                model_features = list(model.feature_names)
                # Filter X to only include features the model was trained on
                missing_features = [f for f in model_features if f not in X.columns]
                if missing_features:
                    print(f"[SHAP Debug] Warning: Missing features in X: {missing_features}")
                    # Might need to handle this case if features are missing
                
                available_features = [f for f in model_features if f in X.columns]
                if len(available_features) < len(model_features):
                    print(f"[SHAP Debug] Only {len(available_features)}/{len(model_features)} model features available in dataset")
                
                X_filtered = X[available_features]
                print(f"[SHAP Debug] X shape: {X.shape}, X_filtered shape: {X_filtered.shape}")
                feature_names = available_features
            else:
                print("[SHAP Debug] Model doesn't have feature_names attribute")
                # Try to infer from model's feature_importances_ if available
                if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
                    model_features = list(model.feature_names_in_)
                    print(f"[SHAP Debug] Using {len(model_features)} features from model.feature_names_in_")
                    X_filtered = X[model_features]
                    feature_names = model_features
                else:
                    print("[SHAP Debug] No feature names found in model, using all features (may cause dimension mismatch)")
                    X_filtered = X
            
            # Create explainer with the model
            print("[SHAP Debug] Creating TreeExplainer")
            explainer = shap.TreeExplainer(model)
            
            # Use the filtered X for SHAP calculation
            print(f"[SHAP Debug] Calculating SHAP values for {X_filtered.shape[0]} samples and {X_filtered.shape[1]} features")
            shap_values = explainer.shap_values(X_filtered)
            
            # Handle the case where shap_values is a list (for multiclass models)
            if isinstance(shap_values, list):
                print(f"[SHAP Debug] SHAP values returned as list with {len(shap_values)} elements")
                # For binary classification, typically take the positive class
                shap_values = shap_values[0]
                
            print(f"[SHAP Debug] SHAP calculation successful, shape: {shap_values.shape}")
            return shap_values, feature_names
            
        except Exception as e:
            print(f"[SHAP Debug] Error during SHAP calculation: {str(e)}")
            print("[SHAP Debug] Full traceback:")
            traceback.print_exc()
            raise
            
    elif model_type.lower() == "sklearn":
        # For sklearn models that are tree-based
        if hasattr(model, "feature_importances_"):
            try:
                if hasattr(model, "feature_names_in_"):
                    model_features = list(model.feature_names_in_)
                    print(f"[SHAP Debug] Using {len(model_features)} features from sklearn model")
                    X_filtered = X[model_features]
                    feature_names = model_features
                else:
                    X_filtered = X
                    
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_filtered)
                
                # For binary classification, shap_values might be a list with one element
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
            except Exception as e:
                print(f"[SHAP Debug] Error with sklearn TreeExplainer: {str(e)}")
                traceback.print_exc()
                raise
        else:
            # For other sklearn models, use KernelExplainer
            try:
                explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 100))
                shap_values = explainer.shap_values(X)[1]  # Class 1 explanation
            except Exception as e:
                print(f"[SHAP Debug] Error with KernelExplainer: {str(e)}")
                traceback.print_exc()
                raise
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    return shap_values, feature_names
