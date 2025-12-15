# from xgboost import XGBClassifier
import os, sys
import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from tqdm import tqdm

# from .utils import import_hpbandster_search_cv

# Implementation notes
# 1. Under cross validation, what would be the best way to determine the "best hyperparameter settings"
#    for the model given the data? mode or most frequent selected parameters? 
# 
#    When using nested cross-validation, the main goal is to provide an unbiased estimate of 
#    the model's performance on unseen data. The inner loop helps select the best hyperparameters, 
#    while the outer loop provides an evaluation of the model with those hyperparameters.
#    
#    However, the concept of "best hyperparameters" in nested cross-validation is a bit tricky.
#    Each outer loop iteration might yield slightly different "best" hyperparameters because different training 
#    and validation sets are used in each iteration


# ---- Logistic Regression ----- 

def nested_cv_logistic_regression_with_importance(X, y, **kargs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score, roc_auc_score

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        X = X.values
    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values

    # Define the logistic regression classifier
    log_reg = LogisticRegression(random_state=0, max_iter=1000, solver='saga')

    # Define the hyperparameter grid
    param_grid_log_reg = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2',  ],  # 'elasticnet'
        # 'solver': ['newton-cg', 'lbfgs', 'liblinear']
    }     

    # Nested CV can be time consuming and we don't necessarily want to use it everytime 
    default_hyperparameters = kargs.get("default_hyperparams", None)
    use_nested_cv = kargs.get('use_nested_cv', True)   
    if default_hyperparameters is None: 
        default_hyperparameters = {
            'C': 1, 
            'penalty': 'l2'
        }
    if not use_nested_cv and kargs.get('default_hyperparams', None): 
        print(f"[model] Logistic: using known hyperparameters:\n{default_hyperparameters}\n")
    
    # Setup the inner and outer cross-validation
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    n_folds_inner = kargs.get("n_folds_inner", n_folds_outer)
    inner_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=n_folds_outer, shuffle=True, random_state=0)

    outer_f1_scores = []
    outer_roc_auc_scores = []
    all_feature_importances = []
    best_params_list = []

    # Outer cross-validation loop
    fold = 1
    test_case = np.random.choice(range(n_folds_outer), 1)[0]
    for train_idx, val_idx in outer_cv.split(X, y):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # X_train = X.iloc[train_idx] if is_dataframe else X[train_idx]
        # X_val = X.iloc[val_idx] if is_dataframe else X[val_idx]
        # No need for special indexing for y as it's either an array or series now
        # y_train = y[train_idx]
        # y_val = y[val_idx]

        # Standardize the features
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        if use_nested_cv: 
            # Grid search with cross-validation in the inner loop
            grid_search = GridSearchCV(log_reg, param_grid_log_reg, scoring='f1_macro', cv=inner_cv, n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            best_hyperparameters = grid_search.best_params_
        else: 
            best_hyperparameters = default_hyperparameters

        # Test 
        if fold == test_case: 
            print(f"[model] Fold={fold}: Logistic regression, best params={best_hyperparameters}")
        best_params_list.append(tuple(best_hyperparameters.items()))
        
        # Train the model using the best hyperparameters found
        best_log_reg = LogisticRegression(C=best_hyperparameters['C'],
                                          penalty=best_hyperparameters.get('penalty', 'l2'), 
                                          solver="saga",  # best_hyperparameters['solver'],
                                          random_state=0, max_iter=1000)
        best_log_reg.fit(X_train_scaled, y_train)
        
        # Predict and compute F1 score on the validation set
        y_pred = best_log_reg.predict(X_val_scaled)
        f1 = f1_score(y_val, y_pred, average='macro')
        outer_f1_scores.append(f1)
        
        # Predict probabilities for ROC AUC computation
        y_prob = best_log_reg.predict_proba(X_val_scaled)[:, 1]
        roc_auc = roc_auc_score(y_val, y_prob)
        outer_roc_auc_scores.append(roc_auc)

        # Compute feature importances (coefficients for logistic regression)
        feature_importances = best_log_reg.coef_[0]
        all_feature_importances.append(feature_importances)

        fold += 1

    # Compute average F1 score, average ROC AUC, and average feature importances
    output_dict = {}
    output_dict['f1'] = avg_f1 = np.mean(outer_f1_scores)
    output_dict['auc'] = output_dict['roc_auc'] = avg_roc_auc = np.mean(outer_roc_auc_scores)
    output_dict['feature_importance'] = avg_feature_importances = np.mean(all_feature_importances, axis=0)

    feature_names = kargs.get("feature_names", [])
    if len(feature_names) > 0: 
        assert len(feature_names) == X.shape[1]

        # Displaying the feature importances
        output_dict['feature_importance_df'] = \
            pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_feature_importances
        })    

    # Determine the most frequently selected hyperparameters
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_params = dict(most_common_params)

    # Also return the model with the best parameters (by mode)
    output_dict['model'] = LogisticRegression(C=most_common_params['C'],
                                          penalty=most_common_params.get('penalty', 'l2'), 
                                          solver="saga",  # Todo: depending on penalty type, choose a better optimizer
                                          random_state=0, max_iter=1000)
    
    # return avg_f1, avg_roc_auc, avg_feature_importances
    return output_dict


# Running the function for logistic regression to get the results
# avg_f1_log_reg, avg_roc_auc_log_reg, avg_feature_importances_log_reg = nested_cv_logistic_regression_with_importance(X_diabetes, y_diabetes_binary)
# avg_f1_log_reg, avg_roc_auc_log_reg


# ----- Random Forest -------

def nested_cv_random_forest_with_importance(X, y, **kargs):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score, roc_auc_score

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        X = X.values
    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values

    # Define the random forest classifier
    rf_clf = RandomForestClassifier(random_state=0)

    # Define the hyperparameter grid
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Nested CV can be time consuming and we don't necessarily want to use it everytime 
    default_hyperparameters = kargs.get("default_hyperparams", None)
    use_nested_cv = kargs.get('use_nested_cv', True)
    if default_hyperparameters is None:
        # Default hyperparameters (you can adjust these based on domain knowledge or past experiments)
        default_hyperparameters = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
    if not use_nested_cv and kargs.get('default_hyperparams', None): 
        print(f"[model] Random Forest: using known hyperparameters:\n{default_hyperparameters}\n")

    # Setup the inner and outer cross-validation
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    n_folds_inner = kargs.get("n_folds_inner", n_folds_outer)
    inner_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=n_folds_outer, shuffle=True, random_state=0)

    outer_f1_scores = []
    outer_roc_auc_scores = []
    all_feature_importances = []
    best_params_list = []

    # Outer cross-validation loop
    fold = 1
    test_case = np.random.choice(range(n_folds_outer), 1)[0]
    for train_idx, val_idx in outer_cv.split(X, y):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # X_train = X.iloc[train_idx] if is_dataframe else X[train_idx]
        # X_val = X.iloc[val_idx] if is_dataframe else X[val_idx]
        # # No need for special indexing for y as it's either an array or series now
        # y_train = y[train_idx]
        # y_val = y[val_idx]

        if use_nested_cv: 
        
            # Grid search with cross-validation in the inner loop
            grid_search = GridSearchCV(rf_clf, param_grid_rf, scoring='f1_macro', cv=inner_cv, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_hyperparameters = grid_search.best_params_
        else: 
            # Use default hyperparameters
            best_hyperparameters = default_hyperparameters

        best_params_list.append(tuple(best_hyperparameters.items()))
        
        # Train the model using the best hyperparameters found
        best_rf = RandomForestClassifier(**best_hyperparameters, random_state=0)
        best_rf.fit(X_train, y_train)

         # Test 
        if fold == test_case: print(f"[model] Fold={fold}: Random forest, best params={best_hyperparameters}")
        best_params_list.append(tuple(best_hyperparameters.items()))
        
        # Predict and compute F1 score on the validation set
        y_pred = best_rf.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')
        outer_f1_scores.append(f1)
        
        # Predict probabilities for ROC AUC computation
        y_prob = best_rf.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_prob)
        outer_roc_auc_scores.append(roc_auc)

        # Compute feature importances (using Random Forest's feature importance attribute)
        feature_importances = best_rf.feature_importances_
        all_feature_importances.append(feature_importances)

        fold += 1

    # Compute average F1 score, average ROC AUC, and average feature importances
    output_dict = {}
    output_dict['f1'] = avg_f1 = np.mean(outer_f1_scores)
    output_dict['auc'] = output_dict['roc_auc'] = avg_roc_auc = np.mean(outer_roc_auc_scores)
    output_dict['feature_importance'] = avg_feature_importances = np.mean(all_feature_importances, axis=0)

    feature_names = kargs.get("feature_names", [])
    if len(feature_names) > 0: 
        assert len(feature_names) == X.shape[1]

        # Displaying the feature importances
        output_dict['feature_importance_df'] = \
            pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_feature_importances
        })    

    # Determine the most frequently selected hyperparameters
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_params = dict(most_common_params)

    # Also return the model with the best parameters (by mode)
    output_dict['model'] = RandomForestClassifier(**most_common_params, random_state=0)
    
    # return avg_f1, avg_roc_auc, avg_feature_importances, most_common_params
    return output_dict

# Running the function for Random Forest to get the results
# avg_f1_rf_v2, avg_roc_auc_rf_v2, avg_feature_importances_rf_v2, most_common_params_rf = nested_cv_random_forest_with_importance_v2(X_diabetes, y_diabetes_binary)

# avg_f1_rf_v2, avg_roc_auc_rf_v2, most_common_params_rf

# Gradient Boosting Tree ------------------------------


def nested_cv_gradient_boosting_with_importance(X, y, **kargs):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score, roc_auc_score

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        X = X.values
    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values
    
    # Define the gradient boosting classifier
    gb_clf = GradientBoostingClassifier(random_state=0)

    # Define the hyperparameter grid
    param_grid_gb = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2]
    }

    # Nested CV can be time consuming and we don't necessarily want to use it everytime 
    default_hyperparameters = kargs.get("default_hyperparams", None)
    use_nested_cv = kargs.get('use_nested_cv', True)
    if default_hyperparameters is None:
        # Default hyperparameters (you can adjust these based on domain knowledge or past experiments)
        default_hyperparameters = {
            'n_estimators': 100,
            'learning_rate': 0.01,
            'max_depth': 5,
            'subsample': 0.9,
            'min_samples_split': 2,
            'min_samples_leaf': 1
    }
    if not use_nested_cv and kargs.get('default_hyperparams', None): 
        print(f"[model] GBT: using known hyperparameters:\n{default_hyperparameters}\n")
    
    # Setup the inner and outer cross-validation
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    n_folds_inner = kargs.get("n_folds_inner", n_folds_outer)
    inner_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=n_folds_outer, shuffle=True, random_state=0)

    outer_f1_scores = []
    outer_roc_auc_scores = []
    all_feature_importances = []
    best_params_list = []

    # Outer cross-validation loop
    fold = 1 
    test_case = np.random.choice(range(n_folds_outer), 1)[0]
    for train_idx, val_idx in outer_cv.split(X, y):
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        # X_train = X.iloc[train_idx] if is_dataframe else X[train_idx]
        # X_val = X.iloc[val_idx] if is_dataframe else X[val_idx]
        # # No need for special indexing for y as it's either an array or series now
        # y_train = y[train_idx]
        # y_val = y[val_idx]

        if use_nested_cv: 
            # Grid search with cross-validation in the inner loop
            grid_search = GridSearchCV(gb_clf, param_grid_gb, scoring='f1_macro', cv=inner_cv, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_hyperparameters = grid_search.best_params_
        else: 
            best_hyperparameters = default_hyperparameters

        # Test 
        if fold == test_case: 
            print(f"[model] Fold={fold}: Gradient Boosting Tree, best params={best_hyperparameters}")
        
        best_params_list.append(tuple(best_hyperparameters.items()))

        # Train the model using the best hyperparameters found
        best_gb = GradientBoostingClassifier(**best_hyperparameters, random_state=0)
        best_gb.fit(X_train, y_train)
        
        # Predict and compute F1 score on the validation set
        y_pred = best_gb.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')
        outer_f1_scores.append(f1)
        
        # Predict probabilities for ROC AUC computation
        y_prob = best_gb.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_prob)
        outer_roc_auc_scores.append(roc_auc)

        # Compute feature importances (using Gradient Boosting's feature importance attribute)
        feature_importances = best_gb.feature_importances_
        all_feature_importances.append(feature_importances)

        fold +=1 

    # Compute average F1 score, average ROC AUC, and average feature importances
    output_dict = {}
    output_dict['f1'] = avg_f1 = np.mean(outer_f1_scores)
    output_dict['auc'] = output_dict['roc_auc'] = avg_roc_auc = np.mean(outer_roc_auc_scores)
    output_dict['feature_importance'] = avg_feature_importances = np.mean(all_feature_importances, axis=0)

    feature_names = kargs.get("feature_names", [])
    if len(feature_names) > 0: 
        assert len(feature_names) == X.shape[1]

        # Displaying the feature importances
        output_dict['feature_importance_df'] = \
            pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_feature_importances
        })    

    # Determine the most frequently selected hyperparameters
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_params = dict(most_common_params)

    # Also return the model with the best parameters (by mode)
    output_dict['model'] = GradientBoostingClassifier(**most_common_params, random_state=0)
    
    # return avg_f1, avg_roc_auc, avg_feature_importances, most_common_params
    return output_dict

# Running the function for Gradient Boosting to get the results
# avg_f1_gb, avg_roc_auc_gb, avg_feature_importances_gb, most_common_params_gb = nested_cv_gradient_boosting_with_importance(X_diabetes, y_diabetes_binary)

# avg_f1_gb, avg_roc_auc_gb, most_common_params_gb


# XGBoost
######### 

def xgb_model_selection_and_training(X, y, **kargs): 
    from sklearn.metrics import classification_report
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
    from sklearn.model_selection import train_test_split
    # from dask_ml.model_selection import train_test_split # faster than sklearn.model_selection's train_test_split

    # Design: Suppose that the input is a dataframe ...
    # -------------------------------------------
    # # Separate the data into features and target
    # non_feature_cols = kargs.get("non_feature_cols", ['label', 'transcript_id'])
    # col_label = kargs.get("col_label", "label")

    # X = df.drop(columns=non_feature_cols)
    # y = df[col_label]
    # # Encode the categorical variables
    # X = pd.get_dummies(X, drop_first=True)  # Todo: more robust way to cope with dummy variables?
    # -------------------------------------------

    # Define the classifier
    clf = XGBClassifier(random_state=0, eval_metric='mlogloss')
    # NOTE: use_label_encoder=False => UserWarning: `use_label_encoder` is deprecated in 1.7.0.

    param_grid = kargs.get("param_grid", {})
    if not param_grid: 
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1],
            'colsample_bytree': [0.8, 0.9, 1],
            'gamma': [0, 0.1, 0.2]
        }

    scoring = kargs.get("scoring", "f1_macro")
    n_folds = kargs.get("n_folds", 5)

    # Define the grid search
    grid_search = GridSearchCV(clf, param_grid, scoring=scoring, cv=n_folds, n_jobs=-1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the grid search
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Instantiate the classifier with the best parameters
    best_clf = XGBClassifier(
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        gamma=best_params['gamma'],
        n_estimators=best_params['n_estimators'],
        random_state=0, use_label_encoder=False, eval_metric='mlogloss'
    )

    # Fit the best classifier on the training data
    best_clf.fit(X_train, y_train)

    # Predict the labels for the test data
    y_pred = best_clf.predict(X_test)

    # Compute the feature importances
    feature_importances = best_clf.feature_importances_

    # Calculate the F1 score on the test set
    f1_score_value = classification_report(y_test, y_pred, output_dict=True)['macro avg']['f1-score']

    return f1_score_value, feature_importances

def nested_cv_xgb(X, y, **kargs):
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score, KFold,  GridSearchCV
    from sklearn.metrics import classification_report

    # Design: Suppose that the input is a dataframe ...
    # -------------------------------------------
    # # Separate the data into features and target
    # non_feature_cols = kargs.get("non_feature_cols", ['label', 'transcript_id'])
    # col_label = kargs.get("col_label", "label")

    # X = df.drop(columns=non_feature_cols)
    # y = df[col_label]
    # # Encode the categorical variables
    # X = pd.get_dummies(X, drop_first=True)  # Todo: more robust way to cope with dummy variables?
    # -------------------------------------------
    X = pd.DataFrame(X)
    
    # Define the classifier
    clf = XGBClassifier(random_state=0, eval_metric='mlogloss')
    # NOTE: use_label_encoder=False => UserWarning: `use_label_encoder` is deprecated in 1.7.0.
    
    # Parameters grid for hyperparameter tuning
    param_grid = kargs.get("param_grid", {})
    if not param_grid: 
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1],
            'colsample_bytree': [0.8, 1],
            'gamma': [0, 0.1]
        }
    
    # Inner CV for hyperparameter tuning
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)
    grid_search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=inner_cv, n_jobs=-1)

    # Outer CV for model evaluation
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=0)
    
    f1_scores = []
    feature_importances_list = []
    
    for train_idx, valid_idx in outer_cv.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        # y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        # Find best hyperparameters using inner CV
        grid_search.fit(X_train, y_train)
        
        # Train the model with the best hyperparameters on the training data
        best_clf = grid_search.best_estimator_
        best_clf.fit(X_train, y_train)
        
        # Predict and calculate F1 score on the validation data
        y_pred = best_clf.predict(X_valid)
        f1 = classification_report(y_valid, y_pred, output_dict=True)['macro avg']['f1-score']
        
        f1_scores.append(f1)
        feature_importances_list.append(best_clf.feature_importances_)
    
    # Average F1 score and feature importances
    avg_f1_score = np.mean(f1_scores)
    avg_feature_importances = np.mean(feature_importances_list, axis=0)
    
    return avg_f1_score, avg_feature_importances
    # NOTE: The function returns the average F1 score and average feature importances over the outer CV folds.

def to_array(X, y): 
    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        X = X.values
        # NOTE: Why converting X to a numpy array? 
        #       XGBoost imposes restrictions on these feature names: they must not contain 
        #       certain characters like [, ], or <.

    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values

    return (X, y)

def nested_cv_xgboost_with_importance(X, y, **kargs):
    # import xgboost as xgb
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score, KFold,  GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score, roc_auc_score

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        X = X.values
        # NOTE: Why converting X to a numpy array? 
        #       XGBoost imposes restrictions on these feature names: they must not contain 
        #       certain characters like [, ], or <.

    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values

    # Define the XGBoost classifier
    xgb_clf = XGBClassifier(random_state=0, eval_metric="logloss")
    # NOTE: use_label_encoder=False => UserWarning: `use_label_encoder` is deprecated in 1.7.0.

    # Define the hyperparameter grid
    param_grid_xgb = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]  
    }

    # Nested CV can be time consuming and we don't necessarily want to use it everytime 
    default_hyperparameters = kargs.get("default_hyperparams", None)
    use_nested_cv = kargs.get('use_nested_cv', True)
    if default_hyperparameters is None or not default_hyperparameters:
        # Default hyperparameters
        default_hyperparameters = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 0,
            'eval_metric': "logloss"
        }
        # NOTE: 'subsample': 
        #          Subsample ratio of the training instances. 
        #          Setting it to 0.5 means that XGBoost randomly collects half of the data instances to grow trees
        #      'colsample_bytree': 
        #         The fraction of features that can be selected for any given tree to train
    if not use_nested_cv: 
        if kargs.get('default_hyperparams', None): 
            print(f"[model] XGBoost: using known hyperparameters:\n{default_hyperparameters}\n")
        else: 
            print(f"[model] XGBoost: using default hyperparameters:\n{default_hyperparameters}\n")
    
    # Setup the inner and outer cross-validation
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    n_folds_inner = kargs.get("n_folds_inner", n_folds_outer)
    inner_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=n_folds_outer, shuffle=True, random_state=0)

    outer_f1_scores = []
    outer_roc_auc_scores = []
    all_feature_importances = []
    best_params_list = []

    # Outer cross-validation loop
    fold = 0
    test_case = np.random.choice(range(n_folds_outer), 1)[0]
    for train_idx, val_idx in outer_cv.split(X, y):
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # X_train = X.iloc[train_idx] if is_dataframe else X[train_idx]
        # X_val = X.iloc[val_idx] if is_dataframe else X[val_idx]
        # # No need for special indexing for y as it's either an array or series now
        # y_train = y[train_idx]
        # y_val = y[val_idx]

        if use_nested_cv:
            # Grid search with cross-validation in the inner loop
            grid_search = GridSearchCV(xgb_clf, param_grid_xgb, scoring='f1_macro', cv=inner_cv, n_jobs=-1)
            # NOTE: scoring 
            #       f1_macro, f1_micro, f1_weighted

            grid_search.fit(X_train, y_train)
            best_hyperparameters = grid_search.best_params_
        else: 
            # Use default hyperparameters
            best_hyperparameters = default_hyperparameters

        # Test 
        if fold == test_case: 
            print(f"[model] Fold={fold}: XGBoost, best params={best_hyperparameters}, via nested cv? {use_nested_cv}")
        best_params_list.append(tuple(best_hyperparameters.items()))
        
        # Train the model using the best hyperparameters found
        best_xgb = XGBClassifier(eval_metric="logloss", random_state=0)
        # best_xgb = XGBClassifier(n_estimators = best_hyperparameters['n_estimators'],
        #                             max_depth = best_hyperparameters['max_depth'],
        #                             learning_rate = best_hyperparameters['learning_rate'],
        #                             subsample = best_hyperparameters['subsample'],
        #                             colsample_bytree = best_hyperparameters['colsample_bytree'],
        #                             random_state=0, eval_metric="logloss") # use_label_encoder=False,
        best_xgb.set_params(**best_hyperparameters)
        best_xgb.fit(X_train, y_train)
        
        # Predict and compute F1 score on the validation set
        y_pred = best_xgb.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')  #
        outer_f1_scores.append(f1)
        
        # Predict probabilities for ROC AUC computation
        y_prob = best_xgb.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_prob)
        outer_roc_auc_scores.append(roc_auc)

        # Compute feature importances
        feature_importances = best_xgb.feature_importances_
        all_feature_importances.append(feature_importances)

        fold += 1

    # Compute average F1 score, average ROC AUC, and average feature importances
    output_dict = {}
    output_dict['f1'] = avg_f1 = np.mean(outer_f1_scores)
    output_dict['auc'] = output_dict['roc_auc'] = avg_roc_auc = np.mean(outer_roc_auc_scores)
    output_dict['feature_importance'] = avg_feature_importances = np.mean(all_feature_importances, axis=0)

    feature_names = kargs.get("feature_names", [])
    if len(feature_names) > 0: 
        assert len(feature_names) == X.shape[1]
        print(f"[test] Example feature names:\n{feature_names[:10]}\n")

        # Displaying the feature importances
        output_dict['feature_importance_df'] = \
            pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_feature_importances
        })    

    # Determine the most frequently selected hyperparameters
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_params = dict(most_common_params)

    # Also return the model with the best parameters (by mode)
    new_xgb = XGBClassifier(eval_metric="logloss", random_state=0)
    # output_dict['model'] = XGBClassifier(n_estimators=most_common_params['n_estimators'],
    #                                 max_depth=most_common_params['max_depth'],
    #                                 learning_rate=most_common_params['learning_rate'],
    #                                 subsample=most_common_params['subsample'],
    #                                 colsample_bytree=most_common_params['colsample_bytree'],
    #                                 random_state=0, eval_metric="logloss")
    output_dict['model'] = new_xgb.set_params(**most_common_params)

    # return avg_f1, avg_roc_auc, avg_feature_importances
    return output_dict

def compute_average_fpr_fnr(y_true, y_pred):
    """
    Computes the average False Positive Rate (FPR) and False Negative Rate (FNR)
    for multiclass classification.

    Parameters:
    - y_true: array-like of shape (n_samples,), True labels.
    - y_pred: array-like of shape (n_samples,), Predicted labels.

    Returns:
    - average_fpr: float, Average False Positive Rate across all classes.
    - average_fnr: float, Average False Negative Rate across all classes.
    """
    from sklearn.metrics import confusion_matrix

    # Compute the confusion matrix for multiclass classification
    conf_mat = confusion_matrix(y_true, y_pred)

    # Initialize lists to store FPR and FNR for each class
    fpr_list = []
    fnr_list = []

    # Calculate FPR and FNR for each class
    for i in range(conf_mat.shape[0]):  # Iterate over each class
        tn = conf_mat.sum() - conf_mat[i, :].sum() - conf_mat[:, i].sum() + conf_mat[i, i]
        fp = conf_mat[:, i].sum() - conf_mat[i, i]
        fn = conf_mat[i, :].sum() - conf_mat[i, i]
        tp = conf_mat[i, i]

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

        fpr_list.append(fpr)
        fnr_list.append(fnr)

    # Compute the average FPR and FNR across all classes
    average_fpr = np.mean(fpr_list)
    average_fnr = np.mean(fnr_list)

    return average_fpr, average_fnr


# hpbandster_wrapper.py
def import_hpbandster_search_cv():
    try:
        from hpbandster_sklearn import HpBandSterSearchCV
        return HpBandSterSearchCV
    except ImportError as e:
        error_message = str(e)
        if "_check_fit_params" in error_message:
            print(f"ImportError: {e}. Applying monkey patch for _check_fit_params.")
            
            # Import the necessary function from compat.py
            from .compat import _check_method_params
            
            # Monkey patch the sklearn.utils.validation module
            print("[action] Patching sklearn.utils.validation._check_fit_params")
            import sklearn.utils.validation
            sklearn.utils.validation._check_fit_params = _check_method_params
            
            # Try importing HpBandSterSearchCV again after the patch
            from hpbandster_sklearn import HpBandSterSearchCV
            return HpBandSterSearchCV
        else:
            raise e


def nested_cv_xgboost_with_importance_via_optuna_v0(X, y, **kargs):
    """
    Performs nested cross-validation with hyperparameter optimization using Optuna
    on an XGBoost classifier. Includes pruning, early stopping, and parallelization.
    Calculates performance metrics and feature importances.

    NOTE: This implementation is similar to nested_cv_xgboost_with_importance_via_optuna()
          but uses XGBoost's native library calls. 

    Parameters
    ----------
    X : array-like or pandas DataFrame
        Feature matrix.
    y : array-like or pandas Series/DataFrame
        Target vector.
    **kargs : dict
        Additional keyword arguments.

    Returns
    -------
    output_dict : dict
        Dictionary containing performance metrics, feature importances, and the best model.
    """
    import optuna
    from optuna.integration import XGBoostPruningCallback
    import xgboost as xgb
    from sklearn.metrics import (f1_score, roc_auc_score, precision_score, recall_score,
                                 confusion_matrix)
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from collections import Counter
    import multiprocessing
    import warnings

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Extract optional arguments
    feature_names = kargs.get("feature_names", [])
    scoring = kargs.get('scoring', 'f1_macro')  # Default scoring metric
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    n_folds_inner = kargs.get("n_folds_inner", 3)
    use_nested_cv = kargs.get('use_nested_cv', True)
    n_trials = kargs.get('n_trials', 20)  # Number of trials for Optuna
    n_jobs = kargs.get('n_jobs', 1)  # Use 1 core by default to avoid issues

    # Check if X is a DataFrame
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        if len(feature_names) == 0:
            feature_names = list(X.columns)
        X = X.values

    # Check if y is a DataFrame or Series
    is_y_dataframe = isinstance(y, (pd.DataFrame, pd.Series))
    if is_y_dataframe:
        y = y.values.ravel()

    classes = np.unique(y)
    n_classes = len(classes)

    # Set the objective and evaluation metric based on the number of classes
    xgb_objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    xgb_eval_metric = "logloss" if n_classes == 2 else "mlogloss"

    # Default hyperparameters
    default_hyperparameters = kargs.get("default_hyperparams", None)
    if default_hyperparameters is None or not default_hyperparameters:
        default_hyperparameters = {
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'objective': xgb_objective,
            'eval_metric': xgb_eval_metric,
            'tree_method': 'hist',  # Faster histogram-based algorithm
            'seed': 0,
            'n_jobs': n_jobs,
        }
        if n_classes > 2:
            default_hyperparameters['num_class'] = n_classes  # Add this line

    # Setup the inner and outer cross-validation
    inner_cv = StratifiedKFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    outer_cv = StratifiedKFold(n_splits=n_folds_outer, shuffle=True, random_state=0)

    # Initialize lists to collect metrics and hyperparameters
    outer_f1_scores = []
    outer_roc_auc_scores = []
    outer_fp_rates = []
    outer_fn_rates = []
    outer_precisions = []
    outer_recalls = []
    all_feature_importances = []
    best_params_list = []
    cv_scores = []

    # Define the scoring function
    from sklearn.metrics import get_scorer
    scoring_func = get_scorer(scoring)

    # Suppress Optuna's verbosity
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Outer cross-validation loop
    fold = 0
    test_case = np.random.choice(range(n_folds_outer), 1)[0]

    for train_idx, val_idx in tqdm(outer_cv.split(X, y), total=n_folds_outer, desc="Outer CV"):
        X_train_full, X_val, y_train_full, y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]

        # Split X_train_full into training and validation sets for early stopping
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )

        if use_nested_cv:
            # Define the Optuna objective function
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
                    'objective': xgb_objective,
                    'eval_metric': xgb_eval_metric,
                    'tree_method': 'hist',
                    'seed': 0,
                    'n_jobs': n_jobs,
                }
                if n_classes > 2:
                    params['num_class'] = n_classes  # Add this line

                # Create DMatrix for training
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dvalid = xgb.DMatrix(X_valid, label=y_valid)
                evals = [(dvalid, 'validation')]

                # Use early stopping and pruning
                pruning_callback = XGBoostPruningCallback(trial, f"validation-{xgb_eval_metric}")

                bst = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=500,
                    evals=evals,
                    early_stopping_rounds=20,
                    callbacks=[pruning_callback],
                    verbose_eval=False
                )

                # Predict on the validation set
                preds = bst.predict(dvalid)
                if n_classes == 2:
                    preds_class = (preds > 0.5).astype(int)
                else:
                    preds_class = np.argmax(preds, axis=1)

                # Compute the score using the user-specified scoring function
                score = scoring_func._score_func(y_valid, preds_class, **scoring_func._kwargs)
                return score

            # Set up Optuna study with pruner and parallelization
            study_name = f"xgboost_opt_fold_{fold}"
            db_filename = f"optuna_study_fold_{fold}.db"
            storage_name = f"sqlite:///{db_filename}"
            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                study_name=study_name,
                storage=storage_name,
                load_if_exists=True,
            )

            # Optimize the objective function
            study.optimize(
                objective,
                n_trials=n_trials,
                n_jobs=1,  # Set to 1 to avoid issues with nested parallelism
                show_progress_bar=True
            )

            # Retrieve the best hyperparameters
            best_hyperparameters = study.best_params
            # Ensure fixed parameters are included
            best_hyperparameters.update({
                'objective': xgb_objective,
                'eval_metric': xgb_eval_metric,
                'tree_method': 'hist',
                'seed': 0,
                'n_jobs': n_jobs,
            })
            if n_classes > 2:
                best_hyperparameters['num_class'] = n_classes  # Add this line

            # Delete the .db file after optimization
            try:
                os.remove(db_filename)
            except OSError as e:
                print(f"Error deleting the database file: {e}")

            # For debugging or inspection
            if fold == test_case:
                print(f"\n[model] XGBoost: Fold={fold}")
                print(f"... Best hyperparameters:\n{best_hyperparameters}\n")
                print(f"... Best {scoring}: {study.best_value}\n")
                trials_df = study.trials_dataframe()
                print(f"... Trials DataFrame:\n{trials_df.head()}\n")

        else:
            # Use default hyperparameters
            best_hyperparameters = default_hyperparameters

        best_params_list.append(tuple(best_hyperparameters.items()))

        # Train the model with the best hyperparameters
        # Create DMatrix for training and validation
        dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)
        dval = xgb.DMatrix(X_val, label=y_val)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        evals = [(dvalid, 'validation')]

        # Use early stopping
        bst = xgb.train(
            best_hyperparameters,
            dtrain_full,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=False
        )

        # Predict on the validation set
        preds = bst.predict(dval)
        if n_classes == 2:
            y_pred = (preds > 0.5).astype(int)
        else:
            y_pred = np.argmax(preds, axis=1)

        # Compute metrics
        f1 = f1_score(y_val, y_pred, average='macro')
        outer_f1_scores.append(f1)

        precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
        outer_precisions.append(precision)

        recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
        outer_recalls.append(recall)

        # ROC AUC Score
        if n_classes == 2:
            roc_auc = roc_auc_score(y_val, preds)
        else:
            roc_auc = roc_auc_score(y_val, preds, multi_class='ovr')
        outer_roc_auc_scores.append(roc_auc)

        # Compute confusion matrix
        if n_classes == 2:
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        else:
            # Compute average FPR and FNR for multiclass
            cm = confusion_matrix(y_val, y_pred, labels=classes)
            fp = cm.sum(axis=0) - np.diag(cm)
            fn = cm.sum(axis=1) - np.diag(cm)
            tn = cm.sum() - (fp + fn + np.diag(cm))
            fpr = np.mean(fp / (fp + tn + 1e-8))
            fnr = np.mean(fn / (fn + np.diag(cm) + 1e-8))
        outer_fp_rates.append(fpr)
        outer_fn_rates.append(fnr)

        # Collect cross-validation scores
        cv_scores.append((best_hyperparameters, f1))

        # Compute feature importances
        feature_importances = bst.get_score(importance_type='gain')
        # Map importances to feature indices
        importances_array = np.zeros(X.shape[1])
        for feat, importance in feature_importances.items():
            idx = int(feat[1:])  # Feature names are like 'f0', 'f1', etc.
            importances_array[idx] = importance
        all_feature_importances.append(importances_array)

        fold += 1

    # Aggregate performance metrics
    output_dict = {
        'f1': np.mean(outer_f1_scores),
        'auc': np.mean(outer_roc_auc_scores),
        'roc_auc': np.mean(outer_roc_auc_scores),  # Alias
        'precision': np.mean(outer_precisions),
        'recall': np.mean(outer_recalls),
        'fpr': np.mean(outer_fp_rates),
        'fnr': np.mean(outer_fn_rates),
        'feature_importance': np.mean(all_feature_importances, axis=0),
    }

    if len(feature_names) > 0:
        output_dict['feature_importance_df'] = pd.DataFrame({
            'Feature': feature_names,
            'Importance': output_dict['feature_importance']
        })

    # Determine the most common and best-scoring hyperparameters
    print("\n[model] Choosing the best hyperparameter setting ...")
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_hyperparams = dict(most_common_params)
    print(f"... Most common hyperparameters:\n{most_common_hyperparams}\n")

    cv_scores_sorted = sorted(cv_scores, key=lambda x: x[1], reverse=True)
    best_scoring_params = cv_scores_sorted[0][0]
    best_score = cv_scores_sorted[0][1]
    output_dict['best_scoring_hyperparams'] = best_scoring_params
    print(f"... Highest scoring hyperparameters (score={best_score}):\n{best_scoring_params}\n")
    for i in range(min(3, len(cv_scores_sorted))):
        p, s = cv_scores_sorted[i][0], cv_scores_sorted[i][1]
        print(f"...... Rank #{i+1}: score={s}, setting: {p}")

    # Choose the best hyperparameters
    output_dict['best_hyperparams'] = best_hyperparams = output_dict['best_scoring_hyperparams']

    # Initialize the best model with the best hyperparameters
    # Train on the full dataset
    dtrain_full = xgb.DMatrix(X, label=y)
    bst_full = xgb.train(
        best_hyperparams,
        dtrain_full,
        num_boost_round=500,
        evals=[(dtrain_full, 'train')],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    output_dict['best_model'] = output_dict['model'] = bst_full

    return output_dict


def nested_cv_xgboost_with_importance_via_optuna(X, y, **kargs):
    """
    Performs nested cross-validation with hyperparameter optimization using Optuna
    on an XGBoost classifier. Includes early stopping and parallelization.
    Calculates performance metrics and feature importances.

    Parameters
    ----------
    X : array-like or pandas DataFrame
        Feature matrix.     
    y : array-like or pandas Series/DataFrame   
        Target vector.

    Memo
    ----
    Hyperparameter Optimization with Optuna:
        - Define the Objective Function:
            The objective function is defined inside the outer loop.
            It suggests hyperparameters using trial.suggest_* methods.
            Initializes an XGBClassifier with these hyperparameters.
            Performs cross-validation using cross_val_score on the training data.
            Returns the mean cross-validation score as the objective to maximize.
        - Create and Optimize the Optuna Study:
            A study is created with optuna.create_study(direction='maximize').
            The study is optimized using study.optimize(objective, n_trials=n_trials), 
            where n_trials is the number of hyperparameter sets to try.
        - Retrieve Best Hyperparameters:
            After optimization, study.best_params contains the best hyperparameters found.
            Additional fixed hyperparameters (e.g., objective, random_state, etc.) are added to best_hyperparameters.
        - Model Training and Evaluation: 
            The best model is initialized with XGBClassifier(**best_hyperparameters).
            The model is trained on the X_train data.
            Predictions are made on the validation set X_val.
            Performance metrics (F1 score, ROC AUC, precision, recall, etc.) are calculated.
        - Collecting and Aggregating Results:
            Performance metrics and feature importances are collected across folds.
            The most common and best-scoring hyperparameters are determined.
            An output dictionary output_dict is created to store all relevant results.

    Output Messages: 
        - E.g. 
           [I 2024-10-10 21:37:37,916] Trial 16 finished with value: 0.9138 and parameters: {...}
           
           The value reported (0.9138 in this case) corresponds to the mean of the scoring metric 
           used in the objective function during that trial.

           Specifically, the value is the mean cross-validation score obtained using the hyperparameters

            + Hyperparameter Suggestion: For each trial, Optuna suggests a set of hyperparameters.
            + Model Initialization: An XGBClassifier is initialized with these hyperparameters.
            + Cross-Validation: 
                cross_val_score performs cross-validation on the training data using the specified scoring metric.
                The scoring metric is passed to cross_val_score.
                By default, scoring='f1_macro' as set in this function
    """
    import optuna
    import xgboost as xgb
    from sklearn.metrics import (
        f1_score, 
        roc_auc_score, 
        precision_score, recall_score,
        confusion_matrix
    )
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from collections import Counter
    # import numpy as np
    # import pandas as pd
    # from tqdm import tqdm
    import multiprocessing
    import warnings

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Extract optional arguments
    feature_names = kargs.get("feature_names", [])
    scoring = kargs.get('scoring', 'f1_macro')  # Default scoring metric
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    n_folds_inner = kargs.get("n_folds_inner", 3)
    use_nested_cv = kargs.get('use_nested_cv', False)
    n_trials = kargs.get('n_trials', 20)  # Number of trials for Optuna
    n_jobs = kargs.get('n_jobs', -1)  # Use all available cores by default

    # Check if X is a DataFrame
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        if len(feature_names) == 0:
            feature_names = list(X.columns)
        X = X.values

    # Check if y is a DataFrame or Series
    is_y_dataframe = isinstance(y, (pd.DataFrame, pd.Series))
    if is_y_dataframe:
        y = y.values.ravel()

    classes = np.unique(y)
    n_classes = len(classes)

    # Set the objective and evaluation metric based on the number of classes
    xgb_objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    xgb_eval_metric = "logloss" if n_classes == 2 else "mlogloss"

    # Default hyperparameters
    default_hyperparameters = kargs.get("default_hyperparams", None)
    if default_hyperparameters is None or not default_hyperparameters:
        default_hyperparameters = {
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'objective': xgb_objective,
            'eval_metric': xgb_eval_metric,
            'tree_method': 'hist',  # Faster histogram-based algorithm
            'random_state': 0,
            'n_jobs': n_jobs,
            'early_stopping': 20,  # Set early_stopping here
        }
        if n_classes > 2:
            default_hyperparameters['num_class'] = n_classes

    # Setup the inner and outer cross-validation
    inner_cv = StratifiedKFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    outer_cv = StratifiedKFold(n_splits=n_folds_outer, shuffle=True, random_state=0)

    # Initialize lists to collect metrics and hyperparameters
    outer_f1_scores = []
    outer_roc_auc_scores = []
    outer_fp_rates = []
    outer_fn_rates = []
    outer_precisions = []
    outer_recalls = []
    all_feature_importances = []
    best_params_list = []
    cv_scores = []

    # Define the scoring function
    from sklearn.metrics import get_scorer
    scoring_func = get_scorer(scoring)

    # Suppress Optuna's verbosity
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Outer cross-validation loop
    fold = 0
    test_case = np.random.choice(range(n_folds_outer), 1)[0]

    for train_idx, val_idx in tqdm(outer_cv.split(X, y), total=n_folds_outer, desc="Outer CV"):
        X_train_full, X_val, y_train_full, y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]

        # Split X_train_full into training and validation sets for early stopping
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )

        if use_nested_cv:
            # Define the Optuna objective function
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
                    'objective': xgb_objective,
                    'eval_metric': xgb_eval_metric,
                    'tree_method': 'hist',
                    'random_state': 0,
                    'n_jobs': n_jobs,
                    'early_stopping': 20,  # Set early_stopping here
                }
                if n_classes > 2:
                    params['num_class'] = n_classes

                model = xgb.XGBClassifier(**params)

                # Use early stopping by setting early_stopping during initialization
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    verbose=False
                )

                # Predict on the validation set
                preds = model.predict(X_valid)

                # Compute the score using the user-specified scoring function
                score = scoring_func._score_func(y_valid, preds, **scoring_func._kwargs)
                return score

            # Set up Optuna study with parallelization
            study_name = f"xgboost_opt_fold_{fold}"
            db_filename = f"optuna_study_fold_{fold}.db"
            storage_name = f"sqlite:///{db_filename}"
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=storage_name,
                load_if_exists=True,
            )

            # Optimize the objective function
            study.optimize(
                objective,
                n_trials=n_trials,
                n_jobs=1,  # Set to 1 to avoid issues with nested parallelism
                show_progress_bar=True
            )

            # Retrieve the best hyperparameters
            best_hyperparameters = study.best_params
            # Ensure fixed parameters are included
            best_hyperparameters.update({
                'objective': xgb_objective,
                'eval_metric': xgb_eval_metric,
                'tree_method': 'hist',
                'random_state': 0,
                'n_jobs': n_jobs,
                'early_stopping': 20,  # Set early_stopping here
            })
            if n_classes > 2:
                best_hyperparameters['num_class'] = n_classes

            # Delete the .db file after optimization
            try:
                os.remove(db_filename)
            except OSError as e:
                print(f"Error deleting the database file: {e}")

            # For debugging or inspection
            if fold == test_case:
                print(f"\n[model] XGBoost: Fold={fold}")
                print(f"... Best hyperparameters:\n{best_hyperparameters}\n")
                print(f"... Best {scoring}: {study.best_value}\n")
                trials_df = study.trials_dataframe()
                print(f"... Trials DataFrame:\n{trials_df.head()}\n")

        else:
            # Use default hyperparameters
            best_hyperparameters = default_hyperparameters

        best_params_list.append(tuple(best_hyperparameters.items()))

        # Train the model with the best hyperparameters
        model = xgb.XGBClassifier(**best_hyperparameters)
        model.fit(
            X_train_full, y_train_full,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )

        # Predict on the validation set
        y_pred = model.predict(X_val)

        # Compute metrics
        f1 = f1_score(y_val, y_pred, average='macro')
        outer_f1_scores.append(f1)

        precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
        outer_precisions.append(precision)

        recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
        outer_recalls.append(recall)

        # ROC AUC Score
        if n_classes == 2:
            y_prob = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_prob)
        else:
            y_prob = model.predict_proba(X_val)
            roc_auc = roc_auc_score(y_val, y_prob, multi_class='ovr')
        outer_roc_auc_scores.append(roc_auc)

        # Compute confusion matrix
        if n_classes == 2:
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        else:
            # Compute average FPR and FNR for multiclass
            cm = confusion_matrix(y_val, y_pred, labels=classes)
            fp = cm.sum(axis=0) - np.diag(cm)
            fn = cm.sum(axis=1) - np.diag(cm)
            tn = cm.sum() - (fp + fn + np.diag(cm))
            fpr = np.mean(fp / (fp + tn + 1e-8))
            fnr = np.mean(fn / (fn + np.diag(cm) + 1e-8))
        outer_fp_rates.append(fpr)
        outer_fn_rates.append(fnr)

        # Collect cross-validation scores
        cv_scores.append((best_hyperparameters, f1))

        # Compute feature importances
        feature_importances = model.feature_importances_
        all_feature_importances.append(feature_importances)

        fold += 1

    # Aggregate performance metrics
    output_dict = {
        'f1': np.mean(outer_f1_scores),
        'auc': np.mean(outer_roc_auc_scores),
        'roc_auc': np.mean(outer_roc_auc_scores),  # Alias
        'precision': np.mean(outer_precisions),
        'recall': np.mean(outer_recalls),
        'fpr': np.mean(outer_fp_rates),
        'fnr': np.mean(outer_fn_rates),
        'feature_importance': np.mean(all_feature_importances, axis=0),
    }

    if len(feature_names) > 0:
        output_dict['feature_importance_df'] = pd.DataFrame({
            'Feature': feature_names,
            'Importance': output_dict['feature_importance']
        })

    # Determine the most common and best-scoring hyperparameters
    print("\n[model] Choosing the best hyperparameter setting ...")
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_hyperparams = dict(most_common_params)
    print(f"... Most common hyperparameters:\n{most_common_hyperparams}\n")

    cv_scores_sorted = sorted(cv_scores, key=lambda x: x[1], reverse=True)
    best_scoring_params = cv_scores_sorted[0][0]
    best_score = cv_scores_sorted[0][1]
    output_dict['best_scoring_hyperparams'] = best_scoring_params
    print(f"... Highest scoring hyperparameters (score={best_score}):\n{best_scoring_params}\n")
    for i in range(min(3, len(cv_scores_sorted))):
        p, s = cv_scores_sorted[i][0], cv_scores_sorted[i][1]
        print(f"...... Rank #{i+1}: score={s}, setting: {p}")

    # Choose the best hyperparameters
    output_dict['best_hyperparams'] = best_hyperparams = output_dict['best_scoring_hyperparams']

    # Initialize the best model with the best hyperparameters
    best_model = xgb.XGBClassifier(**best_hyperparams)
    best_model.fit(
        X, y,
        verbose=False
    )

    output_dict['best_model'] = output_dict['model'] = best_model

    return output_dict


def nested_cv_xgboost_with_importance_via_bohb(X, y, **kargs):
    """
    BOHB stands for Bayesian Optimization and Hyperband. It combines two powerful concepts:
    - Bayesian Optimization
    - Hyperband

    Memo
    ----
    * If you're using scikit-learn version 0.22 or newer, the internal _fit_and_score function 
      has a different signature than in earlier versions. 
      - hpbandster_sklearn was designed to work with older versions of scikit-learn. 
      - To ensure compatibility, you should downgrade scikit-learn to version 0.21.3.

    * Resource parameters,
        resource_name='n_samples', # can be either 'n_samples' or a string corresponding to an estimator attribute, eg. 'n_estimators' for an ensemble
        resource_type=float, # if specified, the resource value will be cast to that type before being passed to the estimator, otherwise it will be derived automatically
        min_budget=0.7,
        max_budget=1
        ... 
        - If not given sufficient resource (e.g. low min_budget), the model performance can be low
    
    References
    ----------
    1. https://dzone.com/articles/bayesian-optimization-and-hyperband-bohb-hyperpara
    2. https://automl.github.io/HpBandSter/build/html/quickstart.html
    """
    # from sklearn.pipeline import Pipeline
    # from sklearn.metrics import classification_report
    import xgboost as xgb
    # from hyperopt import hp
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.metrics import confusion_matrix, precision_score, recall_score
    from sklearn.metrics import log_loss
    # from sklearn.metrics import multilabel_confusion_matrix
    from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
    from collections import Counter

    # from hpbandster.core.worker import Worker
    # from hpbandster.optimizers import BOHB 

    # Use the wrapper function to import HpBandSterSearchCV
    HpBandSterSearchCV = import_hpbandster_search_cv()

    from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
    # from ConfigSpace import UniformIntegerHyperparameter
    # import ConfigSpace as CS
    # import ConfigSpace.hyperparameters as CSH

    feature_names = kargs.get("feature_names", [])

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        if len(feature_names) == 0: feature_names = list(X.columns)
        X = X.values
        # NOTE: Why converting X to a numpy array? 
        #       XGBoost imposes restrictions on these feature names: they must not contain 
        #       certain characters like [, ], or <.

    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values
    classes = np.unique(y)
    n_classes = len(classes)

    objective = "binary:logistic"
    if n_classes > 2: 
        objective = "multi:softprob"

    # Define the XGBoost classifier (this is incorporated into the objective function)
    model = xgb.XGBClassifier(objective=objective, random_state=0, eval_metric="logloss", importance_type='gain')
        
    # NOTE: use_label_encoder=False => UserWarning: `use_label_encoder` is deprecated in 1.7.0.

    # Define the hyperparameter grid (this is defined through the objective function)
    config_space = {
        "objective": objective, # "binary:logistic", 
        'n_estimators': Integer("n_estimators", bounds=(25, 1000), q=25),
        # UniformIntegerHyperparameter(name='n_estimator', lower=10, upper=1000, log=False), 
        # # hp.quniform('n_estimators', 100, 1000, log=True),
       
        'max_depth': Integer("max_depth", bounds=(1, 11), q=1), # hp.quniform('max_depth', 1, 10, q=1),
        'subsample': Float("subsample", bounds=(0.05, 1.0)), # hp.uniform('subsample', 0.05, 1.0),
        'learning_rate': Float("learning_rate", bounds=(0.001, 0.1), log=True), # hp.loguniform('learning_rate', 0.001, 0.1), # cf in optuna: trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        'gamma': Float("gamma", bounds=(0.01, 1.0), log=True),  # hp.loguniform('gamma', 0.01, 1.0),
        'colsample_bytree': Float("colsample_bytree", bounds=(0.5, 1.0), log=False),   # hp.uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': Float("reg_alpha", bounds=(0.01, 1.0), log=True),  # hp.loguniform('reg_alpha', 0.01, 1.0),
        'reg_lambda': Float("reg_lambda", bounds=(0.01, 1.0), log=True) # hp.loguniform('reg_lambda', 0.01, 1.0),
    }
    search_space = kargs.get('search_space', config_space)
    # NOTE: 
    #   `colsample_bytree` is the subsample ratio of columns when constructing each tree.
    # 6 * 6 * 4 * 3 * 3
    scoring = kargs.get('scoring', 'f1_macro') # 'roc_auc' 
    # NOTE: this scoring is only for the use of HpBandSterSearchCV() 
    #       XGBoost's eval_metric has a different allowable metrics

    # Nested CV can be time consuming and we don't necessarily want to use it everytime 
    default_hyperparameters = kargs.get("default_hyperparams", None)
    use_nested_cv = kargs.get('use_nested_cv', True)
    if default_hyperparameters is None or not default_hyperparameters:
        # Default hyperparameters
        default_hyperparameters = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 0,
            'eval_metric': "logloss"
        }
        # NOTE: 'subsample': 
        #          Subsample ratio of the training instances. 
        #          Setting it to 0.5 means that XGBoost randomly collects half of the data instances to grow trees
        #      'colsample_bytree': 
        #         The fraction of features that can be selected for any given tree to train
    if not use_nested_cv: 
        if kargs.get('default_hyperparams', None): 
            print(f"[model] XGBoost: using known hyperparameters:\n{default_hyperparameters}\n")
        else: 
            print(f"[model] XGBoost: using default hyperparameters:\n{default_hyperparameters}\n")
    
    # Setup the inner and outer cross-validation
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    n_folds_inner = kargs.get("n_folds_inner", n_folds_outer)
    
    # inner_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    # outer_cv = KFold(n_splits=n_folds_outer, shuffle=True, random_state=0)
    inner_cv = StratifiedKFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    outer_cv = StratifiedKFold(n_splits=n_folds_outer, shuffle=True, random_state=0)

    outer_f1_scores = []
    outer_roc_auc_scores = []
    outer_fp_rates = []
    outer_fn_rates = []
    outer_precisions = []
    outer_recalls = []
    all_feature_importances = []
    best_params_list = []

    # Outer cross-validation loop
    search = None
    max_iter = 120
    
    fold = 0
    test_case = np.random.choice(range(n_folds_outer), 1)[0]
    cv_scores = []
    for train_idx, val_idx in tqdm(outer_cv.split(X, y), total=n_folds_outer):
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if use_nested_cv:
            param_distributions = ConfigurationSpace(
                    name="nmd_opt_xgboost",
                    seed=42, 
                    space=search_space,
            )
            search = HpBandSterSearchCV(model, 
                        param_distributions, 
                        scoring=scoring,
                        random_state=0, 
                            warm_start=False,
                            resource_name='n_samples', # can be either 'n_samples' or a string corresponding to an estimator attribute, eg. 'n_estimators' for an ensemble
                            resource_type=float, # if specified, the resource value will be cast to that type before being passed to the estimator, otherwise it will be derived automatically
                            min_budget=0.75,
                            max_budget=1,
                                n_jobs=5, 
                                verbose=1).fit(X_train, y_train) # n_iter=10,      

            best_hyperparameters = search.best_params_

        else: 
            # Use default hyperparameters
            best_hyperparameters = default_hyperparameters

        # Test 
        if fold == test_case: 
            print(f"[model] XGBoost: Fold={fold}")
            if use_nested_cv: 
                print(f"... best hyperparams:\n{best_hyperparameters}\n")
                # print(f"... Best/min log loss:", search.best_value)
                print(f"... best score ({scoring}): {search.best_score_}")
                print()

                cv_results = pd.DataFrame(search.cv_results_)
                print(f"... CV results")
                print(cv_results.head()); print()

                # hist = pd.DataFrame(search.history_)
                # print(f"... history:")
                # print(hist.head())
                # NOTE: 'BayesSearchCV' object has no attribute 'history_'

        best_params_list.append(tuple(best_hyperparameters.items()))

        best_model = None
        y_pred = y_prob = None
        if use_nested_cv: 
            # Predict and compute F1 score on the validation set
            y_pred = search.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='macro')  #
            outer_f1_scores.append(f1)

            # Predict probabilities for ROC AUC computation
            if n_classes == 2:
                y_prob = search.predict_proba(X_val)[:, 1]
                roc_auc = roc_auc_score(y_val, y_prob)
            else: 
                y_prob = best_model.predict_proba(X_val) # shape: (n_samples, n_classes)
                roc_auc = roc_auc_score(y_val, y_prob, multi_class='ovr') # either 'ovr' (One-vs-Rest) or 'ovo' (One-vs-One)

            outer_roc_auc_scores.append(roc_auc)

            best_model = search.best_estimator_
        else: 
            # Refit the model using the best hyperparameters found

            best_model = xgb.XGBClassifier(objective=objective, 
                                           random_state=0, eval_metric="logloss", importance_type='gain') # metric: aucpr, logloss
            best_model.set_params(**best_hyperparameters)
            best_model.fit(X_train, y_train)
        
            # Predict and compute F1 score on the validation set
            y_pred = best_model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='macro')  #
            outer_f1_scores.append(f1)
            
            # Predict probabilities for ROC AUC computation
            if n_classes == 2: 
                y_prob = best_model.predict_proba(X_val)[:, 1]
                roc_auc = roc_auc_score(y_val, y_prob)
            else: 
                y_prob = best_model.predict_proba(X_val) # shape: (n_samples, n_classes)
                roc_auc = roc_auc_score(y_val, y_prob, multi_class='ovr') # either 'ovr' (One-vs-Rest) or 'ovo' (One-vs-One)
                # NOTE: multi_class='ovr' computes the ROC AUC score as a macro-average over all classes

            outer_roc_auc_scores.append(roc_auc)

        # ----- Other metrics ----- 

        # Compute confusion matrix
        if n_classes == 2: 
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

            # Calculate False Positive Rate and False Negative Rate
            fpr = fp / (fp + tn)  # False Positive Rate
            fnr = fn / (fn + tp)  # False Negative Rate
        else: 
            average_fpr, average_fnr = compute_average_fpr_fnr(y_val, y_pred)
            fpr = average_fpr
            fnr = average_fnr

        outer_fp_rates.append(fpr)
        outer_fn_rates.append(fnr)

        # Calculate Precision and Recall
        precision = precision_score(y_val, y_pred, average='macro')
        recall = recall_score(y_val, y_pred, average='macro')
        outer_precisions.append(precision)
        outer_recalls.append(recall)

        # CV Scores 
        best_params_score = (best_hyperparameters, f1)
        cv_scores.append(best_params_score)

        # Compute feature importances
        feature_importances = best_model.feature_importances_  # weights/frequency by default
        all_feature_importances.append(feature_importances)

        fold += 1

    # Compute average F1 score, average ROC AUC, and average feature importances
    output_dict = {}
    output_dict['f1'] = avg_f1 = np.mean(outer_f1_scores)
    output_dict['auc'] = output_dict['roc_auc'] = avg_roc_auc = np.mean(outer_roc_auc_scores)
    output_dict['precision'] = np.mean(outer_precisions)
    output_dict['recall'] = np.mean(outer_recalls)
    output_dict['fpr'] = np.mean(outer_fp_rates)
    output_dict['fnr'] = np.mean(outer_fn_rates)

    output_dict['feature_importance'] = avg_feature_importances = np.mean(all_feature_importances, axis=0)
    # output_dict['search'] = search # save a copy of the hyperparameter search result

    if len(feature_names) > 0: 
        assert len(feature_names) == X.shape[1]
        print(f"[test] Example feature names:\n{feature_names[:10]}\n")

        # Displaying the feature importances
        output_dict['feature_importance_df'] = \
            pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_feature_importances
        })    

    # Select the "best" hyperparameter settings from CV iterations
    print("[model] Choosing the best hyperparameter setting ...")
    # A. Choose the most frequently selected hyperparameters
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_params = dict(most_common_params)
    print(f"... most common hyperparam:\n{most_common_params}\n")

    # B. Choose the hyperparam setting that led to highest performance scores
    cv_scores_sorted = sorted(cv_scores, key=lambda x: x[1], reverse=True)
    best_scoring_params = cv_scores_sorted[0][0]
    best_score = cv_scores_sorted[0][1]
    output_dict['best_scoring_hyperparams'] = best_scoring_params
    print(f"... highest scoring hyperparams (score={best_score}):\n{best_scoring_params}\n")
    for i in range(3): 
        p, s = cv_scores_sorted[i][0], cv_scores_sorted[i][1]
        print(f"...... rank #{i+1}: score={s}, setting: {p}")

    # How to choose the best? 
    output_dict['best_hyperparams'] = best_hyperparams = output_dict['best_scoring_hyperparams']

    # Also return the model with the best parameters (by mode)
    # new_xgb = xgb.dask.DaskXGBClassifier(eval_metric="logloss", random_state=0)  # objective ='reg:squarederror'
    new_xgb = xgb.XGBClassifier(objective=objective, random_state=0, eval_metric='logloss', importance_type='gain') # logloss, aucpr
    # NOTE: metric name may not be compatible with "scoring" parameter
    #       - Unknown metric function f1_macro

    output_dict['best_model']=output_dict['model'] = new_xgb.set_params(**best_hyperparams) # this is not fitted yet

    # return avg_f1, avg_roc_auc, avg_feature_importances
    return output_dict

def nested_cv_xgboost_with_importance_via_bayes_opt_optuna(X, y, **kargs):
    """
    
    Memo
    ----
    1. https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html
    """
    import xgboost as xgb
    import optuna
    # from collections import Counter
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score, roc_auc_score, log_loss
    from sklearn.model_selection import cross_val_score, KFold,  GridSearchCV    

    def objective(trial):
        search_space_default = {
            "objective": "binary:logistic", # "reg:squarederror",
            "n_estimators": trial.suggest_int("n_estimators", 25, 1000, step=25, log=False),
            "verbosity": 0,
            # "eval_metric": scoring, 
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }
        search_space = params = kargs.get('search_space', search_space_default)

        model = xgb.XGBClassifier(**params) # xgb.XGBRegressor(**params), eval_metric="logloss"
        model.fit(X_train, y_train, verbose=False)
        y_prob = model.predict_proba(X_val)[:, 1]
        loss = log_loss(y_val, y_prob)
        return loss

    feature_names = kargs.get("feature_names", [])

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        if len(feature_names) == 0: feature_names = list(X.columns)
        X = X.values
        # NOTE: Why converting X to a numpy array? 
        #       XGBoost imposes restrictions on these feature names: they must not contain 
        #       certain characters like [, ], or <.

    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values
    classes = np.unique(y)
    n_classes = len(classes)

    # objective = "binary:logistic"
    # if n_classes > 2: 
    #     objective = "multi:softmax"

    # Define the XGBoost classifier (this is incorporated into the objective function)
    # model = xgb.XGBClassifier(random_state=0, eval_metric="logloss") # objective="binary:logistic"
        
    # NOTE: use_label_encoder=False => UserWarning: `use_label_encoder` is deprecated in 1.7.0.

    # Define the hyperparameter grid (this is defined through the objective function)
    # param_grid_xgb = {
    #         "objective": "binary:logistic", # "reg:squarederror",
    #         "n_estimators": 1000,
    #         "verbosity": 0,
    #         "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
    #         "max_depth": trial.suggest_int("max_depth", 1, 10),
    #         "subsample": trial.suggest_float("subsample", 0.05, 1.0),
    #         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
    #         "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    # }
    # search_space = kargs.get('search_space', param_grid_xgb)
    # NOTE: 
    #   `colsample_bytree` is the subsample ratio of columns when constructing each tree.
    # 6 * 6 * 4 * 3 * 3
    scoring = kargs.get('scoring', 'aucpr')

    # Nested CV can be time consuming and we don't necessarily want to use it everytime 
    default_hyperparameters = kargs.get("default_hyperparams", None)
    use_nested_cv = kargs.get('use_nested_cv', True)
    if default_hyperparameters is None or not default_hyperparameters:
        # Default hyperparameters
        default_hyperparameters = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 0,
            'eval_metric': scoring # "logloss"
        }
        # NOTE: 'subsample': 
        #          Subsample ratio of the training instances. 
        #          Setting it to 0.5 means that XGBoost randomly collects half of the data instances to grow trees
        #      'colsample_bytree': 
        #         The fraction of features that can be selected for any given tree to train
    if not use_nested_cv: 
        if kargs.get('default_hyperparams', None): 
            print(f"[model] XGBoost: using known hyperparameters:\n{default_hyperparameters}\n")
        else: 
            print(f"[model] XGBoost: using default hyperparameters:\n{default_hyperparameters}\n")
    
    # Setup the inner and outer cross-validation
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    n_folds_inner = kargs.get("n_folds_inner", n_folds_outer)
    inner_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=n_folds_outer, shuffle=True, random_state=0)

    outer_f1_scores = []
    outer_roc_auc_scores = []
    all_feature_importances = []
    best_params_list = []

    # Outer cross-validation loop
    search = None
    max_iter = 120
    
    fold = 0
    test_case = np.random.choice(range(n_folds_outer), 1)[0]
    cv_scores = []
    for train_idx, val_idx in tqdm(outer_cv.split(X, y), total=n_folds_outer):
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if use_nested_cv:

            search = optuna.create_study(direction='minimize')
            search.optimize(objective, n_trials=30)

            best_hyperparameters = search.best_params

        else: 
            # Use default hyperparameters
            best_hyperparameters = default_hyperparameters

        # Test 
        if fold == test_case: 
            print(f"[model] XGBoost: Fold={fold}")
            if use_nested_cv: 
                print(f"... best hyperparams:\n{best_hyperparameters}\n")
                print(f"... Best/min log loss:", search.best_value)
                # print(f"... best score: {search.best_score_}")
                print()

                # cv_results = pd.DataFrame(search.cv_results_)
                # print(f"... CV results")
                # print(cv_results.head()); print()

                # hist = pd.DataFrame(search.history_)
                # print(f"... history:")
                # print(hist.head())
                # NOTE: 'BayesSearchCV' object has no attribute 'history_'

        best_params_list.append(tuple(best_hyperparameters.items()))
        
        # Train the model using the best hyperparameters found
        model = xgb.XGBClassifier(random_state=0, eval_metric=scoring) # aucpr, logloss
        model.set_params(**best_hyperparameters)
        model.fit(X_train, y_train)
        
        # Predict and compute F1 score on the validation set
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')  #
        outer_f1_scores.append(f1)
        
        # Predict probabilities for ROC AUC computation
        y_prob = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_prob)
        outer_roc_auc_scores.append(roc_auc)

        # CV Scores 
        best_params_score = (best_hyperparameters, f1)
        cv_scores.append(best_params_score)

        # Compute feature importances
        feature_importances = model.feature_importances_
        all_feature_importances.append(feature_importances)

        fold += 1

    # Compute average F1 score, average ROC AUC, and average feature importances
    output_dict = {}
    output_dict['f1'] = avg_f1 = np.mean(outer_f1_scores)
    output_dict['auc'] = output_dict['roc_auc'] = avg_roc_auc = np.mean(outer_roc_auc_scores)
    output_dict['feature_importance'] = avg_feature_importances = np.mean(all_feature_importances, axis=0)
    # output_dict['search'] = search # save a copy of the hyperparameter search result

    if len(feature_names) > 0: 
        assert len(feature_names) == X.shape[1]
        print(f"[test] Example feature names:\n{feature_names[:10]}\n")

        # Displaying the feature importances
        output_dict['feature_importance_df'] = \
            pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_feature_importances
        })    

    # Select the "best" hyperparameter settings from CV iterations
    print("[model] Choosing the best hyperparameter setting ...")
    # A. Choose the most frequently selected hyperparameters
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_params = dict(most_common_params)
    print(f"... most common hyperparam:\n{most_common_params}\n")

    # B. Choose the hyperparam setting that led to highest performance scores
    cv_scores_sorted = sorted(cv_scores, key=lambda x: x[1], reverse=True)
    best_scoring_params = cv_scores_sorted[0][0]
    best_score = cv_scores_sorted[0][1]
    output_dict['best_scoring_hyperparams'] = best_scoring_params
    print(f"... highest scoring hyperparams (score={best_score}):\n{best_scoring_params}\n")
    for i in range(3): 
        p, s = cv_scores_sorted[i][0], cv_scores_sorted[i][1]
        print(f"...... rank #{i+1}: score={s}, setting: {p}")

    # How to choose the best? 
    output_dict['best_hyperparams'] = best_hyperparams = output_dict['best_scoring_hyperparams']

    # Also return the model with the best parameters (by mode)
    # new_xgb = xgb.dask.DaskXGBClassifier(eval_metric="logloss", random_state=0)  # objective ='reg:squarederror'
    new_xgb = xgb.XGBClassifier(random_state=0, eval_metric=scoring) # logloss, aucpr

    output_dict['model'] = new_xgb.set_params(**best_hyperparams)

    # return avg_f1, avg_roc_auc, avg_feature_importances
    return output_dict

# Todo
def nested_cv_xgboost_with_importance_via_hyperband(X, y, **kargs):
    """
    
    Todo
    Since XGBoost does not implement partial_fit as it is, this version does not yet work. 

    """
    import xgboost as xgb
    # from xgboost import XGBClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import cross_val_score, KFold,  GridSearchCV

    from dask_ml.model_selection import HyperbandSearchCV
    import dask.array as da

    # from dask.distributed import Client
    # import joblib
    # from joblib import Parallel, delayed
    from distributed import Client, LocalCluster

    cluster = LocalCluster()
    client = Client(cluster) # 

    feature_names = kargs.get("feature_names", [])

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        if len(feature_names) == 0: feature_names = list(X.columns)
        X = X.values
        # NOTE: Why converting X to a numpy array? 
        #       XGBoost imposes restrictions on these feature names: they must not contain 
        #       certain characters like [, ], or <.

    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values
    unique_classes = da.unique(y)

    # Define the XGBoost classifier
    # xgb_clf = XGBClassifier(random_state=0, eval_metric="logloss")
    xgb_clf = xgb.dask.DaskXGBClassifier(random_state=0, eval_metric="logloss") 
    
    # NOTE: use_label_encoder=False => UserWarning: `use_label_encoder` is deprecated in 1.7.0.

    # Define the hyperparameter grid
    param_grid_xgb = {
        'n_estimators': [25, 50, 100, 200, 300, 400], # [100, 200, 300],
        'max_depth': [2, 3, 4, 5, 6, 7, 8], # [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2, 0.3], # [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0], # [0.8, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0], # [0.8, 1.0]  
    }
    param_grid = kargs.get('param_grid', param_grid_xgb)
    # NOTE: 
    #   `colsample_bytree` is the subsample ratio of columns when constructing each tree.
    # 6 * 6 * 4 * 3 * 3

    # Nested CV can be time consuming and we don't necessarily want to use it everytime 
    default_hyperparameters = kargs.get("default_hyperparams", None)
    use_nested_cv = kargs.get('use_nested_cv', True)
    if default_hyperparameters is None or not default_hyperparameters:
        # Default hyperparameters
        default_hyperparameters = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 0,
            'eval_metric': "logloss"
        }
        # NOTE: 'subsample': 
        #          Subsample ratio of the training instances. 
        #          Setting it to 0.5 means that XGBoost randomly collects half of the data instances to grow trees
        #      'colsample_bytree': 
        #         The fraction of features that can be selected for any given tree to train
    if not use_nested_cv: 
        if kargs.get('default_hyperparams', None): 
            print(f"[model] XGBoost: using known hyperparameters:\n{default_hyperparameters}\n")
        else: 
            print(f"[model] XGBoost: using default hyperparameters:\n{default_hyperparameters}\n")
    
    # Setup the inner and outer cross-validation
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    n_folds_inner = kargs.get("n_folds_inner", n_folds_outer)
    inner_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=n_folds_outer, shuffle=True, random_state=0)

    outer_f1_scores = []
    outer_roc_auc_scores = []
    all_feature_importances = []
    best_params_list = []

    # Outer cross-validation loop
    search = None
    
    fold = 0
    test_case = np.random.choice(range(n_folds_outer), 1)[0]
    cv_scores = []
    for train_idx, val_idx in outer_cv.split(X, y):
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # A rule-of-thumb to determine HyperbandSearchCV's input parameters requires knowing:
        # - the number of examples the longest trained model will see
        # - the number of hyperparameters to evaluate
        # 
        # We need to define 2 parameters
        #   - max_iter, which determines how many times to call partial_fit
        #   - the chunk size of the Dask array, which determines how many data each partial_fit call receives.

        n_params = 20  # sample about n parameters
        n_examples = 20 * len(X_train)  # 50 passes through dataset for best model

        # Inputs to hyperband
        max_iter = n_params
        chunk_size = n_examples // n_params

        # Create a Dask array with given chunk size
        # X_train2 = da.from_array(X_train, chunks=chunk_size)
        # y_train2 = da.from_array(y_train, chunks=chunk_size)

        if use_nested_cv:
            # Grid search with cross-validation in the inner loop
            # search = GridSearchCV(xgb_clf, param_grid, scoring='f1_macro', cv=inner_cv, n_jobs=-1)
            # NOTE: scoring 
            #       f1_macro, f1_micro, f1_weighted
            
            search = HyperbandSearchCV(
                                xgb_clf,  # NOTE: Doesn't work at the moment since xgb_clf does not implement partial_fit()
                                param_grid,
                                max_iter=max_iter,
                                scoring='f1_macro', 
                                patience=True, 
                                aggressiveness=3, 
                                random_state=0 
                            )
            # NOTE: A patience value is automatically selected if patience=True to work well with 
            #       the Hyperband model selection algorithm.

            # with joblib.parallel_backend('dask'):
            search.fit(X_train, y_train, classes=unique_classes) # Use chunked training set
            best_hyperparameters = search.best_params_
        else: 
            # Use default hyperparameters
            best_hyperparameters = default_hyperparameters

        # Test 
        if fold == test_case: 
            print(f"[model] Fold={fold}: XGBoost, best params={best_hyperparameters}, via nested cv? {use_nested_cv}")
            if use_nested_cv: 
                print(f"... best score: {search.best_score_}")

                cv_results = pd.DataFrame(search.cv_results_)
                print(f"... CV results")
                print(cv_results.head())

                # hist = pd.DataFrame(search.history_)
                # print(f"... history:")
                # print(hist.head())
                # NOTE: 'BayesSearchCV' object has no attribute 'history_'

        best_params_list.append(tuple(best_hyperparameters.items()))
        
        # Train the model using the best hyperparameters found
        best_xgb = xgb.dask.DaskXGBClassifier(random_state=0, eval_metric="logloss") 
        # best_xgb = XGBClassifier(n_estimators = best_hyperparameters['n_estimators'],
        #                             max_depth = best_hyperparameters['max_depth'],
        #                             learning_rate = best_hyperparameters['learning_rate'],
        #                             subsample = best_hyperparameters['subsample'],
        #                             colsample_bytree = best_hyperparameters['colsample_bytree'],
        #                             random_state=0, eval_metric="logloss") # use_label_encoder=False,
        best_xgb.set_params(**best_hyperparameters)
        best_xgb.fit(X_train, y_train)
        
        # Predict and compute F1 score on the validation set
        y_pred = best_xgb.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')  #
        outer_f1_scores.append(f1)
        
        # Predict probabilities for ROC AUC computation
        y_prob = best_xgb.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_prob)
        outer_roc_auc_scores.append(roc_auc)

        # CV Scores 
        cv_scores.append(f1)

        # Compute feature importances
        feature_importances = best_xgb.feature_importances_
        all_feature_importances.append(feature_importances)

        fold += 1

    # HyperbandSearchCV details on the amount of training and the number of models created. 
    # These details are available in the metadata attribute
    print(f"[hyperband] Number of models created: {search.metadata['n_models']}")
    print(f"... n(partial_fit calls): {search.metadata['partial_fit_calls']}")

    # Compute average F1 score, average ROC AUC, and average feature importances
    output_dict = {}
    output_dict['f1'] = avg_f1 = np.mean(outer_f1_scores)
    output_dict['auc'] = output_dict['roc_auc'] = avg_roc_auc = np.mean(outer_roc_auc_scores)
    output_dict['feature_importance'] = avg_feature_importances = np.mean(all_feature_importances, axis=0)

    if len(feature_names) > 0: 
        assert len(feature_names) == X.shape[1]
        print(f"[test] Example feature names:\n{feature_names[:10]}\n")

        # Displaying the feature importances
        output_dict['feature_importance_df'] = \
            pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_feature_importances
        })    

    # Determine the most frequently selected hyperparameters
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_params = dict(most_common_params)

    # Also return the model with the best parameters (by mode)
    new_xgb = xgb.dask.DaskXGBClassifier(eval_metric="logloss", random_state=0)  # objective ='reg:squarederror'
    # output_dict['model'] = XGBClassifier(n_estimators=most_common_params['n_estimators'],
    #                                 max_depth=most_common_params['max_depth'],
    #                                 learning_rate=most_common_params['learning_rate'],
    #                                 subsample=most_common_params['subsample'],
    #                                 colsample_bytree=most_common_params['colsample_bytree'],
    #                                 random_state=0, eval_metric="logloss")
    output_dict['model'] = new_xgb.set_params(**most_common_params)

    # return avg_f1, avg_roc_auc, avg_feature_importances
    return output_dict

# Reporting util for different optimizers
def optimize_and_report(optimizer, X, y, title="model", callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers
    
    optimizer = a sklearn or a skopt optimizer
    X = the training set 
    y = our target
    title = a string label for the experiment
    """
    from time import time 
    start = time()
    
    if callbacks is not None:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
        
    d=pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
           + u"\u00B1"+" %.3f") % (time() - start, 
                                   len(optimizer.cv_results_['params']),
                                   best_score,
                                   best_score_std))    
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    return best_params

def nested_cv_xgboost_with_importance_via_bayes_opt(X, y, **kargs):
    """
    
    Memo
    ----
    1. https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html
    """
    # import xgboost as xgb
    from xgboost import XGBClassifier

    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    from skopt.callbacks import DeadlineStopper, DeltaYStopper
    from skopt.plots import plot_objective, plot_histogram
    # NOTE: pip install scikit-optimize

    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import cross_val_score, KFold,  GridSearchCV

    # from dask_ml.model_selection import HyperbandSearchCV
    # import dask.array as da

    # from distributed import Client
    # client = Client(processes=False, threads_per_worker=2,
    #                 n_workers=5, memory_limit='20GB')

    feature_names = kargs.get("feature_names", [])

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        if len(feature_names) == 0: feature_names = list(X.columns)
        X = X.values
        # NOTE: Why converting X to a numpy array? 
        #       XGBoost imposes restrictions on these feature names: they must not contain 
        #       certain characters like [, ], or <.

    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values
    classes = np.unique(y)
    n_classes = len(classes)

    # objective = "binary:logistic"
    # if n_classes > 2: 
    #     objective = "multi:softmax"

    # Define the XGBoost classifier
    xgb_clf = XGBClassifier(random_state=0, eval_metric="logloss") # objective="binary:logistic"

    # pipeline class is used as estimator to enable
    # search over different model types
    # pipe = Pipeline([
    #     ('model', XGBClassifier(random_state=0, eval_metric="logloss"))
    # ])
        
    # NOTE: use_label_encoder=False => UserWarning: `use_label_encoder` is deprecated in 1.7.0.

    # Define the hyperparameter grid
    param_grid_xgb = {
                'learning_rate': Real(0.01, 1.0, 'uniform'),
                'max_depth': Integer(2, 12),
                'subsample': Real(0.1, 1.0, 'uniform'),
                'colsample_bytree': Real(0.1, 1.0, 'uniform'), # subsample ratio of columns by tree
                'reg_lambda': Real(1e-9, 100., 'uniform'), # L2 regularization
                'reg_alpha': Real(1e-9, 100., 'uniform'), # L1 regularization
                'n_estimators': Integer(25, 500)
    }
    search_space = param_grid = kargs.get('search_space', param_grid_xgb)
    # NOTE: 
    #   `colsample_bytree` is the subsample ratio of columns when constructing each tree.
    # 6 * 6 * 4 * 3 * 3
    scoring = kargs.get('scoring', 'f1_macro')

    # Nested CV can be time consuming and we don't necessarily want to use it everytime 
    default_hyperparameters = kargs.get("default_hyperparams", None)
    use_nested_cv = kargs.get('use_nested_cv', True)
    if default_hyperparameters is None or not default_hyperparameters:
        # Default hyperparameters
        default_hyperparameters = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 0,
            'eval_metric': "logloss"
        }
        # NOTE: 'subsample': 
        #          Subsample ratio of the training instances. 
        #          Setting it to 0.5 means that XGBoost randomly collects half of the data instances to grow trees
        #      'colsample_bytree': 
        #         The fraction of features that can be selected for any given tree to train
    if not use_nested_cv: 
        if kargs.get('default_hyperparams', None): 
            print(f"[model] XGBoost: using known hyperparameters:\n{default_hyperparameters}\n")
        else: 
            print(f"[model] XGBoost: using default hyperparameters:\n{default_hyperparameters}\n")
    
    # Setup the inner and outer cross-validation
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    n_folds_inner = kargs.get("n_folds_inner", n_folds_outer)
    inner_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=n_folds_outer, shuffle=True, random_state=0)

    outer_f1_scores = []
    outer_roc_auc_scores = []
    all_feature_importances = []
    best_params_list = []

    # Outer cross-validation loop
    search = None
    max_iter = 120

    # Running the optimizer
    overdone_control = DeltaYStopper(delta=0.0001)                    # We stop if the gain of the optimization becomes too small
    time_limit_control = DeadlineStopper(total_time=60*60*5)          # We impose a time limit (5 hours)
    
    fold = 0
    test_case = np.random.choice(range(n_folds_outer), 1)[0]
    cv_scores = []
    for train_idx, val_idx in tqdm(outer_cv.split(X, y), total=n_folds_outer):
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if use_nested_cv:
            # Grid search with cross-validation in the inner loop
            # search = GridSearchCV(xgb_clf, param_grid, scoring='f1_macro', cv=inner_cv, n_jobs=-1)
            # NOTE: scoring 
            #       f1_macro, f1_micro, f1_weighted

            # Wrapping everything up into the Bayesian optimizer
            search = \
                BayesSearchCV(estimator=xgb_clf,                                    
                                search_space=search_space,                      
                                scoring=scoring,                                  
                                cv=inner_cv,                                           
                                n_iter=max_iter,                              # max number of trials
                                n_points=10,                             # number of hyperparameter sets evaluated at the same time
                                n_jobs=5,                                # number of jobs
                                # iid=False,                             # if not iid it optimizes on the cv score (deprecated)
                                return_train_score=False,                         
                                refit=False,                                      
                                optimizer_kwargs={'base_estimator': 'GP'},    # optmizer parameters: we use Gaussian Process (GP)
                                random_state=0)                               # random state for replicability
            # NOTE: 
            #  - iid: If True, the data is assumed to be identically distributed across the folds, 
            #         and the loss minimized is the total loss per sample, and not the mean loss across the folds 

            # with joblib.parallel_backend('dask'):
            # search.fit(X_train, y_train, classes=unique_classes) # Use chunked training set
            # best_hyperparameters = search.best_params_
            best_hyperparameters = \
                optimize_and_report(
                    search, X_train, y_train, 
                        title='XGBoostClassifier', 
                          callbacks=[overdone_control, time_limit_control])
        else: 
            # Use default hyperparameters
            best_hyperparameters = default_hyperparameters

        # Test 
        if fold == test_case: 
            print(f"[model] XGBoost: Fold={fold}")
            if use_nested_cv: 
                print(f"... best hyperparams:\n{best_hyperparameters}\n")
                print(f"... best score: {search.best_score_}")
                print()

                cv_results = pd.DataFrame(search.cv_results_)
                print(f"... CV results")
                print(cv_results.head()); print()

                # hist = pd.DataFrame(search.history_)
                # print(f"... history:")
                # print(hist.head())
                # NOTE: 'BayesSearchCV' object has no attribute 'history_'

        best_params_list.append(tuple(best_hyperparameters.items()))
        
        # Train the model using the best hyperparameters found
        best_xgb = XGBClassifier(random_state=0, eval_metric="logloss")
        # best_xgb = XGBClassifier(n_estimators = best_hyperparameters['n_estimators'],
        #                             max_depth = best_hyperparameters['max_depth'],
        #                             learning_rate = best_hyperparameters['learning_rate'],
        #                             subsample = best_hyperparameters['subsample'],
        #                             colsample_bytree = best_hyperparameters['colsample_bytree'],
        #                             random_state=0, eval_metric="logloss") # use_label_encoder=False,
        best_xgb.set_params(**best_hyperparameters)
        best_xgb.fit(X_train, y_train)
        
        # Predict and compute F1 score on the validation set
        y_pred = best_xgb.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')  #
        outer_f1_scores.append(f1)
        
        # Predict probabilities for ROC AUC computation
        y_prob = best_xgb.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_prob)
        outer_roc_auc_scores.append(roc_auc)

        # CV Scores 
        best_params_score = (best_hyperparameters, f1)
        cv_scores.append(best_params_score)

        # Compute feature importances
        feature_importances = best_xgb.feature_importances_
        all_feature_importances.append(feature_importances)

        fold += 1

    # Compute average F1 score, average ROC AUC, and average feature importances
    output_dict = {}
    output_dict['f1'] = avg_f1 = np.mean(outer_f1_scores)
    output_dict['auc'] = output_dict['roc_auc'] = avg_roc_auc = np.mean(outer_roc_auc_scores)
    output_dict['feature_importance'] = avg_feature_importances = np.mean(all_feature_importances, axis=0)
    # output_dict['search'] = search # save a copy of the hyperparameter search result

    if len(feature_names) > 0: 
        assert len(feature_names) == X.shape[1]
        print(f"[test] Example feature names:\n{feature_names[:10]}\n")

        # Displaying the feature importances
        output_dict['feature_importance_df'] = \
            pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_feature_importances
        })    

    # Select the "best" hyperparameter settings from CV iterations
    print("[model] Choosing the best hyperparameter setting ...")
    # A. Choose the most frequently selected hyperparameters
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_params = dict(most_common_params)
    print(f"... most common hyperparam:\n{most_common_params}\n")

    # B. Choose the hyperparam setting that led to highest performance scores
    cv_scores_sorted = sorted(cv_scores, key=lambda x: x[1], reverse=True)
    best_scoring_params = cv_scores_sorted[0][0]
    best_score = cv_scores_sorted[0][1]
    output_dict['best_scoring_hyperparams'] = best_scoring_params
    print(f"... highest scoring hyperparam (score={best_score}):\n{best_scoring_params}\n")
    for i in range(3): 
        p, s = cv_scores_sorted[i][0], cv_scores_sorted[i][1]
        print(f"...... rank #{i+1}: score={s}, setting: {p}")

    # How to choose the best? 
    output_dict['best_hyperparams'] = best_hyperparams = output_dict['best_scoring_hyperparams']

    # Also return the model with the best parameters (by mode)
    # new_xgb = xgb.dask.DaskXGBClassifier(eval_metric="logloss", random_state=0)  # objective ='reg:squarederror'
    new_xgb = XGBClassifier(random_state=0, eval_metric="logloss")

    output_dict['model'] = new_xgb.set_params(**best_hyperparams)

    # return avg_f1, avg_roc_auc, avg_feature_importances
    return output_dict


# SVM
######## 

def nested_cv_linear_svm_with_importance(X, y, **kargs):
    # Output: Performance evaluation (e.g. F1 score, ROCAUC)
    #         feature importance scores 

    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        X = X.values
    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values
    
    # Define the SVM classifier
    svm = SVC(kernel='linear', probability=True, random_state=0)

    # Define the hyperparameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    # Nested CV can be time consuming and we don't necessarily want to use it everytime 
    default_hyperparameters = kargs.get("default_hyperparams", None)
    use_nested_cv = kargs.get('use_nested_cv', True)
    if default_hyperparameters is None: 
        default_hyperparameters = {
            'C': 1
        }
    if not use_nested_cv and kargs.get('default_hyperparams', None): 
        print(f"[model] Linear SVM: using known hyperparameters:\n{default_hyperparameters}\n")

    # Setup the inner and outer cross-validation
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    n_folds_inner = kargs.get("n_folds_inner", n_folds_outer)
    inner_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)

    outer_f1_scores = []
    outer_roc_auc_scores = []
    all_feature_importances = []
    best_params_list = []

    # Outer cross-validation loop
    fold = 1
    test_case = np.random.choice(range(n_folds_outer), 1)[0]
    for train_idx, val_idx in outer_cv.split(X, y):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # X_train = X.iloc[train_idx] if is_dataframe else X[train_idx]
        # X_val = X.iloc[val_idx] if is_dataframe else X[val_idx]
        # # No need for special indexing for y as it's either an array or series now
        # y_train = y[train_idx]
        # y_val = y[val_idx]

        # Standardize the features
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        if use_nested_cv: 

            # Grid search with cross-validation in the inner loop
            grid_search = GridSearchCV(svm, param_grid, scoring='f1_macro', cv=inner_cv, n_jobs=-1)
            # f1_macro

            grid_search.fit(X_train_scaled, y_train)
            best_hyperparameters = grid_search.best_params_
        else: 
            best_hyperparameters = default_hyperparameters

        # Test 
        if fold == test_case: 
            print(f"[model] Fold={test_case}: SVM with linear kernel, best params={best_hyperparameters}")
        best_params_list.append(tuple(best_hyperparameters.items()))
        
        # Train the model using the best hyperparameters found
        best_svm = SVC(kernel='linear', C=best_hyperparameters['C'], 
                       probability=True, random_state=0)
        best_svm.fit(X_train_scaled, y_train)
        
        # Predict and compute F1 score on the validation set
        y_pred = best_svm.predict(X_val_scaled)
        f1 = f1_score(y_val, y_pred, average='macro')
        outer_f1_scores.append(f1)
        
        # Predict probabilities for ROC AUC computation
        y_prob = best_svm.decision_function(X_val_scaled)
        roc_auc = roc_auc_score(y_val, y_prob)
        outer_roc_auc_scores.append(roc_auc)

        # Store feature importances
        all_feature_importances.append(np.abs(best_svm.coef_[0]))

        fold += 1

    # Compute average F1 score, average ROC AUC, and average feature importances
    output_dict = {}
    output_dict['f1'] = avg_f1 = np.mean(outer_f1_scores)
    output_dict['auc'] = output_dict['roc_auc'] = avg_roc_auc = np.mean(outer_roc_auc_scores)
    output_dict['feature_importance'] = avg_feature_importances = np.mean(all_feature_importances, axis=0)

    feature_names = kargs.get("feature_names", [])
    if len(feature_names) > 0: 
        assert len(feature_names) == X.shape[1]

        # Displaying the feature importances
        output_dict['feature_importance_df'] = \
            pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_feature_importances
        })

    # Determine the most frequently selected hyperparameters
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_params = dict(most_common_params)

    # Also return the model with the best parameters (by mode)
    output_dict['model'] = SVC(kernel='linear', C=most_common_params['C'], 
                               probability=True, random_state=0)
    
    return output_dict


def nested_cv_linear_svm_v0(X, y, **kargs):
    # import pandas as pd
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X = pd.DataFrame(X)
    
    # Define the classifier
    # Use a pipeline to standardize the data and then apply LinearSVC
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, dual=False, max_iter=10000, probability=True))
    
    # Parameters grid for hyperparameter tuning
    param_grid = {
        'linearsvc__C': [0.001, 0.01, 0.1, 1, 10, 100]
        # 'C': [0.001, 0.01, 0.1, 1, 10, 100],
    }
    
    # Inner CV for hyperparameter tuning
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)
    grid_search = GridSearchCV(clf, param_grid, scoring='f1_macro', cv=inner_cv, n_jobs=-1)

    # Outer CV for model evaluation
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=0)
    
    f1_scores = []
    feature_importances_list = []
    
    fold_num = 1
    for train_idx, valid_idx in outer_cv.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        # Find best hyperparameters using inner CV
        grid_search.fit(X_train, y_train)
        
        # Train the model with the best hyperparameters on the training data
        best_clf = grid_search.best_estimator_
        best_clf.fit(X_train, y_train)
        
        # Predict and calculate F1 score on the validation data
        y_pred = best_clf.predict(X_valid)
        f1 = classification_report(y_valid, y_pred, output_dict=True)['macro avg']['f1-score']
        
        f1_scores.append(f1)
        
        # Extract feature importances based on the coefficients of the hyperplane
        feature_importances = best_clf.named_steps['linearsvc'].coef_[0]
        feature_importances_list.append(feature_importances)
        
        # print(f"[{fold_num}] {feature_importances}")
        fold_num += 1
    
    # Average F1 score and feature importances
    avg_f1_score = np.mean(f1_scores)
    avg_feature_importances = np.mean(feature_importances_list, axis=0)
    
    return avg_f1_score, avg_feature_importances

def svm_with_nonlinear_kernel_shap_explainer(X, y, **kargs): 
    """

    Memo
    ----
    1.  Warning: 
        Using 353 background data samples could cause slower run times. 
        Consider using shap.sample(data, K) or shap.kmeans(data, K) to summarize the background as K samples."
    
        When you use the KernelExplainer, SHAP requires a set of "background" data samples 
        to compute the SHAP values. This background data is used to approximate the expected output of the model. 

        However, if the background dataset is large, computing SHAP values can become very 
        time-consuming because the KernelExplainer needs to perform computations for each 
        instance in the background dataset for each instance you want to explain.

        The warning provides two potential solutions:

        1. shap.sample(data, K): 
          This will randomly sample 
          K instances from your background data. It's a straightforward way to reduce the dataset size.

        2. shap.kmeans(data, K): This is a more sophisticated approach. Instead of randomly sampling, 
            it uses the k-means clustering algorithm to summarize your background data into K representative clusters. 
            The centroids of these clusters are then used as the background dataset. 
            This can often provide a more "representative" summary of your data than random sampling, 
            especially if your data has a lot of variability or distinct subgroups.
    2. SHAP's background dataset
       
        Let's say you want to know the contribution of a specific feature to the prediction. 
        You could "turn off" that feature by setting it to some default or neutral value and see 
        how the prediction changes. But what should that default or neutral value be? 
        This is where the background dataset comes into play.

        The background dataset provides a reference or baseline against which the contributions of 
        individual features can be measured. By averaging over this background dataset, 
        SHAP can estimate what the model would predict if it didn't know the value of a particular feature 
        for the instance being explained.
    
    """
    import shap
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    # from dask_ml.model_selection import train_test_split
    from sklearn.metrics import f1_score
    # import matplotlib.pyplot as plt

    feature_names = kargs.get("feature_names", list(range(X.shape[1])) )
    random_state = kargs.get("random_state", 0)
    test_size = kargs.get("test_size", 0.2)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train an SVM model with RBF kernel
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    svm_rbf = SVC(kernel='rbf', probability=True, random_state=random_state)
    svm_rbf.fit(X_train_scaled, y_train)

    # Predict on the validation set and compute F1 score
    y_pred = svm_rbf.predict(X_val_scaled)
    f1 = f1_score(y_val, y_pred, average='macro')

    # Initialize the JavaScript visualization code for SHAP
    shap.initjs()

    # ---------------------------------------------------------
    # A. Use SHAP KernelExplainer using full training set (computationally expensive)
    # explainer = shap.KernelExplainer(svm_rbf.predict_proba, X_train_scaled)

    # B. Summarize the training data with k-means clustering
    background_data = shap.kmeans(X_train_scaled, k=50)  # for example, use 50 clusters
    explainer = shap.KernelExplainer(svm_rbf.predict_proba, background_data)

    shap_values = explainer.shap_values(X_val_scaled)
    # NOTE: When interpreting the predictions made on the validation set (X_val_scaled), 
    #       it's common to use the training set (X_train_scaled) as the "background dataset" 
    # ---------------------------------------------------------

    # Compute the average feature importances for the positive class
    avg_feature_importances = np.mean(np.abs(shap_values[1]), axis=0)

    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot the SHAP values for the positive class
    shap.summary_plot(shap_values[1], X_val_scaled, feature_names=feature_names)

    save = kargs.get("save", True)
    verbose = kargs.get("verbose", 1)
 
    if save: 
        output_dir_default = os.path.join(os.getcwd(), "plot")
        output_dir = kargs.get("output_dir", output_dir_default)

        ext = kargs.get("ext", "pdf") # # Save the plot to a PDF file by default
        output_file = kargs.get("output_file", f"shap_summary_plot-k50-svm.{ext}")

        output_path = os.path.join(output_dir, output_file)
        if verbose: print(f"[SHAP] Saving SVM model's shap summary plot to:\n{output_path}\n")

        plt.tight_layout()
        plt.savefig(output_path, format='pdf')

    return f1, avg_feature_importances

def nested_cv_rbf_svm_with_importance(X, y, **kargs):
    return nested_cv_rbf_svm_with_shap(X, y, **kargs)
def nested_cv_rbf_svm_with_shap(X, y, **kargs):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    import shap

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        X = X.values
    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values
    
    # Define the SVM classifier with RBF kernel
    svm_rbf = SVC(kernel='rbf', probability=True, random_state=0)

    # Define the hyperparameter grid
    param_grid_rbf = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    }

    # Nested CV can be time consuming and we don't necessarily want to use it everytime 
    default_hyperparameters = kargs.get("default_hyperparams", None)
    use_nested_cv = kargs.get('use_nested_cv', True)
    if default_hyperparameters is None:
        # Default hyperparameters
        default_hyperparameters = {
            'C': 1,
            'gamma': 'scale',
            'kernel': 'rbf',
            # 'probability': True,
        }
    if not use_nested_cv: 
        if kargs.get('default_hyperparams', None): 
            print(f"[model] SVM+RBF: using known hyperparameters:\n{default_hyperparameters}\n")
        else: 
            print(f"[model] SVM+RBF: using default hyperparameters:\n{default_hyperparameters}\n")
    
    # Setup the inner and outer cross-validation
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    n_folds_inner = kargs.get("n_folds_inner", n_folds_outer)
    inner_cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=0)

    outer_f1_scores = []
    outer_roc_auc_scores = []
    all_feature_importances = []
    best_params_list = []

    # Outer cross-validation loop
    fold = 1
    test_case = np.random.choice(range(n_folds_outer), 1)[0]
    for train_idx, val_idx in outer_cv.split(X, y):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # X_train = X.iloc[train_idx] if is_dataframe else X[train_idx]
        # X_val = X.iloc[val_idx] if is_dataframe else X[val_idx]
        # # No need for special indexing for y as it's either an array or series now
        # y_train = y[train_idx]
        # y_val = y[val_idx]

        # Standardize the features
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        if use_nested_cv: 
            # Grid search with cross-validation in the inner loop
            grid_search = GridSearchCV(svm_rbf, param_grid_rbf, scoring='f1_macro', cv=inner_cv, n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            best_hyperparameters = grid_search.best_params_
        else:
            # Use default hyperparameters
            best_hyperparameters = default_hyperparameters

        # Test 
        if fold == test_case: 
            print(f"[model] Fold={fold}: SVM with RBF kernel, best params={best_hyperparameters}")
        best_params_list.append(tuple(best_hyperparameters.items()))
        
        # Train the model using the best hyperparameters found
        best_svm_rbf = SVC(kernel='rbf', C=best_hyperparameters['C'], gamma=best_hyperparameters['gamma'], 
                           probability=True, random_state=0)
        best_svm_rbf.fit(X_train_scaled, y_train)
        
        # Predict and compute F1 score on the validation set
        y_pred = best_svm_rbf.predict(X_val_scaled)
        f1 = f1_score(y_val, y_pred, average='macro')
        outer_f1_scores.append(f1)
        
        # Predict probabilities for ROC AUC computation
        y_prob = best_svm_rbf.decision_function(X_val_scaled)
        roc_auc = roc_auc_score(y_val, y_prob)
        outer_roc_auc_scores.append(roc_auc)

        # Compute SHAP values for feature importances
        explainer = shap.KernelExplainer(best_svm_rbf.predict_proba, shap.kmeans(X_train_scaled, 50))
        shap_values = explainer.shap_values(X_val_scaled)
        feature_importances = np.mean(np.abs(shap_values[1]), axis=0)
        all_feature_importances.append(feature_importances)

        fold += 1

    output_dict = {}

    # Compute average F1 score, average ROC AUC, and average feature importances
    output_dict['f1'] = avg_f1 = np.mean(outer_f1_scores)
    output_dict['auc'] = avg_roc_auc = np.mean(outer_roc_auc_scores)
    output_dict['feature_importance'] = avg_feature_importances = np.mean(all_feature_importances, axis=0)

    feature_names = kargs.get("feature_names", [])
    if len(feature_names) > 0: 
        assert len(feature_names) == X.shape[1]

        # Displaying the feature importances
        output_dict['feature_importance_df'] = \
            pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_feature_importances
        })

    # Determine the most frequently selected hyperparameters
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_params = dict(most_common_params)

    # Also return the model with the best parameters (by mode)
    output_dict['model'] = SVC(kernel='rbf', C=most_common_params['C'], gamma=most_common_params['gamma'], 
                           probability=True, random_state=0)
    
    # return avg_f1, avg_roc_auc, avg_feature_importances
    return output_dict

def nested_cv_linear_svm_with_importance_via_bohb(X, y, **kargs):
    kargs['loss_fn'] = ['hinge', ]
    return nested_cv_sgd_with_importance_via_bohb(X, y, **kargs) 

def nested_cv_logistic_with_importance_via_bohb(X, y, **kargs):
    kargs['loss_fn'] = ['log_loss', ]
    return nested_cv_sgd_with_importance_via_bohb(X, y, **kargs)

def nested_cv_sgd_with_importance_via_bohb(X, y, **kargs):
    # from skopt.space import Real, Categorical, Integer
    from scipy.stats import uniform, loguniform

    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler

    from sklearn.linear_model import SGDClassifier
    from sklearn.calibration import CalibratedClassifierCV
    
    from hpbandster_sklearn import HpBandSterSearchCV
    from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
    import shap

    import joblib
    # from sklearn.externals.joblib import parallel_backend
    # from sklearn.utils import parallel_backend

    feature_names = kargs.get("feature_names", [])
    loss_functions = kargs.get("loss_fn", ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'])

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        if len(feature_names) == 0: feature_names = list(X.columns)
        X = X.values
        # NOTE: Why converting X to a numpy array? 
        #       XGBoost imposes restrictions on these feature names: they must not contain 
        #       certain characters like [, ], or <.

    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values
    unique_classes = np.unique(y)

    # Define SGD classifier
    model = SGDClassifier(tol=1e-3, eta0=0.001)
    
    # NOTE: use_label_encoder=False => UserWarning: `use_label_encoder` is deprecated in 1.7.0.

    # Define the hyperparameter grid
    sgd_search_space = {
        'alpha': Float("alpha", bounds=(1e-6, 1e+2), log=True),    # loguniform(1e-6, 1e+2),  # The higher the value, the stronger the regularization.
        'loss': Categorical("loss", loss_functions),  # ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'], # 'perceptron'
        'penalty': Categorical("penalty", ['l2', 'l1', 'elasticnet', ]),
        'l1_ratio': Float("l1_ratio", bounds=(0, 1)), # uniform(0, 1), # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], # uniform(0.1, 1.0), 
        'fit_intercept': Categorical("fit_intercept", [True, False]), 
        'learning_rate': Categorical("learning_rate", ['constant', 'optimal', 'invscaling', 'adaptive', ]), # 
        'power_t': Float("power_t", bounds=(0.5, 0.99)),  # uniform( 0.5, 0.99 ), 
        'average': Categorical("average", [True, False]), 
        # 'class_weight': Categorical("class_weight", [None, 'balanced', ]),
        # NOTE:  class_weight 'balanced' is not supported for partial_fit. 
        # In order to use 'balanced' weights, use compute_class_weight('balanced', classes=classes, y=y). 
        # In place of y you can use a large enough sample of the full training set target to properly estimate the class frequency distributions.
    }
    search_space = kargs.get('search_space', sgd_search_space)
    scoring = kargs.get('scoring', 'f1_macro') # 'roc_auc'

    # Nested CV can be time consuming and we don't necessarily want to use it everytime 
    default_hyperparameters = kargs.get("default_hyperparams", None)
    use_nested_cv = kargs.get('use_nested_cv', True)
    if default_hyperparameters is None or not default_hyperparameters:
        # Default hyperparameters
        default_hyperparameters = {
            'alpha': 0.0001,
            'loss': 'log_loss',
            'penalty': 'elasticnet',
            'l1_ratio': 0.5,
            'fit_intercept': True, 
            'learning_rate': 'adpative',
            'average': False, 
        }
        # NOTE: 'subsample': 
        #          Subsample ratio of the training instances. 
        #          Setting it to 0.5 means that XGBoost randomly collects half of the data instances to grow trees
        #      'colsample_bytree': 
        #         The fraction of features that can be selected for any given tree to train
    if not use_nested_cv: 
        if kargs.get('default_hyperparams', None): 
            print(f"[model] SGD: using known hyperparameters:\n{default_hyperparameters}\n")
        else: 
            print(f"[model] SGD: using default hyperparameters:\n{default_hyperparameters}\n")
    
    # Setup the inner and outer cross-validation
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    # n_folds_inner = kargs.get("n_folds_inner", n_folds_outer)
    # inner_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=n_folds_outer, shuffle=True, random_state=0)

    outer_f1_scores = []
    outer_roc_auc_scores = []
    all_feature_importances = []
    best_params_list = []

    # Outer cross-validation loop
    search = None
    n_params = 100  # sample about n parameters
    
    fold = 0
    test_case = np.random.choice(range(n_folds_outer), 1)[0]
    cv_scores = []
    for train_idx, val_idx in outer_cv.split(X, y):
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        if use_nested_cv:            
            param_distributions = ConfigurationSpace(
                    name="nmd_opt_sgd",
                    seed=42, 
                    space=search_space,
            )
            search = HpBandSterSearchCV(model, 
                        param_distributions, 
                        scoring=scoring,
                        random_state=0, 
                            warm_start=True,
                            refit=True, # If True, refit an estimator using the best found parameters on the whole dataset
                            resource_name='n_samples', # can be either 'n_samples' or a string corresponding to an estimator attribute, eg. 'n_estimators' for an ensemble
                            resource_type=float, # if specified, the resource value will be cast to that type before being passed to the estimator, otherwise it will be derived automatically
                            min_budget=0.8,
                            max_budget=1,
                                n_jobs=4, 
                                verbose=1).fit(X_train, y_train) # n_iter=10,      

            best_hyperparameters = search.best_params_
        else: 
            # Use default hyperparameters
            best_hyperparameters = default_hyperparameters

        # Test 
        if fold == test_case: 
            print(f"[model] Fold={fold}: SGD, best params={best_hyperparameters}, via nested cv? {use_nested_cv}")
            if use_nested_cv: 
                print(f"... best hyperparams:\n{best_hyperparameters}\n")
                print(f"... best score: {search.best_score_}")

                cv_results = pd.DataFrame(search.cv_results_)
                # cv_results.sort_values(by='test_score', ascending=False, inplace=True)
                print(f"... CV results")
                print(cv_results.head())

                # hist = pd.DataFrame(search.history_)
                # hist.sort_values(by='score', ascending=False, inplace=True)
                # print(f"... history:")
                # print(hist.head())

        best_params_list.append(tuple(best_hyperparameters.items()))

        best_model = None
        if use_nested_cv: 

            # Predict and compute F1 score on the validation set
            y_pred = search.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='macro')  #
            outer_f1_scores.append(f1)
        
            # Get probability predictions
            # ---------------------------
            if best_hyperparameters['loss'] in ['log_loss', 'modified_huber', ]: 
                # Predict probabilities for ROC AUC computation
                y_prob = search.predict_proba(X_val)[:, 1]
            else: 
                # Instantiate a new model with the best hyperparameters and use probability calibration to get 
                # probability predictions
                model = SGDClassifier(tol=1e-3, eta0=0.001)
                model.set_params(**best_hyperparameters)
                print(f"[info] loss fn: {best_hyperparameters['loss']} does not support predict_proba(), use Platt scaling")
                model_calibrated = CalibratedClassifierCV(model, method='sigmoid')
                # NOTE: 'cv' argument is set to None by default => use the default 5-fold cross-validation to 
                #       estimate probabilities

                model_calibrated.fit(X_train, y_train)
                y_prob = model_calibrated.predict_proba(X_val)[:, 1]
            # -------------------
            # Now, we have y_prob

            roc_auc = roc_auc_score(y_val, y_prob)
            outer_roc_auc_scores.append(roc_auc)
                    
            best_model = search.best_estimator_

        else: 
            best_model = SGDClassifier(tol=1e-3, eta0=0.001)
            best_model.set_params(**best_hyperparameters) # user-provided or default

            # Refit the model
            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_val)

            # F1 score 
            f1 = f1_score(y_val, y_pred, average='macro')  #
            outer_f1_scores.append(f1)

            # Probability prediction
            if best_hyperparameters['loss'] in ['log_loss', 'modified_huber', ]: 
                y_prob = best_model.predict_proba(X_val)[:, 1]
            else: 
                # Instantiate a new model with the best hyperparameters and use probability calibration to get 
                # probability predictions
                model = SGDClassifier(tol=1e-3, eta0=0.001)
                model.set_params(**best_hyperparameters) # user-provided or default
                prob_model = CalibratedClassifierCV(model, method='sigmoid')
                prob_model.fit(X_train, y_train)
                y_prob = prob_model.predict_proba(X_val)[:, 1]
            
            # AUC 
            roc_auc = roc_auc_score(y_val, y_prob)
            outer_roc_auc_scores.append(roc_auc)

        # CV Scores 
        best_params_score = (best_hyperparameters, f1)
        cv_scores.append(best_params_score)

        # Compute SHAP values for feature importances
        # explainer = shap.KernelExplainer(best_model.predict_proba, shap.kmeans(X_train, 50))
        # shap_values = explainer.shap_values(X_val)
        # feature_importances = np.mean(np.abs(shap_values[1]), axis=0)
        # all_feature_importances.append(feature_importances)

        # Compute feature importances (coefficients for logistic regression)

        feature_importances = best_model.coef_[0]
        all_feature_importances.append(feature_importances)

        fold += 1
    ### End CV iterations

    # Compute average F1 score, average ROC AUC, and average feature importances
    output_dict = {}
    output_dict['f1'] = avg_f1 = np.mean(outer_f1_scores)
    output_dict['auc'] = output_dict['roc_auc'] = avg_roc_auc = np.mean(outer_roc_auc_scores)
    output_dict['feature_importance'] = avg_feature_importances = np.mean(all_feature_importances, axis=0)

    if len(feature_names) > 0: 
        assert len(feature_names) == X.shape[1]
        print(f"[test] Example feature names:\n{feature_names[:10]}\n")

        # Displaying the feature importances
        output_dict['feature_importance_df'] = \
            pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_feature_importances
        })    

    # Determine the best hyperparameters
    print("[model] Choosing the best hyperparameter setting ...")
    # A. Choose the most frequently selected hyperparameters
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_params = dict(most_common_params)
    print(f"... most common hyperparam:\n{most_common_params}\n")

    # B. Choose the hyperparam setting that led to highest performance scores
    cv_scores_sorted = sorted(cv_scores, key=lambda x: x[1], reverse=True)
    best_scoring_params = cv_scores_sorted[0][0]
    best_score = cv_scores_sorted[0][1]
    output_dict['best_scoring_hyperparams'] = best_scoring_params
    print(f"... highest scoring hyperparam (score={best_score}):\n{best_scoring_params}\n")
    for i in range(3): 
        p, s = cv_scores_sorted[i][0], cv_scores_sorted[i][1]
        print(f"...... rank #{i+1}: score={s}, setting: {p}")
    
    # How to choose the best? 
    output_dict['best_hyperparams'] = best_hyperparams = output_dict['best_scoring_hyperparams']

    # Also return the model with the best parameters (by mode)
    new_model = SGDClassifier(tol=1e-3, eta0=0.001)
    output_dict['model'] = new_model.set_params(**best_hyperparams)

    # return avg_f1, avg_roc_auc, avg_feature_importances
    return output_dict


def nested_cv_linear_svm_with_importance_via_hyperband(X, y, **kargs):
    kargs['loss_fn'] = ['hinge', ]
    return nested_cv_sgd_with_importance_via_hyperband(X, y, **kargs) 

def nested_cv_logistic_with_importance_via_hyperband(X, y, **kargs):
    kargs['loss_fn'] = ['log_loss', ]
    return nested_cv_sgd_with_importance_via_hyperband(X, y, **kargs)

def nested_cv_sgd_with_importance_via_hyperband(X, y, **kargs):
    # from skopt.space import Real, Categorical, Integer
    from scipy.stats import uniform, loguniform

    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import StandardScaler

    from sklearn.linear_model import SGDClassifier
    from sklearn.calibration import CalibratedClassifierCV
    
    from dask_ml.model_selection import HyperbandSearchCV
    import dask.array as da
    import shap

    import joblib
    # from sklearn.externals.joblib import parallel_backend
    from sklearn.utils import parallel_backend
    # from joblib import Parallel, delayed
    # from distributed import Client, LocalCluster

    # from dask.distributed import Client
    # client = Client(processes=False, threads_per_worker=2,
    #                 n_workers=6, memory_limit='20GB')
    # import dask.distributed
    # cluster = dask.distributed.LocalCluster()
    # client = dask.distributed.Client(cluster)

    feature_names = kargs.get("feature_names", [])
    loss_functions = kargs.get("loss_fn", ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'])

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        if len(feature_names) == 0: feature_names = list(X.columns)
        X = X.values
        # NOTE: Why converting X to a numpy array? 
        #       XGBoost imposes restrictions on these feature names: they must not contain 
        #       certain characters like [, ], or <.

    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values
    unique_classes = da.unique(y)

    # Define SGD classifier
    model = SGDClassifier(tol=1e-3, eta0=0.001)
    
    # NOTE: use_label_encoder=False => UserWarning: `use_label_encoder` is deprecated in 1.7.0.

    # Define the hyperparameter grid
    sgd_search_space = {
        'alpha': loguniform(1e-6, 1e+2),  # The higher the value, the stronger the regularization.
        'loss': loss_functions,  # ['hinge', 'log_loss', 'modified_huber', 'squared_hinge'], # 'perceptron'
        'penalty': ['l2', 'l1', 'elasticnet', ],
        'l1_ratio': uniform(0, 1), # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], # uniform(0.1, 1.0), 
        'fit_intercept': [True, False], 
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive', ], # 
        'power_t': uniform( 0.5, 0.99 ), 
        'average': [True, False], 
        # 'class_weight': [None, 'balanced', ],
        # NOTE:  class_weight 'balanced' is not supported for partial_fit. 
        # In order to use 'balanced' weights, use compute_class_weight('balanced', classes=classes, y=y). 
        # In place of y you can use a large enough sample of the full training set target to properly estimate the class frequency distributions.
    }
    search_space = kargs.get('search_space', sgd_search_space)

    # Nested CV can be time consuming and we don't necessarily want to use it everytime 
    default_hyperparameters = kargs.get("default_hyperparams", None)
    use_nested_cv = kargs.get('use_nested_cv', True)
    if default_hyperparameters is None or not default_hyperparameters:
        # Default hyperparameters
        default_hyperparameters = {
            'alpha': 0.0001,
            'loss': 'log_loss',
            'penalty': 'elasticnet',
            'l1_ratio': 0.5,
            'fit_intercept': True, 
            'learning_rate': 'adpative',
            'average': False, 
        }
        # NOTE: 'subsample': 
        #          Subsample ratio of the training instances. 
        #          Setting it to 0.5 means that XGBoost randomly collects half of the data instances to grow trees
        #      'colsample_bytree': 
        #         The fraction of features that can be selected for any given tree to train
    if not use_nested_cv: 
        if kargs.get('default_hyperparams', None): 
            print(f"[model] SGD: using known hyperparameters:\n{default_hyperparameters}\n")
        else: 
            print(f"[model] SGD: using default hyperparameters:\n{default_hyperparameters}\n")
    
    # Setup the inner and outer cross-validation
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    # n_folds_inner = kargs.get("n_folds_inner", n_folds_outer)
    # inner_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=n_folds_outer, shuffle=True, random_state=0)

    outer_f1_scores = []
    outer_roc_auc_scores = []
    all_feature_importances = []
    best_params_list = []

    # Outer cross-validation loop
    search = None
    n_params = 100  # sample about n parameters
    
    fold = 0
    test_case = np.random.choice(range(n_folds_outer), 1)[0]
    cv_scores = []
    for train_idx, val_idx in outer_cv.split(X, y):
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        # Inputs to hyperband
        n_examples = 100 * len(X_train)  # the number of passes through dataset for best model
        max_iter = n_params  # number of times partial_fit will be called
        chunk_size = n_examples // n_params # number of examples each call sees

        # NOTE: A rule-of-thumb to determine HyperbandSearchCV's input parameters requires knowing:
        # - the number of examples the longest trained model will see
        # - the number of hyperparameters to evaluate
        # 
        # We need to define 2 parameters
        #   - max_iter, which determines how many times to call partial_fit
        #   - the chunk size of the Dask array, which determines how many data each partial_fit call receives.

        # Create a Dask array with given chunk size
        X_train2 = da.from_array(X_train, chunks=chunk_size)
        y_train2 = da.from_array(y_train, chunks=chunk_size)

        if use_nested_cv:
            # Grid search with cross-validation in the inner loop
            # search = GridSearchCV(xgb_clf, param_grid, scoring='f1_macro', cv=inner_cv, n_jobs=-1)
            # NOTE: scoring 
            #       f1_macro, f1_micro, f1_weighted
            
            with joblib.parallel_backend('dask'):
                search = HyperbandSearchCV(
                                    model,  # NOTE: Doesn't work at the moment since xgb_clf does not implement partial_fit()
                                    search_space,
                                    max_iter=max_iter,
                                    scoring='f1_macro', 
                                    patience=True, 
                                    aggressiveness=3, 
                                    # random_state=0 
                                )
                # NOTE: A patience value is automatically selected if patience=True to work well with 
                #       the Hyperband model selection algorithm.

                # assert len(X_train) == len(y_train), f"len(X_train): {len(X_train)} =?= {len(y_train)}ß"

                # with joblib.parallel_backend('dask'):
                search.fit(X_train2, y_train2, classes=unique_classes) # Use chunked training set
            best_hyperparameters = search.best_params_
        else: 
            # Use default hyperparameters
            best_hyperparameters = default_hyperparameters

        # Test 
        if fold == test_case: 
            print(f"[model] Fold={fold}: SGD, best params={best_hyperparameters}, via nested cv? {use_nested_cv}")
            if use_nested_cv: 
                print(f"... best hyperparams:\n{best_hyperparameters}\n")
                print(f"... best score: {search.best_score_}")

                cv_results = pd.DataFrame(search.cv_results_)
                cv_results.sort_values(by='test_score', ascending=False, inplace=True)
                print(f"... CV results")
                print(cv_results.head())

                hist = pd.DataFrame(search.history_)
                hist.sort_values(by='score', ascending=False, inplace=True)
                print(f"... history:")
                print(hist.head())

        best_params_list.append(tuple(best_hyperparameters.items()))

        best_model = None
        if use_nested_cv: 

            # Predict and compute F1 score on the validation set
            y_pred = search.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='macro')  #
            outer_f1_scores.append(f1)
        
            # Get probability predictions
            # ---------------------------
            if best_hyperparameters['loss'] in ['log_loss', 'modified_huber', ]: 
                # Predict probabilities for ROC AUC computation
                y_prob = search.predict_proba(X_val)[:, 1]
            else: 
                # Instantiate a new model with the best hyperparameters and use probability calibration to get 
                # probability predictions
                model = SGDClassifier(tol=1e-3, eta0=0.001)
                model.set_params(**best_hyperparameters)
                print(f"[info] loss fn: {best_hyperparameters['loss']} does not support predict_proba(), use Platt scaling")
                model_calibrated = CalibratedClassifierCV(model, method='sigmoid')
                # NOTE: 'cv' argument is set to None by default => use the default 5-fold cross-validation to 
                #       estimate probabilities

                model_calibrated.fit(X_train, y_train)
                y_prob = model_calibrated.predict_proba(X_val)[:, 1]
            # -------------------
            # Now, we have y_prob

            roc_auc = roc_auc_score(y_val, y_prob)
            outer_roc_auc_scores.append(roc_auc)
                    
            best_model = search.best_estimator_

            # HyperbandSearchCV details on the amount of training and the number of models created. 
            # These details are available in the metadata attribute
            print(f"[hyperband] Number of models created: {search.metadata['n_models']}")
            print(f"... n(partial_fit calls): {search.metadata['partial_fit_calls']}")

        else: 
            best_model = SGDClassifier(tol=1e-3, eta0=0.001)
            best_model.set_params(**best_hyperparameters) # user-provided or default

            # Refit the model
            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_val)

            # F1 score 
            f1 = f1_score(y_val, y_pred, average='macro')  #
            outer_f1_scores.append(f1)

            # Probability prediction
            if best_hyperparameters['loss'] in ['log_loss', 'modified_huber', ]: 
                y_prob = best_model.predict_proba(X_val)[:, 1]
            else: 
                # Instantiate a new model with the best hyperparameters and use probability calibration to get 
                # probability predictions
                model = SGDClassifier(tol=1e-3, eta0=0.001)
                model.set_params(**best_hyperparameters) # user-provided or default
                prob_model = CalibratedClassifierCV(model, method='sigmoid')
                prob_model.fit(X_train, y_train)
                y_prob = prob_model.predict_proba(X_val)[:, 1]
            
            # AUC 
            roc_auc = roc_auc_score(y_val, y_prob)
            outer_roc_auc_scores.append(roc_auc)

        # CV Scores 
        best_params_score = (best_hyperparameters, f1)
        cv_scores.append(best_params_score)

        # Compute SHAP values for feature importances
        # explainer = shap.KernelExplainer(best_model.predict_proba, shap.kmeans(X_train, 50))
        # shap_values = explainer.shap_values(X_val)
        # feature_importances = np.mean(np.abs(shap_values[1]), axis=0)
        # all_feature_importances.append(feature_importances)

        # Compute feature importances (coefficients for logistic regression)

        feature_importances = best_model.coef_[0]
        all_feature_importances.append(feature_importances)

        fold += 1
    ### End CV iterations

    # Compute average F1 score, average ROC AUC, and average feature importances
    output_dict = {}
    output_dict['f1'] = avg_f1 = np.mean(outer_f1_scores)
    output_dict['auc'] = output_dict['roc_auc'] = avg_roc_auc = np.mean(outer_roc_auc_scores)
    output_dict['feature_importance'] = avg_feature_importances = np.mean(all_feature_importances, axis=0)

    if len(feature_names) > 0: 
        assert len(feature_names) == X.shape[1]
        print(f"[test] Example feature names:\n{feature_names[:10]}\n")

        # Displaying the feature importances
        output_dict['feature_importance_df'] = \
            pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_feature_importances
        })    

    # Determine the best hyperparameters
    print("[model] Choosing the best hyperparameter setting ...")
    # A. Choose the most frequently selected hyperparameters
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_params = dict(most_common_params)
    print(f"... most common hyperparam:\n{most_common_params}\n")

    # B. Choose the hyperparam setting that led to highest performance scores
    cv_scores_sorted = sorted(cv_scores, key=lambda x: x[1], reverse=True)
    best_scoring_params = cv_scores_sorted[0][0]
    best_score = cv_scores_sorted[0][1]
    output_dict['best_scoring_hyperparams'] = best_scoring_params
    print(f"... highest scoring hyperparam (score={best_score}):\n{best_scoring_params}\n")
    for i in range(3): 
        p, s = cv_scores_sorted[i][0], cv_scores_sorted[i][1]
        print(f"...... rank #{i+1}: score={s}, setting: {p}")
    
    # How to choose the best? 
    output_dict['best_hyperparams'] = best_hyperparams = output_dict['best_scoring_hyperparams']

    # Also return the model with the best parameters (by mode)
    new_model = SGDClassifier(tol=1e-3, eta0=0.001)
    output_dict['model'] = new_model.set_params(**best_hyperparams)

    # return avg_f1, avg_roc_auc, avg_feature_importances
    return output_dict

# Todo: Support partial_fit for SVM
def nested_cv_svm_with_importance_via_hyperband(X, y, **kargs):
    from skopt.space import Real, Categorical, Integer
    from scipy.stats import uniform, loguniform

    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import cross_val_score, KFold,  GridSearchCV

    from sklearn.svm import SVC, LinearSVC
    from dask_ml.model_selection import HyperbandSearchCV
    import dask.array as da
    import shap

    from dask.distributed import Client
    # import joblib
    # from joblib import Parallel, delayed
    from distributed import Client, LocalCluster

    # cluster = LocalCluster()
    # client = Client(cluster) # 
    client = Client(processes=False, threads_per_worker=2,
                    n_workers=6, memory_limit='20GB')

    feature_names = kargs.get("feature_names", [])

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        if len(feature_names) == 0: feature_names = list(X.columns)
        X = X.values
        # NOTE: Why converting X to a numpy array? 
        #       XGBoost imposes restrictions on these feature names: they must not contain 
        #       certain characters like [, ], or <.

    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values
    unique_classes = da.unique(y)

    # Define the XGBoost classifier
    # xgb_clf = XGBClassifier(random_state=0, eval_metric="logloss")
    model = SVC(probability=True, random_state=0)
    
    # NOTE: use_label_encoder=False => UserWarning: `use_label_encoder` is deprecated in 1.7.0.

    # Define the hyperparameter grid
    svm_search_space = {
        'C':  loguniform(1e-6, 1e+6),  # (1e-6, 1e+6, 'log-uniform'),  # Real(1e-6, 1e+3, prior='log-uniform')
        'gamma': loguniform(1e-6, 1e+1),  #  (1e-6, 1e+1, 'log-uniform'),
        'degree': np.arange(1, 8+1),  # integer valued parameter
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid', ],  # categorical parameter
    },
    search_space = param_grid = kargs.get('param_grid', svm_search_space)

    # Nested CV can be time consuming and we don't necessarily want to use it everytime 
    default_hyperparameters = kargs.get("default_hyperparams", None)
    use_nested_cv = kargs.get('use_nested_cv', True)
    if default_hyperparameters is None or not default_hyperparameters:
        # Default hyperparameters
        default_hyperparameters = {
            'C': 1,
            'gamma': 'scale',
            'kernel': 'rbf',
            'probability': True,
        }
        # NOTE: 'subsample': 
        #          Subsample ratio of the training instances. 
        #          Setting it to 0.5 means that XGBoost randomly collects half of the data instances to grow trees
        #      'colsample_bytree': 
        #         The fraction of features that can be selected for any given tree to train
    if not use_nested_cv: 
        if kargs.get('default_hyperparams', None): 
            print(f"[model] SVM: using known hyperparameters:\n{default_hyperparameters}\n")
        else: 
            print(f"[model] SVM: using default hyperparameters:\n{default_hyperparameters}\n")
    
    # Setup the inner and outer cross-validation
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    # n_folds_inner = kargs.get("n_folds_inner", n_folds_outer)
    # inner_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=n_folds_outer, shuffle=True, random_state=0)

    outer_f1_scores = []
    outer_roc_auc_scores = []
    all_feature_importances = []
    best_params_list = []

    # Outer cross-validation loop
    search = None
    n_params = 100  # sample about n parameters
    
    fold = 0
    test_case = np.random.choice(range(n_folds_outer), 1)[0]
    cv_scores = []
    for train_idx, val_idx in outer_cv.split(X, y):
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Inputs to hyperband
        n_examples = 100 * len(X_train)  # the number of passes through dataset for best model
        max_iter = n_params  # number of times partial_fit will be called
        chunk_size = n_examples // n_params # number of examples each call sees

        # NOTE: A rule-of-thumb to determine HyperbandSearchCV's input parameters requires knowing:
        # - the number of examples the longest trained model will see
        # - the number of hyperparameters to evaluate
        # 
        # We need to define 2 parameters
        #   - max_iter, which determines how many times to call partial_fit
        #   - the chunk size of the Dask array, which determines how many data each partial_fit call receives.

        # Create a Dask array with given chunk size
        X_train2 = da.from_array(X_train, chunks=chunk_size)
        y_train2 = da.from_array(y_train, chunks=chunk_size)

        if use_nested_cv:
            # Grid search with cross-validation in the inner loop
            # search = GridSearchCV(xgb_clf, param_grid, scoring='f1_macro', cv=inner_cv, n_jobs=-1)
            # NOTE: scoring 
            #       f1_macro, f1_micro, f1_weighted
            
            search = HyperbandSearchCV(
                                model,  # NOTE: Doesn't work at the moment since xgb_clf does not implement partial_fit()
                                search_space,
                                max_iter=max_iter,
                                scoring='f1_macro', 
                                patience=True, 
                                aggressiveness=3, 
                                random_state=0 
                            )
            # NOTE: A patience value is automatically selected if patience=True to work well with 
            #       the Hyperband model selection algorithm.

            # with joblib.parallel_backend('dask'):
            search.fit(X_train2, y_train2, classes=unique_classes) # Use chunked training set
            best_hyperparameters = search.best_params_
        else: 
            # Use default hyperparameters
            best_hyperparameters = default_hyperparameters

        # Test 
        if fold == test_case: 
            print(f"[model] Fold={fold}: XGBoost, best params={best_hyperparameters}, via nested cv? {use_nested_cv}")
            if use_nested_cv: 
                print(f"... best score: {search.best_score_}")

                cv_results = pd.DataFrame(search.cv_results_)
                print(f"... CV results")
                print(cv_results.head())

                hist = pd.DataFrame(search.history_)
                print(f"... history:")
                print(hist.head())

        best_params_list.append(tuple(best_hyperparameters.items()))
        
        # Refit the model using the best hyperparameters found
        best_svm = SVC(probability=True, random_state=0)
        
        best_svm.set_params(**best_hyperparameters)
        best_svm.fit(X_train, y_train)
        
        # Predict and compute F1 score on the validation set
        y_pred = best_svm.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')  #
        outer_f1_scores.append(f1)
        
        # Predict probabilities for ROC AUC computation
        y_prob = best_svm.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_prob)
        outer_roc_auc_scores.append(roc_auc)

        # CV Scores 
        cv_scores.append(f1)

        # Compute SHAP values for feature importances
        explainer = shap.KernelExplainer(best_svm.predict_proba, shap.kmeans(X_train, 50))
        shap_values = explainer.shap_values(X_val)
        feature_importances = np.mean(np.abs(shap_values[1]), axis=0)
        all_feature_importances.append(feature_importances)

        fold += 1

    # HyperbandSearchCV details on the amount of training and the number of models created. 
    # These details are available in the metadata attribute
    print(f"[hyperband] Number of models created: {search.metadata['n_models']}")
    print(f"... n(partial_fit calls): {search.metadata['partial_fit_calls']}")

    # Compute average F1 score, average ROC AUC, and average feature importances
    output_dict = {}
    output_dict['f1'] = avg_f1 = np.mean(outer_f1_scores)
    output_dict['auc'] = output_dict['roc_auc'] = avg_roc_auc = np.mean(outer_roc_auc_scores)
    output_dict['feature_importance'] = avg_feature_importances = np.mean(all_feature_importances, axis=0)

    if len(feature_names) > 0: 
        assert len(feature_names) == X.shape[1]
        print(f"[test] Example feature names:\n{feature_names[:10]}\n")

        # Displaying the feature importances
        output_dict['feature_importance_df'] = \
            pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_feature_importances
        })    

    # Determine the most frequently selected hyperparameters
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_params = dict(most_common_params)

    # Also return the model with the best parameters (by mode)
    new_model = SVC(probability=True, random_state=0)  # objective ='reg:squarederror'
    output_dict['model'] = new_model.set_params(**most_common_params)

    # return avg_f1, avg_roc_auc, avg_feature_importances
    return output_dict

def nested_cv_svm_with_importance_via_bohb(X, y, **kargs):
    """
    BOHB stands for Bayesian Optimization and Hyperband. It combines two powerful concepts:
    - Bayesian Optimization
    - Hyperband

    Memo
    ----
    1. Resource parameters,
        resource_name='n_samples', # can be either 'n_samples' or a string corresponding to an estimator attribute, eg. 'n_estimators' for an ensemble
        resource_type=float, # if specified, the resource value will be cast to that type before being passed to the estimator, otherwise it will be derived automatically
        min_budget=0.7,
        max_budget=1
        ... 
        - If not given sufficient resource (e.g. low min_budget), the model performance can be low
    
    References
    ----------
    1. https://dzone.com/articles/bayesian-optimization-and-hyperband-bohb-hyperpara
    2. https://automl.github.io/HpBandSter/build/html/quickstart.html
    """
    from sklearn.svm import SVC, LinearSVC
    import shap 

    # from hyperopt import hp
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.metrics import log_loss
    from sklearn.model_selection import cross_val_score, KFold
    from collections import Counter

    # from hpbandster.core.worker import Worker
    # from hpbandster.optimizers import BOHB 

    from hpbandster_sklearn import HpBandSterSearchCV
    from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
    # from ConfigSpace import UniformIntegerHyperparameter
    # import ConfigSpace as CS
    # import ConfigSpace.hyperparameters as CSH

    feature_names = kargs.get("feature_names", [])

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        if len(feature_names) == 0: feature_names = list(X.columns)
        X = X.values
        # NOTE: Why converting X to a numpy array? 
        #       XGBoost imposes restrictions on these feature names: they must not contain 
        #       certain characters like [, ], or <.

    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values
    classes = np.unique(y)
    n_classes = len(classes)

    # Define SVC type (this is incorporated into the objective function)
    model = SVC(probability=True) # kernel='linear'
        
    # NOTE: use_label_encoder=False => UserWarning: `use_label_encoder` is deprecated in 1.7.0.

    # Define the hyperparameter grid (this is defined through the objective function)
    config_space = { 
        'C': Float("C", bounds=(1e-6, 1e+3), log=True), 
        'gamma': Float("gamma", bounds=(1e-6, 1e+2), log=True), 
        'degree': Integer("degree", bounds=(1, 8), q=1), 
        'kernel': Categorical("kernel", ['linear', 'poly', 'rbf', 'sigmoid', ]), # weights=[0.1, 0.8, 3.14]
    }
    search_space = kargs.get('search_space', config_space)
    # NOTE: 
    #   `colsample_bytree` is the subsample ratio of columns when constructing each tree.
    # 6 * 6 * 4 * 3 * 3
    scoring = kargs.get('scoring', 'f1_macro') # 'roc_auc'

    # Nested CV can be time consuming and we don't necessarily want to use it everytime 
    default_hyperparameters = kargs.get("default_hyperparams", None)
    use_nested_cv = kargs.get('use_nested_cv', True)
    if default_hyperparameters is None or not default_hyperparameters:
        # Default hyperparameters
        default_hyperparameters = {
            'C': 1,
            'gamma': 'scale',
            'kernel': 'rbf',
            'probability': True,
            'random_state': 0
        }

    if not use_nested_cv: 
        if default_hyperparameters is not None: 
            print(f"[model] SVM: using known hyperparameters:\n{default_hyperparameters}\n")
        else: 
            print(f"[model] SVM: using default hyperparameters:\n{default_hyperparameters}\n")
    
    # Setup the inner and outer cross-validation
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    n_folds_inner = kargs.get("n_folds_inner", n_folds_outer)
    inner_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=n_folds_outer, shuffle=True, random_state=0)

    outer_f1_scores = []
    outer_roc_auc_scores = []
    all_feature_importances = []
    best_params_list = []

    # Outer cross-validation loop
    search = None
    max_iter = 120
    
    fold = 0
    test_case = np.random.choice(range(n_folds_outer), 1)[0]
    cv_scores = []
    for train_idx, val_idx in tqdm(outer_cv.split(X, y), total=n_folds_outer):
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardize the features
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        if use_nested_cv:
            param_distributions = ConfigurationSpace(
                    name="nmd_opt_svm",
                    seed=42, 
                    space=search_space,
            )
            search = HpBandSterSearchCV(model, 
                        param_distributions, 
                        scoring=scoring,
                        random_state=0, 
                            warm_start=False,
                            refit=True, # If True, refit an estimator using the best found parameters on the whole dataset
                            resource_name='n_samples', # can be either 'n_samples' or a string corresponding to an estimator attribute, eg. 'n_estimators' for an ensemble
                            resource_type=float, # if specified, the resource value will be cast to that type before being passed to the estimator, otherwise it will be derived automatically
                            min_budget=0.7,
                            max_budget=1,
                                n_jobs=3, 
                                verbose=1).fit(X_train, y_train) # n_iter=10,      

            best_hyperparameters = search.best_params_

        else: 
            # Use default hyperparameters
            best_hyperparameters = default_hyperparameters

        best_params_list.append(tuple(best_hyperparameters.items()))

        best_model = None
        if use_nested_cv: 
            # Predict and compute F1 score on the validation set
            y_pred = search.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='macro')  #
            outer_f1_scores.append(f1)

            # Predict probabilities for ROC AUC computation
            y_prob = search.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_prob)
            outer_roc_auc_scores.append(roc_auc)

            best_model = search.best_estimator_
        else: 
            # Refit the model using the best hyperparameters found
            # kernel = best_hyperparameters['kernel']
            best_model = SVC(probability=True)
            best_model.set_params(**best_hyperparameters)
            best_model.fit(X_train, y_train)
        
            # Predict and compute F1 score on the validation set
            y_pred = best_model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='macro')  #
            outer_f1_scores.append(f1)
            
            # Predict probabilities for ROC AUC computation
            y_prob = best_model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_prob)
            outer_roc_auc_scores.append(roc_auc)

        # CV Scores 
        best_params_score = (best_hyperparameters, f1)
        cv_scores.append(best_params_score)

        # Compute SHAP values for feature importances (Deferred to until best hyperparameters are determined)
        # masker = shap.maskers.Independent(X_train, 100)
        # explainer = shap.KernelExplainer(best_model.predict_proba, shap.kmeans(X_train, 50), link="logit") # shap.kmeans(X_train, 50)
        # shap_values = explainer.shap_values(X_val)
        # assert shap_values.ndim == 2
        # feature_importances = np.mean(np.abs(shap_values), axis=0)
        # all_feature_importances.append(feature_importances)

        # Test 
        if fold == test_case: 
            print(f"[model] SVM: Fold={fold}")
            if use_nested_cv: 
                print(f"... best hyperparams:\n{best_hyperparameters}\n")
                # print(f"... Best/min log loss:", search.best_value)
                print(f"... best score: {search.best_score_}")
                print(f"... F1: {f1}, ROCAUC: {roc_auc}")
                print()
                
                cv_results = pd.DataFrame(search.cv_results_)
                print(f"... CV results")
                print(cv_results.head()); print()

                # hist = pd.DataFrame(search.history_)
                # print(f"... history:")
                # print(hist.head())
                # NOTE: 'BayesSearchCV' object has no attribute 'history_'

        fold += 1

    # Compute average F1 score, average ROC AUC, and average feature importances
    output_dict = {}
    output_dict['f1'] = avg_f1 = np.mean(outer_f1_scores)
    output_dict['auc'] = output_dict['roc_auc'] = avg_roc_auc = np.mean(outer_roc_auc_scores)

    # Defer feature importance to later
    output_dict['feature_importance'] = None # avg_feature_importances = np.mean(all_feature_importances, axis=0)
    # output_dict['search'] = search # save a copy of the hyperparameter search result

    # Select the "best" hyperparameter settings from CV iterations
    print("[model] Choosing the best hyperparameter setting ...")
    # A. Choose the most frequently selected hyperparameters
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_params = dict(most_common_params)
    print(f"... most common hyperparam:\n{most_common_params}\n")

    # B. Choose the hyperparam setting that led to highest performance scores
    cv_scores_sorted = sorted(cv_scores, key=lambda x: x[1], reverse=True)
    best_scoring_params = cv_scores_sorted[0][0]
    best_score = cv_scores_sorted[0][1]
    output_dict['best_scoring_hyperparams'] = best_scoring_params
    print(f"... highest scoring hyperparams (score={best_score}):\n{best_scoring_params}\n")
    for i in range(3): 
        p, s = cv_scores_sorted[i][0], cv_scores_sorted[i][1]
        print(f"...... rank #{i+1}: score={s}, setting: {p}")

    # How to choose the best? 
    output_dict['best_hyperparams'] = best_hyperparams = output_dict['best_scoring_hyperparams']

    # Also return the model with the best parameters (by mode)
    print("[model] Using the optimized hyperparams to train a new SVM classifier ...")
    best_model = SVC(probability=True)
    best_model.set_params(**best_hyperparams)
    output_dict['model'] = best_model
    best_model.fit(X, y)

    print("[model] Computing feature importance based on new SVM classifier ...")
    masker = shap.maskers.Independent(X, 100)
    f = lambda x: best_model.predict_proba(x)[:,1]
    explainer = shap.KernelExplainer(f, shap.kmeans(X, 50), link="logit") # shap.kmeans(X_train, 50)
    shap_values = explainer.shap_values(X) # Output shap_values is a list
    if isinstance(shap_values, list): 
        assert len(shap_values) == 2
        shap_values = shap_values[1] # take positive class
    elif isinstance(shap_values, np.ndarray): 
        assert shap_values.ndim == 2
    output_dict['feature_importance'] = feature_importances = np.mean(np.abs(shap_values), axis=0)

    # Feature importance
    if len(feature_names) > 0: 
        assert len(feature_names) == X.shape[1]
        print(f"[test] Example feature names:\n{feature_names[:10]}\n")

        # Displaying the feature importances
        output_dict['feature_importance_df'] = \
            pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        })    

    # return avg_f1, avg_roc_auc, avg_feature_importances
    return output_dict

def nested_cv_svm_with_importance_via_bayes_opt(X, y, **kargs): 
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    from skopt.callbacks import DeadlineStopper, DeltaYStopper
    from skopt.plots import plot_objective, plot_histogram

    from sklearn.svm import SVC, LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    import shap

    # Check if X is a dataframe
    is_dataframe = isinstance(X, pd.DataFrame)
    # Check if y is a dataframe
    is_y_dataframe = isinstance(y, pd.DataFrame)

    # If X is a Dataframe, convert it to a series or array
    if is_dataframe:
        X = X.values
    # If y is a DataFrame, convert it to a series or array
    if is_y_dataframe:
        y = y.iloc[:, 0].values

    # Pipeline class is used as estimator to enable
    # search over different model types
    pipe = Pipeline([
        ('model', SVC(probability=True, random_state=0))
    ])

    linsvc_search = {
        'model': [LinearSVC(max_iter=1000, random_state=0)],
        'model__C': (1e-6, 1e+6, 'log-uniform'),
    }

    svc_search = \
        {
            'model': Categorical([SVC()]), # probability=True, random_state=0
            'model__C': Real(1e-6, 1e+3, prior='log-uniform'),
            'model__gamma': Real(1e-6, 100.0, prior='log-uniform'),
            'model__degree': Integer(1,8),
            'model__kernel': Categorical(['linear', 'poly', 'rbf', 'sigmoid', ]),
        }
    search_space = \
        kargs.get("search_space", 
               # (parameter space, # of evaluations)
               [(svc_search, 120), (linsvc_search, 20)])
    
    # model = SVC(probability=True, random_state=0)

    # Nested CV can be time consuming and we don't necessarily want to use it everytime 
    default_hyperparameters = kargs.get("default_hyperparams", None)
    use_nested_cv = kargs.get('use_nested_cv', True)
    if default_hyperparameters is None:
        # Default hyperparameters
        default_hyperparameters = {
            'C': 1,
            'gamma': 'scale',
            'kernel': 'rbf',
            'probability': True,
            'random_state': 0
        }
    if not use_nested_cv: 
        if default_hyperparameters is not None: 
            print(f"[model] SVM: using known hyperparameters:\n{default_hyperparameters}\n")
        else: 
            print(f"[model] SVM: using default hyperparameters:\n{default_hyperparameters}\n")
    
    # Other search parameters 
    search = None
    scoring = kargs.get("scoring", 'f1_macro')

    # Setup the inner and outer cross-validation
    n_folds_outer = n_folds = kargs.get("n_folds", 5)
    n_folds_inner = kargs.get("n_folds_inner", n_folds_outer)
    inner_cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    outer_cv = KFold(n_splits=n_folds_inner, shuffle=True, random_state=0)
    cv_scores = []

    outer_f1_scores = []
    outer_roc_auc_scores = []
    all_feature_importances = []
    best_params_list = []

    # Running the optimizer
    overdone_control = DeltaYStopper(delta=0.0001)                    # We stop if the gain of the optimization becomes too small
    time_limit_control = DeadlineStopper(total_time=60*60*5)          # We impose a time limit (5 hours)

    # Outer cross-validation loop
    fold = 1
    test_case = np.random.choice(range(n_folds_outer), 1)[0]
    for train_idx, val_idx in outer_cv.split(X, y):

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # X_train = X.iloc[train_idx] if is_dataframe else X[train_idx]
        # X_val = X.iloc[val_idx] if is_dataframe else X[val_idx]
        # # No need for special indexing for y as it's either an array or series now
        # y_train = y[train_idx]
        # y_val = y[val_idx]

        # Standardize the features
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)

        if use_nested_cv: 
            # Grid search with cross-validation in the inner loop
            # grid_search = GridSearchCV(svm_rbf, param_grid_rbf, scoring='f1_macro', cv=inner_cv, n_jobs=-1)
            # grid_search.fit(X_train_scaled, y_train)
            # best_hyperparameters = grid_search.best_params_

            search = \
                BayesSearchCV(estimator=pipe,                                    
                                search_space=search_space,                      
                                scoring=scoring,                                  
                                cv=inner_cv,                                           
                                # n_iter=max_iter,                              # max number of trials
                                n_points=10,                             # number of hyperparameter sets evaluated at the same time
                                n_jobs=5,                                # number of jobs
                                # iid=False,                             # if not iid it optimizes on the cv score (deprecated)
                                return_train_score=False,                         
                                refit=False,                                      
                                optimizer_kwargs={'base_estimator': 'GP'},    # optmizer parameters: we use Gaussian Process (GP)
                                random_state=0)                               # random state for replicability
            # NOTE: 
            #  - iid: If True, the data is assumed to be identically distributed across the folds, 
            #         and the loss minimized is the total loss per sample, and not the mean loss across the folds 

            # with joblib.parallel_backend('dask'):
            # search.fit(X_train, y_train, classes=unique_classes) # Use chunked training set
            # best_hyperparameters = search.best_params_
            best_hyperparameters = \
                optimize_and_report(
                    search, X_train, y_train, 
                        title='SVM_Probabilistic_Classifier', 
                          callbacks=[overdone_control, time_limit_control])
        else:
            # Use default hyperparameters
            best_hyperparameters = default_hyperparameters

        # Test 
        if fold == test_case: 
            print(f"[model] SVM: Fold={fold}")

            if use_nested_cv: 
                print(f"... best hyperparams:\n{best_hyperparameters}\n")
                print(f"... best score: {search.best_score_}")
                print()

                cv_results = pd.DataFrame(search.cv_results_)
                print(f"... CV results")
                print(cv_results.head()); print()

                # hist = pd.DataFrame(search.history_)
                # print(f"... history:")
                # print(hist.head())
    
        best_params_list.append(tuple(best_hyperparameters.items()))
        
        # Train the model using the best hyperparameters found
        best_svm = best_hyperparameters['model']
        # best_svm = SVC(probability=True, random_state=0)
        # best_svm.set_params(**best_hyperparameters)

        best_svm.fit(X_train, y_train)
        
        # Predict and compute F1 score on the validation set
        y_pred = best_svm.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro')
        outer_f1_scores.append(f1)
        
        # Predict probabilities for ROC AUC computation
        y_prob = best_svm.decision_function(X_val)
        roc_auc = roc_auc_score(y_val, y_prob)
        outer_roc_auc_scores.append(roc_auc)

        # Keep track of hyperparameter settings and their performance scores
        best_params_score = (best_hyperparameters, f1)
        cv_scores.append(best_params_score)

        # Compute SHAP values for feature importances
        explainer = shap.KernelExplainer(best_svm.predict_proba, shap.kmeans(X_train, 50))
        shap_values = explainer.shap_values(X_val)
        assert shap_values.ndim == 2
        feature_importances = np.mean(np.abs(shap_values), axis=0)
        all_feature_importances.append(feature_importances)

        fold += 1

    output_dict = {}

    # Compute average F1 score, average ROC AUC, and average feature importances
    output_dict['f1'] = avg_f1 = np.mean(outer_f1_scores)
    output_dict['auc'] = avg_roc_auc = np.mean(outer_roc_auc_scores)
    output_dict['feature_importance'] = avg_feature_importances = np.mean(all_feature_importances, axis=0)

    feature_names = kargs.get("feature_names", [])
    if len(feature_names) > 0: 
        assert len(feature_names) == X.shape[1]

        # Displaying the feature importances
        output_dict['feature_importance_df'] = \
            pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_feature_importances
        })

    # Select the "best" hyperparameter settings from CV iterations
    print("[model] Choosing the best hyperparameter setting ...")
    # A. Choose the most frequently selected hyperparameters
    counter = Counter(best_params_list)
    most_common_params = counter.most_common(1)[0][0]
    output_dict['most_common_hyperparams'] = most_common_params = dict(most_common_params)
    print(f"... most common hyperparam:\n{most_common_params}\n")

    # B. Choose the hyperparam setting that led to highest performance scores
    cv_scores_sorted = sorted(cv_scores, key=lambda x: x[1], reverse=True)
    best_scoring_params = cv_scores_sorted[0][0]
    best_score = cv_scores_sorted[0][1]
    output_dict['best_scoring_hyperparams'] = best_scoring_params
    print(f"... highest scoring hyperparam (score={best_score}):\n{best_scoring_params}\n")
    for i in range(3): 
        p, s = cv_scores_sorted[i][0], cv_scores_sorted[i][1]
        print(f"...... rank #{i+1}: score={s}, setting: {p}")

    # How to choose the best? 
    output_dict['best_hyperparams'] = best_hyperparams = output_dict['best_scoring_hyperparams']

    # Also return the model with the best parameters (by mode)
    model = best_scoring_params['model']  # Using this Pipeline, the most common params include the model itself 
    # SVC(probability=True, random_state=0)
    # model.set_params(**most_common_params)
    output_dict['model'] = model
    
    # return avg_f1, avg_roc_auc, avg_feature_importances
    return output_dict

def demo_nested_cv_xgb(): 
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    X_diabetes = data.data
    y_diabetes = (data.target > data.target.mean()).astype(int)  # Convert to binary classification problem

    # f1_score_result, feature_importances_result = nested_cv_xgb(X_diabetes, y_diabetes)
    output_dict = nested_cv_xgboost_with_importance(X_diabetes, y_diabetes, feature_names=data.feature_names)

    avg_f1 = output_dict['f1']
    avg_roc_auc = output_dict['auc']
    avg_feature_importances = output_dict['feature_importance']

    print(f"[XGB] F1 score: {avg_f1}")
    print(f"...   ROCAUC:   {avg_roc_auc}")
    print(f"...   feature importance scores:\n{avg_feature_importances}\n")
    print(f"{output_dict['feature_importance']}")

    return

def demo_nested_cv_linear_svm(): 
    # Let's test the function using the diabetes dataset from scikit-learn
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    X_diabetes = data.data
    y_diabetes = (data.target > data.target.mean()).astype(int)  # Convert to binary classification problem

    # Testing the updated function with the diabetes dataset
    output_dict = nested_cv_linear_svm_with_importance(X_diabetes, y_diabetes, feature_names=data.feature_names)
    avg_f1 = output_dict['f1']
    avg_roc_auc = output_dict['auc']
    avg_feature_importances = output_dict['feature_importance']

    print(f"[SVM] F1 score: {avg_f1}")
    print(f"...   ROCAUC:   {avg_roc_auc}")
    print(f"... feature importance scores:\n{avg_feature_importances}\n")
    print(f"{output_dict['feature_importance_df']}")

    return 

def demo_svm_nonlinear_kernel_explainer(): 
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    X_diabetes = data.data
    y_diabetes = (data.target > data.target.mean()).astype(int)  # Convert to binary classification problem

    svm_with_nonlinear_kernel_shap_explainer(X_diabetes, y_diabetes)

    # print(f"[SVM] F1 score: {f1_score_result}")
    # print(f"... feature importance scores:\n{feature_importances_result}\n")

def demo_rbf_svm_with_shap(): 
    from sklearn.datasets import load_diabetes

    # Load the diabetes dataset
    data = load_diabetes()
    X_diabetes = data.data
    y_diabetes = data.target
    median_target = np.median(y_diabetes)
    y_diabetes = np.where(y_diabetes > median_target, 1, 0)

    # Running the function to get the results again
    output_dict = nested_cv_rbf_svm_with_shap(X_diabetes, y_diabetes, feature_names=data.feature_names)
    avg_f1 = output_dict['f1']
    avg_roc_auc = output_dict['auc']
    avg_feature_importances = output_dict['feature_importance']

    # Displaying the SHAP-based feature importances for the diabetes dataset using SVM with RBF kernel
    feature_importance_rbf_df = output_dict['feature_importance_df']

    print(f"[SVM] F1 score: {avg_f1}")
    print(f"...   ROCAUC:   {avg_roc_auc}")
    print(f"... feature importance scores:\n{avg_feature_importances}\n")

    print("[demo] feature importance by SVM with RBF kernel ...")
    feature_importance_rbf_df.sort_values(by='Importance', ascending=False, inplace=True)
    print(feature_importance_rbf_df)

    return

def standardize_model_name(model_name):
    # Convert to lowercase and remove hyphens and plus signs
    standardized_name = model_name.lower().replace('-', '').replace('+', '').replace('_', '')
    
    # Dispatch appropriate model
    if "svm" in standardized_name and "rbf" in standardized_name:
        print("Running SVM with Gaussian (RBF) kernel")
        model_name = "rbf_svm"
    elif "xgb" in standardized_name: 
        model_name = "xgboost"
    elif not ("x" in standardized_name) and ("gb" in standardized_name or "gradientboost" in standardized_name):
        print("Running Gradient Boosted Trees")
        model_name = "gb"
    elif model_name == 'rf' or ("random" in standardized_name and "foreset" in standardized_name):
        model_name = "rf"
    else:
        # print("Model not recognized")
        pass 
    return model_name

def nested_cv_and_feaure_importance(X, y, *, model_name='xgboost', **kargs): 

    # model_name = standardize_name(model_name) # standardize model name
    feature_names = kargs.get("feature_names", [])

    output_dict = {}

    model_name = model_name.lower()
    if model_name.startswith("logi"): 
        # output_dict = nested_cv_logistic_regression_with_importance(X, y, **kargs)
        # output_dict = nested_cv_logistic_with_importance_via_hyperband(X, y, **kargs)
        output_dict = nested_cv_logistic_with_importance_via_bohb(X, y, **kargs)
    elif model_name.startswith('sgd'):
        # output_dict = nested_cv_sgd_with_importance_via_hyperband(X, y, **kargs)
        output_dict = nested_cv_sgd_with_importance_via_bohb(X, y, **kargs)

    elif model_name in ("linear_svm", ): 
        # output_dict = nested_cv_linear_svm_with_importance(X, y, **kargs)
        # output_dict = nested_cv_linear_svm_with_importance_via_hyperband(X, y, **kargs)
        output_dict = nested_cv_linear_svm_with_importance_via_bohb(X, y, **kargs)
    elif model_name in ("svm",  ): # "rbf_svm", "rbf+svm", "rbf-svm", 
        # output_dict = nested_cv_rbf_svm_with_shap(X, y, **kargs)
        # output_dict = nested_cv_rbf_svm_with_importance(X, y, **kargs)
        
        # Bayesian optimization
        # output_dict = nested_cv_svm_with_importance_via_bayes_opt(X, y, **kargs)

        # Hyperband 
        # output_dict = nested_cv_svm_with_importance_via_hyperband(X, y, **kargs)
        # NOTE: SVC also doesn't have partial_fit

        # BOHB: Bayesian optimization + hyperband 
        output_dict = nested_cv_svm_with_importance_via_bohb(X, y, **kargs)

    elif model_name.startswith( ("xgb", "xgboost", ) ):
        # output_dict = nested_cv_xgboost_with_importance(X, y, **kargs)
        # output_dict = nested_cv_xgboost_with_importance_via_bayes_opt(X, y, **kargs)
        # output_dict = nested_cv_xgboost_with_importance_via_bohb(X, y, **kargs)
        output_dict = nested_cv_xgboost_with_importance_via_optuna(X, y, **kargs)
    elif model_name.startswith( ("rf", "random_forest") ): 
        output_dict = nested_cv_random_forest_with_importance(X, y, **kargs)
    elif model_name.startswith( ("gb", "gbt", "gradient_boosting") ): 
        output_dict = nested_cv_gradient_boosting_with_importance(X, y, **kargs)
    else: 
        msg = f"Unrecognized model name: {model_name}"
        raise NotImplementedError(msg)

    avg_f1 = output_dict['f1']
    avg_roc_auc = output_dict['auc']
    avg_feature_importances = output_dict['feature_importance']
    most_common_hyperparams = output_dict['most_common_hyperparams']

    print(f"[report] model: {model_name}")

    # Displaying the SHAP-based feature importances for the diabetes dataset using SVM with RBF kernel
    feature_importance_df = output_dict['feature_importance_df']

    print(f"... F1 score: {avg_f1}")
    print(f"... ROCAUC:   {avg_roc_auc}")
    print(f"... feature importance scores:\n{avg_feature_importances}\n")
    print(f"... most common hyperparams:\n{most_common_hyperparams}\n")

    print("Sorting features by importance scores ...")
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    print(feature_importance_df)

    return output_dict

def demo_nested_cv_and_feaure_importance(model_name): 
    from sklearn.datasets import load_diabetes

    # Load the diabetes dataset
    data = load_diabetes()
    X_diabetes = data.data
    y_diabetes = data.target
    median_target = np.median(y_diabetes)
    y_diabetes = np.where(y_diabetes > median_target, 1, 0)

    # model_name = "logistic regression"

    output_dict = nested_cv_and_feaure_importance(X_diabetes, y_diabetes, 
                    model_name=model_name, 
                    use_nested_cv = True,
                    feature_names=data.feature_names)


    return

def demo_nested_cv_xgboost_with_importance_via_optuna(): 
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Data Generation Code
    # Generate a synthetic binary classification dataset
    X, y = make_classification(
        n_samples=1000,        # Number of samples
        n_features=20,         # Total number of features
        n_informative=15,      # Number of informative features
        n_redundant=5,         # Number of redundant features
        n_classes=2,           # Number of classes
        random_state=42        # Seed for reproducibility
    )

    # Optional: Create a DataFrame to include feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)

    # Split the data into training and testing sets (if needed)
    # For this test, we'll use the entire dataset
    # X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

    # Invoke the function with the generated data
    output = nested_cv_xgboost_with_importance_via_optuna(
        X_df, y,
        feature_names=feature_names,
        scoring='f1_macro',
        n_folds=5,
        n_folds_inner=3,
        use_nested_cv=True,
        n_trials=5  # Adjust the number of trials as needed
    )

    # Print the outputs
    print("\n=== Performance Metrics ===")
    print(f"Average F1 Score: {output['f1']}")
    print(f"Average ROC AUC Score: {output['roc_auc']}")
    print(f"Average Precision: {output['precision']}")
    print(f"Average Recall: {output['recall']}")
    print(f"Average False Positive Rate: {output['fpr']}")
    print(f"Average False Negative Rate: {output['fnr']}")

    print("\n=== Feature Importances ===")
    feature_importance_df = output['feature_importance_df']
    print(feature_importance_df.sort_values(by='Importance', ascending=False).head(10))

    print("\n=== Best Hyperparameters ===")
    print("Most Common Hyperparameters:")
    print(output['most_common_hyperparams'])
    print("\nBest Scoring Hyperparameters:")
    print(output['best_scoring_hyperparams'])

    # Access the best model
    best_model = output['best_model']

    # If you have test data, you can make predictions
    # y_pred = best_model.predict(X_test)
    # Evaluate the model on the test data
    # from sklearn.metrics import classification_report
    # print("\n=== Classification Report on Test Data ===")
    # print(classification_report(y_test, y_pred))


def test(): 
  
    # XGB 
    # demo_nested_cv_xgb() 
 
    # SVM 
    # demo_nested_cv_linear_svm()    

    # SVM with non-linear kernel using SHAP as feature explainer
    # demo_svm_nonlinear_kernel_explainer()  # ... v0
    # demo_rbf_svm_with_shap()

    # Model selection -> training -> evaluation -> feature importance
    # demo_nested_cv_and_feaure_importance("svm")
    # Options: logistic regression, rbf_svm, linear_svm, random_forest (rf), xgboost (xgb), gradient_boosting (gb)

    demo_nested_cv_xgboost_with_importance_via_optuna()

    return

if __name__ == "__main__": 
    test()


