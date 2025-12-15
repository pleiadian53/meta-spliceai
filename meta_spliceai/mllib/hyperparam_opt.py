import os, sys
# sys.path.append('..')

import numpy as np
np.random.seed(123)

import logging
logging.basicConfig(level=logging.WARNING)

import pandas as pd
from tqdm import tqdm

from collections import Counter
import matplotlib.pyplot as plt


def demo_bohb0(**kargs):
    """
    This implementation does not work as it is.
    
    References
    ----------
    1. https://automl.github.io/HpBandSter/build/html/auto_examples/example_5_mnist.html
    """
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from hpbandster.core.worker import Worker
    from hpbandster.optimizers import BOHB 

    import xgboost as xgb
    from hyperopt import hp

    # Load data 
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    scoring = kargs.get('scoring', 'aucpr')

    # Define the objective function
    def objective_function(config):
        """Objective function for BoHB."""

        # Create an XGBoost classifier
        clf = xgb.XGBClassifier(**config)

        # Train the classifier
        clf.fit(X_train, y_train, eval_metric=scoring)

        # Evaluate the classifier on the test set
        score = clf.score(X_test, y_test) # eval_metric=scoring

        return -score  # return value should be negative

    # Hyperparameters search space 
    # Define the configuration space
    config_space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, log=True),
        'max_depth': hp.quniform('max_depth', 1, 10, q=1),
        'subsample': hp.uniform('subsample', 0.05, 1.0),
        'learning_rate': hp.loguniform('learning_rate', 0.001, 0.1), # cf in optuna: trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        'gamma': hp.loguniform('gamma', 0.01, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': hp.loguniform('reg_alpha', 0.01, 1.0),
        'reg_lambda': hp.loguniform('reg_lambda', 0.01, 1.0),
    }

    ### Initialize the BOHB Optimizer
    
    ### Create a BoHB optimizer
    optimizer = BOHB(config_space=config_space)

    ### Run the BoHB optimization
    optimizer.run(objective_function, budget=100)

    ### Retrieve the Best Hyperparameters and Score

    # Get the best configuration
    best_config = optimizer.get_best_config()

    # Print the best configuration
    print('Best configuration:', best_config)

    # Best score? 
    print(dir(optimizer))

    return

def demo_bohb(**kargs):
    """
    This implementation does not work as it is.

    Install HpBandSter's Scikit-learn wrapper: 
        pip install hpbandster-sklearn

    Memo
    ----
    1. ConfigSpace
       - https://automl.github.io/ConfigSpace/main/
       - https://automl.github.io/ConfigSpace/main/api/hyperparameters.html
    2. Scoring: 
       The 'scoring' parameter of check_scoring must be a str among {
            'mutual_info_score', 'jaccard_samples', 'recall_weighted', 'neg_mean_squared_log_error', 
            'roc_auc_ovo', 'precision_macro', 'neg_mean_absolute_error', 
            'neg_negative_likelihood_ratio', 'homogeneity_score', 
            'f1_micro', 'neg_root_mean_squared_error', 'precision_samples', 'recall_samples', 
            'fowlkes_mallows_score', 'neg_mean_absolute_percentage_error', 
            'neg_log_loss', 'recall_macro', 'jaccard', 'precision_weighted', '
            rand_score', 'precision', 'roc_auc', 'f1_macro', 'jaccard_macro', 'recall', 
            'max_error', 'neg_brier_score', 'matthews_corrcoef', 'accuracy', 'roc_auc_ovo_weighted', 
            'f1_weighted', 'neg_mean_poisson_deviance', 'r2', 'explained_variance', 
            'average_precision', 'neg_median_absolute_error', 'positive_likelihood_ratio', 
            'adjusted_mutual_info_score', 'f1_samples', 'roc_auc_ovr', 'completeness_score', 
            'precision_micro', 'v_measure_score', 'jaccard_weighted', 'recall_micro', 
            'jaccard_micro', 'roc_auc_ovr_weighted', 'normalized_mutual_info_score', 
            'neg_mean_squared_error', 'adjusted_rand_score', 'f1', 'neg_mean_gamma_deviance', 
            'balanced_accuracy', 'top_k_accuracy'}
    
    References
    ----------
    1. https://automl.github.io/HpBandSter/build/html/auto_examples/example_5_mnist.html
    """
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris
    from hpbandster.core.worker import Worker
    from hpbandster.optimizers import BOHB 

    from hpbandster_sklearn import HpBandSterSearchCV
    from ConfigSpace import ConfigurationSpace, Integer, Float, Categorical, Normal
    from ConfigSpace import UniformIntegerHyperparameter
    # import ConfigSpace as CS
    # import ConfigSpace.hyperparameters as CSH

    import xgboost as xgb
    # from hyperopt import hp

    # Load data 
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    scoring = kargs.get('scoring', 'f1_macro') # 'roc_auc'

    # Define the objective function
    def objective_function(config):
        """Objective function for BoHB."""

        # Create an XGBoost classifier
        clf = xgb.XGBClassifier(**config)

        # Train the classifier
        clf.fit(X_train, y_train, eval_metric=scoring)

        # Evaluate the classifier on the test set
        score = clf.score(X_test, y_test) # eval_metric=scoring

        return -score  # return value should be negative

    # Hyperparameters search space 
    # Define the configuration space
    config_space = {
        "objective": "multi:softprob", # "binary:logistic", 
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

    param_distributions = ConfigurationSpace(
        name="nmd_opt_xgboost",
        seed=42, 
        space=config_space,
    )
    # param_distributions.add_hyperparameter()
    # param_distributions.add_hyperparameter(CSH.UniformIntegerHyperparameter("max_depth", 1, 11))

    ### Initialize the BOHB Optimizer
    
    ### Create a BoHB optimizer
    # optimizer = BOHB(config_space=param_distributions)

    clf = xgb.XGBClassifier() # eval_metric=scoring
    search = HpBandSterSearchCV(clf, 
                param_distributions, 
                scoring=scoring,
                random_state=0, 
                    warm_start=False,
                    resource_name='n_samples', # can be either 'n_samples' or a string corresponding to an estimator attribute, eg. 'n_estimators' for an ensemble
                    resource_type=float, # if specified, the resource value will be cast to that type before being passed to the estimator, otherwise it will be derived automatically
                    min_budget=0.5,
                    max_budget=1,
                        n_jobs=2, 
                        verbose=1).fit(X_train, y_train) # n_iter=10, 

    ### Run the BoHB optimization
    # optimizer.run(objective_function, budget=100)

    ### Retrieve the Best Hyperparameters and Score

    # Get the best configuration
    # best_config = optimizer.get_best_config()
    best_config = search.best_params_

    # Print the best configuration
    print('Best configuration:', best_config)

    # Best score? 
    best_score = search.best_score_
    print('Best score:', best_score)

    # CV results? 
    print("> CV Results:")
    cv_results = pd.DataFrame(search.cv_results_)
    print(f"... CV results")
    print(cv_results.head()); print()

    # Best estimator? 
    print("> Best model")
    print(search.best_estimator_)
    print(search.best_estimator_.get_params()) 

    # history 
    # print("> Search history")
    # print(search.history_)  # 'HpBandSterSearchCV' object has no attribute 'history_

    return

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

def nested_cv_xgboost_with_importance_via_bohb(X, y, **kargs):
    """
    BOHB stands for Bayesian Optimization and Hyperband. It combines two powerful concepts:
    - Bayesian Optimization
    - Hyperband
    
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

    objective = "binary:logistic"
    if n_classes > 2: 
        objective = "multi:softprob"

    # Define the XGBoost classifier (this is incorporated into the objective function)
    model = xgb.XGBClassifier(random_state=0, eval_metric="logloss") # objective="binary:logistic"
        
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
            param_distributions = ConfigurationSpace(
                    name="nmd_opt_xgboost",
                    seed=42, 
                    space=search_space,
            )
            search = HpBandSterSearchCV(model, 
                        param_distributions, 
                        scoring=scoring,
                        random_state=0, 
                            refit=True, # If True, refit an estimator using the best found parameters on the whole dataset
                            warm_start=False,
                            resource_name='n_samples', # can be either 'n_samples' or a string corresponding to an estimator attribute, eg. 'n_estimators' for an ensemble
                            resource_type=float, # if specified, the resource value will be cast to that type before being passed to the estimator, otherwise it will be derived automatically
                            min_budget=0.5,
                            max_budget=1,
                                n_jobs=-1, 
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

            best_model = xgb.XGBClassifier(random_state=0) # aucpr, logloss
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

        # Compute feature importances
        feature_importances = best_model.feature_importances_  # weights/frequency by default
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

def demo_xgboost_bayes_sv(**kargs): 

    from skopt import BayesSearchCV 

    # import xgboost as xgb
    from xgboost import XGBClassifier
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(n_class=10, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

    # Define the hyperparameter grid
    param_grid_xgb = {
        'n_estimators': [25, 50, 100, 200, 300, 400], # [100, 200, 300],
        'max_depth': [2, 4, 6, 8, None], # [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2, 0.3], # [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0], # [0.8, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0], # [0.8, 1.0]  
    }
    param_grid = kargs.get('param_grid', param_grid_xgb)

    xgb_clf = XGBClassifier(random_state=0, eval_metric="logloss")

    opt = BayesSearchCV(
        xgb_clf,
        param_grid,
        n_iter=32,
        cv=5
    )

    opt.fit(X_train, y_train)

    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(X_test, y_test))


    return 

def demo_xgboost_bayes_cv2(): 
    from xgboost import XGBClassifier
    
    from skopt import gp_minimize
    from skopt.space import Real, Integer

    from sklearn.datasets import load_digits
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from functools import partial

    # Data
    X, y = load_digits(n_class=10, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

    # defining the space
    space = [
        Real(0.6, 0.7, name="colsample_bylevel"),
        Real(0.6, 0.7, name="colsample_bytree"),
        Real(0.01, 1, name="gamma"),
        Real(0.0001, 1, name="learning_rate"),
        Real(0.1, 10, name="max_delta_step"),
        Integer(6, 15, name="max_depth"),
        Real(10, 500, name="min_child_weight"),
        Integer(10, 100, name="n_estimators"),
        Real(0.1, 100, name="reg_alpha"),
        Real(0.1, 100, name="reg_lambda"),
        Real(0.4, 0.7, name="subsample"),
    ]

    # function to fit the model and return the performance of the model
    def return_model_assessment(args, X_train, y_train, X_test, y_test):

        global models, train_scores, test_scores, curr_model_hyper_params

        params = {curr_model_hyper_params[i]: args[i] for i, j in enumerate(curr_model_hyper_params)}
        model = XGBClassifier(random_state=42, seed=42)
        model.set_params(**params)
        fitted_model = model.fit(X_train, y_train, sample_weight=None)
        models.append(fitted_model)
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        train_score = f1_score(train_predictions, y_train)
        test_score = f1_score(test_predictions, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        return 1 - test_score

    # collecting the fitted models and model performance
    models = []
    train_scores = []
    test_scores = []
    curr_model_hyper_params = ['colsample_bylevel', 'colsample_bytree', 'gamma', 'learning_rate', 'max_delta_step',
                            'max_depth', 'min_child_weight', 'n_estimators', 'reg_alpha', 'reg_lambda', 'subsample']
    
    objective_function = partial(return_model_assessment, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    # running the algorithm
    n_calls = 50 # number of times you want to train your model
    results = gp_minimize(objective_function, space, base_estimator=None, n_calls=n_calls, 
                          n_random_starts=n_calls-1, random_state=42)


    return results

def demo_svm_bayes_opt(**kargs): 
    # print(__doc__)

    from skopt import BayesSearchCV 
    # pip install scikit-optimize

    from sklearn.datasets import load_digits
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split

    X, y = load_digits(n_class=10, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

    # log-uniform: understand as search over p = exp(x) by varying x
    opt = BayesSearchCV(
        SVC(),
        {
            'C': (1e-6, 1e+6, 'log-uniform'),
            'gamma': (1e-6, 1e+1, 'log-uniform'),
            'degree': (1, 8),  # integer valued parameter
            'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
        },
        n_iter=32,
        cv=3
    )

    opt.fit(X_train, y_train)

    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(X_test, y_test))

    return

def demo_svm_pipeline_bayes_opt(): 
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    from skopt.plots import plot_objective, plot_histogram

    from sklearn.datasets import load_digits
    from sklearn.svm import LinearSVC, SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    X, y = load_digits(n_class=10, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # pipeline class is used as estimator to enable
    # search over different model types
    pipe = Pipeline([
        ('model', SVC())
    ])

    # single categorical value of 'model' parameter is
    # sets the model class
    # We will get ConvergenceWarnings because the problem is not well-conditioned.
    # But that's fine, this is just an example.
    linsvc_search = {
        'model': [LinearSVC(max_iter=1000)],
        'model__C': (1e-6, 1e+6, 'log-uniform'),
    }

    # explicit dimension classes can be specified like this
    svc_search = {
        'model': Categorical([SVC()]),
        'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
        'model__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
        'model__degree': Integer(1,8),
        'model__kernel': Categorical(['linear', 'poly', 'rbf']),
    }

    opt = BayesSearchCV(
        pipe,
        # (parameter space, # of evaluations)
        [(svc_search, 40), (linsvc_search, 16)],
        cv=3
    )

    opt.fit(X_train, y_train)

    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(X_test, y_test))
    print("best params: %s" % str(opt.best_params_))

    best_hyperparams = opt.best_params_
    model = best_hyperparams['model']

    print(f"> model name: {model.__class__.__name__}")
    print(f"... name? {str(model)}")

    return

def demo_sgd_hyperband_cv(): 
    import numpy as np
    from meta_spliceai.mllib import model_selection as ms
    from meta_spliceai.nmd_concept.nmd_targets_analyzer import get_nmd_data
    # from hyperopt import hp
    # from hyperopt.pyll.stochastic import sample

    from dask_ml.model_selection import HyperbandSearchCV
    from dask_ml.datasets import make_classification

    from sklearn.preprocessing import StandardScaler
    
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import SGDClassifier

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score

    from scipy.stats import uniform, loguniform

    import dask.array as da

    import dask.distributed
    cluster = dask.distributed.LocalCluster()
    client = dask.distributed.Client(cluster)

    # Get data
    # X, y = make_classification(chunks=20)
    X, y, meta_data = \
        get_nmd_data(policy='topn', threshold_eff=0.3, topn=300, 
                        verbose=1, 
                        return_x_as_dataframe=True, return_meta_data=True) # return dataframe so that we can get feature names / columns
    feature_names = list(X.columns)
    X = X.values
    print(f"... shape(X): {X.shape}, shape(y): {y.shape}")
    print(f"... type(X): {type(X)}, type(y): {type(y)}")
    f_encoded = list(set(meta_data['features_encoded']) - set(meta_data['features_original']))
    print(f"... encoded features (n={len(f_encoded)}):\n{f_encoded}\n")

    model_name = 'SGD'
    n_folds = 5 

    output_dict = ms.nested_cv_and_feaure_importance(X, y, 
                    model_name=model_name, 
                    n_folds=n_folds,  # n_folds_outer, n_folds_inner,
                    feature_names=feature_names, 
                    use_nested_cv=True)
    # output_dict = ms.nested_cv_sgd_with_importance_via_hyperband(X, y, 
    #                     n_folds=n_folds, 
    #                     feature_names=feature_names, 
    #                     use_nested_cv=True)

    # Show detailed classification performance report
    avg_f1 = output_dict['f1']
    avg_roc_auc = output_dict['auc']
    avg_feature_importances = output_dict['feature_importance']
    most_common_hyperparams = output_dict['most_common_hyperparams']

    # Displaying the SHAP-based feature importances for the diabetes dataset using SVM with RBF kernel
    feature_importance_df = output_dict['feature_importance_df']

    print(f"... F1 score: {avg_f1}")
    print(f"... ROCAUC:   {avg_roc_auc}")
    print(f"... feature importance scores:\n{avg_feature_importances}\n")
    print(f"... most common hyperparams:\n{most_common_hyperparams}\n")

    print("[demo] Sorting features by importance scores ...")
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    print(feature_importance_df)
    
    return


def demo_sgd_hyperband(): 
    import numpy as np

    # from hyperopt import hp
    # from hyperopt.pyll.stochastic import sample

    from dask_ml.model_selection import HyperbandSearchCV
    from dask_ml.datasets import make_classification

    from sklearn.preprocessing import StandardScaler
    
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import SGDClassifier

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, roc_auc_score

    from scipy.stats import uniform, loguniform

    import dask.array as da
    from meta_spliceai.nmd_concept.nmd_targets_analyzer import get_nmd_data
    

    import dask.distributed
    cluster = dask.distributed.LocalCluster()
    client = dask.distributed.Client(cluster)

    # from distributed import Client
    # client = Client(processes=False, threads_per_worker=2,
    #             n_workers=6, memory_limit='20GB')

    # Get data
    # X, y = make_classification(chunks=20)
    X, y = get_nmd_data(policy='topn', threshold_eff=0.3, topn=300, verbose=1)
    print(f"... shape(X): {X.shape}, shape(y): {y.shape}")
    print(f"... type(X): {type(X)}, type(y): {type(y)}")
    # NOTE: If y is a single label, then y is expected to be 1D, not a column vector
    #       If passed column vector => Warning: A column-vector y was passed when a 1d array was expected.
    #                                  Please change the shape of y to (n_samples, )
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    est = SGDClassifier(tol=1e-3, eta0=0.001) # eta0=0.01

    # Wrap the SGDClassifier with CalibratedClassifierCV
    # calibrated_est = CalibratedClassifierCV(est, method='sigmoid')

    param_dist = {'alpha': loguniform(1e-6, 1.0),    # np.logspace(-6, 2, num=1000), # The higher the value, the stronger the regularization.
                  'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'], # , ['log_loss', 'huber', 'modified_huber'], #
                  'penalty': ['l2', 'l1', 'elasticnet', ],
                  'l1_ratio': uniform(0, 1), # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], # uniform(0.1, 1.0), 
                  'fit_intercept': [True, False], 
                  'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive', ], # 
                  'power_t': uniform( 0.5, 0.99 ), 
                  'average': [True, False], 
                  # 'class_weight': [None, 'balanced', ],
                  # NOTE:  class_weight 'balanced' is not supported for partial_fit. In order to use 'balanced' weights, 
                  # use compute_class_weight('balanced', classes=classes, y=y). In place of y you can use a large enough sample 
                  # of the full training set target to properly estimate the class frequency distributions.
                  }

    # Using HyperOpt
    # space = {
    #     'scaler': hp.choice( 's', 
    #         ( None, 'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'MaxAbsScaler' )),	
    #     'loss': hp.choice( 'l', ( 'log', 'modified_huber' )),	# those with predict_proba
    #     'penalty': hp.choice( 'p', ( 'none', 'l1', 'l2', 'elasticnet' )),
    #     'alpha': hp.loguniform( 'a', log( 1e-10 ), log( 1 )),
    #     'l1_ratio': hp.uniform( 'l1r', 0, 1 ),
    #     'fit_intercept': hp.choice( 'i', (True, False )),
    #     'shuffle': hp.choice( 'sh', ( True, False )),
    #     'learning_rate': hp.choice( 'lr', ( 'constant', 'optimal', 'invscaling' )),
    #     'eta0': hp.loguniform( 'eta', log( 1e-10 ), log( 1 )),
    #     'power_t': hp.uniform( 'pt', 0.5, 0.99 ),
    #     'class_weight': hp.choice( 'cw', ( 'balanced', None ))
    # }

    n_params = 100
    n_examples = 100 * len(X_train)  # the number of passes through dataset for best model
    max_iter = n_params  # number of times partial_fit will be called
    chunk_size = n_examples // n_params # number of examples each call sees
    X_train2 = da.from_array(X_train, chunks=chunk_size)
    y_train2 = da.from_array(y_train, chunks=chunk_size)

    print("> Applying Hyperband Search CV ...")
    search = HyperbandSearchCV(est, param_dist, scoring='f1_macro')
    search.fit(X_train2, y_train2, classes=np.unique(y))
    print("> Best hyperparams:", search.best_params_)
    print("> Best score:", search.best_score_)

    # Predict labels
    y_pred = search.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')  
    print(f"> F1 score (test): {f1}")

    # Predict probabilities for ROC AUC computation
    best_params = search.best_params_
    if not best_params['loss'] in ['log_loss', 'modified_huber', ]: 
        print(f"[info] loss fn: {best_params['loss']} may not support predict_proba()")
        
        best_est = SGDClassifier(tol=1e-3) # eta0=0.01
        best_est.set_params(**best_params)
        best_est = CalibratedClassifierCV(best_est, method='sigmoid')
        # Refit the model with the optimal estimator
        best_est.fit(X_train, y_train)

        y_prob = best_est.predict_proba(X_test)[:, 1]  # This method is only available for log loss and modified Huber loss
    else: 
        y_prob = search.predict_proba(X_test)[:, 1]  # This method is only available for log loss and modified Huber loss
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"> ROC AUC (test): {roc_auc}")

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.sort_values(by='test_score', ascending=False, inplace=True)
    print(f"> CV results")
    print(cv_results.head())
    print(f"... columns:\n{list(cv_results.columns)}\n")

    hist = pd.DataFrame(search.history_)
    hist.sort_values(by='score', ascending=False, inplace=True)
    print(f"> history:")
    print(hist.head())

    print("> Feature importance of the best model")
    best_est = search.best_estimator_
    print(best_est.coef_[0])

    return

def demo_xgboost_cv(**kargs): 
    from meta_spliceai.nmd_concept.nmd_targets_analyzer import get_nmd_data
    import time
    from meta_spliceai.mllib.utils import highlight
    from meta_spliceai.system.sys_config import get_data_dir, SequenceIO
    np.set_printoptions(precision=4)

    start_time = time.time()
    SequenceIO.proj_name = 'nmd'

    data_prefix = SequenceIO.data_prefix = "/mnt/SpliceMediator/splice-mediator"
    # Options: SequenceIO.get_data_dir(), 
    #          "/mnt/SpliceMediator/splice-mediator" 
    #          "/mnt/nfs1/splice-mediator"
    print(f"[demo] data prefix:\n{SequenceIO.get_data_dir()}\n")

    framework = kargs.get("framework", "hpbandster")

    X, y, meta_data = \
        get_nmd_data(policy='topn', threshold_eff=0.3, topn=300, 
                        verbose=1, 
                            return_x_as_dataframe=True, return_meta_data=True) # return dataframe so that we can get feature names / columns
    feature_names = list(X.columns)
    X = X.values
    print(f"... shape(X): {X.shape}, shape(y): {y.shape}")
    print(f"... type(X): {type(X)}, type(y): {type(y)}")
    f_encoded = list(set(meta_data['features_encoded']) - set(meta_data['features_original']))
    print(f"... encoded features (n={len(f_encoded)}):\n{f_encoded}\n")

    highlight("> Search for optimal hyperparameter setting for XGB ...")
    if framework == "optuna": 
        nested_cv_xgboost_fn = nested_cv_xgboost_with_importance_via_bayes_opt_optuna
    elif framework.startswith("hpbandster"):
        print(f"... running BOHB")
        nested_cv_xgboost_fn = nested_cv_xgboost_with_importance_via_bohb

    output_dict = nested_cv_xgboost_fn(X, y, feature_names=feature_names)

    # Show detailed classification performance report
    avg_f1 = output_dict['f1']
    avg_roc_auc = output_dict['auc']
    avg_feature_importances = output_dict['feature_importance']

    most_common_hyperparams = output_dict['most_common_hyperparams']
    best_scoring_hyperparams = output_dict['best_scoring_hyperparams']
    # NOTE: If search algorithm searches through real intervals for some hyperparameters (e.g. learning rate), 
    #       chances are that the same hyperparameter setting will never be encountered twice, in which case
    #       it makes more sense to choose the setting that led to highest performance score (rather than choosing
    #       the most common setting across CV iterations)
    best_hyperparams = best_scoring_hyperparams

    # Displaying the SHAP-based feature importances for the diabetes dataset using SVM with RBF kernel
    feature_importance_df = output_dict['feature_importance_df']

    print(f"... F1 score: {avg_f1}")
    print(f"... ROCAUC:   {avg_roc_auc}")
    print(f"... feature importance scores:\n{avg_feature_importances}\n")
    print(f"... most common hyperparams:\n{most_common_hyperparams}\n")
    print(f"... highest scoring hyperparams:\n{best_scoring_hyperparams}\n")

    print("[analyzer] Sorting features by importance scores ...")
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    print(feature_importance_df)


    return

def demo_svm_cv(**kargs): 
    from meta_spliceai.nmd_concept.nmd_targets_analyzer import get_nmd_data
    import time
    from meta_spliceai.utils.utils_sys import highlight
    from meta_spliceai.system.sys_config import get_data_dir, SequenceIO
    from sklearn.datasets import load_breast_cancer
    np.set_printoptions(precision=4)

    start_time = time.time()
    SequenceIO.proj_name = 'nmd'

    data_prefix = SequenceIO.data_prefix = "/mnt/SpliceMediator/splice-mediator"
    # Options: SequenceIO.get_data_dir(), 
    #          "/mnt/SpliceMediator/splice-mediator" 
    #          "/mnt/nfs1/splice-mediator"
    print(f"[demo] data prefix:\n{SequenceIO.get_data_dir()}\n")

    framework = kargs.get("framework", "hpbandster")
    meta_data = {}

    X, y, meta_data = \
        get_nmd_data(policy='topn', threshold_eff=0.3, topn=300, 
                        verbose=1, 
                            return_x_as_dataframe=True, return_meta_data=True) # return dataframe so that we can get feature names / columns
    # X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    feature_names = list(X.columns)
    X = X.values
    print(f"... shape(X): {X.shape}, shape(y): {y.shape}")
    print(f"... type(X): {type(X)}, type(y): {type(y)}")

    if meta_data: 
        f_encoded = list(set(meta_data['features_encoded']) - set(meta_data['features_original']))
        print(f"... encoded features (n={len(f_encoded)}):\n{f_encoded}\n")

    highlight("> Search for optimal hyperparameter setting for XGB ...")
    if framework == "optuna": 
        # nested_cv_xgboost_fn = nested_cv_svm_with_importance_via_bayes_opt_optuna
        raise NotImplementedError
    elif framework.startswith("hpbandster"):
        print(f"... running BOHB")
        nested_cv_xgboost_fn = nested_cv_svm_with_importance_via_bohb

    output_dict = nested_cv_xgboost_fn(X, y, feature_names=feature_names)

    # Show detailed classification performance report
    avg_f1 = output_dict['f1']
    avg_roc_auc = output_dict['auc']
    avg_feature_importances = output_dict['feature_importance']

    most_common_hyperparams = output_dict['most_common_hyperparams']
    best_scoring_hyperparams = output_dict['best_scoring_hyperparams']
    # NOTE: If search algorithm searches through real intervals for some hyperparameters (e.g. learning rate), 
    #       chances are that the same hyperparameter setting will never be encountered twice, in which case
    #       it makes more sense to choose the setting that led to highest performance score (rather than choosing
    #       the most common setting across CV iterations)
    best_hyperparams = best_scoring_hyperparams

    # Displaying the SHAP-based feature importances for the diabetes dataset using SVM with RBF kernel
    feature_importance_df = output_dict['feature_importance_df']

    print(f"... F1 score: {avg_f1}")
    print(f"... ROCAUC:   {avg_roc_auc}")
    print(f"... feature importance scores:\n{avg_feature_importances}\n")
    print(f"... most common hyperparams:\n{most_common_hyperparams}\n")
    print(f"... highest scoring hyperparams:\n{best_scoring_hyperparams}\n")

    print("[analyzer] Sorting features by importance scores ...")
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    print(feature_importance_df)


    return

def demo_sgd_cv(**kargs):
    from meta_spliceai.nmd_concept.nmd_targets_analyzer import get_nmd_data
    import time
    from meta_spliceai.utils.utils_sys import highlight
    from meta_spliceai.system.sys_config import get_data_dir, SequenceIO
    from sklearn.datasets import load_breast_cancer
    np.set_printoptions(precision=4)

    start_time = time.time()

    framework = kargs.get("framework", "hpbandster")
    meta_data = {}

    X, y, meta_data = \
        get_nmd_data(policy='topn', threshold_eff=0.3, topn=300, 
                        verbose=1, 
                            return_x_as_dataframe=True, return_meta_data=True) # return dataframe so that we can get feature names / columns
    # X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    feature_names = list(X.columns)
    X = X.values
    print(f"... shape(X): {X.shape}, shape(y): {y.shape}")
    print(f"... type(X): {type(X)}, type(y): {type(y)}")

    if meta_data: 
        f_encoded = list(set(meta_data['features_encoded']) - set(meta_data['features_original']))
        print(f"... encoded features (n={len(f_encoded)}):\n{f_encoded}\n")

    highlight("> Search for optimal hyperparameter setting for XGB ...")
    if framework == "optuna": 
        # nested_cv_xgboost_fn = nested_cv_svm_with_importance_via_bayes_opt_optuna
        raise NotImplementedError
    elif framework.startswith("hpbandster"):
        print(f"... running BOHB")
        nested_cv_xgboost_fn = nested_cv_sgd_with_importance_via_bohb

    output_dict = nested_cv_xgboost_fn(X, y, feature_names=feature_names)

    # Show detailed classification performance report
    avg_f1 = output_dict['f1']
    avg_roc_auc = output_dict['auc']
    avg_feature_importances = output_dict['feature_importance']

    most_common_hyperparams = output_dict['most_common_hyperparams']
    best_scoring_hyperparams = output_dict['best_scoring_hyperparams']
    # NOTE: If search algorithm searches through real intervals for some hyperparameters (e.g. learning rate), 
    #       chances are that the same hyperparameter setting will never be encountered twice, in which case
    #       it makes more sense to choose the setting that led to highest performance score (rather than choosing
    #       the most common setting across CV iterations)
    best_hyperparams = best_scoring_hyperparams

    # Displaying the SHAP-based feature importances for the diabetes dataset using SVM with RBF kernel
    feature_importance_df = output_dict['feature_importance_df']

    print(f"... F1 score: {avg_f1}")
    print(f"... ROCAUC:   {avg_roc_auc}")
    print(f"... feature importance scores:\n{avg_feature_importances}\n")
    print(f"... most common hyperparams:\n{most_common_hyperparams}\n")
    print(f"... highest scoring hyperparams:\n{best_scoring_hyperparams}\n")

    print("[analyzer] Sorting features by importance scores ...")
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    print(feature_importance_df)

    return 


def test(): 
    import time
    from sklearn.svm import LinearSVC, SVC
    from meta_spliceai.system.sys_config import SequenceIO

    start_time = time.time()

    SequenceIO.proj_name = 'nmd'
    data_prefix = SequenceIO.data_prefix = "/mnt/nfs1/splice-mediator" 
    # Options: SequenceIO.get_data_dir(), 
    #          "/mnt/SpliceMediator/splice-mediator" 
    #          "/mnt/nfs1/splice-mediator"
    print(f"[demo] data prefix:\n{SequenceIO.get_data_dir()}\n")

    # demo_svm_bayes_opt()
    # demo_xgboost_bayes_sv()
    # demo_svm_pipeline_bayes_opt()

    ### Hyperband
    # demo_sgd_hyperband()
    # demo_sgd_hyperband_cv()

    # BOHB: Bayesian Optimization + Hyperband 
    # demo_bohb()

    ### NMD data 
    # demo_xgboost_cv(framework="optuna")  # "hpbandster", "optuna"
    # demo_svm_cv(framework="hpbandster")  # "hpbandster", "optuna"

    demo_sgd_cv(framework="hpbandster")

    # model = SVC()
    # model2 = LinearSVC(max_iter=1000)
    # print(model.__class__.__name__)
    # print(model2.__class__.__name__)

    delta_t = time.time() - start_time
    print(f"[test] Elapsed {delta_t} seconds ...")

if __name__ == "__main__": 
    test()

