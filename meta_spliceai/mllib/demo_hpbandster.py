# from hpbandster_sklearn import HpBandSterSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


# patch_fit_and_score.py
from sklearn.model_selection._validation import _fit_and_score as original_fit_and_score

def patched_fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params,
                          return_train_score=False, return_parameters=False,
                          return_n_test_samples=False, return_times=False, return_estimator=False,
                          error_score='raise-deprecating'):
    return original_fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, fit_params,
                                  return_train_score=return_train_score, return_parameters=return_parameters,
                                  return_n_test_samples=return_n_test_samples, return_times=return_times,
                                  return_estimator=return_estimator, error_score=error_score)


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
            from meta_spliceai.mllib.compat import _check_method_params
            
            # Monkey patch the sklearn.utils.validation module
            print("[action] Patching sklearn.utils.validation._check_fit_params")
            import sklearn.utils.validation
            sklearn.utils.validation._check_fit_params = _check_method_params
            
            # Try importing HpBandSterSearchCV again after the patch
            from hpbandster_sklearn import HpBandSterSearchCV
            return HpBandSterSearchCV
        else:
            raise e

# Use the wrapper function to import HpBandSterSearchCV
HpBandSterSearchCV = import_hpbandster_search_cv()

# Generate a sample dataset
X_train, y_train = make_classification(n_samples=100, n_features=20, random_state=42)

# Define the model and parameter distributions
model = RandomForestClassifier()

# param_distributions = {
#     'n_estimators': [10, 50, 100],
#     'max_depth': [None, 10, 20, 30]
# }
# NOTE: ConfigSpace library, which is used by hpbandster-sklearn, does not support None as a choice for hyperparameters.

param_distributions = {
    'n_estimators': [10, 50, 100],
    'max_depth': [10, 20, 30]  # Removed None
}

# Define the search object
search = HpBandSterSearchCV(
    model,
    param_distributions,
    scoring='accuracy',
    random_state=0,
    warm_start=False,
    resource_name='n_samples',
    resource_type=float,
    min_budget=0.75,
    max_budget=1,
    n_jobs=5,
    verbose=1
)


# Fit the search object
with parallel_backend('threading'):
    search.fit(X_train, y_train)