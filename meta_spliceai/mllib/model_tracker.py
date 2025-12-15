
import os, sys
# sys.path.append('..')

import pickle, json

from joblib import dump, load
import pandas as pd
import numpy as np
# import pickle

# from mllib.experiment_tracker import ExperimentTracker
from meta_spliceai.mllib.performance_tracker import ModelPerformanceTracker

from typing import List, Optional

class ModelTracker(ModelPerformanceTracker):

    model_dir_name = "model"
    meta_file_format = "pkl"
    
    def __init__(self, experiment='model_tracking', 
                        model_name='xgboost', 
                        model_suffix=None,  **kargs): # 
        super().__init__(experiment=experiment, model_name=model_name, model_suffix=model_suffix,  **kargs) 

        self.model_params = kargs.get("model_params", {})
        self.use_joblib_io = kargs.get("use_joblib_io", True)

        # Training parameters
        self.n_folds = kargs.get("n_folds", 5)
        self.use_nested_cv = kargs.get("use_nested_cv", False)
      
        # self.perf_tracker = ModelPerformanceTracker(experiment=experiment,  # default to "model_tracker"
        #                         model_name=model_name, model_suffix=model_suffix)
        # Additional parameters to keep track of the output using ExperimentTracker
        # self.exp_tracker = ExperimentTracker(experiment=experiment, 
        #                                 model_type=model_type,
        #                                     model_name=kargs.get("model_name", None), 
        #                                     model_suffix=kargs.get("model_suffix", None))

        # experiment = self.exp_tracker.experiment
        # model_name = self.exp_tracker.model_name
        # model_suffix = self.exp_tracker.model_suffix
        # model_name_extended = f"{experiment}-{self.model_id}"

    @property 
    def model_dir(self): 
        return os.path.join(self.experiment_dir, ModelTracker.model_dir_name)
    @property
    def output_dir(self): 
        return os.path.join(self.experiment_dir, ModelTracker.model_dir_name)
    
    # @property
    # def experiment(self):
    #     return self.exp_tracker.experiment
    # @property
    # def experiment_dir(self):
    #     return self.exp_tracker.experiment_dir
    
    @property 
    def model_id(self):
        return self.model_name_parameterized

    @property
    def model_name_parameterized(self):
        base_name = super().model_id
        model_params = {} if self.model_params is None else self.model_params

        if isinstance(model_params, dict):
            for key, value in model_params.items():
                base_name += f"-{key}{value}"   
        else: # list of tuples
            for key, value in model_params: 
                base_name += f"-{key}{value}"
        return base_name
    
    def parse_model_id(self, model_id):
        import re

        # Get the base name
        base_name = super().model_id

        # Remove the base name from the model_id
        param_string = model_id[len(base_name)+1:]

        # Split the remaining string by '-'
        params = param_string.split('-')

        # Create a dictionary of the parameters
        model_params = {}
        for param in params:
            # Each parameter is in the format 'keyvalue'
            # Use regex to split it into a key and a value
            match = re.match(r'([a-zA-Z]+)([0-9a-zA-Z.]+)', param)
            if match:
                key, value = match.groups()
                # Add the key and value to the dictionary
                model_params[key] = value

        return base_name, model_params
    
    def name_model_file(self, format='joblib'):  # 'json' for xgb model
        return f"{self.model_name_parameterized}.{format}"

    def name_predicion_file(self, format='parquet', prefix='prediction'):  # 'json' for xgb model
        return f"{prefix}-{self.model_name_parameterized}.{format}"

    def save(self, model, save_fn=None, verbose=1):
        if save_fn is None: 
            save_fn = save_sklearn_model_given_path 
        model_dir = self.model_dir
        output_path = os.path.join(model_dir, self.name_model_file())
        save_fn(model, output_path=output_path, verbose=verbose)

        return output_path
    def save_model(self, model, save_fn=None, verbose=1):
        return self.save(model, save_fn, verbose)
        
    def load(self, load_fn=None, verbose=1, return_path=False): 
        if load_fn is None: load_fn = load_sklearn_model_given_path
        model_dir = self.model_dir
        input_path = os.path.join(model_dir, self.name_model_file())

        model = load_fn(input_path, verbose=verbose)

        if return_path: 
            return model, input_path
        return model
    def load_model(self, load_fn=None, verbose=1, return_path=False):
        return self.load(load_fn, verbose, return_path)
    
    def save_model_performance(self, filepath=None, **kargs):
        return super().save(filepath, **kargs)
    def load_model_performance(self, **kargs):
        return super().load(**kargs)

    def save_predictions(self, model, X, id_key='tx_id', id_value=None, format='parquet', **kargs):

        verbose = kargs.get("verbose", 1)

        # Get the label predictions
        y_pred = model.predict(X)

        # Check if the model has the predict_proba method
        if hasattr(model, 'predict_proba'):
            # Get the probability predictions for the positive class
            y_prob = model.predict_proba(X)[:, 1]
        else:
            # If the model doesn't have the predict_proba method, set y_prob to None
            y_prob = None

        # If id_value is not provided, use X.index as the default id_value
        if id_value is None: id_value = X.index

        # Create a DataFrame with the predictions
        df_pred = pd.DataFrame({
            id_key: id_value,
            'y_pred': y_pred,
            'y_prob': y_prob
        })

        try:
            # Save the model output/prediction
            output_path = os.path.join(self.model_dir, self.name_predicion_file(format=format))
            
            if format in ('csv', 'tsv', 'pandas', ):
                sep = kargs.get("sep", '\t' if format == 'tsv' else ',') 
                df_pred.to_csv(output_path, index=False, sep=sep)
            elif format == 'parquet':
                df_pred.to_parquet(output_path)
            else:
                raise ValueError(f"Unsupported file format: {format}")

                return output_path
        except Exception as e:
            print(f"Error saving predictions: {e}")
            return None

    def load_predictions(self, format='parquet', **kargs):
        try:
            # Load the model output/prediction
            output_path = os.path.join(self.model_dir, self.name_predicion_file(format=format))
            
            if format in ('csv', 'tsv', 'pandas', ):
                sep = kargs.get("sep", '\t' if format == 'tsv' else ',') 
                df_pred = pd.read_csv(output_path, sep=sep)
            elif format == 'parquet':
                df_pred = pd.read_parquet(output_path)
            else:
                raise ValueError(f"Unsupported file format: {format}")

            return df_pred
        except Exception as e:
            print(f"Error loading predictions: {e}")
            return None
    
    def save_metadata(self, states, format=None, verbose=0):
        # import json
        # import pickle
        if format is None: format = ModelTracker.meta_file_format

        try:
            # Save the metadata
            metadata_path = os.path.join(self.model_dir, self.name_model_file(format=format))
            
            if format == 'json':
                with open(metadata_path, 'w') as f:
                    json.dump(states, f, indent=4)
            elif format in ('pkl', 'pickle', ):
                with open(metadata_path, 'wb') as f:
                    pickle.dump(states, f)
            else:
                raise ValueError(f"Unsupported file format: {format}")

            return metadata_path
        except Exception as e:
            print(f"Error saving metadata: {e}")
            return None

    def load_metadata(self, format=None, verbose=0):
        # import json
        # import pickle
        if format is None: format = ModelTracker.meta_file_format

        try:
            # Load the metadata
            metadata_path = os.path.join(self.model_dir, self.name_model_file(format=format))
            
            if format == 'json':
                with open(metadata_path, 'r') as f:
                    states = json.load(f)
            elif format in ('pkl', 'pickle', ):
                with open(metadata_path, 'rb') as f:
                    states = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file format: {format}")

            return states
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return None
        
def save_sklearn_model_given_path(model, output_path, verbose=1):
    from joblib import dump

    model_dir = os.path.dirname(output_path)

    # Ensure the save directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Save the model
    # model_path = output_path
    dump(model, output_path)

    if verbose: 
        print(f"[save] Model saved to:\n{output_path}\n")

    return

def load_sklearn_model_given_path(model_path, verbose=1):
    """

    Memo
    ----
    1. If you are loading a serialized model (like pickle in Python, RDS in R) generated by
        older XGBoost, please export the model by calling `Booster.save_model` from that version
        first, then load it back in current version. See:

        https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html

        for more details about differences between saving model and serializing.
    """

    from joblib import load

    # Load the scikit-learn model from disk
    loaded_model = load(model_path)

    if verbose: 
        print(f"[load] Model read from:\n{model_path}\n")

    return loaded_model
# Alias
load_sklearn_model = load_sklearn_model_given_path

def save_xgboost_model_given_path(model, output_path, verbose=1):
    from joblib import load

    model_dir = os.path.dirname(output_path)
    os.makedirs(model_dir, exist_ok=True)

    model.save_model(output_path)

    if verbose:
        print(f"[save] Model saved to:\n{output_path}\n")
    return

def load_xgboost_model_given_path(model_path, verbose=1):
    import xgboost as xgb

    # Create a new XGBoost model object
    loaded_model = xgb.XGBClassifier()
    # Load the saved model
    loaded_model.load_model(model_path)

    if verbose: 
        print(f"[load] Model read from:\n{model_path}\n")
    return loaded_model

def save_sklearn_model(model, save_dir, params, format='joblib', **kwargs):
    from joblib import dump

    filename = kwargs.get("base_name", "model")
    verbose = kwargs.get("verbose", 1)

    for key, value in params.items():
        filename += f"_{key}{value}"
    filename += f".{format}"

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, filename)
    dump(model, model_path)

    if verbose:
        print(f"Model saved to {model_path}")

    return model_path

def save_xgboost_model(model, save_dir, params, format='json', **kwargs):
    from joblib import load

    filename = kwargs.get("base_name", "xgboost")
    verbose = kwargs.get("verbose", 1)

    for key, value in params.items():
        filename += f"_{key}{value}"
    filename += f".{format}"

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, filename)
    model.save_model(model_path)

    if verbose:
        print(f"Model saved to {model_path}")

    return model_path
# Alias 
load_xgboost_model = load_xgboost_model_given_path

def save_model(model, save_dir, model_name):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the model
    save_path = os.path.join(save_dir, f"{model_name}.joblib")
    dump(model, save_path)
    print(f"Model saved to {save_path}")

    return save_path

def load_model(model_path):
    # Load the model from disk
    loaded_model = load(model_path)
    return loaded_model

# Uility Functions
########################################################

def dict_diff(d1, d2, path=""):
    """
    Compare two dictionaries (mainly used for comparing two metadata sets).

    This function is a utility for the ModelTracker in connection to load_metadata() and save_metadata() methods.
    It recursively compares the keys and values of two dictionaries and prints the differences.
    If the values are tuples, lists, or numpy arrays, it converts them to lists before comparing.

    Parameters:
    d1 (dict): The first dictionary to compare.
    d2 (dict): The second dictionary to compare.
    path (str): The current path in the dictionary (used for recursive calls).

    Returns:
    None
    """
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        raise TypeError(f"Both inputs must be dictionaries, but got {type(d1)} and {type(d2)}")

    n_diff = 0
    for k in d1.keys():
        if k not in d2:
            print(f"{path}: Key {k} not in second dictionary")
            n_diff += 1
        elif isinstance(d1[k], dict) and isinstance(d2[k], dict):
            n_diff += dict_diff(d1[k], d2[k], path = path + f".{k}")
        else:
            v1 = list(d1[k]) if isinstance(d1[k], (tuple, list, np.ndarray)) else d1[k]
            v2 = list(d2[k]) if isinstance(d2[k], (tuple, list, np.ndarray)) else d2[k]
            if v1 != v2:
                print(f"{path}: For key {k}, value {v1} in first dictionary is not equal to value {v2} in second dictionary")
                n_diff += 1

    for k in d2.keys():
        if k not in d1:
            print(f"{path}: Key {k} not in first dictionary")
            n_diff += 1
    return n_diff

def show_model_info(model: object, user_params: Optional[List[str]] = None) -> str:
    """
    Returns a string representation of a model with selected parameters.

    Parameters:
    model (object): The model to be displayed. This should be an instance of a classifier.
    user_params (list, optional): A list of parameter names to be displayed. 
                                  If not provided or empty, a default set of parameters will be used 
                                  based on the class of the model. 
                                  The best datatype for this parameter is a list to ensure consistency.

    Returns:
    str: A string representation of the model with the selected parameters.

    Raises:
    TypeError: If `user_params` is provided but is not a sequence (i.e., not a list, tuple, etc.).
    """

    # Check if user_params is a sequence if it is provided
    if user_params is not None and not isinstance(user_params, (list, tuple)):
        raise TypeError("`user_params` must be a sequence (list, tuple, etc.)")

    # Define the default parameters you're interested in for each classifier
    default_params_dict = {
        'XGBClassifier': ['n_estimators', 'max_depth'],
        'SVC': ['C', 'kernel'],
        'LogisticRegression': ['C']
    }

    # Get the class name of the model
    model_info = f"{model.__class__.__name__}"

    # Use the user-specified parameters if provided, otherwise use the default parameters
    params = user_params if user_params else default_params_dict.get(model.__class__.__name__, [])

    # Add the values of the parameters to the string
    for param in params:
        if hasattr(model, param):
            model_info += f", {param}={getattr(model, param)}"

    return model_info

# Function to generate example dataset
def generate_example_data():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Define the test function
def test_model_persistence():
    from sklearn.ensemble import RandomForestClassifier

    # Step 1: Instantiate a model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    # Step 2: Generate example dataset
    X_train, X_test, y_train, y_test = generate_example_data()

    # Step 3: Fit the model
    model.fit(X_train, y_train)

    # Step 4: Saving model
    mt = ModelTracker(experiment='model_tracking', model_name='rf')
    mt.model_params = {'n_estimators': 100, 'max_depth': 10}

    # save_dir = 'saved_models'
    # saved_model_path = save_sklearn_model(model, save_dir, params=mt.model_params)
    model_path = mt.save(model, verbose=1)

    # Assert that the model file exists
    assert os.path.exists(model_path)

    # Step 5: Loading model and ensuring it continues to work as expected
    # loaded_model = load_sklearn_model(model_path)
    loaded_model = mt.load(verbose=1)

    # Assert that the loaded model is an instance of RandomForestClassifier
    assert isinstance(loaded_model, RandomForestClassifier)

    # Assert that the loaded model predicts with the same accuracy as the original model
    y_pred_original = model.predict(X_test)
    y_pred_loaded = loaded_model.predict(X_test)
    assert (y_pred_original == y_pred_loaded).all()

    print("[info] All tests passed successfully.")

    return

# Define the test function
def test_model_persistence_xgboost(use_sklearn=False):
    import xgboost as xgb

    # Step 1: Instantiate a model
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)

    # Step 2: Generate example dataset
    X_train, X_test, y_train, y_test = generate_example_data()

    # Step 3: Fit the model
    model.fit(X_train, y_train)

    # Step 4: Saving model
    params = {'n_estimators': 100, 'max_depth': 3}
    mt = ModelTracker(experiment='model_tracking', model_name='xgboost')
    mt.model_params = params

    # saved_model_path = save_xgboost_model(model, save_dir, params)
    save_fn = save_sklearn_model_given_path if use_sklearn else save_xgboost_model_given_path 
    model_path = mt.save(model, save_fn=save_fn, verbose=1)

    # Assert that the model file exists
    assert os.path.exists(model_path)

    # Step 5: Loading model and ensuring it continues to work as expected
    load_fn = load_sklearn_model_given_path if use_sklearn else load_xgboost_model_given_path
    
    # loaded_model = load_xgboost_model(saved_model_path)
    loaded_model = mt.load(load_fn=load_fn, verbose=1)

    # Assert that the loaded model is an instance of XGBClassifier
    assert isinstance(loaded_model, xgb.XGBClassifier)

    # Assert that the loaded model predicts with the same accuracy as the original model
    y_pred_original = model.predict(X_test)
    y_pred_loaded = loaded_model.predict(X_test)
    assert (y_pred_original == y_pred_loaded).all()

    print("All tests passed successfully.")

def test_load_and_save_pretrained_model(**kargs):
    from meta_spliceai.system.model_config import BiotypeModel

    model_name = kargs.get("model_name", BiotypeModel.model_name) # the ML algorithm used as the classifier (e.g. XGBoost)
    model_suffix = kargs.get("model_suffix", BiotypeModel.model_suffix) # Supplementary model identifier 
    
    experiment = BiotypeModel.model_output_dir
    model_type = BiotypeModel.model_type

    print(f"Experiment: {experiment}")
    print(f"Model type: {model_type}")
    print(f"Model name: {model_name}")
    print(f"Model suffix: {model_suffix}")
    
    tracker = \
        ModelTracker(experiment=experiment, 
                        model_type=model_type, 
                            model_name=model_name, model_suffix=model_suffix) 
                                    

    model_path = tracker.model_dir 

    print("Model path: ", model_path)

    model = None 
    try:                               
        print("[I/O] Loading pre-trained biotype classifier ...")
        model = tracker.load(verbose=1) # Load the pre-trained model
        print(f"Model: {model}")
        print(f"Model parameters: {model.get_params()}")
    except FileNotFoundError as e: 
        if not overwrite:
            print(e)
        # Train the model from scratch
        print("(load_or_train_model) Training a new instance of biotype classifier ...")

        # Do model training ...

    return


def test(): 

    # Model persistence for general sklearn-produced models
    # test_model_persistence()

    # Model persistence for xgboost 
    # test_model_persistence_xgboost(use_sklearn=True)

    test_load_and_save_pretrained_model()

    return

if __name__ == "__main__":
    test() 


