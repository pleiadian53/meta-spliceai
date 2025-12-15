import os, sys
# sys.path.append('..')

import json
import re, time, collections
import pandas as pd 
import numpy as np
import pickle

from meta_spliceai.mllib.experiment_tracker import ExperimentTracker

class ModelPerformanceTracker:
    
    def __init__(self, experiment='model_tracking', model_type='descriptor', **kargs):
        # Initialize an empty dictionary to store model performances
        self.performances = {}
        self.results = {}

        # Additional parameters to keep track of the output using ExperimentTracker
        self.exp_tracker = ExperimentTracker(experiment=experiment, 
                                        model_type=model_type,
                                            model_name=kargs.get("model_name", None), 
                                            model_suffix=kargs.get("model_suffix", None))

    @property
    def experiment(self):
        return self.exp_tracker.experiment
    @property
    def experiment_dir(self): 
        return self.exp_tracker.experiment_dir  # self.exp_tracker.experiment will be part of the path
    @property
    def output_dir(self): 
        return self.exp_tracker.experiment_dir  # self.exp_tracker.experiment will be part of the path
    @property
    def model_id(self): 
        return self.exp_tracker.model_id

    @property
    def model_name(self): 
        return self.exp_tracker.model_name
    @property
    def model_suffix(self):
        return self.exp_tracker.model_suffix
    
    def add_performance(self, model_identifier, metrics):
        """
        Add a new model's performance metrics.
        
        Args:
        - model_identifier (dict): A dictionary representing the model's configuration (including experimental settings)
        - metrics (dict): A dictionary with performance metrics, e.g. {'f1': 0.9, 'roc_auc': 0.95}.
        """
        # Convert the model_identifier dictionary to a tuple of key-value pairs for hashing
        model_key = tuple(sorted(model_identifier.items()))
        
        if model_key in self.performances:
            raise ValueError(f"Performance for model configuration {model_identifier} already exists!")
        
        self.performances[model_key] = metrics
    
    def get_performance(self, model_identifier):
        """
        Get the performance metrics for a specific model configuration.
        
        Args:
        - model_identifier (dict): A dictionary representing the model's configuration.
        
        Returns:
        - dict: A dictionary with the model's performance metrics.
        """
        model_key = tuple(sorted(model_identifier.items()))
        
        return self.performances.get(model_key, None)

    def list_all_performances(self):
        """
        List all model configurations and their performance metrics.
        
        Returns:
        - dict: A dictionary with all model configurations and their respective metrics.
        """
        return {tuple(key): value for key, value in self.performances.items()}

    def get_top_configs_v0(self, metric: str, top_n: int = 1):
        # Convert the performance data dictionary into a list of tuples
        sorted_list = sorted(self.performances.items(), key=lambda x: x[1][metric], reverse=True)
        
        # Extract the top configurations based on the desired metric
        top_configs = []
        for i in range(min(top_n, len(sorted_list))):
            config_dict = dict(sorted_list[i][0])
            top_configs.append((config_dict, sorted_list[i][1]))
        
        return top_configs
    
    def get_top_configs_v1(self, metric: str, top_n: int = 1, constraints: dict = None):
        """
        Get top model configurations ranked by a specified metric, subject to given constraints.
        
        Args:
        - metric (str): The performance metric to rank models by.
        - top_n (int): The number of top configurations to return.
        - constraints (dict): A dictionary of constraints that configurations must satisfy.
                            The value in each key-value pair is a tuple: (comparison_operator, value)
                            e.g., {'topn_eff': ('<', 1000)} means configurations must have topn_eff < 1000.
                                {'topn_eff': ('>', 1000)} means configurations must have topn_eff > 1000.
        
        Returns:
        - list of tuples: Each tuple contains a configuration dictionary and its metrics dictionary.
        """
        if constraints is None:
            constraints = {}

        # Function to check if a constraint is satisfied
        def meets_constraint(config_value, operator, constraint_value):
            if operator == '<':
                return config_value < constraint_value
            elif operator == '<=':
                return config_value <= constraint_value
            elif operator == '=':
                return config_value == constraint_value
            elif operator == '>':
                return config_value > constraint_value
            elif operator == '>=':
                return config_value >= constraint_value
            else:
                raise ValueError(f"Unsupported comparison operator: {operator}")

        # Filter configurations based on constraints
        filtered_configs = {}
        for config, metrics in self.performances.items():
            config_dict = dict(config)  # Convert tuple back to dictionary for easier handling
            
            # Check if configuration meets all constraints
            meets_constraints = True
            for key, (operator, constraint_value) in constraints.items():
                if key in config_dict and not meets_constraint(config_dict[key], operator, constraint_value):
                    meets_constraints = False
                    break
            
            if meets_constraints:
                filtered_configs[config] = metrics

        # Sort the filtered configurations by the specified metric
        sorted_list = sorted(filtered_configs.items(), key=lambda x: x[1][metric], reverse=True)
        
        # Extract the top configurations
        top_configs = []
        for i in range(min(top_n, len(sorted_list))):
            config_dict = dict(sorted_list[i][0])
            top_configs.append((config_dict, sorted_list[i][1]))
        
        return top_configs
    
    def get_top_configs(self, metric: str, top_n: int = 1, constraint=None, separate_configs_and_metrics=True):
        """
        Returns the top model configurations ranked by the specified metric, subject to a given constraint.

        Args:
        - metric (str): The performance metric to rank by.
        - top_n (int): Number of top configurations to return.
        - constraint (str, optional): Pandas query string to apply as a constraint.
        
        Returns:
        - DataFrame of top configurations and their metrics.

        Example: Get the top configuration with a constraint
        
                top_config_with_constraint = \
                    tracker.get_top_configs(metric='f1', top_n=1, constraint='topn <= 1500')
        """
        # Convert self.performances to a DataFrame (or use self.results if data available)
        data = []
        for config, metrics in self.performances.items():
            combined = dict(config)  # Convert tuple keys back to a dictionary
            combined.update(metrics)  # Merge with the metrics dictionary
            data.append(combined)
        df = pd.DataFrame(data)
        # print(f"(get_top_configs) dataframe from performance:\n{df.head()}\n")
        # print(f"... columns:\n{list(df.columns)}\n")

        # Apply constraint if provided
        if constraint:
            df = df.query(constraint)

        # Sort by the specified metric in descending order and select the top_n entries
        assert metric in df.columns
        top_configs = df.nlargest(top_n, metric)

        if separate_configs_and_metrics: 
            # Convert the filtered, sorted DataFrame back to the two-dictionary format
            result = []
            for _, row in top_configs.iterrows():
                # Split the row back into model configuration and metrics
                model_config = {key: row[key] for key in row.index if key in self.performances.keys()}
                metrics = {key: row[key] for key in row.index if key not in self.performances.keys()}
                result.append((model_config, metrics))

            return result # list-of-two-dictionary-representation (so that configurations and metrics are separated)
        
        return top_configs # dataframe

    def add_result(self, hyperparameters, performance_metrics):
        # Adding new results by combining both dictionaries
        combined_dict = {**hyperparameters, **performance_metrics}
        self.results.append(combined_dict)

    def to_dataframe(self, verbose=0):
        expanded_dicts = [{**dict(key), **value} for key, value in self.performances.items()]
        if verbose: 
            print(f"[tracker] Found n={len(expanded_dicts)} performance records")
            assert len(expanded_dicts) == len(self.performances)
        return pd.DataFrame(expanded_dicts) # Convert list of dictionaries to DataFrame

    def save_to_csv(self, filepath=None, *, metric='auc', topn=None, verbose=1, sep='\t'):
        # Unpack the tuple keys back to dictionaries
        expanded_dicts = [{**dict(key), **value} for key, value in self.performances.items()]

        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(expanded_dicts)

        # Convert feature importance dataframes to json format
        df = convert_dataframe_columns(df)

        # Sort configurations by the given metric
        df.sort_values(by=metric, ascending=False, ignore_index=True, inplace=True)

        if topn is not None: 
            df = df.iloc[:topn]

        if filepath is None:
            output_dir = self.experiment_dir
            output_file =  f'configs_to_perf_scores-{self.model_id}.csv'
            filepath = os.path.join(output_dir, output_file)

        if verbose: 
            print(f"[performance_tracker] Saving n={df.shape[0]} performance records to:\n{filepath}\n")
        
        # Save DataFrame to CSV
        df.to_csv(filepath, sep=sep, index=False)

    def save(self, filepath=None, **kargs):
        from pathlib import Path
        
        format = kargs.get('format', None)
        metric = kargs.get('metric', 'auc')
        topn = kargs.get('topn', None)
        if filepath is None:
            output_dir = self.experiment_dir
            output_file =  f'configs_to_perf_scores-{self.model_id}'
            filepath = os.path.join(output_dir, output_file)
            print(f"[test] output_dir:\n{output_dir}\n")
            print(f"[test] filepath:\n{filepath}\n")

        df = self.to_dataframe() # Convert performance dictionary to dataframe for data analytics
        assert df.shape[0] == len(self.performances)

        # Sort configurations by the given metric
        df.sort_values(by=metric, ascending=False, ignore_index=True, inplace=True)

        if topn is not None: 
            df = df.iloc[:topn]

        if format is not None: 
            assert len(format) > 0
            filepath = check_and_correct_file_extension(filepath, format)

        create_dir_if_not_exist = kargs.get('create_dir', True)
        parent_dir = os.path.dirname(filepath)
        if not os.path.exists(parent_dir) and create_dir_if_not_exist:
            print(f"[I/O] Directory does not exist yet at:\n{parent_dir}\n")
            Path(parent_dir).mkdir(parents=True, exist_ok=True) 

        print(f"[save] Saving performance data to:\n{filepath}\n")
        if format.lower() == 'json': 
            df.to_json(filepath)
        elif format.lower() == 'csv':
            sep = kargs.get("sep", '\t')

            # Convert feature importance dataframes to json format
            df = convert_dataframe_columns(df)
            df.to_csv(filepath, sep=sep, index=False)
        elif format in ('pkl', 'pickle'):
            print(f"[output] Saving performance dictionary in pickle format at:\n{filepath}\n")
            # from meta_spliceai.utils.utils_data import save_pickle
            with open(filepath, 'wb') as handle:
                pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else: 
            raise NotImplementedError
        
        # Test: save performance dictionary
        output_dir = os.path.dirname(filepath)
        output_path = os.path.join(output_dir, "performance_dict.pickle")
        print(f"[test] Saving performance dictionary (size={len(self.performances)}) to:\n{output_path}\n")
        with open(output_path, 'wb') as handle:
            pickle.dump(self.performances, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return filepath

    def load_with_constraints(self, constraints=None, relaxed=True, **kargs):
        # filepath = kargs.get('file_path', None)
        # raise_exception = kargs.get("raise_excpetion", True)
        # sep = kargs.get("sep", "\t")
        # rank = kargs.get("rank", True)
        # format = kargs.get("format", None)
        metric= kargs.get("metric", "auc")
        threshold = kargs.pop("threshold", None) # performance threshold
        maximize_topn = kargs.pop("maximize_topn", True)
        verbose = kargs.get("verbose", 1)
        
        df_perf = ModelPerformanceTracker.load(self, **kargs)
        shape0 = df_perf.shape
        # NOTE: Don't use self.load(**kargs) because the load() method will also be defined in the subclass
        #       If a subclass instance (ModelTracker) calls .load() through self.load() (self refers to ModelTracker), 
        #       then it will use ModelTracker's version of load() method, which is not what we want
        df_perf0 = df_perf.copy()
        
        if constraints: # e.g. topn <= topn_eff
            df_perf = df_perf.query(constraints).copy()
            # NOTE: .query() typically creates a new copy of the data, it's possible that under certain 
            #       scenarios, it might end up returning a view into the original DataFrame. Hence, call .copy() to 
            #       avoid Pandas thinking it's working with a view.

        # Rank the configurations by the given metric
        performance_ranked = False
        if threshold:
            df_perf = df_perf[df_perf[metric] >= threshold] 

            if maximize_topn:
                # Sort the filtered DataFrame first by 'topn' in descending order and then by the metric in descending order 
                df_perf.sort_values(by=['topn', metric], ascending=[False, False], ignore_index=True, inplace=True)
            else: 
                df_perf.sort_values(by=metric, ascending=False, ignore_index=True, inplace=True)
        else:
            df_perf.sort_values(by=metric, ascending=False, ignore_index=True, inplace=True)
            
        print(f"(load_with_constraints) Constraints={constraints}")
        print(f"> columns:\n{list(df_perf.columns)}\n")

        if verbose: 
            # print(f"(load_with_constraints) constraints={constraints}")
            # print(f"... shape(df_perf): {df_perf.shape} -> {df_perf_filtered.shape}")

            # --- Test ---
            if df_perf.shape == shape0: 
                if constraints: 
                    print("[info] No configurations were filtered out by the constraints.")
            else: 
                print(f"[info] Filtered {shape0[0] - df_perf.shape[0]} configurations by constraints.")
                print(f"[info] Constraints: {constraints}")
            
            columns = ['threshold_eff', 'topn', 'topn_eff', 'f1', 'auc']
            print(df_perf[columns].head(25))
            print()
            # NOTE: Example chart 
            #       threshold_eff  topn  topn_eff        f1       auc
            #   0             0.2   648       759  0.949776  0.991247
            #   1             0.1   648       759  0.948159  0.991091
            #   2             0.8   648       945  0.891834  0.958380
            #   3             0.9   648       958  0.860040  0.944213
            # NOTE: topn_eff is the number of NMD-fated transcripts with efficiency score >= threshold (e.g. 0)   

            
        if df_perf.empty: 
            print(f"[info] No configurations found that satisfy the constraints: {constraints}")
            if relaxed: 
                # If no qualified configurations exist, return all configurations for now
                # Returning the rows with closest configurations would be nice [todo]
                return df_perf0
                
        return df_perf

    def load(self, filepath=None, *, sep='\t', format=None, rank=True, metric='auc', raise_exception=True, verbose=0): 
        
        if filepath is None:
            output_dir = self.experiment_dir
            output_file =  f'configs_to_perf_scores-{self.model_id}'
            filepath = os.path.join(output_dir, output_file)

        if format is not None: 
            assert len(format) > 0
            filepath = check_and_correct_file_extension(filepath, format)

        if verbose: 
            print(f"[tracker] Loading performance dataframe with format={format} from:\n{filepath}\n")
   
        df = None
        if format.lower() == 'csv': 
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, sep=sep, header=0) 
            else: 
                msg = f"(ModelPerformanceTracker.load) Performance tracker does not exist at:\n{filepath}\n"
                # if raise_exception: 
                #     raise FileNotFoundError(msg)
                # else:
                #     print(msg)
                raise FileNotFoundError(msg)

            df = convert_json_columns_to_df(df)

        elif format.lower() == 'json': 
            # Load DataFrame from JSON file
            df = pd.read_json(filepath)  
        elif format in ('pkl', 'pickle', ):
            with open(filepath, 'rb') as handle:
                df = pickle.load(handle) 

        # Sort configurations by the given metric
        if rank: 
            df.sort_values(by=metric, ascending=False, ignore_index=True, inplace=True)

        # Todo: Convert dataframe back to dictionary 
              
        return df

    @staticmethod 
    def save_to_csv2(performances, filepath, metric='auc', topn=None, verbose=1):
        # Unpack the tuple keys back to dictionaries
        expanded_dicts = [{**dict(key), **value} for key, value in performances.items()]

        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(expanded_dicts)

        # Convert feature importance dataframes to json format
        df = convert_dataframe_columns(df)

        # Sort configurations by the given metric
        df.sort_values(by=metric, ascending=False, ignore_index=True, inplace=True)

        if topn is not None: 
            df = df.iloc[:topn]

        if verbose: 
            print(f"[performance_tracker] Saving n={df.shape[0]} performance records to:\n{filepath}\n")
        
        # Save DataFrame to CSV
        df.to_csv(filepath, sep='\t', index=False)


def convert_dataframe_columns(df, **kargs):

    format = kargs.get('format', 'json')
    cols = kargs.get('target_cols', ['feature', 'importance'])

    if format == 'json': 
        for col in df.columns:
            # Check for non-null values in the column
            non_null_values = df[col].dropna()

            # If the column has any non-null values and the first value is a DataFrame
            if len(non_null_values) > 0 and isinstance(non_null_values.iloc[0], pd.DataFrame):
                df[col] = df[col].apply(lambda x: x.to_json() if isinstance(x, pd.DataFrame) else x)
    elif format.startswith('dict'):

        for col in df.columns: 
            # Check for non-null values in the column
            non_null_values = df[col].dropna()

            # If the column has any non-null values and the first value is a DataFrame
            if len(non_null_values) > 0 and isinstance(non_null_values.iloc[0], pd.DataFrame):
                df[col] = df[col].apply(lambda x: x.set_index(cols[0])[cols[1]].to_dict() if isinstance(x, pd.DataFrame) else x)

    return df

def convert_json_columns_to_df(df):

    for col in df.columns:
        # Extract non-null values from the column
        non_null_values = df[col].dropna()

        if len(non_null_values) == 0:
            continue

        if isinstance(non_null_values.iloc[0], str):

            # Check if the first non-null value in the column is a valid JSON string
            try:
                # print(non_null_values.iloc[0])
                parsed = json.loads(non_null_values.iloc[0])
                
                # Check if the parsed JSON can be converted to a DataFrame
                if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                    df[col] = df[col].apply(lambda x: pd.read_json(x) if isinstance(x, str) else x)
            except json.JSONDecodeError:
                # If it's not a valid JSON, just continue to the next column
                continue

    return df

def check_and_correct_file_extension(file_path, desired_extension):
    """
    Check if the file path has the correct extension, if not, update it to the desired extension.
    
    Parameters:
        file_path (str): The path to the file.
        desired_extension (str): The desired file extension (e.g., 'json').
        
    Returns:
        str: The corrected file path.
    """
    # Extract the current file extension
    current_extension = os.path.splitext(file_path)[1][1:]
    
    # Check if the current extension matches the desired extension
    if current_extension != desired_extension:
        corrected_file_path = os.path.splitext(file_path)[0] + '.' + desired_extension
        return corrected_file_path
    
    return file_path

def demo_save_and_load_to_csv(): 
    # from meta_spliceai.utils.utils_df import dataframes_equal
    # from .utils import dataframes_equal
    from meta_spliceai.mllib.utils import dataframes_equal  

    # Create a main DataFrame
    df_main = pd.DataFrame({
        'model': ['Model1', 'Model2'],
        'accuracy': [0.9, 0.85]
    })

    # Create DataFrames to nest within the main DataFrame
    df1 = pd.DataFrame({
        'feature': ['a', 'b'],
        'importance': [0.1, 0.2]
    })

    df2 = pd.DataFrame({
        'feature': ['c', 'd'],
        'importance': [0.3, 0.4]
    })

    # Add dataframes to the main DataFrame
    df_main['top_features'] = [df1, df2]

    df_main = convert_dataframe_columns(df_main)  # nested dataframes -> json
    print(df_main)
    
    df_main.to_csv('model_performance.csv', sep='\t', index=False)

    # Read from CSV
    df_read = pd.read_csv('model_performance.csv', sep='\t', header=0)
    print(df_read)

    print(dataframes_equal(df_main, df_read))

    return

def demo(): 
    tracker = ModelPerformanceTracker()

    # Add some performance metrics for specific model configurations
    tracker.add_performance({'C': 0.1, 'penalty': 'l2'}, {'f1': 0.9, 'roc_auc': 0.95, 'mcc': 0.6})
    tracker.add_performance({'C': 1, 'penalty': 'l1'}, {'f1': 0.85, 'roc_auc': 0.9, 'mcc': 0.55})

    # Retrieve performance metrics for a specific configuration
    print(tracker.get_performance({'C': 0.1, 'penalty': 'l2'}))

    # List all performances
    print(tracker.list_all_performances())

def demo_top_performance(): 
    # Demo
    tracker = ModelPerformanceTracker()

    # Sample data
    configs = [{"param1": 1, "param2": "A"}, {"param1": 2, "param2": "B"}, {"param1": 3, "param2": "A"}]
    performances = [{"F1": 0.9, "ROCAUC": 0.88, "MCC": 0.6}, {"F1": 0.92, "ROCAUC": 0.89, "MCC": 0.62}, {"F1": 0.88, "ROCAUC": 0.85, "MCC": 0.59}]

    # Adding data to tracker
    for config, perf in zip(configs, performances):
        tracker.add_performance(config, perf)

    # Retrieving top 2 configurations based on F1 score
    metric = "F1"
    top_configs = tracker.get_top_configs(metric, 1)
    print(f"> Retrieving top 2 configurations based on {metric} score:\n{top_configs}\n")

    return

def test_misc(): 
    from constants import col_tid as target
    print(target)

    from constants import SequenceMarkers
    print(SequenceMarkers.markers)


def test(): 
    # demo()
    # demo_top_performance()

    # test_misc()
    # demo_save_and_load_to_csv()

    file_path = "prefix/x/y/file_name.json"
    format = "pickle"
    file_path = check_and_correct_file_extension(file_path, format)
    print(f"> Correct path:\n{file_path}\n")

if __name__ == "__main__":
    test()
