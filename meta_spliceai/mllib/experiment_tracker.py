import os, sys
from pathlib import Path

import meta_spliceai.system.sys_config as config
import pandas as pd


class ExperimentTracker(object):

    home_dir = os.path.expanduser("~")
    data_prefix = config.get_proj_dir()
    experiment_root = 'experiments'
    # Options: "<proj_dir>/experiments/"
    #          e.g. f"{home_dir}/work/nmd/experiments" 

    def __init__(self, experiment="nmd_eff", 
                    model_type="descriptor", model_name=None, model_suffix=None):
        self.experiment = experiment
        self.model_type = model_type  # e.g. descriptor, sequence 
        self.model_name = model_name  # e.g. xgboost, transformer
        self.model_suffix = model_suffix
        self.check_datadir()

    def check_datadir(self): 

        # Standardize the root of data prefix
        basename = os.path.basename(ExperimentTracker.data_prefix)
        experiment_root = ExperimentTracker.experiment_root
        if basename != ExperimentTracker.experiment_root: 
            ExperimentTracker.data_prefix = f"{ExperimentTracker.data_prefix}/{experiment_root}"

        # Other rules ...
            
    @property
    def model_id(self): 
        name, suffix = self.model_name, self.model_suffix
        return name if suffix is None else f"{name}-{suffix}" 

    @property
    def experiment_dir(self): 
        parent_dir = os.path.join(ExperimentTracker.data_prefix, self.experiment) 
        # Path(parent_dir).mkdir(parents=True, exist_ok=True)
        expr_dir = os.path.join(parent_dir, self.model_type)

        if self.model_name is not None: 
            expr_dir = os.path.join(expr_dir, self.model_name)
        return expr_dir
    

def meta_data_tracker(data, filepath, id_key):
    """
    Tracks metadata by saving/updating it in a CSV file.

    Parameters:
    - data (dict or pd.Series): The metadata to track.
    - filepath (str): The path to the CSV file.
    - id_key (str): The key in `data` used as the unique identifier.
    """
    # Convert dict to DataFrame if necessary
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    elif isinstance(data, pd.Series):
        data = pd.DataFrame([data])

    # Check if file exists
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)

        # Check if the entry with the same ID exists
        if id_key in df.columns and data[id_key].iloc[0] in df[id_key].values:
            # Update existing row
            df.loc[df[id_key] == data[id_key].iloc[0]] = data.iloc[0]
        else:
            # Concatenate as new row
            df = pd.concat([df, data], ignore_index=True)
    else:
        # Create a new DataFrame and save it
        df = data

    # Save the DataFrame back to CSV
    df.to_csv(filepath, index=False)


def meta_data_tracker_v0(data, file_path, id_key):
    """
    Track metadata in a CSV file. 

    Parameters:
    - data (dict or pd.Series): The metadata to save.
    - file_path (str): Path to the CSV file.
    - id_key (str): The key in the data dict/Series used to identify unique rows.

    Memo: 
    
    As of pandas 2.0, append (previously deprecated) was removed.
    You need to use concat instead (for most applications):
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    """
    # Convert dict to Series if necessary
    if isinstance(data, dict):
        data = pd.Series(data)

    # Check if the file exists
    if os.path.exists(file_path):
        # Read existing data
        df = pd.read_csv(file_path)

        # Check if the row with the same id_key exists
        if id_key in df and data[id_key] in df[id_key].values:
            # Update the existing row
            df.loc[df[id_key] == data[id_key]] = data
        else:
            # Append as a new row
            df = df.append(data, ignore_index=True)
    else:
        # Create a new DataFrame and write it to a file
        df = pd.DataFrame([data])

    # Save the DataFrame to CSV
    df.to_csv(file_path, index=False)