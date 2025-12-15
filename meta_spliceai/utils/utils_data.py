import os, sys
import csv
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from collections.abc import Sequence

# Enable importing files from parent dir
# getting the name of the directory where the this file is present.
current_dir = Path.cwd() # os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parent_dir = os.path.dirname(current_dir)
# adding the parent directory to the sys.path.
sys.path.append(parent_dir) 

def is_numeric(input_var):
    if isinstance(input_var, (int, float, np.number)):
        return True
    elif isinstance(input_var, str) and input_var.isnumeric():
        return True
    elif isinstance(input_var, str) and input_var.replace(".", "", 1).isnumeric():
        return True
    return False

def get_label_counts(df, col_label='label', dtype=None, verbose=0): 
    from tabulate import tabulate
    table = tabulate(df[col_label].value_counts().to_frame(), headers='keys', tablefmt='psql')
    if verbose: 
        if dtype is not None: 
            print(f"> In {dtype} set, label counts:\n{table}\n")
        else: 
            print(f"> Label counts:\n{table}\n")
    return table

def process_path(path=None, file_name=None, ext='.csv', data_dir='data', f_basename='test', makedir=True, verbose=0): 
    
    if not ext.startswith('.'): ext = '.' + ext

    # If a full path to the data file is not given, then use `data_dir` as the default data directory
    prefix = os.path.join(os.getcwd(), data_dir)
    if path: 
        # `path` can be either a directory or a full path (including the file)
        if os.path.isdir(path): 
            prefix = path # overwrite default prefix
            if not file_name: 
                raise ValueError("Input directory given but missing file name.")
            # path = os.path.join(path, file_name)
        else: # a full path is given
            prefix, file_name = os.path.dirname(path), os.path.basename(path)
    else: 
        # prefix = os.path.join(os.getcwd(), data_dir)  # use the default data directory
        if not file_name: 
            raise ValueError("Missing file name.")
    
    # if file name is given, include the file extension
    if ext and file_name.find(ext) < 0: # [note] it's possilbe that extension is embedded in the file name, which breaks this logic
        file_name = file_name + ext
        
    path = os.path.join(prefix, file_name)

    if makedir: Path(prefix).mkdir(parents=True, exist_ok=True)
    if verbose: print(f"(process_path) prefix:\n{prefix}\nfull path:\n{path}\n")

    return path

def is_sequence(x):
    # from collections.abc import Sequence
    if isinstance(x, str): 
        return False 
    if isinstance(x, np.ndarray): 
        return x.ndim > 0
    return isinstance(x, Sequence)

def is_simple_dictionary(adict):
    if not isinstance(adict, dict): return False 

    # tval = True # not a simple dictionary where values are non-sequence data type (e.g. int, float)
    # NOTE: a string is normally conidered as sequence, but here, it's not 
    n_non_sequence_detected = 0
    lengths = []
    for k, v in adict.items(): 
        if is_sequence(v): 
            lengths.append(len(v))
        else: 
            n_non_sequence_detected +=1
    
    n_uniq_length = len(set(lengths))
    if n_non_sequence_detected > 0 or n_uniq_length != 1: 
        return True
    return False

def order_class_names(label_names):
    import numbers

    # Canonicalize `label_names` to a dictionary where the keys the class names and the values are 
    # their integer encodings
    standard_class_map = label_names
    
    keys = list(label_names.keys())
    if isinstance(keys[0], numbers.Number): 
        standard_class_map = {name: code for code, name in label_names.items()}
    else: 
        assert isinstance(keys[0], str)
 
    ordered_names = [name for name, code in sorted(standard_class_map.items(), key=lambda x: x[1])]
        
    return ordered_names

def encode_labels(df, col_label='label', label_names={}, mode="to_int", verbose=0, return_labels_only=False): 
    import numbers
    import collections

    # maps label names to integers
    if not label_names: 
        # For convenience of the NMD project, assign a default labeling map 
        label_names = {0: 'nmd_ineff', 1: 'nmd_eff'}  # Todo: configuration

    if isinstance(label_names, (list, tuple, np.ndarray)): 
        # assuming that the elements in label_names are ordered 
        label_names = {i: c for i, c in enumerate(label_names)}

    if verbose: 
        print("> Before label encoding, label counts:") 
        print(get_label_counts(df))

    # NOTE: NO need to use groupby
    # for r, dfg in df.groupby(col_label): 
    #   ...
    unique_labels = df[col_label].unique()
    labels = df[col_label].values
    if mode == "to_int": # convert label names to integers
        if isinstance(unique_labels[0], str): 
            assert set(unique_labels) == set(label_names.values())

            if return_labels_only: 
                label_codes = {name: code for code, name in label_names.items()}
                labels = np.array([label_codes[c] for c in df[col_label].values])
            else: 
                for class_code, class_name in label_names.items(): 
                    df.loc[df[col_label]==class_name, col_label] = class_code
        else: 
            # otherwise, the labels must have been encoded, in which case the work is done
            assert isinstance(unique_labels[0], numbers.Number), f"unusual label dtype: {type(unique_labels[0])}"
    else: 
        # print(f"[debug] type(unique_labels[0]): {type(unique_labels[0])}") # numpy.int64
        if isinstance(unique_labels[0], numbers.Number):  # convert label codes to their names
            
            if return_labels_only: 
                labels = np.array([label_names[int(c)] for c in df[col_label].values])
            else: 
                for class_code, class_name in label_names.items(): 
                    df.loc[df[col_label]==class_code, col_label] = class_name
        else: 
            assert isinstance(unique_labels[0], str)

    if verbose:
        print("> After label encoding, label counts:") 
        if return_labels_only: 
            print(collections.Counter(labels))
        else: 
            print(get_label_counts(df))
    
    if return_labels_only: 
        # return df[col_label].values
        return labels
    return df

def to_dataframe(X, y, *, feature_cols=[], col_label='label', f_prefix='col', ): 
    return xy_to_dataframe(X, y, feature_cols=feature_cols, col_label=col_label, f_prefix=f_prefix)
def xy_to_dataframe(X, y, feature_cols=[], col_label='label', f_prefix='col'):
    import pandas as pd

    assert X.shape[0] == len(y)

    if not feature_cols: 
        feature_cols = [f'{f_prefix}_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols) 
    
    df[col_label] = y

    return df

def to_xy(df, target_cols, *, feature_cols=[], non_feature_cols=[]): 
    return toXy(df, target_cols, feature_cols=feature_cols, non_feature_cols=non_features_cols)
def toXy(df, target_cols, *, feature_cols=[], non_feature_cols=[]):
    if len(feature_cols) > 0: 
        dfX = df[feature_cols]
    else: 
        dfX = df.drop(target_cols, axis=1) if len(non_feature_cols) == 0 else df.drop(non_feature_cols, axis=1)
    X = dfX.values
    y = df[target_cols].values
    return (X, np.squeeze(y))

def dict_to_dataframe(adict: dict, columns=[]) -> pd.DataFrame: 
    if len(columns) == 0: # if the column names are not given ...
        columns = [0, 1] # ... set a default (although not recommended)
    if len(columns) != 2: 
        raise ValueError(f"A simple dictionary can only be converted to a two-column dataframe but given: {columns}")
            
    try: 
        df = pd.DataFrame(list(adict.items()), columns=columns) # keys are the columns
    except Exception as e: 
        print(e); print("\n>Analyzing the input...")
        for k, v in adict.items(): 
            print(f"{k} => {v}")
        raise ValueError
    return df
def dataframe_to_dict(df, use_cols=[], simple_dict=True, col_as_key=None):

    if len(use_cols) > 0: df = df[use_cols]
    columns = df.columns

    # Handle special case, 2-column dataframe, first
    if len(columns) == 2 and simple_dict: 
        if col_as_key in columns: 
            col_as_value = columns.drop(col_as_key)[0]
        return df.set_index(col_as_key)[col_as_value].to_dict()
    
    return df.to_dict()

# [alias]
def save_csv(df, **kargs):
    kargs['ext'] = '.csv'
    return save_dataframe(df, **kargs)
def save_dataframe(obj, output_path=None, output_file=None, ext='.csv', data_dir='data', **kargs): 
    """
    Save the input `df` as a dataframe but with index removed. If index is desirable, then do 
    df.reset_index() prior to this call. 

    """
    def is_simple_dictionary(adict):
        if not isinstance(adict, dict): return False 

        # tval = True # not a simple dictionary where values are non-sequence data type (e.g. int, float)
        # NOTE: a string is normally conidered as sequence, but here, it's not 
        n_non_sequence_detected = 0
        lengths = []
        for k, v in adict.items(): 
            if is_sequence(v): 
                lengths.append(len(v))
            else: 
                n_non_sequence_detected +=1
        
        n_uniq_length = len(set(lengths))
        if n_non_sequence_detected > 0 or n_uniq_length != 1: 
            return True
        return False

    import pandas as pd
    # from pathlib import Path
    verbose = kargs.pop('verbose', 1)
    verify = kargs.pop('verify', 0)
    dry_run = kargs.pop('test', 0)
    if not 'sep' in kargs: kargs['sep'] = '\t'

    output_path = process_path(path=output_path, file_name=output_file, 
                                        ext=ext, data_dir=data_dir)

    # if not dry_run: 
    #     pass
    # else: 
    #     # don't care 
    #     output_path = None

    if verbose: 
        print(f"[save] Output path:\n{output_path}\n") # os.path.dirname(output_path)

    if isinstance(obj, pd.DataFrame): 
        if not dry_run: 
            kargs['index'] = False
            obj.to_csv(output_path, **kargs) # head=None
        df = obj
    elif isinstance(obj, pd.Series): 
        # [todo] Handle Series where index are meaningful vs index carrying no information
        if not dry_run: 
            kargs['index'] = False
            obj.to_csv(output_path, **kargs) 
        df = obj
    elif isinstance(obj, dict): 
        columns = kargs.pop('columns', [])

        if is_simple_dictionary(obj): # treating keys and values as two separate columns
            df = dict_to_dataframe(obj, columns=columns)
            
            # Alternative way to initializing a dataframe with a "simple" dictionary
            # df = pd.DataFrame(list(obj.items()))

        else:  # keys as columns
            # `df` is a dictionary with the following layout
            #            
            #    { key1: [ ... ], 
            #      key2: [ ... ], 
            #        ...
            #      keyN: [ ... ]}
            #
            #  
            columns = list(obj.keys())
            df = pd.DataFrame(obj, columns=columns) # keys are the columns
        
        if verbose: 
            print(f"[save] Saving input dictionary to dataframe at:\n{output_path}\n")
            print(f"[debug] shape(df): {df.shape}")

        if not dry_run: 
            kargs['index'] = False
            df.to_csv(output_path, **kargs) # head=None

    elif isinstance(obj, (list, tuple, np.ndarray)): 
        columns = kargs.get('columns', [])
        if len(columns) == 0: 
            s = pd.Series(obj)
            if not dry_run: s.to_csv(output_path, index=None, header=None)
            df = s
        elif len(columns) == 1:  
            # Save the list like a regular dataframe
            df = pd.DataFrame(obj, columns=columns)

            if not dry_run: 
                kargs['index'] = False
                df.to_csv(output_path, **kargs) # NOTE: by default, index is included and header is also included
        else: 
            msg = f"A list can only be saved as a single column but given columns={columns}"
            raise ValueError(msg)

    # df.to_csv(output_path, index=False, sep=sep) # head=None
    assert dry_run or (output_path.find(ext) > 0)
    # if dry_run: 
    #     print(f"[save] Input dataframe would have been saved to:\n{output_path}\n")

    return df

def save_to_excel(df, filename, sheet_name='Sheet1', index=False):
    """
    Save a Pandas DataFrame to an Excel file.

    Args:
    df (pd.DataFrame): The DataFrame to save.
    filename (str): The name of the file to save the DataFrame to.
    sheet_name (str): The name of the sheet in the Excel file. Defaults to 'Sheet1'.
    index (bool): Whether to write row names (indexes). Defaults to False.
    """
    # Ensure the filename ends with '.xlsx'
    if not filename.endswith('.xlsx'):
        filename += '.xlsx'

    # Save the DataFrame to an Excel file
    df.to_excel(filename, sheet_name=sheet_name, index=index)

    return filename

def load_from_excel(filename, sheet_name='Sheet1'):
    """
    Load an Excel file into a Pandas DataFrame.

    Args:
    filename (str): The name of the Excel file to load.
    sheet_name (str or int): The name or index of the sheet to load. Defaults to 'Sheet1'.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    # Load the Excel file
    df = pd.read_excel(filename, sheet_name=sheet_name)
    return df


def save_pickle(obj, output_path=None, output_file=None, ext='.pkl', data_dir='data', verbose=1):
    """

    Memo
    ----
    1. pickle.HIGHEST_PROTOCOL
       It picks the highest protocol version your version of Python supports: 
       docs.python.org/3/library/pickle.html#data-stream-format
    """
    # from pathlib import Path
    output_path = process_path(path=output_path, file_name=output_file, 
                                    ext=ext, data_dir=data_dir)
    if verbose > 0: 
        print(f"[save] Output prefix:\n{os.path.dirname(output_path)}\nSaved pickle ({ext}) to:\n{output_path}\n")

    with open(output_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return
def load_pickle(input_path=None, input_file=None, ext='.pkl', data_dir='data', verbose=0): 
    input_path = process_path(path=input_path, file_name=input_file, 
                                    ext=ext, data_dir=data_dir, makedir=False)
    if not os.path.exists(input_path): 
        raise FileNotFoundError(f"Invalid input path:\n{input_path}\n")

    obj = None
    with open(input_path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

def demo_save_load_pickle(adict, path=None, file_name='test'): 
    import pickle

    output_path = process_path(path=path, file_name=file_name, ext='pkl', data_dir='test')

    a = {'hello': 'world'}
    with open(output_path, 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_path, 'rb') as handle:
        b = pickle.load(handle)

    print(a == b)

    return

def get_example_dataset(name="titanic"): 
    from pandas import DataFrame
    from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder

    name = name.lower()

    data = DataFrame() # dummy 
    if name.startswith("tita"): 
        # Titanic
        variables = [
            'pclass', 'survived', 'sex', 'age', 'sibsp',
            'parch', 'fare', 'cabin', 'embarked',
            ]

        data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl',
                        usecols=variables,
                        na_values='?',
                        dtype={'fare': float, 'age': float},
                        )
        data.dropna(subset=['embarked', 'fare'], inplace=True)
        data['age'] = data['age'].fillna(data['age'].mean())

        def get_first_cabin(row):
            try:
                return row.split()[0]
            except:
                return 'N'

        data['cabin'] = data['cabin'].apply(get_first_cabin).str[0]

        encoder = RareLabelEncoder(variables='cabin', n_categories=2)
        data = encoder.fit_transform(data)

        # convert categorical variables to numbers
        encoder = OrdinalEncoder(
                        encoding_method='arbitrary',
                        variables=["sex", "cabin", "embarked"],
                        )
        data = encoder.fit_transform(data)

        print(data.head())

    print(f"[data] shape: {data.shape}")
    return data

import os

def get_sample_sizes(dataset_path, splits=None):
    """
    Traverse the directory structure to compute sample sizes for training, testing, and validation sets.

    Parameters:
    - dataset_path: path to the root directory containing train, test, and validation directories

    Returns:
    - A dictionary with the sample sizes for each split and class.
    """
    if splits is None: splits = ['train', 'test', 'validation']
    sample_sizes = {}

    total_samples = 0

    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if os.path.exists(split_path):
            sample_sizes[split] = {}
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if os.path.isdir(class_path):
                    num_samples = len(os.listdir(class_path))
                    sample_sizes[split][class_name] = num_samples
                    total_samples += num_samples

    sample_sizes["total"] = total_samples
    return sample_sizes

def demo_datastructure(): 
    from tabulate import tabulate
    from pathlib import Path
    import os, sys

    # getting the name of the directory where the this file is present.
    current_dir = Path.cwd() # os.path.dirname(os.path.realpath(__file__))
    # Getting the parent directory name where the current directory is present.
    parent_dir = os.path.dirname(current_dir)
    # adding the parent directory to the sys.path.
    sys.path.append(parent_dir)
    from sys_config import get_data_dir

    # A simple dictionary? 
    details = {
        'Ankit' : 22,
        'Golu' : 21,
        'hacker' : 23
    }
    print(is_simple_dictionary(details)) 
    df = save_dataframe(details, test=True)
    print(tabulate(df, headers='keys', tablefmt='psql'))

    details = {
        'Name' : ['Ankit', 'Aishwarya', 'Shaurya', 'Shivangi'],
        'Age' : [23, 21, 22, 21],
        'University' : ['BHU', 'JNU', 'DU', 'BHU'],
    }
    print(is_simple_dictionary(details)) 
    df = save_dataframe(details, test=True)
    print(tabulate(df, headers='keys', tablefmt='psql'))

    details = {
        'Name' : ['Ankit', 'Aishwarya',  ],
        'Age' : [23, 21, 22, 21],
        'University' : ['BHU', 'JNU', 'DU', 'BHU'],
    }
    print(f"An ill-formed multiset dictionary that cannot be used to initialize pd.DataFrame():\n{details}\n")
    print(is_simple_dictionary(details)) # treat this as a "simple" dictionary
    try: 
        pd.DataFrame(details)
    except: 
        print(f"Could not initialize DataFrame with dict:\n{details}\n")
    print("> But we could treat it as a 'simple' dictionary ...")
    df = save_dataframe(details, test=True, verbose=0)
    print(tabulate(df, headers='keys', tablefmt='psql'))

    # Saving data 
    print("\n> Testing data saving ...\n")

    details = {
        'Name' : ['Ankit', 'Aishwarya', 'Shaurya', 'Shivangi'],
        'Age' : [23, 21, 22, 21],
        'University' : ['BHU', 'JNU', 'DU', 'BHU'],
    }
    df = pd.DataFrame(details)

    output_file = "test.csv"
    output_dir = os.path.join(get_data_dir('nmd'), 'test')
    assert os.path.exists(output_dir)
    output_path = os.path.join(output_dir, output_file)   

    save_dataframe(df, output_path=output_path, sep='|', index=False)

    output_file = "test-no-header.csv"
    output_dir = os.path.join(get_data_dir('nmd'), 'test')
    save_dataframe(df, output_path=output_dir, output_file=output_file, sep='|', index=False, header=False)

    return

def test(): 
    import os, sys
    from pathlib import Path

    # ----- Dataset ----- 
    # df = get_example_dataset(name='titanic')

    demo_probing_dataset_directory()

    # ----- Misc -----
    # demo_datastructure()


    # getting the name of the directory where the this file is present.
    current_dir = Path.cwd() # os.path.dirname(os.path.realpath(__file__))
    # Getting the parent directory name where the current directory is present.
    parent_dir = os.path.dirname(current_dir)
    # adding the parent directory to the sys.path.
    # sys.path.append(parent_dir)

    # output_dir = os.path.join(parent_dir, 'test') 
    # assert os.path.exists(output_dir)
    # output_path = os.path.join(output_dir, 'test.csv')
    # adict = {'a': 3, 'b': 4, 'c': 5, 'd': 20, 'e': 18}
    # columns = ['attribute', 'value']
    # print(adict)
    # save_dataframe(adict, output_path=output_path, sep='\t', columns=columns, verbose=1)

    return

if __name__ == "__main__": 
    test()