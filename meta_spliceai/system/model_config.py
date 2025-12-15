import sys, os, re
from pathlib import Path 

# Refactored to __init__
# current_dir = Path.cwd() # os.path.dirname(os.path.realpath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir) 

import numpy as np
import pandas as pd 
import dask.dataframe as dd
# import gene_analyzer as ga

import system.sys_config as config


class BiotypeModel(object): 

    # Default settings for biotype classifier
    labeling_concept = 'biotype_3way'
    model_output_dir = 'biotype_3way'  # The value of model_output_dir in this class is also used as ... 
    # ... the value for self.experiment in ModelTracker; this gives as a directory structure like 
    # <project_dir>/biotype_3way/... to keep track of model outputs

    model_type = 'descriptor'
    model_name = 'xgboost'
    model_suffix = 'bohb'

    # Feature parameters 
    ftype = 'seq-featurized'

    # Training parameters
    n_folds = 5
    use_nested_cv = False

    # Transcriptomic data IDs:
    # - These IDs allow us to index into the transcriptomic data used to train a biotype classifier
    source_biotype = '3way'
    source_suffix = 'trainset'

class NMDEffModel(object): 
    labeling_concept = base_concept = 'nmd_eff'
    Z = [-0.1, 0.1]    
    model_output_dir = f'{base_concept}-z{Z[1]}'  # The value of model_output_dir in this class is also used as ... 

    model_type = 'descriptor'
    model_name = 'xgboost'
    model_suffix = 'bohb'

    # Feature parameters 
    ftype = 'seq-featurized'

    # Training parameters
    n_folds = 5
    use_nested_cv = False

    # Transcriptomic data IDs:
    # - These IDs allow us to index into the transcriptomic data used to train a biotype classifier
    source_biotype = 'nmd_eff'
    source_suffix = 'trainset'
    
    test_biotype = 'nmd_eff'
    test_suffix = 'testset'

    






