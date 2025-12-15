import matplotlib.pyplot as plt
import numpy as np 
import scipy as sp
from scipy import signal
import random
import math
import itertools
from tqdm import tqdm 

import os, sys
from pathlib import Path

# Enable importing files from parent dir
# getting the name of the directory where the this file is present.
current_dir = Path.cwd() # os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name where the current directory is present.
parent_dir = os.path.dirname(current_dir)
# adding the parent directory to the sys.path.
sys.path.append(parent_dir)

# Scikit-learn stuff
from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectKBest, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Import Pandas: https://pandas.pydata.org/
import pandas as pd
from tabulate import tabulate

# Import Seaborn: https://seaborn.pydata.org/
import seaborn as sns

import utils.utils_classifier as uclf
from tabulate import tabulate

# from ..utils_sys import highlight
# NOTE: This won't work because it treats utils_sys as a parent package

from utils.utils_sys import highlight
from utils.utils_df import display


def make_ge_dataset(**kargs): 
    from data_pipeline import generate_ge_centric_dataset

    save = kargs.get("save", False)

    # select a subset of rows
    n_samples = kargs.get('n_samples', 2500) # subsampling (e.g. for rapid experiments) 

    # select a subset of columns
    fs_method = kargs.pop("fs_method", "MI")
    fs_model = kargs.pop("fs_model", "LASSO") # model used for wrapper-based FS method (e.g. RFE)
    n_features = kargs.get("n_features", 100)
    custom_feature_cols = kargs.get("custom_feature_cols", [])

    col_tid = 'transcript_id'
    col_gid = 'gene_id'
    col_label = 'label'
    gene_as_feature = kargs.get("gene_as_feature", True)
    target_cols = kargs.get("target_cols", [col_label, ])
    meta_cols = kargs.get("meta_cols", [col_tid, ] if gene_as_feature else [col_tid, col_gid, ]) # or, [col_tid, ], if gene is considered a categorical feature
    cat_cols = kargs.get("cat_cols", [col_gid, ] if gene_as_feature else []) 

    df, cols_dict = generate_ge_centric_dataset(**kargs)
    
    return df, cols_dict

class WithinGroupShuffleSplit(object): 
    """
    
    Related 
    ------- 
    sklearn.model_selection.GroupShuffleSplit

    Memo
    ----
    1. Groupby + sampling 
       - https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.sample.html
    """
    def __init__(self, n_splits=5, train_size=0.8, val_size=None, test_size=None, train_val_test_split=True, random_state=None):
        self.n_splits = n_splits 
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.train_val_test_split = train_val_test_split
        
        if train_val_test_split: 
            if test_size is None: 
                # msg = f"In train_val_test split mode, test_size must be specified | train_size: {train_size}, test_size: {test_size}"
                # raise ValueError(msg)
                self.test_size = 0.1
            if val_size is None: 
                self.val_size = 1 - train_size - test_size 
                if self.val_size <= 0: 
                    msg = f"Invalid validation set ratio: {self.val_size} | train_size: {train_size}, test_size: {test_size}"
                    raise ValueError(msg)
            # All sizes must sum to <= 1.0
            assert self.train_size + self.val_size + self.test_size <= 1.0
        else: 
            if train_size is None and test_size is None: 
                raise ValueError("Either train_size or test_size, or both, need to be specified.")
            if test_size is None: 
                self.test_size = 1.0 - self.train_size
            if train_size is None: 
                self.train_size = 1.0 - self.test_size
            # All sizes must sum to <= 1.0
            assert self.train_size + self.test_size <= 1.0

        self.random_state = random_state

    def split(self, df, group, target='label'): 
        group_cols = group
        seed = int(np.random.rand(1)[0] * 2**32)-1 - self.n_splits

        if self.train_val_test_split: 
            for i in range(self.n_splits): 
                df_train = df.groupby(group_cols).sample(frac=self.train_size, random_state=seed+i)
                X_train = df_train.drop(target, axis=1)
                y_train = df_train[target]

                val_test_size = 1 - self.train_size
                df_val_test = df[~df.index.isin(df_train.index)]
                test_size_adjusted = self.test_size/(val_test_size+0.0)

                # A. val and test set can simply be a random subset of the data not in training set
                df_test = df_val_test.sample(frac=test_size_adjusted, random_state=seed+i+1)
                X_test = df_test.drop(target, axis=1)
                y_test = df_test[target]

                df_val = df_val_test[~df_val_test.index.isin(df_test.index)]
                X_val = df_val.drop(target, axis=1)
                y_val = df_val[target]

                # B. Ensure that test set has most of the subgroups as much as possible ... 
                # df_test = df_val_test.groupby(group_cols).sample(frac=test_size_adjusted, random_state=seed+i)
                # X_test = df_test.drop(target, axis=1)
                # y_test = df_test[target]
                # # ... while the rest serves as validation set
                # df_val = df_val_test[~df_val_test.index.isin(df_test.index)]
                # X_val = df_val.drop(target, axis=1)
                # y_val = df_val[target]

                yield X_train, y_train, X_val, y_val, X_test, y_test
        else: # the usual train-test split
            for i in range(self.n_splits): 
                df_train = df.groupby(group_cols).sample(frac=self.train_size, random_state=seed+i)
                X_train = df_train.drop(target, axis=1)
                y_train = df_train[target]

                df_test = df[~df.index.isin(df_train.index)]
                X_test = df_test.drop(target, axis=1)
                y_test = df_test[target]

                yield X_train, y_train, X_test, y_test

class WithinGroupShuffleSplit2(WithinGroupShuffleSplit): 
    """
    
    Related 
    ------- 
    sklearn.model_selection.GroupShuffleSplit

    Memo
    ----
    1. Groupby + sampling 
       - https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.sample.html
    """
    def __init__(self, group, n_splits=5, train_size=0.8, test_size=0.2, random_state=None):
        super().__init__(n_splits=n_splits, train_size=train_size, test_size=test_size, train_val_test_split=False, random_state=random_state)
        # super(WithinGroupShuffleSplit2, self).__init__(...)
        self.group_cols = group  # the column used to define the group IDs

    def split(self, X, y): 
        group_cols = self.group_cols
        seed = int(np.random.rand(1)[0] * 2**32)-1 - self.n_splits

        # Todo: stratify by 'y'
        for i in range(self.n_splits): 
            X_train = X.groupby(group_cols).sample(frac=self.train_size, random_state=seed+i)
            train_index = np.array(X_train.index)

            test_index = np.array(X.drop(index=train_index).index)
            # X_test = X[~X.index.isin(train_index)]
            # test_index = np.array(X_test.index)
          
            # print(f"shape(X_train): {X_train.shape}, train_index size: {len(train_index)}, ex:\n{train_index[:10]}\n")
            # assert len(set(train_index).intersection(test_index)) == 0 # ... ok
            yield train_index, test_index

            
def demo_stratify_on_ge_data(): 
    def show_cols_dict(n=10):
        for feature_type, cols in cols_dict.items(): 
            print(f"[{feature_type}]")
            print(f"... {cols[:n]}")
    def verify_data(df_ge, cat_cols, title='', pos_label=1, neg_label=0, stdout=True): 
        msg = "> GE data summary statistics ...\n" if not title else title + '\n'
        sampled_num_cols = np.random.choice(cols_dict['num_cols'], min(10, len(cols_dict['num_cols'])), replace=False)
        adict = {col:[] for col in sampled_num_cols}
        # --- 
        n0 = df_ge.shape[0]
        for col in sampled_num_cols: 
            n_nz = np.sum(df_ge[col] > 0)
            adict[col].append( round(n_nz/n0 * 100, 2) ) # count percentage of rows with non-zero values
        df_nonzero = pd.DataFrame(adict, index=['n_nz', ])
        dfx = pd.concat([df_ge[ list(sampled_num_cols)].describe(), df_nonzero])
        msg += str(dfx); msg += '\n'; msg += '-' * 80 + '\n'
        # --- 
        genes = df_ge[col_gid].unique()
        trpts = df_ge[col_tid].unique()
        msg += f"... n(genes): {len(genes)}, n(trpts): {len(trpts)}\n"
        label_counts = df_ge[col_label].value_counts()
        msg += f"... n(pos): {label_counts[pos_label]}, n(neg): {label_counts[neg_label]}\n"
        if stdout: print(msg)
        return msg

    # from gene_analyzer import load_gene_expression_data
    from sys_config import get_data_dir
    from utils.utils_encoder import MultipleEncoder
    from lightgbm import LGBMClassifier
    from gene_analyzer import GEMatrix
    from fast_ml.model_development import train_valid_test_split
    
    highlight("> Load gene expression labeled", symbol="#", border=1) 
    data_dir = get_data_dir(proj_name="nmd")
    version="GRCh38.106"

    col_gid = 'gene_id'
    col_tid = 'transcript_id'
    col_label = 'label'

    # a wrapper of data_pipeline.generate_ge_centric_dataset() that, by default, generates a random subset of GE data
    df, cols_dict = make_ge_dataset(version=version, prefix=data_dir, save=False, verbose=1, 
        on_feature_selection=False)
    print(f"> Raw GE labeled data: shape: {df.shape}")
    verify_data(df, cat_cols=cols_dict['cat_cols'])

    meta_cols = cols_dict['meta_cols']
    target_cols = cols_dict['target_cols']
    non_feature_cols = list(meta_cols) + list(target_cols)
    feature_cols = df.columns.drop(non_feature_cols)
    dfXy = df.drop(meta_cols, axis=1)

    group_cols = [col_gid, ]

    # Get the shape of all the datasets
    # X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(dfXy, target = col_label, 
    #                                                                         train_size=0.8, valid_size=0.1, test_size=0.1)

    # print(f"> shape(dfXy): {dfXy.shape}")
    # print(X_train.shape), print(y_train.shape) # `X_train` contains features only, labels are separated out
    # print(X_valid.shape), print(y_valid.shape)
    # print(X_test.shape), print(y_test.shape)

    # stratify by genes 
    train_size, val_size, test_size = 0.7, 0.15, 0.15
    split_X_y = False

    if split_X_y: 
        X = df.drop(target_cols, axis=1) 
        # NOTE: want ID column to be preserved
        # X = df[feature_cols] # .reset_index(drop=True)
        N = X.shape[0]
        if len(target_cols) == 1: 
            y = df[target_cols].values.reshape((N, ))
        else: 
            y = df[target_cols]
        # print(f"shape(X): {X.shape}, shape(y): {y.shape}")

        wgss = WithinGroupShuffleSplit2(group=group_cols, n_splits=3, train_size = train_size, random_state=53)
        for i, (train, test) in enumerate(wgss.split(X, y)):

            highlight(f"Fold = {i}", symbol='=', border=1)
            # print(X.index[:100])
            # print(f"X index > min: {min(X.index)}, max: {max(X.index)} | cv(train) > min: {min(train)}, max: {max(test)}")

            X_train, X_test = X.iloc[train].reset_index(drop=True), X.iloc[test].reset_index(drop=True)
            # X_train, X_test = X.iloc[train], X.iloc[test]  # this leads to NaN when category_encoder is applied? 
            y_train, y_test = y[train], y[test]

            # Need to train on all genes => determine the train set first
            genes_train = X_train[col_gid].unique()
            trpts_train = X_train[col_tid].unique()
            print(f"... n_genes(train): {len(genes_train)}, n_trpts(train): {len(trpts_train)}")
            print(f"... indice(train):\n{train}\n")

            genes_test = X_test[col_gid].unique()
            trpts_test = X_test[col_tid].unique()
            print(f"... n_genes(test): {len(genes_test)}, n_trpts(test): {len(trpts_test)}")
            # print(f"... indice(test):\n{X_test.index[:10]}\n")

            print("-" * 80)
            print(f"> shape(train): {X_train.shape}"); print(f"> shape(test): {X_test.shape}")
            
            # assert len(set(X_train.index).intersection(X_test.index)) == 0, f"intersected? size={len(set(X_train.index).intersection(X_test.index))}"

    else: 
        wgss = WithinGroupShuffleSplit(n_splits=3, train_size=train_size, val_size=val_size, test_size=test_size, random_state=53)

        for i, (X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(wgss.split(df, group=group_cols, target=col_label)):

            highlight(f"Fold = {i}", symbol='=', border=1)
            # Need to train on all genes => determine the train set first
            genes_train = X_train[col_gid].unique()
            trpts_train = X_train[col_tid].unique()
            print(f"... n_genes(train): {len(genes_train)}, n_trpts(train): {len(trpts_train)}")
            print(f"... indice(train):\n{X_train.index[:10]}\n")

            genes_val = X_val[col_gid].unique()
            trpts_val = X_val[col_tid].unique()
            print(f"... n_genes(val): {len(genes_val)}, n_trpts(val): {len(trpts_val)}")
            # print(f"... indice(val):\n{X_val.index[:10]}\n")

            genes_test = X_test[col_gid].unique()
            trpts_test = X_test[col_tid].unique()
            print(f"... n_genes(test): {len(genes_test)}, n_trpts(test): {len(trpts_test)}")
            # print(f"... indice(test):\n{X_test.index[:10]}\n")

            print("-" * 80)
            print(f"> shape(train): {X_train.shape}"); print(f"> shape(val): {X_val.shape}"); print(f"> shape(test): {X_test.shape}")
            
            assert len(set(X_train.index).intersection(X_val.index)) == 0
            assert len(set(X_train.index).intersection(X_test.index)) == 0

    return

def test(): 

    # Applying train-test-split while stratifying by genes 
    demo_stratify_on_ge_data()

    return

if __name__ == "__main__": 
    test()