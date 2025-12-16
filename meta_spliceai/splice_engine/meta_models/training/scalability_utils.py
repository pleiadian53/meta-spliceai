#!/usr/bin/env python3
"""Utilities for improving memory efficiency and scalability of meta-model training.

This module provides helper functions to optimize the memory usage and scalability
of meta-model training pipelines, especially when dealing with large datasets
containing many features (e.g., k-mers).

Key functionalities:
- Feature selection to reduce dimensionality
- Chunked/streaming data processing
- Memory-efficient feature encoding
- Sparse matrix representations for k-mers
"""
from __future__ import annotations

import gc
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
import xgboost as xgb

from meta_spliceai.splice_engine.utils_kmer import is_kmer_feature


logger = logging.getLogger(__name__)


def select_features(
    X: pd.DataFrame, 
    y: np.ndarray, 
    max_features: int = 1000,
    method: str = 'model',
    model_kwargs: Optional[Dict] = None,
    random_state: int = 42,
    exclude_features: List[str] = None,
    force_include_features: List[str] = None,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Select most important features to reduce dimensionality.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Target values
    max_features : int, default=1000
        Maximum number of features to select
    method : str, default='model'
        Method to use for feature selection:
        - 'model': Use a tree-based model to select features
        - 'mutual_info': Use mutual information to select features
    model_kwargs : Dict, optional
        Additional parameters for the model (when method='model')
    random_state : int, default=42
        Random state for reproducibility
    exclude_features : List[str], optional
        List of features to exclude from selection (will be dropped)
    force_include_features : List[str], optional
        List of features to always include regardless of importance
        
    Returns
    -------
    X_selected : pd.DataFrame
        DataFrame with selected features
    selected_features : List[str]
        List of selected feature names
    feature_info : Dict
        Dictionary with information about feature selection
    """
    feature_names = list(X.columns)
    feature_info = {
        'original_feature_count': len(feature_names),
        'method': method,
        'max_features': max_features,
        'exclude_features': exclude_features or [],
        'force_include_features': force_include_features or [],
    }
    
    # Start with all features
    all_features = set(feature_names)
    
    # Remove excluded features
    if exclude_features:
        all_features = all_features - set(exclude_features)
        
    # Always include forced features
    force_include = set()
    if force_include_features:
        force_include = set(force_include_features) & all_features
    
    # Features eligible for selection
    eligible_features = list(all_features - force_include)
    
    # Create X with only eligible features for selection
    X_eligible = X[eligible_features]
    
    # Count features by type for reporting
    kmer_features = [f for f in eligible_features if is_kmer_feature(f)]
    non_kmer_features = [f for f in eligible_features if not is_kmer_feature(f)]
    
    feature_info['original_kmer_features'] = len(kmer_features)
    feature_info['original_non_kmer_features'] = len(non_kmer_features)
    
    # Calculate number of features to select (accounting for forced inclusions)
    n_to_select = max(1, min(max_features - len(force_include), len(eligible_features)))
    
    selected_from_eligible = []
    
    if method == 'model':
        # Use tree-based model for feature selection
        default_model_kwargs = {
            'n_estimators': 50,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
        }
        if model_kwargs:
            default_model_kwargs.update(model_kwargs)
            
        # Use XGBoost for feature selection
        model = xgb.XGBClassifier(**default_model_kwargs)
        
        try:
            selector = SelectFromModel(
                model, 
                max_features=n_to_select, 
                threshold=-np.inf,  # Always select max_features
            )
            selector.fit(X_eligible, y)
            
            # Get selected feature indices
            selected_indices = selector.get_support(indices=True)
            selected_from_eligible = [eligible_features[i] for i in selected_indices]
            
            # Get feature importances
            importances = selector.estimator_.feature_importances_
            feature_info['importance_method'] = 'feature_importances_'
            feature_info['importances'] = {
                eligible_features[i]: float(importances[i]) 
                for i in range(len(eligible_features))
            }
        except Exception as e:
            warnings.warn(f"Error in model-based feature selection: {e}. "
                          f"Falling back to mutual information.")
            method = 'mutual_info'
    
    if method == 'mutual_info':
        # Use mutual information for feature selection
        try:
            mi_scores = mutual_info_classif(
                X_eligible, y, random_state=random_state
            )
            
            # Get indices of top features by mutual information
            selected_indices = np.argsort(mi_scores)[-n_to_select:]
            selected_from_eligible = [eligible_features[i] for i in selected_indices]
            
            feature_info['importance_method'] = 'mutual_info_classif'
            feature_info['importances'] = {
                eligible_features[i]: float(mi_scores[i]) 
                for i in range(len(eligible_features))
            }
        except Exception as e:
            # If mutual information fails, fall back to selecting features randomly
            warnings.warn(f"Error in mutual information feature selection: {e}. "
                          f"Selecting features randomly.")
            
            # Select randomly (but deterministically)
            np.random.seed(random_state)
            selected_indices = np.random.choice(
                len(eligible_features), size=n_to_select, replace=False
            )
            selected_from_eligible = [eligible_features[i] for i in selected_indices]
            
            feature_info['importance_method'] = 'random'
            feature_info['importances'] = {}
    
    # Combine force-included features with selected features
    selected_features = list(force_include) + selected_from_eligible
    
    # Create final DataFrame with only selected features
    X_selected = X[selected_features].copy()
    
    # Update feature info
    selected_kmers = [f for f in selected_features if is_kmer_feature(f)]
    selected_non_kmers = [f for f in selected_features if not is_kmer_feature(f)]
    
    feature_info['selected_feature_count'] = len(selected_features)
    feature_info['selected_kmer_features'] = len(selected_kmers)
    feature_info['selected_non_kmer_features'] = len(selected_non_kmers)
    feature_info['force_included_count'] = len(force_include)
    
    # Clean up
    gc.collect()
    
    return X_selected, selected_features, feature_info


def load_dataset_chunked(
    dataset_path: Union[str, Path], 
    chunksize: int = 10000,
    max_rows: Optional[int] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
    feature_subset: Optional[List[str]] = None,
    label_col: str = "label",
    required_cols: Optional[List[str]] = None,
):
    """Load dataset in chunks to reduce memory footprint.
    
    Parameters
    ----------
    dataset_path : str or Path
        Path to dataset file or directory with Parquet files
    chunksize : int, default=10000
        Number of rows to load per chunk
    max_rows : int, optional
        Maximum number of rows to load in total
    sample_frac : float, optional
        Fraction of rows to sample from each chunk
    random_state : int, default=42
        Random state for reproducibility when sampling
    feature_subset : List[str], optional
        Only load these specific columns (features)
    label_col : str, default="label"
        Name of the label column
    required_cols : List[str], optional
        Additional required columns to always load
        
    Yields
    ------
    pd.DataFrame
        Chunk of dataset with selected features
    """
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from glob import glob
    import os
    
    # Convert path to Path object
    dataset_path = Path(dataset_path)
    
    # Initialize random number generator
    rng = np.random.RandomState(random_state)
    
    # Determine if path is a directory or file
    if dataset_path.is_dir():
        # Find all parquet files in the directory
        parquet_files = sorted(glob(str(dataset_path / "*.parquet")))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {dataset_path}")
    else:
        # Single parquet file
        parquet_files = [str(dataset_path)]
    
    # Prepare columns to load
    columns = None
    if feature_subset:
        # Always include label column and required columns
        columns = list(feature_subset)
        if label_col and label_col not in columns:
            columns.append(label_col)
        if required_cols:
            for col in required_cols:
                if col not in columns:
                    columns.append(col)
    
    # Track total rows loaded
    total_rows_loaded = 0
    
    # Process each parquet file
    for parquet_file in parquet_files:
        # Get file size for reporting
        file_size_mb = os.path.getsize(parquet_file) / (1024 * 1024)
        logger.info(f"Processing file {os.path.basename(parquet_file)} ({file_size_mb:.1f} MB)")
        
        # Use pyarrow for more efficient parquet reading
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_file, columns=columns)
            num_rows = table.num_rows
            
            # Calculate how many chunks we need
            num_chunks = (num_rows + chunksize - 1) // chunksize
            
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunksize
                end_idx = min(start_idx + chunksize, num_rows)
                
                # Load this chunk
                chunk = table.slice(start_idx, end_idx - start_idx).to_pandas()
                
                # Apply sampling if requested
                if sample_frac is not None:
                    chunk = chunk.sample(frac=sample_frac, random_state=rng)
                
                # Check if we've reached max_rows
                if max_rows and total_rows_loaded + len(chunk) > max_rows:
                    # Truncate the chunk
                    rows_needed = max_rows - total_rows_loaded
                    chunk = chunk.iloc[:rows_needed]
                    
                # Update total rows counter
                total_rows_loaded += len(chunk)
                
                logger.info(f"  Yielding chunk {chunk_idx+1}/{num_chunks} with {len(chunk)} rows")
                yield chunk
                
                # Break if we've reached max_rows
                if max_rows and total_rows_loaded >= max_rows:
                    logger.info(f"Reached maximum row limit ({max_rows}), stopping.")
                    return
                
                # Force garbage collection
                gc.collect()
                
        except ImportError:
            # Fall back to pandas if pyarrow is not available
            logger.warning("PyArrow not available, falling back to pandas (slower).")
            
            # Use pandas to read parquet file in chunks
            for chunk in pd.read_parquet(
                parquet_file, columns=columns, chunksize=chunksize
            ):
                # Apply sampling if requested
                if sample_frac is not None:
                    chunk = chunk.sample(frac=sample_frac, random_state=rng)
                
                # Check if we've reached max_rows
                if max_rows and total_rows_loaded + len(chunk) > max_rows:
                    # Truncate the chunk
                    rows_needed = max_rows - total_rows_loaded
                    chunk = chunk.iloc[:rows_needed]
                
                # Update total rows counter
                total_rows_loaded += len(chunk)
                
                yield chunk
                
                # Break if we've reached max_rows
                if max_rows and total_rows_loaded >= max_rows:
                    logger.info(f"Reached maximum row limit ({max_rows}), stopping.")
                    return
                
                # Force garbage collection
                gc.collect()


def convert_kmers_to_sparse(df, kmer_prefix='6mer_', feature_names=None):
    """Convert k-mer features to sparse matrix format to save memory.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing k-mer features
    kmer_prefix : str, default='6mer_'
        Prefix identifying k-mer features
    feature_names : list, optional
        List of feature names to consider, if None use all df.columns
        
    Returns
    -------
    tuple
        (sparse_matrix, non_kmer_df, kmer_feature_names, non_kmer_feature_names)
    """
    if feature_names is None:
        feature_names = df.columns.tolist()
    
    # Identify k-mer and non-k-mer features
    kmer_features = [f for f in feature_names if is_kmer_feature(f, prefix=kmer_prefix)]
    non_kmer_features = [f for f in feature_names if f not in kmer_features]
    
    logger.info(f"Converting {len(kmer_features)} k-mer features to sparse format")
    
    # Extract non-k-mer features as regular DataFrame
    non_kmer_df = df[non_kmer_features].copy() if non_kmer_features else pd.DataFrame(index=df.index)
    
    # Convert k-mer features to sparse matrix
    if kmer_features:
        kmer_matrix = sparse.csr_matrix(df[kmer_features].values)
        
        # Print memory usage reduction
        dense_size = df[kmer_features].values.nbytes / (1024 * 1024)  # MB
        sparse_size = sum(x.nbytes for x in [kmer_matrix.data, kmer_matrix.indices, kmer_matrix.indptr]) / (1024 * 1024)  # MB
        logger.info(f"Memory reduction: dense={dense_size:.2f} MB → sparse={sparse_size:.2f} MB ({sparse_size/dense_size*100:.1f}%)")
    else:
        # No k-mer features, return empty sparse matrix
        kmer_matrix = sparse.csr_matrix((len(df), 0))
    
    return kmer_matrix, non_kmer_df, kmer_features, non_kmer_features


def optimize_memory_usage(df):
    """Optimize memory usage of DataFrame by downcasting numeric types.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to optimize
        
    Returns
    -------
    pd.DataFrame
        Optimized DataFrame with reduced memory usage
    """
    start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    logger.info(f"Memory usage before optimization: {start_mem:.2f} MB")
    
    # Make a copy to avoid modifying the original DataFrame
    df_opt = df.copy()
    
    for col in df_opt.columns:
        col_type = df_opt[col].dtype
        
        # Process numerical columns only
        if col_type != 'object':
            # Integer types
            if np.issubdtype(col_type, np.integer):
                # Find min/max to determine the smallest possible dtype
                col_min = df_opt[col].min()
                col_max = df_opt[col].max()
                
                # Determine best integer type
                if col_min >= 0:  # unsigned
                    if col_max < 255:
                        df_opt[col] = df_opt[col].astype(np.uint8)
                    elif col_max < 65535:
                        df_opt[col] = df_opt[col].astype(np.uint16)
                    elif col_max < 4294967295:
                        df_opt[col] = df_opt[col].astype(np.uint32)
                    else:
                        df_opt[col] = df_opt[col].astype(np.uint64)
                else:  # signed
                    if col_min > -128 and col_max < 127:
                        df_opt[col] = df_opt[col].astype(np.int8)
                    elif col_min > -32768 and col_max < 32767:
                        df_opt[col] = df_opt[col].astype(np.int16)
                    elif col_min > -2147483648 and col_max < 2147483647:
                        df_opt[col] = df_opt[col].astype(np.int32)
                    else:
                        df_opt[col] = df_opt[col].astype(np.int64)
            
            # Float types
            elif np.issubdtype(col_type, np.floating):
                # Use float32 instead of float64 for significant memory savings
                df_opt[col] = df_opt[col].astype(np.float32)
                
                # For very small ranges, could even use float16 but this is risky
                # Uncomment if extreme memory optimization is needed
                # col_min = df_opt[col].min()
                # col_max = df_opt[col].max()
                # if col_min > -65500 and col_max < 65500:
                #     df_opt[col] = df_opt[col].astype(np.float16)
    
    # Calculate memory savings
    end_mem = df_opt.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    reduction = 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
    
    logger.info(f"Memory usage after optimization: {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    
    return df_opt


def optimize_sparse_matrix(sparse_matrix):
    """Optimize a sparse matrix to use minimal memory.
    
    Parameters
    ----------
    sparse_matrix : scipy.sparse.spmatrix
        Sparse matrix to optimize
        
    Returns
    -------
    scipy.sparse.csr_matrix
        Optimized sparse matrix
    """
    # Ensure we have a CSR matrix
    if not isinstance(sparse_matrix, sparse.csr_matrix):
        sparse_matrix = sparse.csr_matrix(sparse_matrix)
    
    # Calculate initial memory usage
    initial_size = sum(x.nbytes for x in [sparse_matrix.data, sparse_matrix.indices, sparse_matrix.indptr]) / (1024 * 1024)  # MB
    
    # Determine optimal data type for indices and indptr
    if sparse_matrix.shape[1] < 256:
        indices_dtype = np.uint8
    elif sparse_matrix.shape[1] < 65536:
        indices_dtype = np.uint16
    else:
        indices_dtype = np.uint32
    
    if sparse_matrix.shape[0] + 1 < 256:
        indptr_dtype = np.uint8
    elif sparse_matrix.shape[0] + 1 < 65536:
        indptr_dtype = np.uint16
    else:
        indptr_dtype = np.uint32
    
    # Convert indices and indptr to smaller types if possible
    sparse_matrix.indices = sparse_matrix.indices.astype(indices_dtype)
    sparse_matrix.indptr = sparse_matrix.indptr.astype(indptr_dtype)
    
    # Determine if data can be stored as float32 instead of float64
    if sparse_matrix.data.dtype == np.float64:
        sparse_matrix.data = sparse_matrix.data.astype(np.float32)
    
    # Calculate final memory usage
    final_size = sum(x.nbytes for x in [sparse_matrix.data, sparse_matrix.indices, sparse_matrix.indptr]) / (1024 * 1024)  # MB
    reduction = 100 * (initial_size - final_size) / initial_size if initial_size > 0 else 0
    
    logger.info(f"Sparse matrix memory usage: {initial_size:.2f} MB → {final_size:.2f} MB ({reduction:.1f}% reduction)")
    
    return sparse_matrix
