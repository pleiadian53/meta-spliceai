#!/usr/bin/env python3
"""Memory-efficient data loading for meta-model training.

This module provides functions to load and process datasets in chunks,
greatly reducing memory footprint for large datasets.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Generator

import numpy as np
import pandas as pd
from scipy import sparse

from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _preprocess_features_for_model

# Import scalability utilities if available
try:
    from meta_spliceai.splice_engine.meta_models.training import scalability_utils
    SCALABILITY_UTILS_AVAILABLE = True
except ImportError:
    SCALABILITY_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_dataset_chunked(
    dataset_path: Union[str, Path],
    *,
    row_cap: int = 0,
    chunksize: int = 10000,
    feature_subset: Optional[List[str]] = None,
    random_state: int = 42,
    required_cols: Optional[List[str]] = None,
    optimize_memory: bool = True,
    use_sparse_kmers: bool = True,
    kmer_prefix: str = '6mer_',
) -> Tuple[pd.DataFrame, List[str]]:
    """Load dataset in memory-efficient chunks and return a combined DataFrame.
    
    This is a wrapper around scalability_utils.load_dataset_chunked that
    provides a similar API to the original datasets.load_dataset function.
    
    Parameters
    ----------
    dataset_path : str or Path
        Path to dataset file or directory with Parquet files
    row_cap : int, default=0
        Maximum number of rows to load (0 = no limit)
    chunksize : int, default=10000
        Number of rows to load per chunk
    feature_subset : List[str], optional
        Only load these specific columns (features)
    random_state : int, default=42
        Random state for reproducibility when sampling
    required_cols : List[str], optional
        Additional required columns to always load
    optimize_memory : bool, default=True
        Apply memory optimization to numeric columns
    use_sparse_kmers : bool, default=True
        Convert k-mer features to sparse format
    kmer_prefix : str, default='6mer_'
        Prefix for identifying k-mer features
        
    Returns
    -------
    tuple
        (DataFrame with processed features, list of feature names)
    """
    if not SCALABILITY_UTILS_AVAILABLE:
        logger.warning("Scalability utilities not available, falling back to standard loader")
        return datasets.load_dataset(dataset_path, row_cap=row_cap, random_state=random_state)
    
    # Set up required columns if not provided
    if required_cols is None:
        required_cols = ['splice_type', 'chrom', 'gene_id', 'transcript_id']
    
    # If max_rows is 0, use None to indicate no limit
    max_rows = row_cap if row_cap > 0 else None
    
    # Generate sample fraction if row_cap is specified
    sample_frac = None
    if row_cap > 0:
        # Try to estimate dataset size to calculate sampling fraction
        try:
            # Use the get_dataset_size function from datasets module if available
            if hasattr(datasets, 'get_dataset_size'):
                total_rows = datasets.get_dataset_size(dataset_path)
                if total_rows > row_cap:
                    sample_frac = row_cap / total_rows
                    logger.info(f"Sampling {sample_frac:.4f} of rows to meet row cap of {row_cap}")
        except Exception as e:
            logger.warning(f"Error estimating dataset size: {e}. Will use max_rows instead.")
    
    # Use chunked loading
    chunks = []
    for chunk_idx, chunk in enumerate(scalability_utils.load_dataset_chunked(
        dataset_path,
        chunksize=chunksize,
        max_rows=max_rows,
        sample_frac=sample_frac,
        random_state=random_state,
        feature_subset=feature_subset,
        required_cols=required_cols,
    )):
        # Process each chunk
        logger.info(f"Processing chunk {chunk_idx+1}")
        
        # Map original label column (splice_type) to label for compatibility
        if "splice_type" in chunk.columns and "label" not in chunk.columns:
            logger.info("Mapping 'splice_type' column to 'label' for compatibility")
            # Encode string labels to numeric values
            label_mapping = {
                'donor': 0,      # Donor sites
                'acceptor': 1,   # Acceptor sites
                '0': 2,          # Neither (current format)
                'neither': 2,    # Neither (possible future format)
                0: 2            # Handle numeric values too
            }
            # Apply the mapping
            chunk["label"] = chunk["splice_type"].map(label_mapping).astype(int)
            logger.info(f"Encoded labels distribution: {chunk['label'].value_counts().to_dict()}")
            
            # Check for unmapped labels
            if chunk["label"].isna().any():
                unmapped = chunk["splice_type"][chunk["label"].isna()].unique()
                logger.warning(f"Found unmapped labels: {unmapped}. Using default value 2.")
                chunk["label"] = chunk["label"].fillna(2).astype(int)
        
        # Apply memory optimization if requested
        if optimize_memory:
            chunk = scalability_utils.optimize_memory_usage(chunk)
        
        chunks.append(chunk)
        
        # Clear memory between chunks
        gc.collect()
    
    # Combine chunks
    if not chunks:
        raise ValueError("No data loaded")
    
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Combined dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Get feature names (columns except special ones)
    feature_names = [
        col for col in df.columns
        if col not in ['label', 'chrom', 'gene_id', 'transcript_id']
    ]
    
    # Preprocess features
    df = _preprocess_features_for_model(df, feature_names)
    # Feature names remain the same after preprocessing
    
    return df, feature_names


def load_and_preprocess_sparse_dataset(
    dataset_path: Union[str, Path],
    *,
    row_cap: int = 0,
    chunksize: int = 10000,
    feature_subset: Optional[List[str]] = None,
    random_state: int = 42,
    required_cols: Optional[List[str]] = None,
    kmer_prefix: str = '6mer_',
) -> Tuple[
    sparse.csr_matrix,  # X_kmer (sparse k-mer features)
    pd.DataFrame,       # X_non_kmer (dense non-k-mer features)
    List[str],          # kmer_feature_names
    List[str],          # non_kmer_feature_names
    np.ndarray,         # y (labels)
    Dict                # metadata (chrom, gene_id, etc.)
]:
    """Load dataset with separate sparse k-mer and dense non-k-mer features.
    
    This provides the most memory-efficient way to load large datasets.
    
    Parameters
    ----------
    dataset_path : str or Path
        Path to dataset file or directory with Parquet files
    row_cap : int, default=0
        Maximum number of rows to load (0 = no limit)
    chunksize : int, default=10000
        Number of rows to load per chunk
    feature_subset : List[str], optional
        Only load these specific columns (features)
    random_state : int, default=42
        Random state for reproducibility when sampling
    required_cols : List[str], optional
        Additional required columns to always load
    kmer_prefix : str, default='6mer_'
        Prefix for identifying k-mer features
        
    Returns
    -------
    tuple
        (X_kmer, X_non_kmer, kmer_feature_names, non_kmer_feature_names, y, metadata)
    """
    if not SCALABILITY_UTILS_AVAILABLE:
        logger.warning("Scalability utilities not available, falling back to standard loader")
        df, feature_names = datasets.load_dataset(dataset_path, row_cap=row_cap, random_state=random_state)
        # Basic processing without sparse conversion
        y = df['label'].values
        metadata = {
            'chrom': df.get('chrom', None),
            'gene_id': df.get('gene_id', None),
            'transcript_id': df.get('transcript_id', None),
        }
        return (None, df[feature_names], [], feature_names, y, metadata)
    
    # Set up required columns if not provided
    if required_cols is None:
        required_cols = ['splice_type', 'chrom', 'gene_id', 'transcript_id']
    
    # If max_rows is 0, use None to indicate no limit
    max_rows = row_cap if row_cap > 0 else None
    
    # Use chunked loading to get initial feature list
    logger.info("Loading first chunk to identify features")
    chunk_gen = scalability_utils.load_dataset_chunked(
        dataset_path,
        chunksize=min(chunksize, 1000),  # Use smaller chunk just to get feature names
        max_rows=1000,  # Only need a small sample to identify features
        random_state=random_state,
        feature_subset=feature_subset,
        required_cols=required_cols,
    )
    first_chunk = next(chunk_gen)
    
    # Process first chunk to get feature names
    _, all_feature_names = _preprocess_features_for_model(first_chunk, 
        [col for col in first_chunk.columns if col not in required_cols]
    )
    
    # Identify k-mer and non-k-mer features
    kmer_features = [f for f in all_feature_names if f.startswith(kmer_prefix)]
    non_kmer_features = [f for f in all_feature_names if not f.startswith(kmer_prefix)]
    
    logger.info(f"Identified {len(kmer_features)} k-mer features and {len(non_kmer_features)} non-k-mer features")
    
    # Initialize data structures for combining chunks
    kmer_chunks = []
    non_kmer_chunks = []
    labels = []
    metadata_chunks = {col: [] for col in required_cols if col != 'label'}
    
    # Now load full dataset in chunks and process
    chunk_gen = scalability_utils.load_dataset_chunked(
        dataset_path,
        chunksize=chunksize,
        max_rows=max_rows,
        random_state=random_state,
        feature_subset=list(set(all_feature_names + required_cols)),
        required_cols=required_cols,
    )
    
    for chunk_idx, chunk in enumerate(chunk_gen):
        logger.info(f"Processing chunk {chunk_idx+1}")
        
        # Preprocess features for this chunk
        chunk, _ = _preprocess_features_for_model(chunk, all_feature_names)
        
        # Store labels
        labels.append(chunk['label'].values)
        
        # Store metadata
        for col in metadata_chunks:
            if col in chunk:
                metadata_chunks[col].append(chunk[col].values)
        
        # Convert k-mer features to sparse matrix
        if kmer_features:
            kmer_matrix = sparse.csr_matrix(chunk[kmer_features].values)
            kmer_chunks.append(kmer_matrix)
        
        # Store non-k-mer features
        if non_kmer_features:
            non_kmer_df = scalability_utils.optimize_memory_usage(chunk[non_kmer_features])
            non_kmer_chunks.append(non_kmer_df)
        
        # Clear memory
        del chunk
        gc.collect()
    
    # Combine all chunks
    y = np.concatenate(labels)
    
    # Combine k-mer sparse matrices
    if kmer_features and kmer_chunks:
        X_kmer = sparse.vstack(kmer_chunks)
        X_kmer = scalability_utils.optimize_sparse_matrix(X_kmer)
    else:
        X_kmer = sparse.csr_matrix((len(y), 0))
    
    # Combine non-k-mer DataFrames
    if non_kmer_features and non_kmer_chunks:
        X_non_kmer = pd.concat(non_kmer_chunks, ignore_index=True)
    else:
        X_non_kmer = pd.DataFrame(index=range(len(y)))
    
    # Combine metadata
    metadata = {}
    for col in metadata_chunks:
        if metadata_chunks[col]:
            metadata[col] = np.concatenate(metadata_chunks[col])
    
    logger.info(f"Loaded dataset with {len(y)} samples")
    logger.info(f"X_kmer shape: {X_kmer.shape}, X_non_kmer shape: {X_non_kmer.shape}")
    
    # Clean up to free memory
    del kmer_chunks, non_kmer_chunks, labels, metadata_chunks
    gc.collect()
    
    return (X_kmer, X_non_kmer, kmer_features, non_kmer_features, y, metadata)
