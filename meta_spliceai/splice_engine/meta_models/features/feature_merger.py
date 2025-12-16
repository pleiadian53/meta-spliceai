"""
Feature merging utilities for meta-models.

This module provides functions for merging different feature sources
(sequence-based k-mer features, probability scores, context scores)
to create comprehensive feature sets for meta-model training.
"""

import pandas as pd
import polars as pl
from typing import Dict, List, Tuple, Optional, Union, Set, Any

from meta_spliceai.splice_engine.meta_models.io.datasets import load_splice_positions
from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
from meta_spliceai.splice_engine.utils_doc import (
    print_emphasized, print_with_indent
)


def merge_features_with_positions(
    featurized_dfs: Dict[str, Union[pd.DataFrame, pl.DataFrame]],
    data_handler: Optional[MetaModelDataHandler] = None,
    positions_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
    use_shared_dir: bool = False,
    output_subdir: Optional[str] = None,
    join_columns: List[str] = ['gene_id', 'position', 'strand', 'splice_type'],
    convert_to_pandas: bool = True,
    verbose: int = 1
) -> Dict[str, pd.DataFrame]:
    """
    Merge k-mer features with probability and context score features.
    
    This function takes k-mer featurized DataFrames and merges them with
    the rich set of features from positions_df (probability scores,
    context scores, and derived features).
    
    Parameters
    ----------
    featurized_dfs : Dict[str, Union[pd.DataFrame, pl.DataFrame]]
        Dictionary of k-mer featurized DataFrames, keyed by prediction type
    data_handler : Optional[MetaModelDataHandler], optional
        Data handler for loading positions data, by default None.
        If None and positions_df is also None, a new handler will be created.
    positions_df : Optional[Union[pd.DataFrame, pl.DataFrame]], optional
        Pre-loaded positions DataFrame, by default None
    use_shared_dir : bool, optional
        Whether to use the shared directory, by default False
    output_subdir : Optional[str], optional
        Output subdirectory, by default None
    join_columns : List[str], optional
        Columns to join on, by default ['gene_id', 'position']
    convert_to_pandas : bool, optional
        Whether to convert all results to pandas, by default True
    verbose : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of merged feature DataFrames, keyed by prediction type
    """
    # Load positions_df if not provided
    if positions_df is None:
        if verbose >= 1:
            print_emphasized("[workflow] Loading positions data for feature merging...")
            
        positions_df = load_splice_positions(
            data_handler=data_handler,
            use_shared_dir=use_shared_dir,
            output_subdir=output_subdir,
            aggregated=True,
            enhanced=True,
            verbose=verbose
        )
        
        if verbose >= 1:
            print_with_indent(f"Loaded positions data with {positions_df.shape[0]} rows and {positions_df.shape[1]} columns")
    
    # Convert positions_df to pandas if it's a polars DataFrame
    if isinstance(positions_df, pl.DataFrame):
        if verbose >= 1:
            print_with_indent("Converting positions_df from polars to pandas...")
        positions_df = positions_df.to_pandas()
    
    # Get the set of columns in positions_df to avoid duplicate columns during merge
    positions_columns = set(positions_df.columns)
    
    # Process each prediction type
    merged_dfs = {}
    for pred_type, featurized_df in featurized_dfs.items():
        if featurized_df is None or featurized_df.shape[0] == 0:
            if verbose >= 1:
                print_with_indent(f"Skipping empty {pred_type} DataFrame")
            continue
        
        # Convert to pandas if needed
        if isinstance(featurized_df, pl.DataFrame):
            featurized_df = featurized_df.to_pandas()
        
        if verbose >= 1:
            print_with_indent(f"Merging {pred_type} features ({featurized_df.shape[0]} rows) with positions data...")
        
        # Determine columns to drop during merge to avoid duplicates
        # Keep all columns from featurized_df, and all non-duplicated columns from positions_df
        featurized_columns = set(featurized_df.columns)
        
        # Get overlap columns (excluding join columns)
        overlap_columns = featurized_columns.intersection(positions_columns) - set(join_columns)
        
        if overlap_columns and verbose >= 1:
            print_with_indent(f"Found {len(overlap_columns)} overlapping columns: {overlap_columns}")
            print_with_indent("Keeping versions from featurized data")
        
        # Create a copy of positions_df with overlapping columns dropped
        positions_df_for_merge = positions_df.drop(columns=list(overlap_columns), errors='ignore')
        
        # Perform the merge
        try:
            merged_df = pd.merge(
                featurized_df, 
                positions_df_for_merge,
                on=join_columns,
                how='inner'
            )
            
            if verbose >= 1:
                print_with_indent(f"Merged DataFrame shape: {merged_df.shape}")
                print_with_indent(f"Retained {merged_df.shape[0]/featurized_df.shape[0]:.2%} of rows after merge")
            
            merged_dfs[pred_type] = merged_df
            
        except Exception as e:
            if verbose >= 1:
                print_with_indent(f"Error merging {pred_type} features: {str(e)}")
                print_with_indent(f"Join columns in featurized_df: {join_columns} present in DataFrame: {all(col in featurized_df.columns for col in join_columns)}")
                print_with_indent(f"Join columns in positions_df: {join_columns} present in DataFrame: {all(col in positions_df.columns for col in join_columns)}")
    
    return merged_dfs


def merge_kmer_with_position_features(
    kmer_result: Dict[str, Any],
    data_handler: Optional[MetaModelDataHandler] = None,
    use_shared_dir: bool = False,
    output_subdir: Optional[str] = None,
    join_columns: List[str] = ['gene_id', 'position', 'strand', 'splice_type'],
    drop_sequence: bool = True,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Merge k-mer features with position-based features for meta-model training.
    
    This function takes the result from make_kmer_featurized_dataset
    and enhances it by merging with probability scores, context scores,
    and derived features from the positions DataFrame.
    
    Parameters
    ----------
    kmer_result : Dict[str, Any]
        Result dictionary from make_kmer_featurized_dataset
    data_handler : Optional[MetaModelDataHandler], optional
        Data handler for loading positions data, by default None
    use_shared_dir : bool, optional
        Whether to use the shared directory, by default False
    output_subdir : Optional[str], optional
        Output subdirectory, by default None
    join_columns : List[str], optional
        Columns to join on, by default ['gene_id', 'position']
    drop_sequence : bool, optional
        Whether to drop the sequence column to save memory, by default True
    verbose : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    Dict[str, Any]
        Enhanced result dictionary with merged features
    """
    if verbose >= 1:
        print_emphasized("[workflow] Creating complete feature set...")
    
    # Extract components from kmer_result
    harmonized_dfs = kmer_result.get('harmonized_dfs', {})
    feature_sets = kmer_result.get('feature_sets', {})
    
    # Merge with positions data
    merged_dfs = merge_features_with_positions(
        featurized_dfs=harmonized_dfs,
        data_handler=data_handler,
        use_shared_dir=use_shared_dir,
        output_subdir=output_subdir,
        join_columns=join_columns,
        convert_to_pandas=True,
        verbose=verbose
    )
    
    # Remove sequence column if requested to save memory
    if drop_sequence:
        for pred_type, df in merged_dfs.items():
            if 'sequence' in df.columns:
                if verbose >= 1:
                    print_with_indent(f"Dropping 'sequence' column from {pred_type} DataFrame to save memory")
                merged_dfs[pred_type] = df.drop(columns=['sequence'])
    
    # Update the result dictionary
    result = kmer_result.copy()
    result['merged_dfs'] = merged_dfs
    
    # Add information about the feature counts
    feature_counts = {}
    for pred_type, df in merged_dfs.items():
        feature_counts[pred_type] = df.shape[1] - len(join_columns)  # Exclude join columns
    
    result['feature_counts'] = feature_counts
    
    if verbose >= 1:
        print_with_indent("Complete feature set created with the following feature counts:")
        for pred_type, count in feature_counts.items():
            print_with_indent(f"  {pred_type}: {count} features")
    
    return result
