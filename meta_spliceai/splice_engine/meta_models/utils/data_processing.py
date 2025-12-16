"""
Data processing utilities for meta models.

This module provides helper functions for processing and validating datasets
used in meta model training and evaluation.
"""

import pandas as pd
import polars as pl
from typing import Union, List, Tuple, Optional, Dict, Any, Set

# Import original functionality without modifying it
from meta_spliceai.splice_engine.analysis_utils import (
    check_and_subset_invalid_transcript_ids as original_check_and_subset_invalid_transcript_ids,
)
from meta_spliceai.splice_engine.splice_error_analyzer import (
    filter_and_validate_ids as original_filter_and_validate_ids,
    count_unique_ids as original_count_unique_ids,
    concatenate_dataframes as original_concatenate_dataframes,
    downsample_dataframe as original_downsample_dataframe,
    is_dataframe_empty,
    display_feature_set,
    subsample_dataframe,
    display_dataframe_in_chunks,
    print_emphasized,
    print_with_indent,
    print_section_separator
)


def check_and_subset_invalid_transcript_ids(
    df: Union[pd.DataFrame, pl.DataFrame],
    col_tid: str = 'transcript_id',
    verbose: int = 1
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Check for and extract rows with invalid transcript IDs.
    
    This is a wrapper around the original check_and_subset_invalid_transcript_ids function
    that maintains compatibility with the existing codebase.
    
    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        DataFrame to check
    col_tid : str, optional
        Column name for transcript ID, by default 'transcript_id'
    verbose : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        DataFrame with invalid transcript IDs
    """
    return original_check_and_subset_invalid_transcript_ids(df, col_tid=col_tid, verbose=verbose)


def filter_and_validate_ids(
    df: Union[pd.DataFrame, pl.DataFrame],
    col_tid: str = 'transcript_id',
    verbose: int = 1
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Filter out rows with invalid transcript IDs.
    
    This is a wrapper around the original filter_and_validate_ids function
    that maintains compatibility with the existing codebase.
    
    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        DataFrame to filter
    col_tid : str, optional
        Column name for transcript ID, by default 'transcript_id'
    verbose : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        Filtered DataFrame
    """
    return original_filter_and_validate_ids(df, col_tid=col_tid, verbose=verbose)


def count_unique_ids(
    df: Union[pd.DataFrame, pl.DataFrame],
    col_tid: str = 'transcript_id'
) -> None:
    """
    Count and print the number of unique gene and transcript IDs.
    
    This is a wrapper around the original count_unique_ids function
    that maintains compatibility with the existing codebase.
    
    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        DataFrame to analyze
    col_tid : str, optional
        Column name for transcript ID, by default 'transcript_id'
    """
    return original_count_unique_ids(df, col_tid=col_tid)


def concatenate_dataframes(
    df1: Union[pd.DataFrame, pl.DataFrame],
    df2: Union[pd.DataFrame, pl.DataFrame],
    axis: int = 0
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Concatenate two DataFrames, handling both pandas and polars.
    
    This is a wrapper around the original concatenate_dataframes function
    that maintains compatibility with the existing codebase.
    
    Parameters
    ----------
    df1 : Union[pd.DataFrame, pl.DataFrame]
        First DataFrame
    df2 : Union[pd.DataFrame, pl.DataFrame]
        Second DataFrame
    axis : int, optional
        Axis along which to concatenate, by default 0
        
    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        Concatenated DataFrame
    """
    return original_concatenate_dataframes(df1, df2, axis=axis)


def downsample_dataframe(
    df: Union[pd.DataFrame, pl.DataFrame],
    sample_fraction: float = 0.1,
    max_sample_size: int = 10000,
    verbose: int = 1
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Downsample a DataFrame to a specified fraction or size.
    
    This is a wrapper around the original downsample_dataframe function
    that maintains compatibility with the existing codebase.
    
    Parameters
    ----------
    df : Union[pd.DataFrame, pl.DataFrame]
        DataFrame to downsample
    sample_fraction : float, optional
        Fraction of rows to keep, by default 0.1
    max_sample_size : int, optional
        Maximum number of rows to keep, by default 10000
    verbose : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        Downsampled DataFrame
    """
    return original_downsample_dataframe(
        df, 
        sample_fraction=sample_fraction, 
        max_sample_size=max_sample_size, 
        verbose=verbose
    )


# Export utility functions for use in the new codebase
__all__ = [
    'check_and_subset_invalid_transcript_ids',
    'filter_and_validate_ids',
    'count_unique_ids',
    'concatenate_dataframes',
    'downsample_dataframe',
    'is_dataframe_empty',
    'display_feature_set',
    'subsample_dataframe',
    'display_dataframe_in_chunks',
    'print_emphasized',
    'print_with_indent',
    'print_section_separator'
]
