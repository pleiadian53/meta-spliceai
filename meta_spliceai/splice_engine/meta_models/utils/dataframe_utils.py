"""
Utilities for working with dataframes in either Pandas or Polars format.
Provides a unified interface to common operations regardless of the underlying dataframe type.
"""

import pandas as pd
import polars as pl
from typing import Any, List, Union, Tuple, Optional


def is_dataframe_empty(df):
    """
    Check if a DataFrame is empty.

    Parameters:
    - df (pd.DataFrame, pl.DataFrame, or SparkDataFrame): The input DataFrame.

    Returns:
    - bool: True if the DataFrame is empty, False otherwise.
    """
    if isinstance(df, pd.DataFrame):
        return df.empty
    elif isinstance(df, pl.DataFrame):
        return df.is_empty()
    elif isinstance(df, SparkDataFrame):
        return df.rdd.isEmpty()
    else:
        raise ValueError("Unsupported DataFrame type")


def is_pandas_dataframe(df: Any) -> bool:
    """
    Check if a dataframe is a pandas DataFrame.
    
    Args:
        df: The dataframe to check
        
    Returns:
        True if pandas DataFrame, False otherwise
    """
    return isinstance(df, pd.DataFrame)


def filter_dataframe(df: Any, column: str, values: List[Any]) -> Any:
    """
    Filter a dataframe by column values, works with either pandas or polars.
    
    Args:
        df: Pandas or Polars dataframe
        column: Column name to filter on
        values: List of values to include
        
    Returns:
        Filtered dataframe (same type as input)
    """
    if is_pandas_dataframe(df):
        return df[df[column].isin(values)]
    else:
        # Assume polars
        return df.filter(pl.col(column).is_in(values))


def get_row_count(df: Any) -> int:
    """
    Get the number of rows in a dataframe, works with either pandas or polars.
    
    Args:
        df: Pandas or Polars dataframe
        
    Returns:
        Number of rows
    """
    if is_pandas_dataframe(df):
        return len(df)
    else:
        # Assume polars
        return df.shape[0]


def get_shape(df: Any) -> Tuple[int, int]:
    """
    Get the shape (rows, columns) of a dataframe, works with either pandas or polars.
    
    Args:
        df: Pandas or Polars dataframe
        
    Returns:
        Tuple of (rows, columns)
    """
    return df.shape


def get_first_row(df: Any) -> Optional[Any]:
    """
    Get the first row of a dataframe, works with either pandas or polars.
    
    Args:
        df: Pandas or Polars dataframe
        
    Returns:
        First row or None if empty
    """
    if get_row_count(df) == 0:
        return None
        
    if is_pandas_dataframe(df):
        return df.iloc[0]
    else:
        # Assume polars
        return df.head(1)


def has_column(df: Any, column: str) -> bool:
    """
    Check if a dataframe has a specific column.
    
    Args:
        df: Pandas or Polars dataframe
        column: Column name to check
        
    Returns:
        True if column exists, False otherwise
    """
    return column in df.columns
