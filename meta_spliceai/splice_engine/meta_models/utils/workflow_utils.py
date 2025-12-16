"""
Workflow utilities for meta_models package.

Provides common workflow functions for formatting, printing, timing, and other
utilities specific to the meta_models package workflows.
"""

import time
from typing import Any, Callable, Dict, Optional, TypeVar, cast
from rich.console import Console
from rich.panel import Panel

# Initialize rich console for prettier output
console = Console()

# Type variable for generic function signatures
T = TypeVar('T')


def print_with_indent(message: str, indent_level: int = 1, indent_size: int = 4) -> None:
    """
    Print a message with consistent indentation.
    
    Parameters
    ----------
    message : str
        The message to print
    indent_level : int, default=0
        Number of indentation levels
    indent_size : int, default=4
        Number of spaces per indentation level
    """
    indent = ' ' * (indent_level * indent_size)
    console.print(f"{indent}{message}")


def print_section_separator() -> None:
    """Print a visually distinct section separator line."""
    console.print(Panel("-" * 85, style="bold"))


def print_emphasized(text: str, style: str = 'bold', edge_effect: bool = True, 
                     symbol: str = '=') -> None:
    """
    Print text with emphasis using styling and optional edge decorations.
    
    Parameters
    ----------
    text : str
        The text to emphasize
    style : str, default='bold'
        Style to apply ('bold', 'underline', 'red', 'green', etc.)
    edge_effect : bool, default=True
        Whether to add a line of symbols above and below the text
    symbol : str, default='='
        Symbol to use for edge effect lines
    """
    styles = {
        'bold': '\033[1m',
        'underline': '\033[4m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m'
    }
    end_style = '\033[0m'  # Reset to default style

    if edge_effect:
        edge_line = symbol * len(text)
        print(edge_line)
        print(f"{styles.get(style, '')}{text}{end_style}")
        print(edge_line)
    else:
        print(f"{styles.get(style, '')}{text}{end_style}")


def time_function(func: Callable[..., T]) -> Callable[..., Dict[str, Any]]:
    """
    Decorator for timing function execution and capturing results.
    
    Parameters
    ----------
    func : Callable
        The function to time
        
    Returns
    -------
    Callable
        Wrapped function that returns timing info along with original results
        
    Examples
    --------
    @time_function
    def process_data(data):
        # processing logic
        return result
        
    timed_result = process_data(my_data)
    print(f"Processing took {timed_result['execution_time']:.2f} seconds")
    result = timed_result['result']
    """
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        # Record start time
        start_time = time.time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Record end time
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Package results with timing info
        if isinstance(result, dict):
            # If result is already a dict, add timing info to it
            result['execution_time'] = execution_time
            return result
        else:
            # Otherwise, wrap the result in a dict with timing info
            return {
                'result': result,
                'execution_time': execution_time
            }
    
    return wrapper


def align_and_append(df1, df2, on_columns=None, how='inner', 
                    suffixes=('', '_right'), verbosity=1):
    """
    Align two dataframes on specific columns and append df2 to df1.
    
    This is a utility function for consistently merging dataframes in the
    meta-models workflow, with appropriate error handling and verbosity.
    
    Parameters
    ----------
    df1 : pd.DataFrame or pl.DataFrame
        First dataframe
    df2 : pd.DataFrame or pl.DataFrame
        Second dataframe to append to df1
    on_columns : list, default=None
        List of column names to align on, if None will use index
    how : str, default='inner'
        Type of join to perform ('inner', 'left', 'right', 'outer')
    suffixes : tuple, default=('', '_right')
        Suffixes to use for overlapping columns
    verbosity : int, default=1
        Controls output detail level
        
    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Combined dataframe
    """
    from meta_spliceai.splice_engine.meta_models.utils.dataframe_utils import (
        is_pandas_dataframe, is_polars_dataframe
    )
    
    if on_columns is None:
        if verbosity >= 2:
            print_with_indent("No alignment columns specified, using indices", indent_level=1)
        # Default behavior depends on dataframe type
        if is_pandas_dataframe(df1):
            import pandas as pd
            return pd.concat([df1, df2], axis=0)
        elif is_polars_dataframe(df1):
            return df1.vstack(df2)
        else:
            raise TypeError("Unsupported dataframe type")
    
    if verbosity >= 2:
        print_with_indent(f"Aligning dataframes on columns: {on_columns}", indent_level=1)
        print_with_indent(f"Join type: {how}", indent_level=1)
    
    # Handle pandas dataframes
    if is_pandas_dataframe(df1) and is_pandas_dataframe(df2):
        return df1.merge(df2, on=on_columns, how=how, suffixes=suffixes)
    
    # Handle polars dataframes
    if is_polars_dataframe(df1) and is_polars_dataframe(df2):
        return df1.join(df2, on=on_columns, how=how)
    
    # Mixed dataframes - convert to pandas for consistency
    if verbosity >= 1:
        print_with_indent("Warning: Mixed dataframe types, converting to pandas", indent_level=1)
    
    if is_polars_dataframe(df1):
        df1 = df1.to_pandas()
    if is_polars_dataframe(df2):
        df2 = df2.to_pandas()
    
    return df1.merge(df2, on=on_columns, how=how, suffixes=suffixes)
