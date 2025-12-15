import pandas as pd
import polars as pl
from pyspark.sql import DataFrame as SparkDataFrame

from .utils_doc import (
    print_emphasized, 
    print_with_indent, 
    display_dataframe_in_chunks
)


def smart_read_csv(file_path, **kwargs):
    """
    Read a CSV/TSV file with automatic separator detection.
    
    Parameters
    ----------
    file_path : str
        Path to the file to read
    **kwargs : dict
        Additional arguments to pass to the reader function
        
    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        Loaded DataFrame
    """
    # Try reading first few lines to detect separator
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
    
    # Guess the separator by counting occurrences
    potential_separators = [',', '\t', ';', '|']
    counts = {sep: first_line.count(sep) for sep in potential_separators}
    likely_separator = max(counts.items(), key=lambda x: x[1])[0]
    
    # Default to comma if no clear separator found
    separator = likely_separator if counts[likely_separator] > 0 else ','
    
    # Use appropriate library based on file extension or kwargs
    use_polars = kwargs.pop('use_polars', True)
    
    if use_polars:
        # Set default schema overrides for genomic data - chromosomes should be strings
        schema_overrides = kwargs.pop('schema_overrides', {})
        
        # Ensure chromosome columns are always read as strings
        for chrom_col in ['chrom', 'seqname', 'chromosome']:
            if chrom_col not in schema_overrides:
                schema_overrides[chrom_col] = pl.Utf8
        
        return pl.read_csv(file_path, separator=separator, schema_overrides=schema_overrides, **kwargs)
    else:
        return pd.read_csv(file_path, sep=separator, **kwargs)


def get_n_unique(df, column_name):
    """
    Get the number of unique values in a specified column.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - column_name (str): The column name for which to get the number of unique values.

    Returns:
    - int: The number of unique values in the specified column.
    """
    if isinstance(df, pd.DataFrame):
        return df[column_name].nunique()
    elif isinstance(df, pl.DataFrame):
        return df[column_name].n_unique()
    else:
        raise TypeError("Input must be a Pandas or Polars DataFrame")


def get_unique_values(df, column_name):
    """
    Get the unique values in a specified column.
    """
    if isinstance(df, pd.DataFrame):
        # For pandas DataFrame
        return df[column_name].unique()
    elif isinstance(df, pl.DataFrame):
        # For Polars DataFrame
        return df.select(column_name).unique().to_series().to_list()
    else:
        raise TypeError("Input must be a Pandas or Polars DataFrame")


def drop_columns(df, columns):
    """
    Drop specified columns from a DataFrame.

    Parameters:
    - df (pd.DataFrame, pl.DataFrame, or SparkDataFrame): The input DataFrame.
    - columns (list): List of columns to drop.

    Returns:
    - pd.DataFrame, pl.DataFrame, or SparkDataFrame: The DataFrame with specified columns dropped.
    """
    if isinstance(df, pd.DataFrame):
        return df.drop(columns=columns)
    elif isinstance(df, pl.DataFrame):
        return df.drop(columns)
    elif isinstance(df, SparkDataFrame):
        return df.drop(*columns)
    else:
        raise ValueError("Unsupported DataFrame type")


def join_and_remove_duplicates(df1, df2, on, how: str = "inner", verbose: int = 1):
    """
    Join two dataframes on specified columns and remove duplicate columns in the latter dataframe.

    Parameters:
    - df1 (pd.DataFrame, pl.DataFrame, or SparkDataFrame): The first dataframe.
    - df2 (pd.DataFrame, pl.DataFrame, or SparkDataFrame): The second dataframe.
    - on (list): List of columns to join on.
    - how (str): Type of join to perform ('inner', 'left', 'right', 'outer').
    - verbose (int): Verbosity level. If > 0, print information about incorporated columns.

    Returns:
    - pd.DataFrame, pl.DataFrame, or SparkDataFrame: The combined dataframe with duplicates removed.
    """

    import pandas as pd

    # ------------------------------------------------------------------
    # Polars branch (preferred) ----------------------------------------
    # ------------------------------------------------------------------
    if isinstance(df1, pl.DataFrame):
        # Ensure df2 is also Polars for an efficient join
        if isinstance(df2, pd.DataFrame):
            df2 = pl.from_pandas(df2)
        elif not isinstance(df2, pl.DataFrame):
            raise TypeError("df2 must be DataFrame-like (pandas or polars)")

        suffix = "_dup"
        join_how = {
            "inner": "inner",
            "left": "left",
            "right": "right",
            "outer": "outer",
        }[how]

        combined_df = df1.join(df2, on=on, how=join_how, suffix=suffix)

        # Drop the duplicate columns coming from df2
        dup_cols = [c for c in combined_df.columns if c.endswith(suffix)]
        combined_df = combined_df.drop(dup_cols)

        # Logging
        if verbose > 0:
            incorporated = [c for c in df2.columns if c not in on and c in combined_df.columns]
            print(f"[info] Columns from df2 incorporated into df1: {incorporated}")

        return combined_df

    # ------------------------------------------------------------------
    # Pandas fallback ---------------------------------------------------
    # ------------------------------------------------------------------
    if isinstance(df1, pd.DataFrame):
        if isinstance(df2, pl.DataFrame):
            df2 = df2.to_pandas()
        suffix = "_dup"
        combined_df = df1.merge(df2, on=on, how=how, suffixes=("", suffix))
        dup_cols = [c for c in combined_df.columns if c.endswith(suffix)]
        combined_df.drop(columns=dup_cols, inplace=True)

        if verbose > 0:
            incorporated = [c for c in df2.columns if c not in on and c in combined_df.columns]
            print(f"[info] Columns from df2 incorporated into df1: {incorporated}")

        return combined_df

    # ------------------------------------------------------------------
    # Spark fallback (unchanged) ---------------------------------------
    # ------------------------------------------------------------------
    if isinstance(df1, SparkDataFrame):
        # Convert both to pandas – spark joins are heavyweight in driver
        df1_pd = df1.toPandas()
        df2_pd = df2.toPandas() if isinstance(df2, SparkDataFrame) else (
            df2.to_pandas() if isinstance(df2, pl.DataFrame) else df2
        )
        suffix = "_dup"
        combined_pd = df1_pd.merge(df2_pd, on=on, how=how, suffixes=("", suffix))
        dup_cols = [c for c in combined_pd.columns if c.endswith(suffix)]
        combined_pd.drop(columns=dup_cols, inplace=True)
        if verbose > 0:
            incorporated = [c for c in df2_pd.columns if c not in on and c in combined_pd.columns]
            print(f"[info] Columns from df2 incorporated into df1: {incorporated}")
        return SparkDataFrame.from_pandas(combined_pd)

    raise TypeError("Unsupported DataFrame type for df1")


def is_empty(df):
    return is_dataframe_empty(df)


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


def concatenate_dataframes(df1, df2, axis=0, verbose=1):
    """
    Concatenate two DataFrames, ensuring consistent column order for row-wise concatenation.

    Parameters:
    - df1 (pd.DataFrame or pl.DataFrame): The first DataFrame.
    - df2 (pd.DataFrame or pl.DataFrame): The second DataFrame.
    - axis (int): The axis to concatenate along (0 for row-wise, 1 for column-wise).
    - verbose (int): Verbosity level for logging.

    Returns:
    - pd.DataFrame or pl.DataFrame: The concatenated DataFrame.
    """
    if verbose > 0:
        print(f"[info] Concatenating DataFrames along axis {axis}")
        print(f"Columns in df1: {df1.columns}")
        print(f"Columns in df2: {df2.columns}")

    if axis == 0:  # Row-wise concatenation
        if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
            # Align column order for Pandas DataFrames
            df2 = df2[df1.columns] if list(df1.columns) != list(df2.columns) else df2
            return pd.concat([df1, df2], axis=axis)
        elif isinstance(df1, pl.DataFrame) and isinstance(df2, pl.DataFrame):
            # Align column order for Polars DataFrames
            if list(df1.columns) != list(df2.columns):
                df2 = df2.select(df1.columns)
            return df1.vstack(df2)  # vstack is sensitive to column order
        else:
            raise ValueError("Both DataFrames must be of the same type (either Pandas or Polars)")

    elif axis == 1:  # Column-wise concatenation
        if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
            return pd.concat([df1, df2], axis=axis)
        elif isinstance(df1, pl.DataFrame) and isinstance(df2, pl.DataFrame):
            return df1.hstack(df2)
        else:
            raise ValueError("Both DataFrames must be of the same type (either Pandas or Polars)")
    else:
        raise ValueError("Unsupported axis value")


def concatenate_dataframes_v1(dfs, axis=0, ignore_index=True):
    """
    Concatenate a list of DataFrames vertically or horizontally.

    Parameters:
    - dfs (list): List of DataFrames to concatenate.
    - axis (int): 0 for vertical (default), 1 for horizontal concatenation.
    - ignore_index (bool): Whether to ignore the index for vertical concatenation.

    Returns:
    - pd.DataFrame or pl.DataFrame: The concatenated DataFrame.
    """
    if all(isinstance(df, pd.DataFrame) for df in dfs):
        return pd.concat(dfs, axis=axis, ignore_index=(ignore_index and axis == 0))
    elif all(isinstance(df, pl.DataFrame) for df in dfs):
        how = "vertical" if axis == 0 else "horizontal"
        return pl.concat(dfs, how=how)
    else:
        raise ValueError("All DataFrames must be of the same type (Pandas or Polars).")


def subsample_dataframe(df, columns=None, num_rows=10, random=False):
    """
    Subset a DataFrame with specified columns and display a specified number of rows.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - columns (list): List of columns to subset.
    - num_rows (int): Number of rows to display (default is 10).
    - random (bool): If True, display random rows. If False, display top rows (default is False).

    Returns:
    - pd.DataFrame or pl.DataFrame: The subset DataFrame.
    """
    if columns is None:
        columns = df.columns

    if isinstance(df, pd.DataFrame):
        subset_df = df[columns]
        if random:
            subset_df = subset_df.sample(n=num_rows)
        else:
            subset_df = subset_df.head(num_rows)
    elif isinstance(df, pl.DataFrame):
        subset_df = df.select(columns)
        if random:
            subset_df = subset_df.sample(n=num_rows)
        else:
            subset_df = subset_df.head(num_rows)
    else:
        raise ValueError("Unsupported DataFrame type")

    return subset_df

def subset_dataframe_by_conditions(df, conditions):
    return subset_dataframe(df, conditions)


def subset_dataframe(df, conditions):
    """
    Subset a DataFrame based on column-specific conditions.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - conditions (dict): A dictionary where keys are column names and values are sets of allowed values.

    Returns:
    - pd.DataFrame or pl.DataFrame: The subset DataFrame.
    """
    is_polars = isinstance(df, pl.DataFrame)

    if is_polars:
        for col, values in conditions.items():
            df = df.filter(pl.col(col).is_in(values))
    elif isinstance(df, pd.DataFrame):
        for col, values in conditions.items():
            df = df[df[col].isin(values)]
    else:
        raise ValueError("Unsupported DataFrame type")

    return df


def estimate_dataframe_size(df, file_format='csv'):
    """
    Estimate the disk size of a DataFrame before saving it.
    
    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The DataFrame for which size is estimated.
    - file_format (str): File format to estimate size ('csv' or 'parquet'). Default is 'csv'.

    Returns:
    - size_in_mb (float): Estimated disk size in MB.
    """
    import tempfile
    import os
    
    # Create a temporary file to save the DataFrame
    with tempfile.NamedTemporaryFile(delete=True, suffix=f".{file_format}") as tmp_file:
        file_path = tmp_file.name

        # Save the DataFrame temporarily
        if isinstance(df, pd.DataFrame):
            if file_format == 'csv':
                df.to_csv(file_path, index=False)
            elif file_format == 'tsv':
                df.to_csv(file_path, index=False, sep='\t')
            elif file_format == 'parquet':
                df.to_parquet(file_path, index=False)
        elif isinstance(df, pl.DataFrame):
            if file_format == 'csv':
                df.write_csv(file_path)
            elif file_format == 'tsv':
                df.write_csv(file_path, separator='\t')
            elif file_format == 'parquet':
                df.write_parquet(file_path)

        # Get the size of the file in MB
        size_in_bytes = os.path.getsize(file_path)
        size_in_mb = size_in_bytes / (1024 * 1024)
    
    return size_in_mb


def estimate_memory_usage(df):
    """
    Estimate the memory usage of a DataFrame in MB.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The DataFrame to estimate memory usage for.

    Returns:
    - memory_in_mb (float): Estimated memory usage in MB.
    """
    if isinstance(df, pd.DataFrame):
        # Pandas DataFrame: Use memory_usage(deep=True) for accurate size estimation
        memory_in_bytes = df.memory_usage(deep=True).sum()
    elif isinstance(df, pl.DataFrame):
        # Polars DataFrame: Use estimated_size() method
        memory_in_bytes = df.estimated_size()
    else:
        raise ValueError("Input must be a Pandas or Polars DataFrame.")
    
    # Convert bytes to megabytes
    memory_in_mb = memory_in_bytes / (1024 * 1024)
    return memory_in_mb


def analyze_dataframe_properties(df, title="DataFrame Properties"):
    """
    Analyze and display properties of a DataFrame, automatically detecting whether it's
    a Polars or Pandas DataFrame.

    Parameters
    ----------
    df : polars.DataFrame or pandas.DataFrame
        The DataFrame to analyze
    title : str, optional
        Title to display before the analysis (default: "DataFrame Properties")

    Returns
    -------
    dict
        Dictionary containing the analyzed properties
    """
    import polars as pl
    import pandas as pd
    
    print_emphasized(f"[analysis] {title}")
    
    # Determine DataFrame type
    if isinstance(df, pl.DataFrame):
        df_type = "Polars"
        total_rows = df.shape[0]
        null_counts = {col: df[col].null_count() for col in df.columns}
        properties = {
            "Type": df_type,
            "Shape": df.shape,
            "Columns": df.columns,
            "Data Types": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            "Memory Usage": df.estimated_size(),  # in bytes
            "Null Count": {
                col: {
                    'count': count,
                    'percentage': (count / total_rows * 100) if total_rows > 0 else 0
                } for col, count in null_counts.items()
            },
            "Sample": df.head(5)
        }
        
        # Basic statistics for numeric columns
        # Polars numeric types include Float32, Float64, Int32, Int64, etc.
        numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) 
                       if any(t in str(dtype).lower() for t in ['float', 'int', 'decimal'])]
        if numeric_cols:
            properties["Numeric Statistics"] = df.select(numeric_cols).describe()
            
    elif isinstance(df, pd.DataFrame):
        df_type = "Pandas"
        total_rows = df.shape[0]
        null_counts = df.isnull().sum()
        properties = {
            "Type": df_type,
            "Shape": df.shape,
            "Columns": df.columns.tolist(),
            "Data Types": df.dtypes.to_dict(),
            "Memory Usage": df.memory_usage(deep=True).sum(),  # in bytes
            "Null Count": {
                col: {
                    'count': count,
                    'percentage': (count / total_rows * 100) if total_rows > 0 else 0
                } for col, count in null_counts.items()
            },
            "Sample": df.head(5),
            "Numeric Statistics": df.describe()
        }
    else:
        raise ValueError("Input must be either a Polars or Pandas DataFrame")

    # Display properties
    print_with_indent(f"DataFrame Type: {df_type}", indent_level=1)
    print_with_indent(f"Shape: {properties['Shape']}", indent_level=1)
    print_with_indent("Columns:", indent_level=1)
    for col in properties['Columns']:
        print_with_indent(f"- {col}: {properties['Data Types'][col]}", indent_level=2)
    
    print_with_indent("Memory Usage:", indent_level=1)
    memory_mb = properties['Memory Usage'] / (1024 * 1024)  # Convert to MB
    print_with_indent(f"- {memory_mb:.2f} MB", indent_level=2)
    
    print_with_indent("Null Values:", indent_level=1)
    for col, null_info in properties['Null Count'].items():
        count = null_info['count']
        percentage = null_info['percentage']
        if count > 0:  # Only show columns with null values
            print_with_indent(
                f"- {col}: {count:,} nulls ({percentage:.2f}% of total)", 
                indent_level=2)
    
    print_with_indent("Sample Data:", indent_level=1)
    display_dataframe_in_chunks(properties['Sample'], num_rows=5, num_columns=5)
    
    if "Numeric Statistics" in properties:
        print_with_indent("Numeric Statistics:", indent_level=1)
        display_dataframe_in_chunks(properties['Numeric Statistics'], num_rows=5, num_columns=5)
    
    return properties


def demo_join_and_remove_duplicates():

    # Example Input DataFrames
    data1 = {
        "gene_id": ["gene1", "gene1", "gene2"],
        "transcript_id": ["tx1", "tx1", "tx2"],
        "splice_type": ["acceptor", "donor", "acceptor"],
        "position": [100, 200, 150],
    }

    data2 = {
        "gene_id": ["gene1", "gene2", "gene3"],
        "transcript_id": ["tx1", "tx2", "tx3"],
        "transcript_length": [1000, 900, 800],
        "gc_content": [0.45, 0.42, 0.38],
    }

    # Create Pandas DataFrames
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    # Perform the join
    augmented_df = join_and_remove_duplicates(df1, df2, on=["gene_id", "transcript_id"], how='left', verbose=1)
    print(augmented_df)



def demo(): 

    demo_join_and_remove_duplicates()

    return


def is_pandas_dataframe(df):
    """
    Check if a dataframe is a pandas DataFrame.
    
    Parameters
    ----------
    df : Any
        The dataframe to check
        
    Returns
    -------
    bool
        True if pandas DataFrame, False otherwise
    """
    return isinstance(df, pd.DataFrame)


def filter_dataframe(df, column, values):
    """
    Filter a dataframe by column values, works with either pandas or polars.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        The dataframe to filter
    column : str
        Column name to filter on
    values : list
        List of values to include
        
    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Filtered dataframe (same type as input)
    """
    if is_pandas_dataframe(df):
        return df[df[column].isin(values)]
    else:
        # Assume polars
        return df.filter(pl.col(column).is_in(values))


def get_row_count(df):
    """
    Get the number of rows in a dataframe, works with either pandas or polars.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        The dataframe to count rows for
        
    Returns
    -------
    int
        Number of rows
    """
    if is_pandas_dataframe(df):
        return len(df)
    else:
        # Assume polars
        return df.shape[0]


def get_shape(df):
    """
    Get the shape (rows, columns) of a dataframe, works with either pandas or polars.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        The dataframe to get shape of
        
    Returns
    -------
    tuple
        Tuple of (rows, columns)
    """
    return df.shape


def get_first_row(df):
    """
    Get the first row of a dataframe, works with either pandas or polars.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        The dataframe to get first row from
        
    Returns
    -------
    Any
        First row or None if empty
    """
    if get_row_count(df) == 0:
        return None
        
    if is_pandas_dataframe(df):
        return df.iloc[0]
    else:
        # Assume polars
        return df.head(1)


def has_column(df, column):
    """
    Check if a dataframe has a specific column.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        The dataframe to check
    column : str
        Column name to check
        
    Returns
    -------
    bool
        True if column exists, False otherwise
    """
    return column in df.columns


if __name__ == "__main__":
    demo()
