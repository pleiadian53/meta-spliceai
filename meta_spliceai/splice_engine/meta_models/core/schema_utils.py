import polars as pl
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

# Schema definitions
DEFAULT_POSITION_SCHEMA = {
    'gene_id': pl.Utf8,
    'transcript_id': pl.Utf8,  # Important: Use Utf8 type for transcript_id
    'position': pl.Int64,      # Current position being evaluated
    'predicted_position': pl.Int64,  # Predicted position (can be null for FN/TN)
    'true_position': pl.Int64,      # Actual annotated position (can be null for FP/TN)
    'pred_type': pl.Utf8,      # TP, FP, FN, or TN
    'score': pl.Float64,       # Score at this position
    'strand': pl.Utf8,         # + or -
    'donor_score': pl.Float64, # Donor probability score
    'acceptor_score': pl.Float64, # Acceptor probability score
    'neither_score': pl.Float64,  # Neither probability score
    'splice_type': pl.Utf8     # 'donor', 'acceptor', or None
}

# Error schema for error DataFrames
DEFAULT_ERROR_SCHEMA = {
    'gene_id': pl.Utf8,
    'transcript_id': pl.Utf8,
    'error_type': pl.Utf8,     # FP or FN
    'position': pl.Int64,      # Position where the error occurred
    'window_start': pl.Int64,  # Start of surrounding window
    'window_end': pl.Int64,    # End of surrounding window
    'strand': pl.Utf8,         # + or -
    'splice_type': pl.Utf8     # 'donor', 'acceptor', or None
}

# Helper function to ensure a DataFrame has all required columns with correct types
def ensure_schema(df, schema):
    """
    Ensure the DataFrame has all columns specified in the schema and in the correct order.
    If a column is missing, add it with null values.
    Returns a DataFrame with exactly the columns specified in the schema, in the same order.
    """
    # First ensure all required columns exist (add missing ones)
    for col_name, col_type in schema.items():
        if col_name not in df.columns:
            if col_type == pl.Int64:
                df = df.with_columns(pl.lit(None).cast(pl.Int64).alias(col_name))
            elif col_type == pl.Float64:
                df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col_name))
            elif col_type == pl.Utf8:
                df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col_name))
            else:
                # For other/custom types, let Polars infer the type
                df = df.with_columns(pl.lit(None).alias(col_name))
    
    # Filter to only keep columns that are in the schema
    df = df.select([col for col in schema.keys()])
    
    return df


def prepare_dataframes_for_stacking(dataframes, string_columns=None, int_columns=None, float_columns=None, verbose=0):
    """
    Prepare a list of dataframes for stacking (vertical concatenation) by ensuring they have
    compatible schemas. This solves common issues like Null vs String type mismatches.
    
    Parameters
    ----------
    dataframes : list of pl.DataFrame
        List of dataframes to prepare for stacking
    string_columns : list of str, optional
        List of column names that should be ensured to have string (Utf8) type
        Common examples include 'gene_id', 'transcript_id', and other identifier columns
        that may contain nulls in some dataframes and strings in others
    int_columns : list of str, optional
        List of column names that should be ensured to have integer (Int64) type
        Common examples include 'position', 'true_position', and other position columns
        that may contain nulls in some dataframes
    float_columns : list of str, optional
        List of column names that should be ensured to have float (Float64) type
        Common examples include 'score', 'probability', and other numeric columns
    verbose : int, optional
        Verbosity level for debugging output
        
    Returns
    -------
    list of pl.DataFrame
        List of dataframes with compatible schemas ready for stacking
    """
    if not dataframes:
        return []
        
    if len(dataframes) == 1:
        return dataframes
    
    # Filter out any empty dataframes
    dataframes = [df for df in dataframes if df.shape[0] > 0]
    if not dataframes:
        return []
    
    # Verify all dataframes have the same set of columns
    all_columns_sets = [set(df.columns) for df in dataframes]
    all_same_columns = all(cols == all_columns_sets[0] for cols in all_columns_sets)
    
    if not all_same_columns and verbose > 0:
        print("[WARNING] Not all dataframes have the same columns (this may indicate a bug)")
        all_columns = set().union(*all_columns_sets)
        for i, cols_set in enumerate(all_columns_sets):
            missing = all_columns - cols_set
            if missing and verbose > 1:
                print(f"  Dataframe {i} is missing columns: {missing}")
    
    # Initialize default column type mappings if not provided
    if string_columns is None:
        string_columns = ['gene_id', 'transcript_id', 'gene_name', 'gene_symbol', 
                         'transcript_name', 'id', 'name', 'source']
    
    if int_columns is None:
        int_columns = ['position', 'true_position', 'predicted_position', 'count']
    
    if float_columns is None:
        float_columns = ['score', 'probability', 'height', 'ratio', 'strength']
    
    # Create a unified schema
    unified_schema = {}
    all_columns = set()
    
    # Collect all unique columns across dataframes
    for df in dataframes:
        all_columns.update(df.columns)
    
    # Determine the best type for each column
    for col in all_columns:
        # Handle explicitly typed columns first
        if col in string_columns:
            unified_schema[col] = pl.Utf8
            continue
        elif col in int_columns:
            unified_schema[col] = pl.Int64
            continue
        elif col in float_columns:
            unified_schema[col] = pl.Float64
            continue
        
        # For other columns, try to infer types
        # Collect all non-null dtypes for this column across dataframes
        col_dtypes = []
        for df in dataframes:
            if col in df.columns:
                dtype_str = str(df.select(pl.col(col)).dtypes[0])
                if 'null' not in dtype_str.lower():
                    col_dtypes.append(df.select(pl.col(col)).dtypes[0])
        
        # If we found any non-null types, use the first one
        if col_dtypes:
            unified_schema[col] = col_dtypes[0]
        # Otherwise use a type based on column name patterns
        else:
            # Check if the column name suggests a particular type
            if any(pattern in col.lower() for pattern in ['score', 'prob', 'height', 'context', 'value', 'ratio', 'diff', 'mean', 'strength']):
                unified_schema[col] = pl.Float64
            elif any(col.startswith(prefix) for prefix in ['is_', 'has_']):
                unified_schema[col] = pl.Boolean
            elif any(pattern in col.lower() for pattern in ['position', 'index', 'count', 'pos', 'idx']):
                unified_schema[col] = pl.Int64
            elif any(col.endswith(suffix) for suffix in ['_id', '_name']):
                unified_schema[col] = pl.Utf8
            else:
                # Default to Utf8 as the safest option for unknown columns
                unified_schema[col] = pl.Utf8
    
    if verbose > 2:
        print(f"[DEBUG] Unified schema created with {len(unified_schema)} columns")
    
    # Apply the unified schema to all dataframes
    prepared_dataframes = []
    for i, df in enumerate(dataframes):
        # Select only the columns needed and ensure types
        try:
            # Ensure columns are properly typed based on their designated type
            # Handle string columns
            for col in string_columns:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Utf8))
            
            # Handle integer columns        
            for col in int_columns:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Int64))
                    
            # Handle float columns
            for col in float_columns:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Float64))
                    
            # Then use ensure_schema for the full schema alignment
            prepared_df = ensure_schema(df, unified_schema)
            prepared_dataframes.append(prepared_df)
            
        except Exception as e:
            if verbose > 0:
                print(f"[ERROR] Failed to prepare dataframe {i} for stacking: {e}")
                
    return prepared_dataframes

def extend_schema(base_schema, *dataframes, context_prefix='context_', default_type=pl.Float64):
    """
    Create an extended schema that preserves both the base schema and additional columns 
    from multiple dataframes.
    
    Parameters
    ----------
    base_schema : dict
        The base schema dictionary to start with (e.g., DEFAULT_POSITION_SCHEMA)
    *dataframes : list of pl.DataFrame
        One or more polars DataFrames whose columns should be included in the extended schema
    context_prefix : str, optional
        Prefix that identifies context columns, which will use default_type if not specified,
        by default 'context_'
    default_type : pl.DataType, optional
        Default type to use for new context columns, by default pl.Float64
    
    Returns
    -------
    dict
        An extended schema that includes all columns from base_schema and dataframes
    
    Examples
    --------
    >>> extended = extend_schema(DEFAULT_POSITION_SCHEMA, donor_df, acceptor_df)
    >>> # Use the extended schema
    >>> donor_df = ensure_schema(donor_df, extended)
    """
    # Start with a copy of the base schema
    extended_schema = base_schema.copy()
    
    # Process each dataframe
    for df in dataframes:
        if df is None or df.is_empty():
            continue
            
        # Add any additional columns from the dataframe
        for col in df.columns:
            if col not in extended_schema:
                # Determine the column type
                if col.startswith(context_prefix):
                    # Special handling for context columns
                    col_type = default_type
                else:
                    # Use the dataframe's schema for the column type
                    col_type = df.schema[col]
                    
                extended_schema[col] = col_type
    
    return extended_schema


def analyze_schema_mismatch(df1: pl.DataFrame, df2: pl.DataFrame, name1: str = "DataFrame1", 
                          name2: str = "DataFrame2") -> Dict:
    """
    Analyze schema differences between two polars DataFrames to identify issues that would prevent stacking.
    This is useful for diagnosing why operations like df1.vstack(df2) fail with schema errors.
    
    Parameters
    ----------
    df1 : pl.DataFrame
        First DataFrame (e.g., donor_error_df)
    df2 : pl.DataFrame
        Second DataFrame (e.g., acceptor_error_df)
    name1 : str, optional
        Name for the first DataFrame for reporting, by default "DataFrame1"
    name2 : str, optional
        Name for the second DataFrame for reporting, by default "DataFrame2"
    
    Returns
    -------
    Dict
        Dictionary with detailed analysis of schema mismatches including:
        - columns_only_in_df1: Columns present only in the first DataFrame
        - columns_only_in_df2: Columns present only in the second DataFrame
        - type_mismatches: Columns with type inconsistencies
        - mismatches_detail: Detailed information about each type mismatch
        - fixable: Whether the mismatch is likely fixable with schema normalization
        - recommendation: Suggested approach to fix the schema mismatch
    
    Examples
    --------
    >>> results = analyze_schema_mismatch(donor_error_df, acceptor_error_df, "donor", "acceptor")
    >>> print(f"Type mismatches: {results['type_mismatches']}")
    >>> print(f"Recommendation: {results['recommendation']}")
    """
    # Initialize result dictionary
    result = {
        "columns_only_in_df1": [],
        "columns_only_in_df2": [],
        "type_mismatches": [],
        "mismatches_detail": [],
        "fixable": True,
        "recommendation": ""
    }
    
    # Get column sets and identify differences
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    only_in_df1 = cols1 - cols2
    only_in_df2 = cols2 - cols1
    common_cols = cols1.intersection(cols2)
    
    # Store columns that are only in one DataFrame
    result["columns_only_in_df1"] = list(only_in_df1)
    result["columns_only_in_df2"] = list(only_in_df2)
    
    # Check for type mismatches in common columns
    for col in common_cols:
        dtype1 = df1.select(pl.col(col)).dtypes[0]
        dtype2 = df2.select(pl.col(col)).dtypes[0]
        
        if dtype1 != dtype2:
            result["type_mismatches"].append(col)
            
            # Check for null vs non-null issues (most common problem)
            is_null_issue = False
            null_string_issue = False
            
            # Check if one is Null and the other is a concrete type
            if "null" in str(dtype1).lower() and "null" not in str(dtype2).lower():
                is_null_issue = True
                if "utf8" in str(dtype2).lower() or "string" in str(dtype2).lower():
                    null_string_issue = True
            elif "null" in str(dtype2).lower() and "null" not in str(dtype1).lower():
                is_null_issue = True
                if "utf8" in str(dtype1).lower() or "string" in str(dtype1).lower():
                    null_string_issue = True
            
            # Sample values to see what's in the data
            try:
                sample1 = df1.select(pl.col(col)).head(5).to_pandas()[col].tolist()
            except:
                sample1 = ["<error getting values>"]
                
            try:
                sample2 = df2.select(pl.col(col)).head(5).to_pandas()[col].tolist()
            except:
                sample2 = ["<error getting values>"]
                
            # Are all values null in either DataFrame?
            all_null_df1 = df1.select(pl.col(col).is_null().all()).item()
            all_null_df2 = df2.select(pl.col(col).is_null().all()).item()
            
            # Add detailed info
            detail = {
                "column": col,
                f"{name1}_type": str(dtype1),
                f"{name2}_type": str(dtype2),
                "is_null_issue": is_null_issue,
                "null_string_issue": null_string_issue,
                f"{name1}_sample": sample1,
                f"{name2}_sample": sample2,
                f"{name1}_all_null": all_null_df1,
                f"{name2}_all_null": all_null_df2,
                "fixable": True  # Most schema issues are fixable
            }
            
            result["mismatches_detail"].append(detail)
    
    # Generate recommendations based on analysis
    recommendations = []
    if result["columns_only_in_df1"] or result["columns_only_in_df2"]:
        recommendations.append(
            f"Use ensure_schema() to normalize both dataframes and include all columns. "
            f"There are {len(result['columns_only_in_df1'])} columns only in {name1} and "
            f"{len(result['columns_only_in_df2'])} columns only in {name2}."
        )
    
    if result["type_mismatches"]:
        null_string_cols = [d["column"] for d in result["mismatches_detail"] if d["null_string_issue"]]
        if null_string_cols:
            recommendations.append(
                f"Convert Null types to Utf8 for columns: {', '.join(null_string_cols)}. "
                f"This is a common issue when one dataframe has string values and another has nulls."
            )
        
        other_type_mismatch_cols = [col for col in result["type_mismatches"] if col not in null_string_cols]
        if other_type_mismatch_cols:
            recommendations.append(
                f"Ensure consistent types for: {', '.join(other_type_mismatch_cols)}. "
                f"Use a common schema dictionary with explicit types and ensure_schema()."
            )
    
    # Check if we believe this is fixable
    complex_type_issues = False
    for detail in result["mismatches_detail"]:
        # Most Null vs concrete type issues are fixable, but other mismatches might be complex
        if not detail["is_null_issue"]:
            complex_type_issues = True
            detail["fixable"] = False
    
    if complex_type_issues:
        recommendations.append(
            "Some columns have complex type mismatches that may not be automatically fixable. "
            "Consider manual type conversion or removing problematic columns."
        )
        result["fixable"] = False
    
    # Combine recommendations
    result["recommendation"] = " ".join(recommendations)
    
    # If no issues found
    if not any([result["columns_only_in_df1"], result["columns_only_in_df2"], result["type_mismatches"]]):
        result["recommendation"] = f"No schema issues detected that would prevent stacking {name1} and {name2}."
    
    return result