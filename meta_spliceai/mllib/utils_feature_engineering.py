import pandas as pd 
import numpy as np

def sanitize_column_names(df):
    # Replace or remove problematic characters
    sanitized_columns = df.columns.str.replace("[", "_").str.replace("]", "_").str.replace("<", "_")
    df.columns = sanitized_columns
    return df


def sanitize_column_names_with_mapping(df, adict={}, verbose=0):
    """
    Sanitize column names and return the sanitized DataFrame along with a mapping 
    from original column names to sanitized names.
    """
    if not adict: 
        adict = {"[": "_", 
                "]": "_", 
                "<": "_"}

    # Replace or remove problematic characters
    sanitized_columns = df.columns.str.replace("[", adict["["]).str.replace("]", adict["]"]).str.replace("<", adict["<"])
    
    # Create a mapping from original to sanitized column names
    column_mapping = dict(zip(sanitized_columns, df.columns))

    if verbose: 
        for col, col_mapped in column_mapping.items(): 
            if col.find('_') > 0: 
                print(col, "-> ", col_mapped)
    
    df.columns = sanitized_columns
    return df, column_mapping

def update_feature_names(df, replace_dict, col='Feature'):
    """
    Update feature names in a DataFrame's 'Feature' column based on a replacement dictionary.
    
    Parameters:
    - df: DataFrame containing the feature importance scores. Expected to have a 'Feature' column.
    - replace_dict: Dictionary specifying the characters to replace and their replacements.
    
    Returns:
    - Updated DataFrame
    """
    # Make sure the DataFrame has a 'Feature' column
    if col not in df.columns:
        raise ValueError(f"The DataFrame must contain a {col} column.")
    
    # Apply the replacements to the 'Feature' column
    for old_char, new_char in replace_dict.items():
        df[col] = df[col].str.replace(old_char, new_char)
    
    return df
