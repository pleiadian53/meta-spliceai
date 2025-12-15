# filepath: /path/to/meta-spliceai/meta_spliceai/splice_engine/analysis_utils.py

import os
import re
import random
import math
import polars as pl
import pandas as pd
import numpy as np

# Plotting 
import seaborn as sns
import matplotlib.pyplot as plt

import shap

from tabulate import tabulate
from .model_evaluator import ModelEvaluationFileHandler

from .utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator, 
    display_dataframe, 
    display_dataframe_in_chunks
)

from meta_spliceai.splice_engine.extract_gene_sequences import build_gene_id_to_name_map


def check_duplicates(df, subset, return_rows=False, verbose=0, example_limit=5):
    """
    Check for duplicate rows in a DataFrame, converting to Pandas if necessary, with verbose analysis.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - subset (list): List of columns to check for duplicates.
    - return_rows (bool): If True, return the duplicate rows. If False, return the number of duplicate rows.
    - verbose (int): Verbosity level. 
                     0: No extra output.
                     1: Print example duplicate groups.
                     2: Return a DataFrame highlighting differing values in duplicate groups.
    - example_limit (int): Number of example duplicate groups to show in verbose mode.

    Returns:
    - int or pd.DataFrame or pl.DataFrame: The number of duplicate rows or the duplicate rows themselves, retaining the input type.
    """
    is_polars = isinstance(df, pl.DataFrame)

    # Convert to Pandas for simplicity if input is Polars
    if is_polars:
        df = df.to_pandas()

    # Find duplicates using Pandas
    duplicates_pd = df[df.duplicated(subset=subset, keep=False)]

    if verbose > 0:
        # Identify groups of duplicates based on the subset
        duplicate_groups = duplicates_pd.groupby(subset)

        if verbose == 1:
            print("\n[Verbose Mode] Example duplicate groups:")
            for group_key, group in list(duplicate_groups)[0:example_limit]:
                print(f"\nDuplicate group for {subset}: {group_key}")
                print(group)
                differing_columns = [
                    col for col in df.columns if col not in subset and group[col].nunique() > 1
                ]
                if differing_columns:
                    print(f"  Differing columns: {differing_columns}")
                else:
                    print("  No differing columns detected.")

        elif verbose == 2:
            print("\n[Verbose Mode Level 2] Highlighting differing columns in duplicate groups:")
            highlight_df_list = []
            for group_key, group in list(duplicate_groups)[0:example_limit]:
                differing_columns = [
                    col for col in df.columns if col not in subset and group[col].nunique() > 1
                ]
                if differing_columns:
                    group = group.copy()
                    group['__group_key'] = str(group_key)  # Add the group key for context
                    diff_columns = {
                        f"{col}_diff": group[col].astype(str) for col in differing_columns
                    }
                    group = pd.concat([group, pd.DataFrame(diff_columns, index=group.index)], axis=1)
                    # NOTE: instead of adding columns one by one, all new columns (*_diff) are created together using pd.concat. This avoids excessive fragmentation and improves performance
                    
                    highlight_df_list.append(group)
            if highlight_df_list:
                highlight_df = pd.concat(highlight_df_list, ignore_index=True)
                print(highlight_df)
            else:
                print("No differing columns detected in the example groups.")

    if return_rows:
        # Convert back to Polars if the input was Polars
        if is_polars:
            return pl.from_pandas(duplicates_pd)
        else:
            return duplicates_pd
    else:
        # Return the count of duplicate rows
        return duplicates_pd.shape[0]


# Example Implementation of check_duplicates
def handle_duplicates(df, subset, action='drop'):
    """
    Handle duplicates in a DataFrame.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - subset (list): List of columns to check for duplicates.
    - action (str): Action to perform on duplicates ('drop' or 'list').

    Returns:
    - pd.DataFrame or pl.DataFrame: The DataFrame with duplicates handled.
    """
    if isinstance(df, pd.DataFrame):
        if action == 'drop':
            return df.drop_duplicates(subset=subset)
        elif action == 'list':
            return df[df.duplicated(subset=subset, keep=False)]
        else:
            raise ValueError("Unsupported action")
    elif isinstance(df, pl.DataFrame):
        if action == 'drop':
            return df.unique(subset=subset)
        elif action == 'list':
            return df.filter(pl.col(subset).is_duplicated())
        else:
            raise ValueError("Unsupported action")
    else:
        raise ValueError("Unsupported DataFrame type")


def find_missing_combinations(df1, df2, on_columns, verbose=1):
    """
    Find the combinations of the specified columns that are in df1 but not in df2.

    Parameters:
    - df1 (pd.DataFrame or pl.DataFrame): The first dataframe.
    - df2 (pd.DataFrame or pl.DataFrame): The second dataframe.
    - on_columns (list): List of columns to join on.

    Returns:
    - pd.DataFrame or pl.DataFrame: The combinations of the specified columns that are in df1 but not in df2.
    """
    is_polars = isinstance(df1, pl.DataFrame)

    if is_polars:
        df1 = df1.to_pandas()
    if isinstance(df2, pl.DataFrame):
        df2 = df2.to_pandas()

    # Find the combinations that are in df1 but not in df2
    missing_combinations = df1.merge(df2[on_columns], on=on_columns, how='left', indicator=True)
    # Perform a left join between df1 and df2 on the specified columns (on_columns).
    # The 'indicator=True' parameter adds a column '_merge' to the result, indicating the source of each row:
    # - 'left_only' if the row is only in df1
    # - 'right_only' if the row is only in df2
    # - 'both' if the row is in both df1 and df2

    missing_combinations = missing_combinations[missing_combinations['_merge'] == 'left_only']
    missing_combinations = missing_combinations[on_columns]

    if is_polars:
        missing_combinations = pl.from_pandas(missing_combinations)

    if verbose: 
        # Total number of missing combinations
        num_missing_combinations = missing_combinations.shape[0]
        print(f"[info] Total number of missing combinations: {num_missing_combinations}")

    return missing_combinations


def validate_ids(df, col_gid='gene_id', col_tid='transcript_id', valid_prefixes=None, verbose=1):
    """
    Validate gene and transcript IDs against a set of prefixes for different annotation systems.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - col_gid (str): The column name for gene IDs (default is 'gene_id').
    - col_tid (str): The column name for transcript IDs (default is 'transcript_id').
    - valid_prefixes (dict): A dictionary with 'gene' and 'transcript' keys containing lists of valid prefixes.
                             Example: {'gene': ['ENSG', 'NM'], 'transcript': ['ENST', 'XM']}
    - verbose (int): Verbosity level (default is 1).

    Returns:
    - pd.DataFrame or pl.DataFrame: A DataFrame containing rows with invalid gene or transcript IDs.
    """
    if valid_prefixes is None:
        valid_prefixes = {
            'gene': ['ENSG'],  # Default Ensembl gene prefixes
            'transcript': ['ENST']  # Default Ensembl transcript prefixes
        }

    is_polars = isinstance(df, pl.DataFrame)

    # Convert Polars DataFrame to Pandas DataFrame if necessary
    if is_polars:
        df = df.to_pandas()

    # Compile regex patterns for valid prefixes
    gene_pattern = f"^({'|'.join(map(re.escape, valid_prefixes['gene']))})"
    transcript_pattern = f"^({'|'.join(map(re.escape, valid_prefixes['transcript']))})"

    # Identify invalid rows
    invalid_rows = df[
        df[col_gid].isnull() |
        df[col_tid].isnull() |
        (df[col_gid] == '') |
        (df[col_tid] == '') |
        (~df[col_gid].astype(str).str.match(gene_pattern)) |
        (~df[col_tid].astype(str).str.match(transcript_pattern))
    ]

    num_invalid_rows = invalid_rows.shape[0]

    if verbose > 0:
        print(f"[info] Number of invalid rows: {num_invalid_rows}")
        if verbose > 1:
            print("[info] Example invalid rows:")
            print(invalid_rows.head(10))

    # Convert back to Polars DataFrame if necessary
    if is_polars:
        invalid_rows = pl.from_pandas(invalid_rows)

    return invalid_rows


def check_data_integrity_polars(df, col_gid='gene_id', col_tid='transcript_id', verbose=1):
    """
    Check the data integrity of a DataFrame by verifying the format of gene IDs and transcript IDs.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - col_gid (str): The column name for gene IDs (default is 'gene_id').
    - col_tid (str): The column name for transcript IDs (default is 'transcript_id').
    - verbose (int): Verbosity level (default is 1).

    Returns:
    - None
    """
    # Check for abnormal gene IDs and transcript IDs
    abnormal_rows = df.filter(
        (pl.col(col_gid).is_null()) | 
        (pl.col(col_tid).is_null()) | 
        (pl.col(col_gid) == '') | 
        (pl.col(col_tid) == '') | 
        (pl.col(col_gid).str.contains('^ENSG') == False) | 
        (pl.col(col_tid).str.contains('^ENST') == False)
    )

    num_abnormal_rows = abnormal_rows.shape[0]

    if num_abnormal_rows > 0:
        if verbose > 0:
            print(f"[info] Number of abnormal rows: {num_abnormal_rows}")
            if verbose > 1:
                print("[info] Example abnormal rows:")
                print(abnormal_rows.head(10))

    return


def check_data_integrity(df, col_gid='gene_id', col_tid='transcript_id', verbose=1):
    """
    Check the data integrity of a DataFrame by verifying the format of gene IDs and transcript IDs.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - col_gid (str): The column name for gene IDs (default is 'gene_id').
    - col_tid (str): The column name for transcript IDs (default is 'transcript_id').
    - verbose (int): Verbosity level (default is 1).

    Returns:
    - pd.DataFrame or pl.DataFrame: A DataFrame containing abnormal rows.
    """
    is_polars = isinstance(df, pl.DataFrame)

    # Convert Polars DataFrame to Pandas DataFrame if necessary
    if is_polars:
        df = df.to_pandas()

    # Check for abnormal gene IDs and transcript IDs
    abnormal_rows = df[
        df[col_gid].isnull() |
        df[col_tid].isnull() |
        (df[col_gid] == '') |
        (df[col_tid] == '') |
        (~df[col_gid].astype(str).str.startswith('ENSG')) |
        (~df[col_tid].astype(str).str.startswith('ENST'))
    ]

    num_abnormal_rows = abnormal_rows.shape[0]

    if num_abnormal_rows > 0:
        if verbose > 0:
            print(f"[info] Number of abnormal rows: {num_abnormal_rows}")
            if verbose > 1:
                print("[info] Example abnormal rows:")
                print(abnormal_rows.head(10))

    # Convert back to Polars DataFrame if necessary
    if is_polars:
        abnormal_rows = pl.from_pandas(abnormal_rows)

    return abnormal_rows


def check_and_subset_invalid_transcript_ids(df, col_gid='gene_id', col_tid='transcript_id', verbose=1):
    """
    Check data integrity and subset rows where transcript IDs are invalid (e.g., null or 0).

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - col_gid (str): The column name for gene IDs (default is 'gene_id').
    - col_tid (str): The column name for transcript IDs (default is 'transcript_id').
    - verbose (int): Verbosity level (default is 1).

    Returns:
    - pd.DataFrame or pl.DataFrame: A DataFrame containing rows with invalid transcript IDs.
    """
    is_polars = isinstance(df, pl.DataFrame)

    # Convert Polars DataFrame to Pandas DataFrame if necessary
    if is_polars:
        df = df.to_pandas()

    # Check for rows with invalid transcript IDs
    invalid_transcript_df = df[
        df[col_tid].isnull() |
        (df[col_tid] == 0)
    ]

    # Check for rows with general data integrity issues
    abnormal_rows = df[
        df[col_gid].isnull() |
        df[col_tid].isnull() |
        (df[col_gid] == '') |
        (df[col_tid] == '') |
        (~df[col_gid].astype(str).str.startswith('ENSG')) |
        (~df[col_tid].astype(str).str.startswith('ENST'))
    ]

    # Combine both checks into a unified DataFrame
    result_df = pd.concat([invalid_transcript_df, abnormal_rows]).drop_duplicates()

    num_invalid_transcripts = invalid_transcript_df.shape[0]
    num_abnormal_rows = abnormal_rows.shape[0]
    num_combined = result_df.shape[0]

    if verbose > 0:
        print(f"[info] Rows with invalid transcript IDs: {num_invalid_transcripts}")
        print(f"[info] Rows with general abnormalities: {num_abnormal_rows}")
        print(f"[info] Total abnormal rows (combined): {num_combined}")
        if verbose > 1:
            print("[info] Example abnormal rows:")
            print(result_df.head(10))

    # Convert back to Polars DataFrame if necessary
    if is_polars:
        result_df = pl.from_pandas(result_df)

    return result_df


def remove_invalid_transcript_ids(df, col_gid='gene_id', col_tid='transcript_id', verbose=1):
    """
    Remove rows with invalid gene IDs or transcript IDs from the input DataFrame.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - col_gid (str): The column name for gene IDs (default is 'gene_id').
    - col_tid (str): The column name for transcript IDs (default is 'transcript_id').
    - verbose (int): Verbosity level (default is 1).

    Returns:
    - pd.DataFrame or pl.DataFrame: The filtered DataFrame with valid gene IDs and transcript IDs.
    """
    is_polars = isinstance(df, pl.DataFrame)

    # Convert Polars DataFrame to Pandas DataFrame if necessary
    if is_polars:
        df = df.to_pandas()

    # Identify rows with invalid gene IDs or transcript IDs
    invalid_rows = df[
        df[col_gid].isnull() |
        df[col_tid].isnull() |
        (df[col_gid] == '') |
        (df[col_tid] == '') |
        (df[col_tid] == 0) |
        (~df[col_gid].astype(str).str.startswith('ENSG')) |
        (~df[col_tid].astype(str).str.startswith('ENST'))
    ]

    num_invalid_rows = invalid_rows.shape[0]

    # Filter out invalid rows
    valid_df = df.drop(invalid_rows.index)

    if verbose > 0:
        print(f"[info] Number of rows removed: {num_invalid_rows}")
        print(f"[info] Number of remaining rows: {valid_df.shape[0]}")
        print(f"[info] Within the remaining rows, the number of unique genes and transcripts are:")
        count_unique_ids(valid_df, col_gid=col_gid, col_tid=col_tid, verbose=verbose)

    # Convert back to Polars DataFrame if necessary
    if is_polars:
        valid_df = pl.from_pandas(valid_df)

    return valid_df


def filter_and_validate_ids(df, col_gid='gene_id', col_tid='transcript_id', valid_prefixes=None, verbose=1, return_invalid_rows=False):
    """
    Filter out rows with invalid gene or transcript IDs, validate against prefixes, and show statistics.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - col_gid (str): The column name for gene IDs (default is 'gene_id').
    - col_tid (str): The column name for transcript IDs (default is 'transcript_id').
    - valid_prefixes (dict): A dictionary with 'gene' and 'transcript' keys containing lists of valid prefixes.
                             Example: {'gene': ['ENSG', 'NM'], 'transcript': ['ENST', 'XM']}
    - verbose (int): Verbosity level (default is 1).
    - return_invalid_rows (bool): If True, return both valid and invalid DataFrames.

    Returns:
    - pd.DataFrame or tuple(pd.DataFrame, pd.DataFrame):
      The filtered DataFrame with valid IDs, or both valid and invalid DataFrames if return_invalid_rows=True.
    """
    if valid_prefixes is None:
        valid_prefixes = {
            'gene': ['ENSG'],  # Default Ensembl gene prefixes
            'transcript': ['ENST']  # Default Ensembl transcript prefixes
        }

    is_polars = isinstance(df, pl.DataFrame)

    # Convert Polars DataFrame to Pandas DataFrame if necessary
    if is_polars:
        df = df.to_pandas()

    # Compile regex patterns for valid prefixes
    gene_pattern = f"^({'|'.join(map(re.escape, valid_prefixes['gene']))})"
    transcript_pattern = f"^({'|'.join(map(re.escape, valid_prefixes['transcript']))})"

    # Statistics before filtering
    total_rows = df.shape[0]
    total_genes = df[col_gid].nunique()
    total_transcripts = df[col_tid].nunique()

    # Identify rows with invalid gene or transcript IDs
    invalid_rows = df[
        df[col_gid].isnull() |
        df[col_tid].isnull() |
        (df[col_gid] == '') |
        (df[col_tid] == '') |
        (df[col_tid] == 0) |
        (~df[col_gid].astype(str).str.match(gene_pattern)) |
        (~df[col_tid].astype(str).str.match(transcript_pattern))
    ]

    num_invalid_rows = invalid_rows.shape[0]

    # Filter out invalid rows
    valid_df = df.drop(invalid_rows.index)

    # Statistics after filtering
    remaining_rows = valid_df.shape[0]
    remaining_genes = valid_df[col_gid].nunique()
    remaining_transcripts = valid_df[col_tid].nunique()

    if verbose > 0:
        print(f"[info] Total rows before filtering: {total_rows}")
        print(f"[info] Total genes before filtering: {total_genes}")
        print(f"[info] Total transcripts before filtering: {total_transcripts}")
        print(f"[info] Rows with invalid IDs: {num_invalid_rows}")
        print(f"[info] Genes associated with invalid rows: {total_genes - remaining_genes}")
        print(f"[info] Transcripts associated with invalid rows: {total_transcripts - remaining_transcripts}")
        print(f"[info] Remaining rows (valid): {remaining_rows}")
        print(f"[info] Remaining genes (valid): {remaining_genes}")
        print(f"[info] Remaining transcripts (valid): {remaining_transcripts}")

        if verbose > 1:
            print("[info] Example invalid rows:")
            print(invalid_rows.head(10))

    # Convert back to Polars DataFrame if necessary
    if is_polars:
        valid_df = pl.from_pandas(valid_df)
        invalid_rows = pl.from_pandas(invalid_rows)

    if return_invalid_rows:
        return valid_df, invalid_rows

    return valid_df


def count_unique_ids(df, col_gid='gene_id', col_tid='transcript_id', verbose=1, return_ids=False):
    """
    Count the number of unique genes and transcripts in a DataFrame.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - col_gid (str): The column name for gene IDs (default is 'gene_id').
    - col_tid (str): The column name for transcript IDs (default is 'transcript_id').
    - verbose (int): Verbosity level (default is 1).
    - return_ids (bool): If True, return the unique gene and transcript IDs.

    Returns:
    - dict: A dictionary with the counts of unique genes and transcripts.
    """
    if col_gid not in df.columns:
        raise ValueError(f"Column '{col_gid}' not found in DataFrame.")
    if col_tid not in df.columns:
        raise ValueError(f"Column '{col_tid}' not found in DataFrame.")

    unique_gene_ids = unique_transcript_ids = None
    if isinstance(df, pd.DataFrame):
        num_unique_genes = df[col_gid].nunique()
        num_unique_transcripts = df[col_tid].nunique()

        if return_ids: 
            unique_gene_ids = df[col_gid].unique().tolist()
            unique_transcript_ids = df[col_tid].unique().tolist()
    elif isinstance(df, pl.DataFrame):
        num_unique_genes = df.select(pl.col(col_gid).n_unique()).to_series()[0]
        num_unique_transcripts = df.select(pl.col(col_tid).n_unique()).to_series()[0]

        if return_ids: 
            unique_gene_ids = df.select(pl.col(col_gid).unique()).to_series().to_list()
            unique_transcript_ids = df.select(pl.col(col_tid).unique()).to_series().to_list()
    else:
        raise ValueError("Unsupported DataFrame type")

    result = {
        'num_unique_genes': num_unique_genes,
        'num_unique_transcripts': num_unique_transcripts
    }

    if return_ids:
        result['unique_gene_ids'] = unique_gene_ids
        result['unique_transcript_ids'] = unique_transcript_ids

    if verbose:
        print(f"[info] Number of unique genes: {num_unique_genes}")
        print(f"[info] Number of unique transcripts: {num_unique_transcripts}")

    return result


# Verify if the label column has two classes and find out the number of respective sample sizes
def verify_label_classes(df, label_col='label'):
    """
    Verify if the label column has two classes and find out the number of respective sample sizes.

    Parameters:
    - df (pl.DataFrame): The input Polars DataFrame.
    - label_col (str): The column name for the label (default is 'label').

    Returns:
    - dict: A dictionary with the class counts.
    """
    is_polars = isinstance(df, pl.DataFrame)

    if is_polars:
        # Convert to Pandas DataFrame for easier manipulation
        df = df.to_pandas()

    # Get the unique classes and their counts
    class_counts = df[label_col].value_counts()

    if len(class_counts) != 2:
        raise ValueError(f"The label column '{label_col}' does not have exactly two classes.")

    print(f"[info] Class counts for '{label_col}':")
    for class_value, count in class_counts.items():
        print(f"  Class {class_value}: {count} samples")

    return class_counts.to_dict()


def filter_analysis_dataframe(df=None, gene_ids=None, transcript_ids=None, verbose=0):
    """
    Filter the input analysis DataFrame by genes and/or transcripts.

    Parameters:
    - df (pl.DataFrame or pd.DataFrame): The input DataFrame containing splice site predictions.
    - gene_ids (list): List of gene IDs to filter by (default is None).
    - transcript_ids (list): List of transcript IDs to filter by (default is None).
    - verbose (int): Verbosity level (default is 0).

    Returns:
    - filtered_df (pl.DataFrame or pd.DataFrame): The filtered DataFrame.
    """
    if df is None: 
        from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
        from meta_spliceai.splice_engine.splice_error_analyzer import ErrorAnalyzer

        mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')
     
        # Load the original analysis sequence DataFrame with all prediction types
        df = mefd.load_analysis_sequences(aggregated=True)  # Output is a Polars DataFrame

    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    if verbose:
        print(f"[info] Shape before filtering: {df.shape}")
        print(f"[info] Columns: {list(df.columns)}")

    # Apply filtering
    if gene_ids is not None and transcript_ids is not None:
        filtered_df = df[(df['gene_id'].isin(gene_ids)) & (df['transcript_id'].isin(transcript_ids))]
    elif gene_ids is not None:
        filtered_df = df[df['gene_id'].isin(gene_ids)]
    elif transcript_ids is not None:
        filtered_df = df[df['transcript_id'].isin(transcript_ids)]
    else:
        filtered_df = df

    if verbose:
        print(f"[info] Shape after filtering: {filtered_df.shape}")

    if is_polars:
        filtered_df = pl.DataFrame(filtered_df)

    return filtered_df


def find_transcripts_with_both_labels(df, num_transcripts=10, label_col='label', col_tid='transcript_id'):
    """
    Find all transcripts for which their associated labels have both 0 and 1 (i.e., both positive and negative examples exist).

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame containing splice site predictions.
    - num_transcripts (int): The number of transcripts to return (default is 10).

    Returns:
    - pd.DataFrame or pl.DataFrame: The DataFrame containing the selected transcripts.
    """
    from .analysis_utils import classify_features  # Assumes this function is available

    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    # Group by transcript_id and gene_id
    grouped = df.groupby([col_tid, 'gene_id'])

    # Iterate through the groups and find transcripts with both labels
    selected_transcripts = []
    for (transcript_id, gene_id), group in grouped:
        if set(group[label_col]) == {0, 1}:
            selected_transcripts.append(group)

        # Limit the number of transcripts to return
        if len(selected_transcripts) >= num_transcripts:
            break

    # Concatenate the selected transcripts into a single DataFrame
    result_df = pd.concat(selected_transcripts, ignore_index=True)

    if is_polars:
        result_df = pl.DataFrame(result_df)

    return result_df


def show_predicted_splice_sites(df, num_transcripts=10, focused_pred_types=None, show_only_focused=True):
    """
    Show for each transcript all the associated predicted splice sites, their splice types, and their associated prediction types.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame containing splice site predictions.
    - num_transcripts (int): The number of transcripts to show (default is 10).
    - focused_pred_types (str or list): Prediction types to highlight (default is None).
    - show_only_focused (bool): Show only transcripts with the focused prediction types (default is False).

    Returns:
    - None
    """
    from meta_spliceai.splice_engine.utils_bio import normalize_strand

    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    # Ensure focused_pred_types is a list
    if focused_pred_types is not None and isinstance(focused_pred_types, str):
        focused_pred_types = [focused_pred_types]

    # Group by transcript_id and gene_id
    grouped = df.groupby(['transcript_id', 'gene_id'])

    # Iterate through the groups and display the information
    n_transcripts_found = 0

    tx_set = {}
    for (transcript_id, gene_id), group in grouped:
        strand = normalize_strand(group['strand'].iloc[0])

        # Check if the group contains all the focused prediction types
        if show_only_focused:
            if not all(group['pred_type'].value_counts().get(pt, 0) > 0 for pt in focused_pred_types):
                continue
        
        tx_set[transcript_id] = gene_id

        print_emphasized(f"Transcript ID: {transcript_id}, Gene ID: {gene_id} (strand: {strand})")

        # Sort the group by position based on the strand
        if strand == '+':
            group = group.sort_values(by='position', ascending=True)
        else:
            group = group.sort_values(by='position', ascending=False)

        for _, row in group.iterrows():
            row_info = f"  Position: {row['position']}, Splice Type: {row['splice_type']}, Prediction Type: {row['pred_type']}"
            if focused_pred_types and row['pred_type'] in focused_pred_types:
                row_info += " *"
            print_with_indent(row_info, indent_level=1)
        print()

        # Limit the number of transcripts to show
        num_transcripts -= 1
        if num_transcripts == 0:
            break

        n_transcripts_found += 1

    print_with_indent(f"Number of transcripts found: {n_transcripts_found} (requested={num_transcripts})", indent_level=1)

    return tx_set


def subsample_genes_v0(df, gene_col='gene_id', N=10):
    """
    Subsample N genes from the DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - gene_col (str): The column name for gene IDs (default is 'gene_id').
    - N (int): The number of genes to subsample (default is 10).

    Returns:
    - pd.DataFrame: Subsampled DataFrame containing only rows from the selected genes.
    """
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    # Get unique gene IDs
    unique_genes = df[gene_col].unique()

    # Randomly sample N genes
    sampled_genes = pd.Series(unique_genes).sample(n=N, random_state=42).tolist()

    # Subset the DataFrame to retain only rows from the sampled genes
    subsampled_df = df[df[gene_col].isin(sampled_genes)]

    if is_polars:
        subsampled_df = pl.DataFrame(subsampled_df)

    return subsampled_df


def subsample_genes(
    df, 
    gene_col='gene_id', 
    N=10, 
    custom_genes=None, 
    gtf_file_path=None, 
    remove_version=True, 
    use_polars=True, 
    **kargs, 
):
    """
    Subsample N genes from the DataFrame and optionally include custom genes.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        The input DataFrame.
    gene_col : str
        The column name for gene IDs (default is 'gene_id').
    N : int
        The number of genes to subsample (default is 10).
    custom_genes : list or set, optional
        A list or set of custom genes to include in the subsample. Can be gene IDs or gene names.
    gtf_file_path : str, optional
        Path to the GTF file for mapping gene names to IDs if needed.
    remove_version : bool
        If True, remove version suffix from gene IDs.
    use_polars : bool
        If True, use Polars for processing.
    verbose : int
        Verbosity level.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Subsampled DataFrame containing only rows from the selected genes.
    """
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    verbose = kargs.get('verbose', 1)

    # Get unique gene IDs
    unique_genes = df[gene_col].unique()

    # Randomly sample N genes
    sampled_genes = pd.Series(unique_genes).sample(n=N, random_state=42).tolist()

    # Add custom genes if provided
    if custom_genes:
        # Determine if custom genes are IDs or names
        if all(str(gene).startswith('ENSG') for gene in custom_genes):
            gene_col_to_use = 'gene_id'
            additional_genes = custom_genes
        else:
            gene_col_to_use = 'gene_name'
            if gene_col_to_use not in df.columns:
                if gtf_file_path is None:
                    raise ValueError("GTF file path must be provided to map gene names to IDs.")
                additional_genes = map_gene_names_to_ids(
                    custom_genes, 
                    gtf_file_path, 
                    remove_version, 
                    use_polars, 
                    verbose
                )
            else:
                additional_genes = df[df[gene_col_to_use].isin(custom_genes)][gene_col].unique()

        # Add additional genes to the sampled list
        sampled_genes.extend(additional_genes)

        # Test: Verify if custom genes exist in the input DataFrame
        existing_genes = set(df[gene_col].unique())
        missing_genes = set(additional_genes) - existing_genes

        if missing_genes:
            print(f"[warning] The following custom genes were not found in the input DataFrame: {missing_genes}")
        else:
            print("[info] All custom genes were found in the input DataFrame.")

    # Ensure uniqueness of sampled genes
    sampled_genes = list(set(sampled_genes))

    # Subset the DataFrame to retain only rows from the sampled genes
    subsampled_df = df[df[gene_col].isin(sampled_genes)]

    # Check the number of subsampled genes
    if verbose: 
        max_display = kargs.get('max_display', 20)

        # Check the final number of subsampled genes including custom genes
        unique_genes = subsampled_df['gene_id'].unique()
        num_unique_genes = len(unique_genes)

        # Display the genes, limited to max_display
        print(f"[info] Displaying up to {max_display} genes among {num_unique_genes} unique genes")
        for gene_id in sampled_genes[:max_display]:
            gene_name = df.loc[df[gene_col] == gene_id, 'gene_name'].iloc[0] if 'gene_name' in df.columns else "N/A"
            print(f"{gene_id}: {gene_name}")

    if is_polars:
        subsampled_df = pl.DataFrame(subsampled_df)

    return subsampled_df


def map_gene_names_to_ids(
    gene_names, 
    gtf_file_path, 
    remove_version=True, 
    use_polars=True, 
    verbose=1
):
    """
    Map gene names to Ensembl gene IDs using a GTF file.

    Parameters
    ----------
    gene_names : list
        List of gene names to map.
    gtf_file_path : str
        Path to the GTF file.
    remove_version : bool
        If True, remove version suffix from gene IDs.
    use_polars : bool
        If True, use Polars for processing.
    verbose : int
        Verbosity level.

    Returns
    -------
    list
        List of mapped Ensembl gene IDs.
    """
    # Build the gene_id -> gene_name map
    gene_id_to_name = build_gene_id_to_name_map(
        gtf_file_path=gtf_file_path,
        remove_version=remove_version,
        use_polars=use_polars,
        verbose=verbose
    )

    # Reverse the map to get gene_name -> gene_id
    gene_name_to_id = {v: k for k, v in gene_id_to_name.items()}

    # Map the provided gene names to gene IDs
    mapped_gene_ids = [gene_name_to_id.get(name) for name in gene_names if name in gene_name_to_id]

    if verbose:
        print(f"[info] Mapped {len(mapped_gene_ids)} gene names to Ensembl IDs.")

    return mapped_gene_ids


def subset_non_motif_features(df, label_col='label', max_unique_for_categorical=10):
    """
    Subset the DataFrame to retain only non-motif features.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - label_col (str): The column name for the label/class (default is 'label').
    - max_unique_for_categorical (int): Threshold for the maximum number of unique values
      to classify a numerical column as categorical (default is 10).

    Returns:
    - pd.DataFrame: DataFrame with only non-motif features.
    """
    feature_categories = classify_features(df, label_col=label_col, max_unique_for_categorical=max_unique_for_categorical)

    # Combine all non-motif features
    non_motif_features = (
        feature_categories['id_columns'] +
        feature_categories['class_labels'] +
        feature_categories['categorical_features'] +
        feature_categories['numerical_features'] +
        feature_categories['derived_categorical_features']
    )

    # Subset the DataFrame to retain only non-motif features
    non_motif_df = df[non_motif_features]

    return non_motif_df


def classify_features_v0(df, label_col='label', max_unique_for_categorical=10):
    """
    Classify features into categories: motif, ID, label, categorical, and numerical.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - label_col (str): The column name for the label/class (default is 'label').
    - max_unique_for_categorical (int): Threshold for the maximum number of unique values
      to classify a numerical column as categorical (default is 10).

    Returns:
    - dict: A dictionary with feature categories.
    """
    # import re

    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    feature_categories = {
        'motif_features': [col for col in df.columns if re.match(r'^\d+mer_.*', col)],
        'id_columns': [col for col in df.columns if col.endswith('_id') or col == 'id'],
        'class_labels': [label_col] if label_col in df.columns else [],
        'categorical_features': [],
        'numerical_features': [],
        'derived_categorical_features': []  # To store numericals classified as categorical due to unique value threshold
    }

    for col in df.columns:
        if col in feature_categories['motif_features'] or col in feature_categories['id_columns'] or col == label_col:
            continue

        col_data = df[col]
        unique_values = set(col_data.dropna().unique())

        if isinstance(col_data.dtype, pd.CategoricalDtype) or col_data.dtype == 'object' or unique_values <= {True, False}:
            feature_categories['categorical_features'].append(col)
        elif pd.api.types.is_numeric_dtype(col_data):
            if len(unique_values) <= max_unique_for_categorical:
                feature_categories['derived_categorical_features'].append(col)
            else:
                feature_categories['numerical_features'].append(col)

    # List all unique values for all categorical features
    feature_categories['unique_values'] = {
        col: df[col].unique().tolist() for col in feature_categories['categorical_features'] + feature_categories['derived_categorical_features']
    }

    return feature_categories


def classify_features(data, label_col='label', max_unique_for_categorical=10, **kargs):
    """
    Classify features into motif, ID, label, categorical, and numerical.
    Accepts either a single DataFrame or a tuple (X, y).

    Parameters:
    - data: pd.DataFrame or tuple(pd.DataFrame, pd.Series/np.array)
        Either a DataFrame containing features and label column, or a tuple (X, y)
        where X is a DataFrame and y is the corresponding label series or array.
    - label_col: str
        Column name for the label if `data` is a single DataFrame.
    - max_unique_for_categorical: int
        Threshold to classify numeric columns as categorical based on unique values.

    Returns:
    - dict: A dictionary with categorized features.
    """
    # import re

    # Convert Polars DataFrame to Pandas
    if isinstance(data, tuple):
        X, y = data
        df = X.copy()

        if y is not None:
            if isinstance(y, pd.DataFrame):
                if y.shape[1] != 1:
                    raise ValueError("y DataFrame must have exactly one column.")
                y = y.iloc[:, 0]
        df[label_col] = y
    elif isinstance(data, (pd.DataFrame, pl.DataFrame)):
        df = data.to_pandas() if isinstance(data, pl.DataFrame) else data.copy()
    else:
        raise TypeError("data must be a DataFrame or tuple (X, y).")

    kmer_pattern = kargs.get("kmer_pattern", r'^\d+mer_.*')  # Regex for k-mers

    feature_categories = {
        'motif_features': [col for col in df.columns if re.match(kmer_pattern, col)],
        'id_columns': [col for col in df.columns if col.endswith('_id') or col == 'id'],
        'class_labels': [label_col] if label_col in df.columns else [],
        'categorical_features': [],
        'numerical_features': [],
        'derived_categorical_features': []
    }

    # Classify each column
    for col in df.columns:
        if (col in feature_categories['motif_features'] or
            col in feature_categories['id_columns'] or
            col == label_col):
            continue

        col_data = df[col]
        unique_values = set(col_data.dropna().unique())

        if (isinstance(col_data.dtype, pd.CategoricalDtype) or
            col_data.dtype == 'object' or
            unique_values <= {True, False}):
            feature_categories['categorical_features'].append(col)
        elif pd.api.types.is_numeric_dtype(col_data):
            if len(unique_values) <= max_unique_for_categorical:
                feature_categories['derived_categorical_features'].append(col)
            else:
                feature_categories['numerical_features'].append(col)

    # Collect unique values for categorical features
    cat_cols = feature_categories['categorical_features'] + feature_categories['derived_categorical_features']
    feature_categories['unique_values'] = {
        col: df[col].unique().tolist() for col in cat_cols
    }

    return feature_categories


def filter_kmer_features(df, keep_kmers=0, random_seed=42):
    """
    Filter k-mer columns from the input dataframe.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - keep_kmers (int): How many k-mer features to keep. Default = 0 (remove all).
    - random_seed (int): Random seed for reproducibility when picking a subset of k-mers.

    Returns:
    - pd.DataFrame or pl.DataFrame: The filtered DataFrame.
    """
    is_polars = isinstance(df, pl.DataFrame)

    # Classify the features
    feature_dict = classify_features(df)
    kmer_cols = feature_dict['motif_features']

    if keep_kmers <= 0:
        # Remove all k-mer features
        if is_polars:
            return df.drop(kmer_cols)
        else:
            return df.drop(columns=kmer_cols)
    else:
        # Keep a subset (either the first N or random sample)
        random.seed(random_seed)
        if keep_kmers >= len(kmer_cols):
            # Nothing to drop
            return df
        else:
            # Randomly pick some subset of k-mers
            keep_subset = random.sample(kmer_cols, k=keep_kmers)
            drop_subset = [col for col in kmer_cols if col not in keep_subset]
            if is_polars:
                return df.drop(drop_subset)
            else:
                return df.drop(columns=drop_subset)


def diagnose_variable_type(df, column_name, max_unique_for_categorical=10):
    """
    Diagnose the type of a variable and explain the rationale.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - column_name (str): The name of the column to diagnose.
    - max_unique_for_categorical (int): Threshold for the maximum number of unique values
      to classify a numerical column as categorical (default is 10).

    Returns:
    - str: The inferred type of the variable (e.g., 'categorical', 'numerical', etc.).
    - str: A diagnosis explaining why the variable is classified as such.
    """
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    col_data = df[column_name]

    unique_values = set(col_data.dropna().unique())
    unique_values_str = f"Unique values: {list(unique_values)[:5]}{'...' if len(unique_values) > 5 else ''}"

    # Handle categorical or boolean-like columns
    if isinstance(col_data.dtype, pd.CategoricalDtype) or col_data.dtype == 'object' or unique_values <= {True, False}:
        return 'categorical', (
            f"The column '{column_name}' is classified as categorical because it contains boolean values or non-numeric types. "
            f"{unique_values_str}"
        )
    
    # Handle boolean-like values with nulls
    if unique_values <= {True, False, None} or unique_values <= {True, False, float('nan')}:
        return 'categorical', (
            f"The column '{column_name}' is classified as categorical because it contains boolean-like values with missing data. "
            f"{unique_values_str}"
        )

    # Handle binary numerical columns
    if unique_values <= {0, 1} or unique_values <= {0, 1, None, float('nan')}:
        return 'binary', (
            f"The column '{column_name}' is classified as binary because it contains only 0 and 1 values, optionally with nulls. "
            f"{unique_values_str}"
        )

    # Handle numerical columns
    if pd.api.types.is_numeric_dtype(col_data):
        if len(unique_values) <= max_unique_for_categorical:
            return 'categorical', (
                f"The column '{column_name}' is classified as categorical because it has fewer than {max_unique_for_categorical} unique numeric values. "
                f"{unique_values_str}"
            )
        return 'numerical', (
            f"The column '{column_name}' is classified as numerical because it contains numeric data types. "
            f"{unique_values_str}"
        )

    # Handle columns with all null values
    if col_data.isnull().all():
        return 'unknown', f"The column '{column_name}' is classified as unknown because it contains only null values."

    # Fallback for unclear cases
    return 'unknown', (
        f"The column '{column_name}' could not be classified clearly. {unique_values_str}"
    )


def identify_null_columns(df, verbose=1):
    """
    Identify columns containing null values, including NaN, None, or other empty-like values, and their null ratios.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - verbose (int): Verbosity level (default is 1).

    Returns:
    - dict: A dictionary with columns containing null values and their null ratios.
    """
    is_polars = isinstance(df, pl.DataFrame)

    # Convert Polars DataFrame to Pandas DataFrame if necessary
    if is_polars:
        df = df.to_pandas()

    null_columns = {col: df[col].isnull().mean() for col in df.columns if df[col].isnull().any()}

    # Check for NaN explicitly in case they are not picked up by isnull
    for col in df.columns:
        if col not in null_columns and df[col].dtype != 'object':
            if df[col].isna().any():
                null_columns[col] = df[col].isna().mean()

    if verbose > 0:
        if not null_columns:
            print("[info] No columns with null or NaN values found.")
        else:
            print("[info] Columns with null or NaN values:")
            for col, ratio in null_columns.items():
                print(f"  Column: {col}, Null Ratio: {ratio:.2%}")

    return null_columns


def label_analysis_dataset(df, pred_type_col='pred_type', label_col='label', positive_class='FP', verbose=1):
    """
    Create encoded labels for the input DataFrame based on the values in another column.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - pred_type_col (str): The column name for the prediction type (default is 'pred_type').
    - label_col (str): The column name for the label (default is 'label').
    - positive_class (str): The value in pred_type_col to be encoded as 1 (default is 'FP').

    Returns:
    - df (pd.DataFrame or pl.DataFrame): The DataFrame with the new label column.
    """
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    # Check if the input dataframe is already labeled
    if label_col in df.columns:
        # Check if the label column is binary and contains only 0 and 1
        if df[label_col].nunique() == 2 and set(df[label_col].unique()) == {0, 1}:  # Binary label column
            if verbose > 0:
                print(f"[info] The dataset is already labeled with a binary label column '{label_col}'.")
            return df

    # Check if the prediction type column exists
    if pred_type_col not in df.columns:
        raise ValueError(f"Prediction type column '{pred_type_col}' not found in DataFrame.")

    # Encode the pred_type column and introduce it as a new label column
    df[label_col] = df[pred_type_col].apply(lambda x: 1 if x == positive_class else 0)

    # Drop the original pred_type column
    df = df.drop(columns=[pred_type_col])

    if is_polars:
        df = pl.DataFrame(df)

    return df


def analyze_data_labels(df, label_col='label', verbose=1, handle_missing=None, max_classes_for_classification=10, display_columns=None):
    """
    Analyze a dataset to determine if it is for classification, regression, or cluster analysis.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - label_col (str): The column name for the label/class (default is 'label').
    - verbose (int): Verbosity level (default is 1).
    - handle_missing (callable): A user-specified function to handle rows with missing labels.
        Default: None. If not specified, rows with missing labels are separated as a test set.
    - max_classes_for_classification (int): Maximum number of unique values in a label column
        to consider it as a classification problem (default is 10).
    - display_columns (list): Subset of columns to display in example rows. If None, defaults to
        all columns excluding motif features (e.g., columns matching patterns like '2mer_*', '3mer_*', '4mer_*').

    Returns:
    - dict: A dictionary with analysis results and optionally a training and test set.
    """
    is_polars = isinstance(df, pl.DataFrame)

    # Convert Polars DataFrame to Pandas DataFrame if necessary
    if is_polars:
        df = df.to_pandas()

    analysis_result = {}

    # Classify features into categories
    feature_categories = classify_features(df, label_col)

    # Check if label column exists
    if label_col not in df.columns:
        analysis_result['type'] = 'cluster_analysis'
        analysis_result['missing_label_reason'] = "Label column is missing"
        if verbose > 0:
            print(f"[info] The dataset is for cluster analysis (label column '{label_col}' is missing).")
        if verbose > 1:
            print("[summary] Analysis result:")
            print("Type: Cluster Analysis")
            print("Reason: Label column is missing")
        return analysis_result

    # Handle missing labels
    num_missing = df[label_col].isnull().sum()
    if num_missing > 0:
        analysis_result['missing_labels'] = num_missing
        if verbose > 0:
            print(f"[warning] Label column '{label_col}' contains {num_missing} missing values.")

        if handle_missing:
            df = handle_missing(df)
        else:
            # Default: Separate rows with missing labels as a test set
            test_set = df[df[label_col].isnull()]
            df = df[df[label_col].notnull()]
            analysis_result['test_set'] = test_set

            if verbose > 0:
                print(f"[info] Separated {len(test_set)} rows with missing labels as a test set.")

    # Automatically exclude motif features for display if display_columns is None
    if display_columns is None:
        display_columns = [col for col in df.columns if col not in feature_categories['motif_features']]

    # Check if label column is categorical (classification) or numerical (regression)
    if isinstance(df[label_col].dtype, pd.CategoricalDtype) or df[label_col].dtype == 'object' or \
       (pd.api.types.is_numeric_dtype(df[label_col]) and df[label_col].nunique() <= max_classes_for_classification):
        analysis_result['type'] = 'classification'
        class_counts = df[label_col].value_counts()
        analysis_result['class_counts'] = class_counts.to_dict()
        analysis_result['num_classes'] = len(class_counts)

        if verbose > 0:
            print(f"[info] The dataset is for classification.")
            print(f"[info] Number of instances for each class: {class_counts.to_dict()}")

            if verbose > 1:
                if analysis_result['num_classes'] == 2:
                    # Ensure positive_class is always 1 and negative_class is always 0
                    if 1 in class_counts.index and 0 in class_counts.index:
                        positive_class = 1
                        negative_class = 0
                    else:
                        positive_class = class_counts.index[0]
                        negative_class = class_counts.index[1]

                    # positive_examples = df[df[label_col] == positive_class][display_columns].head(10)
                    # negative_examples = df[df[label_col] == negative_class][display_columns].head(10)
                    positive_examples = df[df[label_col] == positive_class][display_columns].sample(n=10)
                    negative_examples = df[df[label_col] == negative_class][display_columns].sample(n=10)
                    print(f"[info] Example positive examples (class '{positive_class}'):")
                    display_dataframe_in_chunks(positive_examples, num_rows=positive_examples.shape[0], num_columns=5)
                    print(f"[info] Example negative examples (class '{negative_class}'):")
                    display_dataframe_in_chunks(negative_examples, num_rows=positive_examples.shape[0], num_columns=5)
                else:
                    print(f"[info] Multi-class classification with {analysis_result['num_classes']} classes.")
                    for class_value in class_counts.index[:3]:  # Show examples for up to 3 classes
                        examples = df[df[label_col] == class_value][display_columns].sample(n=10)
                        print(f"[info] Examples for class '{class_value}':")
                        display_dataframe_in_chunks(examples, num_rows=positive_examples.shape[0], num_columns=5)

    elif pd.api.types.is_numeric_dtype(df[label_col]):
        analysis_result['type'] = 'regression'
        if verbose > 0:
            print(f"[info] The dataset is for regression.")
            if verbose > 1:
                print(f"[info] Summary statistics for '{label_col}':")
                print(df[label_col].describe())
    else:
        analysis_result['type'] = 'unknown'
        if verbose > 0:
            print(f"[warning] Unable to determine the dataset type for '{label_col}'.")

    if verbose > 1:
        print("[summary] Analysis result:")
        for key, value in analysis_result.items():
            if key == 'class_counts':
                print(f"  {key}: {value}")
            elif key == 'test_set':
                print(f"  {key}: {len(value)} rows")
            else:
                print(f"  {key}: {value}")

    if verbose > 0:
        print("[info] Feature categories:")
        total_features = len(df.columns) - len(feature_categories['id_columns']) - len(feature_categories['class_labels'])
        sample_size = len(df)
        feature_sample_ratio = sample_size / total_features if total_features > 0 else None
        
        for category, cols in feature_categories.items():
            if isinstance(cols, list):
                # Display list-based categories (e.g., motif_features, id_columns, etc.)
                print(f"  {category}: {cols[:5]}{'...' if len(cols) > 5 else ''} ({len(cols)} total)")

                if category == "class_labels":
                    for col in cols:
                        unique_values = df[col].unique()
                        print(f"    Unique values in '{col}': {unique_values}")

            elif isinstance(cols, dict):
                # Display dictionary-based categories (e.g., unique_values)
                print(f"  {category}: Displaying details for dictionary-based features")
                for cat_col, unique_vals in cols.items():
                    print(f"    {cat_col}: {unique_vals[:5]}{'...' if len(unique_vals) > 5 else ''} ({len(unique_vals)} total)")
            else:
                print(f"  {category}: Unsupported data type for display")
        
        print(f"[info] Total features (excluding meta data and labels): {total_features}")
        # print(f"[info] Class labels: {feature_categories['class_labels']}")
        print(f"[info] Sample size: {sample_size}")
        print(f"[info] Sample-to-feature ratio: {feature_sample_ratio:.2f}")


    return analysis_result


def impute_missing_values(df, imputation_map, verbose=1):
    """
    Impute missing values in the DataFrame using a user-specified dictionary of default values.

    Parameters:
    - df (pd.DataFrame or pl.DataFrame): The input DataFrame.
    - imputation_map (dict): A dictionary where keys are column names and values are the default values to impute.
    - verbose (int): Verbosity level (default is 1).

    Returns:
    - pd.DataFrame or pl.DataFrame: The DataFrame with missing values imputed.
    """
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    for col, value in imputation_map.items():
        if col in df.columns:
            num_missing = df[col].isnull().sum()
            df[col] = df[col].fillna(value)
            if verbose > 0:
                print(f"[info] Imputed {num_missing} missing values in column '{col}' with value '{value}'.")
        else:
            if verbose > 0:
                print(f"[warning] Column '{col}' not found in DataFrame. No imputation performed.")

    if is_polars:
        df = pl.DataFrame(df)

    return df


def subset_training_data(df, group_cols, target_sample_size, verbose=1):
    """
    Subset a training dataset to approximately achieve a specified sample size, 
    ensuring complete transcript-specific data.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - group_cols (list): List of column names to define groups (e.g., ['gene_id', 'transcript_id']).
    - target_sample_size (int): Desired total number of samples in the subset.
    - verbose (int): Verbosity level (default is 1).

    Returns:
    - pd.DataFrame: A subset of the original DataFrame.
    """
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        df = df.to_pandas()

    grouped = df.groupby(group_cols)
    sampled_data = []
    remaining_sample_size = target_sample_size

    # Calculate approximate samples per group
    group_sizes = grouped.size()
    total_groups = len(group_sizes)
    samples_per_group = max(1, remaining_sample_size // total_groups)

    for group, group_df in grouped:
        group_size = len(group_df)

        # Ensure complete transcript-specific data
        if group_size <= samples_per_group:
            sampled_data.append(group_df)
            remaining_sample_size -= group_size
        else:
            # Ensure all splice_type variations are included
            splice_types = group_df['splice_type'].unique()
            sampled_group_df = pd.DataFrame()
            for splice_type in splice_types:
                splice_type_df = group_df[group_df['splice_type'] == splice_type]
                if len(splice_type_df) <= samples_per_group // len(splice_types):
                    sampled_group_df = pd.concat([sampled_group_df, splice_type_df])
                else:
                    sampled_group_df = pd.concat([sampled_group_df, splice_type_df.sample(n=samples_per_group // len(splice_types), random_state=42)])
            sampled_data.append(sampled_group_df)
            remaining_sample_size -= len(sampled_group_df)

        if remaining_sample_size <= 0:
            break

    # Concatenate sampled groups
    sampled_df = pd.concat(sampled_data, axis=0)

    # Add additional samples if needed
    if len(sampled_df) < target_sample_size:
        remaining_samples = target_sample_size - len(sampled_df)
        additional_samples = df.loc[~df.index.isin(sampled_df.index)].sample(n=remaining_samples, random_state=42)
        sampled_df = pd.concat([sampled_df, additional_samples], axis=0)

    if verbose > 0:
        print(f"[info] Target sample size: {target_sample_size}")
        print(f"[info] Achieved sample size: {len(sampled_df)}")
        print(f"[info] Remaining samples added: {remaining_samples if len(sampled_df) < target_sample_size else 0}")

    if is_polars:
        sampled_df = pl.DataFrame(sampled_df)

    return sampled_df


def sort_performance_by_metric_and_error_type(df, metric='f1_score', error_type='FP', df_type='polars'):
    """
    Sort the rows of the DataFrame by a specified performance metric in ascending order
    and then by error type (either FP or FN).

    Parameters:
    - df (pl.DataFrame or pd.DataFrame): The input DataFrame.
    - metric (str): The performance metric to sort by (default is 'f1_score').
    - error_type (str): The error type to sort by (either 'FP' or 'FN', default is 'FP').
    - df_type (str): The type of DataFrame ('polars' or 'pandas', default is 'polars').

    Returns:
    - pl.DataFrame or pd.DataFrame: The sorted DataFrame.
    """
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame columns.")
    if error_type not in ['FP', 'FN']:
        raise ValueError(f"Error type '{error_type}' must be either 'FP' or 'FN'.")

    if df_type == 'polars':
        # Sort by the specified metric in ascending order
        sorted_df = df.sort(metric, descending=False)

        # Further sort by the specified error type
        sorted_df = sorted_df.sort(error_type, descending=False)

    elif df_type == 'pandas':
        # Sort by the specified metric in ascending order
        sorted_df = df.sort_values(by=metric, ascending=True)

        # Further sort by the specified error type
        sorted_df = sorted_df.sort_values(by=error_type, ascending=True)

    else:
        raise ValueError(f"Unsupported DataFrame type '{df_type}'. Use 'polars' or 'pandas'.")

    return sorted_df


def abridge_sequence(sequence, max_length=100, display=False):
    """
    Print a potentially long DNA sequence, showing only the first and last 100nt with '...' in the middle.

    Parameters:
    - sequence (str): The DNA sequence to print.
    - max_length (int): The number of nucleotides to show at the beginning and end (default is 100).
    """
    if len(sequence) <= 2 * max_length:
        truncated_sequence = sequence
        # print_with_indent(f"Sequence: {sequence}")
    else:
        truncated_sequence = f"{sequence[:max_length]}...{sequence[-max_length:]}"
        # print_with_indent(f"Sequence: {sequence[:max_length]}...{sequence[-max_length:]}")

    if display:
        print(f"Sequence: {truncated_sequence}")

    return truncated_sequence


####################################################################################################


def plot_feature_distributions(
    X, 
    y, 
    feature_list, 
    label_col="label",
    label_text_0="TP",   # Label for class 0
    label_text_1="FP",   # Label for class 1
    plot_type="box",     # Options: "box", "violin", or fallback "histplot" for numeric features.
    n_cols=3, 
    figsize=None, 
    title="Feature Distributions",
    output_path=None, 
    show_plot=False, 
    verbose=1,
    top_k_motifs=None,        # If provided, only plot top N motif features (k-mers)
    kmer_pattern=r'^\d+mer_.*', # Regex to identify motif features
    max_features=60,  # Maximum number of features to include in the visualization
    use_swarm_for_motifs=True  # New parameter: overlay swarm plot for motif features
):
    """
    Create plots for features based on their type, integrating feature classification.
    For numeric features, produces box/violin/histograms.
    For categorical features, produces countplots.
    Optionally, restricts motif (k-mer) features to the top N based on mean difference,
    and for motif features, optionally overlays a swarm plot to display all data points.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or np.array
        Binary labels corresponding to X's rows.
    feature_list : list
        List of column names in X to plot.
    label_col : str
        Name to use for the label column when classifying features.
    label_text_0 : str
        Text label for class 0 (default: "TP").
    label_text_1 : str
        Text label for class 1 (default: "FP").
    plot_type : str
        "box", "violin", or fallback "histplot" for numeric features.
    n_cols : int
        Number of columns in the subplot grid.
    figsize : tuple
        (width, height) in inches. If None, auto-calculated.
    title : str
        The overall figure title.
    output_path : str or None
        If given, save the figure to this path.
    show_plot : bool
        Whether to display the plot.
    verbose : int
        Level of verbosity.
    top_k_motifs : int or None
        If provided, only the top N motif features (matching kmer_pattern) will be plotted.
    kmer_pattern : str
        Regular expression to identify motif (k-mer) feature names.
    use_swarm_for_motifs : bool
        If True, for motif features, overlay a swarm plot (showing all data points) on top of the boxplot.
    
    Returns
    -------
    None

    Updates
    -------
    - 2025-03-11: Added use_swarm_for_motifs parameter to overlay swarm plot for motif features.
    """
    from .analysis_utils import classify_features  # Assumes this function is available

    # Combine X and y to classify features
    temp_df = X.copy()
    temp_df[label_col] = y
    feature_categories = classify_features(temp_df, label_col=label_col)
    
    # Retrieve classified feature lists
    categorical_vars = (feature_categories.get("categorical_features", []) +
                        feature_categories.get("derived_categorical_features", []))
    numerical_vars = feature_categories.get("numerical_features", [])
    motif_vars = feature_categories.get("motif_features", [])

    # Set an upper limit on the number of features to plot
    # If there are too many features, select the top ones based on mean difference
    if len(feature_list) > max_features:
        if verbose > 0:
            print(f"Too many features ({len(feature_list)}) for visualization. Selecting top {max_features} based on mean difference.")
        
        # Calculate mean difference for each feature
        mean_diffs = {}
        for feature in feature_list:
            mean_pos = X.loc[y == 1, feature].mean()
            mean_neg = X.loc[y == 0, feature].mean()
            mean_diffs[feature] = abs(mean_pos - mean_neg)

        # Select top features
        sorted_features = sorted(mean_diffs.items(), key=lambda x: x[1], reverse=True)
        feature_list = [f[0] for f in sorted_features[:max_features]]
        
        if verbose > 0:
            print(f"Selected top {len(feature_list)} features based on mean difference between classes.")

    # If top_k_motifs is provided, select top N motif features based on absolute difference in means.
    if top_k_motifs is not None:
        motif_in_list = [feat for feat in feature_list if re.match(kmer_pattern, feat)]
        if len(motif_in_list) > 0:
            df_for_calc = X.copy()
            df_for_calc[label_col] = y
            diff_dict = {}
            for feat in motif_in_list:
                group_means = df_for_calc.groupby(label_col)[feat].mean()
                if group_means.shape[0] < 2:
                    continue
                diff = abs(group_means.iloc[0] - group_means.iloc[1])
                diff_dict[feat] = diff
            top_motifs = sorted(diff_dict, key=lambda k: diff_dict[k], reverse=True)[:top_k_motifs]
            if verbose:
                print(f"[info] Selected top {top_k_motifs} motif features based on mean difference: {top_motifs}")
            non_motif = [feat for feat in feature_list if not re.match(kmer_pattern, feat)]
            feature_list = non_motif + top_motifs

    # Determine grid layout for plots
    n_features = len(feature_list)
    n_rows = math.ceil(n_features / n_cols)
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # Ensure y is a pandas Series aligned with X
    y_series = pd.Series(y, index=X.index, name=label_col)

    for i, feat in enumerate(feature_list):
        ax = axes[i]
        if feat not in X.columns:
            if verbose:
                print(f"[warning] Feature '{feat}' not found in X; skipping.")
            ax.set_visible(False)
            continue

        plot_df = pd.DataFrame({feat: X[feat], label_col: y_series})
        dtype = X[feat].dtype

        # For numeric (and motif) features
        if feat in numerical_vars or feat in motif_vars:
            if plot_type == "box":
                if feat in motif_vars and use_swarm_for_motifs:
                    # Plot boxplot without outlier markers
                    sns.boxplot(data=plot_df, x=label_col, y=feat, ax=ax, showfliers=False)
                    sample_df = plot_df.sample(min(500, len(plot_df)), random_state=42)
                    sns.stripplot(data=sample_df, x=label_col, y=feat, ax=ax,
                                color='black', size=4, alpha=0.6, jitter=True)
                    
                    # Swarm plot is slow for large datasets
                    # sns.swarmplot(data=plot_df, x=label_col, y=feat, ax=ax, color='black', size=4, alpha=0.8)
                else:
                    sns.boxplot(data=plot_df, x=label_col, y=feat, ax=ax)
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels([label_text_0, label_text_1], fontsize=9)
            elif plot_type == "violin":
                sns.violinplot(data=plot_df, x=label_col, y=feat, ax=ax)
                ax.set_xticks([0, 1])
                ax.set_xticklabels([label_text_0, label_text_1], fontsize=9)
            else:
                sns.histplot(data=plot_df, x=feat, hue=label_col, kde=True, ax=ax)
                
                # Update the legend with custom labels
                legend = ax.get_legend()
                if legend:
                    legend.set_title(None)
                    for t, l in zip(legend.texts, [label_text_0, label_text_1]):
                        t.set_text(l)
            
            ax.set_title(feat, fontsize=10)
        elif feat in categorical_vars:
            # For general categorical features (including binary)
            sns.countplot(data=plot_df, x=feat, hue=label_col, ax=ax)
            ax.set_title(feat, fontsize=10)
            ax.tick_params(axis='x', labelrotation=45)
            
            # Update the legend with custom labels for categorical features
            legend = ax.get_legend()
            if legend:
                legend.set_title(None)
                for t, l in zip(legend.texts, [label_text_0, label_text_1]):
                    t.set_text(l)
        else:
            if verbose:
                print(f"[warning] Skipping feature '{feat}' as its type is not handled.")
            ax.set_visible(False)

    # Hide any extra subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path is not None:
        if verbose:
            print(f"[output] Saving feature distribution plot to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)  # Close the figure to prevent memory leaks


def plot_feature_distributions_v1(
    X, 
    y, 
    feature_list, 
    label_col="label",
    label_text_0="TP",   # Label for class 0
    label_text_1="FP",   # Label for class 1
    plot_type="box",     # Options: "box", "violin", or fallback "histplot" for numeric features.
    n_cols=2,            # Changed default from 3 to 2 columns
    figsize=None, 
    title="Feature Distributions",
    output_path=None, 
    show_plot=False, 
    verbose=1,
    top_k_motifs=None,        # If provided, only plot top N motif features (k-mers)
    kmer_pattern=r'^\d+mer_.*', # Regex to identify motif features
    use_swarm_for_motifs=True,  # New parameter: overlay swarm plot for motif features
    show_feature_stats=True,   # New parameter: show mean/median for each class
    annotate_sparse=True      # New parameter: add annotations for sparse features
):
    """
    Create plots for features based on their type, integrating feature classification.
    For numeric features, produces box/violin/histograms.
    For categorical features, produces countplots.
    Optionally, restricts motif (k-mer) features to the top N based on mean difference,
    and for motif features, optionally overlays a swarm plot to display all data points.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or np.array
        Binary labels corresponding to X's rows.
    feature_list : list
        List of column names in X to plot.
    label_col : str
        Name to use for the label column when classifying features.
    label_text_0 : str
        Text label for class 0 (default: "TP").
    label_text_1 : str
        Text label for class 1 (default: "FP").
    plot_type : str
        "box", "violin", or fallback "histplot" for numeric features.
    n_cols : int
        Number of columns in the subplot grid (default: 2).
    figsize : tuple
        (width, height) in inches. If None, auto-calculated.
    title : str
        The overall figure title.
    output_path : str or None
        If given, save the figure to this path.
    show_plot : bool
        Whether to display the plot.
    verbose : int
        Level of verbosity.
    top_k_motifs : int or None
        If provided, only the top N motif features (matching kmer_pattern) will be plotted.
    kmer_pattern : str
        Regular expression to identify motif (k-mer) feature names.
    use_swarm_for_motifs : bool
        If True, for motif features, overlay a swarm plot (showing all data points) on top of the boxplot.
    show_feature_stats : bool
        If True, add annotations with mean/median values for each class.
    annotate_sparse : bool
        If True, add explanatory annotations for sparse features (when boxplot boxes are not visible).
    
    Returns
    -------
    None

    Updates
    -------
    - 2025-03-11: Added use_swarm_for_motifs parameter to overlay swarm plot for motif features.
    - 2025-03-26: Enhanced visualization with custom legend labels for categorical plots, feature stats,
                  and better handling of sparse data in k-mer features.
    - 2025-03-27: Improved subplot spacing and legend placement to prevent overcrowding.
                  Reduced default columns to 2 and positioned legends outside plots.
    """
    from .analysis_utils import classify_features  # Assumes this function is available

    # Combine X and y to classify features
    temp_df = X.copy()
    temp_df[label_col] = y
    feature_categories = classify_features(temp_df, label_col=label_col)
    
    # Retrieve classified feature lists
    categorical_vars = (feature_categories.get("categorical_features", []) +
                        feature_categories.get("derived_categorical_features", []))
    numerical_vars = feature_categories.get("numerical_features", [])
    motif_vars = feature_categories.get("motif_features", [])
    
    # If top_k_motifs is provided, select top N motif features based on absolute difference in means.
    if top_k_motifs is not None:
        motif_in_list = [feat for feat in feature_list if re.match(kmer_pattern, feat)]
        if len(motif_in_list) > 0:
            df_for_calc = X.copy()
            df_for_calc[label_col] = y
            diff_dict = {}
            for feat in motif_in_list:
                group_means = df_for_calc.groupby(label_col)[feat].mean()
                if group_means.shape[0] < 2:
                    continue
                diff = abs(group_means.iloc[0] - group_means.iloc[1])
                diff_dict[feat] = diff
            top_motifs = sorted(diff_dict, key=lambda k: diff_dict[k], reverse=True)[:top_k_motifs]
            if verbose:
                print(f"[info] Selected top {top_k_motifs} motif features based on mean difference: {top_motifs}")
            non_motif = [feat for feat in feature_list if not re.match(kmer_pattern, feat)]
            feature_list = non_motif + top_motifs

    # Determine grid layout for plots
    n_features = len(feature_list)
    n_rows = math.ceil(n_features / n_cols)
    
    # Increase figure size to accommodate legends outside plots
    if figsize is None:
        # Wider figure to accommodate legends outside plots (25% wider)
        figsize = (6.5 * n_cols, 5 * n_rows)  

    # Create figure with more spacing between subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # Increase horizontal spacing for legend placement
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  
    
    # Handle case when there's only one row
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Ensure y is a pandas Series aligned with X
    y_series = pd.Series(y, index=X.index, name=label_col)

    for i, feat in enumerate(feature_list):
        ax = axes[i]
        if feat not in X.columns:
            if verbose:
                print(f"[warning] Feature '{feat}' not found in X; skipping.")
            ax.set_visible(False)
            continue

        plot_df = pd.DataFrame({feat: X[feat], label_col: y_series})
        dtype = X[feat].dtype
        
        # Get basic stats for annotations
        stats = {}
        if show_feature_stats:
            for label_val, label_name in [(0, label_text_0), (1, label_text_1)]:
                group_data = plot_df[plot_df[label_col] == label_val][feat]
                stats[label_name] = {
                    'mean': group_data.mean(),
                    'median': group_data.median(),
                    'unique_vals': len(group_data.unique()),
                    'count': len(group_data)
                }

        # For numeric (and motif) features
        if feat in numerical_vars or feat in motif_vars:
            is_sparse = False
            # Check if data is sparse (few unique values)
            if feat in motif_vars:
                unique_vals = len(plot_df[feat].unique())
                is_sparse = unique_vals <= 3  # Consider it sparse if 3 or fewer unique values
            
            if plot_type == "box":
                if feat in motif_vars and use_swarm_for_motifs:
                    # Plot boxplot without outlier markers
                    sns.boxplot(data=plot_df, x=label_col, y=feat, ax=ax, showfliers=False)
                    
                    # Use a sample for large datasets to prevent overcrowding
                    sample_size = min(500, len(plot_df))
                    sample_df = plot_df.sample(sample_size, random_state=42)
                    
                    # Add stripplot to show individual data points
                    sns.stripplot(data=sample_df, x=label_col, y=feat, ax=ax,
                                 color='black', size=3, alpha=0.6, jitter=True)
                    
                    # If it's a sparse feature, add annotation to explain dots
                    if is_sparse and annotate_sparse:
                        ax.text(0.98, 0.02, "Note: Each dot = one sample\n"
                                          "Few unique values may result in\n"
                                          "missing boxes in the plot.",
                                transform=ax.transAxes, ha='right', va='bottom',
                                fontsize=7, bbox=dict(facecolor='white', alpha=0.8))
                else:
                    sns.boxplot(data=plot_df, x=label_col, y=feat, ax=ax)
                
                # Add x-axis labels
                ax.set_xticks([0, 1])
                ax.set_xticklabels([label_text_0, label_text_1], fontsize=9)
                
                # Move legend outside the plot for boxplots
                legend = ax.get_legend()
                if legend:
                    legend.set_title(None)
                    for t, l in zip(legend.texts, [label_text_0, label_text_1]):
                        t.set_text(l)
                    # Move legend to outside the plot
                    ax.legend(bbox_to_anchor=(1.05, 0.95), loc='upper left', borderaxespad=0)
                
            elif plot_type == "violin":
                sns.violinplot(data=plot_df, x=label_col, y=feat, ax=ax)
                ax.set_xticks([0, 1])
                ax.set_xticklabels([label_text_0, label_text_1], fontsize=9)
                
                # Move legend outside the plot for violinplots
                legend = ax.get_legend()
                if legend:
                    legend.set_title(None)
                    for t, l in zip(legend.texts, [label_text_0, label_text_1]):
                        t.set_text(l)
                    # Move legend to outside the plot
                    ax.legend(bbox_to_anchor=(1.05, 0.95), loc='upper left', borderaxespad=0)
                
            else:
                sns.histplot(data=plot_df, x=feat, hue=label_col, kde=True, ax=ax)
                # Update the legend with custom labels and move outside the plot
                legend = ax.get_legend()
                if legend:
                    legend.set_title(None)
                    for t, l in zip(legend.texts, [label_text_0, label_text_1]):
                        t.set_text(l)
                    # Move legend to outside the plot
                    ax.legend(bbox_to_anchor=(1.05, 0.95), loc='upper left', borderaxespad=0)
            
            # Add feature statistics as annotations - positioned at the top
            if show_feature_stats and not is_sparse:
                stat_text = []
                for label_name in [label_text_0, label_text_1]:
                    mean = stats[label_name]['mean']
                    median = stats[label_name]['median']
                    stat_text.append(f"{label_name}: mean={mean:.2f}, median={median:.2f}")
                
                # Place stats at the bottom of the plot
                ax.text(0.02, 0.02, "\n".join(stat_text),
                       transform=ax.transAxes, ha='left', va='bottom',
                       fontsize=7, bbox=dict(facecolor='white', alpha=0.7))
            
            # Add extra padding to y-axis to prevent annotations from overlapping with data
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            ax.set_ylim(y_min, y_max + 0.1 * y_range)  # Add 10% more space at the top
            
            # Use more descriptive title
            if feat in motif_vars:
                # For k-mer features, extract k-mer size and pattern from feature name
                match = re.match(r'(\d+)mer_(.*)', feat)
                if match:
                    k, pattern = match.groups()
                    ax.set_title(f"{k}-mer: {pattern}", fontsize=10)
                else:
                    ax.set_title(f"Motif: {feat}", fontsize=10)
            else:
                ax.set_title(feat, fontsize=10)
                
        elif feat in categorical_vars:
            # For general categorical features (including binary)
            sns.countplot(data=plot_df, x=feat, hue=label_col, ax=ax)
            
            # Update the legend with custom labels and move it OUTSIDE the plot
            legend = ax.get_legend()
            if legend:
                legend.set_title(None)
                for t, l in zip(legend.texts, [label_text_0, label_text_1]):
                    t.set_text(l)
                # Move legend to outside the plot
                ax.legend(bbox_to_anchor=(1.05, 0.95), loc='upper left', borderaxespad=0)
            
            # Calculate and display percentage distribution
            if show_feature_stats:
                # Get category counts for each class
                cat_stats = []
                for label_val, label_name in [(0, label_text_0), (1, label_text_1)]:
                    class_count = (plot_df[label_col] == label_val).sum()
                    for cat_val in plot_df[feat].unique():
                        count = ((plot_df[feat] == cat_val) & (plot_df[label_col] == label_val)).sum()
                        pct = (count / class_count) * 100 if class_count > 0 else 0
                        cat_stats.append(f"{label_name}, {cat_val}: {pct:.1f}%")
                
                # Add text annotation with percentages (at the BOTTOM)
                ax.text(0.02, 0.02, "\n".join(cat_stats[:4]),  # Show first 4 to avoid overcrowding
                       transform=ax.transAxes, ha='left', va='bottom',
                       fontsize=7, bbox=dict(facecolor='white', alpha=0.7))
                
            ax.set_title(feat, fontsize=10)
            ax.tick_params(axis='x', labelrotation=45)
        else:
            if verbose:
                print(f"[warning] Skipping feature '{feat}' as its type is not handled.")
            ax.set_visible(False)

    # Hide any extra subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, y=0.98)
    # Use tight_layout but with more padding for legends
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])

    if output_path is not None:
        if verbose:
            print(f"[output] Saving feature distribution plot to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)  # Close the figure to prevent memory leaks


def compute_feature_attributions(
    model,
    x_row,        # 1D vector or row of shape (n_features,)
    X_train=None, # Optional reference for shap background, or pass the entire training set
    feature_names=None
):
    """
    Compute local SHAP attributions for a single row x_row using a trained XGBoost (or other) model.

    Returns
    -------
    shap_values_row : np.array of shape (n_features,)
    feature_names : list of str

    Memo
    ----
    - Do local attributions for just one row with TreeExplainer
        explainer = shap.TreeExplainer(xgb_model)
        
        # for a single row x_row
        shap_values_single = explainer.shap_values(x_row.reshape(1, -1))
        # => shape (1, n_features) for a 2-class model or (1, 2, n_features) sometimes
        
        shap_values_array = shap_values_single[0]  # shape (n_features,) for the positive class
    """
    # Create a small shap explainer
    # Typically you do something like:
    explainer = shap.Explainer(model, X_train, feature_names=feature_names)
    
    # x_row is shape (n_features,) or (1, n_features)
    # We might need to reshape to (1, -1) for the explainer
    if x_row.ndim == 1:
        x_row = x_row.reshape(1, -1)

    shap_values_single = explainer(x_row)  # => shap.Explanation object
    # convert to np array
    shap_values_array = shap_values_single.values[0]  # shape (n_features,)

    return shap_values_array, feature_names


def plot_pairwise_scatter(
    X, 
    y, 
    features, 
    pos_label=1, 
    neg_label=0, 
    figsize=(8,6),
    output_path=None,
    show_plot=False,
    verbose=1, 
    **kargs
):
    """
    Create a pairwise scatter (pairplot) for the given features, 
    coloring points by the label.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or np.array
        Binary labels.
    features : list
        List of feature names (columns in X).
    pos_label : int
        Positive label in y.
    neg_label : int
        Negative label in y.
    figsize : tuple
        Overall figure dimension. 
        Note: pairplot internally sets subplots. 
    output_path : str or None
        Path to save the figure. If None, no saving.
    show_plot : bool
        Whether to display the plot with plt.show().
    verbose : int
        Level of verbosity.

    Returns
    -------
    None
    """
    # import matplotlib.pyplot as plt
    # import seaborn as sns

    plot_df = X[features].copy()
    plot_df["label"] = y.values

    use_inf_as_na = kargs.get("use_inf_as_na", True)

    # Convert infinite values to NaN to avoid warnings
    if use_inf_as_na:
        plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Seaborn pairplot doesn't directly respect figsize from the user;
    # Instead we can approximate with 'height' param.
    n_features = len(features)
    # E.g. height=2.5 => total width ~ 2.5*n_features
    # We'll just rely on user to specify
    
    g = sns.pairplot(
        plot_df, 
        hue="label", 
        corner=True, 
        diag_kind="hist",
        height=min(figsize[0]/n_features, figsize[1]/n_features) if n_features>1 else figsize[0], 
        plot_kws={"alpha": 0.7}
    )
    g.fig.suptitle("Pairwise Scatter", y=1.02)

    # Save if output_path
    if output_path is not None:
        if verbose:
            print(f"[output] Saving pairwise scatter to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')

    # Show or close
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to prevent memory leaks


def plot_shap_beeswarm(
    shap_values, 
    X, 
    max_display=20, 
    output_path=None,
    show_plot=False,
    verbose=1
):
    """
    Plot a SHAP beeswarm summary (shap.summary_plot with plot_type='dot')
    for a given set of shap_values and corresponding data X.

    Parameters
    ----------
    shap_values : np.array or shap.Explanation
        The SHAP values for each sample, shape (n_samples, n_features).
    X : pd.DataFrame or np.array
        The data used to compute shap_values. 
        If pd.DataFrame, the columns are used as feature names.
    max_display : int
        Number of top features to display in the plot.
    output_path : str or None
        Where to save the figure. e.g. "shap_beeswarm.pdf"
    show_plot : bool
        If True, display the figure.
    verbose : int
        Verbosity level.

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    # shap.summary_plot is the typical method for "beeswarm"
    # We'll intercept the show by shap.plots._force_matplotlib_show or show=False
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        # fallback
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # By default shap.summary_plot tries to show plot immediately
    # We can force show=False, then do a manual save
    shap.summary_plot(
        shap_values, 
        features=X, 
        feature_names=feature_names, 
        plot_type="dot",  # dot is the typical beeswarm style
        max_display=max_display, 
        show=False
    )

    if output_path is not None:
        if verbose:
            print(f"[output] Saving SHAP beeswarm plot to: {output_path}")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to prevent memory leaks


def plot_shap_global_importance_v0(df, title, output_path, top_k=20):
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance_score', y='feature', data=df.head(top_k), palette="viridis")
    plt.xlabel("Mean Absolute SHAP")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_shap_global_importance(df, title, output_path, top_k=20, colormap="viridis"):
    """
    Plots the top_k features from df (which has 'feature' and 'importance_score' columns)
    using a continuous color gradient. 
    """
    import matplotlib.cm as cm

    # Select top k
    df_top = df.head(top_k)

    # Get a matplotlib colormap (e.g. 'viridis', 'plasma', 'coolwarm', etc.)
    cmap = plt.colormaps[colormap]

    # Normalize importance scores to [0..1]
    max_val = df_top['importance_score'].max()
    min_val = df_top['importance_score'].min()

    # Avoid division by zero if all scores are identical
    denom = (max_val - min_val) if max_val != min_val else 1e-9

    normalized = (df_top['importance_score'] - min_val) / denom
    colors = [cmap(val) for val in normalized]

    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x='importance_score',
        y='feature',
        data=df_top,
        palette=colors,   # pass the RGBA list
        edgecolor='black'
    )
    plt.xlabel("Mean Absolute SHAP", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.title(title, fontsize=16, pad=15)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_feature_importance(
    df,
    title,
    output_path,
    top_k=20,
    colormap="viridis",
    use_continuous_color=False,
    figure_size=(10, 8),
    verbose=1,
    rank_by_abs=False  # New parameter  
):
    """
    Plots feature importance for the top_k features in a horizontal barplot.
    The input DataFrame `df` is assumed to have columns: 'feature' and 'importance_score'.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ['feature', 'importance_score'].
    title : str
        Title of the plot.
    output_path : str
        File path where the plot is saved (e.g. "importance_plot.pdf").
    top_k : int, default=20
        Number of top features to plot.
    colormap : str, default="viridis"
        The name of the matplotlib colormap or Seaborn palette to use.
        e.g. "viridis", "plasma", "coolwarm", "Blues", etc.
    use_continuous_color : bool, default=False
        If True, map the importance scores to a continuous colormap gradient.
        If False, pass the colormap string directly to Seaborn as a discrete palette.
    figure_size : tuple, default=(10, 8)
        The figure size in inches.
    verbose : int, default=1
        Verbosity level. Set to 0 to suppress print statements.
    rank_by_abs : bool, default=False
        If True, sort features by the absolute value of importance_score (descending).
        If False, sort features by raw importance_score (descending).

    Returns
    -------
    None. (The plot is saved to `output_path`.)
    """
    import matplotlib.cm as cm

    # -------------------------------
    # 1) Sort the DataFrame
    # -------------------------------
    if rank_by_abs:
        # Sort by absolute value in descending order
        df_sorted = df.sort_values(
            by="importance_score",
            key=lambda x: x.abs(),  # apply abs() before sorting
            ascending=False
        )
    else:
        # Sort by raw importance in descending order
        df_sorted = df.sort_values(by="importance_score", ascending=False)

    df_top = df_sorted.head(top_k)

    if verbose > 0:
        print(f"[info] Plotting top {top_k} features from a total of {len(df)}.")
        sort_mode = "absolute values" if rank_by_abs else "raw values"
        print(f"[info] Sorting features by {sort_mode} of importance scores.")

    fig = plt.figure(figsize=figure_size)

    # -------------------------------
    # 2) Prepare the colors/palette
    # -------------------------------
    if use_continuous_color:
        cmap = plt.colormaps[colormap]

        # If you are *also* using rank_by_abs, note that the highest absolute
        # value will map to the darkest color. Negative or positive won't matter
        # for the color intensity, only magnitude.
        # But the sign does affect how the bar extends (the bar is still
        # drawn in the positive direction, since we're plotting "importance_score"
        # on the x-axis).
        scores = df_top["importance_score"].values
        min_val, max_val = scores.min(), scores.max()
        denom = (max_val - min_val) if max_val != min_val else 1e-9
        norm_scores = (scores - min_val) / denom

        rgba_colors = [cmap(val) for val in norm_scores]

        sns.barplot(
            x="importance_score",
            y="feature",
            hue="feature",
            data=df_top,
            palette=rgba_colors,
            edgecolor="black",
            legend=False
        )
    else:
        sns.barplot(
            x="importance_score",
            y="feature",
            hue="feature",
            data=df_top,
            palette=colormap, 
            edgecolor="black",
            legend=False
        )

    plt.xlabel("Importance Score", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(title, fontsize=14, pad=10)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    if verbose > 0:
        print(f"[info] Feature importance plot saved to: {output_path}")

    plt.close(fig)  # Close the figure to prevent memory leaks


####################################################################################################
# Demonstration functions
####################################################################################################

def demo_sort_performance_by_metric_and_error_type(**kargs): 
    from .performance_analyzer import PerformanceAnalyzer

    print("Loading performance data...")
    
    # Replace this with your data loading logic
    eval_dir = kargs.get('eval_dir', PerformanceAnalyzer.eval_dir)
    separator = kargs.get('separator', kargs.get('sep', '\t'))

    # Initialize ModelEvaluationFileHandler
    mefd = ModelEvaluationFileHandler(eval_dir, separator=separator)

    # Load the performance dataframe
    performance_df = mefd.load_performance_df(aggregated=True)

    # Example usage
    sorted_full_performance_df = sort_performance_by_metric_and_error_type(performance_df, metric='f1_score', error_type='FP', df_type='polars')
    display_dataframe_in_chunks(sorted_full_performance_df, title="Sorted Performance DataFrame")


def demo_count_unique_ids(**kargs):

    # Example usage
    df_trainset = pd.DataFrame({
        'gene_id': ['gene1', 'gene2', 'gene1', 'gene3'],
        'transcript_id': ['tx1', 'tx2', 'tx1', 'tx3'],
        'position': [1, 2, 3, 4],
        'score': [0.1, 0.2, 0.3, 0.4],
        'splice_type': ['type1', 'type2', 'type1', 'type3'],
        'chrom': ['chr1', 'chr2', 'chr1', 'chr3'],
        'strand': ['+', '-', '+', '-']
    })

    unique_counts = count_unique_ids(df_trainset)
    print(unique_counts)


def demo_check_duplicates(**kargs):

    # Pandas DataFrame
    pd_df = pd.DataFrame({
        "gene_id": ["gene1", "gene2", "gene1", "gene3"],
        "transcript_id": ["tx1", "tx2", "tx1", "tx3"],
        "score": [0.9, 0.8, 0.9, 0.7]
    })

    # Polars DataFrame
    pl_df = pl.DataFrame({
        "gene_id": ["gene1", "gene2", "gene1", "gene3"],
        "transcript_id": ["tx1", "tx2", "tx1", "tx3"],
        "score": [0.9, 0.8, 0.9, 0.7]
    })

    # Subset to check for duplicates
    subset = ["gene_id", "transcript_id"]

    # Pandas
    num_duplicates_pd = check_duplicates(pd_df, subset=subset, return_rows=False)
    print(f"Number of duplicates in Pandas DataFrame: {num_duplicates_pd}")

    # Polars
    num_duplicates_pl = check_duplicates(pl_df, subset=subset, return_rows=False)
    print(f"Number of duplicates in Polars DataFrame: {num_duplicates_pl}")

    # Get duplicate rows (Polars example)
    duplicate_rows_pl = check_duplicates(pl_df, subset=subset, return_rows=True, verbose=2)
    print("Duplicate rows in Polars DataFrame:")
    print(duplicate_rows_pl)
    
    print('=' * 40)

    # Another example DataFrame
    pd_df = pd.DataFrame({
        "gene_id": ["gene1", "gene2", "gene1", "gene3", "gene1"],
        "transcript_id": ["tx1", "tx2", "tx1", "tx3", "tx1"],
        "score": [0.9, 0.8, 0.95, 0.7, 0.9],
        "status": ["active", "inactive", "inactive", "active", "active"]
    })

    count = check_duplicates(pd_df, subset=["gene_id", "transcript_id"], verbose=0)
    print(f"Number of duplicate rows: {count}")

    print('-' * 40)
    check_duplicates(pd_df, subset=["gene_id", "transcript_id"], verbose=1, example_limit=2)

    print('-' * 40)
    check_duplicates(pd_df, subset=["gene_id", "transcript_id"], verbose=2, example_limit=2)


def demo_valid_ids(**kargs): 

    data = {
        "gene_id": ["ENSG00000286505", "NM_001256789", "XYZ12345", "12345", None],
        "transcript_id": ["ENST000001", "XM_005123456", "ABC56789", 0, ""],
        "position": [2147, 2416, 38603, 1836, 12345],
        "score": [0.626, 0.817, 0.584, 0.744, 0.50],
    }

    df = pd.DataFrame(data)

    valid_prefixes = {
        'gene': ['ENSG', 'NM', 'XYZ'],  # Ensembl, RefSeq, and custom prefixes
        'transcript': ['ENST', 'XM', 'ABC']  # Ensembl, RefSeq, and custom prefixes
    }

    invalid_rows = validate_ids(df, col_gid='gene_id', col_tid='transcript_id', valid_prefixes=valid_prefixes, verbose=2)


def display_dictionary(dictionary, indent=2):
    """
    Display a dictionary in a readable format.

    Parameters:
    - dictionary (dict): The input dictionary.
    - indent (int): The number of spaces for indentation (default is 2).

    Returns:
    - None
    """
    for key, values in dictionary.items():
        print(f"{key}:")
        for value in values:
            print(f"{' ' * indent}{value}")


def demo_plot_motif_feature_importance(): 
    from meta_spliceai.splice_engine.splice_error_analyzer import ErrorAnalyzer
    import matplotlib.pyplot as plt

    pred_type = 'FN'
    experiment_name = 'hard_genes'
    model_type = 'xgboost'

    analyzer = ErrorAnalyzer(experiment='hard_genes')
    input_dir = analyzer.set_analysis_output_dir(
        pred_type=pred_type, experiment=experiment_name)

    # Load the motif importance data
    file_name = f"{pred_type.lower()}_vs_tp-{model_type.lower()}-motif-importance-shap-full.tsv"

    input_path = os.path.join(input_dir, file_name)
    print(f"[info] Loading motif importance data from: {input_path}")
    motif_importance_data = pd.read_csv(input_path, sep='\t')

    # Sort the data by importance score for better visualization
    sorted_motif_data = motif_importance_data.sort_values(by="importance_score", ascending=False)

    # Select top 10 motifs for clear storytelling
    top_10_motifs = sorted_motif_data.head(10)

    # Plot the top 10 motifs
    plt.figure(figsize=(10, 6))
    plt.barh(top_10_motifs["motif"], top_10_motifs["importance_score"], align="center")
    plt.xlabel("Importance Score (Mean Absolute SHAP Value)")
    plt.ylabel("Motif (k-mer)")
    plt.title(f"Top Motifs by Importance Score in {pred_type} vs. TP Classification")
    plt.gca().invert_yaxis()
    
    save_path = os.path.join(input_dir, f"{pred_type.lower()}_vs_tp-{model_type.lower()}-motif-scores.pdf")

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[info] Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def demo_analyze_training_data(**kargs): 
    from .splice_error_analyzer import ErrorAnalyzer

    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')

    pred_type = kargs.get('pred_type', 'FN')
    error_label = kargs.get('error_label', 'FP')
    correct_label = kargs.get('correct_label', 'TP')
    splice_type = kargs.get('splice_type', 'any')
    col_label = kargs.get('col_label', 'label')

    df_trainset = mefd.load_featurized_dataset(aggregated=True, pred_type=pred_type)
    analysis_result = \
        analyze_data_labels(df_trainset, label_col=col_label, verbose=2, handle_missing=None)

    print("[info] type(df_trainset): {}".format(type(df_trainset)))

    print_emphasized("Identifying null columns ...")
    identify_null_columns(df_trainset, verbose=2)

    print_emphasized("Imputing missing values ...")
    df_trainset = \
        impute_missing_values(
            df_trainset, 
            imputation_map={'num_overlaps': 0, 'has_consensus': False}, 
            verbose=1)

    print_emphasized("Checking feature types ...")
    feature_categories = classify_features(df_trainset, label_col='label')
    
    display_dictionary(feature_categories['unique_values'])

    print_emphasized("Diagnosing feature types ...")
    var_type, msg = diagnose_variable_type(df_trainset, column_name='has_consensus')
    print_with_indent(var_type, indent_level=1)
    print_with_indent(msg, indent_level=2)

    print_emphasized("Subsetting training data ...")
    df_subset = subset_training_data(df_trainset, group_cols=['gene_id', ], target_sample_size=200, verbose=1)

    subject="featurized_dataset_test"
    featurized_dataset_path = \
            mefd.save_featurized_dataset(
                df_subset, aggregated=True, pred_type=pred_type, subject=subject)


def demo_make_sampled_dataset(**kargs):
    from .splice_error_analyzer import ErrorAnalyzer
    from .utils_df import (
        estimate_dataframe_size, 
        estimate_memory_usage
    )

    subject = "analysis_sequences_test"
    mefd = ModelEvaluationFileHandler(ErrorAnalyzer.eval_dir, separator='\t')
    # splice_pos_df = mefd.load_splice_positions(aggregated=True)

    # Use run_spliceai_workflow.error_analysis_workflow() to generate the error analysis data
    analysis_sequence_df = mefd.load_analysis_sequences(aggregated=True)  # Output is a Polars DataFrame

    # Shape of the DataFrame
    print(f"[info] Shape of the analysis sequence DataFrame: {analysis_sequence_df.shape}")

    df_subset = subset_training_data(analysis_sequence_df, group_cols=['gene_id', ], target_sample_size=500, verbose=1)

    memory_usage_mb = estimate_memory_usage(df_subset)
    print_with_indent(f"Estimated memory usage of df_subset: {memory_usage_mb:.2f} MB", indent_level=1)

    estimated_size_mb = estimate_dataframe_size(df_subset, file_format='tsv')
    print_with_indent(f"Estimated disk size of df_subset: {estimated_size_mb:.2f} MB", indent_level=1)

    analysis_sequences_path = \
        mefd.save_analysis_sequences(df_subset, aggregated=True, subject=subject)

    print(f"[info] Saved sampled analysis sequences to: {analysis_sequences_path}")


def demo_impute_missing_values(**kargs):

    # Example usage
    df_trainset = pd.DataFrame({
        'gene_id': ['gene1', 'gene2', 'gene1', 'gene3'],
        'transcript_id': ['tx1', 'tx2', 'tx1', 'tx3'],
        'position': [1, 2, None, float('nan')],
        'score': [0.1, None, 0.3, 0.4],
        'splice_type': ['type1', 'type2', 'type1', 'type3'],
        'chrom': ['chr1', 'chr2', 'chr1', 'chr3'],
        'strand': ['+', '-', '+', '-']
    })

    imputation_map = {
        'position': 0,
        'score': 0.0
    }

    df_imputed = impute_missing_values(df_trainset, imputation_map, verbose=2)

    display_dataframe_in_chunks(df_imputed, title="Imputed DataFrame")



####################################################################################################
# Main

def demo(): 

    # demo_sort_performance_by_metric_and_error_type()
    # demo_count_unique_ids()

    # demo_check_duplicates()

    # demo_valid_ids(**kargs)

    print_emphasized("Analyze training data ...")
    # demo_analyze_training_data(pred_type='FN')    

    print_emphasized("Impute missing values ...")
    # demo_impute_missing_values()

    print_emphasized("Make sampled dataset ...")
    # demo_make_sampled_dataset()

    print_emphasized("Plot motif feature importance ...")
    demo_plot_motif_feature_importance()


if __name__ == "__main__":
    demo()