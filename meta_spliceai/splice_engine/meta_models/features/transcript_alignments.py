"""Transcript alignment utilities for meta_models package.

This module provides functions for aligning transcript IDs to positions in the
genomic sequence, which is critical for proper splice site analysis.
"""

import os
import pandas as pd
import polars as pl
from typing import Dict, List, Union, Optional, Tuple, Set, Any

# Import Analyzer from core for path standardization
from meta_spliceai.splice_engine.meta_models.core.analyzer import Analyzer
from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
from meta_spliceai.splice_engine.utils_doc import (
    display_dataframe_in_chunks, 
    print_emphasized
)


def retrieve_splice_site_analysis_data(
    gtf_file=None, 
    load_datasets=None, 
    data_handler=None,
    use_shared_dir=False,
    output_subdir=None,
    data_dir=None, 
    verbose=1,
    **kargs
):
    """
    Retrieve a comprehensive set of data sources needed for splice site analysis.
    
    This function loads various datasets required for transcript alignment and splice
    site analysis using the meta_models package's refactored analyzers.
    
    Parameters
    ----------
    gtf_file : str, optional
        Path to the GTF file, if None uses the default
    load_datasets : list, optional
        List of dataset names to load, if None loads all available datasets
    data_handler : MetaModelDataHandler, optional
        Existing data handler to use, if None creates a new one
    use_shared_dir : bool, default=False
        Whether to use shared evaluation directory
    output_subdir : str, optional
        Additional subdirectory for output
    verbose : int, default=1
        Verbosity level
    **kargs : dict
        Additional keyword arguments
    
    Returns
    -------
    dict
        Dictionary containing the requested datasets
    """
    from meta_spliceai.splice_engine.meta_models.core.analyzers.feature import FeatureAnalyzer
    from meta_spliceai.splice_engine.meta_models.core.analyzers.splice import SpliceAnalyzer
    from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
    
    if verbose >= 1:
        print_emphasized(f"[i/o] Loading error analysis data ...")

    # Initialize data handler if not provided
    if data_handler is None:
        # Get separator from kwargs with fallback
        separator = kargs.get('separator', kargs.get('sep', '\t'))
        
        # Create a new data handler
        data_handler = MetaModelDataHandler(Analyzer.eval_dir, separator=separator)

    # Create result container
    result_set = {}

    # Initialize analyzers
    feature_analyzer = FeatureAnalyzer(data_dir=data_dir,gtf_file=gtf_file)
    splice_analyzer = SpliceAnalyzer(data_dir=data_dir)
    
    # Define datasets to load with refactored functions
    available_datasets = {
        'error_df': lambda: data_handler.load_error_analysis(aggregated=True, verbose=verbose),
        'gene_features': lambda: feature_analyzer.retrieve_gene_features(verbose=verbose),
        'transcript_features': lambda: feature_analyzer.retrieve_transcript_features(verbose=verbose),
        'splice_sites_df': lambda: splice_analyzer.retrieve_splice_sites(
            verbose=verbose, 
            column_names={'site_type': 'splice_type'}
        ),
        'position_df': lambda: data_handler.load_splice_positions(aggregated=True, enhanced=True)
    }

    # Add aliases for backward compatibility
    aliases = {
        'gene_feature_df': 'gene_features',
        'transcript_feature_df': 'transcript_features'
    }

    # Load only the specified datasets
    if load_datasets is None:
        load_datasets = list(available_datasets.keys())

    # Resolve aliases
    load_datasets = [aliases.get(dataset, dataset) for dataset in load_datasets]

    # Load each dataset
    for dataset in load_datasets:
        if dataset in available_datasets:
            if verbose >= 1:
                print(f"[info] Loading {dataset}...")
                
            result_set[dataset] = available_datasets[dataset]()
            
            if verbose >= 2:
                print(f"[info] Loaded {dataset} with columns: {result_set[dataset].columns}")
                display_dataframe_in_chunks(result_set[dataset].head(5), title=f"{dataset} Sample")
        else:
            if verbose >= 1:
                print(f"[warning] Dataset {dataset} is not available for loading.")

    return result_set



def align_transcript_ids(
        df, splice_sites_df, gene_feature_df, 
        position_col='position', tolerance=2, throw_exception=True, return_unmatched=False):
    """
    Align transcript IDs to a general dataframe using splice site annotations, accounting
    for absolute vs relative positions and tolerance for minor discrepancies. Includes a
    validation check to ensure every gene in the input dataframe has a match in the splice
    site annotations.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with a position-like column for alignment.
      Required columns: ['gene_id', position_col, 'strand'], where `position_col` represents
        a relative position within the gene, and `strand` specifies the gene's strand.
    - splice_sites_df (pd.DataFrame): DataFrame containing splice site annotations.
      Required columns: ['gene_id', 'position', 'transcript_id'], where `position`
        is the absolute genomic coordinate of the splice site.
    - position_col (str): Name of the column in `df` representing the relative position.
    - tolerance (int): Number of nucleotides to allow for matching splice site positions (default: 0).
    - gene_feature_df (pd.DataFrame): DataFrame containing gene-level features, including
      'gene_id', 'start', and 'end' for determining the absolute positions.

    Returns:
    - pd.DataFrame: A new DataFrame with an added `transcript_id` column, representing the transcript
      associated with each splice site position.
    """
    # Base case: Check if transcript_id column exists and has no null values
    if 'transcript_id' in df.columns and df['transcript_id'].notna().all():
        print("[info] 'transcript_id' already exists and is non-null. Returning the original DataFrame.")
        if return_unmatched: 
            return df, set()

        return df

    # Ensure required columns are present
    required_gene_cols = ['gene_id', 'start', 'end', 'strand']
    for col in required_gene_cols:
        if col not in gene_feature_df.columns:
            raise ValueError(f"Column '{col}' is missing from gene_feature_df.")
    required_splice_cols = ['gene_id', 'position', 'transcript_id']
    for col in required_splice_cols:
        if col not in splice_sites_df.columns:
            raise ValueError(f"Column '{col}' is missing from splice_sites_df.")
    if position_col not in df.columns:
        raise ValueError(f"Column '{position_col}' is missing from df.")

    is_polars = False
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
        is_polars = True
    if isinstance(splice_sites_df, pl.DataFrame):
        splice_sites_df = splice_sites_df.to_pandas()
    if isinstance(gene_feature_df, pl.DataFrame):
        gene_feature_df = gene_feature_df.to_pandas()

    # Merge df with gene_feature_df to get gene start, end, and strand
    df = df.merge(
        gene_feature_df[['gene_id', 'start', 'end', ]],   # 'strand'
        on='gene_id',
        how='left'
    )

    # Compute absolute positions based on strand
    def compute_absolute_position(row):
        relative_position = row[position_col]
        strand = row['strand']
        if strand == '+':
            return row['start'] + relative_position
        elif strand == '-':
            return row['end'] - relative_position
        else:
            raise ValueError(f"Invalid strand value: {strand}")

    # Add an absolute position column to df
    df['absolute_position'] = df.apply(compute_absolute_position, axis=1)

    # Initialize the transcript_id column
    df['transcript_id'] = None

    # Perform alignment with tolerance
    unmatched_genes = set()  # Track genes that fail the validation check
    aligned_rows = []  # To store the expanded rows

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        gene_id = row['gene_id']
        abs_pos = row['absolute_position']
        splice_type = row['splice_type'] if 'splice_type' in row else None

        # Find matching splice sites within the tolerance range
        matching_sites = splice_sites_df[
            (splice_sites_df['gene_id'] == gene_id) &
            (abs(splice_sites_df['position'] - abs_pos) <= tolerance)
        ]

        # Add optional condition for splice_type if it is given
        if splice_type is not None:
            # Todo: Unify the column name
            # matching_sites = matching_sites[matching_sites['site_type'] == splice_type]
            matching_sites = matching_sites[matching_sites['splice_type'] == splice_type]
            
        if not matching_sites.empty:
            # Assign the first matching transcript_id
            # df.at[idx, 'transcript_id'] = matching_sites.iloc[0]['transcript_id']

            # Keep track of all matching transcripts
            # => Create a row for each matched transcript
            for transcript_id in matching_sites['transcript_id']:
                new_row = row.copy()
                new_row['transcript_id'] = transcript_id
                aligned_rows.append(new_row)

        else:
            unmatched_genes.add(gene_id)

            print(f"Could not find a matching splice site for gene {gene_id} at position {abs_pos}, where site_type={splice_type}")
            
            # Test
            if splice_type is None:
                display(splice_sites_df[splice_sites_df['gene_id'] == gene_id].sort_values('position').head(20))
            else: 
                display(splice_sites_df[(splice_sites_df['gene_id'] == gene_id) & (splice_sites_df['splice_type'] == splice_type)].sort_values('position').head(20))

    # Validation check: Ensure all genes in df have matches in splice_sites_df
    df_gene_ids = set(df['gene_id'])
    splice_site_gene_ids = set(splice_sites_df['gene_id'])
    missing_genes = df_gene_ids - splice_site_gene_ids

    if missing_genes:
        raise ValueError(
            f"Some genes in the input dataframe are missing from splice_sites_df: {missing_genes}"
        )

    msg = (f"[warning] Some splice site positions for genes in the input dataframe "
              f"could not be matched (even with tolerance={tolerance}): {unmatched_genes}"
           )

    if unmatched_genes:
        if throw_exception: 
            raise ValueError(msg)
        else: 
            print(msg)

            # remove unmatched genes
            print_emphasized(f"Removing unmatched genes from the input dataframe (n={len(unmatched_genes)})...")
            df = df[~df['gene_id'].isin(unmatched_genes)]

    # Create a DataFrame from aligned rows
    aligned_df = pd.DataFrame(aligned_rows)

    if is_polars:
        aligned_df = pl.DataFrame(aligned_df)

    if return_unmatched: 
        return aligned_df, unmatched_genes

    return aligned_df

