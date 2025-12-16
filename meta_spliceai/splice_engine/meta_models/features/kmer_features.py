"""
K-mer feature extraction for meta models.

This module handles the extraction of k-mer features from sequence data,
which are critical for training meta models that correct base model predictions.
"""

import os
import pandas as pd
import polars as pl
from typing import List, Dict, Any, Optional, Union, Tuple, Set, Literal

# Import original functionality without modifying it
from meta_spliceai.splice_engine.splice_error_analyzer import (
    featurize_gene_sequences,
    harmonize_features as original_harmonize_features
)

# Import meta-model specific utilities
from meta_spliceai.splice_engine.meta_models.features.gene_selection import (
    subset_analysis_sequences, 
    SubsetPolicy
)
# Import moved to function level to avoid circular imports
# Will import as needed in functions that require it
from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
from meta_spliceai.splice_engine.utils_doc import (
    print_emphasized, print_with_indent
)


def make_kmer_features(
    sequence_df: Union[pd.DataFrame, pl.DataFrame],
    kmer_sizes: List[int] = [6],
    return_feature_set: bool = True,
    verbose: int = 1
) -> Tuple[Union[pd.DataFrame, pl.DataFrame], Optional[List[str]]]:
    """
    Extract k-mer features from sequence data.
    
    This is a wrapper around the original featurize_gene_sequences function
    that maintains compatibility with the existing codebase.
    
    Parameters
    ----------
    sequence_df : Union[pd.DataFrame, pl.DataFrame]
        DataFrame containing sequence data with a 'sequence' column
    kmer_sizes : List[int], optional
        List of k-mer sizes to extract, by default [6]
    return_feature_set : bool, optional
        Whether to return the feature set in addition to the featurized DataFrame,
        by default True
    verbose : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    Tuple[Union[pd.DataFrame, pl.DataFrame], Optional[List[str]]]
        Featurized DataFrame and list of feature names if return_feature_set=True
    """
    return featurize_gene_sequences(
        sequence_df,
        kmer_sizes=kmer_sizes,
        return_feature_set=return_feature_set,
        verbose=verbose
    )


def harmonize_feature_sets(
    dataframes: List[Union[pd.DataFrame, pl.DataFrame]],
    feature_sets: List[List[str]]
) -> List[Union[pd.DataFrame, pl.DataFrame]]:
    """
    Ensure all DataFrames have the same feature columns.
    
    This is a wrapper around the original harmonize_features function
    that maintains compatibility with the existing codebase.
    
    Parameters
    ----------
    dataframes : List[Union[pd.DataFrame, pl.DataFrame]]
        List of DataFrames to harmonize
    feature_sets : List[List[str]]
        List of feature sets for each DataFrame
        
    Returns
    -------
    List[Union[pd.DataFrame, pl.DataFrame]]
        List of harmonized DataFrames
    """
    return original_harmonize_features(dataframes, feature_sets)


def make_kmer_featurized_dataset(
    data_handler: MetaModelDataHandler,
    *,
    n_genes: int = 1000,
    subset_policy: SubsetPolicy = "error_total",
    custom_genes: Optional[List[str]] = None,
    kmer_sizes: List[int] = [6],
    use_shared_dir: bool = False,
    output_subdir: Optional[str] = None,
    # This parameter is kept for compatibility but not used for filtering in meta-models
    splice_type: Optional[Literal["donor", "acceptor"]] = None,
    add_gene_features: bool = True,
    use_effective_counts: bool = True,
    filter_by_pred_types: Optional[List[str]] = None,
    fill_missing_value: float = 0.0,
    downsample_options: Optional[Dict[str, Dict[str, float]]] = None,
    verbose: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate k-mer features from gene sequences for meta-model training.
    
    This function implements a memory-efficient workflow for generating k-mer features:
    1. Gene selection based on specified policy
    2. Sequence featurization (k-mer extraction)
    3. Feature harmonization across prediction types
    
    The function returns intermediate results that can be further processed
    by other components in the training pipeline.
    
    Parameters
    ----------
    data_handler : MetaModelDataHandler
        Handler for accessing meta-model data
    n_genes : int, optional
        Number of genes to include, by default 1000
    subset_policy : SubsetPolicy, optional
        Strategy for gene selection, by default "error_total"
        Supported values:
            'random'       - Random sample of genes
            'hard'         - Use PerformanceAnalyzer logic (delegated)
            'error_fp'     - Genes with the most false positives
            'error_fn'     - Genes with the most false negatives
            'error_total'  - Genes with the highest FP+FN count
    custom_genes : Optional[List[str]], optional
        List of specific gene IDs to include, by default None

    kmer_sizes : List[int], optional
        Sizes of k-mers to extract, by default [6]
    use_shared_dir : bool, optional
        Whether to use shared evaluation directory, by default False
    output_subdir : Optional[str], optional
        Additional subdirectory for output, by default None
    splice_type : Literal["donor", "acceptor", "any"], optional
        Filter by splice type, by default "any"
    add_gene_features : bool, optional
        Whether to add gene-level features, by default True
    use_effective_counts : bool, optional
        Whether to use effective error counts (deduplicated), by default True
    filter_by_pred_types : Optional[List[str]], optional
        Only include specific prediction types (TP, TN, FP, FN), by default None
    fill_missing_value : float, optional
        Value to use for missing features, by default 0.0
    downsample_options : Optional[Dict[str, Dict[str, float]]], optional
        Dictionary of downsampling options for each prediction type, by default None
    verbose : int, optional
        Verbosity level, by default 1
    **kwargs
        Additional parameters
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing the following keys:
        - 'harmonized_dfs': Dictionary of harmonized DataFrames with k-mer features by prediction type
        - 'feature_sets': Dictionary of feature sets by prediction type
        - 'analysis_sequence_df': Original analysis sequence DataFrame
        - 'selected_gene_ids': Set of gene IDs selected for analysis
    """
    # Import here to avoid circular imports
    from meta_spliceai.splice_engine.meta_models.features.sequence_featurization import (
        featurize_analysis_sequences,
        harmonize_all_feature_sets
    )

    # Step 1: Gene Selection
    if verbose >= 1:
        print_emphasized(f"[workflow] Selecting genes for featurization...")
        print_with_indent(f"Policy: {subset_policy}, Target count: {n_genes}")
        
    # Process custom genes if provided
    additional_gene_ids = []
    if custom_genes:
        if verbose >= 1:
            print_with_indent(f"Including {len(custom_genes)} custom genes")
        additional_gene_ids = custom_genes
    
    # Use subset_analysis_sequences to select genes and filter the analysis sequence data
    analysis_sequence_df, selected_gene_ids = subset_analysis_sequences(
        data_handler=data_handler,
        n_genes=n_genes,
        subset_policy=subset_policy,
        aggregated=True,
        additional_gene_ids=additional_gene_ids,
        use_effective_counts=use_effective_counts,
        verbose=verbose
    )
    
    # Add gene features if requested
    if add_gene_features and "strand" not in analysis_sequence_df.columns:
        if verbose >= 1:
            print_with_indent("Adding gene features (strand)...")
        
        try:
            # Load gene features
            gene_feature_df = data_handler.load_gene_features()
            
            # Join with analysis sequences
            analysis_sequence_df = analysis_sequence_df.join(
                gene_feature_df.select(['gene_id', 'strand']),
                on='gene_id',
                how='left'
            )
        except Exception as e:
            if verbose >= 1:
                print(f"[warning] Failed to add gene features: {str(e)}")
    
    # Log summary of selected genes
    if verbose >= 1:
        final_unique_gene_ids = analysis_sequence_df.select(pl.col('gene_id')).unique()
        final_num_unique_genes = final_unique_gene_ids.height
        print_emphasized(f"[info] Final number of unique gene IDs: {final_num_unique_genes}")
        
        # Check for missing custom genes
        if custom_genes:
            final_gene_ids_set = set(final_unique_gene_ids.to_series().to_list())
            missing_genes = [gene for gene in custom_genes if gene not in final_gene_ids_set]
            if missing_genes:
                print(f"[warning] The following custom genes were not found: {missing_genes}")
    
    # Step 2: Split sequences by prediction type
    if verbose >= 1:
        print_emphasized(f"[workflow] Splitting sequences by prediction type...")
    
    pred_type_dfs = {}
    
    # Get unique prediction types
    if filter_by_pred_types:
        pred_types = filter_by_pred_types
    else:
        pred_types = analysis_sequence_df.select(pl.col('pred_type')).unique().to_series().to_list()
    
    if verbose >= 1:
        print_with_indent(f"Found prediction types: {pred_types}")
    
    # Split by prediction type
    for pred_type in pred_types:
        type_df = analysis_sequence_df.filter(pl.col('pred_type') == pred_type)
        if type_df.shape[0] > 0:
            pred_type_dfs[pred_type] = type_df
            if verbose >= 1:
                print_with_indent(f"Type {pred_type}: {type_df.shape[0]} sequences")
    
    # Step 3: Featurize sequences
    if verbose >= 1:
        print_emphasized(f"[workflow] Featurizing sequences...")
    
    # Set default downsampling options if not provided
    if downsample_options is None:
        downsample_options = {
            'TN': {'fraction': 0.5, 'max_size': 50000},
            'FN': {'fraction': 1.0, 'max_size': 50000}
        }
    
    # Create a list of columns we may need to reference for downstream processing
    # These will be useful for feature generation and metadata tracking
    metadata_columns = ['gene_id', 'position', 'sequence', 'pred_type', 'score', 'splice_type']
    
    # Featurize sequences for each prediction type
    featurized_dfs, feature_sets = featurize_analysis_sequences(
        sequence_dfs=pred_type_dfs,
        kmer_sizes=kmer_sizes,
        downsample_options=downsample_options,
        verbose=verbose
    )
    
    # Step 4: Harmonize feature sets
    if verbose >= 1:
        print_emphasized(f"[workflow] Harmonizing feature sets...")
    
    harmonized_dfs = harmonize_all_feature_sets(
        featurized_dfs=featurized_dfs,
        feature_sets=feature_sets,
        default_value=fill_missing_value,
        verbose=verbose
    )
    
    # Return intermediate results for further processing
    if verbose >= 1:
        print_emphasized(f"[workflow] K-mer feature generation complete")
        print_with_indent(f"Generated features for {len(harmonized_dfs)} prediction types")
        
        # Print statistics about the feature sets
        for pred_type, df in harmonized_dfs.items():
            if df is not None and df.shape[0] > 0:
                if isinstance(df, pl.DataFrame):
                    shape = df.shape
                else:  # pandas DataFrame
                    shape = df.shape
                print_with_indent(f"  {pred_type}: {shape[0]} sequences, {shape[1]} features")
    
    # Return intermediate results for further processing
    return {
        'featurized_dfs': featurized_dfs,
        'harmonized_dfs': harmonized_dfs,
        'feature_sets': feature_sets,
        'analysis_sequence_df': analysis_sequence_df,
        'selected_gene_ids': selected_gene_ids
    }
