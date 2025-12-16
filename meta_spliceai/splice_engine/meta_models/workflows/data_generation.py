"""
Data generation workflows for meta models.

This module provides the core functionality for generating training data for meta models,
refactored from the original splice_error_analyzer implementation with improved structure.
"""

import os
import pandas as pd
import polars as pl
from typing import Union, Optional, Dict, List, Any, Tuple

# Import original functionality without modifying it
from meta_spliceai.splice_engine.splice_error_analyzer import SpliceAnalyzer
from meta_spliceai.splice_engine.extract_genomic_features import FeatureAnalyzer
from meta_spliceai.splice_engine.model_evaluator import ModelEvaluationFileHandler
from meta_spliceai.splice_engine.splice_error_analyzer import ErrorAnalyzer

# Import refactored components
from meta_spliceai.splice_engine.meta_models.core.data_types import MetaModelConfig
from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
from meta_spliceai.splice_engine.meta_models.features.genomic_features import (
    incorporate_gene_level_features,
    incorporate_length_features,
    incorporate_performance_features,
    incorporate_overlapping_gene_features,
    incorporate_distance_features
)
from meta_spliceai.splice_engine.meta_models.utils.data_processing import (
    check_and_subset_invalid_transcript_ids,
    filter_and_validate_ids,
    count_unique_ids,
    is_dataframe_empty,
    display_feature_set,
    subsample_dataframe,
    display_dataframe_in_chunks,
    print_emphasized,
    print_with_indent,
    print_section_separator
)

# Import original make_training_data_with_analysis_sequences to use directly
from meta_spliceai.splice_engine.splice_error_analyzer import (
    make_training_data_with_analysis_sequences as original_make_training_data_with_analysis_sequences
)


def make_kmer_featurized_dataset(
    gtf_file: Optional[str] = None,
    pred_type: str = 'FP',
    error_label: Optional[str] = None,
    correct_label: str = 'TP',
    kmer_sizes: List[int] = [6],
    subset_genes: bool = True,
    subset_policy: str = 'hard',
    custom_genes: List[str] = [],
    n_genes: int = 1000,
    fn_sample_fraction: float = 1.0,
    fn_max_sample_size: int = 50000,
    tn_sample_fraction: float = 0.5,
    tn_max_sample_size: int = 50000,
    overwrite: bool = True,
    col_tid: str = 'transcript_id',
    fa: Optional[FeatureAnalyzer] = None,
    mefd: Optional[ModelEvaluationFileHandler] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Generate a featurized dataset with k-mer features for meta model training.
    
    This function is a refactored version of the original make_kmer_featurized_dataset
    with improved parameter handling and documentation.
    
    Parameters
    ----------
    gtf_file : Optional[str], optional
        Path to GTF file, by default None
    pred_type : str, optional
        Prediction type (FP, FN), by default 'FP'
    error_label : Optional[str], optional
        Label for error class, by default None (uses pred_type)
    correct_label : str, optional
        Label for correct class, by default 'TP'
    kmer_sizes : List[int], optional
        List of k-mer sizes to extract, by default [6]
    subset_genes : bool, optional
        Whether to subset genes, by default True
    subset_policy : str, optional
        Policy for subsetting genes (random, hard, top), by default 'hard'
    custom_genes : List[str], optional
        List of custom genes to include, by default []
    n_genes : int, optional
        Number of genes to include, by default 1000
    fn_sample_fraction : float, optional
        Fraction of FN data to sample, by default 1.0
    fn_max_sample_size : int, optional
        Maximum FN sample size, by default 50000
    tn_sample_fraction : float, optional
        Fraction of TN data to sample, by default 0.5
    tn_max_sample_size : int, optional
        Maximum TN sample size, by default 50000
    overwrite : bool, optional
        Whether to overwrite existing files, by default True
    col_tid : str, optional
        Column name for transcript ID, by default 'transcript_id'
    fa : Optional[FeatureAnalyzer], optional
        FeatureAnalyzer instance, by default None
    mefd : Optional[ModelEvaluationFileHandler], optional
        ModelEvaluationFileHandler instance, by default None
    
    Returns
    -------
    pd.DataFrame
        Featurized dataset for meta model training
    """
    # Import here to avoid circular imports
    from meta_spliceai.splice_engine.splice_error_analyzer import make_kmer_featurized_dataset as original_function
    
    # Set the error_label to pred_type if None
    if error_label is None:
        error_label = pred_type
    
    # Call the original function with all parameters
    return original_function(
        gtf_file=gtf_file,
        pred_type=pred_type,
        error_label=error_label,
        correct_label=correct_label,
        kmer_sizes=kmer_sizes,
        subset_genes=subset_genes,
        subset_policy=subset_policy,
        custom_genes=custom_genes,
        n_genes=n_genes,
        fn_sample_fraction=fn_sample_fraction,
        fn_max_sample_size=fn_max_sample_size,
        tn_sample_fraction=tn_sample_fraction,
        tn_max_sample_size=tn_max_sample_size,
        overwrite=overwrite,
        col_tid=col_tid,
        fa=fa,
        mefd=mefd,
        **kwargs
    )


def run_training_data_generation(
    config: Optional[MetaModelConfig] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Run the full training data generation workflow for meta models.
    
    This function is a refactored version of run_training_data_generation_workflow
    with improved parameter handling via a configuration object.
    
    Parameters
    ----------
    config : Optional[MetaModelConfig], optional
        Configuration object for meta model training, by default None
    **kwargs : dict
        Additional keyword arguments to override config settings
        
    Returns
    -------
    pd.DataFrame
        Final featurized dataset for meta model training
    
    Notes
    -----
    This function performs the following steps:
    1. Extract k-mer features from analysis sequence data
    2. Check data integrity and validate transcript IDs
    3. Incorporate gene-level features from GTF data
    4. Add exon-intron length features
    5. Add performance profile features
    6. Add overlapping gene features
    7. Add splice site distance features
    8. Save the featurized dataset
    9. Process analysis sequences
    """
    # Create a configuration object if not provided
    if config is None:
        config = MetaModelConfig(**kwargs)
    else:
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Extract configuration parameters for readability
    gtf_file = config.gtf_file
    pred_type = config.pred_type
    error_label = config.error_label
    correct_label = config.correct_label
    kmer_sizes = config.kmer_sizes
    subset_genes = config.subset_genes
    subset_policy = config.subset_policy
    n_genes = config.n_genes
    custom_genes = config.custom_genes
    col_tid = config.col_tid
    overwrite = config.overwrite
    
    # Print configuration parameters
    print_emphasized("[workflow] Starting meta model training data generation workflow...")
    print_with_indent(f"[params] pred_type: {pred_type}, kmer_sizes: {kmer_sizes}, subset_genes: {subset_genes}, subset_policy: {subset_policy}, n_genes: {n_genes}", indent_level=1)
    print_with_indent(f"[params] error_label: {error_label}, correct_label: {correct_label}", indent_level=1)
    print_with_indent(f"[params] custom_genes: {custom_genes}", indent_level=1)
    
    # Initialize handlers
    fa = FeatureAnalyzer(gtf_file=gtf_file, overwrite=overwrite, col_tid=col_tid)
    format_type = fa.format
    separator = '\t' if format_type == 'tsv' else ','
    
    # Initialize the data handler
    data_handler = MetaModelDataHandler(ErrorAnalyzer.eval_dir, separator=separator)
    
    # Stage 1: Generate the main featurized dataset with k-mers
    print_emphasized("[workflow] Extract kmer features from analysis sequence data...")
    
    # Create featurized dataset with k-mer features
    make_kmer_featurized_dataset(
        gtf_file=gtf_file,
        pred_type=pred_type,
        error_label=error_label,
        correct_label=correct_label,
        kmer_sizes=kmer_sizes,
        subset_genes=subset_genes,
        subset_policy=subset_policy,
        custom_genes=custom_genes,
        n_genes=n_genes,
        fn_sample_fraction=config.fn_sample_fraction,
        fn_max_sample_size=config.fn_max_sample_size,
        tn_sample_fraction=config.tn_sample_fraction,
        tn_max_sample_size=config.tn_max_sample_size,
        overwrite=overwrite,
        col_tid=col_tid,
        fa=fa,
        mefd=data_handler.file_handler,
    )
    
    # Subject identifier for featurized dataset
    subject = "seq_featurized"
    
    # Load the featurized dataset
    df_trainset = data_handler.load_featurized_artifact(
        aggregated=True,
        subject=subject,
        error_label=error_label,
        correct_label=correct_label
    )
    
    # Display feature columns
    columns0 = display_feature_set(df_trainset, max_kmers=100)
    print_emphasized(f"[info] Columns in df_trainset prior to incorporating features from various data sources:\n{columns0}\n")
    
    # Print dataset shape
    shape0 = df_trainset.shape
    print_with_indent(f"shape(df_trainset): {shape0}", indent_level=1)
    
    # Count unique IDs
    count_unique_ids(df_trainset, col_tid=col_tid)
    
    # Check data integrity
    df_abnormal = check_and_subset_invalid_transcript_ids(df_trainset, col_tid=col_tid, verbose=1)
    if not is_dataframe_empty(df_abnormal):
        print("[test] Abnormal data in training data (df_trainset):")
        cols = ['gene_id', col_tid, 'position', 'score', 'splice_type', 'chrom', 'strand']
        display_dataframe_in_chunks(subsample_dataframe(df_abnormal, columns=cols, num_rows=10, random=True), title="Abnormal data")
        df_trainset = filter_and_validate_ids(df_trainset, col_tid=col_tid, verbose=1)
    
    # Incorporate gene-level features
    print_emphasized("[workflow] Extracting gene-level features from GTF file...")
    df_trainset = incorporate_gene_level_features(df_trainset, fa=fa)
    
    # Incorporate length features
    print_emphasized("[info] Incorporating exon-intron length features...")
    df_trainset = incorporate_length_features(df_trainset, fa=fa)
    
    # Incorporate performance features
    print_emphasized("[i/o] Extracting performance-profile features...")
    df_trainset = incorporate_performance_features(df_trainset, fa=fa)
    
    # Incorporate overlapping gene features
    sa = SpliceAnalyzer()
    print_emphasized("[info] Incorporating overlapping gene-specific features...")
    df_trainset = incorporate_overlapping_gene_features(df_trainset, sa=sa)
    
    # Incorporate splice site distance features
    print_emphasized("[info] Incorporating splice site features...")
    df_trainset = incorporate_distance_features(df_trainset, fa)
    
    # Save the featurized dataset
    featurized_dataset_path = data_handler.save_featurized_artifact(
        df_trainset, 
        aggregated=True,
        subject=subject,
        error_label=error_label, 
        correct_label=correct_label
    )
    print_emphasized(f"[info] Saved featurized dataset ({error_label}_vs_{correct_label}) to {featurized_dataset_path}")
    print_with_indent(f"Columns: {display_feature_set(df_trainset, max_kmers=100)}", indent_level=1)
    
    # Process analysis sequences
    df_trainset = original_make_training_data_with_analysis_sequences(
        fa=fa,
        pred_type=pred_type,
        error_label=error_label,
        correct_label=correct_label
    )
    
    return df_trainset
