"""
Dataset loading utilities for meta models.

This module provides individual functions for loading different datasets required
for meta model training and analysis. It centralizes dataset access while allowing
selective loading of only the required datasets.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union, Any, List, Callable

import pandas as pd
import polars as pl

from meta_spliceai.splice_engine.meta_models.core.analyzer import Analyzer
from meta_spliceai.splice_engine.meta_models.core.analyzers.feature import FeatureAnalyzer
from meta_spliceai.splice_engine.meta_models.core.analyzers.splice import SpliceAnalyzer
from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
from meta_spliceai.splice_engine.utils_doc import print_emphasized
from meta_spliceai.system.sys_config import get_proj_dir


def get_data_handler(
    data_dir: Optional[str] = None,
    use_shared_dir: bool = False,
    output_subdir: Optional[str] = None,
    meta_subdir: str = 'meta_models',
    separator: str = '\t',
    **kwargs
) -> MetaModelDataHandler:
    """
    Create and configure a MetaModelDataHandler with the appropriate directories.
    
    Args:
        data_dir: Base data directory. If None, uses Analyzer.data_dir
        use_shared_dir: Whether to use shared directory for evaluation data
        output_subdir: Optional subdirectory for evaluation output
        meta_subdir: Subdirectory for meta model data (default: 'meta_models')
        separator: Separator used in data files
        **kwargs: Additional keyword arguments
    
    Returns:
        Configured MetaModelDataHandler instance
    """
    # Default base directories from Analyzer
    data_dir = data_dir or Analyzer.data_dir  # Shared genomic data
    eval_dir = Analyzer.eval_dir              # Default evaluation directory
    
    # Handle directory configuration based on use_shared_dir parameter
    if not use_shared_dir:
        # If not using shared dir and output_subdir is specified,
        # create a user-specific evaluation directory
        if output_subdir is not None:
            # This creates a user-specific evaluation directory
            eval_dir = os.path.join(eval_dir, output_subdir)
    
    # Get separator from kwargs with fallback
    separator = kwargs.get('separator', kwargs.get('sep', separator))
    
    # Create and return a new data handler with the appropriate directories
    # MetaModelDataHandler will automatically create meta_dir as eval_dir/meta_subdir
    return MetaModelDataHandler(eval_dir, meta_subdir=meta_subdir, separator=separator)


def load_error_analysis(
    data_handler: Optional[MetaModelDataHandler] = None,
    data_dir: Optional[str] = None,
    use_shared_dir: bool = False,
    output_subdir: Optional[str] = None,
    aggregated: bool = True,
    verbose: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    Load error analysis data.
    
    Args:
        data_handler: Optional pre-configured data handler
        data_dir: Base data directory. If None, uses Analyzer.data_dir
        use_shared_dir: Whether to use shared directory for evaluation data
        output_subdir: Optional subdirectory for output
        aggregated: Whether to load aggregated data
        verbose: Verbosity level
        **kwargs: Additional keyword arguments passed to load_error_analysis
    
    Returns:
        DataFrame containing error analysis data
    """
    if verbose >= 1:
        print_emphasized(f"[i/o] Loading error analysis data ...")
    
    # Initialize data handler if not provided
    if data_handler is None:
        data_handler = get_data_handler(
            data_dir=data_dir,
            use_shared_dir=use_shared_dir,
            output_subdir=output_subdir,
            **kwargs
        )
    
    # Load and return error analysis data
    return data_handler.load_error_analysis(aggregated=aggregated, verbose=verbose, **kwargs)


def load_splice_positions(
    data_handler: Optional[MetaModelDataHandler] = None,
    data_dir: Optional[str] = None,
    use_shared_dir: bool = False,
    output_subdir: Optional[str] = None,
    aggregated: bool = True,
    enhanced: bool = True,
    verbose: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    Load splice position data.
    
    Args:
        data_handler: Optional pre-configured data handler
        data_dir: Base data directory. If None, uses Analyzer.data_dir
        use_shared_dir: Whether to use shared directory for evaluation data
        output_subdir: Optional subdirectory for output
        aggregated: Whether to load aggregated data
        enhanced: Whether to load enhanced position data
        verbose: Verbosity level
        **kwargs: Additional keyword arguments passed to load_splice_positions
    
    Returns:
        DataFrame containing splice position data
    """
    if verbose >= 1:
        print_emphasized(f"[i/o] Loading splice position data ...")
    
    # Initialize data handler if not provided
    if data_handler is None:
        data_handler = get_data_handler(
            data_dir=data_dir,
            use_shared_dir=use_shared_dir,
            output_subdir=output_subdir,
            **kwargs
        )
    
    # Load and return splice position data
    return data_handler.load_splice_positions(
        aggregated=aggregated, 
        enhanced=enhanced,
        verbose=verbose,
        **kwargs
    )


def load_gene_features(
    data_dir: Optional[str] = None,
    gtf_file: Optional[str] = None,
    verbose: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    Load gene feature data.
    
    Args:
        data_dir: Base data directory. If None, uses Analyzer.data_dir
        gtf_file: Path to GTF file with gene annotations
        verbose: Verbosity level
        **kwargs: Additional keyword arguments passed to retrieve_gene_features
    
    Returns:
        DataFrame containing gene feature data
    """
    if verbose >= 1:
        print_emphasized(f"[i/o] Loading gene feature data ...")
    
    # Initialize feature analyzer
    feature_analyzer = FeatureAnalyzer(data_dir=data_dir, gtf_file=gtf_file)
    
    # Load and return gene features
    return feature_analyzer.retrieve_gene_features(verbose=verbose, **kwargs)


def load_transcript_features(
    data_dir: Optional[str] = None,
    gtf_file: Optional[str] = None,
    verbose: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    Load transcript feature data.
    
    Args:
        data_dir: Base data directory. If None, uses Analyzer.data_dir
        gtf_file: Path to GTF file with transcript annotations
        verbose: Verbosity level
        **kwargs: Additional keyword arguments passed to retrieve_transcript_features
    
    Returns:
        DataFrame containing transcript feature data
    """
    if verbose >= 1:
        print_emphasized(f"[i/o] Loading transcript feature data ...")
    
    # Initialize feature analyzer
    feature_analyzer = FeatureAnalyzer(data_dir=data_dir, gtf_file=gtf_file)
    
    # Load and return transcript features
    return feature_analyzer.retrieve_transcript_features(verbose=verbose, **kwargs)


def load_splice_sites(
    data_dir: Optional[str] = None,
    verbose: int = 1,
    column_rename: Optional[Dict[str, str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load splice site data.
    
    Args:
        data_dir: Base data directory. If None, uses Analyzer.data_dir
        verbose: Verbosity level
        column_rename: Optional dictionary mapping original column names to new ones
        **kwargs: Additional keyword arguments passed to retrieve_splice_sites
    
    Returns:
        DataFrame containing splice site data
    """
    if verbose >= 1:
        print_emphasized(f"[i/o] Loading splice site data ...")
    
    # Default column rename if not provided
    if column_rename is None:
        column_rename = {'site_type': 'splice_type'}
    
    # Initialize splice analyzer
    splice_analyzer = SpliceAnalyzer(data_dir=data_dir)
    
    # Load and return splice sites
    return splice_analyzer.retrieve_splice_sites(
        verbose=verbose, 
        column_names=column_rename,
        **kwargs
    )


def retrieve_splice_site_analysis_data(
    data_dir: Optional[str] = None,
    gtf_file: Optional[str] = None,
    data_handler: Optional[MetaModelDataHandler] = None,
    use_shared_dir: bool = False,
    output_subdir: Optional[str] = None,
    verbose: int = 1,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Load all datasets needed for splice site analysis.
    
    This is a convenience function that loads all five datasets used in 
    splice site analysis. For more selective loading, use the individual
    loading functions.
    
    Args:
        data_dir: Base data directory. If None, uses Analyzer.data_dir
        gtf_file: Path to GTF file with gene/transcript annotations
        data_handler: Optional pre-configured data handler
        use_shared_dir: Whether to use shared directory for evaluation data
        output_subdir: Optional subdirectory for output
        verbose: Verbosity level
        **kwargs: Additional keyword arguments passed to loading functions
    
    Returns:
        Dictionary containing all loaded datasets
    """
    if verbose >= 1:
        print_emphasized(f"[i/o] Loading all splice site analysis datasets ...")
    
    # Initialize data handler if not provided
    if data_handler is None:
        data_handler = get_data_handler(
            data_dir=data_dir,
            use_shared_dir=use_shared_dir,
            output_subdir=output_subdir,
            **kwargs
        )
    
    # Create result container
    result_set = {}
    
    # Load all datasets
    result_set['error_df'] = load_error_analysis(
        data_handler=data_handler, 
        verbose=verbose, 
        **kwargs
    )
    
    result_set['gene_features'] = load_gene_features(
        data_dir=data_dir, 
        gtf_file=gtf_file, 
        verbose=verbose, 
        **kwargs
    )
    
    result_set['transcript_features'] = load_transcript_features(
        data_dir=data_dir, 
        gtf_file=gtf_file, 
        verbose=verbose, 
        **kwargs
    )
    
    result_set['splice_sites_df'] = load_splice_sites(
        data_dir=data_dir, 
        verbose=verbose, 
        **kwargs
    )
    
    result_set['position_df'] = load_splice_positions(
        data_handler=data_handler, 
        verbose=verbose, 
        **kwargs
    )
    
    # Add other datasets here if needed
    
    return result_set


def get_data_dir(
    data_source: str = 'ensembl',
    data_dir: Optional[str] = None,
    data_handler: Optional[MetaModelDataHandler] = None,
    verbose: int = 0
) -> str:
    """
    Get the appropriate data directory for the specified data source.
    
    This function uses a consistent approach to resolve data directories using
    the project structure conventions. It supports multiple data sources and
    handles both explicit paths and data handler contexts.
    
    Args:
        data_source: Data source identifier (e.g., 'ensembl')
        data_dir: Explicitly provided data directory (highest priority)
        data_handler: MetaModelDataHandler instance with context (medium priority)
        verbose: Verbosity level for logging
    
    Returns:
        Resolved data directory path
    """
    import os
    
    # Priority 1: Explicitly provided data_dir
    if data_dir is not None:
        if verbose > 0:
            print(f"Using explicitly provided data directory: {data_dir}")
        return data_dir
    
    # Priority 2: Derive from data_handler context
    if data_handler is not None:
        if hasattr(data_handler, 'eval_dir'):
            # Navigate from eval_dir back to project root, then to data source
            # Typically eval_dir is like <proj_dir>/data/ensembl/spliceai_eval
            parent_dir = os.path.dirname(data_handler.eval_dir)
            if os.path.basename(parent_dir) == data_source:
                # Already at <proj_dir>/data/ensembl level
                result_dir = parent_dir
            else:
                # Need to find data/<source> within project structure
                result_dir = os.path.join(parent_dir, data_source)
            
            if verbose > 0:
                print(f"Derived data directory from data_handler: {result_dir}")
            return result_dir
    
    # Priority 3: Use project directory and data source convention
    try:
        proj_dir = get_proj_dir()
        result_dir = os.path.join(proj_dir, 'data', data_source)
        if verbose > 0:
            print(f"Using project-based data directory: {result_dir}")
        return result_dir
    except Exception as e:
        if verbose > 0:
            print(f"Error getting project directory: {e}")
    
    # Priority 4: Fall back to Analyzer.data_dir (global default)
    result_dir = Analyzer.data_dir
    if verbose > 0:
        print(f"Falling back to Analyzer.data_dir: {result_dir}")
    return result_dir


def load_chromosome_sequences(
    target_genes: Optional[List[str]] = None,
    data_dir: Optional[str] = None,
    data_handler: Optional[MetaModelDataHandler] = None,
    data_source: str = 'ensembl',
    seq_type: str = 'standard',
    verbose: int = 1,
    **kwargs
) -> pl.DataFrame:
    """
    Load gene sequence data from chromosome-specific files.
    
    This function loads gene sequences from chromosome-specific Parquet files,
    potentially filtering for specific target genes if provided.
    
    Args:
        target_genes: Optional list of gene IDs to filter for
        data_dir: Base data directory. If None, uses a resolved data directory
        data_handler: Optional pre-configured data handler
        data_source: Data source identifier (default: 'ensembl')
        seq_type: Type of sequence files to load ('standard' or 'minmax')
        verbose: Verbosity level
        **kwargs: Additional keyword arguments
    
    Returns:
        DataFrame containing gene sequences
    """
    import os
    import glob
    
    # Get the data directory using our utility function
    data_dir = get_data_dir(
        data_source=data_source,
        data_dir=data_dir,
        data_handler=data_handler,
        verbose=verbose if verbose > 1 else 0
    )
    
    # Determine pattern for gene sequence files based on sequence type
    if seq_type.lower() == 'minmax':
        gene_seq_pattern = os.path.join(data_dir, 'gene_sequence_minmax_*.parquet')
        main_file = os.path.join(data_dir, 'gene_sequence_minmax.parquet')
    else:  # standard
        gene_seq_pattern = os.path.join(data_dir, 'gene_sequence_*.parquet')
        # Exclude minmax files from standard pattern search
        exclude_pattern = os.path.join(data_dir, 'gene_sequence_minmax_*.parquet')
        main_file = os.path.join(data_dir, 'gene_sequence.parquet')
    
    if verbose > 0:
        print_emphasized("Loading Chromosome Sequences")
        print(f"Looking for gene sequence files matching: {gene_seq_pattern}")
    
    # First check if main combined file exists
    use_main_file = False
    if os.path.exists(main_file):
        if verbose > 0:
            print(f"Found main sequence file: {main_file}")
        try:
            # Try loading from main file first
            main_df = pl.read_parquet(main_file)
            if target_genes is not None:
                filtered_df = main_df.filter(
                    pl.col('gene_id').is_in(target_genes) | 
                    pl.col('gene_name').is_in(target_genes)
                )
                if filtered_df.height > 0:
                    if verbose > 0:
                        print(f"Found {filtered_df.height} target genes in main sequence file")
                    return filtered_df
            else:
                # If no target genes specified and main file exists, use it
                use_main_file = True
                return main_df
        except Exception as e:
            if verbose > 0:
                print(f"Error loading main sequence file: {e}")
    
    # If main file doesn't exist or doesn't contain our target genes, check chromosome files
    if verbose > 0:
        if not use_main_file:
            print(f"Looking for chromosome-specific {seq_type} sequence files...")

    # Find all chromosome sequence files
    seq_files = glob.glob(gene_seq_pattern)
    
    # For standard sequences, exclude minmax files
    if seq_type.lower() == 'standard' and 'exclude_pattern' in locals():
        exclude_files = set(glob.glob(exclude_pattern))
        seq_files = [f for f in seq_files if f not in exclude_files]
    
    if not seq_files:
        raise FileNotFoundError(f"No gene sequence files found matching pattern: {gene_seq_pattern}")
    
    if verbose > 0:
        print(f"Found {len(seq_files)} chromosome-specific {seq_type} sequence files")
    
    # Process each chromosome file
    all_sequences = []
    
    for file_path in seq_files:
        chrom = os.path.basename(file_path).replace('gene_sequence_', '').replace('.parquet', '')
        
        if verbose > 1:
            print(f"Loading sequences from chromosome {chrom}")
        
        try:
            seq_df = pl.read_parquet(file_path)
            
            # If target genes are specified, filter for them
            if target_genes is not None:
                seq_df = seq_df.filter(
                    pl.col('gene_id').is_in(target_genes) | 
                    pl.col('gene_name').is_in(target_genes)
                )
            
            if seq_df.height > 0:
                all_sequences.append(seq_df)
                if verbose > 1:
                    print(f"  Found {seq_df.height} relevant sequence entries in chromosome {chrom}")
        except Exception as e:
            if verbose > 0:
                print(f"Error loading chromosome {chrom} sequences: {e}")
    
    # Combine all sequences
    if all_sequences:
        combined_df = pl.concat(all_sequences)
        if verbose > 0:
            print(f"Combined {combined_df.height} sequence entries from {len(all_sequences)} chromosomes")
        return combined_df
    else:
        if verbose > 0:
            print("No relevant sequences found")
        return pl.DataFrame()
