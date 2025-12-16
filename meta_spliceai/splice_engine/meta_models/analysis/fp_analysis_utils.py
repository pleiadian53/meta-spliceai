#!/usr/bin/env python
"""
FP Analysis Utilities

Helper functions for FP reduction analysis that mirror the structure of FN rescue analysis.
"""

import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Local imports
from meta_spliceai.splice_engine.meta_models.utils.workflow_utils import (
    print_emphasized, 
    print_with_indent
)

# Import detailed counting function from FN analysis
from meta_spliceai.splice_engine.meta_models.analysis.shared_analysis_utils import (
    get_detailed_splice_site_counts
)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100


def analyze_genes_with_most_fps_by_type(
    positions_df: pl.DataFrame, 
    output_dir: Optional[str] = None,
    splice_type: Optional[str] = None, 
    top_n: int = 10,
    gene_types: Optional[List[str]] = None,
    gene_features_path: Optional[str] = None,
    project_dir: Optional[str] = None,
    use_detailed_counts: bool = False,
    verbose: int = 1
) -> pd.DataFrame:
    """
    Analyze genes with the highest FP rates, with optional filtering by gene type
    and/or splice site type.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing all positions with prediction types
    output_dir : str, optional
        Directory to save output visualizations and statistics
    splice_type : Optional[str], optional
        Type of splice site to analyze ('donor' or 'acceptor'). If None, all splice sites are included.    
    top_n : int, optional
        Number of top genes to analyze, by default 10
    gene_types : Optional[List[str]], optional
        List of gene types to include (e.g., ["protein_coding", "lncRNA"]). 
        If None, all gene types are considered.
    gene_features_path : Optional[str], optional
        Path to the gene features TSV file. If None, will try to find in default location.
    project_dir : Optional[str], optional
        Project directory root, used to find default file locations if paths not provided.
    use_detailed_counts : bool, optional
        Whether to use detailed counts for additional metrics, by default False
    verbose : int, optional
        Verbosity level (0=silent, 1=basic info, 2=detailed), by default 1
        
    Returns
    -------
    pd.DataFrame
        DataFrame with gene-level FP statistics
    """
    # If no gene type filtering is needed and we're using the function in a simpler way
    if gene_types is None and gene_features_path is None:
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Filter to the specific splice type if provided
        if splice_type:
            filtered_df = positions_df.filter(pl.col('splice_type') == splice_type)
            type_label = f"{splice_type.capitalize()} "
            file_prefix = f"top_genes_{splice_type}_"
        else:
            filtered_df = positions_df  # No filtering by splice type
            type_label = ""  # No splice type in labels
            file_prefix = "top_genes_all_"
        
        # Filter to FPs
        fp_df = filtered_df.filter(pl.col('pred_type') == 'FP')
        
        # Get FP counts by gene
        gene_fp_counts = (
            fp_df
            .group_by('gene_id')
            .agg(pl.len().alias('fp_count'))
            .sort('fp_count', descending=True)
            .to_pandas()
        )
        
        # Get total positions per gene for context
        gene_total_counts = (
            filtered_df
            .group_by('gene_id')
            .agg(pl.len().alias('total_positions'))
            .to_pandas()
        )
        
        # Merge to get percentage
        gene_stats = gene_fp_counts.merge(gene_total_counts, on='gene_id', how='left')
        gene_stats['fp_percentage'] = (gene_stats['fp_count'] / gene_stats['total_positions']) * 100
        gene_stats = gene_stats.sort_values('fp_count', ascending=False).head(top_n)
        gene_stats = gene_stats.set_index('gene_id')
        
        # Create visualization if output_dir is provided
        if output_dir:
            plt.figure(figsize=(12, 8))
            ax = gene_stats['fp_count'].head(top_n).plot(kind='bar')
            ax.set_title(f'Genes with Most {type_label}FPs')
            ax.set_xlabel('Gene ID')
            ax.set_ylabel('FP Count')
            
            # Add gene ID labels
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{file_prefix}fps.png'))
            plt.close()  # Close the figure to prevent memory leaks
            
            # Save the data
            gene_stats.to_csv(os.path.join(output_dir, f'{file_prefix}fps.csv'))
        
        return gene_stats
    
    # If gene_types filtering is needed, use the enhanced approach
    # Resolve project directory if needed
    if project_dir is None:
        # Try to infer project directory from system config
        try:
            from meta_spliceai.system.config import Config
            project_dir = Config.PROJ_DIR
        except ImportError:
            # Fallback to using HOME environment variable instead of hardcoded path
            project_dir = os.path.join(os.environ.get('HOME', ''), 'work', 'splice-surveyor')
            if verbose >= 1:
                print(f"[warning] Could not import Config, using fallback project dir: {project_dir}")
    
    # Derive default gene features path if not provided
    if gene_features_path is None:
        gene_features_path = os.path.join(
            project_dir, "data", "ensembl", "spliceai_analysis", "gene_features.tsv"
        )
        if verbose >= 1:
            print(f"[info] Using default gene features path: {gene_features_path}")
    
    # Load gene features
    if verbose >= 1:
        print(f"[i/o] Loading gene features from: {gene_features_path}")
    
    if not os.path.exists(gene_features_path):
        raise FileNotFoundError(f"Gene features file not found: {gene_features_path}")
        
    gene_features_df = pl.read_csv(
        gene_features_path,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}
    )
    
    # Display gene type distribution if verbose
    if verbose >= 2 and 'gene_type' in gene_features_df.columns:
        gene_type_counts = gene_features_df.group_by('gene_type').agg(
            pl.count('gene_id').alias('count')
        ).sort('count', descending=True)
        print("[info] Gene type distribution:")
        print(gene_type_counts.head(10))
    
    # Filter gene features by requested gene types
    if gene_types:
        filtered_gene_features = gene_features_df.filter(
            pl.col('gene_type').is_in(gene_types)
        )
        filtered_gene_ids = filtered_gene_features.select('gene_id').to_series().to_list()
        
        if verbose >= 1:
            print(f"[info] Filtered to {len(filtered_gene_ids)} genes of types: {', '.join(gene_types)}")
        
        # Filter positions dataframe to only include genes of the requested types
        filtered_positions_df = positions_df.filter(
            pl.col('gene_id').is_in(filtered_gene_ids)
        )
        
        if verbose >= 1:
            original_count = positions_df.height
            filtered_count = filtered_positions_df.height
            print(f"[info] Filtered positions from {original_count:,} to {filtered_count:,} rows ({filtered_count/original_count*100:.1f}%)")
    else:
        filtered_positions_df = positions_df
    
    # Filter by splice_type if specified
    if splice_type:
        splice_filtered_df = filtered_positions_df.filter(pl.col('splice_type') == splice_type)
        splice_label = f" {splice_type.capitalize()}"
        file_suffix = f"_{splice_type}"
        if verbose >= 1:
            print(f"[info] Filtered to {splice_filtered_df.height:,} {splice_type} positions")
    else:
        splice_filtered_df = filtered_positions_df  # No filtering by splice type
        splice_label = ""  # No splice type label
        file_suffix = "_all"  # Use 'all' in filenames
        if verbose >= 1:
            print(f"[info] Using all {splice_filtered_df.height:,} positions (both donor and acceptor)")
    
    # Filter to just the FPs
    fp_df = splice_filtered_df.filter(pl.col('pred_type') == 'FP')
    
    if use_detailed_counts:
        # Get detailed counts with all metrics using the same function as FN analysis
        counts_df = get_detailed_splice_site_counts(splice_filtered_df)
        
        # Sort by FP count and take top N
        initial_result = counts_df.sort_values("fp_count", ascending=False).head(top_n)
        
        # Add additional metrics specific to FP analysis
        initial_result['fp_percentage'] = (initial_result['fp_count'] / initial_result['total_positions']) * 100
        if 'tp_count' in initial_result.columns and 'fp_count' in initial_result.columns:
            # Avoid division by zero
            denominator = initial_result['tp_count'] + initial_result['fp_count']
            initial_result['precision'] = initial_result.apply(
                lambda x: x['tp_count'] / denominator[x.name] * 100 if denominator[x.name] > 0 else 0, 
                axis=1
            )
    else:
        # Use simpler counting approach
        # Get FP counts by gene
        gene_fp_counts = (
            fp_df
            .group_by('gene_id')
            .agg(pl.len().alias('fp_count'))
            .sort('fp_count', descending=True)
            .limit(top_n)  # Apply the limit in polars before converting to pandas
            .to_pandas()
        )
        
        # Get total positions per gene for context
        gene_total_counts = (
            splice_filtered_df
            .group_by('gene_id')
            .agg(pl.len().alias('total_positions'))
            .to_pandas()
        )
        
        # Merge to get percentage
        initial_result = gene_fp_counts.merge(gene_total_counts, on='gene_id', how='left')
        initial_result['fp_percentage'] = (initial_result['fp_count'] / initial_result['total_positions']) * 100
        
        # Ensure index doesn't cause problems later
        initial_result = initial_result.reset_index(drop=True)
    
    # Add gene type information to the result
    gene_types_df = gene_features_df.select(['gene_id', 'gene_type', 'gene_name']).to_pandas()
    result = initial_result.merge(gene_types_df, on='gene_id', how='left')
    
    # Ensure we maintain only top_n genes after the merge
    if len(result) > top_n:
        result = result.sort_values('fp_count', ascending=False).head(top_n)
    
    # Reset index to avoid duplicate columns when saving/loading
    result = result.reset_index(drop=True)
    
    # Make sure gene_id is a string to avoid type issues when filtering
    result['gene_id'] = result['gene_id'].astype(str)
    
    # Print results
    if gene_types:
        print(f"\nTop {top_n} {', '.join(gene_types)} genes with most{splice_label} False Positives:")
    else:
        print(f"\nTop {top_n} genes with most{splice_label} False Positives:")
    print(result)
    
    # Create visualization if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot distribution by gene type
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=result, x='gene_id', y='fp_count', hue='gene_type')
        plt.title(f'Genes with Most{splice_label} FPs by Gene Type')
        plt.ylabel('FP Count')
        plt.xlabel('Gene ID')
        plt.xticks(rotation=45, ha='right')
        
        # Place legend in upper right corner instead of using 'best' to avoid performance warning
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'top_genes{file_suffix}_fps_by_gene_type.png'))
        plt.close()
        
        # Plot as a horizontal bar chart with gene names instead of IDs
        plt.figure(figsize=(14, 10))
        result_with_names = result.copy()
        # Use gene name if available, otherwise use gene_id
        result_with_names['display_name'] = result_with_names.apply(
            lambda x: f"{x['gene_name']} ({x['gene_id']})" if pd.notna(x['gene_name']) else x['gene_id'], 
            axis=1
        )
        ax = sns.barplot(data=result_with_names, y='display_name', x='fp_count', hue='gene_type')
        plt.title(f'Top {top_n} Genes with Most{splice_label} FPs')
        plt.xlabel('FP Count')
        plt.ylabel('Gene')
        
        # Place legend in lower right corner instead of using 'best' to avoid performance warning
        ax.legend(loc='lower right', bbox_to_anchor=(1, 0))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'top_genes{file_suffix}_fps_horizontal.png'))
        plt.close()
        
        # Save the summary to CSV
        summary_path = os.path.join(output_dir, f'top_genes{file_suffix}_fps_by_gene_type.csv')
        result.to_csv(summary_path, index=False)
        
        output_type = splice_type if splice_type else "all"
        print(f"[output] Saved gene-level {output_type} FP summary to: {summary_path}")
    
    return result


def analyze_genes_with_most_fps(
    positions_df: pl.DataFrame, 
    output_dir: Optional[str] = None, 
    top_n: int = 10,
    gene_types: Optional[List[str]] = None,
    gene_features_path: Optional[str] = None,
    project_dir: Optional[str] = None,
    verbose: int = 1, 
    use_detailed_counts: bool = False
) -> pd.DataFrame:
    """
    Analyze genes with the highest overall FP rates and generate summary statistics.
    This is now a wrapper around analyze_genes_with_most_fps_by_type with splice_type=None.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing all positions with prediction types
    output_dir : str, optional
        Directory to save output visualizations and statistics
    top_n : int, optional
        Number of top genes to analyze, by default 10
    gene_types : Optional[List[str]], optional
        List of gene types to include (e.g., ["protein_coding", "lncRNA"]). 
        If None, all gene types are considered.
    gene_features_path : Optional[str], optional
        Path to the gene features TSV file. If None, will try to find in default location.
    project_dir : Optional[str], optional
        Project directory root, used to find default file locations if paths not provided.
    verbose : int, optional
        Verbosity level (0=silent, 1=basic info, 2=detailed), by default 1
    use_detailed_counts : bool, optional
        Whether to use detailed counts for additional metrics, by default False
        
    Returns
    -------
    pd.DataFrame
        DataFrame with gene-level FP statistics
    """
    # Call analyze_genes_with_most_fps_by_type with splice_type=None to analyze all splice sites
    return analyze_genes_with_most_fps_by_type(
        positions_df=positions_df,
        output_dir=output_dir,
        splice_type=None,  # Analyze all splice sites (both donor and acceptor)
        top_n=top_n,
        gene_types=gene_types,
        gene_features_path=gene_features_path,
        project_dir=project_dir,
        use_detailed_counts=use_detailed_counts,
        verbose=verbose
    )
