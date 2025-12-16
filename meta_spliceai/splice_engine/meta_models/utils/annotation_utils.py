"""
Utilities for analyzing and processing splice site annotations.
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Any

def analyze_splicing_patterns(enhanced_df, gene_types=None, title=""):
    """
    Analyze splicing patterns for specified gene types
    
    Parameters
    ----------
    enhanced_df : pl.DataFrame
        Enhanced splice sites dataframe
    gene_types : list, optional
        List of gene types to include, by default None (all types)
    title : str, optional
        Title for the analysis, by default ""
        
    Returns
    -------
    pl.DataFrame
        Filtered dataframe based on gene_types (if provided)
    """
    if isinstance(enhanced_df, pd.DataFrame):
        enhanced_df = pl.DataFrame(enhanced_df)

    if gene_types:
        df = enhanced_df.filter(pl.col('gene_type').is_in(gene_types))
        type_str = ", ".join(gene_types)
        print(f"\n=== {title} (Gene Types: {type_str}) ===")
    else:
        df = enhanced_df
        print(f"\n=== {title} (All Gene Types) ===")
    
    # 1. Count genes with valid splice sites and average transcripts per gene
    gene_counts = df.select(['gene_id', 'gene_type']).unique()
    gene_transcript_counts = df.select(['gene_id', 'transcript_id', 'gene_type']).unique()
    
    # Group by gene_type and count unique genes
    genes_by_type = gene_counts.group_by('gene_type').agg(
        pl.n_unique('gene_id').alias('gene_count')
    ).sort('gene_count', descending=True)
    
    # Calculate average transcripts per gene by gene_type
    transcripts_per_gene = gene_transcript_counts.group_by(['gene_type', 'gene_id']).agg(
        pl.n_unique('transcript_id').alias('transcript_count')
    )
    
    avg_transcripts = transcripts_per_gene.group_by('gene_type').agg(
        pl.mean('transcript_count').alias('avg_transcripts_per_gene'),
        pl.count('gene_id').alias('gene_count')
    ).sort('gene_count', descending=True)
    
    # Merge with gene names if available
    gene_names = None
    if 'gene_name' in df.columns:
        # Add gene name information
        gene_names = df.select(['gene_id', 'gene_name']).unique()
    
    print(f"Total genes with splice sites: {gene_counts.height}")
    print(f"Breakdown by gene type:")
    print(genes_by_type.head(10))
    
    print(f"\nAverage transcripts per gene by gene type:")
    print(avg_transcripts.head(10))
    
    # 3. Find alternatively spliced genes (genes with multiple transcripts)
    # Group by gene_id, count transcripts, and sort
    alt_spliced_genes = transcripts_per_gene.filter(
        pl.col('transcript_count') > 1
    ).sort('transcript_count', descending=True)
    
    print(f"\nTotal genes with alternative splicing: {alt_spliced_genes.height}")
    
    # Get top 10 alternatively spliced genes
    top_alt_spliced = alt_spliced_genes.head(10)
    
    # Add gene names if available
    if gene_names is not None:
        top_alt_spliced = top_alt_spliced.join(
            gene_names, on='gene_id', how='left'
        )
    
    print(f"\nTop 10 genes with most alternative transcripts:")
    print(top_alt_spliced)
    
    # Extra: Count splice site types (donor vs acceptor) by gene type
    if 'site_type' in df.columns:
        site_type_by_gene = df.group_by(['gene_type', 'site_type']).agg(
            pl.count('gene_id').alias('site_count')
        ).sort(['gene_type', 'site_count'], descending=[False, True])
        
        print(f"\nSplice site types by gene type:")
        print(site_type_by_gene.head(20))
    
    return df