"""
GTF Utilities for MetaSpliceAI

This module provides utility functions for working with GTF files,
particularly for extracting gene information and mapping genes to chromosomes.
"""

import os
import polars as pl
from typing import Dict, List, Union, Optional, Set


def get_gene_chromosomes(
    gtf_file: str, 
    gene_ids: Optional[List[str]] = None,
    gene_name_col: str = "gene_name",
    gene_id_col: str = "gene_id"
) -> Dict[str, str]:
    """
    Extract chromosome information for specified genes from a GTF file.
    
    This function scans a GTF file and creates a mapping from gene IDs or names
    to their corresponding chromosomes. It can search by both gene ID and gene name.
    
    Parameters
    ----------
    gtf_file : str
        Path to the GTF file to extract chromosome information from
    gene_ids : List[str], optional
        List of gene IDs or gene names to look up. If None, returns info for all genes.
    gene_name_col : str, optional
        Column name for gene names in the GTF, by default "gene_name"
    gene_id_col : str, optional
        Column name for gene IDs in the GTF, by default "gene_id"
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping gene IDs/names to chromosome names
        
    Examples
    --------
    >>> get_gene_chromosomes("path/to/gtf_file.gtf", ["STMN2", "UNC13A"])
    {'STMN2': '1', 'UNC13A': '19'}
    
    >>> get_gene_chromosomes("path/to/gtf_file.gtf", ["ENSG00000104435"])
    {'ENSG00000104435': '1'}
    """
    if not os.path.exists(gtf_file):
        raise FileNotFoundError(f"GTF file not found: {gtf_file}")
    
    # Use different query optimization based on whether gene_ids is provided
    if gene_ids is None:
        # For all genes, use a more memory-efficient approach with streaming
        return _get_all_gene_chromosomes(gtf_file, gene_name_col, gene_id_col)
    else:
        # For specific genes, use a more targeted approach
        return _get_specific_gene_chromosomes(gtf_file, gene_ids, gene_name_col, gene_id_col)


def _get_specific_gene_chromosomes(
    gtf_file: str, 
    gene_ids: List[str], 
    gene_name_col: str,
    gene_id_col: str
) -> Dict[str, str]:
    """
    Get chromosome information for specific genes using an optimized query.
    """
    # Convert all gene IDs to lowercase for case-insensitive matching if needed
    # gene_ids_lower = [g.lower() for g in gene_ids]
    gene_ids_set = set(gene_ids)
    
    # Read only gene lines from GTF file with minimal columns
    query = (
        pl.scan_csv(gtf_file, 
                   separator='\t', 
                   has_header=False,
                   comment_prefix='#',
                   new_columns=['seqname', 'source', 'feature', 'start', 'end', 
                               'score', 'strand', 'frame', 'attribute'],
                   schema_overrides={'seqname': pl.Utf8, 'source': pl.Utf8, 
                                    'feature': pl.Utf8, 'score': pl.Utf8,
                                    'strand': pl.Utf8, 'frame': pl.Utf8, 
                                    'attribute': pl.Utf8})
        .filter(pl.col('feature') == 'gene')
        .select(['seqname', 'attribute'])
    )
    
    # Extract different gene information from the attribute column
    # Note: This is a simplified parser and might need adjustment for different GTF formats
    gene_df = query.with_columns([
        pl.col('attribute').str.extract(f'{gene_id_col} "(.*?)"', 1).alias(gene_id_col),
        pl.col('attribute').str.extract(f'{gene_name_col} "(.*?)"', 1).alias(gene_name_col)
    ]).collect()
    
    # Initialize results dictionary
    gene_to_chrom = {}
    
    # Find matches by gene ID
    for row in gene_df.iter_rows(named=True):
        gene_id = row[gene_id_col]
        gene_name = row[gene_name_col]
        chrom = row['seqname']
        
        # Check if this gene ID or name is in our target list
        if gene_id in gene_ids_set:
            gene_to_chrom[gene_id] = chrom
        
        if gene_name in gene_ids_set:
            gene_to_chrom[gene_name] = chrom
            
    return gene_to_chrom


def _get_all_gene_chromosomes(
    gtf_file: str, 
    gene_name_col: str,
    gene_id_col: str
) -> Dict[str, str]:
    """
    Get chromosome information for all genes using streaming to reduce memory usage.
    """
    # Read only gene lines from GTF file with minimal columns
    query = (
        pl.scan_csv(gtf_file, 
                   separator='\t', 
                   has_header=False,
                   comment_prefix='#',
                   new_columns=['seqname', 'source', 'feature', 'start', 'end', 
                               'score', 'strand', 'frame', 'attribute'],
                   schema_overrides={'seqname': pl.Utf8, 'source': pl.Utf8, 
                                    'feature': pl.Utf8, 'score': pl.Utf8,
                                    'strand': pl.Utf8, 'frame': pl.Utf8, 
                                    'attribute': pl.Utf8})
        .filter(pl.col('feature') == 'gene')
        .select(['seqname', 'attribute'])
    )
    
    # Extract gene IDs and names from the attribute column
    gene_df = query.with_columns([
        pl.col('attribute').str.extract(f'{gene_id_col} "(.*?)"', 1).alias(gene_id_col),
        pl.col('attribute').str.extract(f'{gene_name_col} "(.*?)"', 1).alias(gene_name_col)
    ]).collect()
    
    # Create gene ID to chromosome mapping
    id_to_chrom = {row[gene_id_col]: row['seqname'] for row in gene_df.iter_rows(named=True)}
    
    # Create gene name to chromosome mapping and merge with ID mapping
    name_to_chrom = {row[gene_name_col]: row['seqname'] for row in gene_df.iter_rows(named=True) 
                    if row[gene_name_col] and row[gene_name_col] != row[gene_id_col]}
    
    # Combine both dictionaries
    gene_to_chrom = {**id_to_chrom, **name_to_chrom}
    
    return gene_to_chrom


def get_chromosome_genes(
    gtf_file: str, 
    chromosomes: Optional[List[str]] = None,
    feature_type: str = 'gene'
) -> Dict[str, List[str]]:
    """
    Get all gene IDs for specified chromosomes.
    
    Parameters
    ----------
    gtf_file : str
        Path to the GTF file
    chromosomes : List[str], optional
        List of chromosome names to filter by. If None, includes all chromosomes.
    feature_type : str, optional
        Type of feature to extract, by default 'gene'
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping chromosome names to lists of gene IDs
    """
    if not os.path.exists(gtf_file):
        raise FileNotFoundError(f"GTF file not found: {gtf_file}")
    
    # Read gene lines from GTF file
    query = (
        pl.scan_csv(gtf_file, 
                   separator='\t', 
                   has_header=False,
                   comment_prefix='#',
                   new_columns=['seqname', 'source', 'feature', 'start', 'end', 
                               'score', 'strand', 'frame', 'attribute'],
                   schema_overrides={'seqname': pl.Utf8, 'source': pl.Utf8, 
                                    'feature': pl.Utf8, 'score': pl.Utf8,
                                    'strand': pl.Utf8, 'frame': pl.Utf8, 
                                    'attribute': pl.Utf8})
        .filter(pl.col('feature') == feature_type)
    )
    
    # Apply chromosome filter if specified
    if chromosomes:
        query = query.filter(pl.col('seqname').is_in(chromosomes))
    
    # Extract gene IDs from attribute column
    gene_df = query.with_columns(
        pl.col('attribute').str.extract('gene_id "(.*?)"', 1).alias('gene_id')
    ).select(['seqname', 'gene_id']).collect()
    
    # Group by chromosome
    chrom_to_genes = {}
    for chrom, group in gene_df.group_by('seqname'):
        chrom_to_genes[chrom] = group['gene_id'].to_list()
    
    return chrom_to_genes
