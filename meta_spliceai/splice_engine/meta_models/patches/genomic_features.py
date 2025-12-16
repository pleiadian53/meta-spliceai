"""
Patches for genomic feature extraction functions to address file format issues.
"""

import os
import pandas as pd
import polars as pl
from typing import Union, Optional, Dict, List, Any, Tuple

from meta_spliceai.splice_engine.extract_genomic_features import (
    compute_intron_lengths as original_compute_intron_lengths
)


def patched_compute_intron_lengths(gtf_file_path: str) -> pd.DataFrame:
    """
    Patched version of compute_intron_lengths that handles format mismatches.
    
    Parameters
    ----------
    gtf_file_path : str
        Path to GTF file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with intron lengths
    """
    # Get the analysis directory path
    analysis_dir = os.path.dirname(gtf_file_path)
    if not analysis_dir:
        analysis_dir = '.'
    
    # Path to the exon dataframe
    exon_df_path = os.path.join(analysis_dir, 'exon_df_from_gtf.tsv')
    
    # Use the new smart_read_csv function from utils_df to handle format mismatches
    from meta_spliceai.splice_engine.utils_df import smart_read_csv
    
    # Load the exon dataframe and convert to pandas for compatibility
    exon_df = smart_read_csv(exon_df_path, use_polars=True).to_pandas()
    
    # Fix types
    exon_df['start'] = exon_df['start'].astype(int)
    exon_df['end'] = exon_df['end'].astype(int)
    
    # Group by transcript_id
    grouped = exon_df.groupby('transcript_id')
    
    # Compute intron lengths for each transcript
    intron_lengths = []
    for transcript_id, exons in grouped:
        # Sort exons by start position
        exons = exons.sort_values('start')
        
        # Extract coordinates and strand
        coords = list(zip(exons['start'], exons['end']))
        strand = exons['strand'].iloc[0]
        gene_id = exons['gene_id'].iloc[0]
        
        # Calculate intron lengths
        for i in range(len(coords) - 1):
            intron_start = coords[i][1] + 1
            intron_end = coords[i + 1][0] - 1
            intron_length = intron_end - intron_start + 1
            
            intron_lengths.append({
                'transcript_id': transcript_id,
                'gene_id': gene_id,
                'intron_start': intron_start,
                'intron_end': intron_end,
                'intron_length': intron_length,
                'strand': strand,
                'intron_index': i if strand == '+' else len(coords) - 2 - i
            })
    
    # Create DataFrame from the list of dictionaries
    intron_lengths_df = pd.DataFrame(intron_lengths)
    
    # Add statistics about intron lengths for each transcript
    stats = intron_lengths_df.groupby('transcript_id')['intron_length'].agg(['mean', 'median', 'min', 'max', 'count'])
    stats.columns = ['mean_intron_length', 'median_intron_length', 'min_intron_length', 'max_intron_length', 'num_introns']
    stats = stats.reset_index()
    
    # Merge stats back to the original DataFrame
    intron_lengths_df = pd.merge(intron_lengths_df, stats, on='transcript_id')
    
    return intron_lengths_df
