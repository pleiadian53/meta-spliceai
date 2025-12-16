"""
Utilities for splice junction analysis from enhanced SpliceAI predictions.

This module provides functions for identifying and analyzing splice junctions 
from enhanced SpliceAI predictions with all probability scores.
"""

import polars as pl
from typing import Tuple, Optional, List, Dict, Any


def identify_splice_junctions(
    positions_df: pl.DataFrame, 
    donor_threshold: float = 0.9, 
    acceptor_threshold: float = 0.9
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Identify splice junctions by pairing high-confidence donor and acceptor sites.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing splice site positions and their scores
    donor_threshold : float, optional
        Threshold for high-confidence donor sites, by default 0.9
    acceptor_threshold : float, optional
        Threshold for high-confidence acceptor sites, by default 0.9
    
    Returns
    -------
    tuple
        (junctions_df, donor_sites, acceptor_sites) - DataFrames containing junctions and high-confidence sites
    """
    # Get donor sites with high confidence
    donor_sites = positions_df.filter(pl.col('donor_score') > donor_threshold)
    # Get acceptor sites with high confidence
    acceptor_sites = positions_df.filter(pl.col('acceptor_score') > acceptor_threshold)
    
    # Form junctions by pairing donor and acceptor sites by gene and transcript
    junction_list = []
    
    # Process each gene separately
    for gene_id in positions_df['gene_id'].unique():
        gene_donors = donor_sites.filter(pl.col('gene_id') == gene_id).sort('position')
        gene_acceptors = acceptor_sites.filter(pl.col('gene_id') == gene_id).sort('position')
        
        if gene_donors.height == 0 or gene_acceptors.height == 0:
            continue
        
        # For each transcript, find the donor-acceptor pairs
        for transcript_id in positions_df.filter(pl.col('gene_id') == gene_id)['transcript_id'].unique():
            # Get sites for this transcript
            transcript_donors = gene_donors.filter(pl.col('transcript_id') == transcript_id)
            transcript_acceptors = gene_acceptors.filter(pl.col('transcript_id') == transcript_id)
            
            if transcript_donors.height == 0 or transcript_acceptors.height == 0:
                continue
            
            # Convert to lists for easier pairing
            donor_positions = transcript_donors['position'].to_list()
            acceptor_positions = transcript_acceptors['position'].to_list()
            
            # Match each donor to the next available acceptor
            # Since positions are stored as strand-aware relative positions,
            # donor sites will always come before acceptor sites when sorted
            for donor_pos in donor_positions:
                donor_score = transcript_donors.filter(pl.col('position') == donor_pos)['donor_score'].item()
                
                # Find the nearest acceptor downstream of the donor
                valid_acceptors = [pos for pos in acceptor_positions if pos > donor_pos]
                
                if valid_acceptors:
                    # Get the nearest acceptor (smallest position that's greater than donor_pos)
                    nearest_acceptor = min(valid_acceptors)
                    acceptor_score = transcript_acceptors.filter(pl.col('position') == nearest_acceptor)['acceptor_score'].item()
                    
                    junction_list.append({
                        'gene_id': gene_id,
                        'transcript_id': transcript_id,
                        'donor_position': donor_pos,
                        'acceptor_position': nearest_acceptor,
                        'donor_score': donor_score,
                        'acceptor_score': acceptor_score,
                        'junction_length': nearest_acceptor - donor_pos,
                        'strand': positions_df.filter(
                            (pl.col('gene_id') == gene_id) & 
                            (pl.col('position') == donor_pos)
                        )[0, 'strand'] if 'strand' in positions_df.columns else '+'
                    })
    
    # Create a DataFrame from the junctions or return empty DataFrame
    if junction_list:
        junctions_df = pl.DataFrame(junction_list)
    else:
        junctions_df = pl.DataFrame(schema={
            'gene_id': pl.Utf8, 
            'transcript_id': pl.Utf8, 
            'donor_position': pl.Int64, 
            'acceptor_position': pl.Int64,
            'donor_score': pl.Float64, 
            'acceptor_score': pl.Float64, 
            'junction_length': pl.Int64,
            'strand': pl.Utf8
        })
    
    return junctions_df, donor_sites, acceptor_sites


def report_junction_statistics(
    junctions_df: pl.DataFrame, 
    donor_sites: pl.DataFrame, 
    acceptor_sites: pl.DataFrame, 
    sample_size: int = 10
) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """
    Generate and print statistical reports for identified splice junctions.
    
    Parameters
    ----------
    junctions_df : pl.DataFrame
        DataFrame containing identified splice junctions
    donor_sites : pl.DataFrame
        DataFrame containing high-confidence donor sites
    acceptor_sites : pl.DataFrame
        DataFrame containing high-confidence acceptor sites
    sample_size : int, optional
        Number of sample junctions to display, by default 10
        
    Returns
    -------
    tuple
        (junction_counts, junctions_sample) - DataFrames with statistics for further analysis
    """
    # Report basic statistics
    print(f"Found {donor_sites.height} high-confidence donor sites and {acceptor_sites.height} high-confidence acceptor sites")
    
    # Check if any junctions were found
    if junctions_df.height == 0:
        print("\nNo high-confidence junctions found with the current thresholds.")
        return None, None
    
    # Report identified junctions
    print(f"\nIdentified {junctions_df.height} high-confidence junctions across all transcripts")
    
    # Display junctions sorted by gene and junction length
    print(f"\nSample of high-confidence junctions (sorted by gene and junction length, showing top {sample_size}):")
    junctions_sample = junctions_df.sort(['gene_id', 'junction_length']).head(sample_size)
    print(junctions_sample)
    
    # Count junctions per gene
    junction_counts = junctions_df.group_by('gene_id').agg(
        pl.len().alias('junction_count')
    ).sort('junction_count', descending=True)
    
    print("\nJunction counts per gene:")
    print(junction_counts)
    
    # Additional statistics that might be useful
    if 'junction_length' in junctions_df.columns:
        avg_length = junctions_df['junction_length'].mean()
        min_length = junctions_df['junction_length'].min()
        max_length = junctions_df['junction_length'].max()
        print(f"\nJunction length statistics:")
        print(f"  Average length: {avg_length:.2f}")
        print(f"  Minimum length: {min_length}")
        print(f"  Maximum length: {max_length}")
    
    return junction_counts, junctions_sample
