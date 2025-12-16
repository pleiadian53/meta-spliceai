#!/usr/bin/env python3
"""
Module for mapping genomic positions to transcripts and calculating transcript-level metrics.

This module provides functions to load and process transcript annotations,
map genomic positions to transcript IDs, and calculate transcript-level 
top-k accuracy metrics.
"""
from __future__ import annotations

import os
import time
import hashlib
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from meta_spliceai.system.config import Config
from typing import Dict, Tuple, List, Optional, Union, Any

from meta_spliceai.splice_engine.meta_models.evaluation.top_k_metrics import calculate_site_top_k_accuracy


def load_splice_site_annotations(splice_sites_path: str) -> pd.DataFrame:
    """
    Load splice site annotations from the TSV file.
    
    Parameters
    ----------
    splice_sites_path : str
        Path to the splice site annotations TSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing splice site annotations
    """
    return pd.read_csv(splice_sites_path, sep='\t')


def load_transcript_features(transcript_features_path: str) -> pd.DataFrame:
    """
    Load transcript features from the TSV file.
    
    Parameters
    ----------
    transcript_features_path : str
        Path to the transcript features TSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing transcript features
    """
    return pd.read_csv(transcript_features_path, sep='\t')


def load_gene_features(gene_features_path: str) -> pd.DataFrame:
    """
    Load gene features from the TSV file.
    
    Parameters
    ----------
    gene_features_path : str
        Path to the gene features TSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing gene features
    """
    return pd.read_csv(gene_features_path, sep='\t')


# Cache directory for mapped transcript data - store near annotation files
CACHE_DIR = os.path.join(Config.DATA_DIR, 'ensembl', 'cache')


def _generate_cache_key(positions, chromosomes, splice_sites_path, transcript_features_path):
    """Generate a unique cache key based on input data."""
    # Create a hash based on the file modification times and data fingerprint
    hash_input = []
    
    # Add the file modification timestamps
    if os.path.exists(splice_sites_path):
        hash_input.append(str(os.path.getmtime(splice_sites_path)))
    
    if os.path.exists(transcript_features_path):
        hash_input.append(str(os.path.getmtime(transcript_features_path)))
    
    # Add a fingerprint of the input data (first 100 positions and chromosomes)
    sample_size = min(100, len(positions))
    hash_input.extend([str(p) for p in positions[:sample_size]])
    hash_input.extend([str(c) for c in chromosomes[:sample_size]])
    
    # Add length of data
    hash_input.append(str(len(positions)))
    
    # Create a hash
    key = hashlib.md5('_'.join(hash_input).encode()).hexdigest()
    return key


def save_transcript_mapping_cache(transcript_ids, cache_key):
    """Save transcript mapping results to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"tx_map_{cache_key}.pkl")
    
    with open(cache_path, 'wb') as f:
        pickle.dump(transcript_ids, f)
    
    print(f"Saved transcript mapping to cache: {cache_path}")


def load_transcript_mapping_cache(cache_key):
    """Load transcript mapping results from cache if available."""
    cache_path = os.path.join(CACHE_DIR, f"tx_map_{cache_key}.pkl")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                transcript_ids = pickle.load(f)
            print(f"Loaded transcript mapping from cache: {cache_path}")
            return transcript_ids
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    return None


def map_positions_to_transcripts(
    positions: np.ndarray,
    chromosomes: np.ndarray,
    splice_sites_df: pd.DataFrame,
    transcript_features_df: pd.DataFrame,
    site_type_col: str = 'site_type',
    splice_sites_path: str = None,
    transcript_features_path: str = None,
    use_cache: bool = True
) -> np.ndarray:
    """
    Map genomic positions to transcript IDs using the exact schema of the splice_sites.tsv file.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of genomic positions
    chromosomes : np.ndarray
        Array of chromosomes
    splice_sites_df : pd.DataFrame
        DataFrame with splice site annotations
    transcript_features_df : pd.DataFrame
        DataFrame with transcript features
    site_type_col : str, default='site_type'
        Name of column in splice_sites_df containing site type (donor/acceptor)
        This may be 'site_type' or 'splice_type' depending on annotation source
        
    Returns
    -------
    np.ndarray
        Array of transcript IDs for each position ("unknown" if not found)
    """
    # Check if we should use caching and have the required file paths
    if use_cache and splice_sites_path and transcript_features_path:
        # Generate cache key based on input data and annotation files
        start_time = time.time()
        cache_key = _generate_cache_key(positions, chromosomes, splice_sites_path, transcript_features_path)
        
        # Try to load from cache
        cached_transcript_ids = load_transcript_mapping_cache(cache_key)
        if cached_transcript_ids is not None and len(cached_transcript_ids) == len(positions):
            # Cache hit - return the cached mapping
            return cached_transcript_ids
        else:
            print(f"Cache miss or invalid cache. Performing full transcript mapping...")
    
    print(f"Mapping {len(positions)} positions to transcripts...")
    transcript_ids = np.array(["unknown"] * len(positions), dtype=object)
    
    # 1. First try exact position matching
    print("  Mapping by exact match to splice sites...")
    for idx, (chrom, pos) in enumerate(zip(chromosomes, positions)):
        exact_matches = splice_sites_df[
            (splice_sites_df['chrom'] == chrom) &
            (splice_sites_df['position'] == pos)
        ]
        if len(exact_matches) > 0:
            transcript_ids[idx] = exact_matches.iloc[0]['transcript_id']
    
    # 2. For positions not matched, try matching by transcript boundaries
    unmatched_indices = np.where(transcript_ids == "unknown")[0]
    print(f"  Mapping by transcript boundaries...")
    for idx in unmatched_indices:
        chrom, pos = chromosomes[idx], positions[idx]
        tx_matches = transcript_features_df[
            (transcript_features_df['chrom'] == chrom) &
            (transcript_features_df['start'] <= pos) &
            (transcript_features_df['end'] >= pos)
        ]
        if len(tx_matches) > 0:
            transcript_ids[idx] = tx_matches.iloc[0]['transcript_id']
    
    # 3. For remaining positions, find closest splice site within 50bp
    unmatched_indices = np.where(transcript_ids == "unknown")[0]
    print(f"  Mapping by closest splice site (within 50bp)...")
    for idx in unmatched_indices:
        chrom, pos = chromosomes[idx], positions[idx]
        # Find all splice sites on the same chromosome
        chrom_splice_sites = splice_sites_df[splice_sites_df['chrom'] == chrom].copy()
        if len(chrom_splice_sites) > 0:
            # Calculate distance to each splice site
            chrom_splice_sites['distance'] = abs(chrom_splice_sites['position'] - pos)
            # Find the closest splice site within 50bp
            closest_site = chrom_splice_sites[chrom_splice_sites['distance'] <= 50].sort_values('distance')
            if len(closest_site) > 0:
                transcript_ids[idx] = closest_site.iloc[0]['transcript_id']
    
    # Report mapping statistics
    mapped_count = sum(transcript_ids != "unknown")
    print(f"  Mapped {mapped_count}/{len(positions)} positions to transcripts ({mapped_count/len(positions):.1%})")
    if mapped_count < len(positions):
        print(f"Warning: {len(positions) - mapped_count} positions ({(len(positions) - mapped_count)/len(positions):.1%}) could not be mapped to any transcript")
    
    # Save to cache if enabled
    if use_cache and splice_sites_path and transcript_features_path:
        save_transcript_mapping_cache(transcript_ids, cache_key)
        end_time = time.time()
        print(f"Mapping completed in {end_time - start_time:.2f} seconds")
    
    return transcript_ids


def calculate_transcript_level_top_k(
    df: pd.DataFrame, 
    splice_sites_path: str,
    transcript_features_path: str,
    gene_features_path: Optional[str] = None,
    position_col: str = 'position',
    chrom_col: str = 'chrom',
    label_col: str = 'label',
    prob_donor_col: str = 'prob_donor',
    prob_acceptor_col: str = 'prob_acceptor',
    site_type_col: str = 'site_type',
    donor_label: int = 0,
    acceptor_label: int = 1,
    neither_label: int = 2,
    use_cache: bool = True,
    ) -> Dict[str, float]:
    """
    Calculate transcript-level top-k accuracy by mapping positions to transcripts.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with genomic positions, labels, and predicted probabilities
        Must contain columns: position_col, chrom_col, label, prob_donor, prob_acceptor
    splice_sites_path : str
        Path to splice site annotations file (splice_sites.tsv)
        Expected columns: chrom, start, end, position, strand, site_type, gene_id, transcript_id
    transcript_features_path : str
        Path to transcript features file (transcript_features.tsv)
        Expected columns: chrom, start, end, strand, transcript_id, transcript_name, transcript_type, transcript_length, gene_id
    gene_features_path : Optional[str]
        Path to gene features file for gene-level aggregation (gene_features.tsv)
        Expected columns: start, end, score, strand, gene_id, gene_name, gene_type, gene_length, chrom
    donor_label : int, default=0
        Value in df['label'] that represents donor sites
    acceptor_label : int, default=1
        Value in df['label'] that represents acceptor sites
    position_col : str, default='position'
        Column name for genomic positions
    chrom_col : str, default='chrom'
        Column name for chromosomes
    site_type_col : str, default='site_type'
        Name of column in annotation files containing splice site type
        Could be 'site_type' or 'splice_type' depending on annotation source
        
    Returns
    -------
    Dict[str, float]
        Dictionary with transcript-level top-k accuracy metrics
    """
    # Check if required columns exist
    required_cols = [position_col, chrom_col, 'label', 'prob_donor', 'prob_acceptor']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check if annotation files exist
    for filepath in [splice_sites_path, transcript_features_path]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Annotation file not found: {filepath}")
    
    # Load annotation files
    print(f"Loading splice site annotations from {splice_sites_path}")
    splice_sites_df = load_splice_site_annotations(splice_sites_path)
    
    print(f"Loading transcript features from {transcript_features_path}")
    transcript_features_df = load_transcript_features(transcript_features_path)
    
    # Check if site_type_col exists in splice_sites_df and adapt if necessary
    if site_type_col not in splice_sites_df.columns:
        if 'site_type' in splice_sites_df.columns:
            site_type_col = 'site_type'
            print(f"Using 'site_type' column for splice site types")
        elif 'splice_type' in splice_sites_df.columns:
            site_type_col = 'splice_type'
            print(f"Using 'splice_type' column for splice site types")
        else:
            raise ValueError(f"Neither '{site_type_col}' nor 'site_type'/'splice_type' column found in splice sites data")
    
    transcript_ids = map_positions_to_transcripts(
        positions=df[position_col].values,
        chromosomes=df[chrom_col].values,
        splice_sites_df=splice_sites_df,
        transcript_features_df=transcript_features_df,
        site_type_col=site_type_col,
        splice_sites_path=splice_sites_path,
        transcript_features_path=transcript_features_path,
        use_cache=use_cache
    )
    
    # Add transcript IDs to predictions DataFrame
    df_with_tx = df.copy()
    df_with_tx['transcript_id'] = transcript_ids
    
    # Filter out unknown transcripts
    known_transcripts = df_with_tx[df_with_tx['transcript_id'] != 'unknown']
    unknown_count = len(df_with_tx) - len(known_transcripts)
    if unknown_count > 0:
        print(f"Warning: {unknown_count} positions ({unknown_count/len(df_with_tx):.1%}) could not be mapped to any transcript")
    
    if len(known_transcripts) == 0:
        print("No positions could be mapped to transcripts, cannot calculate transcript-level metrics")
        return {"transcript_error": "No positions mapped to transcripts"}
    
    print(f"Calculating top-k accuracy for {len(known_transcripts)} positions across "
          f"{known_transcripts['transcript_id'].nunique()} transcripts")
    
    # Count donors and acceptors per transcript for reporting
    donor_counts = known_transcripts[known_transcripts['label'] == donor_label].groupby('transcript_id').size()
    acceptor_counts = known_transcripts[known_transcripts['label'] == acceptor_label].groupby('transcript_id').size()
    
    print(f"Found {len(donor_counts)} transcripts with donors and {len(acceptor_counts)} with acceptors")
    
    # Calculate top-k accuracy using transcript IDs as group
    tx_metrics = calculate_site_top_k_accuracy(
        known_transcripts,
        group_col='transcript_id',
        donor_label=donor_label,
        acceptor_label=acceptor_label
    )
    
    # Add additional metrics
    tx_metrics['n_transcripts_with_donors'] = len(donor_counts)
    tx_metrics['n_transcripts_with_acceptors'] = len(acceptor_counts)
    tx_metrics['avg_donors_per_transcript'] = donor_counts.mean() if len(donor_counts) > 0 else 0
    tx_metrics['avg_acceptors_per_transcript'] = acceptor_counts.mean() if len(acceptor_counts) > 0 else 0
    tx_metrics['pct_positions_mapped'] = len(known_transcripts) / len(df_with_tx) if len(df_with_tx) > 0 else 0
    
    # Add gene-level aggregation if gene features are provided
    if gene_features_path and os.path.exists(gene_features_path):
        try:
            gene_features_df = load_gene_features(gene_features_path)
            
            # Link transcript_id to gene_id using transcript features
            tx_to_gene = dict(zip(transcript_features_df['transcript_id'], transcript_features_df['gene_id']))
            # Create a copy to avoid SettingWithCopyWarning
            known_transcripts = known_transcripts.copy()
            known_transcripts['gene_id'] = known_transcripts['transcript_id'].map(tx_to_gene)
            
            # Calculate gene-level aggregated metrics
            genes_with_transcripts = known_transcripts['gene_id'].nunique()
            print(f"Aggregating metrics across {genes_with_transcripts} genes")
            
            tx_metrics['n_genes_with_transcripts'] = genes_with_transcripts
            tx_metrics['avg_transcripts_per_gene'] = known_transcripts['transcript_id'].nunique() / genes_with_transcripts if genes_with_transcripts > 0 else 0
        except Exception as e:
            print(f"Warning: Gene-level aggregation failed: {e}")
    
    # Add prefix to metrics
    return {f"transcript_{k}": v for k, v in tx_metrics.items()}


def report_transcript_top_k(metrics: Dict[str, float]) -> str:
    """
    Generate a formatted report of transcript-level top-k accuracy metrics.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of transcript-level top-k accuracy metrics
        
    Returns
    -------
    str
        Formatted report
    """
    # Check for error condition
    if 'transcript_error' in metrics:
        return f"Transcript-level metrics unavailable: {metrics['transcript_error']}"
    
    # Basic metrics
    n_transcripts = metrics.get('transcript_n_groups', 0)
    n_donor_tx = metrics.get('transcript_n_transcripts_with_donors', 0)
    n_acceptor_tx = metrics.get('transcript_n_transcripts_with_acceptors', 0)
    
    # Format the report
    report = []
    report.append(f"Top-k Accuracy (Transcript-level, {n_transcripts} transcripts):")
    report.append(f"  Donor:     {metrics.get('transcript_donor_top_k', 0):.4f}")
    report.append(f"  Acceptor:  {metrics.get('transcript_acceptor_top_k', 0):.4f}")
    report.append(f"  Combined:  {metrics.get('transcript_combined_top_k', 0):.4f}")
    
    # Additional statistics
    report.append("\nTranscript Statistics:")
    report.append(f"  Transcripts with donors:    {n_donor_tx}")
    report.append(f"  Transcripts with acceptors: {n_acceptor_tx}")
    report.append(f"  Avg donors per transcript:    {metrics.get('transcript_avg_donors_per_transcript', 0):.2f}")
    report.append(f"  Avg acceptors per transcript: {metrics.get('transcript_avg_acceptors_per_transcript', 0):.2f}")
    
    # Mapping quality
    mapped_pct = metrics.get('transcript_pct_positions_mapped', 0) * 100
    report.append(f"  Positions mapped to transcripts: {mapped_pct:.1f}%")
    
    # Gene-level aggregation if available
    if 'transcript_n_genes_with_transcripts' in metrics:
        n_genes = metrics.get('transcript_n_genes_with_transcripts', 0)
        avg_tx_per_gene = metrics.get('transcript_avg_transcripts_per_gene', 0)
        report.append(f"\nGene Statistics:")
        report.append(f"  Genes with transcripts: {n_genes}")
        report.append(f"  Avg transcripts per gene: {avg_tx_per_gene:.2f}")
    
    return "\n".join(report)
