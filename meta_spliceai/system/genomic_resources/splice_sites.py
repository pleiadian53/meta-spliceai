"""Splice site extraction from GTF annotations.

This module provides functions for extracting canonical splice sites (donors and acceptors)
from GTF files, with enhanced metadata including gene names, biotypes, and exon information.

The extracted splice sites include:
- Core columns: chrom, start, end, position, strand, site_type, gene_id, transcript_id
- Enhanced columns: gene_name, gene_biotype, transcript_biotype, exon_id, exon_number, exon_rank

This is a refactored and enhanced version of the splice site extraction logic originally
in meta_spliceai/splice_engine/extract_genomic_features.py.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import polars as pl
import gffutils
from tqdm import tqdm


def extract_splice_sites_from_gtf(
    gtf_path: Union[str, Path],
    consensus_window: int = 2,
    output_file: Optional[Union[str, Path]] = None,
    save: bool = True,
    return_df: bool = True,
    verbosity: int = 1,
    **kwargs
) -> Union[pd.DataFrame, str]:
    """Extract enhanced splice sites from a GTF file.
    
    Identifies canonical splice sites (donors and acceptors) from exon boundaries,
    including enhanced metadata from transcript and exon attributes.
    
    Parameters
    ----------
    gtf_path : str or Path
        Path to the GTF annotation file
    consensus_window : int, default=2
        Window size around splice sites for consensus sequence (±N nucleotides)
    output_file : str or Path, optional
        Path to save the splice sites TSV file. If None and save=True, 
        saves to 'splice_sites_enhanced.tsv' in the same directory as the GTF.
    save : bool, default=True
        Whether to save the results to a file
    return_df : bool, default=True
        Whether to return a DataFrame. If False, returns list of dicts.
    verbosity : int, default=1
        Output verbosity level (0=silent, 1=normal, 2=detailed)
    **kwargs : dict
        Additional arguments (e.g., 'sep' for output separator)
        
    Returns
    -------
    pd.DataFrame or str
        If return_df=True, returns DataFrame with splice site annotations.
        If return_df=False and save=True, returns path to output file.
        
    Notes
    -----
    Output DataFrame columns (14 total):
    - chrom: Chromosome
    - start, end: Genomic coordinates (window around splice site)
    - position: Exact splice site position (1-based)
    - strand: '+' or '-'
    - site_type: 'donor' or 'acceptor'
    - gene_id: Ensembl gene ID
    - transcript_id: Ensembl transcript ID
    - gene_name: Human-readable gene symbol (e.g., "BRCA1")
    - gene_biotype: Gene classification (e.g., "protein_coding")
    - transcript_biotype: Transcript classification
    - exon_id: Ensembl exon ID
    - exon_number: Exon number from GTF
    - exon_rank: Exon position in transcription order (1-based)
    
    Strand-Specific Splice Site Logic:
    - Positive strand (+):
        * Donor Site: Located at exon_end + 1 (first base of intron), except for the last exon
        * Acceptor Site: Located at exon_start - 1 (last base of intron), except for the first exon
    - Negative strand (-):
        * Donor Site: Located at exon_start - 1 (first base of intron in transcription order)
        * Acceptor Site: Located at exon_end + 1 (last base of intron in transcription order)
        
    Examples
    --------
    >>> from meta_spliceai.system.genomic_resources import extract_splice_sites_from_gtf
    >>> 
    >>> # Extract with default settings
    >>> df = extract_splice_sites_from_gtf(
    ...     "data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf",
    ...     consensus_window=2,
    ...     output_file="splice_sites_enhanced.tsv"
    ... )
    >>> 
    >>> # Check enhanced columns
    >>> print(df.columns)
    >>> print(f"Total splice sites: {len(df)}")
    >>> print(f"Genes with names: {df['gene_name'].notna().sum()}")
    """
    gtf_path = Path(gtf_path)
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF file not found: {gtf_path}")
    
    # Determine output file
    if save and output_file is None:
        output_file = gtf_path.parent / "splice_sites_enhanced.tsv"
    
    if verbosity >= 1:
        print(f"[splice_sites] Extracting splice sites from: {gtf_path.name}")
        print(f"[splice_sites] Consensus window: ±{consensus_window} nucleotides")
    
    # Build GTF database
    db_file = gtf_path.parent / "annotations.db"
    if verbosity >= 1:
        print(f"[splice_sites] Building GTF database...")
    
    db = _build_or_load_gtf_database(gtf_path, db_file, verbosity=verbosity)
    
    # Extract splice sites with enhanced metadata
    if verbosity >= 1:
        print(f"[splice_sites] Extracting splice sites from transcripts...")
    
    all_splice_sites = []
    transcript_counter = 0
    
    for transcript in tqdm(
        db.features_of_type('transcript'), 
        desc="Processing transcripts",
        disable=(verbosity < 1)
    ):
        # Extract transcript-level attributes
        transcript_id = transcript.attributes.get('transcript_id', [transcript.id])[0]
        gene_id = transcript.attributes.get('gene_id', [''])[0]
        
        # ✨ NEW: Extract enhanced transcript-level attributes
        gene_name = transcript.attributes.get('gene_name', [''])[0]
        # Handle both 'gene_biotype' and 'gene_type' (different GTF formats)
        gene_biotype = (
            transcript.attributes.get('gene_biotype', [''])[0] or
            transcript.attributes.get('gene_type', [''])[0]
        )
        transcript_biotype = (
            transcript.attributes.get('transcript_biotype', [''])[0] or
            transcript.attributes.get('transcript_type', [''])[0]
        )
        
        strand = transcript.strand
        chrom = transcript.chrom
        
        # Extract exons, sorted by genomic coordinates
        exons = list(db.children(transcript, featuretype='exon', order_by='start'))
        
        # Skip transcripts with less than 2 exons (no splicing)
        if len(exons) < 2:
            continue
        
        # Adjust exon order for negative strand to match transcription order
        if strand == '-':
            exons = exons[::-1]
        
        # Iterate over exons to find splice sites
        for i in range(len(exons)):
            exon = exons[i]
            exon_start = exon.start  # 1-based
            exon_end = exon.end      # 1-based
            
            # ✨ NEW: Extract exon-specific attributes
            exon_id = exon.attributes.get('exon_id', [''])[0]
            exon_number = exon.attributes.get('exon_number', [''])[0]
            exon_rank = i + 1  # Position in transcription order
            
            # Validate exon order (sanity check)
            if strand == '+':
                if i > 0:
                    assert exons[i].start > exons[i - 1].start, \
                        f"Exon order error: {exons[i-1].start} >= {exons[i].start}"
            elif strand == '-':
                if i > 0:
                    assert exons[i].start < exons[i - 1].start, \
                        f"Exon order error: {exons[i-1].start} <= {exons[i].start}"
            
            # Extract splice sites based on strand
            if strand == '+':
                # Positive strand: Acceptor site: exon_start - 1 (except first exon)
                if i > 0:
                    position = exon_start - 1  # 1-based
                    start = position - consensus_window
                    end = position + consensus_window
                    
                    all_splice_sites.append({
                        # Core columns
                        'chrom': chrom,
                        'start': max(start, 1),  # Ensure start >= 1
                        'end': end,
                        'position': position,
                        'strand': strand,
                        'site_type': 'acceptor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id,
                        
                        # ✨ Enhanced columns
                        'gene_name': gene_name,
                        'gene_biotype': gene_biotype,
                        'transcript_biotype': transcript_biotype,
                        'exon_id': exon_id,
                        'exon_number': exon_number,
                        'exon_rank': exon_rank
                    })
                
                # Donor site: exon_end + 1 (except last exon)
                if i < len(exons) - 1:
                    position = exon_end + 1  # 1-based
                    start = position - consensus_window
                    end = position + consensus_window
                    
                    all_splice_sites.append({
                        # Core columns
                        'chrom': chrom,
                        'start': max(start, 1),
                        'end': end,
                        'position': position,
                        'strand': strand,
                        'site_type': 'donor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id,
                        
                        # ✨ Enhanced columns
                        'gene_name': gene_name,
                        'gene_biotype': gene_biotype,
                        'transcript_biotype': transcript_biotype,
                        'exon_id': exon_id,
                        'exon_number': exon_number,
                        'exon_rank': exon_rank
                    })
            
            elif strand == '-':
                # Negative strand: Acceptor site: exon_end + 1 (except first exon in transcription order)
                if i > 0:
                    position = exon_end + 1  # 1-based
                    start = position - consensus_window
                    end = position + consensus_window
                    
                    all_splice_sites.append({
                        # Core columns
                        'chrom': chrom,
                        'start': max(start, 1),
                        'end': end,
                        'position': position,
                        'strand': strand,
                        'site_type': 'acceptor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id,
                        
                        # ✨ Enhanced columns
                        'gene_name': gene_name,
                        'gene_biotype': gene_biotype,
                        'transcript_biotype': transcript_biotype,
                        'exon_id': exon_id,
                        'exon_number': exon_number,
                        'exon_rank': exon_rank
                    })
                
                # Donor site: exon_start - 1 (except last exon in transcription order)
                if i < len(exons) - 1:
                    position = exon_start - 1  # 1-based
                    start = position - consensus_window
                    end = position + consensus_window
                    
                    all_splice_sites.append({
                        # Core columns
                        'chrom': chrom,
                        'start': max(start, 1),
                        'end': end,
                        'position': position,
                        'strand': strand,
                        'site_type': 'donor',
                        'gene_id': gene_id,
                        'transcript_id': transcript_id,
                        
                        # ✨ Enhanced columns
                        'gene_name': gene_name,
                        'gene_biotype': gene_biotype,
                        'transcript_biotype': transcript_biotype,
                        'exon_id': exon_id,
                        'exon_number': exon_number,
                        'exon_rank': exon_rank
                    })
        
        transcript_counter += 1
    
    if verbosity >= 1:
        print(f"[splice_sites] ✓ Processed {transcript_counter} transcripts")
        print(f"[splice_sites] ✓ Extracted {len(all_splice_sites)} splice sites")
    
    # Validation: Check unique genes and transcripts
    if verbosity >= 1 and all_splice_sites:
        df_temp = pd.DataFrame(all_splice_sites)
        unique_genes = df_temp['gene_id'].nunique()
        unique_transcripts = df_temp['transcript_id'].nunique()
        
        # Count splice sites by type
        donors = len(df_temp[df_temp['site_type'] == 'donor'])
        acceptors = len(df_temp[df_temp['site_type'] == 'acceptor'])
        
        print(f"[splice_sites] ✓ Unique genes: {unique_genes:,}")
        print(f"[splice_sites] ✓ Unique transcripts: {unique_transcripts:,}")
        print(f"[splice_sites] ✓ Donor sites: {donors:,} | Acceptor sites: {acceptors:,}")
        
        # Sanity check: donors and acceptors should be roughly balanced
        if donors > 0 and acceptors > 0:
            ratio = donors / acceptors
            if 0.8 <= ratio <= 1.2:
                print(f"[splice_sites] ✓ Donor/Acceptor ratio: {ratio:.2f} (balanced)")
            else:
                print(f"[splice_sites] ⚠ Donor/Acceptor ratio: {ratio:.2f} (imbalanced - check data)")
    
    # Save or return results
    if save and output_file:
        splice_sites_df = pd.DataFrame(all_splice_sites)
        sep = kwargs.get('sep', '\t')
        splice_sites_df.to_csv(output_file, index=False, sep=sep)
        
        if verbosity >= 1:
            print(f"[splice_sites] ✓ Saved to: {output_file}")
        
        if not return_df:
            return str(output_file)
    
    if return_df:
        return pd.DataFrame(all_splice_sites)
    else:
        return all_splice_sites


def _build_or_load_gtf_database(
    gtf_path: Path,
    db_file: Path,
    verbosity: int = 1
) -> gffutils.FeatureDB:
    """Build or load GTF database for efficient querying.
    
    Parameters
    ----------
    gtf_path : Path
        Path to GTF file
    db_file : Path
        Path to database file
    verbosity : int
        Output verbosity
        
    Returns
    -------
    gffutils.FeatureDB
        GTF feature database
    """
    if db_file.exists():
        if verbosity >= 2:
            print(f"[splice_sites] Loading existing database: {db_file.name}")
        return gffutils.FeatureDB(str(db_file))
    
    if verbosity >= 1:
        print(f"[splice_sites] Creating database (this may take a few minutes)...")
    
    db = gffutils.create_db(
        str(gtf_path),
        dbfn=str(db_file),
        force=False,
        keep_order=True,
        merge_strategy='merge',
        sort_attribute_values=True,
        disable_infer_genes=True,
        disable_infer_transcripts=True
    )
    
    if verbosity >= 1:
        print(f"[splice_sites] ✓ Database created: {db_file.name}")
    
    return db


# Convenience function for backward compatibility
def extract_splice_sites_workflow(
    data_prefix: Union[str, Path],
    gtf_file: Union[str, Path],
    consensus_window: int = 2,
    output_path: Optional[Union[str, Path]] = None,
    output_format: str = 'tsv',
    **kwargs
) -> str:
    """Legacy wrapper for extract_splice_sites_from_gtf.
    
    This function maintains backward compatibility with the original
    extract_splice_sites_workflow in extract_genomic_features.py.
    
    Parameters
    ----------
    data_prefix : str or Path
        Directory prefix for output files
    gtf_file : str or Path
        Path to GTF annotation file
    consensus_window : int, default=2
        Window size around splice sites
    output_path : str or Path, optional
        Full path to output file
    output_format : str, default='tsv'
        Output format ('tsv' or 'csv')
        
    Returns
    -------
    str
        Path to the generated splice sites file
        
    Examples
    --------
    >>> output_file = extract_splice_sites_workflow(
    ...     data_prefix="data/mane/GRCh38",
    ...     gtf_file="data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf",
    ...     consensus_window=2,
    ...     output_path="data/mane/GRCh38/splice_sites_enhanced.tsv"
    ... )
    """
    data_prefix = Path(data_prefix)
    
    # Determine output path
    if output_path is None:
        output_path = data_prefix / f"splice_sites_enhanced.{output_format}"
    else:
        output_path = Path(output_path)
    
    # Call the new function
    result = extract_splice_sites_from_gtf(
        gtf_path=gtf_file,
        consensus_window=consensus_window,
        output_file=output_path,
        save=True,
        return_df=False,
        verbosity=kwargs.get('verbose', 1),
        sep='\t' if output_format == 'tsv' else ','
    )
    
    return result

