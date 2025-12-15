"""
Extract gene features from MANE GTF format.

MANE GTF files don't have explicit 'gene' feature lines - they only have
'transcript', 'exon', and 'CDS' features. This module aggregates transcript-level
features to create gene-level features.
"""

import polars as pl
from pathlib import Path
from typing import Optional


def extract_gene_features_from_mane_gtf(
    gtf_file_path: str,
    output_file: Optional[str] = None,
    verbose: int = 1
) -> pl.DataFrame:
    """
    Extract gene-level features from a MANE GTF file by aggregating transcripts.
    
    MANE GTF files have this structure:
    - transcript features (with gene_id, gene_name in attributes)
    - exon features (child of transcript)
    - CDS features (child of transcript)
    
    But NO explicit 'gene' features. We aggregate transcript features to create
    gene-level features.
    
    Parameters
    ----------
    gtf_file_path : str
        Path to the MANE GTF file
    output_file : str, optional
        Path to save the gene features TSV
    verbose : int
        Verbosity level
    
    Returns
    -------
    pl.DataFrame
        Gene features with columns: gene_id, gene_name, gene_type, chrom, 
        start, end, strand, gene_length, num_transcripts
    
    Examples
    --------
    >>> gene_df = extract_gene_features_from_mane_gtf(
    ...     'data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf'
    ... )
    >>> print(f"Extracted {gene_df.height} genes")
    """
    if verbose >= 1:
        print(f"[mane] Extracting gene features from MANE GTF: {gtf_file_path}")
    
    # Define GTF columns
    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    
    # Load GTF file
    gtf_df = pl.read_csv(
        gtf_file_path,
        separator='\t',
        comment_prefix='#',
        has_header=False,
        new_columns=columns,
        schema_overrides={'seqname': pl.Utf8}
    )
    
    if verbose >= 2:
        print(f"[mane] Loaded {gtf_df.height:,} rows from GTF")
        print(f"[mane] Feature types: {gtf_df['feature'].unique().to_list()}")
    
    # Filter for transcript features (these have gene information)
    transcript_df = gtf_df.filter(pl.col('feature') == 'transcript')
    
    if verbose >= 2:
        print(f"[mane] Found {transcript_df.height:,} transcript features")
    
    # Extract gene_id, gene_name from attributes
    transcript_df = transcript_df.with_columns([
        pl.col('attribute').str.extract(r'gene_id "([^"]+)"').alias('gene_id'),
        pl.col('attribute').str.extract(r'gene_name "([^"]+)"').alias('gene_name'),
        pl.col('attribute').str.extract(r'transcript_id "([^"]+)"').alias('transcript_id')
    ])
    
    # MANE is primarily protein-coding, but we can try to infer gene_type
    # For now, default to 'protein_coding' (MANE focus)
    transcript_df = transcript_df.with_columns([
        pl.lit('protein_coding').alias('gene_type')
    ])
    
    # Aggregate transcripts to gene level
    # For each gene:
    # - min(start) across all transcripts
    # - max(end) across all transcripts
    # - first strand (should be consistent)
    # - count of transcripts
    gene_df = transcript_df.group_by('gene_id').agg([
        pl.col('gene_name').first().alias('gene_name'),
        pl.col('gene_type').first().alias('gene_type'),
        pl.col('seqname').first().alias('chrom'),
        pl.col('start').min().alias('start'),
        pl.col('end').max().alias('end'),
        pl.col('strand').first().alias('strand'),
        pl.col('transcript_id').n_unique().alias('num_transcripts')
    ])
    
    # Calculate gene length
    gene_df = gene_df.with_columns([
        (pl.col('end') - pl.col('start') + 1).alias('gene_length')
    ])
    
    # Select and order columns
    gene_df = gene_df.select([
        'gene_id',
        'gene_name',
        'gene_type',
        'chrom',
        'start',
        'end',
        'strand',
        'gene_length',
        'num_transcripts'
    ])
    
    # Sort by chromosome and position
    gene_df = gene_df.sort(['chrom', 'start'])
    
    if verbose >= 1:
        print(f"[mane] Extracted {gene_df.height:,} unique genes")
        print(f"[mane] Columns: {gene_df.columns}")
        
        # Show strand distribution
        strand_counts = gene_df.group_by('strand').agg(pl.count()).sort('strand')
        print(f"[mane] Strand distribution:")
        for row in strand_counts.iter_rows(named=True):
            print(f"  {row['strand']}: {row['count']:,} genes")
    
    # Save if requested
    if output_file:
        gene_df.write_csv(output_file, separator='\t')
        if verbose >= 1:
            print(f"[mane] Saved gene features to: {output_file}")
    
    return gene_df


def is_mane_gtf(gtf_file_path: str) -> bool:
    """
    Check if a GTF file is in MANE format (no 'gene' features).
    
    Parameters
    ----------
    gtf_file_path : str
        Path to GTF file
    
    Returns
    -------
    bool
        True if MANE format (no 'gene' features), False otherwise
    """
    # Read first 1000 lines to check
    columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
    
    sample_df = pl.read_csv(
        gtf_file_path,
        separator='\t',
        comment_prefix='#',
        has_header=False,
        new_columns=columns,
        n_rows=1000
    )
    
    # Check if there are any 'gene' features
    has_gene_features = (sample_df['feature'] == 'gene').any()
    
    # MANE format: no 'gene' features, but has 'transcript' features
    has_transcript_features = (sample_df['feature'] == 'transcript').any()
    
    return (not has_gene_features) and has_transcript_features


if __name__ == '__main__':
    # Test on MANE GTF
    import sys
    
    gtf_file = 'data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf'
    output_file = 'data/mane/GRCh38/gene_features.tsv'
    
    print("=" * 80)
    print("Extract Gene Features from MANE GTF")
    print("=" * 80)
    print()
    
    # Check if MANE format
    print("Checking GTF format...")
    is_mane = is_mane_gtf(gtf_file)
    print(f"Is MANE format: {is_mane}")
    print()
    
    # Extract gene features
    gene_df = extract_gene_features_from_mane_gtf(
        gtf_file,
        output_file=output_file,
        verbose=2
    )
    
    print()
    print("Sample of extracted genes:")
    print(gene_df.head(10))
    print()
    
    print("=" * 80)
    print("✅ Complete!")
    print("=" * 80)


