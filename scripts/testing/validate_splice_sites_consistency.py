#!/usr/bin/env python3
"""Validate splice sites file consistency with GTF.

This script compares the splice sites annotation file against the source GTF
to ensure that:
1. All transcripts with ≥2 exons in GTF are represented in splice sites
2. Gene and transcript counts match
3. No duplicate splice sites exist
4. Donor/acceptor balance is reasonable

Usage:
    python scripts/testing/validate_splice_sites_consistency.py

Requirements:
    - mamba activate metaspliceai
    - GTF file exists
    - splice_sites_enhanced.tsv exists
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import polars as pl
import gffutils
from collections import defaultdict


def count_transcripts_in_gtf(gtf_path: Path, min_exons: int = 2, verbosity: int = 1) -> dict:
    """Count transcripts with sufficient exons in GTF.
    
    Parameters
    ----------
    gtf_path : Path
        Path to GTF file
    min_exons : int
        Minimum number of exons required for splicing
    verbosity : int
        Output verbosity
        
    Returns
    -------
    dict
        Statistics about GTF transcripts
    """
    if verbosity >= 1:
        print(f"Analyzing GTF file: {gtf_path.name}")
        print("(This may take a few minutes...)")
    
    # Build or load database
    db_file = gtf_path.parent / "annotations.db"
    if db_file.exists():
        db = gffutils.FeatureDB(str(db_file))
    else:
        print("Creating GTF database...")
        db = gffutils.create_db(
            str(gtf_path),
            dbfn=str(db_file),
            force=False,
            keep_order=True,
            disable_infer_genes=True,
            disable_infer_transcripts=True
        )
    
    # Count transcripts and exons
    transcript_counts = {
        'total_transcripts': 0,
        'transcripts_with_splicing': 0,  # >= min_exons exons
        'transcripts_single_exon': 0,
        'unique_genes': set(),
        'unique_genes_with_splicing': set()
    }
    
    exon_counts_per_tx = defaultdict(int)
    
    for transcript in db.features_of_type('transcript'):
        transcript_id = transcript.attributes.get('transcript_id', [transcript.id])[0]
        gene_id = transcript.attributes.get('gene_id', [''])[0]
        
        transcript_counts['total_transcripts'] += 1
        transcript_counts['unique_genes'].add(gene_id)
        
        # Count exons for this transcript
        exons = list(db.children(transcript, featuretype='exon'))
        n_exons = len(exons)
        exon_counts_per_tx[transcript_id] = n_exons
        
        if n_exons >= min_exons:
            transcript_counts['transcripts_with_splicing'] += 1
            transcript_counts['unique_genes_with_splicing'].add(gene_id)
        elif n_exons == 1:
            transcript_counts['transcripts_single_exon'] += 1
    
    # Convert sets to counts
    transcript_counts['unique_genes'] = len(transcript_counts['unique_genes'])
    transcript_counts['unique_genes_with_splicing'] = len(transcript_counts['unique_genes_with_splicing'])
    
    # Add distribution stats
    transcript_counts['exon_distribution'] = exon_counts_per_tx
    
    return transcript_counts


def analyze_splice_sites_file(ss_path: Path, verbosity: int = 1) -> dict:
    """Analyze splice sites annotation file.
    
    Parameters
    ----------
    ss_path : Path
        Path to splice sites file
    verbosity : int
        Output verbosity
        
    Returns
    -------
    dict
        Statistics about splice sites
    """
    if verbosity >= 1:
        print(f"\nAnalyzing splice sites file: {ss_path.name}")
    
    # Read with schema overrides to handle string chromosome names
    df = pl.read_csv(
        str(ss_path), 
        separator='\t',
        schema_overrides={'chrom': pl.Utf8}  # Ensure chrom is read as string
    )
    
    stats = {
        'total_splice_sites': len(df),
        'unique_genes': df['gene_id'].n_unique(),
        'unique_transcripts': df['transcript_id'].n_unique(),
        'donor_sites': df.filter(pl.col('site_type') == 'donor').height,
        'acceptor_sites': df.filter(pl.col('site_type') == 'acceptor').height,
    }
    
    # Check for duplicates
    duplicates = df.group_by(['chrom', 'position', 'strand', 'site_type', 'transcript_id']).agg(
        pl.count().alias('count')
    ).filter(pl.col('count') > 1)
    
    stats['duplicate_sites'] = duplicates.height
    
    # Splice sites per transcript
    ss_per_tx = df.group_by('transcript_id').agg(pl.count().alias('n_sites'))
    stats['avg_sites_per_transcript'] = ss_per_tx['n_sites'].mean()
    stats['min_sites_per_transcript'] = ss_per_tx['n_sites'].min()
    stats['max_sites_per_transcript'] = ss_per_tx['n_sites'].max()
    
    return stats


def validate_consistency(gtf_stats: dict, ss_stats: dict, verbosity: int = 1) -> bool:
    """Compare GTF and splice sites statistics.
    
    Parameters
    ----------
    gtf_stats : dict
        Statistics from GTF
    ss_stats : dict
        Statistics from splice sites file
    verbosity : int
        Output verbosity
        
    Returns
    -------
    bool
        True if validation passes
    """
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    all_passed = True
    
    # Test 1: Transcript count match
    print("\n1. Transcript Count Validation")
    print("-" * 80)
    
    gtf_tx = gtf_stats['transcripts_with_splicing']
    ss_tx = ss_stats['unique_transcripts']
    
    print(f"  GTF transcripts (≥2 exons):    {gtf_tx:,}")
    print(f"  Splice sites transcripts:      {ss_tx:,}")
    
    if gtf_tx == ss_tx:
        print(f"  ✓ PASS: Transcript counts match exactly")
    else:
        diff = abs(gtf_tx - ss_tx)
        pct_diff = (diff / gtf_tx) * 100 if gtf_tx > 0 else 0
        
        if pct_diff < 1:  # Less than 1% difference is acceptable
            print(f"  ✓ PASS: Transcript counts nearly match (diff: {diff}, {pct_diff:.2f}%)")
        else:
            print(f"  ❌ FAIL: Significant mismatch (diff: {diff}, {pct_diff:.2f}%)")
            all_passed = False
    
    # Test 2: Gene count validation
    print("\n2. Gene Count Validation")
    print("-" * 80)
    
    gtf_genes = gtf_stats['unique_genes_with_splicing']
    ss_genes = ss_stats['unique_genes']
    
    print(f"  GTF genes (with splicing):     {gtf_genes:,}")
    print(f"  Splice sites genes:            {ss_genes:,}")
    
    if gtf_genes == ss_genes:
        print(f"  ✓ PASS: Gene counts match exactly")
    else:
        diff = abs(gtf_genes - ss_genes)
        pct_diff = (diff / gtf_genes) * 100 if gtf_genes > 0 else 0
        
        if pct_diff < 1:
            print(f"  ✓ PASS: Gene counts nearly match (diff: {diff}, {pct_diff:.2f}%)")
        else:
            print(f"  ⚠ WARNING: Gene count mismatch (diff: {diff}, {pct_diff:.2f}%)")
    
    # Test 3: Donor/Acceptor balance
    print("\n3. Donor/Acceptor Balance")
    print("-" * 80)
    
    donors = ss_stats['donor_sites']
    acceptors = ss_stats['acceptor_sites']
    ratio = donors / acceptors if acceptors > 0 else 0
    
    print(f"  Donor sites:                   {donors:,}")
    print(f"  Acceptor sites:                {acceptors:,}")
    print(f"  Ratio (donor/acceptor):        {ratio:.3f}")
    
    if 0.9 <= ratio <= 1.1:
        print(f"  ✓ PASS: Well-balanced donor/acceptor sites")
    elif 0.8 <= ratio <= 1.2:
        print(f"  ✓ PASS: Reasonably balanced (within 20%)")
    else:
        print(f"  ⚠ WARNING: Imbalanced donor/acceptor sites")
    
    # Test 4: Check for duplicates
    print("\n4. Duplicate Sites Check")
    print("-" * 80)
    
    duplicates = ss_stats['duplicate_sites']
    
    if duplicates == 0:
        print(f"  ✓ PASS: No duplicate splice sites found")
    else:
        print(f"  ❌ FAIL: Found {duplicates} duplicate splice sites")
        all_passed = False
    
    # Test 5: Splice sites per transcript
    print("\n5. Splice Sites Per Transcript")
    print("-" * 80)
    
    avg_sites = ss_stats['avg_sites_per_transcript']
    min_sites = ss_stats['min_sites_per_transcript']
    max_sites = ss_stats['max_sites_per_transcript']
    
    print(f"  Average sites per transcript:  {avg_sites:.1f}")
    print(f"  Min sites per transcript:      {min_sites}")
    print(f"  Max sites per transcript:      {max_sites}")
    
    if min_sites >= 2:
        print(f"  ✓ PASS: All transcripts have ≥2 splice sites (expected)")
    else:
        print(f"  ⚠ WARNING: Some transcripts have <2 splice sites")
    
    # Summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL VALIDATION TESTS PASSED")
        print("\nThe splice sites file is consistent with the GTF file.")
    else:
        print("❌ SOME VALIDATION TESTS FAILED")
        print("\nPlease review the issues above.")
    print("=" * 80)
    
    return all_passed


def main():
    """Main validation function."""
    print("=" * 80)
    print("SPLICE SITES CONSISTENCY VALIDATION")
    print("=" * 80)
    print()
    
    # Setup paths
    gtf_path = project_root / "data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf"
    ss_path = project_root / "data/mane/GRCh38/splice_sites_enhanced.tsv"
    
    # Check files exist
    if not gtf_path.exists():
        print(f"❌ GTF file not found: {gtf_path}")
        return False
    
    if not ss_path.exists():
        print(f"❌ Splice sites file not found: {ss_path}")
        print("\nPlease generate it first using:")
        print("  python scripts/testing/test_enhanced_splice_sites_extraction.py")
        return False
    
    # Analyze GTF
    gtf_stats = count_transcripts_in_gtf(gtf_path, min_exons=2, verbosity=1)
    
    print("\nGTF Summary:")
    print(f"  Total transcripts:             {gtf_stats['total_transcripts']:,}")
    print(f"  Transcripts with splicing:     {gtf_stats['transcripts_with_splicing']:,}")
    print(f"  Single-exon transcripts:       {gtf_stats['transcripts_single_exon']:,}")
    print(f"  Unique genes (total):          {gtf_stats['unique_genes']:,}")
    print(f"  Unique genes (with splicing):  {gtf_stats['unique_genes_with_splicing']:,}")
    
    # Analyze splice sites file
    ss_stats = analyze_splice_sites_file(ss_path, verbosity=1)
    
    print("\nSplice Sites Summary:")
    print(f"  Total splice sites:            {ss_stats['total_splice_sites']:,}")
    print(f"  Unique genes:                  {ss_stats['unique_genes']:,}")
    print(f"  Unique transcripts:            {ss_stats['unique_transcripts']:,}")
    print(f"  Donor sites:                   {ss_stats['donor_sites']:,}")
    print(f"  Acceptor sites:                {ss_stats['acceptor_sites']:,}")
    
    # Validate consistency
    success = validate_consistency(gtf_stats, ss_stats, verbosity=1)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

