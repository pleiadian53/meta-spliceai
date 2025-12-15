#!/usr/bin/env python3
"""
Regenerate complete splice_sites_enhanced.tsv for GRCh37.

This script extracts splice sites from ALL chromosomes in the GRCh37 GTF,
not just chr21 and chr22.

Date: November 2, 2025
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from meta_spliceai.system.genomic_resources import Registry
from meta_spliceai.system.genomic_resources.derive import GenomicDataDeriver


def main():
    """Regenerate complete splice sites for GRCh37."""
    
    print(f"\n{'='*80}")
    print("REGENERATING COMPLETE SPLICE SITES FOR GRCh37")
    print(f"{'='*80}\n")
    
    # Set up paths
    registry = Registry(build='GRCh37', release='87')
    gtf_path = Path(registry.resolve('gtf'))
    data_dir = registry.data_dir
    output_path = data_dir / 'splice_sites_enhanced.tsv'
    
    print(f"Input GTF:    {gtf_path}")
    print(f"Output dir:   {data_dir}")
    print(f"Output file:  {output_path}")
    print()
    
    # Verify GTF exists
    if not gtf_path.exists():
        print(f"[ERROR] GTF file not found: {gtf_path}")
        return 1
    
    print("Extracting splice sites from ALL chromosomes...")
    print("This will take 5-10 minutes...")
    print()
    
    # Create deriver
    deriver = GenomicDataDeriver(
        data_dir=data_dir,
        registry=registry,
        verbosity=2  # Detailed output
    )
    
    # Extract splice sites (NO chromosome filter!)
    # This uses extract_splice_sites_workflow internally, which creates the enhanced version
    result = deriver.derive_splice_sites(
        output_filename='splice_sites_enhanced.tsv',
        consensus_window=2,  # Standard window
        target_chromosomes=None,  # ALL chromosomes!
        force_overwrite=True  # Force regeneration
    )
    
    if not result.get('success'):
        print(f"\n[ERROR] Splice site extraction failed: {result.get('error')}")
        return 1
    
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}\n")
    
    # Verify the output
    import polars as pl
    
    df = result['splice_sites_df']
    if df is None:
        df = pl.read_csv(str(output_path), separator='\t')
    
    print(f"Total splice sites: {df.height:,}")
    print(f"Unique genes: {df['gene_id'].n_unique():,}")
    print()
    
    # Check chromosome distribution
    chrom_dist = df.group_by('chrom').agg(pl.len()).sort('chrom')
    
    print("Chromosome distribution:")
    print(chrom_dist)
    print()
    
    # Verify all major chromosomes present
    expected_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
    missing_chroms = []
    
    for chrom in expected_chroms:
        count = df.filter(pl.col('chrom') == (int(chrom) if chrom.isdigit() else chrom)).height
        if count == 0:
            missing_chroms.append(chrom)
    
    if missing_chroms:
        print(f"⚠️  Warning: Missing chromosomes: {', '.join(missing_chroms)}")
    else:
        print("✅ All expected chromosomes present!")
    
    print()
    print(f"Output saved to: {output_path}")
    print()
    print("Next step: Re-run multi-gene test with complete data")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

