#!/usr/bin/env python3
"""
Generate gene_features.tsv for GRCh37

Extracts gene-level features from GTF including gene_type (biotype),
which is needed for biotype-specific gene sampling.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from meta_spliceai.system.genomic_resources import Registry
from meta_spliceai.system.genomic_resources.derive import GenomicDataDeriver


def main():
    """Generate gene_features.tsv for GRCh37."""
    
    print(f"\n{'='*80}")
    print("GENERATING GENE FEATURES FOR GRCh37")
    print(f"{'='*80}\n")
    
    # Setup
    build = 'GRCh37'
    release = '87'
    registry = Registry(build=build, release=release)
    
    print(f"Build: {build}")
    print(f"Release: {release}")
    print(f"Data directory: {registry.data_dir}")
    print()
    
    # Check if file already exists
    gene_features_file = registry.data_dir / 'gene_features.tsv'
    if gene_features_file.exists():
        print(f"⚠️  gene_features.tsv already exists at {gene_features_file}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
        print()
    
    # Create deriver
    deriver = GenomicDataDeriver(
        data_dir=registry.data_dir,
        registry=registry,
        verbosity=2  # Detailed output
    )
    
    print("Extracting gene features from GTF...")
    print("This will take a few minutes...\n")
    
    # Extract gene features
    result = deriver.derive_gene_features(
        output_filename='gene_features.tsv',
        target_chromosomes=None,  # All chromosomes
        force_overwrite=True
    )
    
    if not result['success']:
        print(f"\n❌ Failed to generate gene features: {result.get('error')}")
        return 1
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}\n")
    
    # Verify the output
    import polars as pl
    
    gene_features_df = pl.read_csv(
        gene_features_file,
        separator='\t',
        schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8}
    )
    
    print(f"✅ Generated: {gene_features_file}")
    print(f"   Total genes: {gene_features_df.height:,}")
    print(f"   Unique chromosomes: {gene_features_df['chrom'].n_unique()}")
    print()
    
    # Show gene type distribution
    if 'gene_type' in gene_features_df.columns:
        print("Gene type distribution (top 10):")
        biotype_counts = gene_features_df.group_by('gene_type').agg(
            pl.count()
        ).sort('count', descending=True)
        
        for row in biotype_counts.head(10).iter_rows(named=True):
            print(f"  {row['gene_type']:30s}: {row['count']:6,} genes")
    else:
        print("⚠️  Warning: gene_type column not found in output")
    
    print()
    print("✅ gene_features.tsv is ready for use!")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

