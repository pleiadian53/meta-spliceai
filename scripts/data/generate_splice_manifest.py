#!/usr/bin/env python3
"""
Generate a data manifest for splice site annotations with comprehensive statistics.

This script profiles splice site annotation files and compares them against source
GTF files to produce a detailed data manifest suitable for documentation and data
science workflows.

Usage:
    python generate_splice_manifest.py [--splice-file PATH] [--gtf-file PATH] [--output PATH]
"""

import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple


def count_unique_values(file_path: str, column: int, skip_header: bool = True) -> int:
    """
    Count unique values in a specific column of a TSV file.
    
    Args:
        file_path: Path to the TSV file
        column: Column number (1-indexed)
        skip_header: Whether to skip the first line
    
    Returns:
        Number of unique values
    """
    cmd = f"tail -n +{'2' if skip_header else '1'} {file_path} | cut -f{column} | sort -u | wc -l"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return int(result.stdout.strip())


def count_gtf_features(gtf_path: str, feature_type: str, id_field: str) -> int:
    """
    Count unique gene or transcript IDs in a GTF file.
    
    Args:
        gtf_path: Path to GTF file
        feature_type: Feature type to filter ('gene' or 'transcript')
        id_field: ID field to extract ('gene_id' or 'transcript_id')
    
    Returns:
        Number of unique IDs
    """
    cmd = f"awk '$3==\"{feature_type}\"' {gtf_path} | sed -E 's/.*{id_field} \"([^\"]+)\".*/\\1/' | sort -u | wc -l"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return int(result.stdout.strip())


def count_total_lines(file_path: str, skip_header: bool = True) -> int:
    """Count total data lines in a file."""
    cmd = f"tail -n +{'2' if skip_header else '1'} {file_path} | wc -l"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return int(result.stdout.strip())


def verify_subset(splice_file: str, gtf_file: str, column: int, 
                  feature_type: str, id_field: str) -> Tuple[bool, int]:
    """
    Verify if IDs in splice file are a subset of GTF file.
    
    Returns:
        Tuple of (is_subset, count_of_mismatches)
    """
    # Extract IDs from splice file
    splice_ids = f"/tmp/splice_{feature_type}.txt"
    gtf_ids = f"/tmp/gtf_{feature_type}.txt"
    
    cmd1 = f"tail -n +2 {splice_file} | cut -f{column} | sort -u > {splice_ids}"
    cmd2 = f"awk '$3==\"{feature_type}\"' {gtf_file} | sed -E 's/.*{id_field} \"([^\"]+)\".*/\\1/' | sort -u > {gtf_ids}"
    cmd3 = f"comm -13 {gtf_ids} {splice_ids} | wc -l"
    
    subprocess.run(cmd1, shell=True, check=True)
    subprocess.run(cmd2, shell=True, check=True)
    result = subprocess.run(cmd3, shell=True, capture_output=True, text=True)
    
    mismatches = int(result.stdout.strip())
    return (mismatches == 0, mismatches)


def generate_manifest(splice_file: str, gtf_file: str, output_file: str = None) -> Dict:
    """
    Generate comprehensive manifest statistics.
    
    Args:
        splice_file: Path to splice sites TSV file
        gtf_file: Path to source GTF file
        output_file: Optional path to save markdown manifest
    
    Returns:
        Dictionary containing all statistics
    """
    print("Analyzing splice site annotation file...")
    
    # Count splice site statistics
    splice_genes = count_unique_values(splice_file, column=7)
    splice_transcripts = count_unique_values(splice_file, column=8)
    total_splice_sites = count_total_lines(splice_file)
    
    print("Analyzing GTF file...")
    
    # Count GTF statistics
    gtf_genes = count_gtf_features(gtf_file, "gene", "gene_id")
    gtf_transcripts = count_gtf_features(gtf_file, "transcript", "transcript_id")
    
    print("Verifying subset relationships...")
    
    # Verify subset relationships
    genes_subset, genes_mismatch = verify_subset(splice_file, gtf_file, 7, "gene", "gene_id")
    trans_subset, trans_mismatch = verify_subset(splice_file, gtf_file, 8, "transcript", "transcript_id")
    
    # Calculate percentages
    gene_coverage = (splice_genes / gtf_genes * 100) if gtf_genes > 0 else 0
    transcript_coverage = (splice_transcripts / gtf_transcripts * 100) if gtf_transcripts > 0 else 0
    
    stats = {
        'gtf_genes': gtf_genes,
        'gtf_transcripts': gtf_transcripts,
        'splice_genes': splice_genes,
        'splice_transcripts': splice_transcripts,
        'total_splice_sites': total_splice_sites,
        'gene_coverage_pct': gene_coverage,
        'transcript_coverage_pct': transcript_coverage,
        'genes_subset': genes_subset,
        'transcripts_subset': trans_subset,
        'genes_mismatch': genes_mismatch,
        'transcripts_mismatch': trans_mismatch,
        'excluded_genes': gtf_genes - splice_genes,
        'excluded_transcripts': gtf_transcripts - splice_transcripts
    }
    
    # Generate markdown manifest if requested
    if output_file:
        generate_markdown_manifest(stats, splice_file, gtf_file, output_file)
    
    return stats


def generate_markdown_manifest(stats: Dict, splice_file: str, gtf_file: str, output_file: str):
    """Generate a markdown manifest file."""
    splice_name = Path(splice_file).name
    gtf_name = Path(gtf_file).name
    date_str = datetime.now().strftime("%Y-%m-%d")
    
    manifest = f"""# Splice Sites Data Manifest

## Overview
This manifest documents the splice site annotations derived from Ensembl GRCh37.87 GTF file.

**Generated:** {date_str}  
**Source GTF:** {gtf_name}  
**Annotation File:** {splice_name}

## Data Statistics

### Source GTF File ({gtf_name})
- **Total Genes:** {stats['gtf_genes']:,}
- **Total Transcripts:** {stats['gtf_transcripts']:,}

### Splice Sites Annotation ({splice_name})
- **Unique Genes:** {stats['splice_genes']:,} ({stats['gene_coverage_pct']:.1f}% of GTF genes)
- **Unique Transcripts:** {stats['splice_transcripts']:,} ({stats['transcript_coverage_pct']:.1f}% of GTF transcripts)
- **Total Splice Sites:** {stats['total_splice_sites']:,} rows

### Subset Verification
{'✅' if stats['genes_subset'] and stats['transcripts_subset'] else '❌'} **All genes and transcripts in {splice_name} are {"" if stats['genes_subset'] and stats['transcripts_subset'] else "NOT "}confirmed to be present in the source GTF file** ({"strict subset with 0 mismatches" if stats['genes_mismatch'] == 0 and stats['transcripts_mismatch'] == 0 else f"genes: {stats['genes_mismatch']} mismatches, transcripts: {stats['transcripts_mismatch']} mismatches"})

## Interpretation

### Gene Coverage ({stats['gene_coverage_pct']:.1f}%)
The splice sites annotation includes {stats['splice_genes']:,} of the {stats['gtf_genes']:,} genes in the GTF. The ~{stats['excluded_genes']:,} excluded genes likely represent:
- Single-exon genes without splice junctions
- Non-coding RNA genes without canonical splicing
- Pseudogenes or other features lacking introns

### Transcript Coverage ({stats['transcript_coverage_pct']:.1f}%)
The {"higher" if stats['transcript_coverage_pct'] > stats['gene_coverage_pct'] else "lower"} transcript coverage ({stats['splice_transcripts']:,} of {stats['gtf_transcripts']:,}) indicates that:
- Multi-isoform genes are well-represented
- Most protein-coding transcripts with splice junctions are captured
- The ~{stats['excluded_transcripts']:,} missing transcripts may be single-exon isoforms or non-canonical variants

## File Schema

### {splice_name} Columns
1. `chrom` - Chromosome identifier
2. `start` - Splice site start position (0-based)
3. `end` - Splice site end position
4. `position` - Canonical splice position
5. `strand` - Genomic strand (+/-)
6. `site_type` - Splice site type (donor/acceptor)
7. `gene_id` - Ensembl gene identifier (ENSG...)
8. `transcript_id` - Ensembl transcript identifier (ENST...)

## Use Cases
This annotation file is suitable for:
- Training splice site prediction models (e.g., SpliceAI, meta-models)
- Gene-aware cross-validation splits
- Transcript isoform analysis
- Alternative splicing pattern detection
- Splice site density and complexity studies

## Generation Script
The manifest statistics were generated using: `generate_splice_manifest.py`

## Notes
- Genome build: GRCh37 (hg19)
- Ensembl version: 87
- Coordinate system: 0-based for start positions
- All gene/transcript IDs validated against source GTF
"""
    
    with open(output_file, 'w') as f:
        f.write(manifest)
    
    print(f"\n✅ Manifest saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate splice site data manifest with comprehensive statistics"
    )
    parser.add_argument(
        '--splice-file',
        default='~/work/meta-spliceai/data/ensembl/GRCh37/splice_sites_enhanced.tsv',
        help='Path to splice sites TSV file'
    )
    parser.add_argument(
        '--gtf-file',
        default='~/work/meta-spliceai/data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf',
        help='Path to source GTF file'
    )
    parser.add_argument(
        '--output',
        default='~/work/meta-spliceai/data/ensembl/GRCh37/SPLICE_SITES_MANIFEST.md',
        help='Output path for manifest file'
    )
    
    args = parser.parse_args()
    
    # Expand paths
    splice_file = Path(args.splice_file).expanduser()
    gtf_file = Path(args.gtf_file).expanduser()
    output_file = Path(args.output).expanduser()
    
    # Validate input files
    if not splice_file.exists():
        print(f"❌ Error: Splice file not found: {splice_file}")
        return 1
    
    if not gtf_file.exists():
        print(f"❌ Error: GTF file not found: {gtf_file}")
        return 1
    
    # Generate manifest
    stats = generate_manifest(str(splice_file), str(gtf_file), str(output_file))
    
    # Print summary
    print("\n" + "="*60)
    print("SPLICE SITE DATA MANIFEST SUMMARY")
    print("="*60)
    print(f"GTF Genes:              {stats['gtf_genes']:>10,}")
    print(f"GTF Transcripts:        {stats['gtf_transcripts']:>10,}")
    print(f"Splice Site Genes:      {stats['splice_genes']:>10,} ({stats['gene_coverage_pct']:>5.1f}%)")
    print(f"Splice Site Transcripts:{stats['splice_transcripts']:>10,} ({stats['transcript_coverage_pct']:>5.1f}%)")
    print(f"Total Splice Sites:     {stats['total_splice_sites']:>10,}")
    print(f"Genes Subset Valid:     {'✅ Yes' if stats['genes_subset'] else '❌ No'}")
    print(f"Transcripts Subset Valid: {'✅ Yes' if stats['transcripts_subset'] else '❌ No'}")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    exit(main())
