#!/usr/bin/env python3
"""
Comprehensive Analysis of Splice Site Annotation Dataset

This script provides detailed biological and statistical analysis of splice_sites.tsv,
generating insights for computational biologists working with the MetaSpliceAI system.

Documentation: docs/data/splice_sites/splice_site_annotations.md
Output: scripts/data/output/splice_sites_summary.txt

Usage:
    # Full analysis
    python scripts/data/analyze_splice_sites.py

    # Quick mode (skip complex computations)
    python scripts/data/analyze_splice_sites.py --quick

    # Custom input/output
    python scripts/data/analyze_splice_sites.py -i custom_file.tsv -o output_dir/

Author: MetaSpliceAI Analysis Suite
Date: 2025-10-04
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import argparse
import os
import sys
from pathlib import Path


def load_splice_sites(file_path):
    """Load splice sites data with proper column types."""
    print(f"📊 Loading splice sites data from: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        sys.exit(1)
    
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    print(f"✅ Loaded {len(df):,} splice sites")
    print(f"📋 Columns: {', '.join(df.columns)}")
    return df


def analyze_chromosome_distribution(df):
    """Analyze splice site distribution across chromosomes."""
    print("\n" + "="*60)
    print("📍 CHROMOSOME DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Count splice sites per chromosome
    chrom_counts = df['chrom'].value_counts().sort_index()
    chrom_counts_sorted = df['chrom'].value_counts()
    
    print(f"\n🧬 Splice Sites by Chromosome (Sorted by Count):")
    print("-" * 50)
    for chrom, count in chrom_counts_sorted.head(15).items():
        percentage = (count / len(df)) * 100
        print(f"Chr {chrom:>2}: {count:>8,} sites ({percentage:5.2f}%)")
    
    # Chromosome-level statistics
    print(f"\n📈 Chromosome Statistics:")
    print(f"   • Total chromosomes: {len(chrom_counts)}")
    print(f"   • Highest: Chr {chrom_counts_sorted.index[0]} ({chrom_counts_sorted.iloc[0]:,} sites)")
    print(f"   • Lowest: Chr {chrom_counts_sorted.index[-1]} ({chrom_counts_sorted.iloc[-1]:,} sites)")
    print(f"   • Average per chromosome: {chrom_counts.mean():.0f} sites")
    
    return chrom_counts_sorted


def analyze_gene_characteristics(df):
    """Analyze gene-level characteristics and splice site patterns."""
    print("\n" + "="*60)
    print("🧬 GENE-LEVEL ANALYSIS")
    print("="*60)
    
    # Count splice sites per gene
    gene_splice_counts = df['gene_id'].value_counts()
    
    print(f"\n🏆 Top 20 Genes by Splice Site Count:")
    print("-" * 60)
    print("Rank | Gene ID           | Splice Sites | Transcripts")
    print("-" * 60)
    
    for i, (gene_id, count) in enumerate(gene_splice_counts.head(20).items(), 1):
        transcript_count = len(df[df['gene_id'] == gene_id]['transcript_id'].unique())
        print(f"{i:>4} | {gene_id:<17} | {count:>11,} | {transcript_count:>10}")
    
    # Gene statistics
    print(f"\n📊 Gene-Level Statistics:")
    print(f"   • Total unique genes: {len(gene_splice_counts):,}")
    print(f"   • Average splice sites per gene: {gene_splice_counts.mean():.1f}")
    print(f"   • Median splice sites per gene: {gene_splice_counts.median():.1f}")
    print(f"   • Max splice sites in one gene: {gene_splice_counts.max():,}")
    print(f"   • Min splice sites in one gene: {gene_splice_counts.min()}")
    
    # Distribution of splice sites per gene
    splice_site_bins = [1, 5, 10, 25, 50, 100, 250, 500, 1000, float('inf')]
    splice_site_labels = ['1', '2-5', '6-10', '11-25', '26-50', '51-100', '101-250', '251-500', '501-1000', '>1000']
    
    print(f"\n📈 Distribution of Splice Sites per Gene:")
    print("-" * 40)
    for i in range(len(splice_site_bins)-1):
        if i == 0:
            mask = gene_splice_counts == splice_site_bins[i]
        else:
            mask = (gene_splice_counts > splice_site_bins[i]) & (gene_splice_counts <= splice_site_bins[i+1])
        
        count = mask.sum()
        percentage = (count / len(gene_splice_counts)) * 100
        print(f"{splice_site_labels[i]:>8} sites: {count:>6,} genes ({percentage:5.1f}%)")
    
    return gene_splice_counts


def analyze_transcript_isoforms(df):
    """Analyze transcript isoforms and alternative splicing patterns."""
    print("\n" + "="*60)
    print("📜 TRANSCRIPT & ISOFORM ANALYSIS")
    print("="*60)
    
    # Count transcripts per gene
    gene_transcript_counts = df.groupby('gene_id')['transcript_id'].nunique().sort_values(ascending=False)
    
    print(f"\n🏆 Top 20 Genes by Transcript Count (Alternative Splicing):")
    print("-" * 70)
    print("Rank | Gene ID           | Transcripts | Total Splice Sites")
    print("-" * 70)
    
    for i, (gene_id, transcript_count) in enumerate(gene_transcript_counts.head(20).items(), 1):
        splice_site_count = len(df[df['gene_id'] == gene_id])
        print(f"{i:>4} | {gene_id:<17} | {transcript_count:>10} | {splice_site_count:>17,}")
    
    # Transcript statistics
    total_transcripts = df['transcript_id'].nunique()
    total_genes = df['gene_id'].nunique()
    avg_transcripts_per_gene = gene_transcript_counts.mean()
    
    print(f"\n📊 Transcript Statistics:")
    print(f"   • Total unique transcripts: {total_transcripts:,}")
    print(f"   • Total unique genes: {total_genes:,}")
    print(f"   • Average transcripts per gene: {avg_transcripts_per_gene:.2f}")
    print(f"   • Median transcripts per gene: {gene_transcript_counts.median():.1f}")
    print(f"   • Max transcripts for one gene: {gene_transcript_counts.max()}")
    
    # Distribution of transcript counts per gene
    transcript_bins = [1, 2, 3, 5, 10, 20, 50, 100, float('inf')]
    transcript_labels = ['1', '2', '3', '4-5', '6-10', '11-20', '21-50', '51-100', '>100']
    
    print(f"\n📈 Distribution of Transcripts per Gene:")
    print("-" * 45)
    for i in range(len(transcript_bins)-1):
        if i < 2:  # For 1 and 2 transcripts, exact match
            mask = gene_transcript_counts == transcript_bins[i]
        else:
            mask = (gene_transcript_counts > transcript_bins[i-1]) & (gene_transcript_counts <= transcript_bins[i])
        
        count = mask.sum()
        percentage = (count / len(gene_transcript_counts)) * 100
        print(f"{transcript_labels[i]:>8} transcripts: {count:>6,} genes ({percentage:5.1f}%)")
    
    return gene_transcript_counts


def analyze_splice_site_types(df):
    """Analyze splice site type distribution and patterns."""
    print("\n" + "="*60)
    print("🔀 SPLICE SITE TYPE ANALYSIS")
    print("="*60)
    
    # Basic site type distribution
    site_type_counts = df['site_type'].value_counts()
    print(f"\n📊 Splice Site Type Distribution:")
    print("-" * 35)
    for site_type, count in site_type_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{site_type.capitalize():>8} sites: {count:>9,} ({percentage:5.1f}%)")
    
    # Strand distribution
    strand_counts = df['strand'].value_counts()
    print(f"\n🧭 Strand Distribution:")
    print("-" * 30)
    for strand, count in strand_counts.items():
        percentage = (count / len(df)) * 100
        print(f"Strand {strand}: {count:>9,} ({percentage:5.1f}%)")
    
    # Combined analysis: site type by strand
    print(f"\n🔄 Site Type by Strand:")
    print("-" * 40)
    cross_tab = pd.crosstab(df['site_type'], df['strand'], margins=True)
    print(cross_tab)
    
    return site_type_counts, strand_counts


def analyze_positional_patterns(df):
    """Analyze positional patterns and genomic distribution."""
    print("\n" + "="*60)
    print("📍 POSITIONAL PATTERN ANALYSIS")
    print("="*60)
    
    # Chromosome size analysis (approximate based on max positions)
    chrom_max_pos = df.groupby('chrom')['position'].max().sort_values(ascending=False)
    
    print(f"\n📏 Approximate Chromosome Sizes (based on max splice site position):")
    print("-" * 60)
    print("Chromosome | Max Position  | Splice Sites | Density (sites/Mb)")
    print("-" * 60)
    
    chrom_counts = df['chrom'].value_counts()
    for chrom in chrom_max_pos.head(10).index:
        max_pos = chrom_max_pos[chrom]
        site_count = chrom_counts[chrom]
        density = site_count / (max_pos / 1_000_000)  # sites per Mb
        print(f"Chr {chrom:>2}     | {max_pos:>12,} | {site_count:>11,} | {density:>15.1f}")
    
    return chrom_max_pos


def analyze_gene_complexity(df):
    """Analyze gene complexity metrics."""
    print("\n" + "="*60)
    print("🧮 GENE COMPLEXITY ANALYSIS")
    print("="*60)
    
    # Create gene-level summary statistics
    gene_stats = df.groupby('gene_id').agg({
        'transcript_id': 'nunique',
        'position': 'count',
        'site_type': lambda x: len(set(x)),
        'chrom': 'first'
    }).rename(columns={
        'transcript_id': 'transcript_count',
        'position': 'splice_site_count',
        'site_type': 'site_type_diversity',
        'chrom': 'chromosome'
    })
    
    # Calculate complexity score (transcripts * splice sites)
    gene_stats['complexity_score'] = gene_stats['transcript_count'] * gene_stats['splice_site_count']
    
    # Sort by complexity
    complex_genes = gene_stats.sort_values('complexity_score', ascending=False)
    
    print(f"\n🏆 Top 15 Most Complex Genes (Transcripts × Splice Sites):")
    print("-" * 80)
    print("Rank | Gene ID           | Transcripts | Splice Sites | Complexity Score")
    print("-" * 80)
    
    for i, (gene_id, row) in enumerate(complex_genes.head(15).iterrows(), 1):
        print(f"{i:>4} | {gene_id:<17} | {row['transcript_count']:>10} | {row['splice_site_count']:>11,} | {row['complexity_score']:>15,}")
    
    # Complexity statistics
    print(f"\n📊 Gene Complexity Statistics:")
    print(f"   • Average complexity score: {gene_stats['complexity_score'].mean():.1f}")
    print(f"   • Median complexity score: {gene_stats['complexity_score'].median():.1f}")
    print(f"   • Max complexity score: {gene_stats['complexity_score'].max():,}")
    
    return gene_stats


def generate_summary_report(df, output_dir):
    """Generate a comprehensive summary report."""
    print("\n" + "="*60)
    print("📝 GENERATING SUMMARY REPORT")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic statistics
    total_sites = len(df)
    unique_genes = df['gene_id'].nunique()
    unique_transcripts = df['transcript_id'].nunique()
    unique_chromosomes = df['chrom'].nunique()
    
    # Generate summary file
    summary_file = os.path.join(output_dir, 'splice_sites_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("SPLICE SITES DATASET SUMMARY REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: 2025-10-04\n")
        f.write(f"Data source: splice_sites.tsv\n\n")
        
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total splice sites:      {total_sites:,}\n")
        f.write(f"Unique genes:           {unique_genes:,}\n")
        f.write(f"Unique transcripts:     {unique_transcripts:,}\n")
        f.write(f"Unique chromosomes:     {unique_chromosomes}\n")
        f.write(f"Avg transcripts/gene:   {unique_transcripts/unique_genes:.2f}\n")
        f.write(f"Avg splice sites/gene:  {total_sites/unique_genes:.1f}\n\n")
        
        # Top genes by splice sites
        f.write("TOP 10 GENES BY SPLICE SITE COUNT\n")
        f.write("-" * 40 + "\n")
        gene_counts = df['gene_id'].value_counts().head(10)
        for gene_id, count in gene_counts.items():
            transcripts = df[df['gene_id'] == gene_id]['transcript_id'].nunique()
            f.write(f"{gene_id}: {count:,} sites, {transcripts} transcripts\n")
        
        f.write(f"\nReport saved to: {summary_file}\n")
    
    print(f"✅ Summary report saved to: {summary_file}")
    return summary_file


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Comprehensive splice site dataset analysis')
    parser.add_argument('--input', '-i', 
                       default='/Users/pleiadian53/work/splice-surveyor/data/ensembl/splice_sites.tsv',
                       help='Path to splice_sites.tsv file')
    parser.add_argument('--output', '-o',
                       default='/Users/pleiadian53/work/splice-surveyor/scripts/data/output',
                       help='Output directory for reports and analysis')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick analysis (skip detailed computations)')
    
    args = parser.parse_args()
    
    print("🧬 SPLICE SITE DATASET COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    print("MetaSpliceAI Computational Biology Analysis Suite")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print("=" * 60)
    
    # Load data
    df = load_splice_sites(args.input)
    
    # Run analyses
    try:
        # 1. Chromosome distribution
        chrom_analysis = analyze_chromosome_distribution(df)
        
        # 2. Gene characteristics
        gene_analysis = analyze_gene_characteristics(df)
        
        # 3. Transcript isoforms
        transcript_analysis = analyze_transcript_isoforms(df)
        
        # 4. Splice site types
        site_type_analysis = analyze_splice_site_types(df)
        
        if not args.quick:
            # 5. Positional patterns
            positional_analysis = analyze_positional_patterns(df)
            
            # 6. Gene complexity
            complexity_analysis = analyze_gene_complexity(df)
        
        # 7. Generate summary report
        summary_file = generate_summary_report(df, args.output)
        
        print("\n" + "="*60)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📊 Analyzed {len(df):,} splice sites")
        print(f"🧬 Covering {df['gene_id'].nunique():,} genes")
        print(f"📜 Across {df['transcript_id'].nunique():,} transcripts")
        print(f"📄 Summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()