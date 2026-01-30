#!/usr/bin/env python3
"""
Enhance splice site annotations with gene and transcript features.

This script adds gene and transcript information to splice site annotations
and can perform genomic data analysis on splicing patterns across gene types.

The script is a standalone utility wrapper around the functionality in 
meta_spliceai.splice_engine.meta_models.utils.annotation_utils
"""

import os
import sys
import argparse
import polars as pl
from pathlib import Path

from meta_spliceai.splice_engine.meta_models.utils import (
    enhance_splice_sites_with_features,
    analyze_splicing_patterns
)

def find_project_root():
    """Find the project root directory"""
    # Start from current directory and move up until we find the project root
    current_dir = Path.cwd()
    while current_dir.name:
        if (current_dir / '.git').exists() or (current_dir / 'setup.py').exists() or current_dir.name == 'meta-spliceai':
            return str(current_dir)
        current_dir = current_dir.parent
    
    # Fallback if we can't detect automatically
    # Use a more generic approach that doesn't expose usernames
    home_dir = str(Path.home())
    return os.path.join(home_dir, "work", "meta-spliceai")

def main():
    parser = argparse.ArgumentParser(description='Enhance splice site annotations with gene and transcript features')
    parser.add_argument('--input', '-i', type=str, help='Input splice sites TSV file path')
    parser.add_argument('--output', '-o', type=str, help='Output enhanced splice sites TSV file path')
    parser.add_argument('--gtf', '-g', type=str, help='GTF file path for feature extraction')
    parser.add_argument('--analyze', '-a', action='store_true', help='Analyze splicing patterns after enhancement')
    parser.add_argument('--gene-types', type=str, nargs='+', help='Gene types to analyze (e.g., protein_coding lincRNA)')
    parser.add_argument('--title', type=str, default='Splicing Analysis', help='Title for the analysis')
    
    args = parser.parse_args()
    
    # Get project directory
    project_dir = find_project_root()
    print(f"Project directory: {project_dir}")
    
    # Default file paths based on project structure if not provided
    splice_sites_path = args.input or os.path.join(project_dir, "data", "ensembl", "splice_sites.tsv")
    enhanced_output_path = args.output or os.path.join(project_dir, "data", "ensembl", "splice_sites_enhanced.tsv")
    gtf_path = args.gtf or os.path.join(project_dir, "data", "ensembl", "Homo_sapiens.GRCh38.112.gtf")
    
    # Default paths for feature files
    gene_features_path = os.path.join(project_dir, "data", "ensembl", "spliceai_analysis", "gene_features.tsv")
    transcript_features_path = os.path.join(project_dir, "data", "ensembl", "spliceai_analysis", "transcript_features.tsv")
    
    # Check if input files exist
    if not os.path.exists(splice_sites_path):
        print(f"ERROR: Splice sites file not found: {splice_sites_path}")
        print("Please specify a valid file path with --input")
        sys.exit(1)
        
    if not os.path.exists(gtf_path):
        print(f"ERROR: GTF file not found: {gtf_path}")
        print("Please specify a valid file path with --gtf")
        sys.exit(1)
    
    print(f"Reading splice sites from: {splice_sites_path}")
    print(f"Using GTF file: {gtf_path}")
    print(f"Output will be written to: {enhanced_output_path}")
    
    # Enhance splice sites with gene and transcript information
    enhanced_df = enhance_splice_sites_with_features(
        splice_sites_path=splice_sites_path,
        gene_features_path=gene_features_path,
        transcript_features_path=transcript_features_path,
        gene_types_to_keep=args.gene_types,
        verbose=1
    )
    
    # Save enhanced dataframe if needed
    if enhanced_output_path:
        enhanced_df.write_csv(enhanced_output_path, separator='\t')
        print(f"\nEnhanced splice sites saved to: {enhanced_output_path}")
        
    print(f"Added feature columns: {[col for col in enhanced_df.columns if col not in ['chrom', 'start', 'end', 'strand', 'site_type', 'gene_id', 'transcript_id']]}")
    
    # Analyze splicing patterns if requested
    if args.analyze:
        gene_types = args.gene_types if args.gene_types else None
        analyze_splicing_patterns(enhanced_df, gene_types=gene_types, title=args.title)
        
        # Also analyze protein-coding genes if gene_types wasn't specified
        if not gene_types:
            analyze_splicing_patterns(enhanced_df, gene_types=["protein_coding"], title="Protein-Coding Genes")

if __name__ == "__main__":
    main()
