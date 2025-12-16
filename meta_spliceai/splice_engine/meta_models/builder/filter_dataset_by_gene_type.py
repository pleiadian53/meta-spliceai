#!/usr/bin/env python3
"""
Filter training dataset to keep only specified gene types.

This utility script creates a filtered version of a training dataset by keeping only
genes of specified types (e.g., protein_coding and lncRNA) and their associated
position-centric training instances.

This is useful when:
1. Initial dataset includes diverse gene types for exploration
2. Analysis shows certain gene types are unreliable or low-quality
3. Production training requires focus on high-confidence gene annotations
4. Meta-model performance analysis suggests filtering specific gene types

Usage:
    python -m meta_spliceai.splice_engine.meta_models.builder.filter_dataset_by_gene_type \
        input_dataset output_dataset --gene-types protein_coding lncRNA

Author: Splice-Surveyor Development Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from typing import List, Set

def filter_dataset(
    input_dir: str,
    output_dir: str,
    gene_types: List[str],
    verbose: bool = True
) -> None:
    """
    Filter a training dataset to keep only specified gene types.
    
    This function processes all batch files in a training dataset and creates
    a new filtered dataset containing only genes of the specified types.
    The enhanced gene manifest is also filtered and updated accordingly.
    
    Parameters
    ----------
    input_dir : str
        Path to input dataset directory (should contain master/ subdirectory)
    output_dir : str
        Path to output directory for filtered dataset
    gene_types : List[str]
        List of gene types to keep (e.g., ['protein_coding', 'lncRNA'])
    verbose : bool
        Whether to print progress information
        
    Raises
    ------
    FileNotFoundError
        If input directory, master subdirectory, or gene manifest is not found
    ValueError
        If batch files don't contain required gene_id column
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    input_master = input_path / "master"
    output_master = output_path / "master"
    
    # Validate input
    if not input_master.exists():
        raise FileNotFoundError(f"Input master directory not found: {input_master}")
    
    manifest_path = input_master / "gene_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Gene manifest not found: {manifest_path}")
    
    # Create output directory
    output_master.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"🔍 Filtering dataset: {input_dir} -> {output_dir}")
        print(f"Gene types to keep: {', '.join(gene_types)}")
        print("=" * 60)
    
    # Load manifest and determine genes to keep
    if verbose:
        print("📊 Loading gene manifest...")
    
    manifest = pd.read_csv(manifest_path)
    original_gene_count = len(manifest)
    
    # Validate manifest has required columns
    required_cols = ['gene_id', 'gene_type']
    missing_cols = [col for col in required_cols if col not in manifest.columns]
    if missing_cols:
        raise ValueError(f"Gene manifest missing required columns: {missing_cols}")
    
    # Filter manifest
    filtered_manifest = manifest[manifest['gene_type'].isin(gene_types)]
    genes_to_keep = set(filtered_manifest['gene_id'].values)
    
    if len(filtered_manifest) == 0:
        raise ValueError(f"No genes found with specified types: {gene_types}")
    
    if verbose:
        print(f"   Original genes: {original_gene_count:,}")
        print(f"   Filtered genes: {len(filtered_manifest):,}")
        print(f"   Genes removed: {original_gene_count - len(filtered_manifest):,} ({((original_gene_count - len(filtered_manifest))/original_gene_count)*100:.1f}%)")
        
        print(f"\\n   Gene types kept:")
        for gene_type in gene_types:
            count = len(manifest[manifest['gene_type'] == gene_type])
            if count > 0:
                pct = (count / original_gene_count) * 100
                print(f"     {gene_type}: {count:,} genes ({pct:.1f}%)")
        
        # Show what's being removed
        removed_manifest = manifest[~manifest['gene_type'].isin(gene_types)]
        if len(removed_manifest) > 0:
            print(f"\\n   Gene types removed:")
            removed_types = removed_manifest['gene_type'].value_counts()
            for gene_type, count in removed_types.items():
                pct = (count / original_gene_count) * 100
                print(f"     {gene_type}: {count} genes ({pct:.1f}%)")
    
    # Process batch files
    batch_files = list(input_master.glob("batch_*.parquet"))
    if not batch_files:
        raise FileNotFoundError(f"No batch files found in {input_master}")
    
    batch_files.sort()
    total_original_records = 0
    total_filtered_records = 0
    
    if verbose:
        print(f"\\n📦 Processing {len(batch_files)} batch files...")
    
    for i, batch_file in enumerate(batch_files, 1):
        if verbose:
            print(f"   Processing {batch_file.name} ({i}/{len(batch_files)})...")
        
        # Load batch
        batch_df = pd.read_parquet(batch_file)
        original_records = len(batch_df)
        
        # Filter batch
        if 'gene_id' not in batch_df.columns:
            raise ValueError(f"gene_id column not found in {batch_file.name}")
        
        filtered_batch = batch_df[batch_df['gene_id'].isin(genes_to_keep)]
        filtered_records = len(filtered_batch)
        
        # Save filtered batch (only if it has records)
        if filtered_records > 0:
            output_batch_path = output_master / batch_file.name
            filtered_batch.to_parquet(output_batch_path, index=False)
        elif verbose:
            print(f"     ⚠️  Skipping empty batch after filtering")
        
        total_original_records += original_records
        total_filtered_records += filtered_records
        
        if verbose:
            retention_pct = (filtered_records / original_records) * 100 if original_records > 0 else 0
            print(f"     Records: {filtered_records:,} / {original_records:,} ({retention_pct:.1f}% retained)")
    
    # Update manifest file indices and save
    if verbose:
        print(f"\\n📋 Updating gene manifest...")
    
    # Reset global indices for filtered manifest
    filtered_manifest = filtered_manifest.copy()
    filtered_manifest['global_index'] = range(len(filtered_manifest))
    
    # Save filtered manifest
    output_manifest_path = output_master / "gene_manifest.csv"
    filtered_manifest.to_csv(output_manifest_path, index=False)
    
    if verbose:
        print(f"   Saved filtered manifest: {len(filtered_manifest):,} genes")
        print(f"   Manifest saved to: {output_manifest_path}")
    
    # Calculate splice site statistics if available
    splice_stats = ""
    if 'total_splice_sites' in manifest.columns:
        original_sites = manifest['total_splice_sites'].sum()
        filtered_sites = filtered_manifest['total_splice_sites'].sum()
        splice_retention = (filtered_sites / original_sites) * 100 if original_sites > 0 else 0
        splice_stats = f"Splice sites: {filtered_sites:,} / {original_sites:,} ({splice_retention:.1f}% retained)\\n"
    
    # Summary
    if verbose:
        print(f"\\n" + "=" * 60)
        print(f"✅ FILTERING COMPLETE")
        print(f"=" * 60)
        print(f"Input dataset: {input_dir}")
        print(f"Output dataset: {output_dir}")
        print(f"")
        print(f"Genes: {len(filtered_manifest):,} / {original_gene_count:,} ({(len(filtered_manifest)/original_gene_count)*100:.1f}% retained)")
        print(f"Records: {total_filtered_records:,} / {total_original_records:,} ({(total_filtered_records/total_original_records)*100:.1f}% retained)")
        if splice_stats:
            print(splice_stats.strip())
        print(f"Batch files: {len([f for f in output_master.glob('batch_*.parquet')])} created")
        print(f"")
        print(f"Gene types in filtered dataset:")
        for gene_type in gene_types:
            count = len(filtered_manifest[filtered_manifest['gene_type'] == gene_type])
            if count > 0:
                pct = (count / len(filtered_manifest)) * 100
                print(f"  {gene_type}: {count:,} genes ({pct:.1f}%)")

def main():
    """Main entry point for the dataset filtering utility."""
    parser = argparse.ArgumentParser(
        description="Filter training dataset by gene types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter to protein-coding and lncRNA genes
  python -m meta_spliceai.splice_engine.meta_models.builder.filter_dataset_by_gene_type \\
      train_pc_5000_3mers_diverse train_pc_lnc_filtered --gene-types protein_coding lncRNA
  
  # Filter to protein-coding only
  python -m meta_spliceai.splice_engine.meta_models.builder.filter_dataset_by_gene_type \\
      train_pc_7000_3mers_opt train_pc_only --gene-types protein_coding
      
  # Filter with custom gene types
  python -m meta_spliceai.splice_engine.meta_models.builder.filter_dataset_by_gene_type \\
      input_dataset output_dataset --gene-types protein_coding lncRNA processed_pseudogene

Common Gene Types:
  - protein_coding: Standard protein-coding genes (highest reliability)
  - lncRNA: Long non-coding RNAs (regulatory functions)
  - processed_pseudogene: Processed pseudogenes (lower reliability)
  - unprocessed_pseudogene: Unprocessed pseudogenes (variable reliability)
  - transcribed_unprocessed_pseudogene: Active unprocessed pseudogenes
  - IG_V_gene: Immunoglobulin V genes (specialized)
  - TR_V_gene: T-cell receptor V genes (specialized)
        """
    )
    
    parser.add_argument(
        "input_dir",
        help="Input dataset directory (should contain master/ subdirectory with gene_manifest.csv)"
    )
    
    parser.add_argument(
        "output_dir", 
        help="Output directory for filtered dataset (will be created if it doesn't exist)"
    )
    
    parser.add_argument(
        "--gene-types",
        nargs="+",
        default=["protein_coding", "lncRNA"],
        help="Gene types to keep (default: protein_coding lncRNA). "
             "Use space-separated list for multiple types."
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress information (default: True)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress information"
    )
    
    args = parser.parse_args()
    
    verbose = args.verbose and not args.quiet
    
    try:
        filter_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            gene_types=args.gene_types,
            verbose=verbose
        )
        
        if verbose:
            print(f"\\n🎉 Dataset filtering completed successfully!")
            print(f"Filtered dataset available at: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())





