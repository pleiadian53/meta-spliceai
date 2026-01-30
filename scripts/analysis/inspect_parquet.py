#!/usr/bin/env python3
"""
Script to inspect the contents of parquet files to understand
available transcript/gene information.
"""
import pandas as pd
import sys

def inspect_parquet(file_path):
    """Inspect the contents of a parquet file."""
    print(f"Inspecting: {file_path}")
    df = pd.read_parquet(file_path)
    
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")
    
    print("\nCheck specific columns:")
    
    if 'gene_id' in df.columns:
        print("\nSample gene_id values:")
        sample = df['gene_id'].head(5).tolist()
        print(sample)
        # Count unique gene IDs
        unique_genes = df['gene_id'].nunique()
        print(f"Number of unique gene_ids: {unique_genes}")
    else:
        print("No gene_id column found")
    
    if 'transcript_id' in df.columns:
        print("\nSample transcript_id values:")
        sample = df['transcript_id'].head(5).tolist()
        print(sample)
        # Check for empty/null transcript IDs
        empty_transcript_ids = df['transcript_id'].isna().sum()
        print(f"Number of empty/null transcript_ids: {empty_transcript_ids} ({empty_transcript_ids/len(df)*100:.2f}%)")
        # Count unique transcript IDs
        unique_transcripts = df['transcript_id'].nunique()
        print(f"Number of unique transcript_ids: {unique_transcripts}")
    else:
        print("No transcript_id column found")
    
    # Check for position and chromosome information
    pos_cols = [col for col in df.columns if 'pos' in col.lower()]
    if pos_cols:
        print("\nPosition-related columns:", pos_cols)
    
    chrom_cols = [col for col in df.columns if 'chrom' in col.lower()]
    if chrom_cols:
        print("\nChromosome-related columns:", chrom_cols)
    
    # Check for splice type column
    if 'splice_type' in df.columns:
        print("\nSplice type distribution:")
        print(df['splice_type'].value_counts())
    
    print("\nFirst few rows sample:")
    print(df.head(2))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_parquet(sys.argv[1])
    else:
        inspect_parquet("/path/to/meta-spliceai/train_pc_1000/master/batch_00001.parquet")
