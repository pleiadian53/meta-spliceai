#!/usr/bin/env python3
"""
Check the structure of the splice dataset to help with top-k implementation.
"""

import pandas as pd
import polars as pl
import os
import sys

def check_parquet_structure(path):
    """Check the column structure of parquet files."""
    if os.path.isdir(path):
        print(f"Reading directory: {path}")
        # Use polars for directory of parquet files
        df = pl.read_parquet(os.path.join(path, "*.parquet"), n_rows=5)
        df_pd = df.to_pandas()
    else:
        print(f"Reading file: {path}")
        df_pd = pd.read_parquet(path, engine="pyarrow")
    
    print("Dataset shape:", df_pd.shape)
    print("\nColumns:")
    for col in df_pd.columns:
        print(f"  - {col} (type: {df_pd[col].dtype})")
    
    print("\nFirst few rows:")
    print(df_pd.head(2).to_string())
    
    # Check if there are transcript identifiers
    transcript_cols = [col for col in df_pd.columns if "transcript" in col.lower()]
    if transcript_cols:
        print("\nTranscript-related columns:")
        for col in transcript_cols:
            print(f"  - {col}")
            unique_values = df_pd[col].unique()
            print(f"    Unique values (up to 5): {unique_values[:5]}")
    else:
        print("\nNo transcript-related columns found")
    
    # Check for gene identifiers
    gene_cols = [col for col in df_pd.columns if "gene" in col.lower()]
    if gene_cols:
        print("\nGene-related columns:")
        for col in gene_cols:
            print(f"  - {col}")
            unique_values = df_pd[col].unique()
            print(f"    Unique values (up to 5): {unique_values[:5]}")
    
    # Check label structure
    if "label" in df_pd.columns:
        print("\nLabel distribution:")
        print(df_pd["label"].value_counts().to_string())
        print("\nUnique labels:", df_pd["label"].unique())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_parquet_structure(sys.argv[1])
    else:
        check_parquet_structure("train_pc_1000/master")
