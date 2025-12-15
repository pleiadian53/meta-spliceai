#!/usr/bin/env python
"""Detailed test script for debugging the leakage probe functionality."""
import sys
import json
from pathlib import Path
import pandas as pd
import polars as pl

from meta_spliceai.splice_engine.meta_models.training import leakage_probe as _leak


def debug_leakage_probe(dataset_path, manifest_csv, threshold=0.9, sample=5000):
    """Debug the leakage probe functionality step by step."""
    print(f"\n--- DEBUGGING LEAKAGE PROBE ---")
    print(f"Dataset: {dataset_path}")
    print(f"Feature manifest: {manifest_csv}")
    
    # Step 1: Check if manifest file exists and print its contents
    if Path(manifest_csv).exists():
        print(f"\n1. Feature manifest exists. Contents:")
        manifest_df = pd.read_csv(manifest_csv)
        print(f"   - Shape: {manifest_df.shape}")
        print(f"   - First 5 features: {manifest_df['feature'].tolist()[:5]}")
    else:
        print(f"\n1. ERROR: Feature manifest doesn't exist at {manifest_csv}")
        return
    
    # Step 2: Check if dataset exists and inspect its structure
    dataset_path = Path(dataset_path)
    if dataset_path.is_dir():
        parquet_files = list(dataset_path.glob("*.parquet"))
        print(f"\n2. Dataset directory exists with {len(parquet_files)} parquet files")
        if parquet_files:
            # Load first file to inspect schema
            try:
                first_file = parquet_files[0]
                df_sample = pd.read_parquet(first_file, engine="pyarrow")
                print(f"   - First parquet file: {first_file}")
                print(f"   - Sample shape: {df_sample.shape}")
                print(f"   - Columns: {df_sample.columns.tolist()[:10]}...")
                print(f"   - Has 'splice_type' column: {'splice_type' in df_sample.columns}")
                
                # Check splice_type values
                if 'splice_type' in df_sample.columns:
                    print(f"   - splice_type values: {df_sample['splice_type'].value_counts().to_dict()}")
                    print(f"   - splice_type dtype: {df_sample['splice_type'].dtype}")
            except Exception as e:
                print(f"   - Error reading parquet: {str(e)}")
    else:
        print(f"\n2. ERROR: Dataset directory doesn't exist at {dataset_path}")
        return
    
    # Step 3: Check feature availability in dataset
    print("\n3. Checking feature availability:")
    features = manifest_df['feature'].tolist()
    available_features = [f for f in features if f in df_sample.columns]
    missing_features = [f for f in features if f not in df_sample.columns]
    print(f"   - Available features: {len(available_features)}/{len(features)}")
    if missing_features:
        print(f"   - Missing features: {missing_features}")
    
    # Step 4: Try running leakage_probe directly
    print("\n4. Running leakage_probe directly:")
    try:
        result = _leak.probe_leakage(
            dataset_path=dataset_path,
            manifest_csv=manifest_csv,
            method="pearson",
            threshold=threshold,
            sample=sample,
            return_all=True
        )
        print(f"   - Result shape: {result.shape}")
        print(f"   - Result columns: {result.columns.tolist()}")
        if not result.empty:
            print(f"   - Top 5 correlations:")
            for _, row in result.head(5).iterrows():
                print(f"     * {row['feature']}: {row['correlation']:.4f}")
        else:
            print("   - Result is empty")
    except Exception as e:
        print(f"   - Error running probe_leakage: {str(e)}")
    
    print("\n--- DEBUG COMPLETE ---")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_leakage_probe_debug.py <dataset_path> <manifest_csv> [threshold] [sample]")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    manifest_csv = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
    sample = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
    
    debug_leakage_probe(dataset_path, manifest_csv, threshold, sample)
