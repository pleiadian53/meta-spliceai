#!/usr/bin/env python3
"""
Extract real error counts from parquet training data for presentation.

Usage:
    python -m meta_spliceai.splice_engine.meta_models.analysis.get_real_error_counts \
      --dataset train_pc_1000/master \
      --sample-size 50000
"""

import argparse
import pandas as pd
from pathlib import Path

def get_real_error_counts(dataset_dir: str, sample_size: int = 50000) -> dict:
    """Extract real error counts from parquet data."""
    
    data_path = Path(dataset_dir)
    parquet_files = list(data_path.glob("batch_*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Read a sample from the first file
    print(f"Reading sample from {parquet_files[0]}")
    df = pd.read_parquet(parquet_files[0])
    
    # Take a sample for faster processing
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Sample size: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Find relevant columns (adapt these based on your actual columns)
    splice_col = None
    donor_col = None
    acceptor_col = None
    
    for col in df.columns:
        if 'splice_type' in col.lower():
            splice_col = col
        elif 'donor' in col.lower() and 'score' in col.lower():
            donor_col = col
        elif 'acceptor' in col.lower() and 'score' in col.lower():
            acceptor_col = col
    
    if not all([splice_col, donor_col, acceptor_col]):
        print("Available columns:")
        for col in df.columns:
            print(f"  - {col}")
        return {"error": "Could not find required columns"}
    
    print(f"Using columns: {splice_col}, {donor_col}, {acceptor_col}")
    
    # Calculate errors
    threshold = 0.5
    
    # Predictions
    pred_donor = df[donor_col] > threshold
    pred_acceptor = df[acceptor_col] > threshold
    pred_neither = (~pred_donor) & (~pred_acceptor)
    
    # True labels
    true_donor = df[splice_col] == 'donor'
    true_acceptor = df[splice_col] == 'acceptor'
    true_neither = df[splice_col] == 'neither'
    
    # Count errors
    errors = {
        'donor_fp': sum(pred_donor & true_neither),
        'donor_fn': sum(true_donor & pred_neither),
        'acceptor_fp': sum(pred_acceptor & true_neither),
        'acceptor_fn': sum(true_acceptor & pred_neither),
        'sample_size': len(df)
    }
    
    return errors

def main():
    parser = argparse.ArgumentParser(description="Get real error counts")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=50000)
    
    args = parser.parse_args()
    
    try:
        errors = get_real_error_counts(args.dataset, args.sample_size)
        
        if 'error' in errors:
            print(f"Error: {errors['error']}")
            return
        
        print(f"\n🎯 Real Error Counts (from {errors['sample_size']} examples):")
        print(f"Donor False Positives: {errors['donor_fp']}")
        print(f"Donor False Negatives: {errors['donor_fn']}")
        print(f"Acceptor False Positives: {errors['acceptor_fp']}")
        print(f"Acceptor False Negatives: {errors['acceptor_fn']}")
        
        # Scale to full dataset if needed
        print(f"\n📊 For presentation slide, you can use these counts:")
        print(f"error_data = {{")
        print(f"    'Donor': {{'False Positives': {errors['donor_fp']}, 'False Negatives': {errors['donor_fn']}}},")
        print(f"    'Acceptor': {{'False Positives': {errors['acceptor_fp']}, 'False Negatives': {errors['acceptor_fn']}}}")
        print(f"}}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 