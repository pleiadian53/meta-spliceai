#!/usr/bin/env python3
"""
Script to analyze splice position datasets for null value patterns.
Specifically analyzes the structure of donor and acceptor features
across different prediction types.
"""

import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Tuple
import sys

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load the splice positions dataset."""
    try:
        return pd.read_csv(filepath, sep='\t')
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        sys.exit(1)

def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Extract donor and acceptor column names."""
    donor_cols = [col for col in df.columns if col.startswith('donor_') 
                and col not in ['donor_acceptor_peak_ratio']]
    acceptor_cols = [col for col in df.columns if col.startswith('acceptor_')]
    return donor_cols, acceptor_cols

def analyze_pred_type(df: pd.DataFrame, pred_type: str) -> Dict:
    """Analyze null patterns for a specific prediction type."""
    type_df = df[df['pred_type'] == pred_type]
    if len(type_df) == 0:
        return None

    results = {
        'total_rows': len(type_df),
        'splice_type_counts': type_df['splice_type'].value_counts().to_dict(),
        'donor_rows': {},
        'acceptor_rows': {}
    }

    donor_cols, acceptor_cols = get_feature_columns(df)

    # Analyze donor rows
    donor_rows = type_df[type_df['splice_type'] == 'donor']
    results['donor_rows'] = {
        'count': len(donor_rows),
        'donor_columns_null_counts': {
            col: donor_rows[col].isnull().sum() for col in donor_cols
        },
        'acceptor_columns_null_counts': {
            col: donor_rows[col].isnull().sum() for col in acceptor_cols
        }
    }

    # Analyze acceptor rows
    acceptor_rows = type_df[type_df['splice_type'] == 'acceptor']
    results['acceptor_rows'] = {
        'count': len(acceptor_rows),
        'donor_columns_null_counts': {
            col: acceptor_rows[col].isnull().sum() for col in donor_cols
        },
        'acceptor_columns_null_counts': {
            col: acceptor_rows[col].isnull().sum() for col in acceptor_cols
        }
    }

    return results

def print_analysis_results(results: Dict, pred_type: str):
    """Print analysis results in a formatted way."""
    if not results:
        print(f"No rows found for prediction type: {pred_type}")
        return

    print(f"\n=== Analysis Results for {pred_type} ===")
    print(f"Total rows: {results['total_rows']}")
    print("\nSplice type distribution:")
    for splice_type, count in results['splice_type_counts'].items():
        print(f"- {splice_type}: {count}")

    # Print donor rows analysis
    print(f"\nDonor rows analysis (total: {results['donor_rows']['count']})")
    print("\nDonor columns in donor rows:")
    for col, nulls in results['donor_rows']['donor_columns_null_counts'].items():
        print(f"- {col}: {nulls} nulls out of {results['donor_rows']['count']} rows")
    print("\nAcceptor columns in donor rows:")
    for col, nulls in results['donor_rows']['acceptor_columns_null_counts'].items():
        print(f"- {col}: {nulls} nulls out of {results['donor_rows']['count']} rows")

    # Print acceptor rows analysis
    print(f"\nAcceptor rows analysis (total: {results['acceptor_rows']['count']})")
    print("\nDonor columns in acceptor rows:")
    for col, nulls in results['acceptor_rows']['donor_columns_null_counts'].items():
        print(f"- {col}: {nulls} nulls out of {results['acceptor_rows']['count']} rows")
    print("\nAcceptor columns in acceptor rows:")
    for col, nulls in results['acceptor_rows']['acceptor_columns_null_counts'].items():
        print(f"- {col}: {nulls} nulls out of {results['acceptor_rows']['count']} rows")

def main():
    parser = argparse.ArgumentParser(description='Analyze splice positions dataset for null patterns')
    parser.add_argument('filepath', help='Path to the splice positions TSV file')
    parser.add_argument('--pred-type', default='TP', 
                    help='Prediction type to analyze (default: TP)')
    args = parser.parse_args()

    # Load and analyze the dataset
    df = load_dataset(args.filepath)
    results = analyze_pred_type(df, args.pred_type)
    print_analysis_results(results, args.pred_type)

if __name__ == '__main__':
    main()
