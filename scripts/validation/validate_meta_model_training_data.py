#!/usr/bin/env python3
"""
Transcript Feature Validation Script

This script validates the relationship between pred_type and transcript-related features
in splice site prediction datasets. It verifies that:
1. Non-splice sites (TN, FP) have missing/empty transcript features
2. Splice sites (TP, FN) have complete transcript features

Usage:
    python validate_meta_model_training_data.py <parquet_file>
    python validate_meta_model_training_data.py <parquet_file> --detailed
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
import warnings

# Suppress the specific deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*DataFrameGroupBy.apply.*")


def identify_transcript_columns(df):
    """Identify transcript-related columns in the dataset."""
    transcript_cols = []
    
    # Primary transcript columns
    primary_cols = ['transcript_id', 'transcript_length', 'transcript_count']
    for col in primary_cols:
        if col in df.columns:
            transcript_cols.append(col)
    
    # Additional transcript-related columns
    additional_patterns = ['tx_', 'transcript_', 'has_tx_info']
    for col in df.columns:
        for pattern in additional_patterns:
            if pattern in col.lower() and col not in transcript_cols:
                transcript_cols.append(col)
                break
    
    # Gene-level features that might be related
    gene_patterns = ['gene_', 'exon_', 'intron_', 'splice_']
    gene_cols = []
    for col in df.columns:
        for pattern in gene_patterns:
            if pattern in col.lower():
                gene_cols.append(col)
                break
    
    return transcript_cols, gene_cols


def is_empty_value(series, col_name):
    """Check if values are considered 'empty' for transcript features."""
    if col_name == 'transcript_id':
        # Empty string or null
        return series.isnull() | (series == '')
    elif col_name in ['transcript_count', 'has_tx_info']:
        # Zero or null
        return series.isnull() | (series == 0)
    elif col_name in ['transcript_length', 'tx_start', 'tx_end']:
        # Should be null for non-splice sites
        return series.isnull()
    else:
        # Default to null check
        return series.isnull()


def analyze_transcript_completeness(df, transcript_cols):
    """Analyze transcript feature completeness by pred_type."""
    results = {}
    
    for col in transcript_cols:
        if col not in df.columns:
            continue
            
        # Calculate empty values using semantic understanding
        empty_mask = is_empty_value(df[col], col)
        
        # Create analysis DataFrame manually to avoid deprecation warning
        analysis_data = []
        for pred_type in df['pred_type'].unique():
            mask = df['pred_type'] == pred_type
            total_count = mask.sum()
            empty_count = empty_mask[mask].sum()
            non_empty_count = total_count - empty_count
            empty_percentage = (empty_count / total_count * 100) if total_count > 0 else 0
            
            analysis_data.append({
                'pred_type': pred_type,
                'total_count': total_count,
                'empty_count': empty_count,
                'non_empty_count': non_empty_count,
                'empty_percentage': round(empty_percentage, 2)
            })
        
        analysis = pd.DataFrame(analysis_data).set_index('pred_type')
        results[col] = analysis
    
    return results


def get_feature_summary(df, transcript_cols):
    """Generate a summary of feature patterns by pred_type."""
    summary = {
        'Non-splice sites (TN, FP)': {},
        'Splice sites (TP, FN)': {}
    }
    
    # Analyze non-splice sites (TN, FP)
    non_splice_mask = df['pred_type'].isin(['TN', 'FP'])
    non_splice_data = df[non_splice_mask]
    
    # Analyze splice sites (TP, FN)
    splice_mask = df['pred_type'].isin(['TP', 'FN'])
    splice_data = df[splice_mask]
    
    for col in transcript_cols:
        if col not in df.columns:
            continue
            
        # Non-splice sites analysis
        if len(non_splice_data) > 0:
            ns_values = non_splice_data[col]
            ns_unique = ns_values.value_counts().head(3)
            ns_nulls = ns_values.isnull().sum()
            ns_total = len(ns_values)
            
            if col == 'transcript_id':
                if (ns_values == '').all():
                    summary['Non-splice sites (TN, FP)'][col] = 'Empty string ""'
                elif ns_nulls == ns_total:
                    summary['Non-splice sites (TN, FP)'][col] = 'NULL (missing)'
                else:
                    summary['Non-splice sites (TN, FP)'][col] = f'Mixed: {dict(ns_unique)}'
            elif col in ['transcript_length', 'tx_start', 'tx_end']:
                if ns_nulls == ns_total:
                    summary['Non-splice sites (TN, FP)'][col] = 'NULL (missing)'
                else:
                    summary['Non-splice sites (TN, FP)'][col] = f'Mixed: {dict(ns_unique)}'
            elif col in ['transcript_count', 'has_tx_info']:
                if (ns_values == 0).all():
                    summary['Non-splice sites (TN, FP)'][col] = '0'
                else:
                    summary['Non-splice sites (TN, FP)'][col] = f'Mixed: {dict(ns_unique)}'
            else:
                summary['Non-splice sites (TN, FP)'][col] = f'Values: {dict(ns_unique)}'
        
        # Splice sites analysis
        if len(splice_data) > 0:
            s_values = splice_data[col]
            s_unique = s_values.value_counts().head(3)
            s_nulls = s_values.isnull().sum()
            s_total = len(s_values)
            
            if col == 'transcript_id':
                if s_nulls == 0 and all(str(v).startswith('ENST') for v in s_values.dropna()):
                    examples = list(s_unique.index[:2])
                    summary['Splice sites (TP, FN)'][col] = f'Valid Ensembl transcript IDs (e.g., "{examples[0]}")'
                else:
                    summary['Splice sites (TP, FN)'][col] = f'Values: {dict(s_unique)}'
            elif col == 'transcript_length':
                if s_nulls == 0 and pd.api.types.is_numeric_dtype(s_values):
                    min_val, max_val = s_values.min(), s_values.max()
                    examples = list(s_unique.index[:2])
                    summary['Splice sites (TP, FN)'][col] = f'Numeric values (e.g., {examples[0]}, {examples[1] if len(examples) > 1 else examples[0]})'
                else:
                    summary['Splice sites (TP, FN)'][col] = f'Values: {dict(s_unique)}'
            elif col == 'transcript_count':
                if s_nulls == 0 and pd.api.types.is_numeric_dtype(s_values):
                    min_val, max_val = s_values.min(), s_values.max()
                    summary['Splice sites (TP, FN)'][col] = f'Positive integers ({int(min_val)}-{int(max_val)})'
                else:
                    summary['Splice sites (TP, FN)'][col] = f'Values: {dict(s_unique)}'
            elif col in ['tx_start', 'tx_end']:
                if s_nulls == 0 and pd.api.types.is_numeric_dtype(s_values):
                    summary['Splice sites (TP, FN)'][col] = 'Genomic coordinates'
                else:
                    summary['Splice sites (TP, FN)'][col] = f'Values: {dict(s_unique)}'
            elif col == 'has_tx_info':
                if (s_values == 1).all():
                    summary['Splice sites (TP, FN)'][col] = '1'
                else:
                    summary['Splice sites (TP, FN)'][col] = f'Values: {dict(s_unique)}'
            else:
                summary['Splice sites (TP, FN)'][col] = f'Values: {dict(s_unique)}'
    
    return summary


def validate_transcript_expectations(df, transcript_cols):
    """Validate that transcript features follow expected patterns."""
    validation_results = {
        'passed': True,
        'issues': [],
        'summary': {}
    }
    
    # Check pred_type distribution
    pred_type_counts = df['pred_type'].value_counts()
    validation_results['summary']['pred_type_distribution'] = pred_type_counts.to_dict()
    
    # Analyze transcript completeness
    transcript_analysis = analyze_transcript_completeness(df, transcript_cols)
    
    for col, analysis in transcript_analysis.items():
        validation_results['summary'][col] = analysis.to_dict()
        
        # Validation rules
        for pred_type in analysis.index:
            empty_pct = analysis.loc[pred_type, 'empty_percentage']
            
            if pred_type in ['TN', 'FP']:  # Non-splice sites
                if empty_pct < 100:
                    validation_results['passed'] = False
                    validation_results['issues'].append(
                        f"{col}: {pred_type} should have 100% empty values but has {empty_pct:.1f}%"
                    )
            elif pred_type in ['TP', 'FN']:  # Splice sites
                if empty_pct > 0:
                    validation_results['passed'] = False
                    validation_results['issues'].append(
                        f"{col}: {pred_type} should have 0% empty values but has {empty_pct:.1f}%"
                    )
    
    return validation_results


def print_feature_summary(df, transcript_cols):
    """Print a summary of feature patterns."""
    print("\n" + "="*60)
    print("TRANSCRIPT FEATURE PATTERNS")
    print("="*60)
    
    feature_summary = get_feature_summary(df, transcript_cols)
    
    for category, features in feature_summary.items():
        if features:  # Only print if there are features
            print(f"\n{category}:")
            for col, pattern in features.items():
                print(f"  • {col}: {pattern}")


def print_detailed_analysis(df, transcript_cols, gene_cols):
    """Print detailed analysis of transcript and gene features."""
    print("\n" + "="*80)
    print("DETAILED TRANSCRIPT FEATURE ANALYSIS")
    print("="*80)
    
    # Basic dataset info
    print(f"\nDataset Overview:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Pred_type distribution:")
    for pred_type, count in df['pred_type'].value_counts().items():
        print(f"    {pred_type}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Transcript columns analysis
    if transcript_cols:
        print(f"\nTranscript-related columns ({len(transcript_cols)}):")
        transcript_results = analyze_transcript_completeness(df, transcript_cols)
        
        for col, analysis in transcript_results.items():
            print(f"\n  {col}:")
            for pred_type in ['TP', 'FN', 'TN', 'FP']:
                if pred_type in analysis.index:
                    row = analysis.loc[pred_type]
                    print(f"    {pred_type}: {row['empty_count']:,} empty / {row['total_count']:,} total ({row['empty_percentage']:.1f}%)")
                    
                    # Show sample values for this pred_type and column
                    sample_vals = df[df['pred_type'] == pred_type][col].value_counts().head(3)
                    if len(sample_vals) > 0:
                        print(f"         Sample values: {dict(sample_vals)}")
    
    # Gene columns analysis (basic null analysis)
    if gene_cols:
        print(f"\nGene-related columns ({len(gene_cols)}) - null analysis:")
        
        for col in gene_cols[:3]:  # Show first 3 gene columns
            if col in df.columns:
                print(f"\n  {col}:")
                # Manual calculation to avoid deprecation warnings
                for pred_type in ['TP', 'FN', 'TN', 'FP']:
                    if pred_type in df['pred_type'].values:
                        subset = df[df['pred_type'] == pred_type][col]
                        total = len(subset)
                        nulls = subset.isnull().sum()
                        null_pct = (nulls / total * 100) if total > 0 else 0
                        print(f"    {pred_type}: {nulls:,} nulls / {total:,} total ({null_pct:.1f}%)")
        
        if len(gene_cols) > 3:
            print(f"\n  ... and {len(gene_cols) - 3} more gene-related columns")


def print_summary_report(validation_results):
    """Print a summary validation report."""
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    if validation_results['passed']:
        print("✅ VALIDATION PASSED")
        print("   All transcript features follow expected patterns:")
        print("   - Non-splice sites (TN, FP) have empty/missing transcript features")
        print("   - Splice sites (TP, FN) have complete transcript features")
    else:
        print("❌ VALIDATION FAILED")
        print("   Issues found:")
        for issue in validation_results['issues']:
            print(f"   - {issue}")
    
    print(f"\nDataset composition:")
    for pred_type, count in validation_results['summary']['pred_type_distribution'].items():
        print(f"  {pred_type}: {count:,}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate transcript features in splice site datasets'
    )
    parser.add_argument('file', help='Path to parquet file to validate')
    parser.add_argument('--detailed', '-d', action='store_true',
                       help='Show detailed analysis of all transcript features')
    
    args = parser.parse_args()
    
    # Check if file exists
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File {args.file} not found")
        sys.exit(1)
    
    try:
        # Load the dataset
        print(f"Loading dataset: {args.file}")
        df = pd.read_parquet(args.file)
        
        # Check required columns
        if 'pred_type' not in df.columns:
            print("Error: 'pred_type' column not found in dataset")
            sys.exit(1)
        
        # Identify transcript and gene columns
        transcript_cols, gene_cols = identify_transcript_columns(df)
        
        if not transcript_cols:
            print("Warning: No transcript-related columns found")
            print("Available columns:", df.columns.tolist())
            sys.exit(1)
        
        print(f"Found {len(transcript_cols)} transcript-related columns:")
        for col in transcript_cols:
            print(f"  - {col}")
        
        # Perform validation
        validation_results = validate_transcript_expectations(df, transcript_cols)
        
        # Print results
        print_summary_report(validation_results)
        
        # Always show feature patterns summary
        print_feature_summary(df, transcript_cols)
        
        if args.detailed:
            print_detailed_analysis(df, transcript_cols, gene_cols)
        
        # Exit with appropriate code
        sys.exit(0 if validation_results['passed'] else 1)
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
