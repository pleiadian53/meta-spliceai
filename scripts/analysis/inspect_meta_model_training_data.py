#!/usr/bin/env python3
"""
Parquet Data Analyzer

A reusable script to analyze parquet files, especially those containing feature-rich datasets
for machine learning models. Analyzes file structure, columns, null values, data types,
and provides detailed summary statistics.

Usage:
    python inspect_meta_model_training_data.py [pattern] [--detailed] [--show-samples]
    
Examples:
    python inspect_meta_model_training_data.py                                    # Analyze all *.parquet files
    python inspect_meta_model_training_data.py "training_dataset_*.parquet"       # Analyze training datasets
    python inspect_meta_model_training_data.py --detailed                        # Include detailed statistics
    python inspect_meta_model_training_data.py --show-samples                    # Show sample values
"""

import pandas as pd
import os
import glob
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

def find_parquet_files(pattern="*.parquet"):
    """Find all parquet files matching the given pattern."""
    files = glob.glob(pattern)
    return sorted(files)

def categorize_columns(columns):
    """Categorize columns into different feature types."""
    categories = {
        'core_features': [],
        'sequence_features': [],
        'genomic_features': [],
        'other_features': []
    }
    
    # Core splice prediction features
    core_keywords = [
        'score', 'probability', 'donor', 'acceptor', 'splice', 'position', 
        'context', 'neither', 'pred_type', 'signal', 'peak', 'surge', 'diff'
    ]
    
    # Sequence composition features
    sequence_keywords = ['6mer_', 'gc_content', 'sequence_length', 'sequence_complexity']
    
    # Genomic annotation features
    genomic_keywords = [
        'gene_', 'transcript_', 'exon', 'intron', 'tx_', 'chrom', 'strand',
        'window_', 'overlap', 'absolute_position', 'distance_'
    ]
    
    for col in columns:
        col_lower = col.lower()
        
        if any(keyword in col_lower for keyword in sequence_keywords):
            categories['sequence_features'].append(col)
        elif any(keyword in col_lower for keyword in genomic_keywords):
            categories['genomic_features'].append(col)
        elif any(keyword in col_lower for keyword in core_keywords):
            categories['core_features'].append(col)
        else:
            categories['other_features'].append(col)
    
    return categories

def analyze_parquet_file(filepath, detailed=False, show_samples=False):
    """Analyze a single parquet file and return comprehensive statistics."""
    try:
        df = pd.read_parquet(filepath)
        
        # Basic info
        info = {
            'filepath': filepath,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'success': True,
            'error': None
        }
        
        # Memory usage
        memory_usage = df.memory_usage(deep=True).sum()
        info['memory_usage_mb'] = memory_usage / (1024 * 1024)
        
        # Data types analysis
        dtypes = df.dtypes
        info['data_types'] = {
            'numeric': len(dtypes[dtypes.isin(['int64', 'float64', 'int32', 'float32'])]),
            'object': len(dtypes[dtypes == 'object']),
            'other': len(dtypes[~dtypes.isin(['int64', 'float64', 'int32', 'float32', 'object'])])
        }
        
        # Column categorization
        info['column_categories'] = categorize_columns(df.columns.tolist())
        
        # Null value analysis
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0].sort_values(ascending=False)
        info['null_columns'] = {}
        info['null_summary'] = {}
        
        if len(null_cols) > 0:
            for col, count in null_cols.items():
                percentage = (count / len(df)) * 100
                info['null_columns'][col] = {
                    'count': count,
                    'percentage': percentage
                }
            info['null_summary'] = {
                'total_columns_with_nulls': len(null_cols),
                'columns_with_nulls': list(null_cols.index),
                'total_null_values': null_cols.sum(),
                'columns_completely_null': len(null_cols[null_cols == len(df)])
            }
        else:
            info['null_summary'] = {
                'total_columns_with_nulls': 0,
                'columns_with_nulls': [],
                'total_null_values': 0,
                'columns_completely_null': 0
            }
        
        # Detailed statistics if requested
        if detailed:
            info['dtypes_detailed'] = df.dtypes.astype(str).to_dict()
            
            # Numeric column statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                info['numeric_stats'] = {}
                for col in numeric_cols:
                    series = df[col].dropna()
                    if len(series) > 0:
                        info['numeric_stats'][col] = {
                            'min': float(series.min()),
                            'max': float(series.max()),
                            'mean': float(series.mean()),
                            'std': float(series.std()) if len(series) > 1 else 0.0,
                            'non_null_count': len(series)
                        }
            
            # String/object column statistics
            object_cols = df.select_dtypes(include=['object']).columns
            if len(object_cols) > 0:
                info['object_stats'] = {}
                for col in object_cols:
                    series = df[col].dropna()
                    info['object_stats'][col] = {
                        'unique_values': series.nunique(),
                        'non_null_count': len(series),
                        'most_common': series.value_counts().head(3).to_dict() if len(series) > 0 else {}
                    }
        
        # Sample values if requested
        if show_samples:
            info['sample_values'] = {}
            for col in df.columns:
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    if df[col].dtype == 'object':
                        info['sample_values'][col] = str(non_null_values.iloc[0])
                    else:
                        info['sample_values'][col] = float(non_null_values.iloc[0])
                else:
                    info['sample_values'][col] = "All null"
        
        return info
        
    except Exception as e:
        return {
            'filepath': filepath,
            'success': False,
            'error': str(e),
            'shape': None,
            'columns': [],
            'null_columns': {},
            'null_summary': {},
            'column_categories': {}
        }

def print_detailed_summary(analysis_results, detailed=False, show_samples=False):
    """Print a comprehensive summary of the parquet file analysis."""
    print("Analysis of Parquet Files")
    print("=" * 50)
    
    successful_files = [r for r in analysis_results if r['success']]
    failed_files = [r for r in analysis_results if not r['success']]
    
    print(f"\nFound {len(analysis_results)} files:")
    print(f"  Successfully analyzed: {len(successful_files)}")
    print(f"  Failed to analyze: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for result in failed_files:
            print(f"  - {result['filepath']}: {result['error']}")
    
    if not successful_files:
        print("\nNo files were successfully analyzed.")
        return
    
    # Overall statistics
    total_rows = sum(r['shape'][0] for r in successful_files)
    total_memory = sum(r.get('memory_usage_mb', 0) for r in successful_files)
    all_columns = set()
    for result in successful_files:
        all_columns.update(result['columns'])
    
    print(f"\nOverall Statistics:")
    print(f"  Total rows across all files: {total_rows:,}")
    print(f"  Total unique columns: {len(all_columns)}")
    print(f"  Total memory usage: {total_memory:.2f} MB")
    
    # File-by-file summary
    print(f"\nPer-file Summary:")
    for result in successful_files:
        filename = Path(result['filepath']).name
        rows, cols = result['shape']
        null_cols = len(result['null_columns'])
        memory_mb = result.get('memory_usage_mb', 0)
        
        print(f"  {filename}:")
        print(f"    Shape: {rows:,} rows × {cols:,} columns")
        print(f"    Memory usage: {memory_mb:.2f} MB")
        print(f"    Columns with nulls: {null_cols}")
        
        # Data type breakdown
        if 'data_types' in result:
            dt = result['data_types']
            print(f"    Data types: {dt['numeric']} numeric, {dt['object']} object, {dt['other']} other")
        
        # Column category breakdown
        if 'column_categories' in result:
            cats = result['column_categories']
            print(f"    Feature categories:")
            print(f"      Core features: {len(cats['core_features'])}")
            print(f"      Sequence features: {len(cats['sequence_features'])}")
            print(f"      Genomic features: {len(cats['genomic_features'])}")
            print(f"      Other features: {len(cats['other_features'])}")
        
        # Top null columns
        if result['null_columns']:
            print(f"    Top null value columns:")
            sorted_nulls = sorted(result['null_columns'].items(), 
                                key=lambda x: x[1]['percentage'], reverse=True)
            for col, info in sorted_nulls[:5]:  # Show top 5
                print(f"      - {col}: {info['count']:,} nulls ({info['percentage']:.2f}%)")
            if len(sorted_nulls) > 5:
                print(f"      ... and {len(sorted_nulls) - 5} more columns with nulls")
    
    # Combined column analysis for single file or when files have similar structure
    if len(successful_files) == 1:
        result = successful_files[0]
        
        print(f"\nColumn Analysis:")
        categories = result.get('column_categories', {})
        
        for cat_name, columns in categories.items():
            if columns:
                print(f"\n{cat_name.replace('_', ' ').title()} ({len(columns)} columns):")
                for i, col in enumerate(sorted(columns), 1):
                    print(f"  {i:3d}. {col}")
        
        # Detailed null value analysis
        if result['null_columns']:
            print(f"\nDetailed Null Value Analysis:")
            print(f"Columns with null values ({len(result['null_columns'])} out of {result['shape'][1]} columns):")
            print("-" * 70)
            
            sorted_nulls = sorted(result['null_columns'].items(), 
                                key=lambda x: x[1]['percentage'], reverse=True)
            for col, info in sorted_nulls:
                print(f"{col:<40} {info['count']:>8,} nulls ({info['percentage']:>6.2f}%)")
        
        # Sample values if requested
        if show_samples and 'sample_values' in result:
            print(f"\nSample Values (first non-null value per column):")
            print("-" * 50)
            for col, sample in sorted(result['sample_values'].items()):
                sample_str = str(sample)[:50] + "..." if len(str(sample)) > 50 else str(sample)
                print(f"{col:<30} {sample_str}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze parquet files for ML datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python analyze_parquet.py                                    # Analyze all *.parquet files
  python analyze_parquet.py "training_dataset_*.parquet"       # Analyze training datasets
  python analyze_parquet.py --detailed                        # Include detailed statistics
  python analyze_parquet.py --show-samples                    # Show sample values"""
    )
    
    parser.add_argument(
        'pattern', 
        nargs='?', 
        default='*.parquet',
        help='File pattern to match (default: *.parquet)'
    )
    
    parser.add_argument(
        '--detailed', 
        action='store_true',
        help='Include detailed per-column statistics'
    )
    
    parser.add_argument(
        '--show-samples', 
        action='store_true',
        help='Show sample values for each column'
    )
    
    args = parser.parse_args()
    
    # Find files
    files = find_parquet_files(args.pattern)
    
    if not files:
        print(f"No parquet files found matching pattern: {args.pattern}")
        print("\nMake sure you're in the correct directory and the files exist.")
        return
    
    print(f"Found {len(files)} files matching pattern '{args.pattern}'")
    
    # Analyze each file
    results = []
    for filepath in files:
        print(f"Analyzing {filepath}...", end=" ")
        result = analyze_parquet_file(filepath, detailed=args.detailed, show_samples=args.show_samples)
        if result['success']:
            print("✓")
        else:
            print("✗")
        results.append(result)
    
    # Print summary
    print_detailed_summary(results, detailed=args.detailed, show_samples=args.show_samples)

if __name__ == "__main__":
    main()
