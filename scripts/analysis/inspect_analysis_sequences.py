#!/usr/bin/env python3
"""
Analysis Sequences Data Analyzer

A reusable script to analyze analysis_sequences_*.tsv files.
Analyzes file structure, columns, null values, and provides summary statistics.

Usage:
    python inspect_analysis_sequences.py [pattern] [--detailed]
    
Examples:
    python inspect_analysis_sequences.py                           # Analyze all analysis_sequences_*.tsv files
    python inspect_analysis_sequences.py "analysis_sequences_*"    # Same as above
    python inspect_analysis_sequences.py "custom_prefix_*"         # Analyze files with custom prefix
    python inspect_analysis_sequences.py --detailed               # Include detailed per-file statistics
"""

import pandas as pd
import os
import glob
import argparse
from pathlib import Path
from collections import defaultdict

def find_analysis_files(pattern="analysis_sequences_*.tsv"):
    """Find all files matching the given pattern."""
    files = glob.glob(pattern)
    return sorted(files)

def analyze_file(filepath, detailed=False):
    """Analyze a single TSV file and return summary statistics."""
    try:
        df = pd.read_csv(filepath, sep='\t', low_memory=False)
        
        # Basic info
        info = {
            'filepath': filepath,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'success': True,
            'error': None
        }
        
        # Null value analysis
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0]
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
                'columns_with_nulls': list(null_cols.index)
            }
        
        # Detailed statistics if requested
        if detailed:
            info['data_types'] = df.dtypes.to_dict()
            info['memory_usage'] = df.memory_usage(deep=True).sum()
            
            # Sample of non-null values for each column
            info['sample_values'] = {}
            for col in df.columns:
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    info['sample_values'][col] = str(non_null_values.iloc[0])
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
            'null_summary': {}
        }

def print_summary(analysis_results, detailed=False):
    """Print a comprehensive summary of the analysis."""
    print("Analysis of Analysis Sequences Files")
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
    all_columns = set()
    for result in successful_files:
        all_columns.update(result['columns'])
    
    print(f"\nOverall Statistics:")
    print(f"  Total rows across all files: {total_rows:,}")
    print(f"  Total unique columns: {len(all_columns)}")
    
    # File-by-file summary
    print(f"\nPer-file Summary:")
    for result in successful_files:
        filename = Path(result['filepath']).name
        rows, cols = result['shape']
        null_cols = len(result['null_columns'])
        print(f"  {filename}:")
        print(f"    Shape: {rows:,} rows × {cols} columns")
        print(f"    Columns with nulls: {null_cols}")
        
        if result['null_columns']:
            print(f"    Null value details:")
            for col, info in result['null_columns'].items():
                print(f"      - {col}: {info['count']:,} nulls ({info['percentage']:.2f}%)")
    
    # Column analysis
    print(f"\nAll Columns ({len(all_columns)} total):")
    for i, col in enumerate(sorted(all_columns), 1):
        print(f"  {i:2d}. {col}")
    
    # Null value pattern analysis
    null_pattern_analysis = defaultdict(list)
    for result in successful_files:
        for col in result['null_summary'].get('columns_with_nulls', []):
            null_pattern_analysis[col].append(Path(result['filepath']).name)
    
    if null_pattern_analysis:
        print(f"\nNull Value Patterns:")
        print(f"Columns that contain null values (and in which files):")
        for col, files in sorted(null_pattern_analysis.items()):
            print(f"  {col}:")
            print(f"    Present in {len(files)}/{len(successful_files)} files")
            if len(files) <= 5:  # Show all files if <= 5
                for file in files:
                    print(f"      - {file}")
            else:  # Show first few and count for many files
                for file in files[:3]:
                    print(f"      - {file}")
                print(f"      ... and {len(files)-3} more files")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze analysis_sequences_*.tsv files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python analyze_sequences.py                           # Analyze all analysis_sequences_*.tsv files
  python analyze_sequences.py "custom_prefix_*"         # Analyze files with custom prefix
  python analyze_sequences.py --detailed               # Include detailed statistics"""
    )
    
    parser.add_argument(
        'pattern', 
        nargs='?', 
        default='analysis_sequences_*.tsv',
        help='File pattern to match (default: analysis_sequences_*.tsv)'
    )
    
    parser.add_argument(
        '--detailed', 
        action='store_true',
        help='Include detailed per-file statistics'
    )
    
    args = parser.parse_args()
    
    # Find files
    files = find_analysis_files(args.pattern)
    
    if not files:
        print(f"No files found matching pattern: {args.pattern}")
        print("\nMake sure you're in the correct directory and the files exist.")
        return
    
    print(f"Found {len(files)} files matching pattern '{args.pattern}'")
    
    # Analyze each file
    results = []
    for filepath in files:
        print(f"Analyzing {filepath}...", end=" ")
        result = analyze_file(filepath, detailed=args.detailed)
        if result['success']:
            print("✓")
        else:
            print("✗")
        results.append(result)
    
    # Print summary
    print_summary(results, detailed=args.detailed)

if __name__ == "__main__":
    main()
