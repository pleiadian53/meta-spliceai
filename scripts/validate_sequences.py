#!/usr/bin/env python3
"""
Analysis Sequence Validation Tool

This script validates sequence files by checking for duplicate combinations of
gene_id, position, strand, and splice_type, which should uniquely identify a splice site context.

It supports both pandas and polars (faster) for processing large datasets.

Typical usage:
    python validate_sequences.py --input consolidated_sequences.tsv --output report.txt
"""

import pandas as pd
import sys
import time
import argparse
from collections import Counter
from pathlib import Path
from typing import Union, Dict, Any, Tuple

# Optional polars support for better performance
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate sequence files by checking for duplicates and integrity issues"
    )
    parser.add_argument(
        "--input", 
        type=str,
        default="consolidated_sequences.tsv",
        help="Input file path (default: consolidated_sequences.tsv)"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Output file path for detailed report (optional)"
    )
    parser.add_argument(
        "--use-polars", 
        action="store_true",
        help="Use polars instead of pandas for processing (faster for large files)"
    )
    parser.add_argument(
        "--show-duplicates", 
        action="store_true",
        help="Show sample duplicate entries in console output"
    )
    parser.add_argument(
        "--save-duplicates", 
        type=str,
        help="Save all duplicate entries to specified file"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed validation information"
    )
    return parser.parse_args()


def analyze_with_pandas(
    file_path: Union[str, Path],
    verbose: bool = False,
    show_duplicates: bool = False,
    save_duplicates: Union[str, Path, None] = None,
    output_report: Union[str, Path, None] = None
) -> Tuple[bool, Dict[str, Any]]:
    """Analyze sequence file using pandas.
    
    Parameters
    ----------
    file_path: str or Path
        Path to the input file
    verbose: bool
        Whether to show detailed information
    show_duplicates: bool
        Whether to print sample duplicate entries
    save_duplicates: str or Path, optional
        Path to save duplicate entries
    output_report: str or Path, optional
        Path to save validation report
        
    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        Success flag and validation statistics
    """
    print(f"Analyzing file: {file_path}")
    start_time = time.time()
    
    # Read the consolidated dataset
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"Error reading file: {e}")
        return False, {"error": str(e)}
    
    print(f"\nDataset Overview:")
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Define composite key columns
    key_columns = ['gene_id', 'position', 'strand', 'splice_type']
    
    # Verify key columns exist
    missing_cols = [col for col in key_columns if col not in df.columns]
    if missing_cols:
        print(f"\nError: Missing required columns: {', '.join(missing_cols)}")
        return False, {"missing_columns": missing_cols}

    # Check for duplicates
    duplicates = df.duplicated(subset=key_columns, keep=False)
    duplicate_count = duplicates.sum()
    unique_combinations = df.drop_duplicates(subset=key_columns).shape[0]
    
    print(f"\nDuplicate Analysis:")
    print(f"Unique combinations: {unique_combinations:,}")
    print(f"Duplicate rows: {duplicate_count:,}")
    
    # Prepare statistics dict
    stats = {
        "total_rows": len(df),
        "unique_combinations": unique_combinations,
        "duplicate_rows": duplicate_count,
        "processing_time": time.time() - start_time,
        "has_duplicates": duplicate_count > 0
    }
        
    if duplicate_count > 0:
        # Get duplicate entries
        duplicate_df = df[duplicates].copy()
        
        # Group duplicates and count occurrences
        duplicate_groups = duplicate_df.groupby(key_columns).size().reset_index(name='count')
        duplicate_groups = duplicate_groups[duplicate_groups['count'] > 1]
        
        stats["duplicate_groups"] = len(duplicate_groups)
        stats["duplicate_distribution"] = Counter(duplicate_groups['count'])
        
        print("\nDuplicate Statistics:")
        print(f"Number of duplicate groups: {len(duplicate_groups):,}")
        print("\nOccurrence distribution:")
        occurrence_counts = Counter(duplicate_groups['count'])
        for count, freq in sorted(occurrence_counts.items()):
            print(f"  Rows appearing {count} times: {freq:,} groups")
        
        if show_duplicates:
            print("\nSample of duplicates (first 5 groups):")
            for i, (_, group) in enumerate(duplicate_df.groupby(key_columns)):
                if i >= 5:
                    break
                print("\nDuplicate Group:")
                print(group.to_string())
        
        # Analyze patterns in duplicates
        if verbose:
            print("\nPatterns in duplicates:")
            for col in key_columns:
                value_counts = duplicate_df[col].value_counts()
                print(f"\nMost common {col} values in duplicates:")
                print(value_counts.head())
        
        # Save duplicates to file if requested
        if save_duplicates:
            duplicate_df.to_csv(save_duplicates, sep='\t', index=False)
            print(f"\nDetailed duplicate entries saved to: {save_duplicates}")
        
    else:
        print("\n✓ No duplicates found in the dataset!")
        print("All composite key combinations are unique.")
    
    # Additional validation statistics
    if verbose:
        print("\nAdditional Statistics:")
        print("Counts by splice_type:")
        print(df['splice_type'].value_counts())
        print("\nCounts by strand:")
        print(df['strand'].value_counts())
        
        # Sequence length validation if sequence column exists
        if 'sequence' in df.columns:
            df['seq_length'] = df['sequence'].str.len()
            stats["sequence_stats"] = df['seq_length'].describe().to_dict()
            print("\nSequence length statistics:")
            print(df['seq_length'].describe())
    
    # Generate output report if requested
    if output_report:
        with open(output_report, 'w') as f:
            f.write(f"Validation Report for {file_path}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total rows: {len(df):,}\n")
            f.write(f"Unique combinations: {unique_combinations:,}\n")
            f.write(f"Duplicate rows: {duplicate_count:,}\n\n")
            
            if duplicate_count > 0:
                f.write("RESULT: DUPLICATES FOUND - Dataset contains duplicate entries\n")
                f.write(f"Number of duplicate groups: {len(duplicate_groups):,}\n\n")
                f.write("Occurrence distribution:\n")
                for count, freq in sorted(occurrence_counts.items()):
                    f.write(f"  Rows appearing {count} times: {freq:,} groups\n")
            else:
                f.write("RESULT: VALIDATION PASSED - All combinations are unique\n")
                
            print(f"\nReport saved to: {output_report}")
    
    elapsed = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed:.2f} seconds")
    
    # Return success (no duplicates) and statistics
    return not stats["has_duplicates"], stats


def analyze_with_polars(
    file_path: Union[str, Path],
    verbose: bool = False,
    show_duplicates: bool = False,
    save_duplicates: Union[str, Path, None] = None,
    output_report: Union[str, Path, None] = None
) -> Tuple[bool, Dict[str, Any]]:
    """Analyze sequence file using polars (faster for large files).
    
    Parameters
    ----------
    file_path: str or Path
        Path to the input file
    verbose: bool
        Whether to show detailed information
    show_duplicates: bool
        Whether to print sample duplicate entries
    save_duplicates: str or Path, optional
        Path to save duplicate entries
    output_report: str or Path, optional
        Path to save validation report
        
    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        Success flag and validation statistics
    """
    print(f"Analyzing file: {file_path} (using polars)")
    start_time = time.time()
    
    # Read the consolidated dataset
    try:
        df = pl.read_csv(file_path, separator='\t')
    except Exception as e:
        print(f"Error reading file: {e}")
        return False, {"error": str(e)}
    
    print(f"\nDataset Overview:")
    print(f"Total rows: {df.height:,}")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Define composite key columns
    key_columns = ['gene_id', 'position', 'strand', 'splice_type']
    
    # Verify key columns exist
    missing_cols = [col for col in key_columns if col not in df.columns]
    if missing_cols:
        print(f"\nError: Missing required columns: {', '.join(missing_cols)}")
        return False, {"missing_columns": missing_cols}
    
    # Check for duplicates
    rows_before = df.height
    unique_df = df.unique(subset=key_columns)
    unique_combinations = unique_df.height
    duplicate_count = rows_before - unique_combinations
    
    print(f"\nDuplicate Analysis:")
    print(f"Unique combinations: {unique_combinations:,}")
    print(f"Duplicate rows: {duplicate_count:,}")
    
    # Prepare statistics dict
    stats = {
        "total_rows": rows_before,
        "unique_combinations": unique_combinations,
        "duplicate_rows": duplicate_count,
        "processing_time": time.time() - start_time,
        "has_duplicates": duplicate_count > 0
    }
    
    if duplicate_count > 0:
        # Find duplicates - get entries that appear more than once
        counts = (
            df.group_by(key_columns)
            .count()
            .filter(pl.col("count") > 1)
            .sort("count", descending=True)
        )
        
        stats["duplicate_groups"] = counts.height
        
        print("\nDuplicate Statistics:")
        print(f"Number of duplicate groups: {counts.height:,}")
        
        # Calculate occurrence distribution
        occurrence_distribution = counts.group_by("count").count().rename({"count_right": "frequency"})
        stats["duplicate_distribution"] = {row["count"]: row["frequency"] for row in occurrence_distribution.iter_rows(named=True)}
        
        print("\nOccurrence distribution:")
        for row in occurrence_distribution.iter_rows(named=True):
            print(f"  Rows appearing {row['count']} times: {row['frequency']:,} groups")
        
        # Show sample duplicates if requested
        if show_duplicates:
            print("\nTop duplicate groups:")
            print(counts.head(5))
            
            # For each of the top 5 duplicate groups, show the actual rows
            if verbose:
                for i, group_key in enumerate(counts.head(5).iter_rows(named=True)):
                    if i >= 5:
                        break
                        
                    # Create filter expression for this group
                    filter_expr = None
                    for col in key_columns:
                        col_expr = pl.col(col) == group_key[col]
                        filter_expr = col_expr if filter_expr is None else filter_expr & col_expr
                    
                    # Get and print the group rows
                    group_rows = df.filter(filter_expr)
                    print(f"\nDuplicate Group {i+1} ({group_key['count']} occurrences):")
                    print(group_rows)
        
        # Save duplicates to file if requested
        if save_duplicates:
            # To find all duplicate rows, we need to join back to original DataFrame
            # where the count > 1 for the key columns
            df_with_counts = df.join(
                counts.select([*key_columns, "count"]),
                on=key_columns,
                how="inner"
            )
            
            df_with_counts.write_csv(save_duplicates, separator='\t')
            print(f"\nDetailed duplicate entries saved to: {save_duplicates}")
    else:
        print("\n✓ No duplicates found in the dataset!")
        print("All composite key combinations are unique.")
    
    # Additional validation statistics
    if verbose:
        print("\nAdditional Statistics:")
        print("Counts by splice_type:")
        splice_counts = df.group_by("splice_type").count().sort("count", descending=True)
        print(splice_counts)
        
        print("\nCounts by strand:")
        strand_counts = df.group_by("strand").count().sort("count", descending=True)
        print(strand_counts)
        
        # Sequence length validation if sequence column exists
        if "sequence" in df.columns:
            seq_lengths = df.select(pl.col("sequence").str.lengths().alias("seq_length"))
            length_stats = seq_lengths.describe()
            
            stats["sequence_stats"] = {
                "min": length_stats.filter(pl.col("statistic") == "min")["seq_length"].item(),
                "max": length_stats.filter(pl.col("statistic") == "max")["seq_length"].item(),
                "mean": length_stats.filter(pl.col("statistic") == "mean")["seq_length"].item(),
                "std": length_stats.filter(pl.col("statistic") == "std")["seq_length"].item(),
            }
            
            print("\nSequence length statistics:")
            print(length_stats)
    
    # Generate output report if requested
    if output_report:
        with open(output_report, 'w') as f:
            f.write(f"Validation Report for {file_path}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total rows: {rows_before:,}\n")
            f.write(f"Unique combinations: {unique_combinations:,}\n")
            f.write(f"Duplicate rows: {duplicate_count:,}\n\n")
            
            if duplicate_count > 0:
                f.write("RESULT: DUPLICATES FOUND - Dataset contains duplicate entries\n")
                f.write(f"Number of duplicate groups: {counts.height:,}\n\n")
                f.write("Occurrence distribution:\n")
                for row in occurrence_distribution.iter_rows(named=True):
                    f.write(f"  Rows appearing {row['count']} times: {row['frequency']:,} groups\n")
            else:
                f.write("RESULT: VALIDATION PASSED - All combinations are unique\n")
                
            print(f"\nReport saved to: {output_report}")
    
    elapsed = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed:.2f} seconds")
    
    # Return success (no duplicates) and statistics
    return not stats["has_duplicates"], stats


def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        # Determine which processing method to use
        if args.use_polars:
            if not HAS_POLARS:
                print("Warning: Polars requested but not installed. Falling back to pandas.")
                print("Install polars for better performance: pip install polars")
                success, stats = analyze_with_pandas(
                    file_path=args.input,
                    verbose=args.verbose,
                    show_duplicates=args.show_duplicates,
                    save_duplicates=args.save_duplicates,
                    output_report=args.output
                )
            else:
                success, stats = analyze_with_polars(
                    file_path=args.input,
                    verbose=args.verbose,
                    show_duplicates=args.show_duplicates,
                    save_duplicates=args.save_duplicates,
                    output_report=args.output
                )
        else:
            success, stats = analyze_with_pandas(
                file_path=args.input,
                verbose=args.verbose,
                show_duplicates=args.show_duplicates,
                save_duplicates=args.save_duplicates,
                output_report=args.output
            )
        
        return 0 if success else 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
