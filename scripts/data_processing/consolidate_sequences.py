#!/usr/bin/env python3
"""
Consolidate Analysis Sequence Files

This script consolidates multiple chunked analysis sequence files into a single file,
performing validation to ensure no duplicate sequences exist in the consolidated output.

Typical usage:
    python consolidate_sequences.py --input-dir /path/to/sequence/files --output consolidated_sequences.tsv
"""

import pandas as pd
import polars as pl
import glob
import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Set, Dict, Union


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Consolidate multiple analysis sequence files into a single file"
    )
    parser.add_argument(
        "--input-dir", 
        type=str,
        default=".",
        help="Directory containing analysis sequence files (default: current directory)"
    )
    parser.add_argument(
        "--pattern", 
        type=str,
        default="analysis_sequences_*.tsv",
        help="File pattern to match (default: analysis_sequences_*.tsv)"
    )
    parser.add_argument(
        "--output", 
        type=str,
        default="consolidated_sequences.tsv",
        help="Output file path (default: consolidated_sequences.tsv)"
    )
    parser.add_argument(
        "--deduplicate", 
        action="store_true",
        help="Remove duplicates from the consolidated output"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be done without writing output file"
    )
    parser.add_argument(
        "--use-polars", 
        action="store_true",
        help="Use polars instead of pandas for processing (faster for large files)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show detailed processing information"
    )
    return parser.parse_args()


def find_sequence_files(input_dir: Union[str, Path], pattern: str) -> List[Path]:
    """Find analysis sequence files in the specified directory.
    
    Parameters
    ----------
    input_dir : str or Path
        Directory to search for files
    pattern : str
        Glob pattern to match files
        
    Returns
    -------
    List[Path]
        List of paths to matching files
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_path}")
        
    # Find all matching files
    files = list(input_path.glob(pattern))
    files.sort()  # Sort for consistent processing order
    return files


def consolidate_with_pandas(
    files: List[Path],
    output_file: Union[str, Path],
    deduplicate: bool = True,
    dry_run: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """Consolidate sequence files using pandas.
    
    Parameters
    ----------
    files : List[Path]
        List of files to consolidate
    output_file : str or Path
        Path to output file
    deduplicate : bool
        If True, remove duplicates from output
    dry_run : bool
        If True, don't write output file
    verbose : bool
        If True, print detailed progress information
        
    Returns
    -------
    pd.DataFrame
        Consolidated DataFrame
    """
    if not files:
        raise ValueError("No files to process")
    
    print(f"Found {len(files)} files to process")
    
    # Read and combine all files
    dfs = []
    total_rows = 0
    composite_key = ['gene_id', 'position', 'strand', 'splice_type']
    
    for idx, file in enumerate(files, 1):
        if verbose:
            print(f"Processing file {idx}/{len(files)}: {file.name}")
        else:
            print(f"Processing file {idx}/{len(files)}... ", end="")
            
        try:
            df = pd.read_csv(file, sep='\t')
            rows_before = len(df)
            total_rows += rows_before
            
            # Check for duplicates within this file
            if verbose:
                duplicates = df.duplicated(subset=composite_key, keep=False)
                if duplicates.any():
                    print(f"  Warning: Found {duplicates.sum()} duplicates in {file.name}")
            
            dfs.append(df)
            if verbose:
                print(f"  Added {rows_before} rows from {file.name}")
            else:
                print(f"OK ({rows_before} rows)")
                
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
            continue
    
    # Combine all dataframes
    print("\nCombining data...")
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined into {len(combined_df)} rows")
    
    # Validation checks
    print("\nValidating...")
    rows_before_dedup = len(combined_df)
    
    if deduplicate:
        combined_df = combined_df.drop_duplicates(
            subset=composite_key
        )
        rows_after_dedup = len(combined_df)
        duplicates_removed = rows_before_dedup - rows_after_dedup
        
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate entries")
        else:
            print("✓ No duplicates found in consolidated dataset")
    else:
        # Just check for duplicates without removing
        unique_combinations = combined_df.drop_duplicates(
            subset=composite_key
        ).shape[0]
        
        if rows_before_dedup != unique_combinations:
            duplicates_found = rows_before_dedup - unique_combinations
            print(f"Warning: Found {duplicates_found} duplicate entries in consolidated dataset")
            
            if verbose:
                # Find and display duplicates
                duplicates = combined_df[combined_df.duplicated(
                    subset=composite_key, 
                    keep=False
                )].sort_values(by=composite_key)
                print("\nSample duplicate entries:")
                print(duplicates.head(10))
        else:
            print("✓ No duplicates found in consolidated dataset")
    
    # Save consolidated file
    if not dry_run:
        try:
            combined_df.to_csv(output_file, sep='\t', index=False)
            print(f"\nSaved consolidated dataset with {len(combined_df)} rows to: {output_file}")
        except Exception as e:
            print(f"\nError saving to {output_file}: {e}")
    else:
        print(f"\n[DRY RUN] Would save {len(combined_df)} rows to: {output_file}")
    
    return combined_df


def consolidate_with_polars(
    files: List[Path],
    output_file: Union[str, Path],
    deduplicate: bool = True,
    dry_run: bool = False,
    verbose: bool = False
) -> pl.DataFrame:
    """Consolidate sequence files using polars (faster for large files).
    
    Parameters
    ----------
    files : List[Path]
        List of files to consolidate
    output_file : str or Path
        Path to output file
    deduplicate : bool
        If True, remove duplicates from output
    dry_run : bool
        If True, don't write output file
    verbose : bool
        If True, print detailed progress information
        
    Returns
    -------
    pl.DataFrame
        Consolidated DataFrame
    """
    if not files:
        raise ValueError("No files to process")
    
    print(f"Found {len(files)} files to process")
    composite_key = ['gene_id', 'position', 'strand', 'splice_type']
    
    # Read and combine all files
    dfs = []
    total_rows = 0
    
    for idx, file in enumerate(files, 1):
        if verbose:
            print(f"Processing file {idx}/{len(files)}: {file.name}")
        else:
            print(f"Processing file {idx}/{len(files)}... ", end="")
            
        try:
            df = pl.read_csv(file, separator='\t')
            rows_before = df.height
            total_rows += rows_before
            
            # Check for duplicates within this file if verbose
            if verbose:
                dup_count = df.height - df.unique(subset=composite_key).height
                if dup_count > 0:
                    print(f"  Warning: Found {dup_count} duplicates in {file.name}")
            
            dfs.append(df)
            if verbose:
                print(f"  Added {rows_before} rows from {file.name}")
            else:
                print(f"OK ({rows_before} rows)")
                
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
            continue
    
    # Combine all dataframes
    print("\nCombining data...")
    combined_df = pl.concat(dfs)
    rows_before_dedup = combined_df.height
    print(f"Combined into {rows_before_dedup} rows")
    
    # Validation checks
    print("\nValidating...")
    
    if deduplicate:
        rows_before = combined_df.height
        combined_df = combined_df.unique(subset=composite_key)
        rows_after = combined_df.height
        duplicates_removed = rows_before - rows_after
        
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate entries")
        else:
            print("✓ No duplicates found in consolidated dataset")
    else:
        # Just check for duplicates without removing
        unique_df = combined_df.unique(subset=composite_key)
        unique_count = unique_df.height
        
        if rows_before_dedup != unique_count:
            duplicates_found = rows_before_dedup - unique_count
            print(f"Warning: Found {duplicates_found} duplicate entries in consolidated dataset")
            
            if verbose:
                # Find and display duplicates - only in verbose mode to save processing time
                counts = (
                    combined_df.group_by(composite_key)
                    .count()
                    .filter(pl.col("count") > 1)
                    .sort("count", descending=True)
                )
                print("\nTop duplicate groups:")
                print(counts.head(10))
        else:
            print("✓ No duplicates found in consolidated dataset")
    
    # Save consolidated file
    if not dry_run:
        try:
            combined_df.write_csv(output_file, separator='\t')
            print(f"\nSaved consolidated dataset with {combined_df.height} rows to: {output_file}")
        except Exception as e:
            print(f"\nError saving to {output_file}: {e}")
    else:
        print(f"\n[DRY RUN] Would save {combined_df.height} rows to: {output_file}")
    
    return combined_df


def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        # Find sequence files
        files = find_sequence_files(args.input_dir, args.pattern)
        if not files:
            print(f"No files found matching pattern '{args.pattern}' in {args.input_dir}")
            return 1
        
        # Process files with selected backend
        if args.use_polars:
            consolidate_with_polars(
                files=files,
                output_file=args.output,
                deduplicate=args.deduplicate,
                dry_run=args.dry_run,
                verbose=args.verbose
            )
        else:
            consolidate_with_pandas(
                files=files,
                output_file=args.output,
                deduplicate=args.deduplicate,
                dry_run=args.dry_run,
                verbose=args.verbose
            )
        
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

