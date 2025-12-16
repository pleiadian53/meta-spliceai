#!/usr/bin/env python3
"""
Dataset Schema Validation Utility

Validates that all batch files in a dataset have consistent schemas,
particularly focusing on k-mer feature columns to prevent schema mismatches.

Usage:
    python validate_dataset_schema.py --dataset train_pc_5000_3mers_diverse/master
"""

import argparse
import glob
import pandas as pd
import polars as pl
from pathlib import Path
from typing import List, Dict, Set, Tuple
import sys


def get_standard_kmers(k: int) -> Set[str]:
    """Generate standard k-mers (no ambiguous nucleotides)."""
    import itertools
    
    bases = ['A', 'C', 'G', 'T']
    kmers = set()
    
    for kmer_tuple in itertools.product(bases, repeat=k):
        kmers.add(''.join(kmer_tuple))
    
    return kmers


def analyze_batch_schema(batch_file: str, verbose: bool = True) -> Dict:
    """Analyze the schema of a single batch file."""
    try:
        df = pd.read_parquet(batch_file)
        
        # Get all k-mer columns
        kmer_cols = [col for col in df.columns if col.startswith(('1mer_', '2mer_', '3mer_', '4mer_', '5mer_', '6mer_'))]
        
        # Group by k-mer size
        kmer_groups = {}
        for col in kmer_cols:
            # Extract k-mer size and sequence
            parts = col.split('_', 1)
            if len(parts) == 2:
                kmer_size = parts[0]  # e.g., '3mer'
                kmer_seq = parts[1]   # e.g., 'AAA' or 'NNN'
                if kmer_size not in kmer_groups:
                    kmer_groups[kmer_size] = []
                kmer_groups[kmer_size].append(kmer_seq)
        
        # Check for ambiguous k-mers
        ambiguous_kmers = {}
        for kmer_size, sequences in kmer_groups.items():
            ambiguous = [seq for seq in sequences if 'N' in seq]
            if ambiguous:
                ambiguous_kmers[kmer_size] = ambiguous
        
        result = {
            'file': batch_file,
            'total_columns': len(df.columns),
            'kmer_columns': len(kmer_cols),
            'kmer_groups': kmer_groups,
            'ambiguous_kmers': ambiguous_kmers,
            'has_issues': bool(ambiguous_kmers)
        }
        
        if verbose:
            print(f"📊 {Path(batch_file).name}: {len(df.columns)} cols, {len(kmer_cols)} k-mers")
            if ambiguous_kmers:
                print(f"   ⚠️  Ambiguous k-mers: {ambiguous_kmers}")
        
        return result
        
    except Exception as e:
        return {
            'file': batch_file,
            'error': str(e),
            'has_issues': True
        }


def validate_dataset_schema(dataset_path: str, fix_issues: bool = False, verbose: bool = True) -> Dict:
    """Validate schema consistency across all batch files in a dataset."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Find all batch files
    batch_files = glob.glob(str(dataset_path / "batch_*.parquet"))
    if not batch_files:
        raise ValueError(f"No batch files found in {dataset_path}")
    
    batch_files.sort()
    
    if verbose:
        print(f"🔍 Validating schema for {len(batch_files)} batch files in {dataset_path}")
    
    # Analyze each batch
    analyses = []
    for batch_file in batch_files:
        analysis = analyze_batch_schema(batch_file, verbose=verbose)
        analyses.append(analysis)
    
    # Check for consistency
    issues = [a for a in analyses if a.get('has_issues', False)]
    
    if verbose:
        print(f"\n📋 Schema Analysis Summary:")
        print(f"   Total batches: {len(batch_files)}")
        print(f"   Batches with issues: {len(issues)}")
    
    if issues:
        print(f"\n🚨 SCHEMA INCONSISTENCIES DETECTED:")
        for issue in issues:
            if 'error' in issue:
                print(f"   ❌ {Path(issue['file']).name}: {issue['error']}")
            elif issue['ambiguous_kmers']:
                print(f"   ⚠️  {Path(issue['file']).name}: Ambiguous k-mers found")
                for kmer_type, ambiguous in issue['ambiguous_kmers'].items():
                    print(f"      {kmer_type}: {ambiguous}")
        
        if fix_issues:
            print(f"\n🔧 Attempting to fix schema issues...")
            fix_dataset_schema(dataset_path, analyses, verbose=verbose)
    
    else:
        print(f"\n✅ All batches have consistent schema!")
    
    return {
        'total_batches': len(batch_files),
        'issues_found': len(issues),
        'analyses': analyses
    }


def fix_dataset_schema(dataset_path: Path, analyses: List[Dict], verbose: bool = True) -> None:
    """Fix schema issues by removing ambiguous k-mers."""
    # Get standard k-mers for each size
    standard_kmers = {}
    for k in [1, 2, 3, 4, 5, 6]:
        standard_kmers[f'{k}mer'] = get_standard_kmers(k)
    
    fixed_count = 0
    
    for analysis in analyses:
        if not analysis.get('has_issues', False) or 'error' in analysis:
            continue
        
        batch_file = analysis['file']
        ambiguous_kmers = analysis.get('ambiguous_kmers', {})
        
        if not ambiguous_kmers:
            continue
        
        if verbose:
            print(f"🔧 Fixing {Path(batch_file).name}...")
        
        try:
            # Load the batch
            df = pd.read_parquet(batch_file)
            
            # Collect all columns to remove
            columns_to_remove = []
            for kmer_type, ambiguous in ambiguous_kmers.items():
                for ambiguous_kmer in ambiguous:
                    col_name = f"{kmer_type}_{ambiguous_kmer}"
                    if col_name in df.columns:
                        columns_to_remove.append(col_name)
            
            if columns_to_remove:
                # Remove ambiguous k-mer columns
                df_fixed = df.drop(columns=columns_to_remove)
                
                # Save the fixed batch
                df_fixed.to_parquet(batch_file, index=False)
                
                if verbose:
                    print(f"   ✅ Removed {len(columns_to_remove)} ambiguous k-mer columns")
                    print(f"   📊 New shape: {df_fixed.shape}")
                
                fixed_count += 1
        
        except Exception as e:
            print(f"   ❌ Failed to fix {Path(batch_file).name}: {e}")
    
    if verbose:
        print(f"\n🎉 Fixed {fixed_count} batch files")


def main():
    parser = argparse.ArgumentParser(description="Validate dataset schema consistency")
    parser.add_argument("--dataset", required=True, help="Path to dataset directory")
    parser.add_argument("--fix", action="store_true", help="Automatically fix schema issues")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        result = validate_dataset_schema(args.dataset, fix_issues=args.fix, verbose=args.verbose)
        
        if result['issues_found'] > 0:
            print(f"\n⚠️  Found {result['issues_found']} batches with schema issues")
            if not args.fix:
                print("   Use --fix to automatically resolve issues")
            sys.exit(1)
        else:
            print(f"\n✅ Dataset schema validation passed!")
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()





