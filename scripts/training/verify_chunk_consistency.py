#!/usr/bin/env python3
"""
Verification script to ensure mini-batch optimization maintains chunk consistency.

This script checks that:
1. Chunk files have the expected naming convention
2. Chunk files contain the expected number of genes
3. File structure is consistent before and after mini-batch optimization

Usage:
    python scripts/training/verify_chunk_consistency.py --meta-dir /path/to/meta_models
"""

import os
import sys
import argparse
from pathlib import Path
import polars as pl
from collections import defaultdict

def parse_chunk_filename(filename):
    """
    Parse chunk filename to extract chromosome and gene range.
    
    Expected format: analysis_sequences_chr2_chunk_1_500.tsv
    
    Returns:
        tuple: (chromosome, start_gene, end_gene) or None if invalid
    """
    parts = filename.replace('.tsv', '').split('_')
    try:
        # Find indices
        chr_idx = parts.index('sequences') + 1  # chromosome comes after 'sequences'
        chunk_idx = parts.index('chunk')
        
        chromosome = parts[chr_idx]
        start_gene = int(parts[chunk_idx + 1])
        end_gene = int(parts[chunk_idx + 2])
        
        return (chromosome, start_gene, end_gene)
    except (ValueError, IndexError):
        return None

def verify_chunk_files(meta_dir, expected_chunk_size=500):
    """
    Verify that chunk files follow expected naming and content conventions.
    
    Args:
        meta_dir: Path to meta_models directory containing chunk files
        expected_chunk_size: Expected number of genes per chunk (default 500)
    
    Returns:
        dict: Verification results with status and details
    """
    results = {
        'status': 'PASS',
        'total_chunks': 0,
        'chunks_by_chromosome': defaultdict(list),
        'issues': [],
        'statistics': {}
    }
    
    # Find all chunk files
    chunk_files = list(Path(meta_dir).glob('analysis_sequences_*_chunk_*.tsv'))
    results['total_chunks'] = len(chunk_files)
    
    if len(chunk_files) == 0:
        results['status'] = 'WARNING'
        results['issues'].append('No chunk files found in meta_dir')
        return results
    
    print(f"Found {len(chunk_files)} chunk files in {meta_dir}")
    print("-" * 80)
    
    for chunk_file in sorted(chunk_files):
        filename = chunk_file.name
        
        # Parse filename
        parsed = parse_chunk_filename(filename)
        if not parsed:
            results['status'] = 'FAIL'
            results['issues'].append(f"Invalid filename format: {filename}")
            continue
        
        chromosome, start_gene, end_gene = parsed
        expected_genes = end_gene - start_gene + 1
        
        # Read chunk file and count unique genes
        try:
            df = pl.read_csv(chunk_file, separator='\t')
            actual_genes = df['gene_id'].n_unique()
            total_rows = df.height
            
            # Store chunk info
            results['chunks_by_chromosome'][chromosome].append({
                'file': filename,
                'start': start_gene,
                'end': end_gene,
                'expected_genes': expected_genes,
                'actual_genes': actual_genes,
                'total_rows': total_rows
            })
            
            # Verify gene count consistency
            if actual_genes > expected_genes:
                results['status'] = 'FAIL'
                results['issues'].append(
                    f"{filename}: More genes than expected ({actual_genes} > {expected_genes})"
                )
            elif actual_genes < expected_genes * 0.95:  # Allow 5% variance for edge cases
                results['status'] = 'WARNING'
                results['issues'].append(
                    f"{filename}: Fewer genes than expected ({actual_genes} < {expected_genes})"
                )
            
            print(f"✓ {filename}")
            print(f"  Range: genes {start_gene}-{end_gene} (expected {expected_genes} genes)")
            print(f"  Content: {actual_genes} unique genes, {total_rows} total rows")
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['issues'].append(f"Error reading {filename}: {str(e)}")
            print(f"✗ {filename}: Error - {str(e)}")
    
    print("-" * 80)
    
    # Check for gaps in chunk sequences
    for chromosome, chunks in results['chunks_by_chromosome'].items():
        chunks_sorted = sorted(chunks, key=lambda x: x['start'])
        
        print(f"\nChromosome {chromosome}: {len(chunks_sorted)} chunks")
        
        for i, chunk in enumerate(chunks_sorted):
            # Check if this chunk connects to the next one
            if i < len(chunks_sorted) - 1:
                next_chunk = chunks_sorted[i + 1]
                expected_next_start = chunk['end'] + 1
                
                if next_chunk['start'] != expected_next_start:
                    results['status'] = 'WARNING'
                    issue = (
                        f"{chromosome}: Gap between {chunk['file']} (ends at {chunk['end']}) "
                        f"and {next_chunk['file']} (starts at {next_chunk['start']})"
                    )
                    results['issues'].append(issue)
                    print(f"  ⚠ {issue}")
            
            # Check chunk size consistency
            if chunk['expected_genes'] != expected_chunk_size and i < len(chunks_sorted) - 1:
                # Allow last chunk to be smaller
                results['status'] = 'WARNING'
                issue = f"{chunk['file']}: Unexpected chunk size ({chunk['expected_genes']} != {expected_chunk_size})"
                results['issues'].append(issue)
                print(f"  ⚠ {issue}")
    
    # Generate statistics
    all_chunks = [chunk for chunks in results['chunks_by_chromosome'].values() for chunk in chunks]
    if all_chunks:
        results['statistics'] = {
            'total_genes': sum(c['actual_genes'] for c in all_chunks),
            'total_rows': sum(c['total_rows'] for c in all_chunks),
            'avg_genes_per_chunk': sum(c['actual_genes'] for c in all_chunks) / len(all_chunks),
            'avg_rows_per_chunk': sum(c['total_rows'] for c in all_chunks) / len(all_chunks),
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Verify chunk file consistency after mini-batch optimization'
    )
    parser.add_argument(
        '--meta-dir',
        type=str,
        required=True,
        help='Path to meta_models directory containing chunk files'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Expected chunk size (default: 500)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.meta_dir):
        print(f"Error: Directory not found: {args.meta_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("CHUNK CONSISTENCY VERIFICATION")
    print("=" * 80)
    print(f"Meta directory: {args.meta_dir}")
    print(f"Expected chunk size: {args.chunk_size} genes")
    print()
    
    results = verify_chunk_files(args.meta_dir, args.chunk_size)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Status: {results['status']}")
    print(f"Total chunks: {results['total_chunks']}")
    print(f"Chromosomes: {len(results['chunks_by_chromosome'])}")
    
    if results['statistics']:
        print(f"\nStatistics:")
        print(f"  Total unique genes: {results['statistics']['total_genes']}")
        print(f"  Total rows: {results['statistics']['total_rows']}")
        print(f"  Avg genes per chunk: {results['statistics']['avg_genes_per_chunk']:.1f}")
        print(f"  Avg rows per chunk: {results['statistics']['avg_rows_per_chunk']:.1f}")
    
    if results['issues']:
        print(f"\nIssues found ({len(results['issues'])}):")
        for issue in results['issues']:
            print(f"  • {issue}")
    else:
        print("\n✓ No issues found - all chunks are consistent!")
    
    print("=" * 80)
    
    # Exit with appropriate code
    if results['status'] == 'FAIL':
        sys.exit(1)
    elif results['status'] == 'WARNING':
        sys.exit(0)  # Warnings are acceptable
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()








