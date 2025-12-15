#!/usr/bin/env python3
"""
Split gene building approach for very large datasets.
Builds multiple smaller datasets that can be combined later.
"""
import subprocess
import sys
from pathlib import Path
import pandas as pd

def split_gene_list(gene_file, n_splits=4, output_prefix="genes_split"):
    """Split a gene list into multiple files."""
    # Read gene list
    if gene_file.endswith('.tsv'):
        df = pd.read_csv(gene_file, sep='\t')
    else:
        df = pd.read_csv(gene_file)
    
    gene_col = 'gene_id'  # Adjust if needed
    genes = df[gene_col].tolist()
    
    # Calculate split sizes
    chunk_size = len(genes) // n_splits
    remainder = len(genes) % n_splits
    
    splits = []
    start_idx = 0
    
    for i in range(n_splits):
        # Add one extra gene to first few splits if there's remainder
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        chunk_genes = genes[start_idx:end_idx]
        
        # Save to file
        output_file = f"{output_prefix}_{i+1}.tsv"
        chunk_df = pd.DataFrame({gene_col: chunk_genes})
        chunk_df.to_csv(output_file, sep='\t', index=False)
        
        splits.append({
            'file': output_file,
            'genes': len(chunk_genes),
            'start': start_idx,
            'end': end_idx
        })
        
        start_idx = end_idx
        print(f"Split {i+1}: {len(chunk_genes)} genes -> {output_file}")
    
    return splits

def run_split_build(split_info, base_cmd_args):
    """Run incremental builder for one split."""
    split_file = split_info['file']
    split_num = split_file.split('_')[-1].replace('.tsv', '')
    output_dir = f"train_pc_split_{split_num}"
    
    cmd = [
        'python', '-m', 
        'meta_spliceai.splice_engine.meta_models.builder.incremental_builder',
        '--n-genes', str(split_info['genes']),
        '--subset-policy', 'error_total', 
        '--gene-ids-file', split_file,
        '--gene-col', 'gene_id',
        '--batch-size', '50',        # Very conservative
        '--batch-rows', '8000',      # Very conservative
        '--run-workflow',
        '--kmer-sizes', '3',
        '--output-dir', output_dir,
        '--overwrite', '-v'
    ]
    
    print(f"\n{'='*60}")
    print(f"BUILDING SPLIT {split_num}: {split_info['genes']} genes")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd)
    return result.returncode == 0

def combine_splits(split_dirs, final_output_dir="train_pc_1000_combined"):
    """Combine multiple split datasets into a single master dataset."""
    from pathlib import Path
    import shutil
    import os
    
    final_master_dir = Path(final_output_dir) / "master"
    final_master_dir.mkdir(parents=True, exist_ok=True)
    
    batch_num = 1
    
    for split_dir in split_dirs:
        master_dir = Path(split_dir) / "master"
        if not master_dir.exists():
            print(f"Warning: {master_dir} not found, skipping")
            continue
            
        # Copy all parquet files
        for parquet_file in master_dir.glob("*.parquet"):
            dest_file = final_master_dir / f"batch_{batch_num:05d}.parquet"
            print(f"Copying {parquet_file} -> {dest_file}")
            
            try:
                # Try hard link first (faster)
                os.link(parquet_file, dest_file)
            except OSError:
                # Fall back to copy
                shutil.copy2(parquet_file, dest_file)
            
            batch_num += 1
    
    print(f"\nCombined dataset created at: {final_output_dir}")
    return final_output_dir

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_gene_build.py <gene_file.tsv> [n_splits]")
        print("Example: python split_gene_build.py additional_genes.tsv 4")
        sys.exit(1)
    
    gene_file = sys.argv[1]
    n_splits = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    
    print(f"Splitting {gene_file} into {n_splits} parts...")
    splits = split_gene_list(gene_file, n_splits)
    
    # Build each split
    successful_dirs = []
    for i, split_info in enumerate(splits):
        print(f"\n\nProcessing split {i+1}/{len(splits)}")
        if run_split_build(split_info, []):
            split_num = split_info['file'].split('_')[-1].replace('.tsv', '')
            successful_dirs.append(f"train_pc_split_{split_num}")
            print(f"✓ Split {split_num} completed successfully")
        else:
            print(f"✗ Split {i+1} failed")
            break
    
    # Combine successful splits
    if successful_dirs:
        print(f"\n\nCombining {len(successful_dirs)} successful splits...")
        final_dir = combine_splits(successful_dirs)
        print(f"Final combined dataset: {final_dir}")
    else:
        print("No successful splits to combine") 