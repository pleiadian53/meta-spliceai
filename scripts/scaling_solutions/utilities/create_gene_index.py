#!/usr/bin/env python3
"""
Gene-Position Index Creator
===========================
Creates an index file that maps gene IDs to their positions in the training dataset
for efficient gene-to-feature-vector lookups.

This addresses the position-centric nature of the training data where each gene
has thousands of positions, each with its own feature vector.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import polars as pl
import pandas as pd


def create_gene_position_index(
    dataset_dir: str | Path,
    output_path: str | Path | None = None,
    verbose: int = 1,
) -> Path:
    """Create a gene-to-position index for quick lookups.
    
    Parameters
    ----------
    dataset_dir
        Path to the master dataset directory containing Parquet files.
    output_path
        Path for the index file. If None, creates 'gene_position_index.csv' in dataset_dir.
    verbose
        Verbosity level.
        
    Returns
    -------
    Path
        Path to the generated index file.
    """
    dataset_dir = Path(dataset_dir)
    if output_path is None:
        output_path = dataset_dir / "gene_position_index.csv"
    
    if verbose:
        print(f"[index] Creating gene-position index from {dataset_dir} ...")
    
    # Find all parquet files
    parquet_paths = sorted(dataset_dir.glob("*.parquet"))
    if not parquet_paths:
        raise RuntimeError(f"No parquet files found in {dataset_dir}")
    
    index_data = []
    total_positions = 0
    
    for i, pq_path in enumerate(parquet_paths, 1):
        if verbose:
            print(f"[index] Processing {pq_path.name} ({i}/{len(parquet_paths)}) ...")
        
        # Read only essential columns to keep memory usage low
        # Note: 'position' column contains strand-dependent relative coordinates, not absolute genomic coordinates
        df = pl.read_parquet(pq_path, columns=["gene_id", "position", "chrom", "splice_type"])
        
        # Normalize splice_type encoding: anything not "donor" or "acceptor" becomes "neither"
        # This handles: "0", None, null, empty strings, etc.
        df = df.with_columns([
            pl.when(pl.col("splice_type").is_in(["donor", "acceptor"]))
            .then(pl.col("splice_type"))
            .otherwise(pl.lit("neither"))
            .alias("splice_type")
        ])
        
        if verbose:
            # Debug: show unique splice_type values before and after normalization
            original_types = df.select("splice_type").unique().to_series().to_list()
            print(f"[index] Original splice_type values: {original_types}")
            
            # Show final normalized values
            final_types = df.select("splice_type").unique().to_series().to_list()
            print(f"[index] Normalized splice_type values: {final_types}")
        
        # Group by gene_id and collect position information
        gene_groups = df.group_by("gene_id").agg([
            pl.col("position").alias("positions"),
            pl.col("chrom").first().alias("chrom"),
            pl.col("splice_type").alias("splice_types"),
            pl.len().alias("position_count")
        ])
        
        # Add file information
        gene_groups = gene_groups.with_columns([
            pl.lit(i).alias("file_index"),
            pl.lit(pq_path.name).alias("file_name")
        ])
        
        index_data.append(gene_groups)
        total_positions += df.height
    
    if not index_data:
        raise RuntimeError("No gene data found in Parquet files")
    
    # Combine all index information
    index_df = pl.concat(index_data)
    
    # Remove duplicates (genes that appear in multiple files)
    index_df = index_df.unique(subset=["gene_id"], maintain_order=True)
    
    # Add global index
    index_df = index_df.with_row_index("global_index")
    
    # Convert nested data to JSON strings for CSV compatibility
    import json
    index_df = index_df.with_columns([
        pl.col("positions").map_elements(lambda x: json.dumps(x.to_list()), return_dtype=pl.Utf8).alias("positions_json"),
        pl.col("splice_types").map_elements(lambda x: json.dumps(x.to_list()), return_dtype=pl.Utf8).alias("splice_types_json")
    ])
    
    # Reorder columns for better readability
    index_df = index_df.select([
        "global_index", "gene_id", "chrom", "position_count", 
        "positions_json", "splice_types_json", "file_index", "file_name"
    ])
    
    # Write index
    index_df.write_csv(output_path)
    
    if verbose:
        print(f"[index] Generated index with {index_df.height:,} unique genes")
        print(f"[index] Total positions across all genes: {total_positions:,}")
        print(f"[index] Index saved to: {output_path}")
        
        # Show statistics
        avg_positions = index_df.select(pl.col("position_count").mean()).item()
        max_positions = index_df.select(pl.col("position_count").max()).item()
        min_positions = index_df.select(pl.col("position_count").min()).item()
        
        print(f"[index] Average positions per gene: {avg_positions:.1f}")
        print(f"[index] Min positions per gene: {min_positions}")
        print(f"[index] Max positions per gene: {max_positions}")
    
    return output_path


def load_gene_positions(
    index_path: str | Path,
    gene_id: str,
    dataset_dir: str | Path | None = None,
) -> pl.DataFrame:
    """Load all positions for a specific gene using the index.
    
    Parameters
    ----------
    index_path
        Path to the gene-position index file.
    gene_id
        Ensembl gene ID to look up.
    dataset_dir
        Path to the dataset directory. If None, inferred from index_path.
        
    Returns
    -------
    pl.DataFrame
        DataFrame containing all positions and features for the gene.
    """
    index_path = Path(index_path)
    if dataset_dir is None:
        dataset_dir = index_path.parent
    
    # Load the index
    index_df = pl.read_csv(index_path)
    
    # Find the gene in the index
    gene_info = index_df.filter(pl.col("gene_id") == gene_id)
    
    if gene_info.height == 0:
        raise ValueError(f"Gene {gene_id} not found in index")
    
    if gene_info.height > 1:
        print(f"Warning: Gene {gene_id} appears in multiple files, using first occurrence")
    
    gene_row = gene_info.row(0)
    file_name = gene_row[gene_info.columns.index("file_name")]
    file_path = Path(dataset_dir) / file_name
    
    # Load the specific file and filter by gene_id
    df = pl.read_parquet(file_path)
    gene_data = df.filter(pl.col("gene_id") == gene_id)
    
    return gene_data


def main():
    parser = argparse.ArgumentParser(
        description="Create gene-position index for training datasets"
    )
    parser.add_argument(
        "dataset_dir",
        help="Path to the master dataset directory"
    )
    parser.add_argument(
        "--output",
        help="Output path for index file (default: dataset_dir/gene_position_index.csv)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=1,
        help="Increase verbosity"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = create_gene_position_index(
            dataset_dir=args.dataset_dir,
            output_path=args.output,
            verbose=args.verbose
        )
        print(f"✅ Index created successfully: {output_path}")
        
    except Exception as e:
        print(f"❌ Error creating index: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 