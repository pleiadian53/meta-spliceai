#!/usr/bin/env python3
"""Test script to demonstrate effective error counting.

This script loads the splice positions test file and analyzes the difference
between raw transcript-level error counts and effective position-level error counts.

Run with:
    python -m meta_spliceai.splice_engine.meta_models.analysis.test_error_counting
"""
import os
import sys
from pathlib import Path

import polars as pl

from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
from meta_spliceai.splice_engine.meta_models.analysis.error_counting import (
    count_effective_errors,
    load_positions_with_effective_error_counts
)


def main():
    # Create data handler
    data_handler = MetaModelDataHandler()
    
    # Get path to test file
    test_file_path = os.path.join(
        data_handler.meta_dir, "full_splice_positions_enhanced_test.tsv"
    )
    
    print(f"\nTest file: {test_file_path}")
    print(f"File exists: {os.path.isfile(test_file_path)}\n")
    
    if not os.path.isfile(test_file_path):
        print("Error: Test file not found. Please ensure the file exists.")
        return
    
    # Load the test file directly
    print("Loading test positions DataFrame...")
    positions_df = pl.read_csv(
        test_file_path,
        separator="\t",
        infer_schema_length=0,
        schema_overrides={"chrom": pl.Utf8},
    )
    
    print(f"Loaded {positions_df.height} rows with {len(positions_df.columns)} columns")
    
    # Get raw error counts by gene
    print("\n1. Raw error counts by gene (include transcript duplicates):")
    raw_counts = count_effective_errors(
        positions_df,
        count_by_position=False,
        verbose=0
    )
    
    # Get effective error counts by gene
    print("\n2. Effective error counts by gene (deduplicated by position):")
    effective_counts = count_effective_errors(
        positions_df,
        count_by_position=True,
        verbose=2
    )
    
    # Get top genes from either count method
    print("\n3. Side-by-side comparison for top genes:")
    combined_genes = (
        set(raw_counts.select("gene_id").head(10).to_series().to_list())
        | set(effective_counts.select("gene_id").head(10).to_series().to_list())
    )
    
    # Create a comparison table
    print("\nRaw vs Effective error counts:")
    print("{:<16} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Gene ID", "Raw FP", "Eff FP", "Raw FN", "Eff FN", "Raw Total", "Eff Total"
    ))
    print("-" * 80)
    
    for gene_id in sorted(combined_genes):
        # Get raw counts
        raw_row = raw_counts.filter(pl.col("gene_id") == gene_id)
        raw_fp = raw_row["FP"].item() if "FP" in raw_row.columns else 0
        raw_fn = raw_row["FN"].item() if "FN" in raw_row.columns else 0 
        raw_total = raw_row["total_errors"].item()
        
        # Get effective counts
        eff_row = effective_counts.filter(pl.col("gene_id") == gene_id)
        eff_fp = eff_row["FP"].item() if "FP" in eff_row.columns else 0
        eff_fn = eff_row["FN"].item() if "FN" in eff_row.columns else 0
        eff_total = eff_row["total_effective_errors"].item()
        
        # Calculate reduction percentages
        fp_pct = ((raw_fp - eff_fp) / raw_fp * 100) if raw_fp > 0 else 0
        fn_pct = ((raw_fn - eff_fn) / raw_fn * 100) if raw_fn > 0 else 0
        total_pct = ((raw_total - eff_total) / raw_total * 100) if raw_total > 0 else 0
        
        # Print the row
        print("{:<16} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            gene_id, 
            f"{raw_fp}", 
            f"{eff_fp} ({fp_pct:.0f}%)", 
            f"{raw_fn}", 
            f"{eff_fn} ({fn_pct:.0f}%)", 
            f"{raw_total}", 
            f"{eff_total} ({total_pct:.0f}%)"
        ))
    
    # Display the top 5 genes with highest raw error counts
    print("\nTop 5 genes by raw error count:")
    print(raw_counts.head(5))
    
    # Display the top 5 genes with highest effective error counts
    print("\nTop 5 genes by effective error count:")
    print(effective_counts.head(5))
    
    # Compare changes in ranking
    print("\n4. Changes in gene rankings:")
    
    # Get top 10 gene IDs by each counting method
    top_raw = raw_counts.select("gene_id").head(10).to_series().to_list()
    top_effective = effective_counts.select("gene_id").head(10).to_series().to_list()
    
    # Find genes that appear in one top list but not the other
    raw_only = [g for g in top_raw if g not in top_effective]
    effective_only = [g for g in top_effective if g not in top_raw]
    
    # Count genes that changed rank
    rank_changed = 0
    for i, gene in enumerate(top_raw):
        if gene in top_effective and top_effective.index(gene) != i:
            rank_changed += 1
            
    print(f"Genes in top 10 by raw count only: {raw_only}")
    print(f"Genes in top 10 by effective count only: {effective_only}")
    print(f"Genes that changed rank position: {rank_changed}")
    
    # Calculate reduction percentage for total errors in top genes
    top_genes = set(top_raw) & set(top_effective)  # Genes in both top 10 lists
    total_raw = sum(raw_counts.filter(pl.col("gene_id").is_in(top_genes))["total_errors"])
    total_eff = sum(effective_counts.filter(pl.col("gene_id").is_in(top_genes))["total_effective_errors"])
    reduction_pct = (total_raw - total_eff) / total_raw * 100 if total_raw > 0 else 0
    
    print(f"\nFor genes in both top 10 lists:")
    print(f"  Total raw errors: {total_raw}")
    print(f"  Total effective errors: {total_eff}")
    print(f"  Reduction: {reduction_pct:.1f}%")
    
    # Get full stats with convenience function
    print("\n5. Using convenience function:")
    result = load_positions_with_effective_error_counts(
        data_handler=data_handler,
        n_genes=5,
        verbose=1
    )
    
    print("\nTop 5 genes by effective error count:")
    print(result['top_genes'])
    
    print("\nDone!")


if __name__ == "__main__":
    main()
