"""Error counting utilities for meta-model analysis.

This module provides functions for counting errors in splice site prediction
data, with special attention to counting unique errors (i.e., by genomic position)
rather than transcript-level occurrences.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import pandas as pd
import polars as pl

from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler


def count_effective_errors(
    positions_df: pl.DataFrame,
    *,
    group_by: List[str] = ["gene_id"],
    count_by_position: bool = True,
    error_types: List[str] = ["FP", "FN"],
    verbose: int = 1
) -> pl.DataFrame:
    """Count effective errors, avoiding duplicate counts for the same genomic position.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing splice site predictions with at least gene_id, position, and pred_type columns
    group_by : List[str], optional
        Columns to group by for error counts, by default ["gene_id"]
    count_by_position : bool, optional
        If True, count unique positions rather than transcript-level occurrences, by default True
    error_types : List[str], optional
        Error types to count, by default ["FP", "FN"]
    verbose : int, optional
        Verbosity level (0=silent), by default 1
        
    Returns
    -------
    pl.DataFrame
        DataFrame with error counts per grouping level
    """
    if "pred_type" not in positions_df.columns:
        raise ValueError("positions_df must contain a 'pred_type' column")
    
    if "position" not in positions_df.columns and count_by_position:
        raise ValueError("positions_df must contain a 'position' column when count_by_position=True")
    
    # Check that all group_by columns exist
    for col in group_by:
        if col not in positions_df.columns:
            raise ValueError(f"Column '{col}' specified in group_by not found in positions_df")

    # Filter to error rows
    error_mask = positions_df["pred_type"].is_in(error_types)
    errors_df = positions_df.filter(error_mask)
    
    # Prepare for counting unique positions per group
    if count_by_position:
        # For each group, we want unique position counts
        # We'll add the required key columns for uniqueness
        unique_key_cols = group_by + ["position"]
        
        # Create a deduplicated version with only unique positions per group
        unique_errors_df = errors_df.unique(subset=unique_key_cols)
        
        # Count unique errors by type for each group
        result = (
            unique_errors_df
            .group_by(group_by + ["pred_type"])
            .agg(pl.count().alias("effective_count"))
            .pivot(values="effective_count", index=group_by, columns="pred_type", aggregate_function="sum")
        )
        
        # Add a total effective errors column
        # First make sure all error type columns exist and have nulls filled with 0
        for err_type in error_types:
            if err_type in result.columns:
                result = result.with_columns(pl.col(err_type).fill_null(0))
            else:
                result = result.with_columns(pl.lit(0).alias(err_type))
                
        # Create the total by directly summing the columns (avoiding list of expressions)
        result = result.with_columns(
            sum(pl.col(err_type) for err_type in error_types).alias("total_effective_errors")
        )
        
        # Sort by total errors descending
        result = result.sort("total_effective_errors", descending=True)
        
    else:
        # Standard count including transcript-level duplicates
        result = (
            errors_df
            .group_by(group_by + ["pred_type"])
            .agg(pl.count().alias("count"))
            .pivot(values="count", index=group_by, columns="pred_type", aggregate_function="sum")
        )
        
        # Add a total errors column
        # First make sure all error type columns exist and have nulls filled with 0
        for err_type in error_types:
            if err_type in result.columns:
                result = result.with_columns(pl.col(err_type).fill_null(0))
            else:
                result = result.with_columns(pl.lit(0).alias(err_type))
                
        # Create the total by directly summing the columns (avoiding list of expressions)
        result = result.with_columns(
            sum(pl.col(err_type) for err_type in error_types).alias("total_errors")
        )
        
        # Sort by total errors descending
        result = result.sort("total_errors", descending=True)

    # We already handle null values earlier in the function, so this is redundant
    # and can be removed
    
    if verbose:
        # Print comparison stats if we're doing position-based counting
        if count_by_position:
            raw_counts = (
                errors_df
                .group_by("pred_type")
                .agg(pl.count().alias("raw_count"))
            )
            
            effective_counts = (
                unique_errors_df
                .group_by("pred_type")
                .agg(pl.count().alias("effective_count"))
            )
            
            print(f"Raw error counts: {raw_counts}")
            print(f"Effective error counts: {effective_counts}")
            
            # Calculate reduction percentage
            for error_type in error_types:
                if error_type in raw_counts["pred_type"]:
                    raw_count = raw_counts.filter(pl.col("pred_type") == error_type)["raw_count"].item()
                    effective_count = effective_counts.filter(pl.col("pred_type") == error_type)["effective_count"].item()
                    reduction = (raw_count - effective_count) / raw_count * 100 if raw_count > 0 else 0
                    print(f"{error_type}: Raw={raw_count}, Effective={effective_count}, Reduction={reduction:.1f}%")
    
    return result


def load_positions_with_effective_error_counts(
    data_handler: Optional[MetaModelDataHandler] = None,
    *,
    aggregated: bool = True,
    group_by: List[str] = ["gene_id"],
    n_genes: Optional[int] = None,
    error_types: List[str] = ["FP", "FN"],
    enhanced: bool = True,
    verbose: int = 1,
    **kwargs
) -> Dict[str, pl.DataFrame]:
    """Load positions DataFrame and compute effective error counts.
    
    Parameters
    ----------
    data_handler : Optional[MetaModelDataHandler], optional
        Data handler to use for loading positions, by default None (creates a new one)
    aggregated : bool, optional
        Whether to load aggregated data, by default True
    group_by : List[str], optional
        Columns to group by for error counts, by default ["gene_id"]
    n_genes : Optional[int], optional
        If provided, return the top N genes by effective error count, by default None
    error_types : List[str], optional
        Error types to count, by default ["FP", "FN"]
    enhanced : bool, optional
        Whether to load enhanced positions, by default True
    verbose : int, optional
        Verbosity level, by default 1
    **kwargs
        Additional arguments passed to data_handler.load_splice_positions
        
    Returns
    -------
    Dict[str, pl.DataFrame]
        Dictionary containing:
        - 'positions': The loaded positions DataFrame
        - 'error_counts': DataFrame with effective error counts
        - 'top_genes': List of top gene IDs by effective error count (if n_genes specified)
    """
    # Create data handler if not provided
    if data_handler is None:
        data_handler = MetaModelDataHandler()
    
    if verbose:
        print(f"[error_counting] Loading positions DataFrame (enhanced={enhanced})...")
    
    # Load positions DataFrame
    positions_df = data_handler.load_splice_positions(
        aggregated=aggregated,
        enhanced=enhanced,
        **kwargs
    )
    
    if verbose:
        print(f"[error_counting] Loaded positions DataFrame with {positions_df.height} rows")
    
    # Count effective errors
    error_counts = count_effective_errors(
        positions_df,
        group_by=group_by,
        error_types=error_types,
        verbose=verbose
    )
    
    result = {
        'positions': positions_df,
        'error_counts': error_counts
    }
    
    # Get top genes if requested
    if n_genes is not None and n_genes > 0:
        total_col = "total_effective_errors" if "total_effective_errors" in error_counts.columns else "total_errors"
        top_genes = error_counts.sort(total_col, descending=True).limit(n_genes)
        result['top_genes'] = top_genes
        
        if "gene_id" in group_by:
            top_gene_ids = top_genes.select("gene_id").to_series().to_list()
            if verbose:
                print(f"[error_counting] Top {len(top_gene_ids)} genes by effective error count: {top_gene_ids[:5]}...")
            result['top_gene_ids'] = top_gene_ids
    
    return result
