#!/usr/bin/env python3
"""
Memory-optimized dataset loading for large training datasets.

This module provides memory-efficient alternatives to the standard datasets.py
for handling very large datasets (>2GB, >3M rows) that cause OOM issues.
"""

import os
import logging
from pathlib import Path
from typing import Sequence, List, Dict, Any, Optional, Union, Tuple
import polars as pl
import pandas as pd

logger = logging.getLogger(__name__)


def estimate_dataset_size_efficiently(source: Union[str, Path]) -> Tuple[int, int]:
    """
    Estimate dataset size without loading full data into memory.
    
    Returns:
        tuple: (estimated_row_count, file_count)
    """
    from . import datasets
    
    paths = datasets._collect_parquet_paths(source)
    
    if len(paths) == 1 and paths[0].is_dir():
        # Directory case - get individual files
        parquet_files = list(paths[0].glob("*.parquet"))
    else:
        # Explicit file list
        parquet_files = paths
    
    print(f"[memory_optimized] Estimating size from {len(parquet_files)} files...")
    
    # Sample a few files to estimate average rows per file
    sample_size = min(3, len(parquet_files))
    sample_files = parquet_files[:sample_size]
    
    total_sample_rows = 0
    for sample_file in sample_files:
        try:
            # Use pandas to get quick row count without full load
            df_sample = pd.read_parquet(sample_file, columns=['gene_id'])  # Just read one column
            total_sample_rows += len(df_sample)
        except Exception as e:
            logger.warning(f"Could not sample {sample_file}: {e}")
    
    if total_sample_rows == 0:
        logger.warning("Could not estimate dataset size, falling back to conservative estimate")
        return 1000000, len(parquet_files)  # Conservative fallback
    
    # Estimate total rows
    avg_rows_per_file = total_sample_rows / sample_size
    estimated_total_rows = int(avg_rows_per_file * len(parquet_files))
    
    print(f"[memory_optimized] Estimated {estimated_total_rows:,} rows from {len(parquet_files)} files")
    
    return estimated_total_rows, len(parquet_files)


def load_dataset_streaming(
    source: Union[str, Path],
    *,
    columns: Optional[Sequence[str]] = None,
    batch_size: int = 50000,
    max_memory_gb: float = 8.0,
    lazy: bool = False,
    rechunk: bool = True,
) -> pl.DataFrame:
    """
    Load large dataset using memory-optimized streaming approach.
    
    This function processes large datasets by:
    1. Estimating dataset size without full loading
    2. Processing files in batches to stay within memory limits
    3. Using efficient concatenation strategies
    
    Parameters
    ----------
    source : str or Path
        File or directory produced by the builder
    columns : Sequence[str], optional
        Optional subset of columns to read
    batch_size : int, default=50000
        Number of rows to process per batch
    max_memory_gb : float, default=8.0
        Maximum memory to use (GB)
    lazy : bool, default=False
        If True, return LazyFrame
    rechunk : bool, default=True
        Whether to rechunk the final result
        
    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        The loaded dataset
    """
    from . import datasets
    from meta_spliceai.splice_engine.meta_models.builder import preprocessing as prep
    
    print(f"[memory_optimized] Loading dataset with memory limit: {max_memory_gb:.1f}GB")
    
    # Get file paths
    paths = datasets._collect_parquet_paths(source)
    
    if len(paths) == 1 and paths[0].is_dir():
        parquet_files = sorted(list(paths[0].glob("*.parquet")))
        print(f"[memory_optimized] Found {len(parquet_files)} files in directory")
    else:
        parquet_files = paths
        print(f"[memory_optimized] Processing {len(parquet_files)} explicit files")
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {source}")
    
    # Estimate size to determine strategy
    estimated_rows, file_count = estimate_dataset_size_efficiently(source)
    
    # Determine if we need batch processing
    estimated_memory_gb = (estimated_rows * 1200 * 8) / (1024**3)  # Rough estimate: 1200 cols * 8 bytes
    
    if estimated_memory_gb <= max_memory_gb:
        print(f"[memory_optimized] Estimated memory {estimated_memory_gb:.1f}GB <= limit, using standard loading")
        return _load_dataset_standard(parquet_files, columns, lazy, rechunk)
    else:
        print(f"[memory_optimized] Estimated memory {estimated_memory_gb:.1f}GB > limit, using batch processing")
        return _load_dataset_batched(parquet_files, columns, batch_size, max_memory_gb, lazy, rechunk)


def _load_dataset_standard(
    parquet_files: List[Path],
    columns: Optional[Sequence[str]],
    lazy: bool,
    rechunk: bool
) -> pl.DataFrame:
    """Standard loading approach for smaller datasets."""
    from . import datasets
    from meta_spliceai.splice_engine.meta_models.builder import preprocessing as prep
    
    # Use the existing safe scan approach but with explicit file list and schema handling
    def "". _safe_scan_parquet(pattern):
        try:
            return pl.scan_parquet(pattern, missing_columns='insert', extra_columns='ignore')
        except TypeError as e:
            if 'extra_columns' in str(e) or 'missing_columns' in str(e):
                try:
                    return pl.scan_parquet(pattern, missing_columns='insert')
                except TypeError:
                    return pl.scan_parquet(pattern)
            raise
    
    pattern_list = [str(p) for p in parquet_files]
    lf = _safe_scan_parquet(pattern_list)
    
    # Apply column filtering
    if columns is None:
        drop_set = set(prep.DEFAULT_DROP_COLUMNS) - {"splice_type", "gene_id", "chrom", "position", "transcript_id"}
        try:
            schema_obj = lf.collect_schema()
        except AttributeError:
            schema_obj = lf.schema if hasattr(lf, "schema") else lf.collect().schema
        
        if isinstance(schema_obj, dict):
            schema_names = list(schema_obj.keys())
        else:
            if hasattr(schema_obj, 'names'):
                schema_names = list(schema_obj.names) if isinstance(schema_obj.names, (list, tuple)) else list(schema_obj.names())
            else:
                schema_names = list(schema_obj)
        
        keep_cols = [c for c in schema_names if c not in drop_set]
        lf = lf.select(keep_cols)
    else:
        lf = lf.select(columns)
    
    if lazy:
        return lf
    
    # Collect with streaming
    print(f"[memory_optimized] Collecting {len(parquet_files)} files...")
    df = lf.collect(streaming=True)
    
    if rechunk:
        df = df.rechunk()
    
    print(f"[memory_optimized] Loaded dataset: {df.shape}")
    return df


def _load_dataset_batched(
    parquet_files: List[Path],
    columns: Optional[Sequence[str]],
    batch_size: int,
    max_memory_gb: float,
    lazy: bool,
    rechunk: bool
) -> pl.DataFrame:
    """Batch processing approach for large datasets."""
    from meta_spliceai.splice_engine.meta_models.builder import preprocessing as prep
    
    print(f"[memory_optimized] Processing {len(parquet_files)} files in batches...")
    
    # Use smaller batches for very large datasets to reduce memory pressure
    if len(parquet_files) > 30:
        files_per_batch = 3  # Very small batches for large datasets
    else:
        files_per_batch = 5  # Standard batch size
    
    print(f"[memory_optimized] Processing {files_per_batch} files per batch")
    
    # Initialize with first batch
    first_batch_files = parquet_files[:files_per_batch]
    print(f"[memory_optimized] Processing initial batch (1/{(len(parquet_files) + files_per_batch - 1) // files_per_batch}) ({len(first_batch_files)} files)...")
    
    # Load first batch with robust schema handling
    try:
        pattern_list = [str(p) for p in first_batch_files]
        lf = pl.scan_parquet(pattern_list, missing_columns='insert', extra_columns='ignore')
    except TypeError:
        # Fallback for older Polars versions
        try:
            lf = pl.scan_parquet(pattern_list, missing_columns='insert')
        except TypeError:
            lf = pl.scan_parquet(pattern_list)
    
    # Apply column filtering
    if columns is None:
        drop_set = set(prep.DEFAULT_DROP_COLUMNS) - {"splice_type", "gene_id", "chrom", "position", "transcript_id"}
        try:
            schema_obj = lf.collect_schema()
        except AttributeError:
            schema_obj = lf.schema if hasattr(lf, "schema") else lf.collect().schema
        
        if isinstance(schema_obj, dict):
            schema_names = list(schema_obj.keys())
        else:
            if hasattr(schema_obj, 'names'):
                schema_names = list(schema_obj.names) if isinstance(schema_obj.names, (list, tuple)) else list(schema_obj.names())
            else:
                schema_names = list(schema_obj)
        
        keep_cols = [c for c in schema_names if c not in drop_set]
        lf = lf.select(keep_cols)
    else:
        lf = lf.select(columns)
    
    # Collect first batch
    final_df = lf.collect(streaming=True)
    print(f"[memory_optimized] Initial batch loaded: {final_df.shape}")
    
    # Process remaining batches incrementally
    for i in range(files_per_batch, len(parquet_files), files_per_batch):
        batch_files = parquet_files[i:i + files_per_batch]
        batch_num = (i // files_per_batch) + 1
        total_batches = (len(parquet_files) + files_per_batch - 1) // files_per_batch
        
        print(f"[memory_optimized] Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)...")
        
        # Load next batch with robust schema handling
        try:
            pattern_list = [str(p) for p in batch_files]
            lf = pl.scan_parquet(pattern_list, missing_columns='insert', extra_columns='ignore')
        except TypeError:
            # Fallback for older Polars versions
            try:
                lf = pl.scan_parquet(pattern_list, missing_columns='insert')
            except TypeError:
                lf = pl.scan_parquet(pattern_list)
        
        lf = lf.select(keep_cols)  # Use same columns as first batch
        batch_df = lf.collect(streaming=True)
        
        print(f"[memory_optimized] Batch {batch_num} loaded: {batch_df.shape}")
        
        # Concatenate incrementally using lazy operations
        print(f"[memory_optimized] Concatenating with existing data...")
        combined_lf = pl.concat([final_df.lazy(), batch_df.lazy()], how="vertical")
        
        # Clear previous dataframes from memory
        del final_df, batch_df
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        # Collect the combined result
        final_df = combined_lf.collect(streaming=True)
        print(f"[memory_optimized] Combined dataset: {final_df.shape}")
    
    # Skip rechunking for very large datasets to avoid OOM
    if rechunk and final_df.height < 2_000_000:
        print(f"[memory_optimized] Rechunking final dataset...")
        final_df = final_df.rechunk()
    elif rechunk:
        print(f"[memory_optimized] Skipping rechunking for large dataset ({final_df.height:,} rows) to avoid OOM")
    
    print(f"[memory_optimized] Final dataset: {final_df.shape}")
    
    if lazy:
        return final_df.lazy()
    
    return final_df


def load_dataset_with_memory_management(
    source: Union[str, Path],
    *,
    columns: Optional[Sequence[str]] = None,
    lazy: bool = False,
    rechunk: bool = True,
    max_memory_gb: float = 8.0,
    fallback_to_standard: bool = True,
) -> pl.DataFrame:
    """
    Main entry point for memory-optimized dataset loading.
    
    This function automatically detects if memory optimization is needed
    and falls back to standard loading for smaller datasets.
    """
    try:
        return load_dataset_streaming(
            source,
            columns=columns,
            lazy=lazy,
            rechunk=rechunk,
            max_memory_gb=max_memory_gb,
        )
    except Exception as e:
        if fallback_to_standard:
            logger.warning(f"Memory-optimized loading failed ({e}), falling back to standard loader")
            from . import datasets
            return datasets.load_dataset(source, columns=columns, lazy=lazy, rechunk=rechunk)
        else:
            raise
