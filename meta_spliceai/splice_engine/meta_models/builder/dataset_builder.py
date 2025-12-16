"""Dataset Builder
===================
Utility for assembling a scalable training dataset from the many
*_analysis_sequences_*.tsv[.gz] files produced by
``run_enhanced_splice_prediction_workflow``.

Highlights
----------
• Memory-efficient: streams TSVs in Lazily with polars `scan_csv`.
• Supports gene sub-sampling (top-N by FP/FN or explicit list).
• Optional on-the-fly k-mer feature extraction using existing
  ``make_kmer_features`` helper.
• Incremental write to Parquet (single file or sharded) so the
  resulting dataset can be memory-mapped by downstream ML libraries.

Example
-------
>>> from meta_spliceai.splice_engine.meta_models.dataset_builder import build_training_dataset
>>> build_training_dataset(
...     analysis_tsv_dir="/path/to/eval/meta_models",
...     output_path="/tmp/splice_training.parquet",
...     mode="fp",          # genes with most false positives
...     top_n_genes=500,
...     kmer_sizes=[6, 5],
... )
"""
from __future__ import annotations

import os
import gzip
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Iterable, Literal, Set

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import itertools

import pandas as pd

# Import concise column name constants
import meta_spliceai.splice_engine.meta_models.constants as const

# Optional: only import when k-mer features requested
from meta_spliceai.splice_engine.meta_models.features.kmer_features import (
    make_kmer_features,
)

# Base set of columns expected in every analysis TSV.
# We include concise Gene/Transcript identifiers via the shared constants
# module so that updates propagate consistently across the codebase.
EXPECTED_MIN_COLUMNS = [
    const.col_gid,
    const.col_tid,
    "position",
    "donor_score",
    "neither_score",
    "acceptor_score",
    "pred_type",
    "splice_type",
    "relative_donor_probability",
    "splice_probability",
    "donor_acceptor_logodds",
    "splice_neither_logodds",
    "probability_entropy",
    "chrom",
    "sequence",  # may be removed later if k-mer encoding replaces it
]

# Columns that must always be interpreted as 64-bit integers to ensure
# schema consistency across all analysis chunks.
_INT_COLUMNS: list[str] = [
    "position",
    "predicted_position",
    "true_position",
    "window_start",
    "window_end",
    "transcript_count",
    "donor_is_local_peak",
    "acceptor_is_local_peak",
]
# Always treat *chrom* as string to preserve non-numeric IDs like X/Y/MT.
_DTYPE_OVERRIDES = {col: pl.Int64 for col in _INT_COLUMNS} | {"chrom": pl.Utf8}

MODE = Literal["fp", "fn", "error_total", "none"]

__all__ = [
    "build_training_dataset",
]


def _discover_tsv_files(root: Path) -> List[Path]:
    """Return list of *analysis_sequences* TSV/TSV.GZ files inside *root*.

    Backup copies created by the schema-fix utility carry the extensions
    ``.bak`` or ``.bak.gz``.  Reading them alongside the active files doubles
    the dataset size and can cause unnecessary RAM spikes.  We therefore
    exclude any file whose *name* ends with those suffixes.
    """
    return [
        f
        for f in root.rglob("*analysis_sequences_*.tsv*")
        if not f.name.endswith((".bak", ".bak.gz"))
    ]


def _gene_error_counts(files: List[Path]) -> Dict[str, Dict[str, int]]:
    """Return per-gene counts of TP/FP/FN/TN by scanning only *gene_id* & *pred_type* columns."""
    counts: Dict[str, Dict[str, int]] = {}
    for f in files:
        # read with polars scan – very fast and memory efficient
        scan = (
            pl.scan_csv(
                f,
                separator="\t",
                null_values="NA",
                with_column_names=lambda cols: cols,
            )
            .select(["gene_id", "pred_type"])
        )
        for chunk in scan.collect(streaming=True).iter_rows(named=True):
            g = chunk["gene_id"]
            p = chunk["pred_type"]
            g_counts = counts.setdefault(g, {"FP": 0, "FN": 0, "TP": 0, "TN": 0})
            if p in g_counts:
                g_counts[p] += 1
    return counts


def _select_top_genes(counts: Dict[str, Dict[str, int]], *, mode: MODE, top_n: int) -> Set[str]:
    metric = {
        "fp": lambda d: d["FP"],
        "fn": lambda d: d["FN"],
        "error_total": lambda d: d["FP"] + d["FN"],
        "none": lambda d: 0,
    }[mode]
    if mode == "none":
        return set(counts.keys())
    selected = sorted(counts.items(), key=lambda kv: metric(kv[1]), reverse=True)[:top_n]
    return {g for g, _ in selected}


def _iter_batches(
    files: List[Path],
    required_cols: List[str],
    selected_genes: Optional[Set[str]],
    batch_rows: int,
) -> Iterable[pl.DataFrame]:
    buffer = []
    total = 0
    for f in files:
        # Load the full TSV so that downstream enrichers have access to
        # probability/context columns that are *not* part of EXPECTED_MIN_COLUMNS.
        # We only verify later that the mandatory columns exist.
        scan = pl.scan_csv(
            f,
            separator="\t",
            null_values="NA",
            with_column_names=lambda cols: cols,
            dtypes=_DTYPE_OVERRIDES,  # enforce critical integer columns
        )
        if selected_genes is not None:
            scan = scan.filter(pl.col("gene_id").is_in(selected_genes))
        df = scan.collect(streaming=True)  # polars DataFrame
        if df.height == 0:
            continue
        buffer.append(df)
        total += df.height
        if total >= batch_rows:
            # Harmonise schemas across buffered DataFrames before concatenation.
            union_cols: set[str] = set().union(*[df.columns for df in buffer])
            ordered_cols = sorted(union_cols)
            # Build a mapping col -> dtype from the first DF that contains it
            col_dtypes: dict[str, pl.datatypes.DataType] = {}
            for _df in buffer:
                for _c in _df.columns:
                    col_dtypes.setdefault(_c, _df[_c].dtype)
            aligned: list[pl.DataFrame] = []
            for df_buf in buffer:
                miss = [c for c in ordered_cols if c not in df_buf.columns]
                if miss:
                    expressions = [
                        (
                            pl.lit(None).cast(col_dtypes[c]) if c in col_dtypes and col_dtypes[c] != pl.Null else pl.lit(None)
                        ).alias(c)
                        for c in miss
                    ]
                    df_buf = df_buf.with_columns(expressions)
                aligned.append(df_buf.select(ordered_cols))
            yield pl.concat(aligned)
            buffer = []
            total = 0
    if buffer:
        union_cols_fin: set[str] = set().union(*[df.columns for df in buffer])
        ordered_cols_fin = sorted(union_cols_fin)
        # Build dtype map once more for final flush
        col_dtypes_fin: dict[str, pl.datatypes.DataType] = {}
        for _df in buffer:
            for _c in _df.columns:
                col_dtypes_fin.setdefault(_c, _df[_c].dtype)
        aligned_fin: list[pl.DataFrame] = []
        for df_buf in buffer:
            miss = [c for c in ordered_cols_fin if c not in df_buf.columns]
            if miss:
                expressions = [
                    (
                        pl.lit(None).cast(col_dtypes_fin[c]) if c in col_dtypes_fin and col_dtypes_fin[c] != pl.Null else pl.lit(None)
                    ).alias(c)
                    for c in miss
                ]
                df_buf = df_buf.with_columns(expressions)
            aligned_fin.append(df_buf.select(ordered_cols_fin))
        yield pl.concat(aligned_fin)


def _ensure_expected_columns(df, expected: List[str]):
    """Check that expected columns are present in DataFrame (supports both pandas and polars)."""
    # Handle both pandas and polars DataFrames
    if hasattr(df, 'columns'):
        columns = df.columns
    else:
        raise ValueError("DataFrame must have a 'columns' attribute")
    
    missing = [c for c in expected if c not in columns]
    if missing:
        # Add detailed debugging information
        available_cols = sorted(list(columns))
        expected_cols = sorted(expected)
        found_cols = sorted([c for c in expected if c in columns])
        
        error_msg = f"Dataset missing expected columns: {missing[:5]}…\n"
        error_msg += f"Available columns ({len(available_cols)}): {available_cols[:10]}...\n"
        error_msg += f"Expected columns ({len(expected_cols)}): {expected_cols}\n"
        error_msg += f"Found columns ({len(found_cols)}): {found_cols}\n"
        error_msg += f"Missing columns ({len(missing)}): {missing}"
        
        raise ValueError(error_msg)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def build_training_dataset(
    analysis_tsv_dir: str,
    output_path: str,
    *,
    mode: MODE = "none",  # fp/fn/error_total selects top genes
    top_n_genes: int | None = None,
    target_genes: Optional[List[str]] = None,
    kmer_sizes: Optional[List[int]] = None,
    batch_rows: int = 500_000,
    keep_sequence: bool = False,
    compression: str | None = "zstd",
    initial_schema_cols: Optional[List[str]] = None,
    transcript_config = None,  # NEW: Optional transcript-aware configuration
    verbose: int = 1,
) -> None:

    """Assemble Parquet training dataset.

    Parameters
    ----------
    analysis_tsv_dir : str
        Directory containing many *_analysis_sequences_*.tsv(.gz) files.
    output_path : str
        Destination Parquet file (will be overwritten).
    mode : {"fp", "fn", "error_total", "none"}
        Strategy to choose *top_n_genes*.
    top_n_genes : int | None
        If provided, restrict dataset to these many genes by *mode*.
    target_genes : list[str] | None
        Explicit set of genes to keep (overrides *mode/top_n_genes*).
    kmer_sizes : list[int] | None
        Extract k-mer features; if None, skip.
    keep_sequence : bool
        If False and kmer_sizes is given, drop raw sequence column.
    transcript_config : TranscriptAwareConfig | None
        Optional transcript-aware position identification configuration.
        If None, uses current genomic-only behavior (backward compatible).
        NOTE: Currently passed through but not used in this function.
        Position deduplication happens in sequence_data_utils.py during inference.
    """
    # Log transcript-aware configuration if provided
    if transcript_config is not None and verbose:
        print(f"[dataset-builder] Transcript-aware config received: {transcript_config.mode}")
        print(f"[dataset-builder] Note: Position deduplication will use transcript-aware grouping during inference")
    
    root = Path(analysis_tsv_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    files = _discover_tsv_files(root)
    if verbose:
        print(f"[dataset-builder] Found {len(files)} analysis TSVs under {root}")

    # Determine gene subset ------------------------------------------------
    selected_genes: Optional[Set[str]] = None
    if target_genes is not None:
        selected_genes = set(target_genes)
    elif top_n_genes is not None and mode != "none":
        if verbose:
            print("[dataset-builder] Counting gene errors to select top genes …")
        counts = _gene_error_counts(files)
        selected_genes = _select_top_genes(counts, mode=mode, top_n=top_n_genes)
        if verbose:
            print(f"  → selected {len(selected_genes)} genes via mode '{mode}'")

    # Required columns -----------------------------------------------------
    req_cols = EXPECTED_MIN_COLUMNS.copy()
    if kmer_sizes is None and not keep_sequence and "sequence" in req_cols:
        req_cols.remove("sequence")

    # Parquet writer setup --------------------------------------------------
    out_path = Path(output_path)
    if out_path.exists():
        out_path.unlink()
    writer: pq.ParquetWriter | None = None
    schema_cols: list[str] | None = initial_schema_cols.copy() if initial_schema_cols else None

    # Pre-compute full k-mer column set to enforce consistent schema --------
    _all_kmer_cols: list[str] = []
    if kmer_sizes is not None and len(kmer_sizes) > 0:
        _alphabet = ("A", "C", "G", "T")
        for k in kmer_sizes:
            _all_kmer_cols.extend(
                [f"{k}mer_{''.join(p)}" for p in itertools.product(_alphabet, repeat=k)]
            )

    for batch_idx, pdf_batch in enumerate(
        _iter_batches(files, req_cols, selected_genes, batch_rows), start=1
    ):
        # Optionally add k-mer features -----------------------------------
        if kmer_sizes is not None:
            pd_batch = pdf_batch.to_pandas()
            
            # CRITICAL: Preserve essential metadata columns before k-mer extraction
            essential_metadata_cols = ['gene_id', 'transcript_id', 'position', 'pred_type', 'splice_type', 
                                     'donor_score', 'acceptor_score', 'neither_score', 'chrom']
            preserved_metadata = {}
            for col in essential_metadata_cols:
                if col in pd_batch.columns:
                    preserved_metadata[col] = pd_batch[col].copy()
            
            # Use featurize_gene_sequences directly to have control over column preservation
            from meta_spliceai.splice_engine.sequence_featurizer import featurize_gene_sequences
            pd_batch, _ = featurize_gene_sequences(
                pd_batch,
                kmer_sizes=kmer_sizes,
                return_feature_set=True,
                drop_source_columns=False,  # CRITICAL: Preserve all metadata columns
                verbose=0,
            )
            
            # CRITICAL: Restore essential metadata columns that may have been dropped
            for col, data in preserved_metadata.items():
                if col not in pd_batch.columns:
                    pd_batch[col] = data
                    if verbose >= 2:
                        print(f"  • Restored essential column: {col}")
            if not keep_sequence and "sequence" in pd_batch.columns:
                pd_batch = pd_batch.drop(columns=["sequence"])

            # Ensure *all* k-mer columns are present even if absent in data --
            missing_cols = [c for c in _all_kmer_cols if c not in pd_batch.columns]
            if missing_cols:
                # Add all missing k-mer columns in one shot to avoid fragmentation
                filler_df = pd.DataFrame({mc: 0.0 for mc in missing_cols}, index=pd_batch.index)
                pd_batch = pd.concat([pd_batch, filler_df], axis=1)

            # CRITICAL: Ensure all k-mer columns have consistent float64 dtype
            # This prevents schema mismatch errors between batches where some k-mer
            # columns might be int64 (when no missing data) vs float64 (when filled)
            for kmer_col in _all_kmer_cols:
                if kmer_col in pd_batch.columns:
                    pd_batch[kmer_col] = pd_batch[kmer_col].astype('float64')

            # CRITICAL: Ensure boolean columns have consistent int64 dtype
            # These come from enhanced workflow as Int8 but need to be int64 for Parquet schema consistency
            _boolean_cols = ["donor_is_local_peak", "acceptor_is_local_peak"]
            for bool_col in _boolean_cols:
                if bool_col in pd_batch.columns:
                    pd_batch[bool_col] = pd_batch[bool_col].astype('int64')

            # Reorder columns so that k-mer cols are at the end in a stable order
            non_kmer_cols = [c for c in pd_batch.columns if c not in _all_kmer_cols]
            pd_batch = pd_batch[non_kmer_cols + _all_kmer_cols]

            # ------------------------------------------------------------------
            # Align with schema of first batch to keep Parquet writer happy
            # ------------------------------------------------------------------
            if schema_cols is not None:
                # 1) Add any columns that were in the schema but missing here
                missing = [c for c in schema_cols if c not in pd_batch.columns]
                if missing:
                    filler_dict: dict[str, float | str] = {}
                    for mc in missing:
                        is_kmer = mc.startswith(tuple(str(k) + 'mer_' for k in (kmer_sizes or [])))
                        filler_dict[mc] = 0.0 if is_kmer else ""
                    pd_batch = pd.concat([pd_batch, pd.DataFrame(filler_dict, index=pd_batch.index)], axis=1)
                # 2) Drop unexpected extra columns
                extra = [c for c in pd_batch.columns if c not in schema_cols]
                if extra:
                    pd_batch = pd_batch.drop(columns=extra)
                # 3) Reorder to match schema exactly
                pd_batch = pd_batch[schema_cols]

            # ------------------------------------------------------------------
            # Normalise string-like categorical columns (always) ---------------
            # ------------------------------------------------------------------
            # 1) splice_type: convert any legacy numeric encoding → strings
            if "splice_type" in pd_batch.columns:
                # Fail-safe logic: anything that is not "donor" or "acceptor" should be "neither"
                # This handles None/null values from the base prediction workflow and any legacy encodings
                # Also handles "0" or 0 which may result from missing value handling logic
                
                # First, convert to string to handle None values
                pd_batch["splice_type"] = pd_batch["splice_type"].astype(str)
                
                # Apply fail-safe logic: if not donor or acceptor, then it's neither
                # This covers None (becomes "None"), "0", 0, "nan", etc.
                pd_batch["splice_type"] = pd_batch["splice_type"].apply(
                    lambda x: x if x in ["donor", "acceptor"] else "neither"
                )

            # 2) Ensure Arrow-friendly nullable string dtype for key columns
            _str_cols = [
                "gene_id",
                "splice_type",
                "pred_type",
                "chrom",
                "strand",
            ]
            for _c in _str_cols:
                if _c in pd_batch.columns:
                    pd_batch[_c] = pd_batch[_c].astype("string").fillna("")

            table = pa.Table.from_pandas(pd_batch, preserve_index=False)
        else:
            if not keep_sequence and "sequence" in pdf_batch.columns:
                pdf_batch = pdf_batch.drop("sequence")
            table = pa.Table.from_pylist(pdf_batch.to_dicts())

        # Validate that all required columns are present ------------------
        expected_cols = EXPECTED_MIN_COLUMNS.copy()
        if not keep_sequence and "sequence" in expected_cols:
            expected_cols.remove("sequence")
        _ensure_expected_columns(table.to_pandas().head(1), expected_cols)

        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression=compression)
            schema_cols = table.schema.names
            
            # CRITICAL: For k-mer features, ensure schema_cols includes ALL possible k-mers
            # This prevents schema mismatch when subsequent batches have k-mers not in first batch
            if _all_kmer_cols:
                # Add any k-mer columns that might not have been in the first batch
                missing_kmer_cols = [c for c in _all_kmer_cols if c not in schema_cols]
                if missing_kmer_cols:
                    if verbose >= 2:
                        print(f"  • Added {len(missing_kmer_cols)} missing k-mer columns to schema")
                    schema_cols.extend(missing_kmer_cols)
        try:
            writer.write_table(table)
        except ValueError as e:
            if "Table schema does not match" in str(e):
                # Provide a concise error message for schema mismatches with many columns
                current_cols = set(table.schema.names)
                expected_cols = set(schema_cols) if schema_cols else set()
                missing = expected_cols - current_cols
                extra = current_cols - expected_cols
                
                error_msg = f"Schema mismatch in batch {batch_idx}:\n"
                if missing:
                    missing_sample = list(missing)[:5]
                    error_msg += f"  Missing columns: {len(missing)} total"
                    if len(missing) <= 5:
                        error_msg += f" {missing_sample}"
                    else:
                        error_msg += f" (sample: {missing_sample}...)"
                    error_msg += "\n"
                if extra:
                    extra_sample = list(extra)[:5]  
                    error_msg += f"  Extra columns: {len(extra)} total"
                    if len(extra) <= 5:
                        error_msg += f" {extra_sample}"
                    else:
                        error_msg += f" (sample: {extra_sample}...)"
                    error_msg += "\n"
                
                # Suggest resolution
                error_msg += "\nResolution: Delete the partially written Parquet file and re-run:"
                error_msg += f"\n  rm {out_path}"
                error_msg += f"\n  <re-run your command>"
                
                raise ValueError(error_msg) from e
            else:
                raise
        if verbose:
            print(f"  • wrote batch {batch_idx} with {table.num_rows:,} rows")

    if writer is not None:
        writer.close()
        if verbose:
            print(f"[dataset-builder] Completed: {out_path} written.")
    else:
        print("[dataset-builder] No data matched the criteria.")
