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

MODE = Literal["fp", "fn", "error_total", "none"]

__all__ = [
    "build_training_dataset",
]


def _discover_tsv_files(root: Path) -> List[Path]:
    """Return sorted list of analysis TSV/TSV.GZ files inside *root*.

    The original glob required an underscore before ``analysis_sequences`` which
    excluded files that begin directly with that stem (e.g.
    ``analysis_sequences_1_chunk_1_500.tsv``).  We drop the underscore so both
    forms are matched.
    """
    return sorted(root.rglob("*analysis_sequences_*.tsv*"))


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
        )
        if selected_genes is not None:
            scan = scan.filter(pl.col("gene_id").is_in(selected_genes))
        df = scan.collect(streaming=True)  # polars DataFrame
        if df.height == 0:
            continue
        buffer.append(df)
        total += df.height
        if total >= batch_rows:
            yield pl.concat(buffer)
            buffer = []
            total = 0
    if buffer:
        yield pl.concat(buffer)


def _ensure_expected_columns(df: pl.DataFrame, expected: List[str]):
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing[:5]}…")


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
    """
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

    for batch_idx, pdf_batch in enumerate(
        _iter_batches(files, req_cols, selected_genes, batch_rows), start=1
    ):
        # Optionally add k-mer features -----------------------------------
        if kmer_sizes is not None:
            pd_batch = pdf_batch.to_pandas()
            pd_batch, _ = make_kmer_features(
                pd_batch,
                kmer_sizes=kmer_sizes,
                return_feature_set=True,   # ensure tuple output for safe unpacking
                verbose=0,
            )
            if not keep_sequence and "sequence" in pd_batch.columns:
                pd_batch = pd_batch.drop(columns=["sequence"])

            # Ensure Arrow-friendly dtypes ---------------------------------
            _str_cols = [
                "gene_id",
                "splice_type",
                "pred_type",
                "chrom",
            ]
            for _c in _str_cols:
                if _c in pd_batch.columns:
                    # ensure nullable string dtype and replace nulls with ""
                    pd_batch[_c] = (
                        pd_batch[_c]
                        .astype("string")
                        .fillna("")
                    )

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
        writer.write_table(table)
        if verbose:
            print(f"  • wrote batch {batch_idx} with {table.num_rows:,} rows")

    if writer is not None:
        writer.close()
        if verbose:
            print(f"[dataset-builder] Completed: {out_path} written.")
    else:
        print("[dataset-builder] No data matched the criteria.")
