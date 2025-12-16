"""Memory-efficient Parquet deduplication for meta-model datasets.

The incremental builder can produce duplicate rows when the same splice site
appears in multiple transcripts.  This helper removes *exact* duplicates based
on a user-supplied subset of columns while touching only one Parquet file at a
time to conserve RAM.

Usage (new directory):
    python -m meta_spliceai.splice_engine.meta_models.builder.deduplicate_dataset \
        train_pc_1000/master                               \
        --dst train_pc_1000/master_dedup                   \
        --subset-cols chrom position strand

In-place:
    python -m meta_spliceai.splice_engine.meta_models.builder.deduplicate_dataset \
        train_pc_1000/master --inplace
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl
import pyarrow.dataset as ds

__all__ = ["deduplicate_parquet_dataset", "deduplicate_inplace"]

# ---------------------------------------------------------------------------
# Core helpers ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def _iter_parquet_files(directory: Path) -> Iterable[Path]:
    """Yield *.parquet files in *directory* (non-recursive)."""
    for path in sorted(directory.glob("*.parquet")):
        if path.is_file():
            yield path


def _dedup_file(src_path: Path, *, subset_cols: Sequence[str], compression: str, dest_path: Path | None = None) -> None:
    """Deduplicate *src_path* → *dest_path* (or overwrite in place)."""
    if dest_path is None:
        dest_path = src_path
    df = pl.read_parquet(src_path).unique(subset=subset_cols)
    df.write_parquet(dest_path, compression=compression)


# ---------------------------------------------------------------------------
# Public API -----------------------------------------------------------------
# ---------------------------------------------------------------------------

def deduplicate_parquet_dataset(
    src: str | Path,
    dst: str | Path | None = None,
    *,
    subset_cols: Sequence[str] = ("chrom", "position", "strand"),
    compression: str = "zstd",
    verbose: bool = True,
) -> Path:
    """Deduplicate each Parquet file in *src* and write to *dst*.

    Only the subset of columns is compared; all other columns are retained.
    The function does *not* attempt a global dedup across files – duplicates
    spanning multiple batches are rare and memory-expensive to track.
    """
    src = Path(src)
    if not src.is_dir():
        raise NotADirectoryError(src)

    if dst is None:
        dst = src.with_name(src.name + "_dedup")
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    for f in _iter_parquet_files(src):
        out_path = dst / f.name
        if verbose:
            print(f"[dedup] {f.name} → {out_path.name}")
        _dedup_file(f, subset_cols=subset_cols, compression=compression, dest_path=out_path)
    if verbose:
        print(f"[dedup] Completed – output at {dst}")
    return dst


def deduplicate_inplace(
    directory: str | Path,
    *,
    subset_cols: Sequence[str] = ("chrom", "position", "strand"),
    compression: str = "zstd",
    verbose: bool = True,
) -> None:
    """Deduplicate every Parquet file in *directory* in place (destructive)."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    for f in _iter_parquet_files(directory):
        tmp = f.with_suffix(".tmp.parquet")
        if verbose:
            print(f"[dedup] {f.name} (in-place)")
        _dedup_file(f, subset_cols=subset_cols, compression=compression, dest_path=tmp)
        os.replace(tmp, f)  # atomic on POSIX
    if verbose:
        print("[dedup] Completed in-place deduplication.")


# ---------------------------------------------------------------------------
# CLI entry-point ------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Deduplicate Parquet dataset produced by the meta-model builder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("src", help="Directory containing Parquet files (master dataset).")
    parser.add_argument("--dst", help="Output directory for deduped files (omit for _dedup suffix).")
    parser.add_argument(
        "--subset-cols",
        nargs="+",
        default=["chrom", "position", "strand"],
        help="Columns that jointly define row identity.",
    )
    parser.add_argument("--inplace", action="store_true", help="Overwrite files in place (destructive).")
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec for output.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress messages.")
    args = parser.parse_args()

    if args.inplace and args.dst:
        parser.error("--inplace and --dst are mutually exclusive")

    if args.inplace:
        deduplicate_inplace(
            args.src,
            subset_cols=args.subset_cols,
            compression=args.compression,
            verbose=not args.quiet,
        )
    else:
        deduplicate_parquet_dataset(
            args.src,
            dst=args.dst,
            subset_cols=args.subset_cols,
            compression=args.compression,
            verbose=not args.quiet,
        )
