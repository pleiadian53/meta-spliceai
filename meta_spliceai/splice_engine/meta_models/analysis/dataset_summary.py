"""Quick summary statistics for a meta-model training dataset.

Run as a module:

    python -m meta_spliceai.splice_engine.meta_models.analysis.dataset_summary \
           /path/to/dataset/master  -v

The script prints:
• row count
• unique gene / transcript counts (if columns present)
• class balance for *label* column (if present)
• null-value counts per column
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import polars as pl
from rich.console import Console
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _scan_dataset(dataset_dir: str | Path) -> pl.LazyFrame:
    """Return a Polars *lazy* scan across **all** Parquet files in *dataset_dir*."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    # Accept both flat file layout and hive-partition style sub-directories
    glob_pattern = str(dataset_dir / "**" / "*.parquet")
    return pl.scan_parquet(glob_pattern)


def _null_counts(lf: pl.LazyFrame, columns: Sequence[str]) -> dict[str, int]:
    nulls: dict[str, int] = {}
    for c in columns:
        null_count = (
            lf.select(pl.col(c).is_null().sum().alias("cnt"))
            .collect()
            .item()
        )
        nulls[c] = null_count
    return nulls


def summarize(dataset_dir: str | Path, verbose: int = 0) -> None:
    lf = _scan_dataset(dataset_dir)
    # Fast schema fetch without expensive resolution
    cols: list[str] = list(lf.collect_schema().keys())

    # Basic counts ---------------------------------------------------------
    basic_exprs: list[pl.Expr] = [pl.len().alias("rows")]

    if "gene_id" in cols:
        basic_exprs.append(pl.col("gene_id").n_unique().alias("genes"))
    if "transcript_id" in cols:
        basic_exprs.append(pl.col("transcript_id").n_unique().alias("transcripts"))
    if "label" in cols:
        basic_exprs.extend(
            [
                pl.col("label").sum().alias("positives"),
                (pl.count() - pl.col("label").sum()).alias("negatives"),
            ]
        )

    basic = lf.select(basic_exprs).collect().row(0, named=True)

    # Null counts ----------------------------------------------------------
    nulls = _null_counts(lf, cols)

    # ------------------------------------------------------------------
    # Pretty print ------------------------------------------------------
    # ------------------------------------------------------------------
    table = Table(title="Dataset Summary", show_lines=True)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for k, v in basic.items():
        table.add_row(k, f"{v:,}")

    # Separate section for null counts
    table.add_row("[bold]Null counts per column[/bold]", "")
    for col in cols:
        n_null = nulls[col]
        if n_null:
            pct = n_null / basic["rows"] * 100 if basic["rows"] else 0
            table.add_row(f"  {col}", f"{n_null:,}  ({pct:0.2f}%)")
    console.print(table)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quick summary statistics for a meta-model training dataset (Parquet).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dataset_dir", help="Directory containing Parquet files (master dataset or batches).")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (not used yet).")
    return p.parse_args()


def main() -> None:
    ns = _parse_cli()
    summarize(ns.dataset_dir, verbose=ns.verbose)


if __name__ == "__main__":
    main()
