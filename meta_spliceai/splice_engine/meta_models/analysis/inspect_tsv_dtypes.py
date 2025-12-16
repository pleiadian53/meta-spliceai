#!/usr/bin/env python
"""Quick utility to inspect column dtypes of TSV/CSV files

Run for any file(s):

    python -m meta_spliceai.splice_engine.meta_models.analysis.inspect_tsv_dtypes \
        data/ensembl/spliceai_analysis/genomic_gtf_feature_set.tsv \
        data/ensembl/spliceai_analysis/gene_features.tsv

It uses *polars.scan_csv* so the file is not fully loaded into memory; only
column names and dtypes are read.  Any column whose dtype is ``Utf8`` but is
expected to be numeric will appear clearly in the output and should be
regenerated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import polars as pl
try:
    from rich import print as rprint
except ImportError:  # fallback to plain print if rich not available
    rprint = print  # type: ignore


NUMERIC_HINTS: tuple[str, ...] = (
    "start",
    "end",
    "position",
    "length",
    "distance",
    "probability",
    "score",
)


def inspect_file(path: Path) -> None:
    """Print the Polars schema of *path* and flag suspicious Utf8 columns."""
    if not path.exists():
        rprint(f"[red]File not found:[/red] {path}")
        return

    try:
        schema = pl.scan_csv(path, separator="\t", null_values="NA").schema
    except Exception as exc:  # pragma: no cover – robust utility
        rprint(f"[red]Failed to read {path}: {exc}[/red]")
        return

    rprint(f"[bold]{path}[/bold]")
    for col, dtype in schema.items():
        warn = (
            dtype == pl.Utf8
            and any(h in col for h in NUMERIC_HINTS)
        )
        mark = "⚠️" if warn else " "
        rprint(f" {mark} {col:<25} {str(dtype)}")
    rprint()


def parse_args(argv: List[str]) -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(description="Inspect column dtypes in TSV/CSV files using Polars.")
    p.add_argument("files", nargs="+", help="One or more TSV/CSV files")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:  # pragma: no cover
    ns = parse_args(sys.argv[1:] if argv is None else argv)
    for f in ns.files:
        inspect_file(Path(f).expanduser())


if __name__ == "__main__":  # pragma: no cover
    main()
