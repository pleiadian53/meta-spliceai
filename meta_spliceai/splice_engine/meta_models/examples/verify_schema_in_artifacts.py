"""Diagnose mismatched column dtypes across analysis TSV chunks.

Example
-------
$ python -m meta_spliceai.splice_engine.meta_models.examples.verify_schema_in_artifacts \
      --root /home/bchiu/work/splice-surveyor/data/ensembl/spliceai_eval/meta_models \
      --pattern "analysis_sequences_*.tsv"

Reports any file where a column's dtype differs from the inferred reference
schema (taken from the first matching file). Helps identify the root cause of
`polars.exceptions.SchemaError` during concat.
"""
from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import polars as pl

_TSV_EXTS = (".tsv", ".tsv.gz")
_CSV_EXTS = (".csv", ".csv.gz")


def _detect_sep(path: str) -> str:
    lower = path.lower()
    if lower.endswith(_TSV_EXTS):
        return "\t"
    if lower.endswith(_CSV_EXTS):
        return ","
    return "\t"


def _infer_schema(path: str) -> Dict[str, pl.datatypes.DataType]:
    """Return dict column->dtype using only the header row."""
    sep = _detect_sep(path)
    df = pl.read_csv(path, separator=sep, null_values="NA", n_rows=0)
    return dict(zip(df.columns, df.dtypes))


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect inconsistent column dtypes across TSV/CSV chunks.")
    ap.add_argument("--root", required=True, help="Directory containing the chunked artifacts.")
    ap.add_argument("--pattern", default="analysis_sequences_*.tsv", help="Glob pattern relative to --root.")
    args = ap.parse_args()

    root = Path(args.root)
    paths = sorted(glob.glob(str(root / args.pattern)))
    if not paths:
        print("No files matched pattern.")
        return

    print(f"Found {len(paths)} files. Gathering dtypes …")
    # Build dtype map col -> set[DataType]
    dtype_map: Dict[str, set] = defaultdict(set)
    for p in paths:
        for col, dt in _infer_schema(p).items():
            dtype_map[col].add(dt)
    # Hard-coded tolerant mix: chrom Int64|Utf8 is acceptable
    TOLERANT_MIX = {"chrom": {pl.Int64, pl.Utf8}}
    print("Column dtype summary (unique dtypes per column):")
    for col, dtypes in sorted(dtype_map.items()):
        print(f"  {col}: {', '.join(str(t) for t in dtypes)}")
    # Identify problematic columns
    mismatches = {
        col: list(dtypes)
        for col, dtypes in dtype_map.items()
        if len(dtypes) > 1 and dtypes != TOLERANT_MIX.get(col, set())
    }
    if not mismatches:
        print("\n✅ All columns have consistent (or tolerated) dtypes.")
        return

    print(f"\n⚠️  Columns with *un-tolerated* dtype variation ({len(mismatches)} columns):")
    for col, dtypes in mismatches.items():
        print(f"  {col}: {', '.join(str(t) for t in dtypes)}")


if __name__ == "__main__":
    main()