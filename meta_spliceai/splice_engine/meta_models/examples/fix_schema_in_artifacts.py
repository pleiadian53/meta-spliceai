"""Quick schema-fix utility for analysis_sequences chunks.

Main goal: make `chrom` uniformly Utf8 so numeric chromosomes ("1", "2") and
non-numeric ones ("X", "Y", "MT") coexist without forcing the column to Int64
in some files. Optionally enforces Int64 on *_is_local_peak columns.

Usage
-----
$ python -m meta_spliceai.splice_engine.meta_models.examples.fix_schema_in_artifacts \
      --root /path/to/meta_models \
      --pattern "analysis_sequences_*.tsv" \
      --backup
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
from pathlib import Path

import polars as pl


def _fix_file(path: Path, backup: bool = False) -> None:
    # Determine delimiter ignoring optional .gz compression suffix
    fname = path.name.lower().removesuffix(".gz")
    sep = "\t" if fname.endswith(".tsv") else ","
    original = pl.read_csv(path, separator=sep, null_values="NA")

    df = original.with_columns(
        pl.when(pl.col("chrom").is_not_null())
          .then(pl.col("chrom").cast(pl.Utf8, strict=False).str.replace_all(r"^chr", ""))
          .otherwise(pl.col("chrom"))
          .alias("chrom")
    )

    # Cast peak columns; track whether dtype changed
    modified = False
    for col in ("donor_is_local_peak", "acceptor_is_local_peak"):
        if col in df.columns:
            if df[col].dtype != pl.Int64:
                df = df.with_columns(pl.col(col).cast(pl.Int64, strict=False))
                modified = True

    # Detect if chrom dtype changed
    if "chrom" in df.columns and df["chrom"].dtype != original["chrom"].dtype:
        modified = True

    if not modified:
        return False  # nothing to write

    if backup:
        shutil.copy2(path, f"{path}.bak")
    df.write_csv(path, separator=sep)
    print("✔ fixed", path.name)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Force consistent dtypes in analysis chunks.")
    ap.add_argument("--root", required=True, help="Directory containing TSV/CSV chunks.")
    ap.add_argument("--pattern", default="analysis_sequences_*.tsv", help="Glob pattern relative to --root.")
    ap.add_argument("--backup", action="store_true", help="Keep a *.bak copy before overwriting.")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(glob.glob(str(root / args.pattern)))
    if not files:
        print("No matching files – nothing to do.")
        return

    print(f"Processing {len(files)} files …")
    fixed = backed_up = 0
    for f in files:
        if _fix_file(Path(f), backup=args.backup):
            fixed += 1
            if args.backup:
                backed_up += 1
    print(f"Done. {fixed} file(s) modified, {backed_up} backups created.")


if __name__ == "__main__":
    main()