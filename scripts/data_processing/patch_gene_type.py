#!/usr/bin/env python
"""One-off patch: fill missing *gene_type* in existing Parquet dataset.

Usage
-----
conda run -n surveyor python scripts/patch_gene_type.py <DATASET_ROOT> [--inplace]

DATASET_ROOT is the directory produced by *incremental_builder* (e.g.
``train_pc_1000``).  The script processes all ``*.parquet`` files inside the
``<root>/master`` sub-directory, fills missing ``gene_type`` values using the
lookup table ``gene_features.tsv`` and saves the patched files to
``<root>/master_patched`` by default.  Use ``--inplace`` to overwrite the
original files *after* a safety confirmation.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys, pathlib

# ------------------------------------------------------------------
# Ensure project root (directory containing 'meta_spliceai') is on sys.path
# ------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if (PROJECT_ROOT / "meta_spliceai").exists():
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
from rich.progress import track

from meta_spliceai.splice_engine.meta_models.builder.builder_utils import (
    fill_missing_gene_type,
)


def patch_dataset(root: Path, *, inplace: bool = False, expected: str = "protein_coding") -> None:
    master_dir = root / "master"
    if not master_dir.exists():
        raise SystemExit(f"master directory not found: {master_dir}")

    out_dir = master_dir if inplace else root / "master_patched"
    if not inplace:
        out_dir.mkdir(parents=True, exist_ok=True)

    parquet_paths = sorted(master_dir.glob("*.parquet"))
    if not parquet_paths:
        raise SystemExit("No Parquet files found in master directory.")

    violating_genes: set[str] = set()
    missing_in_tsv: set[str] = set()

    for p in track(parquet_paths, description="Patching gene_type …"):
        df = pl.read_parquet(p)
        df = fill_missing_gene_type(df)
        # collect diagnostics before writing
        sub = df.select(["gene_id", "gene_type"]).unique()
        violating_genes.update(
            sub.filter(~pl.col("gene_type").eq(expected)).select("gene_id").to_series().to_list()
        )
        missing_in_tsv.update(
            sub.filter(pl.col("gene_type").eq("unknown")).select("gene_id").to_series().to_list()
        )
        out_path = out_dir / p.name
        df.write_parquet(out_path)

    print(
        f"[done] Patched {len(parquet_paths)} files → {'in-place' if inplace else out_dir}"
    )

    print(f"[diag] Genes with gene_type != '{expected}': {len(violating_genes)}")
    if violating_genes:
        sample = sorted(list(violating_genes))[:10]
        print("       examples:", ", ".join(sample), "…" if len(violating_genes) > 10 else "")

    print(f"[diag] Genes missing in gene_features.tsv (gene_type='unknown'): {len(missing_in_tsv)}")
    if missing_in_tsv:
        sample2 = sorted(list(missing_in_tsv))[:10]
        print("       examples:", ", ".join(sample2), "…" if len(missing_in_tsv) > 10 else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Back-fill missing gene_type in dataset")
    parser.add_argument(
        "root",
        help=(
            "Dataset root directory. If given as an absolute path (starts with '/' or '~'), it is "
            "used as-is; otherwise it is resolved relative to the current working directory "
            "(e.g. 'train_pc_1000')."
        ),
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite files inside <root>/master instead of writing to <root>/master_patched>",
    )
    parser.add_argument(
        "--expected-gene-type",
        default="protein_coding",
        help="Expected dominant gene_type for diagnostics (default: protein_coding)",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if args.inplace:
        confirm = input(
            f"[danger] This will OVERWRITE Parquet files in {root/'master'}. Proceed? [y/N]: "
        )
        if confirm.lower() != "y":
            print("Aborted.")
            return

    patch_dataset(root, inplace=args.inplace, expected=args.expected_gene_type)


if __name__ == "__main__":
    main()
