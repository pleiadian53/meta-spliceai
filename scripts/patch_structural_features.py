#!/usr/bin/env python
"""One-off patch: fill missing *structural* columns (gene / transcript level).

Operates on the numeric columns listed in ``builder_utils._FEATURE_LEVEL``.
For patching the `gene_type` column, please use `patch_gene_type.py`.

Usage
-----
conda run -n surveyor python scripts/patch_structural_features.py <DATASET_ROOT> [--inplace]

Without ``--inplace`` the patched Parquet files are written to
``<root>/master_patched``; otherwise they overwrite the originals after a
safety confirmation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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

from meta_spliceai.splice_engine.meta_models.builder import builder_utils as bu

# The set of columns we intend to fix / diagnose
FEATURE_COLS = list(getattr(bu, "_FEATURE_LEVEL").keys())


def patch_dataset(root: Path, *, inplace: bool = False) -> None:
    master_dir = root / "master"
    if not master_dir.exists():
        raise SystemExit(f"master directory not found: {master_dir}")

    out_dir = master_dir if inplace else root / "master_patched"
    if not inplace:
        out_dir.mkdir(parents=True, exist_ok=True)

    parquet_paths = sorted(master_dir.glob("*.parquet"))
    if not parquet_paths:
        raise SystemExit("No Parquet files found in master directory.")

    # Diagnostics accumulators
    null_counts = defaultdict(int)

    for p in track(parquet_paths, description="Patching structural features …"):
        df = pl.read_parquet(p)
        df = bu.fill_missing_structural_features(df)

        # accumulate null counts after patch
        for col in FEATURE_COLS:
            if col in df.columns:
                null_counts[col] += df[col].null_count()
        out_path = out_dir / p.name
        df.write_parquet(out_path)

    print(
        f"[done] Patched {len(parquet_paths)} files → {'in-place' if inplace else out_dir}"
    )

    print("[diag] Remaining nulls after patch:")
    for col in FEATURE_COLS:
        cnt = null_counts.get(col, 0)
        print(f"  • {col}: {cnt} nulls")


def main() -> None:
    parser = argparse.ArgumentParser(description="Back-fill structural features in dataset")
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
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if args.inplace:
        confirm = input(
            f"[danger] This will OVERWRITE Parquet files in {root/'master'}. Proceed? [y/N]: "
        )
        if confirm.lower() != "y":
            print("Aborted.")
            return

    patch_dataset(root, inplace=args.inplace)


if __name__ == "__main__":
    main()
