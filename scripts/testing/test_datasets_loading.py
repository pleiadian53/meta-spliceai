#!/usr/bin/env python
"""Quick sanity-check script for meta-model dataset utilities.

Usage
-----
    conda run -n surveyor python scripts/test_datasets_loading.py [DATASET_DIR]

Without an argument it defaults to ``train_pc_1000`` in the project root.

The script exercises the high-level helpers in
``splice_engine.meta_models.training.datasets`` to ensure that:

1. Polars can read the (possibly incomplete) master Parquet dataset.
2. ``prepare_xy`` successfully converts the frame into feature matrix *X* and
   label vector *y* with the updated label mapping ("neither"/"0"/etc.).
3. ``train_valid_test_split`` works and preserves class stratification.

It prints basic diagnostics: shapes, class counts, and a preview of *X*.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from meta_spliceai.splice_engine.meta_models.training import datasets as ds

pl.Config.set_tbl_rows(10)  # nicer console preview


def main(dataset_root: str | Path = "train_pc_1000") -> None:
    root = Path(dataset_root)
    master_dir = root / "master"

    if not master_dir.exists():
        raise SystemExit(f"Master dataset dir not found: {master_dir}")

    print("\n[step] Loading Parquet dataset …")
    df = ds.load_dataset(master_dir)
    print(df)
    print(f"[info] Loaded DataFrame with shape: {df.shape}\n")

    # ------------------------------------------------------------------
    # Column-wise statistics
    # ------------------------------------------------------------------
    print("[step] Column statistics …")

    # Build a small helper DF for numeric stats to avoid huge prints
    numeric_cols = [c for c, d in df.schema.items() if d in (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    )]
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    if numeric_cols:
        num_stats = df.select(numeric_cols).describe()
        float_cols = [c for c, dt in num_stats.schema.items() if dt in (pl.Float32, pl.Float64)]
        if float_cols:
            num_stats = num_stats.with_columns([pl.col(c).round(3) for c in float_cols])
        # Polars puts stats as rows; transpose for readability
        print("\nNumeric summary (mean/std/min/median/max):")
        print(num_stats)

    print("\nCategorical summary (unique count + top-5 frequencies):")
    for col in cat_cols:
        ser = df[col]
        uniq = ser.n_unique()
        print(f"  • {col}: {uniq} unique, nulls={ser.null_count()}")
        top = (
            df.select(col)
            .group_by(col)
            .len()
            .sort("len", descending=True)
            .head(5)
        )
        for val, cnt in top.iter_rows():
            print(f"     {repr(val)}: {cnt}")


        # ------------------------------------------------------------------
    # Persist summary to disk
    # ------------------------------------------------------------------
    summary_path = root / "column_stats.txt"
    with summary_path.open("w") as fp:
        fp.write("Numeric summary (per-column):\n")
        for col in numeric_cols:
            s = df[col]
            try:
                fp.write(
                    f"  {col}: count={s.len()}, nulls={s.null_count()}, "
                    f"mean={s.mean():.3g}, std={s.std():.3g}, "
                    f"min={s.min():.3g}, median={s.median():.3g}, max={s.max():.3g}\n"
                )
            except Exception as exc:  # fallback if stat not supported
                fp.write(f"  {col}: stats error -> {exc}\n")
        fp.write("\n")
        fp.write("Categorical summary (unique count + top-5 frequencies):\n")
        for col in cat_cols:
            ser = df[col]
            uniq = ser.n_unique()
            fp.write(f"  • {col}: {uniq} unique, nulls={ser.null_count()}\n")
            top = (
                df.select(col)
                .group_by(col)
                .len()
                .sort("len", descending=True)
                .head(5)
            )
            for val, cnt in top.iter_rows():
                fp.write(f"     {repr(val)}: {cnt}\n")
    print(f"[info] Column statistics written to {summary_path}\n")

    print("[step] Converting to X, y …")
    X, y = ds.prepare_xy(df, return_X="polars")
    print("   X shape:", X.shape)
    unique, counts = np.unique(y, return_counts=True)
    print("   y distribution:", dict(zip(unique, counts)))

    print("\n[step] Train/valid/test split …")
    X_tr, X_val, X_te, y_tr, y_val, y_te = ds.train_valid_test_split(
        X, y, test_size=0.2, valid_size=0.1, random_state=0
    )
    print("   Train:", X_tr.shape, "Valid:", X_val.shape, "Test:", X_te.shape)

    print("\n[success] Dataset helpers appear to be working.\n")


if __name__ == "__main__":
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else "train_pc_1000"
    main(dataset_arg)
