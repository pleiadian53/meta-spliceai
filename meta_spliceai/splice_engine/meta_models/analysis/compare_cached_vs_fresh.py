#!/usr/bin/env python3
"""Compare cached donor/acceptor probabilities with fresh predictions.

This helps detect stale cached columns that no longer match the current
meta-model weights.

Example
-------
python -m meta_spliceai.splice_engine.meta_models.analysis.compare_cached_vs_fresh \
       --dataset train_pc_1000/master \
       --run-dir runs/gene_cv_sigmoid \
       --sample 150000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _lazyframe_sample


_CACHED_COLS = ["donor_meta", "acceptor_meta"]


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Compare cached meta probabilities with fresh model predictions.")
    p.add_argument("--dataset", required=True, help="Parquet file or directory of shards")
    p.add_argument("--run-dir", required=True, help="Directory holding trained model artefacts")
    p.add_argument("--sample", type=int, default=200_000, help="Optional row sample for speed (None = all)")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    predict_fn, feature_names = _cutils._load_model_generic(run_dir)

    # --------------------------------------------------------------
    # Build LazyFrame, select required columns
    # --------------------------------------------------------------
    ds_path = Path(args.dataset)
    if ds_path.is_dir():
        lf = pl.scan_parquet(str(ds_path / "*.parquet"), missing_columns="insert")
    else:
        lf = pl.scan_parquet(str(ds_path), missing_columns="insert")

    required_cols_all = feature_names + _CACHED_COLS + ["splice_type"]
    available_cols = [c for c in required_cols_all if c in lf.columns]
    missing_cols = set(_CACHED_COLS) - set(available_cols)
    lf = lf.select(available_cols)
    if missing_cols:
        print(f"[INFO] Cached columns not found in dataset: {', '.join(missing_cols)}. Will compute fresh only.")

    if args.sample is not None:
        lf = _lazyframe_sample(lf, args.sample, seed=42)

    df = lf.collect(streaming=True).to_pandas()
    n = len(df)
    print(f"Loaded {n:,} rows for comparison (sample={args.sample}).")

    # --------------------------------------------------------------
    # Cached probabilities
    # --------------------------------------------------------------
    if set(_CACHED_COLS).issubset(df.columns):
        cached_d_max = float(df["donor_meta"].max())
        cached_a_max = float(df["acceptor_meta"].max())
        print(f"Cached donor_meta   max: {cached_d_max:.6f}")
        print(f"Cached acceptor_meta max: {cached_a_max:.6f}")
    else:
        print("Cached donor/acceptor columns not present in dataset.")
        cached_d_max = cached_a_max = float("nan")

    # --------------------------------------------------------------
    # Fresh predictions
    # --------------------------------------------------------------
    X = df[feature_names].to_numpy(dtype=np.float32)
    proba = predict_fn(X)  # shape (n, 3)

    fresh_d_max = float(proba[:, 1].max())
    fresh_a_max = float(proba[:, 2].max())
    print(f"Fresh donor_meta   max: {fresh_d_max:.6f}")
    print(f"Fresh acceptor_meta max: {fresh_a_max:.6f}\n")

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------
    if not np.isnan(cached_d_max):
        print("Donor diff (cached - fresh):", cached_d_max - fresh_d_max)
        print("Acceptor diff (cached - fresh):", cached_a_max - fresh_a_max)
        if abs(cached_d_max - fresh_d_max) < 1e-6 and abs(cached_a_max - fresh_a_max) < 1e-6:
            print("\n[OK] Cached columns are in sync with current model.")
        else:
            print("\n[WARNING] Cached columns differ from current model. Consider regenerating cached predictions.")


if __name__ == "__main__":  # pragma: no cover
    main()
