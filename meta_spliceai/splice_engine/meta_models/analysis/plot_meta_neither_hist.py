#!/usr/bin/env python3
"""Plot histogram of meta-model probabilities on *neither* rows.

This diagnostic script helps visualise how confident the meta model is on
true-negative (non-splice-site) positions.

Example
-------
python -m meta_spliceai.splice_engine.meta_models.analysis.plot_meta_neither_hist \
       --dataset train_pc_1000/master \
       --run-dir runs/gene_cv_sigmoid \
       --class donor \
       --sample 200000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _lazyframe_sample  # reuse existing util


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Plot histogram of meta probabilities on negatives (splice_type==0).")
    p.add_argument("--dataset", required=True, help="Parquet file or directory of shards used for evaluation")
    p.add_argument("--run-dir", required=True, help="Directory containing trained meta-model artefacts")
    p.add_argument("--class", dest="cls", choices=["donor", "acceptor"], default="donor")
    p.add_argument("--sample", type=int, default=100_000, help="Optional row sample for speed (None = all rows)")
    p.add_argument("--out-png", default=None, help="Output PNG path (defaults to run-dir/hist_neither_<class>.png)")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)

    # ------------------------------------------------------------------
    # Load model and feature names
    # ------------------------------------------------------------------
    predict_fn, feature_names = _cutils._load_model_generic(run_dir)

    # ------------------------------------------------------------------
    # Load dataset lazily – select only required columns
    # ------------------------------------------------------------------
    ds_path = Path(args.dataset)
    if ds_path.is_dir():
        lf = pl.scan_parquet(str(ds_path / "*.parquet"), missing_columns="insert")
    else:
        lf = pl.scan_parquet(str(ds_path), missing_columns="insert")

    required_cols = ["splice_type"] + feature_names
    lf = lf.select(required_cols)

    if args.sample is not None:
        lf = _lazyframe_sample(lf, args.sample, seed=42)

    df = lf.collect(streaming=True).to_pandas()

    # ------------------------------------------------------------------
    # Compute probabilities
    # ------------------------------------------------------------------
    X = df[feature_names].to_numpy(np.float32)
    proba = predict_fn(X)  # shape (n_rows, 3)

    class_idx = 1 if args.cls == "donor" else 2

    mask_neg = df["splice_type"].isin(["0", 0, "neither", "Neither", "NEITHER"])
    scores = proba[mask_neg, class_idx]

    print(f"max {args.cls}-prob on negatives:", scores.max())
    print(f"share ≥0.95:", (scores >= 0.95).mean())
    print(f"share ≥0.90:", (scores >= 0.90).mean())

    thr = 0.03          # replace with real donor/acceptor thresholds
    print(f"share ≥ {thr}:", (scores >= thr).mean())
    

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    bins = np.linspace(0.0, 1.0, 51)
    plt.hist(scores, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
    plt.title(f"Meta probabilities on negatives – class={args.cls}  n={scores.size:,}")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.tight_layout()

    if args.out_png is None:
        args.out_png = run_dir / f"hist_neither_{args.cls}.png"
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"[plot_neither_hist] wrote → {out_path}")

    # ------------------------------------------------------------------
    # Tail / log-scale plot for rare high-probability negatives
    # ------------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    bins_tail = np.linspace(0.8, 1.0, 41)
    plt.hist(scores, bins=bins_tail, color="darkorange", edgecolor="black", alpha=0.8)
    plt.xlabel("Probability (0.8 – 1.0)")
    plt.ylabel("Count (log scale)")
    plt.yscale("log")
    plt.title(f"High-tail on negatives – class={args.cls}")
    plt.tight_layout()

    tail_path = out_path.with_name(out_path.stem + "_tail" + out_path.suffix)
    plt.savefig(tail_path, dpi=150)
    print(f"[plot_neither_hist] wrote → {tail_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
