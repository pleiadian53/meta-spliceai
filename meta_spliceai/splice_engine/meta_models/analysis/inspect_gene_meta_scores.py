#!/usr/bin/env python3
"""Inspect high-scoring meta-model rows for a single gene.

Prints the top-N rows (by probability) for the requested class so you can
see which positions are producing FP spikes.

Example
-------
python -m meta_spliceai.splice_engine.meta_models.analysis.inspect_gene_meta_scores \
       --dataset train_pc_1000/master \
       --run-dir runs/gene_cv_sigmoid \
       --gene ENSG00000087053 \
       --class donor \
       --top 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _lazyframe_sample


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Show top-scoring meta rows for a gene.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--gene", required=True, help="Ensembl gene_id to inspect")
    p.add_argument("--class", dest="cls", choices=["donor", "acceptor"], default="donor")
    p.add_argument("--top", type=int, default=20, help="Number of rows to show")
    p.add_argument("--sample", type=int, default=None, help="Optional random sample before filtering (speed)")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)

    predict_fn, feature_names = _cutils._load_model_generic(run_dir)

    ds_path = Path(args.dataset)
    if ds_path.is_dir():
        lf = pl.scan_parquet(str(ds_path / "*.parquet"), missing_columns="insert")
    else:
        lf = pl.scan_parquet(str(ds_path), missing_columns="insert")

    base_cols = [
        "gene_id",
        "splice_type",
        "strand",
        "gene_start",
        "gene_end",
    ]
    coord_candidates = ["rel_pos", "position"]
    desired_cols = list(dict.fromkeys(base_cols + coord_candidates + feature_names))  # preserve order, drop dups
    available_cols = [c for c in desired_cols if c in lf.columns]
    lf = lf.select(available_cols)

    # pushdown: filter by gene before collect to minimise memory
    lf = lf.filter(pl.col("gene_id") == args.gene)

    if args.sample is not None:
        lf = _lazyframe_sample(lf, args.sample, seed=42)

    df = lf.collect(streaming=True).to_pandas()
    if df.empty:
        print(f"[inspect_gene] No rows for gene {args.gene}")
        return

    X = df[feature_names].to_numpy(np.float32)
    proba = predict_fn(X)

    class_idx = 1 if args.cls == "donor" else 2
    df["meta_prob"] = proba[:, class_idx]

    df_sorted = df.sort_values("meta_prob", ascending=False).head(args.top)
    coord_col = "rel_pos" if "rel_pos" in df_sorted.columns else ("position" if "position" in df_sorted.columns else None)
    cols_to_show = [c for c in [coord_col, "splice_type", "meta_prob"] if c]
    print(df_sorted[cols_to_show].to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":  # pragma: no cover
    main()
