#!/usr/bin/env python3
"""Compare SpliceAI raw probabilities vs meta-model probabilities **per gene**.

The goal is to see *where* (which genes) the meta-model actually changes the
predicted splice propensity.  It aggregates both the raw SpliceAI max score and
the meta-model probability for every position, then computes gene-level means
and accuracy deltas.

Output CSV columns
------------------
* gene_id
* n_sites               – number of positions for this gene (after any sampling)
* mean_p_spliceai       – mean of max(raw_donor, raw_acceptor, raw_neither)
* mean_p_meta           – mean of meta-model probability
* delta_mean_prob
* acc_spliceai          – fraction of positions SpliceAI calls *splice site* (donor/acceptor)
* acc_meta              – same for meta-model (`p_meta >= threshold`)
* delta_acc

Usage
-----
$ conda run -n surveyor python -m \
    meta_spliceai.splice_engine.meta_models.training.gene_score_delta \
    --dataset models/xgb_pc1000/dataset_trimmed.parquet \
    --model-dir models/xgb_pc1000 \
    --out-csv models/xgb_pc1000/gene_score_delta.csv \
    --threshold 0.5 --sample 100000
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import polars as pl

from meta_spliceai.splice_engine.meta_models.builder import preprocessing as _prep

__all__: List[str] = [
    "compute_gene_score_delta",
]


RAW_PROB_COLS = ["donor_score", "acceptor_score", "neither_score"]


def _load_model(model_dir: Path) -> object:  # noqa: D401 – returns estimator
    with open(model_dir / "model.pkl", "rb") as fh:
        return pickle.load(fh)


def compute_gene_score_delta(
    *,
    dataset_path: str | Path,
    model_dir: str | Path,
    threshold: float = 0.5,
    sample: int | None = None,
) -> pd.DataFrame:
    """Return a gene-level comparison DataFrame as described above."""
    dataset_path = Path(dataset_path)
    model_dir = Path(model_dir)

    # ------------------------------------------------------------------
    # Load artefacts needed to reconstruct X in the original feature order
    # ------------------------------------------------------------------
    feature_manifest = pd.read_csv(model_dir / "feature_manifest.csv")
    feature_cols: list[str] = feature_manifest["feature"].tolist()

    model = _load_model(model_dir)

    # ------------------------------------------------------------------
    # Load dataset – only required columns to keep memory low
    # ------------------------------------------------------------------
    # Combine columns while preserving order and removing duplicates
    select_cols = list(dict.fromkeys([
        *feature_cols,
        *RAW_PROB_COLS,
        "gene_id",
        "splice_type",
    ]))

    if dataset_path.is_dir():
        lf = pl.scan_parquet(str(dataset_path / "*.parquet"), missing_columns="insert")
    else:
        lf = pl.scan_parquet(str(dataset_path), missing_columns="insert")

    # Filter select_cols to only include columns that actually exist in the dataset
    available_cols = lf.collect_schema().names()
    missing_cols = [col for col in select_cols if col not in available_cols]
    if missing_cols:
        print(f"Warning: Missing columns in dataset (will be added with zeros later): {missing_cols}")
    existing_select_cols = [col for col in select_cols if col in available_cols]
    
    lf = lf.select(existing_select_cols)

    if sample is not None and sample > 0:
        try:
            lf = lf.sample(n=sample, seed=42)
        except AttributeError:
            lf = lf.limit(sample)

    # Collect to pandas for model inference – limited columns so memory is OK
    df = lf.collect(streaming=True).to_pandas()

    # ------------------------------------------------------------------
    # Prepare X for meta-model (same preprocessing as during training)
    # ------------------------------------------------------------------
    X_df, _ = _prep.prepare_training_data(
        pl.from_pandas(df),  # reuse existing pipeline (expects polars)
        label_col="splice_type",
        return_type="pandas",
    )

    # Ensure all expected features exist in the dataset
    for col in feature_cols:
        if col not in X_df.columns:
            print(f"Warning: Adding missing column '{col}' with zeros")
            X_df[col] = 0.0
    
    # Ensure column order matches training
    X_df = X_df[feature_cols]
    p_meta = model.predict_proba(X_df.values)[:, 1]

    # ------------------------------------------------------------------
    # Compute per-row metrics
    # ------------------------------------------------------------------
    p_max_spliceai = df[RAW_PROB_COLS].max(axis=1)
    type_spliceai = df[RAW_PROB_COLS].idxmax(axis=1).str.replace("_score", "", regex=False)

    pred_spliceai = (type_spliceai != "neither").astype(int)
    pred_meta = (p_meta >= threshold).astype(int)

    splice_col = df["splice_type"]
    if pd.api.types.is_numeric_dtype(splice_col):
        true_label = (splice_col.values > 0).astype(int)
    else:
        true_label = (splice_col != "neither").astype(int)

    # Build aggregation DataFrame
    agg_df = pd.DataFrame({
        "gene_id": df["gene_id"],
        "p_spliceai": p_max_spliceai,
        "p_meta": p_meta,
        "acc_spliceai": (pred_spliceai == true_label).astype(int),
        "acc_meta": (pred_meta == true_label).astype(int),
    })

    grouped = agg_df.groupby("gene_id").agg(
        n_sites=("gene_id", "size"),
        mean_p_spliceai=("p_spliceai", "mean"),
        mean_p_meta=("p_meta", "mean"),
        acc_spliceai=("acc_spliceai", "mean"),
        acc_meta=("acc_meta", "mean"),
    ).reset_index()

    grouped["delta_mean_prob"] = grouped["mean_p_meta"] - grouped["mean_p_spliceai"]
    grouped["delta_acc"] = grouped["acc_meta"] - grouped["acc_spliceai"]

    return grouped.sort_values("delta_mean_prob", ascending=False)


def _main(argv: list[str] | None = None) -> None:  # pragma: no cover – CLI only
    p = argparse.ArgumentParser(description="Compare per-gene SpliceAI vs meta-model scores.")
    p.add_argument("--dataset", required=True, help="Parquet dataset (file or directory)")
    p.add_argument("--model-dir", required=True, help="Directory containing model.pkl & feature_manifest.csv")
    p.add_argument("--out-csv", required=True, help="Where to write the per-gene comparison CSV")
    p.add_argument("--threshold", type=float, default=0.5, help="meta-model decision threshold (default 0.5)")
    p.add_argument("--sample", type=int, default=None, help="optional row sample for speed")
    args = p.parse_args(argv)

    try:
        res_df = compute_gene_score_delta(
            dataset_path=args.dataset,
            model_dir=args.model_dir,
            threshold=args.threshold,
            sample=args.sample,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[gene_score_delta] ERROR: {exc}")
        sys.exit(2)

    out_path = Path(args.out_csv)
    res_df.to_csv(out_path, index=False)
    print(f"[gene_score_delta] wrote {len(res_df)} rows → {out_path}")
    print(res_df.head(20).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    _main()
