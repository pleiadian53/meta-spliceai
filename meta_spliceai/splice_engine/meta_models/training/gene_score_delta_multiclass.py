#!/usr/bin/env python3
"""Compare per-gene probability scores before/after a **3-class** meta-model.

Outputs a CSV with, for each gene_id:

* number of candidate positions (n_sites)
* mean donor / acceptor / neither probabilities from **raw SpliceAI**
* mean donor / acceptor / neither probabilities from **meta-model**
* mean of channel-max probability for raw vs meta (p_max)
* accuracy of raw vs meta predictions (argmax) and the delta
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl

from meta_spliceai.splice_engine.meta_models.builder import preprocessing as _prep
from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.training.label_utils import LABEL_MAP_STR, encode_labels

# ---------------------------------------------------------------------------
# Backward-compatibility alias – remove after full refactor
# ---------------------------------------------------------------------------
_LABEL_MAP_STR = LABEL_MAP_STR

RAW_PROB_COLS = ["donor_score", "acceptor_score", "neither_score"]



def _load_model(model_dir: Path):
    """Load multiclass meta-model trained either via in-RAM (pickle) or
    external-memory (native XGBoost JSON).
    Returns a callable with signature `(np.ndarray) -> np.ndarray` that outputs
    class probabilities shape (n, 3).
    """
    pkl_path = model_dir / "model_multiclass.pkl"
    json_path = model_dir / "model_multiclass.json"
    if pkl_path.exists():
        with open(pkl_path, "rb") as fh:
            model = pickle.load(fh)
        return model.predict_proba
    elif json_path.exists():
        import xgboost as xgb  # local import to keep deps optional
        booster = xgb.Booster()
        booster.load_model(str(json_path))

        # load feature names for validation
        csv_path_feat = model_dir / "feature_manifest.csv"
        json_path_feat = model_dir / "train.features.json"
        if csv_path_feat.exists():
            feature_names = pd.read_csv(csv_path_feat)["feature"].tolist()
        elif json_path_feat.exists():
            feature_names = json.loads(json_path_feat.read_text())["feature_names"]
        else:
            feature_names = None

        def _predict(X: np.ndarray) -> np.ndarray:
            if feature_names is not None:
                dmat = xgb.DMatrix(X, feature_names=feature_names, missing=np.nan)
            else:
                dmat = xgb.DMatrix(X, missing=np.nan)
            return booster.predict(dmat)
        return _predict
    else:
        raise FileNotFoundError("No model_multiclass.pkl or .json found in model_dir")


def _encode_labels(s: pd.Series) -> np.ndarray:
    """Wrapper around label_utils.encode_labels (kept for backward compatibility)."""
    return encode_labels(s.to_numpy())


def compute_gene_score_delta_multiclass(
    *,
    dataset_path: str | Path,
    model_dir: str | Path,
    sample: int | None = None,
) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    model_dir = Path(model_dir)

    # Load feature manifest (CSV or JSON)
    csv_path = model_dir / "feature_manifest.csv"
    json_path = model_dir / "train.features.json"  # created by convert_dataset_to_libsvm
    if csv_path.exists():
        feature_manifest = pd.read_csv(csv_path)
        feature_cols: List[str] = feature_manifest["feature"].tolist()
    elif json_path.exists():
        feature_cols = json.loads(Path(json_path).read_text())["feature_names"]
    else:
        raise FileNotFoundError("feature_manifest.csv or train.features.json not found in model_dir")

    predict_fn = _load_model(model_dir)

    # Columns needed from dataset
    select_cols = list(
        dict.fromkeys([
            *feature_cols,
            *RAW_PROB_COLS,
            "gene_id",
            "splice_type",
        ])
    )

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

    if sample and sample > 0:
        try:
            lf = lf.sample(n=sample, seed=42)
        except AttributeError:
            lf = lf.limit(sample)

    df = lf.collect(streaming=True).to_pandas()

    # Drop rows with missing or unrecognised splice_type to avoid NaNs downstream
    df = df.dropna(subset=["splice_type"])
    df = df[df["splice_type"].isin(_LABEL_MAP_STR.keys())]

    # Build X for meta-model
    X_df, _ = _prep.prepare_training_data(
        pl.from_pandas(df),
        label_col="splice_type",
        return_type="pandas",
    )
    
    # Ensure all expected features exist in the dataset
    for col in feature_cols:
        if col not in X_df.columns:
            print(f"Warning: Adding missing column '{col}' with zeros")
            X_df[col] = 0.0
    
    # Handle non-numeric columns like 'chrom' to preserve feature count compatibility
    for col in feature_cols:
        if col in X_df.columns:
            try:
                X_df[col] = pd.to_numeric(X_df[col], errors='raise')
            except (ValueError, TypeError):
                print(f"Warning: Converting non-numeric column '{col}' to dummy values")
                unique_vals = X_df[col].dropna().unique()
                val_map = {val: idx for idx, val in enumerate(unique_vals)}
                X_df[col] = X_df[col].map(val_map).fillna(-1).astype(float)
    
    # Now select all features, which should all be available and numeric
    X_df = X_df[feature_cols]

    p_meta = predict_fn(X_df.values)  # shape (n, 3)

    # Per-row computations
    p_max_spliceai = df[RAW_PROB_COLS].max(axis=1)
    p_max_meta = p_meta.max(axis=1)

    raw_pred = df[RAW_PROB_COLS].idxmax(axis=1).map({
        "donor_score": 1,
        "acceptor_score": 2,
        "neither_score": 0,
    }).to_numpy()
    meta_pred = p_meta.argmax(axis=1)

    y_true = _encode_labels(df["splice_type"])

    # Build aggregation frame with per-row fields
    agg = pd.DataFrame({
        "gene_id": df["gene_id"],
        "donor_raw": df["donor_score"],
        "acceptor_raw": df["acceptor_score"],
        "neither_raw": df["neither_score"],
        "donor_meta": p_meta[:, 1],
        "acceptor_meta": p_meta[:, 2],
        "neither_meta": p_meta[:, 0],
        "p_max_raw": p_max_spliceai,
        "p_max_meta": p_max_meta,
        "acc_raw": (raw_pred == y_true).astype(int),
        "acc_meta": (meta_pred == y_true).astype(int),
        "y_true": y_true,
        "raw_pred": raw_pred,
        "meta_pred": meta_pred,
    })

    basic = agg.groupby("gene_id").agg(
        n_sites=("gene_id", "size"),
        mean_donor_raw=("donor_raw", "mean"),
        mean_donor_meta=("donor_meta", "mean"),
        mean_acceptor_raw=("acceptor_raw", "mean"),
        mean_acceptor_meta=("acceptor_meta", "mean"),
        mean_neither_raw=("neither_raw", "mean"),
        mean_neither_meta=("neither_meta", "mean"),
        mean_p_max_raw=("p_max_raw", "mean"),
        mean_p_max_meta=("p_max_meta", "mean"),
        acc_raw=("acc_raw", "mean"),
        acc_meta=("acc_meta", "mean"),
    )

    # per-gene precision/recall/F1 and fix/regress counts
    from sklearn.metrics import precision_score, recall_score, f1_score
    rows = []
    for gid, sub in agg.groupby("gene_id"):
        y = sub["y_true"].to_numpy()
        r_pred = sub["raw_pred"].to_numpy()
        m_pred = sub["meta_pred"].to_numpy()
        # --- top-k accuracy (hit@k) ---
        def _hit_at_k(scores: np.ndarray, k: int) -> int:
            if len(scores) <= k:
                k = len(scores)
            # indices sorted descending by score
            top_idx = np.argpartition(-scores, kth=k-1)[:k]
            return int((y[top_idx] != 0).any())

        # combined splice probability used for ranking
        raw_splice_score = np.maximum(sub["donor_raw"].to_numpy(), sub["acceptor_raw"].to_numpy())
        meta_splice_score = np.maximum(sub["donor_meta"].to_numpy(), sub["acceptor_meta"].to_numpy())

        # --- splice-site binary labels (donor OR acceptor = 1, neither = 0)
        y_bin = (y != 0).astype(int)
        r_bin = (r_pred != 0).astype(int)
        m_bin = (m_pred != 0).astype(int)

        metrics_row = {
            "gene_id": gid,
            "precision_raw": precision_score(y, r_pred, average="macro", zero_division=0),
            "precision_meta": precision_score(y, m_pred, average="macro", zero_division=0),
            "recall_raw": recall_score(y, r_pred, average="macro", zero_division=0),
            "recall_meta": recall_score(y, m_pred, average="macro", zero_division=0),
            "f1_raw": f1_score(y, r_pred, average="macro", zero_division=0),
            "f1_meta": f1_score(y, m_pred, average="macro", zero_division=0),
            "precision_splice_raw": precision_score(y_bin, r_bin, zero_division=0),
            "precision_splice_meta": precision_score(y_bin, m_bin, zero_division=0),
            "recall_splice_raw": recall_score(y_bin, r_bin, zero_division=0),
            "recall_splice_meta": recall_score(y_bin, m_bin, zero_division=0),
            "f1_splice_raw": f1_score(y_bin, r_bin, zero_division=0),
            "f1_splice_meta": f1_score(y_bin, m_bin, zero_division=0),
            "n_fixed": int(((r_pred != y) & (m_pred == y)).sum()),
            "n_regressed": int(((r_pred == y) & (m_pred != y)).sum()),
        }
        for k in (1, 2, 5):
            metrics_row[f"top{k}_raw"] = _hit_at_k(raw_splice_score, k)
            metrics_row[f"top{k}_meta"] = _hit_at_k(meta_splice_score, k)
            metrics_row[f"delta_top{k}"] = metrics_row[f"top{k}_meta"] - metrics_row[f"top{k}_raw"]

        rows.append(metrics_row)
    metrics_df = pd.DataFrame(rows).set_index("gene_id")

    grouped = basic.join(metrics_df, how="left")

    # delta columns for macro metrics
    for base in ["precision", "recall", "f1", "precision_splice", "recall_splice", "f1_splice"]:
        grouped[f"delta_{base}"] = grouped[f"{base}_meta"] - grouped[f"{base}_raw"]

    # delta for probability-derived means and accuracy
    for ch in ("donor", "acceptor", "neither", "p_max", "acc"):
        grouped[f"delta_{ch}"] = grouped[f"mean_{ch}_meta" if ch != "acc" else "acc_meta"] - grouped[
            f"mean_{ch}_raw" if ch != "acc" else "acc_raw"
        ]

    grouped = grouped.reset_index()

    return grouped.sort_values("delta_f1", ascending=False)


def _main(argv: List[str] | None = None) -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Per-gene score delta for multiclass meta-model.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model-dir", "--run-dir", dest="model_dir", required=True, help="Directory containing trained model artefacts (alias: --run-dir)")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--sample", type=int, default=None, help="Optional row sample for speed")
    args = p.parse_args(argv)

    res_df = compute_gene_score_delta_multiclass(
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        sample=args.sample,
    )
    res_df.to_csv(args.out_csv, index=False)
    print(f"[gene_score_delta_multiclass] wrote {len(res_df):,} rows → {args.out_csv}")


if __name__ == "__main__":
    _main()
