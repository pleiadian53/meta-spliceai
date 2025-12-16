#!/usr/bin/env python3
"""Train a *three-class* meta-model (donor / acceptor / neither).

Three execution modes are supported, from quick debugging to full-scale
external-memory training.

1. Quick subset (default)
   Uses the built-in ``--row-cap`` (100 000 rows by default) so that the
   feature matrix fits comfortably in ≤10 GB RAM.
   ``SS_MAX_ROWS`` provides the same cap as an environment variable.

   Example quick run (CPU, default settings):
   ```bash
   python -m meta_spliceai.splice_engine.meta_models.training.demo_train_meta_model_multiclass \
       --dataset train_pc_1000/master \
       --out-dir runs/multiclass_quick
   ```

2. Full in-memory training
   Disable the cap with ``--row-cap 0`` **and** make sure ``SS_MAX_ROWS`` is
   unset.  Combine with `hist` / `gpu_hist` to reduce RAM.

   CPU histogram:
   ```bash
   python -m meta_spliceai.splice_engine.meta_models.training.demo_train_meta_model_multiclass \
       --dataset train_full/master \
       --out-dir runs/multiclass_hist \
       --row-cap 0 \
       --tree-method hist --max-bin 256
   ```

   GPU histogram:
   ```bash
   python -m meta_spliceai.splice_engine.meta_models.training.demo_train_meta_model_multiclass \
       --dataset train_full/master \
       --out-dir runs/multiclass_gpu \
       --row-cap 0 \
       --tree-method gpu_hist --max-bin 256
   ```

3. External-memory training (no row cap, lowest RAM)
   Streams the entire dataset from disk by first converting to LibSVM and then
   using XGBoost external-memory.

   ```bash
   python -m meta_spliceai.splice_engine.meta_models.training.demo_train_meta_model_multiclass \
       --dataset train_full/master \
       --out-dir runs/multiclass_extmem \
       --tree-method hist --max-bin 256 \
       --external-memory
   ```

Notes
-----
• ``SS_MAX_ROWS`` (if set) overrides ``--row-cap``.  Use ``export SS_MAX_ROWS=0``
  or ``unset SS_MAX_ROWS`` to disable capping completely.
• Only histogram algorithms (`hist` / `gpu_hist`) honour ``--max-bin``.
• The trained model, feature manifest, and metrics are saved in *out-dir*.
"""
from __future__ import annotations

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from meta_spliceai.splice_engine.meta_models.builder import preprocessing
from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.training.label_utils import LABEL_MAP_STR, encode_labels

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------




def _encode_labels(y_raw: pd.Series | np.ndarray) -> np.ndarray:
    """Wrapper around label_utils.encode_labels (kept for compatibility)."""
    if isinstance(y_raw, pd.Series):
        arr = y_raw.to_numpy()
    else:
        arr = y_raw
    return encode_labels(arr)


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def _split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    *,
    test_size: float,
    valid_size: float,
    random_state: int,
) -> Tuple[np.ndarray, ...]:
    return datasets.train_valid_test_split(
        X,
        y,
        test_size=test_size,
        valid_size=valid_size,
        random_state=random_state,
        groups=groups,
    )


def main(argv: list[str] | None = None) -> None:  # pragma: no cover – CLI only
    p = argparse.ArgumentParser(description="Multiclass (donor / acceptor / neither) meta-model demo.")
    p.add_argument("--dataset", required=True, help="Master dataset directory or single Parquet file")
    p.add_argument("--out-dir", "--run-dir", dest="out_dir", required=True, help="Directory to save model, metrics, and artefacts (alias: --run-dir)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group-col", dest="group_col", default="gene_id", help="Column used to define non-overlapping groups when splitting (e.g. gene_id, chrom)")
    p.add_argument("--holdout-chroms", nargs="*", default=[], help="If provided, hold out these chromosome names entirely for TEST set (requires --group-col chrom)")
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--valid-size", type=float, default=0.15)
    p.add_argument("--row-cap", type=int, default=100_000, help="Optional row cap to keep memory bounded (set 0 to disable)")
    
    # --tree-method selects the underlying XGBoost algorithm:
    #   auto     : let XGBoost decide based on data & hardware (classic default)
    #   hist     : force CPU histogram algorithm (memory-friendly, scales to wide data)
    #   gpu_hist : force GPU histogram algorithm (fast if dataset fits GPU VRAM)
    p.add_argument(
        "--tree-method",
        choices=["auto", "hist", "gpu_hist"],
        default="auto",
        help="XGBoost tree_method – 'hist'/'gpu_hist' scale better on wide data; 'auto' lets XGBoost choose automatically",
    )
    # When tree_method=hist/gpu_hist, max_bin is required
    
    p.add_argument(
        "--max-bin",
        type=int,
        default=256,
        help="Number of histogram bins (only relevant when tree_method resolves to hist/gpu_hist)",
    )
    p.add_argument("--external-memory", action="store_true", help="Train with XGBoost external-memory from on-disk LibSVM; processes full dataset and overrides row cap")
    p.add_argument("--n-estimators", type=int, default=600, help="Max boosting rounds (upper bound when using early stopping)")
    p.add_argument("--early-stopping", type=int, default=0, help="Enable early stopping with given patience (validation fraction fixed at 10%)")
    p.add_argument("--leakage-probe", action="store_true", help="Run correlation-based leakage probe (may be slow)")
    p.add_argument("--delta-sample", type=int, default=100000, help="Row sample for gene score delta calculation (0 = full dataset)")
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------
    # External-memory fast-path
    # -------------------------------------
    if getattr(args, "external_memory", False):
        print("[demo] External-memory mode enabled – full dataset will be used; ignoring --row-cap")
        from meta_spliceai.splice_engine.meta_models.training.external_memory_utils import (
            convert_dataset_to_libsvm,
            train_external_memory_model,
        )
        libsvm_path = convert_dataset_to_libsvm(args.dataset, out_dir / "train.libsvm", verbose=1)
        model_path, evals = train_external_memory_model(
            libsvm_path,
            model_out_dir=out_dir,
            tree_method=args.tree_method,
            max_bin=args.max_bin,
            n_estimators=args.n_estimators,
            early_stopping_rounds=args.early_stopping if args.early_stopping > 0 else None,
            random_state=args.seed,
        )
        with open(out_dir / "metrics.json", "w") as fh:
            json.dump(evals, fh, indent=2)
        print(f"[demo] External-memory training complete. Model → {model_path}")

        # ---------------- Diagnostics & Reports ----------------
        from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils

        _cutils.richer_metrics(args.dataset, out_dir)
        delta_sample = args.delta_sample if args.delta_sample > 0 else None
        _cutils.gene_score_delta(args.dataset, out_dir, sample=delta_sample)
        _cutils.shap_importance(args.dataset, out_dir)
        if args.leakage_probe:
            _cutils.leakage_probe(args.dataset, out_dir)
        print("[demo] Diagnostics complete – see generated CSV/JSON files in run-dir.")
        return

    # -------------------------------------
    # 1. Load & preprocess dataset
    # -------------------------------------
    # Limit rows via env var to reuse datasets.load_dataset internal sampling
    # Honour row cap only if >0 and env var not already set by user
    # import os
    if args.row_cap > 0 and not os.getenv("SS_MAX_ROWS"):
        os.environ["SS_MAX_ROWS"] = str(args.row_cap)
    elif args.row_cap <= 0 and os.getenv("SS_MAX_ROWS"):
        print(f"[warn] Using pre-set SS_MAX_ROWS={os.getenv('SS_MAX_ROWS')} (overrides --row-cap=0)")

    df = datasets.load_dataset(args.dataset)
    X_df, y_series = preprocessing.prepare_training_data(df, label_col="splice_type", return_type="pandas", verbose=1)

    X = X_df.values
    y = _encode_labels(y_series)

    # Select grouping column (gene-wise by default). Allows chromosome-wise (`chrom`) or any column present in DataFrame.
    group_col = args.group_col
    if group_col not in df.columns:
        # Graceful fallback for common chromosome aliases
        chrom_aliases = ["chrom", "seqname", "chromosome"]
        if group_col == "chrom":
            for alias in chrom_aliases:
                if alias in df.columns:
                    group_col = alias
                    break
    
    groups = df[group_col].to_numpy() if group_col in df.columns else None
    if groups is None:
        print(f"[warn] Group column '{args.group_col}' not found – proceeding with stratified (non-group) split.")

    if args.holdout_chroms and group_col.startswith("chrom"):
        # Custom chromosome hold-out logic: test = specified chroms
        test_mask = np.isin(df[group_col].to_numpy(), args.holdout_chroms)
        if test_mask.sum() == 0:
            raise ValueError(f"No rows matched hold-out chromosomes {args.holdout_chroms} using column '{group_col}'.")
        train_val_mask = ~test_mask

        X_train_val, X_test = X[train_val_mask], X[test_mask]
        y_train_val, y_test = y[train_val_mask], y[test_mask]

        # Use gene_id grouping (if available) to avoid leakage within train/valid split
        inner_groups = df["gene_id"].to_numpy()[train_val_mask] if "gene_id" in df.columns else None
        # Derive VALID set
        rel_valid = args.valid_size / (1 - args.test_size)
        if inner_groups is not None:
            from sklearn.model_selection import GroupShuffleSplit
            gss = GroupShuffleSplit(n_splits=1, test_size=rel_valid, random_state=args.seed)
            (train_idx, valid_idx) = next(gss.split(X_train_val, y_train_val, inner_groups))
        else:
            from sklearn.model_selection import train_test_split
            train_idx, valid_idx = train_test_split(
                np.arange(len(X_train_val)),
                test_size=rel_valid,
                random_state=args.seed,
                stratify=y_train_val,
            )
        X_train, X_valid = X_train_val[train_idx], X_train_val[valid_idx]
        y_train, y_valid = y_train_val[train_idx], y_train_val[valid_idx]
    else:
        splits = _split(X, y, groups, test_size=args.test_size, valid_size=args.valid_size, random_state=args.seed)
        X_train, X_valid, X_test, y_train, y_valid, y_test = splits
    

    # -------------------------------------
    # 2. Build & train multiclass XGBoost
    # -------------------------------------
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method=args.tree_method,
        max_bin=args.max_bin if args.tree_method in {"hist", "gpu_hist"} else None,
        random_state=args.seed,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    # -------------------------------------
    # 3. Evaluate
    # -------------------------------------
    def _eval(proba: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        pred = proba.argmax(axis=1)
        return {
            "accuracy": accuracy_score(y_true, pred),
            "macro_f1": f1_score(y_true, pred, average="macro"),
        }

    metrics = {
        "test": _eval(model.predict_proba(X_test), y_test),
        "valid": _eval(model.predict_proba(X_valid), y_valid),
    }

    print(json.dumps(metrics, indent=2))

    # -------------------------------------
    # 4. Persist artefacts (model + feature manifest)
    # -------------------------------------
    import pickle

    with open(out_dir / "model_multiclass.pkl", "wb") as fh:
        pickle.dump(model, fh)

    feature_manifest = pd.DataFrame({"feature": X_df.columns})
    feature_manifest.to_csv(out_dir / "feature_manifest.csv", index=False)

    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Saved model & metrics to {out_dir}")

    # ---------------- Diagnostics & Reports ----------------
    from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils

    _cutils.richer_metrics(args.dataset, out_dir)
    _cutils.gene_score_delta(args.dataset, out_dir)
    _cutils.shap_importance(args.dataset, out_dir)
    if args.leakage_probe:
        _cutils.leakage_probe(args.dataset, out_dir)
    print("[demo] Diagnostics complete – see generated CSV/JSON files in run-dir.")


if __name__ == "__main__":
    main()
