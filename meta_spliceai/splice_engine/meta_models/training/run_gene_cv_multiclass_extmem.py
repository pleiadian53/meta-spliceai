#!/usr/bin/env python3
"""Gene-aware K-fold CV (external-memory) for the 3-class meta-model.

This is the gene-centric counterpart to `run_loco_cv_multiclass_extmem.py`.
Instead of leaving chromosomes out, it performs cross-validation where train/
valid/test splits are *stratified by gene_id* so that records from the same
gene never appear in multiple splits.

All folds are trained with XGBoost external-memory using the global LibSVM
file, keeping RAM usage low regardless of dataset size.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils

################################################################################
# Helpers
################################################################################

from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels  # robust label encoder

################################################################################
# Main
################################################################################

def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(description="Gene-wise K-fold CV (external-memory) for the 3-way meta-classifier.")
    p.add_argument("--dataset", required=True, help="Master dataset directory or single Parquet file for metadata")
    p.add_argument("--libsvm", required=True, help="Path to the *global* LibSVM file (from convert_dataset_to_libsvm)")
    p.add_argument("--out-dir", required=True, help="Output directory for fold metrics & artefacts")

    p.add_argument("--n-folds", type=int, default=5, help="Number of gene-based CV folds")
    p.add_argument("--valid-size", type=float, default=0.1, help="Fraction of TRAIN rows used for validation within each fold")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--tree-method", default="hist", choices=["hist", "approx"], help="GPU unsupported in external-memory mode")
    p.add_argument("--max-bin", type=int, default=256)
    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--device", default="auto", help="XGBoost device parameter (cuda|cpu|auto). GPU unavailable with external-memory.")

    # Diagnostics
    p.add_argument("--annotations", required=True, help="Splice-site annotations for meta splice evaluation (parquet/csv/tsv)")
    p.add_argument("--neigh-sample", type=int, default=0, help="If >0 run neighbour-window diagnostic with this many random genes")
    p.add_argument("--neigh-window", type=int, default=10, help="Neighbourhood size for neighbour-window diagnostic")
    p.add_argument("--diag-sample", type=int, default=25_000, help="Row sample for diagnostics (0 = full)")
    p.add_argument("--leakage-probe", action="store_true", help="Run correlation leakage probe after training")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure full metadata is loaded – external-memory mode can handle it.
    import os as _os
    _os.environ.setdefault("SS_MAX_ROWS", "0")

    # ------------------------------------------------------------------
    # 1. Read metadata (only gene_id & labels needed)
    # ------------------------------------------------------------------
    df_meta = datasets.load_dataset(args.dataset, columns=["gene_id", "splice_type"])
    if "gene_id" not in df_meta.columns:
        raise KeyError("Dataset must contain gene_id column for gene-aware CV")

    y = _encode_labels(df_meta["splice_type"].to_pandas())
    genes = df_meta["gene_id"].to_numpy()

    # ------------------------------------------------------------------
    # 2. Load global external-memory DMatrix once
    # ------------------------------------------------------------------
    libsvm_uri = f"{Path(args.libsvm)}?format=libsvm"
    dall = xgb.DMatrix(libsvm_uri, missing=np.nan)  # feature names auto-loaded via .features.json manifest
    if dall.num_row() != len(y):
        raise ValueError("Row count mismatch between LibSVM file and dataset metadata")

    gkf = GroupKFold(n_splits=args.n_folds)
    fold_rows: List[Dict[str, object]] = []
    last_model = None

    for fold_idx, (train_val_idx, test_idx) in enumerate(gkf.split(np.empty(len(y)), y, groups=genes)):
        print(f"[Gene-CV-ext] Fold {fold_idx+1}/{args.n_folds}  test_rows={len(test_idx)}")

        # Split TRAIN into TRAIN/VALID using groups again to avoid leakage
        rel_valid = args.valid_size / (1.0 - 1.0/args.n_folds)  # adjust because we already carved out test fold
        gss = GroupShuffleSplit(n_splits=1, test_size=rel_valid, random_state=args.seed)
        (train_idx, valid_idx) = next(gss.split(train_val_idx, y[train_val_idx], groups=genes[train_val_idx]))
        train_idx = train_val_idx[train_idx]
        valid_idx = train_val_idx[valid_idx]

        # Build sliced DMatrices
        dtrain = dall.slice(train_idx)
        dval = dall.slice(valid_idx)
        dtest = dall.slice(test_idx)

        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "tree_method": args.tree_method,
            "max_bin": args.max_bin if args.tree_method == "hist" else None,
            "eval_metric": "mlogloss",
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": args.seed,
        }
        if args.device != "auto":
            params["device"] = args.device

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=args.n_estimators,
            evals=[(dval, "eval")],
            verbose_eval=False,
        )
        last_model = bst  # keep reference for saving later

        proba = bst.predict(dtest)
        pred = proba.argmax(axis=1)
        acc = accuracy_score(y[test_idx], pred)
        macro_f1 = f1_score(y[test_idx], pred, average="macro")

        splice_mask = y[test_idx] != 0
        if splice_mask.any():
            splice_acc = accuracy_score(y[test_idx][splice_mask], pred[splice_mask])
            splice_macro_f1 = f1_score(y[test_idx][splice_mask], pred[splice_mask], average="macro")
        else:
            splice_acc = np.nan
            splice_macro_f1 = np.nan

        # Top-K accuracy
        k = int(splice_mask.sum())
        if k > 0:
            splice_prob = proba[:, 1] + proba[:, 2]
            top_idx = np.argsort(-splice_prob)[:k]
            top_k_correct = (y[test_idx][top_idx] != 0).sum()
            top_k_acc = top_k_correct / k
        else:
            top_k_acc = np.nan

        cm = confusion_matrix(y[test_idx], pred, labels=[0, 1, 2])
        label_names = ["neither", "donor", "acceptor"]
        import pandas as _pd
        print("Class distribution (true labels):", {
            name: int((y[test_idx] == i).sum()) for i, name in enumerate(label_names)
        })
        print(_pd.DataFrame(cm, index=label_names, columns=label_names))

        row = {
            "fold": fold_idx,
            "test_rows": len(test_idx),
            "test_accuracy": acc,
            "test_macro_f1": macro_f1,
            "splice_accuracy": splice_acc,
            "splice_macro_f1": splice_macro_f1,
            "top_k_accuracy": top_k_acc,
        }
        fold_rows.append(row)
        with open(out_dir / f"metrics_fold{fold_idx}.json", "w") as fh:
            json.dump(row, fh, indent=2)

    # ------------------------------------------------------------------
    # 3. Persist artefacts and aggregate metrics
    # ------------------------------------------------------------------
    model_path = out_dir / "model_multiclass.json"
    if last_model is not None:
        last_model.save_model(str(model_path))

    # copy feature manifest next to LibSVM into run_dir for convenience
    from shutil import copy2
    manifest_src = Path(args.libsvm).with_suffix(".features.json")
    manifest_dst = out_dir / "train.features.json"
    if manifest_src.exists():
        copy2(manifest_src, manifest_dst)

    df_metrics = pd.DataFrame(fold_rows)
    df_metrics.to_csv(out_dir / "gene_cv_metrics.csv", index=False)

    mean_metrics = df_metrics[["test_accuracy", "test_macro_f1", "splice_accuracy", "splice_macro_f1", "top_k_accuracy"]].mean()
    with open(out_dir / "metrics_aggregate.json", "w") as fh:
        json.dump(mean_metrics.to_dict(), fh, indent=2)

    print("\nGene-CV (external-memory) done.  Per-fold metrics:\n")
    print(df_metrics)
    print("\nAverage across folds:\n", mean_metrics.to_string())

    # ---------------- Diagnostics ----------------
    diag_sample = None if args.diag_sample == 0 else args.diag_sample
    _cutils.richer_metrics(args.dataset, out_dir, sample=diag_sample)
    _cutils.gene_score_delta(args.dataset, out_dir, sample=diag_sample)
    _cutils.shap_importance(args.dataset, out_dir, sample=diag_sample)
    _cutils.base_vs_meta(args.dataset, out_dir, sample=diag_sample)

    # --- suggested threshold ---
    thresh = args.threshold if getattr(args, "threshold", None) is not None else 0.9
    # Auto suggestion only if threshold not provided
    if getattr(args, "threshold", None) is None:
        th_path = out_dir / "threshold_suggestion.txt"
        if th_path.exists():
            try:
                import pandas as _pd
                _th_df = _pd.read_csv(th_path, sep="\t")
                if "best_threshold" in _th_df.columns:
                    thresh = float(_th_df.loc[0, "best_threshold"])
                    if abs(thresh - 0.9) > 1e-3:
                        print(f"[meta_splice_eval] Using suggested threshold {thresh:.3f} instead of default 0.9")
            except Exception as _e:
                print("[warning] could not parse threshold_suggestion.txt:", _e)

    from pathlib import Path as _Path
    base_tsv = _Path(args.base_tsv) if getattr(args, "base_tsv", None) else _Path("data/ensembl/spliceai_eval/full_splice_performance.tsv")
    if not base_tsv.exists():
        base_tsv = None

    try:
        _cutils.meta_splice_performance(
            dataset_path=args.dataset,
            run_dir=out_dir,
            annotations_path=args.annotations,
            threshold=thresh,
            base_tsv=base_tsv,
            errors_only=args.errors_only,
            sample=None,
            verbose=1,
        )
    except Exception as e:
        print("[warning] meta_splice_performance failed:", e)
    if args.neigh_sample > 0:
        try:
            _cutils.neighbour_window_diagnostics(
                args.dataset,
                out_dir,
                annotations_path=args.annotations,
                n_sample=args.neigh_sample,
                window=args.neigh_window,
            )
        except Exception as e:
            print("[warning] neighbour_window_diagnostics failed:", e)


    if args.leakage_probe:
        _cutils.leakage_probe(args.dataset, out_dir)
    print("[Gene-CV-ext] Diagnostics complete – see generated CSV/JSON artefacts in run-dir.")


if __name__ == "__main__":  # pragma: no cover
    main()
