#!/usr/bin/env python3
"""Gene-aware K-fold CV driver for the 3-class meta-model (in-memory).

This mirrors `run_loco_cv_multiclass.py` but uses *gene_id* groupings instead
of chromosome groups and trains with in-memory `XGBClassifier`.

For very large datasets consider `run_gene_cv_multiclass_extmem.py`, which
streams from on-disk LibSVM, but for moderate data sizes this script provides a
convenient, fast workflow with all post-training diagnostics.

Usage
-----
Example with the *train_pc_1000* dataset::

    python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_multiclass \
        --dataset train_pc_1000/master \  # directory of Parquet shards
        --out-dir runs/gene_cv_pc1000 \  # where fold results are written
        --annotations data/ensembl/splice_sites.tsv

Common options::

    --n-folds 5                 # number of gene-wise CV folds (default 5)
    --valid-size 0.1            # fraction of TRAIN rows held out for in-fold validation
    --tree-method hist          # XGBoost backend (hist | gpu_hist | approx)
    --device cuda               # force GPU; "auto" chooses based on availability

For a GPU run with 8 folds you might use::

    python run_gene_cv_multiclass.py \
        --dataset ./train_pc_1000/master \
        --out-dir /scratch/runs/gene_cv_pc1000_gpu \
        --annotations /data/ensembl/splice_sites.tsv \
        --n-folds 8 --tree-method gpu_hist --device cuda

"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from xgboost import XGBClassifier

from meta_spliceai.splice_engine.meta_models.builder import preprocessing
from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels
from meta_spliceai.splice_engine.utils_kmer import is_kmer_feature as _is_kmer
from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils

################################################################################
# CLI
################################################################################

def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(description="Gene-wise K-fold CV for the 3-way meta-classifier (in-memory).")
    p.add_argument("--dataset", required=True, help="Dataset directory or single Parquet file")
    p.add_argument("--out-dir", required=True, help="Directory to save fold metrics & aggregates")

    p.add_argument("--gene-col", default="gene_id", help="Column containing gene IDs for grouping (default: gene_id)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--valid-size", type=float, default=0.1, help="Fraction of TRAIN rows for validation within each fold")
    p.add_argument("--row-cap", type=int, default=100_000, help="Row cap for RAM (0 = disabled)")

    p.add_argument("--diag-sample", type=int, default=25_000, help="Rows to sample for diagnostics (0 = full)")
    p.add_argument("--annotations", required=True, help="Splice-site annotations for meta splice evaluation (parquet/csv/tsv)")
    p.add_argument("--neigh-sample", type=int, default=0, help="If >0 run neighbour-window diagnostic with this many random genes")
    p.add_argument("--neigh-window", type=int, default=10, help="Neighbourhood size for neighbour-window diagnostic")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--tree-method", default="hist", choices=["hist", "gpu_hist", "approx"], help="Underlying XGBoost algorithm")
    p.add_argument("--max-bin", type=int, default=256)
    p.add_argument("--device", default="auto", help="xgboost device parameter (cuda|cpu|auto)")
    p.add_argument("--n-estimators", type=int, default=800)

    # Site-level eval tuning
    p.add_argument("--threshold", type=float, help="Override probability threshold for site-level evaluation")
    p.add_argument("--base-tsv", help="Path to base model full_splice_performance.tsv for comparison")
    p.add_argument("--errors-only", action="store_true", help="Evaluate only rows where base model was FP/FN (uses artifacts pred_type)")

    p.add_argument("--leakage-probe", action="store_true", help="Run correlation leakage probe afterward")
    return p.parse_args(argv)

################################################################################
# Main
################################################################################

def main(argv: List[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Respect row-cap via SS_MAX_ROWS env var consumed by datasets.load_dataset
    if args.row_cap > 0 and not os.getenv("SS_MAX_ROWS"):
        os.environ["SS_MAX_ROWS"] = str(args.row_cap)

    # ------------------------------------------------------------------
    # 1. Load & prepare dataset
    # ------------------------------------------------------------------
    df = datasets.load_dataset(args.dataset)
    if args.gene_col not in df.columns:
        raise KeyError(f"Column '{args.gene_col}' not found in dataset")

    X_df, y_series = preprocessing.prepare_training_data(
        df, label_col="splice_type", return_type="pandas", verbose=1
    )
    X = X_df.values
    y = _encode_labels(y_series)

    genes = df[args.gene_col].to_numpy()

    # Quick feature overview
    feature_names = list(X_df.columns)
    non_kmer_feats = [f for f in feature_names if not _is_kmer(f)]  # noqa: E501
    kmer_feats = [f for f in feature_names if _is_kmer(f)]

    print(f"[Gene-CV] Feature matrix: {len(feature_names)} columns – {len(non_kmer_feats)} non-k-mer features")
    if kmer_feats:
        sample_kmer = random.sample(kmer_feats, k=min(3, len(kmer_feats)))
        print("       Example k-mer feats:", ", ".join(sample_kmer))

    # ------------------------------------------------------------------
    # 2. Gene-wise K-fold CV
    # ------------------------------------------------------------------
    gkf = GroupKFold(n_splits=args.n_folds)
    fold_rows: List[Dict[str, object]] = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(gkf.split(X, y, groups=genes)):
        print(f"[Gene-CV] Fold {fold_idx + 1}/{args.n_folds}  test_rows={len(test_idx)}")

        # Split TRAIN into TRAIN/VALID with groups again
        rel_valid = args.valid_size / (1.0 - 1.0 / args.n_folds)
        gss = GroupShuffleSplit(n_splits=1, test_size=rel_valid, random_state=args.seed)
        (train_idx, valid_idx) = next(gss.split(train_val_idx, y[train_val_idx], groups=genes[train_val_idx]))
        train_idx = train_val_idx[train_idx]
        valid_idx = train_val_idx[valid_idx]

        # Model
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            tree_method=args.tree_method,
            max_bin=args.max_bin if args.tree_method in {"hist", "gpu_hist"} else None,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=args.seed,
            n_jobs=-1,
            device=args.device if args.device != "auto" else None,
        )

        model.fit(
            X[train_idx],
            y[train_idx],
            eval_set=[(X[valid_idx], y[valid_idx])],
            verbose=False,
        )

        proba = model.predict_proba(X[test_idx])
        pred = proba.argmax(axis=1)
        acc = accuracy_score(y[test_idx], pred)
        macro_f1 = f1_score(y[test_idx], pred, average="macro")

        # Splice-site-only metrics (exclude class 0="neither")
        splice_mask = y[test_idx] != 0
        if splice_mask.any():
            splice_acc = accuracy_score(y[test_idx][splice_mask], pred[splice_mask])
            splice_macro_f1 = f1_score(y[test_idx][splice_mask], pred[splice_mask], average="macro")
        else:
            splice_acc = np.nan
            splice_macro_f1 = np.nan

        # Top-K accuracy over splice sites (donor+acceptor)
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
            "top_k": k,
        }
        fold_rows.append(row)
        with open(out_dir / f"metrics_fold{fold_idx}.json", "w") as fh:
            json.dump(row, fh, indent=2)

    # ------------------------------------------------------------------
    # 3. Train final model on full data & save artefacts
    # ------------------------------------------------------------------
    final_model = XGBClassifier(
        n_estimators=args.n_estimators,
        tree_method=args.tree_method,
        max_bin=args.max_bin if args.tree_method in {"hist", "gpu_hist"} else None,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=args.seed,
        n_jobs=-1,
        device=args.device if args.device != "auto" else None,
    )
    final_model.fit(X, y)

    import pickle

    pd.DataFrame({"feature": feature_names}).to_csv(out_dir / "feature_manifest.csv", index=False)
    with open(out_dir / "model_multiclass.pkl", "wb") as fh:
        pickle.dump(final_model, fh)

    # ------------------------------------------------------------------
    # 4. Aggregate metrics
    # ------------------------------------------------------------------
    df_metrics = pd.DataFrame(fold_rows)
    df_metrics.to_csv(out_dir / "gene_cv_metrics.csv", index=False)

    print("\nGene-CV results by fold:\n", df_metrics)
    mean_metrics = df_metrics[["test_accuracy", "test_macro_f1", "splice_accuracy", "splice_macro_f1", "top_k_accuracy"]].mean()
    print("\nAverage across folds:\n", mean_metrics.to_string())

    with open(out_dir / "metrics_aggregate.json", "w") as fh:
        json.dump(mean_metrics.to_dict(), fh, indent=2)

    # ------------------------------------------------------------------
    # 5. Diagnostics & post-training analysis
    # ------------------------------------------------------------------
    diag_sample = None if args.diag_sample == 0 else args.diag_sample
    _cutils.richer_metrics(args.dataset, out_dir, sample=diag_sample)
    _cutils.gene_score_delta(args.dataset, out_dir, sample=diag_sample)
    _cutils.shap_importance(args.dataset, out_dir, sample=diag_sample)
    _cutils.probability_diagnostics(args.dataset, out_dir, sample=diag_sample)
    _cutils.base_vs_meta(args.dataset, out_dir, sample=diag_sample)

    # --- determine optimal threshold from suggestion file ---
    thresh = args.threshold if args.threshold is not None else 0.9
    # If user did not specify threshold, try auto suggestion
    if args.threshold is None:
        th_path = out_dir / "threshold_suggestion.txt"
        if th_path.exists():
            try:
                import pandas as _pd
                # threshold_suggestion.txt is written as key\tvalue pairs (no header)
                _th_df = _pd.read_csv(th_path, sep="\t", header=None, names=["metric", "value"])
                cand = _th_df.loc[_th_df["metric"] == "best_threshold", "value"]
                if not cand.empty:
                    thresh = float(cand.iloc[0])
                    if abs(thresh - 0.9) > 1e-3:
                        print(f"[meta_splice_eval] Using suggested threshold {thresh:.3f} instead of default 0.9")
            except Exception as _e:
                print("[warning] could not parse threshold_suggestion.txt:", _e)

    # --- base model performance TSV ---
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
            sample=diag_sample,  # limit rows to avoid OOM
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

    print("[Gene-CV] Diagnostics complete – see generated CSV/JSON artefacts in run-dir.")


if __name__ == "__main__":  # pragma: no cover
    main()
