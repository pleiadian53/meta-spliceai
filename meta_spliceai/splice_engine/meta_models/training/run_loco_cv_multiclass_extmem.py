#!/usr/bin/env python3
"""LOCO-CV driver **using XGBoost external-memory mode**.

Why a separate script?
----------------------
`run_loco_cv_multiclass.py` keeps everything in RAM.  Once the dataset grows
past millions of rows this becomes impractical.  The present script:

1. Assumes a *single* LibSVM file stored on fast disk/SSD.
2. Loads it **once** into an external-memory `DMatrix` (just file handles, no
   dense matrix in RAM).
3. Slices that `DMatrix` per LOCO fold → near-constant memory.

Usage
-----
Convert the dataset to LibSVM (one-time):
```bash
conda run -n surveyor python - <<'PY'
from meta_spliceai.splice_engine.meta_models.training import external_memory_utils as emu
emu.convert_dataset_to_libsvm(
    dataset_dir="train_pc_5000/master",
    out_path="data/train_pc_5000.libsvm",
    chunk_rows=250_000,
)
PY
```
Then run LOCO-CV:
```bash
conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_extmem \
    --dataset train_pc_5000/master \
    --libsvm data/train_pc_5000.libsvm \
    --out-dir runs/loco_cv_extmem \
    --tree-method hist --max-bin 256 --n-estimators 1200
```
Outputs: one JSON per fold, `loco_metrics.csv`, and overall `metrics_aggregate.json`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils

from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.training import chromosome_split as csplit

# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(description="LOCO-CV (external-memory) for the 3-way meta-classifier.")
    p.add_argument("--dataset", required=True, help="Dataset directory or single Parquet file (for metadata only)")
    p.add_argument("--libsvm", required=True, help="Path to the *global* LibSVM file produced by convert_dataset_to_libsvm")
    p.add_argument("--out-dir", required=True, help="Output directory for fold metrics & aggregates")

    p.add_argument("--group-col", default="chrom", help="Column with chromosome labels (default: chrom)")
    p.add_argument("--base-tsv", help="Path to base model full_splice_performance.tsv for comparison")
    p.add_argument("--errors-only", action="store_true", help="Evaluate only rows where base model was FP/FN (uses artifacts pred_type)")
    p.add_argument("--gene-col", default="gene_id", help="Column with gene IDs")
    p.add_argument("--valid-size", type=float, default=0.15)
    p.add_argument("--min-rows-test", type=int, default=1_000)
    p.add_argument("--heldout-chroms", default="", help="Comma-separated chromosomes for a fixed test set instead of LOCO-CV")
    p.add_argument("--diag-sample", type=int, default=25_000, help="Row sample for diagnostics (0 = full)")
    p.add_argument("--neigh-sample", type=int, default=0, help="If >0 run neighbour-window diagnostic with this many random genes")
    p.add_argument("--neigh-window", type=int, default=10, help="Neighbourhood size for neighbour-window diagnostic")
    p.add_argument("--annotations", required=True, help="Splice-site annotation file (parquet/tsv/csv) for meta splice evaluation")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--tree-method", default="hist", choices=["hist", "approx"], help="GPU is unsupported in external-memory mode")
    p.add_argument("--max-bin", type=int, default=256)
    p.add_argument("--n-estimators", type=int, default=800)
    p.add_argument("--device", default="auto", help="XGBoost device parameter (cuda|cpu|auto). GPU unavailable with external-memory, keep default.")

    p.add_argument("--leakage-probe", action="store_true", help="Run leakage correlation probe after training")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels  # robust label encoder


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1. Load metadata (no dense feature matrix!)
    # ---------------------------------------------------------------------
    df = datasets.load_dataset(args.dataset, columns=[args.group_col, args.gene_col, "splice_type"])

    if args.group_col not in df.columns:
        raise KeyError(f"Column '{args.group_col}' not found")
    if args.gene_col not in df.columns:
        raise KeyError(f"Column '{args.gene_col}' not found")

    y = _encode_labels(df["splice_type"].to_pandas())
    chrom = df[args.group_col].to_numpy()
    genes = df[args.gene_col].to_numpy()

    # ---------------------------------------------------------------------
    # 2. Load global external-memory DMatrix once
    # ---------------------------------------------------------------------
    libsvm_uri = f"{Path(args.libsvm)}?format=libsvm"
    dall = xgb.DMatrix(libsvm_uri, missing=np.nan)  # feature_names auto-loaded via .features.json manifest

    # Sanity check length
    if dall.num_row() != len(y):
        raise ValueError("Row count mismatch between LibSVM file and dataset metadata")

    fold_rows: list[Dict[str, object]] = []
    last_model = None

    if args.heldout_chroms:
        heldout_list = [c.strip() for c in args.heldout_chroms.split(',') if c.strip()]
        tr_idx, val_idx, te_idx, *_ = csplit.holdout_split(
            np.empty(len(chrom)),  # feature matrix placeholder, not used to compute indices
            y,
            chrom_array=chrom,
            holdout_chroms=heldout_list,
            valid_size=args.valid_size,
            gene_array=genes,
            seed=args.seed,
        )
        loops = [("-".join(heldout_list), tr_idx, val_idx, te_idx)]
    else:
        loops = csplit.loco_cv_splits(
            None,  # X unused
            y,
            chrom_array=chrom,
            gene_array=genes,
            valid_size=args.valid_size,
            min_rows=args.min_rows_test,
            seed=args.seed,
        )

    for held_out, tr_idx, val_idx, te_idx in loops:
        print(f"[LOCO-ext] Held-out = {held_out}  rows={len(te_idx)}")

        dtrain = dall.slice(tr_idx)
        dval = dall.slice(val_idx)
        dtest = dall.slice(te_idx)

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

        pred = bst.predict(dtest).argmax(axis=1)
        acc = accuracy_score(y[te_idx], pred)
        macro_f1 = f1_score(y[te_idx], pred, average="macro")

        splice_mask = y[te_idx] != 0
        if splice_mask.any():
            splice_acc = accuracy_score(y[te_idx][splice_mask], pred[splice_mask])
            splice_macro_f1 = f1_score(y[te_idx][splice_mask], pred[splice_mask], average="macro")
        else:
            splice_acc = np.nan
            splice_macro_f1 = np.nan

        # Top-K accuracy
        k = int(splice_mask.sum())
        if k > 0:
            splice_prob = proba[:, 1] + proba[:, 2]
            top_idx = np.argsort(-splice_prob)[:k]
            top_k_correct = (y[te_idx][top_idx] != 0).sum()
            top_k_acc = top_k_correct / k
        else:
            top_k_acc = np.nan

        cm = confusion_matrix(y[te_idx], pred, labels=[0, 1, 2])
        

        label_names = ["neither", "donor", "acceptor"]
        import pandas as _pd
        print("Class distribution (true labels):", {
            name: int((y[te_idx] == i).sum()) for i, name in enumerate(label_names)
        })
        print(_pd.DataFrame(cm, index=label_names, columns=label_names))

        row = {
            "held_out": held_out,
            "test_rows": len(te_idx),
            "test_accuracy": acc,
            "test_macro_f1": macro_f1,
            "splice_accuracy": splice_acc,
            "splice_macro_f1": splice_macro_f1,
            "top_k_accuracy": top_k_acc,
        }
        fold_rows.append(row)
        with open(out_dir / f"metrics_{held_out}.json", "w") as fh:
            json.dump(row, fh, indent=2)

    # Save last trained model for diagnostics
    model_json_path = out_dir / "model_multiclass.json"
    bst.save_model(str(model_json_path))

    # Ensure feature manifest is present in run_dir
    from shutil import copy2
    manifest_src = Path(args.libsvm).with_suffix(".features.json")
    manifest_dst = out_dir / "train.features.json"
    if manifest_src.exists():
        copy2(manifest_src, manifest_dst)

    df_metrics = pd.DataFrame(fold_rows)
    df_metrics.to_csv(out_dir / "loco_metrics.csv", index=False)

    mean_metrics = df_metrics[["test_accuracy", "test_macro_f1", "splice_accuracy", "splice_macro_f1", "top_k_accuracy"]].mean()
    with open(out_dir / "metrics_aggregate.json", "w") as fh:
        json.dump(mean_metrics.to_dict(), fh, indent=2)

    print("\nLOCO-CV (external-memory) done.  Per-fold metrics:\n")
    print(df_metrics)
    print("\nAverage across folds:\n", mean_metrics.to_string())

    # ---------------- Diagnostics ----------------
    diag_sample = None if args.diag_sample == 0 else args.diag_sample
    _cutils.richer_metrics(args.dataset, out_dir, sample=diag_sample)
    _cutils.gene_score_delta(args.dataset, out_dir, sample=diag_sample)
    _cutils.shap_importance(args.dataset, out_dir, sample=diag_sample)
    _cutils.base_vs_meta(args.dataset, out_dir, sample=diag_sample)

    # suggested threshold
    thresh = 0.9
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
    base_tsv = _Path("data/ensembl/spliceai_eval/full_splice_performance.tsv")
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
            sample=diag_sample,
            verbose=1,
        )
    except Exception as e:
        print("[warning] meta_splice_performance failed:", e)
    if getattr(args, 'neigh_sample', 0) > 0:
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
    # Neighbour-window diagnostic optional above


    if args.leakage_probe:
        _cutils.leakage_probe(args.dataset, out_dir)
    print("[LOCO-ext] Diagnostics complete – see generated CSV/JSON artefacts in run-dir.")


if __name__ == "__main__":  # pragma: no cover
    main()
