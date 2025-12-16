"""Demo script to showcase the enhanced splice-site **inference** workflow.

This small utility proves that:

1. `run_enhanced_splice_inference_workflow` produces an **enriched feature
   matrix** whose *feature columns* match *exactly* those used to train the
   meta-model.
2. A pre-trained meta-model (loaded via ``classifier_utils._load_model_generic``)
   can be applied to the new matrix to obtain donor / acceptor / neither
   probabilities which can then be converted into a splice-type prediction.

Run ``python -m meta_spliceai.splice_engine.meta_models.workflows.demo_splice_inference \
        --run-dir /path/to/model_run  \
        --eval-dir /path/to/spliceai_eval``

The script prints (1) a schema comparison summary, (2) a small table of example
predictions, and (3) the location of the generated artefacts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import polars as pl

# Evaluation metrics
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_curve,
    auc,
    roc_auc_score,
    average_precision_score,
)

from meta_spliceai.splice_engine.meta_models.workflows.splice_inference_workflow import (
    run_enhanced_splice_inference_workflow,
)
from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _map_argmax_to_splice_type(argmax_arr: np.ndarray) -> List[str]:
    """Convert arg-max indices 0/1/2 to splice-type strings."""
    mapping = {0: "neither", 1: "donor", 2: "acceptor"}
    return [mapping.get(int(ix), "?") for ix in argmax_arr]


# ---------------------------------------------------------------------------
# Main demo logic
# ---------------------------------------------------------------------------

def demo(
    run_dir: Path,
    eval_dir: Path,
    *,
    target_genes: List[str] | None = None,
    max_positions_per_gene: int = 0,
    max_analysis_rows: int = 500_000,
    train_schema_dir: Path | None = None,
    verbosity: int = 1,
) -> None:
    # 1) Load the pre-trained meta-model -------------------------------------------------
    predict_fn, feature_names = _cutils._load_model_generic(run_dir)
    if verbosity:
        print(f"[demo] Loaded meta-model from {run_dir}. Feature count = {len(feature_names):,}")

    # 2) Run enhanced inference workflow ------------------------------------------------
    artefact_dir = run_enhanced_splice_inference_workflow(
        train_schema_dir=str(train_schema_dir) if train_schema_dir else None,
        covered_pos=None,          # predict for *all* ambiguous positions
        t_low=0.02,
        t_high=0.80,
        target_genes=target_genes,
        max_positions_per_gene=max_positions_per_gene,
        max_analysis_rows=max_analysis_rows,
        verbosity=max(0, verbosity - 1),
    )
    feature_master = artefact_dir / "features" / "master"
    if not feature_master.exists():
        raise FileNotFoundError("Feature dataset not found at " + str(feature_master))
    if verbosity:
        print(f"[demo] Feature dataset located at {feature_master}")

    # 3) Load a small sample of the dataset --------------------------------------------
    first_parquet = next(iter(feature_master.glob("*.parquet")))
    df = pl.read_parquet(first_parquet, n_rows=5_000)
    if verbosity:
        print(f"[demo] Loaded {df.height:,} rows for demonstration")

    # 4) Confirm feature columns match --------------------------------------------------
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise RuntimeError(
            "The generated feature matrix is missing the following columns required by the model: "
            + ", ".join(missing)
        )
    extra = [c for c in df.columns if c in feature_names]
    if verbosity:
        print(f"[demo] ✔ Feature column check passed ({len(extra)}/{len(feature_names)} present).")

    # 5) Make predictions ---------------------------------------------------------------
        # Convert to numeric – cast each feature to Float32 (coercing errors to null) and
    # replace any null / NaN with 0.0 before exporting to NumPy. This prevents failures
    # when some columns contain empty strings or other non-numeric placeholders.
    X_numeric = (
        df
        .select([pl.col(c).cast(pl.Float32, strict=False) for c in feature_names])
        .fill_null(0.0)
    )
    X_sample = X_numeric.to_numpy()
    proba = predict_fn(X_sample)
    y_pred_ix = np.argmax(proba, axis=1)
    y_pred_str = _map_argmax_to_splice_type(y_pred_ix)

    # 6) Build comparison frame (base vs meta) ---------------------------------------
    # Base-model raw scores (present in feature matrix)
    if {"donor_score", "acceptor_score", "neither_score"}.issubset(set(df.columns)):
        base_scores = np.stack([
            df["neither_score"].to_numpy(),
            df["donor_score"].to_numpy(),
            df["acceptor_score"].to_numpy(),
        ], axis=1)
        base_pred_ix = np.argmax(base_scores, axis=1)
    else:
        raise RuntimeError("donor_score / acceptor_score / neither_score columns missing from dataset – cannot build comparison.")

    base_pred_str = _map_argmax_to_splice_type(base_pred_ix)

    comp_df = pl.DataFrame({
        "donor_score": df["donor_score"],
        "acceptor_score": df["acceptor_score"],
        "neither_score": df["neither_score"],
        "donor_meta": proba[:, 1],
        "acceptor_meta": proba[:, 2],
        "neither_meta": proba[:, 0],
        "pred": base_pred_str,
        "pred_meta": y_pred_str,
    })

    # 7) Evaluation metrics & curves (if splice_type present) ------------------------
    metrics: dict[str, float] = {}
    if "splice_type" in df.columns:
        truth_raw = df["splice_type"].cast(pl.Utf8).to_list()
        truth_norm: list[str] = []
        for v in truth_raw:
            if v == "donor" or v == "1":
                truth_norm.append("donor")
            elif v == "acceptor" or v == "2":
                truth_norm.append("acceptor")
            else:
                truth_norm.append("neither")
        truth_arr = np.array(truth_norm)

        # Macro-F1 -------------------------------------------------------------------
        metrics["f1_base"] = float(f1_score(truth_arr, np.array(base_pred_str), average="macro"))
        metrics["f1_meta"] = float(f1_score(truth_arr, np.array(y_pred_str), average="macro"))

        if verbosity:
            print("\n[demo] Classification report – META model:")
            print(classification_report(truth_arr, y_pred_str, digits=3))

        # ROC / PR curves for donor & acceptor ---------------------------------------
        label_map = {"donor": 1, "acceptor": 2}
        for lbl, idx in label_map.items():
            y_true_bin = (truth_arr == lbl).astype(int)
            if y_true_bin.sum() == 0:
                continue  # skip if no positives for this class
            y_score_meta = proba[:, idx]
            try:
                fpr, tpr, roc_thr = roc_curve(y_true_bin, y_score_meta)
                prec, rec, pr_thr = precision_recall_curve(y_true_bin, y_score_meta)
                metrics[f"auroc_{lbl}"] = float(roc_auc_score(y_true_bin, y_score_meta))
                metrics[f"auprc_{lbl}"] = float(average_precision_score(y_true_bin, y_score_meta))

                # Write curve data --------------------------------------------------
                # Harmonise lengths just in case (sklearn usually gives len(fpr)=len(tpr)=len(thr)+1)
                fpr_adj = fpr[1:]
                tpr_adj = tpr[1:]
                n = min(len(fpr_adj), len(tpr_adj), len(roc_thr))
                roc_df = pl.DataFrame({
                    "fpr": fpr_adj[:n],
                    "tpr": tpr_adj[:n],
                    "threshold": roc_thr[:n],
                })
                # Align lengths for PR curve as well
                prec_adj = prec[1:]
                rec_adj = rec[1:]
                m = min(len(prec_adj), len(rec_adj), len(pr_thr))
                pr_df = pl.DataFrame({
                    "recall": rec_adj[:m],
                    "precision": prec_adj[:m],
                    "threshold": pr_thr[:m],
                })
                roc_out = artefact_dir / f"roc_curve_{lbl}.tsv"
                pr_out = artefact_dir / f"pr_curve_{lbl}.tsv"
                roc_df.write_csv(roc_out, separator="\t")
                pr_df.write_csv(pr_out, separator="\t")
                if verbosity:
                    print(f"[demo] Saved ROC & PR curves for {lbl} to {roc_out.name}, {pr_out.name}")
            except ValueError:
                # Handle edge cases where metrics cannot be computed
                pass

        if verbosity and metrics:
            print(f"\n[demo] Macro-F1: base={metrics['f1_base']:.3f}, meta={metrics['f1_meta']:.3f}")

    # 8) Persist to TSV ---------------------------------------------------------------
    out_tsv = artefact_dir / "demo_score_comparison.tsv"
    comp_df.write_csv(out_tsv, separator="\t")
    if verbosity:
        print(f"[demo] Wrote comparison table to {out_tsv}")

    # 8b) Save metrics JSON -----------------------------------------------------------
    import json as _json

    metrics_out = artefact_dir / "demo_metrics.json"
    with open(metrics_out, "w") as fh:
        _json.dump(metrics, fh, indent=2)
    if verbosity:
        print(f"[demo] Wrote metrics to {metrics_out}")

    # 9) Output a few example rows ----------------------------------------------------
    print("\n[demo] Sample comparison rows:")
    print(comp_df.head(10))

    print("\n[demo] SUCCESS – inference workflow + meta-model prediction ran without issues.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo the splice inference workflow + meta-model.")
    p.add_argument("--run-dir", required=True, help="Directory containing the trained meta-model (model_multiclass.*)")
    p.add_argument("--eval-dir", required=False, help="SpliceAI evaluation directory (optional – for future use)")
    p.add_argument("--genes", nargs="*", default=None, help="Optional list of Ensembl gene IDs to limit the demo run")
    p.add_argument("--train-schema-dir", required=True,
                   help="Directory of *training* dataset whose columns define the canonical feature schema (e.g. /path/to/train_pc_1000)")
    p.add_argument("--max-positions-per-gene", type=int, default=0,
                   help="Limit the number of least-confident positions sampled per gene (0 = unlimited)")
    p.add_argument("--max-analysis-rows", type=int, default=500000,
                   help="Global cap on total positions to analyse (0 = unlimited)")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    ns = _parse_args(argv or sys.argv[1:])
    demo(
        run_dir=Path(ns.run_dir).expanduser(),
        eval_dir=Path(ns.eval_dir).expanduser() if ns.eval_dir else Path.cwd(),
        target_genes=ns.genes,
        max_positions_per_gene=ns.max_positions_per_gene,
        max_analysis_rows=ns.max_analysis_rows,
        train_schema_dir=Path(ns.train_schema_dir).expanduser(),
        verbosity=ns.verbose,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
