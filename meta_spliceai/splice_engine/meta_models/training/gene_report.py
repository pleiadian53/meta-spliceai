#!/usr/bin/env python3
"""Quick summary helper for a *gene-wise* CV run directory.

Example
-------
$ python -m meta_spliceai.splice_engine.meta_models.training.gene_report \
      --run-dir runs/gene_cv_pc1000 --top-genes 15

The script prints:
• Fold table sorted by macro-F1
• Aggregate headline metrics
• Hardest *N* genes (via gene_deltas.csv)
• Threshold suggestion, SHAP top features, neighbour-window diagnostic summary

The script is *read-only*: it never touches training artefacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils

__all__ = ["summarise_run"]

# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path):
    with path.open("r") as fh:
        return json.load(fh)


def summarise_run(
    run_dir: Path,
    *,
    top_genes: int = 20,
    dataset_path: str | Path | None = None,
    delta_sample: int | None = None,
) -> None:  # pragma: no cover
    """Print a concise summary for *run_dir* (created by gene-CV scripts)."""
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    # 1. Fold overview --------------------------------------------------------
    cv_csv = run_dir / "gene_cv_metrics.csv"
    if not cv_csv.exists():
        print("[warn] gene_cv_metrics.csv not found – is this a gene-CV run dir?")
        return
    df_folds = pd.read_csv(cv_csv)
    print("\n=== Fold-level metrics (sorted by macro-F1 ascending) ===")
    print(df_folds.sort_values("test_macro_f1").to_string(index=False))

    # 2. Aggregate metrics ----------------------------------------------------
    agg_json = run_dir / "metrics_aggregate.json"
    if agg_json.exists():
        print("\n=== Aggregate metrics ===")
        print(json.dumps(_load_json(agg_json), indent=2))

    # 3. Worst fold details ---------------------------------------------------
    worst = df_folds.nsmallest(1, "test_macro_f1").iloc[0]
    fold_json = run_dir / f"metrics_fold{int(worst.fold)}.json"
    if fold_json.exists():
        print(f"\n=== Detailed metrics for weakest fold (fold {int(worst.fold)}) ===")
        print(json.dumps(_load_json(fold_json), indent=2))

    # 4. Gene deltas / hard genes -------------------------------------------
    delta_csv = run_dir / "gene_deltas.csv"
    if not delta_csv.exists() and dataset_path:
        try:
            print("[info] gene_deltas.csv not found – computing gene deltas now …")
            delta_csv = _cutils.gene_score_delta(dataset_path, run_dir, sample=delta_sample)
            print(f"[info] Wrote gene deltas → {delta_csv}")
        except Exception as exc:
            print(f"[warn] Failed to compute gene deltas: {exc}")
    if delta_csv.exists():
        df_delta = pd.read_csv(delta_csv)
        metric = "delta_p_max" if "delta_p_max" in df_delta.columns else None
        if metric:
            print(f"\n=== Hardest {min(top_genes, len(df_delta))} genes (by {metric}) ===")
            print(df_delta.sort_values(metric).head(top_genes)[["gene_id", metric]].to_string(index=False))

    # 5. Base vs Meta ---------------------------------------------------------
    cmp_file = run_dir / "compare_base_meta.json"
    if cmp_file.exists():
        print("\n=== Base vs Meta ===")
        print(json.dumps(_load_json(cmp_file), indent=2))

    # 6. Threshold suggestion -------------------------------------------------
    thr_file = run_dir / "threshold_suggestion.txt"
    if thr_file.exists():
        kv = dict(l.split("\t") for l in thr_file.read_text().splitlines() if "\t" in l)
        print("\n=== Suggested splice-site threshold ===")
        print(f"best_threshold ≈ {kv.get('best_threshold', 'NA')}  |  F1 ≈ {kv.get('F1', 'NA')}")

    # 7. SHAP top features ----------------------------------------------------
    shap_csv = run_dir / "shap_importance.csv"
    if shap_csv.exists():
        df_shap = pd.read_csv(shap_csv)
        if "importance" in df_shap.columns:
            print("\n=== Top features by SHAP/gain importance (top 10) ===")
            print(
                df_shap.sort_values("importance", ascending=False)
                .head(10)
                .to_string(index=False, float_format=lambda x: f"{x:.4f}")
            )

    # 8. Neighbour-window diagnostic -----------------------------------------
    neigh_tsv = run_dir / "neighbour_window_meta_scores.tsv"
    if neigh_tsv.exists():
        df_neigh = pd.read_csv(neigh_tsv, sep="\t")
        center_rows = df_neigh[df_neigh["rel_pos"] == 0]
        if not center_rows.empty:
            pred = (
                center_rows[["meta_neither", "meta_donor", "meta_acceptor"]]
                .idxmax(axis=1)
                .str.replace("meta_", "")
            )
            acc = (pred == center_rows["true_center_type"]).mean()
            print("\n=== Neighbour-window diagnostic ===")
            print(f"Centres evaluated: {len(center_rows)}  |  correct label @ centre: {acc*100:.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(description="Summarise a gene-CV run directory.")
    p.add_argument("--run-dir", required=True, help="Path to run directory (e.g. runs/gene_cv_cpu)")
    p.add_argument("--top-genes", type=int, default=20, help="How many hardest genes to list")
    p.add_argument("--dataset", help="Original dataset path – used to compute gene deltas if missing")
    p.add_argument("--delta-sample", type=int, default=None, help="Row sample when computing deltas")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    summarise_run(Path(args.run_dir), top_genes=args.top_genes, dataset_path=args.dataset, delta_sample=args.delta_sample)


if __name__ == "__main__":  # pragma: no cover
    main()
