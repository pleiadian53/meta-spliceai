#!/usr/bin/env python3
"""Quick summary helper for a LOCO-CV run directory.

Example
-------
$ python -m meta_spliceai.splice_engine.meta_models.training.loco_report \
      --run-dir runs/loco_cv_cpu --top-genes 15

This prints:
• Fold table sorted by macro-F1
• Aggregate headline metrics
• Weakest chromosome (if any)
• Hardest *N* genes

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
#  Core helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path):
    with path.open("r") as fh:
        return json.load(fh)


def summarise_run(run_dir: Path, *, top_genes: int = 20, dataset_path: str | Path | None = None, delta_sample: int | None = None) -> None:  # pragma: no cover
    """Print a concise summary for *run_dir* (created by the LOCO scripts)."""
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    # 1. Fold-level overview --------------------------------------------------
    loco_csv = run_dir / "loco_metrics.csv"
    if not loco_csv.exists():
        print("[warn] loco_metrics.csv not found – is this a LOCO run directory?")
        return
    df_folds = pd.read_csv(loco_csv)
    print("\n=== Fold-level metrics (sorted by macro-F1 ascending) ===")
    print(df_folds.sort_values("test_macro_f1").to_string(index=False))

    # 2. Aggregate metrics ----------------------------------------------------
    agg_json = run_dir / "metrics_aggregate.json"
    if agg_json.exists():
        print("\n=== Aggregate metrics ===")
        print(json.dumps(_load_json(agg_json), indent=2))

    # 3. Identify weakest chromosome -----------------------------------------
    weakest_row = df_folds.nsmallest(1, "test_macro_f1").iloc[0]
    chrom_file = run_dir / f"metrics_{weakest_row.held_out}.json"
    if chrom_file.exists():
        print(f"\n=== Detailed metrics for weakest fold (chrom {weakest_row.held_out}) ===")
        print(json.dumps(_load_json(chrom_file), indent=2))

    # 4. Hard genes -----------------------------------------------------------
    gene_files: List[Path] = list(run_dir.glob("metrics_ENSG*.json"))
    if gene_files:
        df_genes = pd.DataFrame([_load_json(p) for p in gene_files])

        # Choose an available metric column (priority order)
        metric_priority = [
            "macro_f1",
            "f1",
            "test_macro_f1",
            "accuracy",
            "acc",
        ]
        sort_col = next((c for c in metric_priority if c in df_genes.columns), None)
        if sort_col is None:
            print("[warn] No recognised metric columns in gene JSON files; skipping gene ranking.")
        else:
            df_genes = df_genes.sort_values(sort_col)
            print(f"\n=== Hardest {min(top_genes, len(df_genes))} genes (by {sort_col}) ===")
            cols = [c for c in ["gene_id", sort_col, "n_rows"] if c in df_genes.columns]
            print(df_genes[cols].head(top_genes).to_string(index=False))

    # 5. Probability deltas (raw vs meta) ------------------------------------
    delta_csv = run_dir / "gene_deltas.csv"
    if not delta_csv.exists() and dataset_path:
        try:
            print("[info] gene_deltas.csv not found – computing gene deltas now ...")
            delta_csv = _cutils.gene_score_delta(dataset_path, run_dir, sample=delta_sample)
            print(f"[info] Wrote gene deltas → {delta_csv}")
        except Exception as exc:
            print(f"[warn] Failed to compute gene deltas: {exc}")
            delta_csv = run_dir / "gene_deltas.csv"  # reset

    # Also accept legacy .tsv if .csv not present
    if not delta_csv.exists():
        alt_tsv = run_dir / "gene_score_delta.tsv"
        if alt_tsv.exists():
            delta_csv = alt_tsv
    if delta_csv.exists():
        df_delta = pd.read_csv(delta_csv)
        if "delta_p_max" in df_delta.columns:
            print("\n=== Genes with largest improvement in max-prob (top 10) ===")
            print(df_delta.sort_values("delta_p_max", ascending=False)[["gene_id", "delta_p_max", "delta_acc"]].head(10).to_string(index=False))
            print("\n=== Genes with largest drop in max-prob (bottom 10) ===")
            print(df_delta.sort_values("delta_p_max").head(10)[["gene_id", "delta_p_max", "delta_acc"]].to_string(index=False))

            # Detailed mean probabilities before/after (top 5 only for brevity)
            detail_cols = [
                "gene_id",
                "mean_donor_raw",
                "mean_donor_meta",
                "mean_acceptor_raw",
                "mean_acceptor_meta",
                "mean_neither_raw",
                "mean_neither_meta",
                "mean_p_max_raw",
                "mean_p_max_meta",
            ]
            overlap = [c for c in detail_cols if c in df_delta.columns]
            if "delta_top1" in df_delta.columns:
                print("\n=== Genes with Top-1 hit improvements (meta > raw) ===")
                print(df_delta[df_delta["delta_top1"]>0][["gene_id","top1_raw","top1_meta","delta_top1"]].head(10).to_string(index=False))
                print("\n=== Genes that lost Top-1 hit (meta < raw) ===")
                print(df_delta[df_delta["delta_top1"]<0][["gene_id","top1_raw","top1_meta","delta_top1"]].head(10).to_string(index=False))

            if "delta_f1" in df_delta.columns:
                print("\n=== Genes with largest F1 gain (top 10) ===")
                print(
                    df_delta.sort_values("delta_f1", ascending=False)[[
                        "gene_id", "f1_raw", "f1_meta", "delta_f1", "n_fixed", "n_regressed"
                    ]].head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}")
                )
                print("\n=== Genes with worst F1 drop (bottom 10) ===")
                print(
                    df_delta.sort_values("delta_f1").head(10)[[
                        "gene_id", "f1_raw", "f1_meta", "delta_f1", "n_fixed", "n_regressed"
                    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}")
                )
            if len(overlap) == len(detail_cols):
                print("\n=== Detailed probabilities (top 5 improved genes) ===")
                print(
                    df_delta.sort_values("delta_p_max", ascending=False)
                    .head(5)[detail_cols]
                    .to_string(index=False, float_format=lambda x: f"{x:.3f}")
                )
                print("\n=== Detailed probabilities (top 5 worst genes) ===")
                print(
                    df_delta.sort_values("delta_p_max")
                    .head(5)[detail_cols]
                    .to_string(index=False, float_format=lambda x: f"{x:.3f}")
                )
        else:
            print("[warn] gene_deltas.csv present but column 'delta_p_max' missing.")

    # 6. Base vs Meta comparison ---------------------------------------------
    cmp_file = run_dir / "compare_base_meta.json"
    if cmp_file.exists():
        print("\n=== Base vs Meta ===")
        print(json.dumps(_load_json(cmp_file), indent=2))

    # 7. Threshold suggestion --------------------------------------------------
    thr_file = run_dir / "threshold_suggestion.txt"
    if thr_file.exists():
        with thr_file.open() as fh:
            lines = fh.read().strip().splitlines()
        if len(lines) >= 2:
            kv = dict(l.split("\t") for l in lines if "\t" in l)
            print("\n=== Suggested splice-site threshold ===")
            print(f"best_threshold ≈ {kv.get('best_threshold', 'NA')}  |  F1 ≈ {kv.get('F1', 'NA')}")

    # 8. SHAP importance -------------------------------------------------------
    shap_csv = run_dir / "shap_importance.csv"
    if shap_csv.exists():
        df_shap = pd.read_csv(shap_csv)
        if "importance" in df_shap.columns:
            print("\n=== Top features by SHAP/gain importance (top 10) ===")
            print(df_shap.sort_values("importance", ascending=False).head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 9. Neighbour-window diagnostic ------------------------------------------
    neigh_tsv = run_dir / "neighbour_window_meta_scores.tsv"
    if neigh_tsv.exists():
        df_neigh = pd.read_csv(neigh_tsv, sep="\t")
        center_rows = df_neigh[df_neigh["rel_pos"] == 0].copy()
        if not center_rows.empty:
            pred = center_rows[["meta_neither", "meta_donor", "meta_acceptor"]].idxmax(axis=1).str.replace("meta_", "")
            acc = (pred == center_rows["true_center_type"]).mean()
            print("\n=== Neighbour-window diagnostic ===")
            print(f"Centres evaluated: {len(center_rows)}  |  correct label @ centre: {acc*100:.1f}%")



# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(description="Summarise a LOCO-CV run directory.")
    p.add_argument("--run-dir", required=True, help="Path to run directory (e.g. runs/loco_cv_cpu)")
    p.add_argument("--top-genes", type=int, default=20, help="How many lowest-F1 genes to list")
    p.add_argument("--dataset", help="Path to original dataset used for training; if provided, gene deltas will be computed if missing")
    p.add_argument("--delta-sample", type=int, default=None, help="Optional row sample when computing gene deltas")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    summarise_run(Path(args.run_dir), top_genes=args.top_genes, dataset_path=args.dataset, delta_sample=args.delta_sample)


if __name__ == "__main__":  # pragma: no cover
    main()
