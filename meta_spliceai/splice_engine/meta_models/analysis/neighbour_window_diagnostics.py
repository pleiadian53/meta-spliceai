#!/usr/bin/env python3
"""Neighbour-window diagnostic to visualise how the meta-model scores bases
around true splice sites.

Usage
-----
python -m meta_spliceai.splice_engine.meta_models.analysis.neighbour_window_diagnostics \
    --run-dir runs/gene_cv_pc1000/fold0 \
    # --dataset is optional; if omitted the script resolves to
    # data/ensembl/spliceai_eval/meta_models/full_splice_positions_enhanced.parquet (or .tsv)
    --dataset data/ensembl/spliceai_eval/meta_models/full_splice_positions_enhanced.parquet \
    --annotations data/ensembl/splice_sites.tsv \
    --genes ENSG00000123456,ENSG00000234567 \
    --window 10

The script prints a small table for every focal splice site and saves a CSV in
*run-dir*/neighbour_window_meta_scores.tsv for downstream plotting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

# Internal helpers from the training package
from meta_spliceai.splice_engine.meta_models.training import (
    classifier_utils as _cutils,
)

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(
        description="Visualise meta-model probability profile in a ±N nt window around true splice sites.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True, help="Directory that holds the trained meta model")
    p.add_argument("--dataset", help="Optional per-base feature file; defaults to MetaModelDataHandler eval dir")
    p.add_argument("--annotations", required=True, help="Splice-site truth file (TSV/Parquet)")

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--genes", help="Comma-separated ENSEMBL gene IDs to inspect")
    grp.add_argument("--n-sample", type=int, help="Randomly pick this many genes from annotations")

    p.add_argument("--window", type=int, default=10, help="Neighbourhood size on each side")
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for ✓ marking")
    p.add_argument("--plot", action="store_true", help="Generate PNG line plots for each window")
    p.add_argument("--out-dir", help="Directory to store CSV/plots (defaults to --run-dir)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# Main logic
# --------------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model & feature manifest
    # ------------------------------------------------------------------
    predict_fn, feat_names = _cutils._load_model_generic(run_dir)
    print(f"[diag] Loaded meta model with {len(feat_names)} features")

    # ------------------------------------------------------------------
    # 2. Load annotations & choose genes/sites
    # ------------------------------------------------------------------
    ann_path = Path(args.annotations)
    ann_ext = ann_path.suffix.lower()
    if ann_ext == ".parquet":
        ann_df = pl.read_parquet(ann_path)
    else:
        sep = "\t" if ann_ext == ".tsv" else ","
        # Force 'chrom' to string so mixed numeric/chrX values don't trigger dtype errors
        ann_df = pl.read_csv(ann_path, 
            separator=sep, 
            schema_overrides={"chrom": pl.Utf8}
        )

    splice_df = ann_df.filter(pl.col("site_type").is_in(["donor", "acceptor"]))

    if args.genes:
        gene_list: List[str] = [g.strip() for g in args.genes.split(",") if g.strip()]
    else:
        gene_list = (
            splice_df.select("gene_id").unique().sample(args.n_sample, seed=42).get_column("gene_id").to_list()
        )
    splice_df = splice_df.filter(pl.col("gene_id").is_in(gene_list))
    print(f"[diag] Selected {len(gene_list)} genes with {splice_df.shape[0]} splice sites")

    # ------------------------------------------------------------------
    # 3. Load per-base feature rows covering ±window around each site
    # ------------------------------------------------------------------
    # Resolve dataset path – user provided or default
    if args.dataset:
        ds_path = Path(args.dataset)
    else:
        try:
            from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler

            handler = MetaModelDataHandler()
            ds_path = Path(handler.meta_dir) / "full_splice_positions_enhanced.parquet"
            if not ds_path.exists():
                # Fall back to TSV
                ds_path = ds_path.with_suffix(".tsv")
        except Exception as e:
            sys.exit(f"Unable to determine default dataset path – pass --dataset explicitly. ({e})")

    if not ds_path.exists():
        sys.exit(
            f"Dataset file {ds_path} not found – please provide full_splice_positions_enhanced.[parquet|tsv] or check MetaModelDataHandler.eval_dir"
        )

    # Determine columns to load – we need feature columns + identifiers
    req_cols = {"gene_id", "position", "strand", *feat_names}

    # ------------------------------------------------------------------
    # Robustly open dataset regardless of misleading file extension
    # ------------------------------------------------------------------
    if ds_path.is_dir():
        # Directory - scan all parquet files in the directory
        try:
            lf = pl.scan_parquet(str(ds_path / "*.parquet"), missing_columns="insert")
        except Exception as e:
            sys.exit(f"Failed to read parquet files from directory {ds_path}: {e}")
    elif ds_path.suffix == ".parquet":
        # Expected Parquet – but be defensive in case it is actually CSV
        try:
            lf = pl.scan_parquet(str(ds_path), missing_columns="insert")
        except Exception:
            lf = pl.scan_csv(str(ds_path))
            print(
                f"[diag] WARNING: Failed to read {ds_path.name} as Parquet despite .parquet extension – treated as CSV instead.",
                file=sys.stderr,
            )
    else:
        # Non-Parquet extension → presume delimited text but fall back to Parquet
        try:
            lf = pl.scan_csv(str(ds_path))
        except Exception as _csv_exc:
            try:
                lf = pl.scan_parquet(str(ds_path), missing_columns="insert")
                print(
                    f"[diag] NOTICE: Detected Parquet signature in {ds_path.name} (extension '{ds_path.suffix}'); using Parquet reader.",
                    file=sys.stderr,
                )
            except Exception:
                # Re-raise original CSV error for clarity
                raise _csv_exc
    available = set(lf.columns)
    missing = req_cols - available
    if missing:
        print(f"[Neighbor Diagnostics] Warning: Missing columns {sorted(missing)}, proceeding with available columns")
        # Filter to only use available columns
        req_cols = req_cols & available
        # Update feat_names to only include available features
        feat_names = [f for f in feat_names if f in available]
        print(f"[Neighbor Diagnostics] Reduced feature set to {len(feat_names)} available features")

    # Build a Polars LazyFrame containing only rows within any site ± window
    # Create a lookup dictionary of gene -> list[positions]
    gene_to_pos: dict[str, list[int]] = {}
    for row in splice_df.select(["gene_id", "position"]).iter_rows(named=True):
        gene_to_pos.setdefault(row["gene_id"], []).append(row["position"])

    filters = []
    for gid, pos_list in gene_to_pos.items():
        cond = (pl.col("gene_id") == gid) & (
            pl.any([pl.col("position").is_between(p - args.window, p + args.window) for p in pos_list])
        )
        filters.append(cond)

    lf_subset = lf.select(list(req_cols))
    if filters:
        lf_subset = lf_subset.filter(pl.any(filters))
    df_rows = lf_subset.collect(streaming=True).to_pandas()
    print(f"[diag] Loaded {len(df_rows)} neighbour rows for scoring")

    # ------------------------------------------------------------------
    # 4. Predict meta probabilities
    # ------------------------------------------------------------------
    # Ensure all required features are present, add missing ones with zeros
    from meta_spliceai.splice_engine.meta_models.training.classifier_utils import add_missing_features_with_zeros
    df_rows = add_missing_features_with_zeros(
        df_rows, feat_names, context_name="Neighbor Diagnostics"
    )
    
    X = df_rows[feat_names].to_numpy(np.float32)
    meta_proba = predict_fn(X)
    df_rows[["meta_neither", "meta_donor", "meta_acceptor"]] = meta_proba

    # ------------------------------------------------------------------
    # 5. Produce per-site diagnostic tables
    # ------------------------------------------------------------------
    records = []
    thr = args.threshold

    # Merge truth splice rows into df_rows for easy lookup
    truth_set = {(r["gene_id"], r["position"]): r["site_type"] for r in splice_df.iter_rows(named=True)}

    for gid in gene_list:
        site_pos_list = gene_to_pos.get(gid, [])
        for pos in site_pos_list:
            center_key = (gid, pos)
            true_type = truth_set[center_key]
            # Extract window rows for this site
            mask = (df_rows["gene_id"] == gid) & (df_rows["position"].between(pos - args.window, pos + args.window))
            win = df_rows.loc[mask].copy()
            win["rel_pos"] = win["position"] - pos
            win.sort_values("rel_pos", inplace=True)
            # Print quick text table
            donor_vals = win["meta_donor"].round(3).to_list()
            accept_vals = win["meta_acceptor"].round(3).to_list()
            neither_vals = win["meta_neither"].round(3).to_list()
            rel = win["rel_pos"].to_list()
            center_idx = win.index[win["rel_pos"] == 0].tolist()[0]
            pred_label = ["neither", "donor", "acceptor"][meta_proba[mask].argmax(axis=1)[win.index.get_loc(center_idx)]]
            correct = "✓" if (pred_label == true_type and win.loc[center_idx, f"meta_{true_type}"] >= thr) else "✗"
            print(f"\n[diag] {gid} {true_type} @ pos {pos} ({correct})")
            print("rel_pos : " + " ".join(f"{r:>6}" for r in rel))
            print("donor   : " + " ".join(f"{v:>6.3f}" for v in donor_vals))
            print("accept  : " + " ".join(f"{v:>6.3f}" for v in accept_vals))
            print("neither : " + " ".join(f"{v:>6.3f}" for v in neither_vals))
            # Optional PNG plot
            if args.plot:
                fig, ax = plt.subplots(figsize=(5, 2.8))
                ax.plot(win["rel_pos"], win["meta_donor"], label="donor", color="#1f77b4")
                ax.plot(win["rel_pos"], win["meta_acceptor"], label="acceptor", color="#ff7f0e")
                ax.plot(win["rel_pos"], win["meta_neither"], label="neither", color="#2ca02c")
                ax.axvline(0, color="k", lw=0.8, ls="--")
                ax.set_xlabel("Relative position")
                ax.set_ylabel("Probability")
                ax.set_title(f"{gid} {true_type} centre @ {pos}")
                ax.set_ylim(0, 1)
                ax.legend(frameon=False, fontsize="small")
                fig.tight_layout()
                png_path = out_dir / f"neigh_{gid}_{pos}.png"
                fig.savefig(png_path, dpi=160)
                plt.close(fig)
                print(f"[diag]   -> plot saved to {png_path.name}")
            # Collect for CSV
            for _, row in win.iterrows():
                rec = {
                    "gene_id": gid,
                    "center_pos": pos,
                    "rel_pos": int(row["rel_pos"]),
                    "strand": row["strand"],
                    "true_center_type": true_type,
                    "meta_donor": row["meta_donor"],
                    "meta_acceptor": row["meta_acceptor"],
                    "meta_neither": row["meta_neither"],
                }
                records.append(rec)
    pd.DataFrame.from_records(records).to_csv(out_dir / "neighbour_window_meta_scores.tsv", sep="\t", index=False)
    print(f"[diag] Saved detailed table to {out_dir / 'neighbour_window_meta_scores.tsv'}")


if __name__ == "__main__":  # pragma: no cover
    main()
