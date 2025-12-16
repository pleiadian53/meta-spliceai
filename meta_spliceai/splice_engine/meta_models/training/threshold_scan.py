#!/usr/bin/env python3
"""Standalone threshold-sweep utility for the 3-class splice-site meta-model.

This is a thin wrapper around
``meta_spliceai.splice_engine.meta_models.training.classifier_utils.probability_diagnostics``
which already performs:

1. Calibration-curve + probability-histogram visualisation for donor and
   acceptor classes.
2. A binary relevance sweep (splice-site vs non-splice) across thresholds
   0.05‒0.90 computing F1.
3. Writing the best threshold suggestion to
   ``<run-dir>/threshold_suggestion.txt``.

The script simply exposes that functionality on the command line so that it can
be run *after* any training job without modifying the original training
scripts.

Example
-------
>>> python -m meta_spliceai.splice_engine.meta_models.training.threshold_scan \
        --dataset train_pc_1000/master \
        --run-dir runs/gene_cv_pc1000 \
        --sample 200000

The results will appear inside *run-dir*:
    probability_diagnostics.png   – calibration + histograms
    threshold_suggestion.txt      – tab-separated best threshold & F1
"""
from __future__ import annotations

import numpy as np
import argparse
from pathlib import Path

from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(description="Threshold sweep & probability diagnostics for a trained meta-model.")
    p.add_argument("--dataset", required=True, help="Master dataset directory (or single Parquet file)")
    p.add_argument("--run-dir", required=True, help="Directory containing trained model artefacts")
    p.add_argument("--sample", type=int, default=100_000, help="Row sample cap for *positive* rows (0 = keep all)")
    p.add_argument("--neg-ratio", type=float, default=None,
                   help="If given, sample negatives so that N_neg = N_pos * ratio."
                        " Set to 1 for balanced, 7 to mimic 1:7 pos:neg, etc.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    sample = None if args.sample == 0 else args.sample

    neg_ratio = args.neg_ratio
    if neg_ratio is None:
        # -----------------------------------------------------------
        # Auto-estimate ratio = N_neg / N_pos from dataset metadata
        # -----------------------------------------------------------
        import polars as pl
        ds_path = Path(args.dataset)
        if ds_path.is_dir():
            lf_cnt = pl.scan_parquet(str(ds_path / "*.parquet")).select("splice_type")
        else:
            lf_cnt = pl.scan_parquet(str(ds_path)).select("splice_type")
        counts_df = lf_cnt.group_by("splice_type").len().collect(streaming=True)
        # Handle both string and numeric label encodings
        label_col = counts_df["splice_type"]
        len_col = counts_df["len"]
        label_to_count = dict(zip(label_col.to_list(), len_col.to_list()))
        if all(isinstance(k, (int, np.integer)) for k in label_to_count):
            # Canonical numeric: 0 = neither, 1 = donor, 2 = acceptor
            n_neg = label_to_count.get(0, 0)
            n_pos = label_to_count.get(1, 0) + label_to_count.get(2, 0)
        else:
            # Mixed textual / numeric string labels
            n_neg = 0
            n_pos = 0
            for k, v in label_to_count.items():
                if k in ("0", 0, "neither"):
                    n_neg += v
                else:
                    n_pos += v
        neg_ratio = (n_neg / n_pos) if (n_pos and n_neg) else None
        print(f"[threshold_scan] label counts: pos={n_pos:,} neg={n_neg:,}")
        if neg_ratio is None:
            print("[threshold_scan] Warning: Could not determine neg_ratio automatically – defaulting to 1.0")
            neg_ratio = 1.0
        else:
            print(f"[threshold_scan] Auto-detected neg_ratio ≈ {neg_ratio:.2f}")

    # probability_diagnostics returns None; we just call it for side effects.
    _cutils.probability_diagnostics(
        args.dataset,
        args.run_dir,
        sample=sample,
        neg_ratio=neg_ratio,
    )

    out_path_png = Path(args.run_dir) / "probability_diagnostics.png"
    out_path_txt = Path(args.run_dir) / "threshold_suggestion.txt"

    print("[threshold_scan] Diagnostics written to:")
    print("  •", out_path_png)
    print("  •", out_path_txt)


if __name__ == "__main__":  # pragma: no cover
    main()
