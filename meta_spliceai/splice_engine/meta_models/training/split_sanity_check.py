#!/usr/bin/env python3
"""Sanity‐check dataset splitting.

Diagnostics provided:
1. **Majority‐class baseline** accuracy for train/valid/test – useful to
   contextualise model accuracy.
2. Verify that each `gene_id` is present in *only one* split (important to
   avoid leakage via correlated positions within the same gene).

The script performs **the same group‐aware split** logic as the training
pipeline (see ``datasets.train_valid_test_split``) so results should mirror the
real run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import polars as pl

from meta_spliceai.splice_engine.meta_models.training import datasets

_SPLITS = ("train", "valid", "test")


def _majority_baseline(y: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    counts = np.bincount(y, minlength=2)
    return counts.max() / counts.sum()


def _compute_stats(y: np.ndarray) -> Dict[str, float]:
    n_pos = int(y.sum())
    n_total = int(y.size)
    baseline = _majority_baseline(y)
    return dict(n_samples=n_total, n_positive=n_pos, baseline_acc=baseline)


def _split(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None, *, test_size: float, valid_size: float, random_state: int) -> Tuple[np.ndarray, ...]:
    return datasets.train_valid_test_split(
        X,
        y,
        test_size=test_size,
        valid_size=valid_size,
        random_state=random_state,
        groups=groups,
        return_groups=True,
    )


def main(argv: list[str] | None = None) -> None:  # pragma: no cover – CLI only
    p = argparse.ArgumentParser(description="Split sanity check: majority baseline and gene exclusivity.")
    p.add_argument("--dataset", required=True, help="Dataset directory or Parquet file")
    p.add_argument("--sample", type=int, default=None, help="Optional row sample for speed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--valid-size", type=float, default=0.15)
    args = p.parse_args(argv)

    # ---------------------------------------------------------------
    # Load dataset (only minimal columns)
    # ---------------------------------------------------------------
    cols_needed = ["splice_type", "gene_id"]
    lf = datasets.load_dataset(args.dataset, columns=cols_needed, lazy=True)
    if args.sample is not None and args.sample > 0:
        try:
            lf = lf.sample(n=args.sample, seed=args.seed)
        except AttributeError:
            lf = lf.limit(args.sample)
    df = lf.collect(streaming=True)

    y_bin = (df["splice_type"].to_numpy() > 0).astype(int)
    genes = df["gene_id"].to_numpy() if "gene_id" in df.columns else None

    # dummy X for splitting (we only need row indices)
    X_dummy = np.zeros((len(df), 1), dtype=float)

    splits = _split(
        X_dummy,
        y_bin,
        groups=genes,
        test_size=args.test_size,
        valid_size=args.valid_size,
        random_state=args.seed,
    )

    X_train, X_valid, X_test, y_train, y_valid, y_test, g_train, g_valid, g_test = splits

    # ---------------------------------------------------------------
    # 1. Majority‐class baseline
    # ---------------------------------------------------------------
    stats = {
        "train": _compute_stats(y_train),
        "valid": _compute_stats(y_valid),
        "test": _compute_stats(y_test),
    }

    # ---------------------------------------------------------------
    # 2. Gene exclusivity check
    # ---------------------------------------------------------------
    exclusivity_ok = True
    duplicates: dict[str, set[str]] = {}
    if genes is not None:
        sets = {
            "train": set(g_train),
            "valid": set(g_valid),
            "test": set(g_test),
        }
        for a in _SPLITS:
            for b in _SPLITS:
                if a >= b:
                    continue
                dup = sets[a].intersection(sets[b])
                if dup:
                    exclusivity_ok = False
                    duplicates[f"{a}&{b}"] = dup

    # ---------------------------------------------------------------
    # Print report
    # ---------------------------------------------------------------
    print("[split_sanity_check] Majority baseline:")
    for split in _SPLITS:
        s = stats[split]
        print(f"  {split:<5}  n={s['n_samples']:,}  pos={s['n_positive']:,}  baseline_acc={s['baseline_acc']:.3f}")

    if genes is None:
        print("No 'gene_id' column present – cannot test exclusivity.")
    elif exclusivity_ok:
        print("All gene_ids are exclusive to their split – OK.")
    else:
        print("[WARNING] Found duplicate gene_ids across splits")
        for pair, dups in duplicates.items():
            print(f"  {pair}: {len(dups)} duplicates (showing up to 5): {list(dups)[:5]}")
        sys.exit(2)


if __name__ == "__main__":
    main()
