#!/usr/bin/env python3
"""Diagnostic utility to detect obvious label leakage via near-perfect feature–target correlation.

Given a *feature manifest* (produced by ``demo_train_meta_model``) and the
associated Parquet dataset, this script computes the Pearson (default) or
Spearman correlation between every **numeric** feature and the binary label
``splice_type`` (donor / acceptor / neither).  Any feature with
``|ρ| ≥ threshold`` is flagged as suspicious and written to an output CSV (and
shown on stdout).

Usage
-----
$ conda run -n surveyor python -m \
    meta_spliceai.splice_engine.meta_models.training.leakage_probe \
    path/to/dataset.parquet  path/to/feature_manifest.csv  output/suspects.csv \
    --method pearson   --threshold 0.99   --sample 200000

Arguments
~~~~~~~~~
* **dataset** – Trimmed or raw dataset (can be a directory of Parquet shards).
* **manifest_csv** – The CSV emitted by the training run listing features → X.
* **out_csv** – Destination for the list of flagged features.

Options
~~~~~~~
* ``--method``    correlation metric: *pearson* (default) or *spearman*.
* ``--threshold`` absolute correlation cut-off (default 0.99).
* ``--sample``    randomly sample *n* rows for speed (default: use all rows).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr
from polars.exceptions import ComputeError

from meta_spliceai.splice_engine.meta_models import training as _training_pkg  # noqa: F401 – ensures package init

__all__: List[str] = [
    "probe_leakage",
]


def _compute_corr(series_x: pd.Series, series_y: pd.Series, method: str) -> float:
    if method == "pearson":
        # Avoid repeated RuntimeWarnings when a vector has zero variance
        with np.errstate(divide="ignore", invalid="ignore"):
            coef = series_x.corr(series_y, method="pearson")
        return coef
    if method == "spearman":
        # fallback to scipy for spearman to ignore NaNs uniformly
        coef, _ = spearmanr(series_x, series_y, nan_policy="omit")
        return coef
    raise ValueError(f"Unknown method: {method}")


def probe_leakage(dataset_path: Path | str, manifest_csv: Path | str, *, method: str = "pearson", threshold: float = 0.99, sample: int | None = None, top_n: int | None = None, return_all: bool = False) -> pd.DataFrame:
    """Return a DataFrame of feature correlations with splice_type.
    
    Parameters
    ----------
    dataset_path : Path | str
        Path to dataset
    manifest_csv : Path | str
        Path to feature manifest CSV
    method : str, default="pearson"
        Correlation method
    threshold : float, default=0.99
        Correlation threshold to mark features as leaky
    sample : int | None, default=None
        Number of samples to use
    top_n : int | None, default=None
        Limit results to top N highest correlations
    return_all : bool, default=False
        If True, return all features regardless of correlation value
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature correlations and leakage status
    """
    dataset_path = Path(dataset_path)
    manifest_csv = Path(manifest_csv)

    feature_df = pd.read_csv(manifest_csv)
    feature_cols: list[str] = feature_df["feature"].tolist()

    # Ensure label is present
    columns_to_load = feature_cols + ["splice_type"]

    # Load dataset lazily via Polars
    if dataset_path.is_dir():
        lf = pl.scan_parquet(str(dataset_path / "*.parquet"), missing_columns="insert")
    else:
        lf = pl.scan_parquet(str(dataset_path), missing_columns="insert")

    # Use helper function for robust column selection
    from .classifier_utils import select_available_columns
    lf, missing_cols, existing_cols = select_available_columns(
        lf, columns_to_load, context_name="Leakage Probe"
    )

    if sample is not None and sample > 0:
        # Polars >=1.31 has .sample on LazyFrame
        try:
            lf = lf.sample(n=sample, seed=42)
        except AttributeError:
            # Fallback: collect first then sample via pandas (less memory-friendly)
            pass

    # For memory-efficient Pearson we stream per feature via Polars.
    if method == "pearson":
        lf_corr = lf
        if sample is not None and sample > 0:
            try:
                lf_corr = lf_corr.sample(n=sample, seed=42)
            except AttributeError:
                lf_corr = lf_corr.limit(sample)
        records: list[dict[str, float]] = []  # store corr for all numeric cols
        const_feats: list[str] = []
        target = "splice_type"
        for col in feature_cols:
            try:
                corr_val = (
                    lf_corr.select(pl.corr(pl.col(col).cast(pl.Float64), pl.col(target).cast(pl.Float64)))
                    .collect(streaming=True)
                    .item()
                )
            except ComputeError:
                const_feats.append(col)
                continue
            except Exception:
                continue
            if corr_val is None or np.isnan(corr_val):
                continue
            records.append({"feature": col, "corr": float(corr_val)})
        if records:
            # summary about constant features
            if const_feats:
                print(
                    f"[leakage_probe] Skipped {len(const_feats)} constant/zero-variance features (e.g. {', '.join(const_feats[:5])})"
                )
            df_res = pd.DataFrame(records)
            # Rename column for clarity
            df_res = df_res.rename(columns={"corr": "correlation"})
            df_res["abs_corr"] = df_res["correlation"].abs()
            df_res["is_leaky"] = df_res["abs_corr"] >= threshold
            df_res = df_res.sort_values("abs_corr", ascending=False)
            
            # Either return all features or filter by top_n
            if not return_all and top_n is not None and top_n > 0:
                df_res = df_res.head(top_n)
            # Filter out non-leaky features unless return_all is True
            elif not return_all:
                df_res = df_res[df_res["is_leaky"]]
                
            return df_res.reset_index(drop=True).drop(columns=["abs_corr"])
        # No hits – possible non-numeric label; fall back to pandas path
        try:
            print("[leakage_probe] Falling back to pandas correlation – non-numeric target detected or no high-ρ hits via lazy path")
        except Exception:
            pass

    # Fallback path (Spearman, non-numeric Pearson, or unsupported Pearson) – materialise sample then use pandas
    if sample is not None and sample > 0:
        try:
            lf = lf.sample(n=sample, seed=42)
        except AttributeError:
            lf = lf.limit(sample)
    df = lf.collect(streaming=True).to_pandas()

    # Support both numeric (0/1/2) and string ("donor"/"acceptor"/"neither") label encodings
    splice_col = df["splice_type"]
    print(f"[leakage_probe] Processing splice_type with values: {splice_col.value_counts().head(5).to_dict()}")
    
    # Handle various formats of splice_type column (numeric, string, or mixed)
    if pd.api.types.is_numeric_dtype(splice_col):
        # Pure numeric format
        y_series = pd.Series((splice_col.values > 0).astype(int), name="y")
    else:
        # String or mixed format
        # Check for '0' string representation which should be treated as neither/0
        y_values = splice_col.apply(lambda x: 0 if str(x) in ['0', 'neither'] else 1).values
        y_series = pd.Series(y_values, name="y")

    records: list[dict[str, float]] = []  # store all numeric feature correlations
    const_feats: list[str] = []
    for col in feature_cols:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if s.nunique() <= 1:
            const_feats.append(col)
            continue
        coef = _compute_corr(s.astype(float), y_series, method)
        if np.isnan(coef):
            continue
        records.append({"feature": col, "corr": coef})

    if not records:
        return pd.DataFrame(columns=["feature", "correlation", "is_leaky"])
    df_res = pd.DataFrame(records)
    # Rename column for clarity
    df_res = df_res.rename(columns={"corr": "correlation"})
    df_res["abs_corr"] = df_res["correlation"].abs()
    df_res["is_leaky"] = df_res["abs_corr"] >= threshold
    df_res = df_res.sort_values("abs_corr", ascending=False)
    
    # Either return all features or filter by top_n
    if not return_all and top_n is not None and top_n > 0:
        df_res = df_res.head(top_n)
    # Filter out non-leaky features unless return_all is True
    elif not return_all:
        df_res = df_res[df_res["is_leaky"]]
        
    return df_res.reset_index(drop=True).drop(columns=["abs_corr"])


def _main(argv: list[str] | None = None) -> None:  # pragma: no cover – CLI only
    p = argparse.ArgumentParser(description="Detect possible label leakage via near-perfect feature-label correlation.")
    p.add_argument("dataset", help="Path to dataset (file or directory of Parquet shards)")
    p.add_argument("feature_manifest", help="CSV listing features actually used (feature_manifest.csv)")
    p.add_argument("out_csv", help="Where to save flagged features (CSV)")
    p.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
    p.add_argument("--return-all", action="store_true", help="Return all features, not just potentially leaky ones")
    p.add_argument("--threshold", type=float, default=0.99, help="absolute correlation cutoff")
    p.add_argument("--top-n", type=int, default=50, help="number of top features to output (0 = all)")
    p.add_argument("--sample", type=int, default=None, help="random sample size for speed")
    args = p.parse_args(argv)

    try:
        suspects_df = probe_leakage(
            args.dataset,
            args.feature_manifest,
            method=args.method,
            threshold=args.threshold,
            sample=args.sample,
            top_n=args.top_n if args.top_n != 0 else None,
            return_all=args.return_all,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[leakage_probe] ERROR: {exc}")
        sys.exit(2)

    if suspects_df.empty:
        print(f"[leakage_probe] No features with |ρ| ≥ {args.threshold}")
    else:
        suspects_df.to_csv(args.out_csv, index=False)
        print(f"[leakage_probe] FLAGGED {len(suspects_df)} features → {args.out_csv}")
        print(suspects_df.head(20).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    _main()
