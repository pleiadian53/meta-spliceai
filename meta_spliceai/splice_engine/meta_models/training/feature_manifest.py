#!/usr/bin/env python3
"""Generate a *feature manifest* for a training dataset.

The manifest lists every feature column that ultimately enters the meta-model
(X) together with a placeholder *description* column so human annotators can
clarify each feature’s meaning later.

Why?  Reproducibility & interpretability.  By storing the exact set of features
used in a training run you can:
  • verify which inputs drove perfect metrics,
  • map SHAP importances back to real feature names,
  • track feature drift between dataset versions.

Usage
-----
$ conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.training.feature_manifest \
        path/to/dataset.parquet  output/feature_manifest.csv

The script automatically applies the *same* preprocessing pipeline used by
``Trainer`` so the feature list matches the model exactly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.builder import preprocessing as _prep

__all__: List[str] = [
    "build_feature_manifest",
]


def build_feature_manifest(dataset_path: str | Path) -> pd.DataFrame:
    """Return a DataFrame with columns *feature* and *description*.

    Parameters
    ----------
    dataset_path
        Path to a (possibly sharded) Parquet dataset *before* preprocessing.

    Notes
    -----
    1. We don’t need the full dataset — one shard is enough, but using the
       helper ensures consistent column selection regardless of how many files
       are passed.
    2. The description field is left blank for manual editing.
    """
    # Load full dataset lazily, then apply the canonical pipeline to obtain X.
    df = datasets.load_dataset(dataset_path)

    X_df, _ = _prep.prepare_training_data(
        df,
        label_col="splice_type",
        return_type="pandas",
    )

    manifest = pd.DataFrame({
        "feature": X_df.columns,
        "description": "",  # placeholder for human annotation
    })

    return manifest


def _main(argv: list[str] | None = None) -> None:  # pragma: no cover – CLI only
    p = argparse.ArgumentParser(description="Generate a feature manifest CSV.")
    p.add_argument("dataset", help="Path to (sharded) Parquet dataset")
    p.add_argument(
        "out_csv",
        help="Where to write the CSV (default: <dataset>/feature_manifest.csv)",
        nargs="?",
        default=None,
    )
    args = p.parse_args(argv)

    out_path = Path(args.out_csv) if args.out_csv else Path(args.dataset).with_suffix("").parent / "feature_manifest.csv"

    try:
        manifest_df = build_feature_manifest(args.dataset)
        manifest_df.to_csv(out_path, index=False)
        print(f"[feature_manifest] wrote {len(manifest_df)} rows → {out_path}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[feature_manifest] ERROR: {exc}")
        sys.exit(2)


if __name__ == "__main__":  # pragma: no cover
    _main()
