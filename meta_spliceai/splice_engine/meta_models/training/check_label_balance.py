#!/usr/bin/env python3
"""Quick sanity-check: verify positive/negative label balance in train/valid/test.

Run this after generating / pruning the dataset to make sure we still have BOTH
positive and negative examples in each split; otherwise the model may achieve
perfect 1.0 metrics for the wrong reason (label leakage, unbalanced sampling, …).

Usage
-----
$ conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.training.check_label_balance \
        path/to/dataset.parquet

Optional flags allow tweaking the split sizes to mirror *Trainer* defaults.
The script exits with a **non-zero status** if any split is single-class.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.builder import preprocessing as _prep


def _value_counts(arr: np.ndarray) -> pd.Series:
    """Return counts for {0, 1} as a pandas Series."""
    return pd.Series(arr).value_counts().sort_index()


def _summarise(name: str, y_bin: np.ndarray) -> None:
    counts = _value_counts(y_bin)
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    total = pos + neg
    frac_pos = pos / total if total else 0.0
    print(f"{name:5s}: total={total:,}  pos={pos:,}  neg={neg:,}  pos%={frac_pos:5.1%}")
    if pos == 0 or neg == 0:
        raise ValueError(f"Split '{name}' is single-class – check sampling or leakage!")


def check_label_balance(
    dataset_path: Path | str,
    *,
    test_size: float = 0.2,
    valid_size: float | None = None,
    random_state: int = 42,
    label_col: str = "splice_type",
    group_col: str = "gene_id",
) -> None:
    """Load *dataset_path* and assert balanced labels in all splits."""
    df = datasets.load_dataset(dataset_path)

    # Preprocess to mirror training pipeline & preserve feature names
    _, y_series = _prep.prepare_training_data(
        df,
        label_col=label_col,
        return_type="pandas",
    )
    y_bin = (y_series.values > 0).astype(int)

    # Feature matrix not needed for balance check; pass dummy X with correct length
    X_dummy = np.empty((len(y_bin), 0))

    groups: Sequence[int] | None = (
        df[group_col].to_pandas().values if group_col in df.columns else None
    )

    splits = datasets.train_valid_test_split(
        X_dummy,
        y_bin,
        test_size=test_size,
        valid_size=valid_size,
        random_state=random_state,
        groups=groups,
        return_groups=False,
    )
    X_train, X_valid, X_test, y_train, y_valid, y_test = splits

    print("Label distribution (binary >0 as positive):")
    _summarise("train", y_train)
    if y_valid is not None:
        _summarise("valid", y_valid)
    _summarise("test", y_test)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Verify positive/negative label balance in each split.")
    p.add_argument("dataset", help="Path to the pruned dataset Parquet file")
    p.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split (default 0.2)")
    p.add_argument(
        "--valid-size",
        type=float,
        default=None,
        help="Optional absolute fraction for validation split (default None)",
    )
    p.add_argument("--random-state", type=int, default=42, help="Random seed (default 42)")
    args = p.parse_args()

    try:
        check_label_balance(
            args.dataset,
            test_size=args.test_size,
            valid_size=args.valid_size,
            random_state=args.random_state,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(2)
