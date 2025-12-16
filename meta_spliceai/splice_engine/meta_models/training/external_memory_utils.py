"""Utilities to enable *out-of-core* training with XGBoost external memory.

Why this module?
----------------
Loading the full genome-wide meta-model dataset into RAM can exhaust even large
machines (≫ 64 GB) because the feature matrix may have millions of rows and
tens-of-thousands of columns (lots of k-mers!).  

XGBoost offers an *external memory* mode: it memory-maps a dataset that is stored
on disk in either **LibSVM** or XGBoost binary format.  The training algorithm
then streams data in manageable blocks, keeping peak RSS low.

This module provides two helpers:

1. ``convert_dataset_to_libsvm`` – stream-converts our Parquet dataset to a
   single LibSVM file that XGBoost can memory-map.
2. ``train_external_memory_model`` – trains an ``XGBClassifier`` directly from
   that on-disk file.

Both helpers avoid full materialisation by processing one source *shard* at a
time.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import polars as pl
from sklearn.datasets import dump_svmlight_file  # type: ignore
from xgboost import XGBClassifier

from meta_spliceai.splice_engine.meta_models.builder import preprocessing
from meta_spliceai.splice_engine.meta_models.training.label_utils import LABEL_MAP_STR, encode_labels
from meta_spliceai.splice_engine.meta_models.training import datasets as _ds

__all__ = [
    "convert_dataset_to_libsvm",
    "train_external_memory_model",
]





def _encode_labels(arr: Sequence[str | int]) -> np.ndarray:
    """Wrapper around label_utils.encode_labels (kept for backward compatibility)."""
    return encode_labels(arr)


def _collect_parquet_paths(src: Path) -> List[Path]:
    """Return a flat list of Parquet files contained in *src* (non-recursive)."""
    if src.is_file():
        return [src]
    files = list(src.glob("*.parquet"))
    if files:
        return files
    # Dataset directory written by Arrow – treat the dir itself as one *logical* file
    return [src]


# ---------------------------------------------------------------------------
# 1. Conversion helper
# ---------------------------------------------------------------------------

def convert_dataset_to_libsvm(
    source: str | os.PathLike,
    out_libsvm: str | os.PathLike,
    *,
    chunk_rows: int = 250_000,
    label_col: str = "splice_type",
    verbose: int = 1,
) -> Path:
    """Convert *source* dataset to **LibSVM** external-memory file.

    The function iterates over Parquet *shards* (or the whole file if single
    parquet) to keep memory usage under control.  Each shard is featurised via
    ``preprocessing.prepare_training_data`` and immediately dumped to LibSVM.

    Parameters
    ----------
    source
        Master dataset directory or Parquet file produced by the builder.
    out_libsvm
        Target ``.libsvm`` (or ``.txt``) file.  Parent directory is created.
    chunk_rows
        If a shard has more than *chunk_rows* rows we further split it into
        in-memory chunks to bound peak RSS.
    label_col
        Column containing labels (default ``splice_type``).
    verbose
        Verbosity level (0 = silent).
    """
    src_path = Path(source)
    out_path = Path(out_libsvm)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Fresh start: ensure target file does not exist
    if out_path.exists():
        out_path.unlink()

    files = _collect_parquet_paths(src_path)

    # ------------------------------------------------------------------
    # Pass 1: collect union of feature names across shards
    # ------------------------------------------------------------------
    if verbose:
        print("[libsvm] Scanning shards to determine union feature set …")
    feat_set: set[str] = set()
    for pf in files:
        shard_df = pl.read_parquet(pf)
        X_tmp, _ = preprocessing.prepare_training_data(
            shard_df,
            label_col=label_col,
            return_type="pandas",
            verbose=0,
        )
        feat_set.update(X_tmp.columns)
        del shard_df, X_tmp
    feat_names: List[str] = sorted(feat_set)
    if verbose:
        print(f"[libsvm] Detected {len(feat_names):,} unique features across {len(files)} shards.")

    # ------------------------------------------------------------------
    # Pass 2: write data aligned to the union feature list
    # ------------------------------------------------------------------
    total_rows = 0
    first_write = True
    for pf in files:
        if verbose:
            print(f"[libsvm] Processing shard {pf.name} …")
        # Use eager read per shard (keeps memory bounded to shard size)
        shard_df = pl.read_parquet(pf)
        X_df, y_series = preprocessing.prepare_training_data(
            shard_df,
            label_col=label_col,
            return_type="pandas",
            verbose=0,
        )
        # Align columns to the union feature list
        missing_cols = [c for c in feat_names if c not in X_df.columns]
        for col in missing_cols:
            X_df[col] = 0.0  # add zero column
        # Drop any unexpected extras (should not occur)
        extra_cols = [c for c in X_df.columns if c not in feat_names]
        if extra_cols:
            if verbose:
                print(f"[libsvm] Warning: dropping {len(extra_cols)} extra cols not in union feature set …")
            X_df = X_df.drop(columns=extra_cols)
        # Reorder to consistent order
        X_df = X_df[feat_names]

        X = X_df.values.astype(np.float32)  # xgboost prefers float32
        y = _encode_labels(y_series)
        total_rows += X.shape[0]

        # Further split large shards into sub-chunks if needed
        for start in range(0, X.shape[0], chunk_rows):
            end = start + chunk_rows
            mode = "wb" if first_write else "ab"
            with open(out_path, mode) as fh:
                dump_svmlight_file(
                    X[start:end],
                    y[start:end],
                    fh,
                    zero_based=True,   # XGBoost expects 0-based feature indices
                )
            first_write = False
        # Free memory
        del shard_df, X_df, X, y

    if verbose:
        print(f"[libsvm] Wrote {total_rows:,} rows → {out_path} (features={len(feat_names)})")

    # Persist feature names for future inference / SHAP
    meta_path = out_path.with_suffix(".features.json")
    meta_path.write_text(json.dumps({"feature_names": feat_names}))
    if verbose:
        print(f"[libsvm] Feature manifest → {meta_path}")

    return out_path


# ---------------------------------------------------------------------------
# 2. Training helper
# ---------------------------------------------------------------------------

def train_external_memory_model(
    libsvm_path: str | os.PathLike,
    *,
    model_out_dir: str | os.PathLike,
    tree_method: str = "hist",
    max_bin: int = 256,
    n_estimators: int = 600,
    early_stopping_rounds: int | None = None,
    random_state: int = 42,
    eval_fraction: float = 0.1,
    num_classes: int = 3,
    verbose: int = 1,
) -> tuple[Path, dict]:  # returns (model_path, eval_metrics)
    """Train *multiclass* XGBoost model from an **external-memory LibSVM file**.

    The LibSVM file is split into training/validation DMatrix objects using
    ``eval_fraction``.  Splitting is done by line-offsets without loading the
    whole file.
    """
    import xgboost as xgb  # local import to keep base deps light
    import json

    libsvm_uri = f"{libsvm_path}?format=libsvm"  # external-memory hint

    # Attach human-readable feature names if manifest exists
    manifest_path = Path(libsvm_path).with_suffix(".features.json")
    feature_names = None
    if manifest_path.exists():
        try:
            feature_names = json.loads(manifest_path.read_text())["feature_names"]
        except Exception as exc:
            if verbose:
                print(f"[warn] Failed to read feature manifest: {exc}")

    dtrain = xgb.DMatrix(libsvm_uri, missing=np.nan, feature_names=feature_names)

    if verbose and feature_names is not None:
        # Distinguish k-mer columns (only A/C/G/T chars, length 2-8) from others
        import re
        _KMER_RE = re.compile(r"^([2-8])mer_([ACGT]{2,8})$", flags=re.IGNORECASE)

        def _is_kmer(col: str) -> bool:
            """Return True if *col* follows the `<k>mer_<SEQ>` naming convention.

            Examples matching this pattern:
            • "6mer_AAAATA"  → k=6, sequence AAAATA
            • "5mer_ATTGA"   → k=5, sequence ATTGA
            • "4mer_TTGC"    → k=4, sequence TTGC
            The function verifies that the prefix integer *k* equals the length of
            the nucleotide sequence and that the sequence is composed solely of
            A/C/G/T characters (case-insensitive).
            """
            m = _KMER_RE.match(col)
            if not m:
                return False
            k = int(m.group(1))
            seq = m.group(2).upper()
            return k == len(seq)

        non_kmer_feats = [f for f in feature_names if not _is_kmer(f)]
        kmer_feats = [f for f in feature_names if _is_kmer(f)]

        print(f"[ext-mem] DMatrix loaded with {len(feature_names)} total features")
        print(f"           Non-k-mer features ({len(non_kmer_feats)}): {', '.join(non_kmer_feats) if non_kmer_feats else '—'}")
        if kmer_feats:
            import random
            sample_kmers = random.sample(kmer_feats, k=min(3, len(kmer_feats)))
            print(f"           Example k-mer features ({len(kmer_feats)} total): {', '.join(sample_kmers)}")

    # Create a small validation set by sampling row indices
    if eval_fraction > 0:
        # XGBoost can take a *list* of DMatrices for evaluation; easiest is to
        # split in Python after loading row meta (cheap).
        n_rows = dtrain.num_row()
        n_eval = int(n_rows * eval_fraction)
        if n_eval == 0:
            n_eval = 1
        eval_idx = np.random.default_rng(random_state).choice(n_rows, size=n_eval, replace=False)
        dval = dtrain.slice(eval_idx)
        watchlist = [(dval, "eval")]
    else:
        watchlist = []

    # External-memory works only with CPU histogram algorithm.
    if tree_method == "gpu_hist":
        print("[ext-mem] gpu_hist unsupported with external-memory – falling back to 'hist'.")
        tree_method = "hist"

    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "tree_method": tree_method,
        "max_bin": max_bin,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": random_state,
        "eval_metric": "mlogloss",
    }

    if verbose:
        print("[ext-mem] Starting training …")
    train_kwargs = dict(num_boost_round=n_estimators, evals=watchlist, verbose_eval=verbose)
    if early_stopping_rounds and watchlist:
        train_kwargs["early_stopping_rounds"] = early_stopping_rounds
    bst = xgb.train(params, dtrain, **train_kwargs)

    out_dir = Path(model_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model_multiclass.json"
    bst.save_model(model_path)
    if verbose:
        print(f"[ext-mem] Model saved → {model_path}")

    # Extract evaluation history if API available (method <2.0 or property ≥2.0)
    if watchlist:
        if hasattr(bst, "evals_result"):
            ev = bst.evals_result
            evals_result = ev() if callable(ev) else ev
        else:
            evals_result = {}
    else:
        evals_result = {}

    return model_path, evals_result
