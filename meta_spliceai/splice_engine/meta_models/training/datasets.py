"""Dataset loading & splitting helpers for meta-model training.

This module is intentionally *lightweight*: it converts the enriched Parquet
files produced by ``meta_models.builder`` into «X, y» objects that are ready
for scikit-learn / XGBoost pipelines while keeping Polars as the internal
format for speed.

High-level responsibilities
---------------------------
1. Locate one or more Parquet files (individual batches or the master
   dataset directory) and read them into a single Polars ``DataFrame``.
2. Convert that Polars frame into feature matrix *X* and label vector *y*
   via ``builder.preprocessing.prepare_training_data``.
3. Provide convenience helpers for typical dataset splits (train/valid/test,
   cross-validation folds).

All heavy feature/label decisions (dropping leakage columns, imputing nulls,
label mapping, …) are delegated to the **preprocessing** module so that this
file remains small and easy to maintain.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import polars as pl
from polars.exceptions import ComputeError
from sklearn.model_selection import KFold, train_test_split  # type: ignore

from meta_spliceai.splice_engine.meta_models.builder import preprocessing as prep

__all__ = [
    "load_dataset",
    "prepare_xy",
    "train_valid_test_split",
    "cv_splits",
]

# ---------------------------------------------------------------------------
#  Loading helpers
# ---------------------------------------------------------------------------

def _collect_parquet_paths(source: str | Path) -> List[Path]:
    """Return a list of ``.parquet`` files given a *source* which can be:

    * Single Parquet file
    * Directory containing multiple Parquet files (non-recursive)
    * Directory representing a *dataset* written by ``pyarrow.dataset.write_dataset``
      (i.e. *source* itself is **not** a file but contains part files).  In this
      case we rely on Polars' ability to read a *dataset directory* directly.
    """
    src = Path(source)
    if src.is_file():
        return [src]
    # if directory has *.parquet files inside treat them as a multi-file dataset
    files = list(src.glob("*.parquet"))
    if files:
        return files
    # otherwise assume arrow dataset directory; Polars can read it directly
    return [src]


def load_dataset(
    source: str | Path,
    *,
    columns: Sequence[str] | None = None,
    lazy: bool = False,
    rechunk: bool = True,
) -> pl.DataFrame:
    """Read Parquet file(s) into a *single* Polars DataFrame.

    Parameters
    ----------
    source
        File or directory produced by the builder.
    columns
        Optional subset of columns to read.
    lazy
        If ``True`` returns a ``LazyFrame``; call ``.collect()`` later.  This is
        useful when chaining further Polars operations before materialising.
    rechunk
        Whether to call ``rechunk()`` at the end for contiguous memory layout.
    """
    paths = _collect_parquet_paths(source)

    def _safe_scan_parquet(pattern):
        """Call ``pl.scan_parquet`` with a graceful fallback for older Polars.

        Polars ≥0.20 supports the *missing_columns* keyword; earlier releases do
        not.  We attempt the new signature first, then transparently retry
        without the parameter when a ``TypeError`` is raised.  This keeps the
        codebase compatible across heterogeneous environments (e.g. CPU/GPU
        machines that may have different Polars versions).
        """
        try:
            return pl.scan_parquet(pattern, missing_columns='insert', extra_columns='ignore')
        except TypeError as _e:  # pragma: no cover
            # Try without missing_columns
            try:
                return pl.scan_parquet(pattern, extra_columns='ignore')
            except TypeError:
                # Very old polars - no schema handling parameters
                return pl.scan_parquet(pattern)

    if len(paths) == 1 and paths[0].is_dir():
        print(f"[load_dataset] Reading dataset directory: {paths[0]}")
        lf = _safe_scan_parquet(str(paths[0] / "*.parquet"))
    else:
        # Multiple explicit files -> glob them explicitly
        pattern_list = [str(p) for p in paths]
        print("[load_dataset] Reading dataset file(s):", ", ".join(pattern_list))
        lf = _safe_scan_parquet(pattern_list)

    if columns is None:
        # Keep everything except columns that preprocessing will certainly drop,
        # but *retain* the label and group columns needed later (splice_type & gene_id).
        # Retain mandatory grouping/metadata columns needed by downstream evaluation
        # scripts (e.g. LOCO-CV expects both 'chrom' and 'gene_id').
        # For transcript-level top-k accuracy, we also need to preserve 'position' and 'transcript_id'
        drop_set = set(prep.DEFAULT_DROP_COLUMNS) - {"splice_type", "gene_id", "chrom", "position", "transcript_id"}
        # Polars >=0.20 provides LazyFrame.collect_schema(). For older versions we fall back
        try:
            schema_obj = lf.collect_schema()
        except AttributeError:  # Older Polars
            schema_obj = lf.schema if hasattr(lf, "schema") else lf.collect().schema
        if isinstance(schema_obj, dict):
            schema_names = list(schema_obj.keys())
        else:
            # Fallbacks: Polars Schema has .names() method or .names attribute depending on version
            if hasattr(schema_obj, 'names'):
                schema_names = list(schema_obj.names) if isinstance(schema_obj.names, (list, tuple)) else list(schema_obj.names())
            else:
                schema_names = list(schema_obj)
        keep_cols = [c for c in schema_names if c not in drop_set]
        lf = lf.select(keep_cols)
    else:
        lf = lf.select(columns)

    if lazy:
        return lf  # type: ignore[return-value]

    # Auto-cap rows to mitigate OOM unless overridden. This samples *before* materialisation.
    max_rows_env = os.getenv("SS_MAX_ROWS")
    try:
        row_cap = int(max_rows_env) if max_rows_env else 0
    except ValueError:
        row_cap = 0
    DEFAULT_ROW_CAP = 500_000  # heuristic for ≈2‒4 GB RSS depending on width

    # Determine dataset size efficiently
    row_count = lf.select(pl.count()).collect(streaming=True).item()
    
    # Handle row cap logic: 0 means use full dataset, negative means use default
    if row_cap == 0:
        # Use full dataset (no sampling)
        print(f"[load_dataset] Using full dataset with {row_count:,} rows")
    elif row_cap < 0:
        # Use default row cap
        row_cap = DEFAULT_ROW_CAP
        print(f"[load_dataset] Using default row cap of {row_cap:,} rows")
    if row_cap > 0 and row_count > row_cap:
        print(f"[load_dataset] Row cap {row_cap:,} activated – sampling from {row_count:,} rows …")

        # ----------------------------------------------------------------------------
        # Prefer gene-level sampling: pick random genes until accumulated rows >= cap.
        # This avoids correlated examples across splits. If gene_id is absent we fall
        # back to the deterministic head() behaviour.
        # ----------------------------------------------------------------------------
        # Robustly extract column names across Polars versions
        try:
            schema_obj2 = lf.collect_schema()
        except AttributeError:
            schema_obj2 = lf.schema if hasattr(lf, "schema") else lf.collect().schema
        if isinstance(schema_obj2, dict):
            _schema_names = list(schema_obj2.keys())
        else:
            if hasattr(schema_obj2, "names"):
                _schema_names = list(schema_obj2.names) if isinstance(schema_obj2.names, (list, tuple)) else list(schema_obj2.names())
            else:
                _schema_names = list(schema_obj2)

        if "gene_id" in _schema_names:
            import numpy as _np  # local import to avoid global dependency at top

            # Compute per-gene row counts (streaming to keep memory low)
            gene_counts_polars = (
                lf.group_by("gene_id")
                .agg(pl.count().alias("n_rows"))
                .collect(streaming=True)
            )
            # Memory-optimized conversion: Polars → NumPy → Pandas
            # Direct .to_pandas() fails on M1 Macs with pyarrow 18.x + pandas 2.1.x
            import pandas as _pd
            gene_counts_numpy = gene_counts_polars.to_numpy()
            gene_counts_df = _pd.DataFrame(gene_counts_numpy, columns=gene_counts_polars.columns)

            ids = gene_counts_df["gene_id"].to_numpy()
            counts = gene_counts_df["n_rows"].to_numpy()
            rng = _np.random.default_rng(42)
            selected: list[str] = []
            remaining = row_cap
            for idx in rng.permutation(len(ids)):
                gid = ids[idx]
                n = counts[idx]
                if n == 0:
                    continue
                if n > remaining:
                    # Would exceed cap – stop here
                    break
                selected.append(gid)
                remaining -= n
                if remaining <= 0:
                    break
            lf = lf.filter(pl.col("gene_id").is_in(selected))
            # Safety guard: if still above cap, deterministic head
            if remaining < 0:
                lf = lf.limit(row_cap)
        else:
            # Fallback: deterministic head
            lf = lf.limit(row_cap)

    # Use streaming collection to keep peak RSS lower.
    df = lf.collect(streaming=True)
    return df.rechunk() if rechunk else df

# ---------------------------------------------------------------------------
#  Conversion to X, y
# ---------------------------------------------------------------------------

def prepare_xy(
    df: pl.DataFrame,
    *,
    label_col: str = "splice_type",
    return_X: str = "pandas",  # 'pandas' or 'polars'
    **prep_kwargs,
) -> Tuple[np.ndarray | pl.DataFrame, np.ndarray]:
    """Apply the standard preprocessing pipeline and return (X, y).

    ``prep_kwargs`` are forwarded to ``preprocessing.prepare_training_data``.
    """
    X, y_series = prep.prepare_training_data(
        df,
        label_col=label_col,
        return_type=return_X,
        **prep_kwargs,
    )

    if return_X == "pandas":
        X_mat = X.values  # pandas → NumPy for sklearn/xgboost
        y_vec = y_series.values
        return X_mat, y_vec
    else:  # polars
        return X, y_series.values  # keep X as Polars; y → NumPy

# ---------------------------------------------------------------------------
#  Splitting utilities
# ---------------------------------------------------------------------------

def train_valid_test_split(
    X,
    y,
    *,
    test_size: float = 0.2,
    valid_size: float | None = None,
    random_state: int = 42,
    groups=None,
    return_groups: bool = False,
):
    """Return (X_train, X_valid, X_test, y_train, y_valid, y_test).

    If *valid_size* is ``None`` the function returns only train/test splits
    (valid sets are ``None``).
    """
    # If *groups* are provided use group-aware splitting to avoid leakage.
    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit  # local import to avoid heavy dependency at top

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        (train_val_idx, test_idx) = next(gss.split(X, y, groups))
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        groups_train_val = groups[train_val_idx]
    else:
        # First carve out the TEST set directly according to *test_size*
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        groups_train_val = None

    if valid_size is None:
        if return_groups:
            g_train_val = groups_train_val
            g_test = groups[test_idx] if groups is not None else None
            return (X_train_val, None, X_test, y_train_val, None, y_test, g_train_val, None, g_test)
        return X_train_val, None, X_test, y_train_val, None, y_test

    # Now carve VALID out of the remaining pool so that its *absolute* share
    # matches ``valid_size`` relative to the *original* dataset.
    rel_valid = valid_size / (1 - test_size)

    if groups_train_val is not None:
        gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_valid, random_state=random_state)
        (train_idx, valid_idx) = next(gss_val.split(X_train_val, y_train_val, groups_train_val))
        X_train, X_valid = X_train_val[train_idx], X_train_val[valid_idx]
        y_train, y_valid = y_train_val[train_idx], y_train_val[valid_idx]
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_val,
            y_train_val,
            test_size=rel_valid,
            random_state=random_state,
            stratify=y_train_val,
        )

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def cv_splits(
    X,
    y,
    *,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    groups=None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, valid_idx)`` for cross-validation."""
    if groups is None:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return kf.split(X, y)

    try:
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=n_splits)
        return gkf.split(X, y, groups)
    except ImportError:
        raise ImportError("GroupKFold requires scikit-learn >=0.22")
