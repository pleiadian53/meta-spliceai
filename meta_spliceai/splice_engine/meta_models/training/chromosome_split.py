"""Utilities for chromosome-wise train/valid/test splitting.

This helper module provides:
1.  **LOCO-CV** (leave-one-chromosome-out) iterator that yields
    `(train_idx, valid_idx, test_idx, held_out)` tuples.
2.  Convenience function for a *single* chromosome hold-out split that matches
    the signature of our existing `datasets.train_valid_test_split` helper.
3.  Helpers to inspect chromosome distribution and automatically group very
    small chromosomes together so the test set remains large enough for stable
    metrics.

All functions operate purely on NumPy arrays of indices and therefore add zero
heavy dependencies or memory overhead.  They also accept Polars DataFrames
(with the minimal `.to_numpy()` conversions) so they slot into the existing
pipeline without surprises.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

__all__ = [
    "chromosome_counts",
    "group_chromosomes",
    "loco_cv_splits",
    "holdout_split",
]


# ---------------------------------------------------------------------------
#  Basic helpers
# ---------------------------------------------------------------------------

def chromosome_counts(chrom_array: np.ndarray) -> Dict[str, int]:
    """Return a mapping ``chrom -> row_count`` given an array of chromosome IDs."""
    uniq, counts = np.unique(chrom_array, return_counts=True)
    return dict(zip(uniq.tolist(), counts.tolist()))


def group_chromosomes(
    chrom_array: np.ndarray,
    *,
    min_rows: int = 1000,
    random_state: int | None = None,
) -> List[Sequence[str]]:
    """Return a list of chromosome groups for LOCO-CV.

    Each *group* is a ``Sequence[str]`` (usually a list) of chromosome names
    that will be *held out together* as the test set in one CV fold.

    Strategy
    --------
    • Chromosomes with ≥ ``min_rows`` are treated individually (1-element group).
    • Remaining *small* chromosomes are bucketed together so that each bucket
      reaches at least ``min_rows`` rows.  Simple greedy fill is used because
      perfect balancing is not crucial here.
    """
    rng = np.random.default_rng(random_state)

    counts = chromosome_counts(chrom_array)

    big_chroms = [c for c, n in counts.items() if n >= min_rows]
    small_chroms = [c for c, n in counts.items() if n < min_rows]

    # Shuffle small ones for random (but reproducible) grouping
    small_chroms = rng.permutation(small_chroms).tolist()

    groups: List[List[str]] = [[c] for c in big_chroms]

    # Greedy pack the small chromosomes
    bucket: List[str] = []
    bucket_rows = 0
    for chrom in small_chroms:
        bucket.append(chrom)
        bucket_rows += counts[chrom]
        if bucket_rows >= min_rows:
            groups.append(bucket)
            bucket = []
            bucket_rows = 0
    if bucket:  # leftover
        groups.append(bucket)

    # Deterministic ordering: sort groups by *total* rows descending so that
    # early CV folds use the biggest tests (helps catch issues sooner).
    groups.sort(key=lambda g: -sum(counts[c] for c in g))
    return groups


# ---------------------------------------------------------------------------
#  Splitting utilities
# ---------------------------------------------------------------------------

def _train_valid_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    valid_size: float,
    seed: int,
    gene_groups: np.ndarray | None = None,
):
    """Internal helper to carve out a validation set from *X/y*.

    If *gene_groups* is provided, it uses ``GroupShuffleSplit`` to avoid overlap
    between genes; otherwise it falls back to a stratified ``train_test_split``.
    """
    if valid_size == 0:
        return np.arange(len(X)), np.array([], dtype=int)

    # Ensure validation size is appropriate for the dataset size
    # Limit relative validation size to be at most 40% of available samples
    rel_valid = min(valid_size, 0.4)
    
    if gene_groups is not None:
        # Make sure we have enough unique gene groups for a meaningful split
        unique_groups = np.unique(gene_groups)
        if len(unique_groups) < 3:  # Need at least some groups for train and valid
            # Fall back to simple split without groups
            train_idx, valid_idx = train_test_split(
                np.arange(len(X)),
                test_size=rel_valid,
                random_state=seed,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
        else:
            try:
                gss = GroupShuffleSplit(n_splits=1, test_size=rel_valid, random_state=seed)
                train_idx, valid_idx = next(gss.split(X, y, gene_groups))
            except ValueError as e:
                # If GroupShuffleSplit fails (e.g., due to imbalanced groups),
                # fall back to a simple stratified split
                print(f"Warning: GroupShuffleSplit failed ({e}), falling back to stratified split")
                train_idx, valid_idx = train_test_split(
                    np.arange(len(X)),
                    test_size=rel_valid,
                    random_state=seed,
                    stratify=y if len(np.unique(y)) > 1 else None,
                )
    else:
        train_idx, valid_idx = train_test_split(
            np.arange(len(X)),
            test_size=rel_valid,
            random_state=seed,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
    return train_idx, valid_idx


def holdout_split(
    X: np.ndarray,
    y: np.ndarray,
    chrom_array: np.ndarray,
    *,
    holdout_chroms: Iterable[str],
    valid_size: float = 0.15,
    gene_array: np.ndarray | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/valid/test split adhering to a *fixed* chromosome hold-out.

    Parameters
    ----------
    X, y
        Full data matrices.
    chrom_array
        ``np.ndarray`` of chromosome IDs (same length as *X*).
    holdout_chroms
        Iterable of chromosome names to reserve *entirely* for the test set.
    valid_size
        Fraction of the **original** dataset allocated to *validation*.
    gene_array
        Optional array of gene IDs used for group-aware splitting within the
        train/valid portion.
    seed
        Random seed for reproducibility.
    """
    test_mask = np.isin(chrom_array, list(holdout_chroms))
    if test_mask.sum() == 0:
        raise ValueError("holdout_split: no rows matched the requested chromosomes")

    train_val_mask = ~test_mask

    X_train_val, X_test = X[train_val_mask], X[test_mask]
    y_train_val, y_test = y[train_val_mask], y[test_mask]
    inner_genes = gene_array[train_val_mask] if gene_array is not None else None

    train_idx_rel, valid_idx_rel = _train_valid_split(
        X_train_val,
        y_train_val,
        valid_size=valid_size / (1 - test_mask.mean()),
        seed=seed,
        gene_groups=inner_genes,
    )

    # Map *relative* indices back to absolute indices in the original arrays
    train_idx = np.flatnonzero(train_val_mask)[train_idx_rel]
    valid_idx = np.flatnonzero(train_val_mask)[valid_idx_rel]
    test_idx = np.flatnonzero(test_mask)

    return (
        train_idx,
        valid_idx,
        test_idx,
        X[train_idx],
        X[valid_idx],
        X[test_idx],
    )


def loco_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    chrom_array: np.ndarray,
    *,
    valid_size: float = 0.15,
    gene_array: np.ndarray | None = None,
    min_rows: int = 1000,
    seed: int = 42,
) -> Iterator[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield LOCO-CV splits.

    Each iteration yields ``(held_out_label, train_idx, valid_idx, test_idx)``.
    ``held_out_label`` is a *string* identifying the chromosome (or chromosome
    *group*) that forms the test set in that fold.
    """
    chrom_groups = group_chromosomes(chrom_array, min_rows=min_rows, random_state=seed)

    for group in chrom_groups:
        label = ",".join(group) if len(group) > 1 else group[0]
        test_mask = np.isin(chrom_array, group)
        if test_mask.sum() == 0:
            continue  # skip empty (should not happen)
        train_val_mask = ~test_mask

        # Split VALID from TRAIN using gene-wise grouping when available
        inner_genes = gene_array[train_val_mask] if gene_array is not None else None
        
        # Safety check: if train_val_mask is empty or nearly empty, skip this fold
        train_val_count = train_val_mask.sum()
        if train_val_count < 10:  # Need at least a few samples for a meaningful split
            print(f"Warning: Skipping fold {label} due to insufficient train/validation samples ({train_val_count})")
            continue
            
        # Calculate adjusted validation size, ensuring we don't get division by zero
        # Cap at 0.4 to avoid empty training sets
        if test_mask.mean() >= 0.99:  # Almost all samples in test set
            adj_valid_size = 0.0  # No validation set if almost all samples are in test
        else:
            # Scale valid_size by the proportion of non-test samples, but cap at 0.4
            adj_valid_size = min(valid_size / max(0.01, 1 - test_mask.mean()), 0.4)
            
        train_idx_rel, valid_idx_rel = _train_valid_split(
            X[train_val_mask],
            y[train_val_mask],
            valid_size=adj_valid_size,
            seed=seed,
            gene_groups=inner_genes,
        )

        train_idx = np.flatnonzero(train_val_mask)[train_idx_rel]
        valid_idx = np.flatnonzero(train_val_mask)[valid_idx_rel]
        test_idx = np.flatnonzero(test_mask)

        yield label, train_idx, valid_idx, test_idx
