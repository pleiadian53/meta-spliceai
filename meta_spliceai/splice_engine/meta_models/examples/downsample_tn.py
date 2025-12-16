# downsample_tn.py
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

def load_parquet(path: str | pathlib.Path,
                 columns: list[str] | None = None) -> pd.DataFrame:
    """Zero-copy load of a Parquet file (Arrow → pandas)."""
    return pq.read_table(path, columns=columns).to_pandas(integer_object_nulls=True)

def compute_neighborhood_flags(df: pd.DataFrame, window_nt: int = 50) -> np.ndarray:
    """Return a boolean mask: True for TN rows located within ±``window_nt`` bases of any non-TN.

    The implementation sorts positions once and then uses ``np.searchsorted`` for an
    *O(N log N)* overall complexity, which is faster than nested Python loops for big genes.
    """
    pos = df['position'].to_numpy()
    non_tn_pos = np.sort(df.loc[df['pred_type'] != 'TN', 'position'].to_numpy())

    if non_tn_pos.size == 0:
        # All rows are TN – trivial mask
        return np.zeros(len(df), dtype=bool)

    # For each position, find index of the first non-TN ≥ (pos - window) and check
    # whether it's within +window too.
    lo = np.searchsorted(non_tn_pos, pos - window_nt, side='left')
    hi = np.searchsorted(non_tn_pos, pos + window_nt, side='right')
    has_neighbor = hi > lo  # at least one non-TN in range
    return has_neighbor & (df['pred_type'] == 'TN')

def downsample_tn(
    in_path: str | pathlib.Path,
    out_path: str | pathlib.Path | None = None,
    hard_prob_thresh: float = 0.10,
    window_nt: int = 50,
    easy_neg_ratio: float = 1.0,
    rng: np.random.Generator | None = None
) -> pd.DataFrame:
    """
    Load a MetaSpliceAI training set, downsample true negatives, and
    optionally write a trimmed Parquet.

    Returns
    -------
    pandas.DataFrame
        The filtered frame.
    """
    rng = rng or np.random.default_rng()
    df = load_parquet(in_path)

    # --- masks --------------------------------------------------------------
    is_tn = df['pred_type'] == 'TN'
    is_non_tn = ~is_tn

    # Hard negatives: high donor/acceptor scores
    hard_neg = is_tn & (
        df[['donor_score', 'acceptor_score']].max(axis=1) >= hard_prob_thresh
    )

    # Neighborhood negatives: close to any non-TN
    neigh_neg = compute_neighborhood_flags(df, window_nt=window_nt)

    # Retain: non-TNs, hard negatives, neighborhood negatives
    keep_mask = is_non_tn | hard_neg | neigh_neg

    # Randomly sample remaining easy TNs
    remaining_easy = df.index[~keep_mask & is_tn]
    n_easy_to_keep = int(easy_neg_ratio * is_non_tn.sum())
    if n_easy_to_keep > 0 and remaining_easy.size > 0:
        n_easy_to_keep = min(n_easy_to_keep, remaining_easy.size)
        chosen = rng.choice(remaining_easy, size=n_easy_to_keep, replace=False)
        keep_mask.loc[chosen] = True

    df_trim = df.loc[keep_mask].reset_index(drop=True)

    # Persist if requested
    if out_path is not None:
        pq.write_table(pa.Table.from_pandas(df_trim), out_path)

    return df_trim


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser(description="Downsample TN rows in MetaSpliceAI training data")
    p.add_argument("in_parquet", help="Path to original Parquet")
    p.add_argument("out_parquet", help="Path to write trimmed Parquet")
    p.add_argument("--hard_prob", type=float, default=0.10,
                   help="Threshold for hard negatives (default 0.10)")
    p.add_argument("--window", type=int, default=50,
                   help="Neighbourhood window in nts (default 50)")
    p.add_argument("--easy_ratio", type=float, default=1.0,
                   help="Easy-TN : non-TN ratio (default 1.0)")
    args = p.parse_args(sys.argv[1:])

    _ = downsample_tn(
        args.in_parquet,
        args.out_parquet,
        hard_prob_thresh=args.hard_prob,
        window_nt=args.window,
        easy_neg_ratio=args.easy_ratio,
    )
