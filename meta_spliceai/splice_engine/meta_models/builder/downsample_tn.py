# downsample_tn.py
from __future__ import annotations
import pathlib
import numpy as np
import polars as pl

def _scan_min_cols(path: str | pathlib.Path):
    """Return a lazy scan over the minimal columns required for TN down-sampling."""
    # Add row_count to preserve row indices while streaming.
    return (
        pl.scan_parquet(str(path))
        .select(["pred_type", "donor_score", "acceptor_score", "position"])
        .with_row_count("row_nr")
    )

def compute_neighborhood_flags(position: np.ndarray,
                               pred_type: np.ndarray,
                               window_nt: int = 50) -> np.ndarray:
    """Boolean mask for TN rows within ±``window_nt`` bases of any non-TN row."""
    non_tn_pos = np.sort(position[pred_type != 'TN'])
    if non_tn_pos.size == 0:
        return np.zeros(position.size, dtype=bool)
    lo = np.searchsorted(non_tn_pos, position - window_nt, side='left')
    hi = np.searchsorted(non_tn_pos, position + window_nt, side='right')
    has_neighbor = hi > lo
    return has_neighbor & (pred_type == 'TN')

def downsample_tn(
    in_path: str | pathlib.Path,
    out_path: str | pathlib.Path | None = None,
    *,
    hard_prob_thresh: float = 0.10,
    window_nt: int = 50,
    easy_neg_ratio: float = 1.0,
    rng: np.random.Generator | None = None,
) -> pl.DataFrame:
    """
    Load a MetaSpliceAI training set, downsample true negatives, and
    optionally write a trimmed Parquet.

    Returns
    -------
    polars.DataFrame
    pandas.DataFrame
        The filtered frame.
    """
    rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------
    # Stage 1 – determine row indices to keep (streaming, low-mem) ------
    # ------------------------------------------------------------------
    min_df = _scan_min_cols(in_path).collect(streaming=True)

    pred_type = min_df["pred_type"].to_numpy()
    is_tn = pred_type == "TN"
    is_non_tn = ~is_tn

    donor = min_df["donor_score"].to_numpy()
    acceptor = min_df["acceptor_score"].to_numpy()
    hard_neg = is_tn & (np.maximum(donor, acceptor) >= hard_prob_thresh)

    pos = min_df["position"].to_numpy()
    neigh_neg = compute_neighborhood_flags(pos, pred_type, window_nt=window_nt)

    keep_mask = is_non_tn | hard_neg | neigh_neg

    # ------------------ sample easy TNs ---------------------------------
    remaining_easy_idx = np.where(~keep_mask & is_tn)[0]
    n_easy_to_keep = int(easy_neg_ratio * is_non_tn.sum())
    if n_easy_to_keep > 0 and remaining_easy_idx.size > 0:
        n_easy_to_keep = min(n_easy_to_keep, remaining_easy_idx.size)
        chosen_idx = rng.choice(remaining_easy_idx, size=n_easy_to_keep, replace=False)
        keep_mask[chosen_idx] = True

    keep_indices = set(np.where(keep_mask)[0].tolist())
    
    # Safety check: ensure no indices exceed the dataset size
    max_valid_index = len(min_df) - 1
    keep_indices = {idx for idx in keep_indices if idx <= max_valid_index}
    
    if len(keep_indices) == 0:
        raise ValueError("No valid rows to keep after downsampling")

    # ------------------------------------------------------------------
    # Stage 2 – boolean mask filtering (avoids .is_in() streaming bug) ----
    # ------------------------------------------------------------------
    # Create boolean mask for streaming-compatible filtering
    # This approach avoids .is_in() entirely, which is the root cause of the bug
    
    total_rows = len(min_df)
    keep_mask = np.zeros(total_rows, dtype=bool)
    keep_mask[list(keep_indices)] = True
    
    # Read full dataset and filter using boolean mask
    # Since downsampling typically reduces dataset size significantly, 
    # memory usage should be manageable for most realistic datasets
    df_full = pl.read_parquet(str(in_path))
    df_trim = df_full.filter(pl.Series("_keep", keep_mask))

    if out_path is not None:
        df_trim.write_parquet(out_path, compression="zstd")

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
