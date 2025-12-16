#!/usr/bin/env python3
"""Compare per-gene splice-site performance between *base* and *meta* models.

This diagnostic complements :pymod:`eval_meta_splice` which evaluates **one**
model at a time (typically the *meta* model).  In many workflows we want to
quantify the *improvement* relative to the raw SpliceAI (*base*) predictions
that served as input.  This module provides a lightweight helper that merges
the two ``full_splice_performance*.tsv`` tables and computes delta metrics.

The implementation is intentionally kept dependency–light and runs entirely in
memory via *Polars* which is already a core dependency of MetaSpliceAI.

Example
-------
>>> from meta_spliceai.splice_engine.meta_models.training import compare_splice_performance as csp
>>> merged_df = csp.compare_splice_performance(
...     meta_tsv="run1/full_splice_performance_meta.tsv",
...     base_tsv="run1/full_splice_performance.tsv",
...     out_tsv="run1/perf_meta_vs_base.tsv",
...     include_tns=True,
... )
>>> merged_df.head()

CLI usage
~~~~~~~~~
A small command-line wrapper is provided for ad-hoc use::

    python -m meta_spliceai.splice_engine.meta_models.training.compare_splice_performance \
        --meta-tsv run1/full_splice_performance_meta.tsv \
        --base-tsv run1/full_splice_performance.tsv \
        --out-tsv run1/perf_meta_vs_base.tsv
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any
import warnings

import polars as pl

__all__ = [
    "compare_splice_performance",
]


def _canonicalise_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure required columns are present with standard names.

    Older versions of the base-model evaluation use *splice_type* instead of
    *site_type* – we normalise here so the caller doesn't have to worry.
    """
    if "site_type" not in df.columns and "splice_type" in df.columns:
        df = df.rename({"splice_type": "site_type"})
    return df


_REQUIRED_COLS: List[str] = [
    "gene_id",
    "site_type",
    # core counts
    "TP",
    "FP",
    "FN",
    # core metrics
    "precision",
    "recall",
    "f1_score",
]

_OPTIONAL_TN_COLS: List[str] = [
    "TN",
    "specificity",
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "topk_acc",
]


from typing import Optional
import pandas as pd
import numpy as np

from .classifier_utils import _lazyframe_sample  # re-use existing helper

# Copy required functions from eval_meta_splice to break circular import
def _abs_positions(idx_arr: np.ndarray, gene_start: int, gene_end: int, strand: str) -> np.ndarray:
    """Convert relative indices to absolute genomic coordinates."""
    if strand == "+":
        return gene_start + idx_arr
    else:
        # On the minus strand the relative index counts from the *end*
        return gene_end - idx_arr

def _infer_rel_pos(df: pl.DataFrame) -> pl.Series:
    """Best-effort inference of relative position column name."""
    for cand in ("rel_position", "rel_pos", "position"):
        if cand in df.columns:
            return df[cand]
    raise KeyError("Dataset must contain a relative position column (rel_position / rel_pos / position)")

def _build_pred_results(
    df: pd.DataFrame,
    proba: np.ndarray,
    *,
    missing_policy: str = "skip",
) -> Dict[str, Dict[str, Any]]:
    """Convert per-row DataFrame + meta probabilities into pred_results dict required
    by the existing evaluation helpers.
    """
    pred_dict: Dict[str, Dict[str, Any]] = {}

    # Attach meta probabilities to df
    df = df.copy()
    # Class order: 0=neither, 1=donor, 2=acceptor
    df["donor_meta"] = proba[:, 1]
    df["acceptor_meta"] = proba[:, 2]

    # Ensure necessary metadata columns are present
    required_meta = {"gene_id", "strand", "gene_start", "gene_end"}
    missing = required_meta - set(df.columns)
    if missing:
        raise KeyError(f"Dataset missing columns required for meta evaluation: {missing}")

    # Relative position
    rel_pos = _infer_rel_pos(pl.from_pandas(df))
    df["rel_pos"] = rel_pos.to_numpy()

    # Build per-gene arrays
    for gid, sub in df.groupby("gene_id", sort=False):
        first_row = sub.iloc[0]
        gene_len = int(first_row["gene_end"] - first_row["gene_start"] + 1)
        if missing_policy == "zero":
            donor_arr = np.zeros(gene_len, dtype=np.float32)
            accept_arr = np.zeros(gene_len, dtype=np.float32)
        else:  # "skip" or "predict"
            donor_arr = np.full(gene_len, np.nan, dtype=np.float32)
            accept_arr = np.full(gene_len, np.nan, dtype=np.float32)

        # ------------------------------------------------------------------
        # Some datasets store *rel_pos* as 1-based inclusive coordinates.
        # Guard against out-of-bounds by detecting that pattern and shifting.
        # ------------------------------------------------------------------
        rel_idx = sub["rel_pos"].astype(int).to_numpy()
        if rel_idx.max() >= gene_len:
            # Off-by-one – convert to 0-based by subtracting 1
            rel_idx = rel_idx - 1
        donor_arr[rel_idx] = sub["donor_meta"].values.astype(np.float32)
        accept_arr[rel_idx] = sub["acceptor_meta"].values.astype(np.float32)

        # Absolute positions covered by the dataset rows (needed for missing-policy "skip")
        abs_idx = _abs_positions(rel_idx, int(first_row["gene_start"]), int(first_row["gene_end"]), str(first_row["strand"]))

        # Collect truth splice-site positions from dataset itself (optional)
        truth_donor: set[int] = set()
        truth_acceptor: set[int] = set()
        if "site_type" in sub.columns or "splice_type" in sub.columns:
            _stype_col = "site_type" if "site_type" in sub.columns else "splice_type"
            for pos, st in zip(abs_idx, sub[_stype_col].astype(str).tolist()):
                if st == "donor":
                    truth_donor.add(int(pos))
                elif st == "acceptor":
                    truth_acceptor.add(int(pos))
        pred_dict[gid] = {
            "strand": first_row["strand"],
            "gene_start": int(first_row["gene_start"]),
            "gene_end": int(first_row["gene_end"]),
            "donor_prob": donor_arr,
            "acceptor_prob": accept_arr,
            "covered_abs": set(abs_idx.tolist()),
            "truth_donor": truth_donor,
            "truth_acceptor": truth_acceptor,
        }
    return pred_dict

# Import _vectorised_site_metrics from eval_meta_splice - this should be safe now
# since we broke the cycle by copying _build_pred_results
try:
    from .eval_meta_splice import _vectorised_site_metrics
except ImportError:
    # Fallback - define a simple version if still having issues
    def _vectorised_site_metrics(ann_df, pred_dict, **kwargs):
        raise NotImplementedError("_vectorised_site_metrics not available due to import issues")


def _load_annotations(path: Path) -> pl.DataFrame:
    """Read splice-site annotations with sensible defaults.

    The Ensembl TSV has *chrom* as a mix of numeric and 'X', 'Y', so we
    force that column to Utf8 to avoid the Polars type inference error.
    """
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=10000,
        schema_overrides={"chrom": pl.Utf8},
    )


def _recompute_base(
    *,
    dataset_path: Path,
    annotations_path: Path,
    threshold: float,
    window: int,
    sample: Optional[int],
    include_tns: bool,
) -> pl.DataFrame:
    """Return base-model per-gene performance table built from SpliceAI raw scores."""
    cols = [
        "gene_id",
        "strand",
        "gene_start",
        "gene_end",
        "rel_position",
        "rel_pos",
        "position",
        "donor_score",
        "acceptor_score",
        "neither_score",
    ]
    if dataset_path.is_dir():
        lf = pl.scan_parquet(str(dataset_path / "*.parquet"), missing_columns="insert")
    else:
        lf = pl.scan_parquet(str(dataset_path), missing_columns="insert")
    lf = lf.select([c for c in cols if c in lf.columns])
    if sample is not None:
        lf = _lazyframe_sample(lf, sample, seed=42)
    df = lf.collect(streaming=True).to_pandas()

    proba = df[["neither_score", "donor_score", "acceptor_score"]].to_numpy(np.float32)
    pred_dict = _build_pred_results(df, proba, missing_policy="skip")

    ann_df = _load_annotations(annotations_path)
    res_pd = _vectorised_site_metrics(
        ann_df=ann_df,
        pred_dict=pred_dict,
        threshold=threshold,
        window=window,
        include_tns=include_tns,
    )
    return pl.from_pandas(res_pd)


def compare_splice_performance(
    *,
    meta_tsv: str | Path,
    base_tsv: str | Path | None = None,
    out_tsv: str | Path | None = None,
    include_tns: bool | None = None,
    # --- recompute options ---
    recompute_base: bool = False,
    dataset_path: str | Path | None = None,
    annotations_path: str | Path | None = None,
    threshold: float = 0.9,
    window: int = 2,
    sample: int | None = None,
    verbose: int = 0,
) -> pl.DataFrame:
    """Return *merged* DataFrame with *_base*, *_meta* and *_delta* columns.

    Parameters
    ----------
    meta_tsv, base_tsv
        Paths to the two TSV files produced by :pyfunc:`eval_meta_splice.
        meta_splice_performance`.  The *base* file is usually generated by the
        standalone SpliceAI evaluation script.
    out_tsv
        Optional path – if provided, the merged table is written to this TSV.
    include_tns
        If *None* (default) we infer from the presence of the TN-related columns
        in *meta_tsv*.  Set *True* / *False* to force behaviour.
    """
    meta_tsv = Path(meta_tsv)

    if not meta_tsv.exists():
        raise FileNotFoundError(f"meta TSV not found: {meta_tsv}")

    # ---------------------------------------------------------------
    # Load META dataframe first – needed to decide TN handling.
    # ---------------------------------------------------------------
    if verbose >= 1:
        print(f"Loading meta TSV: {meta_tsv}")
    meta_df = pl.read_csv(
        meta_tsv,
        separator="\t",
        infer_schema_length=10000,
        schema_overrides={"chrom": pl.Utf8},
    )

    # Determine whether TN-related cols should be considered
    if include_tns is None:
        include_tns = all(col in meta_df.columns for col in _OPTIONAL_TN_COLS)

    # ---------------------------------------------------------------
    # Obtain *base_df*: either from provided TSV or recomputed on-the-fly.
    # ---------------------------------------------------------------
    if recompute_base:
        if dataset_path is None or annotations_path is None:
            raise ValueError("--dataset and --annotations must be provided when --recompute-base is set")
        base_df = _recompute_base(
            dataset_path=Path(dataset_path),
            annotations_path=Path(annotations_path),
            threshold=threshold,
            window=window,
            sample=sample,
            include_tns=include_tns,
        )
    else:
        base_tsv = Path(base_tsv)  # type: ignore[arg-type]
        if not base_tsv.exists():
            raise FileNotFoundError(f"base TSV not found: {base_tsv}")
        base_df = pl.read_csv(
            base_tsv,
            separator="\t",
            infer_schema_length=10000,
            schema_overrides={"chrom": pl.Utf8},
        )

    meta_df = _canonicalise_columns(meta_df)
    base_df = _canonicalise_columns(base_df)

    # Determine whether TN-related cols should be considered
    if include_tns is None:
        include_tns = all(col in meta_df.columns for col in _OPTIONAL_TN_COLS)
    extra_cols = _OPTIONAL_TN_COLS if include_tns else []

    needed_cols = _REQUIRED_COLS + extra_cols
    missing_meta = [c for c in needed_cols if c not in meta_df.columns]
    missing_base = [c for c in needed_cols if c not in base_df.columns]
    if missing_meta:
        raise KeyError(f"meta TSV missing columns: {missing_meta}")
    if missing_base:
        raise KeyError(f"base TSV missing columns: {missing_base}")

    # Select required columns
    meta_sel = meta_df.select(needed_cols)
    base_sel = base_df.select(needed_cols)

    # ------------------ rename base cols with *_base suffix ------------------
    base_sel = base_sel.rename({
        c: f"{c}_base" for c in needed_cols if c not in {"gene_id", "site_type"}
    })
    meta_sel = meta_sel.rename({
        c: f"{c}_meta" for c in needed_cols if c not in {"gene_id", "site_type"}
    })

    # Inner join on keys
    merged = base_sel.join(meta_sel, on=["gene_id", "site_type"], how="inner")

    # Compute deltas (meta – base)
    metrics_for_delta = [
        "TP", "FP", "FN", "precision", "recall", "f1_score",
    ] + (extra_cols if include_tns else [])

    for m in metrics_for_delta:
        merged = merged.with_columns(
            (pl.col(f"{m}_meta") - pl.col(f"{m}_base")).alias(f"{m}_delta")
        )

    # Persist if requested
    if out_tsv is not None:
        out_tsv = Path(out_tsv)
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        merged.write_csv(out_tsv, separator="\t")
        print(f"[compare_splice_performance] wrote merged TSV to {out_tsv}")

    # High-level console summary – analogous to eval_meta_splice
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_f1_base = merged["f1_score_base"].mean()
        mean_f1_meta = merged["f1_score_meta"].mean()
    print(f"[compare_splice_performance] Global F1: base {mean_f1_base:.3f} → meta {mean_f1_meta:.3f} (Δ {mean_f1_meta - mean_f1_base:+.3f})")

    tp_delta = merged["TP_delta"].sum()
    fp_delta = merged["FP_delta"].sum()
    fn_delta = merged["FN_delta"].sum()
    print(f"[compare_splice_performance] Aggregate TP Δ {tp_delta:+}, FP Δ {fp_delta:+}, FN Δ {fn_delta:+}")

    improved = (merged["f1_score_delta"] > 0).sum()
    worsened = (merged["f1_score_delta"] < 0).sum()
    unchanged = (merged["f1_score_delta"] == 0).sum()
    print(f"[compare_splice_performance] Genes improved: {improved}, worsened: {worsened}, unchanged: {unchanged}")

    return merged


# -----------------------------------------------------------------------------
# CLI helper
# -----------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Compare base vs meta splice-site performance TSVs")
    ap.add_argument("--meta-tsv", required=True, help="Path to full_splice_performance_meta.tsv")
    ap.add_argument("--base-tsv", default=None, help="Path to full_splice_performance.tsv (base model)")
    ap.add_argument("--out-tsv", default=None, help="Save merged TSV with *_delta columns to this path")
    ap.add_argument("--include-tns", action="store_true", help="Expect TN-related columns and include their deltas")

    # Recompute-base specific
    ap.add_argument("--recompute-base", action="store_true", help="Recompute base performance from raw SpliceAI scores instead of loading TSV")
    ap.add_argument("--dataset", default=None, help="Parquet file/directory with the feature matrix rows (required with --recompute-base)")
    ap.add_argument("--annotations", default=None, help="Splice-site truth annotations file (required with --recompute-base)")
    ap.add_argument("--threshold", type=float, default=0.9, help="Probability threshold to call a site when recomputing base (default 0.9)")
    ap.add_argument("--window", type=int, default=2, help="Consensus window when matching predictions to truth (default 2)")
    ap.add_argument("--sample", type=int, default=None, help="Optional row sample when recomputing base")

    args = ap.parse_args()

    compare_splice_performance(
        meta_tsv=args.meta_tsv,
        base_tsv=args.base_tsv,
        out_tsv=args.out_tsv,
        include_tns=args.include_tns if args.include_tns else None,
        recompute_base=args.recompute_base,
        dataset_path=args.dataset,  # e.g. train_pc_1000/master/ or train_pc_1000/master/*.parquet
        annotations_path=args.annotations,
        threshold=args.threshold,
        window=args.window,
        sample=args.sample,
    )
