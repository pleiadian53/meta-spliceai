"""
Genomic feature extraction for meta models.

This module handles the extraction of genomic features such as gene-level features,
length features, performance features, and other genomic context information.
These features are important for meta models that correct base model predictions.
"""

import pandas as pd
import polars as pl
from typing import Union
import os
from pathlib import Path

# Std lib
import os

# Package constants
from meta_spliceai.splice_engine.meta_models import constants as const

# Local utilities re-used from other sub-modules -----------------------------
from meta_spliceai.splice_engine.extract_genomic_features import (
    FeatureAnalyzer,
    SpliceAnalyzer,  # defined in the same module
    run_genomic_gtf_feature_extraction,
    compute_total_lengths,
    compute_distances_with_strand_adjustment,
)

from meta_spliceai.splice_engine.utils_df import join_and_remove_duplicates

# Safe fallback imports for helper diagnostics (optional)
try:
    from meta_spliceai.splice_engine.utils_doc import print_emphasized, print_with_indent
except ModuleNotFoundError:  # during unit tests utils_doc might be absent
    def print_emphasized(msg):
        print(msg)

    def print_with_indent(msg, indent_level=1):
        print(" " * (indent_level * 2) + msg)


def merge_contextual_features(pos_df: pl.DataFrame,
                              gene_meta_df: pl.DataFrame|None = None,
                              kmer_df: pl.DataFrame|None = None,
                              on_gid: str = "gene_id") -> pl.DataFrame:
    """
    Append extra feature blocks (gene-level stats, pre-computed k-mer counts) to
    the per-position frame produced by `enhanced_process_predictions_with_all_scores`.

    Parameters
    ----------
    pos_df : Polars DataFrame
        Must contain column `gene_id`.
    gene_meta_df : Polars DataFrame, optional
        Frame with one row per gene_id (length, GC%, exon_count, etc.).
    kmer_df : Polars DataFrame, optional
        Frame with k-mer frequencies keyed by `gene_id`.
    """
    pass


def incorporate_gene_level_features(
    df_trainset: Union[pd.DataFrame, pl.DataFrame],
    fa=None,
    **kwargs
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Incorporate gene-level features from GTF data.
    
    This is a wrapper around the original incorporate_gene_level_features function
    that maintains compatibility with the existing codebase.
    
    Parameters
    ----------
    df_trainset : Union[pd.DataFrame, pl.DataFrame]
        DataFrame with base features
    fa : FeatureAnalyzer, optional
        FeatureAnalyzer instance, by default None
        
    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        DataFrame with added gene-level features
    """
    # ------------------------------------------------------------------
    # If no FeatureAnalyzer supplied, create a lightweight one
    # ------------------------------------------------------------------
    if fa is None:
        fa = FeatureAnalyzer(**kwargs)

    col_tid = kwargs.get("col_tid", const.col_tid)
    col_gid = kwargs.get("col_gid", const.col_gid)

    # 1) Gene-level features from GTF
    out_path = os.path.join(fa.analysis_dir, "genomic_gtf_feature_set.tsv")
    gene_fs_df = run_genomic_gtf_feature_extraction(
        fa.gtf_file, output_file=out_path, overwrite=getattr(fa, "overwrite", False)
    )

    # ------------------------------------------------------------------
    # Ensure (gene_id, transcript_id) is unique to avoid row inflation
    # ------------------------------------------------------------------
    rows_before_dedup = gene_fs_df.shape[0]
    gene_fs_df = gene_fs_df.unique(subset=[col_gid, col_tid])
    removed = rows_before_dedup - gene_fs_df.shape[0]
    if removed:
        print_with_indent(
            f"[gene-level] Warning – removed {removed:,} duplicate rows in GTF feature table",
            indent_level=2,
        )

    print_with_indent(
        f"[gene-level] Merging {gene_fs_df.shape[0]:,} unique transcript rows of gene features",
        indent_level=1,
    )

    df_trainset = join_and_remove_duplicates(
        df_trainset, gene_fs_df, on=[col_gid, col_tid], how="left", verbose=1
    )

    return df_trainset


def incorporate_length_features(
    df_trainset: Union[pd.DataFrame, pl.DataFrame],
    fa=None,
    **kwargs
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Incorporate exon-intron length features.
    
    This is a wrapper around the original incorporate_length_features function
    that maintains compatibility with the existing codebase.
    
    Parameters
    ----------
    df_trainset : Union[pd.DataFrame, pl.DataFrame]
        DataFrame with base features
    fa : FeatureAnalyzer, optional
        FeatureAnalyzer instance, by default None
        
    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        DataFrame with added length features
    """
    if fa is None:
        fa = FeatureAnalyzer(**kwargs)

    col_tid = kwargs.get("col_tid", const.col_tid)

    # Try to use cached total lengths file first (much faster)
    from meta_spliceai.system.genomic_resources import Registry
    try:
        registry = Registry()
        cached_lengths_path = Path(registry.cfg.data_root) / "spliceai_analysis" / "total_intron_exon_lengths.tsv"
    except Exception:
        # Fallback: use FeatureAnalyzer's data directory
        cached_lengths_path = Path(fa.data_dir) / "spliceai_analysis" / "total_intron_exon_lengths.tsv"
    
    if cached_lengths_path.exists():
        print_with_indent(
            f"[length] Loading cached total lengths from {cached_lengths_path}",
            indent_level=1,
        )
        # Use pandas for consistency with compute_total_lengths output
        length_df = pd.read_csv(cached_lengths_path, sep='\t')
    else:
        print_with_indent(
            f"[length] Computing total lengths from GTF (no cache found)",
            indent_level=1,
        )
        length_df = compute_total_lengths(fa.gtf_file)

    print_with_indent(
        f"[length] Merging total exon/intron lengths (rows={length_df.shape[0]:,})",
        indent_level=1,
    )

    df_trainset = join_and_remove_duplicates(
        df_trainset, length_df, on=[col_tid], how="left", verbose=1
    )

    return df_trainset


def incorporate_performance_features(
    df_trainset: Union[pd.DataFrame, pl.DataFrame],
    fa=None,
    **kwargs
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Incorporate performance profile features.
    
    This is a wrapper around the original incorporate_performance_features function
    that maintains compatibility with the existing codebase.
    
    Parameters
    ----------
    df_trainset : Union[pd.DataFrame, pl.DataFrame]
        DataFrame with base features
    fa : FeatureAnalyzer, optional
        FeatureAnalyzer instance, by default None
        
    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        DataFrame with added performance features
    """
    if fa is None:
        fa = FeatureAnalyzer(**kwargs)

    perf_df = fa.retrieve_gene_level_performance_features(**kwargs)

    # Ensure one row per (gene_id, splice_type) before merging
    required_cols = [const.col_gid, "splice_type", "n_splice_sites"]
    missing_cols = set(required_cols) - set(perf_df.columns)
    if missing_cols:
        raise ValueError(
            "retrieve_gene_level_performance_features must return columns: "
            + ", ".join(sorted(missing_cols))
        )

    rows_before = perf_df.shape[0]
    perf_df = (
        perf_df[required_cols]
        .groupby([const.col_gid, "splice_type"], as_index=False)
        .agg({"n_splice_sites": "max"})
    )
    removed = rows_before - perf_df.shape[0]
    if removed:
        print_with_indent(
            f"[performance] Warning – removed {removed:,} duplicate rows in performance table",
            indent_level=2,
        )

    print_with_indent(
        f"[performance] Merging performance features (rows={perf_df.shape[0]:,})",
        indent_level=1,
    )

    df_trainset = join_and_remove_duplicates(
        df_trainset, perf_df, on=[const.col_gid, "splice_type"], how="left", verbose=1
    )

    return df_trainset


def incorporate_overlapping_gene_features(
    df_trainset: Union[pd.DataFrame, pl.DataFrame],
    sa=None,
    **kwargs
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Incorporate overlapping gene-specific features.
    
    This is a wrapper around the original incorporate_overlapping_gene_features function
    that maintains compatibility with the existing codebase.
    
    Parameters
    ----------
    df_trainset : Union[pd.DataFrame, pl.DataFrame]
        DataFrame with base features
    sa : SpliceAnalyzer, optional
        SpliceAnalyzer instance, by default None
        
    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        DataFrame with added overlapping gene features
    """
    if sa is None:
        sa = SpliceAnalyzer()

    overlapping_df = sa.retrieve_overlapping_gene_metadata(
        output_format="dataframe", to_pandas=True
    )

    overlapping_df = overlapping_df[["gene_id_1", "num_overlaps"]].rename(
        columns={"gene_id_1": "gene_id"}
    )

    # Ensure one row per gene_id to avoid row inflation during merge
    rows_before = overlapping_df.shape[0]
    overlapping_df = (
        overlapping_df
        .groupby("gene_id", as_index=False)
        .agg({"num_overlaps": "max"})  # keep the maximum count if duplicates exist
    )
    removed = rows_before - overlapping_df.shape[0]
    if removed:
        print_with_indent(
            f"[overlap] Warning – removed {removed:,} duplicate rows in overlap table",
            indent_level=2,
        )

    print_with_indent(
        f"[overlap] Merging overlapping-gene stats (rows={overlapping_df.shape[0]:,})",
        indent_level=1,
    )

    df_trainset = join_and_remove_duplicates(
        df_trainset, overlapping_df, on=[const.col_gid], how="left", verbose=1
    )

    return df_trainset


def incorporate_distance_features(
    df_trainset: Union[pd.DataFrame, pl.DataFrame],
    fa=None,
    **kwargs
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Incorporate splice site distance features.
    
    This is a wrapper around the original incorporate_distance_features function
    that maintains compatibility with the existing codebase.
    
    Parameters
    ----------
    df_trainset : Union[pd.DataFrame, pl.DataFrame]
        DataFrame with base features
    fa : FeatureAnalyzer, optional
        FeatureAnalyzer instance, by default None
        
    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        DataFrame with added distance features
    """
    # if fa is None:
    #     fa = FeatureAnalyzer(**kwargs)

    col_tid = kwargs.get("col_tid", const.col_tid)

    print_with_indent("[distance] Computing splice-site distances …", indent_level=1)

    # ------------------------------------------------------------------
    # DEBUG: report dtypes of critical columns to trace string vs numeric
    # ------------------------------------------------------------------
    if isinstance(df_trainset, pl.DataFrame):
        dbg_cols = [c for c in ("gene_start", "gene_end", "tx_start", "tx_end", "position") if c in df_trainset.columns]
        dtype_map = {c: str(df_trainset[c].dtype) for c in dbg_cols}
        print_with_indent(f"[debug] column dtypes before distance step: {dtype_map}", indent_level=2)
    else:  # pandas
        dbg_cols = [c for c in ("gene_start", "gene_end", "tx_start", "tx_end", "position") if c in df_trainset.columns]
        dtype_map = {c: str(df_trainset[c].dtype) for c in dbg_cols}
        print_with_indent(f"[debug] column dtypes before distance step: {dtype_map}", indent_level=2)

    # Skip computation if transcript_id is absent – insufficient info for transcript boundary distances.
    if const.col_tid not in df_trainset.columns:
        print_with_indent(
            f"[distance] Skipped – '{const.col_tid}' column missing so transcript-boundary distances cannot be computed.",
            indent_level=2,
        )
        return df_trainset

    df_out = compute_distances_with_strand_adjustment(
        df_trainset, match_col="position", col_tid=col_tid
    )

    # ------------------------------------------------------------------
    # Add presence flags so the model can differentiate "large distance" vs
    # "unknown distance" when transcript/gene context is missing.
    # ------------------------------------------------------------------
    if isinstance(df_out, pl.DataFrame):
        has_tx_flag = (
            pl.when(pl.col("tx_start").is_null())
            .then(0)
            .otherwise(1)
            if "tx_start" in df_out.columns
            else pl.lit(0)
        ).alias("has_tx_info")

        has_gene_flag = (
            pl.when(pl.col("gene_start").is_null())
            .then(0)
            .otherwise(1)
            if "gene_start" in df_out.columns
            else pl.lit(0)
        ).alias("has_gene_info")

        df_out = df_out.with_columns([has_tx_flag, has_gene_flag])
    else:
        # pandas branch
        df_out["has_tx_info"] = 1 if "tx_start" in df_out.columns else 0
        df_out.loc[df_out["tx_start"].isna(), "has_tx_info"] = 0 if "tx_start" in df_out.columns else 0

        df_out["has_gene_info"] = 1 if "gene_start" in df_out.columns else 0
        df_out.loc[df_out["gene_start"].isna(), "has_gene_info"] = 0 if "gene_start" in df_out.columns else 0

    # ------------------------------------------------------------------
    # Optional NaN imputation for ML-friendliness
    # ------------------------------------------------------------------
    impute = kwargs.get("impute_missing", True)
    sentinel_offset = kwargs.get("sentinel_offset", 500)

    if impute:
        if isinstance(df_out, pl.DataFrame):
            # Compute max distance ignoring nulls
            max_start = df_out.select(pl.max("distance_to_start")).item()
            max_end = df_out.select(pl.max("distance_to_end")).item()
            non_nulls = [v for v in (max_start, max_end) if v is not None]
            max_dist = max(non_nulls) if non_nulls else 0
            sentinel = max_dist + sentinel_offset

            df_out = df_out.with_columns([
                pl.col("distance_to_start").fill_null(sentinel),
                pl.col("distance_to_end").fill_null(sentinel),
            ])
        else:
            max_dist = max(
                df_out["distance_to_start"].max(skipna=True),
                df_out["distance_to_end"].max(skipna=True),
            )
            sentinel = (0 if pd.isna(max_dist) else max_dist) + sentinel_offset

            df_out["distance_to_start"].fillna(sentinel, inplace=True)
            df_out["distance_to_end"].fillna(sentinel, inplace=True)

    return df_out
