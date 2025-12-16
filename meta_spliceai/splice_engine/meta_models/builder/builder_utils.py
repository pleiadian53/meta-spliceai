"""Utility helpers for the *incremental builder* pipeline.

These functions are intended for **sanity-checks** and quick integrity
verification during development runs.  They are lightweight and should not be
considered a full testing framework – for that, use `pytest` or a similar tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, List
import os

import polars as pl

__all__: list[str] = [
    "verify_gene_selection",
    "fill_missing_gene_type",
    "fill_missing_structural_features",
]


def _load_gene_features_df() -> pl.DataFrame:
    """Return the `gene_features.tsv` table as a Polars DataFrame.

    The file is searched in two likely locations:

    1. Configured via `meta_spliceai.system.config.Config.DATA_DIR`.
    2. Fallback to `~/work/splice-surveyor/data/ensembl/spliceai_analysis/gene_features.tsv`.
    """
    try:
        from meta_spliceai.system.config import Config  # local import to avoid hard dep at module load time

        gf_path = Path(Config.DATA_DIR) / "ensembl" / "spliceai_analysis" / "gene_features.tsv"
    except Exception:
        gf_path = (
            Path(os.environ.get("HOME", ""))
            / "work"
            / "splice-surveyor"
            / "data"
            / "ensembl"
            / "spliceai_analysis"
            / "gene_features.tsv"
        )

    if not gf_path.exists():
        raise FileNotFoundError(
            "Unable to locate gene_features.tsv required for verification checks. "
            "Generate the table or disable verification functions."
        )

    return pl.read_csv(gf_path, separator="\t", schema_overrides={"chrom": pl.Utf8})


from functools import lru_cache
from typing import Dict, List


def fill_missing_gene_type(df: pl.DataFrame, *, sentinel: str = "unknown") -> pl.DataFrame:
    """Back-fill null ``gene_type`` values using *gene_features.tsv*.

    Parameters
    ----------
    df
        Input Polars DataFrame containing at least a ``gene_id`` column and
        optionally a ``gene_type`` column.
    sentinel
        Fallback value when the lookup table is also missing the gene.
    """

    if "gene_type" not in df.columns or df["gene_type"].null_count() == 0:
        return df  # nothing to fix

    # Lazily load & cache the mapping table
    @lru_cache(maxsize=1)
    def _gf() -> pl.DataFrame:
        return _load_gene_features_df().select(["gene_id", "gene_type"]).rename({"gene_type": "gene_type_ref"})

    gf = _gf()
    # Left join to bring in reference types
    df2 = (
        df.join(gf, on="gene_id", how="left")
        .with_columns(
            pl.coalesce([
                pl.col("gene_type"),
                pl.col("gene_type_ref"),
                pl.lit(sentinel),
            ]).alias("gene_type")
        )
        .drop("gene_type_ref")
    )
    return df2


# ---------------------------------------------------------------------------
#  Structural features back-fill
# ---------------------------------------------------------------------------

# Map column → level so we know which ID to join on
# Column → level mapping (target names inside dataset)
_FEATURE_LEVEL: Dict[str, str] = {
    # transcript-level
    "transcript_length": "transcript",
    "tx_start": "transcript",
    "tx_end": "transcript",
    "num_exons": "transcript",
    "avg_exon_length": "transcript",
    "median_exon_length": "transcript",
    "total_exon_length": "transcript",
    "total_intron_length": "transcript",
    "absolute_position": "transcript",
    # gene-level
    "gene_start": "gene",
    "gene_end": "gene",
    "gene_length": "gene",
    "n_splice_sites": "gene",
    "num_overlaps": "gene",
}

# Possible name differences between feature tables and training dataset
# Additional gene-level sources
_GENE_PERF_COLS = {"n_splice_sites"}
_GENE_OVERLAP_COLS = {"num_overlaps"}

_GENE_ALIASES: Dict[str, str] = {
    "gene_start": "start",
    "gene_end": "end",
}

_TRANSCRIPT_ALIASES: Dict[str, str] = {
    "tx_start": "start",
    "tx_end": "end",
}


@lru_cache(maxsize=1)
def _load_performance_features_df() -> pl.DataFrame:
    try:
        from meta_spliceai.system.config import Config
        pf_path = Path(Config.DATA_DIR) / "ensembl" / "spliceai_analysis" / "performance_df_features.tsv"
    except Exception:
        pf_path = (
            Path(os.environ.get("HOME", ""))
            / "work" / "splice-surveyor" / "data" / "ensembl" / "spliceai_analysis" / "performance_df_features.tsv"
        )
    if not pf_path.exists():
        raise FileNotFoundError("performance_df_features.tsv not found – required for n_splice_sites back-fill.")
    return pl.read_csv(pf_path, separator="\t").select(["gene_id", "n_splice_sites"])


@lru_cache(maxsize=1)
def _load_overlap_counts_df() -> pl.DataFrame:
    try:
        from meta_spliceai.system.config import Config
        ov_path = Path(Config.DATA_DIR) / "ensembl" / "overlapping_gene_counts.tsv"
    except Exception:
        ov_path = (
            Path(os.environ.get("HOME", "")) / "work" / "splice-surveyor" / "data" / "ensembl" / "overlapping_gene_counts.tsv"
        )
    if not ov_path.exists():
        raise FileNotFoundError("overlapping_gene_counts.tsv not found – required for num_overlaps back-fill.")
    df = pl.read_csv(ov_path, separator="\t", schema_overrides={"chrom": pl.Utf8})
    if "gene_id" in df.columns and "num_overlaps" in df.columns:
        return df.select(["gene_id", "num_overlaps"])
    # Otherwise derive counts from pairwise table (gene_id_1 / gene_id_2)
    if {"gene_id_1", "gene_id_2"}.issubset(df.columns):
        agg = (
            df.group_by("gene_id_1")
            .agg(pl.col("gene_id_2").n_unique().alias("num_overlaps"))
            .rename({"gene_id_1": "gene_id"})
        )
        return agg
    raise ValueError("overlapping_gene_counts.tsv has unexpected schema – cannot derive num_overlaps")


# ---------------------------------------------------------------------------
# Splice-site table → authoritative n_splice_sites
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_splice_sites_df() -> pl.DataFrame:
    """Load splice_sites.tsv (or enhanced version) containing at least gene_id, position columns.
    
    Automatically uses splice_sites_enhanced.tsv if available via Registry.
    """
    try:
        # Use Registry for systematic path resolution (prefers enhanced version)
        from meta_spliceai.system.genomic_resources import Registry
        registry = Registry()
        ss_path = Path(registry.resolve("splice_sites"))
        if not ss_path or not ss_path.exists():
            raise FileNotFoundError(f"splice_sites file not found via Registry: {ss_path}")
    except Exception as e:
        # Fallback to SystemConfig
        try:
            from meta_spliceai.system.config import Config
            ss_path = Path(Config.DATA_DIR) / "ensembl" / "splice_sites.tsv"
        except Exception:
            # Last resort: environment variable
            from meta_spliceai.system.config import find_project_root
            proj_root = find_project_root()
            ss_path = proj_root / "data" / "ensembl" / "splice_sites.tsv"
        
        if not ss_path.exists():
            raise FileNotFoundError(f"splice_sites.tsv not found at {ss_path} – cannot compute n_splice_sites")
    
    return (
        pl.read_csv(ss_path, separator="\t", schema_overrides={"chrom": pl.Utf8})
        .select(["gene_id", "position"])
        .unique()
    )


def _compute_n_splice_sites_df() -> pl.DataFrame:
    """Return DataFrame with gene_id, n_splice_sites (unique positions)."""
    return (
        _load_splice_sites_df()
        .group_by("gene_id")
        .agg(pl.count().alias("n_splice_sites"))
    )


def update_n_splice_sites(df: pl.DataFrame, *, sentinel: int = -1) -> pl.DataFrame:
    """Override *n_splice_sites* column using authoritative splice_sites.tsv.

    If the gene is absent from the splice-site table the value is set to *sentinel*.
    """
    if "gene_id" not in df.columns:
        return df
    sites_df = _compute_n_splice_sites_df()
    _df = df.clone()
    if "n_splice_sites" in _df.columns:
        _df = _df.drop("n_splice_sites")
    _df = _df.join(sites_df, on="gene_id", how="left")
    return _df.with_columns(pl.col("n_splice_sites").fill_null(sentinel))


@lru_cache(maxsize=1)
def _load_transcript_features_df() -> pl.DataFrame:
    """Load `transcript_features.tsv` with robust path resolution."""
    try:
        from meta_spliceai.system.config import Config

        tf_path = (
            Path(Config.DATA_DIR)
            / "ensembl"
            / "spliceai_analysis"
            / "transcript_features.tsv"
        )
    except Exception:
        tf_path = (
            Path(os.environ.get("HOME", ""))
            / "work"
            / "splice-surveyor"
            / "data"
            / "ensembl"
            / "spliceai_analysis"
            / "transcript_features.tsv"
        )

    if not tf_path.exists():
        raise FileNotFoundError("transcript_features.tsv not found – required for structural back-fill.")

    return pl.read_csv(tf_path, separator="\t", schema_overrides={"chrom": pl.Utf8})


def fill_missing_structural_features(df: pl.DataFrame) -> pl.DataFrame:
    """Back-fill numeric structural columns (gene/transcript level).

    * Gene-level columns are filled via a left join on ``gene_id`` using
      *gene_features.tsv*.
    * Transcript-level columns are filled via a left join on ``transcript_id``
      using *transcript_features.tsv*.
    """

    # Identify which columns actually need filling (exist + have nulls)
    cols_to_fix: List[str] = [c for c in df.columns if c in _FEATURE_LEVEL and df[c].null_count() > 0]
    if not cols_to_fix:
        return df

    out = df

    # ----- gene-level primary (gene_features.tsv) -----
    g_cols = [c for c in cols_to_fix if _FEATURE_LEVEL[c] == "gene"]
    if g_cols and "gene_id" in out.columns:
        gf = _load_gene_features_df()
        # Build list of reference columns actually present in gene_features.tsv
        ref_pairs = []  # (target_col, ref_col)
        for c in g_cols:
            ref = _GENE_ALIASES.get(c, c)
            if ref in gf.columns:
                ref_pairs.append((c, ref))
        if ref_pairs:
            gf_subset_cols = ["gene_id"] + list({ref for _, ref in ref_pairs})
            gf_subset = gf.select(gf_subset_cols)
            out = out.join(gf_subset, on="gene_id", how="left", suffix="_ref")
            for tgt, ref in ref_pairs:
                # If names differ we joined with original ref name, else suffix _ref
                ref_col_name = ref if ref != tgt else f"{tgt}_ref"
                if ref_col_name in out.columns:
                    out = out.with_columns(
                        pl.coalesce([pl.col(tgt), pl.col(ref_col_name)]).alias(tgt)
                    ).drop(ref_col_name)

    # ----- secondary gene-level tables (performance & overlap) -----
    # Always refresh n_splice_sites from authoritative performance table
    if any(c in _GENE_PERF_COLS for c in g_cols):
            try:
                pf = _load_performance_features_df()
                out = out.join(pf, on="gene_id", how="left", suffix="_perf")
                if "n_splice_sites_perf" in out.columns:
                    out = out.with_columns(
                        pl.when(pl.col("n_splice_sites_perf").is_not_null())
                        .then(pl.col("n_splice_sites_perf"))
                        .otherwise(pl.col("n_splice_sites"))
                        .alias("n_splice_sites")
                    ).drop("n_splice_sites_perf")
            except FileNotFoundError as e:
                print(f"[warn] structural back-fill: {e}")
    # Only fetch overlap counts when still missing
    missing_gene_cols = [c for c in g_cols if out[c].null_count() > 0]
    if missing_gene_cols and any(c in _GENE_OVERLAP_COLS for c in missing_gene_cols):
            try:
                ov = _load_overlap_counts_df()
                out = out.join(ov, on="gene_id", how="left", suffix="_ov")
                if "num_overlaps_ov" in out.columns:
                    out = out.with_columns(
                        pl.coalesce([pl.col("num_overlaps"), pl.col("num_overlaps_ov")]).alias("num_overlaps")
                    ).drop("num_overlaps_ov")
            except FileNotFoundError as e:
                print(f"[warn] structural back-fill: {e}")

    # ----- transcript-level -----
    t_cols = [c for c in cols_to_fix if _FEATURE_LEVEL[c] == "transcript"]
    if t_cols and "transcript_id" in out.columns:
        try:
            tf = _load_transcript_features_df()
            ref_pairs_t: List[tuple[str, str]] = []
            for c in t_cols:
                ref = _TRANSCRIPT_ALIASES.get(c, c)
                if ref in tf.columns:
                    ref_pairs_t.append((c, ref))
            if ref_pairs_t:
                tf_subset_cols = ["transcript_id"] + list({ref for _, ref in ref_pairs_t})
                tf_subset = tf.select(tf_subset_cols)
                out = out.join(tf_subset, on="transcript_id", how="left", suffix="_ref")
                for tgt, ref in ref_pairs_t:
                    ref_col_name = ref if ref != tgt else f"{tgt}_ref"
                    if ref_col_name in out.columns:
                        out = out.with_columns(
                            pl.coalesce([pl.col(tgt), pl.col(ref_col_name)]).alias(tgt)
                        ).drop(ref_col_name)
        except FileNotFoundError as e:
            print(f"[warn] fill_missing_structural_features: {e}; skipping transcript back-fill")

    out = out.with_columns(
        pl.when(pl.col("num_overlaps").is_null())
        .then(pl.lit(0))
        .otherwise(pl.col("num_overlaps"))
        .alias("num_overlaps")
    )

    SENTINEL = -1

    out = out.with_columns(
        [
            # transcript-level fall-backs (always applied when still null)
            pl.coalesce([
                pl.col("transcript_length"),
                pl.col("gene_length"),
                pl.lit(SENTINEL),
            ]).alias("transcript_length"),

            pl.coalesce([
                pl.col("tx_start"),
                pl.col("gene_start"),
                pl.lit(SENTINEL),
            ]).alias("tx_start"),

            pl.coalesce([
                pl.col("tx_end"),
                pl.col("gene_end"),
                pl.lit(SENTINEL),
            ]).alias("tx_end"),

            # exon-derived stats: 0 if absent
            *[
                pl.col(c).fill_null(0).alias(c)
                for c in [
                    "num_exons",
                    "avg_exon_length",
                    "median_exon_length",
                    "total_exon_length",
                    "total_intron_length",
                ]
            ],

            # absolute_position: keep NaN (XGBoost OK) or set to -1
            pl.col("absolute_position").fill_null(SENTINEL).alias("absolute_position"),

            # indicator
            (pl.col("transcript_id").is_null() | (pl.col("transcript_id") == "")).cast(pl.Int8).alias("missing_transcript_feats"),
        ]
    )   

    # Final authoritative update of n_splice_sites using splice_sites.tsv
    out = update_n_splice_sites(out)
    return out



def verify_gene_selection(
    data_handler,  # MetaModelDataHandler but we avoid the import type to keep this utility light
    gene_ids: Sequence[str],
    expected_gene_types: Sequence[str] | None,
    *,
    raise_error: bool = True,
    verbose: int = 1,
) -> bool:
    """Check that *all* selected genes fall within the expected gene types.

    Parameters
    ----------
    data_handler
        Instance providing access paths; currently unused but accepted for API
        symmetry (future extensions might rely on it).
    gene_ids
        List of Ensembl gene IDs selected by the builder.
    expected_gene_types
        Allowed gene types (e.g., ["protein_coding"]).  If *None*, verification
        is skipped.
    raise_error
        If *True*, raise ``AssertionError`` when a violation is detected.
    verbose
        Verbosity level: 0 = silent, 1 = summary, 2 = list offending IDs.
    """

    if expected_gene_types is None:
        return True  # nothing to verify

    expected_gene_types = set(expected_gene_types)

    gf_df = _load_gene_features_df().select(["gene_id", "gene_type"])

    # Filter gene_features to only selected ids
    subset_df = gf_df.filter(pl.col("gene_id").is_in(list(gene_ids)))

    # Identify any genes whose type is *not* in expected set
    invalid_df = subset_df.filter(~pl.col("gene_type").is_in(list(expected_gene_types)))
    n_invalid = invalid_df.height

    if n_invalid == 0:
        if verbose:
            print(f"[verify] ✔ Gene selection check passed ({len(gene_ids)} genes, all expected types).")
        return True

    msg = (
        f"[verify] ✗ Gene selection check failed – {n_invalid} / {len(gene_ids)} genes do not match expected gene_types."
    )

    if verbose:
        print(msg)
        if verbose > 1:
            offending_ids: List[str] = invalid_df.select("gene_id").to_series().to_list()
            print("Offending gene IDs:", ", ".join(offending_ids[:20]), "…" if len(offending_ids) > 20 else "")

    if raise_error:
        raise AssertionError(msg)

    return False
