"""Meta-model package constants

Short, consistent identifiers for column names and shared defaults.
Import as::

    import meta_spliceai.splice_engine.meta_models.constants as const

then access, e.g. ``const.col_tid``.
"""
from __future__ import annotations

from typing import Final, Tuple

# ---------------------------------------------------------------------------
# Core column name aliases (concise)
# ---------------------------------------------------------------------------
# Gene / transcript identifiers
col_gid: Final[str] = "gene_id"
col_tid: Final[str] = "transcript_id"

# ---------------------------------------------------------------------------
# Package-wide defaults that many modules rely on
# ---------------------------------------------------------------------------
DEFAULT_KMER_SIZES: Final[Tuple[int, ...]] = (6,)

# Minimum set of columns for efficient data loading during inference workflow
# 
# NOTE: This is ONLY for workflow efficiency - loading core columns from analysis_sequences files.
# The actual feature matrix for meta-model prediction will contain 120+ features including:
# - Raw scores (donor_score, acceptor_score, neither_score)
# - Probability-derived features (relative_donor_probability, splice_probability, entropy, etc.)
# - Context scores (context_score_p1, context_neighbor_mean, context_asymmetry, etc.)
# - Genomic features (gene_length, transcript_length, n_splice_sites, etc.)
# - Sequence motifs (k-mer counts, sequence-derived features)
# - Advanced features (signal_strength, peak_height_ratio, etc.)
#
# The StandardizedFeaturizer generates the complete feature matrix from these core columns
# plus additional genomic data sources.
EXPECTED_MIN_COLUMNS: Final[Tuple[str, ...]] = (
    col_gid,
    col_tid,
    "position",
    "splice_type",
    "pred_type",
    "donor_score",
    "acceptor_score", 
    "neither_score",
)

__all__ = [
    "col_gid",
    "col_tid",
    "DEFAULT_KMER_SIZES",
    "EXPECTED_MIN_COLUMNS",
]
