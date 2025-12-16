"""Meta-model feature preparation utilities.

This module provides a *single* place to:
1.  Decide which columns are valid input features and which must be removed to
    avoid information leakage or over-fitting.
2.  Generate the machine-learning label vector (y) from the canonical
    `splice_type` column produced by the builder pipeline.
3.  Impute missing values so that downstream libraries such as XGBoost are not
    tripped up by nulls.

The functions operate on **Polars** DataFrames, matching the rest of the
builder stack, but can return NumPy / pandas objects for compatibility with ML
frameworks.

Example
-------
>>> import polars as pl
>>> from meta_spliceai.splice_engine.meta_models.builder import preprocessing as prep
>>> df = pl.read_parquet("training_dataset.parquet")
>>> X, y = prep.prepare_training_data(df)

Both ``X`` and ``y`` are pandas objects ready for e.g. ``xgboost.DMatrix``.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl

# Import centralized feature schema
from meta_spliceai.splice_engine.meta_models.builder.feature_schema import (
    encode_categorical_features,
    CATEGORICAL_FEATURES,
    is_kmer_feature,
    validate_feature_types
)

# ---------------------------------------------------------------------------
#  Column taxonomy
# ---------------------------------------------------------------------------

# Columns whose presence would leak ground-truth information
LEAKAGE_COLUMNS: List[str] = [
    "splice_type",         # the *target* label itself
    "pred_type",           # directly derived from splice_type
    "true_position",       # exact coordinate of the real splice site
    "predicted_position",  # tightly correlated with label
    # "absolute_position",   # tightly correlated with label
]

# Columns that mostly encode *where* rather than *what*
METADATA_COLUMNS: List[str] = [
    "position",           # Absolute position - high cardinality, poor generalization
    "absolute_position",
    "window_start",
    "window_end",
    "gene_id",
    "transcript_id",
    "gene_type",
    "strand",
    "transcript_count",
    "gene_name",          # Gene name - metadata not used by model
]

# ALTERNATIVE: If you wanted positional features, these would be better:
# RELATIVE_POSITION_FEATURES = [
#     "relative_position",      # position / gene_length (0.0 to 1.0)
#     "distance_to_start",      # distance from gene start (log-transformed)
#     "distance_to_end",        # distance from gene end (log-transformed)  
#     "is_first_exon",          # boolean indicator
#     "is_last_exon",           # boolean indicator
# ]
# These would be lower cardinality and more biologically meaningful

# Raw sequence string (only useful for DL models)
SEQUENCE_COLUMNS: List[str] = ["sequence"]

# Potentially redundant statistical summaries
REDUNDANT_COLUMNS: List[str] = ["score"]

# ---------------------------------------------------------------------------
#  Label utilities
# ---------------------------------------------------------------------------

from meta_spliceai.splice_engine.meta_models.training.label_utils import LABEL_MAP_STR

# Canonical mapping (neither=0, donor=1, acceptor=2) plus a legacy string "0"
# alias for non-splice sites that some older artefacts use.
_DEFAULT_LABEL_MAP: Dict[str, int] = {**LABEL_MAP_STR, "0": 0}


def make_label_series(df: pl.DataFrame, *, label_col: str = "splice_type", label_map: Dict[str, int] | None = None) -> pd.Series:
    """Return ``y`` as a *pandas* Series suitable for scikit-learn / XGBoost.

    Parameters
    ----------
    df
        Source Polars DataFrame.
    label_col
        Name of the categorical column holding splice-type labels.
    label_map
        Optional explicit mapping from categorical label → numeric class.
        If omitted, ``{"neither": 0, "acceptor": 1, "donor": 2}`` is used.
    """
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in DataFrame.")

    # Apply fail-safe logic: anything that is not "donor" or "acceptor" should be "neither"
    # This handles None/null values and any legacy encodings (including "0" or 0)
    normalized_df = df.with_columns(
        pl.when(pl.col(label_col).is_in(["donor", "acceptor"]))
        .then(pl.col(label_col))
        .otherwise(pl.lit("neither"))
        .alias(label_col)
    )

    mapping = _DEFAULT_LABEL_MAP if label_map is None else label_map
    
    # Polars → pandas conversion happens here – keeps rest of pipeline Polars
    # Direct numpy conversion to avoid pandas/pyarrow compatibility issues
    import pandas as pd
    labels_list = normalized_df[label_col].to_list()
    
    # Convert to numpy array first, then to pandas Series
    # This avoids pandas/pyarrow compatibility issues with lists
    labels_array = np.array(labels_list, dtype=str)
    y = pd.Series(labels_array).map(mapping).astype(np.int8)
    return y

# ---------------------------------------------------------------------------
#  Column filtering
# ---------------------------------------------------------------------------

def drop_unwanted_columns(
    df: pl.DataFrame,
    *,
    drop_leakage: bool = True,
    drop_metadata: bool = True,
    drop_sequence: bool = True,
    drop_redundant: bool = True,
    extra_drop: Iterable[str] | None = None,
    preserve_transcript_columns: bool = False,
    encode_chrom: bool = False,
    verbose: int = 0,
) -> pl.DataFrame:
    """Return a new DataFrame with problematic columns removed."""
    to_drop: List[str] = []
    if drop_leakage:
        to_drop.extend(LEAKAGE_COLUMNS)
    if drop_metadata:
        if preserve_transcript_columns or encode_chrom:
            # Keep position and chrom columns needed for transcript mapping or as features
            excluded_cols = []
            if preserve_transcript_columns:
                excluded_cols.append('position')
                excluded_cols.append('chrom')
                excluded_cols.append('transcript_id')  # CRITICAL: Also preserve transcript_id
            elif encode_chrom:  # Only add 'chrom' if it wasn't already added
                excluded_cols.append('chrom')  # This ensures 'chrom' is excluded even if not preserving transcripts
            to_drop.extend([col for col in METADATA_COLUMNS if col not in excluded_cols])
        else:
            to_drop.extend(METADATA_COLUMNS)
    if drop_sequence:
        to_drop.extend(SEQUENCE_COLUMNS)
    if drop_redundant:
        to_drop.extend(REDUNDANT_COLUMNS)
    if extra_drop:
        to_drop.extend(extra_drop)

    present = [c for c in to_drop if c in df.columns]
    return df.drop(present)

# ---------------------------------------------------------------------------
#  Imputation
# ---------------------------------------------------------------------------

def _compute_fill_value(series: pl.Series) -> object:
    """Return a scalar fill value appropriate for *series* dtype."""
    if series.dtype.is_numeric():
        med = series.median()
        if med is None or (isinstance(med, float) and np.isnan(med)):
            return 0
        return med
    # Fallback for categorical / string – use mode or placeholder
    mode_s = series.mode()
    if len(mode_s) > 0:
        return mode_s[0]
    return "UNKNOWN"


def impute_nulls(df: pl.DataFrame) -> pl.DataFrame:
    """Fill NA/Null values *column-wise* using median (numeric) or mode."""
    new_df = df
    for name in df.columns:
        series = df[name]
        if series.null_count() == 0 and not (
            series.dtype == pl.Utf8 and (series == "").sum() > 0
        ):
            # No nulls (and, for strings, no blank strings) -> skip
            continue

        if series.dtype == pl.Utf8 or series.dtype == pl.Categorical:
            # Treat both null and blank strings as missing → "unknown"
            new_df = new_df.with_columns(
                pl.when(pl.col(name).is_null() | (pl.col(name) == ""))
                .then(pl.lit("unknown"))
                .otherwise(pl.col(name))
                .alias(name)
            )
            continue

        # Numeric or other dtypes → use median/mode fallback
        fill_val = _compute_fill_value(series)
        try:
            new_df = new_df.with_columns(pl.col(name).fill_null(fill_val))
        except ValueError as e:
            # Log and skip column on unexpected incompatibility
            print(
                f"[warn] impute_nulls: could not fill column '{name}' (dtype={series.dtype}) -> {e}"
            )
    return new_df

# ---------------------------------------------------------------------------
#  End-to-end helper
# ---------------------------------------------------------------------------

def prepare_training_data(
    df: pl.DataFrame,
    *,
    label_col: str = "splice_type",
    label_map: Dict[str, int] | None = None,
    drop_leakage: bool = True,
    drop_metadata: bool = True,
    drop_sequence: bool = True,
    drop_redundant: bool = True,
    extra_drop: Sequence[str] | None = None,
    impute: bool = True,
    return_type: str = "pandas",  # 'pandas' | 'polars'
    verbose: int = 0,
    preserve_transcript_columns: bool = False,
    encode_chrom: bool = False,
) -> Tuple[pd.DataFrame | pl.DataFrame, pd.Series]:
    """Full pipeline: filter → impute → split features/labels.

    Returns
    -------
    X
        Feature matrix (pandas or Polars depending on *return_type*).
    y
        Label vector as a pandas Series.
    """
    # 1. Build label first before we drop the column (to keep API symmetric)
    y = make_label_series(df, label_col=label_col, label_map=label_map)

    # 2. Remove the *label* column to avoid leakage, then drop other unwanted cols
    df = drop_unwanted_columns(
        df,
        drop_leakage=drop_leakage,
        drop_metadata=drop_metadata,
        drop_sequence=drop_sequence,
        drop_redundant=drop_redundant,
        extra_drop=extra_drop,
        preserve_transcript_columns=preserve_transcript_columns,
        encode_chrom=encode_chrom,
        verbose=verbose,
    )
    X_df = df

    if verbose:
        import re
        kmer_re = re.compile(r"^\d+mer_")
        nkmer = len([c for c in X_df.columns if kmer_re.match(c)])
        nfeat = len(X_df.columns)
        print(f"Preprocessing: {nfeat} features ({nfeat - nkmer} non-kmer, {nkmer} k-mer)")
        
        # Show non-kmer columns
        non_kmers = [c for c in X_df.columns if not kmer_re.match(c)]
        if non_kmers:
            print("Non-kmer columns:")
            print(", ".join(sorted(non_kmers)))
    
    # Encode categorical features using centralized schema
    if encode_chrom and 'chrom' in X_df.columns:
        if X_df.select('chrom').dtypes[0] == pl.Utf8:
            if verbose:
                print(f"\nApplying categorical encoding from centralized schema...")
            
            X_df = encode_categorical_features(
                X_df,
                features_to_encode=['chrom'],
                verbose=verbose
            )
    
    # Safeguard: Verify no metadata columns are in feature set (unless specifically preserved or encoded)
    preserved_columns = []
    if preserve_transcript_columns:
        preserved_columns.extend(['position', 'chrom', 'transcript_id'])  # Include transcript_id
    if encode_chrom:  # Always check independently 
        preserved_columns.append('chrom')
        
    training_metadata_cols = [col for col in METADATA_COLUMNS if col in X_df.columns and col not in preserved_columns]
    if training_metadata_cols:
        raise ValueError(f"Metadata columns found in feature set: {training_metadata_cols}. This is likely an error.")
        
    # Ensure preserved columns are marked clearly
    preserved_metadata_cols = [col for col in preserved_columns if col in X_df.columns]
    if preserve_transcript_columns and preserved_metadata_cols and verbose:
        print(f"\nPreserved metadata columns (for transcript mapping only, NOT used for training): {', '.join(preserved_metadata_cols)}")
    elif encode_chrom and 'chrom' in X_df.columns and verbose:
        print(f"\nUsing 'chrom' as a training feature with numeric encoding")

    # 3. Domain-specific fix: ensure gene_type is populated
    try:
        from .builder_utils import fill_missing_gene_type  # local import
        X_df = fill_missing_gene_type(X_df)
        from .builder_utils import fill_missing_structural_features
        X_df = fill_missing_structural_features(X_df)
    except Exception as exc:  # non-fatal – continue without enrichment
        print(f"[warn] fill_missing_gene_type / structural failed: {exc}")

    # 4. Impute missing values
    if impute:
        X_df = impute_nulls(X_df)

    # 4. Convert if desired
    if return_type == "pandas":
        # MEMORY OPTIMIZATION: Convert via numpy to avoid pandas/pyarrow compatibility issues
        # and reduce memory footprint. This is more efficient than creating a pandas DataFrame
        # with object columns, especially for large datasets.
        import pandas as pd
        
        # Convert Polars DataFrame to numpy arrays (memory-efficient)
        # This creates a contiguous numpy array instead of pandas' block manager
        X_numpy = X_df.to_numpy()
        
        # Create pandas DataFrame from numpy (more efficient than dict of lists)
        X = pd.DataFrame(X_numpy, columns=X_df.columns)
        
    elif return_type == "polars":
        X = X_df
    elif return_type == "numpy":
        # NEW: Direct numpy conversion for maximum memory efficiency
        # Returns (numpy_array, column_names) instead of DataFrame
        X = X_df.to_numpy()
    else:
        raise ValueError("return_type must be 'pandas', 'polars', or 'numpy'")

    return X, y

# ---------------------------------------------------------------------------
#  Convenience: expose default drop list for documentation/testing
# ---------------------------------------------------------------------------

DEFAULT_DROP_COLUMNS: List[str] = (
    LEAKAGE_COLUMNS + METADATA_COLUMNS + SEQUENCE_COLUMNS + REDUNDANT_COLUMNS
)
