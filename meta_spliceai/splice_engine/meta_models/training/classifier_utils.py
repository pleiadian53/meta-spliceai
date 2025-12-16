#!/usr/bin/env python3
"""Utility helpers for post-training diagnostics of the meta-model.

Features implemented
--------------------
1. Richer evaluation metrics (accuracy, macro-F1, ROC-AUC OVR).
2. Label-leakage probe wrapper (re-exports leakage_probe.probe_leakage).
3. Gene-level score deltas via gene_score_delta_multiclass.compute_…
4. SHAP-based feature importance report.

All helpers auto-detect whether the model is saved as a *pickle* (sklearn
wrapper) or native XGBoost JSON and adjust accordingly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Any, List
import json
import os
import warnings

import numpy as np
import polars as pl
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef
from sklearn.base import ClassifierMixin

from . import explainers  # SHAP/permutation utilities
from . import gene_score_delta_multiclass as _gene_delta
from . import leakage_probe as _leak
from .label_utils import LABEL_MAP_STR, encode_labels as _encode_labels, swap_0_2


def select_available_columns(
    lazy_frame: pl.LazyFrame, 
    required_columns: List[str], 
    context_name: str = "Data Processing",
    verbose: bool = True
) -> tuple[pl.LazyFrame, List[str], List[str]]:
    """
    Robustly select only available columns from a Polars LazyFrame.
    
    This function handles the common pattern of attempting to select columns that may not exist
    in the dataset, which occurs frequently when feature manifests include more columns than
    are present in the actual parquet files (e.g., k-mers with ambiguous nucleotides).
    
    Parameters
    ----------
    lazy_frame : pl.LazyFrame
        The Polars LazyFrame to select columns from
    required_columns : List[str]
        List of column names that we want to select
    context_name : str, default="Data Processing"
        Context name for warning messages (e.g., "Feature Importance Analysis")
    verbose : bool, default=True
        Whether to print warning messages for missing columns
        
    Returns
    -------
    tuple[pl.LazyFrame, List[str], List[str]]
        - Selected LazyFrame with only available columns
        - List of missing column names
        - List of existing column names that were selected
        
    Examples
    --------
    >>> # Basic usage
    >>> lf_selected, missing, existing = select_available_columns(
    ...     lf, ['donor_score', 'acceptor_score', '3mer_GTN'], 
    ...     context_name="Feature Analysis"
    ... )
    
    >>> # Handle missing columns in model preprocessing
    >>> lf_selected, missing, existing = select_available_columns(
    ...     lf, feature_names + ['splice_type'], 
    ...     context_name="Model Training"
    ... )
    """
    available_cols = lazy_frame.columns
    missing_cols = [col for col in required_columns if col not in available_cols]
    existing_cols = [col for col in required_columns if col in available_cols]
    
    if missing_cols and verbose:
        print(f"[{context_name}] Warning: Missing columns {missing_cols}, proceeding with available columns")
    
    # Select only existing columns to avoid ColumnNotFoundError
    lf_selected = lazy_frame.select(existing_cols)
    
    return lf_selected, missing_cols, existing_cols


def add_missing_features_with_zeros(
    dataframe: pd.DataFrame, 
    required_features: List[str],
    context_name: str = "Data Processing",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Add missing feature columns to a pandas DataFrame with zero values.
    
    This function handles the common pattern of ensuring all required features are present
    in a DataFrame before model inference, adding missing features with zeros (which is
    statistically valid for k-mer features where zero indicates no occurrence).
    
    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame to add missing columns to
    required_features : List[str]  
        List of feature names that must be present
    context_name : str, default="Data Processing"
        Context name for warning messages (e.g., "Model Inference")
    verbose : bool, default=True
        Whether to print warning messages for missing features
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all required features present (missing ones filled with zeros)
        
    Examples
    --------
    >>> # Basic usage
    >>> df_complete = add_missing_features_with_zeros(
    ...     df, feature_names, context_name="Model Prediction"
    ... )
    
    >>> # Silent operation
    >>> df_complete = add_missing_features_with_zeros(
    ...     df, feature_names, verbose=False
    ... )
    """
    df = dataframe.copy()
    
    for feature in required_features:
        if feature not in df.columns:
            if verbose:
                print(f"[{context_name}] Warning: Adding missing feature '{feature}' with zeros")
            df[feature] = 0.0
    
    return df

# Note: meta_evaluation_utils imports are moved inside functions to avoid circular imports

# ---------------------------------------------------------------------------
# Backward-compatibility alias – remove after full refactor
# ---------------------------------------------------------------------------
_LABEL_MAP_STR = LABEL_MAP_STR

# ---------------------------------------------------------------------------
#  Model wrapper for independent-sigmoid ensembles
# ---------------------------------------------------------------------------
class SigmoidEnsemble:
    """Lightweight wrapper exposing predict_proba for 3-class sigmoid ensemble.

    Parameters
    ----------
    models : list of length 3
        Binary classifiers ordered by canonical class index
        (0=neither, 1=donor, 2=acceptor).  Each must implement
        ``predict_proba`` returning [:,1] as *P(class = 1)*.
    feature_names : list[str]
        Feature column names corresponding to training matrix order.
    """
    def __init__(self, models: list, feature_names: list[str]):
        if len(models) != 3:
            raise ValueError("SigmoidEnsemble expects exactly 3 binary models")
        self.models = models
        self.feature_names = feature_names

    # ------------------------------------------------------------------
    # API compatible with scikit-learn estimator used elsewhere
    # ------------------------------------------------------------------
    def predict_proba(self, X):  # noqa: D401 – simple def
        """Return stacked probabilities shape (n,3)."""
        # import numpy as np
        parts = [m.predict_proba(X)[:, 1] for m in self.models]
        return np.column_stack(parts)

    def __getstate__(self):  # for pickling
        return {
            "models": self.models,
            "feature_names": self.feature_names,
        }

    def __setstate__(self, state):
        self.models = state["models"]
        self.feature_names = state["feature_names"]

    # Convenience to mimic sklearn interface
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class CalibratedSigmoidEnsemble(SigmoidEnsemble):
    """SigmoidEnsemble with an **optional calibration layer** for splice-site prob.

    Parameters
    ----------
    models : list[sklearn.base.BaseEstimator]
        Three binary XGBoost models as in SigmoidEnsemble.
    feature_names : list[str]
        Feature columns.
    calibrator : fitted sklearn regressor supporting ``predict_proba`` or
        callable that maps ``s = p_d + p_a`` onto calibrated probability.

    Memo
    ----
    1. Example calibrator: 
       IsotonicRegression(out_of_bounds='clip').fit(s_train, y_bin)
    """

    def __init__(self, models: list, feature_names: list[str], calibrator):
        super().__init__(models, feature_names)
        self.calibrator = calibrator

    def _calibrate(self, s: np.ndarray) -> np.ndarray:  # noqa: D401
        # Most classifiers expose predict_proba; regressors (e.g. IsotonicRegression)
        # expose predict(); custom lambdas are callable.
        if hasattr(self.calibrator, "predict_proba"):
            return self.calibrator.predict_proba(s.reshape(-1, 1))[:, 1]
        elif hasattr(self.calibrator, "predict"):
            return self.calibrator.predict(s.reshape(-1, 1))
        elif callable(self.calibrator):
            return self.calibrator(s)
        else:
            raise TypeError(f"Unsupported calibrator type: {type(self.calibrator)}")

    # ------------------------------------------------------------------
    # Pickling helpers – include calibrator so that loaded model works.
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = super().__getstate__()
        state["calibrator"] = self.calibrator
        return state

    def __setstate__(self, state):
        self.calibrator = state.pop("calibrator", None)
        super().__setstate__(state)

    def predict_proba(self, X):
        # import numpy as np
        proba = super().predict_proba(X)  # shape (n,3)
        s = proba[:, 1] + proba[:, 2]
        if self.calibrator is None:
            # fallback: no calibration available
            return proba
        s_cal = self._calibrate(s)
        # rescale donor/acceptor proportionally; avoid divide-by-zero
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(s > 0, s_cal / s, 0.0)
        proba[:, 1] *= scale
        proba[:, 2] *= scale
        proba[:, 0] = 1.0 - s_cal  # neither prob
        return proba


class PerClassCalibratedSigmoidEnsemble:
    """Wrapper for three binary models with a separate calibrator for each class.

    This applies individual calibration to each class probability, then
    normalizes the results to ensure they sum to 1.0, producing well-distributed
    probabilities suitable for multi-class classification and threshold-based 
    decisions.
    """
    def __init__(
        self,
        models: List[ClassifierMixin],
        feature_names: List[str],
        calibrators: List[Any],  # One calibrator per class
        threshold_donor: float = None,
        threshold_acceptor: float = None,
        threshold_neither: float = None,
        optimal_threshold: float = None,
    ):
        self.models = models
        self.feature_names = feature_names
        self.calibrators = calibrators
        self.threshold_donor = threshold_donor
        self.threshold_acceptor = threshold_acceptor
        self.threshold_neither = threshold_neither
        self.optimal_threshold = optimal_threshold
        
    def get_base_models(self):
        """Return the underlying base models for SHAP analysis.
        
        This exposes the three binary XGBoost models to the SHAP TreeExplainer.
        
        Returns
        -------
        List of the three underlying binary models (neither, donor, acceptor).
        """
        return self.models

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get class probabilities with per-class calibration.

        Returns
        -------
        array of shape (n_samples, 3)
            Probabilities for each sample, with columns
            [neither, donor, acceptor]. Values are calibrated and normalized
            to sum to 1.0 per row.
        """
        # Get raw probabilities from the three binary classifiers
        raw_proba = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])  # (n, 3)

        # Apply separate calibration to each class
        calibrated_proba = np.zeros_like(raw_proba)
        for i, calibrator in enumerate(self.calibrators):
            if calibrator is not None:
                # For Platt scaling (LogisticRegression), we need to reshape
                if hasattr(calibrator, 'predict_proba'):
                    # Using predict_proba for models like LogisticRegression (Platt scaling)
                    calibrated_proba[:, i] = calibrator.predict_proba(
                        raw_proba[:, i].reshape(-1, 1)
                    )[:, 1]  # Use the positive class probability
                else:
                    # For isotonic regression which returns direct values
                    calibrated_proba[:, i] = calibrator.predict(raw_proba[:, i])
            else:
                # No calibrator for this class, keep raw probability
                calibrated_proba[:, i] = raw_proba[:, i]
        
        # Apply temperature scaling to soften extreme probabilities
        # Temperature > 1 makes probabilities less extreme
        temperature = 1.5
        tempered_proba = np.power(calibrated_proba, 1/temperature)

        # Normalize to ensure probabilities sum to 1.0 per sample
        row_sums = tempered_proba.sum(axis=1).reshape(-1, 1)
        # Handle zero-sum edge case
        row_sums[row_sums == 0] = 1.0
        
        return tempered_proba / row_sums


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _preprocess_features_for_model(df, feature_names):
    """Preprocess features to ensure they're compatible with ML models.
    
    This function handles different types of features appropriately:
    - K-mer features: NaN values replaced with 0 (no occurrence)
    - Categorical features: Converted to numeric indices preserving ordinality if possible
    - Already numeric features: Left unchanged
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features to preprocess
    feature_names : list[str]
        List of feature column names to process
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all features properly converted to numeric types
        suitable for model input
    
    Notes
    -----
    - For categorical features with inherent ordering (ordinal), you may want to
      provide a custom mapping dictionary instead of relying on automatic conversion
    - Features already in numeric form are left untouched
    - Missing values in non-k-mer features are replaced with -1
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Process each feature as needed
    for col in feature_names:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataframe, adding with zeros")
            df[col] = 0
            continue
            
        # Skip already numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Handle different feature types
        if _is_kmer(col):
            # For k-mer features, missing values should be 0 (no occurrence)
            print(f"Converting non-numeric k-mer column '{col}' to float with 0 for NaN values")
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        elif df[col].dtype.name == 'category':
            # Preserve existing categorical encodings if possible
            if all(pd.api.types.is_numeric_dtype(cat) for cat in df[col].cat.categories):
                # Categorical with numeric categories - preserve as is
                df[col] = df[col].cat.codes.astype(float)
            else:
                # Categorical with non-numeric categories - create mapping
                print(f"Converting categorical column '{col}' to numeric codes")
                df[col] = df[col].cat.codes.astype(float)
        elif df[col].dtype == object:
            # Handle string/object columns by mapping to integers
            print(f"Converting non-numeric column '{col}' to float with numeric encoding")
            unique_values = df[col].dropna().unique()
            value_map = {val: idx for idx, val in enumerate(unique_values)}
            df[col] = df[col].map(value_map).fillna(-1)
        else:
            # Other non-numeric dtypes - try standard conversion
            print(f"Converting column '{col}' with dtype {df[col].dtype} to numeric")
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)
    
    return df

def _is_kmer(feature_name: str) -> bool:
    """Check if a feature name corresponds to a k-mer pattern.
    
    Parameters
    ----------
    feature_name : str
        The name of the feature to check
        
    Returns
    -------
    bool
        True if the feature name matches k-mer pattern, False otherwise
        
    Examples
    --------
    >>> _is_kmer("6mer_GGATCN")
    True
    >>> _is_kmer("4mer_TGGA")
    True
    >>> _is_kmer("donor_score")
    False
    """
    # Use regex to identify standard k-mer patterns in MetaSpliceAI
    # This pattern matches digits followed by 'mer' (e.g., 3mer, 4mer, 6mer)
    import re
    kmer_pattern = re.compile(r'\d+mer')
    return bool(kmer_pattern.search(feature_name))


def _load_model(run_dir: Path):
    """Load model from run_dir - convenience function.
    
    Returns the loaded model object directly.
    """
    import pickle
    import xgboost as xgb

    pkl_path = Path(run_dir) / "model_multiclass.pkl"
    json_path = Path(run_dir) / "model_multiclass.json"

    if pkl_path.exists():
        with open(pkl_path, "rb") as fh:
            model = pickle.load(fh)
        return model
    elif json_path.exists():
        booster = xgb.Booster()
        booster.load_model(str(json_path))
        return booster
    else:
        raise FileNotFoundError("No model_multiclass.[pkl|json] found in run_dir")


def _load_model_generic(run_dir: Path):
    """Return (predict_proba_fn, feature_names).

    *predict_proba_fn* must accept ``np.ndarray`` X and return ``np.ndarray``
    probabilities *shape (n,3)*.
    """
    import pickle
    import xgboost as xgb

    pkl_path = run_dir / "model_multiclass.pkl"
    json_path = run_dir / "model_multiclass.json"

    if pkl_path.exists():
        with open(pkl_path, "rb") as fh:
            model = pickle.load(fh)
        predict_fn = model.predict_proba
    elif json_path.exists():
        booster = xgb.Booster()
        booster.load_model(str(json_path))
        # We'll attach feature names after we load the manifest below.
        predict_fn = None  # type: ignore

    else:
        raise FileNotFoundError("No model_multiclass.[pkl|json] found in run_dir")

    # feature manifest (CSV for in-RAM, JSON for ext-mem)
    csv_path = run_dir / "feature_manifest.csv"
    json_path_manifest = run_dir / "train.features.json"
    if csv_path.exists():
        feature_names = pd.read_csv(csv_path)["feature"].tolist()
    elif json_path_manifest.exists():
        feature_names = json.loads(json_path_manifest.read_text())["feature_names"]
    else:
        raise FileNotFoundError("feature manifest not found in run_dir")

    # If model loaded from JSON, bind feature names now
    import xgboost as xgb
    if predict_fn is None:
        def predict_fn(X: np.ndarray) -> np.ndarray:  # type: ignore[override]
            dmat = xgb.DMatrix(X, feature_names=feature_names, missing=np.nan)
            return booster.predict(dmat)

    return predict_fn, feature_names


def _legacy_encode_labels(arr: Sequence[Any]) -> np.ndarray:
    """Convert raw *splice_type* values to canonical integers 0/1/2.

    Canonical mapping: 0 = neither (non-splice), 1 = donor, 2 = acceptor.

    The legacy on-disk datasets occasionally use *numeric* labels with the
    *alternate* mapping 0 = donor, 1 = acceptor, 2 = neither.  When we detect
    that pattern we automatically remap to the canonical order and emit a
    RuntimeWarning so that callers are aware of the silent fix.
    """
    import warnings

    arr = np.asarray(arr)

    # ------------------------------------------------------------------
    # Case 1 – non-integer dtype.  Could be strings *or* object-dtype numbers.
    # ------------------------------------------------------------------
    if arr.dtype.kind not in ("i", "u"):
        # Mixed / string/object labels.
        conv: list[int] = []
        unknown: set[str] = set()
        for x in arr:
            # Direct ints
            if isinstance(x, (int, np.integer)):
                conv.append(int(x))
                continue
            # Numeric strings e.g. "0", "1"
            try:
                xi = int(x)
                conv.append(xi)
                continue
            except (ValueError, TypeError):
                pass
            # Canonical string labels (case-insensitive)
            val = _LABEL_MAP_STR.get(str(x).lower())  # type: ignore[arg-type]
            if val is None:
                unknown.add(str(x))
            else:
                conv.append(val)
        if unknown:
            raise ValueError(
                f"Unrecognised splice_type labels: {', '.join(sorted(unknown))}."
            )
        arr = np.asarray(conv, dtype=int)

    # ------------------------------------------------------------------
    # Case 2 – numeric labels 0/1/2.  Validate & remap if necessary.
    # ------------------------------------------------------------------
    uniq = np.unique(arr)
    if not set(uniq).issubset({0, 1, 2}):
        raise ValueError(
            f"Unexpected numeric splice_type labels {uniq}. Expected subset of {{0,1,2}}."
        )

    # Heuristic: in canonical encoding *neither* is label 0 and therefore
    #    counts[0] should be the largest class (or tied).  If instead label 2
    #    dominates we assume the legacy mapping (0=donor, 2=neither).
    counts = np.bincount(arr.astype(int), minlength=3)
    if counts[0] >= counts[1] and counts[0] >= counts[2]:
        # already canonical (0 = neither)
        return arr.astype(int)

    # Legacy pattern detected → remap 0↔2
    remapped = arr.copy()
    remapped[arr == 0] = 2
    remapped[arr == 2] = 0
    warnings.warn(
        "Detected legacy numeric encoding (0=donor,2=neither); auto-remapped to canonical order.",
        RuntimeWarning,
    )
    return remapped.astype(int)


# -----------------------------------------------------------------------------
# Sampling helper avoiding LazyFrame.sample
# -----------------------------------------------------------------------------

def _lazyframe_sample_gene_aware(lf: pl.LazyFrame, n: int, seed: int = 42) -> pl.LazyFrame:
    """Return ~n rows from *lf* by sampling complete genes.
    
    This ensures gene structure is preserved by sampling at the gene level,
    not at the row level.
    """
    try:
        # Check if gene_id column exists
        schema = lf.collect_schema()
        if "gene_id" not in schema.names():
            # Fallback to row-based sampling if no gene_id
            print("[Warning] No gene_id column found, falling back to row-based sampling")
            return _lazyframe_sample(lf, n, seed)
        
        # Get unique genes and their row counts
        gene_counts = (
            lf.group_by("gene_id")
            .agg(pl.count().alias("n_rows"))
            .collect(streaming=True)
        )
        
        # Sample genes until we reach approximately n rows
        import numpy as np
        rng = np.random.default_rng(seed)
        
        gene_ids = gene_counts["gene_id"].to_list()
        row_counts = gene_counts["n_rows"].to_list()
        
        selected_genes = []
        total_rows = 0
        
        # Shuffle gene order for random sampling
        indices = rng.permutation(len(gene_ids))
        
        for idx in indices:
            gene_id = gene_ids[idx]
            n_rows = row_counts[idx]
            
            if total_rows + n_rows > n * 1.1:  # Allow 10% overage
                break
                
            selected_genes.append(gene_id)
            total_rows += n_rows
            
            if total_rows >= n:
                break
        
        print(f"[Gene-aware sampling] Selected {len(selected_genes)} genes with {total_rows} rows (target: {n})")
        
        # Filter to selected genes
        return lf.filter(pl.col("gene_id").is_in(selected_genes))
        
    except Exception as e:
        print(f"[Warning] Gene-aware sampling failed: {e}, falling back to row-based sampling")
        return _lazyframe_sample(lf, n, seed)

def _lazyframe_sample(lf: pl.LazyFrame, n: int, seed: int = 42) -> pl.LazyFrame:
    """Return ~n randomly sampled rows from *lf* without using LazyFrame.sample.

    Uses row hash to create a reproducible pseudo-random subset.
    """
    try:
        # total row count (cheap; only metadata read)
        total_rows = lf.select(pl.count()).collect(streaming=True).to_series(0).item()
        
        # Handle edge cases
        if total_rows == 0:
            # Empty dataframe - return as is
            return lf
        if total_rows <= n:
            # Already small enough - return as is
            return lf
        if n <= 0:
            # Invalid sample size - return empty
            return lf.limit(0)
            
        # Calculate step size with better bounds checking
        step = max(1, total_rows // n)
        
        # Ensure step doesn't exceed total_rows
        if step >= total_rows:
            step = max(1, total_rows - 1)
            
        rng = np.random.default_rng(seed)
        # choose random offset to avoid bias, ensure it's within bounds
        max_offset = min(step - 1, total_rows - 1)
        offset = rng.integers(0, max(1, max_offset + 1))
        
        # Add additional bounds check
        if offset >= total_rows:
            offset = 0
            
        lf_sub = (
            lf.with_row_count("_idx")
            .filter(((pl.col("_idx") + offset) % step) == 0)
            .drop("_idx")
            .limit(n)
        )
        return lf_sub
        
    except Exception as e:
        # Fallback for any polars errors - just limit the dataframe
        print(f"[WARNING] _lazyframe_sample failed with {e}, falling back to simple limit")
        try:
            return lf.limit(n)
        except Exception:
            # Last resort - return original dataframe
            print(f"[WARNING] Even simple limit failed, returning original dataframe")
            return lf

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def probability_diagnostics(
    dataset_path: str | Path,
    run_dir: str | Path,
    *,
    sample: int | None = 100_000,
    n_bins: int = 15,
    neg_ratio: float | None = None,
) -> None:
    """Generate probability histograms & calibration curves for donor/acceptor.

    Parameters
    ----------
    dataset_path : str or Path
        Parquet file or directory containing the row-level feature matrix.
    run_dir : str or Path
        Directory with the trained meta model artefacts.
    sample : int or None, default 100_000
        Cap on *positive* (splice-site) rows to include. Use ``0`` or ``None``
        to keep all positive rows. Negative (neither) rows are sampled
        separately according to *neg_ratio*.
    n_bins : int, default 15
        Number of bins for calibration plots.
    neg_ratio : float or None, default None
        When set to a positive value, the diagnostics will **keep all positive
        rows (up to *sample*) and randomly sample negatives so that
        ``n_neg = n_pos * neg_ratio``.  A ratio of ``1`` yields a balanced
        validation set.  If ``None`` the old behaviour is preserved: a uniform
        row sample irrespective of class labels.
    """
    import matplotlib.pyplot as plt  # heavy import only when needed
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import f1_score

    run_dir = Path(run_dir)
    predict_fn, feature_names = _load_model_generic(run_dir)

    # ---------------------------------------------------------------
    # Load data lazily with optional stratified sampling
    # ---------------------------------------------------------------
    needed_cols = feature_names + ["splice_type"]
    
    # Preserve gene_id for gene-aware sampling if environment variable is set
    preserve_gene_id = os.environ.get('SS_PRESERVE_GENE_ID') == '1'
    if preserve_gene_id:
        needed_cols.append("gene_id")
    
    if Path(dataset_path).is_dir():
        lf_all = pl.scan_parquet(str(Path(dataset_path) / "*.parquet"), missing_columns="insert")
    else:
        lf_all = pl.scan_parquet(str(dataset_path), missing_columns="insert")
    
    # Use helper function for robust column selection
    lf_all, missing_cols, existing_cols = select_available_columns(
        lf_all, needed_cols, context_name="Probability Diagnostics"
    )

    if neg_ratio is None:
        # Legacy: uniform sample of *sample* rows irrespective of label
        if sample is not None and sample != 0:
            lf_all = _lazyframe_sample_gene_aware(lf_all, sample, seed=42)
        
        # Remove gene_id after sampling if it was only added for sampling
        if preserve_gene_id and "gene_id" not in feature_names and "gene_id" in existing_cols:
            final_cols = [col for col in existing_cols if col != "gene_id"]
            lf_all = lf_all.select(final_cols)
        
        df = lf_all.collect(streaming=True).to_pandas()
    else:
        # Stratified: keep positives (optionally capped) and sample negatives
        pos_lf = lf_all.filter(pl.col("splice_type") != "neither")
        if sample not in (None, 0):
            pos_lf = _lazyframe_sample_gene_aware(pos_lf, sample, seed=42)
        # Count positives to determine how many negatives are needed
        n_pos = pos_lf.select(pl.count()).collect(streaming=True).to_series()[0]
        n_neg_target = int(n_pos * neg_ratio)

        neg_lf = lf_all.filter(pl.col("splice_type") == "neither")
        # If dataset has fewer negatives than requested, keep them all
        if n_neg_target > 0:
            neg_lf = _lazyframe_sample_gene_aware(neg_lf, n_neg_target, seed=123)
        lf_comb = pl.concat([pos_lf, neg_lf])
        
        # Remove gene_id after sampling if it was only added for sampling
        if preserve_gene_id and "gene_id" not in feature_names and "gene_id" in existing_cols:
            final_cols = [col for col in existing_cols if col != "gene_id"]
            lf_comb = lf_comb.select(final_cols)
        
        df = lf_comb.collect(streaming=True).to_pandas()


    # Prepare y labels
    y_true = _encode_labels(df["splice_type"].to_numpy())
    
    # Preprocess features using the standardized helper function
    df = _preprocess_features_for_model(df, feature_names)
    
    # Convert to numpy array for prediction
    X = df[feature_names].to_numpy(np.float32)
    proba = predict_fn(X)

    # For each class (1=donor, 2=acceptor) plot prob distribution & calibration
    class_map = {1: "donor", 2: "acceptor"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for idx, cls in enumerate([1, 2]):
        ax_hist = axes[idx, 0]
        ax_cal = axes[idx, 1]
        p = proba[:, cls]
        ax_hist.hist(p, bins=n_bins, color="skyblue", edgecolor="k")
        ax_hist.set_title(f"{class_map[cls]} probability distribution")
        ax_hist.set_xlabel("predicted P")
        ax_hist.set_ylabel("count")

        frac_pos_true, mean_pred = calibration_curve((y_true == cls).astype(int), p, n_bins=n_bins, strategy="uniform")
        ax_cal.plot(mean_pred, frac_pos_true, "s-", label="calibration")
        ax_cal.plot([0, 1], [0, 1], "k--", alpha=0.6)
        ax_cal.set_title(f"{class_map[cls]} calibration")
        ax_cal.set_xlabel("mean predicted P")
        ax_cal.set_ylabel("fraction positives")
        ax_cal.legend()

    fig.tight_layout()
    out_png = run_dir / "probability_diagnostics.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # Threshold sweep on donor+acceptor vs rest (binary relevance for F1)
    y_binary = (y_true != 0).astype(int)  # 1 for splice site, 0 for neither
    proba_site = proba[:, 1:].max(axis=1)  # best of donor/acceptor
    # Helper to choose highest-threshold within eps of best F1
    def _best_threshold(y_bin: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
        grid = np.linspace(0.05, 0.95, 46)
        f1s = [f1_score(y_bin, (scores >= t).astype(int)) for t in grid]
        arr = np.array(f1s)
        best = arr.max()
        eps = 0.002
        cand = grid[arr >= best - eps]
        return cand.max(), best

    # Global (any splice) threshold
    best_t, best_f1 = _best_threshold(y_binary, proba_site)
    print(f"[diag] Best binary F1={best_f1:.3f} at threshold≈{best_t:.2f}")

    # Per-class thresholds
    t_donor, f1_donor = _best_threshold((y_true == 1).astype(int), proba[:, 1])
    t_acceptor, f1_acceptor = _best_threshold((y_true == 2).astype(int), proba[:, 2])
    # ------------------------------------------------------------
    # Save threshold suggestion + model metadata for later scripts
    # ------------------------------------------------------------
    # Inspect the saved model to infer output layer & calibration
    output_layer: str = "unknown"
    calibrated: bool | str = False
    calib_method: str = "none"
    try:
        import pickle
        pkl_path = run_dir / "model_multiclass.pkl"
        with open(pkl_path, "rb") as fh:
            _mdl = pickle.load(fh)

    except Exception:
        _mdl = None

    if _mdl is not None:
        from meta_spliceai.splice_engine.meta_models.training.classifier_utils import (
            SigmoidEnsemble, CalibratedSigmoidEnsemble
        )
        if isinstance(_mdl, (SigmoidEnsemble, CalibratedSigmoidEnsemble)):
            output_layer = "sigmoid_ovr"
        else:
            # assume multiclass softmax booster
            output_layer = "softmax"
        if isinstance(_mdl, CalibratedSigmoidEnsemble):
            calibrated = True
            cal = _mdl.calibrator
            try:
                from sklearn.isotonic import IsotonicRegression
                from sklearn.linear_model import LogisticRegression
                if isinstance(cal, IsotonicRegression):
                    calib_method = "isotonic"
                elif isinstance(cal, LogisticRegression):
                    calib_method = "platt"
                else:
                    calib_method = type(cal).__name__
            except Exception:
                calib_method = type(cal).__name__ if cal is not None else "none"
        else:
            calibrated = False
    # Write tab-separated metadata + suggestion
    meta_lines = [
        f"output_layer\t{output_layer}",
        f"calibrated\t{calibrated}",
        f"calib_method\t{calib_method}",
        f"threshold_global\t{best_t:.3f}",
        f"F1_global\t{best_f1:.3f}",
        f"threshold_donor\t{t_donor:.3f}",
        f"F1_donor\t{f1_donor:.3f}",
        f"threshold_acceptor\t{t_acceptor:.3f}",
        f"F1_acceptor\t{f1_acceptor:.3f}",
    ]
    (run_dir / "threshold_suggestion.txt").write_text("\n".join(meta_lines) + "\n")


# -----------------------------------------------------------------------------
#  Threshold loading helper (used by downstream evaluation scripts)
# -----------------------------------------------------------------------------

def load_thresholds(run_dir: str | Path) -> Dict[str, float]:
    """Parse ``threshold_suggestion.txt`` into a {key: value} dict with floats.

    Unknown / non-float values are skipped. Returns an empty dict if the file
    does not exist.
    """
    from pathlib import Path

    run_dir = Path(run_dir)
    thr_path = run_dir / "threshold_suggestion.txt"
    out: Dict[str, float] = {}
    if not thr_path.exists():
        return out
    for ln in thr_path.read_text().splitlines():
        if "\t" not in ln:
            continue
        k, v = ln.split("\t", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            # Ignore non-numeric entries (e.g., output_layer)
            continue
    return out

def richer_metrics(dataset_path: str | Path, run_dir: str | Path, *, sample: int | None = 100_000, plot_curves: bool = True) -> Dict[str, float]:
    """Return dict with accuracy, macro-F1, Cohen's kappa, Matthews CC, ROC-AUC on a (possibly sampled) slice.

    Parameters
    ----------
    dataset_path
        Master dataset directory or a single Parquet file.
    run_dir
        Directory containing the trained model artefacts.
    sample
        Number of rows to sample for evaluation. ``None`` → use entire dataset
        (may be heavy).
    """
    run_dir = Path(run_dir)
    predict_fn, feature_names = _load_model_generic(run_dir)

    # load dataset lazily, select needed columns only
    feat_cols = feature_names
    select_cols = feat_cols + ["splice_type"]
    
    # Preserve gene_id for gene-aware sampling if environment variable is set
    preserve_gene_id = os.environ.get('SS_PRESERVE_GENE_ID') == '1'
    if preserve_gene_id:
        select_cols.append("gene_id")

    if Path(dataset_path).is_dir():
        lf = pl.scan_parquet(str(Path(dataset_path) / "*.parquet"), missing_columns="insert")
    else:
        lf = pl.scan_parquet(str(dataset_path), missing_columns="insert")

    # Use helper function for robust column selection
    lf, missing_cols, existing_cols = select_available_columns(
        lf, select_cols, context_name="Richer Metrics", verbose=False
    )
    
    # Sample if requested - use gene-aware sampling to preserve gene structure
    if sample is not None:
        try:
            # Check if gene_id is available for gene-aware sampling
            if "gene_id" in existing_cols:
                lf = _lazyframe_sample_gene_aware(lf, sample, seed=42)
                # Remove gene_id after sampling if it was only added for sampling
                if preserve_gene_id and "gene_id" not in feature_names:
                    lf = lf.select([col for col in existing_cols if col != "gene_id"])
            else:
                print("[Info] Gene-aware sampling not available: gene_id column not found, using row-based sampling")
                lf = _lazyframe_sample(lf, sample, seed=42)
        except Exception as e:
            print(f"[Warning] Gene-aware sampling failed: {e}")
            lf = _lazyframe_sample(lf, sample, seed=42)
    
    # Try to collect with streaming for memory efficiency
    try:
        print(f"Loading dataset with {len(select_cols)} columns")
        df = lf.collect(streaming=True).to_pandas()
    except pl.ColumnNotFoundError:
        print("Fallback: loading dataset file by file due to schema issues")
        # Fallback: some shards miss columns despite union schema → load file-by-file
        from glob import glob
        files: list[str]
        if Path(dataset_path).is_dir():
            files = sorted(glob(str(Path(dataset_path) / "*.parquet")))
        else:
            files = [str(dataset_path)]
        parts: list[pd.DataFrame] = []
        want = set(select_cols)
        rows_remaining = sample if sample is not None else None
        per_file_quota = None if rows_remaining is None else max(1, rows_remaining // max(1, len(files)))
        for fp in files:
            # read only intersecting columns
            schema_cols = set(pl.read_parquet(fp, n_rows=0).columns)
            use_cols = [c for c in select_cols if c in schema_cols]
            if "splice_type" not in use_cols:
                use_cols.append("splice_type")
            pdf = pl.read_parquet(fp, columns=use_cols).to_pandas()
            if rows_remaining is not None and len(pdf) > per_file_quota:
                pdf = pdf.sample(per_file_quota, random_state=42)
            parts.append(pdf)
            if rows_remaining is not None:
                rows_remaining -= len(pdf)
                if rows_remaining <= 0:
                    break
        df = pd.concat(parts, ignore_index=True, copy=False)
        # Add any still-missing feature cols as zeros
        for col in feat_cols:
            if col not in df.columns:
                df[col] = 0.0

    # Drop rows where splice_type is missing or unrecognized
    df = df.dropna(subset=["splice_type"])
    df = df[df["splice_type"].isin(_LABEL_MAP_STR.keys())]

    # Determine if 'chrom' is in the dataset but not in feat_cols (potential issue)  
    chrom_col = None
    for col_name in df.columns:
        if col_name.lower() in ['chrom', 'chromosome', 'chr']:
            chrom_col = col_name
            break
    
    if chrom_col and chrom_col not in feat_cols and predict_fn is not None:
        # Get expected feature count from the model if possible
        expected_features = None
        try:
            # Try to determine expected feature count from the model
            if hasattr(predict_fn, 'models') and predict_fn.models:
                if hasattr(predict_fn.models[0], 'feature_names_in_'):
                    expected_features = predict_fn.models[0].feature_names_in_
                elif hasattr(predict_fn.models[0], 'n_features_in_'):
                    expected_features = predict_fn.models[0].n_features_in_
        except (AttributeError, IndexError):
            pass
            
        if expected_features is not None and len(feat_cols) < expected_features:
            print(f"Warning: Model expects {expected_features} features but only {len(feat_cols)} provided")
            print(f"Adding chromosome column '{chrom_col}' to features to maintain compatibility")
            feat_cols = list(feat_cols) + [chrom_col]
    
    # Handle non-numeric columns to preserve feature count compatibility
    for col in feat_cols:
        if col in df.columns:
            try:
                # Try to convert column to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Check for NaN values from failed conversions
                if df[col].isna().any():
                    # Use different defaults based on feature type
                    if _is_kmer(col):
                        # For k-mers, 0 represents no occurrence which is more semantically correct
                        print(f"Warning: Column '{col}' has non-numeric values that were converted to NaN, replacing with 0")
                        df[col] = df[col].fillna(0)
                    else:
                        # For other features, use -1 as before
                        print(f"Warning: Column '{col}' has non-numeric values that were converted to NaN, replacing with -1")
                        df[col] = df[col].fillna(-1)
            except (ValueError, TypeError):
                # For columns like 'chrom' which may contain non-numeric values (e.g., 'X', 'Y')
                print(f"Warning: Converting non-numeric column '{col}' to dummy values")
                # Map unique values to integers to maintain feature count
                unique_vals = df[col].dropna().unique()
                val_map = {val: idx for idx, val in enumerate(unique_vals)}
                # Apply mapping and fill NaN with -1
                df[col] = df[col].map(val_map).fillna(-1).astype(float)
    
    # Ensure all feature columns exist in the dataframe
    for col in feat_cols:
        if col not in df.columns:
            print(f"Warning: Missing column '{col}', adding with zeros")
            df[col] = 0.0

    # Use all original feature columns to maintain exact feature count
    X = df[feat_cols].values.astype(np.float32)
    y = _encode_labels(df["splice_type"].values)

    proba = predict_fn(X)
    pred = proba.argmax(axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }

    # ---------------- Curves (one-vs-rest) ----------------
    if plot_curves:
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve, precision_recall_curve

            label_names = ["neither", "donor", "acceptor"]
            plt.figure(figsize=(6, 5))
            for i, lab in enumerate(label_names):
                if (y == i).sum() == 0:
                    continue  # class absent in sample
                fpr, tpr, _ = roc_curve(y == i, proba[:, i])
                plt.plot(fpr, tpr, label=lab)
                # save CSV for exact coordinates
                _csv = pd.DataFrame({"fpr": fpr, "tpr": tpr})
                _csv.to_csv(run_dir / f"roc_{lab}.csv", index=False)
            plt.plot([0, 1], [0, 1], "k--", lw=1)
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC curve (OvR)")
            plt.legend(); plt.tight_layout()
            plt.savefig(run_dir / "roc_curve.png", dpi=150)
            plt.close()

            plt.figure(figsize=(6, 5))
            for i, lab in enumerate(label_names):
                if (y == i).sum() == 0:
                    continue
                prec, rec, _ = precision_recall_curve(y == i, proba[:, i])
                plt.plot(rec, prec, label=lab)
                _csv = pd.DataFrame({"recall": rec, "precision": prec})
                _csv.to_csv(run_dir / f"pr_{lab}.csv", index=False)
            plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve (OvR)")
            plt.legend(); plt.tight_layout()
            plt.savefig(run_dir / "pr_curve.png", dpi=150)
            plt.close()
        except ImportError:  # matplotlib not installed
            warnings.warn("matplotlib missing – ROC/PR curves skipped", RuntimeWarning)
    # roc_auc_score for multiclass may fail if a class is missing – guard
    try:
        metrics["ovr_roc_auc"] = float(roc_auc_score(y, proba, multi_class="ovr"))
    except ValueError as exc:  # e.g. only two classes present in sample
        warnings.warn(str(exc), RuntimeWarning)

    # persist
    (run_dir / "metrics_richer.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def leakage_probe(dataset_path: str | Path, run_dir: str | Path, *, threshold: float = 0.95, sample: int | None = 50_000, method: str = "pearson") -> pd.DataFrame:
    """Run correlation-based leakage probe; returns DataFrame and writes TSV.
    
    Produces a comprehensive report of feature correlations with the target variable,
    marking features that exceed the threshold as potential leakage sources.
    
    Parameters
    ----------
    dataset_path : str | Path
        Path to dataset file or directory
    run_dir : str | Path
        Directory for output files
    threshold : float, default=0.95
        Correlation threshold to mark features as leaky
    sample : int | None, default=50_000
        Number of samples to use for correlation calculation
    method : str, default="pearson"
        Correlation method ("pearson" or "spearman")
    
    Returns
    -------
    pd.DataFrame
        DataFrame with feature correlations and leakage status
    """
    run_dir = Path(run_dir)
    manifest_csv = run_dir / "feature_manifest.csv"
    if not manifest_csv.exists():
        # try JSON manifest – convert to temp CSV in memory
        json_path = run_dir / "train.features.json"
        if not json_path.exists():
            raise FileNotFoundError("feature manifest not found")
        features = json.loads(json_path.read_text())["feature_names"]
        manifest_csv = run_dir / "_tmp_manifest.csv"
        pd.DataFrame({"feature": features}).to_csv(manifest_csv, index=False)

    # Get comprehensive correlation report (returns all features, not just potential leaky ones)
    df_hits = _leak.probe_leakage(dataset_path, manifest_csv, method=method, threshold=threshold, 
                               sample=sample, return_all=True)
    
    # Save the comprehensive report
    report_path = run_dir / "feature_correlations.csv"
    df_hits.to_csv(report_path, index=False)
    
    # Also save the leaky features report for backward compatibility
    leaky_features = df_hits[df_hits["is_leaky"]]
    if not leaky_features.empty:
        out_path = run_dir / "leakage_probe.tsv"
        leaky_features.to_csv(out_path, sep="\t", index=False)
        print(f"[WARNING] Found {len(leaky_features)} potentially leaky features with correlation >= {threshold}:")
        for _, row in leaky_features.iterrows():
            print(f"  - {row['feature']}: correlation = {row['correlation']:.4f}")
    else:
        print(f"No leaky features found (threshold = {threshold})")
        
    print(f"Full correlation report saved to: {report_path}")
    return df_hits


def gene_score_delta(dataset_path: str | Path, run_dir: str | Path, *, sample: int | None = None) -> Path:
    """Compute per-gene deltas and write CSV; returns path."""
    out_csv = Path(run_dir) / "gene_deltas.csv"
    res_df = _gene_delta.compute_gene_score_delta_multiclass(dataset_path=dataset_path, model_dir=run_dir, sample=sample)
    res_df.to_csv(out_csv, index=False)
    return out_csv


def shap_importance(dataset_path: str | Path, run_dir: str | Path, *, sample: int = 20_000, background_size: int = 100, chunk_size: int = 1000, memory_limit_mb: int = 8000) -> Path:
    """Compute SHAP feature importance.

    Parameters
    ----------
    dataset_path : str | Path
        Path to dataset file or directory
    run_dir : str | Path
        Path to model directory
    sample : int, default=20_000
        Number of samples to use for SHAP analysis
    background_size : int, default=100
        Number of samples to use for SHAP background
    chunk_size : int, default=1000
        Number of samples to process at once to reduce memory usage
    memory_limit_mb : int, default=8000
        Memory limit in MB before falling back to gain importance
    
    Returns
    -------
    Path
        Path to output CSV file
    
    Notes
    -----
    For custom ensemble models, we aggregate SHAP values across all base models.
    For large models, we use chunking to reduce memory usage.
    """
    run_dir = Path(run_dir)
    predict_fn, feature_names = _load_model_generic(run_dir)

    # ------------------------------------------------------------------
    # 0. Memory-budget guard: SHAP creates dense matrices of shape (rows, fmts)
    #    which can explode quickly.  Rough float32 footprint = 4 bytes.
    #    We double it to account for SHAP outputs and overhead.
    # ------------------------------------------------------------------
    if sample is not None:
        expected_bytes = sample * len(feature_names) * 4 * 2.5  # safety factor
        if expected_bytes > memory_limit_mb * 1_000_000:  # MB to bytes
            import xgboost as _xgb, pandas as _pd, pickle as _pkl
            print(
                f"[shap_importance] Requested sample too large (rows={sample:,}, "
                f"features={len(feature_names)}) – using gain importance instead.)"
            )
            model_path_json = run_dir / "model_multiclass.json"
            if model_path_json.exists():
                booster = _xgb.Booster(); booster.load_model(str(model_path_json))
            else:
                with open(run_dir / "model_multiclass.pkl", "rb") as fh:
                    loaded_model = _pkl.load(fh)
                # --------------------------------------------------------------
                # Support various ensemble types with binary models inside
                # --------------------------------------------------------------
                if hasattr(loaded_model, "get_booster"):
                    booster = loaded_model.get_booster()
                    score_dict = booster.get_score(importance_type="gain")
                elif hasattr(loaded_model, "models") or hasattr(loaded_model, "get_base_models"):
                    # Aggregate gain across the three binary boosters
                    score_dict: dict[str, float] = {}
                    
                    # Get the base models - handle both SigmoidEnsemble and PerClassCalibratedSigmoidEnsemble
                    if hasattr(loaded_model, "get_base_models"):
                        models = loaded_model.get_base_models()
                    else:
                        models = loaded_model.models
                        
                    for mdl in models:  # type: ignore[attr-defined]
                        b = mdl.get_booster()
                        for feat, val in b.get_score(importance_type="gain").items():
                            score_dict[feat] = score_dict.get(feat, 0.0) + val
                else:
                    raise AttributeError("Loaded model does not expose .get_booster nor .models for importance computation")
            imp_df = _pd.DataFrame({
                "feature": feature_names,
                "importance": [score_dict.get(f, 0.0) for f in feature_names],
            })
            imp_df.sort_values("importance", ascending=False, inplace=True)
            out_csv = run_dir / "shap_importance.csv"
            imp_df.to_csv(out_csv, index=False)
            return out_csv

    # sample dataset rows – let Polars insert missing columns automatically
    if Path(dataset_path).is_dir():
        lf = pl.scan_parquet(str(Path(dataset_path) / "*.parquet"), missing_columns="insert")
    else:
        lf = pl.scan_parquet(str(dataset_path), missing_columns="insert")
    # Preserve gene_id for gene-aware sampling, then select features
    if "gene_id" not in feature_names:
        # Include gene_id temporarily for sampling
        lf_with_gene_id = lf.select(feature_names + ["gene_id"])
        lf_sampled = _lazyframe_sample_gene_aware(lf_with_gene_id, sample, seed=42)
        # Remove gene_id after sampling
        lf = lf_sampled.select(feature_names)
    else:
        lf = lf.select(feature_names)
        lf = _lazyframe_sample_gene_aware(lf, sample, seed=42)
    try:
        # First collect as polars dataframe, then use the same preprocessing as training
        df_pl = lf.collect(streaming=True)
        
        # Apply the same preprocessing pipeline used during training
        # This includes proper chromosome encoding for mixed string/numeric chromosomes
        from meta_spliceai.splice_engine.meta_models.builder.preprocessing import prepare_training_data
        
        # We need to add a dummy splice_type column to use prepare_training_data
        if 'splice_type' not in df_pl.columns:
            df_pl = df_pl.with_columns(pl.lit("neither").alias("splice_type"))
        
        # Use the same preprocessing as training with chromosome encoding
        df_processed, _ = prepare_training_data(
            df_pl,
            label_col="splice_type",
            encode_chrom=True,  # This handles chromosome string->numeric conversion
            return_type="pandas",
            verbose=0
        )
        
        # Select only the features we need for SHAP
        available_features = [f for f in feature_names if f in df_processed.columns]
        missing_features = [f for f in feature_names if f not in df_processed.columns]
        
        if missing_features:
            print(f"[shap_importance] Warning: {len(missing_features)} features missing from processed data")
            # Add missing features as zeros
            for feature in missing_features:
                df_processed[feature] = 0.0
        
        # Ensure column order matches feature_names
        df = df_processed[feature_names]
        
        # Convert to float32 array for SHAP
        import numpy as np  # Import numpy here to ensure it's available
        X = df.values.astype(np.float32)
        
        # Print info for diagnostics
        print(f"[shap_importance] Loaded {X.shape[0]} samples with {X.shape[1]} features")
    except (pl.ColumnNotFoundError, UnboundLocalError, AttributeError, ValueError) as e:
        # Handle data loading or numpy reference errors
        print(f"[shap_importance] Error processing dataset: {e}")
        print("[shap_importance] Falling back to file-by-file processing with proper preprocessing...")
        # Fallback: iterate files and manually union columns, then apply proper preprocessing
        import numpy as np  # Ensure numpy is available
        from glob import glob
        from meta_spliceai.splice_engine.meta_models.builder.preprocessing import prepare_training_data
        
        files: list[str]
        if Path(dataset_path).is_dir():
            files = sorted(glob(str(Path(dataset_path) / "*.parquet")))
        else:
            files = [str(dataset_path)]
            
        parts: list[pl.DataFrame] = []
        rows_remaining = sample if sample is not None else None
        per_file_quota = None if rows_remaining is None else max(1, rows_remaining // max(1, len(files)))
        
        for fp in files:
            # Read with polars to maintain data types
            df_part = pl.read_parquet(fp)
            
            # Sample if needed
            if rows_remaining is not None and len(df_part) > per_file_quota:
                df_part = df_part.sample(per_file_quota, seed=42)
                
            parts.append(df_part)
            if rows_remaining is not None:
                rows_remaining -= len(df_part)
                if rows_remaining <= 0:
                    break
        
        # Combine all parts
        df_combined = pl.concat(parts, how="vertical_relaxed")
        
        # Add dummy splice_type column if missing
        if 'splice_type' not in df_combined.columns:
            df_combined = df_combined.with_columns(pl.lit("neither").alias("splice_type"))
        
        # Apply the same preprocessing as training
        try:
            df_processed, _ = prepare_training_data(
                df_combined,
                label_col="splice_type",
                encode_chrom=True,  # This handles chromosome string->numeric conversion
                return_type="pandas",
                verbose=0
            )
            
            # Select only the features we need, adding zeros for missing ones
            for col in feature_names:
                if col not in df_processed.columns:
                    df_processed[col] = 0.0
            
            X = df_processed[feature_names].values.astype(np.float32)
            print(f"[shap_importance] Fallback processing successful: {X.shape[0]} samples with {X.shape[1]} features")
            
        except Exception as preprocessing_error:
            print(f"[shap_importance] Preprocessing also failed: {preprocessing_error}")
            print("[shap_importance] Using basic preprocessing as last resort...")
            # Last resort: basic preprocessing
            df_basic = df_combined.to_pandas()
            for col in feature_names:
                if col not in df_basic.columns:
                    df_basic[col] = 0.0
                elif df_basic[col].dtype == object:
                    # Handle chromosomes specifically
                    if col == 'chrom':
                        # Basic chromosome encoding
                        chrom_map = {str(i): i for i in range(1, 23)}
                        chrom_map.update({'X': 23, 'Y': 24, 'MT': 25, 'M': 25})
                        df_basic[col] = df_basic[col].map(lambda x: chrom_map.get(str(x), 100))
                    else:
                        # Generic object->numeric conversion
                        unique_vals = df_basic[col].dropna().unique()
                        val_map = {val: idx for idx, val in enumerate(unique_vals)}
                        df_basic[col] = df_basic[col].map(val_map).fillna(0)
                        
            X = df_basic[feature_names].values.astype(np.float32)

    # We need model object, not just predict_fn, for SHAP
    import pickle, xgboost as xgb
    model_path_json = run_dir / "model_multiclass.json"
    if model_path_json.exists():
        model = xgb.Booster(); model.load_model(str(model_path_json))
    else:
        with open(run_dir / "model_multiclass.pkl", "rb") as fh:
            model = pickle.load(fh)

    try:
        # Check if we're dealing with a custom ensemble model that requires special handling
        if hasattr(model, 'get_base_models'):
            # Get underlying binary models for PerClassCalibratedSigmoidEnsemble
            print("[INFO] Processing custom ensemble model with specialized SHAP handling")
            binary_models = model.get_base_models()
            
            # Use chunking to process data in batches to avoid memory issues
            print(f"[INFO] Processing {len(X)} samples in chunks of {chunk_size}")
            
            # Initialize an array to store aggregated SHAP values
            import shap, json, os, gc
            import numpy as np  # Import numpy again to ensure it's available
            print("Loading model...")
            clf = _load_model(run_dir)
            from tqdm import tqdm
            
            # First get background data for explainers
            if len(X) > background_size:
                bg_indices = np.random.choice(len(X), background_size, replace=False)
                background_data = X[bg_indices]
            else:
                background_data = X
                
            # Create explainers for each binary model - INITIALIZE VARIABLE FIRST
            explainers = []
            for i, binary_model in enumerate(binary_models):
                try:
                    explainer = shap.TreeExplainer(binary_model, background_data, model_output="probability")
                    explainers.append(explainer)
                except Exception as e:
                    print(f"[WARNING] Could not create explainer for model {i}: {e}")
                    # Fall back to gain importance for this model
                    explainers.append(None)
            
            # Process data in chunks to save memory
            total_chunks = (len(X) + chunk_size - 1) // chunk_size
            all_shap_values = None
            
            for chunk_idx in tqdm(range(total_chunks), desc="Processing SHAP chunks"):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(X))
                X_chunk = X[start_idx:end_idx]
                
                # Calculate and aggregate SHAP values for each binary model
                chunk_shap_values = None
                for i, (explainer, binary_model) in enumerate(zip(explainers, binary_models)):
                    if explainer is not None:
                        try:
                            # Get SHAP values for this model
                            model_shap = explainer.shap_values(X_chunk)
                            
                            # For binary classification, shap_values might return [negative, positive]
                            # We want the positive class (index 1)
                            if isinstance(model_shap, list) and len(model_shap) > 1:
                                model_shap = model_shap[1]
                            
                            # Initialize or add to aggregated values
                            if chunk_shap_values is None:
                                chunk_shap_values = model_shap
                            else:
                                chunk_shap_values += model_shap
                        except Exception as e:
                            print(f"[WARNING] SHAP calculation failed for model {i} chunk {chunk_idx}: {e}")
                
                # Aggregate this chunk's SHAP values with the total
                if chunk_shap_values is not None:
                    if all_shap_values is None:
                        all_shap_values = chunk_shap_values
                    else:
                        all_shap_values = np.vstack((all_shap_values, chunk_shap_values))
                
                # Force garbage collection between chunks
                import gc
                gc.collect()
            
            # Calculate feature importance from aggregated SHAP values
            if all_shap_values is not None:
                mean_abs_shap = np.mean(np.abs(all_shap_values), axis=0)
                imp_df = pd.DataFrame({
                    "feature": feature_names,
                    "importance": mean_abs_shap
                })
                imp_df.sort_values("importance", ascending=False, inplace=True)
            else:
                # Fall back to gain importance if SHAP failed completely
                raise Exception("SHAP calculation failed for all models/chunks")
        else:
            # Standard approach for supported model types
            imp_df = explainers.shap_feature_importance(model, X, feature_names, background_size=background_size)
    except Exception as exc:
        # Frequently arises from a faulty transformers/Keras import path when
        # SHAP auto-detects HuggingFace models even though they're not used
        # here.  Gracefully degrade by skipping SHAP and falling back to basic
        # gain-based importance from the booster – always available.
        import xgboost as xgb
        print("[warning] SHAP importance failed – falling back to in-model gain: ", exc)
        score_dict: dict[str, float] = {}
        if isinstance(model, xgb.Booster):
            # Raw booster available directly
            booster: xgb.Booster = model  # type: ignore[no-redef]
            score_dict = booster.get_score(importance_type="gain")
        elif hasattr(model, "get_booster"):
            booster = model.get_booster()  # type: ignore[attr-defined]
            score_dict = booster.get_score(importance_type="gain")
        elif hasattr(model, "models"):
            # Aggregate gain across underlying binary boosters (SigmoidEnsemble)
            for mdl in model.models:  # type: ignore[attr-defined]
                b = mdl.get_booster()
                for feat, val in b.get_score(importance_type="gain").items():
                    score_dict[feat] = score_dict.get(feat, 0.0) + val
        elif hasattr(model, "get_base_models"):
            # Handle PerClassCalibratedSigmoidEnsemble
            for mdl in model.get_base_models():
                try:
                    b = mdl.get_booster()
                    for feat, val in b.get_score(importance_type="gain").items():
                        score_dict[feat] = score_dict.get(feat, 0.0) + val
                except Exception as e:
                    print(f"[WARNING] Failed to get gain importance from model: {e}")
        else:
            raise AttributeError("Loaded model does not expose .get_booster, .models, or .get_base_models for importance computation")
        # Map XGBoost's f0…fN keys back to original feature names
        index_map = {f"f{idx}": name for idx, name in enumerate(feature_names)}
        named_scores = {index_map.get(k, k): v for k, v in score_dict.items()}
        # Ensure every feature is present
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": [named_scores.get(name, 0.0) for name in feature_names],
        })
        imp_df.sort_values("importance", ascending=False, inplace=True)

    out_csv = run_dir / "shap_importance.csv"
    imp_df.to_csv(out_csv, index=False)
    return out_csv
# -----------------------------------------------------------------------------
#  Meta splice-site performance helper (donor/acceptor)
# -----------------------------------------------------------------------------

def meta_splice_performance(
    dataset_path: str | Path,
    run_dir: str | Path,
    annotations_path: str | Path | None = None,
    *,
    threshold: float | None = None,
    threshold_donor: float | None = None,
    threshold_acceptor: float | None = None,
    consensus_window: int = 2,
    sample: int | None = None,
    base_tsv: str | Path | None = None,
    errors_only: bool = False,
    error_artifact: str | Path | None = None,
    include_tns: bool = False,
    verbose: int = 2,
) -> Path:
    """Convenience wrapper around eval_meta_splice.meta_splice_performance.

    Returns the output TSV path produced by the evaluation.
    """
    # Import the module and access the comprehensive function directly
    # (avoid conflict with simplified function of same name)
    import meta_spliceai.splice_engine.meta_models.training.eval_meta_splice as _eval_module
    
    # Get the first (comprehensive) function by inspecting the module source
    # The comprehensive function has more complex logic and more parameters
    import inspect
    import types
    
    # Find the comprehensive meta_splice_performance function (first definition)
    _core = None
    for name, obj in inspect.getmembers(_eval_module, inspect.isfunction):
        if name == 'meta_splice_performance':
            # Check if this is the comprehensive version by parameter count
            sig = inspect.signature(obj)
            if len(sig.parameters) > 15:  # Comprehensive version has many parameters
                _core = obj
                break
    
    if _core is None:
        # Fallback to any meta_splice_performance function
        _core = getattr(_eval_module, 'meta_splice_performance')

    # If annotations_path is None, try to find default splice sites file
    if annotations_path is None:
        # Try to find splice sites file in common locations
        potential_paths = [
            Path("data/ensembl/splice_sites.tsv"),
            Path("data/ensembl/splice_sites.parquet"),
            Path("data/ensembl/spliceai_eval/splice_sites.tsv"),
            Path("data/ensembl/spliceai_eval/splice_sites.parquet"),
        ]
        
        for path in potential_paths:
            if path.exists():
                annotations_path = path
                break
        
        if annotations_path is None:
            # Create a warning but continue - the function can work without annotations
            warnings.warn(
                "No splice sites annotations file found. Meta splice performance evaluation will use dataset-internal truth columns only.",
                RuntimeWarning
            )

    try:
        return _core(
            dataset_path=dataset_path,
            run_dir=run_dir,
            annotations_path=annotations_path,
            threshold=threshold,
            threshold_donor=threshold_donor,
            threshold_acceptor=threshold_acceptor,
            consensus_window=consensus_window,
            sample=sample,
            base_tsv=base_tsv,
            error_artifact=error_artifact,
            errors_only=errors_only,
            include_tns=include_tns,
            verbose=verbose,
        )
    except Exception as e:
        # Enhanced error handling with more specific information
        run_dir = Path(run_dir)
        error_msg = f"meta_splice_performance failed: {e}"
        
        # Log detailed error information
        if verbose >= 1:
            print(f"[ERROR] {error_msg}")
            print(f"[DEBUG] Dataset path: {dataset_path}")
            print(f"[DEBUG] Run directory: {run_dir}")
            print(f"[DEBUG] Annotations path: {annotations_path}")
            
            # Check if basic files exist
            model_pkl = run_dir / "model_multiclass.pkl"
            model_json = run_dir / "model_multiclass.json"
            feature_manifest = run_dir / "feature_manifest.csv"
            
            print(f"[DEBUG] Model pkl exists: {model_pkl.exists()}")
            print(f"[DEBUG] Model json exists: {model_json.exists()}")
            print(f"[DEBUG] Feature manifest exists: {feature_manifest.exists()}")
            
            if verbose >= 2:
                import traceback
                traceback.print_exc()
        
        # Try to create a minimal output file to prevent downstream failures
        try:
            minimal_output = run_dir / "full_splice_performance_meta.tsv"
            if not minimal_output.exists():
                # Create minimal header-only file
                minimal_data = pd.DataFrame(columns=[
                    "gene_id", "site_type", "TP", "FP", "FN", "precision", "recall", "f1_score"
                ])
                minimal_data.to_csv(minimal_output, sep="\t", index=False)
                if verbose >= 1:
                    print(f"[DEBUG] Created minimal output file: {minimal_output}")
        except Exception:
            pass
            
        raise RuntimeError(error_msg) from e

# -----------------------------------------------------------------------------
#  Base vs Meta comparison helper
# -----------------------------------------------------------------------------

def base_vs_meta(
    dataset_path: str | Path,
    run_dir: str | Path,
    *,
    threshold: float = 0.0,
    sample: int | None = 100_000,
) -> Dict[str, float]:
    """Compare SpliceAI raw arg-max predictions (base) with meta-model predictions.

    Saves a JSON summary (compare_base_meta.json) in *run_dir* and returns a
    metrics dict containing accuracy and macro-F1 for both models.
    """
    run_dir = Path(run_dir)
    predict_fn, feature_names = _load_model_generic(run_dir)

    required_cols = [
        "donor_score",
        "acceptor_score",
        "neither_score",
        "splice_type",
    ] + feature_names
    # Deduplicate while preserving order to avoid Polars DuplicateError
    required_cols = list(dict.fromkeys(required_cols))

    # Load required columns lazily via Polars for efficiency
    if Path(dataset_path).is_dir():
        lf = pl.scan_parquet(str(Path(dataset_path) / "*.parquet"), missing_columns="insert")
    else:
        lf = pl.scan_parquet(str(dataset_path), missing_columns="insert")
    
    # Use helper function for robust column selection
    lf, missing_cols, existing_cols = select_available_columns(
        lf, required_cols, context_name="Base vs Meta"
    )
    if sample is not None:
        lf = _lazyframe_sample_gene_aware(lf, sample, seed=42)
    df = lf.collect(streaming=True).to_pandas()

    # Filter to recognised labels only
    df = df.dropna(subset=["splice_type"])
    df = df[df["splice_type"].isin(_LABEL_MAP_STR.keys())]

    # Base predictions – argmax over the three raw SpliceAI scores
    base_scores = df[["neither_score", "donor_score", "acceptor_score"]].to_numpy(float)
    base_pred = base_scores.argmax(axis=1)
    if threshold > 0.0:
        max_score = base_scores.max(axis=1)
        base_pred[max_score < threshold] = -1  # treat below-threshold as incorrect

    # Meta predictions
    # Preprocess features using the standardized helper function
    df = _preprocess_features_for_model(df, feature_names)
    
    # Convert to numpy array for prediction
    X = df[feature_names].to_numpy(np.float32)
    meta_proba = predict_fn(X)
    meta_pred = meta_proba.argmax(axis=1)



    y = _encode_labels(df["splice_type"].to_numpy())

    acc_canon = accuracy_score(y, meta_pred)
    acc_swap = accuracy_score(y, swap_0_2(meta_pred))
    if acc_swap > acc_canon:
        warnings.warn(
            "meta predictions appear to follow legacy numeric order (0=donor,2=neither); remapping to canonical.",
            RuntimeWarning,
        )
        meta_pred = swap_0_2(meta_pred)
        acc_canon = acc_swap  # for later stats

    # Extra per-splice-site stats
    splice_mask = np.isin(y, [1, 2])  # donor or acceptor
    # Show mapping for verification
    mapping_str = ", ".join(f"{k}:{v}" for k, v in _LABEL_MAP_STR.items())
    print(f"[base_vs_meta] Canonical label map ⇒ {mapping_str}")

    stats: Dict[str, float] = {
        "base_accuracy": float(accuracy_score(y, base_pred)),
        "meta_accuracy": float(accuracy_score(y, meta_pred)),
        "base_macro_f1": float(f1_score(y, base_pred, average="macro")),
        "meta_macro_f1": float(f1_score(y, meta_pred, average="macro")),
        "base_splice_accuracy": float(accuracy_score(y[splice_mask], base_pred[splice_mask])) if splice_mask.any() else float("nan"),
        "meta_splice_accuracy": float(accuracy_score(y[splice_mask], meta_pred[splice_mask])) if splice_mask.any() else float("nan"),
        "base_fp": int(((base_pred != 0) & (y == 0)).sum()),
        "meta_fp": int(((meta_pred != 0) & (y == 0)).sum()),
        "base_fn": int(((base_pred == 0) & (y != 0)).sum()),
        "meta_fn": int(((meta_pred == 0) & (y != 0)).sum()),
    }
    (run_dir / "compare_base_meta.json").write_text(json.dumps(stats, indent=2))
    return stats

# -----------------------------------------------------------------------------
#  Compare splice performance TSVs helper (meta vs base)
# -----------------------------------------------------------------------------

def _generate_base_tsv_from_raw_scores(
    dataset_path: str | Path,
    run_dir: str | Path,
    annotations_path: str | Path | None = None,
    out_tsv: str | Path | None = None,
    sample: int | None = None,
    verbose: int = 1,
) -> Path:
    """Generate base model performance TSV using raw SpliceAI scores.
    
    This function creates a performance evaluation using only the raw donor_score,
    acceptor_score, and neither_score columns to evaluate the base SpliceAI model.
    
    Parameters
    ----------
    dataset_path : str | Path
        Path to the dataset containing raw SpliceAI scores
    run_dir : str | Path
        Run directory for output
    annotations_path : str | Path | None
        Optional annotations file
    out_tsv : str | Path | None
        Output TSV path (defaults to run_dir/full_splice_performance.tsv)
    sample : int | None
        Optional sample size for faster computation
    verbose : int
        Verbosity level
        
    Returns
    -------
    Path
        Path to the generated base performance TSV
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    run_dir = Path(run_dir)
    dataset_path = Path(dataset_path)
    
    if out_tsv is None:
        out_tsv = run_dir / "full_splice_performance.tsv"
    else:
        out_tsv = Path(out_tsv)
    
    if verbose >= 1:
        print(f"[_generate_base_tsv] Generating base performance from raw scores...")
        print(f"[_generate_base_tsv] Dataset: {dataset_path}")
        print(f"[_generate_base_tsv] Output: {out_tsv}")
    
    try:
        # Load dataset
        if dataset_path.is_dir():
            # Handle directory of parquet files
            import polars as pl
            lf = pl.scan_parquet(str(dataset_path / "*.parquet"))
            if sample is not None:
                lf = _lazyframe_sample_gene_aware(lf, sample, seed=42)
            df = lf.collect(streaming=True).to_pandas()
        else:
            # Single parquet file
            df = pd.read_parquet(dataset_path)
            if sample is not None and len(df) > sample:
                df = df.sample(n=sample, random_state=42)
        
        if verbose >= 1:
            print(f"[_generate_base_tsv] Loaded {len(df)} rows")
        
        # Check required columns
        required_cols = ["gene_id", "splice_type", "donor_score", "acceptor_score"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")
        
        # Add neither_score if missing
        if "neither_score" not in df.columns:
            df["neither_score"] = 1.0 - df["donor_score"] - df["acceptor_score"]
        
        # Generate base predictions using argmax
        raw_scores = df[["neither_score", "donor_score", "acceptor_score"]].values
        base_pred = raw_scores.argmax(axis=1)
        
        # Convert to splice type predictions
        pred_map = {0: "neither", 1: "donor", 2: "acceptor"}
        df["base_pred"] = [pred_map[p] for p in base_pred]
        
        # Compute per-gene, per-site-type metrics
        records = []
        for gene_id, gdf in df.groupby("gene_id"):
            for stype in ("donor", "acceptor"):
                truth_mask = gdf["splice_type"] == stype
                pred_mask = gdf["base_pred"] == stype
                
                tp = int((pred_mask & truth_mask).sum())
                fp = int((pred_mask & ~truth_mask).sum())
                fn = int((~pred_mask & truth_mask).sum())
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                
                records.append({
                    "gene_id": gene_id,
                    "site_type": stype,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "precision": round(prec, 3),
                    "recall": round(rec, 3),
                    "f1_score": round(f1, 3),
                })
        
        # Create output DataFrame
        result_df = pd.DataFrame(records)
        
        # Save to TSV
        result_df.to_csv(out_tsv, sep="\t", index=False)
        
        if verbose >= 1:
            print(f"[_generate_base_tsv] Generated {len(result_df)} performance records")
            print(f"[_generate_base_tsv] Saved to: {out_tsv}")
        
        return out_tsv
        
    except Exception as e:
        if verbose >= 1:
            print(f"[_generate_base_tsv] Failed to generate base TSV: {e}")
        raise


def _generate_missing_performance_files(
    dataset_path: str | Path,
    run_dir: str | Path,
    annotations_path: str | Path | None = None,
    verbose: int = 1,
) -> tuple[bool, bool]:
    """Generate missing performance TSV files if possible.
    
    Returns
    -------
    tuple[bool, bool]
        (meta_tsv_exists, base_tsv_exists) after generation attempt
    """
    run_dir = Path(run_dir)
    meta_tsv = run_dir / "full_splice_performance_meta.tsv"
    base_tsv = run_dir / "full_splice_performance.tsv"
    
    meta_exists = meta_tsv.exists()
    base_exists = base_tsv.exists()
    
    # Try to generate meta TSV if missing
    if not meta_exists:
        if verbose >= 1:
            print("[DEBUG] Attempting to generate missing meta performance TSV")
        try:
            meta_splice_performance(
                dataset_path=dataset_path,
                run_dir=run_dir,
                annotations_path=annotations_path,
                verbose=max(0, verbose - 1),
                sample=15000,  # Use smaller sample to avoid issues
            )
            meta_exists = meta_tsv.exists()
            if verbose >= 1:
                print(f"[DEBUG] Meta TSV generation: {'successful' if meta_exists else 'failed'}")
        except Exception as e:
            if verbose >= 1:
                print(f"[DEBUG] Meta TSV generation failed: {e}")
    
    # Try to generate base TSV if missing (basic approach using raw scores)
    if not base_exists and meta_exists:
        if verbose >= 1:
            print("[DEBUG] Attempting to generate missing base performance TSV from raw scores")
        try:
            # Use the base TSV generation capability from the comprehensive evaluation function
            # This will use raw SpliceAI scores to generate base performance metrics
            base_performance_file = _generate_base_tsv_from_raw_scores(
                dataset_path=dataset_path,
                run_dir=run_dir,
                annotations_path=annotations_path,
                out_tsv=base_tsv,
                sample=15000,
                verbose=max(0, verbose - 1),
            )
            base_exists = base_tsv.exists()
            if verbose >= 1:
                print(f"[DEBUG] Base TSV generation: {'successful' if base_exists else 'failed'}")
        except Exception as e:
            if verbose >= 1:
                print(f"[DEBUG] Base TSV generation failed: {e}")
    
    return meta_exists, base_exists


def enhanced_compare_splice_performance(
    meta_tsv: str | Path | None = None,
    base_tsv: str | Path | None = None,
    run_dir: str | Path | None = None,
    dataset_path: str | Path | None = None,
    annotations_path: str | Path | None = None,
    out_tsv: str | Path | None = None,
    verbose: int = 1,
) -> bool:
    """Enhanced wrapper for compare_splice_performance with fallback mechanisms.
    
    Parameters
    ----------
    meta_tsv, base_tsv : str | Path | None
        Paths to performance TSV files. If None and run_dir provided, will look for
        default names in run_dir.
    run_dir : str | Path | None
        Run directory containing model artifacts and performance files.
    dataset_path : str | Path | None
        Dataset path for generating missing TSV files.
    annotations_path : str | Path | None
        Annotations path for generating missing TSV files.
    out_tsv : str | Path | None
        Output path for comparison results.
    verbose : int
        Verbosity level.
        
    Returns
    -------
    bool
        True if comparison was successful, False otherwise.
    """
    
    if run_dir is not None:
        run_dir = Path(run_dir)
        
        # Auto-detect TSV paths if not provided
        if meta_tsv is None:
            meta_tsv = run_dir / "full_splice_performance_meta.tsv"
        if base_tsv is None:
            base_tsv = run_dir / "full_splice_performance.tsv"
        if out_tsv is None:
            out_tsv = run_dir / "perf_meta_vs_base.tsv"
    
    # Check if files exist
    meta_exists = meta_tsv is not None and Path(meta_tsv).exists()
    base_exists = base_tsv is not None and Path(base_tsv).exists()
    
    if verbose >= 1:
        print(f"[DEBUG] Meta TSV exists: {meta_exists}, path: {meta_tsv}")
        print(f"[DEBUG] Base TSV exists: {base_exists}, path: {base_tsv}")
    
    # Try to generate missing files if we have the necessary information
    if (not meta_exists or not base_exists) and dataset_path is not None and run_dir is not None:
        if verbose >= 1:
            print("[DEBUG] Attempting to generate missing performance comparison files")
        
        try:
            meta_exists, base_exists = _generate_missing_performance_files(
                dataset_path=dataset_path,
                run_dir=run_dir,
                annotations_path=annotations_path,
                verbose=verbose,
            )
        except Exception as e:
            if verbose >= 1:
                print(f"[DEBUG] Failed to generate missing files: {e}")
    
    if verbose >= 1:
        print(f"[DEBUG] After generation attempt: Meta TSV exists: {meta_exists}, Base TSV exists: {base_exists}")
    
    # Proceed with comparison if we have the files
    if meta_exists and base_exists:
        try:
            from .compare_splice_performance import compare_splice_performance
            
            compare_splice_performance(
                meta_tsv=meta_tsv,
                base_tsv=base_tsv,
                out_tsv=out_tsv,
                verbose=verbose,
            )
            
            if verbose >= 1:
                print(f"[compare_splice_performance] Successfully generated comparison: {out_tsv}")
            return True
            
        except Exception as e:
            if verbose >= 1:
                print(f"[compare_splice_performance] Comparison failed: {e}")
                if verbose >= 2:
                    import traceback
                    traceback.print_exc()
            return False
    else:
        if verbose >= 1:
            print(f"[compare_splice_performance] Cannot proceed - missing TSV files")
            print(f"    Meta TSV: {meta_tsv} (exists: {meta_exists})")
            print(f"    Base TSV: {base_tsv} (exists: {base_exists})")
        return False


from .compare_splice_performance import compare_splice_performance  # noqa: E402  -- keep import near bottom to avoid circular deps

# Collect previously defined exported symbols (may be empty if not yet declared)
_prev_all = globals().get("__all__", [])

# Explicitly enumerate the public helpers exported by this module.
__all__ = list(dict.fromkeys(
    _prev_all + [
        # Core diagnostics
        "meta_splice_performance",
        "meta_splice_performance_correct",
        "meta_splice_performance_argmax",
        "probability_diagnostics",
        "richer_metrics",
        "leakage_probe",
        "gene_score_delta",
        "shap_importance",
        "base_vs_meta",
        "neighbour_window_diagnostics",
        # Comparison helpers
        "compare_splice_performance",
        "enhanced_compare_splice_performance",
        "_generate_missing_performance_files",
    ]
))

# -----------------------------------------------------------------------------
#  Neighbour-window diagnostic wrapper
# -----------------------------------------------------------------------------

def neighbour_window_diagnostics(
    dataset_path: str | Path,
    run_dir: str | Path,
    *,
    annotations_path: str | Path | None = None,
    genes: list[str] | None = None,
    n_sample: int | None = 5,
    window: int = 10,
) -> None:
    """Run neighbour-window diagnostic (optional, may be slow).

    Parameters
    ----------
    dataset_path
        Path to the training dataset (passed through so callers don't need to
        worry – can be *None* to use default MetaModelDataHandler location).
    run_dir
        Directory that contains the fold's trained model artefacts.
    annotations_path
        Truth splice-site annotation file (TSV/CSV/Parquet). If None, will try
        to find default splice sites file.
    genes, n_sample, window
        Selection parameters forwarded to the diagnostic script.  If *genes* is
        provided, it overrides *n_sample*.
    """
    from meta_spliceai.splice_engine.meta_models.analysis import neighbour_window_diagnostics as _nwd
    
    # Handle missing annotations_path
    if annotations_path is None:
        # Try to find splice sites file in common locations
        potential_paths = [
            Path("data/ensembl/splice_sites.tsv"),
            Path("data/ensembl/splice_sites.parquet"),
            Path("data/ensembl/spliceai_eval/splice_sites.tsv"),
            Path("data/ensembl/spliceai_eval/splice_sites.parquet"),
        ]
        
        for path in potential_paths:
            if path.exists():
                annotations_path = path
                break
        
        if annotations_path is None:
            print("[neigh_diag] Skipped (no splice sites annotations found)")
            return

    try:
        argv = [
            "--run-dir",
            str(run_dir),
            "--annotations",
            str(annotations_path),
            "--window",
            str(window),
            "--plot",  # enable PNG plots when called from training pipeline
        ]
        if dataset_path is not None:
            argv += ["--dataset", str(dataset_path)]

        if genes is not None:
            argv += ["--genes", ",".join(genes)]
        elif n_sample is not None and n_sample > 0:
            argv += ["--n-sample", str(n_sample)]
        else:
            print("[neigh_diag] Skipped (n_sample=0 and no genes provided)")
            return

        print("[neigh_diag] Running neighbour-window diagnostic…")
        _nwd.main(argv)
        print("[neigh_diag] Done.")
        
    except Exception as e:
        # Enhanced error handling for common issues
        error_msg = str(e)
        
        if "schema names differ" in error_msg:
            print(f"[neigh_diag] WARNING: Schema mismatch detected in parquet files")
            print(f"[neigh_diag] This often indicates corrupted or incompatible parquet files")
            print(f"[neigh_diag] Dataset path: {dataset_path}")
            print(f"[neigh_diag] Annotations path: {annotations_path}")
            print(f"[neigh_diag] Try regenerating the feature dataset or using TSV format")
        elif "columns" in error_msg.lower() and "not found" in error_msg.lower():
            print(f"[neigh_diag] WARNING: Missing required columns in dataset")
            print(f"[neigh_diag] This may indicate an incompatible dataset format")
        else:
            print(f"[neigh_diag] WARNING: Neighbour window diagnostic failed: {e}")
            
        # Don't re-raise the exception - this is optional diagnostic
