"""Model explanation utilities for meta-model training.

Currently provides SHAP-based feature importance for tree models (e.g.
XGBoost, LightGBM) and a permutation-based fallback for any scikit-learn
estimator implementing ``predict``.

Example
-------
>>> from meta_spliceai.splice_engine.meta_models.training import explainers
>>> imp_df = explainers.shap_feature_importance(model, X_test, feature_names)
>>> imp_df.head()
"""
from __future__ import annotations

from typing import Sequence, Any

import numpy as np
import pandas as pd

# Third-party libraries are imported lazily to keep import cost low

__all__ = [
    "shap_feature_importance",
    "permutation_feature_importance",
]


def _maybe_import_shap():  # pragma: no cover
    try:
        import shap  # type: ignore

        return shap
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'shap' package is required for SHAP explanations. Install via 'pip install shap'."
        ) from exc


def shap_feature_importance(
    model: Any,
    X: np.ndarray,
    feature_names: Sequence[str] | None = None,
    *,
    background_size: int = 100,
) -> pd.DataFrame:
    """Return mean(|SHAP|) feature importance for *model* evaluated on *X*.

    Parameters
    ----------
    model
        Fitted estimator. Tree-based models (e.g. XGBoost, LightGBM, scikit-learn
        GradientBoosting) are supported.
    X
        2-D feature matrix. Should be a **NumPy** array; pass ``X.values`` if you
        have a pandas/Polars frame.
    feature_names
        Optional list of feature names. If ``None`` defaults to ``[f0, f1, …]``.
    background_size
        Number of rows randomly sampled from *X* for the background dataset used
        by the TreeExplainer. Lower values speed up computation and reduce
        memory overhead.

    Returns
    -------
    pandas.DataFrame
        Columns: ``feature``, ``importance``. Sorted descending by importance.
    """

    shap = _maybe_import_shap()

    if X.ndim != 2:
        raise ValueError("X must be a 2-D matrix")

    rng = np.random.default_rng(42)
    bg_idx = rng.choice(X.shape[0], size=min(background_size, X.shape[0]), replace=False)
    background = X[bg_idx]

    # TreeExplainer supports most gradient-boosting & tree ensembles
    explainer = shap.TreeExplainer(model, data=background)
    shap_values = explainer.shap_values(X, check_additivity=False)

    # TreeExplainer returns list for multiclass; collapse with mean(|v|)
    if isinstance(shap_values, list):
        # Typical format for multiclass: list length = n_classes, each (n_samples, n_features)
        shap_values_arr = np.stack([np.abs(v).mean(axis=0) for v in shap_values]).mean(axis=0)
    else:
        # XGBoost >= 2.0 may return 3-D array: (n_samples, n_features, n_classes)
        if shap_values.ndim == 3:  # collapse samples first, then classes
            shap_values_arr = np.abs(shap_values).mean(axis=0).mean(axis=-1)
        else:
            shap_values_arr = np.abs(shap_values).mean(axis=0)

    # Ensure feature_names aligns with SHAP vector length
    if feature_names is None or len(feature_names) != len(shap_values_arr):
        feature_names = [f"f{i}" for i in range(len(shap_values_arr))]

    imp_df = pd.DataFrame({"feature": feature_names, "importance": shap_values_arr})
    imp_df.sort_values("importance", ascending=False, inplace=True)
    imp_df.reset_index(drop=True, inplace=True)
    return imp_df


def permutation_feature_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str] | None = None,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Permutation importance fallback when SHAP is unavailable or unsuitable."""

    from sklearn.inspection import permutation_importance  # type: ignore

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    imp_df = pd.DataFrame({"feature": feature_names, "importance": result.importances_mean})
    imp_df.sort_values("importance", ascending=False, inplace=True)
    imp_df.reset_index(drop=True, inplace=True)
    return imp_df
