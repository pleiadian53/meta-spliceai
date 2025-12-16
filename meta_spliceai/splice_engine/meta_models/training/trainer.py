"""High-level training orchestration for meta-models.

Typical usage
-------------
>>> from meta_spliceai.splice_engine.meta_models.training import trainer as tr
>>> T = tr.Trainer(model_spec="xgboost", out_dir="models/xgb_v1")
>>> T.fit("/data/train_pc_1000/master")
>>> T.save()

Responsibilities
----------------
1. Load and split the dataset via ``datasets`` helpers.
2. Instantiate the estimator via ``models.get_model``.
3. Train → validate → test, collecting core metrics.
4. Persist artefacts: model pickle, metrics JSON, optional feature importance.

The class is deliberately *simple*; more advanced workflows (nested CV,
hyper-parameter search, ensemble stacking, …) can be layered on top without
changing this core.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import (  # pylint: disable=import-error
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.utils import Bunch  # type: ignore

from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.training import models as model_registry
from meta_spliceai.splice_engine.meta_models.builder import preprocessing

__all__ = ["Trainer"]


class Trainer:  # pylint: disable=too-many-instance-attributes
    """Encapsulate a single training run."""

    def __init__(
        self,
        model_spec: str | dict[str, Any] | None = None,
        *,
        out_dir: str | Path = "trained_model",
        label_col: str = "splice_type",
        group_col: str = "gene_id",
        test_size: float = 0.2,
        valid_size: float | None = None,
        random_state: int = 42,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.label_col = label_col
        self.test_size = test_size
        self.valid_size = valid_size
        self.group_col = group_col
        self.random_state = random_state

        self.model = model_registry.get_model(model_spec)
        self.metrics: Dict[str, float] = {}
        self.splits: Dict[str, Sequence[int]] | None = None
        self.feature_names_: Sequence[str] | None = None
        self.feature_importance_df = None  # type: ignore
        # Keep test data & groups for downstream analysis
        self._X_test: np.ndarray | None = None
        self._y_test: np.ndarray | None = None
        self._g_test: np.ndarray | None = None
        self.gene_metrics_df = None  # type: ignore

    # ------------------------------------------------------------------
    # Core workflow ----------------------------------------------------
    # ------------------------------------------------------------------

    def fit(self, dataset_path: str | Path, *, columns: Sequence[str] | None = None) -> "Trainer":
        """Load dataset, split, and fit the estimator."""
        df = datasets.load_dataset(dataset_path, columns=columns)

        # Run preprocessing pipeline; keep pandas DataFrame to preserve column names
        X_df, y_series = preprocessing.prepare_training_data(
            df,
            label_col=self.label_col,
            return_type="pandas",
        )
        self.feature_names_ = list(X_df.columns)

        # Convert to NumPy for model training
        X = X_df.values
        y = (y_series.values > 0).astype(int)
        groups = df[self.group_col].to_pandas().values if self.group_col in df.columns else None

        # Request group-aware splits and retrieve group vectors for metrics
        if groups is not None:
            (X_train, X_valid, X_test, y_train, y_valid, y_test, g_train, g_valid, g_test) = datasets.train_valid_test_split(
                X,
                y,
                test_size=self.test_size,
                valid_size=self.valid_size,
                random_state=self.random_state,
                groups=groups,
                return_groups=True,
            )
        else:
            (X_train, X_valid, X_test, y_train, y_valid, y_test) = datasets.train_valid_test_split(
                X,
                y,
                test_size=self.test_size,
                valid_size=self.valid_size,
                random_state=self.random_state,
            )

        # Save test set for later benchmarking
        self._X_test, self._y_test = X_test, y_test
        if groups is not None:
            self._g_test = g_test
        else:
            self._g_test = None

        self.model.fit(X_train, y_train)

        self.metrics = self._eval_block(X_test, y_test, prefix="test")
        if X_valid is not None:
            self.metrics.update(self._eval_block(X_valid, y_valid, prefix="valid"))

        # Gene-wise metrics
        if self._g_test is not None:
            self.gene_metrics_df = _gene_level_metrics(y_test, self.model.predict(X_test), self._g_test)
            self.metrics["gene_avg_f1"] = float(self.gene_metrics_df["f1_score"].mean())
        return self

    # ------------------------------------------------------------------
    # Evaluation -------------------------------------------------------
    # ------------------------------------------------------------------

    def _eval_block(self, X, y, *, prefix: str) -> Dict[str, float]:
        probas = self.model.predict_proba(X)[:, 1]
        preds = (probas >= 0.5).astype(int)
        return {
            f"{prefix}_auc_roc": _safe_metric(roc_auc_score, y, probas),
            f"{prefix}_auc_pr": _safe_metric(average_precision_score, y, probas),
            f"{prefix}_accuracy": _safe_metric(accuracy_score, y, preds),
        }

    def evaluate(self, X, y) -> Dict[str, float]:  # noqa: D401
        """Return core metrics on (X, y) without updating internal state."""
        return self._eval_block(X, y, prefix="eval")

    # ------------------------------------------------------------------
    # Persistence ------------------------------------------------------
    # ------------------------------------------------------------------

    def save(self) -> None:  # noqa: D401
        """Persist model, metrics, and training metadata to *out_dir*."""
        # model
        with open(self.out_dir / "model.pkl", "wb") as fh:
            pickle.dump(self.model, fh)

        # metrics
        with open(self.out_dir / "metrics.json", "w") as fh:
            json.dump(self.metrics, fh, indent=2)

        # minimal meta
        meta = dict(
            label_col=self.label_col,
            test_size=self.test_size,
            valid_size=self.valid_size,
            random_state=self.random_state,
            model_spec=getattr(self.model, "__class__", object).__name__,
        )
        with open(self.out_dir / "run_meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)

    # ------------------------------------------------------------------
    # Convenience & analysis ------------------------------------------
    # ------------------------------------------------------------------

    def feature_importance(
        self,
        *,
        max_samples: int = 1000,
        background_samples: int = 100,
    ) -> "pd.DataFrame":
        """Return SHAP mean(|value|) feature importance on a test subset.

        The result is cached in ``self.feature_importance_df``.
        """

        if self._X_test is None:
            raise RuntimeError("Trainer.fit() must be called before feature_importance().")

        import numpy as np
        from meta_spliceai.splice_engine.meta_models.training import explainers

        X = self._X_test
        if max_samples and X.shape[0] > max_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(X.shape[0], size=max_samples, replace=False)
            X_sample = X[idx]
        else:
            X_sample = X

        # Use feature names only if they match X dimension
        feat_names = (
            self.feature_names_ if self.feature_names_ and len(self.feature_names_) == X_sample.shape[1] else None
        )
        imp_df = explainers.shap_feature_importance(
            self.model,
            X_sample,
            feature_names=feat_names,
            background_size=background_samples,
        )
        self.feature_importance_df = imp_df
        return imp_df

    def to_sklearn_bunch(self) -> Bunch:
        """Return a *Bunch* containing model, metrics, and miscellaneous info."""
        return Bunch(
            model=self.model,
            metrics=self.metrics,
            feature_names=self.feature_names_,
            gene_metrics=self.gene_metrics_df,
        )

    # ------------------------------------------------------------------
    # Baseline comparison ------------------------------------------------
    # ------------------------------------------------------------------

    def compare_baseline(self, baseline_tsv: str | Path, *, key: str = "f1_score") -> Any:  # noqa: D401
        """Return a DataFrame with baseline vs meta-model gene-level metric and Δ."""
        import polars as pl  # local import to keep dependency optional

        if self.gene_metrics_df is None:
            raise RuntimeError("fit() must be called before baseline comparison.")

        # Polars <1.2 used 'sep', newer versions use 'separator'
        try:
            base_df = pl.read_csv(
                baseline_tsv,
                separator="\t",
                infer_schema_length=10000,
                schema_overrides={"chrom": pl.Utf8},
            )
        except Exception:
            # Fallback – use pandas which has more permissive parser, then convert
            import pandas as pd
            base_df = pl.from_pandas(pd.read_csv(baseline_tsv, sep="\t"))
        meta_df = pl.from_pandas(self.gene_metrics_df)

        joined = base_df.join(meta_df, on="gene_id", how="inner", suffix="_meta")
        if key not in joined.columns or f"{key}_meta" not in joined.columns:
            raise KeyError(f"Metric '{key}' missing from either baseline or meta data.")

        joined = joined.with_columns(
            (pl.col(f"{key}_meta") - pl.col(key)).alias("delta")
        )
        return joined.to_pandas()


# ---------------------------------------------------------------------------
#  Helpers ------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _gene_level_metrics(y_true, y_pred, gene_ids):
    """Return a pandas DataFrame with per-gene confusion counts and F1."""
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score

    # Binary reduction: treat class > 0 as positive (splice site) for simplicity
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)

    data = []
    for gid in np.unique(gene_ids):
        mask = gene_ids == gid
        if mask.sum() == 0:
            continue
        yt = y_true_bin[mask]
        yp = y_pred_bin[mask]
        TP = int(((yt == 1) & (yp == 1)).sum())
        FP = int(((yt == 0) & (yp == 1)).sum())
        FN = int(((yt == 1) & (yp == 0)).sum())
        TN = int(((yt == 0) & (yp == 0)).sum())
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        data.append(dict(gene_id=gid, TP=TP, FP=FP, FN=FN, TN=TN, precision=prec, recall=rec, f1_score=f1))

    return pd.DataFrame(data)


def _safe_metric(func, y_true, y_pred):
    try:
        return float(func(y_true, y_pred))
    except Exception:  # pragma: no cover – rare edge cases (e.g., single-class)
        return float("nan")
