"""Incremental / online-learning utilities for very large training datasets.

This module complements the classic `Trainer` class by supporting models that
implement scikit-learn's ``partial_fit`` API.  The core idea is to *stream* the
Parquet training dataset in manageable record batches so that only a fraction
of the data is held in memory at any one time.

Typical usage
-------------
>>> from meta_spliceai.splice_engine.meta_models.training.incremental import IncrementalTrainer
>>> trainer = IncrementalTrainer(
...     model_spec="sgd_logistic",        # online logistic regression
...     batch_size=250_000,              # ≈ memory of ~2.5×10^5 rows
...     out_dir="models/sgd_full"        # where to write the model & metrics
... )
>>> trainer.fit("train_pc_20000/master").save()

Design notes
~~~~~~~~~~~~
* Only *classification* is supported for now.
* ``partial_fit`` models must be compatible with sparse SciPy matrices – we
  convert Pandas/Polars frames via ``.to_numpy()`` to keep dependencies light.
* Evaluation is performed on an *optional* validation Parquet directory (small
  subset) because scanning the entire training dataset again would be slow.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence
import json
import pickle

import numpy as np
import pyarrow.dataset as ds
from sklearn.linear_model import SGDClassifier  # pylint: disable=import-error

# Optional heavyweights --------------------------------------------------
try:
    import lightgbm as lgb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    lgb = None  # noqa: N816

try:
    from xgboost.dask import DaskXGBClassifier  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    DaskXGBClassifier = None
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

__all__ = ["get_online_model", "IncrementalTrainer"]

# ---------------------------------------------------------------------------
# Model factory ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def get_online_model(name: str | Dict[str, Any] | None = None):
    """Return an *incremental* estimator that supports ``partial_fit``.

    * ``sgd_logistic``  - SGDClassifier(loss="log_loss")
    * ``sgd_hinge``     - linear SVM (hinge)
    * ``sgd_pa``        - passive-aggressive classifier

    A ``dict`` spec is also accepted, allowing overrides::

        {"name": "sgd_logistic", "alpha": 1e-4, "penalty": "l1"}
    """
    if name is None:
        name = "sgd_logistic"

    if isinstance(name, dict):
        model_name = name.get("name", "sgd_logistic")
        kwargs = {k: v for k, v in name.items() if k != "name"}
    else:
        model_name = name
        kwargs = {}

    model_name = model_name.lower()
    if model_name == "sgd_logistic":
        return SGDClassifier(loss="log_loss", **kwargs)
    if model_name == "sgd_hinge":
        return SGDClassifier(loss="hinge", **kwargs)
    if model_name in ("sgd_pa", "passive_aggressive"):
        return SGDClassifier(loss="hinge", alpha=0.0001, **kwargs)  # PA approx.
    if model_name in ("lightgbm", "lgbm"):
        if lgb is None:
            raise ImportError("lightgbm is not installed – pip install lightgbm")
        return lgb.LGBMClassifier(**kwargs)
    if model_name in ("xgboost_dask", "dask_xgb"):
        if DaskXGBClassifier is None:
            raise ImportError("dask-xgboost is not available – pip install xgboost[dask]")
        return DaskXGBClassifier(**kwargs)

    raise ValueError(f"Unknown online model '{model_name}'")


# ---------------------------------------------------------------------------
# Utility – iterate Parquet rows in mini-batches --------------------------------
# ---------------------------------------------------------------------------

def _iter_batches(parquet_dir: str | Path, *, columns: Sequence[str], batch_size: int) -> Iterator[np.ndarray]:
    """Yield dictionary {col: ndarray} in *row* chunks from a Parquet directory."""
    dataset = ds.dataset(parquet_dir)
    scanner = dataset.scanner(columns=list(columns), batch_size=batch_size)
    for record_batch in scanner.to_batches():
        # Convert only required columns to numpy to minimise memory copy
        arrays = record_batch.columns
        yield {col: arr.to_numpy() for col, arr in zip(record_batch.schema.names, arrays)}


# ---------------------------------------------------------------------------
# Incremental trainer ---------------------------------------------------------
# ---------------------------------------------------------------------------

class IncrementalTrainer:
    """Trainer that works with *very large* datasets via ``partial_fit``.

    Parameters
    ----------
    model_spec
        String or dict understood by :func:`get_online_model`.
    batch_size
        Number of rows per mini-batch; tune according to available RAM.
    target_col
        Name of the target label column.
    out_dir
        Where to write ``model.pkl`` & ``metrics.json``.
    """

    def __init__(
        self,
        *,
        model_spec: str | Dict[str, Any] | None = None,
        batch_size: int = 250_000,
        target_col: str = "label",
        out_dir: str | Path = "models/online",
        classes: Sequence[int] | None = (0, 1),
    ) -> None:
        self.model = get_online_model(model_spec)
        self.batch_size = batch_size
        self.target_col = target_col
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.classes_ = np.asarray(classes)
        self.metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Core fit ----------------------------------------------------------
    # ------------------------------------------------------------------

    def fit(self, parquet_dir: str | Path, *, validation_dir: str | Path | None = None) -> "IncrementalTrainer":
        """Stream training data from *parquet_dir* and update the model."""
        cols = [self.target_col]
        first_pass = True
        for batch in _iter_batches(parquet_dir, columns=cols + ["features"], batch_size=self.batch_size):
            X = np.vstack(batch["features"]).astype(np.float32)
            y = batch[self.target_col].astype(int)
            if first_pass:
                self.model.partial_fit(X, y, classes=self.classes_)
                first_pass = False
            else:
                self.model.partial_fit(X, y)
        # ---------------- evaluation ----------------------------------
        if validation_dir:
            self.metrics = self._evaluate(validation_dir)
        return self

    # ------------------------------------------------------------------
    def _evaluate(self, parquet_dir: str | Path) -> Dict[str, float]:
        cols = [self.target_col]
        y_true_all: List[int] = []
        y_prob_all: List[float] = []
        for batch in _iter_batches(parquet_dir, columns=cols + ["features"], batch_size=self.batch_size):
            X = np.vstack(batch["features"]).astype(np.float32)
            y_true = batch[self.target_col].astype(int)
            prob = self.model.predict_proba(X)[:, 1]
            y_true_all.append(y_true)
            y_prob_all.append(prob)
        y_true_concat = np.concatenate(y_true_all)
        y_prob_concat = np.concatenate(y_prob_all)
        return {
            "roc_auc": float(roc_auc_score(y_true_concat, y_prob_concat)),
            "pr_auc": float(average_precision_score(y_true_concat, y_prob_concat)),
            "accuracy": float(accuracy_score(y_true_concat, y_prob_concat > 0.5)),
        }

    # ------------------------------------------------------------------
    def save(self) -> None:
        """Persist model and metrics to *out_dir*."""
        with open(self.out_dir / "model.pkl", "wb") as fh:
            pickle.dump(self.model, fh)
        with open(self.out_dir / "metrics.json", "w") as fh:
            json.dump(self.metrics, fh, indent=2)


# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import argparse, sys, json as _json

    ap = argparse.ArgumentParser(
        description="Incremental (online) trainer for huge Parquet datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("dataset_dir", help="Path to Parquet directory (training).")
    ap.add_argument("--val-dir", help="Optional Parquet dir for validation.")
    ap.add_argument("--model", default="sgd_logistic", help="Model spec string or JSON dict.")
    ap.add_argument("--batch-size", type=int, default=250_000, help="Row chunk size loaded into RAM.")
    ap.add_argument("--target-col", default="label", help="Target column name in Parquet.")
    ap.add_argument("--out-dir", default="models/online", help="Directory to write model & metrics.")
    args = ap.parse_args(sys.argv[1:])

    try:
        model_spec = _json.loads(args.model)
    except (ValueError, _json.JSONDecodeError):
        model_spec = args.model

    trainer = IncrementalTrainer(
        model_spec=model_spec,
        batch_size=args.batch_size,
        target_col=args.target_col,
        out_dir=args.out_dir,
    )
    trainer.fit(args.dataset_dir, validation_dir=args.val_dir).save()
    print(f"[incremental] Training done. Metrics: {json.dumps(trainer.metrics, indent=2)}")
