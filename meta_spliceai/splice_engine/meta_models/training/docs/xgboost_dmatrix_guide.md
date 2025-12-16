# XGBoost `DMatrix` Primer

`DMatrix` is **XGBoost’s core data container**—an efficient, column-major
representation of the feature matrix plus optional metadata (labels, weights,
base margins, groups…).  All training and prediction APIs ultimately consume a
`DMatrix` (or its GPU cousin `DeviceQuantileDMatrix`).

---

## 1  Why not just pass `numpy.ndarray`?

* **Memory layout** XGBoost converts any Python object you pass to its own
  internal format to exploit cache locality and sparsity.  Building it once and
  re-using it avoids repeated conversions.
* **Metadata support** `labels`, `weights`, `base_margin`, `group` (for
  ranking) live *inside* the `DMatrix` and travel with the data.
* **Quantile sketch pre-computation** For histogram algorithms, cut points are
  computed during `DMatrix` construction, saving time in the first boosting
  iteration.
* **External-memory & GPU hooks** Specialised subclasses can stream data from
  disk (`libsvm_uri?format=libsvm`) or keep it on the GPU without host copies.

---

## 2  Creating a `DMatrix`

```python
import numpy as np
import xgboost as xgb

X = np.random.rand(1000, 200).astype(np.float32)
y = (X[:, 0] > 0.5).astype(int)

dtrain = xgb.DMatrix(X, label=y, missing=np.nan)
```

Other inputs:

| Source                              | Example                                                                                         |
|-------------------------------------|-------------------------------------------------------------------------------------------------|
| CSR/CSC sparse matrix               | `xgb.DMatrix(csr_matrix, label=y)`                                                             |
| pandas `DataFrame`                  | `xgb.DMatrix(df, label=df["target"])`                                                        |
| cuDF / cuPy (GPU)                   | `xgb.DMatrix(gpu_df, label=gpu_y)`                                                             |
| Parquet / Arrow dataset (via Arrow) | load to pandas/pyarrow first, then convert                                                     |
| Spark `DataFrame` (cluster)         | use `xgboost.spark.PySparkXGBClassifier` / `PySparkXGBRegressor`; XGBoost builds DMatrices partition-wise |
| **External-memory file**            | `xgb.DMatrix("train.libsvm?format=libsvm", missing=np.nan)`                                   |

*The URI suffix hints XGBoost to use external-memory mode: it memory-maps the
file and streams 4 MB blocks instead of loading everything into RAM.*

> **PySpark note** On a Spark cluster (including Microsoft Fabric) you should use the
> `xgboost.spark` package (`PySparkXGBClassifier` / `Regressor`).  It accepts a
> Spark `DataFrame` directly and internally converts each partition to a
> `DMatrix`, performing distributed training across executors.  The API mirrors
> scikit-learn but scales out via Spark’s shuffle instead of external-memory.
> See <https://xgboost.readthedocs.io/en/stable/tutorials/spark/index.html>.

---

## 3  Key Parameters

| Argument          | Purpose                                                                    |
|-------------------|----------------------------------------------------------------------------|
| `data`            | Features (2-D) or path/URI to external-memory dataset                      |
| `label`           | Target vector (1-D)                                                         |
| `missing`         | Value to treat as *NA*. Faster than `np.nan` comparisons at train‐time.     |
| `weight`          | Per-row weights                                                             |
| `base_margin`     | Initial prediction per row (used for continued training / warm start)       |
| `group`           | Query boundaries for ranking tasks                                          |

Advanced: `feature_names`, `feature_types`, `nthread`, …

---

## 4  `DMatrix` vs `DeviceQuantileDMatrix`

| Use case                    | Object                        | Notes                                           |
|-----------------------------|--------------------------------|-------------------------------------------------|
| CPU / external-memory       | `DMatrix`                      | Supports streaming LibSVM/CSV                   |
| GPU histogram (`gpu_hist`)  | `DeviceQuantileDMatrix`        | Keeps data in GPU memory; avoids host transfer  |

The API is identical; XGBoost picks an efficient backend based on the object
passed.

---

## 5  Typical Workflow

```python
import xgboost as xgb

dtrain = xgb.DMatrix("train.libsvm?format=libsvm", missing=np.nan)
params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "tree_method": "hist",  # or "gpu_hist"
}
model = xgb.train(params, dtrain, num_boost_round=600)
model.save_model("model.json")
```

---

## 6  Reading Back Feature Names

If you pass `feature_names` during construction, they are stored inside the
model and show up in `model.feature_names`.  This is helpful for SHAP or downstream
interpretation.

---

## 7  Further Reading

* XGBoost Python docs: <https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix>
* External-memory tutorial: <https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html>
* GPU data formats: <https://xgboost.readthedocs.io/en/stable/tutorials/gpu_acceleration.html>

*Last updated: 2025-06-20*
