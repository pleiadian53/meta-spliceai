# Troubleshooting: "feature names must have the same length as the number of data columns" (External-Memory LibSVM)

*Last updated: 2025-06-21*

---

## ✂️  Symptom

Running the multi-class meta-model trainer in **external-memory** mode fails at
`xgb.DMatrix()` construction with a message similar to:

```
ValueError: ('feature names must have the same length as the number of data columns, ', 'expected 4165, got 4164')
```

where the second number is exactly **one less** than the first.

---

## 🔍  Root cause

`external_memory_utils.convert_dataset_to_libsvm()` converts each Parquet shard
to a **LibSVM** file using `sklearn.datasets.dump_svmlight_file()`.  Prior to
2025-06-21 the helper passed `zero_based=False`, producing **1-based** feature
indices (**1 … N**).

When you later create a `DMatrix` *and* supply `feature_names`, XGBoost assumes
the LibSVM file uses the **canonical 0-based** convention (**0 … N-1**).  The
parser therefore allocates one extra (all-zero) column, so the internal matrix
has `N + 1` columns while the list of names still has `N`, causing the length
mismatch error shown above.

---

## 🛠️  Fix

Commit `<hash>` (2025-06-21) changed the writer to 0-based indexing:

```diff
-        dump_svmlight_file(X, y, fh, zero_based=False)  # libsvm indexes from 1
+        dump_svmlight_file(X, y, fh, zero_based=True)   # XGBoost expects 0-based indices
```

No other code changes are required.

---

## ✅  Verification

1. Re-run dataset conversion **or** delete the existing `*.libsvm` file so it is
   regenerated with 0-based indices.
2. Execute the trainer again:

   ```bash
   conda run -n surveyor \
     python -m meta_spliceai.splice_engine.meta_models.training.demo_train_meta_model_multiclass \
       --dataset train_pc_1000/master \
       --out-dir runs/multiclass_full \
       --tree-method hist --max-bin 256 \
       --external-memory --early-stopping 250
   ```
3. `xgb.DMatrix()` should now build successfully, and training will proceed.

---

## 📚  References

* [scikit-learn `dump_svmlight_file` docs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.dump_svmlight_file.html)
* [XGBoost external-memory tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html)
* `training/docs/xgboost_dmatrix_guide.md` for a broader overview of `DMatrix` usage.

---

👉  **Takeaway:** when supplying `feature_names` to XGBoost, ensure your LibSVM
file uses **0-based** feature indices so the column count matches the name
list.
