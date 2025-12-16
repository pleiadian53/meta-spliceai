# Handling Out-of-Memory (OOM) During Meta-Model Training

This document captures the memory-related pitfalls we uncovered while
developing the **MetaSpliceAI** meta-model pipeline and the engineering
solutions that keep the workflow functional on moderate-RAM machines.

---

## 1 Feature-Pruning Stage

| Stage | Symptom | Root Cause | Fix |
| ----- | ------- | ---------- | --- |
| `_prune_dataset_shardwise` → `sink_parquet()` | Process killed (exit-code 137) after correlation analysis finished | `LazyFrame.sink_parquet()` buffers the entire DataFrame before writing | • Switched to manual chunked writer via `pyarrow.parquet.ParquetWriter`  
• `row_group_size`, `data_page_size`, `compression="zstd"` tuned to keep memory <1 GB  |
| Sampling for correlation | High RSS when sampling 200 k rows per shard | Wide (~4 k cols) dataset × large sample | New CLI flags  
`--pruning-sample-rows-per-shard` (default 25 k)  
`--pruning-max-corr-cols` (default 1 500) |

---

## 2 Dataset Loading

| Symptom | Root Cause | Fix |
| ------- | ---------- | --- |
| OOM inside `pl.scan_parquet(...).collect()` | Reading **all** columns inc. `sequence` strings | • Drop obviously unused columns at *scan* stage using `DEFAULT_DROP_COLUMNS`  
• `collect(streaming=True)` for pipeline-friendly execution |
| Inconsistent schemas → `ColumnNotFoundError` | Shards missing rare kmers | `missing_columns='insert'` passed to `pl.scan_parquet` (Polars ≥ 1.31) |
| Large materialisation before XGBoost | Dense matrix copies (~4 k cols × many rows) | **Row-cap**: `SS_MAX_ROWS` env var (default 500 k) samples rows with `lf.limit()` before collect |

---

## 3 Model Fitting (XGBoost)

| Symptom | Root Cause | Fix |
| ------- | ---------- | --- |
| `ValueError: could not convert string to float` | Residual string column (`gene_type`) in feature matrix | Added to `METADATA_COLUMNS` so it is dropped in preprocessing |
| OOM even with cap | 500 k × 4 k → 6 GB dense matrix copies | Lower `SS_MAX_ROWS` (e.g. 100 k) or switch to incremental training |

---

## 4 Incremental / Out-of-Core Option

When full in-memory training is impossible, use
`IncrementalTrainer` (scikit-learn `partial_fit` API). Example:

```python
from meta_spliceai.splice_engine.meta_models.training.incremental import IncrementalTrainer
trainer = IncrementalTrainer(model_spec="sgd_logistic", batch_size=250_000)
trainer.fit("train_pc_20000/master").save()
```

For tree models, extend to `xgboost.dask.DaskXGBClassifier`.

---

## 5 Tuning Cheat-Sheet

| Lever | Why | Example |
| ----- | --- | ------- |
| `SS_MAX_ROWS` | Reduce rows collected | `export SS_MAX_ROWS=200000` |
| `--pruning-sample-rows-per-shard` | Fewer rows sampled per shard | `--pruning-sample-rows-per-shard 10000` |
| `--pruning-max-corr-cols` | Limit corr-analysis width | `--pruning-max-corr-cols 800` |
| `batch_size` (Incremental) | Controls RAM per mini-batch | `batch_size=100_000` |

---

## 6 Future Improvements

* **Feature hashing / sparse matrices** to cut dense width.
* **On-disk DMatrix** (XGBoost external memory) for large row counts.
* **Parquet predicate pushdown** to load task-specific columns only.
* **Benchmark Polars ≥2.0**: newer versions have lower peak RSS for joins & collects.
