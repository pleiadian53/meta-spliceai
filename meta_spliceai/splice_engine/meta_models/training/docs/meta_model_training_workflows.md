# Meta-Model Training Workflows

This guide explains how to train **meta-models** that learn to correct raw
SpliceAI predictions (reduce false positives / false negatives) using the
utilities in `splice_engine.meta_models.training/`.

Unlike the dataset-builder guide—which stops after producing a cleaned
`master/` directory—this document focuses on **model fitting and evaluation**.

---

## 1. Quick-Start (1 000-gene demo)

Assume you already have a deduplicated training dataset in
`train_pc_1000/master/` and gene-level SpliceAI performance metrics (usually
`full_splice_performance.tsv`).

```bash
# Train XGBoost meta-model, prune redundant columns, and compare to baseline
python -m meta_spliceai.splice_engine.meta_models.training.demo_train_meta_model \
       train_pc_1000/master                       \
       --prune-features                           \
       --model xgboost                            \
       --out-dir models/xgb_pc1000
```

### What happens under the hood
1. **Memory-safe pruning** – Polars `scan_parquet` inspects a small sample to
   drop zero-variance and highly correlated (|ρ|>0.99) columns, then **streams**
   the full dataset through a column-drop pipeline. No pandas objects are
   created.
2. **Group-wise (gene) splitting** – The default `--group-col gene_id` triggers
   `GroupShuffleSplit`, ensuring that every gene’s rows appear in exactly one of
   train / validation / test splits, preventing information leakage.
3. **Model fitting** – The selected learner (default XGBoost) is trained; full
   feature matrix is loaded once (≈ 6 GB for the 1000-gene example).
4. **Baseline comparison** – After fitting, the script merges per-gene F1 scores
   with the baseline SpliceAI TSV and writes `delta_vs_baseline.csv`.

### Key CLI options
| Flag | Default | Purpose |
|------|---------|---------|
| `dataset` *(positional)* | – | Path to `master/` Parquet directory or file. |
| `baseline_tsv` *(positional, optional)* | *auto* | Gene-level baseline TSV. If omitted the script looks for `full_splice_performance.tsv` under `MetaModelDataHandler.eval_dir`. |
| `--model` | `xgboost` | Any model spec accepted by `Trainer` (`lightgbm`, `sgd_logistic`, JSON string, etc.). |
| `--out-dir` | `models/demo_run` | Directory for model pickle, metrics JSON, and delta CSV. |
| `--prune-features` | *off* | Enable streaming feature pruning. |
| `--corr-thr` | `0.99` | Correlation threshold when pruning. |
| `--group-col` | `gene_id` | Column defining groups for leak-free splits. |
| `--test-size` / `--valid-size` | `0.2` / `0.1` | Override held-out fractions. |

> **Tip** – Pass JSON strings, e.g. `--model '{"learning_rate":0.05,"n_estimators":500}'` to
> fine-tune hyper-parameters without editing code.

---

## 2. Scaling to Larger Datasets

For > 3 million rows—or when RAM is limited—switch to
`IncrementalTrainer`, which calls `partial_fit()` on mini-batches and keeps
memory constant.

```bash
python -m meta_spliceai.splice_engine.meta_models.training.incremental \
       train_pc_large/master                      \
       --model lightgbm                           \
       --batch-size 200_000                       \
       --target splice_type                       \
       --out-dir models/lgbm_large
```

`IncrementalTrainer` shares the same evaluation and baseline-comparison logic
as `Trainer`, so downstream plots are identical.

---

## 3. Cross-Validation (optional)

`Trainer` returns one train/valid/test split. For K-fold gene CV create a small
wrapper:

```python
from sklearn.model_selection import GroupKFold
from meta_spliceai.splice_engine.meta_models.training import Trainer, datasets
import polars as pl, numpy as np

df = datasets.load_dataset("train_pc_1000/master")
X, y_multi = datasets.prepare_xy(df)
y = (y_multi > 0).astype(int)
G = df["gene_id"].to_pandas().values

cv = GroupKFold(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, G)):
    t = Trainer(model_spec="xgboost", out_dir=f"models/cv/fold{fold}")
    t.fit(df[train_idx])  # or supply sliced arrays
    t.compare_baseline("full_splice_performance.tsv")
```

---

## 4. Artefacts and Next Steps

After a successful run you will find:

```
models/xgb_pc1000/
├── model.pkl               # Trained estimator
├── metrics.json            # Overall test/valid metrics
├── delta_vs_baseline.csv   # Per-gene Δ metrics
└── dataset_trimmed.parquet # Only if --prune-features was used
```

Feed `delta_vs_baseline.csv` into your favourite plotting tool to visualise
wins/losses per gene or splice type.

---

_Last updated_: 2025-06-19
