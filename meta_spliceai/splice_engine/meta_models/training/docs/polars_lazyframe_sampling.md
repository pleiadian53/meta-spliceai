# Sampling rows from a Polars **LazyFrame**

> TL;DR – `LazyFrame.sample()` does **not** exist (≤ v1.31).  Use a hash-based filter or add a row index and take every *k*-th row instead.

## The symptom

```python
>>> lf = pl.scan_parquet("dataset/*.parquet")
>>> lf.sample(n=10_000)
AttributeError: 'LazyFrame' object has no attribute 'sample'
```

Polars implements `.sample()` only on **DataFrame** / **Series**.  Calling it on a
`LazyFrame` inevitably triggers the above error and usually leads to people
accidentally switching to eager mode (`collect()`), re-introducing the very
OOM problems lazy execution was meant to fix.

## Why it matters for MetaSpliceAI

During meta-model training we often need to

* draw a validation minibatch for quick metrics,
* feed a modest background dataset to SHAP, or
* just preview a few rows.

Doing that eagerly on millions of rows would blow up memory.

## Recommended workaround (hash-based lazy sampling)

We add a monotonically increasing row counter (or any deterministic hash), then
keep **every `step`-th** row:

```python
import polars as pl
import numpy as np

def lazyframe_sample(lf: pl.LazyFrame, n: int, seed: int = 42) -> pl.LazyFrame:
    """Roughly `n` random rows without materialising the whole dataset."""
    total = lf.select(pl.count()).collect(streaming=True).to_series(0).item()
    if total <= n:
        return lf

    step   = max(1, total // n)
    offset = np.random.default_rng(seed).integers(step)

    return (
        lf.with_row_count("_idx")
          .filter(((pl.col("_idx") + offset) % step) == 0)
          .drop("_idx")
          .limit(n)
    )
```

* Complexity: **O(1)** memory, **O(total_rows)** time (linear scan), still fully
  lazy/streaming.
* Result size is guaranteed ≤ `n` and statistically close to `n`.
* `seed` gives reproducibility.

### Adopted in codebase

The helper now lives in `meta_spliceai.splice_engine.meta_models.training.classifier_utils._lazyframe_sample`
and replaces every previous `.sample()` call.  Use it whenever you need a lazy
sample on Polars ≤ 1.31.

## Alternative strategies

1. **Eager DataFrame sampling**
   ```python
   df = lf.collect()
   df.sample(n)
   ```
   Works for small tables only – **not recommended** for our 800 k×4 k dataset.
2. **`RandomSample` scan filter** (pending Polars feature request)
   Some future Polars release may offer a native lazy sampling expression.  We
   can switch once that lands.
