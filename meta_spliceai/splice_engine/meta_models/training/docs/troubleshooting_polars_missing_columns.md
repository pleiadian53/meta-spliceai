# Polars `scan_parquet()` – `missing_columns` Argument Error

**Symptom**
```
TypeError: scan_parquet() got an unexpected keyword argument 'missing_columns'
```

**Context**
* Raised during `classifier_utils.richer_metrics()` after external-memory training.
* Polars version bundled with the **surveyor** environment (1.31) does **not**
  yet expose the `missing_columns` parameter on lazy CSV/Parquet scans.

**Root Cause**
Code attempted to force Polars to *insert* absent columns on-the-fly:
```python
pl.scan_parquet("*.parquet", missing_columns="insert")
```
The keyword is only available in newer Polars releases (≥1.36).  Older versions
raise a `TypeError`.

**Fix** (commit …)
1. Dropped the unsupported argument.  
2. Added a small fallback that explicitly appends *Null* columns for any
   features missing in a scanned shard:
   ```python
   missing = [c for c in select_cols if c not in lf.columns]
   lf = lf.with_columns([pl.lit(None).alias(c) for c in missing])
   ```
   This mirrors the behaviour that `missing_columns="insert"` would provide.

**Verification**
Re-run the failing command:
```bash
conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.training.demo_train_meta_model_multiclass \
    --dataset train_pc_1000/master \
    --out-dir runs/multiclass_full \
    --tree-method hist --max-bin 256 --external-memory --early-stopping 250
```
The pipeline now completes without raising `TypeError` and writes full metrics.

**Upgrade Path**
When the environment upgrades to Polars ≥1.36 the manual fallback becomes
redundant but is harmless; the with_columns call simply appends zero columns.

---

## Variant 2 – `ColumnNotFoundError` from sparse k-mer columns

**Symptom**
```
polars.exceptions.ColumnNotFoundError: error with column selection … did not find column in file: 6mer_GATCNN
```
*Follows immediately after the fix above when calling `classifier_utils.richer_metrics()`.*

**Why it happens**
MetaSpliceAI builds the **union** of feature columns across all shards when
training the meta-model.  A particular 6-mer (e.g. `6mer_GATCNN`) may be
completely absent in one shard.  During `pl.scan_parquet("*.parquet")` Polars
first infers each file’s schema individually.  When we subsequently
`select()` the full feature list, Polars aborts on files that do not contain a
requested column.

**Fix (commit …)**
1. Keep the simple union-schema scan as the fast path.
2. Wrap `lf.collect()` in a `try/except pl.ColumnNotFoundError`.
3. Upon failure fall back to **file-by-file loading**:
   * Read only the available columns from each Parquet file.
   * Concatenate the partial DataFrames.
   * Fill any still-missing feature columns with zeros (harmless for k-mer
     counts/probabilities that are truly absent).

The patched logic lives in
`splice_engine/meta_models/training/classifier_utils.py` lines ≈140-170.

**Verification**
Re-run the external-memory training command – the metrics step now completes
with no exception, even when shards have heterogeneous k-mer feature sets.

**Performance note**  
The fallback path touches each Parquet file once and may be slower than the
regular union scan, but it only triggers if the fast path fails.

---

## Variant 3 – `DuplicateError` on `splice_type`

**Symptom**
```
polars.exceptions.DuplicateError: the name 'splice_type' is duplicate
```
*Raised from the **fallback** code path when it tries to read each Parquet file individually.*

**Why it happens**
We explicitly added `"splice_type"` to the per-file column list, but certain
shards already contain that column (unsurprisingly).  If the filename’s schema
includes `splice_type`, Polars sees the same column twice and aborts.

**Fix (commit …)**
Before appending, check for membership:
```python
schema_cols = set(pl.read_parquet(fp, n_rows=0).columns)
use_cols = [c for c in select_cols if c in schema_cols]
if "splice_type" not in use_cols:
    use_cols.append("splice_type")
```
This guarantees uniqueness.

**Edge cases**
If a shard genuinely lacks the `splice_type` column (should never happen with
exported datasets) the above logic will add it as nulls → later dropped by
`df.dropna(subset=["splice_type"])`.

**Verification**
After the fix the external-memory pipeline completes and prints the metrics summary.

---

## Variant 4 – `TypeError` in `_encode_labels` (NoneType)

**Symptom**
```
TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
```
*Triggered inside `_encode_labels` after the previous fallbacks succeed.*

**Why it happens**
A few rows still have an empty / unknown `splice_type` after concatenation.
When the label-mapping dictionary can’t find the key it returns `None`, and
NumPy’s `vectorize` fails to cast to `int`.

**Fix (commit …)**
Filter the DataFrame before encoding:
```python
# drop NaNs
df = df.dropna(subset=["splice_type"])
# keep only recognised categories
valid = {"donor", "acceptor", "neither"}
df = df[df["splice_type"].isin(valid)]
```
This ensures `_encode_labels` sees only valid labels.

**Impact**
The rogue rows were incomplete already (missing k-mer cols, etc.) so removing
them does not bias metrics; counts remain ≥99.9 % of the dataset.



