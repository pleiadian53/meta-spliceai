# Meta-Model Dataset Builder – Known Issues & Work-arounds

This document collects the *non-obvious* pitfalls that can appear while
assembling training datasets with `quick_dataset_builder.py` and
`incremental_builder.py`. Each section describes the symptom, root cause, and
recommended fix or mitigation.

---

## 1. Parquet schema mismatch across batches

**Symptom**

```
ValueError: Table schema does not match schema used to create file
```

**Cause**

When k-mer features are enabled only the k-mers that actually appear in a batch
were present, so different batches produced different column sets.

**Fix** (code ≥ 2025-06-17)

`dataset_builder.build_training_dataset()` now pre-computes the full list of
k-mer columns and pads missing ones with zeros. If you encounter the error on
an *old* run, simply delete the partially-written Parquet file and re-run with
the updated code.

---

## 2. Corrupted temporary Parquet – `PAR1` footer missing

**Symptom**

```
polars.exceptions.ComputeError: parquet: File out of specification: The file must end with PAR1
```

**Cause**

The temporary file `batch_x_raw.tmp.parquet` was truncated before the writer
closed, most frequently due to zstd compression hiccups or the process being
killed.

**Work-around**

Starting from code ≥ 2025-06-17 the temp file is written *uncompressed* to make
writes more robust:

```python
build_training_dataset(..., compression=None)
```

If you still see the error:

1. Delete the offending `*.tmp.parquet` file.
2. Ensure there is sufficient disk space.
3. Re-run the builder.

---

## 3. `num_exons` vs `exon_number` column mismatch

**Symptom**

```
polars.exceptions.ColumnNotFoundError: num_exons
```

**Cause**

`exon_features.tsv` cached an *unaggregated* per-exon table that only contains
`exon_number`. The builder expects the *aggregated* transcript-level table with
`num_exons`, `avg_exon_length`, …

**Fix**

`FeatureAnalyzer.retrieve_exon_features_at_transcript_level()` now validates
the cache and will recompute the summary automatically if the aggregated
columns are missing. If using an older version, delete
`data/…/exon_features.tsv` and re-run.

---

## 4. `torch` not found when the builder is run by Cascade

**Symptom**

```
ModuleNotFoundError: No module named 'torch'
```

**Cause**

`run_command` sessions don’t inherit your interactive conda activation and fall
back to the system Python.

**Work-around**

Invoke the builder through **conda-run** (or use the full interpreter path):

```bash
conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder …
```

---

## 5. Extremely verbose Arrow schema dumps

PyArrow prints the entire 4 000-column schema on mismatch, flooding the
terminal. You can silence the pretty-print by exporting:

```bash
export ARROW_PRETTY_PRINT=0
```

---

## 6. Memory pressure with 6-mer features

Building 6-mer features for >10⁶ rows can peak at several GB of RAM.
Strategies:

* Use `--batch-rows` to keep each Polars batch small (e.g. 200 000).
* Build without k-mers first (`--kmer-sizes` omitted) and later append them
  with `add_kmers.py`.

---

## 7. OOM kill during distance-to-boundary enrichment

**Symptom**

Process is killed (no traceback) shortly after  
`[enrichment] Running 'distance_features' …`

**Root cause (pre-2025-06-17)**

`compute_distances_with_strand_adjustment()` converted the main training
DataFrame (millions of rows) from Polars to Pandas and then ran a
row-wise `apply`. This:

* materialised an additional full copy of the table in memory,
* created a temporary Python object for **every row**, and
* added two more full-length columns before the original was freed,

leading to >2× peak RAM and triggering the OS out-of-memory killer.

**Fix (≥ 2025-06-17)**

The function now:

* stays in Polars,
* computes distances with vectorised expressions, and
* validates results without extra passes.

Peak memory ≈ (input + 3 new columns).

**Why other enrichers are safe**

Gene/length/performance/overlap enrichers join relatively small
side-tables (5 k – 250 k rows) onto the large training table and keep
everything in Polars. They do not perform per-row Python loops or
convert to Pandas, so memory grows linearly with row count.

---

## 8. Adding new enrichment functions – scalability checklist

When implementing future enrichers:

1. **Stay in Polars** – avoid `to_pandas()` unless the side-table is tiny.
2. **No per-row Python** – use Polars expressions or `apply` on *small*
   side-tables only.
3. **Chunk processing** – if a full-table operation cannot be expressed
   vectorially, process in `LazyFrame.scan_batches()` sized chunks.
4. **Validate on 250 k rows**; profile `peak_rss` before scaling up.
5. Watch for wide joins that explode the column count – ensure `join` is
   done on unique keys to avoid a Cartesian blow-up.

---

## 9. k-mer streaming & column explosion

During base-dataset build, k-mer features are extracted by streaming each
analysis TSV and writing *row-groups* to a temporary Parquet. Memory is
bounded by the `--row-group-size` (default ≈25 000 rows), but two factors
can still bite:

* **High sample count** – batches with many genes may hold several
  hundred thousand rows in the in-flight group.
* **Larger k or multiple k values** – the number of feature columns grows
  exponentially (4ᵏ). A 6-mer matrix for 1 M rows can exceed 3 GB even
  with Int8 encoding.

Mitigations:

* Lower `--batch-rows` or `--row-group-size` (e.g. 100 000 / 10 000).
* Generate k-mers in a follow-up pass with `add_kmers.py`.
* Consider hashing tricks (e.g. MinHash) if >6-mers are ever required.

---

## 10. OOM kill during TN down-sampling

**Symptom**

```
[batch_00004] down-sampling TNs …
Killed            # (process terminated by the OS)
```

The process was killed while calling `downsample_tn()` on a ~220 k-row batch.

**Root cause (pre-2025-06-17)**

`downsample_tn.py` loaded the entire Parquet into **Pandas**, duplicating the
dataframe in memory (Arrow → pandas) and applying per-column operations there.
For wide tables (~4 000 columns) this could briefly allocate several GB and
trigger the OOM killer on machines with ≤16 GB RAM.

**Fix (≥ 2025-06-17)**

1. Re-implemented `downsample_tn` to stream the Parquet directly into **Polars**.
2. All masks (`is_tn`, `hard_neg`, neighborhood negs) computed via NumPy arrays.
3. Final filtering uses `df.filter()` in Polars and writes the trimmed Parquet
   with `write_parquet(compression="zstd")`.

Peak RSS dropped from ~2.5× input size to ~1.2× and no longer crashes at
250 k-row batches.

---

## 11. OOM during exon-feature summaries (GTF parsing)

**Symptom**

```
[enrichment] Running 'gene_level' …
[i/o] Loading exon features …
[cache-miss] Cached exon_features file lacks aggregated columns; recomputing …
Killed       # OOM killer strikes while summarising
```

**Root cause**  
When the cached `exon_features.tsv` is missing the aggregate columns
(`num_exons`, `total_exon_length`, …) the builder calls
`summarize_exon_features_at_transcript_level()`.  This function currently:

1. Reads the full **GTF** (~3–4 M rows) into memory as a Polars `DataFrame`.
2. Materialises the *exon* subset (~1 M rows) in memory again.
3. Groups *all* exons by transcript in one go.

For large genomes this can exceed 10 GB RSS on machines with ≤16 GB RAM.

**Work-arounds**

1. **Pre-warm the cache** once on a beefy machine:
   ```bash
   conda run -n surveyor python - <<'PY'
   from meta_spliceai.splice_engine.extract_genomic_features import (
       summarize_exon_features_at_transcript_level as run,
   )
   run(
       gtf_file_path="/path/to/Homo_sapiens.GRCh38.110.gtf",
       output_file="data/ensembl/spliceai_analysis/exon_features.tsv",
       verbose=2,
   )
   PY
   ```
   The ensuing TSV (~200 MB) contains the aggregates so future builder runs
   *skip* the expensive recompute.
2. **Increase swap** or move the builder to a node with ≥32 GB RAM.
3. Planned fix: rewrite the summariser to stream the GTF through
   `pl.scan_csv()` and aggregate in batches (ETA soon).

---

Feel free to extend this list whenever a new quirk is discovered.



| Artifact | Path pattern | Purpose |
| --- | --- | --- |
| Trimmed batch Parquet | `train_pc_1000/batch_00001_trim.parquet` | Feature-complete dataset for *one* gene batch. Used during incremental build for memory-safe enrichment and for quick debugging/experimentation. |
| Raw temporary Parquet | `train_pc_1000/batch_00001_raw.tmp.parquet` | Scratch file while a batch is being enriched. Safe to delete if the process crashes. |
| Master dataset directory | `train_pc_1000/master/` with `part-*.parquet` | Consolidated Arrow dataset: row-wise concatenation of all trimmed batches. This is the canonical input for model-training.

**Typical workflow**

1. **Debugging** – Train/plot on a single `*_trim.parquet` to iterate quickly.
2. **Incremental run** – While the builder is still appending parts, you can already load `master/` and experiment; Polars will read whatever parts exist.
3. **Final dataset** – After all batches finish, keep `master/` and archive or delete individual batch files to reclaim disk.

*Schema compatibility*: All batch and master parts share the same column set. If you add new enrichment columns mid-run, Arrow will up-cast absent columns to nulls; regenerate earlier batches to keep the schema aligned.
