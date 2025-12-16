# Data-Integrity & Distance-Feature Guidelines

This document captures design decisions and implementation details that arose while building the meta-model feature pipeline.  It explains _why_ they were adopted and provides reference code snippets for future maintenance.

---

## 1. Preventing Row Inflation when Merging Feature Tables

### Problem
`gene_level` enrichment merges the base dataset with a table of (gene_id, transcript_id, …) genomic features.  If the right-hand table contains **multiple** rows for the same key, a left-join replicates base rows and silently inflates the sample count.

### Solution – Deduplicate Before the Join
```python
rows_before = gene_fs_df.shape[0]
gene_fs_df = gene_fs_df.unique(subset=[col_gid, col_tid])
removed = rows_before - gene_fs_df.shape[0]
if removed:
    print(f"[gene-level] removed {removed:,} duplicate rows before merge")

train_df = join_and_remove_duplicates(train_df, gene_fs_df,
                                      on=[col_gid, col_tid], how="left")
```
*Always* ensure `(gene_id, transcript_id)` uniqueness before merging.

> **Why only these columns?**  The GTF‐derived feature table is **transcript-level** – it does **not** include the splice-site `position`.  A transcript has one strand, so `(gene_id, transcript_id)` (or `(gene_id, transcript_id, strand)` for explicitness) is already unique.  
>
> For the **analysis-sequence / training** rows themselves the unique key is different:  
> `gene_id, strand, position, splice_type` (and `transcript_id` if you want transcript-specific duplication).  Deduplicating the auxiliary transcript table therefore **cannot** collapse legitimate splice-site rows; it only prevents accidental row inflation from duplicate transcript entries.

---

## 2. Aggregating Exon-Level Features Up-Front

When exon-level tables are required (e.g. exon lengths), aggregate to a single row **per transcript** before joining:

```python
agg_df = (
    exon_df.groupby(["gene_id", "transcript_id"])
            .agg(
                total_exon_length = pl.sum("exon_length"),
                num_exons         = pl.count(),
                avg_exon_length   = pl.mean("exon_length"),
            )
)
```
This guarantees uniqueness and keeps joins O(1) per key.

---

## 3. Defining Distance Features for FP / TN Points

Not every predicted position is a true splice site:

* **TP / FN** – belong to annotated splice sites → have `transcript_id` and transcript boundaries.
* **FP / TN** – may lie outside transcripts.

Rules implemented in `incorporate_distance_features`:

1. **Prefer transcript boundaries** when `transcript_id` is present.
2. **Fallback to gene boundaries** if only `gene_id` is known.
3. If neither transcript nor gene context exists, distances are left as *missing* (`NaN`).
4. Two binary flags are added:
   * `has_tx_info` – 1 if `tx_start/tx_end` present.
   * `has_gene_info` – 1 if `gene_start/gene_end` present.
   
Models can learn that “distance unknown” differs from “distance far”.

---

## 4. NaN Imputation Strategy for Distance Columns

Most ML frameworks dislike NaNs.  We impute them with a **sentinel value** slightly larger than the largest observed valid distance.

```python
max_dist    = df[["distance_to_start", "distance_to_end"]].max().max()
sentinel    = max_dist + 500   # 500 nt above max

for col in ("distance_to_start", "distance_to_end"):
    df[col] = df[col].fillna(sentinel)
```

* The sentinel is gene-scale and won’t collide with valid distances.
* `has_tx_info / has_gene_info` still signal that the value was imputed.
* Parameters `impute_missing=True` and `sentinel_offset=500` can be overridden when calling `incorporate_distance_features`.

---

Keep this doc updated as additional data quality or feature-engineering conventions emerge.
