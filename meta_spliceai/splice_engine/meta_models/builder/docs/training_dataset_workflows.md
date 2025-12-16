# Training Dataset Construction Workflows

This guide outlines the different workflows for building a meta-model training dataset, depending on what artifacts you already have.

> **Scope**: This document covers dataset construction. For model training, see the documentation under `splice_engine/meta_models/training/`.

---

### Quick Reference

| Your Situation                               | Recommended Workflow                                     | Key Command                                                 |
| -------------------------------------------- | -------------------------------------------------------- | ----------------------------------------------------------- |
| Starting from scratch                        | **A: End-to-End Build**                                  | `incremental_builder` with `--run-workflow`                 |
| Specific gene list & artifacts exist       | **B: Build for Specific Gene List (Artifacts Must Exist)** | `extract_gene_tx_ids` (optional) & `incremental_builder`    |
| Build was interrupted (some batches exist)   | **C: Resume or Finalize Build**                          | Re-run `incremental_builder` or use `quick_dataset_builder` |
| Dataset is built, needs validation/updates | **D: Maintenance & Validation**                          | `patch_*` and `test_*` scripts                              |

---

### Why an *Incremental* Builder?

Building the full-genome meta-model dataset involves tens of millions of rows.  Trying to materialise everything at once often exhausts RAM and makes the whole process brittle—one crash means starting over.

The *incremental builder* therefore works **batch-by-batch**:

1. Choose a chunk of genes (default 1 000).
2. Build + enrich + down-sample that slice only.
3. Append / hard-link the trimmed batch into a growing `master/` Arrow dataset.
4. Repeat until the requested number of genes is reached.

Because every batch is persisted independently you can:

• Resume after interruptions (already-finished batches are skipped).  
• Extend an existing dataset by running more batches.  
• Control memory footprint via `--batch-size`.

`--patch-dataset` can be applied **once at the end** so that the heavy patch scripts touch the data only after the whole master directory is assembled.

---

## Core Workflows

### A. End-to-End Build (From Scratch)

Use this path when you have no pre-computed SpliceAI artifacts. This single command runs the entire pipeline.

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
       --n-genes 20000                  `# Total genes to process` \
       --subset-policy error_total      `# Gene selection strategy` \
       --batch-size 1000                `# Genes per batch` \
       --run-workflow                   `# ⬅️ Critical: runs SpliceAI + enhancement` \
       --easy-ratio 0.3                 `# TN down-sampling ratio` \
       --output-dir train_pc_20000      `# Output directory for batches/ and master/` \
       --overwrite -v
```

**Process**: Runs SpliceAI → Creates prediction artifacts → Generates `_raw.parquet` → Generates `_trim.parquet` → Assembles `master/` dataset → *(optional)* applies `--patch-dataset` to enrich structural & gene-type columns in place.

### B. Build for a Specific Gene List (Artifacts Must Exist)

Use this path when you have a specific list of gene IDs and their corresponding SpliceAI prediction artifacts (`analysis_sequences_*`, `splice_positions_enhanced_*`, etc.) already exist. The gene list can be generated, for example, by `extract_gene_tx_ids.py` or be a custom list.

**Important**: For this path, `incremental_builder` must be run *without* the `--run-workflow` flag, as it relies on pre-existing SpliceAI artifacts for the specified genes.

**Example Scenario:**

1.  **(Optional) Generate a Gene List:**

    *   To extract gene IDs from existing artifacts (e.g., from a previous SpliceAI run on a larger set of genes for which you have `eval-dir`):
        ```bash
        python -m meta_spliceai.splice_engine.meta_models.examples.extract_gene_tx_ids \
               --eval-dir /path/to/spliceai_eval/meta_models \
               --out gene_ids_from_artifacts.tsv
        ```
    *   To **randomly sample** 1 000 genes directly from the artifacts (new `--n-genes` option):
        ```bash
        python -m meta_spliceai.splice_engine.meta_models.examples.extract_gene_tx_ids \
               --eval-dir /path/to/spliceai_eval/meta_models \
               --n-genes 1000 \
               --out gene_ids_1000.tsv
        ```
    *   To create a **deterministic subset** (e.g., the first 1 000 genes), you can still filter the TSV produced above:
        ```bash
        head -n 1001 gene_ids_from_artifacts.tsv > gene_ids_1000.tsv   # +1 for header
        ```
    *   Alternatively, your `gene_ids.tsv` file could be manually curated or come from another source.

2.  **Run Incremental Builder with the Gene List:**

    Feed the gene list to the `incremental_builder`. Remember to omit `--run-workflow`.

    ```bash
    python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
           --gene-ids-file gene_ids_1000.tsv --gene-col gene_id \
           --batch-size 250 \
           --batch-rows 20000 \
           --kmer-sizes 6 \
           --output-dir train_pc_1000 \
           --overwrite -vv
    ```

### C. Resume an Interrupted Build

If the builder was stopped, you have two options:

1.  **To Continue a Partially-Finished Build:**
    Simply re-run the *same `incremental_builder` command* without `--overwrite`. The script will detect and skip completed batches.

2.  **To Finalize a Build When All Batches are Trimmed (Master Missing):**
    Simply re-run the *same* `incremental_builder` command **without** `--overwrite`. The script detects that every batch's `_trim.parquet` already exists, skips regeneration, and proceeds directly to linking the `master/` dataset.

> **Tip – One-shot fully-patched dataset**  
> Pass `--patch-dataset` to *incremental_builder* and it will automatically invoke the built-in patch scripts **after** linking the `master/` dataset:
>
> ```bash
> python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
>     --n-genes 1000 --subset-policy error_total \
>     --batch-size 500 --patch-dataset \
>     --output-dir train_pc_1000 -v
> ```
>
> This produces `train_pc_1000/master/` that already contains the structural-feature and gene-type columns—no follow-up patch commands necessary.

---

## Dataset Maintenance & Validation

After any build, ensure dataset integrity before training.

1.  **Patch Features** (optional):  
    Automatically run `patch_structural_features.py` and `patch_gene_type.py` to
    fill missing structural/gene-type columns in place.

    ```bash
    python scripts/patch_structural_features.py  train_pc_1000 --inplace
    python scripts/patch_gene_type.py            train_pc_1000 --inplace
    ```

2.  **Deduplicate Rows** (recommended when duplicates ≥10 %):  
    Remove exact duplicates created when the same splice site appears in multiple
    transcripts.  By default a separate `*_dedup/` directory is written.

    ```bash
    python -m meta_spliceai.splice_engine.meta_models.builder.deduplicate_dataset \
           train_pc_1000/master                       \
           --dst train_pc_1000/master_dedup            # or --inplace
    ```

3.  **Run Integrity Checks**:  
    Verify that the dataset has no nulls in critical columns and is structurally
    sound.

    ```bash
    python scripts/test_dataset_integrity.py             train_pc_1000
    python scripts/validate_meta_model_training_data.py  train_pc_1000
    ```

4.  **Quick Dataset Summary** *(optional)*:  
    Get high-level counts, class balance, and null distribution.

    ```bash
    python -m meta_spliceai.splice_engine.meta_models.analysis.dataset_summary \
           train_pc_1000/master_dedup
    ```

---

## Quick Meta-Model Training Demo

Once your `master/` dataset is ready, you can sanity-check meta-model training and
measure improvement over the raw SpliceAI scores in a **single command** using the
helper script `demo_train_meta_model.py`.

```bash
python -m meta_spliceai.splice_engine.meta_models.training.demo_train_meta_model \
       train_pc_1000/master                       \
       --prune-features                           \
       --model xgboost                            \
       --out-dir models/xgb_pc1000
```

Key options
| Flag | Default | Purpose |
|------|---------|---------|
| `dataset` (positional) | – | Path to the `master/` Parquet directory or file. |
| `baseline_tsv` (positional, optional) | *auto* | Gene-level SpliceAI metrics. Omit to let the script auto-locate `full_splice_performance.tsv` under the default evaluation directory. |
| `--model` | `xgboost` | Any model spec accepted by `Trainer` (e.g. `lightgbm`, `sgd_logistic`). JSON strings are also accepted for hyper-params. |
| `--out-dir` | `models/demo_run` | Where to write the trained model, metrics JSON and `delta_vs_baseline.csv`. |
| `--prune-features` | *off* | Stream-prune zero-variance + highly-correlated (|ρ|>0.99) columns before training; avoids loading the full dataset into memory. |
| `--corr-thr` | `0.99` | Correlation threshold for pruning when `--prune-features` is set. |
| `--group-col` | `gene_id` | Column that defines groups for group-aware splitting. |

Why this demo is useful
1. **Memory-safe pruning** – uses Polars `scan_parquet` to inspect only a small
   sample before streaming the full data through a column-drop pipeline; pandas
   is never invoked.
2. **Group-wise (gene-wise) splitting** – `Trainer` employs
   `GroupShuffleSplit`, ensuring that all rows for a gene land in exactly one of
   train/valid/test, thus preventing leakage.
3. **Baseline comparison built-in** – after fitting, the script calculates the Δ
   (default key: `f1_score`) between the meta-model and baseline SpliceAI gene-
   level results and writes `delta_vs_baseline.csv` for quick plotting.
4. **Self-configuring defaults** – When `baseline_tsv` is omitted the script
   automatically discovers `full_splice_performance.tsv` via
   `MetaModelDataHandler.eval_dir`, so you can run the command with only the
   dataset path.

> For datasets >3 million rows or tighter RAM budgets, substitute
> `training.IncrementalTrainer` via the `--model` flag (e.g. `--model
> sgd_logistic`) or use the dedicated CLI wrapper described under
> `training/incremental.py`.

---

## Advanced Topics

-   **Scalability**: The builder streams data using Apache Arrow, keeping memory usage low regardless of dataset size. It is designed to handle 20,000+ genes.
-   **Failure Recovery**: Re-running a command without `--overwrite` allows the builder to resume from the last completed batch.
-   **Parallelism**: You can run multiple builder instances on different sets of genes and merge the final `master/` directories using the `quick_dataset_builder`.

---

_Last updated: 2025-06-19_
