# Dataset Artifacts and Intermediate Files

This document explains the different Parquet files and directories produced
while **incrementally** building the meta-model training dataset.

> **Why “train_pc_1000”?**  
> The output directory is named *train_pc_1000* purely as an **example**: it
> contains training data for **1 000 protein-coding (pc) genes**.  You can pick
> any directory name, e.g. `train_pc_5000`, `train_lncrna_20000`, to reflect the
> gene type and count chosen when running the incremental builder.
>
> Example command (CLI ≥ 2025-06-17):
>
> ```bash
> conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
>     --n-genes      1000 \      # total gene count
>     --subset-policy error_total \   # choose genes with most total errors
>     --batch-size   250  \      # genes per batch
>     --output-dir   train_pc_1000 \
>     --run-workflow \          # run enhanced prediction workflow first
>     --overwrite    \
>     -v                        # increase verbosity
> ```
>
> Adjust the arguments to match your experiment: larger `--n-genes`, different
> `--subset-policy`, multiple `--kmer-sizes`, etc.


| Artifact | Example Path | Purpose |
| --- | --- | --- |
| Trimmed batch Parquet | `train_pc_1000/batch_00001_trim.parquet` | Feature-complete dataset for **one** gene batch after down-sampling the majority class. Handy for debugging or quick experiments while the full build is still running. |
| Raw temporary Parquet | `train_pc_1000/batch_00001_raw.tmp.parquet` | Scratch file written during enrichment of a batch. May be safely deleted if the process crashes; it will be regenerated. |
| Master dataset directory | `train_pc_1000/master/` containing `part-*.parquet` | Consolidated Arrow dataset created by *appending* each trimmed batch. This directory is the canonical input for model‐training. |

## Typical workflow

1. **Debugging** – Iterate quickly on a single `*_trim.parquet` file (e.g. test
   preprocessing, try model hyper-params).
2. **Incremental run** – Even before all batches finish you can point training
   scripts at `master/`; Polars will read whatever `part-*.parquet` files exist
   so far.
3. **Final dataset** – After the builder completes, keep `master/` and archive
   or delete the individual batch files to reclaim disk space.

## Schema compatibility

All batch files and master parts share the same column set. If you add new
enrichment columns *mid-run*, Arrow will up-cast the missing columns in earlier
parts to NULLs. To guarantee a consistent schema, regenerate earlier batches or
run the builder from scratch after major feature additions.

## Artifact Validation and Cleanup

During the build process, various intermediate artifacts are generated in the
`data/ensembl/spliceai_eval/meta_models/` directory. Sometimes these artifacts
may become corrupted due to interrupted processes or have incorrect schemas due
to code changes.

For comprehensive documentation on validating and cleaning up these artifacts, see:
[Artifact Validation and Cleanup](artifact_validation.md)

### Quick Validation Commands

```bash
# Check for corrupted artifacts
python scripts/validate_artifacts.py --list-corrupted

# Clean up corrupted files
python scripts/cleanup_artifacts.py --corrupted-only

# Validate all artifacts and generate report
python scripts/validate_artifacts.py --output-report validation_report.txt
```
