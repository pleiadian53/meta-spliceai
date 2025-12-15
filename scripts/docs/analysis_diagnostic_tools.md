# Diagnostic Scripts for Dataset Integrity

This document describes two helper scripts that quickly sanity-check different stages of the meta-model pipeline.  They live in the `scripts/` folder and are **read-only diagnostics** – they never mutate data.

| Script | Purpose | Typical stage |
|--------|---------|---------------|
| `inspect_analysis_sequences.py` | Summarise the raw *analysis_sequences_* TSV chunks (per-gene splice-site predictions before k-mer featurisation). | Pre-build sanity check – verify row counts & required columns before creating the base training dataset. |
| `inspect_meta_model_training_data.py` | Summarise the final (or intermediate) **Parquet** training dataset after k-mer + feature enrichment. | Post-build sanity check – confirm row counts match the source TSVs and inspect feature composition / null patterns. |

> **Note:** The `inspect_…` prefix makes it clear these scripts are read-only diagnostics aimed at specific pipeline stages.

---

## 1  `inspect_analysis_sequences.py`

Quick summary of every `analysis_sequences_*` TSV in a directory.

### Usage
```bash
python scripts/inspect_analysis_sequences.py \
       --tsv-dir  "/path/to/analysis_sequences" \
       [--pattern "analysis_sequences_*.tsv"] \
       [--max-rows 5]          # example rows to print per file
```

### What it reports
* Total and per-file row/column counts
* Columns with missing values and their percentages
* Lists of all columns observed (helpful for schema drift)
* Aggregated row count across all files – compare this with the final dataset to catch row inflation.

---

## 2  `inspect_meta_model_training_data.py`

Summary of one or more Parquet files (usually the enriched training dataset).

### Usage
```bash
python scripts/inspect_meta_model_training_data.py \
       /path/to/training_dataset_*.parquet \
       [--top-null 20]         # show top-20 null columns
```

### What it reports
* Shape and memory footprint per file and in aggregate
* Feature-category breakdown (core, sequence-derived, genomic, other)
* Detailed null analysis (top columns with missing values)
* Data-type distribution (numeric vs object vs categorical)

---

## Adding new checks
Both scripts follow the same template: iterate over files → load with `polars`/`pandas` → collect basic stats → pretty-print.  If you want additional diagnostics (e.g. value ranges, duplicate-key checks), copy the helper functions at the bottom of each script.

---

## Location
```
meta-spliceai/
└── scripts/
    ├── inspect_analysis_sequences.py
    ├── inspect_meta_model_training_data.py
    └── docs/
        └── analysis_diagnostic_tools.md   # (this file)
