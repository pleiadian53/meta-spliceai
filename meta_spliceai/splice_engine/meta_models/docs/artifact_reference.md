# MetaSpliceAI ― Meta-Models Artifact Map

> Maintainers: this is the single source of truth for **where** every durable
> file produced or consumed by the meta-model workflows is stored.
> Update it whenever a new artifact is added or its location changes.

## 1. Top-Level Directories

| Key (`Analyzer` constant) | Default Path | Purpose |
|--------------------------|-------------|---------|
| `data_dir`               | `/…/data/ensembl` | Root for reference inputs (GTF, FASTA) and **global** lookup tables. |
| `analysis_dir`           | `data_dir/spliceai_analysis` | Gene / transcript / exon feature tables, performance summaries, overlap counts, etc.  **Read-many / write-once**. |
| `eval_dir`               | `data_dir/spliceai_eval` | Baseline SpliceAI prediction outputs and evaluation artefacts. |
| `meta_dir` (run-time, via `MetaModelDataHandler`) | `eval_dir/meta_models` | All files generated **specifically** for the meta-model pipelines (enhanced positions, training Parquets, checkpoints). |

<small>`meta_dir` is derived at runtime; the others are class constants that can be overridden via environment/config.</small>

## 2. Global Reference Tables  (`analysis_dir`)

| File | Created by | Description |
|------|------------|-------------|
| `gene_features.tsv` | `FeatureAnalyzer.retrieve_gene_features()` | Static gene-level attributes extracted from the GTF (length, type, exon count …). |
| `transcript_features.tsv` | `FeatureAnalyzer.retrieve_transcript_features()` | Transcript-level summary per gene. |
| `exon_df_from_gtf.tsv` | `extract_exon_features_from_gtf(cache_file=…)` | Raw exon rows parsed from the GTF (cached). |
| `exon_features.tsv` | **NEW (this run)** via `FeatureAnalyzer.retrieve_exon_dataframe()` | Cleaned/feature-rich exon table (length, ordinal, etc.). |
| `performance_df_features.tsv` | `retrieve_gene_level_performance_features()` | Aggregated metrics from `PerformanceAnalyzer` (hard genes, FN/FP rates). |
| `overlapping_gene_counts.tsv` | `overlap_features` enricher | Counts of overlapping genes on each strand. |

These files are **shared** by all analysis / builder / rescue pipelines and do
not depend on the subset of genes under study.

## 3. Baseline Prediction Artefacts  (`eval_dir`)

| File | Created by | Notes |
|------|------------|-------|
| `full_splice_positions.tsv` | `run_spliceai_workflow.py` (baseline mode) | All annotated positions with baseline SpliceAI outputs (no enhanced probabilities). Shortcut for ranking genes by FN/FP. |
| `analysis_sequences_*.tsv` | Same workflow | Sequence-centric view used as k-mer featurisation input. |

## 4. Meta-Model Specific Artefacts  (`eval_dir/meta_models`)

| File / Folder | Produced by | Description |
|---------------|------------|-------------|
| `full_splice_positions_enhanced.tsv` | `run_enhanced_splice_prediction_workflow()` | Baseline table **with** donor / acceptor / neither probabilities. |
| `analysis_sequences_enhanced_*.tsv` | same | Enhanced sequence-level tables. |
| `training_dataset_*.parquet` | `incremental_builder.build_base_dataset()` | Per-batch enriched datasets (before TN down-sampling). |
| `train_dataset_trimmed/master/*.parquet` | `incremental_builder.incremental_build_training_dataset()` | Final, balanced training set used by the meta-learner. |

## 5. Reference Inputs  (`data_dir`)

| File | Source |
|------|--------|
| `Homo_sapiens.GRCh38.112.gtf` | Ensembl release 112 | Used for feature extraction & exon parsing. |
| `Homo_sapiens.GRCh38.dna.primary_assembly.fa` | Ensembl | Reference genome for sequence context. |

## 6. Quick Lookup

```
<data_dir>
├── Homo_sapiens.GRCh38.112.gtf
├── Homo_sapiens.GRCh38.dna.primary_assembly.fa
├── spliceai_analysis/            # global reference tables
│   ├── gene_features.tsv
│   ├── transcript_features.tsv
│   ├── exon_features.tsv
│   └── …
├── spliceai_eval/                # baseline predictions
│   ├── full_splice_positions.tsv
│   ├── analysis_sequences_*.tsv
│   └── meta_models/
│       ├── full_splice_positions_enhanced.tsv
│       ├── analysis_sequences_enhanced_*.tsv
│       └── train_dataset_trimmed/
│           └── master/*.parquet
└── …
```

---
**Tip:** `MetaModelDataHandler` and `FeatureAnalyzer` expose helper methods to
build paths programmatically—avoid hard-coding paths in new code.
