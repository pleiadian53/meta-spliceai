# MetaSpliceAI Data Organization

This document describes the data organization and file naming conventions used in MetaSpliceAI.

## Directory Structure

```
meta-spliceai/
└── data/
    └── ensembl/                      # Data source (e.g., Ensembl database)
        ├── gene_sequence_*.parquet   # Gene sequence datasets (chunked by chromosome)
        ├── gene_sequence_minmax_*.parquet  # Gene sequence with maximal transcript boundaries
        ├── tx_sequence_*.parquet     # Transcript sequence datasets
        ├── splice_sites.tsv          # Splice site annotations inferred from GTF
        ├── splice_sites_enhanced.tsv # Enhanced splice site annotations with gene/transcript attributes
        ├── ... 
        └── spliceai_eval/            # Evaluation directory for SpliceAI predictions
            ├── full_analysis_sequences.tsv       # Complete sequence data for analysis
            ├── full_splice_errors.tsv            # Comprehensive splice error data
            ├── full_featurized_dataset_*.tsv     # Featurized datasets for model training
            ├── ... 
            └── meta_models/                       # Meta-models directory
                ├── analysis_sequences_*_chunk_*.tsv     # Chunked contextual sequence data
                ├── splice_errors_*_chunk_*.tsv          # Chunked error analysis files
                ├── splice_positions_enhanced_*_chunk_*.tsv  # Enhanced splice site positions data
                ├── full_splice_positions_enhanced.tsv    # Complete set of enhanced splice positions
                ├── fn_rescue_analysis_*/               # Experiment for False Negative rescue analysis
                ├── context_feature_analysis_*/         # Experiment for context feature analysis
                └── ...                                 # Other experiment-specific directories
        └── ...                       # Additional data directories and files
```

## Key Data Files

### Base Sequence Data

| File Pattern | Description |
|--------------|-------------|
| `gene_sequence_*.parquet` | Gene sequences chunked by chromosome. Contains genomic sequences for each gene. |
| `gene_sequence_minmax_*.parquet` | Gene sequences with maximal boundaries from associated transcripts. |
| `tx_sequence_*.parquet` | Transcript sequences. Contains genomic sequences for each transcript. |

### Splice Site Annotations

| File | Description |
|------|-------------|
| `splice_sites.tsv` | Basic splice site annotations inferred from GTF files. Contains position, type, gene, and transcript information. |
| `splice_sites_enhanced.tsv` | Enhanced version with additional gene and transcript attributes alongside the annotated splice sites. |

### Error Analysis Files

| File | Description |
|------|-------------|
| `full_analysis_sequences.tsv` | Complete sequence data used for error analysis. |
| `full_splice_errors.tsv` | Comprehensive data of splice prediction errors (FP, FN). |
| `full_featurized_dataset_fn_tp.tsv` | Features for distinguishing between False Negatives and True Positives. |
| `full_featurized_dataset_fn_tn.tsv` | Features for distinguishing between False Negatives and True Negatives. |

### Meta Model Files

| File Pattern | Description |
|--------------|-------------|
| `analysis_sequences_*_chunk_*.tsv` | Chunked contextual sequence data for meta-model analysis. |
| `splice_errors_*_chunk_*.tsv` | Error analysis files separated by gene ID chunks. |
| `splice_positions_enhanced_*_chunk_*.tsv` | Enhanced splice position data with context-aware features. Used for meta-model training. |

## File Naming Conventions

1. **Base Files**: Use descriptive prefix followed by content type:
   - `gene_sequence_*` - Gene sequences
   - `tx_sequence_*` - Transcript sequences

2. **Chunked Files**: Include chunking information:
   - `*_<chromosome/gene>_chunk_<start>_<end>.tsv`
   - Example: `analysis_sequences_11_chunk_1501_2000.tsv` (chromosome 11, genes 1501-2000)

3. **Enhanced Files**: Add `enhanced` suffix to indicate additional features:
   - `splice_sites_enhanced.tsv`
   - `splice_positions_enhanced_*.tsv`

4. **Experiment Directories**: Use descriptive name with date:
   - `fn_rescue_analysis_YYYYMMDD`
   - Example: `fn_rescue_analysis_20250511`

## Importing Data

The IO module provides utilities for loading these datasets:

```python
from meta_spliceai.splice_engine.meta_models.io import datasets

# Load enhanced splice positions
positions_df = datasets.load_full_positions_data()

# Load specific chunks
chunk_df = datasets.load_positions_chunk(chromosome=11, chunk="1501_2000")
```

## Directory Resolution

The system uses the following directory resolution:

1. `data_dir`: Base data directory (`/home/bchiu/work/meta-spliceai/data/ensembl`)
2. `eval_dir`: Evaluation directory (`data_dir/spliceai_eval`)
3. `meta_dir`: Meta-models directory (`eval_dir/meta_models`)

Classes like `Analyzer` maintain these paths and provide consistent access.
