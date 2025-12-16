# Meta-Model Features

This directory contains feature extraction and processing modules for the MetaSpliceAI meta-models.

## 📋 Module Overview

This package provides comprehensive feature processing capabilities for splice site prediction meta-models:

- **Gene Selection** - Intelligent gene subsetting for training data optimization
- **K-mer Processing** - Full k-mer extraction, filtering, and optimization pipeline
- **Genomic Features** - Gene-level and contextual genomic information extraction
- **Sequence Featurization** - Unified sequence processing and harmonization

## 🎯 Gene Selection System

The gene selection system provides intelligent strategies for subsetting genes to optimize training data size while preserving informative examples.

### Core Modules

- **`gene_selection.py`** - Main gene selection strategies and utilities

### Gene Selection Strategies

1. **`error_total`** - Genes with highest total error counts (FP + FN) - **recommended**
2. **`error_fp`** - Genes with most false positives
3. **`error_fn`** - Genes with most false negatives  
4. **`random`** - Random sampling of genes
5. **`hard`** - Performance-based hard genes (delegated to PerformanceAnalyzer)
6. **`custom`** - User-provided gene list

### Usage in Training Scripts

```python
# In CV scripts like run_gene_cv_sigmoid.py
from meta_spliceai.splice_engine.meta_models.features import subset_analysis_sequences

# Select top 1000 genes by total error count
subset_df, gene_ids = subset_analysis_sequences(
    data_handler,
    n_genes=1000,
    subset_policy="error_total",
    use_effective_counts=True,  # Deduplicate by genomic position
    verbose=1
)
```

### Memory-Efficient Loading

The gene selection system includes **iterative loading** that only loads sequences for selected genes, significantly reducing memory usage:

```python
# Automatically used in subset_analysis_sequences
analysis_df = data_handler.iterative_load_analysis_sequences(
    target_gene_ids=selected_genes,
    show_progress=True
)
```

## 🧬 K-mer Filtering System

The k-mer filtering system provides domain-specific approaches for handling large k-mer feature sets in splice site prediction.

### Core Modules

- **`kmer_filtering.py`** - Main filtering strategies (motif-based, MI-based, sparsity-aware, ensemble)
- **`kmer_filter_config.py`** - Configuration and integration utilities  
- **`kmer_filtering_demo.py`** - Comprehensive demonstration script
- **`kmer_features.py`** - K-mer extraction and processing

### Quick Start

```bash
# Run the demonstration
python -m meta_spliceai.splice_engine.meta_models.features.kmer_filtering_demo

# Use in CV scripts
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000/master \
    --out-dir models/gene_cv_with_filtering \
    --filter-kmers \
    --kmer-filter-strategy ensemble \
    --n-folds 5
```

### Available Strategies

1. **`motif`** - Preserves biologically relevant splice site motifs
2. **`mi`** - Information-theoretic feature selection
3. **`sparsity`** - Handles sparse k-mer patterns
4. **`ensemble`** - Combines multiple strategies
5. **`variance`** - Traditional variance-based filtering

### Preset Configurations

- **`conservative`** - Preserves most biologically relevant features
- **`balanced`** - Good balance between reduction and preservation
- **`aggressive`** - Maximum feature reduction
- **`motif_only`** - Preserves only known splice site motifs
- **`mi_only`** - Information-theoretic selection

## 🧬 Genomic Features System

The genomic features system extracts gene-level and contextual genomic information used during feature enrichment in training data assembly.

### Core Modules

- **`genomic_features.py`** - Main genomic feature extraction and merging
- **`feature_enrichment.py`** - Feature enrichment pipeline integration
- **`feature_merger.py`** - Feature merging and harmonization utilities

### Key Features

1. **Gene-level Features** - Gene length, type, performance statistics
2. **Contextual Features** - Intron/exon context, distance features
3. **Performance Features** - Gene-specific model performance metrics
4. **Feature Merging** - Efficient joining of multiple feature sources

### Usage in Training Pipeline

```python
# Used in incremental_builder.py during feature enrichment
from meta_spliceai.splice_engine.meta_models.features import merge_contextual_features

# Merge multiple feature sources
enriched_df = merge_contextual_features(
    pos_df=positions_df,
    gene_meta_df=gene_metadata,
    kmer_df=kmer_features,
    on_gid="gene_id"
)
```

## 🔄 Sequence Featurization System

The sequence featurization system provides unified processing of sequence data for different prediction types in the meta-model pipeline.

### Core Modules

- **`sequence_featurization.py`** - Main sequence processing and harmonization
- **`transcript_alignments.py`** - Transcript alignment features

### Key Capabilities

1. **Multi-type Processing** - Handle TP, TN, FP, FN sequences uniformly
2. **Feature Harmonization** - Ensure consistent feature sets across types
3. **Downsampling Integration** - Built-in data balancing capabilities
4. **Flexible K-mer Sizes** - Support for multiple k-mer lengths

### Usage Example

```python
from meta_spliceai.splice_engine.meta_models.features import featurize_analysis_sequences

# Process sequences for all prediction types
featurized_dfs, feature_sets = featurize_analysis_sequences(
    sequence_dfs={'TP': tp_df, 'FP': fp_df, 'TN': tn_df, 'FN': fn_df},
    kmer_sizes=[4, 6],
    downsample_options={'FP': {'rate': 0.5}},
    verbose=1
)
```

## 📁 Directory Structure

```
features/
├── gene_selection.py          # Gene selection strategies and utilities
├── kmer_filtering.py          # K-mer filtering strategies  
├── kmer_filter_config.py      # K-mer filtering configuration
├── kmer_filtering_demo.py     # K-mer filtering demonstration
├── kmer_features.py           # K-mer feature extraction
├── sequence_featurization.py  # Sequence processing and harmonization
├── genomic_features.py        # Genomic feature extraction
├── feature_enrichment.py      # Feature enrichment pipeline
├── feature_merger.py          # Feature merging utilities
├── transcript_alignments.py   # Transcript alignment features
└── README.md                  # This file
```

## 🔧 Integration Examples

### Gene Selection in CV Scripts

Gene selection is widely used across training scripts for memory optimization:

```python
# run_gene_cv_sigmoid.py - Binary classification with gene selection
subset_df, gene_ids = subset_analysis_sequences(
    data_handler,
    n_genes=1000,
    subset_policy="error_total",
    additional_gene_ids=custom_genes,
    use_effective_counts=True
)

# run_loco_cv_multiclass_scalable.py - LOCO CV with gene selection  
subset_df, gene_ids = subset_analysis_sequences(
    data_handler,
    n_genes=5000,
    subset_policy="error_fn",  # Focus on false negatives
    verbose=1
)
```

### K-mer Filtering Integration

```python
from meta_spliceai.splice_engine.meta_models.features import (
    add_kmer_filtering_args,
    KmerFilterConfig, 
    integrate_kmer_filtering_in_cv
)

# Add arguments to parser
add_kmer_filtering_args(parser)

# Create configuration from arguments
config = KmerFilterConfig.from_args(args)

# Apply filtering during CV
X_filtered, filtered_features = integrate_kmer_filtering_in_cv(
    X, y, feature_names, config
)
```

### Feature Enrichment in Training Pipeline

```python
# incremental_builder.py - Feature enrichment during dataset assembly
from meta_spliceai.splice_engine.meta_models.features import merge_contextual_features

# Enrich positions with genomic features
enriched_df = merge_contextual_features(
    pos_df=positions_enhanced,
    gene_meta_df=gene_metadata,
    kmer_df=kmer_features
)
```

## 📊 Performance Expectations

### Gene Selection Memory Impact

| Genes Selected | Memory Usage | Typical Use Case |
|----------------|--------------|------------------|
| 1,000 | ~2-4 GB | Quick iteration/debugging |
| 5,000 | ~8-12 GB | Development CV runs |
| 10,000 | ~16-24 GB | Production training |
| 20,000+ | ~32+ GB | Full-scale experiments |

### K-mer Filtering Reduction Rates

| Strategy | Small Dataset | Medium Dataset | Large Dataset |
|----------|---------------|----------------|---------------|
| `motif` | 10-30% | 20-40% | 30-50% |
| `mi` | 30-60% | 50-80% | 70-90% |
| `sparsity` | 20-50% | 40-70% | 60-85% |
| `ensemble` | 40-70% | 60-85% | 80-95% |

## 🔗 Complete Pipeline Example

Here's how the different systems work together in a typical training workflow:

```python
# 1. Gene Selection - Select informative genes
from meta_spliceai.splice_engine.meta_models.features import subset_analysis_sequences

subset_df, gene_ids = subset_analysis_sequences(
    data_handler,
    n_genes=1000,                    # Memory optimization
    subset_policy="error_total",     # Focus on challenging genes
    use_effective_counts=True        # Avoid position duplicates
)

# 2. Feature Enrichment - Add genomic context
from meta_spliceai.splice_engine.meta_models.features import merge_contextual_features

enriched_df = merge_contextual_features(
    pos_df=positions_df,
    gene_meta_df=gene_metadata,      # Gene-level features
    kmer_df=kmer_features           # K-mer features
)

# 3. K-mer Filtering - Optimize feature set
from meta_spliceai.splice_engine.meta_models.features import integrate_kmer_filtering_in_cv

X_filtered, filtered_features = integrate_kmer_filtering_in_cv(
    X, y, feature_names,
    config=KmerFilterConfig(strategy='ensemble')  # Smart filtering
)

# 4. Training - Use optimized data for CV
# (Reduced memory, focused genes, optimized features)
```

## 🚀 Quick Start Commands

```bash
# Test gene selection with different policies
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000/master \
    --n-genes 1000 \
    --gene-subset-policy error_total \
    --out-dir models/test_gene_selection

# Test k-mer filtering demo
python -m meta_spliceai.splice_engine.meta_models.features.kmer_filtering_demo

# CV with both gene selection and k-mer filtering
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000/master \
    --n-genes 1000 \
    --gene-subset-policy error_total \
    --filter-kmers \
    --kmer-filter-strategy ensemble \
    --out-dir models/optimized_training
```

## 📚 Documentation

For detailed documentation, see:
- `docs/kmer_filtering_guide.md` - Comprehensive k-mer filtering guide
- `tests/test_kmer_filtering.py` - K-mer filtering unit tests
- Individual module docstrings for detailed API documentation 