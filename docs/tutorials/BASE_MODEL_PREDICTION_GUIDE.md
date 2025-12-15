# Base Model Prediction Guide

**Tutorial: Using the Base Model for Splice Site Prediction**

---

## Table of Contents

1. [Overview](#overview)
2. [Understanding Modes and Artifacts](#understanding-modes-and-artifacts)
3. [Use Cases](#use-cases)
4. [Quick Start Examples](#quick-start-examples)
5. [Configuration Reference](#configuration-reference)
6. [Output Structure](#output-structure)
7. [Best Practices](#best-practices)
8. [FAQ](#faq)

---

## Overview

The **base model prediction workflow** (`splice_prediction_workflow.py`) is the core component for running splice site predictions using base models like SpliceAI or OpenSpliceAI. This tutorial explains how to use it for various scenarios.

### What is the Base Model Pass?

The "base model pass" refers to running splice site predictions using a pre-trained model (like SpliceAI) on genomic sequences to identify potential splice sites. This is distinct from:

- **Meta-model inference**: Using a trained meta-model to correct base model predictions
- **Training data generation**: Collecting predictions for meta-model training

### Supported Base Models

Currently supported:
- ✅ **SpliceAI** (default)
- 🔄 **OpenSpliceAI** (in development)

---

## Understanding Modes and Artifacts

### Execution Modes

The workflow supports two execution modes that control artifact management:

#### 1. Test Mode (Default)

```python
mode='test'
```

**Characteristics**:
- Artifacts are **overwritable** (always replaced on each run)
- Stored in test-specific subdirectories
- Ideal for development, validation, and experimentation
- Default behavior when `coverage='gene_subset'`

**Artifact Location**:
```
data/ensembl/GRCh37/spliceai_eval/
└── tests/
    └── {test_name}/
        └── meta_models/
            └── predictions/
                ├── full_splice_positions_enhanced.tsv
                └── full_splice_errors.tsv
```

#### 2. Production Mode

```python
mode='production'
```

**Characteristics**:
- Artifacts are **immutable** (never overwritten unless forced)
- Stored in production directories
- Used for full genome coverage
- Automatically activated when `coverage='full_genome'`

**Artifact Location**:
```
data/ensembl/GRCh37/spliceai_eval/
└── meta_models/
    ├── training_data/
    │   ├── analysis_sequences_chr*.tsv
    │   └── full_splice_positions_enhanced.tsv
    └── models/
        └── meta_model_v1.pkl
```

### Coverage Types

| Coverage | Description | Typical Use | Mode |
|----------|-------------|-------------|------|
| `gene_subset` | Small set of genes | Testing, validation | test |
| `chromosome` | Single chromosome | Development, debugging | test |
| `full_genome` | All chromosomes | Production, training | production |

### Default Behavior

**When you run `splice_prediction_workflow.py` without specifying mode**:

```python
# Default configuration
config = SpliceAIConfig()  # mode='test', coverage='gene_subset'
```

- **Mode**: `test`
- **Coverage**: `gene_subset`
- **Test Name**: Auto-generated (e.g., `test_20251105_141420`)
- **Overwrite**: Always (test mode behavior)
- **Location**: `data/.../tests/{test_name}/meta_models/predictions/`

---

## Use Cases

### Use Case 1: Quick Gene-Level Prediction

**Scenario**: You want to predict splice sites for a few specific genes.

**Example**:
```python
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig

# Configure for specific genes
config = SpliceAIConfig(
    mode='test',
    coverage='gene_subset',
    test_name='my_genes_test',
    threshold=0.5,
    consensus_window=2,
    error_window=500
)

# Run predictions
results = run_enhanced_splice_prediction_workflow(
    config=config,
    target_genes=['BRCA1', 'TP53', 'EGFR'],  # Your genes of interest
    verbosity=1
)

# Access results
positions_df = results['positions']
error_analysis_df = results['error_analysis']
```

**Output Location**:
```
data/ensembl/GRCh37/spliceai_eval/tests/my_genes_test/meta_models/predictions/
├── full_splice_positions_enhanced.tsv
└── full_splice_errors.tsv
```

---

### Use Case 2: Validation Testing

**Scenario**: You want to validate model performance on a diverse gene set.

**Example**:
```python
# Sample diverse genes
protein_coding_genes = ['BRCA1', 'TP53', 'EGFR', 'KRAS', 'MYC']
lncrna_genes = ['MALAT1', 'NEAT1', 'XIST']

all_genes = protein_coding_genes + lncrna_genes

# Configure for validation
config = SpliceAIConfig(
    mode='test',
    coverage='gene_subset',
    test_name='validation_run',
    threshold=0.5,
    use_auto_position_adjustments=True  # Enable coordinate adjustment
)

# Run with no TN sampling for complete evaluation
results = run_enhanced_splice_prediction_workflow(
    config=config,
    target_genes=all_genes,
    no_tn_sampling=True,  # Keep all true negatives for accurate metrics
    verbosity=1
)

# Analyze performance
print(f"Total positions: {results['positions'].height:,}")
print(f"Errors analyzed: {results['error_analysis'].height:,}")
```

---

### Use Case 3: Chromosome-Level Analysis

**Scenario**: You want to process all genes on a specific chromosome.

**Example**:
```python
config = SpliceAIConfig(
    mode='test',
    coverage='chromosome',
    test_name='chr21_analysis',
    chromosomes=['21'],  # Specify chromosome
    threshold=0.5
)

# Run on entire chromosome
results = run_enhanced_splice_prediction_workflow(
    config=config,
    target_chromosomes=['21'],
    verbosity=1
)
```

---

### Use Case 4: Full Genome Production Run

**Scenario**: You want to generate complete predictions for meta-model training.

**Example**:
```python
config = SpliceAIConfig(
    mode='production',  # Immutable artifacts
    coverage='full_genome',
    threshold=0.5,
    use_auto_position_adjustments=True,
    do_extract_annotations=True,
    do_extract_splice_sites=True,
    do_extract_sequences=True
)

# Run on all chromosomes
results = run_enhanced_splice_prediction_workflow(
    config=config,
    verbosity=1
)
```

**Output Location**:
```
data/ensembl/GRCh37/spliceai_eval/meta_models/
├── training_data/
│   ├── analysis_sequences_chr1_1_500.tsv
│   ├── analysis_sequences_chr1_501_1000.tsv
│   ├── ...
│   └── full_splice_positions_enhanced.tsv
└── predictions/
    ├── full_splice_positions_enhanced.tsv
    └── full_splice_errors.tsv
```

---

### Use Case 5: Comparing Different Base Models

**Scenario**: You want to compare SpliceAI vs. OpenSpliceAI predictions.

**Example**:
```python
# Test SpliceAI
config_spliceai = SpliceAIConfig(
    gtf_file='data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf',
    genome_fasta='data/ensembl/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa',
    mode='test',
    test_name='spliceai_comparison'
)

results_spliceai = run_enhanced_splice_prediction_workflow(
    config=config_spliceai,
    target_genes=['BRCA1', 'TP53'],
    verbosity=1
)

# Test OpenSpliceAI (when available)
config_openspliceai = SpliceAIConfig(
    gtf_file='data/mane/GRCh38/MANE.v1.0.gtf',
    genome_fasta='data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa',
    mode='test',
    test_name='openspliceai_comparison'
)

# results_openspliceai = run_enhanced_splice_prediction_workflow(
#     config=config_openspliceai,
#     target_genes=['BRCA1', 'TP53'],
#     verbosity=1
# )
```

---

## Quick Start Examples

### Example 1: Minimal Configuration

```python
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)

# Simplest possible usage (uses all defaults)
results = run_enhanced_splice_prediction_workflow(
    target_genes=['BRCA1']
)

print(f"Predictions: {results['positions'].height}")
```

### Example 2: Custom Thresholds

```python
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig

config = SpliceAIConfig(
    threshold=0.2,  # Lower threshold for more sensitive detection
    consensus_window=5,  # Wider window for consensus
    error_window=1000  # Larger window for error analysis
)

results = run_enhanced_splice_prediction_workflow(
    config=config,
    target_genes=['BRCA1', 'TP53']
)
```

### Example 3: With Data Preparation

```python
config = SpliceAIConfig(
    do_extract_annotations=True,  # Extract gene annotations
    do_extract_splice_sites=True,  # Extract splice site annotations
    do_extract_sequences=True,  # Extract genomic sequences
    mode='test',
    test_name='full_pipeline_test'
)

results = run_enhanced_splice_prediction_workflow(
    config=config,
    target_genes=['BRCA1', 'TP53', 'EGFR']
)
```

---

## Configuration Reference

### Essential Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gtf_file` | str | Auto-detected | GTF annotation file |
| `genome_fasta` | str | Auto-detected | Genome FASTA file |
| `mode` | str | `'test'` | Execution mode: `'test'` or `'production'` |
| `coverage` | str | `'gene_subset'` | Coverage: `'gene_subset'`, `'chromosome'`, `'full_genome'` |
| `test_name` | str | Auto-generated | Test identifier for test mode |
| `threshold` | float | `0.5` | Splice site score threshold |
| `consensus_window` | int | `2` | Window for consensus calling |
| `error_window` | int | `500` | Window for error analysis |

### Data Preparation Flags

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `do_extract_annotations` | bool | `False` | Extract gene annotations from GTF |
| `do_extract_splice_sites` | bool | `False` | Extract splice site annotations |
| `do_extract_sequences` | bool | `False` | Extract genomic sequences |
| `do_find_overlaping_genes` | bool | `False` | Find overlapping genes |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_auto_position_adjustments` | bool | `True` | Auto-detect coordinate adjustments |
| `chromosomes` | List[str] | `None` | Specific chromosomes to process |
| `test_mode` | bool | `False` | Legacy test mode (use `mode` instead) |
| `format` | str | `'tsv'` | Output format |
| `seq_format` | str | `'parquet'` | Sequence file format |

---

## Output Structure

### Primary Outputs

#### 1. Full Splice Positions Enhanced (`full_splice_positions_enhanced.tsv`)

Contains all analyzed positions with predictions and features.

**Key Columns**:
- `gene_id`, `transcript_id`: Gene/transcript identifiers
- `chrom`, `position`, `strand`: Genomic coordinates
- `splice_type`: `'donor'` or `'acceptor'`
- `error_type`: `'TP'`, `'FP'`, `'FN'`, or `'TN'`
- `prob_donor`, `prob_acceptor`, `prob_neither`: SpliceAI probabilities
- `context_*`: Contextual probability features
- `donor_diff_*`, `donor_surge_*`: Derived features

**Example**:
```
gene_id          position  splice_type  error_type  prob_donor  prob_acceptor
ENSG00000012048  32889617  donor        TP          0.9856      0.0023
ENSG00000012048  32889805  acceptor     TP          0.0012      0.9912
ENSG00000012048  32890000  donor        FP          0.6234      0.0045
```

#### 2. Full Splice Errors (`full_splice_errors.tsv`)

Contains only positions with errors (FP, FN) for focused analysis.

**Key Columns**:
- `error_type`: `'FP'` or `'FN'`
- `gene_id`, `position`, `splice_type`, `strand`
- `transcript_id`: Associated transcript(s)
- `window_start`, `window_end`: Analysis window

#### 3. Analysis Sequences (Chunk Files)

Contains ±250bp sequences around each position for downstream analysis.

**Location**: `predictions/analysis_sequences_chr*_*.tsv`

**Key Columns**:
- `gene_id`, `position`, `strand`, `splice_type`
- `sequence`: 501bp sequence (±250bp around position)
- `error_type`: Classification
- `window_start`, `window_end`: Sequence coordinates

---

## Best Practices

### 1. Start Small, Scale Up

```python
# ✅ Good: Start with a few genes
results = run_enhanced_splice_prediction_workflow(
    target_genes=['BRCA1', 'TP53'],
    verbosity=1
)

# ❌ Avoid: Starting with full genome without testing
# results = run_enhanced_splice_prediction_workflow()  # Processes everything!
```

### 2. Use Appropriate Modes

```python
# ✅ Good: Test mode for development
config = SpliceAIConfig(mode='test', test_name='my_experiment')

# ✅ Good: Production mode for final runs
config = SpliceAIConfig(mode='production', coverage='full_genome')

# ❌ Avoid: Production mode for testing
# config = SpliceAIConfig(mode='production', coverage='gene_subset')  # Wasteful
```

### 3. Enable Position Adjustments

```python
# ✅ Good: Auto-detect coordinate adjustments
config = SpliceAIConfig(use_auto_position_adjustments=True)

# ⚠️ Caution: Manual adjustments only if you know what you're doing
config = SpliceAIConfig(use_auto_position_adjustments=False)
```

### 4. Use No TN Sampling for Validation

```python
# ✅ Good: Keep all TNs for accurate metrics
results = run_enhanced_splice_prediction_workflow(
    config=config,
    no_tn_sampling=True  # For validation
)

# ✅ Good: Sample TNs for training data generation
results = run_enhanced_splice_prediction_workflow(
    config=config,
    no_tn_sampling=False  # Default, reduces dataset size
)
```

### 5. Monitor Progress

```python
# ✅ Good: Use appropriate verbosity
results = run_enhanced_splice_prediction_workflow(
    config=config,
    verbosity=1  # Normal progress messages
)

# For debugging
results = run_enhanced_splice_prediction_workflow(
    config=config,
    verbosity=2  # Detailed messages
)
```

---

## FAQ

### Q1: What's the difference between test mode and production mode?

**A**: 
- **Test mode**: Artifacts are overwritable, stored in test-specific directories. Use for development, validation, and experimentation.
- **Production mode**: Artifacts are immutable (protected from overwriting), stored in production directories. Use for final runs and training data generation.

### Q2: Why is lncRNA performance lower than protein-coding genes?

**A**: lncRNAs have different splicing patterns than protein-coding genes:
- More variable splice sites
- Non-canonical splicing mechanisms
- Lower conservation

**Run 1 Results**:
- Protein-coding F1: **94.87%** ✅
- lncRNA F1: **58.25%** ⚠️

This is **expected behavior**, not a bug. The meta-model is designed to correct these predictions.

### Q3: What happens if I run the same test twice?

**A**: Depends on the mode:

**Test Mode**:
```python
config = SpliceAIConfig(mode='test', test_name='my_test')
# First run: Creates artifacts
# Second run: Overwrites artifacts (always)
```

**Production Mode**:
```python
config = SpliceAIConfig(mode='production')
# First run: Creates artifacts
# Second run: Skips saving (artifacts exist, immutable)
```

### Q4: How do I process only specific chromosomes?

**A**:
```python
results = run_enhanced_splice_prediction_workflow(
    config=config,
    target_chromosomes=['21', '22', 'X']
)
```

### Q5: Can I use this for variant analysis?

**A**: Not directly. The base model pass is for reference genome predictions. For variant analysis, use the inference workflow (coming soon).

### Q6: What if my genes don't have splice sites?

**A**: The model will correctly predict low scores (near 0) for all positions. This is the expected behavior for genes like tRNA, rRNA, etc.

**Example from Run 1**:
- 5 edge case genes (no splice sites)
- Result: 0 false positives ✅
- Correct behavior validated

### Q7: How long does it take?

**Typical Times**:
- 1-5 genes: 1-5 minutes
- 10-30 genes: 10-30 minutes
- Single chromosome: 1-3 hours
- Full genome: 6-24 hours (depending on hardware)

### Q8: What are the output file sizes?

**Typical Sizes**:
- 30 genes: ~10-50 MB
- Single chromosome: ~500 MB - 2 GB
- Full genome: ~20-50 GB

---

## Related Documentation

- [Artifact Management Guide](../development/ARTIFACT_MANAGEMENT.md)
- [Schema Standardization](../development/SCHEMA_STANDARDIZATION_COMPLETE.md)
- [Validation Test Results](../testing/GENE_CATEGORY_TEST_RESULTS.md)
- [Production Readiness Checklist](../testing/PRODUCTION_READINESS_CHECKLIST.md)

---

**Last Updated**: 2025-11-05  
**Version**: 1.0

