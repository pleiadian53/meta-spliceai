# Base Model Workflow Guide

**Complete guide for using SpliceAI and OpenSpliceAI base models**

---

## Overview

MetaSpliceAI supports multiple base models with **automatic genomic resource routing**:

| Base Model | Build | Annotations | Splice Sites | Framework |
|------------|-------|-------------|--------------|-----------|
| **SpliceAI** | GRCh37 | Ensembl (release 87) | ~2M | Keras |
| **OpenSpliceAI** | GRCh38 | MANE v1.3 | ~370K | PyTorch |

The system **automatically**:
- ✅ Routes to correct genomic build
- ✅ Loads correct splice site annotations
- ✅ Uses correct artifact paths
- ✅ Handles framework differences

---

## Quick Start

### Using SpliceAI (GRCh37)

```python
from meta_spliceai import run_base_model_predictions

# Simple usage
results = run_base_model_predictions(
    base_model='spliceai',  # GRCh37/Ensembl
    target_genes=['BRCA1', 'TP53']
)

print(f"Analyzed {results['positions'].height:,} positions")
print(f"Build: {results['artifact_manager']['summary']['build']}")  # GRCh37
```

### Using OpenSpliceAI (GRCh38)

```python
from meta_spliceai import run_base_model_predictions

# Simple usage
results = run_base_model_predictions(
    base_model='openspliceai',  # GRCh38/MANE
    target_genes=['BRCA1', 'TP53']
)

print(f"Analyzed {results['positions'].height:,} positions")
print(f"Build: {results['artifact_manager']['summary']['build']}")  # GRCh38
```

**That's it!** The system handles everything else automatically.

---

## How It Works

### 1. User Specifies Base Model

```python
config = BaseModelConfig(base_model='openspliceai')
```

### 2. System Routes Automatically

The system determines:
- **Genomic Build**: GRCh38 (for OpenSpliceAI)
- **Annotation Source**: MANE (for OpenSpliceAI)
- **Splice Sites File**: `data/mane/GRCh38/splice_sites_enhanced.tsv`
- **Artifact Path**: `data/mane/GRCh38/openspliceai_eval/`

### 3. Workflow Runs Seamlessly

```python
# Loads OpenSpliceAI PyTorch models
models, metadata = load_base_model_ensemble('openspliceai')

# Uses MANE splice site annotations
ss_annotations = load_splice_sites('GRCh38_MANE')

# Generates predictions
predictions = predict_splice_sites_for_genes(sequences, models)

# Saves to correct location
save_to('data/mane/GRCh38/openspliceai_eval/')
```

---

## Configuration Options

### Basic Configuration

```python
from meta_spliceai import BaseModelConfig

config = BaseModelConfig(
    base_model='openspliceai',  # or 'spliceai'
    mode='test',                # 'test' or 'production'
    threshold=0.5,
    consensus_window=2
)
```

### Advanced Configuration

```python
config = BaseModelConfig(
    # Base model selection
    base_model='openspliceai',
    
    # Execution mode
    mode='test',                    # test: overwritable artifacts
    coverage='gene_subset',         # gene_subset, chromosome, full_genome
    test_name='my_experiment',      # test identifier
    
    # Prediction parameters
    threshold=0.5,                  # splice site threshold
    consensus_window=2,             # consensus calling window
    error_window=500,               # error analysis window
    use_auto_position_adjustments=True,  # auto-detect position shifts
    
    # Data preparation
    do_extract_annotations=False,   # use existing annotations
    do_extract_splice_sites=False,  # use existing splice sites
    do_extract_sequences=False      # use existing sequences
)
```

---

## Complete Examples

### Example 1: Test Mode (Overwritable)

```python
from meta_spliceai import run_base_model_predictions

# Test with OpenSpliceAI
results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1', 'TP53', 'EGFR'],
    mode='test',
    test_name='openspliceai_test_run',
    verbosity=1
)

# Results are saved to:
# data/mane/GRCh38/openspliceai_eval/tests/openspliceai_test_run/
```

### Example 2: Production Mode (Immutable)

```python
from meta_spliceai import run_base_model_predictions

# Production run (full chromosome)
results = run_base_model_predictions(
    base_model='openspliceai',
    target_chromosomes=['21'],
    mode='production',
    coverage='chromosome',
    verbosity=1
)

# Results are saved to:
# data/mane/GRCh38/openspliceai_eval/meta_models/predictions/
# (immutable - won't overwrite existing files)
```

### Example 3: Gene Category Analysis

```python
from meta_spliceai import run_base_model_predictions
import polars as pl

# Sample genes by category
protein_coding_genes = ['BRCA1', 'TP53', 'EGFR', 'MYC', 'KRAS']
lncrna_genes = ['MALAT1', 'NEAT1', 'XIST']

# Test on protein-coding genes
pc_results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=protein_coding_genes,
    mode='test',
    test_name='protein_coding_test'
)

# Test on lncRNA genes
lnc_results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=lncrna_genes,
    mode='test',
    test_name='lncrna_test'
)

# Compare performance
print(f"Protein-coding F1: {calculate_f1(pc_results['positions']):.3f}")
print(f"lncRNA F1: {calculate_f1(lnc_results['positions']):.3f}")
```

### Example 4: Model Comparison

```python
from meta_spliceai import run_base_model_predictions

genes = ['BRCA1', 'TP53']

# Run with SpliceAI (GRCh37)
spliceai_results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=genes,
    mode='test',
    test_name='spliceai_comparison'
)

# Run with OpenSpliceAI (GRCh38)
openspliceai_results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=genes,
    mode='test',
    test_name='openspliceai_comparison'
)

# Note: Direct coordinate comparison requires liftOver
# But can compare F1 scores, precision, recall on same genes
```

---

## Understanding Results

### Results Dictionary

```python
results = {
    'success': True,
    
    # DataFrames
    'positions': pl.DataFrame,        # All analyzed positions
    'error_analysis': pl.DataFrame,   # FP/FN positions
    'analysis_sequences': pl.DataFrame,  # Sequences for analysis
    
    # Paths
    'paths': {
        'eval_dir': str,
        'artifacts_dir': str,
        'positions_artifact': str,
        'errors_artifact': str
    },
    
    # Artifact Manager
    'artifact_manager': {
        'mode': 'test',
        'coverage': 'gene_subset',
        'test_name': 'my_test',
        'summary': {
            'build': 'GRCh38',
            'source': 'mane',
            'base_model': 'openspliceai'
        }
    }
}
```

### Positions DataFrame

```python
positions_df = results['positions']

# Columns:
# - gene_id: Ensembl gene ID
# - position: Genomic position
# - splice_type: 'donor' or 'acceptor'
# - error_type: 'TP', 'TN', 'FP', 'FN'
# - prob_donor: Donor probability
# - prob_acceptor: Acceptor probability
# - prob_neither: Neither probability
# - strand: '+' or '-'
# - chrom: Chromosome
# - context_*: Context features
# - derived features: donor_diff, donor_surge, etc.

# Calculate F1 score
tp = positions_df.filter(pl.col('error_type') == 'TP').height
fp = positions_df.filter(pl.col('error_type') == 'FP').height
fn = positions_df.filter(pl.col('error_type') == 'FN').height

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"F1 Score: {f1:.3f}")
```

---

## Genomic Resources

### Directory Structure

```
data/
├── ensembl/GRCh37/              # SpliceAI resources
│   ├── Homo_sapiens.GRCh37.87.gtf
│   ├── Homo_sapiens.GRCh37.dna.primary_assembly.fa
│   ├── splice_sites_enhanced.tsv  (~2M sites)
│   └── spliceai_eval/            # SpliceAI artifacts
│
└── mane/GRCh38/                 # OpenSpliceAI resources
    ├── MANE.GRCh38.v1.3.refseq_genomic.gtf
    ├── GCF_000001405.40_GRCh38.p14_genomic.fna
    ├── splice_sites_enhanced.tsv  (~370K sites)
    └── openspliceai_eval/        # OpenSpliceAI artifacts
```

### Automatic Routing

```python
# User code
config = BaseModelConfig(base_model='openspliceai')

# System automatically determines:
# - Build: GRCh38
# - Source: mane
# - Data dir: data/mane/GRCh38/
# - Splice sites: data/mane/GRCh38/splice_sites_enhanced.tsv
# - Artifacts: data/mane/GRCh38/openspliceai_eval/
```

---

## Testing Scripts

### Test OpenSpliceAI

```bash
# Test with gene categories
python scripts/testing/test_openspliceai_gene_categories.py

# Output:
# - results/openspliceai_gene_categories_*/
#   - sampled_genes_by_category.tsv
#   - category_performance_summary.tsv
#   - results_summary.json
```

### Compare Models

```bash
# Compare SpliceAI vs OpenSpliceAI
python scripts/testing/compare_base_models.py \
  --spliceai-dir results/base_model_gene_categories_test \
  --openspliceai-dir results/openspliceai_gene_categories_*

# Output:
# - results/base_model_comparison/model_comparison.tsv
```

---

## Best Practices

### 1. Choose the Right Base Model

**Use SpliceAI when**:
- Working with GRCh37 data
- Need comprehensive isoform coverage
- Research applications
- Comparing with published SpliceAI results

**Use OpenSpliceAI when**:
- Working with GRCh38 data
- Clinical applications (MANE focus)
- Need PyTorch framework
- Want canonical transcript predictions

### 2. Use Test Mode for Development

```python
# Test mode: artifacts are overwritable
results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1'],
    mode='test',
    test_name='development_test'
)
```

### 3. Use Production Mode for Final Results

```python
# Production mode: artifacts are immutable
results = run_base_model_predictions(
    base_model='openspliceai',
    target_chromosomes=['21', '22'],
    mode='production',
    coverage='chromosome'
)
```

### 4. Validate on Gene Categories

```python
# Test on different gene types
protein_coding = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=protein_coding_genes
)

lncrna = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=lncrna_genes
)

# Compare performance
```

---

## Troubleshooting

### Issue: Models not found

**Error**: `FileNotFoundError: OpenSpliceAI models not found`

**Solution**:
```bash
./scripts/base_model/download_openspliceai_models.sh
```

### Issue: Splice sites not found

**Error**: `FileNotFoundError: splice_sites_enhanced.tsv not found`

**Solution**:
```bash
./scripts/setup/download_grch38_mane_data.sh
```

### Issue: Wrong genomic build

**Symptom**: Predictions use wrong resources

**Solution**: Ensure `base_model` parameter is set correctly:
```python
# For GRCh37
config = BaseModelConfig(base_model='spliceai')

# For GRCh38
config = BaseModelConfig(base_model='openspliceai')
```

---

## Summary

### ✅ Clean User Interface

**Single parameter controls everything**:
```python
base_model='openspliceai'  # or 'spliceai'
```

**System handles**:
- ✅ Genomic build routing (GRCh37 vs GRCh38)
- ✅ Annotation source (Ensembl vs MANE)
- ✅ Splice site definitions (2M vs 370K)
- ✅ Artifact management
- ✅ Framework differences (Keras vs PyTorch)

### ✅ Consistent API

**Same interface for all models**:
```python
# SpliceAI
run_base_model_predictions(base_model='spliceai', ...)

# OpenSpliceAI
run_base_model_predictions(base_model='openspliceai', ...)

# Future models
run_base_model_predictions(base_model='pangolin', ...)
```

### ✅ Automatic Resource Management

**No manual path configuration needed**:
- Genomic resources loaded automatically
- Artifacts saved to correct locations
- Schema standardization applied
- Build compatibility ensured

---

**Ready to use!** Start with the Quick Start examples above. 🚀

*Last Updated: 2025-11-06*


