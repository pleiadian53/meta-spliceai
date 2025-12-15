# Using run_base_model.py for Full Coverage Testing

**Date**: November 9, 2025  
**Status**: Production Ready

---

## Overview

Yes! `run_base_model.py` has **all the necessary mechanisms** to run any established base model with full coverage mode enabled or disabled. The only difference from `compare_base_models_robust.py` is **gene selection** - you need to provide your own gene list.

---

## Quick Answer to Your Question

> "Will we be able to do the same via run_base_model.py?"

**YES!** ✅

`run_base_model.py` already has:
- ✅ `base_model` parameter (switch between models)
- ✅ `save_nucleotide_scores` parameter (enable/disable full coverage)
- ✅ `target_genes` parameter (specify genes to test)
- ✅ All other configuration options

**The only thing you need to do**: Provide the gene list yourself.

---

## Direct Usage Examples

### Example 1: Quick Test with Known Genes

```python
from meta_spliceai import run_base_model_predictions

# Test SpliceAI with full coverage on 5 genes
results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1', 'TP53', 'EGFR', 'MYC', 'KRAS'],
    save_nucleotide_scores=True,  # Enable full coverage
    mode='test',
    coverage='gene_subset',
    test_name='full_coverage_test',
    threshold=0.5,
    verbosity=1,
    no_tn_sampling=True
)

# Access results
positions = results['positions']
nucleotide_scores = results['nucleotide_scores']  # NEW: Full coverage data
gene_manifest = results['gene_manifest']

print(f"Positions: {positions.height:,}")
print(f"Nucleotide scores: {nucleotide_scores.height:,}")
```

---

### Example 2: Compare Models on Same Genes

```python
from meta_spliceai import run_base_model_predictions

# Define test genes
test_genes = ['BRCA1', 'TP53', 'EGFR', 'MYC', 'KRAS']

# Test SpliceAI
spliceai_results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=test_genes,
    save_nucleotide_scores=True,
    mode='test',
    test_name='spliceai_full_coverage',
    verbosity=1
)

# Test OpenSpliceAI (same genes!)
openspliceai_results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=test_genes,
    save_nucleotide_scores=True,
    mode='test',
    test_name='openspliceai_full_coverage',
    verbosity=1
)

# Compare
print(f"SpliceAI nucleotide scores: {spliceai_results['nucleotide_scores'].height:,}")
print(f"OpenSpliceAI nucleotide scores: {openspliceai_results['nucleotide_scores'].height:,}")
```

---

### Example 3: Standard Mode (No Nucleotide Scores)

```python
from meta_spliceai import run_base_model_predictions

# Standard mode - positions only
results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1', 'TP53'],
    save_nucleotide_scores=False,  # Default - positions only
    mode='test',
    test_name='standard_test',
    verbosity=1
)

# Only positions, no nucleotide scores
positions = results['positions']
nucleotide_scores = results.get('nucleotide_scores')  # Will be empty DataFrame

print(f"Positions: {positions.height:,}")
print(f"Nucleotide scores: {nucleotide_scores.height if nucleotide_scores is not None else 0}")
```

---

### Example 4: With Gene Sampling (Like Robust Script)

```python
from meta_spliceai import run_base_model_predictions
from meta_spliceai.system.genomic_resources import Registry
import polars as pl

# Step 1: Sample genes (like compare_base_models_robust.py does)
registry = Registry(build='GRCh37', release='87')
gene_features = pl.read_csv(
    str(registry.data_dir / "gene_features.tsv"),
    separator='\t',
    schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8}
)

# Sample protein-coding genes
protein_coding = gene_features.filter(
    (pl.col('gene_type') == 'protein_coding') &
    (pl.col('gene_length') >= 5_000) &
    (pl.col('gene_length') <= 500_000)
)

# Random sample (no seed = different genes each run)
sampled_genes = protein_coding.sample(n=10)['gene_name'].to_list()

print(f"Sampled genes: {', '.join(sampled_genes)}")

# Step 2: Run predictions with full coverage
results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=sampled_genes,
    save_nucleotide_scores=True,  # Full coverage
    mode='test',
    test_name='sampled_genes_full_coverage',
    verbosity=1
)

print(f"Nucleotide scores: {results['nucleotide_scores'].height:,}")
```

---

### Example 5: Production Run with Full Coverage

```python
from meta_spliceai import run_base_model_predictions

# Production run on specific genes with full coverage
results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1', 'BRCA2', 'TP53', 'PTEN'],
    save_nucleotide_scores=True,
    mode='production',  # Immutable artifacts
    coverage='gene_subset',
    threshold=0.5,
    use_auto_position_adjustments=True,
    verbosity=1
)

# Save nucleotide scores for downstream analysis
nucleotide_scores = results['nucleotide_scores']
nucleotide_scores.write_csv('production_nucleotide_scores.tsv', separator='\t')
```

---

## Command-Line Usage

You can also create a simple script:

```python
#!/usr/bin/env python3
"""
test_full_coverage.py - Test full coverage mode via run_base_model.py
"""
import sys
from meta_spliceai import run_base_model_predictions

# Get genes from command line or use defaults
if len(sys.argv) > 1:
    genes = sys.argv[1:]
else:
    genes = ['BRCA1', 'TP53', 'EGFR']

print(f"Testing full coverage on: {', '.join(genes)}")

# Run with full coverage
results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=genes,
    save_nucleotide_scores=True,
    mode='test',
    test_name='cli_full_coverage',
    verbosity=1
)

print(f"\nResults:")
print(f"  Positions: {results['positions'].height:,}")
print(f"  Nucleotide scores: {results['nucleotide_scores'].height:,}")
print(f"  Genes processed: {len(results['gene_manifest'].filter(pl.col('status') == 'processed'))}")
```

**Usage**:
```bash
# Default genes
python test_full_coverage.py

# Custom genes
python test_full_coverage.py BRCA1 TP53 KRAS MYC
```

---

## Comparison: Script vs. Direct API

### Using `compare_base_models_robust.py`

**Pros**:
- ✅ Automatic gene sampling (intersection-based)
- ✅ Runs BOTH models automatically
- ✅ Performance comparison built-in
- ✅ Handles missing genes gracefully
- ✅ Complete test harness

**Cons**:
- ❌ Need to edit script to change configuration
- ❌ Less flexible for custom workflows

**Use when**:
- Comparing models
- Production validation
- Standardized testing

---

### Using `run_base_model.py` Directly

**Pros**:
- ✅ Full programmatic control
- ✅ Easy to integrate into custom workflows
- ✅ Can use in Jupyter notebooks
- ✅ Flexible gene selection
- ✅ Can process results immediately

**Cons**:
- ❌ Need to provide gene list yourself
- ❌ Need to call separately for each model
- ❌ Need to implement comparison logic yourself

**Use when**:
- Custom analysis workflows
- Jupyter notebook exploration
- Integration with other tools
- Single model testing
- Programmatic control needed

---

## Key Parameters

### Essential Parameters

```python
run_base_model_predictions(
    base_model='spliceai',           # or 'openspliceai'
    target_genes=['BRCA1', ...],     # YOUR gene list
    save_nucleotide_scores=True,     # Enable full coverage
    mode='test',                     # or 'production'
    verbosity=1                      # 0=minimal, 1=normal, 2=detailed
)
```

### Full Coverage Control

```python
# Disable (default) - positions only
save_nucleotide_scores=False

# Enable - full coverage (all nucleotides)
save_nucleotide_scores=True
```

### Model Selection

```python
# SpliceAI (GRCh37/Ensembl)
base_model='spliceai'

# OpenSpliceAI (GRCh38/MANE)
base_model='openspliceai'
```

---

## Output Structure

### Standard Mode (`save_nucleotide_scores=False`)

```python
results = {
    'success': True,
    'positions': DataFrame,          # TP/FP/TN/FN positions
    'error_analysis': DataFrame,     # Errors only
    'gene_manifest': DataFrame,      # Gene processing status
    'nucleotide_scores': DataFrame,  # EMPTY
    'paths': {...},
    'manifest_summary': {...}
}
```

### Full Coverage Mode (`save_nucleotide_scores=True`)

```python
results = {
    'success': True,
    'positions': DataFrame,          # TP/FP/TN/FN positions
    'error_analysis': DataFrame,     # Errors only
    'gene_manifest': DataFrame,      # Gene processing status
    'nucleotide_scores': DataFrame,  # FULL - all nucleotides ← NEW
    'paths': {
        'nucleotide_scores_artifact': 'path/to/nucleotide_scores.tsv'  ← NEW
    },
    'manifest_summary': {...}
}
```

---

## Performance Considerations

| Configuration | Genes | Runtime | Data Volume | Memory |
|--------------|-------|---------|-------------|--------|
| Standard (5 genes) | 5 | ~1 min | ~2 MB | ~200 MB |
| Full Coverage (5 genes) | 5 | ~3 min | ~50 MB | ~500 MB |
| Standard (20 genes) | 20 | ~5 min | ~10 MB | ~500 MB |
| Full Coverage (20 genes) | 20 | ~15 min | ~200 MB | ~2 GB |

---

## Summary

### Answer to Your Question

> "Will we be able to do the same via run_base_model.py?"

**YES!** ✅ `run_base_model.py` has **ALL** the necessary mechanisms:

1. ✅ **Base model selection**: `base_model='spliceai'` or `'openspliceai'`
2. ✅ **Full coverage control**: `save_nucleotide_scores=True` or `False`
3. ✅ **Gene specification**: `target_genes=[...]`
4. ✅ **All configuration options**: `mode`, `threshold`, `verbosity`, etc.

### The Only Difference

**`compare_base_models_robust.py`**:
- Handles gene sampling automatically
- Runs both models
- Compares results

**`run_base_model.py`**:
- You provide the gene list
- You call it separately for each model
- You implement comparison logic

### When to Use Each

**Use `compare_base_models_robust.py`** when:
- Comparing models
- Standardized testing
- Production validation

**Use `run_base_model.py`** when:
- Custom workflows
- Jupyter notebooks
- Programmatic control
- Single model testing

---

*Last Updated: November 9, 2025*  
*Status: Production Ready*

