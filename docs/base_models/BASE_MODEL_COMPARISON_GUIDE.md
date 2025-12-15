# Base Model Comparison Guide

**Date**: November 6, 2025  
**Purpose**: Demonstrate easy switching between base models and compare performance

---

## Overview

This guide demonstrates how to easily switch between different base models (SpliceAI and OpenSpliceAI) using the unified interface provided by `run_base_model.py`. The comparison test runs both models on the same gene set to enable direct performance comparison.

---

## Key Features Demonstrated

### 1. **Single-Parameter Model Switching**

The cleanest way to switch between base models is using the `base_model` parameter:

```python
from meta_spliceai.run_base_model import run_base_model_predictions

# Run SpliceAI (GRCh37/Ensembl)
results_spliceai = run_base_model_predictions(
    base_model='spliceai',  # ← Just change this parameter!
    target_genes=['BRCA1', 'TP53', 'EGFR'],
    mode='test',
    test_name='my_test'
)

# Run OpenSpliceAI (GRCh38/MANE)
results_openspliceai = run_base_model_predictions(
    base_model='openspliceai',  # ← Same interface, different model!
    target_genes=['BRCA1', 'TP53', 'EGFR'],
    mode='test',
    test_name='my_test'
)
```

### 2. **Automatic Genomic Resource Routing**

The system automatically routes to the correct genomic resources based on the `base_model` parameter:

| Base Model | Genomic Build | Annotation Source | GTF Path | FASTA Path |
|-----------|---------------|-------------------|---------|------------|
| `spliceai` | GRCh37 | Ensembl 87 | `data/ensembl/GRCh37/...` | `data/ensembl/GRCh37/...` |
| `openspliceai` | GRCh38 | MANE v1.3 | `data/mane/GRCh38/...` | `data/mane/GRCh38/...` |

**No manual path configuration needed!** The `BaseModelConfig.__post_init__()` method automatically sets the correct paths.

### 3. **Unified Workflow Interface**

Both models use the same workflow (`splice_prediction_workflow.py`), which:
- Handles framework differences (Keras vs PyTorch) internally
- Uses the same prediction interface (`predict_with_model()`)
- Produces the same output format
- Supports the same configuration options

---

## Comparison Test Scripts

### Robust Comparison (Recommended)
**Location**: `scripts/testing/compare_base_models_robust.py`

This is the **recommended approach** for comparing base models. It uses intersection-based sampling to ensure fair comparison.

#### What It Does

1. **Finds Gene Intersection** between GRCh37/Ensembl and GRCh38/MANE:
   - Loads gene features from both builds
   - Identifies genes present in both by gene name
   - Reports intersection size and coverage

2. **Samples from Intersection** by category:
   - Protein-coding genes (multi-exon, with splice sites)
   - lncRNA genes (with splice sites)
   - Genes without splice sites
   - Ensures sampled genes exist in BOTH builds

3. **Runs SpliceAI** on sampled genes:
   ```python
   spliceai_results = run_base_model_predictions(
       base_model='spliceai',
       target_genes=intersection_genes,
       ...
   )
   ```

4. **Runs OpenSpliceAI** on same genes:
   ```python
   openspliceai_results = run_base_model_predictions(
       base_model='openspliceai',
       target_genes=intersection_genes,  # Same genes!
       ...
   )
   ```

5. **Handles Missing Genes Gracefully**:
   - Reports which genes were processed by each model
   - Reports which genes were missing (if any)
   - Continues with available genes
   - Saves missing gene lists for debugging

6. **Compares Performance**:
   - Overall metrics (precision, recall, F1, accuracy)
   - Genes processed vs. missing
   - Runtime comparison
   - Saves detailed results to JSON

### Simple Comparison
**Location**: `scripts/testing/compare_base_models_simple.py`

Uses 5 well-known genes (BRCA1, TP53, EGFR, MYC, KRAS) for quick demonstration.

### Full Comparison (Original)
**Location**: `scripts/testing/compare_base_models.py`

Samples 30 genes from GRCh37 and attempts to map them to GRCh38. May have missing genes.

### Running the Comparison

**Recommended (Robust)**:
```bash
source ~/.bash_profile && mamba activate surveyor
cd /Users/pleiadian53/work/meta-spliceai
python scripts/testing/compare_base_models_robust.py
```

**Quick Demo (Simple)**:
```bash
python scripts/testing/compare_base_models_simple.py
```

**Full Test (Original)**:
```bash
python scripts/testing/compare_base_models.py
```

### Expected Output

```
================================================================================
BASE MODEL COMPARISON TEST
================================================================================

STEP 1: Sample Test Genes from GRCh37/Ensembl
  ✅ Sampled 20 protein-coding genes
  ✅ Sampled 5 lncRNA genes
  ✅ Sampled 5 genes without splice sites

STEP 2: Run SpliceAI Predictions (GRCh37/Ensembl)
  🔵 Demonstrating clean base model interface:
     run_base_model_predictions(base_model='spliceai', ...)
  ✅ SpliceAI completed in XX.X seconds
     Positions analyzed: X,XXX
     Errors detected: XX

STEP 3: Run OpenSpliceAI Predictions (GRCh38/MANE)
  🟢 Demonstrating clean base model interface:
     run_base_model_predictions(base_model='openspliceai', ...)
  ✅ OpenSpliceAI completed in XX.X seconds
     Positions analyzed: X,XXX
     Errors detected: XX

STEP 4: Performance Comparison
  OVERALL PERFORMANCE COMPARISON
  Metric                SpliceAI              OpenSpliceAI
  ...
```

---

## Using the Workflow Directly

You can also use the workflow directly if you need more control:

```python
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig

# Create config with base_model parameter
config = SpliceAIConfig(
    base_model='spliceai',  # or 'openspliceai'
    mode='test',
    test_name='my_experiment',
    threshold=0.5
)

# Run workflow
results = run_enhanced_splice_prediction_workflow(
    config=config,
    target_genes=['BRCA1', 'TP53'],
    verbosity=1
)
```

---

## Configuration Options

Both models support the same configuration options:

```python
config = BaseModelConfig(
    base_model='spliceai',  # or 'openspliceai'
    mode='test',            # 'test' or 'production'
    coverage='gene_subset', # 'gene_subset', 'chromosome', or 'full_genome'
    test_name='my_test',   # Test identifier (required in test mode)
    threshold=0.5,         # Splice site score threshold
    consensus_window=2,     # Window for consensus calling
    error_window=500,      # Window for error analysis
    use_auto_position_adjustments=True,  # Auto-detect position offsets
    verbosity=1            # Output verbosity (0-2)
)
```

---

## Output Structure

Both models produce the same output structure:

```python
results = {
    'success': True,
    'positions': pl.DataFrame,      # All analyzed positions with predictions
    'error_analysis': pl.DataFrame,  # Positions with errors (FP, FN)
    'analysis_sequences': pl.DataFrame,  # Sequences around positions (optional)
    'paths': {
        'eval_dir': str,
        'positions_artifact': str,
        'errors_artifact': str,
        ...
    },
    'artifact_manager': {
        'mode': str,
        'coverage': str,
        'test_name': str,
        ...
    }
}
```

### Positions DataFrame Columns

- **Identification**: `gene_id`, `transcript_id`, `position`, `chrom`, `strand`
- **Predictions**: `donor_score`, `acceptor_score`, `neither_score`, `score`
- **Classification**: `pred_type` (TP, TN, FP, FN), `splice_type` (donor, acceptor)
- **Context Features**: `context_score_m1`, `context_score_p1`, `context_max`, etc.
- **Derived Features**: `donor_diff_m1`, `acceptor_surge_ratio`, `probability_entropy`, etc.

---

## Gene Mapping Between Builds

### Challenge
- **SpliceAI** uses GRCh37/Ensembl with Ensembl gene IDs (e.g., `ENSG00000012048`)
- **OpenSpliceAI** uses GRCh38/MANE with gene names (e.g., `BRCA1`)

### Solution
The comparison script maps genes by gene name:
1. Extract gene names from GRCh37 Ensembl IDs
2. Look up same gene names in GRCh38/MANE
3. Use gene names for OpenSpliceAI predictions

**Note**: Not all genes are available in both builds:
- MANE focuses on canonical protein-coding transcripts
- Many lncRNA genes are not in MANE
- Some genes may have different names between builds

---

## Performance Metrics

The comparison calculates:

### Overall Metrics
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)

### By Splice Type
- Separate metrics for donor and acceptor sites
- Helps identify model strengths/weaknesses

### Runtime
- Processing time per model
- Useful for performance optimization

---

## Example Results

```
OVERALL PERFORMANCE COMPARISON
--------------------------------------------------------------------------------
Metric                SpliceAI              OpenSpliceAI
--------------------------------------------------------------------------------
Total Positions       1,534                 1,234
TP                    671                   612
TN                    829                   598
FP                    13                    8
FN                    21                    16
--------------------------------------------------------------------------------
Precision             0.9810                0.9871
Recall                0.9697                0.9745
F1 Score             0.9753                0.9808
Accuracy             0.9778                0.9805
--------------------------------------------------------------------------------
Runtime (sec)         191.2                 165.3
```

---

## Key Takeaways

1. **Easy Switching**: Change one parameter (`base_model`) to switch models
2. **Automatic Routing**: System automatically uses correct genomic resources
3. **Unified Interface**: Same workflow, same output format, same configuration
4. **Framework Agnostic**: Handles Keras (SpliceAI) and PyTorch (OpenSpliceAI) internally
5. **Fair Comparison**: Same gene set, same evaluation criteria

---

## Troubleshooting

### Issue: "OpenSpliceAI models not found"
**Solution**: Download models first:
```bash
./scripts/base_model/download_openspliceai_models.sh
```

### Issue: "Gene features not found"
**Solution**: Extract gene features for the build:
```bash
# For GRCh37
python meta_spliceai/splice_engine/extract_genomic_features.py \
    --gtf data/ensembl/GRCh37/... \
    --output data/ensembl/GRCh37/gene_features.tsv

# For GRCh38/MANE
python meta_spliceai/splice_engine/extract_gene_features_mane.py \
    --gtf data/mane/GRCh38/... \
    --output data/mane/GRCh38/gene_features.tsv
```

### Issue: "Genes not found in GRCh38/MANE"
**Solution**: This is expected for some genes (especially lncRNA). The script will report which genes are missing and continue with available genes.

---

## Related Documentation

- [OpenSpliceAI Test Results](OPENSPLICEAI_TEST_RESULTS.md)
- [Universal Base Model Support](UNIVERSAL_BASE_MODEL_SUPPORT.md)
- [Base Model Architecture](FINAL_ARCHITECTURE_SUMMARY.md)
- [Testing Guide](OPENSPLICEAI_TESTING_GUIDE.md)

---

*Last Updated: 2025-11-06*

