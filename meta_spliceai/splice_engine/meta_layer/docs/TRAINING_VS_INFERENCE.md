# Training vs. Inference Data Formats

**Last Updated**: December 2025  
**Status**: ✅ Critical Architecture Document

---

## The "Gap" Problem

### Training Data: Subsampled Positions

Training artifacts contain **subsampled positions** (position-centric representation):

```
Gene ABC1: 10,000 nucleotides
           ↓
Training artifacts: ~1,000 positions (subsampled)
  ├── TPs: ~50 (actual splice sites correctly predicted)
  ├── FPs: ~100 (positions wrongly predicted as splice sites)
  ├── FNs: ~20 (missed actual splice sites)
  └── TNs: ~830 (subsampled from ~9,800 non-splice sites)
  
Total: ~10% of gene included in training data
```

**Why subsampled?**
- TNs (true negatives) vastly outnumber other classes
- Subsampling prevents class imbalance
- Window-based TN sampling preserves locality (see `tn_sampling_mode="window"`)

### Inference Requirement: Full Coverage

For proper gene-level predictions, we need **ALL positions**:

```
Gene ABC1: 10,000 nucleotides
           ↓
Full coverage output: 10,000 positions
  - Position 1: [donor=0.01, acceptor=0.02, neither=0.97]
  - Position 2: [donor=0.02, acceptor=0.01, neither=0.97]
  - ...
  - Position 10000: [donor=0.03, acceptor=0.85, neither=0.12]
  
Output shape: [10000, 3]
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Gene (N nucleotides)                                               │
│         │                                                           │
│         ▼                                                           │
│  Base Model (SpliceAI/OpenSpliceAI)                                 │
│         │                                                           │
│         ▼                                                           │
│  Enhanced Evaluation + TN Subsampling                               │
│         │                                                           │
│         ▼                                                           │
│  Training Artifacts (~10% of positions)                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ analysis_sequences_chr*.tsv                                 │   │
│  │ - ~1000 positions per 10000nt gene                          │   │
│  │ - Features: scores + derived features + 501nt context       │   │
│  │ - Labels: splice_type (donor/acceptor/neither)              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         │                                                           │
│         ▼                                                           │
│  Meta-Layer Training (MetaSpliceModel)                              │
│         │                                                           │
│         ▼                                                           │
│  Trained Model (best_model.pt)                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       INFERENCE PIPELINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Gene (N nucleotides)                                               │
│         │                                                           │
│         ▼                                                           │
│  Base Model with FULL COVERAGE                                      │
│  (save_nucleotide_scores=True)                                      │
│         │                                                           │
│         ▼                                                           │
│  Full Nucleotide Scores (ALL N positions)                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ nucleotide_scores.tsv                                       │   │
│  │ - ALL N positions in gene                                   │   │
│  │ - Raw scores: donor, acceptor, neither                      │   │
│  │ - 501nt context windows for each                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         │                                                           │
│         ▼                                                           │
│  Feature Generation (for ALL positions)                             │
│         │                                                           │
│         ▼                                                           │
│  Meta-Layer Inference (FullCoveragePredictor)                       │
│         │                                                           │
│         ▼                                                           │
│  Recalibrated Scores [N, 3]                                         │
│  - donor_scores: [N]                                                │
│  - acceptor_scores: [N]                                             │
│  - neither_scores: [N]                                              │
│                                                                     │
│  ✓ Length = gene_end - gene_start + 1 = gene_length                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Training (uses subsampled artifacts)

```python
from meta_spliceai.splice_engine.meta_layer import (
    MetaLayerConfig,
    ArtifactLoader,
    MetaLayerDataset
)

# Load SUBSAMPLED training artifacts
config = MetaLayerConfig(base_model='openspliceai')
loader = ArtifactLoader(config)
df = loader.load_analysis_sequences(chromosomes=['21', '22'])

# df contains ~10% of positions per gene
print(f"Training samples: {len(df)}")  # e.g., 50,000 positions
```

### Inference (full coverage)

```python
from meta_spliceai.splice_engine.meta_layer.inference import (
    FullCoveragePredictor,
    predict_full_coverage
)

# Predict on ALL positions
results = predict_full_coverage(
    meta_model_path='models/best_model.pt',
    target_genes=['BRCA1', 'TP53'],
    base_model='openspliceai'
)

# Check output length matches gene length
brca1 = results['BRCA1']
print(f"Gene length: {brca1.gene_length}")
print(f"Donor scores: {len(brca1.donor_scores)}")  # Same as gene_length!
assert len(brca1.donor_scores) == brca1.gene_length
```

---

## Base Model Full Coverage Mode

The base model supports full coverage via `save_nucleotide_scores=True`:

```python
from meta_spliceai import run_base_model_predictions

# Run with FULL COVERAGE
results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1', 'TP53'],
    save_nucleotide_scores=True,  # KEY: All positions
    mode='test',
    no_tn_sampling=True
)

# nucleotide_scores contains ALL positions
nucleotide_scores = results['nucleotide_scores']
print(f"All positions: {nucleotide_scores.height:,}")
```

See: `docs/base_models/RUN_BASE_MODEL_FULL_COVERAGE_EXAMPLES.md`

---

## Validation Checklist

For full coverage inference, verify:

1. **Output length** matches gene length:
   ```python
   assert len(donor_scores) == (gene_end - gene_start + 1)
   ```

2. **Three score vectors** per gene:
   ```python
   assert donor_scores.shape == (gene_length,)
   assert acceptor_scores.shape == (gene_length,)
   assert neither_scores.shape == (gene_length,)
   ```

3. **Scores sum to ~1** at each position:
   ```python
   total = donor_scores + acceptor_scores + neither_scores
   assert np.allclose(total, 1.0, atol=0.01)
   ```

4. **Coverage is complete** (no gaps):
   ```python
   positions = result.to_dataframe()['position'].to_numpy()
   expected = np.arange(gene_start, gene_end + 1)
   assert np.array_equal(positions, expected)
   ```

---

## Evaluation Strategy

### 1. Generalization to Unseen Genes (CV)

Test how well the model generalizes to genes not in training:

```python
# Gene-wise cross-validation
# Split by gene_id to prevent within-gene leakage
from sklearn.model_selection import GroupKFold

gkfold = GroupKFold(n_splits=5)
for train_idx, test_idx in gkfold.split(X, y, groups=gene_ids):
    # Train on some genes, test on others
    ...
```

### 2. Alternative Splice Site Detection (SpliceVarDB)

Test ability to capture variant-induced splice sites:

```python
# Load SpliceVarDB variants
from meta_spliceai.splice_engine.case_studies import SpliceVarDBLoader

variants = SpliceVarDBLoader().load_variants()

# For each splice-inducing variant:
# 1. Run meta-layer inference on the gene
# 2. Check if the alternative splice site is detected
# 3. Compare to base model predictions
```

---

## Key Files

| File | Purpose |
|------|---------|
| `run_base_model.py` | Base model with full coverage option |
| `ArtifactLoader` | Load SUBSAMPLED training artifacts |
| `FullCoveragePredictor` | Full coverage inference pipeline |
| `enhanced_evaluation.py` | TN subsampling logic |

---

*This document clarifies the critical difference between training (subsampled) and inference (full coverage) data formats.*

