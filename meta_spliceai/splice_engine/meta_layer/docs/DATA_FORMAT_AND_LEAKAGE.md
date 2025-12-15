# Data Format and Leakage Prevention

**Last Updated**: December 2025  
**Status**: ✅ Critical Documentation

---

## Data Format

### Training Artifacts: SUBSAMPLED Positions

**Important**: Training artifacts contain a **SUBSET** of nucleotide positions, NOT all positions in a gene.

```
Gene ABC1: 10,000 nucleotides
           ↓
Training artifacts: ~1,000 positions (SUBSAMPLED)
  ├── TPs: ~50 (actual splice sites correctly predicted)
  ├── FPs: ~100 (false positives - positions wrongly predicted as splice)
  ├── FNs: ~20 (false negatives - missed actual splice sites)
  └── TNs: ~830 (subsampled from ~9,800 non-splice sites)

Total: ~10% of gene included in training data
```

**Why subsampled?**
- True negatives (non-splice sites) vastly outnumber other classes (~98%)
- Subsampling prevents severe class imbalance
- Window-based TN sampling preserves locality (see `tn_sampling_mode="window"` in `enhanced_evaluation.py`)

### Per-Position Processing

Each row in the artifacts represents a single genomic position:

```
Input: analysis_sequences_chr21.tsv
Each row = 1 genomic position with:
  - sequence: 501nt context window centered on position
  - features: 43+ derived features from base model scores
  - splice_type: label ('donor', 'acceptor', '')
```

### Training vs. Inference Data

| Context | Data Source | Positions per Gene |
|---------|-------------|-------------------|
| **Training** | `analysis_sequences_*.tsv` | ~10% (subsampled) |
| **Inference** | `nucleotide_scores.tsv` | 100% (all positions) |

**Training Example** (subsampled artifacts):
```python
# Load training artifacts (SUBSAMPLED)
loader = ArtifactLoader(config)
df = loader.load_analysis_sequences(chromosomes=['21'])

# Gene ABC1 with 10,000 nucleotides → ~1,000 positions in artifacts
gene_positions = df.filter(pl.col('gene_id') == 'ABC1')
# → ~1,000 rows (NOT 10,000!)
```

**Full Coverage Inference** (all positions):
```python
from meta_spliceai.splice_engine.meta_layer.inference import predict_full_coverage

# Runs base model with save_nucleotide_scores=True
# Then applies meta-model to ALL positions
results = predict_full_coverage(
    meta_model_path='models/best_model.pt',
    target_genes=['ABC1'],
    base_model='openspliceai'
)

# Gene ABC1: ALL 10,000 positions
abc1 = results['ABC1']
assert len(abc1.donor_scores) == abc1.gene_length  # 10,000 ✅
```

### Output Format

Per-nucleotide predictions, same shape as base model:

| Position | Donor Score | Acceptor Score | Neither Score |
|----------|-------------|----------------|---------------|
| 1000     | 0.01        | 0.02           | 0.97          |
| 1001     | 0.85        | 0.03           | 0.12          |
| ...      | ...         | ...            | ...           |
| 10999    | 0.02        | 0.91           | 0.07          |

**Full coverage output**: `[gene_length, 3]` for each gene

See [`TRAINING_VS_INFERENCE.md`](TRAINING_VS_INFERENCE.md) for the complete pipeline.

---

## Data Leakage Prevention

### Critical: Column Categories

```python
# NEVER use these as features - they leak label information!
LEAKAGE_COLS = [
    'splice_type',         # The target label itself
    'pred_type',           # Base model prediction (derived from label comparison)
    'true_position',       # Exact splice site coordinate
    'predicted_position',  # Tightly correlated with label
    'is_correct',          # Whether base model was correct
    'error_type',          # FP/FN/TP/TN classification
]

# NEVER use these as features - high cardinality, poor generalization
METADATA_COLS = [
    'gene_id',             # Unique per gene
    'transcript_id',       # Unique per transcript
    'position',            # Absolute genomic position
    'absolute_position',
    'window_start',
    'window_end',
    'strand',
    'chrom',
]
```

### Safe Feature Categories

```python
# These are safe to use as features:
SAFE_FEATURES = [
    # Base model scores (the core signal)
    'donor_score', 'acceptor_score', 'neither_score',
    
    # Context scores (neighboring positions)
    'context_score_m2', 'context_score_m1', 
    'context_score_p1', 'context_score_p2',
    
    # Derived probability features
    'relative_donor_probability', 'splice_probability',
    'donor_acceptor_diff', 'probability_entropy',
    
    # Pattern features (computed from scores)
    'donor_is_local_peak', 'acceptor_is_local_peak',
    'donor_signal_strength', 'acceptor_signal_strength',
    # ... etc
]
```

### Leakage Detection

The `MetaLayerDataset` automatically:

1. **Checks for known leakage columns** in the data
2. **Excludes them** from the feature set
3. **Optionally runs correlation-based leakage detection**

```python
# Enable leakage checking (default: True)
dataset = MetaLayerDataset(
    df,
    check_leakage=True,
    leakage_threshold=0.95,  # Flag features with |corr| >= 0.95
    extra_exclude_cols=['my_suspicious_feature']
)
```

---

## Comparison with Tabular Meta-Model

The tabular meta-model (`run_gene_cv_sigmoid.py`) uses the same leakage prevention:

```python
# From meta_spliceai/splice_engine/meta_models/builder/preprocessing.py
from .preprocessing import LEAKAGE_COLUMNS, METADATA_COLUMNS

excluded = set(LEAKAGE_COLUMNS + METADATA_COLUMNS + ["splice_type"])
features = [c for c in all_cols if c not in excluded]
```

The meta_layer package follows this convention exactly.

---

## Common Leakage Patterns

### 1. Using `pred_type` as a Feature ❌

```python
# WRONG: pred_type tells you what the base model predicted
# This is derived by comparing base model output to splice_type
features = ['donor_score', 'pred_type', ...]  # LEAKAGE!
```

### 2. Using Position-Based Features ❌

```python
# WRONG: Absolute position is high-cardinality and memorizable
features = ['position', 'window_start', ...]  # MEMORIZATION RISK!
```

### 3. Using Gene/Transcript IDs ❌

```python
# WRONG: Model can memorize patterns per gene
features = ['gene_id', 'transcript_id', ...]  # OVERFITTING RISK!
```

---

## Recommendations

1. **Always use the schema**: Let `FeatureSchema` define what's safe
2. **Enable leakage checking**: Set `check_leakage=True` (default)
3. **Use gene-wise CV**: Split by gene_id to avoid within-gene leakage
4. **Review feature importance**: High-importance features may indicate leakage

---

*This document should be reviewed alongside the tabular model's leakage handling in `meta_spliceai/splice_engine/meta_models/training/`*

