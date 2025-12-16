# Experiment 002: Direct Delta Prediction

**Date**: December 14, 2025  
**Status**: In Progress  
**Predecessor**: [001_canonical_classification](../001_canonical_classification/)

---

## Motivation

Experiment 001 showed that training for canonical splice site classification does NOT transfer to variant effect detection. The meta-layer achieved 99.11% classification accuracy but only 17% variant detection rate.

**The key insight**: To predict how variants affect splicing, we must train on variant data with delta-focused objectives.

---

## Hypothesis

> By training a model to directly predict delta scores (ref → alt changes) using SpliceVarDB variant data, we can improve variant effect detection beyond the base model.

---

## Approach: Siamese Delta Network

### Architecture

```
                     Siamese Delta Network
    ┌────────────────────────────────────────────────┐
    │                                                │
    │   Reference Seq ──→ [Encoder] ──→ ref_embed    │
    │                         ↓                       │
    │                    (shared weights)             │
    │                         ↓                       │
    │   Alternate Seq ──→ [Encoder] ──→ alt_embed    │
    │                                                │
    │              alt_embed - ref_embed             │
    │                      ↓                         │
    │               [Delta Head]                     │
    │                      ↓                         │
    │        Output: [Δ_donor, Δ_acceptor]           │
    │                                                │
    └────────────────────────────────────────────────┘
```

### Key Design Choices

1. **Siamese Architecture**: Same encoder for ref and alt ensures the model learns the *difference*, not just the sequences
2. **Delta Output**: Directly predict score changes, not absolute scores
3. **Variant-Centered Context**: Window around the variant position
4. **SpliceVarDB Supervision**: Use variant classifications as training signal

---

## Training Data

### Source: SpliceVarDB

Each training sample is a variant with:
- `chrom`, `position`, `ref_allele`, `alt_allele`
- `classification`: Splice-altering, Normal, Low-frequency, Conflicting

### Data Preparation

```python
For each variant in SpliceVarDB:
    1. Extract ref_sequence (501nt centered on variant)
    2. Create alt_sequence (substitute variant)
    3. Get base model delta: base_delta = base_model(alt) - base_model(ref)
    4. Create training sample:
       - Input: (ref_sequence, alt_sequence)
       - Target: base_delta (supervised by base model)
       - Weight: 2.0 if Splice-altering else 1.0
```

### Label Strategy Options

| Option | Target | Pros | Cons |
|--------|--------|------|------|
| **A. Base Model Delta** | `base_model(alt) - base_model(ref)` | Continuous, directly comparable | Limited by base model |
| **B. Classification** | 1 if Splice-altering else 0 | Uses SpliceVarDB labels | Binary, loses magnitude |
| **C. Hybrid** | Multi-task: predict delta + classify | Both signals | More complex |

We'll start with **Option A** as it enables direct comparison with base model.

---

## Model Architecture

### DeltaPredictor (Siamese)

```python
class DeltaPredictor(nn.Module):
    def __init__(self, hidden_dim=256):
        # Shared sequence encoder
        self.encoder = CNNEncoder(input_channels=4, output_dim=hidden_dim)
        
        # Delta prediction head
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # [Δ_donor, Δ_acceptor]
        )
    
    def forward(self, ref_seq, alt_seq):
        # Encode both sequences with shared encoder
        ref_embed = self.encoder(ref_seq)  # [B, hidden_dim]
        alt_embed = self.encoder(alt_seq)  # [B, hidden_dim]
        
        # Compute difference embedding
        diff_embed = alt_embed - ref_embed
        
        # Predict delta scores
        delta = self.delta_head(diff_embed)  # [B, 2]
        
        return delta  # [Δ_donor, Δ_acceptor]
```

### Loss Function

```python
# Weighted MSE loss
loss = weighted_mse(
    predicted_delta,
    target_delta,  # From base model
    weights        # Higher for splice-altering variants
)
```

---

## Training Configuration

```yaml
model:
  architecture: siamese_delta
  encoder: CNN
  hidden_dim: 256
  dropout: 0.1

training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 0.01
  
  # Class weighting
  splice_altering_weight: 2.0
  normal_weight: 1.0
  low_frequency_weight: 0.5

data:
  source: SpliceVarDB
  genome_build: GRCh38
  context_size: 501
  train_chromosomes: [1-20]
  test_chromosomes: [21, 22]
```

---

## Evaluation Metrics

### Primary Metrics

1. **Delta Correlation**: Pearson correlation between predicted and true deltas
2. **Detection Rate**: % of splice-altering variants with |Δ| > threshold
3. **False Positive Rate**: % of normal variants with |Δ| > threshold

### Comparison Framework

```
For each variant in test set:
    base_delta = base_model(alt) - base_model(ref)
    meta_delta = delta_predictor(ref_seq, alt_seq)
    
    Compare:
    - |base_delta| vs |meta_delta| for splice-altering variants
    - Detection rates at various thresholds
```

---

## Expected Outcomes

### Success Criteria

| Metric | Base Model | Target |
|--------|------------|--------|
| Detection Rate (splice-altering) | 67% | >75% |
| Mean |Δ| (splice-altering) | 0.19 | >0.25 |
| False Positive Rate (normal) | ~0% | <5% |

### Failure Modes

1. **Overfitting to training variants**: Poor generalization to new variants
2. **Dominated by base model**: Just learns to copy base model deltas
3. **Ignoring sequence context**: Relies too heavily on variant type

---

## Implementation Plan

1. **VariantDataset**: Load SpliceVarDB, extract sequences, compute base deltas
2. **DeltaPredictor**: Siamese architecture with shared encoder
3. **Training Loop**: Weighted loss, early stopping on validation
4. **Evaluation**: Compare with base model on held-out variants

---

## Files to Create

```
meta_layer/
├── data/
│   └── variant_dataset.py      # SpliceVarDB + sequence extraction
├── models/
│   └── delta_predictor.py      # Siamese delta network
├── training/
│   └── delta_trainer.py        # Delta-specific training loop
└── examples/
    └── train_delta.py          # Example training script
```

---

## References

- [001_canonical_classification](../001_canonical_classification/) - Previous experiment
- [LABELING_STRATEGY.md](../../LABELING_STRATEGY.md) - Delta-learning formulation
- [DELTA_SCORE_IMPLEMENTATION.md](../../DELTA_SCORE_IMPLEMENTATION.md) - Delta computation

