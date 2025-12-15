# Approach A: Paired (Siamese) Delta Prediction

**Status**: TESTED  
**Result**: r=0.38 correlation (insufficient)  
**Last Updated**: December 15, 2025

---

## Overview

Approach A uses a **Siamese architecture** to predict splice site delta scores by comparing reference and alternate sequences.

### Key Characteristics

- **Input**: Both ref_seq AND alt_seq
- **Target**: base_model(alt) - base_model(ref)
- **Inference**: Two encoder forward passes + difference

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      APPROACH A: PAIRED PREDICTION                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    ref_seq ──→ [Shared Encoder] ──→ ref_features                        │
│                                            ↓                             │
│                                     diff = alt - ref                     │
│                                            ↓                             │
│    alt_seq ──→ [Shared Encoder] ──→ alt_features                        │
│                                                                          │
│                                     diff ──→ [Delta Head] ──→ Δ         │
│                                                                          │
│    Output: Δ = [Δ_donor, Δ_acceptor] per position                       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Implementations Tested

### V1: SimpleCNN + Global Pooling

```python
# Output: [B, 2] single delta per variant
class DeltaPredictorV1:
    - encoder: SimpleCNN → global_avg_pool → [B, H]
    - diff: alt_embed - ref_embed
    - delta_head: MLP → [B, 2]
```

**Result**: No learning (r ≈ 0)

### V2: SimpleCNN + Per-Position

```python
# Output: [B, L, 2] delta per position
class DeltaPredictorV2:
    - encoder: SimpleCNN → [B, H, L]
    - diff: alt_features - ref_features
    - delta_head: Conv1d → [B, 2, L]
```

**Result**: Slight improvement (r ≈ 0.1)

### V3: Gated CNN + Dilated Convolutions

```python
# Output: [B, L, 2] with better receptive field
class SimpleCNNDeltaPredictor:
    - encoder: GatedCNN with dilations [1, 2, 4, 8, 16, 32]
    - diff: alt_features - ref_features
    - delta_head: Gated conv → [B, 2, L]
```

**Result**: Best so far (r = 0.36)

### V4: Gated CNN + Calibration

| Calibration | Correlation | Notes |
|-------------|-------------|-------|
| None | r=0.36 | Baseline |
| Output scaling | r=0.22 | Overfitting |
| Temperature | r=-0.03 | No improvement |
| Quantile (τ=0.9) | r=0.38 | Best |

---

## Training Details

### Data

```python
# SpliceVarDB variants (~1000-2000 for quick iteration)
# Balanced: 50% Splice-altering, 50% Normal

train_variants = loader.get_train_test_split(test_chromosomes=['21', '22'])
```

### Targets

```python
# Base model delta (OpenSpliceAI ensemble)
def get_target(variant, base_models, fasta):
    ref_seq = extract_sequence(variant, fasta, context=10000)
    alt_seq = apply_variant(ref_seq, variant)
    
    ref_pred = ensemble_predict(base_models, ref_seq)
    alt_pred = ensemble_predict(base_models, alt_seq)
    
    delta = alt_pred - ref_pred  # [L, 3]
    
    # Extract window around variant
    center = len(delta) // 2
    window = delta[center-50:center+51]  # [101, 2]
    return window[:, [0, 1]]  # donor, acceptor only
```

### Loss

```python
# MSE with sample weights
loss = F.mse_loss(pred, target, reduction='none')
loss = (loss.mean(dim=[1, 2]) * sample_weights).mean()

# Sample weights: 2.0 for Splice-altering, 1.0 for Normal
```

### Hyperparameters

```python
optimizer = AdamW(lr=5e-5, weight_decay=0.02)
scheduler = OneCycleLR(max_lr=1e-3)
epochs = 20-30
batch_size = 32
context_window = 101  # variant ± 50nt
```

---

## Limitations

### 1. Target Accuracy

The fundamental limitation: **targets come from the base model**, which may be wrong.

```
If base_model predicts incorrectly:
  - Splice-altering variant → base_delta ≈ 0 (wrong)
  - Normal variant → base_delta ≠ 0 (wrong)
  
We're training on potentially incorrect labels!
```

### 2. Inference Efficiency

Two forward passes required:
```python
ref_embed = encoder(ref_seq)  # Pass 1
alt_embed = encoder(alt_seq)  # Pass 2
delta = delta_head(alt_embed - ref_embed)
```

### 3. No Variant Context

The model doesn't explicitly know:
- What the original base was
- What it changed to
- Where in the sequence the variant is

---

## Why It Partially Works

Despite limitations, r=0.38 correlation suggests:

1. **Sequence patterns matter**: The CNN learns splice site motifs
2. **Relative changes**: Difference embedding captures change magnitude
3. **Local effects**: Gated CNN captures variant effects on nearby positions

---

## Why It's Not Sufficient

r=0.38 is not good enough because:

1. **Target noise**: ~40% of base model predictions may be wrong
2. **Missing information**: No explicit variant encoding
3. **Magnitude mismatch**: Learned deltas don't match true effect sizes

---

## Recommendations

### For Approach A Improvements

1. **Filter training data**: Only use variants where base model is confident
2. **Ensemble agreement**: Only train on variants where all base models agree
3. **Multi-scale features**: Add global context to local delta prediction

### Versus Approach B

Consider Approach B instead:
- Uses validated SpliceVarDB labels (not base model deltas)
- Single forward pass
- Explicit variant encoding

---

## Files

| File | Description |
|------|-------------|
| `models/delta_predictor_v2.py` | V2 implementation |
| `models/hyenadna_delta_predictor.py` | V3 Gated CNN implementation |
| `docs/experiments/002_delta_prediction/` | Detailed experiment logs |

---

## Conclusion

Approach A (paired prediction) achieves moderate correlation (r=0.38) but is fundamentally limited by relying on base model deltas as targets. **Approach B (single-pass with validated labels)** is recommended for further development.

---

*This document is part of the meta_layer methodology documentation.*

