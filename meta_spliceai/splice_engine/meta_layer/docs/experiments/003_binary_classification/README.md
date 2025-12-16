# Experiment 003: Binary Classification (Multi-Step Step 1)

**Date**: December 15, 2025  
**Status**: Initial Results  
**Approach**: Multi-Step Framework, Step 1

---

## Objective

Test binary classification: "Is this variant splice-altering?"

This is **Step 1 of the Multi-Step Framework**, NOT Approach B (which is delta prediction).

---

## Setup

### Data

- **Source**: SpliceVarDB
- **Classes**: 
  - Positive: `Splice-altering`
  - Negative: `Normal`
  - Excluded: `Low-frequency`, `Conflicting`
- **Train/Test Split**: Chromosomes 21, 22 held out for testing

### Model

`SpliceInducingClassifier` (from `splice_classifier.py`)

```python
SpliceInducingClassifier(
    hidden_dim=128,
    n_layers=6,
    dropout=0.1
)
```

**Architecture**:
```
alt_seq [B, 4, 501] ──→ GatedCNNEncoder ──→ seq_features [B, H]
                                                    ↓
ref_base [B, 4] ──┬──→ VariantEmbed ──→ var_features [B, H]
alt_base [B, 4] ──┘                           ↓
                                        concat [B, 2H]
                                              ↓
                                    Classifier ──→ P(splice-altering)
```

### Hyperparameters

```python
context_size = 501
epochs = 30
batch_size = 32
optimizer = AdamW(lr=1e-4, weight_decay=0.01)
scheduler = OneCycleLR(max_lr=1e-3)
```

---

## Results

### Initial Run (2000 samples, balanced)

| Metric | Value | Notes |
|--------|-------|-------|
| **ROC-AUC** | 0.61 | Above random (0.5) |
| **PR-AUC (AP)** | 0.66 | Primary metric |
| **F1 Score** | 0.53 | **Needs > 0.7** |
| **Accuracy** | 0.60 | (Not a useful metric) |
| **Precision** | 0.63 | |
| **Recall** | 0.45 | |

### Confusion Matrix

```
                 Pred Normal  Pred SA
True Normal:         185        65
True SA:             137       113
```

---

## Analysis

### What's Working

1. **Above random**: AUC 0.61 > 0.5 indicates learning
2. **Balanced training**: Equal class representation helps
3. **Simple architecture**: Gated CNN + variant embedding

### What's Not Working

1. **Low recall (0.45)**: Missing many splice-altering variants
2. **F1 too low (0.53)**: Target is > 0.7
3. **Limited data**: Only 2000 samples

### Why Performance is Limited

1. **Hard problem**: Many variants have subtle effects
2. **Context may be too short**: 501nt might miss distal effects
3. **Variant encoding too simple**: Just ref/alt base, no position info

---

## Potential Improvements

### 1. More Data

```python
# Use all ~50K SpliceVarDB variants
# Requires GPU (RunPods)
```

### 2. Longer Context

```python
# Try 1001nt or 2001nt windows
context_size = 1001
```

### 3. Richer Variant Encoding

```python
# Add:
# - Relative position in window
# - Distance to nearest canonical splice site
# - Base model confidence scores
```

### 4. Data Augmentation

```python
# Reverse complement augmentation
# Sequence masking
```

### 5. Pre-trained Encoder

```python
# HyenaDNA instead of random init CNN
# Requires GPU
```

---

## Next Steps

1. **Try longer context** (local M1)
2. **Add position features** (local M1)
3. **Scale up data + HyenaDNA** (RunPods GPU)

---

## Files

| File | Description |
|------|-------------|
| `models/splice_classifier.py` | Model implementation |
| `docs/methods/MULTI_STEP_FRAMEWORK.md` | Framework description |

---

## Key Takeaway

Binary classification is **learnable** (AUC > 0.5) but **not yet useful** (F1 < 0.7). Need improvements before proceeding to Steps 2-4 of the Multi-Step Framework.

---

*This experiment is part of the Multi-Step Framework development, NOT Approach B.*

