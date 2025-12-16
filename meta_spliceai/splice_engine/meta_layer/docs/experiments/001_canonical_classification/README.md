# Experiment 001: Canonical Splice Site Classification

**Date**: December 14, 2025  
**Status**: Completed  
**Outcome**: Partial Success (improved classification, failed variant detection)

---

## Overview

This experiment evaluates a multimodal meta-learning approach for splice site prediction. The meta-layer combines DNA sequence context (via CNN encoder) with tabular features derived from base model scores to recalibrate splice site predictions.

### Hypothesis

> By learning from base model errors using a multimodal architecture that combines sequence context with prediction features, we can improve both canonical splice site classification AND variant effect detection.

### Key Finding

**The hypothesis was partially validated**: The meta-layer improves canonical splice site classification but does NOT improve variant effect detection on SpliceVarDB.

---

## Experimental Setup

### Architecture

```
Position-Centric Meta-Layer:

Input:
  ├── 501nt DNA sequence (one-hot) → CNN Encoder → [256-dim]
  └── 43 tabular features (base scores, derived) → MLP → [256-dim]
                                                    ↓
                                            Concatenation
                                                    ↓
                                            Fusion MLP
                                                    ↓
                                            Output: [3]
                                            (donor, acceptor, neither)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | OpenSpliceAI (GRCh38/MANE) |
| Sequence Encoder | CNN (4 layers, 32 channels) |
| Hidden Dimension | 256 |
| Training Samples | 20,000 (balanced: 6,666 per class) |
| Training Chromosomes | 1-20 |
| Test Chromosome | 21 |
| Epochs | 15 |
| Batch Size | 64 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | CosineAnnealingLR |

### Labels

- **Source**: GTF/GFF3 annotations (canonical splice sites)
- **Classes**: donor, acceptor, neither
- **Weighting**: Balanced sampling (equal class representation)

### Evaluation

1. **Task 1**: Canonical splice site classification (test set from artifacts)
2. **Task 2**: Variant effect detection (SpliceVarDB, chromosome 21)

---

## Results

### Task 1: Canonical Splice Site Classification ✅

| Metric | Base Model | Meta-Layer | Change |
|--------|------------|------------|--------|
| **Accuracy** | 97.61% | 99.11% | **+1.50%** |
| Donor AP | 0.9981 | 0.9966 | -0.15% |
| Acceptor AP | 0.9972 | 0.9964 | -0.09% |
| Neither AP | 0.9925 | 0.9989 | **+0.64%** |

**Conclusion**: Meta-layer improves overall classification accuracy, especially for "neither" class.

### Task 2: Variant Effect Detection ❌

| Metric | Base Model | Meta-Layer | Change |
|--------|------------|------------|--------|
| **Splice-altering detection rate** | 67% (4/6) | 17% (1/6) | **-50%** |
| Mean delta (splice-altering) | 0.1933 | 0.0228 | **-88%** |
| Mean delta (normal) | 0.0004 | 0.0010 | +150% (but tiny) |
| Mean delta (low-frequency) | 0.0415 | 0.0014 | -97% |

**Conclusion**: Meta-layer produces much smaller delta scores, failing to detect most splice-altering variants.

---

## Analysis

### Why Classification Improved

1. **Sequence context**: The 501nt context provides additional signal beyond base model scores
2. **Feature fusion**: Combining sequence and tabular features captures complementary information
3. **Error correction**: The meta-layer learns to correct base model mistakes in ambiguous regions

### Why Variant Detection Failed

See [ANALYSIS.md](./ANALYSIS.md) for detailed analysis.

**Summary**:
1. **Training objective mismatch**: Model optimized for classification, not for detecting changes
2. **Small context change**: A single variant barely changes the 501nt context
3. **Position-centric limitation**: Model outputs single [3] prediction, cannot find MAX delta in window
4. **No variant-aware training**: SpliceVarDB data not used during training

---

## Files & Artifacts

| File | Location |
|------|----------|
| Model Checkpoint | `data/mane/GRCh38/openspliceai_eval/meta_layer_dev/20251214_161359/checkpoints/meta_layer_phase1.pt` |
| Training Data | Pre-computed artifacts in `data/mane/GRCh38/openspliceai_eval/meta_models/` |
| Test Data | `analysis_sequences_21_*.tsv` |

---

## Lessons Learned

See [LESSONS_LEARNED.md](./LESSONS_LEARNED.md) for detailed insights.

**Key Takeaways**:
1. Classification accuracy ≠ variant effect detection ability
2. Training objective must match evaluation objective
3. Position-centric architectures limit variant effect analysis
4. Need explicit variant-aware training for SpliceVarDB evaluation

---

## Next Steps

→ **Phase 2: Direct Delta Prediction** (see `002_delta_prediction/`)

The next experiment will:
1. Train on (ref, alt) sequence pairs
2. Predict delta scores directly
3. Use SpliceVarDB classifications for supervision/weighting

---

## References

- [LABELING_STRATEGY.md](../../LABELING_STRATEGY.md) - Labeling approaches
- [DELTA_SCORE_IMPLEMENTATION.md](../../DELTA_SCORE_IMPLEMENTATION.md) - Delta score computation
- [TRAINING_VS_INFERENCE.md](../../TRAINING_VS_INFERENCE.md) - Data format differences

