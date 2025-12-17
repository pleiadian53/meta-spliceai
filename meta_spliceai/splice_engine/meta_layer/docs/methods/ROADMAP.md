# Alternative Splice Site Prediction: Methodology Roadmap

**Created**: December 15, 2025  
**Status**: Active Development  
**Last Updated**: December 16, 2025

---

## Overview

This document tracks the progressive development of methods for detecting and predicting **alternative splice sites** induced by genetic variants.

### Goal
Predict whether and how a genetic variant affects splicing patterns, going beyond what current base models (SpliceAI, OpenSpliceAI) can detect.

### Challenge
Base models are trained on canonical splice sites and fail to capture many variant-induced alternative splice sites documented in SpliceVarDB.

---

## Method Taxonomy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ALTERNATIVE SPLICE SITE PREDICTION                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ APPROACH A: Paired Prediction (Siamese)                                │  │
│  │                                                                        │  │
│  │ • Input: ref_seq + alt_seq (BOTH needed)                              │  │
│  │ • Target: base_model(alt) - base_model(ref)                           │  │
│  │ • Output: [L, 2] per-position deltas                                  │  │
│  │                                                                        │  │
│  │ Status: Tested, r=0.38 correlation (not sufficient)                   │  │
│  │ Limitation: Learning from potentially inaccurate base model deltas    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ VALIDATED DELTA: Single-Pass with Ground Truth Targets                 │  │
│  │                                                                        │  │
│  │ • Input: alt_seq + variant_info (ref_base, alt_base)                  │  │
│  │ • Target: SpliceVarDB-validated delta (ground truth filtering)        │  │
│  │ • Output: Δ directly (single forward pass)                            │  │
│  │                                                                        │  │
│  │ Status: ✅ IMPLEMENTED & TESTED - r=0.507 (BEST!) ⭐                   │  │
│  │ Key finding: 8000 samples → r=0.507 (+24% vs 2000 samples)            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ MULTI-STEP FRAMEWORK (Decomposed Problem) ⭐ BEST FOR INTERPRETABILITY │  │
│  │                                                                        │  │
│  │ Step 1: Binary Classification                                         │  │
│  │   • "Is this variant splice-altering?"                                │  │
│  │   • Status: Tested, AUC=0.61, F1=0.53 (needs >0.7)                   │  │
│  │   • Value: Triage 10K variants → 1K candidates                        │  │
│  │                                                                        │  │
│  │ Step 2: Effect Type Classification                                    │  │
│  │   • "What type of effect?" (gain/loss, donor/acceptor)               │  │
│  │   • Status: Implemented, NOT YET TESTED                               │  │
│  │   • Value: Guides ASO design strategy                                 │  │
│  │                                                                        │  │
│  │ Step 3: Position Localization                                         │  │
│  │   • "Where in the window is the effect?"                              │  │
│  │   • Status: NOT YET IMPLEMENTED                                       │  │
│  │   • Value: ⭐ CRITICAL for ASO target design (exact position)         │  │
│  │                                                                        │  │
│  │ Step 4: Delta Magnitude (CONDITIONED on Steps 1-3)                    │  │
│  │   • "How strong is the effect at the identified position?"            │  │
│  │   • Input: alt_seq + effect_type (Step 2) + position (Step 3)        │  │
│  │   • Status: NOT YET IMPLEMENTED (ValidatedDelta is standalone)       │  │
│  │   • TODO: Create ConditionedDeltaPredictor using cascade outputs     │  │
│  │                                                                        │  │
│  │ WHY IMPORTANT: Provides INTERPRETABLE decisions for FDA/stakeholders  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Development Timeline

### Phase 1: Canonical Classification (Completed ❌)

**Dates**: December 8-14, 2025  
**Goal**: Improve splice site classification using meta-learning on base model artifacts

**Approach**:
- Multimodal model: sequence (CNN) + tabular features (base model scores)
- Labels: GTF canonical splice sites (donor, acceptor, neither)
- Sample weights from SpliceVarDB

**Results**:
- Classification accuracy: 99%
- Variant detection: 17% (FAILED)

**Conclusion**: High accuracy on canonical sites does NOT transfer to variant detection.

**Documentation**: `docs/experiments/001_canonical_classification/`

---

### Phase 2: Direct Delta Prediction - Approach A (Completed ⚠️)

**Dates**: December 14-15, 2025  
**Goal**: Predict delta scores directly using Siamese architecture

**Approach**:
- Input: ref_seq + alt_seq (paired)
- Target: base_model(alt) - base_model(ref)
- Architecture: Gated CNN with dilated convolutions

**Variations Tested**:
| Variation | Correlation | Notes |
|-----------|-------------|-------|
| V2 Original | r=-0.04 | No learning |
| V2 + 10x data | r=0.002 | Still no correlation |
| Gated CNN | r=0.36 | Architecture matters |
| + Quantile loss | r=0.38 | Best so far |
| + Scaling | r=0.22 | Overfitting |
| + Temperature | r=-0.03 | No improvement |
| + Multi-task | r=-0.07 | Task interference |

**Conclusion**: Moderate correlation achieved (r=0.38) but:
1. Targets (base model deltas) may be inaccurate
2. Not sufficient for practical use

**Documentation**: `docs/experiments/002_delta_prediction/`

---

### Phase 2B: Validated Delta Prediction (COMPLETED ✅) ⭐ BEST APPROACH

**Status**: Implemented and tested  
**Best Result**: r=0.507 correlation (8000 samples) ⭐  
**Goal**: Use SpliceVarDB classifications to derive ground-truth delta targets

**Key Difference from Phase 2A**:
- Phase 2A: Target = base_model(alt) - base_model(ref) (possibly inaccurate)
- Phase 2B: Target = SpliceVarDB-validated delta (ground truth filtering)

**Approach** (from LABELING_STRATEGY.md):
```
Input: alt_sequence + variant_info (ref_base, alt_base)
Target: Δ derived from SpliceVarDB classification
Output: Δ directly (single forward pass)

Final score = base_scores + Δ
```

**Results by Dataset Size**:
| Samples | Correlation | ROC-AUC | PR-AUC |
|---------|-------------|---------|--------|
| 2,000 | r=0.41 | 0.58 | 0.62 |
| **8,000** | **r=0.507** | **0.589** | **0.633** |

**Key Finding**: More data significantly helps (+24% correlation improvement)

**Documentation**: 
- `docs/experiments/004_validated_delta/README.md`
- `docs/methods/VALIDATED_DELTA_PREDICTION.md`

---

### Phase 2C: Multi-Step Framework (IN PROGRESS)

**Status**: Step 1 tested (needs improvement)  
**Goal**: Decompose the problem into manageable sub-tasks

**Step 1: Binary Classification**
- Question: "Is this variant splice-altering?"
- Results: AUC=0.61, F1=0.53 (needs F1 > 0.7)
- Status: Needs improvement

**Step 2: Effect Type Classification**
- Question: "What type?" (Donor gain/loss, Acceptor gain/loss)
- Status: Not yet implemented

**Step 3: Position Localization**
- Question: "Where in the window?"
- Status: Not yet implemented

**Step 4: Delta Magnitude**
- Question: "How strong?"
- Status: Not yet implemented

**Documentation**: `docs/methods/MULTI_STEP_FRAMEWORK.md`

---

## Current Implementation Status

### Models Implemented

| Model | File | Purpose | Status |
|-------|------|---------|--------|
| `ValidatedDeltaPredictor` ⭐ | `validated_delta_predictor.py` | Single-pass delta (Approach B) | **BEST: r=0.507** |
| `DeltaPredictorV2` | `delta_predictor_v2.py` | Approach A (paired) | Tested |
| `SimpleCNNDeltaPredictor` | `hyenadna_delta_predictor.py` | Gated CNN encoder | Tested |
| `SpliceInducingClassifier` | `splice_classifier.py` | Multi-step Step 1 | Tested |
| `EffectTypeClassifier` | `splice_classifier.py` | Multi-step Step 2 | Implemented, not tested |
| `UnifiedSpliceClassifier` | `splice_classifier.py` | Multi-task | Implemented, not tested |
| `HyenaDNADeltaPredictor` | `hyenadna_delta_predictor.py` | GPU encoder | Implemented, needs GPU |

### Pending Implementations

| Model | Purpose | Priority |
|-------|---------|----------|
| `PositionLocalizer` | Multi-step Step 3 | MEDIUM |
| `DeltaMagnitudePredictor` | Multi-step Step 4 | LOW |

---

## Evaluation Metrics

### Preferred Metrics (for this domain)

| Metric | Why | Notes |
|--------|-----|-------|
| **PR-AUC (AP)** | Handles class imbalance | Primary metric |
| **F1 Score** | Balances precision/recall | Target: > 0.7 |
| **ROC-AUC** | Overall discrimination | Can be misleading |

### Avoid

| Metric | Why |
|--------|-----|
| Accuracy | Misleading with imbalanced data |

---

## Compute Resources

### Available

| Environment | Specs | Suitable For |
|-------------|-------|--------------|
| MacBook M1 | 16GB RAM, MPS | Quick iterations, small models |

### Needed (RunPods)

| Environment | Specs | Suitable For |
|-------------|-------|--------------|
| RTX 4090 | 24GB VRAM | HyenaDNA-small, larger batches |
| A40 | 48GB VRAM | HyenaDNA-medium |
| A100 | 80GB VRAM | HyenaDNA-large, fine-tuning |

---

## Next Steps (Prioritized)

### Completed ✅

1. ~~**Implement Approach B** (Single-Pass Delta)~~ → r=0.507 with 8000 samples
2. ~~**More training data**~~ → Confirmed: +24% improvement with 4x data
3. ~~**Multi-Step Step 1**~~ → AUC=0.61 (implemented, needs improvement)

### HIGH PRIORITY ⭐

| Priority | Task | Why Important | Expected Outcome |
|----------|------|---------------|------------------|
| 1 | **Multi-Step Step 2** (Effect Type) | Enables cascade | 5-class classifier |
| 2 | **Multi-Step Step 3** (Position) | Enables conditioned delta | Position ± 5nt |
| 3 | **ConditionedDeltaPredictor** | Use cascade outputs! | r > 0.70 (vs 0.507) |
| 4 | **Full SpliceVarDB** (50K) | Data scaling works | Better generalization |
| 5 | **HyenaDNA encoder** | Better sequence understanding | +5-10% improvement |

### ⭐ NEW: Cascade vs Standalone Insight

**Current `ValidatedDelta` is standalone** (r=0.507) - it ignores Multi-Step outputs!

```
WRONG (current):  alt_seq → ValidatedDelta → Δ  (learns everything from scratch)
RIGHT (proposed): alt_seq + effect_type + position → ConditionedDelta → Δ  (simpler task)
```

**Expected improvement**: r=0.507 → r>0.70 by conditioning on Steps 1-3 outputs.

### MEDIUM PRIORITY

| Priority | Task | Why Important | Expected Outcome |
|----------|------|---------------|------------------|
| 6 | **Improve Multi-Step Step 1** | Better triage | AUC > 0.75, F1 > 0.7 |
| 7 | **Longer context** (1001nt) | Distant regulatory elements | +5-10% improvement |
| 8 | **Cross-validation** | Robust estimates | Variance estimates |

### Why Multi-Step is High Priority

Multi-Step Framework provides **interpretable outputs** that ValidatedDelta cannot:

```
FDA/Stakeholder Question          ValidatedDelta     Multi-Step
─────────────────────────────     ──────────────     ───────────
"Is this variant pathogenic?"     "Δ=0.35...???"     "YES (92%)" ✅
"What type of effect?"            "Δ_donor=0.35"     "Donor gain" ✅
"Where should we target ASO?"     "max at pos 127"   "pos 127 ± 3nt" ✅
```

For clinical/regulatory approval, **you need to explain your predictions**.

---

## Documentation Structure

```
meta_layer/docs/
├── methods/                      # Methodology development
│   ├── ROADMAP.md                    # This file
│   ├── VALIDATED_DELTA_PREDICTION.md # Single-pass validated delta (recommended)
│   ├── META_RECALIBRATION.md         # Per-position splice score refinement (proposed)
│   ├── MULTI_STEP_FRAMEWORK.md       # Decomposed approach
│   ├── PAIRED_DELTA_PREDICTION.md    # Siamese delta prediction (deprecated)
│   └── GPU_REQUIREMENTS.md           # Compute requirements
│
├── experiments/                  # Detailed experiment logs
│   ├── 001_canonical_classification/
│   ├── 002_delta_prediction/
│   ├── 003_binary_classification/
│   ├── 004_validated_delta/     # ⭐ Best results (r=0.507)
│   ├── GPU_TRAINING_GUIDE.md    # RunPods setup
│   └── DATA_TRANSFER_GUIDE.md   # Data transfer instructions
│
├── setup/                        # Environment setup
│   ├── RUNPODS_COMPLETE_SETUP.md
│   ├── RUNPODS_DISK_CONFIGURATION.md
│   └── RUNPODS_STORAGE_REQUIREMENTS.md
│
├── wishlist/                     # Pending GPU experiments
│   └── GPU_EXPERIMENTS.md
│
└── *.md                         # Architecture, guides, etc.
```

---

## References

- **SpliceVarDB**: Source of ground-truth variant effect labels
- **LABELING_STRATEGY.md**: Detailed approach descriptions
- **ARCHITECTURE.md**: Model architectures
- **Session summaries**: `dev/sessions/`

---

*This roadmap will be updated as methodology development progresses.*

