# Meta-Layer Methodology Documentation

This directory contains documentation for the various methodological approaches being developed for alternative splice site prediction.

## Document Index

| Document | Description | Status |
|----------|-------------|--------|
| [ROADMAP.md](ROADMAP.md) | High-level methodology development roadmap | Active |
| [APPROACH_A_PAIRED.md](APPROACH_A_PAIRED.md) | Siamese/paired delta prediction | Tested (r=0.38) |
| [APPROACH_B_SINGLE_PASS.md](APPROACH_B_SINGLE_PASS.md) | Single-pass validated delta prediction | **BEST (r=0.507)** ⭐ |
| [MULTI_STEP_FRAMEWORK.md](MULTI_STEP_FRAMEWORK.md) | Decomposed classification approach | ⭐ **Best for Interpretability** |
| [GPU_REQUIREMENTS.md](GPU_REQUIREMENTS.md) | Compute resource guide | Active |

---

## 🏆 Method Selection Guide

### TL;DR: Which Method to Use?

| Your Goal | Best Method | Why |
|-----------|-------------|-----|
| **"Should I investigate this variant?"** | Multi-Step Step 1 | Direct yes/no answer |
| **"What kind of effect is this?"** | Multi-Step Step 2 | Donor gain/loss, Acceptor gain/loss |
| **"Where should I target my ASO?"** | Multi-Step Step 3 | Position localization |
| **"How strong is the effect?"** | ValidatedDelta | Continuous delta scores |
| **"Rank variants by severity"** | ValidatedDelta | Quantitative ranking |
| **"Explain to FDA/stakeholders"** | Multi-Step | Interpretable decision trail |

### Two Approaches → One Integrated Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    INTEGRATED CASCADE PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  MULTI-STEP FRAMEWORK (Steps 1-3)                                       │
│  ─────────────────────────────────                                       │
│  Step 1: "Is this pathogenic?" ──────────────────┐                      │
│  Step 2: "What type of effect?" ─────────────────┼──→ CASCADE           │
│  Step 3: "Where exactly?" ───────────────────────┘    OUTPUTS           │
│                                                           │              │
│                                                           ↓              │
│  CONDITIONED DELTA (Step 4) ⭐ NEW                                      │
│  ─────────────────────────────────                                       │
│  Input:  alt_seq + effect_type (Step 2) + position (Step 3)             │
│  Output: Δ magnitude at the identified position                          │
│                                                                          │
│  ⚠️ CURRENT GAP: ValidatedDelta is STANDALONE (ignores Steps 1-3)      │
│  🎯 TODO: Implement ConditionedDeltaPredictor using cascade outputs     │
│                                                                          │
│  Expected: r=0.507 (standalone) → r>0.70 (conditioned)                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Cascading Matters

The key insight of Multi-Step is **using previous answers to simplify subsequent questions**:

| Step | Without Cascade | With Cascade | Simplification |
|------|-----------------|--------------|----------------|
| Step 2 | Predict for ALL variants | Only splice-altering | Cleaner training data |
| Step 3 | Find position anywhere | Find donor (if donor_gain) | Narrower search |
| Step 4 | Predict [L, 2] deltas | Predict Δ at position 127 | **Point estimate!** |

**Current `ValidatedDelta` (r=0.507) is standalone** - it ignores this cascade!  
**Proposed `ConditionedDelta`** would use Steps 1-3 outputs → much simpler task → better performance.

## Quick Reference

### Approach Summary

```
APPROACH A (Paired)               APPROACH B (Single-Pass) ⭐ BEST
─────────────────────────         ─────────────────────────────────
ref_seq ──→ encoder ──┐           alt_seq ──→ encoder ──┐
                      ├─→ diff    ref_base ──→ embed ──┼─→ delta
alt_seq ──→ encoder ──┘           alt_base ──→ embed ──┘

Target: base_delta                Target: validated_delta
Status: r=0.38                    Status: r=0.507 (8K samples) ⭐


TARGET FORMAT (Both Approaches):
─────────────────────────────────
target = [Δ_donor, Δ_acceptor, Δ_neither]  # continuous floats in [-1, 1]
Example: [+0.35, -0.02, -0.33] = donor gain (+0.35)

Validated targets (Approach B):
  Splice-altering: target = base_model(alt) - base_model(ref)  # Trust base model
  Normal:          target = [0.0, 0.0, 0.0]                    # Override!


MULTI-STEP FRAMEWORK
─────────────────────
Step 1: Is splice-altering? → Binary (AUC=0.61, needs >0.7)
Step 2: What type?          → Multi-class (NOT IMPLEMENTED)
Step 3: Where?              → Localization (NOT IMPLEMENTED)
Step 4: How strong?         → Regression (NOT IMPLEMENTED)
```

### Key Differences

| Aspect | Approach A | Approach B | Multi-Step |
|--------|------------|------------|------------|
| Input | ref + alt | alt + var_info | alt + var_info |
| Target | base_delta | validated_delta | classification |
| Forward passes | 2 | 1 | 1-4 |
| Interpretability | Low | Medium | High |

## Current Status

| Method | Correlation | Status | Recommended |
|--------|-------------|--------|-------------|
| Paired Prediction (A) | r=0.38 | Tested | No |
| **Validated Single-Pass (B)** | **r=0.507** | ⭐ **BEST** | **Yes** |
| Binary Classification | AUC=0.61 | Needs improvement | For triage |

**Key Finding**: More data significantly helps. 8000 samples improved correlation by +24%.

## Priority

1. ✅ **DONE**: Validated Single-Pass with 8K samples → r=0.507
2. **HIGH**: Scale to full SpliceVarDB (50K samples) on GPU → Expected: r>0.60
3. **HIGH**: HyenaDNA encoder (GPU required)
4. **MEDIUM**: Improve Binary Classification (F1 > 0.7)
5. **LOW**: Multi-Step Steps 2-4

---

## 🎯 Application to RNA Therapeutics

### Which Methods Are Most Promising?

| Method | Triage | Effect Type | Position | Quantification | Explainability |
|--------|--------|-------------|----------|----------------|----------------|
| **Multi-Step** ⭐ | ✅ Best | ✅ Best | ✅ Best | ⚠️ Indirect | ✅ **Best** |
| **ValidatedDelta** ⭐ | ⚠️ Threshold | ⚠️ Derived | ⚠️ Max pos | ✅ Best | ⚠️ Numbers only |
| Paired Delta (A) | ⚠️ Poor | ⚠️ Derived | ⚠️ Noisy | ⚠️ r=0.38 | ❌ Poor |

### Why Multi-Step is Critical for Drug Discovery

1. **Regulatory Approval**: FDA requires mechanistic understanding
   - ✅ "This variant is pathogenic because it creates a new donor site at position 127"
   - ❌ "Δ_donor = 0.35" (what does this mean?)

2. **ASO Target Design**: Need to know WHERE to target
   - ✅ Multi-Step Step 3 gives position localization
   - ❌ ValidatedDelta gives max delta position (indirect)

3. **Clinical Decisions**: Binary yes/no for treatment decisions
   - ✅ Multi-Step: "P(splice-altering) = 0.92 → TREAT"
   - ❌ ValidatedDelta: "Δ = 0.35 → ??? → need threshold → TREAT?"

### Recommended Workflow: Combined Approach

```
RNA THERAPEUTICS VARIANT SCREENING PIPELINE
────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: TRIAGE (Multi-Step Step 1)                                    │
│  Input:  10,000 candidate variants                                      │
│  Filter: P(splice-altering) > 0.5                                       │
│  Output: 1,000 high-priority variants                                   │
│  Time:   ~1 minute                                                      │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: EFFECT TYPING (Multi-Step Step 2)                             │
│  Input:  1,000 high-priority variants                                   │
│  Output: Classified by effect type                                      │
│    - 400 Donor gain (new cryptic donors)                                │
│    - 200 Donor loss (exon skipping)                                     │
│    - 250 Acceptor gain/loss                                             │
│    - 150 Complex                                                        │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: QUANTIFICATION (ValidatedDelta)                               │
│  Input:  1,000 classified variants                                      │
│  Output: Delta scores [Δ_donor, Δ_acceptor]                             │
│  Use:    Rank by |Δ| for prioritization                                 │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: LOCALIZATION (Multi-Step Step 3) [FUTURE]                     │
│  Input:  Top 100 candidates (by delta magnitude)                        │
│  Output: Exact affected positions ± 5nt                                 │
│  Use:    Design 18-25mer ASO targeting this position                    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: EXPERIMENTAL VALIDATION                                       │
│  Methods: RNA-seq, minigene assays, RT-PCR                              │
│  Top 10 candidates → wet lab                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Limitations & Next Steps

| Limitation | Impact | Solution | Priority |
|------------|--------|----------|----------|
| Step 1 AUC=0.61 | Triage accuracy | More data, HyenaDNA | ⭐ HIGH |
| Step 2 not tested | No effect typing | Run experiments | MEDIUM |
| Step 3 not implemented | No localization | Build model | ⭐ HIGH |
| r=0.507 for ValidatedDelta | ~50% variance explained | Scale to 50K | HIGH |

---

## Related Documentation

- `../experiments/` - Detailed experiment logs
- `../experiments/004_validated_delta/` - Best results
- `../LABELING_STRATEGY.md` - Label derivation strategies
- `../ARCHITECTURE.md` - Model architectures

