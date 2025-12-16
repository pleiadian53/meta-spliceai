# Meta-Layer Methodology Documentation

This directory contains documentation for the various methodological approaches being developed for alternative splice site prediction.

## Document Index

| Document | Description | Status |
|----------|-------------|--------|
| [ROADMAP.md](ROADMAP.md) | High-level methodology development roadmap | Active |
| [APPROACH_A_PAIRED.md](APPROACH_A_PAIRED.md) | Siamese/paired delta prediction | Tested (r=0.38) |
| [APPROACH_B_SINGLE_PASS.md](APPROACH_B_SINGLE_PASS.md) | Single-pass validated delta prediction | **BEST (r=0.507)** ⭐ |
| [GPU_REQUIREMENTS.md](GPU_REQUIREMENTS.md) | Compute resource guide | Active |
| [MULTI_STEP_FRAMEWORK.md](../MULTI_STEP_FRAMEWORK.md) | Decomposed approach | In Progress |

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

| Method | Alternative Splice Sites | New Isoforms | Drug Targets | Why |
|--------|-------------------------|--------------|--------------|-----|
| **ValidatedDelta (B)** ⭐ | ✅ Best | ✅ Good | ✅ Best | Quantitative delta scores enable ranking |
| Multi-Step Framework | ✅ Good | ⚠️ Limited | ✅ Good | Binary decisions for triage |
| Paired Delta (A) | ⚠️ Moderate | ⚠️ Limited | ⚠️ Moderate | Noisy targets limit accuracy |

### Why ValidatedDelta is Best for Drug Discovery

1. **Quantitative Predictions**: Delta scores (not just yes/no) let you rank variants by effect magnitude
2. **Both Gains AND Losses**: Detects donor/acceptor gains and losses (4 effect types)
3. **Ground-Truth Training**: Uses SpliceVarDB-validated labels, not potentially wrong base model predictions
4. **Scalable**: More data → better results. Full SpliceVarDB should achieve r>0.60

### Workflow for Drug Target Discovery

```
1. Screen candidate variants
   └─→ ValidatedDeltaPredictor: Get delta scores

2. Prioritize by effect magnitude
   └─→ Sort by |Δ_donor| + |Δ_acceptor|

3. Identify effect type
   └─→ Δ_donor > 0.1 = "Donor gain" (new splice site)
   └─→ Δ_donor < -0.1 = "Donor loss" (lost splice site)
   └─→ Similar for acceptor

4. Predict new isoforms
   └─→ Donor gain + nearby acceptor = potential new exon
   └─→ Donor loss = potential exon skipping

5. Validate top candidates
   └─→ RNA-seq, minigene assays
```

### Limitations & Future Work

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Current r=0.507 | ~50% variance explained | Scale to 50K samples, use HyenaDNA |
| Point mutations only | Doesn't handle indels well | Extend architecture |
| Position-agnostic | Doesn't predict WHERE the new site is | Multi-Step Framework Step 3 |

---

## Related Documentation

- `../experiments/` - Detailed experiment logs
- `../experiments/004_validated_delta/` - Best results
- `../LABELING_STRATEGY.md` - Label derivation strategies
- `../ARCHITECTURE.md` - Model architectures

