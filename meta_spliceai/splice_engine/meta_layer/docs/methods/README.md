# Meta-Layer Methodology Documentation

This directory contains documentation for the various methodological approaches being developed for alternative splice site prediction.

## Document Index

| Document | Description | Status |
|----------|-------------|--------|
| [ROADMAP.md](ROADMAP.md) | High-level methodology development roadmap | Active |
| [APPROACH_A_PAIRED.md](APPROACH_A_PAIRED.md) | Siamese/paired delta prediction | Tested |
| [APPROACH_B_SINGLE_PASS.md](APPROACH_B_SINGLE_PASS.md) | Single-pass delta prediction | Proposed |
| [MULTI_STEP_FRAMEWORK.md](../MULTI_STEP_FRAMEWORK.md) | Decomposed approach | In Progress |
| [COMPARISON.md](COMPARISON.md) | Method comparison summary | Planned |

## Quick Reference

### Approach Summary

```
APPROACH A (Paired)               APPROACH B (Single-Pass)
─────────────────────────         ─────────────────────────
ref_seq ──→ encoder ──┐           alt_seq ──→ encoder ──┐
                      ├─→ diff    ref_base ──→ embed ──┼─→ delta
alt_seq ──→ encoder ──┘           alt_base ──→ embed ──┘

Target: base_delta                Target: validated_delta
Status: r=0.38                    Status: r=0.41 (BEST)


TARGET FORMAT (Both Approaches):
─────────────────────────────────
target = [Δ_donor, Δ_acceptor, Δ_neither]  # continuous floats in [-1, 1]
Example: [+0.35, -0.02, -0.33] = donor gain (+0.35)

Validated targets:
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

- **Paired Prediction**: Tested, correlation r=0.38
- **Validated Single-Pass**: Tested, correlation r=0.41 (**BEST**)
- **Binary Classification**: Tested, needs improvement (F1=0.53, AUC=0.61)

## Priority

1. **HIGH**: Improve Validated Single-Pass with more data (8k+ samples)
2. **HIGH**: Improve Binary Classification (F1 > 0.7)
3. **MEDIUM**: Test Multi-Step Steps 2-4
4. **LOW**: Compare all approaches

## Related Documentation

- `../experiments/` - Detailed experiment logs
- `../LABELING_STRATEGY.md` - Label derivation strategies
- `../ARCHITECTURE.md` - Model architectures
- `../../dev/sessions/` - Session summaries

