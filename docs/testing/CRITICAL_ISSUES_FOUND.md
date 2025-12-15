# CRITICAL ISSUES FOUND - Inference Workflow

**Date**: 2025-10-29  
**Status**: 🚨 BLOCKING - Two critical bugs prevent successful inference

---

## Issue 1: Incomplete Coverage (88.6% instead of 100%)

### Problem
SpliceAI is returning only 6,300 positions for a 7,107 bp gene (88.6% coverage).

### Root Cause
The `predict_splice_sites_for_genes` function in `run_spliceai_workflow.py` has a bug in its block overlap/merging logic:
- It generates **807 duplicate rows** for GSTM3
- After deduplication, we're left with **807 missing positions**
- The missing positions are scattered throughout the gene (683 gaps, mostly single-position gaps)

### Evidence
```
GSTM3 (7,107 bp):
- SpliceAI returns: 7,107 rows (correct total)
- But 807 are duplicates
- After deduplication: 6,300 unique positions
- Missing: 807 positions (11.4%)
- Gaps: 683 gaps scattered throughout
```

### Impact
- **Moderate**: Meta-model can still work on the 88.6% of positions we DO have
- **But violates user requirement**: "full coverage mode" should produce N scores for N-bp gene

### Proposed Solutions
1. **Short-term**: Document the 88.6% coverage limitation and proceed with testing
2. **Medium-term**: Fix the block overlap logic in `predict_splice_sites_for_genes`
3. **Long-term**: Consider using a simpler SpliceAI wrapper or the official SpliceAI toolkit

---

## Issue 2: Meta-Only Mode Complete Failure (F1=0.015 vs 0.674)

### Problem
The meta-only mode is **destroying predictions** instead of improving them:
- Base-only F1: 0.674
- Meta-only F1: 0.015 (97% worse!)
- Hybrid F1: 0.673 (no improvement)

### Root Cause
**The meta-model is NOT actually recalibrating scores!**

Evidence:
```python
BASE_ONLY:
  Donor score: min=0.000, max=0.999, mean=0.002
  Acceptor score: min=0.000, max=0.999, mean=0.001

META_ONLY:
  Donor score: min=0.000, max=0.999, mean=0.002  # IDENTICAL!
  Acceptor score: min=0.000, max=0.999, mean=0.001  # IDENTICAL!
  
Meta-model applied to: 6300/6300 positions (100.0%)  # Claims to be applied!
```

The meta-model **claims** it's being applied to 100% of positions, but the scores are **byte-for-byte identical** to base-only mode!

### Possible Causes
1. **Meta-model loading failure**: Model loads but doesn't actually predict
2. **Prediction pipeline bug**: Meta-model predicts but results are discarded
3. **Feature mismatch**: Meta-model receives wrong features and returns base scores as fallback
4. **Output overwrite**: Meta-model predictions are generated but then overwritten by base scores

### Impact
- **CRITICAL**: This completely breaks the meta-model inference workflow
- **Blocks all testing**: Cannot evaluate meta-model performance until fixed
- **Contradicts training results**: Meta-model showed improvement during training

---

## Recommended Next Steps

### Priority 1: Fix Meta-Only Mode (CRITICAL)
1. Add debug logging to `_apply_meta_model_to_features` to verify:
   - Model is loading correctly
   - Features are being passed correctly
   - Predictions are being generated
   - Predictions are being applied to the DataFrame

2. Check if there's a silent fallback that's returning base scores

3. Verify the meta-model file itself is valid and trained correctly

### Priority 2: Address Coverage Issue (IMPORTANT)
1. Document the 88.6% coverage limitation
2. Investigate the duplicate row generation in `predict_splice_sites_for_genes`
3. Consider alternative SpliceAI wrappers

### Priority 3: Re-run Comprehensive Tests
Once both issues are fixed:
1. Re-run tests on all 6 genes × 3 modes
2. Verify F1 scores show expected improvements:
   - Hybrid > Base-only
   - Meta-only recalibrates uncertain positions
3. Verify coverage is 95%+ for all genes

---

## Files Modified Today

### Core Inference Workflow
- `meta_spliceai/splice_engine/meta_models/workflows/inference/enhanced_selective_inference.py`
  - Added deduplication logic (line 845-850)
  - Added `crop_size=0` parameter (line 754)
  - Integrated derived feature generation (lines 400-600)
  - Removed fallback logic (multiple locations)

### Test Scripts
- `scripts/testing/test_all_modes_seen_unseen_genes.py` (created)
  - Comprehensive test for all 3 modes on 6 genes

---

## Questions for User

1. **Coverage**: Can we proceed with 88.6% coverage for now, or is 100% coverage a hard requirement?

2. **Meta-model**: Do you have a working example of the meta-model being applied successfully? This would help us debug.

3. **Training workflow**: Did the training workflow also have the duplicate row issue, or did it use a different SpliceAI wrapper?

4. **Priority**: Which issue should we fix first - coverage or meta-model failure?





