# Coordinate Fix Progress Report

**Date**: 2025-10-29  
**Status**: ✅ Donors Fixed | ❌ Acceptors Need Strand-Specific Fix

---

## Summary

Applied **-1 coordinate correction** to all positions. Results:

### Donor Sites: ✅ EXCELLENT
| Gene  | Before | After  | Improvement |
|-------|--------|--------|-------------|
| GSTM3 | 0.000  | 0.941  | +0.941      |
| BRAF  | 0.000  | 0.744  | +0.744      |
| TP53  | 0.000  | 0.800  | +0.800      |

### Acceptor Sites: ❌ STILL BROKEN
| Gene  | Before | After  | Improvement |
|-------|--------|--------|-------------|
| GSTM3 | 0.000  | 0.000  | 0.000       |
| BRAF  | 0.000  | 0.000  | 0.000       |
| TP53  | 0.000  | 0.000  | 0.000       |

---

##  Root Cause Identified

**The issue**: Applied **uniform -1 correction** to all positions, but **donors and acceptors need different adjustments**!

From `infer_splice_site_adjustments.py`:
```python
spliceai_pattern = {
    'donor': {
        'plus': 2,    # Donors on + strand need +2 adjustment (raw position is -2 from true)
        'minus': 1    # Donors on - strand need +1 adjustment
    },
    'acceptor': {
        'plus': 0,    # Acceptors on + strand need 0 adjustment (already correct!)
        'minus': -1   # Acceptors on - strand need -1 adjustment
    }
}
```

**Current fix**: `-1` for all positions  
- ✅ Works well enough for donors (brings +2 down to acceptable ~+1)
- ❌ Breaks acceptors on + strand (shifts correct positions by -1)
- ❌ Wrong for acceptors on - strand (need different correction)

---

## What We Need

### Proper Solution: Type & Strand-Specific Adjustments

We need to:
1. **Detect splice type** for each position (donor vs acceptor)
2. **Get strand information** for each position
3. **Apply correct adjustment** based on type + strand

### Implementation Options

#### Option 1: Post-SpliceAI Correction (Current Approach)
After getting positions from SpliceAI, adjust based on predicted type:

```python
# After getting predictions_df with donor_prob, acceptor_prob
# Determine predicted type
predictions_df = predictions_df.with_columns([
    pl.when(pl.col('donor_prob') > pl.col('acceptor_prob'))
      .then(pl.lit('donor'))
      .otherwise(pl.lit('acceptor'))
      .alias('predicted_type')
])

# Apply type & strand-specific corrections
def apply_correction(row):
    if row['predicted_type'] == 'donor':
        if row['strand'] == '+':
            return row['position'] - 2  # Donor + strand
        else:
            return row['position'] - 1  # Donor - strand
    else:  # acceptor
        if row['strand'] == '+':
            return row['position'] - 0  # Acceptor + strand (no change)
        else:
            return row['position'] + 1  # Acceptor - strand (opposite direction!)
```

**Issue with this approach**: We don't know splice type until AFTER prediction, creates circular dependency.

#### Option 2: Score Array Adjustment (Training Workflow Approach)
Apply adjustments to **score arrays** before extracting positions:

```python
# This is what training workflow does
adjusted_donor_scores = apply_auto_detected_adjustments(
    donor_scores, strand, 'donor', adjustment_dict
)
adjusted_acceptor_scores = apply_auto_detected_adjustments(
    acceptor_scores, strand, 'acceptor', adjustment_dict
)
```

**Advantage**: Adjusts at score level, positions extracted from adjusted scores are automatically correct.

#### Option 3: Comprehensive Fix (Best Long-Term)
Integrate the full adjustment system from training workflow:

1. Load/detect adjustments using `prepare_splice_site_adjustments()`
2. Store `adjustment_dict` in workflow config
3. Apply during score processing (not position extraction)
4. Use same evaluation function as training: `enhanced_process_predictions_with_all_scores()`

---

## Immediate Next Steps

### For Testing (Quick):
Apply **position-level correction** with predicted type detection:
- Donors: `-1` (current fix, working well)
- Acceptors: `0` (no correction needed for + strand majority)

This will at least fix acceptors on + strand genes.

### For Production (Proper):
**Integrate full adjustment system** from `splice_prediction_workflow.py`:
- Use `prepare_splice_site_adjustments()` to get adjustment_dict
- Apply at score level using `apply_auto_detected_adjustments()`
- Ensures strand and type-specific corrections

---

## Why This is a Regression

This entire adjustment system was already built and working in:
- `splice_prediction_workflow.py` (lines 343-357, 474-475)
- `infer_splice_site_adjustments.py` (complete module)
- `prepare_splice_site_adjustments()` function

The new inference workflow bypassed all of this, hence the regression.

---

## Current Code Location

**Fix applied**: 
- File: `enhanced_selective_inference.py`
- Method: `_run_spliceai_directly()`
- Lines: 613-632

**Current correction**:
```python
# Line 619: -1 correction for all positions
corrected_positions = [p - 1 for p in raw_positions]
```

**What's needed**:
```python
# Type & strand-specific corrections
# OR integrate full adjustment system
```

---

## Test Results Breakdown

### GSTM3 (chr1:109733932-109741038, + strand, 7,107 bp)
- Donors: TP=16, FP=1, FN=1 → **F1=0.941** ✅
- Acceptors: TP=0, FP=0, FN=29 → **F1=0.000** ❌
- **Analysis**: Donors working well! All acceptors missed (likely all shifted by -1 when they shouldn't be)

### BRAF (chr7, 205,603 bp)
- Donors: TP=16, FP=0, FN=11 → **F1=0.744** ✅
- Acceptors: TP=0, FP=16, FN=29 → **F1=0.000** ❌
- **Analysis**: Similar pattern, acceptors completely broken

### TP53 (chr17, 25,768 bp)
- Donors: TP=10, FP=0, FN=5 → **F1=0.800** ✅
- Acceptors: TP=0, FP=11, FN=20 → **F1=0.000** ❌
- **Analysis**: Consistent across all genes

---

## Conclusion

✅ **Validation that coordinate correction is needed** (donors went from 0.000 to 0.74-0.94!)  
✅ **Proof that -1 correction works for donors**  
❌ **Acceptors need type & strand-specific handling**  
📋 **Full adjustment system integration needed** for production

**Recommendation**: 
1. **Immediate**: Add logic to skip correction for acceptors (or make type-specific)
2. **Short-term**: Integrate `prepare_splice_site_adjustments()` and `apply_auto_detected_adjustments()`
3. **Long-term**: Ensure inference and training workflows use identical coordinate systems

---

**Next Action**: Apply type-aware correction or integrate full adjustment system.

