# Score-Shifting Bug Analysis

**Date**: 2025-10-31  
**Issue**: Coordinate adjustments are systematically misaligned

## Test Results Summary

From 24 genes tested:
- **+ strand donors**: 0.0% exact match, 53-57% off by ±2 nt
- **+ strand acceptors**: 45.6% exact match ✅
- **- strand donors**: 0.7% exact match, 68-71% off by ±1 nt  
- **- strand acceptors**: 0.0% exact match, 68-69% off by ±1 nt

## Current Implementation

```python
# Adjustment values (default)
adjustment_dict = {
    'donor': {'plus': 2, 'minus': 1},
    'acceptor': {'plus': 0, 'minus': -1}
}

# Current code (lines 681-694)
if strand == '+':
    predictions_df = predictions_df.with_columns([
        pl.col('donor_prob').shift(-donor_plus_adj).fill_null(0).alias('donor_prob'),      # shift(-2)
        pl.col('acceptor_prob').shift(-acceptor_plus_adj).fill_null(0).alias('acceptor_prob'),  # shift(0)
        pl.col('neither_prob').alias('neither_prob')
    ])
```

## The Bug

### Understanding `.shift()` in Polars

After sorting by position:
- `shift(-n)`: Moves values DOWN (from later row indices to earlier row indices)
  - Position i gets value from position i+n
  - Example: `shift(-2)` → position 100 gets score from position 102
  
- `shift(+n)`: Moves values UP (from earlier row indices to later row indices)
  - Position i gets value from position i-n
  - Example: `shift(+2)` → position 100 gets score from position 98

### What the Adjustments Mean

The adjustment values represent **where the model PREDICTS the splice site relative to the TRUE location**:

- **Donor +2 on + strand**: Model predicts 2 nt AFTER the true site
  - True donor at position 100
  - Model gives high score at position 102
  - To FIX: We need position 100 to get the score from position 102
  - **Correct operation**: `shift(-2)` ✅

- **Acceptor 0 on + strand**: Model predicts at the correct location
  - True acceptor at position 100
  - Model gives high score at position 100
  - To FIX: No shift needed
  - **Correct operation**: `shift(0)` ✅

- **Donor +1 on - strand**: Model predicts 1 nt AFTER the true site
  - True donor at position 100
  - Model gives high score at position 101
  - To FIX: We need position 100 to get the score from position 101
  - **Correct operation**: `shift(-1)` ✅

- **Acceptor -1 on - strand**: Model predicts 1 nt BEFORE the true site
  - True acceptor at position 100
  - Model gives high score at position 99
  - To FIX: We need position 100 to get the score from position 99
  - **Correct operation**: `shift(+1)` ❌ **WRONG IN CODE!**

## Root Cause

The current code uses:
```python
pl.col('acceptor_prob').shift(-acceptor_minus_adj)  # shift(-(-1)) = shift(+1)
```

This is CORRECT for negative adjustments!

But wait... let me re-analyze the test results:

### Re-Analysis of Test Results

**+ Strand:**
- Donors (adj=+2): 0% exact, 57% off by ±2 → **Scores are NOT being shifted!**
- Acceptors (adj=0): 46% exact → **Working as expected**

**- Strand:**
- Donors (adj=+1): 0.7% exact, 71% off by ±1 → **Scores are NOT being shifted!**
- Acceptors (adj=-1): 0% exact, 69% off by ±1 → **Scores are NOT being shifted!**

## The REAL Bug

Looking at the pattern, it seems like **the adjustments are NOT being applied at all** or are being applied in the WRONG direction!

Let me check: If donors on + strand are off by +2, and the adjustment is +2, then:
- If NO adjustment applied: predictions would be at position+2 (off by +2) ✅ **MATCHES TEST RESULTS**
- If adjustment applied correctly: predictions would be at position (exact match)

**Conclusion**: The adjustments are either:
1. Not being applied at all, OR
2. Being applied in the opposite direction

### Hypothesis: Wrong Sign Convention

The current code interprets adjustment values as "shift amount", but they might actually mean "offset amount to SUBTRACT"!

If adjustment_dict values represent "model offset from true position":
- Donor +2 on + strand: Model is +2 ahead, so we need to shift scores BACK by 2
- This means: `position` should get score from `position + 2`
- Which is: `shift(-2)` ✅ **CODE IS CORRECT**

But test results show this ISN'T working!

## Debugging Steps

1. **Check if adjustments are being loaded**: Print `self.adjustment_dict` values
2. **Check if shift is being applied**: Add debug logging to see before/after scores
3. **Verify shift direction**: Manually check a few positions

## Possible Issues

1. **Adjustment dict not loaded**: Using wrong default values
2. **Shift not applied**: Code path not executing
3. **Scores overwritten**: Later code overwrites adjusted scores
4. **Column name mismatch**: Using wrong column names (donor_prob vs donor_score)

## Next Steps

1. Add debug logging to `_apply_coordinate_adjustments()`
2. Check what column names are actually in the DataFrame
3. Verify the adjustment_dict values being used
4. Test on a single gene with detailed logging

---

**Status**: Bug identified but root cause unclear. Need more debugging.

