# Score-Shifting: No Improvement After Fix

**Date**: 2025-10-31  
**Status**: ❌ **FIX DID NOT WORK**

## Test Results (After "Fix")

**IDENTICAL** to before:
- **Average F1**: 0.186 (target: ≥0.7)
- **Donor F1**: 0.004 (0.3% exact match)
- **Acceptor F1**: 0.370 (28.5% exact match)
- **Coverage**: 100.0% ✅

### Stratified Results

| Category | Metric | Value |
|----------|--------|-------|
| **+ Strand Donors** | Exact Match | 0.0% |
| **+ Strand Acceptors** | Exact Match | 45.6% |
| **- Strand Donors** | Exact Match | 0.7% |
| **- Strand Acceptors** | Exact Match | 0.0% |

## Analysis

The "fix" (correcting column names from `*_score` to `*_prob`) did NOT improve results. This means:

1. ✅ The column names are now correct (`*_prob` at adjustment time)
2. ✅ The adjustments are being applied (no errors)
3. ❌ But the adjustments are STILL WRONG (same misalignment pattern)

## Possible Root Causes

### 1. Wrong Shift Direction

The current code uses:
```python
pl.col('donor_prob').shift(-donor_plus_adj)  # shift(-2) for +strand donors
```

With `shift(-2)`:
- Position i gets value from position i+2
- This means: position 100 gets score from position 102

**Question**: Is this the correct direction?

If the model predicts donors at position+2 (2 nt AFTER true site):
- True donor at 100
- Model high score at 102
- We want position 100 to get the score from 102
- **This requires**: `shift(-2)` ✅ CORRECT

But test results show donors are STILL off by ±2! This suggests:
- Either the shift is not being applied
- Or the adjustment VALUE is wrong
- Or we need the OPPOSITE shift

### 2. Wrong Adjustment Values

Default values:
```python
adjustment_dict = {
    'donor': {'plus': 2, 'minus': 1},
    'acceptor': {'plus': 0, 'minus': -1}
}
```

**Question**: Are these values correct for SpliceAI?

### 3. Adjustment Applied But Then Overwritten

The adjustments might be applied correctly, but then:
- Later code overwrites the adjusted scores
- Or the adjusted DataFrame is not saved
- Or the wrong DataFrame is used for output

## Next Steps

1. **Add debug logging** to verify:
   - Adjustment values being used
   - Scores before adjustment
   - Scores after adjustment
   - That adjusted DataFrame is actually saved

2. **Test on a single gene** with detailed logging

3. **Verify adjustment values** are correct for SpliceAI

4. **Check if adjustments are being overwritten** downstream

---

**Status**: Investigation needed to find the real bug!

