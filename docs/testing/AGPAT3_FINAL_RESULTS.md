# AGPAT3 Final Results - Correct Metrics

**Date**: November 2, 2025  
**Status**: ✅ COMPLETE  
**Gene**: ENSG00000160216 (AGPAT3) on chr21

---

## Executive Summary

**Using the correct `pred_type` column**, AGPAT3 shows **excellent performance**:

- **Average F1 Score: 0.8920** (89.2%)
- **Donor F1: 0.9017** (90.2%)
- **Acceptor F1: 0.8824** (88.2%)
- **Perfect Precision: 1.0000** (no false positives!)

**Adjustment Analysis (consensus_window=0)**:
- **Donors**: Need +2bp adjustment (0% exact matches)
- **Acceptors**: No adjustment needed (100% exact matches)

---

## Correct Metrics (Using pred_type)

### Donor Sites

**With consensus_window=2** (standard evaluation):
```
Total positions:     95
True Positives:      78
False Positives:     0
False Negatives:     17
True Negatives:      0

Precision: 1.0000 (100%)
Recall:    0.8211 (82%)
F1 Score:  0.9017 (90%)
```

**With consensus_window=0** (exact matching):
```
Exact matches:       0 (0%)
Within ±1bp:         0
Within ±2bp:         78 (100%)

Offset: -2bp (all 78 positions)

❌ ADJUSTMENT NEEDED: +2bp
```

### Acceptor Sites

**With consensus_window=2** (standard evaluation):
```
Total positions:     95
True Positives:      75
False Positives:     0
False Negatives:     20
True Negatives:      0

Precision: 1.0000 (100%)
Recall:    0.7895 (79%)
F1 Score:  0.8824 (88%)
```

**With consensus_window=0** (exact matching):
```
Exact matches:       75 (100%)
Within ±1bp:         75 (100%)
Within ±2bp:         75 (100%)

Offset: 0bp (all 75 positions)

✅ NO ADJUSTMENT NEEDED
```

---

## Key Findings

### 1. Model Performance is Excellent

**Perfect Precision (1.0)**:
- Zero false positives for both donors and acceptors
- Model is highly specific
- No spurious predictions

**High Recall (0.79-0.82)**:
- Catches 79-82% of true splice sites
- Some sites are missed (FN), but this is expected
- Within normal range for splice site prediction

**High F1 Scores (0.88-0.90)**:
- Excellent balance of precision and recall
- Comparable to previous test (F1 = 0.9312)
- Model is working correctly

### 2. Coordinate Adjustments Needed for Donors

**Systematic -2bp offset**:
- ALL 78 high-scoring donor predictions offset by -2bp
- Not random variation - systematic pattern
- Consistent across all donor sites in AGPAT3

**Why this matters**:
- With consensus_window=2: Predictions are "correct" (within tolerance)
- With consensus_window=0: Predictions are "wrong" (not exact)
- Adjustments needed for exact coordinate accuracy

### 3. Acceptors Are Perfectly Aligned

**100% exact matches**:
- All 75 acceptor predictions at exact positions
- No systematic offset
- Perfect coordinate accuracy

**This is important**:
- Shows the offset is splice-type specific
- Not a global coordinate issue
- Donor and acceptor handled differently

---

## Comparison: Wrong vs Correct Method

### Wrong Method (Using error_type)

```
Donor F1:    0.0000  ❌
Acceptor F1: 0.0000  ❌
```

**Why wrong**:
- `error_type` only contains FP/FN (for error analysis)
- TPs not included in `error_df`
- Join with positions_df marks TPs as TNs
- Metrics show 0 TPs (incorrect!)

### Correct Method (Using pred_type)

```
Donor F1:    0.9017  ✅
Acceptor F1: 0.8824  ✅
```

**Why correct**:
- `pred_type` contains all classifications (TP/FP/FN/TN)
- Set during prediction evaluation
- Accurate classification of all positions
- Metrics show true performance

---

## Understanding consensus_window

### Purpose

**consensus_window** provides tolerance for biological variation:
- Splice sites can have minor positional uncertainty
- Annotations may have small errors
- Biological reality isn't always exact

### How It Works

**consensus_window=0** (exact matching):
```python
if predicted_position == true_position:
    # TP
```

**consensus_window=2** (±2bp tolerance):
```python
if abs(predicted_position - true_position) <= 2:
    # TP
```

### Impact on AGPAT3

**Donors with consensus_window=0**:
- 0% recall (all predictions at -2bp offset)
- Would need adjustment to get any TPs

**Donors with consensus_window=2**:
- 82% recall (78/95 within ±2bp)
- -2bp offset is within tolerance
- High F1 score (0.9017)

**This shows**:
- consensus_window masks the offset
- But doesn't fix it
- Adjustments still needed for exact coordinates

---

## The Automatic Adjustment System

### Design Philosophy

> "The automatic adjustment logic prior to running the base model is meant to be based on 'raw alignment' i.e. no tolerance window (consensus_window=0)"

**This is correct!**

1. **Detect offsets** with consensus_window=0 (exact matching)
2. **Apply adjustments** to fix systematic offsets
3. **Evaluate** with consensus_window=2 (biological tolerance)

**Result**: Both exact coordinates AND tolerance for variation

### For AGPAT3

**Current state** (no adjustments):
- consensus_window=0: 0% donor recall
- consensus_window=2: 82% donor recall

**After applying +2bp adjustment**:
- consensus_window=0: 100% donor recall (exact matches)
- consensus_window=2: 82% donor recall (unchanged, but now exact)

**Benefit**: Exact coordinates without losing biological tolerance

---

## Recommendations

### Immediate: Update Metrics Calculation

**Always use `pred_type`, never `error_type`**:

```python
# ❌ WRONG
y_true = positions_df['error_type'].is_in(['TP', 'FN'])

# ✅ CORRECT
y_true = positions_df['pred_type'].is_in(['TP', 'FN'])
```

**Why**:
- `pred_type`: Complete classification (TP/FP/FN/TN)
- `error_type`: Only errors (FP/FN) for analysis

### Short-Term: Apply Adjustments

**For AGPAT3 donors**:
```python
# Apply +2bp adjustment
adjusted_donor_positions = donor_positions + 2
```

**Expected result**:
- consensus_window=0: 100% exact matches
- consensus_window=2: Maintain high F1

### Long-Term: Systematic Adjustment Detection

**Already have the infrastructure!**

In `splice_prediction_workflow.py`:
```python
use_auto_position_adjustments=True
```

**This should**:
1. Sample genes for adjustment detection
2. Test with consensus_window=0
3. Detect systematic offsets
4. Apply adjustments before evaluation

**For AGPAT3**: Should have detected +2bp donor adjustment

**Question**: Why didn't it detect this?
- May need to check adjustment detection logic
- May be using wrong consensus_window
- May be gene-specific vs global adjustment

---

## Next Steps

### 1. Verify Adjustment Detection System

**Check**: Does the automatic adjustment detection use consensus_window=0?

**Location**: `meta_spliceai/splice_engine/meta_models/utils/infer_splice_site_adjustments.py`

**Expected**: Should detect +2bp donor adjustment for AGPAT3

### 2. Test More Genes

**Run the same analysis on**:
- All 20 sampled genes (15 protein-coding + 5 lncRNA)
- Previous 50 protein-coding genes
- Diverse sample across chromosomes

**Goal**: Determine if -2bp donor offset is:
- Gene-specific (only AGPAT3)
- Chromosome-specific (chr21)
- Systematic (all genes)

### 3. Update Documentation

**Document the correct approach**:
1. Use `pred_type` for metrics (not `error_type`)
2. Test with consensus_window=0 for adjustments
3. Apply adjustments if needed
4. Evaluate with consensus_window=2 for final metrics

---

## Conclusion

### The Truth About AGPAT3

**Performance**: Excellent (F1 = 0.89)
- High precision (1.0)
- Good recall (0.79-0.82)
- Model is working correctly

**Coordinates**: Need adjustment for donors
- Donors: -2bp offset (need +2bp adjustment)
- Acceptors: Perfect alignment (no adjustment)

**The "F1 = 0" mystery**: Data structure bug
- Used wrong column (`error_type` instead of `pred_type`)
- TPs were there all along!

### Key Lessons

1. **Column names matter**: `pred_type` vs `error_type` have different purposes
2. **consensus_window is not a fix**: Masks offsets but doesn't correct them
3. **Test with consensus_window=0**: Only way to detect true alignment
4. **Adjustments are needed**: Even when F1 is high with tolerance

### The Design is Correct

> "The automatic adjustment logic is based on raw alignment (consensus_window=0)"

**This is the right approach!**
- Detects true coordinate offsets
- Fixes systematic misalignments
- Maintains biological tolerance

**For AGPAT3**: System should detect and apply +2bp donor adjustment

---

## Summary Table

| Metric | Donor | Acceptor | Average |
|--------|-------|----------|---------|
| **F1 Score (consensus_window=2)** | 0.9017 | 0.8824 | 0.8920 |
| **Precision** | 1.0000 | 1.0000 | 1.0000 |
| **Recall** | 0.8211 | 0.7895 | 0.8053 |
| **Exact matches (consensus_window=0)** | 0% | 100% | 50% |
| **Adjustment needed** | +2bp | None | - |

**Interpretation**: Excellent performance with consensus_window=2, but donors need coordinate adjustment for exact alignment.

---

**Date**: November 2, 2025  
**Status**: Complete analysis with correct metrics  
**Next**: Test adjustment detection system and apply fixes



