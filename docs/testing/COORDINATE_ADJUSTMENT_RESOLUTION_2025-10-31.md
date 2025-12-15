# Coordinate Adjustment Resolution - Final Solution

## Date: 2025-10-31

## Problem Statement

The inference workflow was producing F1 scores of 0.000 for donors and 0.400 overall, despite SpliceAI being a well-documented high-performance model.

## Root Cause

The coordinate adjustment system was applying **incorrect adjustments** that made predictions WORSE, not better:
- Adjustment values: `donor: {plus: 2, minus: 1}, acceptor: {plus: 0, minus: -1}`
- These adjustments were **shifting predictions AWAY from true splice sites**
- The base SpliceAI model predictions were **ALREADY ALIGNED** with our GTF annotations

## Evidence

### VCAM1 (ENSG00000162692) Test Results:

| Configuration | Overall F1 | Donor F1 | Acceptor F1 |
|--------------|------------|----------|-------------|
| With adjustments (+2/+1) | 0.400 | 0.000 | 0.818 |
| **Zero adjustments** | **0.756** | **0.696** | **0.818** |

### Detailed Analysis:
- **Base model predictions**: Match true donor sites (e.g., 100719924, 100720751, 100723340)
- **After +2 adjustment**: Off by +2 from true sites (e.g., 100719926, 100720753, 100723342)
- **Conclusion**: Adjustment was moving predictions in the WRONG direction

## Solution

Set all coordinate adjustments to **ZERO**:
```python
self.adjustment_dict = {
    'donor': {'plus': 0, 'minus': 0},
    'acceptor': {'plus': 0, 'minus': 0}
}
```

## Key Insights from the Investigation

### 1. Correlated Probability Vectors (User's Critical Insight)
The user correctly identified that the three scores at each position (donor, acceptor, neither) are **correlated** and must sum to 1.0. When adjusting coordinates, we must shift the ENTIRE probability vector as a unit, not individual score types independently.

This insight led to the development of the **multi-view adjustment** approach:
- Donor view: All three scores shifted by donor adjustment
- Acceptor view: All three scores shifted by acceptor adjustment
- When evaluating donors, use donor view; when evaluating acceptors, use acceptor view

### 2. Multi-View Implementation
The `score_adjustment_v2.py` module correctly implements:
- `shift_probability_vectors()`: Shifts all three scores together
- `create_splice_type_views()`: Creates separate views for each splice type
- `adjust_predictions_dataframe_v2()`: Adds view columns to DataFrame

This implementation is **correct** and should be retained for future use with models that DO require adjustment (e.g., OpenSpliceAI).

### 3. The Real Problem
The adjustment VALUES were wrong, not the adjustment LOGIC. The multi-view approach works perfectly - it just revealed that the adjustments were making things worse!

## Implications

1. **For SpliceAI with our GTF annotations**: No adjustment needed
2. **For other base models** (e.g., OpenSpliceAI): The multi-view adjustment system is ready to use
3. **For future work**: Always validate adjustment values empirically before applying them

## Files Modified

1. `/Users/pleiadian53/work/meta-spliceai/meta_spliceai/splice_engine/meta_models/utils/score_adjustment_v2.py`
   - New module implementing correlated probability vector adjustment
   - Multi-view approach with separate columns for each splice type

2. `/Users/pleiadian53/work/meta-spliceai/meta_spliceai/splice_engine/meta_models/workflows/inference/enhanced_selective_inference.py`
   - Updated to use zero adjustments by default
   - Integrated multi-view adjustment system

3. `/Users/pleiadian53/work/meta-spliceai/scripts/testing/generate_and_test_20_genes.py`
   - Updated evaluation to use splice-type-specific view columns

## Next Steps

1. ✅ **DONE**: Set adjustments to zero for current dataset
2. **TODO**: Test on the full 20+ gene set to confirm consistent performance
3. **TODO**: Investigate where the original +2/+1 adjustment values came from
4. **TODO**: Document when/how to determine correct adjustment values for new datasets
5. **TODO**: Fix meta-model threshold (0.5 → 0.95) for proper Platt calibration

## Acknowledgment

This resolution was made possible by the user's critical insight about correlated probability vectors. The investigation of this insight led to the discovery that the real problem was not the adjustment logic, but the adjustment values themselves.

