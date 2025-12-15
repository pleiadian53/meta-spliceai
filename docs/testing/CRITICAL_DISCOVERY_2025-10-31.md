# CRITICAL DISCOVERY: Adjustment Direction is Backwards!

## Date: 2025-10-31 13:05

## The Problem

After implementing multi-view adjustment with correlated probability vectors, we discovered that:

1. **Base model (unadjusted) predictions MATCH the true donor sites**:
   - True: 100719924 → Pred (base): 100719924 ✅ EXACT MATCH
   - True: 100720751 → Pred (base): 100720751 ✅ EXACT MATCH
   - True: 100723340 → Pred (base): 100723340 ✅ EXACT MATCH

2. **Donor view (adjusted) predictions are OFF by +2**:
   - True: 100719924 → Pred (donor_view): 100719926 ❌ OFF BY +2
   - True: 100720751 → Pred (donor_view): 100720753 ❌ OFF BY +2
   - True: 100723340 → Pred (donor_view): 100723342 ❌ OFF BY +2

## The Root Cause

Our adjustment logic is:
```python
donor_adj = +2  # "Model predicts 2nt upstream of true site"
shift_amount = -donor_adj = -2  # Shift by -2 to correct
```

But this is **BACKWARDS**! The base model is already predicting at the correct position, so:
- `donor_adj = +2` should mean "we need to shift predictions +2 to align with GTF"
- NOT "model predicts 2nt upstream"

## The Confusion

The `adjustment_dict` values have been interpreted inconsistently:
1. **Our interpretation**: `+2` = "model predicts 2nt upstream, so shift back by -2"
2. **Actual meaning**: `+2` = "shift predictions forward by +2 to align with GTF"

## Evidence

From VCAM1 (ENSG00000162692):
- Base model F1 scores would be PERFECT if we didn't adjust!
- After adjustment, donors are off by exactly +2
- Acceptors (adjustment=0) remain correct at 69.2% exact match

## Next Steps

1. **Option A**: Remove adjustment entirely (base model is already correct)
2. **Option B**: Fix the interpretation of adjustment values
3. **Option C**: Investigate why the adjustment values were set to +2/+1 in the first place

## Key Question

**Where did the `donor: {plus: 2, minus: 1}` values come from?**
- Were they empirically determined from training data?
- Are they based on SpliceAI's documented coordinate system?
- Do they apply to our specific GTF annotations?

We need to trace back to the original source of these adjustment values!

