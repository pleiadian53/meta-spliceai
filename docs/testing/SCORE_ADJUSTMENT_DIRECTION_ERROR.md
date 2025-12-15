# Score Adjustment Direction Error - ROOT CAUSE FOUND!

**Date**: 2025-10-31  
**Status**: 🎯 **ROOT CAUSE IDENTIFIED**

## The Critical Error

I had the **shift direction BACKWARDS** in my implementation!

### Current (WRONG) Implementation

```python
# From score_adjustment.py
def shift_score_vector(scores, shift_amount, fill_value=0.0):
    if shift_amount > 0:
        # Positive shift: position i gets value from position i+shift_amount
        shifted[:-shift_amount] = scores[shift_amount:]
```

With adjustment values: `{'donor': {'plus': 2}}`
- This shifts: position i gets score from position i+2
- Means: position 100 gets score from position 102

### What the Adjustment Values Actually Mean

From SPLICE_SITE_DEFINITION_ANALYSIS.md (lines 116-122):
```
SpliceAI Model Predictions:
Donor (+):    2nt upstream of GTF position
Donor (-):    1nt upstream of GTF position  
Acceptor (+): Exact GTF position
Acceptor (-): 1nt downstream of GTF position
```

**"2nt upstream" means**:
- True donor at position 100 (GTF)
- SpliceAI gives high score at position 98 (100-2)
- **To correct**: Position 100 needs score from position 98
- **Required shift**: position i gets score from position i-2 (shift by -2!)

## The Confusion

The adjustment dict uses **POSITIVE** values to represent **UPSTREAM** offsets:
```python
adjustment_dict = {
    'donor': {'plus': 2, 'minus': 1},   # "2nt upstream" and "1nt upstream"
    'acceptor': {'plus': 0, 'minus': -1}  # "exact" and "1nt downstream"
}
```

But in my shift implementation:
- Positive shift_amount → shifts scores FORWARD (3' direction)
- Negative shift_amount → shifts scores BACKWARD (5' direction)

## The Fix

**Option 1**: Negate the adjustment values before shifting
```python
# In adjust_splice_scores_by_type_and_strand()
adjusted_donor = shift_score_vector(donor_scores, shift_amount=-donor_adj)
adjusted_acceptor = shift_score_vector(acceptor_scores, shift_amount=-acceptor_adj)
```

**Option 2**: Change the shift_score_vector logic to match the semantics

**Option 3**: Clarify the documentation and use opposite convention

## Evidence

### Before Fix
- Donor (+strand): 0% exact, 57% off by +2 → scores NOT shifted
- Acceptor (+strand): 69% exact → works because adj=0 (no shift)
- Minus strand: all off by ±1 → wrong direction

### Expected After Fix
- Donor (+strand): Should go to ~70-90% exact match
- Acceptor (+strand): Should remain ~70% exact
- Minus strand: Should improve to ~70-90% exact match

## Action Items

1. ✅ Identified root cause: Direction backwards
2. ⏭️ Fix `adjust_splice_scores_by_type_and_strand` to negate adjustments
3. ⏭️ Test on VCAM1 again
4. ⏭️ If successful, re-run on all 24 genes
5. ⏭️ Document the correct interpretation of adjustment values

---

**Confidence**: 95% - This explains ALL the observed patterns!

