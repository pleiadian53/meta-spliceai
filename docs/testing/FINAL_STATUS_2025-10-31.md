# Final Status: Score Adjustment Investigation - 2025-10-31

## Summary

After ~6 hours of investigation, I've made significant progress but **the alignment issue persists**.

## What Works ✅

1. **Standalone score adjustment module** (`score_adjustment.py`):
   - Created and tested
   - Correctly shifts score vectors
   - Properly recalculates neither scores
   - All score sums = 1.0 ✓

2. **Integration into inference workflow**:
   - Module is being called
   - Adjustments are being applied (2,543 donor scores changed)
   - Neither scores properly recalculated (sum=1.0 for all positions)

3. **Coverage**:
   - 100% coverage maintained ✓
   - No position collisions ✓

## What Doesn't Work ❌

**F1 scores remain unchanged:**
- Donor F1: 0.000 (unchanged)
- Acceptor F1: 0.818 (unchanged)  
- Overall F1: 0.400 (unchanged)
- Donors still "off by ±2 nt" from annotations

## Critical Findings

### Finding 1: Neither Scores Were Wrong
**Problem**: After shifting donor/acceptor scores, neither scores weren't recalculated.  
**Evidence**: Score sums were ~2.0 for high-scoring positions  
**Fix**: Implemented `adjusted_neither = 1.0 - adjusted_donor - adjusted_acceptor`  
**Result**: ✅ Score sums now = 1.0

### Finding 2: Adjustment Direction Was Backwards
**Problem**: Adjustment values represent "model offset from true position"  
**OpenSpliceAI doc**: "+2 means model predicts 2nt upstream"  
**Fix**: Negated adjustment before shifting: `shift_amount=-donor_adj`  
**Result**: ⚠️ Still testing

### Finding 3: Adjustments ARE Being Applied
**Evidence**:
- Log shows: "Scores changed: donor=2543/19304"
- High donor scores exist in saved file (0.998, 0.994, 0.982, etc.)
- Neither scores properly recalculated

But **F1 scores don't improve!**

## Possible Remaining Issues

### Hypothesis 1: Wrong Shift Direction (AGAIN)
Maybe I have the semantics backwards AGAIN:
- "Model predicts 2nt upstream" could mean different things
- Need to verify with actual example positions

### Hypothesis 2: GTF Annotations Are Wrong
- Maybe the GTF annotations themselves have an offset?
- Need to manually check a few splice sites

### Hypothesis 3: Evaluation Logic Issue
- Maybe the evaluation is comparing wrong columns?
- Maybe it's using `donor_prob` instead of `donor_score`?

### Hypothesis 4: Adjustment Values Are Wrong
- Default: `{donor: {plus: 2}, acceptor: {plus: 0}}`
- Maybe these values are incorrect for our GTF version?
- Need to empirically derive the correct values

## Test Evidence

### VCAM1 (ENSG00000162692, + strand, 19kb)

**Predictions (after adjustment)**:
- 9 positions with donor_score > 0.5
- Example: position 100734770 has donor=0.998
- All score sums = 1.0 ✓

**Annotations from GTF**:
- 14 donor sites, 13 acceptor sites

**Evaluation Results**:
- Donor exact match: 0/14 (0.0%)
- Donor off by ±2: 8/14 (57.1%)
- Acceptor exact match: 9/13 (69.2%)

**Interpretation**:
- Acceptors working (69% exact match) ✓
- Donors NOT working (0% exact, 57% off by ±2) ❌

## Next Steps

### Option 1: Empirically Derive Adjustments
Look at a few actual splice sites and manually check where SpliceAI puts high scores

### Option 2: Check OpenSpliceAI Implementation
Look at how OpenSpliceAI handles coordinate adjustments in their actual code

### Option 3: Test Without Adjustments
Run predictions WITHOUT any adjustments to see baseline

### Option 4: Manual Position Tracing
Pick one specific donor site and trace through:
1. GTF annotation position
2. Where SpliceAI gives high score
3. Where our adjustment puts the high score
4. What the evaluation finds

## Recommendation

I recommend **Option 4** (manual position tracing) to definitively understand what's happening at a single position level.

---

**Time invested**: ~6 hours  
**Lines of code written**: ~800  
**Tests run**: 15+  
**Coffee consumed**: ☕☕☕

**Status**: The bug is elusive but we're close!

