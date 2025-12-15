# Score-Shifting Bug: ROOT CAUSE FOUND!

**Date**: 2025-10-31  
**Status**: ✅ **BUG IDENTIFIED**

## The Bug

**Column Name Mismatch**: The adjustment code uses `donor_prob`, `acceptor_prob`, `neither_prob`, but the actual DataFrame columns are `donor_score`, `acceptor_score`, `neither_score`!

### Evidence

```python
# Predictions file columns (verified):
['donor_score', 'acceptor_score', 'neither_score', ...]

# Adjustment code (lines 686-693):
pl.col('donor_prob').shift(-donor_plus_adj).fill_null(0).alias('donor_prob'),
pl.col('acceptor_prob').shift(-acceptor_plus_adj).fill_null(0).alias('acceptor_prob'),
pl.col('neither_prob').alias('neither_prob')
```

### What Happens

1. `pl.col('donor_prob')` tries to access a column that doesn't exist
2. Polars creates a new column filled with `null` values
3. `.shift()` operates on the null column
4. `.fill_null(0)` fills with zeros
5. `.alias('donor_prob')` creates a NEW column called `donor_prob` (all zeros!)
6. The ORIGINAL `donor_score` column remains **UNCHANGED**!

Result: **No adjustments are applied**, predictions remain misaligned!

## Test Results Explained

- **+ strand donors** (adj=+2): 0% exact, 57% off by ±2 → Unadjusted scores
- **+ strand acceptors** (adj=0): 46% exact → Works because adj=0 (no shift needed)
- **- strand** (adj=+1/-1): 0-0.7% exact, ~70% off by ±1 → Unadjusted scores

The acceptors on + strand have 46% exact match because:
- Adjustment is 0 (no shift)
- Even with the bug, no shift = no change
- Some acceptors happen to align naturally

## The Fix

Change column names from `*_prob` to `*_score`:

```python
# BEFORE (WRONG):
pl.col('donor_prob').shift(-donor_plus_adj).fill_null(0).alias('donor_prob'),
pl.col('acceptor_prob').shift(-acceptor_plus_adj).fill_null(0).alias('acceptor_prob'),
pl.col('neither_prob').alias('neither_prob')

# AFTER (CORRECT):
pl.col('donor_score').shift(-donor_plus_adj).fill_null(0).alias('donor_score'),
pl.col('acceptor_score').shift(-acceptor_plus_adj).fill_null(0).alias('acceptor_score'),
pl.col('neither_score').alias('neither_score')
```

## Expected Results After Fix

With correct column names:
- **+ strand donors**: Should go from 0% → ~70-90% exact match
- **+ strand acceptors**: Should remain ~45% (already working)
- **- strand donors**: Should go from 0.7% → ~70-90% exact match
- **- strand acceptors**: Should go from 0% → ~70-90% exact match

**Overall F1**: Should improve from 0.186 → ~0.7-0.8

## Action Items

1. ✅ Bug identified: Column name mismatch
2. ⏭️ Fix `_apply_coordinate_adjustments()` to use `*_score` instead of `*_prob`
3. ⏭️ Re-test on same 24 genes
4. ⏭️ Verify F1 scores improve to ≥0.7

---

**Confidence**: 99% - This is definitely the bug!

