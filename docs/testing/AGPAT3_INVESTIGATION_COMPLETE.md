# AGPAT3 Investigation - Complete Analysis

**Date**: November 2, 2025  
**Status**: ✅ ROOT CAUSE IDENTIFIED  
**Gene**: ENSG00000160216 (AGPAT3) on chr21

---

## Executive Summary

**The mystery is solved!** AGPAT3 doesn't have 0 TPs - it has a **data structure mismatch**:

1. **`pred_type` column**: Correctly identifies 78 TPs ✅
2. **`error_type` column**: Incorrectly marks them as TNs ❌
3. **Root cause**: `error_df` only contains FP/FN, not TP
4. **When joined**: TPs get NULL → filled as 'TN'

**Additionally**: With `consensus_window=0` test, we confirmed:
- **Donor sites need +2bp adjustment** (0% exact matches)
- **Acceptor sites are perfectly aligned** (100% exact matches)

---

## Investigation Results

### Finding 1: Data Structure Mismatch

**The Bug**:
```python
# positions_df has pred_type='TP' for 78 donors
# error_df only has FP and FN rows (no TP rows!)
# When we join them:
positions_with_errors = positions_df.join(
    error_df.select(['position', 'error_type']),
    on='position',
    how='left'  # Left join!
).with_columns(
    pl.col('error_type').fill_null('TN')  # TPs become TN!
)
```

**Result**:
- TP positions don't exist in `error_df`
- Join returns NULL for `error_type`
- `fill_null('TN')` incorrectly marks them as TN

**Evidence**:
```
Sample donor position:
  position: 159
  pred_type: 'TP'        ✅ Correct!
  error_type: 'TN'       ❌ Wrong!
  donor_score: 0.994
  offset: -2bp
```

### Finding 2: Coordinate Offsets (consensus_window=0 test)

**DONOR SITES**:
```
High-scoring positions: 78
Exact matches (offset=0): 0
Within ±1bp: 0
Within ±2bp: 78

Offset distribution:
  -2bp: 78 positions (100%)

Metrics:
  consensus_window=0: 0.0% recall   ❌
  consensus_window=1: 0.0% recall   ❌
  consensus_window=2: 100.0% recall ✅
```

**Conclusion**: **Donors need +2bp adjustment**

**ACCEPTOR SITES**:
```
High-scoring positions: 75
Exact matches (offset=0): 75
Within ±1bp: 75
Within ±2bp: 75

Offset distribution:
  0bp: 75 positions (100%)

Metrics:
  consensus_window=0: 100.0% recall ✅
  consensus_window=1: 100.0% recall ✅
  consensus_window=2: 100.0% recall ✅
```

**Conclusion**: **Acceptors are perfectly aligned**

---

## Root Causes

### 1. Why error_type is Wrong

**Design of `error_df`**:
- Purpose: Track positions that need sequence extraction for analysis
- Contains: Only FP and FN (errors that need investigation)
- Does NOT contain: TP (correct predictions don't need error analysis)

**Code location**: `enhanced_evaluation.py`, lines 700-710
```python
# Only FP and FN are added to error_list
for pos_type in ['FP', 'FN']:  # Note: TP is not included!
    for pos in pos_dict[pos_type]:
        # ... add to error_list
```

**The problem**: When we join `positions_df` with `error_df`:
- TPs don't have matching rows in `error_df`
- Join returns NULL
- `fill_null('TN')` incorrectly labels them

### 2. Why Donors Need +2bp Adjustment

**Systematic offset**:
- ALL 78 high-scoring donor predictions: offset = -2bp
- This is not random variation
- This is a systematic coordinate mismatch

**Possible causes**:
1. **GTF annotation convention**: Donor site position definition
2. **SpliceAI output convention**: How SpliceAI reports donor positions
3. **Gene-specific**: Might be specific to AGPAT3 or chr21

**Why acceptors don't need adjustment**:
- 100% exact matches
- No systematic offset
- Perfect alignment with annotations

---

## The Correct Interpretation

### What F1 = 0 Actually Means

**It doesn't mean the model failed!**

It means:
1. We're using `error_type` column (which is wrong for TPs)
2. TPs are incorrectly labeled as TNs
3. Metrics calculated from `error_type` show 0 TPs

**The truth** (from `pred_type` column):
- 78 donor TPs (with -2bp offset, within consensus_window=2)
- 75 acceptor TPs (perfect alignment)
- Model is actually working excellently!

### What consensus_window=0 Test Reveals

**Your point was exactly right**:
> "consensus_window is just a solution to ease out such 'predicted deltas' or discrepancies but ultimately, we need to know if adjustments are needed."

**With consensus_window=0**:
- Reveals true coordinate alignment
- Donors: 0% → adjustments needed
- Acceptors: 100% → no adjustments needed

**With consensus_window=2**:
- Masks the -2bp offset
- Both show 100% recall
- But underlying mismatch remains

---

## Solutions

### Solution 1: Fix error_type Column (Immediate)

**Option A**: Include TPs in `error_df`
```python
# In enhanced_evaluation.py, line ~700
for pos_type in ['TP', 'FP', 'FN']:  # Add TP!
    for pos in pos_dict[pos_type]:
        # ... add to error_list
```

**Option B**: Use `pred_type` instead of `error_type`
```python
# In analysis code
# Don't join with error_df
# Use pred_type column directly from positions_df
y_true = (positions_df['pred_type'].is_in(['TP', 'FN'])).to_numpy()
```

**Recommendation**: **Option B** - `pred_type` is the correct classification

### Solution 2: Apply Coordinate Adjustments (Proper Fix)

**For AGPAT3 (and possibly other genes)**:

```python
# Apply +2bp adjustment to donor predictions
adjusted_donor_positions = donor_positions + 2

# Then evaluate with consensus_window=0 for exact matching
# Or use consensus_window=2 for biological variation tolerance
```

**Why this is correct**:
1. Fixes the underlying coordinate mismatch
2. Achieves exact alignment (100% with consensus_window=0)
3. Still allows tolerance for biological variation (consensus_window=2)

---

## Recommendations

### Immediate Actions

1. **Fix metrics calculation**:
   - Use `pred_type` column, not `error_type`
   - This will show true performance (F1 ~1.0 with consensus_window=2)

2. **Document the bug**:
   - `error_type` column is unreliable for TPs
   - Always use `pred_type` for classification

3. **Re-calculate AGPAT3 metrics**:
   - Using `pred_type` column
   - Should show excellent performance

### Long-Term Solutions

1. **Implement coordinate adjustments**:
   - Detect offsets automatically (already have this system!)
   - Apply +2bp to donor predictions for AGPAT3
   - Test if this is gene-specific or systematic

2. **Test more genes**:
   - Check if -2bp donor offset is common
   - Identify patterns (chromosome, biotype, etc.)
   - Build adjustment database

3. **Update evaluation protocol**:
   - Always test with consensus_window=0 first
   - Identify needed adjustments
   - Apply adjustments
   - Then use consensus_window=2 for final evaluation

---

## Key Insights

### 1. Two Different Classification Systems

**`pred_type`** (in `positions_df`):
- Set during prediction evaluation
- Correctly identifies TP/FP/FN/TN
- **This is the correct classification**

**`error_type`** (in `error_df`):
- Only contains FP/FN (for error analysis)
- Does NOT contain TP (not an error!)
- **Should not be used for metrics**

### 2. consensus_window is Not a Fix

**What it does**:
- Provides tolerance for biological variation
- Allows ±2bp flexibility in matching

**What it doesn't do**:
- Fix underlying coordinate mismatches
- Correct systematic offsets

**The right approach**:
1. Apply adjustments to fix systematic offsets
2. Then use consensus_window for biological variation

### 3. Gene-Specific vs Systematic Offsets

**AGPAT3 shows**:
- Donor: -2bp offset (systematic)
- Acceptor: 0bp offset (perfect)

**Questions to answer**:
- Is this specific to AGPAT3?
- Is this specific to chr21?
- Is this common across all genes?

**Previous test (50 genes)**:
- Average F1 = 0.9312
- Suggests most genes don't have large offsets
- But individual genes might vary

---

## Testing Plan

### Phase 1: Fix Metrics (Immediate)

```python
# Use pred_type instead of error_type
positions_df = pl.read_csv("full_splice_positions_enhanced.tsv", separator='\t')

# Calculate metrics using pred_type
for splice_type in ['donor', 'acceptor']:
    type_df = positions_df.filter(pl.col('splice_type') == splice_type)
    
    # Use pred_type, not error_type!
    y_true = type_df['pred_type'].is_in(['TP', 'FN']).to_numpy()
    y_scores = type_df[f'{splice_type}_score'].to_numpy()
    
    # Calculate F1, PR-AUC
    # ...
```

**Expected result**: F1 ~1.0 for AGPAT3 (with consensus_window=2)

### Phase 2: Test Coordinate Adjustments

```python
# Apply +2bp adjustment to donor predictions
donor_df = donor_df.with_columns(
    (pl.col('predicted_position') + 2).alias('adjusted_position')
)

# Re-calculate offset with adjusted positions
donor_df = donor_df.with_columns(
    (pl.col('adjusted_position') - pl.col('true_position')).alias('offset')
)

# Check exact matches
exact_matches = donor_df.filter(pl.col('offset') == 0).height
```

**Expected result**: 100% exact matches after adjustment

### Phase 3: Test More Genes

```python
# Run the same analysis on:
# - All 20 sampled genes (15 protein-coding + 5 lncRNA)
# - Previous 50 protein-coding genes
# - Random sample of 100 genes

# For each gene, calculate:
# - Donor offset distribution
# - Acceptor offset distribution
# - Percentage requiring adjustments
```

**Goal**: Characterize offset patterns across genes

---

## Conclusion

### The Good News

1. **AGPAT3 predictions are excellent!**
   - High scores (0.95-0.998)
   - 78 donor TPs, 75 acceptor TPs
   - Just need coordinate adjustment

2. **The system is working correctly**
   - `pred_type` classification is accurate
   - Predictions are within consensus_window
   - No bugs in prediction logic

3. **We now understand the issue**
   - `error_type` column is misleading
   - Donor sites need +2bp adjustment
   - Acceptor sites are perfectly aligned

### The Path Forward

1. **Immediate**: Use `pred_type` for metrics (not `error_type`)
2. **Short-term**: Apply +2bp adjustment to donors
3. **Long-term**: Build comprehensive adjustment system

### The Answer to Your Question

> "Find out why AGPAT3 has 0 TPs when the -2bp offset should be within tolerance"

**Answer**: 
- AGPAT3 **does** have TPs (78 donors, 75 acceptors)
- They're correctly identified in `pred_type` column
- They're incorrectly labeled in `error_type` column (data structure bug)
- The -2bp offset **is** within tolerance (consensus_window=2)

> "Test if we need score view adjustments with consensus_window=0"

**Answer**:
- **Donors**: YES, need +2bp adjustment (0% exact matches)
- **Acceptors**: NO, perfectly aligned (100% exact matches)
- consensus_window=2 masks this, but adjustments are still needed for exact coordinates

---

**Date**: November 2, 2025  
**Status**: Investigation complete, solutions identified  
**Next Step**: Implement fixes and re-test


