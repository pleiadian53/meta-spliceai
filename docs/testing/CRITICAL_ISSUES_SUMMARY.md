# Critical Issues Summary - Meta-Model Inference

**Date**: 2025-10-30  
**Status**: 🔴 **CRITICAL ISSUES REMAIN**  

---

## Executive Summary

The comprehensive test shows "ALL TESTS PASSED!" but this is **misleading**. The tests ran without crashing, but there are **three critical failures** that make the meta-model inference workflow unusable:

1. ❌ **Incomplete Coverage**: 83-88% instead of 100%
2. ❌ **Duplicate Predictions**: 1.14-1.20× duplication ratio
3. ❌ **Meta-only Mode Failure**: F1 scores near zero (0.004-0.018)

---

## Issue 1: Incomplete Coverage (MOST CRITICAL)

### Symptoms
```
Expected: 11,573 positions → Got: 10,034 positions (86.7% coverage)
Expected: 29,984 positions → Got: 25,411 positions (84.7% coverage)
Expected: 3,107 positions → Got: 2,735 positions (88.0% coverage)
Expected: 9,754 positions → Got: 8,117 positions (83.2% coverage)
```

### Impact
- **12-17% of nucleotide positions are missing predictions**
- Cannot generate complete coverage for meta-model inference
- Violates the core requirement: "predict scores for ALL positions"

### Root Cause
The bug is in `predict_splice_sites_for_genes()` in `run_spliceai_workflow.py`. Despite our investigation showing that:
- SpliceAI model auto-crops output to 5,000 bp (correct)
- Overlapping blocks are intentional (correct)
- Averaging logic exists (correct)

**Something is still causing positions to be skipped.**

### Hypothesis
The issue might be in:
1. **Block boundary handling**: Last block might be truncated
2. **Trimming logic** (lines 372-378): Might be too aggressive
3. **Position calculation** for negative strand genes
4. **Padding logic** in `prepare_input_sequence()`: Might not align with block processing

---

## Issue 2: Duplicate Predictions

### Symptoms
```
ENSG00000187987: 11,573 expected → 11,573 + 1,539 = 13,112 rows (1.15× duplication)
ENSG00000171812: 29,984 expected → 29,984 + 4,573 = 34,557 rows (1.18× duplication)
ENSG00000233730: 3,107 expected → 3,107 + 372 = 3,479 rows (1.14× duplication)
ENSG00000278923: 9,754 expected → 9,754 + 1,637 = 11,391 rows (1.20× duplication)
```

### Impact
- Corrupts evaluation metrics (F1 scores)
- Requires workaround deduplication (masking the root cause)
- Indicates fundamental bug in position tracking

### Root Cause
The tuple key `(gene_id, position)` in `merged_results` doesn't always match `data['absolute_position']`:

```python
# Line 345:
pos_key = (gene_id, absolute_position) if has_absolute_positions else (gene_id, block_start + i + 1)

# But then line 354 stores:
merged_results[pos_key]['absolute_position'] = absolute_position if has_absolute_positions else (block_start + i + 1)
```

If `has_absolute_positions` is True, `pos_key` uses `absolute_position`. But if there's any mismatch in how `absolute_position` is calculated across blocks, we get duplicates.

### Attempted Fix
Added deduplication check in `efficient_results` path (line 411):
```python
if abs_pos not in efficient_results[gene_id]['positions']:
    # append...
```

But this is a **workaround**, not a fix. The real issue is that `trimmed_results` already contains duplicate keys with different `absolute_position` values.

---

## Issue 3: Meta-only Mode Complete Failure

### Symptoms
```
ENSG00000187987:
  base_only: F1 = 0.857 ✅
  hybrid:    F1 = 0.800 ✅
  meta_only: F1 = 0.017 ❌ (should be > base_only)

ENSG00000171812:
  base_only: F1 = 0.727 ✅
  hybrid:    F1 = 0.727 ✅
  meta_only: F1 = 0.004 ❌

ENSG00000278923:
  base_only: F1 = 0.500 ✅
  hybrid:    F1 = 0.500 ✅
  meta_only: F1 = 0.004 ❌
```

### Impact
- **Meta-model is not improving predictions at all**
- Meta-only mode performs **worse than random guessing**
- The entire meta-learning approach appears broken

### Root Cause (Likely)
The meta-model is being applied, but:
1. **Feature mismatch**: Inference features don't match training features
2. **Score corruption**: Duplicates are being averaged, destroying signal
3. **Model not loading**: Despite logs saying "100% positions recalibrated"
4. **Threshold issue**: Meta-model outputs might need different thresholds

### Evidence
From previous tests, meta-only scores were **identical** to base-only scores, suggesting the meta-model isn't actually changing anything.

---

## Issue 4: Misleading Success Message

### Problem
The test script prints "✅ ALL TESTS PASSED!" even when:
- Coverage is 83% (not 100%)
- Meta-only F1 is 0.004 (essentially zero)
- Duplicates are present

### Fix Needed
Update test script to:
```python
# Fail if coverage < 95%
assert coverage >= 0.95, f"Coverage too low: {coverage:.1%}"

# Fail if meta-only doesn't improve over base
assert meta_f1 >= base_f1 * 0.9, f"Meta-model not improving: {meta_f1} vs {base_f1}"

# Fail if duplicates detected
assert n_duplicates == 0, f"Found {n_duplicates} duplicate predictions"
```

---

## Test Results Detail

### ENSG00000187987 (ZSCAN23) - Protein-coding, 11,573 bp
- Coverage: 10,034/11,573 (86.7%) ❌
- Duplicates: 1,539 (1.15×) ❌
- Base F1: 0.857 ✅
- Meta F1: 0.017 ❌

### ENSG00000171812 (COL8A2) - Protein-coding, 29,984 bp
- Coverage: 25,411/29,984 (84.7%) ❌
- Duplicates: 4,573 (1.18×) ❌
- Base F1: 0.727 ✅
- Meta F1: 0.004 ❌

### ENSG00000233730 (LINC01765) - lncRNA, 3,107 bp
- Coverage: 2,735/3,107 (88.0%) ❌
- Duplicates: 372 (1.14×) ❌
- Base F1: 0.000 (expected for lncRNA with no splice sites)
- Meta F1: 0.018

### ENSG00000278923 - lncRNA, 9,754 bp
- Coverage: 8,117/9,754 (83.2%) ❌
- Duplicates: 1,637 (1.20×) ❌
- Base F1: 0.500 ✅
- Meta F1: 0.004 ❌

---

## Priority Actions

### Priority 1: Fix Incomplete Coverage (BLOCKING)
**This is the most critical issue.** Without 100% coverage, we cannot:
- Generate complete feature matrices for meta-model
- Ensure consistent predictions across all positions
- Trust any evaluation metrics

**Action**: Deep dive into `predict_splice_sites_for_genes()` to find where positions are being skipped.

### Priority 2: Fix Duplicate Predictions
**Action**: Find why `trimmed_results` contains duplicate keys with different `absolute_position` values.

### Priority 3: Debug Meta-only Mode Failure
**Action**: Once coverage and duplicates are fixed, investigate why meta-model isn't improving predictions.

### Priority 4: Fix Test Success Criteria
**Action**: Update test script to fail on coverage < 95%, meta F1 < base F1, or duplicates > 0.

---

## Investigation Plan

### Step 1: Add Detailed Logging to `predict_splice_sites_for_genes()`

Add logging to track:
- Number of blocks generated
- Positions covered by each block
- Positions added to `merged_results`
- Positions removed by trimming
- Final position count

### Step 2: Test with Simple Gene

Use a small gene (e.g., 7,107 bp) and manually verify:
- Expected blocks: ceil(7107/5000) = 2 blocks
- Expected positions: 7,107
- Actual positions: ?

### Step 3: Check Strand-Specific Logic

Test positive and negative strand genes separately to see if the coverage issue is strand-specific.

### Step 4: Verify Padding Math

```python
# For 7,107 bp gene:
padded_length = ceil(7107/5000) * 5000 = 10,000
with_flanking = 10,000 + 2*5000 = 20,000
num_blocks = (20,000 - 15,000) // 5000 + 1 = 2

# Block 0: positions 0-4,999 (5,000 positions)
# Block 1: positions 5,000-9,999 (5,000 positions)
# Total: 10,000 positions (but gene is only 7,107!)
```

**Aha!** The padding extends the sequence to 10,000 bp, but the gene is only 7,107 bp. The trimming logic should remove positions 7,107-9,999, leaving exactly 7,107 positions. But we're getting fewer!

---

## Files Modified (So Far)

1. `meta_spliceai/splice_engine/run_spliceai_workflow.py`
   - Added comments explaining SpliceAI auto-cropping (lines 322-324)
   - Fixed DataFrame conversion to use `abs_pos` consistently (lines 428-436)
   - Added deduplication check in `efficient_results` (lines 409-415)

2. `meta_spliceai/splice_engine/meta_models/workflows/inference/enhanced_selective_inference.py`
   - Added deduplication workaround (lines 843-852)

3. `meta_spliceai/splice_engine/utils_kmer.py`
   - Updated regex to include 'N' in k-mers (line 15)

---

## Next Steps

**User Decision Required**:
1. Should we prioritize fixing coverage (100% required) or accept 83-88%?
2. Should we continue debugging or look for a reference implementation?
3. Do you have a working example of SpliceAI generating complete coverage?

**Recommended**: Focus on Issue 1 (coverage) first, as it's blocking everything else.



