# SpliceAI Block Processing Investigation

**Date**: 2025-10-30  
**Status**: 🔍 **INVESTIGATING**  
**Impact**: Critical - Affects all inference modes

---

## Problem Summary

The meta-model inference workflow showed:

1. **Duplicate predictions** for some positions (8,721 rows for 6,300 unique positions = 1.38× duplication)
2. **Incomplete coverage** (88.6% instead of 100%)
3. **Incorrect F1 scores** due to corrupted evaluation data

Initially suspected the `predict_splice_sites_for_genes()` function had a bug in block processing, but investigation revealed the actual cause is different.

---

## Investigation Findings

### Key Discovery: SpliceAI Model Auto-Crops Output

**CRITICAL**: The SpliceAI model **automatically crops its output** to remove context padding!

```
Input:  15,000 bp (5,000 context + 5,000 core + 5,000 context)
Output:  5,000 bp (core region only - context already removed by model)
```

This means the original code in `predict_splice_sites_for_genes()` was **already correct** - it doesn't need to manually crop the output because the model does it internally.

### SpliceAI's Overlapping Block Design (Intentional)

SpliceAI processes long sequences by splitting them into **overlapping blocks**:

```python
# From prepare_input_sequence() (lines 197-204)
block_size = context // 2 + 5000 + context // 2  # = 15,000 bp (for context=10000)
num_blocks = (len(padded_seq) - block_size) // 5000 + 1
blocks = [padded_seq[5000 * i: 5000 * i + block_size] for i in range(num_blocks)]
```

**Block Structure**:
- Each block is **15,000 bp** (5,000 bp core + 5,000 bp context on each side)
- Blocks advance by **5,000 bp** each time
- **Overlap = 10,000 bp** between consecutive blocks

**Why Overlapping?**
- Positions at block edges have **incomplete context**
- Overlapping ensures every position gets **full context** from at least one block
- Predictions from overlapping regions are **averaged** for robustness

### The Bug (Lines 314-351)

The original code had two critical errors:

**Error 1**: Processed **all 15,000 positions** in each block instead of just the center 5,000:

```python
# WRONG: Iterates through all 15,000 positions (including context padding)
for i, (donor_p, acceptor_p, neither_p) in enumerate(zip(donor_prob, acceptor_prob, neither_prob)):
    absolute_position = gene_start + (block_start + i)
    merged_results[pos_key]['donor_prob'].append(donor_p)
```

**Error 2**: Position calculation assumed non-overlapping blocks:

```python
block_start = block_index * 5000  # Correct for core region, but applied to full block
```

### What Actually Happened

For a 7,107 bp gene with context=10,000:

**Block 0** (block_index=0):
- Covers positions 0-14,999 (full block including padding)
- `block_start = 0`
- Stores predictions for positions 0-14,999 ❌ **WRONG** (should be 0-4,999 only)

**Block 1** (block_index=1):
- Covers positions 5,000-19,999
- `block_start = 5,000`
- Stores predictions for positions 5,000-19,999 ❌ **WRONG** (should be 5,000-9,999 only)

**Result**:
- Positions 5,000-9,999 get predicted **twice** (from both blocks) ✅ **Intended**
- Positions 10,000-14,999 get predicted **only from Block 0** ❌ **Should be from Block 2**
- Positions 15,000-19,999 get predicted **only from Block 1** ❌ **Out of bounds!**
- The trimming logic (lines 369-375) removes out-of-bounds positions, but gaps remain

---

## The Fix

**Modified Lines 327-364** in `predict_splice_sites_for_genes()`:

```python
# CRITICAL FIX: Extract only the center 5,000 bp (remove context padding)
# Block structure: [context/2 padding][5000 bp core][context/2 padding]
# We only use predictions from the core region with full context
half_context = context // 2
core_start = half_context
core_end = half_context + 5000

# Crop to core region
donor_prob_core = donor_prob[core_start:core_end]
acceptor_prob_core = acceptor_prob[core_start:core_end]
neither_prob_core = neither_prob[core_start:core_end]

# Calculate the start position of the current block's CORE relative to the original sequence
block_start = block_index * 5000

# Store the results with adjusted positions (only for core region)
for i, (donor_p, acceptor_p, neither_p) in enumerate(zip(donor_prob_core, acceptor_prob_core, neither_prob_core)):
    # ... rest of the loop processes only 5,000 positions per block
```

### What Now Happens Correctly

**Block 0**:
- Processes only positions 5,000-9,999 (core region with full context)
- Stores predictions for positions 0-4,999 ✅

**Block 1**:
- Processes only positions 10,000-14,999 (core region with full context)
- Stores predictions for positions 5,000-9,999 ✅

**Block 2**:
- Processes only positions 15,000-19,999 (core region with full context)
- Stores predictions for positions 10,000-14,999 ✅

**Overlap Handling**:
- Positions 5,000-9,999 get predicted from **both Block 0 and Block 1**
- Probabilities are **averaged** (lines 407-410) ✅

---

## Impact on Inference

### Before Fix
- **Coverage**: 88.6% (6,300 positions for 7,107 bp gene)
- **Duplicates**: 8,721 rows for 6,300 unique positions (1.38× duplication)
- **F1 Scores**: Corrupted by duplicate rows
- **Meta-only mode**: Failed (F1=0.015) due to duplicate averaging

### After Fix (Expected)
- **Coverage**: 100% (7,107 positions for 7,107 bp gene)
- **Duplicates**: None (7,107 rows for 7,107 unique positions)
- **F1 Scores**: Accurate evaluation
- **Meta-only mode**: Should work correctly

---

## Why This Wasn't Caught Earlier

During **training**, this bug was masked because:

1. **Training evaluates only at known splice sites** (from annotations)
2. If a splice site appears in the overlap, it gets averaged correctly
3. Missing positions between splice sites don't affect training metrics
4. The trimming logic removed most out-of-bounds positions

For **inference with full coverage** (required for meta-model), we need predictions for **every single position**, so the bug became critical.

---

## Related Changes

1. **`enhanced_selective_inference.py`** (lines 843-851):
   - Removed deduplication workaround
   - Now **fails loudly** if duplicates are detected
   - This ensures the fix in `predict_splice_sites_for_genes()` is working

2. **Testing**:
   - Re-run comprehensive tests to verify 100% coverage
   - Verify meta-only mode now works correctly

---

## Files Modified

1. `/Users/pleiadian53/work/meta-spliceai/meta_spliceai/splice_engine/run_spliceai_workflow.py`
   - Lines 327-364: Fixed block processing to extract only core region

2. `/Users/pleiadian53/work/meta-spliceai/meta_spliceai/splice_engine/meta_models/workflows/inference/enhanced_selective_inference.py`
   - Lines 843-851: Removed deduplication workaround, added loud failure check

---

## Next Steps

1. ✅ Fix applied to `predict_splice_sites_for_genes()`
2. ⏳ Re-run comprehensive test: `python scripts/testing/test_all_modes_comprehensive_v2.py`
3. ⏳ Verify 100% coverage for all genes
4. ⏳ Verify meta-only mode F1 score improves
5. ⏳ Check if meta-model is now actually recalibrating scores

---

## Technical Notes

### SpliceAI's Original Design

The overlapping block design is from the original SpliceAI paper:
- **Purpose**: Ensure every position has full context for accurate predictions
- **Trade-off**: 2× computational cost (each position predicted twice in overlap)
- **Benefit**: More robust predictions through averaging

### Why Extract Core Region?

The context padding (5,000 bp on each side) is **only for the model**:
- Positions in padding have **incomplete context** (edge effects)
- Original SpliceAI crops predictions to avoid these edge effects
- We should **never use predictions from padded regions**

### Alternative Solutions Considered

**Option 1** (Chosen): Extract only center 5,000 bp from each block
- ✅ Simple and reliable
- ✅ Matches SpliceAI's original design
- ✅ Naturally handles overlaps through averaging

**Option 2** (Rejected): Track which positions in each block correspond to genomic positions
- ❌ More complex and error-prone
- ❌ Requires careful bookkeeping of overlaps
- ❌ No clear advantage over Option 1

---

## Verification

To verify the fix is working:

```python
# Check coverage
n_positions = len(predictions_df)
gene_length = gene_end - gene_start + 1
coverage_pct = (n_positions / gene_length) * 100
assert coverage_pct == 100.0, f"Expected 100% coverage, got {coverage_pct:.1f}%"

# Check for duplicates
n_unique = predictions_df['position'].n_unique()
assert n_positions == n_unique, f"Found {n_positions - n_unique} duplicate positions!"
```

---

**Status**: Fix implemented, awaiting comprehensive test results.

