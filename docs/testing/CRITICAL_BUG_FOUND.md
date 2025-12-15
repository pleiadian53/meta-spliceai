# CRITICAL BUG: Inference Modes Producing Different Position Counts

**Date:** 2025-10-29  
**Status:** 🔴 **BLOCKING** - Must fix before proceeding with testing

---

## Problem Summary

The three inference modes are producing **different numbers of output positions** for the same gene, which is **fundamentally wrong**. All modes should predict scores for **every position** in the gene sequence.

### Observed Behavior

Testing gene **ENSG00000134202** (GSTM3, 7,107 bp):

| Mode | Positions | Expected | Status |
|------|-----------|----------|--------|
| `base_only` | 845 | 7,107 | ❌ Missing 88% |
| `hybrid` | 840 | 7,107 | ❌ Missing 88% |
| `meta_only` | **4,200** | 7,107 | ❌ 59% too many! |

---

## Why This Is Critical

1. **Invalid Comparisons**: Cannot compare performance across modes if they're predicting different positions
2. **Incorrect Metrics**: F1 scores of 0.000 because coordinate systems don't align
3. **Unusable Outputs**: Predictions cannot be used for variant analysis or downstream tasks
4. **Fundamental Design Flaw**: Suggests the inference workflow is not correctly implementing the "predict all positions" requirement

---

## Root Cause Analysis

### Expected Behavior

For a gene of length `L`:
- **Input**: Gene sequence of length `L`
- **Output**: `L` rows, one per nucleotide position
- **Columns**: `[position, donor_score, acceptor_score, neither_score, donor_meta, acceptor_meta, neither_meta, ...]`

### Actual Behavior

The workflow is currently:
1. Running base model pass → generates predictions for **some** positions (not all)
2. Filtering to only positions with "interesting" scores or annotations
3. Outputting only the filtered subset

This is **wrong** for the inference workflow. The filtering logic is appropriate for **training data generation** but not for **inference**.

---

## Impact on Testing

### Why Previous Tests Failed

1. **`test_all_modes_comprehensive_v2.py`**: F1=0.000 for all genes
   - Tried to compare genomic coordinates from `splice_sites_enhanced.tsv` with gene-relative coordinates from predictions
   - Position counts didn't match, so no true positives were found

2. **`test_modes_with_known_genes.py`**: F1=0.000 despite "Correctly captured: 8/8 sites"
   - Base model pass correctly identified splice sites
   - But final output only contained 845/7107 positions
   - Evaluation couldn't find the splice sites in the sparse output

3. **`test_three_modes_comparison.py`**: Crashed with shape mismatch
   - Different modes produced different position counts
   - Cannot perform element-wise score comparisons

---

## Required Fixes

### 1. **Complete Position Coverage** (CRITICAL)

**File**: `enhanced_selective_inference.py`

**Issue**: The workflow is not generating predictions for all positions in the gene sequence.

**Fix**: Ensure that for a gene of length `L`, the final output has exactly `L` rows, with one row per nucleotide position (1 to L).

**Verification**:
```python
gene_length = gene_end - gene_start + 1
assert len(predictions_df) == gene_length, f"Expected {gene_length} positions, got {len(predictions_df)}"
```

### 2. **Consistent Output Schema** (CRITICAL)

All three modes must produce outputs with:
- Same number of rows (all positions)
- Same columns (including metadata)
- Same coordinate system (gene-relative: 1 to L)

### 3. **Mode-Specific Behavior** (IMPORTANT)

- **`base_only`**: `donor_meta = donor_score` (no recalibration)
- **`hybrid`**: Recalibrate only uncertain positions
- **`meta_only`**: Recalibrate ALL positions

But all modes must output predictions for **all** positions.

---

## Next Steps

### Immediate Actions

1. ✅ **Document the bug** (this file)
2. ⏳ **Investigate the filtering logic** in `enhanced_selective_inference.py`
3. ⏳ **Fix the position coverage issue**
4. ⏳ **Re-test all three modes** with position count verification
5. ⏳ **Re-run comprehensive tests** once fixed

### Testing Strategy

After fixing:
1. **Sanity check**: Verify `len(output) == gene_length` for all modes
2. **Consistency check**: Verify all modes have identical position lists
3. **Score check**: Verify base scores are identical, meta scores differ appropriately
4. **Performance check**: Calculate F1 scores and compare across modes

---

## References

- **Inference workflow**: `meta_spliceai/splice_engine/meta_models/workflows/inference/enhanced_selective_inference.py`
- **Integration guide**: `docs/INTEGRATION_GUIDE.md`
- **Test scripts**:
  - `scripts/testing/test_inference_direct.py`
  - `scripts/testing/test_three_modes_comparison.py`
  - `scripts/testing/test_all_modes_comprehensive_v2.py`

---

## Conclusion

The inference workflow has a **fundamental bug** where it's not generating predictions for all positions in the gene sequence. This makes all current test results **invalid** and blocks further progress on:
- Performance evaluation
- Mode comparison
- Variant analysis integration

**Priority**: 🔴 **CRITICAL** - Must fix immediately before any further testing or development.

