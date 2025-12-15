# Diverse Gene Test Plan

**Date**: November 2, 2025  
**Purpose**: Verify base model performance after multi-build system update

---

## Objective

Test the base model (SpliceAI) with a diverse set of genes to:

1. **Verify System Integrity**: Ensure the multi-build system works correctly after the Registry refactor
2. **Test Different Gene Types**: Compare performance on protein-coding vs lncRNA genes
3. **Validate Coordinate Alignment**: Confirm zero adjustments are needed for GRCh37
4. **Explore Performance Boundaries**: Understand how SpliceAI performs on different biotypes

---

## Test Design

### Gene Selection

**Protein-Coding Genes**: 15 randomly sampled
- Expected: High F1 scores (≥0.8), high PR-AUC (≥0.9)
- These are the genes SpliceAI was trained on

**lncRNA Genes**: 5 randomly sampled
- Expected: Lower performance (unknown, exploratory)
- lncRNAs have different splicing patterns than protein-coding genes
- Interesting to see if SpliceAI generalizes

**Total**: 20 genes

### Why This Mix?

1. **Protein-Coding**: Validates that the system still works correctly
2. **lncRNA**: Explores model generalization and potential limitations
3. **Diverse Sample**: Different from the previous test (which used 50 protein-coding genes)

---

## Expected Results

### Coordinate Alignment

**Expected**: Zero adjustments needed

**Reasoning**:
- SpliceAI was trained on GRCh37
- We're evaluating on GRCh37 (Ensembl release 87)
- Predictions should align perfectly with annotations

**If Non-Zero Adjustments Detected**:
- Indicates a coordinate mismatch problem
- Would need to investigate annotation source differences

### Performance Metrics

**Protein-Coding Genes**:
- F1 Score: ≥0.80 (ideally ≥0.85)
- PR-AUC: ≥0.90 (ideally ≥0.95)
- Should match or exceed previous test results

**lncRNA Genes**:
- F1 Score: Unknown (exploratory)
- PR-AUC: Unknown (exploratory)
- Likely lower than protein-coding genes

**Why Lower Performance on lncRNAs?**:
- SpliceAI was primarily trained on protein-coding genes
- lncRNAs have different splicing patterns
- Splice sites may be less conserved
- Lower expression levels in training data

---

## Test Implementation

### Script

`scripts/testing/test_base_model_diverse_genes.py`

### Key Features

1. **Dynamic Gene Sampling**: Reads GTF and samples genes by biotype
2. **Separate Metrics**: Calculates F1/PR-AUC for each biotype independently
3. **Comparison**: Shows performance difference between biotypes
4. **Alignment Check**: Verifies coordinate adjustments

### Configuration

```python
Registry(build='GRCh37', release='87')
```

**Critical**: Must specify `release='87'` because:
- Default release is "112" (for GRCh38)
- GRCh37 uses release "87"
- Without this, GTF path won't resolve correctly

---

## Metrics Calculated

### Per Biotype, Per Splice Type

For each combination of (biotype, splice_type):

1. **Precision**: TP / (TP + FP)
2. **Recall**: TP / (TP + FN)
3. **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
4. **PR-AUC**: Area under precision-recall curve
5. **Counts**: TP, FP, FN, TN

### Summary Comparison

**Protein-Coding vs lncRNA**:
- ΔF1 = F1(protein-coding) - F1(lncRNA)
- ΔPR-AUC = PR-AUC(protein-coding) - PR-AUC(lncRNA)

---

## Success Criteria

### Must Pass

1. ✅ **Workflow Completes**: No errors, all genes processed
2. ✅ **Predictions Generated**: Non-empty positions DataFrame
3. ✅ **Zero Adjustments**: Coordinate alignment is perfect
4. ✅ **High Protein-Coding Performance**: F1 ≥0.80, PR-AUC ≥0.90

### Nice to Have

1. 🎯 **lncRNA Performance**: F1 ≥0.60 (would indicate good generalization)
2. 🎯 **Consistent with Previous Test**: Similar F1 scores to the 50-gene test

### Exploratory

1. 🔍 **Performance Gap**: How much worse is lncRNA vs protein-coding?
2. 🔍 **Splice Type Differences**: Do donors vs acceptors differ by biotype?

---

## Interpretation Guide

### Scenario 1: High Performance on Both Biotypes

**Protein-Coding F1 ≥0.85, lncRNA F1 ≥0.75**

**Interpretation**: SpliceAI generalizes well to lncRNAs
**Conclusion**: Excellent! Model is robust across gene types

### Scenario 2: High Protein-Coding, Lower lncRNA

**Protein-Coding F1 ≥0.85, lncRNA F1 = 0.50-0.70**

**Interpretation**: Expected - lncRNAs have different splicing patterns
**Conclusion**: Normal. SpliceAI is optimized for protein-coding genes

### Scenario 3: High Protein-Coding, Very Low lncRNA

**Protein-Coding F1 ≥0.85, lncRNA F1 <0.50**

**Interpretation**: SpliceAI doesn't generalize well to lncRNAs
**Conclusion**: Model is specialized for protein-coding genes
**Action**: Consider training a separate model for lncRNAs

### Scenario 4: Lower Protein-Coding Performance

**Protein-Coding F1 <0.80**

**Interpretation**: Something is wrong!
**Possible Causes**:
- Coordinate mismatch (check adjustments)
- Wrong genome build
- Data corruption
- System regression

**Action**: Investigate immediately

---

## Previous Test Comparison

### Previous Test (50 Protein-Coding Genes)

**Date**: November 1, 2025
**Genes**: 50 protein-coding genes (hardcoded list)
**Results**:
- Average F1: 0.9312 (excellent!)
- Adjustments: Zero (perfect alignment)
- PR-AUC: Not calculated in that test

### This Test (20 Diverse Genes)

**Date**: November 2, 2025
**Genes**: 15 protein-coding + 5 lncRNA (randomly sampled)
**Expected**:
- Protein-coding F1: Similar to previous test (~0.93)
- lncRNA F1: Unknown (exploratory)
- Adjustments: Zero (same setup)

### Why Different Sample?

1. **Verify Robustness**: Different genes test generalization
2. **Explore Biotypes**: lncRNAs are interesting
3. **Avoid Overfitting**: Not using the same genes repeatedly

---

## Post-Test Actions

### If Test Passes

1. ✅ Document results in `docs/testing/`
2. ✅ Update `PHASE_2_COMPLETE_DIRECTORY_STRUCTURE.md`
3. ✅ Proceed with Phase 3 (incremental builder updates)
4. ✅ Begin OpenSpliceAI integration (Phase 4)

### If Test Fails

1. 🔍 Investigate root cause
2. 🔍 Check coordinate alignment
3. 🔍 Verify data integrity
4. 🔍 Review Registry path resolution
5. 🔍 Fix issues before proceeding

---

## Technical Notes

### Release Number Specification

**Critical Lesson**: Always specify `release` when using Registry for GRCh37

```python
# ❌ WRONG - uses default release "112"
registry = Registry(build='GRCh37')

# ✅ CORRECT - uses GRCh37 release "87"
registry = Registry(build='GRCh37', release='87')
```

**Why This Matters**:
- GTF filename template: `Homo_sapiens.GRCh37.{release}.gtf`
- Without correct release, path won't resolve
- Results in `FileNotFoundError`

### Gene Sampling Strategy

**Random Sampling**: Uses `pandas.sample(random_state=42)` for reproducibility

**Benefits**:
- Different genes each time (if seed changed)
- Avoids bias from hardcoded gene lists
- Tests system with diverse gene characteristics

**Reproducibility**: Fixed seed (42) ensures same genes if re-run

---

## Future Improvements

### Test Suite Expansion

1. **More Biotypes**: Test pseudogenes, miRNAs, etc.
2. **Stratified Sampling**: By chromosome, gene length, exon count
3. **Edge Cases**: Very long genes, single-exon genes, overlapping genes
4. **Cross-Build**: Same genes on GRCh37 vs GRCh38

### Automated Testing

1. **CI/CD Integration**: Run on every commit
2. **Performance Regression**: Alert if F1 drops
3. **Coordinate Alignment**: Automated check for adjustments
4. **Multi-Build**: Test all builds automatically

### Visualization

1. **Performance by Biotype**: Bar charts comparing F1 scores
2. **PR Curves**: Separate curves for each biotype
3. **Coordinate Alignment**: Heatmap of adjustment values
4. **Gene-Level Metrics**: Per-gene F1 scores

---

## Summary

This test validates that:
1. ✅ Multi-build system works correctly after Registry refactor
2. ✅ Base model performance is maintained
3. ✅ Coordinate alignment is perfect (zero adjustments)
4. 🔍 SpliceAI performance on different gene biotypes

**Expected Outcome**: High protein-coding performance, exploratory lncRNA results, zero coordinate adjustments

**Next Steps**: If test passes, proceed with OpenSpliceAI integration



