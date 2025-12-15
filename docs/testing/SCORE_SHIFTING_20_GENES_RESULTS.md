# Score-Shifting Validation: 20+ Genes Results

**Date**: 2025-10-31  
**Test**: Comprehensive validation of score-shifting coordinate adjustment on 24 NEW protein-coding genes

## Executive Summary

✅ **Coverage**: 100% for all 24 genes - Score-shifting maintains full coverage!  
❌ **Alignment**: **CRITICAL FAILURE** - Predictions are systematically misaligned with annotations

**Overall Average F1**: 0.186 (Target: ≥0.7)

## Detailed Results

### By Splice Type

| Splice Type | Average F1 | Avg Exact Match | Genes >70% Exact |
|-------------|------------|-----------------|------------------|
| **Donor**   | 0.004      | 0.3%            | 0/24             |
| **Acceptor**| 0.370      | 28.5%           | 2/24             |

### By Strand

| Strand | N Genes | Avg F1 | Donor Exact | Acceptor Exact |
|--------|---------|--------|-------------|----------------|
| **+**  | 15      | 0.295  | 0.0%        | 45.6%          |
| **-**  | 9       | 0.005  | 0.7%        | 0.0%           |

## Offset Patterns

### + Strand Genes
- **Donors**: Consistently off by **±2 nt** (53-57% of sites)
- **Acceptors**: **45-69% exact matches** ✅ (working for many genes)

### - Strand Genes  
- **Donors**: Off by **±1 nt** (68-71% of sites)
- **Acceptor**: Off by **±1 nt** (68-69% of sites)

## Example Gene Results

### ENSG00000162692 (VCAM1, + strand, 19,304 bp)
- Coverage: 100.0% ✅
- Donor: F1=0.000 (0% exact, 57% off by ±2)
- Acceptor: F1=0.818 (69% exact) ✅
- Overall F1: 0.400

### ENSG00000146648 (EGFR, + strand, 203,289 bp)
- Coverage: 100.0% ✅
- Donor: F1=0.000 (0% exact, 53% off by ±2)
- Acceptor: F1=0.761 (61% exact) ✅
- Overall F1: 0.386

### ENSG00000073756 (PTGS2, - strand, 9,132 bp)
- Coverage: 100.0% ✅
- Donor: F1=0.000 (0% exact, 71% off by ±1)
- Acceptor: F1=0.000 (0% exact, 69% off by ±1)
- Overall F1: 0.000

## Root Cause Analysis

The score-shifting implementation in `_apply_coordinate_adjustments()` has **systematic errors**:

### Issue 1: + Strand Donors
- **Expected**: Adjustment of +2 should align donors with GTF
- **Actual**: Donors are off by ±2 nt → adjustment is NOT being applied or applied incorrectly
- **Acceptors work**: Suggests acceptor adjustment (+3) is correct

### Issue 2: - Strand (Both Types)
- **Expected**: Adjustments should align both donors and acceptors
- **Actual**: Both are off by ±1 nt → adjustments are incorrect or reversed

### Issue 3: Direction of Shift
The current implementation uses:
```python
pl.col('donor_prob').shift(-donor_plus_adj)
```

**Problem**: `shift(-n)` moves values DOWN (from later indices to earlier indices). After sorting by position:
- `shift(-2)` means: position i gets the score from position i+2
- This is equivalent to shifting scores LEFT (to earlier positions)
- But we may need to shift RIGHT (to later positions) for some cases!

## Conclusion

**CRITICAL BUG FOUND**: The score-shifting logic has the **wrong direction** for some adjustments!

### What Works:
✅ 100% coverage maintained  
✅ No position collisions  
✅ Acceptors on + strand (45.6% exact match)

### What Fails:
❌ Donors on + strand (0.0% exact match, off by ±2)  
❌ Both types on - strand (0.0-0.7% exact match, off by ±1)

## Next Steps

1. **Review adjustment values** from GTF analysis
2. **Fix shift direction** in `_apply_coordinate_adjustments()`
3. **Test on same 24 genes** to verify fix
4. **Expect**: F1 ≥ 0.7 for all genes after fix

## Test Genes (24 total)

### + Strand (15 genes)
- ENSG00000162692 (VCAM1), ENSG00000163930 (BAP1), ENSG00000204287 (HLA-A)
- ENSG00000111640 (GAPDH), ENSG00000146648 (EGFR), ENSG00000136997 (MYC)
- ENSG00000147889 (CDKN2A), ENSG00000165731 (RET), ENSG00000107485 (GATA3)
- ENSG00000123374 (CDK2), ENSG00000111276 (CDKN1B), ENSG00000139687 (RB1)
- ENSG00000141736 (ERBB2), ENSG00000100030 (MAPK1), ENSG00000134086 (VHL)
- ENSG00000115414 (FN1), ENSG00000196549 (MME)

### - Strand (9 genes)
- ENSG00000117318 (ID3), ENSG00000138795 (LEF1), ENSG00000113558 (SKP2)
- ENSG00000164362 (TERT), ENSG00000105974 (CAV1), ENSG00000149925 (ALDOA)
- ENSG00000073756 (PTGS2)

---

**Status**: Score-shifting maintains coverage but has critical alignment bugs that must be fixed.

