# Investigation: SpliceAI Coordinate Adjustment Values

## Date: 2025-10-31

## Summary

The documented adjustment values (`donor: {plus: 2, minus: 1}, acceptor: {plus: 0, minus: -1}`) appear to be **INCORRECT** for our current dataset, despite being described as "empirically determined."

## Current Test Results

### VCAM1 (ENSG00000162692, + strand, 19,304 bp)

| Configuration | Overall F1 | Donor F1 | Acceptor F1 | Donor Exact Match |
|--------------|------------|----------|-------------|-------------------|
| **Zero adjustments** | **0.756** | **0.696** | 0.818 | **57.1%** (8/14) |
| With +2/+1 adjustments | 0.400 | 0.000 | 0.818 | 0.0% (0/14) |

**Key Finding**: Base SpliceAI predictions are ALREADY aligned with our GTF annotations!

## Performance Gap Analysis

### Expected vs Actual Performance

**SpliceAI Documentation** (Jaganathan et al., 2019):
- **Top-k accuracy**: 95%
- **PR-AUC**: 0.97
- Evaluated on held-out test set

**Our Results** (VCAM1, zero adjustments):
- **Overall F1**: 0.756 (≈75.6% accuracy)
- **Donor F1**: 0.696
- **Acceptor F1**: 0.818
- **Donor exact match**: 57.1% (8/14 donors)
- **Acceptor exact match**: 69.2% (9/13 acceptors)

**Gap**: ~20% lower than documented performance

## Possible Explanations for Performance Gap

### 1. Evaluation Metric Differences

**Top-k Accuracy** (SpliceAI paper):
- "Fraction of correctly predicted splice sites at the threshold where the number of predicted sites equals the actual number"
- This is a **lenient** metric that adjusts threshold per gene
- Allows for some false positives if there are corresponding false negatives

**F1 Score** (Our evaluation):
- Fixed threshold (0.5)
- Penalizes both false positives AND false negatives equally
- More stringent than top-k accuracy

**Expected relationship**: F1 < Top-k accuracy

### 2. Dataset Differences

**SpliceAI Training/Test Data**:
- Specific genome build (likely GRCh37/hg19 or GRCh38)
- Specific GTF version
- Curated gene set
- May exclude difficult cases

**Our Data**:
- Ensembl GRCh38.112 GTF
- Single test gene (VCAM1)
- May include edge cases

### 3. Gene-Specific Factors

**VCAM1 Characteristics**:
- 14 donors, 13 acceptors
- 19,304 bp length
- May have non-canonical splice sites
- May have weak splice signals

**Missing donors** (6/14 = 43%):
- Could be weak splice sites with scores < 0.5
- Could be non-canonical (non-GT/AG)
- Could be alternatively spliced

### 4. Threshold Selection

**Our threshold**: 0.5 (arbitrary)

**Optimal threshold**: Unknown for this gene

**Impact**: A lower threshold might capture more true positives but also more false positives

## Source of Adjustment Values

### Documentation Trail

1. **`BASE_MODEL_SPLICE_SITE_DEFINITIONS.md`** (lines 127-136):
   ```
   From extensive empirical analysis, SpliceAI predictions require these adjustments
   ```
   - Source: "Empirical analysis from `splice_utils.py`"
   - No details on methodology

2. **`coordinate_reconciliation.py`** (lines 107-113):
   ```python
   # SpliceAI model adjustments (from your analysis)
   'spliceai_model_adjustments': CoordinateOffset(
       donor_plus=2,    # SpliceAI predicts 2nt upstream on plus strand
       donor_minus=1,   # SpliceAI predicts 1nt upstream on minus strand
       acceptor_plus=0, # SpliceAI matches GTF position on plus strand
       acceptor_minus=-1 # SpliceAI predicts 1nt downstream on minus strand
   ),
   ```
   - Comment: "(from your analysis)" - suggests user-provided values

3. **`SPLICE_SITE_DEFINITION_ANALYSIS.md`** (lines 69-82):
   ```python
   # From your splice_utils.py:
   spliceai_adjustments = {
       'donor': {'plus': 2, 'minus': 1},
       'acceptor': {'plus': 0, 'minus': -1}
   }
   ```
   - Described as "Your Current Adjustments"

### Hypothesis

The adjustment values may have been:
1. **Borrowed from OpenSpliceAI analysis** where they were needed
2. **Determined on a different dataset** (different GTF version, genome build)
3. **Incorrectly applied** (direction reversed, interpretation error)
4. **Based on incomplete testing** (tested on minus strand only, generalized incorrectly)

## OpenSpliceAI Comparison

### Known OpenSpliceAI Offsets

From `SPLICE_SITE_DEFINITION_ANALYSIS.md`:
- **Donor**: +1 relative to MetaSpliceAI (both strands)
- **Acceptor**: 0 relative to MetaSpliceAI (both strands)

### Combined Offsets (if using OpenSpliceAI preprocessing + SpliceAI model)

- **Donor (+)**: +1 (OpenSpliceAI) + 2 (SpliceAI) = +3 total
- **Donor (-)**: +1 (OpenSpliceAI) + 1 (SpliceAI) = +2 total
- **Acceptor (+)**: 0 (OpenSpliceAI) + 0 (SpliceAI) = 0 total
- **Acceptor (-)**: 0 (OpenSpliceAI) + (-1) (SpliceAI) = -1 total

**Question**: Are we using OpenSpliceAI preprocessing? If not, why apply these adjustments?

## Recommendations

### 1. Comprehensive Empirical Validation

Test on 20+ genes across:
- Both strands (+ and -)
- Different gene lengths
- Different splice site strengths
- Different chromosomes

For each gene, compare:
- Zero adjustments
- Current adjustments (+2/+1, 0/-1)
- Reversed adjustments (-2/-1, 0/+1)
- Other combinations

### 2. Investigate Minus Strand

Our test was on **+ strand only**. The adjustments may be:
- Correct for minus strand
- Incorrect for plus strand
- Or vice versa

### 3. Check Data Processing Pipeline

Verify:
- Are we using the same GTF file for annotations and predictions?
- Are coordinates being converted (0-based ↔ 1-based) anywhere?
- Are we using OpenSpliceAI preprocessing inadvertently?

### 4. Examine Training Data

If possible, compare:
- Our GTF annotations vs SpliceAI training data annotations
- Coordinate definitions
- Splice site definitions (last exon nt vs first intron nt, etc.)

### 5. Optimize Threshold

Instead of fixed 0.5, try:
- Top-k approach (adjust threshold to match number of true sites)
- ROC curve analysis to find optimal threshold
- Gene-specific thresholds

## Next Steps

1. ✅ **DONE**: Tested VCAM1 with zero adjustments → F1=0.756
2. **TODO**: Test 20+ genes with zero adjustments
3. **TODO**: Test genes on minus strand specifically
4. **TODO**: Compare zero vs current adjustments across full gene set
5. **TODO**: Investigate if adjustments were meant for OpenSpliceAI workflow only
6. **TODO**: Document correct adjustment values for our specific GTF/workflow

## Conclusion

The documented adjustment values appear to be **incorrect for our current workflow**. The base SpliceAI model predictions are already well-aligned with our GTF annotations (F1=0.756), and applying the documented adjustments makes performance WORSE (F1=0.400).

However, our performance (75.6%) is still below SpliceAI's documented 95% top-k accuracy. This gap likely stems from:
1. Different evaluation metrics (F1 vs top-k)
2. Different thresholds (0.5 vs optimized)
3. Gene-specific factors (VCAM1 may be difficult)
4. Small sample size (1 gene tested so far)

**The multi-view adjustment system we built is correct and should be retained** for use with models that DO need adjustments (e.g., OpenSpliceAI). But for SpliceAI with our current GTF, zero adjustments appear optimal.

