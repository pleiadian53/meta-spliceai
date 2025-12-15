# PR-AUC Analysis Plan: Proper Comparison to SpliceAI

## Date: 2025-10-31

## Why PR-AUC Matters

**SpliceAI paper reports**: PR-AUC = 0.97

This is the **correct metric** to compare against, not F1 at a fixed threshold!

### PR-AUC vs F1

| Metric | Threshold | Robustness | What It Measures |
|--------|-----------|------------|------------------|
| **F1** | Fixed (e.g., 0.5) | Sensitive to threshold choice | Performance at ONE operating point |
| **PR-AUC** | All thresholds | Robust | Performance across ALL operating points |

**Key Insight**: Our F1 of 0.756 at threshold=0.5 may not be the optimal threshold! PR-AUC evaluates performance across all possible thresholds.

## Why F1 Can Be Misleading

### Example Scenario

**Gene with weak splice sites**:
- True positives have scores: 0.4, 0.45, 0.48 (below our 0.5 threshold)
- At threshold=0.5: F1 = 0.0 (all missed) ❌
- At threshold=0.3: F1 = 0.9 (all found) ✅
- **PR-AUC**: 0.95 (captures performance across thresholds) ✅

### Our VCAM1 Results Revisited

- **F1 at 0.5**: 0.756 (looks moderate)
- **PR-AUC**: ??? (need to calculate - may be much higher!)

## The ±2bp Tolerance Window

**User's clarification**: This was for handling minor definition differences, NOT for hiding coordinate problems.

**Makes sense**: Different annotation systems may define splice sites slightly differently:
- Last nucleotide of exon vs first nucleotide of intron
- 0-based vs 1-based coordinates
- GT dinucleotide start vs end

**The ±2bp window ensures fair comparison** regardless of these minor differences.

## Calculation Plan

### 1. For Each Gene

```python
# Get all positions and scores
positions = predictions_df['position']
scores = predictions_df['donor_score_donor_view']  # Use view-specific scores

# Create binary labels (1 = true splice site, 0 = not)
labels = [1 if pos in true_donor_positions else 0 for pos in positions]

# Calculate PR curve across all thresholds
precision, recall, thresholds = precision_recall_curve(labels, scores)
pr_auc = auc(recall, precision)

# Or use sklearn's average_precision_score (equivalent)
ap = average_precision_score(labels, scores)
```

### 2. Aggregate Across Genes

- Calculate mean PR-AUC across all test genes
- Calculate median, std, min, max
- Stratify by:
  - Splice type (donor vs acceptor)
  - Strand (+ vs -)
  - Gene characteristics (length, number of sites)

### 3. Compare to SpliceAI

**Target**: PR-AUC = 0.97

**Thresholds for interpretation**:
- PR-AUC ≥ 0.95: ✅ Excellent, matches SpliceAI
- PR-AUC 0.90-0.95: ✅ Good, close to SpliceAI
- PR-AUC 0.80-0.90: ⚠️  Moderate, some gap
- PR-AUC < 0.80: ❌ Needs investigation

## Expected Findings

### Hypothesis 1: Our Performance Is Actually Good

- F1=0.756 at threshold=0.5 may be suboptimal threshold
- PR-AUC may be ≥0.90, much closer to 0.97
- This would validate that zero adjustments are correct

### Hypothesis 2: Performance Varies by Gene

- Some genes may have PR-AUC ≈ 0.97 (perfect)
- Others may be lower due to:
  - Weak splice signals
  - Non-canonical sites
  - Alternative splicing complexity
  - Gene-specific factors

### Hypothesis 3: Donors vs Acceptors

- Acceptors may have higher PR-AUC (we saw F1=0.818)
- Donors may be slightly lower (we saw F1=0.696)
- But both should be >0.90 if model is working correctly

## Implementation Status

✅ **Created**: `scripts/testing/calculate_pr_auc.py`

**Features**:
- Calculates PR-AUC per gene
- Uses view-specific score columns (e.g., `donor_score_donor_view`)
- Aggregates across all test genes
- Compares to SpliceAI's reported 0.97
- Stratifies by splice type
- Provides per-gene breakdown

**Usage**:
```bash
python scripts/testing/calculate_pr_auc.py
```

## Next Steps

1. ✅ **Wait for 20-gene test to complete** (in progress)
2. **Run PR-AUC analysis** on the test genes
3. **Interpret results**:
   - If PR-AUC ≥ 0.90: Zero adjustments validated ✅
   - If PR-AUC < 0.80: Investigate further ⚠️
4. **Document findings** in final report
5. **Update recommendations** based on PR-AUC results

## Key Takeaway

**F1 at a fixed threshold is only one data point**. PR-AUC gives us the full picture across all possible operating points, which is what SpliceAI paper reports and what we should compare against.

Thank you to the user for pointing this out - it's the correct way to evaluate the system!

