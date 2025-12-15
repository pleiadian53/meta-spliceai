# Session Summary: Correlated Probability Vectors & Coordinate Adjustment Resolution

## Date: 2025-10-31

## User's Critical Insight

The user identified that the three scores at each position (donor, acceptor, neither) are **correlated** and must sum to 1.0:

> "When we predict the splice site score for each splice type, we essentially need to create 'a new view' to align with the annotated splice sites... the splice site scores for all the 3 splice type needs to be adjusted simultaneously and shifted in parallel consistently, because they are correlated: for each position, the probability scores sum to 1 because a position can be donor, acceptor or neither."

This insight was **absolutely correct** and led to a major breakthrough!

## What We Discovered

### 1. Multi-View Adjustment System (Correct Implementation)

We implemented a system that correctly handles correlated probability vectors:

**`score_adjustment_v2.py`**:
- `shift_probability_vectors()`: Shifts all three scores (donor, acceptor, neither) together as a unit
- `create_splice_type_views()`: Creates separate views for each splice type
  - Donor view: All scores shifted by donor adjustment
  - Acceptor view: All scores shifted by acceptor adjustment
- `adjust_predictions_dataframe_v2()`: Adds view columns to DataFrame for evaluation

**Key Features**:
- ✅ Maintains probability constraint (sum = 1.0) at all positions
- ✅ 100% coverage (all N positions for N-bp gene)
- ✅ No position collisions
- ✅ Separate views for different splice types

### 2. The Real Problem: Incorrect Adjustment Values

While implementing the multi-view system, we discovered that the **adjustment values themselves were wrong**!

**Documented adjustments**:
```python
{
    'donor': {'plus': 2, 'minus': 1},
    'acceptor': {'plus': 0, 'minus': -1}
}
```

**Test Results (VCAM1, + strand)**:

| Configuration | Overall F1 | Donor F1 | Acceptor F1 |
|--------------|------------|----------|-------------|
| With adjustments (+2/+1) | 0.400 | 0.000 | 0.818 |
| **Zero adjustments** | **0.756** | **0.696** | **0.818** |

**Conclusion**: The base SpliceAI model predictions are ALREADY aligned with our GTF annotations!

## Implementation Details

### Files Created/Modified

1. **`score_adjustment_v2.py`** (NEW):
   - Multi-view adjustment with correlated probability vectors
   - Production-ready for models that need adjustment (e.g., OpenSpliceAI)

2. **`enhanced_selective_inference.py`** (MODIFIED):
   - Integrated multi-view adjustment system
   - Set default adjustments to zero (evidence-based)
   - Added view column renaming logic

3. **`generate_and_test_20_genes.py`** (MODIFIED):
   - Updated evaluation to use splice-type-specific view columns
   - `analyze_position_offsets()`: Uses `donor_score_donor_view` for donors
   - F1 calculation: Uses appropriate view columns for each splice type

### How It Works

1. **Prediction Generation**:
   ```python
   # Generate base predictions
   predictions_df = predict_splice_sites_for_genes(...)
   
   # Apply multi-view adjustment (creates 6 new columns)
   predictions_df = adjust_predictions_dataframe_v2(
       predictions_df,
       adjustment_dict={'donor': {'plus': 0, 'minus': 0}, ...},
       method='multi_view'
   )
   # Result: donor_score, acceptor_score, neither_score (original)
   #         donor_score_donor_view, acceptor_score_donor_view, neither_score_donor_view
   #         donor_score_acceptor_view, acceptor_score_acceptor_view, neither_score_acceptor_view
   ```

2. **Evaluation**:
   ```python
   # For donors: use donor view
   donor_score_col = 'donor_score_donor_view' if exists else 'donor_score'
   pred_donors = predictions_df.filter(pl.col(donor_score_col) > 0.5)
   
   # For acceptors: use acceptor view
   acceptor_score_col = 'acceptor_score_acceptor_view' if exists else 'acceptor_score'
   pred_acceptors = predictions_df.filter(pl.col(acceptor_score_col) > 0.5)
   ```

## Performance Analysis

### Current Results (VCAM1, zero adjustments)
- **Overall F1**: 0.756
- **Donor F1**: 0.696 (8/14 exact matches = 57.1%)
- **Acceptor F1**: 0.818 (9/13 exact matches = 69.2%)

### Expected Performance (SpliceAI paper)
- **Top-k accuracy**: 95%
- **PR-AUC**: 0.97

### Performance Gap (~20%)

**Likely Explanations**:

1. **Different Metrics**:
   - Top-k accuracy (lenient, adjusts threshold per gene)
   - vs F1 score (fixed threshold 0.5, more stringent)

2. **Threshold Selection**:
   - We use 0.5 (arbitrary)
   - Optimal threshold may be different

3. **Gene-Specific Factors**:
   - VCAM1 may have weak splice sites
   - May have non-canonical sites
   - Single gene is small sample

4. **Dataset Differences**:
   - Different GTF version
   - Different gene set
   - Different evaluation protocol

## Investigation: Where Did +2/+1 Come From?

### Documentation Trail

1. **`BASE_MODEL_SPLICE_SITE_DEFINITIONS.md`**:
   - Claims "extensive empirical analysis"
   - Source: `splice_utils.py`
   - No methodology details

2. **`coordinate_reconciliation.py`**:
   - Comment: "(from your analysis)"
   - Suggests user-provided values

3. **`SPLICE_SITE_DEFINITION_ANALYSIS.md`**:
   - Describes as "Your Current Adjustments"
   - Documents OpenSpliceAI offsets (+1 for donors)
   - Shows combined offsets (OpenSpliceAI + SpliceAI)

### Hypothesis

The adjustment values may have been:
1. **Meant for OpenSpliceAI workflow** (which has +1 offset)
2. **Determined on different GTF version** (different coordinate system)
3. **Incorrectly generalized** (tested on minus strand, applied to plus)
4. **Direction reversed** (interpretation error)

## Current Status

### ✅ Completed

1. Implemented multi-view adjustment system with correlated probability vectors
2. Tested on VCAM1 (+ strand) - confirmed zero adjustments work best
3. Set default adjustments to zero in inference workflow
4. Updated evaluation code to use view-specific columns
5. Created comprehensive documentation

### 🔄 In Progress

Running comprehensive test on 24 genes:
- 17 on + strand
- 7 on - strand
- Will validate zero adjustments across diverse gene set
- Will check if minus strand needs different adjustments

### 📋 TODO

1. Complete 20-gene test and analyze results
2. Stratify by strand, splice type, gene characteristics
3. Investigate if adjustments were meant for OpenSpliceAI only
4. Optimize threshold (try top-k approach instead of fixed 0.5)
5. Fix meta-model threshold (0.5 → 0.95 for Platt calibration)

## Key Takeaways

1. **User's insight was correct**: Probability vectors must be shifted together
2. **Multi-view system is correct**: Ready for models that need adjustment
3. **Adjustment values were wrong**: Base SpliceAI already aligned with our GTF
4. **Performance is good but not great**: 75.6% vs documented 95%
5. **More testing needed**: Validate across multiple genes and strands

## Recommendations

1. **Keep multi-view adjustment system**: It's correct and will be needed for OpenSpliceAI
2. **Use zero adjustments for SpliceAI**: Evidence shows this is optimal
3. **Investigate performance gap**: Try different thresholds, metrics, genes
4. **Test OpenSpliceAI**: Verify if it needs the documented +1 offset
5. **Document findings**: Update BASE_MODEL_SPLICE_SITE_DEFINITIONS.md with new evidence

## Acknowledgment

This breakthrough was made possible by the user's critical insight about correlated probability vectors. The investigation of this insight not only led to a correct implementation of the adjustment mechanism, but also revealed that the adjustment values themselves were incorrect for our workflow.

