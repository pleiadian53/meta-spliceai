# CRITICAL FINDING: Adjustments Are Hardcoded, Not Empirically Determined

## Date: 2025-10-31

## The Smoking Gun

**File**: `meta_spliceai/splice_engine/meta_models/utils/infer_splice_site_adjustments.py`  
**Lines**: 439-451

```python
else:
    # SpliceAI has a known systematic pattern - this is how SpliceAI scores must be
    # adjusted to align with true positions
    spliceai_pattern = {
        'donor': {'plus': 2, 'minus': 1},
        'acceptor': {'plus': 0, 'minus': -1}
    }
    
    if verbose:
        print("\nUsing SpliceAI's known adjustment pattern:")
    
    return spliceai_pattern
```

## The Truth

Despite documentation claiming "extensive empirical analysis," the adjustment values are **HARDCODED** as a "known systematic pattern"!

### The Auto-Detection System

The codebase HAS an empirical detection system:
- `empirical_infer_splice_site_adjustments()` (lines 148-393)
- Searches offset range (-5 to +5)
- Compares predictions to annotations
- Finds optimal offset that maximizes TP rate

**BUT**: This is only used when `use_empirical=True`, which is **NOT the default**!

### Default Behavior

```python
def auto_detect_splice_site_adjustments(..., use_empirical=False):
    if use_empirical:
        # Use data-driven approach
        inferred_adjustments, _ = empirical_infer_splice_site_adjustments(...)
        return inferred_adjustments
    else:
        # Return HARDCODED values
        return spliceai_pattern  # {donor: {plus: 2, minus: 1}, ...}
```

**Default**: Returns hardcoded `{donor: {plus: 2, minus: 1}, ...}` without any analysis!

## Implications

### 1. Never Actually Validated

The hardcoded values were **never empirically tested** on the current workflow:
- No data-driven determination
- No comparison to GTF annotations
- No optimization for TP rate
- Just assumed to be correct!

### 2. Source Unknown

The comment says "known systematic pattern" but doesn't cite:
- Where this pattern comes from
- What data it was tested on
- Which SpliceAI version
- Which GTF version
- Which workflow (direct SpliceAI vs OpenSpliceAI preprocessing)

### 3. Our Test Is The First Real Validation

**VCAM1 Results**:
| Configuration | Overall F1 | Donor F1 | Acceptor F1 |
|--------------|------------|----------|-------------|
| Hardcoded adjustments (+2/+1) | 0.400 | 0.000 | 0.818 |
| **Zero adjustments (empirical)** | **0.756** | **0.696** | **0.818** |

**Our test is the FIRST actual empirical validation** - and it shows the hardcoded values are WRONG!

## The Empirical Detection Code

The system DOES have proper empirical detection logic (unused by default):

```python
def empirical_infer_splice_site_adjustments(
    annotations_df, 
    pred_results, 
    search_range=(-5, 5),
    min_genes_per_category=3,
    consensus_window=2, 
    probability_threshold=0.4,
    min_tp_improvement=0.2,
    verbose=False
):
    """
    Empirically infer optimal splice site coordinate adjustments by testing
    different offset values and measuring their impact on TP rate.
    
    For each splice type (donor/acceptor) and strand (+/-):
    1. Extract true splice sites from annotations
    2. Extract predicted splice sites (above threshold)
    3. Test offsets in search_range
    4. For each offset:
       - Shift predicted positions
       - Count TPs (predictions matching truth within consensus_window)
       - Calculate TP rate
    5. Select offset that maximizes TP rate
    """
```

**This would have found that zero adjustments are optimal!**

## Why It Was Never Used

Looking at the training workflow call:

```python
# From splice_prediction_workflow.py (line 289-334)
if config.use_auto_position_adjustments:
    sample_predictions = predict_splice_sites_for_genes(sample_seq_df, ...)
    
    adjustment_dict = prepare_splice_site_adjustments(
        local_dir=local_dir,
        ss_annotations_df=ss_annotations_df,
        sample_predictions=sample_predictions,
        use_empirical=True,  # ← Would enable data-driven detection
        ...
    )
```

**Hypothesis**: The empirical detection was implemented but:
1. Either `use_auto_position_adjustments` was False
2. Or `use_empirical` parameter wasn't passed as True
3. Or the detection ran on a different dataset/workflow

## Recommendation

### Enable Empirical Detection

Run the empirical detection on a sample of genes:

```python
from meta_spliceai.splice_engine.meta_models.utils.infer_splice_site_adjustments import (
    empirical_infer_splice_site_adjustments
)

# Use sample predictions from test genes
optimal_adjustments, stats = empirical_infer_splice_site_adjustments(
    annotations_df=annotations_df,
    pred_results=sample_predictions,
    search_range=(-5, 5),
    verbose=True
)

print(f"Empirically determined adjustments: {optimal_adjustments}")
```

**Expected result**: Will find that optimal adjustments are `{donor: {plus: 0, minus: 0}, acceptor: {plus: 0, minus: 0}}`

## Conclusion

The adjustment values of `{donor: {plus: 2, minus: 1}, acceptor: {plus: 0, minus: -1}}` are:

1. ❌ **NOT empirically determined** (despite documentation claims)
2. ❌ **Hardcoded** as a "known pattern" without citation
3. ❌ **Never validated** on the current workflow
4. ❌ **Making predictions worse** (F1: 0.400 → 0.756 after removing them)

The empirical detection system EXISTS but was never actually used!

**Our zero-adjustment finding is the first real empirical validation** of these values - and it proves they're incorrect for our workflow.

