# Score-Based Adjustment Detection Module

## Date: 2025-10-31

## Overview

Created a new module `infer_score_adjustments.py` that implements **empirical detection of score vector adjustments** using the correlated probability vector paradigm.

## Key Difference from Old Module

### Old: `infer_splice_site_adjustments.py`
- **Adjusted**: Position coordinates
- **Created**: New view of splice site annotations
- **Problems**: Position collisions, coverage loss
- **Example**: Shift position 100 → 98 (donor +2 adjustment)

### New: `infer_score_adjustments.py`
- **Adjusts**: Score vectors (correlated probabilities)
- **Creates**: New views of scores (donor view, acceptor view)
- **Benefits**: 100% coverage, no collisions, preserves probability constraints
- **Example**: Position 100 gets scores from position 98 (all 3 scores shift together)

## Architecture

### Core Function: `empirical_infer_score_adjustments()`

```python
def empirical_infer_score_adjustments(
    annotations_df: pl.DataFrame,      # Ground truth annotations
    pred_results: Dict[str, Dict],     # Predictions per gene
    search_range: Tuple[int, int] = (-5, 5),  # Range of shifts to test
    probability_threshold: float = 0.5,
    min_f1_improvement: float = 0.05,
    verbose: bool = False
) -> Tuple[Dict, Dict]:
    """
    Test different score shift amounts and find optimal adjustments.
    
    For each splice type + strand combination:
    1. Extract genes with that combination
    2. For each shift amount in search_range:
       a. Apply correlated probability vector shift
       b. Evaluate predictions vs annotations
       c. Calculate F1 score
    3. Select shift that maximizes F1
    4. Only use non-zero shift if improvement > threshold
    
    Returns optimal adjustments and detailed statistics.
    """
```

### Uses Correlated Probability Vectors

```python
from score_adjustment_v2 import create_splice_type_views

# Create views with correlated probability shifts
views = create_splice_type_views(
    donor_scores=donor_scores,
    acceptor_scores=acceptor_scores,
    neither_scores=neither_scores,
    strand=strand,
    adjustment_dict=test_adjustment,
    verbose=False
)

# Use appropriate view for evaluation
if splice_type == 'donor':
    adjusted_scores = views['donor_view'][0]  # donor scores from donor view
else:
    adjusted_scores = views['acceptor_view'][1]  # acceptor scores from acceptor view
```

This ensures:
- All three scores (donor, acceptor, neither) shift together
- Probability constraint maintained: sum = 1.0
- No position collisions
- 100% coverage

### Entry Point: `auto_detect_score_adjustments()`

```python
def auto_detect_score_adjustments(
    annotations_df: pl.DataFrame,
    pred_results: Dict[str, Dict[str, Any]],
    use_empirical: bool = True,
    search_range: Tuple[int, int] = (-5, 5),
    threshold: float = 0.5,
    verbose: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    Main entry point for adjustment detection.
    
    If use_empirical=True and predictions available:
        Run empirical detection
    Else:
        Return zero adjustments (assume base model aligned)
    """
```

## Usage

### 1. Basic Usage

```python
from meta_spliceai.splice_engine.meta_models.utils.infer_score_adjustments import (
    auto_detect_score_adjustments
)

# Detect optimal adjustments
adjustments = auto_detect_score_adjustments(
    annotations_df=annotations,    # Polars DataFrame with annotations
    pred_results=predictions,      # Dict of predictions per gene
    use_empirical=True,
    search_range=(-5, 5),
    threshold=0.5,
    verbose=True
)

print(f"Optimal adjustments: {adjustments}")
# Output: {'donor': {'plus': 0, 'minus': 0}, 'acceptor': {'plus': 0, 'minus': 0}}
```

### 2. Integration with Workflow

```python
from meta_spliceai.splice_engine.meta_models.utils.infer_score_adjustments import (
    auto_detect_score_adjustments,
    save_adjustment_dict
)

# Generate sample predictions on subset of genes
sample_predictions = predict_splice_sites_for_genes(
    sample_genes, models=models, context=10000
)

# Detect adjustments
adjustments = auto_detect_score_adjustments(
    annotations_df=splice_sites_df,
    pred_results=sample_predictions,
    use_empirical=True,
    verbose=True
)

# Save for future use
save_adjustment_dict(
    adjustments,
    output_path="data/ensembl/score_adjustments_v2.json"
)
```

### 3. Testing

```bash
# Run the test script
python scripts/testing/test_score_adjustment_detection.py
```

This will:
1. Load predictions for test genes
2. Run empirical detection
3. Compare to old hardcoded values
4. Report differences
5. Save detected adjustments

## Expected Results

Based on our VCAM1 test:
- **Old adjustments**: `{donor: {plus: 2, minus: 1}, acceptor: {plus: 0, minus: -1}}`
- **Expected detection**: `{donor: {plus: 0, minus: 0}, acceptor: {plus: 0, minus: 0}}`

The empirical detection should find that **zero adjustments are optimal** for our current workflow (SpliceAI → GTF annotations directly).

## Advantages Over Old Module

1. **Correct Paradigm**: Shifts scores, not positions
2. **Preserves Coverage**: 100% coverage maintained
3. **No Collisions**: Positions stay fixed
4. **Probability Constraints**: Sum = 1.0 maintained
5. **Splice-Type-Specific**: Uses appropriate view for each type
6. **Data-Driven**: Tests actual F1 improvement, not just position matching

## Integration Points

### Replace Old Module Calls

**Before** (old position-based):
```python
from meta_spliceai.splice_engine.meta_models.utils.infer_splice_site_adjustments import (
    empirical_infer_splice_site_adjustments
)
```

**After** (new score-based):
```python
from meta_spliceai.splice_engine.meta_models.utils.infer_score_adjustments import (
    empirical_infer_score_adjustments
)
```

### Use with score_adjustment_v2.py

The new module is designed to work seamlessly with `score_adjustment_v2.py`:

```python
from meta_spliceai.splice_engine.meta_models.utils.infer_score_adjustments import (
    auto_detect_score_adjustments
)
from meta_spliceai.splice_engine.meta_models.utils.score_adjustment import (
    adjust_predictions_dataframe_v2
)

# Detect adjustments
adjustments = auto_detect_score_adjustments(...)

# Apply to predictions
adjusted_df = adjust_predictions_dataframe_v2(
    predictions_df=predictions,
    adjustment_dict=adjustments,
    method='multi_view',
    verbose=True
)
```

## Next Steps

1. **Run test script** to validate detection works
2. **Compare results** to old hardcoded values
3. **Integrate** into training workflow (`splice_prediction_workflow.py`)
4. **Update** `prepare_splice_site_adjustments()` to use new module
5. **Document** workflow-specific adjustments (SpliceAI vs OpenSpliceAI)

## File Locations

- **New module**: `meta_spliceai/splice_engine/meta_models/utils/infer_score_adjustments.py`
- **Test script**: `scripts/testing/test_score_adjustment_detection.py`
- **Old module**: `meta_spliceai/splice_engine/meta_models/utils/infer_splice_site_adjustments.py` (keep for reference)
- **Score adjustment v2**: `meta_spliceai/splice_engine/meta_models/utils/score_adjustment_v2.py`

## Summary

The new `infer_score_adjustments.py` module implements the **correct paradigm** for adjustment detection:
- Shifts **score vectors** with correlated probabilities
- Maintains 100% coverage and probability constraints
- Uses splice-type-specific views for evaluation
- Empirically determines optimal shifts based on F1 improvement

This module completes the implementation of your insight about correlated probability vectors, making the adjustment detection system consistent with the score-shifting approach we developed.

