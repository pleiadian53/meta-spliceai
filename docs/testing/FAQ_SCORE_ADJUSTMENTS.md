# FAQ: Score Adjustments

## Date: 2025-10-31

## Q1: If adjustments are zero, do we need to create score views?

**Short Answer: NO**

**Explanation:**

When `infer_score_adjustments.py` finds that optimal adjustments are all zero, it means the base model predictions are already aligned with the annotations. In this case:

- ❌ No score shifting needed
- ❌ No need to create `_donor_view` or `_acceptor_view` columns
- ✅ Use base scores directly: `donor_score`, `acceptor_score`, `neither_score`
- ✅ Simpler, faster code

**Code Pattern:**

```python
# Check if adjustments are needed
if all_adjustments_are_zero(adjustment_dict):
    # Use base scores directly
    donor_predictions = predictions_df.filter(
        pl.col('donor_score') > threshold
    )
    acceptor_predictions = predictions_df.filter(
        pl.col('acceptor_score') > threshold
    )
else:
    # Create views and use appropriate view for each splice type
    adjusted_df = adjust_predictions_dataframe_v2(
        predictions_df, 
        adjustment_dict, 
        method='multi_view'
    )
    
    # Use donor view for donor predictions
    donor_predictions = adjusted_df.filter(
        pl.col('donor_score_donor_view') > threshold
    )
    
    # Use acceptor view for acceptor predictions
    acceptor_predictions = adjusted_df.filter(
        pl.col('acceptor_score_acceptor_view') > threshold
    )
```

**Helper Function:**

```python
def all_adjustments_are_zero(adjustment_dict: Dict[str, Dict[str, int]]) -> bool:
    """Check if all adjustments are zero."""
    return all(
        adjustment_dict[splice_type][strand] == 0
        for splice_type in ['donor', 'acceptor']
        for strand in ['plus', 'minus']
    )
```

**Current Status:**

For SpliceAI → GTF workflow:
- ✅ Empirically confirmed: All adjustments are zero
- ✅ No views needed
- ✅ Can simplify inference workflow

---

## Q2: What's the difference between score_adjustment.py and score_adjustment_v2.py? Can they be consolidated?

**Short Answer: YES, already consolidated!**

**History:**

### Version 1 (score_adjustment.py) - INCORRECT ❌

**Problem:** Shifted donor, acceptor, and neither scores **independently**

```python
# WRONG: Independent shifts
adjusted_donor = shift_score_vector(donor_scores, donor_shift)
adjusted_acceptor = shift_score_vector(acceptor_scores, acceptor_shift)
adjusted_neither = neither_scores  # Not adjusted!

# Result: donor + acceptor + neither ≠ 1.0 (BROKEN!)
```

**Why wrong:**
- Breaks probability constraint (sum ≠ 1.0)
- Treats correlated scores as independent
- Produces invalid probability distributions

### Version 2 (score_adjustment_v2.py) - CORRECT ✅

**Solution:** Shifts entire probability vector as a **correlated unit**

```python
# RIGHT: Correlated shift
def shift_correlated_vector(
    donor: float, 
    acceptor: float, 
    neither: float,
    shift: int,
    all_scores: np.ndarray
) -> Tuple[float, float, float]:
    """
    Shift the ENTIRE probability tuple together.
    All three scores move as a unit.
    """
    if shift == 0:
        return (donor, acceptor, neither)
    
    shifted_idx = current_idx + shift
    if 0 <= shifted_idx < len(all_scores):
        # Get entire tuple from shifted position
        return all_scores[shifted_idx]  # (d, a, n)
    else:
        return (0.0, 0.0, 1.0)  # Edge case

# Result: donor + acceptor + neither = 1.0 (CORRECT!)
```

**Why correct:**
- Maintains probability constraint (sum = 1.0)
- Treats correlated scores correctly
- Produces valid probability distributions

### Consolidation (2025-10-31)

**Changes made:**
1. Renamed `score_adjustment.py` → `score_adjustment_v1_deprecated.py`
2. Renamed `score_adjustment_v2.py` → `score_adjustment.py` (canonical)
3. Updated all imports across codebase

**Current state:**
- ✅ Single canonical module: `score_adjustment.py`
- ✅ Implements correlated probability vectors
- ✅ All code uses correct version
- 📦 Old version kept for reference (deprecated)

---

## Q3: When should I use multi-view adjustment?

**Answer: Only when adjustments are non-zero AND different for donor/acceptor**

**Scenarios:**

### Scenario 1: All adjustments zero (SpliceAI → GTF)
```python
adjustment_dict = {
    'donor': {'plus': 0, 'minus': 0},
    'acceptor': {'plus': 0, 'minus': 0}
}
```
**Action:** Skip adjustment entirely, use base scores

### Scenario 2: Same non-zero adjustment for both types
```python
adjustment_dict = {
    'donor': {'plus': 2, 'minus': 2},
    'acceptor': {'plus': 2, 'minus': 2}
}
```
**Action:** Could use single view (all scores shift by same amount)

### Scenario 3: Different adjustments (e.g., OpenSpliceAI?)
```python
adjustment_dict = {
    'donor': {'plus': 1, 'minus': 1},
    'acceptor': {'plus': 0, 'minus': 0}
}
```
**Action:** **Use multi-view** (donor and acceptor need different shifts)

---

## Q4: How do I run empirical detection for a new base model?

**Step 1: Generate sample predictions**

```python
from meta_spliceai.splice_engine.run_spliceai_workflow import (
    predict_splice_sites_for_genes
)

# Generate predictions for sample genes
sample_genes = ['ENSG00000141510', 'ENSG00000157764', ...]  # 10-20 genes
sample_predictions = predict_splice_sites_for_genes(
    gene_ids=sample_genes,
    models=your_base_model,
    context=10000
)
```

**Step 2: Load annotations**

```python
import polars as pl

# Load GTF annotations
annotations_df = pl.read_csv(
    "data/ensembl/splice_sites_enhanced.tsv",
    separator="\t"
)

# Filter to sample genes
annotations_df = annotations_df.filter(
    pl.col('gene_id').is_in(sample_genes)
)
```

**Step 3: Run empirical detection**

```python
from meta_spliceai.splice_engine.meta_models.utils.infer_score_adjustments import (
    auto_detect_score_adjustments,
    save_adjustment_dict
)

# Detect optimal adjustments
adjustments = auto_detect_score_adjustments(
    annotations_df=annotations_df,
    pred_results=sample_predictions,
    use_empirical=True,
    search_range=(-5, 5),
    threshold=0.5,
    verbose=True
)

# Save for future use
save_adjustment_dict(
    adjustments,
    output_path=f"data/ensembl/{model_name}_adjustments.json"
)

print(f"Detected adjustments: {adjustments}")
```

**Step 4: Use detected adjustments**

```python
# In your workflow
from meta_spliceai.splice_engine.meta_models.utils.infer_score_adjustments import (
    load_adjustment_dict
)

adjustment_dict = load_adjustment_dict(
    f"data/ensembl/{model_name}_adjustments.json"
)

# Apply to predictions
if not all_adjustments_are_zero(adjustment_dict):
    predictions_df = adjust_predictions_dataframe_v2(
        predictions_df,
        adjustment_dict,
        method='multi_view'
    )
```

---

## Q5: Why are adjustments workflow-specific?

**Answer: Different base models use different training data and splice site definitions**

**Factors affecting adjustments:**

1. **Training data source**
   - Different genome builds (GRCh37 vs GRCh38)
   - Different annotation databases (GENCODE vs Ensembl)
   - Different splice site definitions

2. **Model architecture**
   - How the model defines splice sites
   - Position encoding conventions
   - Output coordinate system

3. **Our annotation source**
   - We use Ensembl GTF (GRCh38)
   - Specific splice site definition (within introns)
   - Coordinate system

**Examples:**

| Base Model | Training Data | Our Data | Adjustment Needed? |
|------------|--------------|----------|-------------------|
| SpliceAI | GENCODE (GRCh37) | Ensembl GTF (GRCh38) | ❌ No (empirically confirmed) |
| OpenSpliceAI | Unknown | Ensembl GTF (GRCh38) | ✅ Likely (+1 for donors?) |
| Custom model | Your data | Ensembl GTF (GRCh38) | 🔄 Run detection |

**Recommendation:** Always run empirical detection for each base model to ensure optimal alignment.

---

## Summary

1. **Views only needed when adjustments ≠ 0** → For SpliceAI, no views needed
2. **Modules consolidated** → Use `score_adjustment.py` (v1 deprecated)
3. **Multi-view for different adjustments** → Donor and acceptor need different shifts
4. **Empirical detection is easy** → Run on 10-20 sample genes
5. **Adjustments are workflow-specific** → Different models need different adjustments

The correlated probability vector paradigm ensures:
- ✅ 100% coverage (no position collisions)
- ✅ Valid probabilities (sum = 1.0)
- ✅ Correct alignment with annotations
- ✅ Model-agnostic (works with any base model)

