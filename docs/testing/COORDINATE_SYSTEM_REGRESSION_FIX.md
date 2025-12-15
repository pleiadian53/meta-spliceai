# Coordinate System Regression Fix

**Date**: 2025-10-29  
**Issue**: Off-by-one coordinate error in inference workflow  
**Status**: ❌ REGRESSION - Previously solved, now broken again

---

## The Problem

### Observed Issue
All predictions are **+1** from annotations:
```
Predicted: [109737457, 109737656, 109738091, ...]
Annotated: [109737456, 109737655, 109738090, ...]
Offset:    +1         +1         +1         ...
```

### Why This is a Regression

This problem was **already solved** in the training workflow (`splice_prediction_workflow.py`):

1. **Infrastructure exists**: `SpliceCoordinateReconciler`, `prepare_splice_site_adjustments()`
2. **Adjustments documented**: `infer_splice_site_adjustments.py` (lines 36-39)
3. **System in production**: Training workflow uses `splice_site_adjustments=adjustment_dict`

### What Went Wrong

The **new inference workflow** (`enhanced_selective_inference.py`) bypassed this system:
- ❌ Doesn't call `prepare_splice_site_adjustments()`
- ❌ Doesn't use `adjustment_dict`
- ❌ Doesn't pass `splice_site_adjustments` to prediction processing
- ❌ Calls SpliceAI directly without coordinate correction

---

## Known SpliceAI Coordinate Offsets

From `infer_splice_site_adjustments.py` (lines 36-39, 441-444):

```python
# Standard SpliceAI pattern (what the model learned vs GTF positions)
spliceai_pattern = {
    'donor': {
        'plus': 2,   # SpliceAI predicts 2nt upstream on + strand
        'minus': 1   # SpliceAI predicts 1nt upstream on - strand
    },
    'acceptor': {
        'plus': 0,   # SpliceAI matches GTF position on + strand
        'minus': -1  # SpliceAI predicts 1nt downstream on - strand
    }
}
```

**Biological context**:
- **Donor site**: Last nucleotide of exon (T in GT dinucleotide)
- **Acceptor site**: First nucleotide of exon (G in AG dinucleotide)

**SpliceAI's behavior**:
- Predicts donor sites **upstream** of true position
- Predicts acceptor sites **at or downstream** of true position

---

## Current Observations

### Test Data (GSTM3, chr1, + strand)
```
Donor sites observed:
  Predicted: 109737457, 109737656, 109738091, 109738285, 109739429, 109739833, 109740240
  Annotated: 109737456, 109737655, 109738090, 109738284, 109739428, 109739832, 109740239
  Offset:    +1         +1         +1         +1         +1         +1         +1
```

### Analysis

**Expected offset for donors on + strand**: +2 (from SpliceAI pattern)  
**Observed offset**: +1  
**Discrepancy**: 1 nucleotide less than expected

**Possible explanations**:
1. **Different SpliceAI version**: Model might have changed behavior
2. **Position extraction issue**: How we extract positions from SpliceAI dict
3. **Annotation convention**: Our `splice_sites_enhanced.tsv` might use different convention
4. **Gene-relative vs genomic**: Coordinate system confusion

---

## Investigation Required

### 1. Check SpliceAI Output Convention

**Question**: What coordinate system does `predict_splice_sites_for_genes()` use?

```python
# In _run_spliceai_directly():
gene_preds = predictions_dict[gene_id]
positions = gene_preds['positions']  # What are these?
```

**Need to verify**:
- Are these 0-based or 1-based?
- Are these gene-relative or genomic?
- Are these already adjusted or raw?

### 2. Check Annotation Convention

**Question**: What coordinate system does `splice_sites_enhanced.tsv` use?

From extraction code, splice sites are derived from GTF exon boundaries:
- GTF uses **1-based, fully-closed** intervals
- Donor: `exon.end` position (last nucleotide of exon)
- Acceptor: Next `exon.start` position (first nucleotide of next exon)

**Need to verify**:
- Does extraction code apply any offsets?
- Is there a +1 or -1 adjustment during extraction?

### 3. Compare with Training Workflow

**Training workflow** applies adjustments in two places:

**A. During prediction evaluation** (`enhanced_process_predictions_with_all_scores`):
```python
# splice_prediction_workflow.py, line 474-475
predicted_delta_correction=True,  # Enable adjustments
splice_site_adjustments=adjustment_dict,  # Pass detected adjustments
```

**B. In score adjustment** (`adjust_scores` function):
```python
# infer_splice_site_adjustments.py, lines 41-82
if splice_type == 'donor':
    if strand == '+':
        adjusted_scores = np.roll(scores, 2)  # Shift by +2
    else:  # '-'
        adjusted_scores = np.roll(scores, 1)  # Shift by +1
```

---

## Solution Strategy

### Option 1: Apply -1 Correction (Quick Fix)

**Where**: In `_run_spliceai_directly()` after extracting positions

```python
predictions_df = pl.DataFrame({
    'position': [p - 1 for p in gene_preds['positions']],  # Subtract 1
    ...
})
```

**Pros**: Simple, immediate fix  
**Cons**: Doesn't address root cause, hardcoded value

### Option 2: Integrate Adjustment System (Proper Fix)

**Where**: In `enhanced_selective_inference.py` initialization

```python
# In __init__() or run_incremental():
from meta_spliceai.splice_engine.meta_models.utils.splice_utils import (
    prepare_splice_site_adjustments
)

# Load or detect adjustments
adjustment_result = prepare_splice_site_adjustments(
    local_dir=self.output_manager.predictions_root,
    ss_annotations_df=self.splice_sites_df,
    sample_predictions=None,  # Or provide sample
    use_empirical=False,  # Use standard SpliceAI pattern
    verbosity=1
)

self.adjustment_dict = adjustment_result.get('adjustment_dict')
```

**Then apply during prediction**:
- Either adjust positions after SpliceAI
- Or apply during score array processing

**Pros**: Consistent with training workflow, uses existing infrastructure  
**Cons**: More complex integration

### Option 3: Investigate and Fix Root Cause

**Steps**:
1. Trace coordinate convention through entire pipeline
2. Identify where +1 discrepancy originates
3. Fix at source (either in position extraction or annotation extraction)
4. Ensure consistency across training and inference

**Pros**: True fix, prevents future issues  
**Cons**: Time-consuming, might affect other code

---

## Recommended Approach

### Immediate (for testing):
**Apply -1 correction** in `_run_spliceai_directly()` with clear documentation

### Short-term (for production):
**Integrate adjustment system** from training workflow

### Long-term (for maintenance):
**Root cause analysis** and unified coordinate system

---

## Related Files

**Coordinate adjustment infrastructure**:
- `meta_spliceai/splice_engine/meta_models/utils/infer_splice_site_adjustments.py`
- `meta_spliceai/splice_engine/meta_models/utils/splice_utils.py` (prepare_splice_site_adjustments)
- `meta_spliceai/splice_engine/meta_models/openspliceai_adapter/coordinate_reconciliation.py`

**Documentation**:
- `meta_spliceai/splice_engine/meta_models/openspliceai_adapter/SPLICE_SITE_DEFINITION_ANALYSIS.md`
- `meta_spliceai/splice_engine/meta_models/openspliceai_adapter/docs/SPLICE_SITE_DEFINITION_ANALYSIS.md`

**Training workflow**:
- `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py` (lines 343-357, 474-475)

**Inference workflow** (needs fix):
- `meta_spliceai/splice_engine/meta_models/workflows/inference/enhanced_selective_inference.py`
- Method: `_run_spliceai_directly()` (lines ~590-650)

---

## Testing Plan

After applying fix:

1. **Unit test**: Verify positions match annotations
2. **F1 score test**: Expect >0.90 for protein-coding genes
3. **Coverage test**: Verify complete coverage maintained
4. **Cross-validation**: Compare with training workflow outputs

---

**Next Action**: Apply immediate -1 correction and test, then integrate proper adjustment system.

