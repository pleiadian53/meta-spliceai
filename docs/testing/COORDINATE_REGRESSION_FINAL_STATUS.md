# Coordinate System Regression: Final Status & Recommendations

**Date**: 2025-10-29  
**Status**: ✅ Root Cause Identified | ⚠️ Partial Fix Applied | 📋 Action Plan Defined

---

## Executive Summary

### What Was Discovered
A critical **coordinate system regression** was found in the new inference workflow:
- The training workflow (`splice_prediction_workflow.py`) has a complete coordinate adjustment system
- The inference workflow (`enhanced_selective_inference.py`) bypassed this system entirely
- This caused ALL predictions to be misaligned by +1 from annotations (F1=0.000)

### Current Status
- ✅ **Donors Fixed**: F1 scores improved from 0.000 to 0.74-0.94 with -1 correction
- ❌ **Acceptors Broken**: Still F1=0.000 (need type & strand-specific handling)
- ✅ **Infrastructure Identified**: Existing adjustment system documented and located
- ✅ **Root Cause Understood**: Coordinate conventions differ between base models and GTF annotations

---

## Test Results

### Before Any Fix
| Gene  | Donor F1 | Acceptor F1 | Overall F1 |
|-------|----------|-------------|------------|
| GSTM3 | 0.000    | 0.000       | 0.000      |
| BRAF  | 0.000    | 0.000       | 0.000      |
| TP53  | 0.000    | 0.000       | 0.000      |

**Cause**: All predictions were +1 from annotations due to missing coordinate adjustments.

### After Partial Fix (Type-Aware -1 Correction)
| Gene  | Donor F1 | Acceptor F1 | Overall F1 |
|-------|----------|-------------|------------|
| GSTM3 | **0.941**| 0.000       | 0.485      |
| BRAF  | **0.744**| 0.000       | 0.364      |
| TP53  | **0.800**| 0.000       | 0.357      |

**Progress**: Donors excellent! Acceptors need full adjustment system integration.

---

## Root Cause Analysis

### Why This is a Regression

The project **already has** a complete coordinate adjustment system:

1. **Infrastructure** (`meta_spliceai/splice_engine/meta_models/utils/`):
   - `infer_splice_site_adjustments.py` - Core adjustment logic
   - `splice_utils.py` - `prepare_splice_site_adjustments()` function
   - `verify_splice_adjustment.py` - Validation utilities

2. **Documentation** (`meta_spliceai/splice_engine/meta_models/openspliceai_adapter/docs/`):
   - `SPLICE_SITE_DEFINITION_ANALYSIS.md` - Complete analysis
   - `coordinate_reconciliation.py` - Reconciliation system

3. **Production Usage** (`splice_prediction_workflow.py`):
   ```python
   # Lines 343-357: Prepare adjustments
   adjustment_result = prepare_splice_site_adjustments(
       local_dir=local_dir,
       ss_annotations_df=ss_annotations_df,
       sample_predictions=sample_predictions,
       use_empirical=config.use_auto_position_adjustments,
       verbosity=verbosity
   )
   
   # Lines 474-475: Apply adjustments
   enhanced_process_predictions_with_all_scores(
       ...
       predicted_delta_correction=True,
       splice_site_adjustments=adjustment_dict,
       ...
   )
   ```

### What Went Wrong

The new inference workflow (`enhanced_selective_inference.py`) was developed in isolation and:
1. ❌ Didn't import `prepare_splice_site_adjustments()`
2. ❌ Didn't call adjustment preparation
3. ❌ Didn't pass adjustments to prediction processing
4. ❌ Called SpliceAI directly without coordinate correction

**Result**: Regression of already-solved problem.

---

## Why Coordinate Adjustments Are Critical

### Base Model Agnostic Design

The adjustment system was designed to handle **arbitrary base models**:
- **SpliceAI**: Has known systematic offsets (donors +2/+1, acceptors 0/-1)
- **OpenSpliceAI**: Has different offsets (+1 for donors, 0 for acceptors)
- **Future Models**: May have yet other conventions

### The Problem Without Adjustments

Different systems define splice sites differently:

**GTF Annotations** (our ground truth):
- Donor: Last nucleotide of exon (T in GT dinucleotide)
- Acceptor: First nucleotide of exon (G in AG dinucleotide)
- Coordinate system: 1-based, fully-closed intervals

**SpliceAI Training Data**:
- May use different biological definitions
- May use different coordinate conventions
- Systematic offsets observed: +2/+1 for donors, 0/-1 for acceptors

**Without adjustments**:
- Model finds correct biological sites
- But coordinates are off by 1-2 nucleotides
- Appears as if model "doesn't work" (F1=0.000)
- Actually just coordinate system mismatch

---

## Known Coordinate Offsets

From `infer_splice_site_adjustments.py`:

```python
spliceai_pattern = {
    'donor': {
        'plus': 2,    # SpliceAI predicts 2nt upstream on + strand
        'minus': 1    # SpliceAI predicts 1nt upstream on - strand
    },
    'acceptor': {
        'plus': 0,    # SpliceAI matches GTF position on + strand
        'minus': -1   # SpliceAI predicts 1nt downstream on - strand
    }
}
```

**Interpretation**:
- To align SpliceAI with GTF annotations, **shift predictions** by these amounts
- Positive values = shift right (downstream), Negative = shift left (upstream)
- Different for each splice type AND strand

---

## Current Implementation Status

### What's Been Added

1. **Import** (`enhanced_selective_inference.py`, line 52-54):
   ```python
   from meta_spliceai.splice_engine.meta_models.utils.splice_utils import (
       prepare_splice_site_adjustments
   )
   ```

2. **Initialization** (line 209-212):
   ```python
   # Initialize coordinate adjustment system
   self.adjustment_dict = None  # Will be loaded/detected
   ```

3. **Application Method** (lines 539-589):
   ```python
   def _apply_coordinate_adjustments(self, predictions_df):
       # Type-aware correction
       # Donors: -1, Acceptors: 0 (simplified)
   ```

### What's Still Needed

1. **Load/Detect Adjustments** in `run_incremental()`:
   ```python
   # Early in workflow, before processing genes:
   if self.adjustment_dict is None:
       adjustment_result = prepare_splice_site_adjustments(
           local_dir=self.output_manager.predictions_root,
           ss_annotations_df=self.splice_sites_df,
           sample_predictions=None,
           use_empirical=False,  # Use known SpliceAI pattern
           verbosity=self.config.verbose
       )
       self.adjustment_dict = adjustment_result.get('adjustment_dict')
   ```

2. **Apply Full Adjustment System**:
   - Current: Simplified type-aware correction (donors -1, acceptors 0)
   - Needed: Full strand & type-specific adjustments from `adjustment_dict`

3. **Integration with Evaluation**:
   - Consider using `enhanced_process_predictions_with_all_scores()` 
   - This function already has adjustment logic built-in
   - Would ensure consistency with training workflow

---

## Recommended Solution Path

### Option 1: Quick Fix (Current Approach)
**Status**: Partially implemented  
**Pros**: Simple, immediate improvement for donors  
**Cons**: Doesn't handle acceptors or strand-specific cases properly

**What's done**:
- ✅ Type-aware correction (donors get -1)
- ✅ Acceptors on + strand get 0 (no correction)

**What's missing**:
- ❌ Acceptor - strand handling
- ❌ Full strand-specific donor corrections (+2 vs +1)

### Option 2: Proper Integration (Recommended)
**Status**: Not yet implemented  
**Pros**: Uses existing, tested infrastructure  
**Cons**: More complex, requires understanding of evaluation function

**Steps**:
1. Call `prepare_splice_site_adjustments()` during workflow initialization
2. Store `adjustment_dict` in workflow instance
3. Modify `_apply_coordinate_adjustments()` to use full adjustment_dict:
   ```python
   def _apply_coordinate_adjustments(self, predictions_df):
       # Determine predicted type
       predictions_df = predictions_df.with_columns([
           pl.when(pl.col('donor_prob') > pl.col('acceptor_prob'))
             .then(pl.lit('donor'))
             .otherwise(pl.lit('acceptor'))
             .alias('predicted_type')
       ])
       
       # Apply strand & type-specific adjustments
       for splice_type in ['donor', 'acceptor']:
           for strand in ['+', '-']:
               strand_key = 'plus' if strand == '+' else 'minus'
               adjustment = self.adjustment_dict[splice_type][strand_key]
               
               mask = (
                   (pl.col('predicted_type') == splice_type) &
                   (pl.col('strand') == strand)
               )
               predictions_df = predictions_df.with_columns([
                   pl.when(mask)
                     .then(pl.col('position') - adjustment)  # Apply correction
                     .otherwise(pl.col('position'))
                     .alias('position')
               ])
       
       return predictions_df.drop('predicted_type')
   ```

### Option 3: Score-Level Adjustment (Most Correct)
**Status**: Not implemented  
**Pros**: Matches training workflow exactly, adjusts at score level  
**Cons**: Most complex, requires deeper integration

**Approach**:
- Apply adjustments to **score arrays** before position extraction
- Use `apply_auto_detected_adjustments()` function
- This is what training workflow does

---

## Action Items

### Immediate (for testing):
1. ✅ **DONE**: Apply type-aware -1 correction (donors working)
2. ⚠️ **IN PROGRESS**: Debug acceptor issue (why still F1=0.000?)
3. 📋 **TODO**: Add strand-specific acceptor handling

### Short-term (for production):
4. 📋 **TODO**: Integrate `prepare_splice_site_adjustments()` properly
5. 📋 **TODO**: Use full `adjustment_dict` with all type & strand combinations
6. 📋 **TODO**: Test with OpenSpliceAI to verify base-model agnostic design

### Long-term (for maintainability):
7. 📋 **TODO**: Create unified coordinate system documentation
8. 📋 **TODO**: Add coordinate system tests to prevent future regressions
9. 📋 **TODO**: Ensure all workflows (training, inference, evaluation) use same adjustment system

---

## Key Insights

1. **This was already solved**: Training workflow has complete adjustment system
2. **Regression from isolation**: New code developed without checking existing infrastructure
3. **Base model agnostic**: System designed for ANY base model, not just SpliceAI
4. **Type & strand-specific**: One-size-fits-all corrections don't work
5. **Score vs Position**: Adjustments should ideally be at score level, not position level

---

## Related Files

**Core Adjustment System**:
- `meta_spliceai/splice_engine/meta_models/utils/infer_splice_site_adjustments.py` (504 lines)
- `meta_spliceai/splice_engine/meta_models/utils/splice_utils.py` (`prepare_splice_site_adjustments`)
- `meta_spliceai/splice_engine/meta_models/utils/verify_splice_adjustment.py`

**Documentation**:
- `meta_spliceai/splice_engine/meta_models/openspliceai_adapter/SPLICE_SITE_DEFINITION_ANALYSIS.md`
- `meta_spliceai/splice_engine/meta_models/openspliceai_adapter/docs/SPLICE_SITE_DEFINITION_ANALYSIS.md`
- `meta_spliceai/splice_engine/meta_models/openspliceai_adapter/coordinate_reconciliation.py`

**Workflows**:
- `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py` (uses adjustments correctly)
- `meta_spliceai/splice_engine/meta_models/workflows/inference/enhanced_selective_inference.py` (partial fix applied)

**Test Scripts**:
- `scripts/testing/test_base_only_protein_coding.py` (current test)
- `scripts/testing/test_all_modes_comprehensive_v2.py` (comprehensive test)

---

## Next Steps

**For User**:
You've correctly identified this as a regression of an already-solved problem. The question now is:

1. **Continue with simplified fix** (type-aware -1 correction) and debug acceptors?
2. **Integrate full adjustment system** properly using `prepare_splice_site_adjustments()`?
3. **Investigate acceptor issue** first to understand why they're still F1=0.000?

**My Recommendation**:
Option 2 - Integrate the full adjustment system properly. This:
- Uses existing, tested infrastructure
- Handles all splice types and strands correctly
- Works with any base model (SpliceAI, OpenSpliceAI, future models)
- Prevents future regressions
- Takes ~30 minutes to implement properly

Would you like me to proceed with the full integration?

---

**Status**: Ready for next phase - full adjustment system integration or acceptor debugging.

