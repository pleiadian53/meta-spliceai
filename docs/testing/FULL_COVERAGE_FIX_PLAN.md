# Full Coverage Fix Plan

**Date:** 2025-10-29  
**Priority:** 🔴 **CRITICAL**

---

## Problem Statement

The inference workflow is loading **filtered** predictions (only FP/FN positions) instead of **complete** predictions (all positions). This causes:

1. Base-only: 845/7107 positions (88% missing)
2. Hybrid: 840/7107 positions (88% missing)  
3. Meta-only: 4,200/7107 positions (59% too many - bug in duplication)

**Root Cause**: The workflow loads from `analysis_sequences_*.tsv` files which are **filtered** outputs designed for training data generation, not inference.

---

## Solution Overview

### Key Insight

SpliceAI's `predict_splice_sites_for_genes()` already returns predictions for **ALL positions** (N × 3 matrix where N = gene length). The problem is:

1. ✅ **Base model pass generates complete predictions** 
2. ❌ **But then filters them** to only FP/FN positions for training
3. ❌ **Inference workflow loads the filtered files** instead of complete predictions

### The Fix

**Save and load RAW SpliceAI predictions separately for inference use.**

---

## Implementation Plan

### Step 1: Modify Base Model Pass to Save Complete Predictions

**File**: `meta_spliceai/splice_engine/run_spliceai_workflow.py` or the prediction workflow

**Change**: After `predict_splice_sites_for_genes()` returns complete predictions, save them BEFORE filtering:

```python
# BEFORE filtering for training
complete_predictions_file = f"complete_predictions_{gene_id}.parquet"
complete_predictions_path = output_dir / complete_predictions_file
predictions_df.write_parquet(complete_predictions_path)

# THEN do filtering for training data
filtered_df = predictions_df.filter(...)  # existing filtering logic
```

**Output**: `complete_predictions_{gene_id}.parquet` with ALL positions

### Step 2: Modify Inference Workflow to Load Complete Predictions

**File**: `enhanced_selective_inference.py`

**Method**: `_generate_complete_base_model_predictions()`

**Change**: Look for `complete_predictions_*.parquet` files instead of `analysis_sequences_*.tsv`:

```python
# NEW: Look for complete predictions first
search_patterns = [
    "**/complete_predictions_*.parquet",  # NEW - complete predictions
    "**/complete_predictions_*.tsv",       # NEW - fallback format
]

# If not found, fall back to running SpliceAI directly
if not prediction_files:
    self.logger.info("  No cached predictions found - running SpliceAI...")
    # Run SpliceAI and save complete predictions
    result = self._run_spliceai_for_gene(gene_id, gene_info)
```

### Step 3: Ensure All Modes Produce Same Output Dimension

**All three modes must**:
1. Start with complete base predictions (N positions)
2. Generate metadata for all positions
3. Apply mode-specific logic:
   - **base_only**: `donor_meta = donor_score` (no recalibration)
   - **hybrid**: Recalibrate only uncertain positions
   - **meta_only**: Recalibrate ALL positions
4. Output N rows with identical schema

**Verification**:
```python
assert len(output_df) == gene_length, f"Expected {gene_length} positions, got {len(output_df)}"
```

---

## Detailed Changes

### Change 1: Save Complete Predictions in Base Model Pass

**Location**: Where SpliceAI predictions are generated

```python
def _save_complete_and_filtered_predictions(predictions_df, gene_id, output_dir):
    """
    Save both complete (for inference) and filtered (for training) predictions.
    """
    # Save complete predictions for inference
    complete_file = output_dir / f"complete_predictions_{gene_id}.parquet"
    predictions_df.write_parquet(complete_file, compression='zstd')
    logger.info(f"  Saved complete predictions: {complete_file}")
    
    # Filter for training (existing logic)
    filtered_df = predictions_df.filter(
        (pl.col('is_splice_site') == True) |  # True positives
        (pl.col('donor_score') > threshold) |  # False positives
        (pl.col('acceptor_score') > threshold)
    )
    
    # Save filtered predictions for training
    filtered_file = output_dir / f"analysis_sequences_{gene_id}.tsv"
    filtered_df.write_csv(filtered_file, separator='\t')
    logger.info(f"  Saved filtered predictions: {filtered_file}")
    
    return complete_file, filtered_file
```

### Change 2: Load Complete Predictions in Inference

**Location**: `enhanced_selective_inference.py` line ~374

```python
def _load_complete_predictions(self, gene_id: str, complete_output_dir: Path) -> pl.DataFrame:
    """
    Load complete base model predictions for ALL positions.
    
    Priority:
    1. Cached complete_predictions_*.parquet (fastest)
    2. Run SpliceAI directly (if no cache)
    """
    # Look for cached complete predictions
    complete_pred_files = list(complete_output_dir.glob(f"complete_predictions_{gene_id}.parquet"))
    
    if complete_pred_files:
        self.logger.info(f"  Loading cached complete predictions...")
        pred_df = pl.read_parquet(complete_pred_files[0])
        self.logger.info(f"  ✅ Loaded {pred_df.height:,} positions from cache")
        return pred_df
    
    # No cache - run SpliceAI
    self.logger.info(f"  No cache found - running SpliceAI...")
    return self._run_spliceai_and_save(gene_id, gene_info, complete_output_dir)
```

### Change 3: Verify Output Dimensions

**Location**: End of `run_incremental()` method

```python
# CRITICAL: Verify complete coverage
gene_length = gene_info[gene_id]['length']
if gene_final_df.height != gene_length:
    self.logger.error(
        f"  ❌ COVERAGE ERROR: Expected {gene_length} positions, got {gene_final_df.height}"
    )
    raise ValueError(f"Incomplete coverage for {gene_id}")

self.logger.info(f"  ✅ Complete coverage verified: {gene_length} positions")
```

---

## Testing Strategy

### Test 1: Position Count Verification

```python
gene_length = 7107  # GSTM3
for mode in ['base_only', 'hybrid', 'meta_only']:
    df = run_inference(gene_id, mode)
    assert len(df) == gene_length, f"{mode}: Expected {gene_length}, got {len(df)}"
```

### Test 2: Score Consistency

```python
base_df = run_inference(gene_id, 'base_only')
hybrid_df = run_inference(gene_id, 'hybrid')
meta_df = run_inference(gene_id, 'meta_only')

# All should have same positions
assert base_df['position'].to_list() == hybrid_df['position'].to_list()
assert base_df['position'].to_list() == meta_df['position'].to_list()

# Base scores should be identical
assert np.allclose(base_df['donor_score'], hybrid_df['donor_score'])
assert np.allclose(base_df['donor_score'], meta_df['donor_score'])

# Meta scores should differ
assert not np.allclose(base_df['donor_meta'], meta_df['donor_meta'])
```

### Test 3: Mode-Specific Behavior

```python
# Base-only: no recalibration
assert np.allclose(base_df['donor_meta'], base_df['donor_score'])

# Hybrid: partial recalibration
recal_pct = (hybrid_df['is_adjusted'] == 1).sum() / len(hybrid_df)
assert 0 < recal_pct < 1, f"Hybrid should recalibrate some positions, got {recal_pct:.1%}"

# Meta-only: full recalibration
recal_pct = (meta_df['is_adjusted'] == 1).sum() / len(meta_df)
assert recal_pct == 1.0, f"Meta-only should recalibrate all positions, got {recal_pct:.1%}"
```

---

## Timeline

1. **Step 1** (Modify base model pass): 30 min
2. **Step 2** (Modify inference workflow): 45 min
3. **Step 3** (Add verification): 15 min
4. **Testing**: 30 min

**Total**: ~2 hours

---

## Success Criteria

✅ All three modes produce exactly `gene_length` positions  
✅ All modes have identical position lists  
✅ Base scores are identical across modes  
✅ Meta scores differ appropriately  
✅ Metadata features (9/9) preserved in all modes  
✅ F1 scores > 0 (meaningful performance metrics)

---

## Next Steps After Fix

1. Re-run comprehensive test on diverse genes
2. Compare performance across modes
3. Validate meta-model improvements
4. Proceed with variant analysis integration

---

## References

- **Bug report**: `docs/testing/CRITICAL_BUG_FOUND.md`
- **Session summary**: `docs/session_summaries/SESSION_SUMMARY_2025-10-29.md`
- **Inference workflow**: `meta_spliceai/splice_engine/meta_models/workflows/inference/enhanced_selective_inference.py`
- **Base model pass**: `meta_spliceai/splice_engine/run_spliceai_workflow.py`

