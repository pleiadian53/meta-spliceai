# True Negative (TN) Sampling Issue Analysis

## Issue Summary

The `full_splice_positions_enhanced.tsv` file is unexpectedly large (3.4 GB for chromosome 21 alone) because it's storing **ALL nucleotide positions** instead of just a sampled subset of True Negatives.

## Investigation Results

### File Size Analysis
- **File**: `data/mane/GRCh38/openspliceai_eval/meta_models/full_splice_positions_enhanced.tsv`
- **Size**: 3.4 GB
- **Rows**: 4,017,636 positions
- **Chromosome**: 21 only

### Position Type Breakdown
```
TN (True Negatives):  4,017,063 (99.99%)
TP (True Positives):        532 (0.01%)
FP (False Positives):        10 (0.00%)
FN (False Negatives):        30 (0.00%)
```

### Expected vs Actual TN Sampling

**Configuration**:
- TN Sampling: **Enabled** (not disabled)
- TN Sample Factor: **1.2x** (default)
- Expected TN count: ~638 (1.2 × 532 TPs)

**Actual**:
- TN count: **4,017,063**
- Ratio: **7,550x** more TNs than expected!

## Root Cause

The TN sampling logic in `enhanced_evaluate_splice_site_errors` is **not working correctly**. Despite:
1. `no_tn_sampling=False` (sampling should be active)
2. `tn_sample_factor=1.2` (should sample 1.2x the number of TPs)
3. `collect_tn=True` (collect TNs)

The function is collecting **ALL nucleotide positions** as TNs instead of sampling.

## Impact

### Disk Space
For the full genome (all chromosomes):
- Chromosome 21: 3.4 GB (214 genes)
- Estimated full genome: **~150-200 GB** (19,000+ genes)
- This is **unsustainable** for laptop storage

### Memory Usage
- Loading 4M+ rows per chromosome into memory
- Causes severe memory pressure during aggregation
- Led to the thrashing issue we experienced earlier

### Processing Time
- Slower I/O operations
- Longer aggregation times
- Increased risk of out-of-memory errors

## Why This Happens

The `full_splice_positions_enhanced.tsv` file is meant to store:
1. **All True Positives (TPs)**: Correctly predicted splice sites
2. **All False Positives (FPs)**: Incorrectly predicted splice sites
3. **All False Negatives (FNs)**: Missed splice sites
4. **Sampled True Negatives (TNs)**: A small representative sample of non-splice-site positions

However, the current implementation is storing **ALL positions** from the base model predictions, which includes every nucleotide in every gene.

## Solution Options

### Option 1: Fix TN Sampling Logic (Recommended)
**Action**: Debug and fix `enhanced_evaluate_splice_site_errors` to properly sample TNs

**Pros**:
- Maintains intended behavior
- Keeps file sizes manageable
- Preserves statistical representativeness

**Cons**:
- Requires code changes
- Need to rerun chromosome 21 test

**Implementation**:
```python
# In enhanced_evaluate_splice_site_errors
# Ensure TN sampling is actually applied when no_tn_sampling=False
if not no_tn_sampling and collect_tn:
    # Sample TNs based on tn_sample_factor
    target_tn_count = int(len(tp_positions) * tn_sample_factor)
    if len(tn_positions) > target_tn_count:
        tn_positions = sample_tn_positions(
            tn_positions, 
            target_count=target_tn_count,
            mode=tn_sampling_mode,
            proximity_radius=tn_proximity_radius
        )
```

### Option 2: Disable TN Collection Entirely
**Action**: Set `collect_tn=False` in the workflow

**Pros**:
- Immediate fix
- Minimal file size
- No code changes needed

**Cons**:
- Loses TN information entirely
- May impact meta-learning training quality
- Can't evaluate model behavior on non-splice-sites

### Option 3: Use --no-tn-sampling Flag
**Action**: Run with `--no-tn-sampling` to explicitly disable TN collection

**Pros**:
- Quick workaround
- No code changes
- Can toggle on/off as needed

**Cons**:
- Loses TN information
- Flag name is confusing (double negative)
- Not a permanent solution

### Option 4: Store TNs Separately
**Action**: Save TNs to a separate file (`full_splice_positions_tn.tsv`)

**Pros**:
- Keeps main file small
- Preserves TN data if needed
- Can load TNs selectively

**Cons**:
- Requires code changes
- More complex file management
- Still stores massive TN dataset

## Recommended Action Plan

### Immediate (For Current Run)
1. **Accept the current chr21 results** - The file is large but manageable for one chromosome
2. **Do NOT run full genome** with current code - Would generate 150-200 GB of data

### Short-term (Before Full Genome Pass)
1. **Debug TN sampling** in `enhanced_evaluate_splice_site_errors`
2. **Add logging** to verify TN sampling is working:
   ```python
   if verbose >= 1:
       print(f"[tn_sampling] Collected {len(all_tn_positions)} TNs")
       print(f"[tn_sampling] Target TN count: {target_tn_count}")
       print(f"[tn_sampling] Sampling mode: {tn_sampling_mode}")
       print(f"[tn_sampling] Final TN count: {len(sampled_tn_positions)}")
   ```
3. **Test on chr21** again to verify fix
4. **Verify file size** drops from 3.4 GB to ~10-50 MB

### Long-term (For Production)
1. **Add configuration option** for TN sampling strategy
2. **Document TN sampling** behavior clearly
3. **Add validation** to check TN ratio doesn't exceed threshold
4. **Consider adaptive sampling** based on gene length

## File Size Estimates (After Fix)

### Per Chromosome (with proper TN sampling)
```
Positions per chromosome:
- TPs: ~500-3,000 (depends on gene count)
- FPs: ~10-100
- FNs: ~30-200
- TNs: ~600-3,600 (1.2x TPs)

Total: ~1,200-7,000 positions per chromosome
File size: ~10-50 MB per chromosome (vs 3.4 GB currently)
```

### Full Genome (with proper TN sampling)
```
Total positions: ~50,000-150,000 (all chromosomes)
File size: ~500 MB - 2 GB (vs 150-200 GB currently)
```

## Questions for User

1. **Do you need TN positions at all?**
   - For meta-learning training, TNs help the model learn what's NOT a splice site
   - But you may not need ALL TNs, just a representative sample

2. **What's the intended use of `full_splice_positions_enhanced.tsv`?**
   - If it's just for evaluation metrics → Don't need TNs
   - If it's for meta-learning training → Need sampled TNs
   - If it's for analysis → Need sampled TNs

3. **Should we fix the sampling or disable TN collection?**
   - Fix sampling: More work, but preserves intended functionality
   - Disable TNs: Quick fix, but loses information

## Related Files

- **Workflow**: `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py`
- **Evaluation**: `meta_spliceai/splice_engine/meta_models/core/enhanced_evaluation.py`
- **Enhanced workflow**: `meta_spliceai/splice_engine/meta_models/core/enhanced_workflow.py`

## Next Steps

Please advise on preferred solution:
- [ ] Fix TN sampling logic (Option 1)
- [ ] Disable TN collection (Option 2)
- [ ] Use --no-tn-sampling flag (Option 3)
- [ ] Store TNs separately (Option 4)
- [ ] Other approach?

