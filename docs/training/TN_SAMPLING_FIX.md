# TN Sampling Bug Fix

## Issues Identified

### Issue 1: Variable Scope Bug
**Location**: Lines 181, 528, 887, 1194 in `enhanced_evaluation.py`

**Problem**:
```python
# Line 181/887: Initialize at module level
tn_collection = []

# Inside gene loop:
for gene_id, gene_data in pred_results.items():
    # ... process gene ...
    
    # Line 528/1194: Re-initialize (OVERWRITES previous genes!)
    tn_collection = []  
    
    # ... collect TNs for this gene ...
    
    # Line 655/1322: Add to positions_list
    if collect_tn and tn_collection:
        positions_list.extend(tn_collection)
```

**Impact**: The re-initialization inside the loop doesn't actually cause data loss (each gene's TNs are added to `positions_list` before the next iteration), but it's confusing and the logic should be clearer.

### Issue 2: Data Leakage (CRITICAL)
**Location**: Lines 574-628 (donor), 1239-1293 (acceptor)

**Problem**:
```python
# Line 581: Uses TRUE positions (ground truth labels!)
true_donor_positions_for_tn = [p['true_position'] for p in tp_positions ...]

# Line 595: Creates windows around TRUE splice sites
for true_pos in true_donor_positions_for_tn:
    window_start = max(0, true_pos - left_window)
    window_end = min(gene_len, true_pos + right_window + 1)
```

**Impact**: The model learns which TNs are "near splice sites" based on ground truth, not predictions. This leaks information about true splice site locations into the training data.

### Issue 3: All TNs Collected
**Root Cause**: The `no_tn_sampling` check happens INSIDE the `if collect_tn` block, but the default sampling logic doesn't actually limit TNs effectively.

**Current Flow**:
1. Collect ALL TN positions in `tn_positions_all` (lines 436-454, 1116-1134)
2. Check if `no_tn_sampling=False` (line 530, 1196)
3. Calculate `num_tn_to_sample` based on `tn_sample_factor` (line 539, 1210)
4. **BUT**: For "window" mode, it collects ALL TNs in windows, then samples if too many

**Why it fails**: With `error_window=500` and many genes, the windows cover most of the gene sequence, so almost all TNs are "in window" and get collected.

## Proposed Fix

### Strategy: Prediction-Based TN Sampling (No Data Leakage)

```python
def sample_tns_around_predictions(
    tn_positions_all: List[Dict],
    tp_positions: List[Dict],
    fp_positions: List[Dict],
    tn_sample_factor: float = 1.2,
    near_window: int = 50,  # Collect TNs within this distance
    far_sample_rate: float = 0.1,  # Sample 10% of distant TNs
    verbose: int = 1
) -> List[Dict]:
    """
    Sample TN positions based on PREDICTED splice sites to avoid data leakage.
    
    Strategy:
    1. Collect ALL TNs within `near_window` of any PREDICTED splice site (TP or FP)
    2. For TNs beyond `near_window`, subsample at `far_sample_rate`
    3. Ensure total TNs ≈ tn_sample_factor × (num_tp + num_fp + num_fn)
    
    This avoids data leakage because:
    - Uses predicted positions (TP + FP), not true positions
    - Model doesn't know which predictions are correct (TP) vs incorrect (FP)
    - Distant TNs are sampled uniformly, not based on proximity to true sites
    """
    if not tn_positions_all:
        return []
    
    # Get all PREDICTED splice site positions (TP + FP)
    predicted_positions = set()
    for pos in tp_positions + fp_positions:
        if pos.get('predicted_position') is not None:
            predicted_positions.add(pos['predicted_position'])
    
    if not predicted_positions:
        # No predictions - sample uniformly
        num_tp_fp_fn = len(tp_positions) + len(fp_positions)
        target_count = int(num_tp_fp_fn * tn_sample_factor)
        if len(tn_positions_all) <= target_count:
            return tn_positions_all
        return random.sample(tn_positions_all, target_count)
    
    # Categorize TNs as "near" or "far" from predictions
    near_tns = []
    far_tns = []
    
    for tn in tn_positions_all:
        tn_pos = tn['position']
        # Find minimum distance to any predicted splice site
        min_dist = min(abs(tn_pos - pred_pos) for pred_pos in predicted_positions)
        
        if min_dist <= near_window:
            near_tns.append(tn)
        else:
            far_tns.append(tn)
    
    # Collect all near TNs
    sampled_tns = near_tns.copy()
    
    # Sample far TNs
    num_far_to_sample = int(len(far_tns) * far_sample_rate)
    if num_far_to_sample > 0 and far_tns:
        sampled_far = random.sample(far_tns, min(num_far_to_sample, len(far_tns)))
        sampled_tns.extend(sampled_far)
    
    # Apply overall limit based on tn_sample_factor
    num_tp_fp_fn = len(tp_positions) + len(fp_positions)
    target_count = int(num_tp_fp_fn * tn_sample_factor)
    
    if len(sampled_tns) > target_count:
        # Too many TNs - prioritize near TNs, then sample from far
        if len(near_tns) >= target_count:
            # Even near TNs exceed target - sample from near
            sampled_tns = random.sample(near_tns, target_count)
        else:
            # Keep all near, sample remaining from far
            remaining = target_count - len(near_tns)
            sampled_far = random.sample(
                [tn for tn in sampled_tns if tn not in near_tns],
                min(remaining, len(sampled_tns) - len(near_tns))
            )
            sampled_tns = near_tns + sampled_far
    
    if verbose >= 1:
        print(f"[tn_sampling] Collected {len(sampled_tns)} TNs: "
              f"{len([tn for tn in sampled_tns if tn in near_tns])} near predictions, "
              f"{len([tn for tn in sampled_tns if tn not in near_tns])} far")
    
    return sampled_tns
```

### Implementation Changes

**File**: `meta_spliceai/splice_engine/meta_models/core/enhanced_evaluation.py`

**Change 1: Remove module-level tn_collection initialization**
```python
# Line 181, 887: REMOVE these lines
# tn_collection = []  # DELETE
```

**Change 2: Replace TN sampling logic in donor evaluation**
```python
# Replace lines 526-632 with:
if collect_tn and len(tn_positions_all) > 0:
    if no_tn_sampling:
        # No sampling mode - preserve all TN positions
        tn_collection = tn_positions_all
        if verbose >= 1:
            print(f"[info] No TN sampling: preserving all {len(tn_positions_all)} TN positions for gene {gene_id}")
    else:
        # Apply prediction-based sampling to avoid data leakage
        tn_collection = sample_tns_around_predictions(
            tn_positions_all=tn_positions_all,
            tp_positions=tp_positions,
            fp_positions=fp_positions,
            tn_sample_factor=tn_sample_factor,
            near_window=tn_proximity_radius,  # Use proximity_radius as near_window
            far_sample_rate=0.1,  # Sample 10% of distant TNs
            verbose=verbose
        )
else:
    tn_collection = []
```

**Change 3: Apply same fix to acceptor evaluation** (lines 1192-1298)

### Expected Results After Fix

**For chromosome 21**:
```
Before fix:
- Total positions: 4,017,636
- TN positions: 4,017,063 (99.99%)
- TP positions: 532
- File size: 3.4 GB

After fix:
- Total positions: ~1,200-2,000
- TN positions: ~600-1,400 (sampled)
- TP positions: 532
- FP positions: 10
- FN positions: 30
- File size: ~10-20 MB
```

**TN Distribution**:
- ~80-90% near predicted splice sites (within 50 nt)
- ~10-20% far from predictions (sampled at 10%)
- No data leakage (uses predicted positions only)

### Configuration Options

Add new parameters to control TN sampling:

```python
# In SpliceAIConfig
tn_near_window: int = 50  # Distance to consider "near" a prediction
tn_far_sample_rate: float = 0.1  # Sampling rate for distant TNs
tn_sample_factor: float = 1.2  # Overall TN/TP ratio
```

### Testing Plan

1. **Unit test**: Verify no data leakage
   ```python
   def test_no_data_leakage():
       # Ensure sampled TNs don't depend on true_position
       sampled = sample_tns_around_predictions(...)
       for tn in sampled:
           assert tn['true_position'] is None
   ```

2. **Integration test**: Run chr21 with fix
   ```bash
   python scripts/training/run_full_genome_base_model_pass.py \
     --base-model openspliceai \
     --mode test \
     --chromosomes 21 \
     --verbosity 2
   ```

3. **Verify output**:
   - File size: ~10-20 MB (not 3.4 GB)
   - TN count: ~600-1,400 (not 4M)
   - TN distribution: mostly near predictions

## Migration Path

1. **Immediate**: Fix the bug in `enhanced_evaluation.py`
2. **Test**: Run chr21 test to verify fix
3. **Document**: Update workflow docs with new TN sampling strategy
4. **Deploy**: Use for full genome pass

## Benefits

1. **No data leakage**: Uses only predicted positions
2. **Efficient storage**: 10-20 MB vs 3.4 GB per chromosome
3. **Better representation**: TNs near predictions are more informative
4. **Scalable**: Full genome pass becomes feasible on laptop
5. **Theoretically sound**: Model learns from its own predictions, not ground truth

