# TN Sampling Fix - Implementation Complete

## Summary

Fixed the True Negative (TN) sampling bug that was causing `full_splice_positions_enhanced.tsv` to store ALL 4 million nucleotide positions instead of a sampled subset.

## Changes Made

### File Modified
`meta_spliceai/splice_engine/meta_models/core/enhanced_evaluation.py`

### Key Changes

#### 1. Removed Module-Level TN Collection (Lines 181, 887)
**Before:**
```python
tn_collection = []  # Module level - caused confusion
```

**After:**
```python
# Removed - now initialized per-gene
```

#### 2. Fixed Donor Site TN Sampling (Lines 522-615)
**Before (Data Leakage):**
```python
# Line 581: Used TRUE positions - data leakage!
true_donor_positions_for_tn = [p['true_position'] for p in tp_positions ...]

for true_pos in true_donor_positions_for_tn:
    window_start = max(0, true_pos - left_window)
    window_end = min(gene_len, true_pos + right_window + 1)
    # Collect TNs in windows around TRUE splice sites
```

**After (No Leakage):**
```python
# Line 575: Use PREDICTED positions - no leakage!
predicted_donor_positions = [p['predicted_position'] for p in (tp_positions + fp_positions) 
                            if p.get('predicted_position') is not None]

# Use tn_proximity_radius (50) instead of error_window (500)
window_size = tn_proximity_radius

for pred_pos in predicted_donor_positions:
    window_start = max(0, pred_pos - window_size)
    window_end = min(gene_len, pred_pos + window_size + 1)
    # Collect TNs in windows around PREDICTED splice sites
```

#### 3. Fixed Acceptor Site TN Sampling (Lines 1173-1262)
Same fix applied to acceptor evaluation function.

### Three Sampling Modes

#### Mode 1: Random (Default fallback)
```python
if tn_sampling_mode == "random":
    tn_collection = random.sample(tn_positions_all, num_tn_to_sample)
```
- Samples TNs uniformly across the gene
- Simple and unbiased
- Good baseline

#### Mode 2: Proximity
```python
elif tn_sampling_mode == "proximity":
    predicted_positions = [p['predicted_position'] for p in (tp_positions + fp_positions) ...]
    # Sort TNs by distance to nearest prediction
    # Take closest num_tn_to_sample TNs
```
- Prefers TNs near predicted splice sites
- Uses predicted positions (TP + FP) - no data leakage
- Good for learning decision boundaries

#### Mode 3: Window (Recommended)
```python
elif tn_sampling_mode == "window":
    predicted_positions = [p['predicted_position'] for p in (tp_positions + fp_positions) ...]
    window_size = tn_proximity_radius  # Typically 50
    
    # Collect ALL TNs within window_size of any prediction
    # Sample if total exceeds num_tn_to_sample
```
- Collects TNs adjacent to predicted splice sites
- Window size = `tn_proximity_radius` (50) not `error_window` (500)
- Best for contextual learning
- **This is what we're using**

## Expected Results

### Before Fix
```
File: full_splice_positions_enhanced.tsv
Size: 3.4 GB
Rows: 4,017,636

Breakdown:
- TN: 4,017,063 (99.99%)
- TP: 532 (0.01%)
- FP: 10
- FN: 30
```

### After Fix (Expected)
```
File: full_splice_positions_enhanced.tsv
Size: ~10-20 MB (170x smaller!)
Rows: ~1,200-2,000

Breakdown:
- TN: ~600-1,400 (sampled, ~1.2x TP count)
- TP: 532
- FP: 10
- FN: 30
```

### TN Distribution (Expected)
- **Near predictions** (within 50 nt): ~80-90% of sampled TNs
- **Far from predictions**: ~10-20% of sampled TNs
- **Sampling factor**: 1.2x (configurable via `tn_sample_factor`)

## Configuration Parameters

### Current Defaults
```python
collect_tn = True  # Collect TNs
no_tn_sampling = False  # Apply sampling
tn_sample_factor = 1.2  # Collect 1.2x as many TNs as (TP+FP+FN)
tn_sampling_mode = "random"  # Default mode
tn_proximity_radius = 50  # Window size for "window" and "proximity" modes
```

### To Change Sampling Strategy
```python
# In run_base_model_predictions or workflow config:
results = run_base_model_predictions(
    base_model="openspliceai",
    target_chromosomes=["21"],
    config=config,
    no_tn_sampling=False,  # Enable sampling
    # These are passed through kwargs to enhanced_process_predictions_with_all_scores:
    tn_sample_factor=1.5,  # Collect more TNs
    tn_sampling_mode="window",  # Use window-based sampling
    tn_proximity_radius=100,  # Larger windows
)
```

## Why This Fix is Important

### 1. No Data Leakage
**Before**: Used `true_position` to determine which TNs to collect
- Model learns which TNs are "near splice sites" based on ground truth
- Leaks information about true splice site locations

**After**: Uses `predicted_position` (TP + FP)
- Model only knows where IT predicted splice sites
- Doesn't know which predictions are correct (TP) vs incorrect (FP)
- No information leakage

### 2. Efficient Storage
**Before**: 3.4 GB per chromosome → 150-200 GB for full genome
**After**: 10-20 MB per chromosome → 500 MB - 2 GB for full genome

### 3. Better Training Data
- TNs near predictions are more informative (decision boundary)
- Distant TNs are less informative (clearly not splice sites)
- Sampling focuses on informative examples

### 4. Scalable
- Full genome pass now feasible on laptop (16 GB RAM)
- Faster I/O and processing
- Reduced memory pressure

## Testing

### Test Command
```bash
cd /Users/pleiadian53/work/meta-spliceai

python scripts/training/run_full_genome_base_model_pass.py \
  --base-model openspliceai \
  --mode test \
  --coverage full_genome \
  --chromosomes 21 \
  --verbosity 2
```

### Monitor Progress
```bash
bash scripts/training/monitor_tn_fix_test.sh
```

### Verify Results
```bash
# Check file size
ls -lh data/mane/GRCh38/openspliceai_eval/meta_models/full_splice_positions_enhanced.tsv

# Check row count
wc -l data/mane/GRCh38/openspliceai_eval/meta_models/full_splice_positions_enhanced.tsv

# Check TN distribution
awk -F'\t' 'NR>1 {print $52}' data/mane/GRCh38/openspliceai_eval/meta_models/full_splice_positions_enhanced.tsv | sort | uniq -c
```

### Expected Output
```
Size: ~10-20 MB (not 3.4 GB)
Rows: ~1,200-2,000 (not 4M)

Distribution:
  ~600-1,400 TN
  532 TP
  10 FP
  30 FN
```

## Performance Impact

### Metrics Should Be Comparable
The fix should **NOT** significantly change performance metrics because:

1. **Base model is unchanged**: Same predictions, same TP/FP/FN
2. **TN sampling doesn't affect metrics**: Metrics calculated on TP/FP/FN
3. **Using predicted positions**: Even though we're using predictions as anchors, the base model was well-trained, so predicted positions are good proxies for true positions

### Expected Metrics (Should Match Previous Run)
```
F1 Score:          ~0.97
Precision:         ~0.99
Recall:            ~0.96
Accuracy:          ~0.98
ROC-AUC:           ~0.999
Average Precision: ~0.998
```

## Next Steps

1. ✅ **Implementation Complete**: Fix applied to both donor and acceptor evaluation
2. ⏳ **Testing**: Running chr21 test (in progress)
3. ⏳ **Verification**: Check file size and TN count
4. ⏳ **Performance Check**: Verify metrics are comparable
5. ⏳ **Documentation**: Update workflow docs
6. ⏳ **Full Genome**: Run full genome pass with fix

## Files Changed

- `meta_spliceai/splice_engine/meta_models/core/enhanced_evaluation.py`
  - Lines 176-178: Removed module-level `tn_collection`
  - Lines 522-615: Fixed donor TN sampling (use predicted positions)
  - Lines 1173-1262: Fixed acceptor TN sampling (use predicted positions)

## Related Documents

- `docs/training/TN_SAMPLING_ISSUE_ANALYSIS.md` - Original problem analysis
- `docs/training/TN_SAMPLING_FIX.md` - Detailed fix design
- `docs/training/TN_SAMPLING_FIX_IMPLEMENTATION.md` - This document

## Test Status

**Started**: 2025-11-14 12:59 PM
**Status**: Running (gene 71/214)
**Log**: `logs/openspliceai_chr21_tn_fix_test_20251114_125922.log`
**Expected Completion**: ~20-30 minutes

Monitor with:
```bash
bash scripts/training/monitor_tn_fix_test.sh
```


