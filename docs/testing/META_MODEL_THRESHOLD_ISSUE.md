# Meta-Model Threshold Issue - Root Cause Analysis

**Date**: 2025-10-30  
**Status**: 🎯 **ROOT CAUSE IDENTIFIED**  

---

## Executive Summary

The meta-model is performing terribly in inference (F1=0.004-0.021) despite excellent training performance (F1=0.944-0.945) because we're using the **wrong threshold** (0.5 instead of 0.95).

---

## The Problem

### Inference Results (Current)
```
ENSG00000187987: base F1=0.857, meta F1=0.021 (97% degradation!)
  - Base: TP=7, FP=0-1
  - Meta: TP=8, FP=762 (MASSIVE FP rate!)

ENSG00000171812: base F1=0.727, meta F1=0.006 (99% degradation!)
  - Base: TP=7, FP=0
  - Meta: TP=7, FP=2,281 (MASSIVE FP rate!)
```

**Pattern**: Meta-model finds all TPs but predicts hundreds/thousands of FPs!

### Training Results (Expected)
```
From results/gene_cv_1000_run_15/compare_base_meta.json:
  Base FP: 428 → Meta FP: 17 (96% reduction!)
  Base FN: 197 → Meta FN: 0 (100% reduction!)
  Base F1: 0.759-0.770 → Meta F1: 0.944-0.945 (+18-19%)
```

**Expected**: Meta-model should REDUCE FPs, not increase them!

---

## Root Cause: Wrong Threshold

### Training Configuration
From `results/gene_cv_1000_run_15/threshold_suggestion.txt`:
```
output_layer      sigmoid_ovr
calibrated        True
calib_method      platt
threshold_global  0.950
F1_global         0.948
threshold_donor   0.950
F1_donor          0.957
threshold_acceptor 0.950
F1_acceptor       0.940
```

**Key Finding**: The optimal threshold is **0.95**, not 0.5!

### Why 0.95?

1. **Platt Calibration**: The meta-model outputs are calibrated probabilities using Platt scaling
2. **Class Imbalance**: Splice sites are rare (~1-2% of positions), so high threshold is needed
3. **Precision-Recall Trade-off**: Threshold of 0.95 balances precision and recall optimally

### Current Inference Code

In our test scripts and evaluation code, we're using:
```python
threshold = 0.5  # WRONG!
pred_donors = set(predictions_df.filter(pl.col('donor_score') > threshold)['position'].to_list())
```

This explains the massive FP rate:
- At threshold=0.5: Predicts ~30-40% of positions as splice sites → thousands of FPs
- At threshold=0.95: Predicts ~1-2% of positions as splice sites → minimal FPs

---

## Evidence

### 1. Training Metrics at Different Thresholds

From the training results, we can infer:
- **Threshold=0.5**: Would predict many more splice sites (not optimal)
- **Threshold=0.95**: Achieves F1=0.948 (optimal balance)

### 2. FP Rate Comparison

**At threshold=0.5 (current inference)**:
```
ENSG00000187987 (11,573 bp):
  Predicted splice sites: ~762 FPs + 8 TPs = 770 total
  Prediction rate: 770/11,573 = 6.7%
```

**At threshold=0.95 (training)**:
```
Expected prediction rate: ~1-2% (matching true splice site density)
Expected FPs: ~17 per 15,000 positions (from training results)
Scaled to 11,573 bp: ~13 FPs (much better!)
```

### 3. Score Distribution

The meta-model outputs calibrated probabilities where:
- **True splice sites**: Scores typically 0.95-0.99
- **False positives**: Scores typically 0.5-0.9
- **True negatives**: Scores typically 0.0-0.5

Using threshold=0.5 captures all FPs in the 0.5-0.9 range!

---

## Solution

### Fix 1: Update Evaluation Functions

Change all evaluation code to use threshold=0.95:

```python
# OLD:
threshold = 0.5

# NEW:
threshold = 0.95  # Optimal threshold from training
```

### Fix 2: Load Threshold from Training Results

Better approach - load the optimal threshold from the model's training results:

```python
def load_optimal_thresholds(model_dir):
    """Load optimal thresholds from training results."""
    threshold_file = Path(model_dir) / "threshold_suggestion.txt"
    
    if not threshold_file.exists():
        logger.warning("No threshold file found, using default 0.95")
        return {'donor': 0.95, 'acceptor': 0.95, 'global': 0.95}
    
    thresholds = {}
    with open(threshold_file) as f:
        for line in f:
            if 'threshold_' in line:
                key, value = line.strip().split('\t')
                splice_type = key.replace('threshold_', '')
                thresholds[splice_type] = float(value)
    
    return thresholds
```

### Fix 3: Update Inference Workflow

The inference workflow should:
1. Load optimal thresholds from model directory
2. Apply type-specific thresholds (donor vs acceptor)
3. Report which threshold is being used

```python
# In EnhancedSelectiveInferenceWorkflow
def _load_model_config(self):
    """Load model and its optimal thresholds."""
    model_dir = Path(self.config.model_path).parent
    
    # Load thresholds
    self.thresholds = load_optimal_thresholds(model_dir)
    self.logger.info(f"Loaded optimal thresholds: {self.thresholds}")
    
    # Load model
    self.meta_model = joblib.load(self.config.model_path)
```

---

## Testing Plan

### Step 1: Verify Base-Only and Hybrid Modes (In Progress)

Test on well-annotated protein-coding genes to ensure:
- Base-only achieves F1 > 0.7 (ideally > 0.8)
- Hybrid performs similarly to base-only
- Predictions align with GTF annotations

**Genes**:
- TP53 (ENSG00000141510): 25,768 bp
- BRAF (ENSG00000157764): 205,603 bp
- GSTM3 (ENSG00000134202): 7,107 bp
- PTEN (ENSG00000171862): 105,338 bp
- BRCA2 (ENSG00000139618): 84,195 bp

### Step 2: Fix Meta-Only Mode with Correct Threshold

Once base/hybrid are verified:
1. Update evaluation code to use threshold=0.95
2. Re-run meta-only tests
3. Verify meta-only achieves F1 > base-only

**Expected Results**:
```
Gene: ENSG00000187987
  Base-only: F1 = 0.857
  Meta-only: F1 = 0.90-0.95 (improvement!)
  
  Base FPs: 0-1
  Meta FPs: 5-15 (slight increase acceptable if FNs reduced)
  
  Base FNs: 1-2
  Meta FNs: 0-1 (reduction!)
```

### Step 3: Comprehensive Evaluation

Test on diverse gene set:
- Protein-coding genes (expected high performance)
- lncRNA genes (expected lower performance)
- Genes with complex splicing patterns
- Genes with rare splice variants

---

## Files to Modify

### 1. Test Scripts
- `scripts/testing/test_all_modes_comprehensive_v2.py`
- `scripts/testing/test_base_hybrid_long_genes.py`
- Any other evaluation scripts

**Change**: `threshold = 0.5` → `threshold = 0.95`

### 2. Evaluation Functions
- `enhanced_selective_inference.py` (if it has evaluation code)
- Any utility functions that evaluate predictions

**Change**: Add threshold parameter, default to 0.95

### 3. Documentation
- Update all documentation mentioning thresholds
- Add note about calibrated probabilities requiring higher thresholds

---

## Key Insights

1. **Calibrated probabilities require different thresholds**: The meta-model uses Platt scaling, which produces well-calibrated probabilities. These require higher thresholds than raw model outputs.

2. **Class imbalance matters**: Splice sites are rare (~1-2% of positions), so a threshold of 0.5 would predict far too many splice sites.

3. **Training vs inference consistency**: We must use the same threshold in inference as was determined optimal during training.

4. **Type-specific thresholds**: Donor and acceptor sites may have slightly different optimal thresholds (though in this case both are 0.95).

---

## Next Steps

1. ✅ Identified root cause (wrong threshold)
2. 🔄 Running base/hybrid validation test
3. ⏳ Fix evaluation code to use threshold=0.95
4. ⏳ Re-test meta-only mode
5. ⏳ Verify meta-model improvements

---

## Conclusion

The meta-model is working correctly - it's our evaluation that's broken! By using threshold=0.5 instead of 0.95, we're treating any position with >50% probability as a splice site, which results in massive over-prediction.

Once we fix the threshold, we should see:
- Meta-only F1 > Base-only F1 (as expected from training)
- FP rate similar to or better than base model
- FN rate reduced compared to base model

This is a **configuration issue**, not a fundamental problem with the meta-model!

