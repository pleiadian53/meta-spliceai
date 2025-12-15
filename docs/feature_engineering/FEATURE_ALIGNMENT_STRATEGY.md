# Feature Alignment Strategy: Training vs. Inference

## Problem Statement

**User's Observation:**
> "In output of the inference workflow, I still saw feature dimension mismatch."

**Core Issue:**
The feature matrix during **training** (`X_train`) and during **inference** (`X_inference`) must have **identical columns** for the model to make predictions.

## User's Correct Analysis

### Critical Insight #1: Direction Matters (with K-mer Exception!)

| Scenario | Severity | Resolution |
|----------|----------|------------|
| **Inference has MORE features** | ⚠️ Non-critical | ✅ Drop extra features (e.g., rare k-mers) |
| **Inference has FEWER k-mer features** | ⚠️ Non-critical | ✅ Fill with 0 (k-mer not in test sequence) |
| **Inference has FEWER non-k-mer features** | 🚨 **CRITICAL** | ❌ Indicates incomplete feature generation |

**Example:**
- Training: 121 features (64 k-mers + 57 other features)
- Inference (MORE): 130 features → Drop 9 extra k-mers (rare k-mers not in training)
- Inference (FEWER k-mers): 115 features (45 k-mers + 57 other) → **OK** - Fill 19 missing k-mers with 0
- Inference (FEWER non-k-mers): 118 features (64 k-mers + 54 other) → **FAIL** - missing 3 critical features!

**Why K-mers Are Special:**
- Training data (1000 genes) → comprehensive k-mer coverage (all 64 possible 3-mers)
- Test gene (1 gene) → limited k-mer coverage (e.g., only 45 of 64 3-mers in sequence)
- Missing k-mers in test gene → count = 0 (NOT an error!)

### Critical Insight #2: Extra Features Are Acceptable

**Why?** Test data may contain:
- **Unseen k-mers**: Gene sequence contains "GGC" (not in training data)
- **Rare categorical values**: New chromosome (e.g., "chrUn_random")
- **Edge case features**: Extreme context scores

**These are SAFE to drop** because:
1. Model was never trained on them
2. Model doesn't expect them
3. Dropping them doesn't lose information the model can use

### Critical Insight #3: Missing Features Are Fatal

**Why?** Missing features indicate:
- ❌ Incomplete feature extraction pipeline
- ❌ Bug in feature generation code
- ❌ Inconsistency between training and inference workflows

**Example Issues:**
- k-mers not generated during inference
- Genomic features not enriched
- Categorical encoding not applied
- Context scores not computed

## Implementation: `_align_features_with_model()`

### Design Philosophy

```python
def _align_features_with_model(features, model):
    """
    Align inference features with model's expected features.
    
    Two cases:
    1. CRITICAL: Missing features → RAISE ERROR (inference bug)
    2. NON-CRITICAL: Extra features → DROP SILENTLY (expected behavior)
    """
```

### Algorithm

```python
# Step 1: Get model's expected features
expected_features = model.feature_names_in_  # From scikit-learn/LightGBM

# Step 2: Compare
inference_features = set(features.columns)
expected_set = set(expected_features)

missing = expected_set - inference_features  # Model expects, inference doesn't have
extra = inference_features - expected_set     # Inference has, model doesn't expect

# Step 3: Separate k-mers from non-k-mers
missing_kmers = [f for f in missing if is_kmer_feature(f)]
missing_non_kmers = [f for f in missing if not is_kmer_feature(f)]

# Step 4: Handle missing non-k-mers (CRITICAL)
if missing_non_kmers:
    raise ValueError(f"CRITICAL: Missing {len(missing_non_kmers)} non-k-mer features")

# Step 5: Handle missing k-mers (NON-CRITICAL - fill with 0)
if missing_kmers:
    logger.info(f"Missing {len(missing_kmers)} k-mers (not in test sequence, filling with 0)")
    for kmer in missing_kmers:
        features[kmer] = 0

# Step 6: Handle extra features (NON-CRITICAL - drop)
if extra:
    logger.info(f"Dropping {len(extra)} extra features")
    # Will be handled by reindex below

# Step 7: Reindex to match model's expectations (adds missing with 0, drops extra)
features = features.reindex(columns=expected_features, fill_value=0)

return features
```

### Key Features

#### **1. Comprehensive Diagnostics**

```python
logger.info(f"Feature alignment:")
logger.info(f"  Model expects: {len(expected_features)} features")
logger.info(f"  Inference has: {len(inference_features)} features")
logger.info(f"  Common: {len(expected_set & inference_features)} features")

if missing:
    logger.error(f"  ❌ Missing: {len(missing)} features")
    logger.error(f"     First 10: {sorted(missing)[:10]}")
    
    # Diagnose k-mer issues
    missing_kmers = [f for f in missing if is_kmer(f)]
    if missing_kmers:
        logger.error(f"     Missing k-mers: {len(missing_kmers)} (k-mer generation issue?)")
```

#### **2. Actionable Error Messages**

```python
raise ValueError(
    f"CRITICAL: Inference is missing {len(missing)} features that the model expects. "
    f"This indicates incomplete feature generation. "
    f"Missing features (first 10): {sorted(missing)[:10]}"
)
```

#### **3. Graceful Handling of Extra Features**

```python
if extra:
    logger.info(f"  ℹ️  Extra: {len(extra)} features (will drop)")
    logger.debug(f"     First 10: {sorted(extra)[:10]}")
    features = features[expected_features]  # Drop extras
```

## Diagnostic Tool: `diagnose_feature_mismatch.py`

### Purpose

Systematically compare training and inference features to identify mismatches.

### Usage

```bash
cd /Users/pleiadian53/work/meta-spliceai
conda activate surveyor
python scripts/testing/diagnose_feature_mismatch.py
```

### What It Does

1. **Load Model Metadata**
   - Extract `feature_names_in_` from model
   - Load `feature_manifest.json/csv`
   - Load `global_excluded_features.txt`

2. **Analyze Inference Features**
   - Parse analysis file (`analysis_sequences_*.tsv`)
   - Categorize columns (leakage, metadata, k-mers, etc.)
   - Identify potential features

3. **Compare**
   - Missing in inference (CRITICAL)
   - Extra in inference (non-critical)
   - K-mer-specific analysis

4. **Report**
   ```
   ✅ PERFECT MATCH: All 121 features match exactly
   
   OR
   
   ❌ PROBLEM IDENTIFIED:
     Inference is missing 6 features that the model expects
     Missing features: ['AAA', 'AAC', 'AAG', 'AAT', 'tx_start', 'tx_end']
     Root cause: Incomplete feature generation in inference workflow
   ```

## Common Mismatch Scenarios

### Scenario 1: Missing K-mers (Natural - Not an Error!)

**Symptom:**
```
ℹ️  Missing 19 k-mers (not in test sequence, will fill with 0)
   K-mers: ['AAN', 'CAN', 'CNN', 'GAN', 'GNN', ...]
```

**Root Cause:** Test gene sequence doesn't contain all possible k-mers

**Example:**
```
Training data (1000 genes):
  Sequences: AAACCCGGGTTT, AAAGGGCCCTTT, ...
  K-mers found: All 64 possible 3-mers (AAA, AAC, AAG, ..., TTT)

Test gene (1 gene):
  Sequence: AAACCCGGGTTT
  K-mers found: 45 of 64 3-mers
  Missing: AAN, CAN, CNN, GAN, GNN, ... (19 k-mers)
  
Action: Fill missing k-mers with count = 0
Result: ✅ Feature matrix now has all 121 features
```

**Fix:** ✅ **No fix needed** - automatically filled with 0 by alignment function

### Scenario 1b: K-mer Generation Not Running (Error!)

**Symptom:**
```
❌ Missing 64 CRITICAL features
   Features: ['AAA', 'AAC', 'AAG', ..., 'TTT']
```

**Root Cause:** K-mer generation not running during inference (ALL k-mers missing)

**Fix:**
```python
# In _generate_complete_base_model_predictions()
complete_predictions = self._generate_kmer_features(complete_predictions, kmer_sizes=[3])
```

### Scenario 2: Missing Genomic Features

**Symptom:**
```
❌ Missing: 5 features
   ['tx_start', 'tx_end', 'num_overlaps', 'transcript_length', 'gc_content']
```

**Root Cause:** Genomic enrichment not running before feature extraction

**Fix:**
```python
# Enrich BEFORE k-mer generation
complete_predictions = self.genomic_enricher.enrich(complete_predictions)
complete_predictions = self._generate_kmer_features(complete_predictions, kmer_sizes=[3])
```

### Scenario 3: Missing Sequence Features

**Symptom:**
```
❌ Missing: 3 features
   ['gc_content', 'sequence_length', 'sequence_complexity']
```

**Root Cause:** Sequence feature calculation not integrated

**Fix:**
```python
# In _generate_kmer_features()
df = self._calculate_sequence_features(df)  # Add this step
```

### Scenario 4: Extra K-mers (Non-critical)

**Symptom:**
```
ℹ️  Extra: 12 features (will drop)
   ['AAN', 'CNN', 'GNN', ...]  # Rare k-mers not in training
```

**Root Cause:** Test gene has rare k-mer combinations not seen during training

**Fix:** ✅ **No fix needed** - automatically dropped by alignment function

### Scenario 5: Categorical Encoding Mismatch

**Symptom:**
```
❌ Missing: 1 feature
   ['chrom']
```

**Root Cause:** `chrom` not numerically encoded during inference

**Fix:**
```python
# Apply categorical encoding
complete_predictions = self._apply_dynamic_chrom_encoding(complete_predictions)
```

## Best Practices

### **1. Always Call Alignment Before Prediction**

```python
# BAD: Direct prediction
predictions = model.predict_proba(features)  # ❌ May fail with feature mismatch

# GOOD: Align first
features_aligned = self._align_features_with_model(features, model)  # ✅ Guaranteed match
predictions = model.predict_proba(features_aligned)
```

### **2. Log Feature Details**

```python
logger.info(f"Extracted {len(feature_cols)} feature columns")
logger.info(f"Feature categories:")
logger.info(f"  - Base scores: {count_base}")
logger.info(f"  - Probability features: {count_prob}")
logger.info(f"  - Context features: {count_context}")
logger.info(f"  - Genomic features: {count_genomic}")
logger.info(f"  - K-mers: {count_kmers}")
```

### **3. Use Diagnostic Tools**

Before deploying:
```bash
# Run diagnostic
python scripts/testing/diagnose_feature_mismatch.py

# Expected output
✅ PERFECT MATCH: All 121 features match exactly
```

### **4. Document Feature Generation Order**

```python
# CRITICAL: Order matters!
# 1. Load base predictions (donor_score, acceptor_score, neither_score)
# 2. Derive probability features (from base scores)
# 3. Compute context features (from neighboring scores)
# 4. Enrich genomic features (gene_start, tx_start, num_overlaps, etc.)
# 5. Apply categorical encoding (chrom → numeric)
# 6. Generate k-mers (from sequence column)
# 7. Calculate sequence features (gc_content, sequence_complexity)
```

### **5. Save Feature Manifest During Training**

```python
# During training
from meta_spliceai.splice_engine.meta_models.builder.feature_manifest_utils import save_feature_manifests

save_feature_manifests(
    X_train,
    output_dir=model_dir,
    model=model,
    categorical_features=['chrom']
)
```

This creates:
- `feature_manifest.json` - Complete metadata
- `feature_manifest.csv` - Human-readable list
- `global_excluded_features.txt` - Exclusions applied

### **6. Validate Consistency**

```python
# During inference initialization
assert model_path.exists(), "Model not found"
assert (model_path.parent / "feature_manifest.json").exists(), "Feature manifest missing"
assert (model_path.parent / "global_excluded_features.txt").exists(), "Exclusions file missing"
```

## Testing

### Unit Test: Feature Alignment

```python
def test_feature_alignment():
    # Training features
    expected = ['donor_score', 'acceptor_score', 'AAA', 'AAC', 'tx_start']
    
    # Inference has extra k-mer
    inference_features = ['donor_score', 'acceptor_score', 'AAA', 'AAC', 'AAG', 'tx_start']
    
    # Align
    aligned = _align_features_with_model(inference_features, model)
    
    # Should drop 'AAG'
    assert list(aligned.columns) == expected
```

### Integration Test: End-to-End

```python
def test_inference_feature_consistency():
    # Run inference
    results = run_inference(gene_id='ENSG00000141736', mode='hybrid')
    
    # Should complete without feature mismatch errors
    assert results.success
    assert results.meta_model_usage > 0  # Meta-model was applied
```

## Summary

### Key Principles

1. ✅ **Extra features are OK** → Drop them
2. 🚨 **Missing features are FATAL** → Raise error
3. 📊 **Always align before predict** → Guaranteed consistency
4. 🔍 **Use diagnostics** → Catch issues early
5. 📝 **Document feature generation** → Reproducibility

### Implementation Checklist

- [x] `_align_features_with_model()` method implemented
- [x] Alignment called before `model.predict_proba()`
- [x] Comprehensive logging for diagnostics
- [x] Error messages are actionable
- [x] Extra features handled gracefully
- [x] Missing features raise informative errors
- [x] Diagnostic tool (`diagnose_feature_mismatch.py`) created
- [x] Documentation complete

### Expected Behavior

**Success Case:**
```
Feature alignment:
  Model expects: 121 features
  Inference has: 121 features
  Common: 121 features
✅ Features aligned: 121 columns
✅ Meta-model predictions generated for 1500 positions
```

**Extra Features (OK):**
```
Feature alignment:
  Model expects: 121 features
  Inference has: 125 features
  Common: 121 features
  ℹ️  Extra: 4 features (will drop)
     First 10: ['AAU', 'CNN', 'GNN', 'NNN']
✅ Features aligned: 121 columns
✅ Meta-model predictions generated for 1500 positions
```

**Missing Features (ERROR):**
```
Feature alignment:
  Model expects: 121 features
  Inference has: 115 features
  Common: 115 features
  ❌ Missing: 6 features
     First 10: ['AAA', 'AAC', 'AAG', 'AAT', 'tx_start', 'tx_end']
     Missing k-mers: 4 (k-mer generation issue?)
❌ CRITICAL: Inference is missing 6 features that the model expects.
```

---

**Status:** ✅ Implemented and Ready to Test

**Next Steps:**
1. Run diagnostic tool to identify current mismatch
2. Fix identified issues in feature generation
3. Re-test inference workflow
4. Verify alignment succeeds

