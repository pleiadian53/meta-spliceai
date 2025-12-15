# Metadata Features vs. Model Features

**Created:** 2025-10-28  
**Status:** 🎯 Critical Design Distinction

## Problem Statement

**User's Key Insight:**
> "The 9 extra features dropped can be useful metadata for selecting nucleotide position to apply meta model in the future"

These features were being **incorrectly excluded** from the inference workflow because they were treated as model input features, when they should be **preserved as metadata** for downstream analysis and meta-model selection logic.

## The 9 Metadata Features

### Uncertainty Indicators
1. **`is_uncertain`** - Boolean flag for positions requiring meta-model
2. **`is_low_confidence`** - Low confidence in base model prediction
3. **`is_high_entropy`** - High entropy across splice type probabilities
4. **`is_low_discriminability`** - Small difference between top predictions

### Confidence Metrics
5. **`max_confidence`** - Maximum probability across splice types
6. **`score_spread`** - Difference between highest and second-highest scores
7. **`score_entropy`** - Shannon entropy of splice type distribution

### Classification Metadata
8. **`confidence_category`** - Categorical label (high/medium/low confidence)
9. **`predicted_type_base`** - Base model's predicted splice type

## Critical Distinction

### Model Input Features (121 features)
**Purpose:** Features used by the meta-model to make predictions

**Categories:**
- Base scores: `donor_score`, `acceptor_score`, `neither_score` (3)
- Probability-derived features: `probability_entropy`, `donor_acceptor_diff`, etc. (38)
- Context features: `context_score_m1`, `donor_weighted_context`, etc. (12)
- K-mers: `3mer_AAA`, `3mer_AAC`, ..., `3mer_TTT` (64)
- Genomic features: `gene_start`, `tx_start`, `num_overlaps`, etc. (10)
- Sequence features: `gc_content`, `sequence_length`, `sequence_complexity` (3)
- Categorical: `chrom` (1)

**Total:** 121 features

**Usage:** `model.predict_proba(X)` where `X.shape = (n_positions, 121)`

### Metadata Features (9 features)
**Purpose:** Metadata for analysis, visualization, and future meta-model selection

**Categories:**
- Uncertainty flags: 4 features
- Confidence metrics: 3 features
- Classification labels: 2 features

**Usage:** 
- Output columns for analysis
- Filtering positions for selective meta-model application
- Visualization and debugging
- Future adaptive threshold tuning

## Current Implementation Issue

### Problem
In `_align_features_with_model()`, these 9 metadata features are:
1. ✅ Correctly excluded from model input (good!)
2. ❌ Logged as warnings suggesting they shouldn't exist (bad!)
3. ❌ May be dropped from output DataFrames (bad!)

### Current Code
```python
# In _align_features_with_model()
extra_non_kmers = [f for f in extra if not is_kmer_feature(f)]

if extra_non_kmers:
    self.logger.warning(f"    ⚠️  Extra {len(extra_non_kmers)} non-k-mer features (will drop)")
    self.logger.warning(f"       Features: {sorted(extra_non_kmers)[:10]}")
```

**Issue:** This warning suggests these features are problems, when they're actually valuable metadata!

## Proposed Solution

### 1. Define Metadata Features Explicitly

```python
# In meta_spliceai/splice_engine/meta_models/builder/preprocessing.py

# Features that should be preserved as metadata but NOT used for model input
METADATA_FEATURES = [
    # Uncertainty indicators (used for selective meta-model application)
    'is_uncertain',
    'is_low_confidence', 
    'is_high_entropy',
    'is_low_discriminability',
    
    # Confidence metrics (useful for analysis and thresholding)
    'max_confidence',
    'score_spread',
    'score_entropy',
    
    # Classification metadata
    'confidence_category',
    'predicted_type_base',
]

# Features that indicate data leakage (MUST be excluded)
LEAKAGE_COLUMNS = [
    'is_donor', 'is_acceptor', 'is_neither',
    'label', 'y', 'target',
    'is_true_donor', 'is_true_acceptor',
    'site_type', 'is_splice_site'
]

# Features that are identifiers (MUST be excluded from model)
IDENTIFIER_COLUMNS = [
    'gene_id', 'transcript_id', 'position',
    'absolute_position', 'gene_name',
    'tx_name', 'exon_id'
]
```

### 2. Update Feature Alignment Logic

```python
def _align_features_with_model(self, features: pd.DataFrame, model, 
                                 data_with_metadata: Optional[pl.DataFrame] = None) -> tuple:
    """
    Align inference features with model's expected features.
    
    IMPORTANT: Distinguishes between:
    1. Model input features (121) - Must match exactly
    2. Metadata features (9) - Preserved for output but not used in model
    3. Extra features - Safely dropped if not in training
    
    Returns
    -------
    tuple[pd.DataFrame, dict]
        - Aligned features for model input (121 columns)
        - Metadata dict for preserving in output
    """
    from meta_spliceai.splice_engine.meta_models.builder.preprocessing import (
        METADATA_FEATURES, is_kmer_feature
    )
    
    # Get expected features
    expected_features = self._get_expected_features(model)
    
    inference_features = set(features.columns)
    expected_set = set(expected_features)
    
    missing = expected_set - inference_features
    extra = inference_features - expected_set
    
    # Separate extra features into categories
    extra_metadata = [f for f in extra if f in METADATA_FEATURES]
    extra_kmers = [f for f in extra if is_kmer_feature(f) and f not in METADATA_FEATURES]
    extra_other = [f for f in extra if f not in METADATA_FEATURES and not is_kmer_feature(f)]
    
    # Log categorized extras
    self.logger.info(f"  Feature alignment:")
    self.logger.info(f"    Model expects: {len(expected_features)} features")
    self.logger.info(f"    Inference has: {len(inference_features)} features")
    self.logger.info(f"    Common: {len(expected_set & inference_features)} features")
    
    if extra_metadata:
        self.logger.info(f"    ℹ️  Metadata features: {len(extra_metadata)} (preserved for output)")
        self.logger.debug(f"       Features: {sorted(extra_metadata)}")
        
    if extra_kmers:
        self.logger.info(f"    ℹ️  Extra k-mers: {len(extra_kmers)} (not in training, will drop)")
        
    if extra_other:
        self.logger.warning(f"    ⚠️  Unexpected extra features: {len(extra_other)}")
        self.logger.warning(f"       Features: {sorted(extra_other)}")
    
    # Extract metadata before alignment
    metadata_dict = {}
    for meta_col in extra_metadata:
        if meta_col in features.columns:
            metadata_dict[meta_col] = features[meta_col].values
    
    # Handle missing features (k-mers filled with 0, others raise error)
    missing_kmers = [f for f in missing if is_kmer_feature(f)]
    missing_non_kmers = [f for f in missing if not is_kmer_feature(f)]
    
    if missing_non_kmers:
        raise ValueError(
            f"CRITICAL: Missing {len(missing_non_kmers)} non-k-mer features. "
            f"Features: {sorted(missing_non_kmers)[:10]}"
        )
    
    if missing_kmers:
        self.logger.info(f"    ℹ️  Missing {len(missing_kmers)} k-mers (not in test sequence, filling with 0)")
        for kmer in missing_kmers:
            features[kmer] = 0
    
    # Align features to model expectations (drops extras, adds missing with 0)
    features_aligned = features.reindex(columns=expected_features, fill_value=0)
    
    self.logger.info(f"  ✅ Features aligned: {features_aligned.shape[1]} columns")
    
    return features_aligned, metadata_dict
```

### 3. Preserve Metadata in Output

```python
def _apply_meta_model_to_features(self, model, features: pd.DataFrame,
                                    original_data: pl.DataFrame) -> tuple:
    """
    Apply meta-model to feature matrix, preserving metadata.
    
    Returns
    -------
    tuple[np.ndarray, dict]
        - Predictions array (n_positions, 3)
        - Metadata dictionary for output preservation
    """
    try:
        # Align features (returns aligned features + metadata dict)
        features_aligned, metadata_dict = self._align_features_with_model(
            features, model, original_data
        )
        
        # Get predictions from model
        predictions = model.predict_proba(features_aligned)
        
        self.logger.info(f"  ✅ Meta-model predictions generated for {len(predictions)} positions")
        
        # Validate predictions
        if not np.allclose(predictions.sum(axis=1), 1.0, atol=1e-3):
            self.logger.warning("  ⚠️  Predictions don't sum to 1.0 - normalizing")
            row_sums = predictions.sum(axis=1, keepdims=True)
            predictions = predictions / row_sums
        
        return predictions, metadata_dict
        
    except Exception as e:
        self.logger.error(f"  ❌ Meta-model prediction failed: {e}")
        raise
```

### 4. Include Metadata in Final Output

```python
def _create_output_dataframe(self, gene_id: str, predictions: pl.DataFrame,
                               metadata: dict) -> pl.DataFrame:
    """
    Create final output DataFrame with predictions AND metadata.
    """
    # Start with predictions
    output = predictions.clone()
    
    # Add metadata columns
    for col_name, values in metadata.items():
        if len(values) == len(output):
            output = output.with_columns(pl.Series(col_name, values))
    
    return output
```

## Use Cases for Metadata Features

### 1. Adaptive Threshold Selection
```python
# Select positions where meta-model should be applied
uncertain_positions = predictions.filter(
    (pl.col('is_uncertain') == True) | 
    (pl.col('max_confidence') < 0.7) |
    (pl.col('score_entropy') > 0.8)
)
```

### 2. Confidence-Based Filtering
```python
# Filter high-confidence predictions only
high_conf_predictions = predictions.filter(
    pl.col('confidence_category') == 'high'
)
```

### 3. Analysis and Visualization
```python
# Compare base model vs meta-model on different confidence levels
analysis = predictions.group_by('confidence_category').agg([
    pl.col('is_adjusted').mean().alias('meta_model_rate'),
    pl.col('max_confidence').mean().alias('avg_confidence')
])
```

### 4. Performance Evaluation
```python
# Evaluate meta-model impact by confidence level
for category in ['high', 'medium', 'low']:
    subset = predictions.filter(pl.col('confidence_category') == category)
    
    # Calculate F1 score improvement
    base_f1 = calculate_f1(subset, use_meta=False)
    meta_f1 = calculate_f1(subset, use_meta=True)
    
    print(f"{category}: Base F1={base_f1:.3f}, Meta F1={meta_f1:.3f}")
```

### 5. Future Adaptive Meta-Model Selection
```python
# Train a selector model to decide when to apply meta-model
selector_features = [
    'max_confidence', 'score_entropy', 'score_spread',
    'is_low_discriminability'
]

X_selector = predictions.select(selector_features)
y_selector = (predictions['meta_f1'] > predictions['base_f1']).cast(int)

# Train binary classifier: should_apply_meta_model?
selector_model = train_selector(X_selector, y_selector)
```

## Implementation Priority

### Phase 1: Preserve Metadata ✅ HIGH PRIORITY
1. Update `_align_features_with_model()` to return metadata dict
2. Ensure metadata columns are included in final output DataFrames
3. Update logging to distinguish metadata from unexpected features

### Phase 2: Document Usage 📝 MEDIUM PRIORITY
4. Add examples of using metadata for analysis
5. Document best practices for threshold tuning
6. Create visualization tools that leverage metadata

### Phase 3: Adaptive Selection 🚀 FUTURE WORK
7. Train meta-model selector using metadata features
8. Implement dynamic threshold adjustment based on performance
9. Build confidence-aware ensemble strategies

## Testing

### Verify Metadata Preservation
```python
# Test that metadata is preserved in output
result = run_inference(gene_id='ENSG00000134202', mode='hybrid')
output = pl.read_parquet(result.predictions_path)

# Check metadata columns exist
metadata_cols = [
    'is_uncertain', 'is_low_confidence', 'is_high_entropy',
    'is_low_discriminability', 'max_confidence', 'score_spread',
    'score_entropy', 'confidence_category', 'predicted_type_base'
]

for col in metadata_cols:
    assert col in output.columns, f"Metadata column {col} missing!"
    
print("✅ All metadata columns preserved")
```

### Verify Feature Alignment
```python
# Verify model input has exactly 121 features (no metadata)
features_for_model = extract_model_features(output)
assert features_for_model.shape[1] == 121, "Feature count mismatch!"
print("✅ Model features correctly aligned (121 columns)")
```

## Summary

| Feature Type | Count | Used in Model? | In Output? | Purpose |
|--------------|-------|----------------|------------|---------|
| **Model Input** | 121 | ✅ Yes | ✅ Yes | Prediction |
| **Metadata** | 9 | ❌ No | ✅ Yes | Analysis/Selection |
| **Identifiers** | ~7 | ❌ No | ✅ Yes | Tracking |
| **Leakage** | ~8 | ❌ No | ❌ No | Data leakage |

## Key Takeaways

1. 🎯 **Metadata features are valuable** - They enable adaptive meta-model selection
2. 🔧 **Clear separation needed** - Distinguish model features from metadata
3. 📊 **Always preserve metadata** - Include in output for downstream analysis
4. ⚠️ **Don't confuse with leakage** - Metadata is derived from model outputs, not ground truth
5. 🚀 **Future potential** - Enables learned meta-model selection strategies

---

**Action Items:**
- [ ] Update `_align_features_with_model()` to preserve metadata
- [ ] Verify metadata in output DataFrames
- [ ] Update logging messages to reflect metadata vs. unexpected features
- [ ] Document metadata usage examples
- [ ] Create visualization tools leveraging metadata

**Status:** 📋 Documented, awaiting implementation

