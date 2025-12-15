# Metadata Features Status Report

**Created:** 2025-10-28  
**Status:** ✅ 6/9 Preserved, 3/9 To Be Added

## Current Status

### ✅ Metadata Features in Output (6/9)

These metadata features are **currently preserved** in the inference output and available for downstream analysis:

1. **`is_uncertain`** ✅
   - Type: Boolean
   - Usage: 15.3% of positions marked as uncertain in hybrid mode
   - Purpose: Primary flag for selective meta-model application

2. **`is_low_confidence`** ✅
   - Type: Boolean
   - Usage: 0% in current test (high-quality predictions)
   - Purpose: Identifies low-confidence base model predictions

3. **`is_high_entropy`** ✅
   - Type: Boolean
   - Usage: 0% in current test
   - Purpose: High entropy across splice type probabilities

4. **`max_confidence`** ✅
   - Type: Float64
   - Statistics: Mean=0.931, Std=0.157
   - Purpose: Maximum probability across splice types

5. **`score_spread`** ✅
   - Type: Float64
   - Statistics: Mean=0.829, Std=0.348
   - Purpose: Difference between highest and second-highest scores

6. **`confidence_category`** ✅
   - Type: String
   - Values: "high", "medium", "low", etc.
   - Purpose: Categorical confidence classification

### ❌ Missing Metadata Features (3/9)

These features should be added to enhance selective meta-model application:

1. **`is_low_discriminability`** ❌
   - Purpose: Small difference between top predictions
   - Formula: `score_spread < spread_low_threshold`
   - Implementation: Add in `_identify_uncertain_positions()`

2. **`score_entropy`** ❌  
   - Purpose: Shannon entropy of splice type distribution
   - Formula: `-sum(p * log(p))` for probabilities
   - Implementation: Already computed as `entropy`, just needs to be renamed/aliased

3. **`predicted_type_base`** ❌
   - Purpose: Base model's predicted splice type
   - Formula: `argmax(donor_score, acceptor_score, neither_score)`
   - Implementation: Add in output schema creation

## Verification Results

### Test File
- Path: `predictions/test_hybrid/predictions/hybrid/combined_predictions.parquet`
- Mode: Hybrid
- Genes: ENSG00000134202, ENSG00000141736, ENSG00000169239
- Total positions: 28,424

### Feature Distribution

| Feature | Present? | Type | Non-zero % | Mean | Purpose |
|---------|----------|------|------------|------|---------|
| `is_uncertain` | ✅ | Boolean | 15.3% | - | Meta-model trigger |
| `is_low_confidence` | ✅ | Boolean | 0.0% | - | Low confidence flag |
| `is_high_entropy` | ✅ | Boolean | 0.0% | - | High entropy flag |
| `max_confidence` | ✅ | Float | 100% | 0.931 | Confidence metric |
| `score_spread` | ✅ | Float | 100% | 0.829 | Discriminability |
| `confidence_category` | ✅ | String | 100% | - | Categorical label |
| `is_low_discriminability` | ❌ | - | - | - | **TO ADD** |
| `score_entropy` | ❌ | - | - | - | **TO ADD** (alias `entropy`) |
| `predicted_type_base` | ❌ | - | - | - | **TO ADD** |

## Implementation Plan

### Phase 1: Add Missing Features (Quick Wins)

#### 1. `score_entropy` (Alias existing `entropy`)
```python
# In _create_final_output_schema()
output_df = output_df.with_columns([
    pl.col('entropy').alias('score_entropy')  # Already computed!
])
```

#### 2. `predicted_type_base` (Derive from scores)
```python
# In _create_final_output_schema()
output_df = output_df.with_columns([
    pl.when(pl.col('donor_score') >= pl.col('acceptor_score'))
      .when(pl.col('donor_score') >= pl.col('neither_score'))
      .then(pl.lit('donor'))
      .when(pl.col('acceptor_score') >= pl.col('neither_score'))
      .then(pl.lit('acceptor'))
      .otherwise(pl.lit('neither'))
      .alias('predicted_type_base')
])
```

#### 3. `is_low_discriminability` (From score_spread)
```python
# In _identify_uncertain_positions()
uncertain_df = uncertain_df.with_columns([
    (pl.col('score_spread') < spread_low_threshold).alias('is_low_discriminability')
])
```

### Phase 2: Documentation & Testing

1. Update `METADATA_VS_MODEL_FEATURES.md` with current status
2. Add unit tests for metadata preservation
3. Create visualization examples using metadata
4. Document best practices for adaptive thresholding

### Phase 3: Advanced Usage

1. Train meta-model selector using metadata features
2. Implement confidence-based ensembling
3. Build adaptive threshold tuning system
4. Create performance analysis dashboard

## Use Cases (Current & Planned)

### Current Use Cases (6 features available)

#### 1. Selective Meta-Model Application
```python
# Current hybrid mode logic
uncertain = df.filter(pl.col('is_uncertain') == True)
meta_applied = len(uncertain)
total = len(df)
print(f"Meta-model usage: {meta_applied}/{total} ({meta_applied/total*100:.1f}%)")
```

#### 2. Confidence-Based Analysis
```python
# Analyze predictions by confidence category
analysis = df.group_by('confidence_category').agg([
    pl.count().alias('count'),
    pl.col('is_adjusted').mean().alias('meta_model_rate'),
    pl.col('max_confidence').mean().alias('avg_confidence')
])
```

### Future Use Cases (with 3 additional features)

#### 3. Multi-Criteria Uncertainty Detection
```python
# Combine multiple uncertainty indicators
high_uncertainty = df.filter(
    (pl.col('is_uncertain') == True) |
    (pl.col('is_low_discriminability') == True) |
    (pl.col('score_entropy') > 0.8)
)
```

#### 4. Base Model Error Analysis
```python
# Compare base vs meta predictions
errors = df.filter(
    (pl.col('predicted_type_base') != pl.col('splice_type')) &  # Base wrong
    (pl.col('splice_type') == ground_truth)  # Meta correct
)

# Analyze what made base model fail
error_profile = errors.group_by('predicted_type_base').agg([
    pl.col('score_entropy').mean(),
    pl.col('max_confidence').mean(),
    pl.col('is_low_discriminability').mean()
])
```

## Recommendations

### For Immediate Use
1. ✅ Use `is_uncertain` for selective meta-model application
2. ✅ Use `max_confidence` and `score_spread` for filtering
3. ✅ Use `confidence_category` for stratified analysis

### For Enhanced Functionality
1. 🔧 Add `is_low_discriminability` for improved uncertainty detection
2. 🔧 Add `score_entropy` (alias `entropy`) for information-theoretic analysis
3. 🔧 Add `predicted_type_base` for error analysis and debugging

### For Future Development
1. 🚀 Train learned meta-model selector using all 9 metadata features
2. 🚀 Implement adaptive threshold tuning based on validation performance
3. 🚀 Build confidence-calibrated ensemble predictions

## Summary

**Current State:** ✅ **FUNCTIONAL**
- 6/9 metadata features preserved in output
- Sufficient for basic selective meta-model application
- Ready for confidence-based analysis

**Enhancement Needed:** 🔧 **LOW PRIORITY**
- 3 additional features can be easily added
- Would enable more sophisticated uncertainty detection
- Useful for advanced analysis and learned selection strategies

**Overall Assessment:** 
The current implementation successfully preserves the most important metadata features. The missing 3 features are "nice-to-have" enhancements that can be added incrementally without affecting core functionality.

---

**Next Steps:**
1. Add `score_entropy` alias (1 line change)
2. Add `predicted_type_base` derivation (~5 lines)
3. Add `is_low_discriminability` flag (~3 lines)
4. Update test suite to verify all 9 features
5. Document usage examples

**Estimated Effort:** ~30 minutes for all 3 additions


