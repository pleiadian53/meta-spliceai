# Centralized Categorical Feature Encoding - Implementation Summary

## Overview

This document summarizes the implementation of a centralized categorical feature encoding system that ensures consistency between training and inference pipelines.

## Problem Solved

**Before**: Categorical encoding logic was duplicated and hardcoded in multiple locations:
- `preprocessing.py` (training)
- `enhanced_selective_inference.py` (inference)

This created risks of:
- Inconsistency between training and inference
- Maintenance burden when adding new categorical features
- Ambiguity about which features should be treated as categorical vs. numerical

**After**: A single source of truth for categorical feature specifications with reusable encoding functions.

## Implementation

### 1. New Module: `feature_schema.py`

Location: `meta_spliceai/splice_engine/meta_models/builder/feature_schema.py`

**Key Components**:
- `CategoricalFeatureSpec`: Dataclass defining encoding specifications
- `CATEGORICAL_FEATURES`: Registry of all categorical features
- `ALWAYS_NUMERICAL_FEATURES`: List of features that must stay numerical
- `encode_categorical_features()`: Main encoding function
- `is_kmer_feature()`: Helper to identify k-mer features
- `validate_feature_types()`: Validation function

### 2. Updated Files

**`preprocessing.py`** (lines 33-38, 284-294):
- Import centralized encoding functions
- Replace hardcoded chromosome encoding with call to `encode_categorical_features()`

**`enhanced_selective_inference.py`** (lines 46-48, 1203-1240):
- Import centralized encoding functions
- Simplify `_apply_dynamic_chrom_encoding()` to use centralized logic

### 3. Documentation

- **`docs/development/CATEGORICAL_FEATURE_ENCODING.md`**: Comprehensive guide
- **`docs/CATEGORICAL_ENCODING_SUMMARY.md`**: This summary document

### 4. Tests

**`scripts/testing/test_categorical_encoding.py`**: 5 comprehensive tests
- Test 1: Chromosome encoding correctness
- Test 2: K-mer feature detection
- Test 3: Feature type consistency
- Test 4: Encoding idempotence
- Test 5: Feature type validation

**All tests passing** ✅

## Key Design Decisions

### 1. K-mers are ALWAYS Numerical

**Rationale**: K-mer features represent sequence motif counts, not categories.

Even though some rare k-mers might have few distinct values (e.g., 0, 1, 2), they should be treated as continuous numerical features because:
- They represent counts/frequencies
- They have inherent ordering (0 < 1 < 2)
- Machine learning models should learn distance metrics on these values

### 2. Chromosome Encoding Uses Biological Ordering

**Mapping**:
- Autosomes 1-22 → 1-22
- X chromosome → 23
- Y chromosome → 24
- Mitochondrial (MT/M) → 25
- Unknown scaffolds → 100+

**Rationale**: This encoding preserves biological relationships and is more interpretable than arbitrary numeric assignments.

### 3. Encoding is Idempotent

Once a feature is encoded (numeric), calling the encoding function again is a no-op.

**Implementation**: Check if feature is already numeric before encoding.

### 4. Configuration-Based Design

While currently using Python code for specifications, the design supports future migration to YAML/JSON configuration files:

```yaml
categorical_features:
  chrom:
    encoding_type: custom
    custom_mapping:
      '1': 1
      'X': 23
    handle_unknown: use_default
    default_value: 100
```

## Benefits

### 1. Consistency Guarantee ✅
Both training and inference use **identical encoding logic**, eliminating divergence risk.

### 2. Single Source of Truth ✅
All categorical feature specifications in one place: `feature_schema.py`

### 3. Clear Documentation ✅
Self-documenting code with explicit specifications and descriptions.

### 4. Easy Extension ✅
Adding new categorical features requires only:
1. Define a `CategoricalFeatureSpec`
2. Register in `CATEGORICAL_FEATURES`
3. Use `encode_categorical_features()`

### 5. Type Safety ✅
Type hints and dataclasses provide better IDE support and catch errors early.

### 6. Validation Support ✅
`validate_feature_types()` catches misclassified features.

## Usage Examples

### During Training (preprocessing.py)

```python
from meta_spliceai.splice_engine.meta_models.builder.feature_schema import (
    encode_categorical_features
)

# Encode categorical features
if encode_chrom and 'chrom' in X_df.columns:
    X_df = encode_categorical_features(
        X_df,
        features_to_encode=['chrom'],
        verbose=verbose
    )
```

### During Inference (enhanced_selective_inference.py)

```python
from meta_spliceai.splice_engine.meta_models.builder.feature_schema import (
    encode_categorical_features
)

def _apply_dynamic_chrom_encoding(self, df: pl.DataFrame) -> pl.DataFrame:
    """Apply categorical encoding using centralized schema."""
    return encode_categorical_features(
        df,
        features_to_encode=['chrom'],
        verbose=False
    )
```

## Current Categorical Features

| Feature | Encoding Type | Strategy | Currently Used |
|---------|--------------|----------|----------------|
| `chrom` | Custom | Biological ordering (1-25, 100+) | ✅ Yes |
| `strand` | Custom | Binary (+1, -1, 0) | ❌ No (excluded as metadata) |
| `gene_type` | Custom | Ordinal by biotype | ❌ No (excluded as metadata) |

## Testing Results

```
============================================================
SUMMARY
============================================================
Passed: 5/5
Failed: 0/5

🎉 All tests PASSED!
```

**Test Coverage**:
- ✅ Chromosome encoding correctness
- ✅ K-mer feature detection
- ✅ Feature type consistency
- ✅ Encoding idempotence
- ✅ Feature type validation

## Future Enhancements

### 1. YAML/JSON Configuration
Move feature specifications to external configuration files for easier management.

### 2. Dynamic Feature Discovery
Automatically detect categorical vs. numerical features based on data characteristics.

### 3. One-Hot Encoding
Add support for one-hot encoding (currently only ordinal and custom mappings are supported).

### 4. Feature Importance Tracking
Track which categorical features contribute most to model performance.

### 5. Encoding Visualization
Create visualizations showing the distribution of encoded vs. original categorical values.

## Related Documentation

- **`docs/development/CATEGORICAL_FEATURE_ENCODING.md`**: Detailed technical guide
- **`meta_spliceai/splice_engine/meta_models/builder/feature_schema.py`**: Implementation
- **`scripts/testing/test_categorical_encoding.py`**: Test suite

## Conclusion

The centralized categorical encoding system provides a **robust, maintainable, and consistent** approach to handling categorical features across the entire meta-spliceai pipeline. The design is flexible enough to accommodate future enhancements while ensuring that training and inference always use identical encoding logic.

**Key Takeaway**: **K-mers are counts, not categories!** Always treat them as numerical features.

