# Categorical Feature Encoding

## Overview

This document describes the centralized categorical feature encoding system used in the meta-model training and inference pipelines.

## Design Rationale

### The Problem

Prior to this design, categorical encoding was hardcoded in multiple locations:
- `meta_spliceai/splice_engine/meta_models/builder/preprocessing.py` (training)
- `meta_spliceai/splice_engine/meta_models/workflows/inference/enhanced_selective_inference.py` (inference)

This led to several issues:
1. **Duplication**: Same encoding logic repeated in multiple places
2. **Inconsistency risk**: Training and inference could diverge if logic was updated in one place but not the other
3. **Ambiguity**: No clear specification of which features are categorical vs. numerical
4. **Maintenance burden**: Adding new categorical features required code changes in multiple files

### The Solution

We created a **centralized feature schema** in `meta_spliceai/splice_engine/meta_models/builder/feature_schema.py` that:
1. Defines which features are categorical
2. Specifies how each categorical feature should be encoded
3. Provides reusable encoding functions
4. Ensures consistency between training and inference

## Architecture

### Core Components

#### 1. `CategoricalFeatureSpec` Dataclass

Defines the encoding specification for a single categorical feature:

```python
@dataclass
class CategoricalFeatureSpec:
    name: str                          # Feature column name
    encoding_type: str                 # 'ordinal', 'onehot', or 'custom'
    custom_mapping: Optional[Dict]     # Custom mapping for 'custom' type
    handle_unknown: str                # 'error', 'use_default', or 'skip'
    default_value: Optional[int]       # Default value for unknown categories
    description: str                   # Human-readable description
```

#### 2. `CATEGORICAL_FEATURES` Registry

A dictionary mapping feature names to their encoding specifications:

```python
CATEGORICAL_FEATURES: Dict[str, CategoricalFeatureSpec] = {
    'chrom': CHROM_SPEC,
    # 'strand': STRAND_SPEC,      # Currently excluded as metadata
    # 'gene_type': GENE_TYPE_SPEC,  # Currently excluded as metadata
}
```

#### 3. `encode_categorical_features()` Function

The main encoding function that applies the specifications:

```python
def encode_categorical_features(
    df: pl.DataFrame,
    features_to_encode: Optional[List[str]] = None,
    verbose: bool = False
) -> pl.DataFrame:
    """Encode categorical features according to their specifications."""
```

#### 4. `ALWAYS_NUMERICAL_FEATURES` List

Specifies features that should **always** be treated as numerical, even if they have few distinct values:

```python
ALWAYS_NUMERICAL_FEATURES: List[str] = [
    # K-mer counts (even if some rare k-mers have few distinct values)
    # Splice site scores
    'donor_score', 'acceptor_score', 'neither_score',
    # Probability-derived features
    'relative_donor_probability', 'splice_probability', ...
    # Genomic coordinates (even though they're integers)
    'gene_start', 'gene_end', ...
]
```

## Feature Type Guidelines

### Categorical Features

Features that represent **discrete categories** with no inherent numerical relationship:

| Feature | Example Values | Encoding Strategy |
|---------|---------------|-------------------|
| `chrom` | '1', 'X', 'MT' | Custom mapping with biological ordering |
| `strand` | '+', '-', '.' | Binary: +1, -1, 0 (currently excluded) |
| `gene_type` | 'protein_coding', 'lncRNA' | Ordinal by frequency (currently excluded) |

### Numerical Features

Features that represent **quantities or measurements**:

| Feature Category | Examples | Why Numerical? |
|-----------------|----------|----------------|
| **K-mer counts** | 'AAA', 'ACG', 'GGT' | Represent sequence motif frequencies |
| **Splice scores** | `donor_score`, `acceptor_score` | Continuous probabilities [0, 1] |
| **Probabilities** | `splice_probability`, `probability_entropy` | Derived continuous metrics |
| **Genomic coordinates** | `gene_start`, `gene_end` | Integer positions with inherent ordering |
| **Derived metrics** | `gene_length`, `gc_content` | Computed quantities |

### Ambiguous Cases

**K-mer features** are a special case that might appear categorical but should **always be numerical**:

- ✅ **Correct**: Treat as numerical (counts)
- ❌ **Wrong**: Treat as categorical

**Reason**: K-mers represent sequence motif frequencies. Even if some rare k-mers have only a few distinct count values (e.g., 0, 1, 2), they are fundamentally counts and should be treated as continuous numerical features.

**Example**:
```python
# K-mer 'AAA' might have values: [0, 0, 1, 0, 2, 0, 1, ...]
# This is a COUNT, not a category!
```

## Usage

### During Training

In `preprocessing.py`:

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

### During Inference

In `enhanced_selective_inference.py`:

```python
from meta_spliceai.splice_engine.meta_models.builder.feature_schema import (
    encode_categorical_features
)

# Apply same encoding as training
def _apply_dynamic_chrom_encoding(self, df: pl.DataFrame) -> pl.DataFrame:
    return encode_categorical_features(
        df,
        features_to_encode=['chrom'],
        verbose=False
    )
```

## Chromosome Encoding Specification

The `chrom` feature uses a custom mapping with biological ordering:

### Encoding Rules

| Category | Values | Encoded As |
|----------|--------|------------|
| Autosomes | '1'-'22', 'chr1'-'chr22' | 1-22 |
| X chromosome | 'X', 'chrX' | 23 |
| Y chromosome | 'Y', 'chrY' | 24 |
| Mitochondrial | 'MT', 'chrMT', 'M', 'chrM' | 25 |
| Unknown scaffolds | Any other value | 100+ (incremental) |

### Example

```python
chrom_map = {
    '1': 1, 'chr1': 1,
    '2': 2, 'chr2': 2,
    ...
    'X': 23, 'chrX': 23,
    'Y': 24, 'chrY': 24,
    'MT': 25, 'chrMT': 25,
    'GL000225.1': 100,  # Unknown scaffold
    'KI270750.1': 101,  # Unknown scaffold
    ...
}
```

## Adding New Categorical Features

To add a new categorical feature:

### 1. Define the Specification

```python
# In feature_schema.py

MY_FEATURE_SPEC = CategoricalFeatureSpec(
    name='my_feature',
    encoding_type='custom',
    custom_mapping={
        'category_a': 1,
        'category_b': 2,
        'category_c': 3,
    },
    handle_unknown='use_default',
    default_value=99,
    description="Description of my feature"
)
```

### 2. Register in the Registry

```python
CATEGORICAL_FEATURES: Dict[str, CategoricalFeatureSpec] = {
    'chrom': CHROM_SPEC,
    'my_feature': MY_FEATURE_SPEC,  # Add here
}
```

### 3. Update Exclusion Lists (if needed)

If the feature should be excluded from model input, add it to the appropriate exclusion list in `preprocessing.py`:

```python
# If it's metadata
METADATA_COLUMNS.append('my_feature')

# If it causes leakage
LEAKAGE_COLUMNS.append('my_feature')

# If it's redundant
REDUNDANT_COLUMNS.append('my_feature')
```

### 4. Use in Code

```python
# Training (preprocessing.py)
X_df = encode_categorical_features(
    X_df,
    features_to_encode=['chrom', 'my_feature'],
    verbose=verbose
)

# Inference (enhanced_selective_inference.py)
result_df = encode_categorical_features(
    df,
    features_to_encode=['chrom', 'my_feature'],
    verbose=False
)
```

## Benefits

### 1. **Consistency Guarantee**

Both training and inference use the **exact same encoding logic** from the centralized registry, eliminating the risk of divergence.

### 2. **Single Source of Truth**

All categorical feature specifications are defined in one place (`feature_schema.py`), making it easy to understand and maintain.

### 3. **Clear Documentation**

The `CategoricalFeatureSpec` dataclass provides self-documenting code with descriptions and explicit encoding strategies.

### 4. **Type Safety**

The use of type hints and dataclasses provides better IDE support and catches errors at development time.

### 5. **Easy Extension**

Adding new categorical features is straightforward: define a spec, register it, and use it—no need to modify encoding logic.

### 6. **Validation Support**

The `validate_feature_types()` function can catch errors where features are misclassified as categorical vs. numerical.

## Testing

To verify consistent encoding between training and inference:

```python
# Test script
from meta_spliceai.splice_engine.meta_models.builder.feature_schema import (
    encode_categorical_features,
    CATEGORICAL_FEATURES
)

# Create test dataframe with various chromosome names
test_df = pl.DataFrame({
    'chrom': ['1', 'chr2', 'X', 'chrY', 'MT', 'GL000225.1']
})

# Encode
encoded_df = encode_categorical_features(test_df, features_to_encode=['chrom'])

# Verify
expected = pl.DataFrame({
    'chrom': [1, 2, 23, 24, 25, 100]
})

assert encoded_df['chrom'].to_list() == expected['chrom'].to_list()
```

## Future Improvements

### 1. Configuration File Support

Move feature specifications to a YAML/JSON configuration file:

```yaml
# feature_schema.yaml
categorical_features:
  chrom:
    encoding_type: custom
    custom_mapping:
      '1': 1
      'chr1': 1
      'X': 23
      'chrX': 23
    handle_unknown: use_default
    default_value: 100
    description: "Chromosome identifier"
```

### 2. Dynamic Feature Discovery

Automatically detect categorical vs. numerical features based on data characteristics:

```python
def infer_feature_types(df: pl.DataFrame, threshold: int = 10) -> Dict[str, str]:
    """Infer feature types based on cardinality and data type."""
    feature_types = {}
    for col in df.columns:
        unique_count = df.select(pl.col(col).n_unique())[0, 0]
        dtype = df.select(col).dtypes[0]
        
        if dtype == pl.Utf8 or unique_count < threshold:
            feature_types[col] = 'categorical'
        else:
            feature_types[col] = 'numerical'
    
    return feature_types
```

### 3. One-Hot Encoding Support

Currently only ordinal and custom mappings are supported. Add full one-hot encoding:

```python
def apply_onehot_encoding(df: pl.DataFrame, feature: str) -> pl.DataFrame:
    """Apply one-hot encoding, creating multiple binary columns."""
    unique_vals = df.select(pl.col(feature).unique())[feature].to_list()
    
    for val in unique_vals:
        col_name = f"{feature}_{val}"
        df = df.with_columns(
            (pl.col(feature) == val).cast(pl.Int32).alias(col_name)
        )
    
    return df.drop(feature)
```

## Related Files

- `meta_spliceai/splice_engine/meta_models/builder/feature_schema.py` - Centralized feature specifications
- `meta_spliceai/splice_engine/meta_models/builder/preprocessing.py` - Training data preparation
- `meta_spliceai/splice_engine/meta_models/workflows/inference/enhanced_selective_inference.py` - Inference workflow

## References

- [scikit-learn: Encoding categorical features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)
- [Pandas: Categorical data](https://pandas.pydata.org/docs/user_guide/categorical.html)
- [Feature Engineering Best Practices](https://developers.google.com/machine-learning/data-prep/transform/transform-categorical)

