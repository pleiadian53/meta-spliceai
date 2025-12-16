# Feature Preprocessing Improvements

## Background

The MetaSpliceAI meta-model currently faces challenges with mixed data types (categorical and numeric) during training and inference. In particular, we've encountered issues with columns like `'chrom'` causing errors during model diagnostics and evaluation when non-numeric values (e.g., 'X', 'Y') need to be processed by numeric-only models.

Our current solution is a temporary fix in `classifier_utils.py` that converts categorical values to dummy numeric values and ensures feature count compatibility. However, a more principled approach to feature preprocessing is needed for long-term robustness.

## Current Limitations

1. **Inconsistent Feature Handling**: Categorical columns like `'chrom'` are handled differently between training and evaluation
2. **Runtime Feature Mismatch**: Feature count mismatches between trained models and evaluation datasets
3. **Lack of Feature Metadata**: No persistent record of which features were used for training and how they were preprocessed
4. **Brittle Preprocessing**: Ad-hoc preprocessing without a standardized pipeline
5. **Missing Feature Selection Strategy**: No formal approach to determining which features to include in training

## Proposed Solution: Feature Engineering Pipeline

### 1. Dedicated Feature Preprocessor Class

```python
class FeaturePreprocessor:
    def __init__(self, categorical_cols=None, numeric_cols=None):
        self.categorical_cols = categorical_cols or []
        self.numeric_cols = numeric_cols or []
        self.encoders = {}  # Store encoders for each categorical column
        
    def fit(self, df):
        # Auto-detect column types if not specified
        if not self.categorical_cols and not self.numeric_cols:
            self._detect_column_types(df)
            
        # Fit encoders for categorical columns
        for col in self.categorical_cols:
            if col in df.columns:
                self.encoders[col] = self._create_encoder(df[col])
        
        return self
        
    def transform(self, df):
        # Apply preprocessing consistently
        result = df.copy()
        
        # Handle categorical columns
        for col, encoder in self.encoders.items():
            if col in df.columns:
                result[col] = encoder.transform(df[col])
            else:
                # Add missing column with default values
                result[col] = self.encoders[col].default_value
                
        return result
        
    def _detect_column_types(self, df):
        # Logic to auto-detect column types
        # e.g., chromosomes, gene IDs as categorical
        
    def _create_encoder(self, series):
        # Create appropriate encoder based on column characteristics
        # e.g., OrdinalEncoder for chrom
```

### 2. Feature Selection Tracking

- Store feature metadata in a JSON config file alongside the model:
  ```json
  {
    "feature_version": "1.0.0",
    "categorical_features": ["chrom", "gene_type"],
    "numeric_features": ["donor_score", "acceptor_score", ...],
    "encoding_schemes": {
      "chrom": "ordinal",
      "gene_type": "one_hot"
    },
    "selected_features": ["chrom", "donor_score", ...]
  }
  ```

- Provide utilities to load and apply this configuration during both training and inference

### 3. Model-Feature Compatibility Layer

```python
class MetaModelFeatureManager:
    def __init__(self, feature_config_path=None):
        self.feature_config = self._load_config(feature_config_path)
        self.preprocessor = self._build_preprocessor()
        
    def prepare_features(self, df, for_inference=False):
        # Ensure all expected features are present
        # Apply correct preprocessing based on feature type
        # Handle mixed data types appropriately
        
    def _build_preprocessor(self):
        # Create and configure a FeaturePreprocessor based on config
```

### 4. Explicit Categorical Handling

- Use scikit-learn's `OneHotEncoder` or `OrdinalEncoder` for categorical columns
- Implement custom encoders for genomic data:
  - Special handling for sex chromosomes (X, Y, MT)
  - Consistent handling of chromosome naming conventions (e.g., 'chr1' vs. '1')
- Store the encoding scheme with the model

## Implementation Timeline

### Phase 1: Documentation and Planning (Current)
- [x] Document the issue and proposed solutions
- [ ] Review existing feature handling code
- [ ] Define interfaces for the new feature preprocessing system

### Phase 2: Basic Feature Preprocessor
- [ ] Implement `FeaturePreprocessor` class
- [ ] Create feature metadata schema
- [ ] Update training pipeline to save feature metadata

### Phase 3: Integration
- [ ] Implement `MetaModelFeatureManager`
- [ ] Update evaluation scripts to use the new feature preprocessing system
- [ ] Ensure backward compatibility with existing models

### Phase 4: Extended Functionality
- [ ] Add feature importance-based feature selection
- [ ] Implement advanced categorical encoding options
- [ ] Add support for feature interactions

## Integration Points

The new feature preprocessing system should integrate with:

1. `builder_utils.py` - During model building
2. `classifier_utils.py` - For model diagnostics and evaluation
3. `run_gene_cv_sigmoid.py` - For cross-validation
4. `cv_evaluation.py` - For comprehensive evaluation

## Benefits

1. **Consistency**: Same preprocessing applied in training and inference
2. **Robustness**: Proper handling of mixed data types
3. **Transparency**: Clear documentation of feature selection and preprocessing
4. **Maintainability**: Centralized feature logic rather than scattered preprocessing
5. **Extensibility**: Easy to add new feature types and preprocessing methods

## Risks and Mitigations

1. **Risk**: Backward compatibility issues with existing models
   - **Mitigation**: Include version information in metadata and provide compatibility layers

2. **Risk**: Performance impact of more complex preprocessing
   - **Mitigation**: Optimize critical paths, consider caching preprocessed features

3. **Risk**: Increased complexity of the codebase
   - **Mitigation**: Clear documentation and examples, comprehensive unit tests
