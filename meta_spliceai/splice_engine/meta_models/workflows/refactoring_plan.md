# Refactoring Plan for selective_meta_inference.py (Updated)

## Current Issues
- File is ~1800 lines long
- Multiple nested try/except and if/else blocks causing indentation errors
- Mixed responsibilities (configuration, verification, processing, I/O)
- Difficult to navigate and maintain
- Performance issues (DataFrame fragmentation warnings)

## Recent Improvements
- Introduced `StandardizedFeaturizer` for consistent feature generation
- Using standardized metadata columns from `preprocessing.py`
- Fixed k-mer extraction and feature harmonization

## Proposed Module Structure

### 1. **selective_meta_inference.py** (Main Driver - ~300 lines)
- Keep only the main `run_selective_meta_inference()` function
- High-level orchestration logic
- Import and coordinate other modules

### 2. **inference_config.py** (~150 lines)
```python
# Configuration classes and factory functions
- SelectiveInferenceConfig (dataclass)
- SelectiveInferenceResults (dataclass)
- create_selective_config()
- Configuration validation
```

### 3. **inference_verification.py** (~200 lines)
```python
# All verification functions
- verify_selective_featurization()
- verify_no_label_leakage()
- Verification result classes
# Note: get_excluded_columns_for_inference() should use preprocessing.py definitions
```

### 4. **inference_io.py** (~400 lines)
```python
# I/O and data management
- setup_inference_directories()
- track_processed_genes()
- load_processed_genes()
- create_gene_manifest()
- get_test_data_directory()
- Save/load predictions
- Artifact preservation logic
- MLflow integration helpers
```

### 5. **prediction_combiner.py** (~200 lines)
```python
# Prediction combination logic
- combine_predictions_for_complete_coverage()
- Inference mode handling (base_only, hybrid, meta_only)
- Confidence categorization
```

### 6. **selective_feature_processor.py** (~300 lines)
```python
# Feature generation for selective positions (simplified with StandardizedFeaturizer)
- generate_selective_meta_predictions()
- generate_chunked_meta_predictions()
- FeatureGeneratorWrapper (thin wrapper around StandardizedFeaturizer)
# Note: Most feature logic now delegated to StandardizedFeaturizer
```

## Benefits of This Structure

1. **Clear Separation of Concerns**
   - Configuration separate from logic
   - I/O separate from processing
   - Verification as standalone utilities

2. **Easier Testing**
   - Can unit test each module independently
   - Mock dependencies more easily

3. **Reduced Nesting**
   - Each module has focused, flatter structure
   - Fewer deeply nested blocks

4. **Better Error Handling**
   - Errors isolated to specific modules
   - Clearer error propagation

5. **Improved Readability**
   - Developers can find functionality quickly
   - Related functions grouped together

## Implementation Strategy

### Phase 1: Create New Modules (No Breaking Changes)
1. Create new module files with extracted functions
2. Keep original functions in place temporarily
3. Test new modules independently

### Phase 2: Wire Up New Modules
1. Update selective_meta_inference.py to import from new modules
2. Replace inline functions with imports
3. Test integration

### Phase 3: Cleanup
1. Remove duplicated code from main file
2. Add proper docstrings
3. Update imports in other files

## Example Refactored Main Driver

```python
# selective_meta_inference.py (refactored)
from .inference_config import SelectiveInferenceConfig, SelectiveInferenceResults
from .inference_verification import verify_selective_featurization
from .inference_io import setup_inference_directories, track_processed_genes
from .prediction_combiner import combine_predictions_for_complete_coverage
from .selective_feature_processor import generate_selective_meta_predictions

def run_selective_meta_inference(config: SelectiveInferenceConfig) -> SelectiveInferenceResults:
    """Main driver - much cleaner and focused."""
    
    # Step 1: Setup
    directories = setup_inference_directories(config.inference_base_dir)
    results = SelectiveInferenceResults(success=False, config=config)
    
    try:
        # Step 2: Load training coverage
        training_positions = load_training_coverage(config)
        
        # Step 3: Run base inference
        base_predictions = run_base_inference(config, training_positions)
        
        # Step 4: Generate meta predictions
        meta_predictions = generate_selective_meta_predictions(
            config, base_predictions, workflow_results
        )
        
        # Step 5: Combine predictions
        hybrid_predictions = combine_predictions_for_complete_coverage(
            base_predictions, meta_predictions, config
        )
        
        # Step 6: Save results
        save_results(hybrid_predictions, directories, results)
        
        # Step 7: Track genes
        track_processed_genes(directories['manifests'], results.per_gene_stats, config)
        
        results.success = True
        
    except Exception as e:
        handle_error(e, results, config)
        
    return results
```

## File Organization After Refactoring

```
workflows/
├── selective_meta_inference.py       # Main driver (300 lines)
├── inference/                        # New subdirectory
│   ├── __init__.py
│   ├── config.py                     # Configuration
│   ├── verification.py               # Verification utilities
│   ├── io_utils.py                   # I/O operations
│   ├── prediction_combiner.py        # Prediction combination
│   └── feature_processor.py          # Feature processing
├── chunked_meta_processor.py         # Keep as is (uses StandardizedFeaturizer)
└── selective_feature_generator.py    # Can be deprecated (functionality in StandardizedFeaturizer)
```

## Integration with Existing Components

### StandardizedFeaturizer Integration
- Feature processing modules will use `StandardizedFeaturizer` instead of custom logic
- Remove duplicate k-mer extraction code
- Ensure consistent feature generation across training and inference

### Preprocessing Module Integration
- Use `METADATA_COLUMNS`, `LEAKAGE_COLUMNS`, etc. from `preprocessing.py`
- Remove hardcoded column lists
- Leverage existing `drop_unwanted_columns()` function

### Performance Optimizations
- Address DataFrame fragmentation warnings by batch operations
- Use efficient DataFrame operations (avoid iterative `.at[]` updates)
- Consider using Polars for heavy data processing

## Priority Actions

1. **Immediate**: Create the new module structure
2. **Phase 1**: Move configuration and I/O functions
3. **Phase 2**: Move verification and feature processing
4. **Phase 3**: Refactor main driver and test thoroughly

This refactoring will:
- Reduce file size from 1800+ to ~300 lines for the main driver
- Eliminate deep nesting and indentation issues
- Improve testability and maintainability
- Leverage existing standardized components
