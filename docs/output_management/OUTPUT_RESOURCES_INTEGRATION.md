# Output Resources Integration

**Date:** 2025-10-28  
**Status:** ✅ Module Created, Integration Pending  
**Pattern:** Follows `genomic_resources` design

## Summary

Created a centralized `output_resources` module following the same design pattern as `genomic_resources` for consistent system-wide output management.

## What Was Created

### 1. New Module: `meta_spliceai/system/output_resources/`

```
meta_spliceai/system/output_resources/
├── __init__.py          # Public API, singleton pattern
├── config.py            # OutputConfig (like genomic_resources.Config)
├── registry.py          # OutputRegistry (like genomic_resources.Registry)
├── manager.py           # OutputManager (directory management)
└── README.md            # Complete documentation
```

### 2. Key Components

#### **`OutputConfig`** - Configuration
```python
from meta_spliceai.system.output_resources import OutputConfig

config = OutputConfig()
# predictions_base: PROJECT_ROOT/predictions
# artifacts_base: predictions/spliceai_eval/meta_models

# Or from environment
config = OutputConfig.from_env()
# META_SPLICEAI_PREDICTIONS=/custom/path
```

#### **`OutputRegistry`** - Path Resolution
```python
from meta_spliceai.system.output_resources import get_output_registry

registry = get_output_registry()  # Singleton

# Resolve resources
predictions_base = registry.resolve('predictions_base')
artifacts = registry.resolve('artifacts')

# Get gene paths
paths = registry.get_gene_paths('hybrid', 'ENSG00000169239')
# → predictions/hybrid/ENSG00000169239/combined_predictions.parquet
```

#### **`OutputManager`** - Directory Management
```python
from meta_spliceai.system.output_resources import create_output_manager

manager = create_output_manager("hybrid", is_test=False)

# Get all paths for a gene
paths = manager.get_gene_output_paths("ENSG00000169239")

# paths.predictions_file → predictions/hybrid/ENSG00000169239/combined_predictions.parquet
# paths.artifacts_dir → predictions/spliceai_eval/meta_models
# paths.analysis_sequences_dir → predictions/spliceai_eval/meta_models/analysis_sequences
# paths.base_predictions_dir → predictions/spliceai_eval/meta_models/complete_base_predictions
```

## Design Pattern: Parallel to `genomic_resources`

### Comparison

| Feature | genomic_resources | output_resources |
|---------|-------------------|------------------|
| **Config** | `Config` | `OutputConfig` |
| **Registry** | `Registry` | `OutputRegistry` |
| **Manager** | `SystematicGenomicManager` | `OutputManager` |
| **Singleton** | `get_default_manager()` | `get_output_registry()` |
| **resolve()** | ✅ GTF, FASTA, etc. | ✅ predictions, artifacts |
| **Environment vars** | `META_SPLICEAI_DATA` | `META_SPLICEAI_PREDICTIONS` |

### Example: Consistent API

**Genomic Resources:**
```python
from meta_spliceai.system.genomic_resources import create_systematic_manager

manager = create_systematic_manager()
gtf_path = manager.resolve("gtf_file")
```

**Output Resources:**
```python
from meta_spliceai.system.output_resources import get_output_registry

registry = get_output_registry()
predictions_base = registry.resolve("predictions_base")
```

## Integration Plan

### Phase 1: Core Module ✅ COMPLETE
- [x] Create `OutputConfig` class
- [x] Create `OutputRegistry` class
- [x] Create `OutputManager` class
- [x] Write comprehensive README
- [x] Follow `genomic_resources` pattern

### Phase 2: Integrate into Inference Workflow (Next)
- [ ] Update `enhanced_selective_inference.py` to use `OutputManager`
- [ ] Replace `_setup_output_directory()` with `OutputManager`
- [ ] Ensure all metadata features are preserved
- [ ] Update artifact path management

### Phase 3: Update Test Scripts
- [ ] Update `test_diverse_genes_with_metadata.py`
- [ ] Update `test_all_modes_comprehensive.py`
- [ ] Update `test_three_modes_simple.py`

### Phase 4: End-to-End Testing
- [ ] Test all 3 modes with new structure
- [ ] Verify metadata preservation (9/9 features)
- [ ] Verify artifact centralization
- [ ] Performance testing

## Usage Examples

### Example 1: Simple Usage

```python
from meta_spliceai.system.output_resources import create_output_manager

# Create manager
manager = create_output_manager("hybrid", is_test=False)

# Get paths
paths = manager.get_gene_output_paths("ENSG00000169239")

# Save predictions
import polars as pl
predictions_df.write_parquet(paths.predictions_file)

# Output: predictions/hybrid/ENSG00000169239/combined_predictions.parquet
```

### Example 2: Integration with Inference Config

```python
from meta_spliceai.system.output_resources import OutputManager

# Create from inference config
manager = OutputManager.from_config(inference_config, logger)

# Use throughout workflow
for gene_id in config.target_genes:
    paths = manager.get_gene_output_paths(gene_id)
    
    # Save predictions
    predictions_df.write_parquet(paths.predictions_file)
    
    # Save artifacts (shared across modes)
    artifact_path = manager.get_artifact_path(
        'analysis_sequences',
        f'analysis_sequences_{gene_id}.tsv'
    )
    sequences_df.write_csv(artifact_path, separator='\t')
```

### Example 3: Test Mode

```python
# Test outputs automatically go to tests/ subdirectory
manager = create_output_manager("hybrid", is_test=True)

paths = manager.get_gene_output_paths("ENSG00000169239")
# → predictions/hybrid/tests/ENSG00000169239/combined_predictions.parquet
```

## Metadata Features Status

### Current State (from testing)
- **6/9 features** present in output:
  - ✅ `is_uncertain`
  - ✅ `is_low_confidence`
  - ✅ `is_high_entropy`
  - ✅ `max_confidence`
  - ✅ `score_spread`
  - ✅ `confidence_category`

- **3/9 features** missing from output:
  - ❌ `is_low_discriminability`
  - ❌ `score_entropy`
  - ❌ `predicted_type_base`

### Analysis

**Good news:** All 9 features ARE being generated in the code!
- Line 735: `score_entropy`
- Line 747: `predicted_type_base`
- Line 763: `is_low_discriminability`

**Issue:** These features are created in the uncertainty identification step but not flowing through to the final output.

**Solution:** Ensure these columns are preserved when creating final output dataframes in `enhanced_selective_inference.py`.

## Directory Structure (Unchanged)

The new module maintains the clean structure we created:

```
predictions/
├── base_only/
│   ├── {gene_id}/
│   │   └── combined_predictions.parquet
│   └── tests/
│       └── {gene_id}/
├── hybrid/
│   └── {gene_id}/
├── meta_only/
│   └── {gene_id}/
└── spliceai_eval/              # Artifacts (mode-independent)
    └── meta_models/
        ├── analysis_sequences/
        └── complete_base_predictions/
```

## Benefits

### 1. Consistency ✅
- Same pattern as `genomic_resources`
- Familiar API for developers
- Unified system design

### 2. Centralization ✅
- Single source of truth for paths
- Easy to update structure
- Consistent across codebase

### 3. Flexibility ✅
- Environment variable support
- Project root detection
- Configurable paths

### 4. Test Support ✅
- Automatic test separation
- Clean test/production isolation
- Easy to clean up tests

## Next Steps

### Immediate (Today)
1. ✅ Create `output_resources` module
2. Fix metadata preservation in inference workflow
3. Integrate `OutputManager` into `enhanced_selective_inference.py`

### Short-term (This Week)
4. Update all test scripts
5. Run end-to-end test
6. Verify all 9 metadata features present

### Documentation (Ongoing)
7. Update inference workflow docs
8. Add usage examples
9. Create migration guide

## Files Modified/Created

### New Files ✅
1. `meta_spliceai/system/output_resources/__init__.py`
2. `meta_spliceai/system/output_resources/config.py`
3. `meta_spliceai/system/output_resources/registry.py`
4. `meta_spliceai/system/output_resources/manager.py`
5. `meta_spliceai/system/output_resources/README.md`
6. `docs/OUTPUT_RESOURCES_INTEGRATION.md` (this document)

### To Be Modified
1. `meta_spliceai/splice_engine/meta_models/workflows/inference/enhanced_selective_inference.py`
2. `scripts/testing/test_diverse_genes_with_metadata.py`
3. `scripts/testing/test_all_modes_comprehensive.py`
4. `scripts/testing/test_three_modes_simple.py`

## Success Criteria

✅ **Phase 1 Complete:**
- [x] Module structure follows `genomic_resources` pattern
- [x] All three classes implemented (`Config`, `Registry`, `Manager`)
- [x] Singleton pattern implemented
- [x] Comprehensive documentation

🔄 **Phase 2 In Progress:**
- [ ] `OutputManager` integrated into inference workflow
- [ ] All 9 metadata features preserved in output
- [ ] Artifact paths use centralized management

⏳ **Phase 3 Pending:**
- [ ] All test scripts updated
- [ ] End-to-end testing complete
- [ ] Production deployment ready

## Summary

**Created:** Centralized `output_resources` module  
**Pattern:** Parallel to `genomic_resources`  
**Status:** Module complete, integration pending  
**Next:** Integrate into inference workflow and ensure metadata preservation

---

**Version:** 1.0.0  
**Date:** 2025-10-28  
**Author:** System Integration Team


