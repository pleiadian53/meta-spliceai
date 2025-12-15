# Base Model Configuration

**Date:** 2025-10-28  
**Purpose:** Document configurable base model support in output management  
**Status:** ✅ Implemented

## Overview

The `output_resources` module supports **configurable base model names** to organize artifacts by the base model used for predictions. This allows the system to cleanly separate outputs when using different base models (SpliceAI, OpenSpliceAI, Pangolin, etc.).

## Motivation

> **User Insight (2025-10-28):**
> "The reason we use this subdirectory name, `spliceai_eval`, is because we are by default using SpliceAI as the base model, but in the future, if we use other base models like OpenSpliceAI, then we'll have to change the directory name accordingly to keep outputs organized (e.g., `predictions/openspliceai_eval/...`)."

### Problem
- Initially, `spliceai_eval` was hardcoded
- No way to organize outputs by base model
- Future-proofing for multiple base models

### Solution
- Configurable `base_model_name` parameter
- Environment variable support (`META_SPLICEAI_BASE_MODEL`)
- Clean directory structure per base model

## Directory Structure

### Default (SpliceAI)

```
predictions/
├── base_only/
│   └── {gene_id}/
│       └── combined_predictions.parquet
├── hybrid/
│   └── {gene_id}/
├── meta_only/
│   └── {gene_id}/
└── spliceai_eval/           # ← Base model artifacts
    └── meta_models/
        ├── analysis_sequences/
        └── complete_base_predictions/
```

### With OpenSpliceAI

```bash
export META_SPLICEAI_BASE_MODEL=openspliceai
```

```
predictions/
├── base_only/
├── hybrid/
├── meta_only/
└── openspliceai_eval/       # ← OpenSpliceAI artifacts
    └── meta_models/
        ├── analysis_sequences/
        └── complete_base_predictions/
```

### With Pangolin

```bash
export META_SPLICEAI_BASE_MODEL=pangolin
```

```
predictions/
├── base_only/
├── hybrid/
├── meta_only/
└── pangolin_eval/           # ← Pangolin artifacts
    └── meta_models/
        ├── analysis_sequences/
        └── complete_base_predictions/
```

## Usage

### Method 1: Environment Variable (Recommended)

```bash
# Set base model
export META_SPLICEAI_BASE_MODEL=openspliceai

# Run inference
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --genes ENSG00000169239 \
    --mode hybrid

# Artifacts saved to: predictions/openspliceai_eval/meta_models/
```

### Method 2: Programmatic

```python
from meta_spliceai.system.output_resources import create_output_manager

# SpliceAI (default)
manager = create_output_manager("hybrid", base_model_name="spliceai")

# OpenSpliceAI
manager = create_output_manager("hybrid", base_model_name="openspliceai")

# Pangolin
manager = create_output_manager("hybrid", base_model_name="pangolin")

# Get artifact paths
artifact_path = manager.get_artifact_path(
    'analysis_sequences',
    'sequence_data.tsv'
)
# → predictions/{base_model_name}_eval/meta_models/analysis_sequences/sequence_data.tsv
```

### Method 3: From Inference Config

```python
from meta_spliceai.system.output_resources import OutputManager

# Create from config with base model
manager = OutputManager.from_config(
    config=inference_config,
    logger=logger,
    base_model_name="openspliceai"
)
```

### Method 4: OutputConfig Directly

```python
from meta_spliceai.system.output_resources import OutputConfig, OutputRegistry, OutputManager

# Create config with base model
config = OutputConfig(base_model_name="openspliceai")

# Create registry
registry = OutputRegistry(config)

# Verify paths
print(registry.resolve('artifacts'))
# Output: predictions/openspliceai_eval/meta_models
```

## Environment Variables

### Full Set

```bash
# Base predictions directory
export META_SPLICEAI_PREDICTIONS=/mnt/scratch/predictions

# Base model name (affects artifact directory)
export META_SPLICEAI_BASE_MODEL=openspliceai

# Override artifacts directory (overrides base_model_name)
export META_SPLICEAI_ARTIFACTS=/custom/artifacts
```

### Priority Order

1. **`META_SPLICEAI_ARTIFACTS`** (highest priority, overrides everything)
2. **`META_SPLICEAI_BASE_MODEL`** (used to construct artifacts path)
3. **Constructor parameter** `base_model_name`
4. **Default** (`"spliceai"`)

### Examples

#### Example 1: Default SpliceAI

```bash
# No environment variables
python run_inference.py

# Result:
# predictions_base: PROJECT_ROOT/predictions
# artifacts: predictions/spliceai_eval/meta_models
```

#### Example 2: OpenSpliceAI

```bash
export META_SPLICEAI_BASE_MODEL=openspliceai
python run_inference.py

# Result:
# predictions_base: PROJECT_ROOT/predictions
# artifacts: predictions/openspliceai_eval/meta_models
```

#### Example 3: Custom Predictions Directory

```bash
export META_SPLICEAI_PREDICTIONS=/mnt/scratch/predictions
export META_SPLICEAI_BASE_MODEL=openspliceai
python run_inference.py

# Result:
# predictions_base: /mnt/scratch/predictions
# artifacts: /mnt/scratch/predictions/openspliceai_eval/meta_models
```

#### Example 4: Complete Override

```bash
export META_SPLICEAI_ARTIFACTS=/custom/artifacts
python run_inference.py

# Result:
# predictions_base: PROJECT_ROOT/predictions
# artifacts: /custom/artifacts (overrides base_model_name)
```

## Supported Base Models

### Current

| Model | base_model_name | Status |
|-------|----------------|---------|
| SpliceAI | `spliceai` | ✅ Default |
| OpenSpliceAI | `openspliceai` | ✅ Supported |

### Future

| Model | base_model_name | Status |
|-------|----------------|---------|
| Pangolin | `pangolin` | 🔄 Planned |
| SpliceBERT | `splicebert` | 🔄 Planned |
| Custom | `{custom}` | 🔄 Planned |

## Design Rationale

### Why `{base_model_name}_eval`?

1. **Clear Separation** - Each base model gets its own artifact directory
2. **Consistency** - All base model artifacts follow same pattern
3. **Discovery** - Easy to find artifacts by model: `predictions/*_eval/`
4. **Future-Proof** - Supports arbitrary base models

### Why Under `predictions/`?

**Critical Decision:** All output (predictions + artifacts) should be in ONE directory.

**Before (Wrong):**
```
predictions/          # Predictions
  ├── hybrid/
  └── base_only/

predictions_base/     # Artifacts (separate!)
  └── spliceai_eval/
```
❌ Two directories to manage  
❌ Inconsistent structure  
❌ Harder to clean up

**After (Correct):**
```
predictions/          # Everything in one place
  ├── hybrid/         # Predictions
  ├── base_only/
  └── spliceai_eval/  # Artifacts
      └── meta_models/
```
✅ Single directory  
✅ Consistent structure  
✅ Easy cleanup

## API Reference

### `OutputConfig`

```python
class OutputConfig:
    predictions_base: Optional[Path] = None
    base_model_name: str = "spliceai"  # ← Configurable
    artifacts_base: Optional[Path] = None
    use_project_root: bool = True
```

**Constructor:**
```python
# Default
config = OutputConfig()

# Custom base model
config = OutputConfig(base_model_name="openspliceai")

# Complete override
config = OutputConfig(
    predictions_base=Path("/mnt/predictions"),
    base_model_name="openspliceai"
)
```

**Methods:**
```python
# From environment
config = OutputConfig.from_env()
```

### `create_output_manager()`

```python
def create_output_manager(
    mode: str,
    is_test: bool = False,
    base_dir: Optional[Path] = None,
    base_model_name: str = "spliceai"  # ← Configurable
) -> OutputManager
```

**Examples:**
```python
# Default SpliceAI
manager = create_output_manager("hybrid")

# OpenSpliceAI
manager = create_output_manager("hybrid", base_model_name="openspliceai")

# Custom directory + OpenSpliceAI
manager = create_output_manager(
    "hybrid",
    base_dir=Path("/mnt/predictions"),
    base_model_name="openspliceai"
)
```

### `OutputManager.from_config()`

```python
@classmethod
def from_config(
    cls,
    config: EnhancedSelectiveInferenceConfig,
    logger: Optional[logging.Logger] = None,
    base_model_name: str = "spliceai"  # ← Configurable
) -> OutputManager
```

## Testing

### Unit Test

```python
def test_base_model_configuration():
    """Test configurable base model name."""
    
    # Test default
    config = OutputConfig()
    assert config.base_model_name == "spliceai"
    assert "spliceai_eval" in str(config.artifacts_base)
    
    # Test OpenSpliceAI
    config = OutputConfig(base_model_name="openspliceai")
    assert config.base_model_name == "openspliceai"
    assert "openspliceai_eval" in str(config.artifacts_base)
    
    # Test custom
    config = OutputConfig(base_model_name="pangolin")
    assert "pangolin_eval" in str(config.artifacts_base)
```

### Integration Test

```python
def test_inference_with_openspliceai():
    """Test inference workflow with OpenSpliceAI."""
    import os
    
    # Set environment
    os.environ['META_SPLICEAI_BASE_MODEL'] = 'openspliceai'
    
    # Create manager
    manager = create_output_manager("hybrid", is_test=True)
    
    # Verify artifacts path
    artifact_path = manager.get_artifact_path(
        'analysis_sequences',
        'test.tsv'
    )
    assert 'openspliceai_eval' in str(artifact_path)
    
    # Clean up
    del os.environ['META_SPLICEAI_BASE_MODEL']
```

## Migration Guide

### Updating Existing Code

**Before:**
```python
# Hardcoded
artifacts_dir = Path("predictions/spliceai_eval/meta_models")
```

**After:**
```python
# Configurable
from meta_spliceai.system.output_resources import create_output_manager

manager = create_output_manager("hybrid", base_model_name="spliceai")
artifacts_dir = manager.registry.resolve('artifacts')
```

### For Different Base Models

```python
# SpliceAI (default)
manager_spliceai = create_output_manager("hybrid", base_model_name="spliceai")

# OpenSpliceAI
manager_open = create_output_manager("hybrid", base_model_name="openspliceai")

# Compare artifacts paths
print(manager_spliceai.registry.resolve('artifacts'))
# → predictions/spliceai_eval/meta_models

print(manager_open.registry.resolve('artifacts'))
# → predictions/openspliceai_eval/meta_models
```

## Best Practices

### 1. Use Environment Variables for Deployment

```bash
# In production/cluster
export META_SPLICEAI_BASE_MODEL=openspliceai
export META_SPLICEAI_PREDICTIONS=/mnt/scratch/predictions
```

### 2. Explicit Base Model in Scripts

```python
# Good: Explicit
manager = create_output_manager("hybrid", base_model_name="openspliceai")

# Avoid: Implicit default
manager = create_output_manager("hybrid")  # Uses spliceai by default
```

### 3. Document Base Model in Metadata

```python
# Save with predictions
metadata = {
    'base_model': 'openspliceai',
    'base_model_version': '1.0.0',
    'inference_mode': 'hybrid',
    'timestamp': datetime.now().isoformat()
}
```

## Summary

**Feature:** Configurable base model name  
**Purpose:** Organize outputs by base model  
**Implementation:** `base_model_name` parameter throughout  
**Environment Variable:** `META_SPLICEAI_BASE_MODEL`  
**Default:** `"spliceai"`  
**Future:** Support for OpenSpliceAI, Pangolin, custom models

---

**Version:** 1.0.0  
**Date:** 2025-10-28  
**Status:** ✅ Production-ready

