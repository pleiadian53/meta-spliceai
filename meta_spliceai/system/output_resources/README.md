# Output Resources

**Created:** 2025-10-28  
**Purpose:** Centralized management of prediction outputs and artifacts  
**Pattern:** Follows `genomic_resources` design for consistency

## Overview

This module provides centralized management of prediction outputs, following the same design pattern as `genomic_resources` for consistency across the system.

### Key Components

1. **`OutputConfig`** - Configuration (like `genomic_resources.Config`)
2. **`OutputRegistry`** - Path resolution (like `genomic_resources.Registry`)
3. **`OutputManager`** - Directory management (new, specific to outputs)

## Quick Start

### Basic Usage

```python
from meta_spliceai.system.output_resources import create_output_manager

# Create manager for hybrid mode
manager = create_output_manager("hybrid", is_test=False)

# Get paths for a gene
paths = manager.get_gene_output_paths("ENSG00000169239")

# paths.predictions_file
# → Path('predictions/hybrid/ENSG00000169239/combined_predictions.parquet')

# paths.artifacts_dir
# → Path('predictions/spliceai_eval/meta_models')
```

### With Inference Config

```python
from meta_spliceai.system.output_resources import OutputManager

# Create from inference config
manager = OutputManager.from_config(inference_config, logger)

# Use in workflow
paths = manager.get_gene_output_paths(gene_id)
final_df.write_parquet(paths.predictions_file)
```

## Directory Structure

```
predictions/
├── base_only/              # Base-only mode predictions
│   ├── {gene_id}/         # Per-gene predictions
│   │   └── combined_predictions.parquet
│   └── tests/             # Test outputs
│       └── {gene_id}/
│
├── hybrid/                # Hybrid mode predictions
│   └── {gene_id}/
│
├── meta_only/             # Meta-only mode predictions
│   └── {gene_id}/
│
└── spliceai_eval/         # Artifacts (mode-independent)
    └── meta_models/
        ├── analysis_sequences/
        │   └── analysis_sequences_*.tsv
        └── complete_base_predictions/
            └── gene_sequence_*.parquet
```

## Design Principles

### 1. Parallel to `genomic_resources`

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

### 2. Singleton Pattern

Both systems use singletons for consistency:
```python
# Genomic resources singleton
from meta_spliceai.system.genomic_resources import get_default_manager

# Output resources singleton
from meta_spliceai.system.output_resources import get_output_registry
```

### 3. Environment Variables

Both support environment variable overrides:
```bash
# Genomic resources
export META_SPLICEAI_DATA=/custom/data/path

# Output resources
export META_SPLICEAI_PREDICTIONS=/custom/predictions/path
export META_SPLICEAI_ARTIFACTS=/custom/artifacts/path
```

## API Reference

### `OutputConfig`

Configuration for output locations.

**Constructor:**
```python
OutputConfig(
    predictions_base: Optional[Path] = None,  # Defaults to PROJECT_ROOT/predictions
    artifacts_base: Optional[Path] = None,    # Defaults to predictions/spliceai_eval/meta_models
    use_project_root: bool = True
)
```

**Class Methods:**
```python
OutputConfig.from_env()  # Create from environment variables
```

### `OutputRegistry`

Central registry for path resolution.

**Methods:**
```python
resolve(resource_kind: str) -> Path
    # resource_kind: 'predictions_base', 'artifacts', 'analysis_sequences', 'base_predictions'

get_mode_dir(mode: InferenceMode, is_test: bool = False) -> Path
    # Get directory for inference mode

get_gene_dir(mode: InferenceMode, gene_id: str, is_test: bool = False) -> Path
    # Get directory for specific gene

get_gene_paths(mode: InferenceMode, gene_id: str, is_test: bool = False) -> GenePaths
    # Get all paths for a gene

get_artifact_path(artifact_type: str, filename: str) -> Path
    # Get path for specific artifact file
```

### `OutputManager`

Manages output directory structure.

**Methods:**
```python
get_gene_output_paths(gene_id: str) -> GenePaths
    # Get all paths for a gene

get_combined_output_path() -> Path
    # Get path for combined predictions

cleanup_old_predictions(gene_id: str)
    # Remove old predictions for a gene

get_artifact_path(artifact_type: str, filename: str) -> Path
    # Get path for artifact file

log_directory_structure()
    # Log current directory structure
```

**Class Methods:**
```python
OutputManager.from_config(config: EnhancedSelectiveInferenceConfig, logger: Optional[Logger]) -> OutputManager
    # Create from inference config
```

## Usage Examples

### Example 1: Basic Prediction Output

```python
from meta_spliceai.system.output_resources import create_output_manager

# Create manager
manager = create_output_manager("hybrid", is_test=False)

# Get paths
paths = manager.get_gene_output_paths("ENSG00000169239")

# Save predictions
import polars as pl
predictions_df = pl.DataFrame({...})
predictions_df.write_parquet(paths.predictions_file)

print(f"Saved to: {paths.predictions_file}")
# Output: Saved to: predictions/hybrid/ENSG00000169239/combined_predictions.parquet
```

### Example 2: Test Mode

```python
# Create manager for testing
manager = create_output_manager("hybrid", is_test=True)

# Paths automatically go to tests/ subdirectory
paths = manager.get_gene_output_paths("ENSG00000169239")
# → predictions/hybrid/tests/ENSG00000169239/combined_predictions.parquet
```

### Example 3: Artifacts

```python
manager = create_output_manager("hybrid", is_test=False)

# Save analysis sequences (mode-independent)
artifact_path = manager.get_artifact_path(
    'analysis_sequences',
    'analysis_sequences_1_chunk_1_500.tsv'
)

# artifact_path
# → predictions/spliceai_eval/meta_models/analysis_sequences/analysis_sequences_1_chunk_1_500.tsv

# These are shared across all modes!
```

### Example 4: Integration with Inference Workflow

```python
from meta_spliceai.splice_engine.meta_models.workflows.inference import (
    EnhancedSelectiveInferenceWorkflow,
    EnhancedSelectiveInferenceConfig
)
from meta_spliceai.system.output_resources import OutputManager

# Create config
config = EnhancedSelectiveInferenceConfig(
    target_genes=['ENSG00000169239'],
    model_path=Path('results/model.pkl'),
    inference_mode='hybrid'
)

# Create workflow with output manager
workflow = EnhancedSelectiveInferenceWorkflow(config)

# Output manager is automatically created from config
# workflow.output_manager.get_gene_output_paths(gene_id)
```

## Comparison with `genomic_resources`

### Similarities

| Feature | genomic_resources | output_resources |
|---------|-------------------|------------------|
| **Config class** | `Config` | `OutputConfig` |
| **Registry class** | `Registry` | `OutputRegistry` |
| **Singleton** | `get_default_manager()` | `get_output_registry()` |
| **Environment vars** | `META_SPLICEAI_DATA` | `META_SPLICEAI_PREDICTIONS` |
| **resolve()** | ✅ | ✅ |
| **Project root detection** | ✅ | ✅ |

### Differences

| Feature | genomic_resources | output_resources |
|---------|-------------------|------------------|
| **Purpose** | Input data | Output predictions |
| **Manager** | SystematicGenomicManager | OutputManager |
| **Resources** | GTF, FASTA, splice sites | Predictions, artifacts |
| **Modes** | N/A | base_only, hybrid, meta_only |
| **Test separation** | No | Yes (tests/ subdirectory) |

## Migration Guide

### Before (Old Structure)

```python
# Old way: hardcoded paths
output_dir = Path("predictions") / "diverse_test_hybrid" / "predictions"
gene_output = output_dir / f"{gene_id}_hybrid" / "combined_predictions.parquet"
```

### After (New Structure)

```python
# New way: centralized management
from meta_spliceai.system.output_resources import create_output_manager

manager = create_output_manager("hybrid", is_test=True)
paths = manager.get_gene_output_paths(gene_id)
gene_output = paths.predictions_file

# Clean, simple: predictions/hybrid/tests/{gene_id}/combined_predictions.parquet
```

## Environment Variables

### Supported Variables

```bash
# Override predictions base directory
export META_SPLICEAI_PREDICTIONS=/custom/predictions

# Override artifacts directory
export META_SPLICEAI_ARTIFACTS=/custom/artifacts

# Example usage
export META_SPLICEAI_PREDICTIONS=/mnt/scratch/predictions
python run_inference.py --mode hybrid --genes ENSG00000169239
```

### Priority Order

1. **Explicit constructor argument** (highest priority)
2. **Environment variable**
3. **Default** (PROJECT_ROOT/predictions)

## Testing

### Unit Tests

```python
def test_output_config():
    """Test OutputConfig initialization."""
    config = OutputConfig()
    assert config.predictions_base.exists()
    assert config.artifacts_base.exists()

def test_output_registry():
    """Test OutputRegistry path resolution."""
    registry = OutputRegistry()
    
    # Test resolve
    pred_base = registry.resolve('predictions_base')
    assert pred_base.name == 'predictions'
    
    # Test gene paths
    paths = registry.get_gene_paths('hybrid', 'ENSG00000169239')
    assert 'hybrid' in str(paths.predictions_file)
    assert 'ENSG00000169239' in str(paths.predictions_file)

def test_output_manager():
    """Test OutputManager functionality."""
    manager = create_output_manager('hybrid', is_test=False)
    
    paths = manager.get_gene_output_paths('ENSG00000169239')
    assert paths.predictions_file.parent.name == 'ENSG00000169239'
    assert 'hybrid' in str(paths.predictions_file)
```

## Benefits

### 1. Consistency with Genomic Resources ✅
- Same design pattern
- Similar API
- Familiar to developers

### 2. Centralized Management ✅
- Single source of truth
- Easy to update paths
- Consistent across codebase

### 3. Clean Structure ✅
- Flat, predictable paths
- Test separation
- Mode-based organization

### 4. Environment Support ✅
- Override defaults easily
- Cluster-friendly
- Deployment flexibility

## Summary

**Pattern:** Follows `genomic_resources` design  
**Purpose:** Centralized output management  
**Structure:** Clean, flat, mode-based  
**API:** Familiar, consistent, easy to use

---

**Version:** 1.0.0  
**Created:** 2025-10-28  
**Status:** Production-ready

