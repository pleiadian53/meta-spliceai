# Base Model Selection and Genomic Build Routing

**Date**: 2025-11-06  
**Status**: ✅ Implemented and Tested

---

## Overview

MetaSpliceAI now supports multiple base models with automatic genomic build routing:

- **SpliceAI**: Keras-based, trained on GRCh37/Ensembl
- **OpenSpliceAI**: PyTorch-based, trained on GRCh38/MANE

The system automatically routes to the correct genomic resources based on the selected base model, ensuring consistency and correctness.

---

## Architecture

### Unified Model Loading

**Entry Point**: `load_base_model_ensemble()` in `model_utils.py`

```python
from meta_spliceai.splice_engine.meta_models.utils.model_utils import load_base_model_ensemble

# Load SpliceAI (GRCh37)
models, metadata = load_base_model_ensemble('spliceai')

# Load OpenSpliceAI (GRCh38)
models, metadata = load_base_model_ensemble('openspliceai')
```

**Metadata Structure**:
```python
{
    'base_model': 'spliceai' or 'openspliceai',
    'genome_build': 'GRCh37' or 'GRCh38',
    'context': 10000,
    'framework': 'keras' or 'pytorch',
    'num_models': 5,
    'device': 'cpu', 'cuda', or 'mps'
}
```

### Genomic Build Routing

**Logic**: `SpliceAIConfig.get_artifact_manager()`

The system uses the `base_model` parameter as the **primary source of truth** for determining:
1. Genomic build (GRCh37 vs GRCh38)
2. Annotation source (Ensembl vs MANE)
3. Artifact storage paths

**Routing Table**:

| Base Model | Genome Build | Annotation Source | Framework |
|------------|--------------|-------------------|-----------|
| spliceai | GRCh37 | Ensembl | Keras |
| openspliceai | GRCh38 | MANE | PyTorch |

---

## User Interface

### 1. Using `run_base_model_predictions()`

**Simple Usage**:
```python
from meta_spliceai import run_base_model_predictions

# Use SpliceAI (default)
results = run_base_model_predictions(
    target_genes=['BRCA1', 'TP53']
)

# Use OpenSpliceAI
results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1', 'TP53']
)
```

**With Configuration**:
```python
from meta_spliceai import run_base_model_predictions, BaseModelConfig

config = BaseModelConfig(
    base_model='openspliceai',
    mode='test',
    threshold=0.5,
    use_auto_position_adjustments=True
)

results = run_base_model_predictions(
    config=config,
    target_genes=['BRCA1']
)
```

### 2. Using `predict_splice_sites()` (Simplified)

```python
from meta_spliceai import predict_splice_sites

# Quick predictions with SpliceAI
positions = predict_splice_sites('BRCA1')

# Quick predictions with OpenSpliceAI
positions = predict_splice_sites(
    genes=['BRCA1', 'TP53'],
    base_model='openspliceai'
)
```

### 3. Direct Workflow Usage

```python
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig

config = SpliceAIConfig(
    base_model='openspliceai',  # Specify base model
    mode='test',
    coverage='gene_subset'
)

results = run_enhanced_splice_prediction_workflow(
    config=config,
    target_genes=['BRCA1'],
    verbosity=1
)
```

---

## Implementation Details

### Model Loading Flow

```
User Request
    ↓
run_base_model_predictions(base_model='openspliceai')
    ↓
BaseModelConfig(base_model='openspliceai')
    ↓
run_enhanced_splice_prediction_workflow(config)
    ↓
load_base_model_ensemble(base_model='openspliceai')
    ↓
load_openspliceai_ensemble()
    ↓
OpenSpliceAI models loaded on appropriate device
```

### Genomic Resource Routing

```
BaseModelConfig(base_model='openspliceai')
    ↓
config.get_artifact_manager()
    ↓
Infer: build='GRCh38', source='mane'
    ↓
create_artifact_manager_from_workflow_config(
    build='GRCh38',
    source='mane',
    base_model='openspliceai'
)
    ↓
Artifacts stored in: data/mane/GRCh38/openspliceai_eval/
```

### Key Design Decisions

1. **Base Model as Primary Source of Truth**
   - The `base_model` parameter determines genomic build
   - GTF file path does NOT override this (prevents confusion)
   - Ensures consistency between model and data

2. **Automatic Device Selection**
   - PyTorch models auto-detect best device (CUDA > MPS > CPU)
   - Keras models use CPU by default
   - User can override with `device` parameter

3. **Artifact Management**
   - Each base model has separate artifact directories
   - Prevents mixing predictions from different builds
   - Enables easy comparison between models

---

## File Structure

### Modified Files

1. **`meta_spliceai/splice_engine/meta_models/utils/model_utils.py`**
   - Added `load_base_model_ensemble()` - unified loader
   - Added `load_openspliceai_ensemble()` - OpenSpliceAI loader
   - Existing `load_spliceai_ensemble()` - SpliceAI loader

2. **`meta_spliceai/splice_engine/meta_models/core/data_types.py`**
   - Added `base_model` parameter to `SpliceAIConfig`
   - Updated `get_artifact_manager()` to use base_model for routing

3. **`meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py`**
   - Updated model loading to use `load_base_model_ensemble()`
   - Added model metadata logging

4. **`meta_spliceai/run_base_model.py`**
   - Removed `NotImplementedError` for OpenSpliceAI
   - Added model validation and prerequisite checks
   - Updated config creation to include `base_model`

5. **`meta_spliceai/openspliceai/predict/predict.py`**
   - Fixed imports (absolute → relative)

### New Files

1. **`scripts/testing/test_openspliceai_integration.py`**
   - Integration test for model loading
   - Genomic build routing validation
   - Prediction workflow verification

2. **`docs/development/BASE_MODEL_SELECTION_AND_ROUTING.md`** (this file)
   - Comprehensive documentation
   - Usage examples
   - Architecture overview

---

## Testing

### Integration Test

**Location**: `scripts/testing/test_openspliceai_integration.py`

**Run**:
```bash
python scripts/testing/test_openspliceai_integration.py
```

**Tests**:
1. ✅ Model Loading (SpliceAI and OpenSpliceAI)
2. ✅ Genomic Build Routing (GRCh37/Ensembl vs GRCh38/MANE)
3. ⏭️ Prediction Workflow (requires GRCh38 data)

**Results**:
```
================================================================================
TEST SUMMARY
================================================================================
✅ Model Loading: PASSED
✅ Genomic Build Routing: PASSED
⏭️  Prediction Workflow: SKIPPED (requires GRCh38 data)

🎉 OPENSPLICEAI INTEGRATION TEST COMPLETE!
================================================================================
```

### Manual Testing

**Test SpliceAI**:
```python
from meta_spliceai import run_base_model_predictions

results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1'],
    mode='test',
    test_name='spliceai_test'
)

print(f"Build: {results['artifact_manager']['summary']['build']}")
print(f"Source: {results['artifact_manager']['summary']['source']}")
# Expected: Build: GRCh37, Source: ensembl
```

**Test OpenSpliceAI**:
```python
from meta_spliceai import run_base_model_predictions

results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1'],
    mode='test',
    test_name='openspliceai_test'
)

print(f"Build: {results['artifact_manager']['summary']['build']}")
print(f"Source: {results['artifact_manager']['summary']['source']}")
# Expected: Build: GRCh38, Source: mane
```

---

## Prerequisites

### For SpliceAI

- ✅ Models: Included with `spliceai` package
- ✅ Data: GRCh37 genome and Ensembl annotations
- ✅ Framework: Keras/TensorFlow

### For OpenSpliceAI

- ✅ Models: Download with `./scripts/base_model/download_openspliceai_models.sh`
- ❌ Data: GRCh38 genome and MANE annotations (not yet available)
- ✅ Framework: PyTorch (installed)

---

## Future Work

### Immediate (Required for Full OpenSpliceAI Support)

1. **GRCh38 Genomic Resources** (4-8 hours)
   - Download GRCh38 genome FASTA
   - Download MANE annotations GTF
   - Generate splice site annotations for GRCh38
   - Test data preparation workflow

2. **Full Prediction Testing** (2-4 hours)
   - Run OpenSpliceAI predictions on test genes
   - Compare with SpliceAI predictions (after liftover)
   - Validate performance metrics

3. **Liftover Utilities** (4-6 hours)
   - Implement coordinate liftover (GRCh37 ↔ GRCh38)
   - Enable cross-build comparisons
   - Support hybrid workflows

### Long-term Enhancements

1. **Multi-Build Support**
   - Support both GRCh37 and GRCh38 simultaneously
   - Automatic coordinate translation
   - Unified result reporting

2. **Model Ensemble**
   - Combine predictions from multiple base models
   - Weighted averaging based on confidence
   - Consensus calling across builds

3. **Additional Base Models**
   - SpliceAI-10k (extended context)
   - Pangolin
   - Custom fine-tuned models

4. **Performance Optimization**
   - GPU acceleration for OpenSpliceAI
   - Batch processing optimization
   - Caching and memoization

---

## API Reference

### `load_base_model_ensemble()`

```python
def load_base_model_ensemble(
    base_model: str = 'spliceai',
    context: int = 10000,
    device: Optional[str] = None,
    verbosity: int = 1
) -> Tuple[List, Dict[str, Any]]
```

**Parameters**:
- `base_model`: 'spliceai' or 'openspliceai'
- `context`: Context window size (default: 10000)
- `device`: PyTorch device ('cpu', 'cuda', 'mps', or None for auto)
- `verbosity`: Output verbosity level

**Returns**:
- `models`: List of loaded model objects
- `metadata`: Dictionary with model information

### `BaseModelConfig`

```python
@dataclass
class BaseModelConfig:
    # Base model selection
    base_model: str = "spliceai"  # 'spliceai' or 'openspliceai'
    
    # Artifact management
    mode: str = "test"  # 'test' or 'production'
    coverage: str = "gene_subset"  # 'gene_subset', 'chromosome', 'full_genome'
    test_name: Optional[str] = None
    
    # Prediction parameters
    threshold: float = 0.5
    consensus_window: int = 2
    error_window: int = 500
    use_auto_position_adjustments: bool = True
    
    # ... other parameters ...
```

### `run_base_model_predictions()`

```python
def run_base_model_predictions(
    base_model: str = 'spliceai',
    target_genes: Optional[List[str]] = None,
    target_chromosomes: Optional[List[str]] = None,
    config: Optional[BaseModelConfig] = None,
    verbosity: int = 1,
    no_tn_sampling: bool = False,
    **kwargs
) -> Dict[str, Any]
```

**Returns**:
```python
{
    'success': bool,
    'positions': pl.DataFrame,  # All analyzed positions
    'error_analysis': pl.DataFrame,  # FP/FN positions
    'analysis_sequences': pl.DataFrame,  # Sequences for analysis
    'paths': {
        'eval_dir': str,
        'artifacts_dir': str,
        'positions_artifact': str,
        'errors_artifact': str
    },
    'artifact_manager': {
        'mode': str,
        'coverage': str,
        'test_name': str,
        'summary': dict
    }
}
```

---

## Troubleshooting

### Issue: OpenSpliceAI models not found

**Error**:
```
FileNotFoundError: OpenSpliceAI models not found at data/models/openspliceai/
```

**Solution**:
```bash
./scripts/base_model/download_openspliceai_models.sh
```

### Issue: PyTorch not installed

**Error**:
```
ModuleNotFoundError: No module named 'torch'
```

**Solution**:
```bash
mamba install pytorch -c pytorch -y
```

### Issue: Wrong genomic build

**Symptom**: Predictions use wrong genomic resources

**Solution**: Ensure `base_model` parameter is set correctly:
```python
# For GRCh37
config = BaseModelConfig(base_model='spliceai')

# For GRCh38
config = BaseModelConfig(base_model='openspliceai')
```

### Issue: Device not available

**Error**:
```
RuntimeError: CUDA not available
```

**Solution**: OpenSpliceAI will automatically fall back to CPU or MPS. No action needed.

---

## Summary

✅ **Implemented**:
- Unified model loading interface
- Automatic genomic build routing
- Base model parameter in all APIs
- Integration testing
- Comprehensive documentation

✅ **Tested**:
- Model loading (SpliceAI and OpenSpliceAI)
- Genomic build routing (GRCh37/Ensembl vs GRCh38/MANE)
- Configuration and artifact management

⏳ **Pending**:
- GRCh38 genomic resources
- Full prediction workflow testing
- Performance comparison

---

**Last Updated**: 2025-11-06  
**Status**: ✅ Production Ready (pending GRCh38 data)

