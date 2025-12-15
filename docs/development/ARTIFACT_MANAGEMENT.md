# Artifact Management System

## Overview

The artifact management system provides systematic organization and lifecycle management for:
1. **Base model prediction artifacts** (analysis_sequences, splice_positions, splice_errors)
2. **Meta-model training datasets** (feature matrices, labels)
3. **Model checkpoints** (trained meta-models)

This system follows the same architectural patterns as `genomic_resources` for consistency and maintainability.

## Key Concepts

### Execution Modes

**Production Mode (`mode="production"`)**
- For complete, genome-wide data collection
- Artifacts are treated as **immutable cache**
- Overwriting requires explicit `force=True` flag
- Used for meta-model training and production inference
- Default for `coverage="full_genome"`

**Test Mode (`mode="test"`)**
- For iterative development and testing
- Artifacts are **always overwritable**
- Isolated in test-specific directories
- Used for validation and debugging
- Default for `coverage="gene_subset"` or `coverage="chromosome"`

### Coverage Levels

- **`full_genome`**: Complete genome-wide analysis (auto-switches to production mode)
- **`chromosome`**: Single or multiple chromosomes
- **`gene_subset`**: Specific genes for testing

## Directory Structure

```
data/<source>/<build>/<base_model>_eval/
├── meta_models/                    # Production artifacts (immutable)
│   ├── full_splice_positions_enhanced.tsv
│   ├── full_splice_errors.tsv
│   └── analysis_sequences_*.tsv
│
├── tests/                          # Test artifacts (ephemeral)
│   ├── <test_name>/
│   │   ├── sampled_genes.tsv
│   │   └── meta_models/
│   │       └── predictions/
│   │           ├── full_splice_positions_enhanced.tsv
│   │           └── analysis_sequences_*.tsv
│
├── training_data/                  # Meta-model training datasets
│   ├── feature_matrix.parquet
│   ├── labels.parquet
│   └── metadata.json
│
└── models/                         # Meta-model checkpoints
    ├── latest/
    │   ├── model.pt
    │   └── config.json
    └── v1.0/
        ├── model.pt
        └── config.json
```

### Example Paths

**Production artifacts (SpliceAI on GRCh37)**:
```
data/ensembl/GRCh37/spliceai_eval/meta_models/
```

**Test artifacts**:
```
data/ensembl/GRCh37/spliceai_eval/tests/base_model_comprehensive_test/meta_models/predictions/
```

**OpenSpliceAI on GRCh38 (future)**:
```
data/mane/GRCh38/openspliceai_eval/meta_models/
```

## Usage

### In Workflow Configuration

```python
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig

# Test mode (default for gene subsets)
config = SpliceAIConfig(
    mode='test',
    coverage='gene_subset',
    test_name='my_test',
    # ... other params
)

# Production mode (for full genome)
config = SpliceAIConfig(
    mode='production',
    coverage='full_genome',
    # ... other params
)
```

### Direct ArtifactManager Usage

```python
from meta_spliceai.system.artifact_manager import ArtifactManager, ArtifactConfig

# Create manager
config = ArtifactConfig(
    mode='test',
    coverage='gene_subset',
    source='ensembl',
    build='GRCh37',
    base_model='spliceai',
    test_name='my_test'
)
manager = ArtifactManager(config)

# Get artifacts directory
artifacts_dir = manager.get_artifacts_dir(create=True)

# Get specific artifact path
positions_file = manager.get_artifact_path('full_splice_positions_enhanced.tsv')

# Check if should overwrite
if manager.should_overwrite(positions_file):
    # Write artifact
    df.write_csv(positions_file, separator='\t')

# List all artifacts
artifacts = manager.list_artifacts('*.tsv')

# Print summary
manager.print_summary()
```

### From Workflow Config

```python
config = SpliceAIConfig(
    mode='test',
    coverage='gene_subset',
    test_name='my_test',
    # ... other params
)

# Get artifact manager from config
manager = config.get_artifact_manager()
```

## Default Values

### Mode Parameter

**Default: `"test"`**

Rationale:
- **Safe by default**: Test mode prevents accidental overwriting of production artifacts
- **Explicit production**: Users must explicitly set `mode="production"` for genome-wide runs
- **Auto-detection**: Automatically switches to production for `coverage="full_genome"`

### Coverage Parameter

**Default: `"gene_subset"`**

Rationale:
- Most development work involves testing on specific genes
- Matches typical workflow usage patterns
- Clear distinction from production runs

## Overwrite Logic

```python
def should_overwrite(artifact_path: Path, force: bool = False) -> bool:
    """
    Determine if artifact should be overwritten.
    
    Logic:
    1. If force=True: Always overwrite
    2. If file doesn't exist: Always write
    3. If mode='test': Always overwrite
    4. If mode='production': Never overwrite (unless force=True)
    """
```

### Examples

```python
# Test mode - always overwrites
manager = ArtifactManager(ArtifactConfig(mode='test'))
manager.should_overwrite(existing_file)  # True

# Production mode - never overwrites
manager = ArtifactManager(ArtifactConfig(mode='production'))
manager.should_overwrite(existing_file)  # False
manager.should_overwrite(existing_file, force=True)  # True

# New file - always writes
manager.should_overwrite(new_file)  # True (regardless of mode)
```

## Integration with Inference Workflow

The artifact manager optimizes meta-model inference by caching pre-computed features:

```python
# During meta-model inference
manager = config.get_artifact_manager()
production_artifacts = manager.get_artifacts_dir()

if production_artifacts.exists() and not force_regenerate:
    # Fast path: Load pre-computed features
    print("[inference] Loading pre-computed artifacts...")
    X = load_feature_matrix(production_artifacts)
else:
    # Slow path: Run base model inference + feature generation
    print("[inference] Generating artifacts from base model...")
    X = run_base_model_and_generate_features()
    
    # Save for future runs
    if manager.config.mode == 'production':
        save_artifacts(production_artifacts)
```

## Training Data Management

The artifact manager also handles meta-model training data:

```python
# Get training data directory
training_dir = manager.get_training_data_dir(create=True)

# Save feature matrix
feature_matrix_path = training_dir / 'feature_matrix.parquet'
X.write_parquet(feature_matrix_path)

# Save labels
labels_path = training_dir / 'labels.parquet'
y.write_parquet(labels_path)

# Save metadata
metadata_path = training_dir / 'metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
```

## Model Checkpoint Management

```python
# Get model checkpoint directory
checkpoint_dir = manager.get_model_checkpoint_dir(
    model_version='v1.0',
    create=True
)

# Save model
torch.save(model.state_dict(), checkpoint_dir / 'model.pt')

# Save config
with open(checkpoint_dir / 'config.json', 'w') as f:
    json.dump(model_config, f, indent=2)

# Create 'latest' symlink
latest_dir = manager.get_model_checkpoint_dir('latest')
if latest_dir.exists():
    latest_dir.unlink()
latest_dir.symlink_to(checkpoint_dir)
```

## Benefits

### 1. Clear Semantics
- Directory structure clearly indicates purpose and mutability
- Test vs production distinction is explicit

### 2. Optimized Inference
- Production artifacts serve as pre-computed cache
- Avoids redundant base model inference
- Significant speed improvement for meta-model predictions

### 3. Flexible Testing
- Test artifacts are isolated and overwritable
- Rapid iteration during development
- No risk of polluting production data

### 4. Base-Model Specific
- Each base model (spliceai, openspliceai) has its own namespace
- Prevents cross-contamination of artifacts
- Supports multiple base models simultaneously

### 5. Build-Specific
- GRCh37 vs GRCh38 artifacts are properly separated
- Follows same pattern as genomic_resources
- Consistent with existing architecture

## Future Enhancements

### Versioning
Add support for artifact versioning:
```
data/ensembl/GRCh37/spliceai_eval/meta_models/
├── v1_20251104/
├── v2_20251105/
└── latest -> v2_20251105/
```

### Automatic Cleanup
Add utilities to clean up old test artifacts:
```python
manager.cleanup_old_tests(keep_recent=5)
```

### Artifact Validation
Add checksums and validation:
```python
manager.validate_artifacts()  # Check integrity
manager.get_artifact_metadata()  # Get creation date, size, etc.
```

## Related Documentation

- [Genomic Resources](../system/genomic_resources/README.md) - Similar pattern for genomic data
- [Schema Standardization](SCHEMA_STANDARDIZATION_SOLUTION.md) - Column name consistency
- [Base Model Testing](../testing/BASE_MODEL_TEST_RUNNING.md) - Test workflow usage

## Summary

The artifact management system provides:
- **Systematic organization** of base model artifacts and training data
- **Mode-based routing** (production vs test)
- **Optimized inference** through artifact caching
- **Flexible testing** with isolated, overwritable artifacts
- **Consistent architecture** following genomic_resources patterns

This ensures efficient, maintainable, and scalable management of all meta-model artifacts throughout the development and production lifecycle.

