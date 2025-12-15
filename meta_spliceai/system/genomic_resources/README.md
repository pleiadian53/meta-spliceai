# Genomic Resources Management Package

A systematic package for managing genomic reference data (GTF, FASTA) and derived datasets for the MetaSpliceAI project.

## Overview

This package provides a clean, reusable interface for:
1. **Configuration Management** (`config.py`) - YAML-based configuration with environment overrides
2. **Path Resolution** (`registry.py`) - Unified interface for locating genomic resources
3. **Data Download** (`download.py`) - Fetch and verify Ensembl GTF/FASTA files
4. **Data Derivation** (`derive.py`) - Generate derived datasets from GTF/FASTA
5. **Validation** (`validators.py`) - Pre-flight checks for gene selection
6. **CLI** (`cli.py`) - Command-line interface for all operations

## Architecture

### Design Principles

1. **Separation of Concerns**: Original workflow code remains unchanged; refactored code lives in `system/genomic_resources/`
2. **Backward Compatibility**: Existing imports and workflows continue to work
3. **Reusability**: New code is designed for use across multiple workflows
4. **Flexibility**: Supports multiple builds, releases, and data locations

### Module Structure

```
meta_spliceai/system/genomic_resources/
├── __init__.py          # Package exports
├── config.py            # Configuration management
├── registry.py          # Path resolution
├── download.py          # Ensembl file fetching
├── derive.py            # Data derivation (NEW - refactored from workflows)
├── validators.py        # Gene validation
├── cli.py               # Command-line interface
└── README.md            # This file
```

## Refactored Data Preparation Functions

The following functions from `splice_prediction_workflow.py` have been refactored into `derive.py`:

### 1. Gene Annotations (`prepare_gene_annotations` → `GenomicDataDeriver.derive_gene_annotations`)

**Original**: `meta_spliceai.splice_engine.meta_models.workflows.data_preparation.prepare_gene_annotations()`

**Refactored**: `GenomicDataDeriver.derive_gene_annotations()`

**Improvements**:
- Cleaner error handling
- Better path management via Registry
- Consistent return format
- Optional force_overwrite flag

### 2. Splice Sites (`prepare_splice_site_annotations` → `GenomicDataDeriver.derive_splice_sites`)

**Original**: `meta_spliceai.splice_engine.meta_models.workflows.data_preparation.prepare_splice_site_annotations()`

**Refactored**: `GenomicDataDeriver.derive_splice_sites()`

**Improvements**:
- Simplified interface
- Automatic chromosome filtering
- Better file existence checking
- Consistent output format

### 3. Genomic Sequences (`prepare_genomic_sequences` → `GenomicDataDeriver.derive_genomic_sequences`)

**Original**: `meta_spliceai.splice_engine.meta_models.workflows.data_preparation.prepare_genomic_sequences()`

**Refactored**: `GenomicDataDeriver.derive_genomic_sequences()`

**Improvements**:
- Cleaner parameter handling
- Better format support (parquet, tsv, csv)
- Improved error messages
- Per-chromosome file management

### 4. Overlapping Genes (`handle_overlapping_genes` → `GenomicDataDeriver.derive_overlapping_genes`)

**Original**: `meta_spliceai.splice_engine.meta_models.workflows.data_preparation.handle_overlapping_genes()`

**Refactored**: `GenomicDataDeriver.derive_overlapping_genes()`

**Improvements**:
- Simplified interface
- Better caching strategy
- Consistent error handling
- Optional chromosome filtering

## Usage

### Python API

```python
from meta_spliceai.system.genomic_resources import GenomicDataDeriver

# Create deriver instance
deriver = GenomicDataDeriver(verbosity=1)

# Derive individual datasets
result = deriver.derive_gene_annotations()
result = deriver.derive_splice_sites(consensus_window=2)
result = deriver.derive_genomic_sequences(mode='gene', seq_type='full')
result = deriver.derive_overlapping_genes()

# Or derive everything at once
results = deriver.derive_all(
    consensus_window=2,
    target_chromosomes=['1', '2', 'X'],
    force_overwrite=False
)
```

### Command-Line Interface

```bash
# Audit current resources
python -m meta_spliceai.system.genomic_resources.cli audit

# Download GTF and FASTA
python -m meta_spliceai.system.genomic_resources.cli bootstrap \
  --species homo_sapiens \
  --build GRCh38 \
  --release 112

# Derive all datasets
python -m meta_spliceai.system.genomic_resources.cli derive --all

# Derive specific datasets
python -m meta_spliceai.system.genomic_resources.cli derive \
  --splice-sites \
  --consensus-window 2 \
  --chromosomes 1 2 X Y

# Force regeneration
python -m meta_spliceai.system.genomic_resources.cli derive \
  --all \
  --force
```

## Integration with Existing Workflows

### Current State

The original workflow functions in `splice_prediction_workflow.py` **remain unchanged** and continue to work:

```python
# This still works exactly as before
from meta_spliceai.splice_engine.meta_models.workflows.data_preparation import (
    prepare_gene_annotations,
    prepare_splice_site_annotations,
    prepare_genomic_sequences,
    handle_overlapping_genes
)

result = prepare_gene_annotations(local_dir, gtf_file, ...)
```

### Future Migration Path

When ready to migrate to the new system:

```python
# Option 1: Use GenomicDataDeriver directly
from meta_spliceai.system.genomic_resources import GenomicDataDeriver

deriver = GenomicDataDeriver(data_dir=local_dir)
result = deriver.derive_gene_annotations(...)

# Option 2: Use convenience functions
from meta_spliceai.system.genomic_resources import derive_gene_annotations

result = derive_gene_annotations(data_dir=local_dir, ...)
```

## Key Differences from Original Implementation

### 1. Class-Based Design

**Original**: Standalone functions
**Refactored**: `GenomicDataDeriver` class with instance methods

**Benefits**:
- Shared configuration across operations
- Better state management
- Easier testing and mocking

### 2. Consistent Return Format

**Original**: Varied return formats (dict, DataFrame, tuple)
**Refactored**: Always returns `Dict[str, Any]` with:
- `'success'`: bool
- `'<resource>_file'`: str (path)
- `'<resource>_df'`: pl.DataFrame
- `'error'`: str (if failed)

### 3. Better Error Handling

**Original**: Mix of exceptions and silent failures
**Refactored**: Consistent exception handling with informative error messages

### 4. Path Management

**Original**: Hardcoded paths and manual path construction
**Refactored**: Uses `Registry` for systematic path resolution

### 5. Configuration

**Original**: Parameters passed through function calls
**Refactored**: YAML configuration with environment overrides

## Configuration

### YAML Configuration (`configs/genomic_resources.yaml`)

```yaml
species: homo_sapiens
default_build: GRCh38
default_release: "112"
data_root: data/ensembl

builds:
  GRCh38:
    gtf: "Homo_sapiens.GRCh38.{release}.gtf"
    fasta: "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    ensembl_base: "https://ftp.ensembl.org/pub/release-{release}"
```

### Environment Variables

- `SS_SPECIES`: Override species
- `SS_BUILD`: Override build
- `SS_RELEASE`: Override release
- `SS_DATA_ROOT`: Override data root directory
- `SS_GTF_PATH`: Explicit GTF file path
- `SS_FASTA_PATH`: Explicit FASTA file path

## Validation

Pre-flight validation ensures genes have splice sites before running expensive workflows:

```python
from meta_spliceai.system.genomic_resources.validators import validate_gene_selection

valid_genes, invalid_summary = validate_gene_selection(
    gene_ids=['ENSG00000142611', 'ENSG00000290712'],
    data_dir=Path('data/ensembl'),
    min_splice_sites=1,
    fail_on_invalid=False,
    verbose=True
)
```

**Integrated into `incremental_builder.py`**: Automatically validates genes when `--run-workflow` is specified.

## Testing

### Test the Refactored Code

```python
# Test derivation
from meta_spliceai.system.genomic_resources import GenomicDataDeriver

deriver = GenomicDataDeriver(verbosity=2)

# Test individual functions
result = deriver.derive_gene_annotations(force_overwrite=True)
assert result['success'], result['error']

result = deriver.derive_splice_sites(force_overwrite=True)
assert result['success'], result['error']
```

### Test CLI

```bash
# Test audit
python -m meta_spliceai.system.genomic_resources.cli audit

# Test derive (dry run by checking existing files)
python -m meta_spliceai.system.genomic_resources.cli derive --annotations -v
```

## Migration Guide

### For Workflow Developers

**Current Code** (no changes needed):
```python
from meta_spliceai.splice_engine.meta_models.workflows.data_preparation import (
    prepare_gene_annotations
)

result = prepare_gene_annotations(local_dir, gtf_file, do_extract=True)
```

**Future Code** (when ready to migrate):
```python
from meta_spliceai.system.genomic_resources import GenomicDataDeriver

deriver = GenomicDataDeriver(data_dir=local_dir)
result = deriver.derive_gene_annotations(force_overwrite=True)
```

### For New Workflows

Use the refactored code from the start:

```python
from meta_spliceai.system.genomic_resources import GenomicDataDeriver

# Create deriver with your data directory
deriver = GenomicDataDeriver(
    data_dir="path/to/data",
    verbosity=1
)

# Derive what you need
annotations = deriver.derive_gene_annotations()
splice_sites = deriver.derive_splice_sites(consensus_window=2)
sequences = deriver.derive_genomic_sequences(mode='gene')
```

## Benefits of Refactored Design

1. **Maintainability**: Centralized data preparation logic
2. **Reusability**: Can be used across multiple workflows
3. **Testability**: Easier to test individual components
4. **Consistency**: Uniform interface and error handling
5. **Flexibility**: Easy to extend with new derivation methods
6. **Documentation**: Better documented with examples
7. **Configuration**: YAML-based configuration vs. hardcoded paths

## Future Enhancements

Potential additions to the refactored system:

1. **Caching**: Intelligent caching of derived datasets
2. **Parallel Processing**: Multi-threaded derivation for large datasets
3. **Incremental Updates**: Update only changed portions of datasets
4. **Version Control**: Track dataset versions and provenance
5. **Quality Metrics**: Automatic quality checks on derived data
6. **Format Conversion**: Convert between different file formats
7. **Cloud Storage**: Support for S3/GCS/Azure storage backends

## References

- Original implementation: `meta_spliceai/splice_engine/meta_models/workflows/data_preparation.py`
- Workflow integration: `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py`
- Configuration: `configs/genomic_resources.yaml`
- Documentation: `meta_spliceai/system/docs/GENOMIC_RESOURCES_SETUP_SUMMARY.md`
