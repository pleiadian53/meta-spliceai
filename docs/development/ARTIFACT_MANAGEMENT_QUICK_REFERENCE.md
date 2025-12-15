# Artifact Management - Quick Reference

## TL;DR

**Default mode: `test`** (safe, overwritable)  
**Production mode: `production`** (immutable, cached)

## Quick Start

### Test Mode (Default)
```python
config = SpliceAIConfig(
    mode='test',  # Optional - this is the default
    coverage='gene_subset',
    test_name='my_test',
    # ... other params
)
```
**Artifacts go to**: `data/ensembl/GRCh37/spliceai_eval/tests/my_test/`

### Production Mode
```python
config = SpliceAIConfig(
    mode='production',
    coverage='full_genome',
    # ... other params
)
```
**Artifacts go to**: `data/ensembl/GRCh37/spliceai_eval/meta_models/`

## Directory Structure

```
data/<source>/<build>/<base_model>_eval/
├── meta_models/           # Production (immutable)
├── tests/<test_name>/     # Test (overwritable)
├── training_data/         # Meta-model training datasets
└── models/                # Meta-model checkpoints
```

## Mode Comparison

| Feature | Test Mode | Production Mode |
|---------|-----------|-----------------|
| **Default?** | ✅ Yes | ❌ No |
| **Overwrite?** | ✅ Always | ❌ Never (unless force=True) |
| **Use case** | Development, testing | Training, inference |
| **Location** | `tests/<name>/` | `meta_models/` |
| **Auto-switch** | For gene_subset | For full_genome |

## Common Patterns

### Get Artifact Manager
```python
manager = config.get_artifact_manager()
manager.print_summary()
```

### Get Artifacts Directory
```python
artifacts_dir = manager.get_artifacts_dir(create=True)
```

### Check Overwrite Policy
```python
file_path = manager.get_artifact_path('my_artifact.tsv')
if manager.should_overwrite(file_path):
    df.write_csv(file_path, separator='\t')
```

### List Artifacts
```python
artifacts = manager.list_artifacts('*.tsv')
```

## Parameters

### `mode`
- **Type**: `str`
- **Default**: `"test"`
- **Options**: `"test"`, `"production"`
- **When to use production**: Full genome runs, meta-model training

### `coverage`
- **Type**: `str`
- **Default**: `"gene_subset"`
- **Options**: `"full_genome"`, `"chromosome"`, `"gene_subset"`
- **Auto-switches mode**: `full_genome` → `production`

### `test_name`
- **Type**: `Optional[str]`
- **Default**: Auto-generated timestamp
- **Example**: `"base_model_comprehensive_test"`
- **Only used in**: Test mode

## Examples

### Comprehensive Test
```python
config = SpliceAIConfig(
    mode='test',
    coverage='gene_subset',
    test_name='comprehensive_test',
    gtf_file='data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf',
    genome_fasta='data/ensembl/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa',
    eval_dir='results/comprehensive_test',
)
```

### Full Genome Production Run
```python
config = SpliceAIConfig(
    mode='production',  # Or auto-detected from coverage
    coverage='full_genome',
    gtf_file='data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf',
    genome_fasta='data/ensembl/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa',
    eval_dir='data/ensembl/GRCh37/spliceai_eval',
)
```

### Chromosome-Specific Test
```python
config = SpliceAIConfig(
    mode='test',
    coverage='chromosome',
    test_name='chr21_test',
    chromosomes=['21'],
    # ... other params
)
```

## Overwrite Decision Tree

```
Is force=True?
├─ Yes → Overwrite
└─ No
   ├─ File exists?
   │  ├─ No → Write
   │  └─ Yes
   │     ├─ Mode is test? → Overwrite
   │     └─ Mode is production? → Skip
   └─ File doesn't exist → Write
```

## Best Practices

1. **Use test mode for development** - Prevents accidental production overwrites
2. **Name your tests** - Makes it easier to find and manage artifacts
3. **Use production mode for full genome** - Auto-detected from `coverage='full_genome'`
4. **Check artifacts exist before inference** - Speeds up meta-model predictions
5. **Clean up old test artifacts** - Saves disk space

## Related Files

- **Implementation**: `meta_spliceai/system/artifact_manager.py`
- **Config**: `meta_spliceai/splice_engine/meta_models/core/data_types.py`
- **Documentation**: `docs/development/ARTIFACT_MANAGEMENT.md`

## Need Help?

See full documentation: `docs/development/ARTIFACT_MANAGEMENT.md`

