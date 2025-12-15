# Schema Standardization for Genomic Datasets

## Overview

The `schema.py` module provides a formal, reusable solution for standardizing column names across genomic datasets. This ensures consistency throughout the system and prevents bugs caused by synonymous column names.

## Problem Statement

Different parts of the system may use different names for the same concept:
- `site_type` vs `splice_type` (splice site annotations)
- `seqname` vs `chrom` (chromosome names)
- `gene_type` vs `gene_biotype` (gene classification)

This inconsistency can lead to:
- ❌ Silent failures (code looking for wrong column name)
- ❌ Duplicate logic (each module handling renaming separately)
- ❌ Maintenance burden (updating column names in multiple places)

## Solution

A centralized schema standardization module that:
- ✅ Defines standard column names for each dataset type
- ✅ Provides reusable functions for schema standardization
- ✅ Works with both Polars and Pandas DataFrames
- ✅ Is idempotent (safe to call multiple times)
- ✅ Is non-destructive (only renames, doesn't modify data)

## Usage

### Basic Usage

```python
from meta_spliceai.system.genomic_resources import standardize_splice_sites_schema
import polars as pl

# Load splice sites with non-standard column names
ss_df = pl.read_csv('splice_sites.tsv', separator='\t')
# Columns: ['chrom', 'position', 'site_type', 'gene_id']  # ← site_type

# Standardize schema
ss_df = standardize_splice_sites_schema(ss_df, verbose=True)
# Columns: ['chrom', 'position', 'splice_type', 'gene_id']  # ← splice_type
```

### In Workflows

```python
from meta_spliceai.system.genomic_resources import standardize_splice_sites_schema

# After loading splice site annotations
ss_annotations_df = load_splice_sites(file_path)

# Standardize schema for consistency
ss_annotations_df = standardize_splice_sites_schema(
    ss_annotations_df,
    verbose=(verbosity >= 2)
)

# Now safe to use with evaluation code that expects 'splice_type'
evaluate_predictions(predictions, ss_annotations_df)
```

### Standardize Multiple Datasets

```python
from meta_spliceai.system.genomic_resources import standardize_all_schemas

# Standardize all datasets at once
result = standardize_all_schemas(
    splice_sites=ss_df,
    gene_features=gf_df,
    transcript_features=tf_df,
    verbose=True
)

standardized_ss = result['splice_sites']
standardized_gf = result['gene_features']
standardized_tf = result['transcript_features']
```

## Standard Column Mappings

### Splice Sites

| Non-Standard | Standard | Reason |
|--------------|----------|--------|
| `site_type` | `splice_type` | GTF convention → biological terminology |
| `type` | `splice_type` | Generic → specific |

### Gene Features

| Non-Standard | Standard | Reason |
|--------------|----------|--------|
| `seqname` | `chrom` | GTF convention → genomics convention |
| `gene_type` | `gene_biotype` | Alternative naming |
| `biotype` | `gene_biotype` | Short form → full form |

### Transcript Features

| Non-Standard | Standard | Reason |
|--------------|----------|--------|
| `seqname` | `chrom` | GTF convention → genomics convention |
| `transcript_type` | `transcript_biotype` | Alternative naming |
| `biotype` | `transcript_biotype` | Short form → full form |

### Exon Features

| Non-Standard | Standard | Reason |
|--------------|----------|--------|
| `seqname` | `chrom` | GTF convention → genomics convention |

## API Reference

### Core Functions

#### `standardize_splice_sites_schema(df, inplace=False, verbose=False)`

Standardize splice site annotation column names.

**Parameters:**
- `df`: Polars or Pandas DataFrame
- `inplace`: If True, modify DataFrame in place (Pandas only)
- `verbose`: If True, print renaming operations

**Returns:** Standardized DataFrame

**Example:**
```python
ss_df = standardize_splice_sites_schema(ss_df, verbose=True)
# [schema] Standardizing splice_sites columns:
#   site_type → splice_type
```

#### `standardize_gene_features_schema(df, inplace=False, verbose=False)`

Standardize gene feature column names.

#### `standardize_transcript_features_schema(df, inplace=False, verbose=False)`

Standardize transcript feature column names.

#### `standardize_exon_features_schema(df, inplace=False, verbose=False)`

Standardize exon feature column names.

#### `standardize_all_schemas(**datasets, verbose=False)`

Standardize multiple datasets at once.

**Parameters:**
- `splice_sites`: Optional splice sites DataFrame
- `gene_features`: Optional gene features DataFrame
- `transcript_features`: Optional transcript features DataFrame
- `exon_features`: Optional exon features DataFrame
- `verbose`: If True, print renaming operations

**Returns:** Dictionary with standardized DataFrames

### Utility Functions

#### `get_standard_column_mapping(schema_type)`

Get the standard column mapping for a schema type.

**Parameters:**
- `schema_type`: One of 'splice_sites', 'gene_features', 'transcript_features', 'exon_features'

**Returns:** Dictionary mapping non-standard to standard column names

**Example:**
```python
mapping = get_standard_column_mapping('splice_sites')
# {'site_type': 'splice_type', 'type': 'splice_type'}
```

#### `print_standard_schemas()`

Print all standard schema mappings for reference.

**Example:**
```python
from meta_spliceai.system.genomic_resources import print_standard_schemas
print_standard_schemas()
```

**Output:**
```
================================================================================
STANDARD GENOMIC DATASET SCHEMAS
================================================================================

Splice Sites:
----------------------------------------
  site_type            → splice_type
  type                 → splice_type

Gene Features:
----------------------------------------
  seqname              → chrom
  gene_type            → gene_biotype
  biotype              → gene_biotype

...
```

## Design Principles

### 1. Idempotent

Safe to call multiple times:
```python
df = standardize_splice_sites_schema(df)
df = standardize_splice_sites_schema(df)  # Safe - no effect
df = standardize_splice_sites_schema(df)  # Still safe
```

### 2. Non-Destructive

Only renames columns, doesn't modify data:
```python
# Before: ['chrom', 'site_type', 'gene_id']
df = standardize_splice_sites_schema(df)
# After:  ['chrom', 'splice_type', 'gene_id']
# All data preserved, only column name changed
```

### 3. Conflict-Aware

Won't rename if target column already exists:
```python
# DataFrame has both 'site_type' and 'splice_type'
df = standardize_splice_sites_schema(df)
# No renaming - avoids conflict
```

### 4. Framework-Agnostic

Works with both Polars and Pandas:
```python
# Polars
pl_df = standardize_splice_sites_schema(pl_df)

# Pandas
pd_df = standardize_splice_sites_schema(pd_df)
```

## Integration Points

### Where to Use Schema Standardization

1. **After loading data files**:
   ```python
   ss_df = pl.read_csv('splice_sites.tsv', separator='\t')
   ss_df = standardize_splice_sites_schema(ss_df)
   ```

2. **In data preparation workflows**:
   ```python
   # In prepare_splice_site_annotations()
   ss_df = extract_splice_sites(gtf_file)
   ss_df = standardize_splice_sites_schema(ss_df)
   return ss_df
   ```

3. **Before evaluation/analysis**:
   ```python
   # Before calling evaluation functions
   ss_df = standardize_splice_sites_schema(ss_df)
   results = evaluate_predictions(predictions, ss_df)
   ```

4. **In data derivation**:
   ```python
   # In GenomicDataDeriver
   gene_features = extract_gene_features_from_gtf(gtf_file)
   gene_features = standardize_gene_features_schema(gene_features)
   ```

### Where NOT to Use

- ❌ **Before writing to files** - Keep original column names in files for compatibility
- ❌ **In file I/O functions** - Standardize after loading, not during
- ❌ **In GTF parsing** - GTF files should maintain their original schema

## Migration Guide

### Replacing Ad-Hoc Renaming

**Before:**
```python
# Ad-hoc renaming scattered throughout codebase
if 'site_type' in df.columns and 'splice_type' not in df.columns:
    df = df.rename({'site_type': 'splice_type'})
```

**After:**
```python
# Centralized, reusable solution
from meta_spliceai.system.genomic_resources import standardize_splice_sites_schema
df = standardize_splice_sites_schema(df)
```

### Benefits of Migration

1. **Consistency**: All code uses the same standardization logic
2. **Maintainability**: Update mappings in one place
3. **Discoverability**: Clear API for schema standardization
4. **Testing**: Single point to test schema logic
5. **Documentation**: Centralized documentation of standard schemas

## Testing

### Unit Tests

```python
import polars as pl
from meta_spliceai.system.genomic_resources import standardize_splice_sites_schema

def test_standardize_splice_sites():
    """Test splice site schema standardization."""
    # Create DataFrame with non-standard column
    df = pl.DataFrame({
        'chrom': ['1'],
        'position': [1000],
        'site_type': ['donor'],  # Non-standard
        'gene_id': ['ENSG00000000001']
    })
    
    # Standardize
    result = standardize_splice_sites_schema(df)
    
    # Verify
    assert 'splice_type' in result.columns
    assert 'site_type' not in result.columns
    assert result['splice_type'][0] == 'donor'

def test_idempotent():
    """Test that standardization is idempotent."""
    df = pl.DataFrame({
        'chrom': ['1'],
        'site_type': ['donor']
    })
    
    # Apply multiple times
    df1 = standardize_splice_sites_schema(df)
    df2 = standardize_splice_sites_schema(df1)
    df3 = standardize_splice_sites_schema(df2)
    
    # Should all be the same
    assert df1.columns == df2.columns == df3.columns
```

### Integration Tests

```python
def test_workflow_integration():
    """Test schema standardization in full workflow."""
    # Load data with non-standard schema
    ss_df = load_splice_sites('data/splice_sites.tsv')
    
    # Standardize
    ss_df = standardize_splice_sites_schema(ss_df)
    
    # Should work with evaluation code
    results = evaluate_predictions(predictions, ss_df)
    
    # Verify results
    assert results['success']
    assert len(results['positions']) > 0
```

## Troubleshooting

### Issue: Column Not Being Renamed

**Symptom**: `site_type` still exists after standardization

**Possible Causes:**
1. Target column `splice_type` already exists (conflict avoidance)
2. Source column name doesn't match mapping exactly (case-sensitive)

**Solution:**
```python
# Check what columns exist
print(df.columns)

# Check if target already exists
if 'splice_type' in df.columns:
    print("Target column already exists - no renaming needed")

# Verify mapping
from meta_spliceai.system.genomic_resources import get_standard_column_mapping
mapping = get_standard_column_mapping('splice_sites')
print(f"Mapping: {mapping}")
```

### Issue: Wrong Schema Type

**Symptom**: `ValueError: Unknown schema type`

**Solution:**
```python
# Valid schema types
valid_types = ['splice_sites', 'gene_features', 'transcript_features', 'exon_features']

# Use correct type
mapping = get_standard_column_mapping('splice_sites')  # ✅
mapping = get_standard_column_mapping('splice_site')   # ❌ (singular)
```

## Future Enhancements

### Planned Features

1. **Schema Validation**:
   ```python
   validate_schema(df, schema_type='splice_sites', strict=True)
   # Raises error if required columns are missing
   ```

2. **Schema Inference**:
   ```python
   schema_type = infer_schema_type(df)
   # Returns 'splice_sites' based on column patterns
   ```

3. **Custom Mappings**:
   ```python
   custom_mapping = {'my_col': 'standard_col'}
   df = standardize_schema(df, custom_mapping)
   ```

4. **Schema Versioning**:
   ```python
   df = standardize_schema(df, version='v2')
   # Support for schema evolution
   ```

## References

- [Schema Module](schema.py)
- [Genomic Resources README](README.md)
- [Critical Bug Fix: site_type Column](../../docs/testing/CRITICAL_BUG_FIX_SITE_TYPE_COLUMN.md)
- [Splice Prediction Workflow](../../splice_engine/meta_models/workflows/splice_prediction_workflow.py)

## Contributing

### Adding New Mappings

To add a new column mapping:

1. **Update the mapping dictionary** in `schema.py`:
   ```python
   SPLICE_SITE_COLUMN_MAPPING = {
       'site_type': 'splice_type',
       'new_synonym': 'standard_name',  # Add here
   }
   ```

2. **Document the mapping** in this file

3. **Add tests** for the new mapping

4. **Update affected code** to use standardization

### Guidelines

- **Standard names should be**:
  - Clear and unambiguous
  - Consistent with biological/genomics terminology
  - Widely used in the field

- **Mappings should be**:
  - One-to-one (one synonym → one standard)
  - Non-conflicting (no circular mappings)
  - Well-documented (explain the reason)

---

**Last Updated**: November 4, 2025  
**Version**: 1.0.0  
**Status**: ✅ Production Ready

