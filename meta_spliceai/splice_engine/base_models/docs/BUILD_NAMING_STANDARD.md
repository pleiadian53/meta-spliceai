# Standardized Build Naming Convention

**Date**: 2025-11-11  
**Status**: **ADOPTED**

## Overview

This document defines the standard naming convention for genomic builds and annotation sources in the meta-spliceai project.

## Motivation

**Problem**: Ad-hoc build naming leads to confusion and doesn't scale:
```python
# Ad-hoc (before)
if source == 'mane' and build == 'GRCh38':
    build_name = 'GRCh38_MANE'
elif source == 'ensembl' and build == 'GRCh38':
    build_name = 'GRCh38_Ensembl'  # But what about this?
```

**Solution**: Standardized naming convention that is:
- ✅ Generalizable to any source
- ✅ Backward compatible
- ✅ Easy to parse
- ✅ Self-documenting

## Standard Convention

### Format

```
{build}_{source}
```

Where:
- `{build}` = Genomic build (GRCh37, GRCh38, T2T_CHM13, etc.)
- `{source}` = Annotation source (Ensembl, MANE, GENCODE, RefSeq, etc.)

### Special Case

**GRCh37** (without suffix) = Ensembl GRCh37 (historical default)

## Examples

| Source | Build | Standard Name | Description |
|--------|-------|---------------|-------------|
| Ensembl | GRCh37 | `GRCh37` | Default (backward compatible) |
| MANE | GRCh38 | `GRCh38_MANE` | MANE Select on GRCh38 |
| Ensembl | GRCh38 | `GRCh38_Ensembl` | Ensembl on GRCh38 |
| GENCODE | GRCh38 | `GRCh38_GENCODE` | GENCODE on GRCh38 |
| RefSeq | GRCh37 | `GRCh37_RefSeq` | RefSeq on GRCh37 |
| RefSeq | GRCh38 | `GRCh38_RefSeq` | RefSeq on GRCh38 |
| Ensembl | T2T_CHM13 | `T2T_CHM13_Ensembl` | Ensembl on T2T |

## Supported Sources

| Source | Description |
|--------|-------------|
| `Ensembl` | Ensembl genome annotations |
| `MANE` | MANE Select (Matched Annotation from NCBI and EMBL-EBI) |
| `GENCODE` | GENCODE comprehensive gene annotation |
| `RefSeq` | NCBI Reference Sequence Database |
| `UCSC` | UCSC Genome Browser annotations |

## Supported Builds

| Build | Description |
|-------|-------------|
| `GRCh37` | Genome Reference Consortium Human Build 37 |
| `GRCh38` | Genome Reference Consortium Human Build 38 |
| `hg19` | UCSC Human Genome Build 19 (equivalent to GRCh37) |
| `hg38` | UCSC Human Genome Build 38 (equivalent to GRCh38) |
| `T2T_CHM13` | Telomere-to-Telomere CHM13 assembly |

## Usage

### Python API

```python
from meta_spliceai.system.genomic_resources import get_standardized_build_name

# Get standardized build name
build_name = get_standardized_build_name('mane', 'GRCh38')
# Returns: 'GRCh38_MANE'

build_name = get_standardized_build_name('ensembl', 'GRCh38')
# Returns: 'GRCh38_Ensembl'

build_name = get_standardized_build_name('ensembl', 'GRCh37')
# Returns: 'GRCh37' (default case)
```

### Parsing Build Names

```python
from meta_spliceai.system.genomic_resources import parse_build_name

# Parse a build name
info = parse_build_name('GRCh38_MANE')
# Returns: {'build': 'GRCh38', 'source': 'mane'}

info = parse_build_name('GRCh37')
# Returns: {'build': 'GRCh37', 'source': 'ensembl'}
```

### Validation

```python
from meta_spliceai.system.genomic_resources import validate_build_name

# Validate a build name
is_valid = validate_build_name('GRCh38_MANE')
# Returns: True

is_valid = validate_build_name('InvalidBuild')
# Returns: False
```

### Getting Descriptions

```python
from meta_spliceai.system.genomic_resources import get_build_description

# Get human-readable description
desc = get_build_description('GRCh38_MANE')
# Returns: 'Genome Reference Consortium Human Build 38 (MANE)'

desc = get_build_description('GRCh37')
# Returns: 'Genome Reference Consortium Human Build 37 (Ensembl)'
```

## Integration

### GeneSelector

The `GeneSelector` now uses standardized build naming:

```python
from meta_spliceai.system.genomic_resources import GeneSelector

selector = GeneSelector()

# Automatically uses standardized naming
result = selector.sample_genes_for_comparison(
    source1='ensembl',
    source1_build='GRCh37',  # → 'GRCh37'
    source2='mane',
    source2_build='GRCh38',  # → 'GRCh38_MANE'
    ...
)
```

### Registry

The `Registry` class accepts standardized build names:

```python
from meta_spliceai.system.genomic_resources import Registry

# Using standardized names
registry1 = Registry(build='GRCh37', release='87')
registry2 = Registry(build='GRCh38_MANE', release='1.3')
registry3 = Registry(build='GRCh38_Ensembl', release='110')
```

## Benefits

### 1. Generalizability

**Before** (ad-hoc):
```python
if source == 'mane' and build == 'GRCh38':
    build_name = 'GRCh38_MANE'
# What about other sources? Need more if-statements!
```

**After** (standardized):
```python
build_name = get_standardized_build_name(source, build)
# Works for any source!
```

### 2. Self-Documenting

Build names are self-explanatory:
- `GRCh38_MANE` → GRCh38 with MANE annotations
- `GRCh38_Ensembl` → GRCh38 with Ensembl annotations
- `GRCh37_RefSeq` → GRCh37 with RefSeq annotations

### 3. Easy to Parse

```python
# Extract build and source
parsed = parse_build_name('GRCh38_GENCODE')
build = parsed['build']   # 'GRCh38'
source = parsed['source']  # 'gencode'
```

### 4. Backward Compatible

- `GRCh37` (without suffix) still works
- Existing code continues to function
- Gradual migration possible

### 5. Extensible

Easy to add new sources:
```python
# Add new source to SUPPORTED_SOURCES
SUPPORTED_SOURCES['custom'] = 'CustomAnnotation'

# Use immediately
build_name = get_standardized_build_name('custom', 'GRCh38')
# Returns: 'GRCh38_CustomAnnotation'
```

## Migration Guide

### For Existing Code

**Old code** (still works):
```python
registry = Registry(build='GRCh37', release='87')
```

**New code** (recommended):
```python
from meta_spliceai.system.genomic_resources import get_standardized_build_name

build = get_standardized_build_name('ensembl', 'GRCh37')
registry = Registry(build=build, release='87')
```

### For New Base Models

When adding a new base model with a different annotation source:

```python
# Example: New base model using GENCODE on GRCh38
from meta_spliceai.system.genomic_resources import get_standardized_build_name

# Get standardized build name
build_name = get_standardized_build_name('gencode', 'GRCh38')
# Returns: 'GRCh38_GENCODE'

# Use in Registry
registry = Registry(build=build_name, release='44')

# Use in GeneSelector
selector = GeneSelector()
result = selector.sample_genes_for_comparison(
    source1='ensembl',
    source1_build='GRCh37',
    source2='gencode',
    source2_build='GRCh38',
    ...
)
```

## File Organization

### Data Directory Structure

Following the standard naming:

```
data/
├── ensembl/
│   ├── GRCh37/           # Default (Ensembl GRCh37)
│   │   ├── gene_features.tsv
│   │   ├── splice_sites_enhanced.tsv
│   │   └── ...
│   └── GRCh38_Ensembl/   # Ensembl on GRCh38
│       ├── gene_features.tsv
│       └── ...
├── mane/
│   └── GRCh38_MANE/      # MANE on GRCh38
│       ├── gene_features.tsv
│       └── ...
├── gencode/
│   └── GRCh38_GENCODE/   # GENCODE on GRCh38
│       ├── gene_features.tsv
│       └── ...
└── refseq/
    ├── GRCh37_RefSeq/    # RefSeq on GRCh37
    └── GRCh38_RefSeq/    # RefSeq on GRCh38
```

**Note**: The source directory name (e.g., `ensembl/`, `mane/`) is for organization. The actual build name includes the source suffix (e.g., `GRCh38_MANE`).

## Implementation

### Module

**File**: `meta_spliceai/system/genomic_resources/build_naming.py`

**Functions**:
- `get_standardized_build_name(source, build)` - Get standardized name
- `parse_build_name(build_name)` - Parse into components
- `get_build_description(build_name)` - Get human-readable description
- `validate_build_name(build_name)` - Validate format
- `get_build_for_base_model(base_model)` - Get build for known models

### Constants

- `SUPPORTED_SOURCES` - Dictionary of supported annotation sources
- `SUPPORTED_BUILDS` - Dictionary of supported genomic builds

## Examples

### Adding a New Base Model

**Scenario**: Add a new base model using RefSeq annotations on GRCh38

```python
from meta_spliceai.system.genomic_resources import (
    get_standardized_build_name,
    Registry
)

# 1. Get standardized build name
build_name = get_standardized_build_name('refseq', 'GRCh38')
# Returns: 'GRCh38_RefSeq'

# 2. Create registry
registry = Registry(build=build_name, release='110')

# 3. Use in gene selection
from meta_spliceai.system.genomic_resources import GeneSelector

selector = GeneSelector()
result = selector.sample_genes_for_comparison(
    source1='ensembl',
    source1_build='GRCh37',
    source2='refseq',
    source2_build='GRCh38',
    ...
)
```

### Comparing Multiple Sources

```python
sources = [
    ('ensembl', 'GRCh37'),
    ('mane', 'GRCh38'),
    ('ensembl', 'GRCh38'),
    ('gencode', 'GRCh38'),
]

for source, build in sources:
    build_name = get_standardized_build_name(source, build)
    print(f"{source:10s} + {build:10s} → {build_name}")

# Output:
# ensembl    + GRCh37     → GRCh37
# mane       + GRCh38     → GRCh38_MANE
# ensembl    + GRCh38     → GRCh38_Ensembl
# gencode    + GRCh38     → GRCh38_GENCODE
```

## Summary

✅ **Standard Adopted**: `{build}_{source}` format  
✅ **Backward Compatible**: `GRCh37` still works  
✅ **Generalizable**: Works for any source  
✅ **Self-Documenting**: Names are clear  
✅ **Easy to Parse**: Simple string splitting  
✅ **Extensible**: Easy to add new sources  

**Use the standardized naming convention for all new code!**

---

**Module**: `meta_spliceai.system.genomic_resources.build_naming`  
**Date Adopted**: 2025-11-11  
**Status**: Production ready




