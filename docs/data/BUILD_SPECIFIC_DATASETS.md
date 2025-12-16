# Build-Specific Genomic Datasets

## Overview

All genomic annotation-derived datasets are **build-specific** and must be organized under their respective build directories.

## Directory Structure

```
data/
├── ensembl/
│   ├── GRCh37/
│   │   ├── gene_features.tsv          # Build-specific
│   │   ├── splice_sites_enhanced.tsv  # Build-specific
│   │   ├── exon_features.tsv          # Build-specific (if generated)
│   │   ├── transcript_features.tsv    # Build-specific (if generated)
│   │   ├── Homo_sapiens.GRCh37.87.gtf
│   │   └── Homo_sapiens.GRCh37.dna.primary_assembly.fa
│   └── GRCh38/
│       ├── gene_features.tsv          # Build-specific
│       ├── splice_sites_enhanced.tsv  # Build-specific
│       ├── Homo_sapiens.GRCh38.112.gtf
│       └── Homo_sapiens.GRCh38.dna.primary_assembly.fa
└── mane/
    └── GRCh38/
        ├── gene_features.tsv          # Build-specific
        ├── splice_sites_enhanced.tsv  # Build-specific
        ├── MANE.GRCh38.v1.3.refseq_genomic.gff
        └── GCF_000001405.40_GRCh38.p14_genomic.fna
```

## Key Datasets

### 1. `gene_features.tsv`
**Purpose**: Gene-level metadata including biotype information

**Columns**:
- `gene_id`: Ensembl gene ID
- `gene_name`: Gene symbol
- `gene_type`: Gene biotype (protein_coding, lincRNA, etc.)
- `gene_length`: Gene length in base pairs
- `chrom`: Chromosome
- `start`, `end`: Genomic coordinates
- `strand`: +/-

**Generation**: Automatically extracted from GTF during the base model pass workflow (`splice_prediction_workflow.py`).

The workflow includes a data preparation step (step 1.5) that automatically derives gene features if they don't already exist.

**Manual Generation** (if needed):
```bash
python scripts/setup/generate_grch37_gene_features.py
```

**Usage**: 
- Biotype-specific gene sampling
- Gene filtering by type
- Gene metadata lookup

### 2. `splice_sites_enhanced.tsv`
**Purpose**: Comprehensive splice site annotations

**Columns**:
- `chrom`: Chromosome
- `position`: Genomic position
- `site_type`: donor/acceptor
- `gene_id`: Associated gene
- `transcript_id`: Associated transcript
- Additional context columns

**Generation**: Extracted from GTF during data preparation
```bash
python scripts/setup/regenerate_grch37_splice_sites_complete.py
```

**Usage**:
- Ground truth labels for evaluation
- Gene filtering (genes with valid splice sites)
- Coordinate alignment verification

### 3. `exon_features.tsv` (Optional)
**Purpose**: Exon-level annotations

**Generation**: Extracted from GTF when needed

### 4. `transcript_features.tsv` (Optional)
**Purpose**: Transcript-level annotations

**Generation**: Extracted from GTF when needed

## Why Build-Specific?

Different genome builds have:
1. **Different coordinates**: Same gene has different positions in GRCh37 vs GRCh38
2. **Different annotations**: Gene models evolve between releases
3. **Different gene sets**: Some genes added/removed between versions
4. **Different biotypes**: Gene classifications may change

## Migration from Old Structure

**Old (incorrect)**:
```
data/ensembl/gene_features.tsv  # ❌ Not build-specific
data/ensembl/GRCh37/...
data/ensembl/GRCh38/...
```

**New (correct)**:
```
data/ensembl/GRCh37/gene_features.tsv  # ✅ Build-specific
data/ensembl/GRCh38/gene_features.tsv  # ✅ Build-specific
```

## Base Model Data Mapping

| Base Model | Annotation Source | Build | Data Directory |
|------------|------------------|-------|----------------|
| SpliceAI | Ensembl | GRCh37 | `data/ensembl/GRCh37/` |
| OpenSpliceAI | MANE RefSeq | GRCh38 | `data/mane/GRCh38/` |

## Code Implications

### Registry Resolution
The `Registry` class automatically resolves to the correct build-specific directory:

```python
from meta_spliceai.system.genomic_resources import Registry

registry = Registry(build='GRCh37', release='87')
gene_features_file = registry.data_dir / 'gene_features.tsv'
# Resolves to: data/ensembl/GRCh37/gene_features.tsv
```

### No Fallback Logic
Scripts should **not** have fallback logic to search multiple locations. If a build-specific file is missing, it should fail fast with a clear error message:

```python
# ✅ GOOD: Fail fast
gene_features_file = registry.data_dir / 'gene_features.tsv'
if not gene_features_file.exists():
    raise FileNotFoundError(
        f"gene_features.tsv not found at {gene_features_file}\n"
        f"This file is build-specific and should exist at data/<source>/<build>/\n"
        f"Please generate it first."
    )

# ❌ BAD: Fallback logic masks problems
gene_features_file = registry.data_dir / 'gene_features.tsv'
if not gene_features_file.exists():
    gene_features_file = registry.data_dir.parent / 'gene_features.tsv'  # Wrong!
```

## Generation Scripts

### For GRCh37 (Ensembl)
```bash
# Gene features
python scripts/setup/generate_grch37_gene_features.py

# Splice sites
python scripts/setup/regenerate_grch37_splice_sites_complete.py
```

### For GRCh38 (Ensembl)
```bash
# Use GenomicDataDeriver with build='GRCh38', release='112'
```

### For GRCh38 (MANE)
```bash
# Will be implemented for OpenSpliceAI integration
```

## Verification

Check that all required files exist for a build:

```bash
# For GRCh37
ls -lh data/ensembl/GRCh37/gene_features.tsv
ls -lh data/ensembl/GRCh37/splice_sites_enhanced.tsv

# For GRCh38
ls -lh data/ensembl/GRCh38/gene_features.tsv
ls -lh data/ensembl/GRCh38/splice_sites_enhanced.tsv
```

## Related Documentation

- [Annotation Source Directory Structure](../development/ANNOTATION_SOURCE_DIRECTORY_STRUCTURE.md)
- [Base Model Data Mapping](../base_models/BASE_MODEL_DATA_MAPPING.md)
- [Genome Build Compatibility](../base_models/GENOME_BUILD_COMPATIBILITY.md)

