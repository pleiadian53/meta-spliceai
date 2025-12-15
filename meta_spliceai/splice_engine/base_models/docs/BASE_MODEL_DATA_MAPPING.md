# Base Model → Data Directory Mapping

**Date**: November 1, 2025  
**Purpose**: Clear mapping between base models and their data directories

---

## Overview

Each base model requires specific genomic data that matches its training specifications. The directory structure reflects this relationship:

```
data/<annotation_source>/<build>/
```

Where:
- `annotation_source` = Where the annotations came from (ensembl, mane, gencode)
- `build` = Genome build version (GRCh37, GRCh38, etc.)

---

## Base Model Mappings

### SpliceAI

**Training Specifications**:
- Genome Build: **GRCh37** (hg19)
- Annotations: **GENCODE V24lift37**
- Annotation Source: Ensembl-compatible

**Data Directory**: `data/ensembl/GRCh37/`

**Required Files**:
```
data/ensembl/GRCh37/
├── Homo_sapiens.GRCh37.87.gtf              # Ensembl GTF
├── Homo_sapiens.GRCh37.dna.primary_assembly.fa  # Reference genome
├── splice_sites_enhanced.tsv                # Derived splice sites
├── annotations.db                           # Gene annotations
├── gene_features.tsv                        # Gene-level features
└── gene_sequence_*.parquet                  # Extracted sequences
```

**Usage**:
```python
from meta_spliceai.system.genomic_resources import Registry

# For SpliceAI
registry = Registry(build='GRCh37')
# Resolves to: data/ensembl/GRCh37/
```

---

### OpenSpliceAI

**Training Specifications**:
- Genome Build: **GRCh38.p14**
- Annotations: **MANE v1.3 RefSeq**
- Annotation Source: MANE (NCBI RefSeq)

**Data Directory**: `data/mane/GRCh38/`

**Required Files**:
```
data/mane/GRCh38/
├── MANE.GRCh38.v1.3.refseq_genomic.gff      # MANE GFF3 (not GTF!)
├── GCF_000001405.40_GRCh38.p14_genomic.fna  # RefSeq reference genome
├── splice_sites_enhanced.tsv                # Derived splice sites
├── annotations.db                           # Gene annotations
├── gene_features.tsv                        # Gene-level features
└── gene_sequence_*.parquet                  # Extracted sequences
```

**Usage**:
```python
from meta_spliceai.system.genomic_resources import Registry

# For OpenSpliceAI
registry = Registry(build='GRCh38_MANE')
# Resolves to: data/mane/GRCh38/
```

**Important Notes**:
- MANE uses **GFF3 format**, not GTF
- Different transcript set than Ensembl
- Higher confidence, but fewer transcripts

---

## Why This Matters

### 1. Genome Build Mismatch = Performance Drop

**Example**: Using SpliceAI with GRCh38 data
- Expected PR-AUC: 0.97
- Actual PR-AUC: 0.541
- **Performance drop: 44%**

**Solution**: Match the genome build to the model's training data

### 2. Annotation Source Mismatch = Wrong Transcripts

Different annotation sources have different transcript sets:
- **Ensembl**: Comprehensive, includes many isoforms
- **MANE**: High-confidence, one representative transcript per gene
- **GENCODE**: Similar to Ensembl, slightly different curation

**Solution**: Match the annotation source to the model's training data

### 3. Directory Structure Enforces Correctness

The directory structure makes it **impossible** to accidentally mix:
```
data/ensembl/GRCh37/  ← SpliceAI data
data/mane/GRCh38/     ← OpenSpliceAI data
```

They're physically separated, preventing cross-contamination.

---

## Configuration

All mappings are defined in `configs/genomic_resources.yaml`:

```yaml
base_models:
  spliceai:
    annotation_source: "ensembl"  # → data/ensembl/GRCh37/
    training_build: "GRCh37"
    
  openspliceai:
    annotation_source: "mane"     # → data/mane/GRCh38/
    training_build: "GRCh38"
```

---

## Adding New Base Models

To add a new base model:

1. **Identify training specifications**:
   - Genome build (GRCh37, GRCh38, etc.)
   - Annotation source (Ensembl, MANE, GENCODE, etc.)
   - Annotation version

2. **Add to config** (`genomic_resources.yaml`):
```yaml
base_models:
  new_model:
    name: "New Model"
    training_build: "GRCh38"
    training_annotation: "Ensembl 110"
    annotation_source: "ensembl"  # → data/ensembl/GRCh38/
```

3. **Download data** to correct directory:
```bash
# For Ensembl-based model
data/ensembl/GRCh38/

# For MANE-based model
data/mane/GRCh38/

# For GENCODE-based model
data/gencode/GRCh38/
```

4. **Generate derived data**:
```bash
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh38 \
  --dataset splice_sites
```

---

## Quick Reference

| Base Model | Build | Annotation Source | Data Directory |
|------------|-------|-------------------|----------------|
| SpliceAI | GRCh37 | Ensembl (GENCODE V24) | `data/ensembl/GRCh37/` |
| OpenSpliceAI | GRCh38 | MANE v1.3 RefSeq | `data/mane/GRCh38/` |

---

## Testing

Verify correct data directory resolution:

```python
from meta_spliceai.system.genomic_resources import Registry

# Test SpliceAI
r_spliceai = Registry(build='GRCh37')
print(f"SpliceAI data: {r_spliceai.data_dir}")
# Expected: /path/to/data/ensembl/GRCh37

# Test OpenSpliceAI
r_openspliceai = Registry(build='GRCh38_MANE')
print(f"OpenSpliceAI data: {r_openspliceai.data_dir}")
# Expected: /path/to/data/mane/GRCh38
```

---

## Summary

**Key Principle**: The data directory structure directly reflects the base model's training data specifications.

**Structure**: `data/<annotation_source>/<build>/`

**Benefits**:
- ✅ Clear separation between different data sources
- ✅ Prevents accidental mixing of incompatible data
- ✅ Self-documenting (path tells you the source)
- ✅ Easy to add new base models
- ✅ Enforces correctness through physical separation

**Remember**: Always match the genome build AND annotation source to the base model's training data!



