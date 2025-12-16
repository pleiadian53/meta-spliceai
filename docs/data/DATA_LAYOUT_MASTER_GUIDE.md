# Data Layout and Organization - Master Guide

**Last Updated**: November 21, 2025  
**Purpose**: Comprehensive guide to all data directories, datasets, and conventions in the Meta-SpliceAI project

---

## Quick Reference

| Data Type | Location | Used By | Build |
|-----------|----------|---------|-------|
| **SpliceAI Base Model** | `data/ensembl/GRCh37/` | SpliceAI predictions | GRCh37 |
| **OpenSpliceAI Base Model** | `data/mane/GRCh38/` | OpenSpliceAI predictions | GRCh38 |
| **Training Datasets** | `data/train_*` | Meta-model training | Various |
| **Test/Validation Datasets** | `data/test_*` | Model evaluation | Various |
| **Case Studies** | `meta_spliceai/splice_engine/case_studies/` | Variant analysis, disease studies | Various |
| **Pre-trained Models** | `data/models/` | Inference | N/A |

---

## Table of Contents

1. [Base Model Data Directories](#base-model-data-directories)
2. [Training and Evaluation Datasets](#training-and-evaluation-datasets)
3. [Case Study Data](#case-study-data)
4. [Pre-trained Model Weights](#pre-trained-model-weights)
5. [Output and Artifacts](#output-and-artifacts)
6. [Directory Naming Conventions](#directory-naming-conventions)
7. [Data Organization Principles](#data-organization-principles)

---

## Base Model Data Directories

### Overview

Base models require genome-specific reference data that matches their training specifications. The directory structure follows:

```
data/<annotation_source>/<build>/
```

### SpliceAI (GRCh37)

**Location**: `data/ensembl/GRCh37/`

**Purpose**: Reference data for SpliceAI base model predictions

**Structure**:
```
data/ensembl/GRCh37/
├── Homo_sapiens.GRCh37.87.gtf              # Ensembl gene annotations
├── Homo_sapiens.GRCh37.dna.primary_assembly.fa  # Reference genome FASTA
├── Homo_sapiens.GRCh37.dna.primary_assembly.fa.fai  # FASTA index
│
├── splice_sites_enhanced.tsv                # Extracted splice sites (derived)
├── annotations.db                           # SQLite gene annotations DB
├── gene_features.tsv                        # Gene-level features
├── gene_chromosome_map.tsv                  # Gene→chromosome mapping
├── chromosome_sizes.tsv                     # Chromosome lengths
│
├── gene_sequence_*.parquet                  # Extracted gene sequences per chromosome
│   ├── gene_sequence_1.parquet
│   ├── gene_sequence_2.parquet
│   └── ... (one per chromosome)
│
└── spliceai_eval/                           # SpliceAI evaluation artifacts
    ├── meta_models/                         # Meta-model specific outputs
    │   ├── analysis_sequences_*.tsv         # Training features per chromosome
    │   ├── error_analysis_*.tsv             # Prediction errors
    │   ├── splice_positions_*.tsv           # Splice site predictions
    │   └── inference/                       # Inference-specific outputs
    └── gene_manifest.json                   # Processing manifest
```

**Key Files**:
- **GTF**: Ensembl Release 87 gene annotations for GRCh37
- **Reference Genome**: From Ensembl, primary assembly (no patches)
- **Splice Sites**: Derived from GTF, enhanced with features
- **Gene Sequences**: Pre-extracted per chromosome for faster processing

**Usage**:
```python
from meta_spliceai.system.genomic_resources import Registry

# Load SpliceAI resources
registry = Registry(build='GRCh37')
# Automatically resolves to: data/ensembl/GRCh37/
```

**Documentation**: [`docs/base_models/GRCH37_SETUP_COMPLETE_GUIDE.md`](../base_models/GRCH37_SETUP_COMPLETE_GUIDE.md)

---

### OpenSpliceAI (GRCh38 MANE)

**Location**: `data/mane/GRCh38/`

**Purpose**: Reference data for OpenSpliceAI base model predictions

**Structure**:
```
data/mane/GRCh38/
├── MANE.GRCh38.v1.3.refseq_genomic.gff      # MANE RefSeq annotations (GFF3!)
├── GCF_000001405.40_GRCh38.p14_genomic.fna  # RefSeq reference genome
├── GCF_000001405.40_GRCh38.p14_genomic.fna.fai  # FASTA index
│
├── splice_sites_enhanced.tsv                # Extracted splice sites (derived)
├── annotations.db                           # SQLite gene annotations DB
├── gene_features.tsv                        # Gene-level features
├── gene_chromosome_map.tsv                  # Gene→chromosome mapping
├── chromosome_sizes.tsv                     # Chromosome lengths
│
├── gene_sequence_*.parquet                  # Extracted gene sequences per chromosome
│   ├── gene_sequence_1.parquet
│   ├── gene_sequence_2.parquet
│   └── ... (one per chromosome)
│
└── openspliceai_eval/                       # OpenSpliceAI evaluation artifacts
    ├── meta_models/                         # Meta-model specific outputs
    │   ├── analysis_sequences_*.tsv         # Training features per chromosome
    │   ├── error_analysis_*.tsv             # Prediction errors
    │   ├── splice_positions_*.tsv           # Splice site predictions
    │   └── inference/                       # Inference-specific outputs
    └── gene_manifest.json                   # Processing manifest
```

**Key Differences from SpliceAI**:
- Uses **GFF3 format** (not GTF)
- MANE RefSeq annotations (high-confidence, fewer transcripts)
- RefSeq genome naming (NC_000001.11 vs chr1)
- GRCh38.p14 (includes patches)

**Usage**:
```python
from meta_spliceai.system.genomic_resources import Registry

# Load OpenSpliceAI resources
registry = Registry(build='GRCh38_MANE')
# Automatically resolves to: data/mane/GRCh38/
```

**Documentation**: [`docs/base_models/GRCH38_MANE_VALIDATION_COMPLETE.md`](../base_models/GRCH38_MANE_VALIDATION_COMPLETE.md)

---

### Why Separate Directories?

**Genome Build Mismatch = Performance Drop**

Using the wrong data directory with a base model causes severe performance degradation:

| Model | Correct Data | Wrong Data | PR-AUC Drop |
|-------|--------------|------------|-------------|
| SpliceAI | GRCh37 (0.97) | GRCh38 (0.541) | **-44%** |
| OpenSpliceAI | GRCh38 MANE (0.98) | GRCh37 (0.523) | **-47%** |

**Lesson**: Always match the base model to its training data!

**See**: [`docs/base_models/BASE_MODEL_DATA_MAPPING.md`](../base_models/BASE_MODEL_DATA_MAPPING.md)

---

## Training and Evaluation Datasets

### Overview

Training datasets are stored at the **data root level** and follow a naming convention:

```
{purpose}_{gene_type}_{count}_{features}_{version}
```

### Training Datasets

**Location**: `data/train_*/`

#### train_pc_7000_3mers_opt

**Purpose**: Primary production training dataset for meta-model  
**Location**: `data/train_pc_7000_3mers_opt/`

**Characteristics**:
- **Gene Type**: Protein-coding
- **Gene Count**: ~7000 genes
- **Features**: 3-mer sequence features + splice scores
- **Version**: Optimized for production
- **Build**: GRCh37 (derived from SpliceAI analysis)

**Structure**:
```
data/train_pc_7000_3mers_opt/
├── master/
│   ├── training_data.parquet          # Main training data
│   └── feature_manifest.csv           # Feature descriptions
├── batch_*.parquet                    # Batched training data (optional)
└── README.md                          # Dataset documentation
```

**Usage**:
```python
from meta_spliceai.splice_engine.meta_models import load_training_data

data = load_training_data('train_pc_7000_3mers_opt')
```

#### train_pc_100_3mers_diverse

**Purpose**: Small diverse dataset for quick testing  
**Location**: `data/train_pc_100_3mers_diverse/`

**Characteristics**:
- **Gene Count**: ~100 genes
- **Purpose**: Fast iteration, testing, CI/CD
- **Build**: GRCh37

---

### Test/Validation Datasets

**Location**: `data/test_*/`

#### test_pc_1000_3mers

**Purpose**: Held-out test set for model evaluation  
**Location**: `data/test_pc_1000_3mers/`

**Characteristics**:
- **Gene Count**: ~1000 genes
- **Purpose**: Final model evaluation, never used for training
- **Build**: GRCh37

---

### Dataset Naming Convention

| Component | Description | Examples |
|-----------|-------------|----------|
| **Purpose** | `train`, `test`, `eval`, `val` | train, test |
| **Gene Type** | `pc` (protein-coding), `nc` (non-coding), `all` | pc, nc |
| **Count** | Number of genes | 1000, 7000, 100 |
| **Features** | Feature set identifier | 3mers, 5mers, kmers |
| **Version** | Dataset version or variant | opt, diverse, v1, v2 |

**Examples**:
- `train_pc_7000_3mers_opt`: Training, protein-coding, 7000 genes, 3-mers, optimized
- `test_pc_1000_3mers`: Test set, protein-coding, 1000 genes, 3-mers
- `eval_nc_500_5mers_v1`: Evaluation, non-coding, 500 genes, 5-mers, version 1

---

## Case Study Data

### Overview

Case study data includes variant databases, disease-specific datasets, and alternative splicing resources.

**Location**: `meta_spliceai/splice_engine/case_studies/`

### Structure

```
meta_spliceai/splice_engine/case_studies/
├── data_sources/                          # External variant databases
│   ├── datasets/
│   │   ├── splicevardb/                   # SpliceVarDB variant database
│   │   │   ├── splicevardb_20250915/     # Release-dated
│   │   │   │   ├── raw/
│   │   │   │   │   ├── variants_page_*.json
│   │   │   │   │   └── download_metadata.json
│   │   │   │   ├── processed/
│   │   │   │   │   ├── splicevardb_validated_GRCh38.parquet
│   │   │   │   │   ├── splicevardb_validated_GRCh38.tsv
│   │   │   │   │   └── splicevardb_validated_GRCh38.vcf.gz
│   │   │   │   └── README.md
│   │   │   └── latest -> splicevardb_20250915/
│   │   │
│   │   └── clinvar/                       # ClinVar pathogenic variants
│   │       ├── clinvar_20250831/
│   │       │   ├── raw/
│   │       │   │   └── clinvar_20250831.vcf.gz
│   │       │   ├── processed/
│   │       │   │   ├── clinvar_variants.parquet
│   │       │   │   └── clinvar_variants.tsv
│   │       │   └── README.md
│   │       └── latest -> clinvar_20250831/
│   │
│   └── vcf_processing/                    # VCF analysis workflows
│       ├── vcf_to_alternative_splice_sites.py
│       └── vcf_delta_scoring.py
│
└── studies/                               # Specific disease/phenomenon studies
    ├── ALS/                               # ALS-related splicing
    ├── cancer/                            # Cancer splicing alterations
    └── rare_diseases/                     # Rare disease case studies
```

### Key Datasets

#### SpliceVarDB

**Purpose**: Validated splice-altering variants from literature  
**Location**: `meta_spliceai/splice_engine/case_studies/data_sources/datasets/splicevardb/latest/`

**Characteristics**:
- **Build**: GRCh38
- **Variants**: ~2,000 validated splice-altering variants
- **Source**: Literature-curated, experimentally validated
- **Formats**: Parquet, TSV, VCF

**Usage**:
```python
from meta_spliceai.splice_engine.case_studies.data_sources.datasets import load_splicevardb

variants = load_splicevardb()  # Loads latest version
```

**Documentation**: `meta_spliceai/splice_engine/openspliceai_recalibration/DATA_ORGANIZATION.md`

#### ClinVar

**Purpose**: Clinical pathogenic variants for splice site analysis  
**Location**: `meta_spliceai/splice_engine/case_studies/data_sources/datasets/clinvar/latest/`

**Characteristics**:
- **Build**: GRCh38
- **Variants**: Pathogenic/likely pathogenic variants
- **Source**: NCBI ClinVar
- **Use Case**: Clinical validation, pathogenicity assessment

---

## Pre-trained Model Weights

### Location

`data/models/`

### Structure

```
data/models/
├── spliceai/                              # SpliceAI model weights
│   ├── spliceai_1.pt
│   ├── spliceai_2.pt
│   ├── spliceai_3.pt
│   ├── spliceai_4.pt
│   ├── spliceai_5.pt
│   └── config.json
│
└── openspliceai/                          # OpenSpliceAI model weights
    ├── openspliceai_1.pt
    ├── openspliceai_2.pt
    ├── openspliceai_3.pt
    ├── openspliceai_4.pt
    ├── openspliceai_5.pt
    └── config.json
```

### Usage

Models are automatically loaded by the inference engine:

```python
from meta_spliceai.splice_engine import load_base_model

# Automatically finds models in data/models/spliceai/
model = load_base_model('spliceai', device='cuda')
```

---

## Output and Artifacts

### Base Model Outputs

**Location**: `data/<annotation_source>/<build>/<base_model>_eval/meta_models/`

#### Chunk-Level Artifacts

Generated during base model pass (full genome processing):

```
data/mane/GRCh38/openspliceai_eval/meta_models/
├── analysis_sequences_1_chunk_1_500.tsv           # Features for training
├── analysis_sequences_1_chunk_501_1000.tsv
├── analysis_sequences_1_chunk_1001_1500.tsv
├── ... (one per 500-gene chunk)
│
├── error_analysis_1_chunk_1_500.tsv               # Prediction errors
├── splice_positions_1_chunk_1_500.tsv             # All splice predictions
│
└── gene_manifest.json                             # Processing metadata
```

**Purpose**: These are the **training artifacts** for the meta-model layer.

**Format**: TSV files with standardized schema (see [`docs/data/splice_sites/SCHEMA_STANDARDIZATION.md`](splice_sites/SCHEMA_STANDARDIZATION.md))

#### Aggregated Outputs

After chunk processing, data can be aggregated:

```
data/mane/GRCh38/openspliceai_eval/meta_models/
├── analysis_sequences_all_chromosomes.parquet     # All features combined
├── error_analysis_all_chromosomes.parquet         # All errors combined
└── splice_positions_all_chromosomes.parquet       # All predictions combined
```

### Inference Outputs

**Location**: `data/<annotation_source>/<build>/<base_model>_eval/meta_models/inference/`

```
inference/
├── predictions_<run_id>.parquet                   # Inference predictions
├── delta_scores_<run_id>.parquet                  # Delta score analysis
└── alternative_splice_sites_<run_id>.parquet      # Alternative splicing
```

---

## Directory Naming Conventions

### Genome Builds

| Name | Format | Example | Used By |
|------|--------|---------|---------|
| **GRCh37** | Ensembl | `chr1`, `chr2`, ... | SpliceAI |
| **GRCh38** | Ensembl | `chr1`, `chr2`, ... | Most tools |
| **GRCh38_MANE** | RefSeq | `NC_000001.11`, `NC_000002.12`, ... | OpenSpliceAI |

### Annotation Sources

| Source | Full Name | Description |
|--------|-----------|-------------|
| **ensembl** | Ensembl | Comprehensive gene annotations |
| **mane** | Matched Annotation from NCBI and EBI | High-confidence RefSeq+Ensembl |
| **gencode** | GENCODE | Human gene annotation |

### Output Subdirectories

| Directory | Purpose |
|-----------|---------|
| `<model>_eval/` | Base model evaluation outputs |
| `<model>_eval/meta_models/` | Meta-model training data |
| `<model>_eval/meta_models/inference/` | Inference-specific outputs |

---

## Data Organization Principles

### 1. Separation of Concerns

- **Reference Data**: `data/<source>/<build>/` (immutable)
- **Training Data**: `data/train_*/` (versioned)
- **Outputs**: `data/<source>/<build>/<model>_eval/` (derived, can be regenerated)
- **Case Studies**: `meta_spliceai/splice_engine/case_studies/` (external resources)

### 2. Build Isolation

Each genome build has its own directory to prevent cross-contamination:
- SpliceAI (GRCh37) → `data/ensembl/GRCh37/`
- OpenSpliceAI (GRCh38) → `data/mane/GRCh38/`

**Never mix builds!**

### 3. Versioning

- **Datasets**: Include version in directory name (`train_pc_7000_3mers_opt`)
- **External Data**: Use release dates (`splicevardb_20250915`)
- **Symlinks**: Point to latest version (`latest -> splicevardb_20250915/`)

### 4. Documentation

Every major data directory should have:
- `README.md`: Overview, usage, source
- `docs/` subdirectory (optional): Detailed documentation
- Feature manifests: For training datasets

---

## Common Tasks

### Task 1: Run Base Model Pass

**SpliceAI**:
```bash
run_base_model --base-model spliceai --mode production --coverage full_genome
# Uses: data/ensembl/GRCh37/
# Outputs: data/ensembl/GRCh37/spliceai_eval/meta_models/
```

**OpenSpliceAI**:
```bash
run_base_model --base-model openspliceai --mode production --coverage full_genome
# Uses: data/mane/GRCh38/
# Outputs: data/mane/GRCh38/openspliceai_eval/meta_models/
```

### Task 2: Load Training Data

```python
import polars as pl

# Load training features
features = pl.read_parquet('data/train_pc_7000_3mers_opt/master/training_data.parquet')

# Load base model outputs
openspliceai_features = pl.read_csv(
    'data/mane/GRCh38/openspliceai_eval/meta_models/analysis_sequences_1_chunk_1_500.tsv',
    separator='\t'
)
```

### Task 3: Access Case Study Data

```python
from meta_spliceai.splice_engine.case_studies.data_sources.datasets import load_splicevardb

# Load variant database
variants = load_splicevardb()  # Automatically uses latest version
```

---

## Migration Notes

### Current State (As of Nov 2025)

⚠️ **Data directory cleanup needed**:
- Root `data/ensembl/` has mixed GRCh37 and GRCh38 files
- Some legacy analysis directories need archiving
- Training datasets are at root level (correct)

### Cleanup Plan

See: `dev/DATA_DIRECTORY_CLEANUP_PLAN.md` (to be created)

---

## Related Documentation

### Essential Guides
- **Base Models**: [`docs/base_models/BASE_MODEL_DATA_MAPPING.md`](../base_models/BASE_MODEL_DATA_MAPPING.md)
- **SpliceAI Setup**: [`docs/base_models/GRCH37_SETUP_COMPLETE_GUIDE.md`](../base_models/GRCH37_SETUP_COMPLETE_GUIDE.md)
- **OpenSpliceAI Setup**: [`docs/base_models/GRCH38_MANE_VALIDATION_COMPLETE.md`](../base_models/GRCH38_MANE_VALIDATION_COMPLETE.md)
- **Schema Standards**: [`docs/data/splice_sites/SCHEMA_STANDARDIZATION.md`](splice_sites/SCHEMA_STANDARDIZATION.md)

### Deep Dives
- **Case Study Organization**: `meta_spliceai/splice_engine/openspliceai_recalibration/DATA_ORGANIZATION.md`
- **Genomic Resources**: `meta_spliceai/system/genomic_resources/docs/`
- **Dataset Conventions**: `meta_spliceai/splice_engine/case_studies/data_sources/datasets/README.md`

---

## Quick Reference Table

| What Do You Need? | Where Is It? | Documentation |
|-------------------|--------------|---------------|
| Run SpliceAI predictions | `data/ensembl/GRCh37/` | [GRCH37_SETUP](../base_models/GRCH37_SETUP_COMPLETE_GUIDE.md) |
| Run OpenSpliceAI predictions | `data/mane/GRCh38/` | [GRCH38_MANE](../base_models/GRCH38_MANE_VALIDATION_COMPLETE.md) |
| Train meta-model | `data/train_pc_7000_3mers_opt/` | [Training Docs](../training/) |
| Analyze variants | `case_studies/data_sources/datasets/` | [DATA_ORGANIZATION](../../meta_spliceai/splice_engine/openspliceai_recalibration/DATA_ORGANIZATION.md) |
| Access model weights | `data/models/` | Auto-loaded |
| Base model outputs | `data/<source>/<build>/<model>_eval/` | [Output Structure](../reference/INFERENCE_OUTPUT_STRUCTURE.md) |

---

**Questions? See**: [`docs/data/README.md`](README.md) or check specific documentation linked above.

