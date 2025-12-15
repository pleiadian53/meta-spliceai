# Base Models Documentation

This directory contains **user-facing documentation** for base model integration, usage, and configuration in MetaSpliceAI.

> **Note**: Implementation-specific documentation (porting guides, data mappings, internal formats) is in `meta_spliceai/splice_engine/base_models/docs/`

---

## 📚 Quick Start

### For Users

**Want to use a base model?**
→ See [UNIVERSAL_BASE_MODEL_SUPPORT.md](UNIVERSAL_BASE_MODEL_SUPPORT.md) for feature overview

**Comparing SpliceAI and OpenSpliceAI?**
→ See [BASE_MODEL_COMPARISON_GUIDE.md](BASE_MODEL_COMPARISON_GUIDE.md) for comparison guide

**Setting up GRCh37 data?**
→ See [GRCH37_SETUP_COMPLETE_GUIDE.md](GRCH37_SETUP_COMPLETE_GUIDE.md) for setup instructions

### For AI Agents / Porting

**Want to port the base layer to another project?**
→ See package-level docs: `meta_spliceai/splice_engine/base_models/docs/AI_AGENT_PORTING_GUIDE.md`

### For Developers

**Understanding coordinate systems?**
→ See package-level docs: `meta_spliceai/splice_engine/base_models/docs/POSITION_COORDINATE_SYSTEMS.md`

**Understanding data organization?**
→ See package-level docs: `meta_spliceai/splice_engine/base_models/docs/BASE_MODEL_DATA_MAPPING.md`

---

## 📑 Documentation Index

### User Guides (This Directory)

| Document | Purpose |
|----------|---------|
| **[UNIVERSAL_BASE_MODEL_SUPPORT.md](UNIVERSAL_BASE_MODEL_SUPPORT.md)** | Extensibility for custom models |
| **[BASE_MODEL_COMPARISON_GUIDE.md](BASE_MODEL_COMPARISON_GUIDE.md)** | How to compare different base models |
| **[RUN_BASE_MODEL_FULL_COVERAGE_EXAMPLES.md](RUN_BASE_MODEL_FULL_COVERAGE_EXAMPLES.md)** | Usage examples and code snippets |

### Setup & Installation

| Document | Purpose |
|----------|---------|
| **[GRCH37_SETUP_COMPLETE_GUIDE.md](GRCH37_SETUP_COMPLETE_GUIDE.md)** | Complete GRCh37 setup guide |
| **[GRCH37_DOWNLOAD_GUIDE.md](GRCH37_DOWNLOAD_GUIDE.md)** | GRCh37 data download reference |

### Design Rationale & Compatibility

| Document | Purpose |
|----------|---------|
| **[GENE_LOCUS_VS_PREMRNA_RATIONALE.md](GENE_LOCUS_VS_PREMRNA_RATIONALE.md)** | Design decision: gene locus vs pre-mRNA |
| **[NUCLEOTIDE_SCORES_DESIGN_RATIONALE.md](NUCLEOTIDE_SCORES_DESIGN_RATIONALE.md)** | Nucleotide-level scoring design |
| **[GENOME_BUILD_COMPATIBILITY.md](GENOME_BUILD_COMPATIBILITY.md)** | GRCh37/GRCh38 compatibility information |

### Package-Level Documentation

Implementation-specific docs are in `meta_spliceai/splice_engine/base_models/docs/`:

| Document | Purpose |
|----------|---------|
| `POSITION_COORDINATE_SYSTEMS.md` | Absolute vs relative coordinate handling |
| `AI_AGENT_PORTING_GUIDE.md` | Comprehensive 6-stage porting guide |
| `AI_AGENT_PROMPTS.md` | Ready-to-use prompts for AI agents |
| `BASE_LAYER_PORT_VERIFICATION_PROMPTS.md` | Verification prompts for porting |
| `BASE_LAYER_VERIFICATION_SUMMARY.md` | Verification strategy summary |
| `BASE_LAYER_INTEGRATION_GUIDE.md` | Technical integration details |
| `BASE_MODEL_DATA_MAPPING.md` | Data organization and model-to-build mapping |
| `BUILD_NAMING_STANDARD.md` | Naming conventions for builds |
| `COMPARE_BASE_MODELS_ROBUST_USAGE.md` | Technical script usage |
| `GENE_MAPPING_SYSTEM.md` | Cross-build gene identification |
| `GENE_MAPPER_QUICK_REFERENCE.md` | Quick reference for gene mapping |
| `SEQUENCE_INPUT_FORMAT_FOR_BASE_MODELS.md` | Input format specifications |

---

## 🔑 Key Concepts

### Base Models

- **SpliceAI**: Original Keras model, trained on GRCh37/Ensembl
- **OpenSpliceAI**: PyTorch model, trained on GRCh38/MANE
- **Custom Models**: Extensible architecture supports additional models

### Genomic Builds

- **GRCh37**: Human genome build 37 (hg19)
- **GRCh38**: Human genome build 38 (hg38)
- **Coordinate systems**: Different between builds, requires liftOver

### Annotation Sources

- **Ensembl**: Comprehensive annotations, all isoforms
- **MANE**: Matched Annotation from NCBI and EBI, canonical transcripts
- **RefSeq**: NCBI reference sequences

---

## 🏗️ Architecture Overview

```
MetaSpliceAI Base Model System
├── User Interface
│   ├── CLI: run_base_model --base-model <model>
│   └── API: run_base_model_predictions(base_model='spliceai')
│
├── Configuration (model_config.py)
│   ├── BaseModelConfig (abstract)
│   ├── SpliceAIConfig (GRCh37/Ensembl)
│   └── OpenSpliceAIConfig (GRCh38/MANE)
│
├── Model Loading
│   ├── SpliceAI → Keras models (5 models)
│   └── OpenSpliceAI → PyTorch models (5 models)
│
├── Genomic Resources (Registry)
│   ├── GRCh37/Ensembl → data/ensembl/GRCh37/
│   └── GRCh38/MANE → data/mane/GRCh38/
│
└── Artifact Management
    ├── Test mode → tests/{test_name}/
    └── Production mode → meta_models/predictions/
```

---

## 📁 Data Layout

```
data/
├── ensembl/GRCh37/                # SpliceAI data
│   ├── genome.fa                   # Reference genome
│   ├── annotations.gtf             # Ensembl annotations
│   ├── splice_sites_enhanced.tsv   # Splice site annotations
│   └── spliceai_eval/              # Prediction outputs
│       └── meta_models/
│           ├── analysis_positions_chr*.tsv
│           ├── analysis_sequences_chr*.tsv
│           └── gene_manifest.tsv
│
└── mane/GRCh38/                   # OpenSpliceAI data
    ├── genome.fa                   # Reference genome
    ├── annotations.gtf             # MANE annotations
    ├── splice_sites_enhanced.tsv   # Splice site annotations
    └── openspliceai_eval/          # Prediction outputs
        └── meta_models/
            ├── analysis_positions_chr*.tsv
            ├── analysis_sequences_chr*.tsv
            └── gene_manifest.tsv
```

---

## 🚀 Usage Examples

### CLI Usage

```bash
# SpliceAI (GRCh37/Ensembl)
run_base_model --base-model spliceai --mode test --coverage gene_subset

# OpenSpliceAI (GRCh38/MANE)
run_base_model --base-model openspliceai --mode test --coverage gene_subset

# Full genome production run
run_base_model --base-model openspliceai --mode production --coverage full_genome
```

### Python API Usage

```python
from meta_spliceai import run_base_model_predictions

# Run SpliceAI
results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1', 'TP53'],
    mode='test'
)

# Run OpenSpliceAI
results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1', 'TP53'],
    mode='test'
)
```

---

## 📊 Features

### ✅ Implemented

- **Multi-Model Support**: SpliceAI and OpenSpliceAI
- **Automatic Resource Routing**: Build-specific data paths
- **Gene Manifest**: Tracking processed vs. missing genes
- **Nucleotide-Level Scores**: Full splice site landscape
- **Chunk-Level Checkpointing**: Resume interrupted processes
- **Memory-Efficient Processing**: Mini-batch gene processing
- **Cross-Build Gene Mapping**: Map genes between GRCh37 and GRCh38
- **Extensible Configuration**: Easy to add new models

---

## 📚 Related Documentation

### Package-Level Docs
- `meta_spliceai/splice_engine/base_models/docs/` - Implementation details

### In Other Directories
- **Training**: `docs/training/` - Meta-learning on top of base models
- **Feature Engineering**: `docs/feature_engineering/` - Derived features
- **Data Management**: `docs/data/` - Data layout and conventions

---

*Last Updated: December 13, 2025*  
*Project-level Documents: 9*  
*Package-level Documents: 13*
