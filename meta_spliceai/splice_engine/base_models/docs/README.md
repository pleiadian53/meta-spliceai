# Base Models - Package-Level Documentation

This directory contains **implementation-specific documentation** for the base model layer in MetaSpliceAI.

> **User-facing documentation** is in `docs/base_models/` at the project root.

---

## 📑 Documentation Index

### Base Model Specifications

| Document | Purpose |
|----------|---------|
| **[SPLICEAI.md](SPLICEAI.md)** | SpliceAI model (GRCh37/Ensembl) - architecture, usage, configuration |
| **[OPENSPLICEAI.md](OPENSPLICEAI.md)** | OpenSpliceAI model (GRCh38/MANE) - architecture, usage, configuration |

### Core Concepts

| Document | Purpose |
|----------|---------|
| **[POSITION_COORDINATE_SYSTEMS.md](POSITION_COORDINATE_SYSTEMS.md)** | Absolute vs relative coordinate handling, strand-dependent positions |

### Data & Structure

| Document | Purpose |
|----------|---------|
| **[BASE_MODEL_DATA_MAPPING.md](BASE_MODEL_DATA_MAPPING.md)** | Data organization and model-to-build mapping |
| **[BUILD_NAMING_STANDARD.md](BUILD_NAMING_STANDARD.md)** | Naming conventions for builds and datasets |
| **[SEQUENCE_INPUT_FORMAT_FOR_BASE_MODELS.md](SEQUENCE_INPUT_FORMAT_FOR_BASE_MODELS.md)** | Input format specifications |

### Gene Mapping

| Document | Purpose |
|----------|---------|
| **[GENE_MAPPING_SYSTEM.md](GENE_MAPPING_SYSTEM.md)** | Cross-build gene identification system |
| **[GENE_MAPPER_QUICK_REFERENCE.md](GENE_MAPPER_QUICK_REFERENCE.md)** | Quick reference for gene mapping |

### Integration & Porting

| Document | Purpose |
|----------|---------|
| **[BASE_LAYER_INTEGRATION_GUIDE.md](BASE_LAYER_INTEGRATION_GUIDE.md)** | Technical integration details |
| **[AI_AGENT_PORTING_GUIDE.md](AI_AGENT_PORTING_GUIDE.md)** | Comprehensive 6-stage porting guide |
| **[AI_AGENT_PROMPTS.md](AI_AGENT_PROMPTS.md)** | Ready-to-use prompts for AI agents |
| **[BASE_LAYER_PORT_VERIFICATION_PROMPTS.md](BASE_LAYER_PORT_VERIFICATION_PROMPTS.md)** | Verification prompts for porting |
| **[BASE_LAYER_VERIFICATION_SUMMARY.md](BASE_LAYER_VERIFICATION_SUMMARY.md)** | Verification strategy summary |

### Scripts & Usage

| Document | Purpose |
|----------|---------|
| **[COMPARE_BASE_MODELS_ROBUST_USAGE.md](COMPARE_BASE_MODELS_ROBUST_USAGE.md)** | Usage guide for model comparison script |

---

## 🔗 Related Documentation

### Project-Level (User-Facing)
- `docs/base_models/` - User guides, setup instructions, design rationale

### Source Code
- `meta_spliceai/splice_engine/meta_models/core/position_types.py` - Position coordinate utilities
- `meta_spliceai/splice_engine/run_spliceai_workflow.py` - Core prediction workflow
- `meta_spliceai/run_base_model.py` - Main entry point

---

*Last Updated: December 13, 2025*  
*Total Documents: 14*

