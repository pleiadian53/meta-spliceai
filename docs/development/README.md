# Development Documentation

This directory contains technical documentation for Meta-SpliceAI's core systems and workflows.

## 📚 Documentation Structure

### Artifact Management
- **[ARTIFACT_MANAGEMENT.md](ARTIFACT_MANAGEMENT.md)** - Overview of artifact management system
- **[ARTIFACT_MANAGEMENT_IMPLEMENTATION.md](ARTIFACT_MANAGEMENT_IMPLEMENTATION.md)** - Detailed implementation guide
- **[ARTIFACT_MANAGEMENT_QUICK_REFERENCE.md](ARTIFACT_MANAGEMENT_QUICK_REFERENCE.md)** - Quick reference for artifact management

### Base Models & Integration
- **[BASE_MODEL_SELECTION_AND_ROUTING.md](BASE_MODEL_SELECTION_AND_ROUTING.md)** - How base model selection and routing works
- **[OPENSPLICEAI_MODELS_INFO.md](OPENSPLICEAI_MODELS_INFO.md)** - Technical specifications for OpenSpliceAI models
- **[OPENSPLICEAI_QUICK_START.md](OPENSPLICEAI_QUICK_START.md)** - Quick start guide for OpenSpliceAI integration

### Data Structures & Schema
- **[ANNOTATION_SOURCE_DIRECTORY_STRUCTURE.md](ANNOTATION_SOURCE_DIRECTORY_STRUCTURE.md)** - Directory structure for genomic annotations
- **[MANE_VS_ENSEMBL_SPLICE_SITES.md](MANE_VS_ENSEMBL_SPLICE_SITES.md)** - Comparison of MANE and Ensembl splice site annotations

### Features & Functionality
- **[AUTOMATIC_GENE_FEATURES_GENERATION.md](AUTOMATIC_GENE_FEATURES_GENERATION.md)** - Automated gene feature extraction system
- **[MULTI_BUILD_SUPPORT_COMPLETE_2025-11-01.md](MULTI_BUILD_SUPPORT_COMPLETE_2025-11-01.md)** - Multi-genome build support (GRCh37/GRCh38)
- **[REGISTRY_QUICK_REFERENCE.md](REGISTRY_QUICK_REFERENCE.md)** - Quick reference for genomic registry system

### Training & Data Building
- **[INCREMENTAL_BUILDER_USAGE_GUIDE.md](INCREMENTAL_BUILDER_USAGE_GUIDE.md)** - Guide for incremental training data builder
- **[FEATURE_MANIFEST_DESIGN.md](FEATURE_MANIFEST_DESIGN.md)** - Design of feature manifest system for training

### Optimization & Performance
- **[INFERENCE_DATA_REUSE_OPTIMIZATION.md](INFERENCE_DATA_REUSE_OPTIMIZATION.md)** - Optimization to reuse data during inference

### Variant Analysis
- **[VCF_VARIANT_ANALYSIS_WORKFLOW.md](VCF_VARIANT_ANALYSIS_WORKFLOW.md)** - Workflow for analyzing splice-altering variants from VCF files

---

## 🗂️ Related Documentation

### Development Notes (Internal)
Development history, session summaries, and technical investigations are organized in `/dev/`:

- **`dev/sessions/`** - Session summaries and progress logs
- **`dev/implementation_logs/`** - Implementation progress tracking
- **`dev/technical_notes/`** - Technical investigations and debugging notes
- **`dev/archive/`** - Archived or superseded documentation

### Case Studies
For variant analysis examples and workflows, see:
- **`meta_spliceai/splice_engine/case_studies/`** - Implementation
- **`dev/case_studies/`** - Development notes

---

## 🚀 Getting Started

1. **New to the project?** Start with [OPENSPLICEAI_QUICK_START.md](OPENSPLICEAI_QUICK_START.md)
2. **Building training data?** See [INCREMENTAL_BUILDER_USAGE_GUIDE.md](INCREMENTAL_BUILDER_USAGE_GUIDE.md)
3. **Working with variants?** Check [VCF_VARIANT_ANALYSIS_WORKFLOW.md](VCF_VARIANT_ANALYSIS_WORKFLOW.md)
4. **Understanding artifacts?** Read [ARTIFACT_MANAGEMENT_QUICK_REFERENCE.md](ARTIFACT_MANAGEMENT_QUICK_REFERENCE.md)

---

## 📝 Documentation Standards

All documentation in this directory:
- ✅ Reflects **current implementation** (not historical)
- ✅ Is **publicly shareable** (suitable for GitHub)
- ✅ Provides **technical reference** for users and contributors
- ✅ Is **actively maintained**

For development history and internal notes, see `/dev/` directory.

---

**Last Updated**: 2024-11-19  
**Total Documents**: 17
