# MetaSpliceAI Documentation Index

**Central navigation hub for all MetaSpliceAI documentation**

**Last Updated**: January 30, 2026

---

## 🎯 Quick Access

| What do you want to do? | Go here |
|--------------------------|---------|
| **Run splice site predictions** | [Base Layer Entry Points](#base-layer-production) |
| **Train meta-learning models** | [Meta Layer Documentation](#meta-layer-development) |
| **Analyze disease variants** | [Variant Analysis Documentation](#variant-analysis-development) |
| **Find all executable commands** | [Entry Points Reference](./ENTRY_POINTS_REFERENCE.md) |
| **Understand the architecture** | [Project Overview](#architecture-overview) |

---

## 📚 Main Documentation Hubs

### 1. [Entry Points Reference](./ENTRY_POINTS_REFERENCE.md) ⭐

**Complete reference for all user-facing entry points**

- CLI commands registered in `pyproject.toml`
- Runnable examples and test scripts
- Entry point scripts by functional layer
- Recommended workflows
- Status and availability

**Start here if**: You want to know what commands are available

---

### 2. [Meta Layer Documentation](./meta_layer/README.md)

**Multimodal deep learning for splice site recalibration**

- Architecture and design
- Training guides and methods
- Experiment results
- GPU/CPU configurations
- Model zoo

**Start here if**: You want to train or use meta-learning models

---

### 3. [Variant Analysis Documentation](./variant_analysis/README.md)

**Disease validation and clinical variant analysis**

- ClinVar pipeline (primary tool)
- VCF processing workflows
- Database integrations
- Delta score computation
- Disease cohort validation

**Start here if**: You want to analyze clinical or disease variants

---

## 🏗️ Architecture Overview

MetaSpliceAI is organized in three functional layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    METASPLICEAI LAYERS                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │ BASE LAYER   │   │  META LAYER  │   │ CASE STUDIES │   │
│  │ ✅ Production│   │🚧 Development│   │🚧 Development│   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  Splice Site        Multimodal DL       Variant Analysis   │
│  Prediction         Recalibration       & Validation       │
│  (SpliceAI/         (HyenaDNA/CNN)     (ClinVar/          │
│   OpenSpliceAI)                         SpliceVarDB)       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📖 Documentation by Layer

### Base Layer (Production)

**Status**: ✅ **Production-Ready** with full CLI support

#### Core Documents
- **[Universal Base Model Support](./base_models/UNIVERSAL_BASE_MODEL_SUPPORT.md)** - Multi-model architecture
- **[Base Model Comparison Guide](./base_models/BASE_MODEL_COMPARISON_GUIDE.md)** - Comparing models
- **[Genome Build Compatibility](./base_models/GENOME_BUILD_COMPATIBILITY.md)** - Critical build requirements
- **[CLI Splice Sites](./CLI_SPLICE_SITES.md)** - Annotation CLI tool

#### CLI Tools
- `run_base_model` - Run base model predictions
- `evaluate_predictions` - Evaluate results
- `annotate_splice_sites` - Generate splice site annotations

#### Python API
- `meta_spliceai.run_base_model.run_base_model_predictions()`
- `meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow`

**Documentation Location**: `docs/base_models/`

---

### Meta Layer (Development)

**Status**: 🚧 **In Development** - Core functionality present, CLI pending

#### Core Documents
- **[Meta Layer README](./meta_layer/README.md)** ⭐ Start here
- **[Package README](../meta_spliceai/splice_engine/meta_layer/README.md)** - Technical details
- **[Architecture](../meta_spliceai/splice_engine/meta_layer/docs/ARCHITECTURE.md)** - System design
- **[Training Guide](../meta_spliceai/splice_engine/meta_layer/docs/TRAINING_GUIDE.md)** - How to train

#### Methods & Approaches
- **[Validated Delta Prediction](../meta_spliceai/splice_engine/meta_layer/docs/methods/VALIDATED_DELTA_PREDICTION.md)** ⭐ Recommended
- **[Multi-Step Framework](../meta_spliceai/splice_engine/meta_layer/docs/methods/MULTI_STEP_FRAMEWORK.md)** - Staged training
- **[Paired Delta Prediction](../meta_spliceai/splice_engine/meta_layer/docs/methods/PAIRED_DELTA_PREDICTION.md)** - WT/ALT comparison
- **[Roadmap](../meta_spliceai/splice_engine/meta_layer/docs/methods/ROADMAP.md)** - Development plan

#### Experiments
- [Experiment 001](../meta_spliceai/splice_engine/meta_layer/docs/experiments/001_canonical_classification/) - Canonical classification
- [Experiment 002](../meta_spliceai/splice_engine/meta_layer/docs/experiments/002_delta_prediction/) - Delta prediction
- [Experiment 003](../meta_spliceai/splice_engine/meta_layer/docs/experiments/003_binary_classification/) - Binary classification
- [Experiment 004](../meta_spliceai/splice_engine/meta_layer/docs/experiments/004_validated_delta/) ⭐ **Current best**

#### Entry Points
- `meta_layer/examples/train_simple.py` ⭐ Start here
- `meta_layer/examples/test_model.py`
- `meta_layer/tests/test_validated_delta_experiments.py`
- `meta_layer/tests/train_hyenadna_validated_delta.py` (GPU)

**Documentation Location**: `meta_spliceai/splice_engine/meta_layer/docs/`

---

### Case Studies / Variant Analysis (Development)

**Status**: 🚧 **In Development** - Entry points functional, workflows in progress

#### Core Documents
- **[Variant Analysis README](./variant_analysis/README.md)** ⭐ Start here
- **[Package README](../meta_spliceai/splice_engine/case_studies/README.md)** - Technical details
- **[Complete ClinVar Pipeline](../meta_spliceai/splice_engine/case_studies/docs/variant_analysis/COMPLETE_CLINVAR_PIPELINE_README.md)** ⭐ Primary workflow
- **[Entry Points README](../meta_spliceai/splice_engine/case_studies/entry_points/README.md)** - CLI tools

#### Workflows
- **[VCF Variant Analysis Workflow](../meta_spliceai/splice_engine/case_studies/docs/VCF_VARIANT_ANALYSIS_WORKFLOW.md)** - General VCF
- **[VCF to Alternative Splice Sites](../meta_spliceai/splice_engine/case_studies/docs/VCF_TO_ALTERNATIVE_SPLICE_SITES_WORKFLOW.md)** - Alternative splicing
- **[Delta Score Bridge](../meta_spliceai/splice_engine/case_studies/docs/DELTA_SCORE_BRIDGE_IMPLEMENTATION.md)** - Score computation

#### System Design
- **[System Design Analysis (Q1-Q7)](../meta_spliceai/splice_engine/case_studies/docs/SYSTEM_DESIGN_ANALYSIS_Q1_Q7.md)** - Architecture
- **[Variant Splicing Biology (Q10-Q12)](../meta_spliceai/splice_engine/case_studies/docs/VARIANT_SPLICING_BIOLOGY_Q10_Q12.md)** - Biology
- **[Implementation Guide](../meta_spliceai/splice_engine/case_studies/docs/IMPLEMENTATION_GUIDE.md)** - Implementation

#### Entry Points
- `case_studies/entry_points/run_clinvar_pipeline.py` ⭐ Primary tool
- `case_studies/entry_points/run_vcf_column_documenter.py`
- `case_studies/examples/clinvar_openspliceai_workflow.py`

**Documentation Location**: `meta_spliceai/splice_engine/case_studies/docs/`

---

## 📊 Documentation by Type

### User Guides

| Guide | Layer | Audience | Status |
|-------|-------|----------|--------|
| [Entry Points Reference](./ENTRY_POINTS_REFERENCE.md) | All | Users | ✅ Complete |
| [Base Model Comparison](./base_models/BASE_MODEL_COMPARISON_GUIDE.md) | Base | Users | ✅ Complete |
| [Meta Layer README](./meta_layer/README.md) | Meta | Users | ✅ Complete |
| [Variant Analysis README](./variant_analysis/README.md) | Variant | Users | ✅ Complete |
| [CLI Splice Sites](./CLI_SPLICE_SITES.md) | Base | Users | ✅ Complete |

### Technical Documentation

| Document | Layer | Audience | Status |
|----------|-------|----------|--------|
| [Architecture](../meta_spliceai/splice_engine/meta_layer/docs/ARCHITECTURE.md) | Meta | Developers | ✅ Complete |
| [System Design](../meta_spliceai/splice_engine/case_studies/docs/SYSTEM_DESIGN_ANALYSIS_Q1_Q7.md) | Variant | Developers | ✅ Complete |
| [Universal Base Model Support](./base_models/UNIVERSAL_BASE_MODEL_SUPPORT.md) | Base | Developers | ✅ Complete |
| [Training Guide](../meta_spliceai/splice_engine/meta_layer/docs/TRAINING_GUIDE.md) | Meta | Developers | ✅ Complete |

### Tutorials

| Tutorial | Layer | Topic | Status |
|----------|-------|-------|--------|
| [ClinVar Workflow 1-2](../meta_spliceai/splice_engine/case_studies/docs/tutorials/CLINVAR_WORKFLOW_STEPS_1_2_TUTORIAL.md) | Variant | VCF processing | ✅ Complete |
| [ClinVar Workflow 2.5](../meta_spliceai/splice_engine/case_studies/docs/tutorials/CLINVAR_WORKFLOW_STEP_2.5_TUTORIAL.md) | Variant | Advanced | ✅ Complete |
| [Universal VCF Parsing](../meta_spliceai/splice_engine/case_studies/docs/tutorials/UNIVERSAL_VCF_PARSING_TUTORIAL.md) | Variant | VCF parsing | ✅ Complete |
| [HyenaDNA Fine-tuning](../meta_spliceai/splice_engine/meta_layer/docs/methods/HYENADNA_FINETUNING_TUTORIAL.md) | Meta | GPU training | ✅ Complete |

### Method Papers

| Document | Layer | Topic | Status |
|----------|-------|-------|--------|
| [Validated Delta Prediction](../meta_spliceai/splice_engine/meta_layer/docs/methods/VALIDATED_DELTA_PREDICTION.md) | Meta | Delta scores | ✅ Complete |
| [Multi-Step Framework](../meta_spliceai/splice_engine/meta_layer/docs/methods/MULTI_STEP_FRAMEWORK.md) | Meta | Training | ✅ Complete |
| [Paired Delta Prediction](../meta_spliceai/splice_engine/meta_layer/docs/methods/PAIRED_DELTA_PREDICTION.md) | Meta | Variants | ✅ Complete |
| [Meta Recalibration](../meta_spliceai/splice_engine/meta_layer/docs/methods/META_RECALIBRATION.md) | Meta | Recalibration | ✅ Complete |

---

## 🔍 Finding Documentation

### By Task

**I want to...**

- **Run base model predictions** → [Entry Points Reference](./ENTRY_POINTS_REFERENCE.md#base-layer-production)
- **Train a meta-learning model** → [Meta Layer README](./meta_layer/README.md#quick-start)
- **Process ClinVar variants** → [Complete ClinVar Pipeline](../meta_spliceai/splice_engine/case_studies/docs/variant_analysis/COMPLETE_CLINVAR_PIPELINE_README.md)
- **Understand the architecture** → [Architecture](../meta_spliceai/splice_engine/meta_layer/docs/ARCHITECTURE.md)
- **Find examples** → Search `examples/` directories
- **Debug issues** → Check layer-specific troubleshooting guides

### By File Type

```bash
# Find all README files
find docs -name "README.md"
find meta_spliceai -name "README.md"

# Find method documentation
find meta_spliceai/splice_engine/meta_layer/docs/methods -name "*.md"

# Find tutorials
find meta_spliceai/splice_engine/case_studies/docs/tutorials -name "*.md"

# Find experiment results
find meta_spliceai/splice_engine/meta_layer/docs/experiments -name "*.md"
```

---

## 📝 Documentation Standards

### Structure

All major components follow this structure:

```
component/
├── README.md                    # Overview and quick start
├── docs/                        # Detailed documentation
│   ├── ARCHITECTURE.md         # System design
│   ├── TRAINING_GUIDE.md       # How-to guides
│   ├── methods/                # Method descriptions
│   ├── experiments/            # Experiment results
│   └── tutorials/              # Step-by-step tutorials
├── examples/                    # Runnable examples
└── tests/                       # Test/training scripts
```

### Status Indicators

- ✅ **Complete** - Production-ready, fully documented
- 🚧 **In Development** - Functional but incomplete
- 📋 **Planned** - Designed but not implemented
- ⭐ **Recommended** - Preferred/best approach

---

## 🚀 Getting Started Paths

### Path 1: New User (Production Tools)

1. Read [Entry Points Reference](./ENTRY_POINTS_REFERENCE.md)
2. Try [Base Layer Quickstart](#base-layer-quickstart)
3. Explore [Base Model Documentation](./base_models/)

### Path 2: Researcher (Meta-Learning)

1. Read [Meta Layer README](./meta_layer/README.md)
2. Understand [Validated Delta Method](../meta_spliceai/splice_engine/meta_layer/docs/methods/VALIDATED_DELTA_PREDICTION.md)
3. Run [training examples](../meta_spliceai/splice_engine/meta_layer/examples/)

### Path 3: Clinical (Variant Analysis)

1. Read [Variant Analysis README](./variant_analysis/README.md)
2. Follow [Complete ClinVar Pipeline](../meta_spliceai/splice_engine/case_studies/docs/variant_analysis/COMPLETE_CLINVAR_PIPELINE_README.md)
3. Try [ClinVar workflow](../meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py)

### Path 4: Developer (Contributing)

1. Review all three layer READMEs
2. Study [Architecture documents](#technical-documentation)
3. Check [Implementation Guides](#system-design)
4. Review [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## 📞 Getting Help

### By Layer

- **Base Layer**: Check `docs/base_models/` and CLI help (`--help`)
- **Meta Layer**: Check `meta_layer/docs/` and examples
- **Variant Analysis**: Check `case_studies/docs/` and entry point READMEs

### General Resources

- **Main README**: [README.md](../README.md)
- **Getting Started**: [GETTING_STARTED.md](../GETTING_STARTED.md)
- **Project Summary**: [PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md)
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## 🔄 Recent Updates

### January 30, 2026

- ✅ Created [Entry Points Reference](./ENTRY_POINTS_REFERENCE.md)
- ✅ Created [Meta Layer Documentation Hub](./meta_layer/README.md)
- ✅ Created [Variant Analysis Documentation Hub](./variant_analysis/README.md)
- ✅ Created this documentation index
- ✅ Verified all base layer entry points
- ✅ Cataloged meta layer and variant analysis entry points

---

## 📊 Documentation Coverage

| Layer | User Guides | Technical Docs | Tutorials | Examples | Status |
|-------|-------------|----------------|-----------|----------|--------|
| **Base Layer** | ✅ Complete | ✅ Complete | ✅ Complete | ✅ Complete | ✅ **Production** |
| **Meta Layer** | ✅ Complete | ✅ Complete | ✅ Complete | ✅ Complete | 🚧 **Development** |
| **Variant Analysis** | ✅ Complete | ✅ Complete | ✅ Complete | 🚧 Partial | 🚧 **Development** |

---

**Maintained By**: MetaSpliceAI Development Team  
**Last Updated**: January 30, 2026  
**Version**: 0.2.0
