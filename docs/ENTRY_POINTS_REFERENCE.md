# MetaSpliceAI Entry Points Reference

**Purpose**: Comprehensive reference for all user-facing entry points in MetaSpliceAI  
**Last Updated**: January 30, 2026

---

## 📋 Overview

This document catalogs all entry points (executables and runnable modules) available to users of MetaSpliceAI, organized by functional layer and purpose.

### Entry Point Types

1. **CLI Commands** - Registered in `pyproject.toml` as executable commands
2. **Runnable Examples** - Scripts with `main()` functions in `examples/` directories
3. **Test Scripts** - Executable test/training scripts in `tests/` directories
4. **Entry Point Scripts** - Dedicated entry points in `entry_points/` directories

---

## 🏗️ Architecture Layers

MetaSpliceAI is structured in three functional layers:

```
┌──────────────────────────────────────────────────────────────┐
│                         USER ENTRY POINTS                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │  BASE LAYER    │  │  META LAYER    │  │  CASE STUDIES  │ │
│  │  (Production)  │  │  (Development) │  │  (Development) │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  SpliceAI/          Multimodal Meta-      Variant Analysis   │
│  OpenSpliceAI       Learning (DL)         & Validation       │
│  Predictions        Recalibration         Workflows          │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎯 Base Layer (Production-Ready)

### Status: ✅ **Production** - Fully functional with CLI tools

The base layer provides splice site prediction using base models (SpliceAI, OpenSpliceAI).

### CLI Commands (Registered in pyproject.toml)

#### 1. `run_base_model` ⭐ **Primary Entry Point**

**Purpose**: Run base model predictions (SpliceAI/OpenSpliceAI) on genes or chromosomes  
**Module**: `meta_spliceai.cli.run_base_model_cli:main`  
**Status**: ✅ Production

**Usage**:
```bash
# Single gene analysis
run_base_model --genes BRCA1 TP53

# Chromosome analysis  
run_base_model --chromosomes 21 --base-model openspliceai

# Full genome pass (production mode)
run_base_model --base-model openspliceai --mode production --coverage full_genome

# Specific chromosomes for training data
run_base_model --chromosomes 1,2,3 --mode production --verbosity 2
```

**Key Features**:
- Model-agnostic design (works with any base model)
- Memory-efficient processing with mini-batching
- Smart checkpointing and resumption
- Automatic artifact management
- Production vs test modes

**Output**:
- Per-nucleotide splice scores
- Position-level predictions with labels (TP/FP/FN/TN)
- Performance metrics
- Analysis sequences for meta-learning

---

#### 2. `evaluate_predictions`

**Purpose**: Evaluate base model prediction artifacts  
**Module**: `meta_spliceai.cli.evaluate_cli:main`  
**Status**: ✅ Production

**Usage**:
```bash
# Evaluate artifacts directory
evaluate_predictions --artifacts-dir data/mane/GRCh38/openspliceai_eval/meta_models

# Quick summary only
evaluate_predictions --artifacts-dir data/mane/GRCh38/openspliceai_eval/meta_models --summary

# Save metrics to file
evaluate_predictions --artifacts-dir results/ --output-file metrics.json
```

**Output**:
- F1 Score, Precision, Recall
- ROC-AUC, Average Precision
- JSON metrics export

---

#### 3. `annotate_splice_sites`

**Purpose**: Generate and validate splice site annotations from GTF files  
**Module**: `meta_spliceai.cli.splice_sites_cli:main`  
**Status**: ✅ Production

**Usage**:
```bash
# Generate with validation for MANE/GRCh38
annotate_splice_sites --build mane-grch38 --validate

# List available builds
annotate_splice_sites --list-builds

# Custom GTF file
annotate_splice_sites --gtf annotations.gtf --output sites.tsv --validate
```

**Supported Builds**:
- `mane-grch38` - MANE Select v1.3 (OpenSpliceAI)
- `ensembl-grch37` - Ensembl Release 87 (SpliceAI)
- `ensembl-grch38` - Ensembl Release 112

**Output**:
- Enhanced 14-column TSV format
- Consensus sequences
- Validation reports

---

### Python API (Base Layer)

```python
from meta_spliceai.run_base_model import run_base_model_predictions, BaseModelConfig

# Configure and run
config = BaseModelConfig(
    base_model='openspliceai',
    mode='production',
    coverage='full_genome'
)

results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1', 'TP53'],
    config=config,
    verbosity=1
)

# Access results
print(f"Positions: {len(results['positions'])}")
print(f"Metrics: {results['metrics']}")
```

---

## 🧠 Meta Layer (In Development)

### Status: 🚧 **Development** - Core functionality present, CLI pending

The meta layer provides multimodal deep learning for splice site recalibration.

### Planned CLI Commands

**Note**: These are not yet registered in `pyproject.toml` but functionality exists in codebase.

#### 1. `train_meta_model` (Candidate)

**Purpose**: Train multimodal meta-learning model  
**Status**: 🚧 In Development  
**Location**: `meta_spliceai/splice_engine/meta_layer/`

**Proposed Usage**:
```bash
# Train with HyenaDNA (GPU required)
train_meta_model --base-model openspliceai --sequence-encoder hyenadna \
    --config configs/hyenadna.yaml --output-dir models/meta_layer_v1

# Lightweight CPU version  
train_meta_model --base-model openspliceai --sequence-encoder cnn \
    --config configs/lightweight.yaml --output-dir models/meta_layer_v1_lite
```

---

#### 2. `predict_meta_splice` (Candidate)

**Purpose**: Run meta-layer inference for splice site recalibration  
**Status**: 🚧 In Development

**Proposed Usage**:
```bash
# Predict for gene with trained model
predict_meta_splice --model models/meta_layer_v1 --gene BRCA1 --return-exons

# Batch prediction
predict_meta_splice --model models/meta_layer_v1 --genes-file genes.txt
```

---

### Runnable Examples (meta_layer/examples/)

**Location**: `meta_spliceai/splice_engine/meta_layer/examples/`

| Script | Purpose | Status |
|--------|---------|--------|
| `train_simple.py` | Simple training example | ✅ Functional |
| `test_model.py` | Model testing workflow | ✅ Functional |
| `verify_artifacts.py` | Artifact verification | ✅ Functional |
| `run_base_vs_meta_comparison.py` | Base vs meta comparison | ✅ Functional |

**Usage**:
```bash
# Run from project root
python meta_spliceai/splice_engine/meta_layer/examples/train_simple.py

# With custom config
python meta_spliceai/splice_engine/meta_layer/examples/test_model.py --config configs/default.yaml
```

---

### Test/Training Scripts (meta_layer/tests/)

**Location**: `meta_spliceai/splice_engine/meta_layer/tests/`

| Script | Purpose | GPU | Status |
|--------|---------|-----|--------|
| `test_validated_delta_experiments.py` | Validated delta prediction (CPU) | No | ✅ Functional |
| `test_gpu_validated_delta_experiments.py` | Validated delta prediction (GPU) | Yes | ✅ Functional |
| `train_hyenadna_validated_delta.py` | HyenaDNA training | Yes | ✅ Functional |
| `test_gpu_multistep_experiments.py` | Multi-step framework | Yes | ✅ Functional |
| `test_position_localization.py` | Position localization | No | ✅ Functional |

**Usage**:
```bash
# Run validated delta experiments (CPU)
python meta_spliceai/splice_engine/meta_layer/tests/test_validated_delta_experiments.py

# Run GPU training
python meta_spliceai/splice_engine/meta_layer/tests/train_hyenadna_validated_delta.py --gpu
```

---

## 🔬 Case Studies / Variant Analysis (In Development)

### Status: 🚧 **Development** - Entry points available, workflows in progress

Variant analysis and validation workflows for disease-specific splice mutations.

### Entry Point Scripts (case_studies/entry_points/)

**Location**: `meta_spliceai/splice_engine/case_studies/entry_points/`

#### 1. `run_clinvar_pipeline.py` ⭐ **Primary Entry Point**

**Purpose**: Process ClinVar VCF files for variant analysis  
**Status**: ✅ Functional  
**Documentation**: [Complete ClinVar Pipeline README](../case_studies/docs/variant_analysis/COMPLETE_CLINVAR_PIPELINE_README.md)

**Usage**:
```bash
# Simple usage with systematic discovery
python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py \
    clinvar_20250831.vcf.gz results/clinvar_pipeline

# With specific reference genome
python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py \
    clinvar_20250831.vcf.gz results/clinvar_pipeline \
    --reference Homo_sapiens.GRCh38.dna.primary_assembly.fa

# Research mode with all variants
python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py \
    clinvar_20250831.vcf.gz results/research --research-mode

# Pathogenic variants only
python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py \
    clinvar_20250831.vcf.gz results/pathogenic --pathogenic-only
```

**Output**:
- Parsed VCF variants
- WT/ALT sequences ready for delta score computation
- Coordinate-validated variants
- Compatible with base and meta models

---

#### 2. `run_vcf_column_documenter.py`

**Purpose**: Analyze and document VCF column values  
**Status**: ✅ Functional

**Usage**:
```bash
# Basic usage
python meta_spliceai/splice_engine/case_studies/entry_points/run_vcf_column_documenter.py \
    --vcf data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz \
    --output-dir data/ensembl/clinvar/vcf/docs/

# With sample size limit
python meta_spliceai/splice_engine/case_studies/entry_points/run_vcf_column_documenter.py \
    --vcf clinvar.vcf.gz --output-dir docs/ --max-variants 50000
```

**Output**:
- JSON, Markdown, CSV format documentation
- Value frequency analysis
- ClinVar-specific field meanings

---

### Runnable Examples (case_studies/examples/)

**Location**: `meta_spliceai/splice_engine/case_studies/examples/`

| Script | Purpose | Status |
|--------|---------|--------|
| `run_disease_validation_example.py` | Disease-specific validation | 🚧 In Progress |
| `delta_scores_workflow.py` | Delta score computation | 🚧 In Progress |
| `clinvar_openspliceai_workflow.py` | ClinVar + OpenSpliceAI workflow | ✅ Functional |
| `vcf_parsing_tutorial.py` | VCF parsing examples | ✅ Functional |
| `vcf_to_alternative_sites_demo.py` | Alternative splice site detection | 🚧 In Progress |

**Usage**:
```bash
# Run disease validation
python meta_spliceai/splice_engine/case_studies/examples/run_disease_validation_example.py \
    --work-dir ./results --comprehensive

# Run ClinVar workflow
python meta_spliceai/splice_engine/case_studies/examples/clinvar_openspliceai_workflow.py
```

---

## 📊 Entry Points Summary Table

| Entry Point | Layer | Type | Status | Priority |
|-------------|-------|------|--------|----------|
| `run_base_model` | Base | CLI | ✅ Production | ⭐⭐⭐ High |
| `evaluate_predictions` | Base | CLI | ✅ Production | ⭐⭐ Medium |
| `annotate_splice_sites` | Base | CLI | ✅ Production | ⭐⭐ Medium |
| `train_meta_model` | Meta | Candidate CLI | 🚧 Development | ⭐⭐⭐ High |
| `predict_meta_splice` | Meta | Candidate CLI | 🚧 Development | ⭐⭐⭐ High |
| `run_clinvar_pipeline.py` | Variant | Entry Point Script | ✅ Functional | ⭐⭐⭐ High |
| `run_vcf_column_documenter.py` | Variant | Entry Point Script | ✅ Functional | ⭐ Low |
| Meta Layer Examples | Meta | Examples | ✅ Functional | ⭐⭐ Medium |
| Meta Layer Tests | Meta | Test Scripts | ✅ Functional | ⭐ Low |
| Case Studies Examples | Variant | Examples | 🚧 Mixed | ⭐⭐ Medium |

---

## 🚀 Recommended User Workflow

### For Splice Site Prediction (Production-Ready)

```bash
# 1. Generate splice site annotations
annotate_splice_sites --build mane-grch38 --validate

# 2. Run base model predictions
run_base_model --genes BRCA1 TP53 CFTR --base-model openspliceai --mode production

# 3. Evaluate results
evaluate_predictions --artifacts-dir data/mane/GRCh38/openspliceai_eval/meta_models
```

### For Meta-Learning Training (Development)

```bash
# 1. Run base model on training chromosomes
run_base_model --chromosomes 1,2,3 --mode production --coverage chromosome

# 2. Train meta-layer (using examples)
python meta_spliceai/splice_engine/meta_layer/examples/train_simple.py

# 3. Test trained model
python meta_spliceai/splice_engine/meta_layer/examples/test_model.py --model models/meta_v1
```

### For Variant Analysis (Development)

```bash
# 1. Process ClinVar variants
python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py \
    clinvar_20250831.vcf.gz results/clinvar --pathogenic-only

# 2. Run delta score workflow
python meta_spliceai/splice_engine/case_studies/examples/delta_scores_workflow.py \
    --variants results/clinvar/variants.tsv

# 3. Validate with disease cohorts
python meta_spliceai/splice_engine/case_studies/examples/run_disease_validation_example.py \
    --work-dir results/validation --comprehensive
```

---

## 📝 Adding New Entry Points

### To add a CLI command (pyproject.toml):

```toml
[tool.poetry.scripts]
your_command = "meta_spliceai.module.submodule:main"
```

### To create an entry point script:

1. Place in appropriate `entry_points/` directory
2. Add `#!/usr/bin/env python3` shebang
3. Implement `main()` function
4. Make executable: `chmod +x script.py`
5. Document in this reference

---

## 🔍 Entry Point Discovery

### Find all runnable scripts:

```bash
# Find scripts with main() functions
find meta_spliceai -name "*.py" -exec grep -l "def main(" {} \;

# List registered CLI commands
grep -A 10 "\[tool.poetry.scripts\]" pyproject.toml

# Find entry point scripts
find meta_spliceai -path "*/entry_points/*.py" -type f
```

---

## 📚 Related Documentation

- **Base Layer**: [Base Model Support](../docs/base_models/UNIVERSAL_BASE_MODEL_SUPPORT.md)
- **Meta Layer**: [Meta Layer README](../meta_spliceai/splice_engine/meta_layer/README.md)
- **Case Studies**: [Case Studies README](../meta_spliceai/splice_engine/case_studies/README.md)
- **CLI Tools**: [CLI Splice Sites](../docs/CLI_SPLICE_SITES.md)

---

**Last Updated**: January 30, 2026  
**Maintained By**: MetaSpliceAI Development Team  
**Status Legend**: ✅ Production | 🚧 In Development | 📋 Planned
