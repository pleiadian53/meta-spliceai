# Variant Analysis Documentation

**Central documentation hub for MetaSpliceAI's Variant Analysis and Disease Validation (Case Studies)**

---

## 📚 Quick Navigation

### Getting Started
- **[Package README](../../meta_spliceai/splice_engine/case_studies/README.md)** - Main package overview
- **[Entry Points README](../../meta_spliceai/splice_engine/case_studies/entry_points/README.md)** - Command-line tools

### Core Workflows
- **[Complete ClinVar Pipeline](../../meta_spliceai/splice_engine/case_studies/docs/variant_analysis/COMPLETE_CLINVAR_PIPELINE_README.md)** ⭐ Primary workflow
- **[VCF Variant Analysis Workflow](../../meta_spliceai/splice_engine/case_studies/docs/VCF_VARIANT_ANALYSIS_WORKFLOW.md)** - General VCF processing
- **[VCF to Alternative Splice Sites](../../meta_spliceai/splice_engine/case_studies/docs/VCF_TO_ALTERNATIVE_SPLICE_SITES_WORKFLOW.md)** - Alternative splicing

### System Design
- **[System Design Analysis (Q1-Q7)](../../meta_spliceai/splice_engine/case_studies/docs/SYSTEM_DESIGN_ANALYSIS_Q1_Q7.md)** - Architecture decisions
- **[Variant Splicing Biology (Q10-Q12)](../../meta_spliceai/splice_engine/case_studies/docs/VARIANT_SPLICING_BIOLOGY_Q10_Q12.md)** - Biological context
- **[Implementation Guide](../../meta_spliceai/splice_engine/case_studies/docs/IMPLEMENTATION_GUIDE.md)** - Implementation details

### OpenSpliceAI Integration
- **[OpenSpliceAI Variant Analysis (Q8-Q9)](../../meta_spliceai/splice_engine/case_studies/docs/OPENSPLICEAI_VARIANT_ANALYSIS_Q8_Q9.md)** - OpenSpliceAI specifics
- **[Delta Score Bridge Implementation](../../meta_spliceai/splice_engine/case_studies/docs/DELTA_SCORE_BRIDGE_IMPLEMENTATION.md)** - Score computation
- **[OpenSpliceAI Integration](../../meta_spliceai/splice_engine/case_studies/docs/DEV_OPENSPLICEAI_INTEGRATION.md)** - Development notes

### Data Processing
- **[Universal VCF Parser Guide](../../meta_spliceai/splice_engine/case_studies/docs/UNIVERSAL_VCF_PARSER_GUIDE.md)** - VCF parsing
- **[VCF Analysis Tools](../../meta_spliceai/splice_engine/case_studies/docs/VCF_ANALYSIS_TOOLS_GUIDE.md)** - Analysis utilities
- **[Training Data Analysis](../../meta_spliceai/splice_engine/case_studies/docs/TRAINING_DATA_ANALYSIS.md)** - Data QC

### Tutorials
- **[ClinVar Workflow Steps 1-2](../../meta_spliceai/splice_engine/case_studies/docs/tutorials/CLINVAR_WORKFLOW_STEPS_1_2_TUTORIAL.md)** - Part 1
- **[ClinVar Workflow Step 2.5](../../meta_spliceai/splice_engine/case_studies/docs/tutorials/CLINVAR_WORKFLOW_STEP_2.5_TUTORIAL.md)** - Part 2
- **[Universal VCF Parsing](../../meta_spliceai/splice_engine/case_studies/docs/tutorials/UNIVERSAL_VCF_PARSING_TUTORIAL.md)** - VCF tutorial

### Disease-Specific
- **[Disease-Specific Meta-Learning Roadmap](../../meta_spliceai/splice_engine/case_studies/docs/DISEASE_SPECIFIC_META_LEARNING_ROADMAP.md)** - Future directions
- **[Enhanced Alternative Splicing](../../meta_spliceai/splice_engine/case_studies/docs/ENHANCED_ALTERNATIVE_SPLICING_SUMMARY.md)** - Alternative splicing

### Variant Analysis Details
- **[Pipeline Solution Summary](../../meta_spliceai/splice_engine/case_studies/docs/variant_analysis/PIPELINE_SOLUTION_SUMMARY.md)** - Pipeline overview
- **[Context Window Strategy](../../meta_spliceai/splice_engine/case_studies/docs/variant_analysis/CONTEXT_WINDOW_STRATEGY.md)** - Sequence extraction
- **[Enhanced Mechanisms](../../meta_spliceai/splice_engine/case_studies/docs/variant_analysis/ENHANCED_MECHANISMS_SUMMARY.md)** - Advanced features
- **[VCF Column Documenter](../../meta_spliceai/splice_engine/case_studies/docs/variant_analysis/README_VCF_COLUMN_DOCUMENTER.md)** - VCF documentation

---

## 🎯 Overview

The **Variant Analysis** (Case Studies) component provides infrastructure for:

1. **Variant Data Ingestion** - From ClinVar, SpliceVarDB, MutSpliceDB, DBASS
2. **VCF Processing** - Parse, validate, and standardize variants
3. **Delta Score Computation** - WT vs ALT splice score differences
4. **Disease Validation** - Validate models on disease cohorts
5. **Clinical Annotation** - Pathogenic/benign classification

### Key Features

- ✅ **Systematic VCF Processing** - Robust ClinVar pipeline
- ✅ **Coordinate Validation** - Ensure variant accuracy
- ✅ **WT/ALT Sequence Construction** - Ready for delta scores
- ✅ **Multiple Database Support** - ClinVar, SpliceVarDB, MutSpliceDB
- 🚧 **Disease Validation Workflows** - In development
- 🚧 **Meta-Model Integration** - Connect to meta layer

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│               VARIANT ANALYSIS PIPELINE                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐                                         │
│  │ VCF Input       │                                         │
│  │ (ClinVar, etc)  │                                         │
│  └────────┬────────┘                                         │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────┐                                         │
│  │ VCF Parser      │ ← Universal parser                     │
│  │ & Validator     │   + Coordinate validation              │
│  └────────┬────────┘                                         │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────┐                                         │
│  │ WT/ALT Sequence │ ← Extract context sequences            │
│  │ Constructor     │   (501nt windows)                      │
│  └────────┬────────┘                                         │
│           │                                                   │
│           ├──────────┬────────────┐                          │
│           ▼          ▼            ▼                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                    │
│  │ Base     │ │ Meta     │ │ Disease  │                    │
│  │ Model    │ │ Model    │ │ Cohort   │                    │
│  │ Δ Scores │ │ Δ Scores │ │ Analysis │                    │
│  └──────────┘ └──────────┘ └──────────┘                    │
│                                                               │
│  OUTPUT: Delta scores, validation metrics, clinical insights │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Process ClinVar Variants ⭐ Most Common

```bash
# Simple usage with systematic discovery
python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py \
    clinvar_20250831.vcf.gz results/clinvar_pipeline

# With specific reference genome
python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py \
    clinvar_20250831.vcf.gz results/clinvar_pipeline \
    --reference Homo_sapiens.GRCh38.dna.primary_assembly.fa

# Pathogenic variants only (clinical focus)
python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py \
    clinvar_20250831.vcf.gz results/pathogenic --pathogenic-only

# Research mode (all variants)
python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py \
    clinvar_20250831.vcf.gz results/research --research-mode
```

**Output**:
- `variants_parsed.tsv` - Parsed variants
- `variants_wt_alt.tsv` - WT/ALT sequences ready for delta scores
- `coordinate_validation.json` - Validation report

---

### 2. Document VCF Columns

```bash
# Analyze VCF structure
python meta_spliceai/splice_engine/case_studies/entry_points/run_vcf_column_documenter.py \
    --vcf data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz \
    --output-dir data/ensembl/clinvar/vcf/docs/
```

---

### 3. Run Disease Validation (Coming Soon)

```bash
# Validate on disease cohorts
python meta_spliceai/splice_engine/case_studies/examples/run_disease_validation_example.py \
    --work-dir ./results \
    --meta-model ./models/meta_v1.pkl \
    --comprehensive \
    --diseases lung_cancer breast_cancer
```

---

## 📊 Supported Databases

### 1. ClinVar ⭐ Primary Database

**Status**: ✅ Fully Integrated  
**Source**: NCBI ClinVar  
**Size**: ~2.5M variants  
**Focus**: Clinical significance annotations

**Key Fields**:
- CLNSIG - Clinical significance (Pathogenic/Benign)
- CLNREVSTAT - Review status
- GENEINFO - Gene annotations
- MC - Molecular consequence

**Use Cases**:
- Clinical variant validation
- Pathogenic variant analysis
- Benchmarking base and meta models

---

### 2. SpliceVarDB

**Status**: ✅ Supported (via meta_layer)  
**Source**: Sullivan et al. 2024  
**Size**: >50,000 validated variants  
**Focus**: Experimentally validated splice variants

**Key Features**:
- Experimental validation methods
- Splice-altering classifications
- Ground truth for training
- Cryptic splice site annotations

**Integration**: Used by meta_layer for validated delta prediction

---

### 3. MutSpliceDB

**Status**: 🚧 Parser Available  
**Source**: NCI TCGA/CCLE  
**Focus**: Cancer-specific splice mutations

**Key Features**:
- RNA evidence from cancer samples
- Therapeutic target annotations (e.g., MET exon 14)
- Tumor type associations

**Use Cases**:
- Cancer variant validation
- Therapeutic target discovery
- Disease-specific analysis

---

### 4. DBASS5/DBASS3

**Status**: 🚧 Parser Available  
**Focus**: Aberrant splice sites  
**Size**: Curated collection

**Key Features**:
- Cryptic splice site activation
- Strength scores
- Classic examples (CFTR pseudoexons)

**Use Cases**:
- Cryptic site validation
- Deep intronic variant analysis

---

## 📁 Package Structure

```
case_studies/
├── data_sources/          # Database ingesters
│   ├── base.py           # Common infrastructure
│   ├── clinvar.py        # ClinVar ingester
│   ├── splicevardb.py    # SpliceVarDB ingester
│   ├── mutsplicedb.py    # MutSpliceDB ingester
│   ├── dbass.py          # DBASS ingester
│   └── resource_manager.py  # Resource management
│
├── formats/               # Data format handling
│   ├── hgvs_parser.py    # HGVS notation
│   ├── vcf_parser.py     # VCF parsing
│   └── bed_converter.py  # BED format
│
├── workflows/             # Analysis workflows
│   ├── disease_validation.py  # Disease validation
│   ├── delta_score_workflow.py  # Delta computation
│   └── cryptic_site_workflow.py  # Cryptic sites
│
├── tools/                 # Utility tools
│   ├── coordinate_validator.py  # Validate coordinates
│   ├── sequence_extractor.py    # Extract sequences
│   ├── vcf_column_documenter.py # Document VCF
│   └── variant_filter.py        # Filter variants
│
├── entry_points/          # User-facing entry points
│   ├── run_clinvar_pipeline.py ⭐ Primary tool
│   ├── run_vcf_column_documenter.py
│   └── project_root_utils.py  # Path utilities
│
├── examples/              # Example workflows
│   ├── clinvar_openspliceai_workflow.py
│   ├── vcf_parsing_tutorial.py
│   ├── delta_scores_workflow.py
│   └── run_disease_validation_example.py
│
└── docs/                  # Comprehensive documentation
    ├── README.md          # Package overview
    ├── variant_analysis/  # Variant analysis docs
    ├── tutorials/         # Step-by-step tutorials
    └── variant_splicing/  # Biological background
```

---

## 🎓 Learning Path

### Beginner

1. Read [Package README](../../meta_spliceai/splice_engine/case_studies/README.md)
2. Understand [Complete ClinVar Pipeline](../../meta_spliceai/splice_engine/case_studies/docs/variant_analysis/COMPLETE_CLINVAR_PIPELINE_README.md)
3. Run [ClinVar pipeline](../../meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py) on test data
4. Review [VCF parsing tutorial](../../meta_spliceai/splice_engine/case_studies/examples/vcf_parsing_tutorial.py)

### Intermediate

1. Study [System Design Analysis](../../meta_spliceai/splice_engine/case_studies/docs/SYSTEM_DESIGN_ANALYSIS_Q1_Q7.md)
2. Understand [Delta Score Implementation](../../meta_spliceai/splice_engine/case_studies/docs/DELTA_SCORE_BRIDGE_IMPLEMENTATION.md)
3. Work through [ClinVar Workflow Tutorials](../../meta_spliceai/splice_engine/case_studies/docs/tutorials/)
4. Run [OpenSpliceAI workflow](../../meta_spliceai/splice_engine/case_studies/examples/clinvar_openspliceai_workflow.py)

### Advanced

1. Read [Disease-Specific Roadmap](../../meta_spliceai/splice_engine/case_studies/docs/DISEASE_SPECIFIC_META_LEARNING_ROADMAP.md)
2. Implement [disease validation workflow](../../meta_spliceai/splice_engine/case_studies/examples/run_disease_validation_example.py)
3. Study [Variant Splicing Biology](../../meta_spliceai/splice_engine/case_studies/docs/VARIANT_SPLICING_BIOLOGY_Q10_Q12.md)
4. Integrate with meta-layer delta prediction

---

## 🔬 Key Workflows

### Workflow 1: ClinVar Variant Processing ⭐

```bash
# 1. Download ClinVar VCF
wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar_20250831.vcf.gz

# 2. Run pipeline
python meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py \
    clinvar_20250831.vcf.gz results/clinvar --pathogenic-only

# 3. Check outputs
ls results/clinvar/
# → variants_parsed.tsv
# → variants_wt_alt.tsv
# → coordinate_validation.json
```

---

### Workflow 2: Delta Score Computation

```bash
# 1. Get WT/ALT sequences from ClinVar pipeline
# 2. Run base model predictions
run_base_model --variants-file results/clinvar/variants_wt_alt.tsv \
    --output-dir results/base_model_deltas

# 3. Compute delta scores (coming soon)
python meta_spliceai/splice_engine/case_studies/examples/delta_scores_workflow.py \
    --wt-predictions results/base_model_deltas/wt/ \
    --alt-predictions results/base_model_deltas/alt/ \
    --output results/delta_scores.tsv
```

---

### Workflow 3: Disease Cohort Validation

```bash
# Validate on lung cancer cohort
python meta_spliceai/splice_engine/case_studies/examples/run_disease_validation_example.py \
    --work-dir results/lung_cancer \
    --meta-model models/meta_v1.pkl \
    --diseases lung_cancer \
    --databases SpliceVarDB MutSpliceDB \
    --comprehensive
```

---

## 📊 Output Formats

### Parsed Variants TSV

```
CHROM  POS     REF  ALT  GENE     CLNSIG      MC              HGVS
chr7   117..   G    A    CFTR     Pathogenic  splice_donor    c.1521+1G>A
chr17  43..    C    T    BRCA1    Pathogenic  splice_accept   c.5152-1C>T
```

### WT/ALT Sequences TSV

```
VARIANT_ID  WT_SEQ_501NT    ALT_SEQ_501NT   POSITION
var_001     ACGT...ACGT     ACGT...AAGT     250
var_002     TGCA...TGCA     TGCA...CGCA     250
```

### Delta Scores TSV

```
VARIANT_ID  GENE   DELTA_DONOR  DELTA_ACCEPTOR  PREDICTED_EFFECT
var_001     CFTR   -0.85        0.02            Loss of donor
var_002     BRCA1  0.02         -0.92           Loss of acceptor
```

---

## 🚧 Development Status

### Completed ✅

- ClinVar VCF parsing
- Universal VCF parser
- Coordinate validation
- WT/ALT sequence construction
- Reference genome integration
- VCF column documentation tool
- Entry point infrastructure
- Systematic path discovery
- Multiple database parsers (structure)

### In Progress 🚧

- Delta score computation workflow
- Disease validation workflows
- Meta-model integration
- SpliceVarDB full integration
- MutSpliceDB validation
- Cryptic site detection pipeline

### Planned 📋

- Real-time variant analysis API
- Web-based visualization
- Batch processing optimization
- Pre-computed variant database
- Clinical decision support integration

---

## 🤝 Integration Points

### With Base Layer

**Input**: VCF variants → WT/ALT sequences  
**Process**: Base model predictions on WT and ALT  
**Output**: Delta scores for each variant

### With Meta Layer

**Input**: WT/ALT sequences + base model scores  
**Process**: Meta-layer recalibration  
**Output**: Improved delta score predictions

### With Case Studies

**Input**: Disease cohort variants  
**Process**: Validation against experimental data  
**Output**: Model performance metrics

---

## 📞 Support

- **Primary Tool**: [run_clinvar_pipeline.py](../../meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py)
- **Documentation**: [Complete ClinVar Pipeline README](../../meta_spliceai/splice_engine/case_studies/docs/variant_analysis/COMPLETE_CLINVAR_PIPELINE_README.md)
- **Tutorials**: [Tutorial Directory](../../meta_spliceai/splice_engine/case_studies/docs/tutorials/)
- **Examples**: [Examples Directory](../../meta_spliceai/splice_engine/case_studies/examples/)

---

## 🔍 Key Use Cases

### 1. Clinical Variant Analysis

Process pathogenic variants from ClinVar for clinical validation

### 2. Disease Cohort Studies

Validate models on disease-specific mutation cohorts

### 3. Therapeutic Target Discovery

Identify splice-altering mutations for antisense therapy

### 4. Meta-Model Training

Generate training data with validated ground truth

### 5. Biomarker Development

Discover splice variant biomarkers for disease progression

---

**Last Updated**: January 30, 2026  
**Status**: ✅ Core Pipeline Functional, 🚧 Advanced Features In Development  
**Recommended Entry Point**: [run_clinvar_pipeline.py](../../meta_spliceai/splice_engine/case_studies/entry_points/run_clinvar_pipeline.py)
