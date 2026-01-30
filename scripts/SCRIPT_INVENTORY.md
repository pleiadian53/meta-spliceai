# MetaSpliceAI Script Inventory & Management System

## 📋 **Script Organization Philosophy**

This document provides a systematic way to track, categorize, and manage all scripts in the MetaSpliceAI project. Scripts are organized by **purpose** and **workflow stage** to make them easy to find and use.

**⚠️ Updated January 30, 2026**: Scripts have been reorganized into functional subdirectories. All paths in this document reflect the new organization.

## 🗂️ **Script Categories**

### **1. Setup & Environment** (`installation/`, `gpu_env_setup/`)
Scripts for setting up and testing the MetaSpliceAI environment.

| Script | Purpose | Usage | Dependencies |
|--------|---------|-------|--------------|
| `installation/migrate_conda_to_mamba.sh` | Migrate from conda to mamba | One-time setup | conda, mamba |
| `gpu_env_setup/test_gpu_installation.sh` | Basic GPU testing | Setup verification | CUDA, GPU |
| `testing/test_gpu_performance.py` | Comprehensive GPU testing | Performance validation | tensorflow, torch |
| `installation/diagnose_gpu_environment.py` | GPU diagnostics | Troubleshooting | Multiple GPU libs |
| `testing/check_gpu.py` | Quick GPU check | Development | tensorflow |
| `testing/check_versions.py` | Version compatibility | Debugging | Standard libs |
| `installation/fix_ml_dependencies.py` | Fix ML library issues | Troubleshooting | pip, conda |

### **2. Data Preparation & Validation**
Scripts for preparing, validating, and inspecting training data.

| Script | Purpose | Usage | Input | Output |
|--------|---------|-------|-------|--------|
| `analysis/analyze_training_data.py` | Comprehensive data analysis | Data QC | Training datasets | Analysis reports |
| `validation/validate_meta_model_training_data.py` | Training data validation | Pre-training QC | Meta-model data | Validation report |
| `testing/test_dataset_integrity.py` | Dataset integrity checks | Data validation | Any dataset | Pass/fail report |
| `data_processing/validate_sequences.py` | Sequence validation | Data QC | FASTA/sequences | Validation results |
| `testing/check_dataset.py` | Quick dataset check | Development | Dataset files | Summary stats |
| `analysis/inspect_analysis_sequences.py` | Sequence inspection | Data exploration | Analysis sequences | Detailed inspection |
| `analysis/inspect_meta_model_training_data.py` | Training data inspection | Data exploration | Training data | Feature analysis |
| `analysis/inspect_parquet.py` | Parquet file inspection | Data debugging | Parquet files | File contents |

### **3. Feature Engineering & Enhancement**
Scripts for creating and enhancing features for meta-model training.

| Script | Purpose | Usage | Input | Output |
|--------|---------|-------|-------|--------|
| `data_processing/enhance_splice_sites.py` | Enhance splice site annotations | Feature engineering | Splice sites | Enhanced annotations |
| `data_processing/consolidate_sequences.py` | Consolidate sequence data | Data preparation | Multiple sequences | Consolidated data |
| `data_processing/patch_gene_type.py` | Fix gene type annotations | Data correction | Gene annotations | Corrected annotations |
| `data_processing/patch_structural_features.py` | Fix structural features | Data correction | Feature data | Corrected features |
| `data_processing/generate_gene_manifest.py` | Create gene manifest | Data organization | Gene data | Gene manifest |
| `data_processing/query_gene_manifest.py` | Query gene information | Data exploration | Gene manifest | Query results |

### **4. Analysis & Visualization** (`analysis/`)
Scripts for analyzing results and creating visualizations.

| Script | Purpose | Usage | Input | Output |
|--------|---------|-------|-------|--------|
| `analysis/analyze_splice_positions.py` | Splice position analysis | Result analysis | Position data | Analysis plots |
| `analysis/analyze_overlapping_genes.py` | Gene overlap analysis | Data exploration | Gene annotations | Overlap analysis |
| `analysis/plot_splice_sites_hist.py` | Splice site histograms | Visualization | Splice site data | Histogram plots |
| `utilities/meta_model_concept_diagram.py` | Concept diagrams | Documentation | N/A | Concept diagrams |
| `utilities/position_centric_data_repr_diagram.py` | Data representation diagrams | Documentation | N/A | Architecture diagrams |
| `analysis/splice_site_visualization/` | Splice site visualizations | Result visualization | Various | Plots and charts |

### **5. Model Evaluation** (`evaluation/`)
Scripts for evaluating model performance and conducting cross-validation analysis.

| Script | Purpose | Usage | Input | Output |
|--------|---------|-------|-------|--------|
| `analysis/f1_pr_analysis_merged.py` | **F1-based PR analysis** | **Post-CV analysis** | **Gene CV results** | **F1-optimized plots** |
| `evaluation/cv_metrics/` | CV metrics analysis | Performance evaluation | CV results | Metrics reports |
| `evaluation/feature_importance/` | Feature importance analysis | Model interpretation | Trained models | Importance plots |
| `evaluation/overfitting/` | Overfitting analysis | Model validation | Training results | Diagnostic plots |
| `testing/test_transcript_topk.py` | Transcript-level evaluation | Performance testing | Predictions | Top-k accuracy |

### **6. Training & Scaling** (`training/`, `scaling_solutions/`)
Scripts for model training, especially large-scale and multi-GPU training.

| Script | Purpose | Usage | Input | Output |
|--------|---------|-------|-------|--------|
| `training/run_multi_gpu_training.py` | Multi-GPU training | Large-scale training | Training data | Trained models |
| `training/run_multi_gpu_training_ig.py` | Multi-GPU with IG analysis | Advanced training | Training data | Models + IG analysis |
| `testing/test_gpu_performance.py` | GPU performance testing | Hardware validation | N/A | Performance metrics |
| `testing/test_gpu_xgboost.py` | XGBoost GPU testing | Library validation | Sample data | Performance test |
| `scaling_solutions/` | Scaling solutions | Large datasets | Various | Optimized workflows |

### **7. Quality Control & Testing** (`testing/`, `validation/`)
Scripts for testing, validation, and quality control.

| Script | Purpose | Usage | Input | Output |
|--------|---------|-------|-------|--------|
| `validation/pre_flight_checks.py` | Comprehensive pre-run checks | Pre-training validation | System state | Check results |
| `testing/test_datasets_loading.py` | Dataset loading tests | Development testing | Dataset configs | Load test results |
| `testing/test_leakage_probe.py` | Data leakage detection | Quality control | Training data | Leakage analysis |
| `testing/test_leakage_probe_debug.py` | Leakage debugging | Debugging | Training data | Debug analysis |
| `validation/validate_artifacts.py` | Artifact validation | Post-training QC | Model artifacts | Validation report |

### **8. Utilities & Maintenance** (`utilities/`, `maintenance/`, `archive/`)
General utility scripts for maintenance and system operations.

| Script | Purpose | Usage | Input | Output |
|--------|---------|-------|-------|--------|
| `maintenance/cleanup_artifacts.py` | Clean up old artifacts | Maintenance | File system | Cleanup report |
| `installation/build_for_fabric.sh` | Build for Fabric deployment | Deployment | Source code | Deployment package |
| `training/post_training_analysis.sh` | Post-training analysis | Workflow automation | Training results | Analysis suite |
| `installation/deepspeed_patch.py` | DeepSpeed compatibility | Library patching | N/A | Patched environment |
| `testing/check_polars_version.py` | Polars version check | Compatibility | N/A | Version info |
| `utilities/manage_scripts.py` | Script management | Organization | Scripts dir | Management tasks |
| `archive/` | Outdated/temporary scripts | Historical reference | Various | Archived files |

## 🎯 **Usage Patterns by Workflow Stage**

### **Stage 1: Environment Setup**
```bash
# Initial setup
./scripts/installation/migrate_conda_to_mamba.sh
./scripts/gpu_env_setup/test_gpu_installation.sh
python scripts/pre_flight_checks.py
```

### **Stage 2: Data Preparation**
```bash
# Data validation and preparation
python scripts/analysis/analyze_training_data.py
python scripts/validation/validate_meta_model_training_data.py
python scripts/data_processing/enhance_splice_sites.py
```

### **Stage 3: Training**
```bash
# Model training (various scales)
python scripts/training/run_multi_gpu_training.py
python scripts/testing/test_leakage_probe.py  # Quality control
```

### **Stage 4: Evaluation & Analysis**
```bash
# Post-training analysis
python scripts/analysis/f1_pr_analysis_merged.py results/gene_cv_pc_1000_3mers_run_4
python scripts/evaluation/cv_metrics/analyze_cv_results.py
python scripts/testing/test_transcript_topk.py
```

### **Stage 5: Maintenance**
```bash
# System maintenance
python scripts/maintenance/cleanup_artifacts.py
python scripts/validation/validate_artifacts.py
```

## 📊 **Script Usage Frequency & Priority**

### **High-Frequency Scripts** (Daily/Weekly Use)
- `analysis/f1_pr_analysis_merged.py` - Post-CV analysis
- `validation/pre_flight_checks.py` - Pre-training validation
- `analysis/analyze_training_data.py` - Data exploration
- `testing/test_gpu_performance.py` - Performance monitoring

### **Medium-Frequency Scripts** (Monthly/Project Use)
- `validation/validate_meta_model_training_data.py` - Data validation
- `training/run_multi_gpu_training.py` - Large-scale training
- `maintenance/cleanup_artifacts.py` - Maintenance
- `data_processing/enhance_splice_sites.py` - Feature engineering

### **Low-Frequency Scripts** (Setup/Troubleshooting)
- `installation/fix_ml_dependencies.py` - Troubleshooting
- `installation/diagnose_gpu_environment.py` - Hardware issues
- `installation/deepspeed_patch.py` - Library compatibility
- `installation/build_for_fabric.sh` - Deployment

## 🔍 **Finding the Right Script**

### **By Purpose**
- **Need to analyze CV results?** → `analysis/f1_pr_analysis_merged.py`
- **GPU not working?** → `installation/diagnose_gpu_environment.py`
- **Data looks wrong?** → `analysis/analyze_training_data.py`
- **Training failing?** → `validation/pre_flight_checks.py`
- **Need performance metrics?** → `testing/test_gpu_performance.py`

### **By Input Type**
- **Gene CV results** → `analysis/f1_pr_analysis_merged.py`
- **Training datasets** → `analysis/analyze_training_data.py`, `validation/validate_meta_model_training_data.py`
- **Parquet files** → `analysis/inspect_parquet.py`
- **Sequence data** → `data_processing/validate_sequences.py`, `analysis/inspect_analysis_sequences.py`
- **Model artifacts** → `validation/validate_artifacts.py`

### **By Output Type**
- **Need plots/visualizations** → `analysis/f1_pr_analysis_merged.py`, `analysis/plot_splice_sites_hist.py`
- **Need validation reports** → `validation/validate_meta_model_training_data.py`, `testing/test_dataset_integrity.py`
- **Need performance metrics** → `testing/test_gpu_performance.py`, `testing/test_transcript_topk.py`
- **Need debugging info** → `installation/diagnose_gpu_environment.py`, `testing/test_leakage_probe_debug.py`

## 📚 **Documentation Standards**

### **Required Documentation for Each Script**
1. **Header comment** with purpose, usage, and examples
2. **README entry** in appropriate category
3. **Dependencies** clearly listed
4. **Input/Output** format specification
5. **Usage examples** with real commands

### **Documentation Files**
- `SCRIPT_INVENTORY.md` (this file) - Complete script catalog
- `README.md` - Quick reference and directory structure
- Category-specific READMEs in subdirectories
- Individual script documentation (e.g., `README_f1_pr_analysis.md`)

## 🔄 **Maintenance Guidelines**

### **Adding New Scripts**
1. Place in appropriate category directory
2. Add entry to this inventory
3. Create/update category README
4. Add header documentation to script
5. Test and validate functionality

### **Deprecating Scripts**
1. Mark as deprecated in inventory
2. Add deprecation notice to script header
3. Provide migration path to replacement
4. Remove after appropriate notice period

### **Regular Maintenance Tasks**
- Monthly: Review and update script inventory
- Quarterly: Clean up deprecated scripts
- Annually: Reorganize categories if needed
- As needed: Update documentation and examples

---

**Last Updated**: January 30, 2026  
**Total Scripts Tracked**: 60+ (after reorganization)  
**Categories**: 8 main categories across 23 subdirectories  
**Documentation Coverage**: Comprehensive with usage examples  
**Reorganization Status**: ✅ Complete - All scripts categorized and moved
