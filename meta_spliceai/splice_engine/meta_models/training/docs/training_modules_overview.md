# Training Modules Overview

**Document Created:** January 10, 2025  
**Last Updated:** January 10, 2025  
**Package:** `meta_spliceai.splice_engine.meta_models.training`

## Table of Contents

1. [Package Structure](#package-structure)
2. [Core Training Scripts](#core-training-scripts)
3. [Evaluation and Analysis Modules](#evaluation-and-analysis-modules)
4. [Utility and Support Modules](#utility-and-support-modules)
5. [Legacy and Alternative Implementations](#legacy-and-alternative-implementations)
6. [Configuration and Examples](#configuration-and-examples)
7. [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)

---

## Package Structure

The training package contains modules for meta-model training, evaluation, and analysis. The package is organized into several functional categories:

```
training/
├── docs/                           # Documentation
├── configs/                        # Configuration files
├── examples/                       # Example scripts
├── __pycache__/                   # Python cache
│
├── # Core Training Scripts
├── run_gene_cv_sigmoid.py          # Main gene-aware CV with sigmoid ensemble
├── run_loco_cv_multiclass_scalable.py  # Leave-one-chromosome-out CV
├── run_ablation_multiclass.py      # Ablation studies
│
├── # Evaluation and Analysis
├── eval_meta_splice.py             # Meta-model splice site evaluation
├── meta_evaluation_utils.py        # Evaluation utilities
├── gene_score_delta.py             # Per-gene performance comparison (binary)
├── gene_score_delta_multiclass.py  # Per-gene performance comparison (multiclass)
├── compare_splice_performance.py   # Base vs meta performance comparison
│
├── # Utilities and Support
├── classifier_utils.py             # Core classifier utilities and workflows
├── label_utils.py                  # Label encoding/decoding utilities
├── datasets.py                     # Dataset handling utilities
├── feature_manifest.py             # Feature manifest management
├── scalability_utils.py            # Memory and performance optimizations
├── external_memory_utils.py        # External memory training
│
├── # Diagnostics and Quality Control
├── shuffle_label_sanity_multiclass.py  # Label shuffle sanity checks
├── run_comprehensive_diagnostics.py    # Comprehensive diagnostic suite
├── run_diagnostics.py              # Basic diagnostics
├── leakage_probe.py                # Feature leakage detection
├── threshold_scan.py               # Threshold optimization
│
├── # Specialized Analysis
├── meta_splice_predict.py          # Meta-model splice prediction
├── chromosome_split.py             # Chromosome-based data splitting
├── chunked_datasets.py             # Chunked dataset processing
├── incremental.py                  # Incremental learning
├── explainers.py                   # Model explanation utilities
│
└── # Legacy and Alternatives
    ├── run_gene_cv_sigmoid_v0.py    # Legacy gene CV implementation
    ├── eval_meta_splice_v0.py       # Legacy evaluation
    └── ...                         # Other legacy modules
```

---

## Core Training Scripts

### **run_gene_cv_sigmoid.py**
**Primary gene-aware cross-validation script**

- **Purpose**: Train and evaluate meta-models using gene-aware CV to prevent data leakage
- **Architecture**: 3-Independent Sigmoid Ensemble with optional calibration
- **Key Features**: 
  - Gene-aware GroupKFold ensures splice sites from same gene stay in same fold
  - Optional Platt scaling or isotonic regression calibration
  - Comprehensive evaluation with 7 visualization plots
  - Overfitting monitoring and early stopping
  - Feature importance analysis with 4 different methods
- **Usage**: Main production training script for moderate datasets (<10K genes)
- **Related**: Uses `classifier_utils.py`, `gene_score_delta_multiclass.py`

### **run_loco_cv_multiclass_scalable.py**
**Leave-one-chromosome-out cross-validation**

- **Purpose**: Test extreme out-of-distribution generalization by holding out entire chromosomes
- **Architecture**: Same 3-Independent Sigmoid Ensemble as gene CV
- **Key Features**:
  - Memory-efficient processing for large datasets (20K+ genes)
  - Identical analysis pipeline to gene CV (feature parity achieved Jan 2025)
  - Scalable external memory support
  - Conservative performance estimates for production deployment
- **Usage**: Large-scale training and out-of-distribution testing
- **Related**: Shares utilities with gene CV, uses `scalability_utils.py`

### **run_ablation_multiclass.py**
**Ablation study framework**

- **Purpose**: Systematic feature ablation to understand feature importance
- **Features**: Remove feature groups and measure performance impact
- **Usage**: Research and feature engineering optimization

---

## Evaluation and Analysis Modules

### **eval_meta_splice.py**
**Comprehensive meta-model evaluation**

- **Purpose**: Evaluate meta-model performance on splice site prediction tasks
- **Features**:
  - Position-level and gene-level evaluation
  - ROC curves, precision-recall curves
  - Consensus window analysis
  - Threshold optimization
- **Output**: TSV files with detailed metrics, visualization plots
- **Usage**: Post-training evaluation and performance assessment

### **gene_score_delta.py** & **gene_score_delta_multiclass.py**
**Per-gene performance comparison**

- **Purpose**: Compare base model vs meta-model performance at gene level
- **Key Difference**:
  - `gene_score_delta.py`: Binary classification (splice vs non-splice)
  - `gene_score_delta_multiclass.py`: 3-class classification (donor/acceptor/neither)
- **Features**:
  - Gene-level aggregation of probability scores
  - Accuracy delta computation
  - Top-k accuracy analysis
  - Fix/regression counting
- **Usage**: Understanding where meta-model improves performance
- **⚠️ Known Issue**: Feature harmonization problems (see [Troubleshooting](#common-issues-and-troubleshooting))

### **meta_evaluation_utils.py**
**Evaluation utility functions**

- **Purpose**: Shared utilities for evaluation scripts
- **Features**: Metric computation, visualization helpers, file I/O
- **Usage**: Supporting module for evaluation scripts

### **compare_splice_performance.py**
**Base vs meta performance comparison**

- **Purpose**: Systematic comparison of base model and meta-model performance
- **Features**: Side-by-side metrics, statistical significance testing
- **Usage**: Production validation and performance reporting

---

## Utility and Support Modules

### **classifier_utils.py**
**Core classifier utilities and workflows**

- **Purpose**: Central hub for training workflows and utilities
- **Key Functions**:
  - `gene_score_delta()`: Wrapper for gene-level analysis
  - `leakage_probe()`: Feature leakage detection
  - `shap_importance()`: SHAP-based feature importance
  - Dataset loading and preprocessing utilities
- **Features**: Memory optimization, error handling, logging
- **Usage**: Core dependency for most training scripts

### **label_utils.py**
**Label encoding and decoding utilities**

- **Purpose**: Consistent label handling across all modules
- **Features**:
  - String to numeric label encoding
  - Label validation and consistency checks
  - Support for both binary and multiclass labels
- **Constants**: `LABEL_MAP_STR`, `LABEL_MAP_INT`
- **Usage**: Imported by all modules handling labels

### **datasets.py**
**Dataset handling utilities**

- **Purpose**: Dataset loading, validation, and preprocessing
- **Features**: Parquet file handling, schema validation, sampling
- **Usage**: Supporting module for data loading

### **feature_manifest.py**
**Feature manifest management**

- **Purpose**: Track and validate feature sets across training runs
- **Features**: Feature list persistence, validation, compatibility checking
- **Usage**: Ensures consistent feature ordering between training and inference

### **scalability_utils.py**
**Memory and performance optimizations**

- **Purpose**: Handle large-scale datasets efficiently
- **Features**:
  - Memory-efficient data processing
  - Chunked operations
  - Progress monitoring
- **Usage**: Supporting module for large-scale training

---

## Legacy and Alternative Implementations

### **run_gene_cv_sigmoid_v0.py**
**Legacy gene CV implementation**

- **Status**: Deprecated, kept for reference
- **Purpose**: Original gene CV implementation before major refactoring
- **Note**: Use `run_gene_cv_sigmoid.py` for new work

### **eval_meta_splice_v0.py**
**Legacy evaluation implementation**

- **Status**: Deprecated, kept for reference
- **Purpose**: Original evaluation before comprehensive refactoring
- **Note**: Use `eval_meta_splice.py` for new work

---

## Configuration and Examples

### **configs/**
Configuration files for various training scenarios

### **examples/**
Example scripts demonstrating usage patterns

---

## Common Issues and Troubleshooting

### **Feature Harmonization Problems**

**Issue**: Missing 3-mer columns causing Polars schema mismatch errors  
**Symptoms**: 
```
polars.exceptions.ColumnNotFoundError: 3mer_GTN
```

**Root Cause**: Mismatch between features used during training (recorded in `feature_manifest.csv`) and features available in current dataset.

**Common Missing Features**:
- `3mer_GTN`, `3mer_TNN`, `3mer_NNT`, `3mer_NTT` (containing ambiguous nucleotides)

**Solution**: Fixed in January 2025 update to `gene_score_delta.py` and `gene_score_delta_multiclass.py`:
1. Filter column selection to only existing columns
2. Add missing features as zeros during preprocessing
3. Maintain feature order for model compatibility

**Affected Modules**: `gene_score_delta.py`, `gene_score_delta_multiclass.py`  
**Status**: ✅ Fixed  
**Related Docs**: `troubleshooting_polars_missing_columns.md`, `gene_aware_cv/gene_cv_feature_harmonization.md`

### **Memory Issues**

**Issue**: Out-of-memory errors during large-scale training  
**Solutions**: 
- Use `run_loco_cv_multiclass_scalable.py` for large datasets
- Enable `--memory-optimize` flag
- Reduce `--diag-sample` size
**Related Docs**: `oom_issues.md`

### **Polars LazyFrame Issues**

**Issue**: LazyFrame sampling failures  
**Related Docs**: `polars_lazyframe_sampling.md`, `troubleshooting_polars_missing_columns.md`

### **XGBoost DMatrix Compatibility**

**Issue**: Feature name mismatches in XGBoost models  
**Related Docs**: `xgboost_dmatrix_guide.md`, `libsvm_feature_mismatch.md`

---

## Getting Help

1. **Check existing documentation** in `docs/` directory
2. **Review troubleshooting guides** for common issues
3. **Examine example scripts** in `examples/` directory
4. **Use verbose logging** (`--verbose` flag) for debugging
5. **Check module docstrings** for detailed API documentation

---

## Recent Updates

### January 2025
- **✅ Feature Harmonization Fix**: Resolved 3-mer column mismatch issues
- **✅ Complete Feature Parity**: LOCO-CV and Gene-CV now have identical capabilities
- **✅ Critical Evaluation Fix**: Resolved systematic evaluation methodology issues
- **✅ Enhanced Documentation**: Comprehensive module overview and troubleshooting guides

### Legacy
- Sigmoid ensemble architecture implementation
- Gene-aware and chromosome-aware cross-validation
- Comprehensive evaluation and visualization pipeline
- SHAP analysis integration
- Memory optimization features

---

*This document provides a comprehensive overview of the training package modules. For specific usage instructions, refer to individual module documentation and the main README.*