# Meta-Model Training Utility Scripts Reference

This document provides comprehensive usage documentation for all utility scripts in the meta-model training workflow. These utilities replace the previous dynamic `python -c` scripts with robust, reusable tools.

## 📁 **Location**
All utility scripts are located in:
```
meta_spliceai/splice_engine/meta_models/training/utils/
```

## 🚀 **Quick Start**
All utilities can be run using the module syntax:
```bash
# Activate environment first
mamba activate surveyor

# Run any utility
python -m meta_spliceai.splice_engine.meta_models.training.utils.<utility_name> [args]
```

---

## 📊 **1. Dataset Inspector**
**File:** `dataset_inspector.py`  
**Purpose:** Environment checks and dataset quality assessment

### Usage
```bash
# Check environment readiness
python -m meta_spliceai.splice_engine.meta_models.training.utils.dataset_inspector --check-env

# Analyze dataset quality
python -m meta_spliceai.splice_engine.meta_models.training.utils.dataset_inspector \
    --dataset train_pc_5000_3mers_diverse/master \
    --output dataset_quality_report.json \
    --verbose
```

### Key Features
- ✅ Environment validation (dependencies, GPU availability)
- 📊 Batch-by-batch analysis (rows, columns, memory usage)
- 🔍 Schema consistency checking
- 📈 Data type analysis and memory profiling
- 🧬 Splice type distribution analysis
- 💾 Quality scoring and issue detection

### Output
- Console: Detailed quality assessment report
- JSON: Structured quality metrics (if `--output` specified)

---

## 🧬 **2. Chromosome Analyzer**
**File:** `chromosome_analyzer.py`  
**Purpose:** Chromosome-aware analysis for datasets and CV results

### Usage
```bash
# Quick chromosome distribution check
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --dataset train_pc_5000_3mers_diverse/master \
    --analysis quick-distribution

# Comprehensive chromosome distribution analysis
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --dataset train_pc_5000_3mers_diverse/master \
    --analysis distribution \
    --output chromosome_distribution.json

# Analyze chromosome CV performance
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --results-dir results/chromosome_cv_pc_5000_3mers_diverse \
    --analysis performance \
    --output chromosome_performance.json \
    --verbose

# Quick CV results check
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --results-dir results/chromosome_cv \
    --analysis quick-cv
```

### Analysis Types
- **`distribution`**: Full chromosome distribution analysis across dataset
- **`performance`**: Chromosome-specific CV performance analysis
- **`quick-distribution`**: Fast chromosome check (first batch only)
- **`quick-cv`**: Quick CV results summary

### Key Features
- 📊 Chromosome distribution statistics
- 🎯 Performance ranking by chromosome
- ⚠️ Problematic chromosome identification
- 📈 Consistency analysis across chromosomes
- 🔍 Statistical summaries and recommendations

---

## 📈 **3. Performance Analyzer**
**File:** `performance_analyzer.py`  
**Purpose:** Training performance analysis and visualization

### Usage
```bash
# Analyze training performance
python -m meta_spliceai.splice_engine.meta_models.training.utils.performance_analyzer \
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 \
    --output performance_analysis.json \
    --verbose

# Generate legacy-style summary
python -m meta_spliceai.splice_engine.meta_models.training.utils.performance_analyzer \
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 \
    --legacy-summary
```

### Key Features
- 📊 CV metrics analysis (F1, accuracy, precision, recall)
- 📈 Performance visualization and plotting
- 🔍 Fold-by-fold performance breakdown
- ⚖️ Training vs validation performance comparison
- 📋 Comprehensive performance summaries
- 🎯 Performance trend analysis

---

## 🎯 **4. Calibration Checker**
**File:** `calibration_checker.py`  
**Purpose:** Model calibration analysis

### Usage
```bash
# Analyze model calibration
python -m meta_spliceai.splice_engine.meta_models.training.utils.calibration_checker \
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 \
    --output calibration_analysis.json \
    --verbose
```

### Key Features
- 📊 Calibration curve analysis
- 🎯 Reliability diagram generation
- 📈 Brier score calculation
- ⚖️ Per-class calibration assessment
- 🔍 Calibration quality metrics

---

## 🔍 **5. Leakage Validator**
**File:** `leakage_validator.py`  
**Purpose:** Data leakage detection and validation

### Usage
```bash
# Validate existing results for leakage
python -m meta_spliceai.splice_engine.meta_models.training.utils.leakage_validator \
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 \
    --output leakage_report.json \
    --verbose

# Run comprehensive leakage analysis
python -m meta_spliceai.splice_engine.meta_models.training.utils.leakage_validator \
    --run-analysis \
    --dataset train_pc_5000_3mers_diverse/master \
    --output-dir leakage_analysis \
    --threshold 0.95 \
    --verbose
```

### Key Features
- 🔍 Feature leakage detection
- 📊 Cross-validation leakage analysis
- ⚠️ Suspicious feature identification
- 📈 Leakage severity scoring
- 🛡️ Validation recommendations

---

## 🎭 **6. Ensemble Analyzer**
**File:** `ensemble_analyzer.py`  
**Purpose:** Model ensemble analysis

### Usage
```bash
# Analyze ensemble model
python -m meta_spliceai.splice_engine.meta_models.training.utils.ensemble_analyzer \
    --model results/gene_cv_pc_5000_3mers_diverse_run1/models/fold_1/xgb_model.json \
    --output ensemble_analysis.json \
    --verbose
```

### Key Features
- 🌳 Tree ensemble analysis
- 📊 Feature importance aggregation
- 🔍 Model complexity assessment
- 📈 Ensemble diversity metrics
- ⚖️ Individual tree contribution analysis

---

## 🔄 **7. Cross-Dataset Validator**
**File:** `cross_dataset_validator.py`  
**Purpose:** Cross-dataset validation and generalization testing

### Usage
```bash
# Validate model across datasets
python -m meta_spliceai.splice_engine.meta_models.training.utils.cross_dataset_validator \
    --model results/gene_cv_pc_5000_3mers_diverse_run1/models/fold_1/xgb_model.json \
    --datasets train_pc_7000_3mers_opt/master train_pc_1000_3mers/master \
    --sample-size 10000 \
    --output cross_dataset_validation.json \
    --verbose
```

### Key Features
- 🔄 Cross-dataset performance evaluation
- 📊 Generalization assessment
- 🎯 Domain adaptation analysis
- 📈 Performance degradation metrics
- 🔍 Dataset-specific insights

---

## 🧪 **8. Ablation Analyzer**
**File:** `ablation_analyzer.py`  
**Purpose:** Ablation study analysis and visualization

### Usage
```bash
# Analyze ablation study results
python -m meta_spliceai.splice_engine.meta_models.training.utils.ablation_analyzer \
    --ablation-dir results/ablation_study_pc_5000_3mers_diverse \
    --output ablation_analysis.json \
    --plot-dir ablation_plots \
    --verbose

# Generate plots only
python -m meta_spliceai.splice_engine.meta_models.training.utils.ablation_analyzer \
    --ablation-dir results/ablation_study_pc_5000_3mers_diverse \
    --plots-only \
    --plot-dir ablation_plots
```

### Key Features
- 📊 Feature group importance ranking
- 📈 Performance impact visualization
- 🔍 Feature interaction analysis
- 📋 Ablation summary reports
- 🎯 Optimal feature set identification

---

## 🎯 **9. Feature Importance Runner**
**File:** `feature_importance_runner.py`  
**Purpose:** Feature importance analysis using SHAP

### Usage
```bash
# Run feature importance analysis
python -m meta_spliceai.splice_engine.meta_models.training.utils.feature_importance_runner \
    --dataset train_pc_5000_3mers_diverse/master \
    --run-dir results/gene_cv_pc_5000_3mers_diverse_run1 \
    --sample 5000 \
    --verbose
```

### Key Features
- 🎯 SHAP-based feature importance
- 📊 Feature ranking and scoring
- 📈 Importance visualization
- 🔍 Feature interaction detection
- 📋 Comprehensive importance reports

---

## 🧬 **10. Paralog Analyzer**
**File:** `paralog_analyzer.py`  
**Purpose:** Paralog gene identification and analysis

### Usage
```bash
# Analyze paralog genes
python -m meta_spliceai.splice_engine.meta_models.training.utils.paralog_analyzer \
    --gene-features data/ensembl/spliceai_analysis/gene_features.tsv \
    --output paralog_analysis.json \
    --cluster-distance 1000000 \
    --verbose
```

### Key Features
- 🧬 Gene family identification
- 📊 Chromosomal clustering analysis
- 🔍 Pseudogene detection
- 📈 Paralog relationship mapping
- ⚠️ Data leakage risk assessment

### ⚠️ **Performance Note**
The paralog analyzer is computationally intensive, especially during chromosomal clustering analysis. For large gene sets (>60K genes), expect runtime of 10-30 minutes.

---

## 🛠️ **Common Usage Patterns**

### **Pre-Training Workflow**
```bash
# 1. Environment check
python -m meta_spliceai.splice_engine.meta_models.training.utils.dataset_inspector --check-env

# 2. Dataset quality assessment
python -m meta_spliceai.splice_engine.meta_models.training.utils.dataset_inspector \
    --dataset train_pc_5000_3mers_diverse/master --verbose

# 3. Chromosome distribution analysis
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --dataset train_pc_5000_3mers_diverse/master --analysis distribution
```

### **Post-Training Analysis**
```bash
# 1. Performance analysis
python -m meta_spliceai.splice_engine.meta_models.training.utils.performance_analyzer \
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 --verbose

# 2. Calibration check
python -m meta_spliceai.splice_engine.meta_models.training.utils.calibration_checker \
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 --verbose

# 3. Leakage validation
python -m meta_spliceai.splice_engine.meta_models.training.utils.leakage_validator \
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 --verbose
```

### **Chromosome-Aware Analysis**
```bash
# 1. Quick chromosome check
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --dataset train_pc_5000_3mers_diverse/master --analysis quick-distribution

# 2. Chromosome CV performance
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --results-dir results/chromosome_cv_pc_5000_3mers_diverse --analysis performance --verbose
```

---

## 📋 **Output Formats**

### **Console Output**
All utilities provide rich console output with:
- 🎨 Color-coded status indicators
- 📊 Formatted tables and statistics
- ⚠️ Clear warning and error messages
- 🎯 Progress indicators for long operations

### **JSON Output**
When `--output` is specified, utilities generate structured JSON with:
- 📊 Numerical metrics and statistics
- 📈 Analysis results and summaries
- ⚠️ Issues and recommendations
- 🕒 Timestamps and metadata

### **File Outputs**
Some utilities generate additional files:
- 📈 **Performance plots** (PNG format)
- 📊 **CSV reports** (detailed metrics)
- 🎯 **SHAP analysis** (feature importance)
- 📋 **Summary reports** (markdown format)

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **Import Errors**
```bash
# Ensure surveyor environment is activated
mamba activate surveyor

# Verify installation
python -c "import meta_spliceai; print('✅ Package imported successfully')"
```

#### **Dataset Path Issues**
```bash
# Use absolute paths or ensure you're in project root
cd /path/to/meta-spliceai
python -m meta_spliceai.splice_engine.meta_models.training.utils.dataset_inspector \
    --dataset train_pc_5000_3mers_diverse/master
```

#### **Memory Issues**
```bash
# For large datasets, use sampling
python -m meta_spliceai.splice_engine.meta_models.training.utils.feature_importance_runner \
    --dataset train_pc_5000_3mers_diverse/master \
    --run-dir results/gene_cv_pc_5000_3mers_diverse_run1 \
    --sample 1000  # Reduce sample size
```

### **Performance Tips**
- 🚀 **Use sampling** for exploratory analysis
- 💾 **Monitor memory usage** with large datasets
- 🔄 **Run in background** for long operations
- 📊 **Save outputs** to avoid re-computation

---

## 📚 **Integration with Workflows**

These utilities are designed to integrate seamlessly with the existing meta-model training workflows:

- **COMPLETE_META_MODEL_WORKFLOW.md**: Uses utilities throughout the 6-phase workflow
- **CHROMOSOME_AWARE_WORKFLOW.md**: Leverages chromosome_analyzer extensively
- **gene_aware_cv/gene_cv_sigmoid.md**: Integrates performance_analyzer and calibration_checker

### **Workflow Integration Examples**
See the main workflow documentation for complete integration examples and best practices.

---

## 🎯 **Best Practices**

1. **Always activate the surveyor environment** before running utilities
2. **Use verbose mode** (`--verbose`) for detailed insights
3. **Save outputs** (`--output`) for reproducibility
4. **Start with quick analyses** before running comprehensive assessments
5. **Monitor resource usage** for computationally intensive utilities
6. **Validate datasets** before training with dataset_inspector
7. **Check for leakage** after training with leakage_validator
8. **Analyze performance** comprehensively with multiple utilities

---

*This documentation covers all 10 utility scripts that replace the previous dynamic `python -c` commands, providing robust, maintainable, and well-documented tools for meta-model training and evaluation.*



