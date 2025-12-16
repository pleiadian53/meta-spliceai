# Complete Meta-Model Training and Evaluation Workflow

**⚠️ DEPRECATED:** This documentation is outdated. Please use **[Comprehensive Training Guide](COMPREHENSIVE_TRAINING_GUIDE.md)** instead.

**Date:** August 2025 (OUTDATED)  
**Status:** ❌ **SUPERSEDED**  
**Replaced By:** [Comprehensive Training Guide](COMPREHENSIVE_TRAINING_GUIDE.md)  

## 📋 **Overview**

This document provides a complete workflow for training and evaluating meta-models for splice site prediction. The workflow includes:

1. **Pre-training Steps**: Dataset validation, schema checking, and preparation
2. **Training Approaches**: Gene-aware and chromosome-aware cross-validation
3. **Post-training Analysis**: Comprehensive evaluation, diagnostics, and ablation studies
4. **Quality Assurance**: Performance monitoring and validation

## 🛠️ **Utility Scripts Documentation**

This workflow uses 10 specialized utility scripts that replace previous dynamic `python -c` commands:

- 📚 **[Complete Reference](UTILITY_SCRIPTS_REFERENCE.md)**: Comprehensive documentation for all utilities
- 🚀 **[Quick Reference](UTILITY_SCRIPTS_QUICK_REFERENCE.md)**: Essential commands and usage patterns

All utilities are located in `meta_spliceai/splice_engine/meta_models/training/utils/`

## 🎯 **Workflow Summary**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PRE-TRAINING  │    │     TRAINING    │    │ POST-TRAINING   │    │   VALIDATION    │
│                 │    │                 │    │                 │    │                 │
│ • Schema Check  │───▶│ • Gene-aware CV │───▶│ • Evaluation    │───▶│ • Performance   │
│ • Data Quality  │    │ • Chromosome CV │    │ • Diagnostics   │    │ • Ablation      │
│ • Environment   │    │ • Model Training│    │ • Feature Analysis│   │ • Comparison    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🔧 **Phase 1: Pre-Training Preparation**

### **1.1 Environment Setup**

```bash
# Activate the surveyor environment
mamba activate surveyor

# Verify environment
python -m meta_spliceai.splice_engine.meta_models.training.utils.dataset_inspector --check-env
```

### **1.2 Dataset Schema Validation**

**Critical Step**: Always validate dataset schema before training to prevent schema mismatch errors.

```bash
# Validate dataset schema
python meta_spliceai/splice_engine/meta_models/builder/validate_dataset_schema.py \
    --dataset train_pc_5000_3mers_diverse/master

# Auto-fix any schema issues if found
python meta_spliceai/splice_engine/meta_models/builder/validate_dataset_schema.py \
    --dataset train_pc_5000_3mers_diverse/master --fix
```

**Expected Output**:
```
🔍 Validating schema for 20 batch files in train_pc_5000_3mers_diverse/master
📊 batch_00001.parquet: 143 cols, 64 k-mers
📊 batch_00002.parquet: 143 cols, 64 k-mers
...
✅ All batches have consistent schema!
```

### **1.3 Dataset Quality Assessment**

```bash
# Comprehensive dataset quality assessment
python -m meta_spliceai.splice_engine.meta_models.training.utils.dataset_inspector \
    --dataset train_pc_5000_3mers_diverse/master \
    --output dataset_quality_report.json
```

### **1.4 Memory and Resource Planning**

**Memory Requirements**:
- **Small dataset** (< 100K rows): 4-8 GB RAM
- **Medium dataset** (100K-500K rows): 8-16 GB RAM  
- **Large dataset** (> 500K rows): 16-32 GB RAM

**Storage Requirements**:
- **Training outputs**: 2-5 GB per run
- **Diagnostics**: 1-3 GB additional
- **Ablation studies**: 5-15 GB per study

---

## 🚀 **Phase 2: Training Approaches**

### **2.1 Gene-Aware Cross-Validation (Recommended)**

**Use Case**: Standard meta-model training with gene-level generalization

```bash
# Production-ready gene-aware CV training
# Note: The script automatically detects master/ subdirectory if present
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_5000_3mers_diverse \
    --out-dir results/gene_cv_pc_5000_3mers_diverse_run1 \
    --n-estimators 800 \
    --row-cap 0 \
    --calibrate-per-class \
    --auto-exclude-leaky \
    --monitor-overfitting \
    --calibration-analysis \
    --neigh-sample 5000 \
    --early-stopping-patience 30 \
    --verbose
```

**Smart Dataset Detection**: The script automatically:
- ✅ Detects if `master/` subdirectory exists and contains parquet files
- ✅ Uses `master/` subdirectory automatically if found
- ✅ Falls back to the specified path if no `master/` subdirectory exists
- ✅ Provides clear feedback about which path is being used

**Enhanced Features**:
- ✅ **Smart Path Resolution**: Automatically detects and uses `master/` subdirectory
- ✅ **Comprehensive Training Summary**: Automatically generates detailed training reports
- ✅ **Argument Validation**: Validates all command-line arguments before training
- ✅ **System Information**: Captures platform, memory, and environment details
- ✅ **Performance Metrics**: Collects and summarizes all CV performance metrics

**Key Parameters**:
- `--row-cap 0`: Use full dataset (critical for production)
- `--calibrate-per-class`: Enable per-class probability calibration
- `--auto-exclude-leaky`: Automatically remove leaky features
- `--monitor-overfitting`: Track overfitting during training
- `--calibration-analysis`: Comprehensive calibration analysis

### **2.2 Chromosome-Aware Cross-Validation**

**Use Case**: Testing generalization across different chromosomes

```bash
# Chromosome-aware CV training
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_pc_5000_3mers_diverse/master \
    --out-dir results/chromosome_cv_pc_5000_3mers_diverse \
    --n-estimators 400 \
    --row-cap 0 \
    --min-rows-test 10000 \
    --verbose
```

**Key Parameters**:
- `--min-rows-test 10000`: Minimum test set size per chromosome
- Uses Leave-One-Chromosome-Out (LOCO) validation

### **2.3 Memory-Optimized Training**

**For systems with limited memory**:

```bash
# Memory-optimized training
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_5000_3mers_diverse/master \
    --out-dir results/gene_cv_memory_optimized \
    --n-estimators 400 \
    --row-cap 100000 \
    --memory-optimize \
    --max-diag-sample 10000 \
    --neigh-sample 1000 \
    --verbose
```

**Memory Optimization Features**:
- `--memory-optimize`: Enable memory optimization
- `--max-diag-sample 10000`: Limit diagnostic sample size
- `--neigh-sample 1000`: Reduce neighbor analysis sample

---

## 📊 **Phase 3: Post-Training Analysis**

### **3.1 Comprehensive Evaluation**

The training script automatically runs comprehensive evaluation, but you can also run additional analyses:

```bash
# Run comprehensive diagnostics (if not already done)
python -m meta_spliceai.splice_engine.meta_models.training.run_comprehensive_diagnostics \
    --dataset train_pc_5000_3mers_diverse/master \
    --run-dir results/gene_cv_pc_5000_3mers_diverse_run1 \
    --sample 25000 \
    --verbose
```

### **3.2 Ablation Studies**

**Critical for understanding feature importance and model behavior**:

```bash
# Comprehensive ablation study (with sensible defaults)
python -m meta_spliceai.splice_engine.meta_models.training.run_ablation_multiclass \
    --dataset train_pc_5000_3mers_diverse/master \
    --out-dir results/ablation_study_pc_5000_3mers_diverse \
    --n-estimators 800 \
    --verbose

# Alternative: Specify custom modes if needed
python -m meta_spliceai.splice_engine.meta_models.training.run_ablation_multiclass \
    --dataset train_pc_5000_3mers_diverse/master \
    --out-dir results/ablation_study_pc_5000_3mers_diverse \
    --modes full,raw_scores,no_probs,no_kmer,only_kmer,no_spliceai \
    --n-estimators 800 \
    --verbose
```

**Default Ablation Modes** (automatically included):
- `full`: All features (baseline)
- `raw_scores`: Only SpliceAI raw scores
- `no_probs`: Exclude probability-derived features
- `no_kmer`: Exclude k-mer features
- `only_kmer`: Only k-mer features

**Additional Available Modes** (specify with `--modes` if needed):
- `no_spliceai`: Exclude SpliceAI raw scores
- `positional_only`: Only positional features
- `context_only`: Only context window features

**Ablation Study Output**:
- `ablation_results.csv`: Performance comparison across all modes
- `ablation_summary.json`: Detailed statistical analysis
- `ablation_plots/`: Visualization of feature importance
- `feature_contribution_analysis.txt`: Human-readable analysis

**Key Metrics Analyzed**:
- F1 Score (macro and weighted)
- Accuracy and Top-k Accuracy
- Precision and Recall
- AUC and Average Precision
- Statistical significance testing

**Ablation Study Analysis**:

```bash
# Analyze ablation study results using the dedicated utility
python -m meta_spliceai.splice_engine.meta_models.training.utils.ablation_analyzer \
    --ablation-dir results/ablation_study_pc_5000_3mers_diverse \
    --output results/ablation_analysis.json \
    --verbose
```

**Expected Ablation Study Insights**:
- **SpliceAI Scores**: Typically contribute 15-25% to performance
- **K-mer Features**: Usually provide 10-20% improvement
- **Probability Features**: Often add 5-15% performance gain
- **Positional Features**: Generally contribute 5-10%
- **Context Features**: May add 3-8% depending on window size

### **3.3 Feature Importance Analysis**

```bash
# Run comprehensive feature importance analysis
python -m meta_spliceai.splice_engine.meta_models.training.utils.feature_importance_runner \
    --dataset train_pc_5000_3mers_diverse/master \
    --run-dir results/gene_cv_pc_5000_3mers_diverse_run1 \
    --sample 25000
```
### **3.4 Performance Comparison Analysis**

```bash
# Compare base vs meta model performance
python -m meta_spliceai.splice_engine.meta_models.training.eval_meta_splice \
    --dataset train_pc_5000_3mers_diverse/master \
    --run-dir results/gene_cv_pc_5000_3mers_diverse_run1 \
    --sample 25000 \
    --out-tsv results/comprehensive_performance_comparison.tsv \
    --verbose
```

---

## 🔍 **Phase 4: Quality Assurance and Validation**

### **4.1 Performance Validation**

**Check key performance metrics**:

```bash
# Comprehensive performance analysis
python -m meta_spliceai.splice_engine.meta_models.training.utils.performance_analyzer \
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 \
    --output performance_analysis.json
```

### **4.2 Model Quality Checks**

**Verify model calibration and overconfidence**:

```bash
# Comprehensive calibration analysis
python -m meta_spliceai.splice_engine.meta_models.training.utils.calibration_checker \
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 \
    --output calibration_analysis.json
```

### **4.3 Data Leakage Validation**

```bash
# Comprehensive data leakage validation
python -m meta_spliceai.splice_engine.meta_models.training.utils.leakage_validator \
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 \
    --output leakage_validation.json
```

**Data Leakage Validation Checklist**:
- ✅ **Feature Correlation Analysis**: Check for >0.95 correlations with target
- ✅ **Excluded Features Log**: Review features removed due to leakage
- ✅ **Cross-Validation Integrity**: Ensure no data leakage across folds
- ✅ **Temporal Leakage**: Verify no future information in training data
- ✅ **Target Leakage**: Confirm no direct target information in features
```

---

## 📈 **Phase 5: Advanced Analysis and Optimization**

### **5.1 Hyperparameter Optimization**

```bash
# Run hyperparameter optimization (example workflow)
echo "📊 Hyperparameter optimization workflow:"
echo "1. Load your training data"
echo "2. Define parameter grid (n_estimators, max_depth, learning_rate, subsample)"
echo "3. Use GridSearchCV or RandomizedSearchCV with 5-fold CV"
echo "4. Optimize for f1_macro scoring"
echo ""
echo "💡 For automated hyperparameter optimization, consider using:"
echo "   - Optuna for advanced optimization"
echo "   - Hyperopt for Bayesian optimization"
echo "   - scikit-learn's GridSearchCV for exhaustive search"
```

### **5.2 Model Ensemble Analysis**

```bash
# Comprehensive ensemble analysis
python -m meta_spliceai.splice_engine.meta_models.training.utils.ensemble_analyzer \
    --model results/gene_cv_pc_5000_3mers_diverse_run1/model_multiclass.pkl \
    --output ensemble_analysis.json
```

### **5.3 Cross-Dataset Validation**

```bash
# Validate model on different datasets using the cross-dataset validation utility
python -m meta_spliceai.splice_engine.meta_models.training.utils.cross_dataset_validator \
    --model results/gene_cv_pc_5000_3mers_diverse_run1/model_multiclass.pkl \
    --datasets train_pc_7000_3mers_opt/master train_pc_1000_3mers/master \
    --output results/cross_dataset_validation.json \
    --sample-size 25000 \
    --verbose
```

**Cross-Dataset Validation Checklist**:
- ✅ **Different Gene Sets**: Test on datasets with different gene compositions
- ✅ **Different K-mer Sizes**: Validate across various k-mer configurations
- ✅ **Different Splice Densities**: Test on high/low splice density genes
- ✅ **Generalization Assessment**: Ensure model generalizes beyond training data

---

## 📋 **Phase 6: Documentation and Reporting**

### **6.1 Training Summary (Automatically Generated)**

**The training script now automatically generates comprehensive training summaries!**

After training completes, you'll find these files in your output directory:

```bash
# Check the automatically generated training summaries
ls -la results/gene_cv_pc_5000_3mers_diverse_run1/training_summary.*

# View the human-readable summary
cat results/gene_cv_pc_5000_3mers_diverse_run1/training_summary.txt

# View the JSON summary (for programmatic access)
cat results/gene_cv_pc_5000_3mers_diverse_run1/training_summary.json
```

**What's Included in the Training Summary:**

1. **📊 Dataset Information**:
   - Original vs. actual dataset path used
   - Gene manifest location and statistics
   - Gene type distribution

2. **⚙️ Training Parameters**:
   - All command-line arguments used
   - Model configuration settings
   - System resources utilized

3. **📈 Performance Summary**:
   - Mean F1 scores and standard deviations
   - Accuracy metrics
   - Top-k accuracy results

4. **💻 System Information**:
   - Platform and Python version
   - CPU cores and memory
   - Training environment details

5. **📁 Key File Locations**:
   - Trained model path
   - Feature manifest location
   - Gene manifest location
   - All output file paths

### **6.2 Performance Summary**

```bash
# Generate legacy-style performance summary
python -m meta_spliceai.splice_engine.meta_models.training.utils.performance_analyzer \
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 \
    --legacy-summary
```

---

## 🚨 **Troubleshooting Guide**

### **Common Issues and Solutions**

#### **Schema Mismatch Error**
```bash
# Error: polars.exceptions.SchemaError: extra column in file outside of expected schema
# Solution: Run schema validation
python meta_spliceai/splice_engine/meta_models/builder/validate_dataset_schema.py \
    --dataset your_dataset/master --fix
```

#### **Memory Issues**
```bash
# Error: Out of memory during training
# Solution: Use memory-optimized settings
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset your_dataset/master \
    --out-dir results/memory_optimized \
    --memory-optimize \
    --row-cap 50000 \
    --max-diag-sample 5000 \
    --n-estimators 200
```

#### **Missing Dependencies**
```bash
# Error: ModuleNotFoundError: No module named 'polars'
# Solution: Activate correct environment
mamba activate surveyor
```

#### **Poor Performance**
```bash
# Run comprehensive leakage analysis (for new datasets)
python -m meta_spliceai.splice_engine.meta_models.training.utils.leakage_validator \
    --run-analysis --dataset your_dataset/master \
    --output-dir leakage_check --threshold 0.95
```

---

## 📚 **Additional Resources**

### **Related Documentation**
- [Gene-Aware CV Documentation](gene_aware_cv/gene_cv_sigmoid.md)
- [Chromosome-Aware CV Documentation](chrom_aware_cv/chromosome_aware_evaluation.md)
- [Feature Importance Analysis](shap_analysis_troubleshooting.md)
- [Model Evaluation Guide](model_evaluation_and_diagnostics.md)

### **Example Scripts**
- [Training Examples](../examples/)
- [Diagnostic Scripts](../run_comprehensive_diagnostics.py)
- [Ablation Studies](../run_ablation_multiclass.py)

### **Best Practices**
1. **Always validate schema** before training
2. **Use `--row-cap 0`** for production training
3. **Run ablation studies** to understand feature importance
4. **Monitor overfitting** during training
5. **Check for data leakage** in your features
6. **Validate on multiple datasets** when possible

---

## 🎯 **Quick Reference Commands**

### **Standard Training Workflow**
```bash
# 1. Validate schema (script auto-detects master/ subdirectory)
python meta_spliceai/splice_engine/meta_models/builder/validate_dataset_schema.py --dataset your_dataset

# 2. Train model (script auto-detects master/ subdirectory)
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset your_dataset --out-dir results/your_run --row-cap 0 --verbose

# 3. Run ablation study (script auto-detects master/ subdirectory)
python -m meta_spliceai.splice_engine.meta_models.training.run_ablation_multiclass \
    --dataset your_dataset --out-dir results/ablation --cv-strategy gene --modes full,no_kmer,only_kmer

# 4. Check training summary (automatically generated)
cat results/your_run/training_summary.txt
```

### **Memory-Constrained Training**
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset your_dataset/master --out-dir results/memory_optimized \
    --memory-optimize --row-cap 50000 --n-estimators 200
```

### **Quick Performance Check**
```bash
# Quick F1 score check
python -m meta_spliceai.splice_engine.meta_models.training.utils.performance_analyzer \
    --results-dir results/your_run --legacy-summary | grep "Mean F1"
```

---

**Remember**: This workflow ensures robust, well-validated meta-model training with comprehensive analysis and quality assurance! 🚀
