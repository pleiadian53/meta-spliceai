# Chromosome-Aware Cross-Validation Workflow

**Date:** August 2025  
**Status:** ✅ **CHROMOSOME-AWARE CV GUIDE**

## 🛠️ **Utility Scripts**
This workflow extensively uses the `chromosome_analyzer` utility. For complete documentation:
- 📚 **[Complete Reference](UTILITY_SCRIPTS_REFERENCE.md#2-chromosome-analyzer)**
- 🚀 **[Quick Reference](UTILITY_SCRIPTS_QUICK_REFERENCE.md)**  

## 🎯 **Overview**

Chromosome-aware cross-validation (LOCO - Leave-One-Chromosome-Out) is a specialized validation approach that tests model generalization across different chromosomes. This is particularly important for genomic applications where different chromosomes may have distinct characteristics.

## 🔬 **When to Use Chromosome-Aware CV**

### **Use Cases**
- **Genomic generalization testing**: Verify model performance across different chromosomes
- **Chromosome-specific analysis**: Identify chromosomes where the model performs poorly
- **Biological validation**: Ensure the model doesn't overfit to specific chromosomal regions
- **Publication requirements**: Many genomic papers require chromosome-aware validation

### **Advantages**
- **Biological relevance**: Tests real-world generalization scenarios
- **Robust evaluation**: More stringent than gene-aware CV
- **Chromosome insights**: Reveals chromosome-specific performance patterns
- **Publication standard**: Widely accepted in genomic literature

### **Disadvantages**
- **Computational cost**: Higher than gene-aware CV
- **Data requirements**: Needs sufficient data per chromosome
- **Complexity**: More complex setup and interpretation

---

## 🚀 **Chromosome-Aware Training Workflow**

### **Step 1: Pre-Training Validation**

```bash
# 1. Schema validation (script auto-detects master/ subdirectory)
python meta_spliceai/splice_engine/meta_models.builder/validate_dataset_schema.py \
    --dataset train_pc_5000_3mers_diverse

# 2. Chromosome distribution analysis
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --dataset train_pc_5000_3mers_diverse/master \
    --analysis distribution
```

### **Step 2: Chromosome-Aware Training**

```bash
# Production chromosome-aware CV training
# Note: The script automatically detects master/ subdirectory if present
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_pc_5000_3mers_diverse \
    --out-dir results/chromosome_cv_pc_5000_3mers_diverse \
    --n-estimators 400 \
    --row-cap 0 \
    --min-rows-test 10000 \
    --calibrate-per-class \
    --auto-exclude-leaky \
    --monitor-overfitting \
    --calibration-analysis \
    --verbose
```

**Key Parameters**:
- `--min-rows-test 10000`: Minimum test set size per chromosome
- `--row-cap 0`: Use full dataset
- `--calibrate-per-class`: Enable per-class calibration
- `--auto-exclude-leaky`: Remove leaky features

### **Step 3: Chromosome-Specific Analysis**

```bash
# Analyze chromosome-specific performance
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --results-dir results/chromosome_cv_pc_5000_3mers_diverse \
    --analysis performance
```

---

## 📊 **Chromosome-Aware Ablation Studies**

### **Comprehensive Chromosome Ablation**

```bash
# Chromosome-aware ablation study
# Note: The script automatically detects master/ subdirectory if present
python -m meta_spliceai.splice_engine.meta_models.training.run_ablation_multiclass \
    --dataset train_pc_5000_3mers_diverse \
    --out-dir results/chromosome_ablation_study \
    --cv-strategy chromosome \
    --modes full,no_spliceai,no_probs,no_kmer,only_kmer,raw_scores \
    --n-estimators 200 \
    --min-rows-test 5000 \
    --verbose
```

### **Chromosome-Specific Feature Analysis**

```bash
# Analyze feature importance by chromosome
python -m meta_spliceai.splice_engine.meta_models.training.utils.ablation_analyzer \
    --ablation-dir results/chromosome_ablation_study
```

---

## 🔍 **Chromosome-Aware Diagnostics**

### **Chromosome Performance Visualization**

```bash
# Generate chromosome performance plots
python -m meta_spliceai.splice_engine.meta_models.training.utils.performance_analyzer \
    --results-dir results/chromosome_cv_pc_5000_3mers_diverse
```

### **Chromosome-Specific Feature Importance**

```bash
# Analyze feature importance by chromosome
python -m meta_spliceai.splice_engine.meta_models.training.utils.feature_importance_runner \
    --dataset train_pc_5000_3mers_diverse/master \
    --run-dir results/chromosome_cv_pc_5000_3mers_diverse
```

---

## 📈 **Chromosome-Aware Evaluation Metrics**

### **Key Metrics for Chromosome-Aware CV**

```bash
# Comprehensive chromosome-aware evaluation
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --results-dir results/chromosome_cv_pc_5000_3mers_diverse \
    --analysis performance \
    --output chromosome_evaluation.json \
    --verbose
```

---

## 🚨 **Chromosome-Aware Troubleshooting**

### **Common Issues and Solutions**

#### **Insufficient Data per Chromosome**
```bash
# Error: Some chromosomes have too few test samples
# Solution: Adjust minimum test size or combine small chromosomes

python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_pc_5000_3mers_diverse/master \
    --out-dir results/chromosome_cv_adjusted \
    --min-rows-test 5000  # Reduce from 10000 to 5000
```

#### **Chromosome Performance Imbalance**
```bash
# Analyze chromosome-specific issues
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --results-dir results/chromosome_cv_pc_5000_3mers_diverse \
    --analysis performance \
    --verbose
```

#### **Memory Issues with Large Chromosomes**
```bash
# Memory-optimized chromosome-aware training
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_pc_5000_3mers_diverse/master \
    --out-dir results/chromosome_cv_memory_optimized \
    --n-estimators 200 \
    --row-cap 100000 \
    --min-rows-test 5000 \
    --memory-optimize
```

---

## 📋 **Chromosome-Aware Best Practices**

### **1. Data Preparation**
- **Validate chromosome distribution**: Ensure sufficient data per chromosome
- **Check chromosome quality**: Verify no systematic issues in specific chromosomes
- **Balance considerations**: Some chromosomes may be naturally harder to predict

### **2. Training Strategy**
- **Start with gene-aware CV**: Use chromosome-aware CV for final validation
- **Adjust test sizes**: Balance between statistical power and computational cost
- **Monitor convergence**: Chromosome-specific training may need different parameters

### **3. Evaluation Approach**
- **Focus on consistency**: Look for chromosomes with significantly different performance
- **Consider biological factors**: Some chromosomes may have distinct characteristics
- **Report comprehensively**: Include both overall and chromosome-specific metrics

### **4. Interpretation Guidelines**
- **Biological relevance**: Poor performance on specific chromosomes may indicate biological differences
- **Data quality**: Check for systematic issues in problematic chromosomes
- **Model limitations**: Some chromosomes may be inherently harder to predict

---

## 🎯 **Quick Reference Commands**

### **Standard Chromosome-Aware Workflow**
```bash
# 1. Validate schema (script auto-detects master/ subdirectory)
python meta_spliceai/splice_engine/meta_models/builder/validate_dataset_schema.py --dataset your_dataset

# 2. Analyze chromosome distribution
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --dataset your_dataset/master \
    --analysis quick-distribution

# 3. Train with chromosome-aware CV (script auto-detects master/ subdirectory)
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset your_dataset --out-dir results/chromosome_cv --min-rows-test 10000 --verbose

# 4. Analyze results
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --results-dir results/chromosome_cv \
    --analysis quick-cv
```

### **Chromosome-Aware Ablation Study**
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_ablation_multiclass \
    --dataset your_dataset/master --out-dir results/chromosome_ablation \
    --cv-strategy chromosome --modes full,no_kmer,only_kmer --min-rows-test 5000
```

---

**Remember**: Chromosome-aware CV provides the most stringent validation for genomic applications and is essential for publication-quality results! 🧬
