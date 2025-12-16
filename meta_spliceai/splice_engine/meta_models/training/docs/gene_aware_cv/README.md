# Gene-Aware Cross-Validation Documentation

This directory contains documentation related to gene-aware cross-validation approaches in MetaSpliceAI's meta-model training system.

## 📋 Document Organization

### 🚀 Current & Active Documentation

#### **Primary Reference**
- **[COMPREHENSIVE_TRAINING_GUIDE.md](../COMPREHENSIVE_TRAINING_GUIDE.md)** - **MAIN REFERENCE** for all training workflows including gene-aware CV

### 🧬 Gene-Aware CV Specific Documentation

#### **Deep Learning Approaches**
- **[DEEP_LEARNING_CV_README.md](DEEP_LEARNING_CV_README.md)** - Complete guide for deep learning-based gene-aware CV
- **[DEEP_LEARNING_CV_SOLUTION_SUMMARY.md](DEEP_LEARNING_CV_SOLUTION_SUMMARY.md)** - Solution summary for deep learning CV implementation
- **[DEEP_LEARNING_INCOMPATIBILITY_SUMMARY.md](DEEP_LEARNING_INCOMPATIBILITY_SUMMARY.md)** - Analysis of incompatibilities between deep learning and traditional approaches

#### **Traditional ML Approaches**
- **[gene_cv_sigmoid.md](gene_cv_sigmoid.md)** - ⚠️ **DEPRECATED** - Detailed documentation for sigmoid ensemble approach (superseded by Comprehensive Training Guide)
- **[gene_aware_evaluation.md](gene_aware_evaluation.md)** - ⚠️ **OUTDATED** - Original gene-aware evaluation strategy (concepts integrated into Comprehensive Training Guide)

#### **Specialized Features**
- **[gene_cv_feature_harmonization.md](gene_cv_feature_harmonization.md)** - Feature harmonization guide for gene-aware CV
- **[transcript_level_topk.md](transcript_level_topk.md)** - Transcript-level top-k accuracy evaluation

## 🎯 Quick Navigation

### **For New Users**
1. Start with **[COMPREHENSIVE_TRAINING_GUIDE.md](../COMPREHENSIVE_TRAINING_GUIDE.md)** for complete training workflows
2. Use **[DEEP_LEARNING_CV_README.md](DEEP_LEARNING_CV_README.md)** if working with TabNet/TensorFlow models
3. Reference specific feature guides as needed

### **For Deep Learning Research**
- **[DEEP_LEARNING_CV_README.md](DEEP_LEARNING_CV_README.md)** - Complete deep learning CV implementation
- **[DEEP_LEARNING_CV_SOLUTION_SUMMARY.md](DEEP_LEARNING_CV_SOLUTION_SUMMARY.md)** - Technical solution details
- **[DEEP_LEARNING_INCOMPATIBILITY_SUMMARY.md](DEEP_LEARNING_INCOMPATIBILITY_SUMMARY.md)** - Architecture analysis

### **For Traditional ML Workflows**
- **[COMPREHENSIVE_TRAINING_GUIDE.md](../COMPREHENSIVE_TRAINING_GUIDE.md)** - Complete traditional ML training guide
- **[gene_cv_sigmoid.md](gene_cv_sigmoid.md)** - Detailed sigmoid ensemble documentation (deprecated but comprehensive)

## 📊 Document Status Legend

| Status | Meaning | Action Required |
|--------|---------|----------------|
| ✅ **CURRENT** | Up-to-date and actively maintained | Use as primary reference |
| ⚠️ **DEPRECATED** | Superseded by newer documentation | Use for historical reference only |
| ❌ **OUTDATED** | Contains outdated information | Use with caution, verify against current docs |

## 🔄 Migration Notes

### **From Legacy Documentation**
- **gene_cv_sigmoid.md** → **COMPREHENSIVE_TRAINING_GUIDE.md** (comprehensive training workflows)
- **gene_aware_evaluation.md** → **COMPREHENSIVE_TRAINING_GUIDE.md** (evaluation strategies)
- **Deep learning approaches** → **DEEP_LEARNING_CV_README.md** (dedicated deep learning guide)

### **Key Changes**
- **Unified Interface**: All training approaches now use consistent command-line interface
- **Multi-Algorithm Support**: Support for XGBoost, CatBoost, LightGBM, TabNet, TensorFlow
- **Memory Optimization**: Enhanced memory management for large datasets
- **Comprehensive Analysis**: Integrated analysis pipeline with all evaluation methods

## 🚀 Getting Started

### **Quick Start (Traditional ML)**
```bash
# Activate environment
mamba activate surveyor

# Basic gene-aware CV training
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset your_dataset/master \
    --out-dir results/gene_cv \
    --n-estimators 400 \
    --calibrate-per-class \
    --auto-exclude-leaky \
    --verbose
```

### **Quick Start (Deep Learning)**
```bash
# Deep learning gene-aware CV
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_deep_learning \
    --dataset your_dataset/master \
    --out-dir results/deep_learning_cv \
    --algorithm tabnet \
    --n-folds 5 \
    --verbose
```

## 📚 Related Documentation

- **[COMPREHENSIVE_TRAINING_GUIDE.md](../COMPREHENSIVE_TRAINING_GUIDE.md)** - Main training reference
- **[MULTI_INSTANCE_ENSEMBLE_TRAINING.md](../MULTI_INSTANCE_ENSEMBLE_TRAINING.md)** - Large-scale training
- **[MEMORY_SCALABILITY_LESSONS.md](../MEMORY_SCALABILITY_LESSONS.md)** - Memory optimization

---

*Last updated: January 2025*
