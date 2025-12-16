# Utility Scripts Quick Reference Card

## 🚀 **Environment Setup**
```bash
mamba activate surveyor
cd /path/to/meta-spliceai
```

## 📊 **Quick Commands**

### **Dataset Analysis**
```bash
# Environment check
python -m meta_spliceai.splice_engine.meta_models.training.utils.dataset_inspector --check-env

# Dataset quality
python -m meta_spliceai.splice_engine.meta_models.training.utils.dataset_inspector \
    --dataset train_pc_5000_3mers_diverse/master --verbose

# Chromosome distribution
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --dataset train_pc_5000_3mers_diverse/master --analysis quick-distribution
```

### **Training Results Analysis**
```bash
# Performance analysis
python -m meta_spliceai.splice_engine.meta_models.training.utils.performance_analyzer \
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 --verbose

# Calibration check
python -m meta_spliceai.splice_engine.meta_models.training.utils.calibration_checker \
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 --verbose

# Leakage validation
python -m meta_spliceai.splice_engine.meta_models.training.utils.leakage_validator \
    --results-dir results/gene_cv_pc_5000_3mers_diverse_run1 --verbose
```

### **Chromosome-Aware CV Analysis**
```bash
# CV performance by chromosome
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --results-dir results/chromosome_cv_pc_5000_3mers_diverse --analysis performance --verbose

# Quick CV check
python -m meta_spliceai.splice_engine.meta_models.training.utils.chromosome_analyzer \
    --results-dir results/chromosome_cv --analysis quick-cv
```

### **Advanced Analysis**
```bash
# Feature importance
python -m meta_spliceai.splice_engine.meta_models.training.utils.feature_importance_runner \
    --dataset train_pc_5000_3mers_diverse/master \
    --run-dir results/gene_cv_pc_5000_3mers_diverse_run1 --sample 5000

# Ablation analysis
python -m meta_spliceai.splice_engine.meta_models.training.utils.ablation_analyzer \
    --ablation-dir results/ablation_study_pc_5000_3mers_diverse --verbose

# Ensemble analysis
python -m meta_spliceai.splice_engine.meta_models.training.utils.ensemble_analyzer \
    --model results/gene_cv_pc_5000_3mers_diverse_run1/models/fold_1/xgb_model.json --verbose
```

## 🎯 **All 10 Utilities**

| Utility | Purpose | Key Use Case |
|---------|---------|--------------|
| `dataset_inspector` | Environment & dataset quality | Pre-training validation |
| `chromosome_analyzer` | Chromosome-aware analysis | LOCO-CV workflows |
| `performance_analyzer` | Training performance | Post-training evaluation |
| `calibration_checker` | Model calibration | Probability assessment |
| `leakage_validator` | Data leakage detection | Quality assurance |
| `ensemble_analyzer` | Model ensemble analysis | Model interpretation |
| `cross_dataset_validator` | Cross-dataset validation | Generalization testing |
| `ablation_analyzer` | Ablation study analysis | Feature importance |
| `feature_importance_runner` | SHAP analysis | Feature ranking |
| `paralog_analyzer` | Paralog gene detection | Data leakage prevention |

## 📋 **Common Flags**
- `--verbose`: Detailed output
- `--output file.json`: Save structured results
- `--help`: Show usage information

For complete documentation, see: `UTILITY_SCRIPTS_REFERENCE.md`



