# MetaSpliceAI Scripts - Quick Reference Guide

## 🚀 **Most Common Tasks**

### **After Gene CV Training** (Most Frequent)
```bash
# Analyze F1-based PR curves and FP/FN trade-offs
python scripts/f1_pr_analysis_merged.py results/gene_cv_pc_1000_3mers_run_4

# Expected output:
# - pr_curves_f1_optimized.pdf
# - f1_pr_analysis_summary.txt
```

### **Before Training** (Pre-flight Checks)
```bash
# Comprehensive system validation
python scripts/pre_flight_checks.py

# Data validation
python scripts/validate_meta_model_training_data.py
```

### **GPU Issues** (Troubleshooting)
```bash
# Quick GPU check
python scripts/check_gpu.py

# Comprehensive GPU diagnostics
python scripts/diagnose_gpu_environment.py

# Performance testing
python scripts/test_gpu_performance.py
```

### **Data Exploration** (Understanding Your Data)
```bash
# Comprehensive training data analysis
python scripts/analyze_training_data.py

# Inspect specific sequences
python scripts/inspect_analysis_sequences.py

# Check dataset integrity
python scripts/test_dataset_integrity.py
```

## 📁 **Quick Directory Navigation**

```
scripts/
├── 🔧 Setup & Troubleshooting
│   ├── check_gpu.py                    # Quick GPU check
│   ├── diagnose_gpu_environment.py     # Full GPU diagnostics
│   ├── pre_flight_checks.py            # Pre-training validation
│   └── fix_ml_dependencies.py          # Fix library issues
│
├── 📊 Data & Analysis
│   ├── analyze_training_data.py         # Training data analysis
│   ├── validate_meta_model_training_data.py  # Data validation
│   ├── inspect_analysis_sequences.py    # Sequence inspection
│   └── f1_pr_analysis_merged.py        # 🌟 CV results analysis
│
├── 🎯 Model Training & Evaluation
│   ├── run_multi_gpu_training.py        # Multi-GPU training
│   ├── test_transcript_topk.py          # Transcript evaluation
│   └── evaluation/                      # Evaluation scripts
│
└── 🛠️ Utilities
    ├── cleanup_artifacts.py             # Clean old files
    ├── validate_artifacts.py            # Validate outputs
    └── scaling_solutions/               # Large-scale solutions
```

## 🎯 **Problem-Solution Quick Lookup**

| Problem | Script | Command |
|---------|--------|---------|
| **CV results need analysis** | `f1_pr_analysis_merged.py` | `python scripts/f1_pr_analysis_merged.py results/gene_cv_*` |
| **GPU not detected** | `diagnose_gpu_environment.py` | `python scripts/diagnose_gpu_environment.py` |
| **Training data looks wrong** | `analyze_training_data.py` | `python scripts/analyze_training_data.py` |
| **Pre-training validation** | `pre_flight_checks.py` | `python scripts/pre_flight_checks.py` |
| **Library conflicts** | `fix_ml_dependencies.py` | `python scripts/fix_ml_dependencies.py` |
| **Dataset integrity issues** | `test_dataset_integrity.py` | `python scripts/test_dataset_integrity.py` |
| **Performance issues** | `test_gpu_performance.py` | `python scripts/test_gpu_performance.py` |
| **Disk space issues** | `cleanup_artifacts.py` | `python scripts/cleanup_artifacts.py` |

## 📋 **Workflow Checklists**

### **New Project Setup Checklist**
- [ ] `./scripts/installation/migrate_conda_to_mamba.sh` (if needed)
- [ ] `python scripts/check_gpu.py`
- [ ] `python scripts/pre_flight_checks.py`
- [ ] `python scripts/check_versions.py`

### **Before Training Checklist**
- [ ] `python scripts/validate_meta_model_training_data.py`
- [ ] `python scripts/test_dataset_integrity.py`
- [ ] `python scripts/pre_flight_checks.py`
- [ ] `python scripts/test_leakage_probe.py`

### **After Training Checklist**
- [ ] `python scripts/f1_pr_analysis_merged.py results/gene_cv_*`
- [ ] `python scripts/test_transcript_topk.py`
- [ ] `python scripts/validate_artifacts.py`
- [ ] Review generated plots and summaries

### **Troubleshooting Checklist**
- [ ] `python scripts/diagnose_gpu_environment.py`
- [ ] `python scripts/analyze_training_data.py`
- [ ] `python scripts/test_datasets_loading.py`
- [ ] `python scripts/fix_ml_dependencies.py`

## 🔍 **Script Categories at a Glance**

### **🌟 High-Priority Scripts** (Use Weekly)
- `f1_pr_analysis_merged.py` - CV results analysis
- `pre_flight_checks.py` - System validation
- `analyze_training_data.py` - Data exploration
- `test_gpu_performance.py` - Performance monitoring

### **🔧 Setup Scripts** (Use Once/Rarely)
- `diagnose_gpu_environment.py` - GPU troubleshooting
- `fix_ml_dependencies.py` - Library fixes
- `check_versions.py` - Compatibility checks

### **📊 Analysis Scripts** (Use Per Project)
- `validate_meta_model_training_data.py` - Data validation
- `test_transcript_topk.py` - Model evaluation
- `inspect_analysis_sequences.py` - Data inspection

### **🛠️ Utility Scripts** (Use As Needed)
- `cleanup_artifacts.py` - Maintenance
- `validate_artifacts.py` - Output validation
- `test_leakage_probe.py` - Quality control

## 💡 **Pro Tips**

### **Efficient Workflow**
1. **Always start with**: `python scripts/pre_flight_checks.py`
2. **After CV training**: `python scripts/f1_pr_analysis_merged.py results/gene_cv_*`
3. **When in doubt**: `python scripts/analyze_training_data.py`
4. **GPU issues**: `python scripts/diagnose_gpu_environment.py`

### **Time-Saving Commands**
```bash
# Quick system check
python scripts/check_gpu.py && python scripts/check_versions.py

# Full data validation pipeline
python scripts/validate_meta_model_training_data.py && python scripts/test_dataset_integrity.py

# Post-training analysis suite
python scripts/f1_pr_analysis_merged.py results/gene_cv_* && python scripts/test_transcript_topk.py
```

### **Common Patterns**
- **Gene CV results**: Always in `results/gene_cv_pc_1000_3mers_run_*` format
- **Output files**: Look for `.pdf` plots and `.txt` summaries
- **Error logs**: Check console output for detailed error messages
- **Performance**: GPU scripts provide timing and memory usage info

## 📚 **Documentation Hierarchy**

1. **QUICK_REFERENCE.md** (this file) - Fast lookup and common tasks
2. **SCRIPT_INVENTORY.md** - Complete catalog with details
3. **README.md** - Directory overview and structure
4. **Category READMEs** - Detailed documentation for script groups
5. **Individual script docs** - Specific usage guides (e.g., `README_f1_pr_analysis.md`)

---

**💡 Remember**: When you can't remember which script to use, start with this quick reference guide!
