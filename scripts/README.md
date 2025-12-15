# MetaSpliceAI Scripts - Management System

This directory contains 50+ organized scripts for various MetaSpliceAI operations, with a comprehensive management system to help you find and use the right script for any task.

## 🚀 **Quick Start - Most Common Tasks**

### **After Gene CV Training** (Most Frequent)
```bash
# Analyze F1-based PR curves and FP/FN trade-offs
python scripts/f1_pr_analysis_merged.py results/gene_cv_pc_1000_3mers_run_4
```

### **Before Training** (Pre-flight Checks)
```bash
python scripts/pre_flight_checks.py
python scripts/validate_meta_model_training_data.py
```

### **GPU Issues** (Troubleshooting)
```bash
python scripts/diagnose_gpu_environment.py
```

## 📚 **Documentation System**

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** | Fast lookup for common tasks | Daily use, when you need a script quickly |
| **[SCRIPT_INVENTORY.md](SCRIPT_INVENTORY.md)** | Complete catalog of all scripts | Finding specific scripts, understanding capabilities |
| **README.md** (this file) | Overview and navigation | First-time users, getting oriented |
| **Category READMEs** | Detailed docs for script groups | Working within specific domains |

## 📁 **Organized Directory Structure**

```
scripts/
├── 📚 Documentation & Management
│   ├── README.md                    # This overview (start here)
│   ├── QUICK_REFERENCE.md           # Fast lookup guide
│   ├── SCRIPT_INVENTORY.md          # Complete script catalog
│   └── README_f1_pr_analysis.md     # Detailed F1 analysis docs
│
├── 🔧 Setup & Environment
│   ├── installation/                # Installation and setup scripts
│   ├── gpu_env_setup/              # GPU environment setup and testing
│   ├── check_gpu.py                # Quick GPU check
│   ├── diagnose_gpu_environment.py # Full GPU diagnostics
│   └── pre_flight_checks.py        # Comprehensive pre-training checks
│
├── 📊 Data & Analysis
│   ├── f1_pr_analysis_merged.py    # 🌟 CV results analysis (most used)
│   ├── analyze_training_data.py     # Training data analysis
│   ├── validate_meta_model_training_data.py # Data validation
│   ├── analysis/                    # Analysis scripts and visualizations
│   └── [20+ data scripts...]
│
├── 🎯 Model Training & Evaluation
│   ├── evaluation/                  # Model evaluation scripts
│   ├── run_multi_gpu_training.py    # Multi-GPU training
│   ├── scaling_solutions/           # Large-scale training solutions
│   └── [10+ training scripts...]
│
└── 🛠️ Utilities & Maintenance
    ├── cleanup_artifacts.py         # Clean old files
    ├── validate_artifacts.py        # Validate outputs
    └── [15+ utility scripts...]
```

## 🚀 **Quick Start**

### **For New Users:**
```bash
# Basic installation
mamba env create -f environment.yml
mamba activate surveyor

# Test installation
./docs/installation/test_installation.sh
```

### **For Existing Conda Users:**
```bash
# Migrate from conda to mamba
./scripts/installation/migrate_conda_to_mamba.sh
```

### **For GPU Machines:**
```bash
# Test GPU setup
./scripts/gpu_env_setup/test_gpu_installation.sh

# Performance testing
python scripts/gpu_env_setup/test_gpu_performance.py
```

## 📋 **Script Categories**

### **🔧 Installation Scripts (`scripts/installation/`)**
- **Environment setup** and configuration
- **Migration tools** (conda to mamba)
- **Installation verification**

### **🚀 GPU Environment Scripts (`scripts/gpu_env_setup/`)**
- **GPU setup** and configuration
- **Performance testing** and benchmarking
- **Comprehensive verification** and diagnostics

## 🎯 **Common Workflows**

### **New Installation (CPU/GPU):**
1. `mamba env create -f environment.yml`
2. `mamba activate surveyor`
3. `./docs/installation/test_installation.sh`
4. (GPU only) `./scripts/gpu_env_setup/test_gpu_installation.sh`

### **Conda to Mamba Migration:**
1. `./scripts/installation/migrate_conda_to_mamba.sh`
2. Verify with installation tests
3. (GPU only) Test GPU setup

### **GPU Performance Testing:**
1. `./scripts/gpu_env_setup/test_gpu_installation.sh`
2. `python scripts/gpu_env_setup/test_gpu_performance.py`
3. `python scripts/gpu_env_setup/verify_gpu_setup.py`

## 📚 **Documentation**

### **Installation Guides:**
- **Main Installation:** `docs/installation/INSTALLATION.md`
- **GPU Setup:** `docs/gpu_environment_setup.md`

### **Script Documentation:**
- **Installation Scripts:** `scripts/installation/README.md`
- **GPU Setup Scripts:** `scripts/gpu_env_setup/README.md`

## 🤝 **Contributing**

To add new scripts:
1. **Choose appropriate directory** based on functionality
2. **Update relevant README.md** files
3. **Add documentation** in main guides
4. **Test on both GPU and non-GPU** environments
5. **Update this main README** if adding new categories

## 🔧 **Script Standards**

All scripts should:
- ✅ **Be executable** (`chmod +x`)
- ✅ **Include help** (`-h` or `--help`)
- ✅ **Handle errors** gracefully
- ✅ **Provide clear output** with status messages
- ✅ **Work on both GPU and non-GPU** machines
- ✅ **Include documentation** in README files

---

**Note:** These scripts are designed to work across different environments and provide appropriate feedback for each scenario.
