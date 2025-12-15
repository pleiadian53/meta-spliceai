# Installation Documentation

This directory contains comprehensive installation guides and tools for MetaSpliceAI.

## 📋 **Contents**

- **[INSTALLATION.md](INSTALLATION.md)** - Complete installation guide with multiple methods
- **[test_installation.sh](test_installation.sh)** - Comprehensive installation verification script

## 🚀 **Quick Start**

### **1. Choose Installation Method**

| Method | Best For | Complexity |
|--------|----------|------------|
| **Poetry + Mamba** | Modern development | ⭐⭐⭐ |
| **Conda Environment** | Research, stability | ⭐⭐ |
| **Pip Requirements** | Production deployment | ⭐ |

### **2. Run Installation**

```bash
# Method 1: Poetry + Mamba (Recommended)
mamba create -n surveyor python=3.10 -y
mamba activate surveyor
mamba install -c conda-forge -c bioconda bedtools=2.30.0 poetry -y
poetry config virtualenvs.create false
poetry install

# Method 2: Conda Environment
conda env create -f environment.yml
conda activate surveyor

# Method 3: Pip
python -m venv meta-spliceai-env
source meta-spliceai-env/bin/activate
pip install -r requirements.txt
```

### **3. Verify Installation**

```bash
# Run the comprehensive test script
./docs/installation/test_installation.sh
```

## 🎯 **Installation Methods**

### **Poetry + Mamba (Recommended)**
- **Fastest dependency resolution** (mamba is 10x faster than conda)
- **Modern package management** (Poetry handles Python packages)
- **Reproducible builds** (lock files)
- **Better conflict detection** (mamba)

### **Conda Environment**
- **Battle-tested and stable**
- **Handles complex dependencies** (TensorFlow, PyTorch, bioinformatics tools)
- **Includes system tools** (bedtools)

### **Pip Requirements**
- **Fast installation**
- **Minimal footprint**
- **Standard Python workflow**

## 🔧 **Troubleshooting**

### **Common Issues**

1. **GPU Support Problems**
   ```bash
   # Run GPU diagnostics
   python scripts/diagnose_gpu_environment.py
   ```

2. **Package Conflicts**
   ```bash
   # Use Poetry for dependency resolution
   poetry lock
   poetry install
   ```

3. **Memory Issues**
   ```bash
   # Check system resources
   free -h
   df -h
   ```

### **Getting Help**

- Check the [main INSTALLATION.md](INSTALLATION.md) for detailed troubleshooting
- Run the test script to identify specific issues
- Review system requirements in the installation guide

## 📊 **System Requirements**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10 | 3.10.11 |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 10GB | 50GB+ |
| **GPU** | Optional | NVIDIA with CUDA |

## 🎉 **Next Steps**

After successful installation:

1. **Read the main documentation** in `docs/`
2. **Try a sample analysis** to verify functionality
3. **Check GPU setup** if using GPU acceleration
4. **Review package management** in `docs/PACKAGE_MANAGEMENT.md`

---

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md). 