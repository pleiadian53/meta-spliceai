# MetaSpliceAI Installation Guide

## 🎯 **Quick Start (Recommended)**

The fastest way to get MetaSpliceAI running:

```bash
# 1. Install Mamba (fast conda alternative)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniforge3

# Initialize mamba for your shell (makes it available in future sessions)
$HOME/miniforge3/bin/mamba init bash
# For zsh users: $HOME/miniforge3/bin/mamba init zsh

# Reload shell configuration or open a new terminal
source ~/.bashrc  # or source ~/.zshrc for zsh users

# Alternatively: For one-time use without modifying shell config:
# eval "$($HOME/miniforge3/bin/mamba shell hook --shell bash)"

# 2. Create environment from exact working configuration
git clone <repository-url>
cd meta-spliceai
mamba env create -f environment.yml
mamba activate surveyor

# 3. Install the package in editable mode without enforcing pinned deps
#    (avoids pip resolver downgrades and preserves the tested env)
pip uninstall -y meta-spliceai || true
pip install -e . --no-deps

# 4. Test installation
./docs/installation/test_installation.sh

# 5. Optional: Install additional dependencies if needed
# For distributed processing (some legacy components):
# pip install pyspark>=3.5.0  # Faster than mamba for PySpark
# For BigWig file handling (genomic data visualization):
# pip install pyBigWig>=0.3.18
# For Weights & Biases experiment tracking:
# mamba install -v wandb>=0.16.0
# Note: PySpark and pyBigWig install faster via pip
```

## 🚀 **GPU Support (CUDA 12.2)**

**MetaSpliceAI includes full GPU acceleration support!**

```bash
# Base env (CPU-default torch in environment.yml)
mamba env create -f environment.yml
mamba activate surveyor

# Test GPU setup
./scripts/gpu_env_setup/test_gpu_installation.sh
python scripts/gpu_env_setup/test_gpu_performance.py
```

**GPU Features:**
- ✅ **XGBoost GPU acceleration** (5-10x faster training)
- ✅ **TensorFlow GPU support** (10-50x faster inference)
- ✅ **PyTorch CUDA support** (10-100x faster operations)
- ✅ **Multi-GPU support** (parallel processing)

### PyTorch GPU Installation (optional)

By default, `environment.yml` pins `torch==2.7.1` (CPU). Install a CUDA build only if you have a compatible NVIDIA driver:

```bash
# Option A: Conda (recommended for CUDA builds)
mamba install -c pytorch -c nvidia -c conda-forge pytorch=2.7 pytorch-cuda=12.6

# Option B: Pip (from PyTorch CUDA index)
pip uninstall -y torch
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1
```

> Tip: Verify with `python -c "import torch; print(torch.cuda.is_available())"`.

**For detailed GPU setup and testing, see:** `docs/gpu_environment_setup.md`

## 📁 **File Guide: Which File for What?**

| File | Purpose | When to Use |
|------|---------|-------------|
| **`environment.yml`** | ✅ **Environment reproduction** | **Primary** - Use for setting up development/production environments |
| **`pyproject.toml`** | ✅ **Package building & metadata** | Use for building wheels and distributing the package |
| **`requirements.txt`** | ❌ **DEPRECATED** | Don't use - kept for backward compatibility only |

### **environment.yml - Single Source of Truth**
- Contains **exact working versions** tested and verified
- Use for **reproducible environments** across machines
- Handles **system dependencies** (bedtools, bcftools, tabix, CUDA)
- **Fast installation** via mamba
- **GPU-optimized** package versions
- **Includes HuggingFace ecosystem** (transformers, tokenizers) for error modeling
- **VCF analysis tools** (bcftools, tabix) for genomic variant processing

Note: MLflow is included and kept current (3.2.0). You can also install/update explicitly:

```bash
# Using conda-forge
mamba install -c conda-forge mlflow=3.2.0

# Or via pip
pip install -U mlflow==3.2.0
```

### **pyproject.toml - For Distribution**
- Contains **flexible version ranges** for compatibility
- Use for **building wheels** for Microsoft Fabric
- Includes **development tools** and **optional dependencies**
- **Modern Python packaging** standards

## 🏗️ **Installation Methods**

### **Method 1: Mamba (Recommended)**

✅ **Best for development and production environments**

```bash
# Install Miniforge (includes mamba)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniforge3

# Initialize shell (optional - makes mamba available in new terminals)
$HOME/miniforge3/bin/mamba init

# Create environment
git clone <repository>
cd meta-spliceai
mamba env create -f environment.yml
mamba activate surveyor

# Optional: Verify VCF analysis tools (if uncommented in environment.yml)
# bcftools --version
# tabix --version
```

**Why mamba?**
- ⚡ **10x faster** than conda
- 🛡️ **No solver conflicts** (avoids conda-libmamba-solver issues)
- 🎯 **Better dependency resolution**
- 🔬 **Excellent for scientific packages**
- 🚀 **Faster GPU package installation**

### **Method 2: Poetry + Mamba Hybrid**

✅ **Best for package development and distribution**

```bash
# Create base environment with mamba
mamba create -n surveyor python=3.10 bedtools poetry -c conda-forge -c bioconda
mamba activate surveyor

# Configure Poetry to use current environment
poetry config virtualenvs.create false

# Install dependencies via Poetry
poetry install
```

### **Method 3: From Wheel (Microsoft Fabric)**

✅ **Best for deployment in Microsoft Fabric**

```bash
# Build wheel (do this on development machine)
mamba activate surveyor
poetry build

# Install in Fabric (upload wheel file first)
%pip install /path/to/meta_spliceai-0.2.0-py3-none-any.whl

# Or with Fabric extras
%pip install meta_spliceai[fabric]
```

## 🔄 **Migration from Conda to Mamba**

### **For Existing Conda Users**

If you're currently using conda and want to switch to mamba (recommended):

**Option 1: Automated Migration (Recommended)**
```bash
# Use the migration script for easy transition
./scripts/installation/migrate_conda_to_mamba.sh

# Or with custom environment name
./scripts/installation/migrate_conda_to_mamba.sh -e my-env-name

# Or keep the old environment (backup only)
./scripts/installation/migrate_conda_to_mamba.sh -k
```

**Option 2: Manual Migration**
```bash
# 1. Export your current environment (if you want to preserve it)
conda env export --no-builds > environment-conda-backup.yml

# 2. Remove the old conda environment
conda env remove -n surveyor

# 3. Install mamba (if not already installed)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniforge3
source $HOME/miniforge3/bin/activate

# 4. Create new mamba environment
mamba env create -f environment.yml
mamba activate surveyor

# 5. Verify the new environment
./docs/installation/test_installation.sh
```

### **Benefits of Migration**
- ⚡ **Faster package installation** (especially GPU packages)
- 🛡️ **Fewer dependency conflicts**
- 🎯 **Better CUDA package compatibility**
- 🔧 **Improved environment management**

## 🏆 **Microsoft Fabric Integration**

### **Building for Fabric**

```bash
# 1. Ensure you're in the right environment
mamba activate surveyor

# 2. Build wheel with all dependencies
poetry build

# 3. Verify build
ls dist/
# meta_spliceai-0.2.0-py3-none-any.whl
# meta_spliceai-0.2.0.tar.gz
```

### **Installing in Fabric**

```python
# Method 1: Upload wheel file to Fabric workspace
%pip install /FileStore/shared_uploads/meta_spliceai-0.2.0-py3-none-any.whl

# Method 2: From private package index
%pip install meta-spliceai --extra-index-url https://your-private-index

# Method 3: With Fabric-specific dependencies
%pip install meta_spliceai[fabric]
```

### **Fabric Environment Setup**

```python
# In Fabric notebook
import meta_spliceai
print(f"MetaSpliceAI version: {meta_spliceai.__version__}")

# Test core functionality
from meta_spliceai.splice_engine.meta_models.training import run_gene_cv_sigmoid
print("✅ MetaSpliceAI loaded successfully in Fabric!")
```

## **Advanced Installation**

### **GPU Support (CUDA 12.2)**

```bash
# The environment.yml includes CUDA-compatible versions
mamba env create -f environment.yml
mamba activate surveyor

# Verify GPU support
./scripts/gpu_env_setup/test_gpu_installation.sh
python scripts/gpu_env_setup/test_gpu_performance.py

# Or manual verification
python -c "import tensorflow as tf; print('GPU available:', tf.config.list_physical_devices('GPU'))"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### **Automated GPU Setup**

For the easiest GPU environment setup:

```bash
# Automated GPU environment creation
./scripts/gpu_env_setup/install_gpu_environment.sh

# For full CUDA toolkit (development work)
./scripts/gpu_env_setup/install_gpu_environment.sh -m complete

# For custom environment name
./scripts/gpu_env_setup/install_gpu_environment.sh -e my-gpu-env
```

### **Development Setup**

```bash
# Full development environment with optional dependencies
mamba env create -f environment.yml
mamba activate surveyor

# Install development tools
poetry install --group dev

# Install pre-commit hooks
pre-commit install

# Install all optional dependencies
poetry install --extras all
```

### **Minimal Installation**

```bash
# Create minimal environment
mamba create -n surveyor-minimal python=3.10
mamba activate surveyor-minimal

# Install only core dependencies
poetry install --no-dev --no-optional
```

## 🧬 **Error Model Setup**

**For deep learning error analysis with transformers and DNABERT:**

The error model (`meta_spliceai.splice_engine.meta_models.error_model`) requires additional HuggingFace dependencies that are now included in `environment.yml`:

- **Transformers**: 4.45.0+ (DNABERT, HyenaDNA support)
- **Tokenizers**: 0.20.0+ (Fast DNA tokenization)
- **Accelerate**: 1.0.0+ (Multi-GPU acceleration)

### **Quick Error Model Test**
```bash
# Test error model installation
python -c "
from meta_spliceai.splice_engine.meta_models.error_model import ErrorModelConfig, TransformerTrainer
from transformers import AutoTokenizer
print('✅ Error model ready!')
"

# Run minimal workflow test (requires data)
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir results/test \
    --device cpu --batch-size 4 --num-epochs 1 --skip-ig
```

📖 **For detailed error model setup**: See [`docs/installation/ERROR_MODEL_SETUP.md`](ERROR_MODEL_SETUP.md)

## 🧪 **Testing Installation**

### **Quick Test**
```bash
./docs/installation/test_installation.sh
```

### **GPU Testing**
```bash
# Basic GPU installation test
./scripts/gpu_env_setup/test_gpu_installation.sh

# Performance testing
python scripts/gpu_env_setup/test_gpu_performance.py

# Detailed GPU setup verification
python scripts/gpu_env_setup/verify_gpu_setup.py
```

### **Comprehensive Test**
```bash
# Test command-line tools
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder --help
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid --help

# Test imports
python -c "import meta_spliceai; print('✅ Package loaded successfully')"

# Test GPU (if available)
python -c "import tensorflow as tf; print('✅ TensorFlow GPU:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

## ❓ FAQ: Wheels, Fabric, and Environments

### Where is the wheel saved during `pip install -e . --no-deps`?

- Editable installs create a temporary wheel in a pip cache (e.g., `/tmp/pip-ephem-wheel-cache-.../wheels/...`).
- This ephemeral wheel is not intended for redistribution.

### How do I generate a reusable wheel (e.g., for Microsoft Fabric)?

```bash
mamba activate surveyor
poetry build
ls dist/
# dist/meta_spliceai-<version>-py3-none-any.whl
# dist/meta_spliceai-<version>.tar.gz
```

- The reusable wheel is produced in the `dist/` directory.
- Upload the `.whl` to Fabric and install with `%pip install /path/to/meta_spliceai-<version>-py3-none-any.whl`.

### Do I need both `environment.yml` and `pyproject.toml`? What’s their relationship?

- **Yes, both are used and complementary:**
  - **`environment.yml`**: Defines the full compute environment (Python version, Conda/Mamba channels, system tools like `bedtools`, CUDA choices, and most Python libs). Use it to create/update your local/VM environment.
  - **`pyproject.toml`**: Defines the package metadata and how to build/install the `meta_spliceai` package itself (version, name, entry points, optional extras). Use it to build wheels and for `pip install -e . --no-deps` inside the environment created by `environment.yml`.

#### Recommended workflow (local/VM)

```bash
mamba env create -f environment.yml   # or: mamba env update -f environment.yml
mamba activate surveyor
pip install -e . --no-deps            # install code from source into this env
```

#### For Fabric

- Fabric runs a managed Python environment (no Conda/Mamba). Build a wheel locally and upload it:

```bash
# On your dev machine/VM
mamba activate surveyor
poetry build

# In Fabric notebook
%pip install /FileStore/shared_uploads/meta_spliceai-<version>-py3-none-any.whl
```

- If Fabric requires additional runtime libs, create a companion `requirements.txt` (export from your env) and install it before the wheel in Fabric.

## 🔄 **Environment Reproduction**

### **Share Your Environment**
```bash
# Export exact environment (recommended)
mamba env export --no-builds > environment-shared.yml

# Or export with Poetry
poetry export -f requirements.txt --output requirements-shared.txt
```

### **Reproduce Environment**
```bash
# From environment.yml (recommended)
mamba env create -f environment-shared.yml

# From requirements.txt (fallback)
pip install -r requirements-shared.txt
```

## 🔧 **Troubleshooting**

### **Common Issues**

#### **Issue: conda-libmamba-solver error**
```bash
# Solution: Use mamba directly instead of conda
conda config --set solver classic
# Or better: use mamba instead of conda entirely
```

#### **Issue: TensorFlow/NumPy compatibility**
```bash
# Solution: Use our tested versions from environment.yml
mamba env create -f environment.yml  # Uses numpy 2.1.3 + TensorFlow 2.19.0
```

#### **Issue: scikit-learn not found**
```bash
# Check if you're in the right environment
conda info --envs
mamba activate surveyor

# Reinstall if needed
mamba install scikit-learn -c conda-forge
```

#### **Issue: Wheel building fails**
```bash
# Ensure Poetry is configured correctly
poetry config virtualenvs.create false
poetry config virtualenvs.in-project false

# Rebuild
poetry build --verbose
```

#### **Issue: GPU not detected**
```bash
# Run comprehensive GPU diagnostics
python scripts/gpu_env_setup/verify_gpu_setup.py

# Check NVIDIA driver
nvidia-smi

# Verify CUDA installation
nvcc --version
```

### **Getting Help**

1. **Check the test script output**: `./docs/installation/test_installation.sh`
2. **Verify environment**: `mamba list` and `conda info --envs`
3. **Check package versions**: Match versions in `environment.yml`
4. **Review logs**: Look for specific error messages during installation
5. **GPU issues**: Use `./scripts/gpu_env_setup/test_gpu_installation.sh`

## 📊 **Performance Comparison**

| Method | Speed | Reliability | Best For |
|--------|-------|-------------|----------|
| **mamba + environment.yml** | 🚀 Fast | 🛡️ High | Development, Production, GPU |
| **poetry + pyproject.toml** | ⚡ Medium | 🛡️ High | Package building, Distribution |
| **pip + requirements.txt** | 🐌 Slow | ⚠️ Medium | Legacy compatibility |

## 🎯 **Summary**

**For most users:**
```bash
mamba env create -f environment.yml
mamba activate surveyor
```

**For GPU machines:**
```bash
mamba env create -f environment.yml
mamba activate surveyor
./scripts/gpu_env_setup/test_gpu_installation.sh
```

**For Microsoft Fabric:**
```bash
poetry build
# Upload wheel to Fabric
```

**For development:**
```bash
mamba env create -f environment.yml
mamba activate surveyor
poetry install --group dev
```

This approach gives you **fast, reliable installations** with **excellent Microsoft Fabric compatibility** and **full GPU acceleration**! 🚀 