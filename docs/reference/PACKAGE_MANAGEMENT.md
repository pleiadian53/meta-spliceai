# Package Management Strategy for MetaSpliceAI

## 🎯 **Current Recommended Strategy (January 2025)**

After extensive testing, we've established the **optimal approach** for MetaSpliceAI:

### **✅ WORKING SOLUTION: Mamba + Poetry Hybrid**

```bash
# 1. Install Mamba directly (avoid conda conflicts)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniforge3
source $HOME/miniforge3/bin/activate

# 2. Create environment with system dependencies
mamba create -n surveyor python=3.10 bedtools poetry -c conda-forge -c bioconda -y
mamba activate surveyor

# 3. Install Python packages via mamba (faster, better dependency resolution)
mamba install pandas tensorflow scikit-learn matplotlib seaborn polars pyarrow numba shap xgboost biopython gffutils pybedtools -c conda-forge -y

# 4. Poetry for development workflow (optional)
poetry config virtualenvs.create false  # Use mamba environment
poetry install --no-deps  # Install only development tools
```

### **🗂️ File Roles Clarified**

| File | Status | Purpose | When to Use |
|------|--------|---------|-------------|
| **`environment.yml`** | ✅ **PRIMARY** | Exact environment reproduction | **Use for deployment & sharing** |
| **`pyproject.toml`** | ✅ **DEVELOPMENT** | Package metadata & wheel building | **Use for building & publishing** |
| **`requirements.txt`** | ❌ **DEPRECATED** | Legacy pip requirements | **Avoid - use Poetry export instead** |

## 📋 **File-by-File Strategy**

### **1. environment.yml - The Single Source of Truth**

**✅ Use this for environment reproduction:**

```yaml
name: surveyor
channels:
  - conda-forge
  - bioconda
dependencies:
  - python=3.10.14
  - bedtools=2.31.1
  - poetry>=1.6.0
  - pip
  - pip:
      # Core ML/DL frameworks (exact versions for reproducibility)
      - tensorflow==2.19.0     # Supports CUDA 12.2
      - torch==2.7.1+cu126     # GPU support
      - keras==3.10.0
      # Data processing
      - polars==1.31.0
      - pandas==2.3.1
      - pyarrow==19.0.1
      - numpy==2.1.3           # Compatible with TensorFlow 2.19.0
      # ML tools
      - xgboost==3.0.1
      - scikit-learn==1.7.0
      - shap==0.48.0
      - numba==0.61.2          # Supports numpy 2.1+
      # Bioinformatics
      - biopython==1.85
      - gffutils==0.12
      - pybedtools==0.9.1
      # Visualization
      - matplotlib==3.9.2
      - seaborn==0.13.2
      # Utilities
      - tqdm==4.67.1
      - rich==13.9.4
      - h5py==3.12.1
      - openpyxl>=3.1.0,<4.0
```

**Why environment.yml is primary:**
- ✅ **Exact reproducibility** - pin exact versions that work
- ✅ **Cross-platform compatibility** - works on Linux, macOS, Windows
- ✅ **Fast installation** - mamba resolves dependencies efficiently
- ✅ **System dependencies** - handles bedtools, CUDA, etc.

### **2. pyproject.toml - For Development & Distribution**

**✅ Use this for package building and metadata:**

```toml
[tool.poetry]
name = "meta-spliceai"
version = "0.2.0"
description = "Meta-SpliceAI: Meta-learning framework for splice site prediction"
authors = ["Meta-SpliceAI Contributors"]
readme = "README.md"
packages = [{include = "meta_spliceai"}]

[tool.poetry.dependencies]
python = "^3.10"
# Use flexible version ranges for distribution
tensorflow = "^2.19.0"
torch = "^2.7.0"
pandas = "^2.3.0"
numpy = "^2.1.0"
polars = "^1.31.0"
xgboost = "^3.0.0"
scikit-learn = "^1.7.0"
biopython = "^1.85"
# ... other core dependencies

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
black = "^24.0.0"
isort = "^5.12.0"
mypy = "^1.8.0"
jupyter = "^1.0.0"

[tool.poetry.group.fabric.dependencies]
# Microsoft Fabric specific dependencies
azure-identity = "^1.15.0"
azure-storage-blob = "^12.19.0"
pyodbc = "^4.0.39"

[build-system]
requires = ["poetry-core>=1.6.0"]
build-backend = "poetry.core.masonry.api"

# Enable wheel building for Microsoft Fabric
[tool.poetry.build]
generate-setup-file = true

[tool.poetry.scripts]
meta-spliceai = "meta_spliceai.cli:main"
```

**Why pyproject.toml for distribution:**
- ✅ **Modern Python packaging** - follows PEP 518/621 standards
- ✅ **Flexible version ranges** - allows compatible updates
- ✅ **Wheel building** - creates distributable packages
- ✅ **Development groups** - separates dev/test/fabric dependencies

### **3. requirements.txt - DEPRECATED**

**❌ No longer used - here's why:**

```bash
# OLD WAY (problems)
pip install -r requirements.txt  # Slow, poor dependency resolution

# NEW WAY (better)
mamba env create -f environment.yml  # Fast, excellent dependency resolution
# OR
poetry export -f requirements.txt --output requirements-export.txt  # Generate when needed
```

## 🏗️ **Microsoft Fabric Integration**

### **Building Wheels for Fabric**

```bash
# 1. Activate your development environment
mamba activate surveyor

# 2. Build wheel for distribution
poetry build

# 3. Result: wheel files in dist/
ls dist/
# meta_spliceai-0.2.0-py3-none-any.whl
# meta_spliceai-0.2.0.tar.gz
```

### **Installing in Microsoft Fabric**

```python
# In Fabric notebook cell
%pip install /path/to/meta_spliceai-0.2.0-py3-none-any.whl

# Or from private package index
%pip install meta-spliceai --extra-index-url https://your-private-index
```

### **Fabric-Specific Dependencies**

```bash
# Install with Fabric extras
poetry install --extras fabric

# Or build wheel with Fabric dependencies included
poetry build --format wheel
```

## 🔄 **Workflow for Different Use Cases**

### **For Development**
```bash
# Clone repo
git clone <repo>
cd meta-spliceai

# Create environment
mamba env create -f environment.yml
mamba activate surveyor

# Start developing
code .
```

### **For Production Deployment**
```bash
# Export exact environment
mamba env export --no-builds > environment-production.yml

# Deploy
mamba env create -f environment-production.yml
```

### **For Distribution/Publishing**
```bash
# Update version
poetry version patch  # or minor, major

# Build wheel
poetry build

# Publish to PyPI (if public)
poetry publish

# Or distribute wheel file directly
scp dist/*.whl user@server:/path/
```

### **For Microsoft Fabric**
```bash
# Build wheel with all dependencies
poetry build

# Test wheel installation
pip install dist/meta_spliceai-*.whl

# Deploy to Fabric
# Upload wheel to Fabric workspace
```

## 📊 **Comparison: Current vs Previous**

| Aspect | Previous (Multiple Files) | Current (Mamba + Poetry) |
|--------|--------------------------|---------------------------|
| **Reproducibility** | ❌ Version drift between files | ✅ Single source of truth |
| **Speed** | ⚠️ Slow pip dependency resolution | ✅ Fast mamba resolution |
| **Conflicts** | ❌ conda-libmamba-solver issues | ✅ No conflicts |
| **Distribution** | ❌ Complex wheel building | ✅ Simple `poetry build` |
| **Development** | ⚠️ Multiple tools to manage | ✅ Unified workflow |
| **Microsoft Fabric** | ❌ No wheel building | ✅ Direct wheel deployment |

## 🎯 **Action Items**

### **Immediate Updates Needed**

1. **✅ Update pyproject.toml** - modernize dependencies and add Fabric support
2. **✅ Keep environment.yml current** - this is your primary environment file
3. **❌ Deprecate requirements.txt** - add deprecation notice
4. **✅ Add wheel building workflow** - for Microsoft Fabric deployment

### **Next Steps**

1. **Update pyproject.toml** (see updated version below)
2. **Test wheel building** for Fabric deployment
3. **Update CI/CD** to use mamba instead of conda
4. **Document Fabric deployment** process

---

## 🚀 **Why This Approach Works**

1. **Mamba**: Fast, reliable dependency resolution without conflicts
2. **environment.yml**: Single source of truth for exact reproducibility  
3. **Poetry**: Modern tool for package building and distribution
4. **Deprecate requirements.txt**: Eliminates version drift and confusion

This hybrid approach gives you the **best of both worlds**:
- Fast, reliable environments via Mamba
- Modern packaging and distribution via Poetry
- Microsoft Fabric compatibility via wheels
- No more version conflicts or dependency drift 