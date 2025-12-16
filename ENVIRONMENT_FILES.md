# Environment Files Guide

Meta-SpliceAI provides platform-specific environment files for different development environments.

---

## 📋 Available Files

| File | Platform | Use Case |
|------|----------|----------|
| **`environment.yml`** | **M1/M2 Mac** | Local development on Apple Silicon |
| **`environment-linux.yml`** | **Linux/RunPods** | GPU training on cloud instances |

---

## 🔑 Key Differences

### environment.yml (Mac)
```yaml
- tensorflow==2.15.0
- tensorflow-macos==2.15.0  # M1/M2 optimized
```

### environment-linux.yml (Linux)
```yaml
- tensorflow==2.15.0
# tensorflow-macos removed (not available on Linux)
```

---

## 🚀 Usage

### On M1/M2 Mac (Local Development)

```bash
mamba env create -f environment.yml
mamba activate metaspliceai
pip install -e .
```

### On Linux/RunPods (GPU Training)

```bash
mamba env create -f environment-linux.yml
mamba activate metaspliceai

# Upgrade PyTorch for CUDA support
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -e .
```

---

## 🔧 Why Two Files?

**Problem**: `tensorflow-macos` is only available for macOS and will cause installation failures on Linux.

**Solution**: Platform-specific environment files ensure smooth installation across different environments.

---

## 📦 What's Included (Both Files)

- **Python**: 3.10
- **Core**: numpy, pandas, polars, scipy
- **ML Frameworks**: scikit-learn, xgboost, lightgbm, catboost
- **Deep Learning**: PyTorch, TensorFlow
- **Genomics**: biopython, pyfaidx, gffutils, pybedtools
- **Visualization**: matplotlib, seaborn, plotly
- **Splice Models**: spliceai

---

## ⚙️ Updating Environments

When adding new dependencies:

1. Update **both** files to keep them in sync
2. Only difference should be `tensorflow-macos` line
3. Test on both Mac and Linux if possible

---

**See Also**:
- [Installation Guide](docs/installation/INSTALLATION.md)
- [RunPods Setup](dev/runpods/RUNPODS_QUICK_SETUP.md)


