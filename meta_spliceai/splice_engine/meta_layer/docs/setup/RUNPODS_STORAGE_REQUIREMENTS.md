# RunPods Storage Requirements for Meta-Layer GPU Training

**Created**: December 16, 2025  
**Issue**: Environment setup failed due to insufficient disk quota

---

## Problem Summary

RunPods instances have a **hidden storage quota** on `/workspace` that is not shown by `df -h`. The reported "free space" (e.g., 213TB on shared storage) is misleading - each pod has an individual quota.

### Symptoms

```
critical libmamba Write failed
error    libmamba Could not write to file /workspace/miniforge3/pkgs/...: Disk quota exceeded
```

Or with pip:
```
ERROR: Could not install packages due to an OSError: [Errno 122] Disk quota exceeded
```

### Root Cause

PyTorch with CUDA requires significant storage:
- **PyTorch CUDA packages (conda)**: ~4GB download + ~8GB extracted
- **Miniforge base**: ~9GB
- **metaspliceai environment**: ~3GB
- **Meta-SpliceAI repo**: ~3GB
- **Total needed**: **~20-25GB minimum**

A 20GB quota is insufficient.

---

## Minimum Storage Requirements

| Component | Size | Notes |
|-----------|------|-------|
| Miniforge3 base | 9 GB | Core conda installation |
| metaspliceai env | 5-8 GB | Depends on packages |
| PyTorch CUDA (pip) | 2-3 GB | Much smaller than conda |
| Meta-SpliceAI repo | 3 GB | Including data files |
| FASTA reference | 3 GB | GRCh38 primary assembly |
| SpliceVarDB | 7 MB | Training data |
| Package cache | 2-5 GB | Temporary during install |
| **TOTAL** | **~25-30 GB minimum** |

### Recommendation

**Provision RunPods instance with at least 50GB storage** to allow for:
- Model checkpoints (~500MB each)
- Training logs
- Experiment outputs
- Future data files

---

## How to Check Your Quota

The reported disk space is misleading:
```bash
# This shows SHARED storage, not YOUR quota:
df -h /workspace
# Filesystem                   Size  Used Avail Use%
# mfs#...runpod.net:9421       515T  302T  213T  59%  ← MISLEADING!

# Test actual writable space:
dd if=/dev/zero of=/workspace/test_write bs=1M count=100
# If this fails with "Disk quota exceeded", your quota is full
rm -f /workspace/test_write
```

Check actual usage:
```bash
du -sh /workspace/*
```

---

## RunPods Instance Selection

When creating a new RunPods instance:

1. **Go to**: RunPods Dashboard → Deploy → GPU Cloud
2. **Select GPU**: RTX 3090 / RTX 4090 / A100 (24GB+ VRAM recommended)
3. **IMPORTANT**: Look for **"Volume Disk"** or **"Container Disk"** settings
4. **Set storage**: **50GB minimum** (100GB recommended for extensive experiments)

### Pod Configuration Checklist

```
□ GPU: RTX 4090 or better (24GB+ VRAM)
□ Volume Disk: 50GB minimum
□ Container Disk: 20GB (default is fine)
□ Template: PyTorch or Ubuntu
```

---

## Optimized Installation Steps

Once you have sufficient storage:

### 1. Install Miniforge to /workspace

```bash
cd /workspace
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /workspace/miniforge3
rm Miniforge3-Linux-x86_64.sh  # Save 82MB
```

### 2. Configure for /workspace Storage

```bash
cat > /workspace/miniforge3/.condarc << 'EOF'
pkgs_dirs:
  - /workspace/miniforge3/pkgs
envs_dirs:
  - /workspace/miniforge3/envs
channels:
  - conda-forge
  - bioconda
  - defaults
EOF
```

### 3. Create Environment (Minimal - No CUDA from Conda)

```bash
source /workspace/miniforge3/etc/profile.d/mamba.sh
export MAMBA_ROOT_PREFIX=/workspace/miniforge3

# Create env WITHOUT pytorch from conda (saves ~5GB)
mamba create -n metaspliceai python=3.10 \
    numpy=1.26.4 scipy=1.11.4 pandas=2.1.4 \
    polars pyarrow matplotlib-base \
    scikit-learn statsmodels bedtools -y
```

### 4. Install PyTorch via Pip (Smaller)

```bash
mamba activate metaspliceai

# Set pip cache to /workspace
export PIP_CACHE_DIR=/workspace/pip_cache
mkdir -p $PIP_CACHE_DIR

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. Install Remaining Dependencies

```bash
pip install \
    argcomplete argh python-dotenv biopython pyyaml \
    gffutils h5py psutil pybedtools pyfaidx \
    requests rich seaborn tqdm tabulate \
    transformers einops  # For HyenaDNA
```

### 6. Install Meta-SpliceAI

```bash
cd /workspace/meta-spliceai
pip install -e .
```

### 7. Clean Up Caches

```bash
# Free space after installation
mamba clean --all -y
rm -rf /workspace/pip_cache
rm -rf /workspace/miniforge3/pkgs/*
```

---

## Verification

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

from meta_spliceai.splice_engine.meta_layer.models import (
    SimpleCNNDeltaPredictor,
    ValidatedDeltaPredictor,
    HyenaDNADeltaPredictor
)
from meta_spliceai.splice_engine.meta_layer.data import splicevardb_loader
print('✅ All imports OK!')
"
```

---

## Quick Reference

| Issue | Solution |
|-------|----------|
| "Disk quota exceeded" during mamba | Need larger volume disk |
| "bedtools not found" | Add bioconda channel |
| CUDA not available | Install PyTorch via pip with cu121 |
| Imports fail | Run `pip install -e .` in repo |

---

*Document created after encountering 20GB quota limitation on initial RunPods setup.*

