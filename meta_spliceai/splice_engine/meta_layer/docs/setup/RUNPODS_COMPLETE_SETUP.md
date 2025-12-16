# Complete RunPods Setup Guide for Meta-Layer GPU Training

**Purpose**: Step-by-step guide for setting up a RunPods VM for GPU training  
**Last Updated**: December 16, 2025  
**Audience**: AI agents and human developers

---

## ⚠️ CRITICAL: Storage Requirements

**Before creating a RunPods instance, ensure you have sufficient disk quota!**

| Requirement | Value |
|-------------|-------|
| **Volume Disk** | **50GB minimum** (100GB recommended) |
| GPU VRAM | 24GB+ (RTX 4090, A40, A100) |

The `df -h` command shows **misleading** shared storage (515TB). Each pod has a **hidden individual quota**. A 20GB quota is insufficient!

**See**: [RUNPODS_STORAGE_REQUIREMENTS.md](./RUNPODS_STORAGE_REQUIREMENTS.md) for details on storage errors.

---

## 📋 Table of Contents

1. [Quick Reference](#quick-reference)
2. [SSH Key Setup](#ssh-key-setup)
3. [Environment Installation](#environment-installation)
4. [Data Transfer](#data-transfer)
5. [GitHub Authentication](#github-authentication)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## Quick Reference

### Connection Details Template

```bash
# Update these from RunPods dashboard!
RUNPOD_HOST="<IP from SSH over exposed TCP>"
RUNPOD_PORT="<Port from SSH over exposed TCP>"
SSH_KEY="~/.ssh/id_ed25519"
```

### One-Liner Status Check

```bash
# On RunPods VM:
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" && \
echo "Disk: $(df -h /workspace | tail -1 | awk '{print $4}') free" && \
echo "Conda: $(which mamba)" && \
echo "Meta-SpliceAI: $(pip show meta-spliceai 2>/dev/null | grep Version || echo 'Not installed')"
```

---

## SSH Key Setup

### Step 1: Generate SSH Key (On Local Machine)

If you don't have an SSH key:

```bash
# Generate ED25519 key (recommended)
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_ed25519

# Or RSA if ED25519 not supported
ssh-keygen -t rsa -b 4096 -C "your_email@example.com" -f ~/.ssh/id_rsa
```

### Step 2: Upload Public Key to RunPods

1. Go to [RunPods Settings](https://www.runpod.io/console/user/settings)
2. Navigate to "SSH Public Keys"
3. Click "Add SSH Key"
4. Paste your public key:
   ```bash
   # Copy this output:
   cat ~/.ssh/id_ed25519.pub
   ```
5. Save

### Step 3: Configure SSH Config (Local Machine)

Add to `~/.ssh/config`:

```ssh
Host runpod-metaspliceai
    # ⚠️ UPDATE THESE from RunPods dashboard "SSH over exposed TCP"
    HostName <IP_ADDRESS>
    Port <PORT>
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

### Step 4: Test Connection

```bash
ssh runpod-metaspliceai
# Should connect without password prompt
```

---

## Environment Installation

### Step 1: Install Miniforge (On RunPods VM)

```bash
# SSH into RunPods
ssh runpod-metaspliceai

# Install to /workspace (has more disk space than /)
cd /workspace
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /workspace/miniforge3

# Initialize
/workspace/miniforge3/bin/conda init bash
source ~/.bashrc

# Verify
which mamba
# Should show: /workspace/miniforge3/bin/mamba
```

### Step 2: Clone Repository

```bash
cd /workspace
git clone https://github.com/pleiadian53/meta-spliceai.git
cd meta-spliceai
```

### Step 3: Create Environment

```bash
# Use the minimal RunPods config (avoids problematic dependencies)
mamba env create -f environment-runpods-minimal.yml

# Activate
mamba activate metaspliceai

# Install CUDA PyTorch (required for GPU)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install HyenaDNA dependencies
pip install transformers einops

# Install project in editable mode
pip install -e .
```

### Step 4: Verify Installation

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# Test meta_layer imports
from meta_spliceai.splice_engine.meta_layer.models import SimpleCNNDeltaPredictor
print('Meta-layer imports: OK')
"
```

---

## Data Transfer

### Required Files

| File | Size | Purpose | Transfer Method |
|------|------|---------|-----------------|
| `splicevardb.download.tsv` | 6.8 MB | Training data | SCP |
| `Homo_sapiens.GRCh38.dna.primary_assembly.fa` | 2.9 GB | Sequence extraction | Download on VM |

### Step 1: Transfer SpliceVarDB (From Local Machine)

```bash
# Create directory on VM first
ssh runpod-metaspliceai "mkdir -p /workspace/meta-spliceai/meta_spliceai/splice_engine/case_studies/workflows/splicevardb"

# Transfer SpliceVarDB
scp -P <PORT> -i ~/.ssh/id_ed25519 \
    ~/work/meta-spliceai/meta_spliceai/splice_engine/case_studies/workflows/splicevardb/splicevardb.download.tsv \
    root@<HOST>:/workspace/meta-spliceai/meta_spliceai/splice_engine/case_studies/workflows/splicevardb/
```

### Step 2: Download FASTA (On RunPods VM)

```bash
# On RunPods VM
mkdir -p /workspace/meta-spliceai/data/mane/GRCh38
cd /workspace/meta-spliceai/data/mane/GRCh38

# Download from Ensembl (faster than SCP for large files)
wget https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz

# Decompress
gunzip Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz

# Index for pyfaidx
python -c "from pyfaidx import Fasta; Fasta('Homo_sapiens.GRCh38.dna.primary_assembly.fa')"
```

### Step 3: Verify Data

```bash
# On RunPods VM
cd /workspace/meta-spliceai
mamba activate metaspliceai

python -c "
from meta_spliceai.splice_engine.meta_layer.data.splicevardb_loader import load_splicevardb
loader = load_splicevardb(genome_build='GRCh38')
variants = loader.get_all_variants()
print(f'Loaded {len(variants)} SpliceVarDB variants')
"
```

---

## GitHub Authentication

### Why This Matters

For the AI agent on RunPods to push changes back to GitHub (experiment results, model improvements), it needs write access.

### Option A: Deploy Key (Recommended for Repos)

**On RunPods VM:**

```bash
# Generate a new key specifically for this VM
ssh-keygen -t ed25519 -C "runpods-metaspliceai" -f ~/.ssh/github_deploy -N ""

# Show the public key
cat ~/.ssh/github_deploy.pub
```

**On GitHub:**

1. Go to your repo → Settings → Deploy keys
2. Click "Add deploy key"
3. Name: "RunPods Training VM"
4. Paste the public key
5. ✅ Check "Allow write access"
6. Save

**Configure Git on VM:**

```bash
# Configure git to use the deploy key
cat >> ~/.ssh/config << 'EOF'
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/github_deploy
    IdentitiesOnly yes
EOF

# Set git identity
git config --global user.email "your_email@example.com"
git config --global user.name "RunPods Training"

# Test
ssh -T git@github.com
# Should show: "Hi pleiadian53/meta-spliceai! You've successfully authenticated..."
```

### Option B: Personal Access Token (Alternative)

If deploy keys don't work:

```bash
# Generate token at: https://github.com/settings/tokens
# Select: repo (full control)

# Store credentials
git config --global credential.helper store

# Next git push will prompt for username/token
# Username: your_github_username
# Password: <paste_your_token>
```

### Verify Push Access

```bash
cd /workspace/meta-spliceai

# Create a test file
echo "# Test commit from RunPods $(date)" >> /tmp/test_push.md

# Commit and push (dry run)
git status
# Should show your changes

# Actual push (when ready)
# git add -A && git commit -m "Test from RunPods" && git push
```

---

## Verification

### Complete Verification Script

Save as `/workspace/verify_setup.sh`:

```bash
#!/bin/bash
# Complete Meta-SpliceAI RunPods Setup Verification

echo "=== Meta-SpliceAI RunPods Setup Verification ==="
echo ""

# 1. GPU
echo "1. GPU Status:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# 2. Disk Space
echo "2. Disk Space:"
df -h /workspace | tail -1
echo ""

# 3. Conda/Mamba
echo "3. Conda Environment:"
echo "   Mamba: $(which mamba)"
echo "   Active env: $CONDA_DEFAULT_ENV"
echo ""

# 4. PyTorch
echo "4. PyTorch Status:"
python -c "
import torch
print(f'   Version: {torch.__version__}')
print(f'   CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
"
echo ""

# 5. Meta-SpliceAI
echo "5. Meta-SpliceAI:"
pip show meta-spliceai 2>/dev/null | grep -E "^(Name|Version|Location):" || echo "   Not installed"
echo ""

# 6. Data Files
echo "6. Data Files:"
if [ -f "/workspace/meta-spliceai/meta_spliceai/splice_engine/case_studies/workflows/splicevardb/splicevardb.download.tsv" ]; then
    echo "   ✅ SpliceVarDB: OK"
else
    echo "   ❌ SpliceVarDB: MISSING"
fi

if [ -f "/workspace/meta-spliceai/data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa" ]; then
    echo "   ✅ FASTA: OK"
else
    echo "   ❌ FASTA: MISSING"
fi
echo ""

# 7. Git Access
echo "7. GitHub Access:"
ssh -T git@github.com 2>&1 | head -1
echo ""

# 8. Test Import
echo "8. Meta-Layer Imports:"
python -c "
from meta_spliceai.splice_engine.meta_layer.models import (
    SimpleCNNDeltaPredictor,
    ValidatedDeltaPredictor,
    HyenaDNADeltaPredictor
)
print('   ✅ All model imports OK')
" 2>/dev/null || echo "   ❌ Import failed"

echo ""
echo "=== Verification Complete ==="
```

Run with:

```bash
chmod +x /workspace/verify_setup.sh
/workspace/verify_setup.sh
```

---

## Troubleshooting

### Issue: "No space left on device"

```bash
# Check disk usage
df -h

# If / is full but /workspace has space, reinstall mamba to /workspace:
rm -rf ~/miniforge3
# Follow Step 1 of Environment Installation
```

### Issue: CUDA not available

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: SSH connection drops

Add to `~/.ssh/config` on local machine:

```ssh
Host runpod-metaspliceai
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

### Issue: git push rejected

```bash
# Make sure you have the latest
git pull origin main --rebase

# Then push
git push origin main
```

### Issue: Import errors

```bash
# Reinstall meta-spliceai
pip install -e /workspace/meta-spliceai

# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"
```

---

## Quick Start Checklist

```
□ SSH key generated and uploaded to RunPods
□ SSH config updated with RunPods host/port
□ Can SSH without password: ssh runpod-metaspliceai
□ Miniforge installed to /workspace/miniforge3
□ metaspliceai environment created and activated
□ PyTorch with CUDA installed
□ meta-spliceai installed in editable mode
□ SpliceVarDB transferred (6.8 MB)
□ FASTA downloaded (2.9 GB)
□ GitHub deploy key configured (for pushing)
□ Verification script passes
```

---

## Starting Training

Once setup is complete:

```bash
# SSH into VM
ssh runpod-metaspliceai

# Activate environment
mamba activate metaspliceai

# Start tmux session (persists if SSH disconnects)
tmux new -s training

# Navigate to project
cd /workspace/meta-spliceai

# Pull latest code
git pull origin main

# Start training (example)
python meta_spliceai/splice_engine/meta_layer/tests/test_validated_delta_experiments.py \
    --samples 10000 \
    --epochs 50 \
    --batch-size 64

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

---

## Related Documentation

- [GPU_TRAINING_GUIDE.md](../experiments/GPU_TRAINING_GUIDE.md) - Training guide
- [DATA_TRANSFER_GUIDE.md](../experiments/DATA_TRANSFER_GUIDE.md) - Data transfer details
- [GPU_EXPERIMENTS.md](../wishlist/GPU_EXPERIMENTS.md) - Experiment queue

---

*This guide ensures consistent, reproducible setup for GPU training environments.*

