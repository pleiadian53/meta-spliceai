# Data Transfer Guide for GPU Training

**Purpose**: Document how to transfer required data to RunPods for GPU training  
**Last Updated**: December 15, 2025

---

## 📊 Required Data Summary

| Data | Size | Location | Required For |
|------|------|----------|--------------|
| **SpliceVarDB** | 6.8 MB | `meta_spliceai/splice_engine/case_studies/workflows/splicevardb/splicevardb.download.tsv` | ✅ All experiments |
| **FASTA (GRCh38)** | 2.9 GB | `data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa` | ✅ Sequence extraction |
| **Meta-models artifacts** | 2.2 GB | `data/mane/GRCh38/openspliceai_eval/meta_models/` | ✅ **Required for artifact-based training** |
| **GTF annotations** | ~50 MB | `data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf` | ❌ **Not needed if using precomputed artifacts** |

### Important Notes

**GTF/GFF Files**: If you're using **precomputed artifacts** (the `analysis_sequences_*.tsv` files), you **do NOT need** the GTF/GFF file. The labels from GTF annotations are already embedded in the artifacts (in the `splice_type` column). You'd only need GTF/GFF if you were regenerating artifacts from scratch.

**Precomputed Artifacts**: These are **required** for artifact-based training with OpenSpliceAI. The artifacts contain:
- Base model predictions (donor/acceptor/neither scores)
- Derived features (position, context, etc.)
- Ground truth labels (from GTF, already embedded)
- Sequence windows (±250nt around each position)

### Minimum Transfer (~10 MB)
For delta prediction (HyenaDNA, ValidatedDelta) **without artifact-based training**, you only need:
- ✅ SpliceVarDB TSV (6.8 MB)
- ✅ FASTA can be downloaded directly on RunPods (see Option 3)

**For artifact-based training** (recommended), you also need:
- ✅ Precomputed artifacts (2.2 GB) - see transfer instructions below

---

## 🚀 Transfer Options

### Option 1: SCP Transfer (From Local Mac)

Your RunPods connection supports SCP (SSH over exposed TCP):

```bash
# Connection details (update these from RunPods dashboard!)
RUNPOD_HOST="69.30.85.30"
RUNPOD_PORT="22084"
SSH_KEY="~/.ssh/id_ed25519"

# 1. Create directories on RunPods first
ssh -p ${RUNPOD_PORT} -i ${SSH_KEY} root@${RUNPOD_HOST} \
    "mkdir -p /workspace/meta-spliceai/meta_spliceai/splice_engine/case_studies/workflows/splicevardb && \
     mkdir -p /workspace/meta-spliceai/data/mane/GRCh38"

# 2. Transfer SpliceVarDB (small, fast)
scp -P ${RUNPOD_PORT} -i ${SSH_KEY} \
    ~/work/meta-spliceai/meta_spliceai/splice_engine/case_studies/workflows/splicevardb/splicevardb.download.tsv \
    root@${RUNPOD_HOST}:/workspace/meta-spliceai/meta_spliceai/splice_engine/case_studies/workflows/splicevardb/

# 3. Transfer FASTA (large, ~10-30 min depending on connection)
scp -P ${RUNPOD_PORT} -i ${SSH_KEY} \
    ~/work/meta-spliceai/data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    root@${RUNPOD_HOST}:/workspace/meta-spliceai/data/mane/GRCh38/

# 4. Transfer pre-computed artifacts (REQUIRED for artifact-based training)
#    Use rsync for better progress tracking and resumability
echo "Transferring precomputed artifacts (2.2 GB - this will take 15-30 min)..."
ssh -p ${RUNPOD_PORT} -i ${SSH_KEY} root@${RUNPOD_HOST} \
    "mkdir -p /workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval"

rsync -avz --progress \
    -e "ssh -p ${RUNPOD_PORT} -i ${SSH_KEY}" \
    ~/work/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_models/ \
    root@${RUNPOD_HOST}:/workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_models/
```

### Option 2: rsync (Better for Resumable Transfers)

**Recommended for large transfers** (artifacts, FASTA):

```bash
# Connection details
RUNPOD_HOST="69.30.85.30"
RUNPOD_PORT="22084"
SSH_KEY="~/.ssh/id_ed25519"

# Create directories first
ssh -p ${RUNPOD_PORT} -i ${SSH_KEY} root@${RUNPOD_HOST} "
    mkdir -p /workspace/meta-spliceai/meta_spliceai/splice_engine/case_studies/workflows/splicevardb
    mkdir -p /workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval
"

# Transfer SpliceVarDB
rsync -avz --progress \
    -e "ssh -p ${RUNPOD_PORT} -i ${SSH_KEY}" \
    ~/work/meta-spliceai/meta_spliceai/splice_engine/case_studies/workflows/splicevardb/splicevardb.download.tsv \
    root@${RUNPOD_HOST}:/workspace/meta-spliceai/meta_spliceai/splice_engine/case_studies/workflows/splicevardb/

# Transfer FASTA file (if not downloading on RunPods)
rsync -avz --progress \
    -e "ssh -p ${RUNPOD_PORT} -i ${SSH_KEY}" \
    ~/work/meta-spliceai/data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    root@${RUNPOD_HOST}:/workspace/meta-spliceai/data/mane/GRCh38/

# Transfer precomputed artifacts (REQUIRED for artifact-based training)
rsync -avz --progress \
    -e "ssh -p ${RUNPOD_PORT} -i ${SSH_KEY}" \
    ~/work/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_models/ \
    root@${RUNPOD_HOST}:/workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_models/
```

### Option 3: Download FASTA Directly on RunPods (Recommended!)

The FASTA file can be downloaded from Ensembl directly on RunPods - often faster than SCP:

```bash
# SSH into RunPods
ssh -p 40195 -i ~/.ssh/id_ed25519 root@213.192.2.86

# Create directory
mkdir -p /workspace/meta-spliceai/data/mane/GRCh38
cd /workspace/meta-spliceai/data/mane/GRCh38

# Download GRCh38 FASTA from Ensembl (~2.9GB)
wget https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz

# Decompress (takes ~2-3 minutes)
gunzip Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz

# Index the FASTA (required by pyfaidx)
pip install pyfaidx
python -c "from pyfaidx import Fasta; Fasta('Homo_sapiens.GRCh38.dna.primary_assembly.fa')"
```

**Note**: For MANE-specific reference, you may need a different FASTA. The code may use:
- `GCF_000001405.40_GRCh38.p14_genomic.fna` (NCBI RefSeq)
- `Homo_sapiens.GRCh38.dna.primary_assembly.fa` (Ensembl)

Check `meta_layer/core/path_manager.py` for the expected path.

---

## 🔄 Quick Transfer Script

Save this as `transfer_to_runpods.sh` on your local Mac:

```bash
#!/bin/bash
# Transfer essential data to RunPods for meta-layer GPU training

# Configuration (update these!)
RUNPOD_HOST="213.192.2.86"
RUNPOD_PORT="40195"
SSH_KEY="~/.ssh/id_ed25519"
LOCAL_ROOT="$HOME/work/meta-spliceai"
REMOTE_ROOT="/workspace/meta-spliceai"

echo "=== Meta-SpliceAI Data Transfer ==="
echo "Host: ${RUNPOD_HOST}:${RUNPOD_PORT}"

# Create remote directories
echo "Creating remote directories..."
ssh -p ${RUNPOD_PORT} -i ${SSH_KEY} root@${RUNPOD_HOST} "
    mkdir -p ${REMOTE_ROOT}/meta_spliceai/splice_engine/case_studies/workflows/splicevardb
    mkdir -p ${REMOTE_ROOT}/data/mane/GRCh38/openspliceai_eval
"

# Transfer SpliceVarDB (essential, small)
echo "Transferring SpliceVarDB (6.8 MB)..."
rsync -avz --progress \
    -e "ssh -p ${RUNPOD_PORT} -i ${SSH_KEY}" \
    ${LOCAL_ROOT}/meta_spliceai/splice_engine/case_studies/workflows/splicevardb/splicevardb.download.tsv \
    root@${RUNPOD_HOST}:${REMOTE_ROOT}/meta_spliceai/splice_engine/case_studies/workflows/splicevardb/

# Ask about FASTA
read -p "Transfer FASTA from local? (y/n, or download on RunPods): " choice
if [ "$choice" = "y" ]; then
    echo "Transferring FASTA (2.9 GB - this will take a while)..."
    rsync -avz --progress \
        -e "ssh -p ${RUNPOD_PORT} -i ${SSH_KEY}" \
        ${LOCAL_ROOT}/data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
        root@${RUNPOD_HOST}:${REMOTE_ROOT}/data/mane/GRCh38/
else
    echo "Download FASTA on RunPods with:"
    echo "  wget https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
fi

echo "=== Transfer Complete ==="
echo "Next: SSH into RunPods and run training"
```

---

## ✅ Verification

After transfer, verify files on RunPods:

```bash
# SSH into RunPods
ssh -p 40195 -i ~/.ssh/id_ed25519 root@213.192.2.86

# Check SpliceVarDB
ls -lh /workspace/meta-spliceai/meta_spliceai/splice_engine/case_studies/workflows/splicevardb/
# Should show: splicevardb.download.tsv (6.8MB)

# Check FASTA
ls -lh /workspace/meta-spliceai/data/mane/GRCh38/
# Should show: Homo_sapiens.GRCh38.dna.primary_assembly.fa (~2.9GB)

# Check precomputed artifacts (if transferred)
ls -lh /workspace/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_models/ | head -10
# Should show: analysis_sequences_*.tsv files (many files, ~2.2GB total)

# Test loading
cd /workspace/meta-spliceai
mamba activate metaspliceai
python -c "
from meta_spliceai.splice_engine.meta_layer.data.splicevardb_loader import load_splicevardb
loader = load_splicevardb(genome_build='GRCh38')
print(f'Loaded {len(loader.get_all_variants())} variants')
"
```

---

## 🔀 Cursor Remote SSH (Alternative)

Cursor IDE supports remote SSH, which can help with file transfers:

1. Install "Remote - SSH" extension in Cursor
2. Add RunPods to SSH config:
   ```
   Host runpod-metaspliceai
       HostName 213.192.2.86
       Port 40195
       User root
       IdentityFile ~/.ssh/id_ed25519
   ```
3. Connect: `Cmd+Shift+P` → "Remote-SSH: Connect to Host" → Select `runpod-metaspliceai`
4. Open folder: `/workspace/meta-spliceai`
5. Drag and drop files! (Works for small files)

For large files (>100MB), use SCP/rsync instead.

---

## 📁 Data Structure on RunPods

After transfer, your RunPods should have:

```
/workspace/meta-spliceai/
├── meta_spliceai/
│   └── splice_engine/
│       └── case_studies/
│           └── workflows/
│               └── splicevardb/
│                   └── splicevardb.download.tsv  ← REQUIRED (6.8MB)
├── data/
│   └── mane/
│       └── GRCh38/
│           ├── Homo_sapiens.GRCh38.dna.primary_assembly.fa  ← REQUIRED (2.9GB)
│           └── openspliceai_eval/
│               └── meta_models/
│                   ├── analysis_sequences_*.tsv  ← REQUIRED for artifact-based training (2.2GB)
│                   ├── splice_errors_*.tsv
│                   └── gene_manifest.tsv
└── (rest of repo from git clone)
```

---

## 🏃 After Transfer: Start Training

```bash
# On RunPods
cd /workspace/meta-spliceai
mamba activate metaspliceai

# Quick test
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

from meta_spliceai.splice_engine.meta_layer.models import HyenaDNADeltaPredictor
model = HyenaDNADeltaPredictor(model_name='hyenadna-small-32k').to('cuda')
print(f'HyenaDNA loaded on GPU!')
"

# Run training (see GPU_TRAINING_GUIDE.md)
cd meta_spliceai/splice_engine/meta_layer/tests
python test_validated_delta_experiments.py --samples 10000 --epochs 50
```

---

*This guide ensures you can transfer the minimal required data to start GPU experiments.*

