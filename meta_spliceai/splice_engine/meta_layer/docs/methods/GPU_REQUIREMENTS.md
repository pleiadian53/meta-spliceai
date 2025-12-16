# GPU Resource Requirements for Meta-Layer Methods

**Created**: December 15, 2025  
**Updated**: December 16, 2025  
**Purpose**: Guide for deciding which experiments to run locally vs. on RunPods

---

## Quick Reference: What Runs Where

| Method/Model | M1 Mac (16GB) | RTX 4090 (24GB) | A40 (48GB) | A100 (80GB) |
|--------------|---------------|-----------------|------------|-------------|
| **SimpleCNNDeltaPredictor** | ✅ Fast | ✅ Very Fast | ✅ | ✅ |
| **ValidatedDeltaPredictor** | ✅ Fast | ✅ Very Fast | ✅ | ✅ |
| **SpliceInducingClassifier** | ✅ Fast | ✅ Very Fast | ✅ | ✅ |
| HyenaDNA-tiny (1K context) | ⚠️ Slow | ✅ Fast | ✅ | ✅ |
| **HyenaDNA-small (8K context)** | ❌ OOM | ✅ Fast | ✅ | ✅ |
| **HyenaDNA-medium (32K context)** | ❌ OOM | ⚠️ Tight | ✅ Fast | ✅ |
| **HyenaDNA-large (160K context)** | ❌ OOM | ❌ OOM | ⚠️ Slow | ✅ Fast |
| DNABERT-2 | ❌ OOM | ✅ Fast | ✅ | ✅ |
| Full SpliceVarDB (50K) | ⚠️ Hours | ✅ Minutes | ✅ | ✅ |

**Legend**: ✅ Recommended | ⚠️ Possible but slow/tight | ❌ Won't fit

---

## Methods That REQUIRE GPU (RunPods)

### 1. HyenaDNA-Based Encoders 🔴 HIGH PRIORITY

**Why GPU Required**:
- HyenaDNA is a 7M-100M parameter DNA foundation model
- Requires CUDA for efficient state space model computation
- Memory grows with sequence length (O(n) but with large constants)

**Model Sizes**:
| Model | Parameters | Max Context | Min VRAM |
|-------|------------|-------------|----------|
| hyenadna-tiny-1k | 7M | 1,024 | 4GB |
| hyenadna-small-32k | 7M | 32,768 | 8GB |
| hyenadna-medium-160k | 50M | 163,840 | 24GB |
| hyenadna-large-450k | 100M | 450,000 | 48GB |

**Recommended for RunPods**:
```python
# RTX 4090 (24GB) - Best value
from meta_spliceai.splice_engine.meta_layer.models import HyenaDNADeltaPredictor

model = HyenaDNADeltaPredictor(
    model_name='hyenadna-small-32k',  # 32K context
    freeze_encoder=True,               # Start frozen, then unfreeze
    hidden_dim=256
)
```

**Files to Run**:
- `models/hyenadna_delta_predictor.py` - Contains `HyenaDNADeltaPredictor`
- Training script: Create based on `tests/test_validated_delta_experiments.py`

---

### 2. DNABERT-2 Encoder 🟡 MEDIUM PRIORITY

**Why GPU Required**:
- BERT-based transformer architecture
- Attention scales O(n²) with sequence length
- Fine-tuning requires gradient storage

**Specifications**:
- Parameters: ~110M
- Max context: 512 tokens (roughly 2K nucleotides with k-mer tokenization)
- Min VRAM: 8GB (inference), 16GB (training)

**Not Yet Implemented** but architecture is similar:
```python
# Future implementation
class DNABERTDeltaPredictor(nn.Module):
    def __init__(self, freeze_encoder=True):
        self.encoder = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M")
        # ...
```

---

### 3. Full SpliceVarDB Training (~50K variants) 🟡 MEDIUM PRIORITY

**Why GPU Helps**:
- 8K samples on M1 takes ~2 hours for data prep + training
- 50K samples would take ~10+ hours on M1
- GPU parallelization: 10-50x speedup

**Current Experiment Running**:
```bash
# Running on M1 (will take a while)
python -m meta_spliceai.splice_engine.meta_layer.tests.test_validated_delta_experiments --exp more_data
```

**On GPU (much faster)**:
```python
# Same code, but with larger batch and all data
MAX_TRAIN = 50000  # All of SpliceVarDB
BATCH_SIZE = 256   # vs 32-64 on M1
```

---

### 4. Ensemble Training 🟢 LOW PRIORITY (but benefits from GPU)

**What**: Train multiple models with different configurations, then ensemble

**Why GPU Helps**:
- Need to train 5-10 models for proper ensemble
- Each model takes 1-2 hours on M1
- GPU can run all experiments in parallel

---

## Methods That Work Fine on M1 Mac

### ✅ SimpleCNNDeltaPredictor (Current Best)
- 3M parameters
- Runs in ~30 min for 2000 samples
- Already tested and works well

### ✅ ValidatedDeltaPredictor ⭐ BEST
- Same architecture as SimpleCNN
- Uses validated targets from SpliceVarDB
- **Best correlation: r=0.507 (8000 samples)**
- Key finding: More data → better results (+24% with 4x data)

### ✅ SpliceInducingClassifier
- Binary classification (simpler task)
- Fast training, small model

### ✅ Calibrated Predictors (Quantile, Temperature, etc.)
- Same base architecture
- Just different loss functions

---

## RunPods Setup Guide

### Recommended Pod Configuration

| Use Case | Pod Type | GPU | VRAM | $/hr |
|----------|----------|-----|------|------|
| HyenaDNA-small | Community | RTX 4090 | 24GB | ~$0.44 |
| HyenaDNA-medium | Secure | A40 | 48GB | ~$0.79 |
| Full fine-tuning | Secure | A100 | 80GB | ~$1.89 |

### Environment Setup Script

```bash
#!/bin/bash
# runpods_setup.sh

# Clone repo
git clone https://github.com/your-org/meta-spliceai.git
cd meta-spliceai

# Create environment
conda create -n metaspliceai python=3.10 -y
conda activate metaspliceai

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install HyenaDNA dependencies
pip install transformers einops flash-attn

# Install project
pip install -e .

# Download genomic resources (if needed)
# python -m meta_spliceai.system.genomic_resources.cli download --build GRCh38

# Run experiment
python train_hyenadna.py --model small --epochs 50
```

---

## Experiments to Run on RunPods

### Priority 1: HyenaDNA with Validated Delta Targets

```python
# train_hyenadna_validated.py
import torch
from meta_spliceai.splice_engine.meta_layer.models import HyenaDNADeltaPredictor
from meta_spliceai.splice_engine.meta_layer.data.splicevardb_loader import load_splicevardb

# Config for RTX 4090
device = torch.device('cuda')
BATCH_SIZE = 128
MAX_TRAIN = 20000
EPOCHS = 100

# Model
model = HyenaDNADeltaPredictor(
    model_name='hyenadna-small-32k',
    freeze_encoder=True,  # Start frozen
    hidden_dim=256
).to(device)

# Data (same as test_validated_delta_experiments.py)
loader = load_splicevardb(genome_build='GRCh38')
# ... training loop ...
```

### Priority 2: Full SpliceVarDB with Gated CNN

```python
# Same as test_validated_delta_experiments.py but with:
MAX_TRAIN = 40000  # Use most of SpliceVarDB
BATCH_SIZE = 256
EPOCHS = 100
```

### Priority 3: Cross-Validation

```python
# 5-fold cross-validation for robust estimates
for fold in range(5):
    train_variants, val_variants = get_fold(fold)
    model = SimpleCNNDeltaPredictor(...)
    # train and evaluate
```

---

## Summary: What to Run Where

### M1 Mac (Development & Quick Iterations)
1. ✅ Debugging and development
2. ✅ Quick experiments with 2000-8000 samples
3. ✅ SimpleCNN/ValidatedDelta with quantile loss
4. ✅ Binary classification experiments

### RunPods GPU (Production Training)
1. 🔴 HyenaDNA encoder experiments
2. 🔴 Full SpliceVarDB training (50K variants)
3. 🟡 DNABERT-2 (if implemented)
4. 🟡 Cross-validation and hyperparameter search
5. 🟢 Ensemble model training

---

*This guide helps prioritize experiments based on available compute resources.*

