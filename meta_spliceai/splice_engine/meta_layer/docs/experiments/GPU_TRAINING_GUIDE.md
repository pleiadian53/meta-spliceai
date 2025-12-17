# GPU Training Guide for Meta-Layer Experiments

**Purpose**: Jump-start GPU-intensive training on RunPods or similar cloud GPU instances  
**Last Updated**: December 15, 2025  
**Status**: Ready for GPU experiments

---

## 🎯 Quick Context

The **meta_layer** package implements multimodal deep learning to detect **alternative splice sites** induced by genetic variants. We've tested on M1 Mac and now need **GPU resources** for:

1. **HyenaDNA encoder** (pre-trained DNA foundation model)
2. **Full SpliceVarDB dataset** (~50K variants instead of 2K samples)
3. **Larger models and longer context windows**

### Current Best Result (M1 Mac)
- **ValidatedDeltaPredictor**: r=0.41 correlation
- Architecture: Gated CNN, 6 layers, hidden_dim=128
- Training: 2000 samples, 30 epochs

---

## 📋 GPU Experiment Priority List

### Priority 1: HyenaDNA Integration ⭐

**Why**: Pre-trained DNA foundation model should significantly improve performance over random CNN initialization.

```python
from meta_spliceai.splice_engine.meta_layer.models import HyenaDNADeltaPredictor

# RTX 4090 (24GB) - Use small model
model = HyenaDNADeltaPredictor(
    model_name='hyenadna-small-32k',
    hidden_dim=256,
    freeze_encoder=True,  # Start frozen, then fine-tune
    output_dim=2
).to('cuda')

# A40 (48GB) - Can use medium model
model = HyenaDNADeltaPredictor(
    model_name='hyenadna-medium-160k',
    hidden_dim=256,
    freeze_encoder=True,
    output_dim=2
).to('cuda')

# A100 (80GB) - Can fine-tune encoder
model = HyenaDNADeltaPredictor(
    model_name='hyenadna-large-1m',
    hidden_dim=256,
    freeze_encoder=False,  # Fine-tune!
    output_dim=2
).to('cuda')
```

**Required packages**:
```bash
pip install transformers einops
pip install flash-attn --no-build-isolation  # Optional, for A100
```

**Expected improvement**: From r=0.41 → target r=0.60+

---

### Priority 2: Full SpliceVarDB Dataset

**Why**: We've only trained on 2000 samples. Full dataset has ~50,000 variants.

```python
from meta_spliceai.splice_engine.meta_layer.data.splicevardb_loader import load_splicevardb

# Load all data (not just 2000)
loader = load_splicevardb(genome_build='GRCh38')
train_variants, test_variants = loader.get_train_test_split(
    test_chromosomes=['21', '22']
)

# Full training (~40K samples after filtering)
print(f"Training samples: {len(train_variants)}")  # Should be ~40K
```

**Training config for full dataset**:
```python
config = {
    'batch_size': 64,        # GPU can handle larger batches
    'epochs': 50,            # More epochs for larger dataset
    'lr': 1e-4,              # Lower LR for stability
    'weight_decay': 0.01,
    'warmup_epochs': 5,
}
```

---

### Priority 3: Longer Context Windows

**Why**: Current 101nt context may miss distant splice effects. Try 501nt or 1001nt.

```python
# Longer context experiment
model = SimpleCNNDeltaPredictor(
    input_length=1001,      # Was 101
    hidden_dim=256,         # Scale up
    n_layers=8,             # More layers for larger input
    kernel_size=15,
    dropout=0.1
)
```

**Memory requirements**:
| Context | M1 Mac | RTX 4090 | A40 |
|---------|--------|----------|-----|
| 101nt | ✅ | ✅ | ✅ |
| 501nt | ⚠️ OOM | ✅ | ✅ |
| 1001nt | ❌ | ⚠️ batch=16 | ✅ |

---

### Priority 4: ValidatedDeltaPredictor with HyenaDNA

**Why**: Combine best architecture (ValidatedDelta) with best encoder (HyenaDNA).

```python
from meta_spliceai.splice_engine.meta_layer.models import create_validated_delta_predictor

# Create with HyenaDNA encoder
model = create_validated_delta_predictor({
    'model_type': 'validated_delta_with_hyenadna',
    'hyenadna_model': 'small',
    'hidden_dim': 256,
    'freeze_encoder': True,
    'n_layers': 6
})
```

---

### Priority 5: Multi-Step Framework Steps 2-4

**Why**: Binary classification (Step 1) achieved AUC=0.61. Need to test remaining steps.

```python
from meta_spliceai.splice_engine.meta_layer.models import (
    EffectTypeClassifier,
    UnifiedSpliceClassifier
)

# Step 2: Effect Type (gain/loss, donor/acceptor)
effect_model = EffectTypeClassifier(
    hidden_dim=128,
    n_classes=4  # DG, DL, AG, AL
)

# Multi-task: Binary + Effect + Localization
unified_model = UnifiedSpliceClassifier(
    hidden_dim=256,
    use_hyenadna=True
)
```

---

## 🔧 Model Classes Reference

### Available in `meta_layer.models`

| Model | Purpose | GPU Required? |
|-------|---------|---------------|
| `SimpleCNNDeltaPredictor` | Gated CNN delta (Approach A) | No (but faster) |
| `ValidatedDeltaPredictor` | Ground-truth validated delta | No (but faster) |
| `HyenaDNADeltaPredictor` | HyenaDNA encoder + delta | **Yes** |
| `SpliceInducingClassifier` | Binary classification | No |
| `EffectTypeClassifier` | Effect type (4-class) | No |
| `UnifiedSpliceClassifier` | Multi-task unified | Recommended |

### Import Example

```python
from meta_spliceai.splice_engine.meta_layer.models import (
    SimpleCNNDeltaPredictor,
    ValidatedDeltaPredictor,
    HyenaDNADeltaPredictor,
    SpliceInducingClassifier,
    EffectTypeClassifier,
    create_validated_delta_predictor
)
```

---

## 📊 Baseline Results (M1 Mac)

These are the benchmarks to beat on GPU:

| Experiment | Samples | Correlation | Detection | Notes |
|------------|---------|-------------|-----------|-------|
| Gated CNN (Approach A) | 2000 | r=0.36 | 0% | Baseline |
| + Quantile Loss | 2000 | r=0.38 | 20% | Better calibration |
| **ValidatedDelta** | 2000 | **r=0.41** | 18.7% | **Current best** |
| Binary Classifier | 2000 | AUC=0.61 | F1=0.53 | Step 1 |

### Target Metrics on GPU

| Experiment | Target Correlation | Target Detection |
|------------|-------------------|------------------|
| HyenaDNA + ValidatedDelta | r > 0.55 | > 30% |
| Full Dataset (50K) | r > 0.50 | > 25% |
| Longer Context (1001nt) | r > 0.45 | > 22% |
| Multi-Step (all steps) | N/A | F1 > 0.7 |

---

## 🚀 Quick Start on RunPods

### 0. Complete Setup First!

Before training, follow the complete setup guide:

**See**: [../setup/RUNPODS_COMPLETE_SETUP.md](../setup/RUNPODS_COMPLETE_SETUP.md) for:
- SSH key setup
- Environment installation
- Data transfer
- GitHub authentication (for pushing results)

**Data Transfer Only**: [DATA_TRANSFER_GUIDE.md](./DATA_TRANSFER_GUIDE.md)

**Minimum required** (~10 MB to transfer, FASTA can be downloaded on RunPods):
- `splicevardb.download.tsv` (6.8 MB) - SCP from local
- `Homo_sapiens.GRCh38.dna.primary_assembly.fa` (2.9 GB) - Download from Ensembl

### 1. Setup Environment

```bash
# Clone and install
cd /workspace
git clone https://github.com/pleiadian53/meta-spliceai.git
cd meta-spliceai

# Create environment
mamba env create -f environment-runpods-minimal.yml
mamba activate metaspliceai

# Install CUDA PyTorch
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install HyenaDNA dependencies
pip install transformers einops

# Install project
pip install -e .
```

### 2. Run HyenaDNA Experiment

```python
#!/usr/bin/env python
"""GPU training script for HyenaDNA delta prediction."""

import torch
import logging
from pathlib import Path

# Setup
logging.basicConfig(level=logging.INFO)
device = torch.device('cuda')

# Import meta_layer components
from meta_spliceai.splice_engine.meta_layer.models import HyenaDNADeltaPredictor
from meta_spliceai.splice_engine.meta_layer.data.splicevardb_loader import load_splicevardb
from meta_spliceai.splice_engine.base_models import load_base_model_ensemble

# Load base model for generating targets
base_models, metadata = load_base_model_ensemble('openspliceai', context=10000, device='cuda')

# Load data
loader = load_splicevardb(genome_build='GRCh38')
train_variants, test_variants = loader.get_train_test_split(test_chromosomes=['21', '22'])

# Limit for first experiment
train_variants = train_variants.head(10000)
print(f"Training on {len(train_variants)} variants")

# Create model
model = HyenaDNADeltaPredictor(
    model_name='hyenadna-small-32k',
    hidden_dim=256,
    freeze_encoder=True
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop here...
# (See meta_layer/tests/test_validated_delta_experiments.py for full example)
```

### 3. Run Full Dataset Experiment

```bash
# Use existing test script with more data
cd /workspace/meta-spliceai/meta_spliceai/splice_engine/meta_layer/tests

# Start in tmux for persistence
tmux new -s training

# Run with larger dataset
python test_validated_delta_experiments.py \
    --samples 10000 \
    --epochs 50 \
    --batch-size 64 \
    --hidden-dim 256

# Detach: Ctrl+B, then D
```

---

## 📁 Important Paths

### Input Data
```
data/mane/GRCh38/openspliceai_eval/meta_models/     # Artifacts
data/splicevardb/                                    # SpliceVarDB data
```

### Output (use these for saving)
```
models/checkpoints/                                  # Model weights
results/                                            # Evaluation results
logs/                                               # Training logs
```

### Documentation
```
meta_spliceai/splice_engine/meta_layer/docs/
├── experiments/           # Experiment results
├── methods/              # Methodology
│   ├── ROADMAP.md        # Overall roadmap
│   └── GPU_REQUIREMENTS.md
└── wishlist/             # Future experiments
```

---

## 🧪 Experiment Tracking

When running experiments, save results with clear naming:

```python
# Naming convention
experiment_name = f"hyenadna_{model_name}_samples{n_samples}_ep{epochs}_{timestamp}"

# Save checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'best_correlation': best_r,
    'config': config
}, f'models/checkpoints/{experiment_name}.pt')

# Save results
results = {
    'experiment': experiment_name,
    'correlation': correlation,
    'detection_rate': detection_rate,
    'training_time': elapsed_time,
    'gpu': torch.cuda.get_device_name(0)
}
with open(f'results/{experiment_name}.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## ⚠️ Important Notes

### 1. Don't Overwrite Production Artifacts
Use `MetaLayerPathManager` for path management:
```python
from meta_spliceai.splice_engine.meta_layer.core.path_manager import MetaLayerPathManager
pm = MetaLayerPathManager(base_model='openspliceai')
output_dir = pm.get_output_write_dir()  # Timestamped, safe
```

### 2. Monitor GPU Memory
```bash
watch -n 1 nvidia-smi
```

### 3. Use tmux for Long Training
```bash
tmux new -s training
# ... run training ...
# Ctrl+B, D to detach
tmux attach -t training  # to reconnect
```

### 4. Save Checkpoints Frequently
```python
# Save every 5 epochs
if epoch % 5 == 0:
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')
```

---

## 📈 What Has Worked (Don't Repeat Failures)

### ✅ Things That Worked
1. **Gated residual blocks** - Key architectural enabler
2. **Dilated convolutions** - Larger receptive field
3. **LayerNorm** (not BatchNorm)
4. **Quantile loss (τ=0.9)** - Better calibration
5. **SpliceVarDB-validated targets** - Ground truth filtering
6. **Balanced training** - 50% splice-altering variants

### ❌ Things That Didn't Work
1. Simple scaling (overfits)
2. Temperature scaling (no improvement)
3. Multi-task (classification + regression) - task interference
4. MSE loss alone - conservative predictions
5. Canonical classification for variant detection - task mismatch

---

## 📚 Related Documentation

- **ROADMAP.md**: Overall methodology roadmap
- **GPU_REQUIREMENTS.md**: Detailed compute requirements
- **PAIRED_DELTA_PREDICTION.md**: Siamese architecture details (deprecated)
- **VALIDATED_DELTA_PREDICTION.md**: Validated delta approach (recommended)
- **MULTI_STEP_FRAMEWORK.md**: Decomposed problem formulation

---

## 🎯 Summary for AI Agent

**Where we are**: ValidatedDeltaPredictor with Gated CNN achieves r=0.41 on M1 Mac.

**What to do on GPU**:
1. Test HyenaDNA encoder (Priority 1)
2. Scale to full SpliceVarDB dataset (Priority 2)
3. Try longer context windows (Priority 3)
4. Combine ValidatedDelta + HyenaDNA (Priority 4)

**Target**: Improve correlation from r=0.41 to r>0.55 with HyenaDNA.

**Files to modify**: 
- `meta_layer/tests/test_validated_delta_experiments.py` (add GPU experiments)
- `meta_layer/models/hyenadna_delta_predictor.py` (HyenaDNA integration)

---

*This guide enables AI agents to continue GPU-intensive experiments without needing access to dev/ directory.*


