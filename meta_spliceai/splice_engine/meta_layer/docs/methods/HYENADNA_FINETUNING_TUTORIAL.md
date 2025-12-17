# HyenaDNA Fine-tuning for Validated Delta Prediction

**A Comprehensive Tutorial**

*Date: December 2025*  
*Author: Meta-SpliceAI Team*

---

## Table of Contents

1. [Introduction](#introduction)
2. [Background](#background)
3. [Architecture Overview](#architecture-overview)
4. [The Validated Delta Strategy](#the-validated-delta-strategy)
5. [Fine-tuning Methods](#fine-tuning-methods)
6. [Implementation Guide](#implementation-guide)
7. [Training Pipeline](#training-pipeline)
8. [Hyperparameter Recommendations](#hyperparameter-recommendations)
9. [Experiments and Results](#experiments-and-results)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## Introduction

This tutorial covers how to fine-tune HyenaDNA, a pre-trained DNA language model, for predicting splice site probability changes caused by genetic variants. We use the **validated delta target strategy**, which leverages ground truth labels from SpliceVarDB to create high-quality training targets.

### Goals

- Understand why and how to fine-tune HyenaDNA for splice prediction
- Implement gradual unfreezing and discriminative learning rates
- Train models that outperform both frozen encoders and from-scratch training

### Prerequisites

- Understanding of PyTorch and transformer models
- Familiarity with splice site biology
- Access to GPU (A40 or better recommended)

---

## Background

### Why HyenaDNA?

HyenaDNA is a pre-trained DNA language model with several advantages:

| Feature | Benefit |
|---------|---------|
| **Pre-trained on DNA** | Understands biological grammar (codons, motifs, etc.) |
| **Long context** | Can process up to 1M bp (vs 10K for most models) |
| **O(n) complexity** | Efficient attention mechanism (Hyena operator) |
| **Multiple sizes** | Small (7M) to Large (100M) parameters |

### Available Models

| Model | Layers | Dim | Context | Parameters | Use Case |
|-------|--------|-----|---------|------------|----------|
| `hyenadna-tiny-1k` | 2 | 128 | 1K | ~2M | Quick experiments |
| `hyenadna-small-32k` | 4 | 256 | 32K | ~7M | Transfer learning |
| `hyenadna-medium-160k` | 8 | 256 | 160K | ~25M | **Fine-tuning (recommended)** |
| `hyenadna-large-1m` | 12 | 512 | 1M | ~100M | Large-scale fine-tuning |

### The Challenge

Despite being pre-trained on DNA, frozen HyenaDNA underperformed a simple CNN on our task:

| Model | Correlation | ROC-AUC |
|-------|-------------|---------|
| CNN (from scratch) | **0.609** | **0.585** |
| HyenaDNA-small (frozen) | 0.484 | 0.562 |

**Why?** The pre-training task (next-token prediction) doesn't directly teach splice-specific patterns. Fine-tuning allows the model to adapt its representations.

---

## Architecture Overview

### Single-Pass Design

Unlike paired models that need both ref and alt sequences, our model uses only the alternate sequence:

```
                    ┌─────────────────────────────────────────┐
                    │      HyenaDNA Validated Delta           │
                    └─────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
        ▼                               ▼                               ▼
  ┌───────────┐                  ┌───────────┐                  ┌───────────┐
  │  alt_seq  │                  │ ref_base  │                  │ alt_base  │
  │ [B, 4, L] │                  │  [B, 4]   │                  │  [B, 4]   │
  └─────┬─────┘                  └─────┬─────┘                  └─────┬─────┘
        │                               │                               │
        │ one-hot to tokens             └───────────┬───────────────────┘
        │                                           │
        ▼                                           ▼
  ┌─────────────────────┐                    ┌─────────────┐
  │   HyenaDNA Encoder  │                    │   Variant   │
  │   (partial freeze)  │                    │   Embed     │
  │                     │                    │  [8 → H]    │
  │ Layer 1 [FROZEN]    │                    └──────┬──────┘
  │ Layer 2 [FROZEN]    │                           │
  │   ...               │                           │
  │ Layer N-1 [TRAIN]   │                           │
  │ Layer N   [TRAIN]   │                           │
  └──────────┬──────────┘                           │
             │                                      │
             ▼                                      │
       Global Pool                                  │
       [B, H_enc]                                   │
             │                                      │
             ▼                                      │
       ┌───────────┐                                │
       │ Projection│                                │
       │ [H_enc→H] │                                │
       └─────┬─────┘                                │
             │                                      │
             └────────────────┬─────────────────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │   Concat    │
                       │  [B, 2*H]   │
                       └──────┬──────┘
                              │
                              ▼
                       ┌─────────────┐
                       │  Delta Head │
                       │  [2*H → 3]  │
                       └──────┬──────┘
                              │
                              ▼
                       ┌─────────────┐
                       │    Output   │
                       │   [B, 3]    │
                       │ [Δd,Δa,Δn]  │
                       └─────────────┘
```

### Component Details

#### 1. HyenaDNA Encoder

Converts DNA tokens to contextual embeddings:

```python
# Input: one-hot [B, 4, L] → token IDs [B, L]
token_ids = seq.argmax(dim=1)

# HyenaDNA forward pass
outputs = encoder(token_ids, output_hidden_states=True)
hidden = outputs.hidden_states[-1]  # [B, L, H_enc]

# Global average pooling
features = hidden.mean(dim=1)  # [B, H_enc]
```

#### 2. Projection Layer

Maps encoder dimension to hidden dimension with normalization:

```python
self.proj = nn.Sequential(
    nn.Linear(encoder_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout)
)
```

#### 3. Variant Embedding

Encodes ref/alt base identity:

```python
# Concatenate ref and alt one-hot encodings
var_info = torch.cat([ref_base, alt_base], dim=-1)  # [B, 8]

self.variant_embed = nn.Sequential(
    nn.Linear(8, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim)
)
```

#### 4. Delta Head

Predicts the 3-channel delta output:

```python
self.delta_head = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.GELU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.GELU(),
    nn.Linear(hidden_dim // 2, 3)  # [Δ_donor, Δ_acceptor, Δ_neither]
)
```

---

## The Validated Delta Strategy

### The Problem with Naive Targets

Traditional approach uses base model predictions as targets:

```
Target = SpliceAI(alt_seq) - SpliceAI(ref_seq)
```

**Issue**: If a variant is NOT splice-altering but SpliceAI predicts a delta anyway, we're training on incorrect labels!

### Our Solution: Validated Targets

Use SpliceVarDB ground truth to filter and correct targets:

```python
if splicevardb_classification == 'Splice-altering':
    # Trust base model - SpliceVarDB confirms this variant affects splicing
    target = SpliceAI(alt_seq) - SpliceAI(ref_seq)
    
elif splicevardb_classification == 'Normal':
    # Override base model - we KNOW there's no effect
    target = [0.0, 0.0, 0.0]
    
else:  # Low-frequency, Conflicting
    # Skip - uncertain labels
    continue
```

### Target Format

The target is a **continuous 3D vector**:

```
target = [Δ_donor, Δ_acceptor, Δ_neither]
```

| Component | Meaning | Range |
|-----------|---------|-------|
| Δ_donor | Change in donor site probability | ~[-1, +1] |
| Δ_acceptor | Change in acceptor site probability | ~[-1, +1] |
| Δ_neither | Change in non-splice probability | ~[-1, +1] |

**Interpretation Examples**:

| Δ_donor | Δ_acceptor | Meaning |
|---------|------------|---------|
| +0.35 | ~0 | Donor gain (new/enhanced donor site) |
| -0.40 | ~0 | Donor loss (weakened/destroyed) |
| ~0 | +0.42 | Acceptor gain |
| ~0 | -0.38 | Acceptor loss |
| 0 | 0 | No effect (Normal variant) |

---

## Fine-tuning Methods

### Method 1: Frozen Encoder (Transfer Learning)

All encoder weights frozen, only train projection and head.

```python
model = create_hyenadna_model(
    'hyenadna-small-32k',
    freeze_encoder=True
)
```

**Pros**: Fast, stable, no overfitting risk  
**Cons**: Limited adaptation capacity

### Method 2: Gradual Unfreezing

Unfreeze the last N layers while keeping early layers frozen.

```python
model = create_hyenadna_model(
    'hyenadna-medium-160k',
    freeze_encoder=False,
    unfreeze_last_n=2  # Unfreeze last 2 layers
)
```

**Rationale**:
- Early layers learn low-level patterns (base composition, motifs)
- Later layers learn higher-level patterns (task-specific)
- Unfreezing later layers allows task adaptation while preserving general knowledge

### Method 3: Discriminative Learning Rates

Different learning rates for different parts:

```python
param_groups = model.get_param_groups(
    base_lr=5e-5,       # For new layers (head, projection)
    encoder_lr_mult=0.1  # Encoder gets 10x lower LR
)

optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
```

**Why?**
- Pre-trained weights are already good → small updates
- New layers need to learn from scratch → larger updates
- Prevents catastrophic forgetting

### Method 4: Combination (Recommended)

Combine gradual unfreezing with discriminative LR:

```python
model = create_hyenadna_model(
    'hyenadna-medium-160k',
    freeze_encoder=False,
    unfreeze_last_n=2
)

optimizer = model.get_optimizer(
    base_lr=5e-5,
    encoder_lr_mult=0.1,  # Last 2 layers get 5e-6
    weight_decay=0.01
)
```

---

## Implementation Guide

### Installation

```bash
# Create environment
conda create -n metaspliceai python=3.10
conda activate metaspliceai

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install transformers (for HyenaDNA)
pip install transformers

# Install meta-spliceai
cd /path/to/meta-spliceai
pip install -e .
```

### Model Creation

```python
from meta_spliceai.splice_engine.meta_layer.models.hyenadna_validated_delta import (
    create_hyenadna_model,
    create_finetuned_model,
    HyenaDNAConfig
)

# Option 1: Factory function
model = create_hyenadna_model(
    model_name='hyenadna-medium-160k',
    hidden_dim=256,
    freeze_encoder=False,
    unfreeze_last_n=2,
    dropout=0.1
)

# Option 2: Convenience function for fine-tuning
model = create_finetuned_model(unfreeze_last_n=2)

# Option 3: Config object
config = HyenaDNAConfig(
    model_name='hyenadna-medium-160k',
    hidden_dim=256,
    freeze_encoder=False,
    unfreeze_last_n=2
)
model = HyenaDNAValidatedDelta(config)
```

### Forward Pass

```python
# Prepare inputs
alt_seq = torch.tensor(one_hot_seq("ACGT..."), dtype=torch.float32)  # [4, L]
ref_base = torch.tensor([1, 0, 0, 0], dtype=torch.float32)  # A
alt_base = torch.tensor([0, 1, 0, 0], dtype=torch.float32)  # C

# Add batch dimension
alt_seq = alt_seq.unsqueeze(0)  # [1, 4, L]
ref_base = ref_base.unsqueeze(0)  # [1, 4]
alt_base = alt_base.unsqueeze(0)  # [1, 4]

# Predict
delta = model(alt_seq, ref_base, alt_base)  # [1, 3]
print(f"Δ_donor: {delta[0, 0]:.4f}")
print(f"Δ_acceptor: {delta[0, 1]:.4f}")
```

### Optimizer Setup

```python
# For fine-tuning with discriminative LR
optimizer = model.get_optimizer(
    base_lr=5e-5,
    encoder_lr_mult=0.1,
    weight_decay=0.01
)

# For frozen encoder (simpler)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)
```

---

## Training Pipeline

### Using the Training Script

```bash
# Transfer learning (frozen encoder)
python -m meta_spliceai.splice_engine.meta_layer.tests.train_hyenadna_validated_delta \
    --mode frozen \
    --model hyenadna-small-32k

# Fine-tuning last 2 layers (recommended)
python -m meta_spliceai.splice_engine.meta_layer.tests.train_hyenadna_validated_delta \
    --mode finetune \
    --unfreeze 2 \
    --model hyenadna-medium-160k

# Deep fine-tuning (last 4 layers)
python -m meta_spliceai.splice_engine.meta_layer.tests.train_hyenadna_validated_delta \
    --mode finetune \
    --unfreeze 4 \
    --lr 3e-5 \
    --encoder-lr-mult 0.05

# Full fine-tuning (all layers trainable)
python -m meta_spliceai.splice_engine.meta_layer.tests.train_hyenadna_validated_delta \
    --mode full \
    --lr 1e-5
```

### Custom Training Loop

```python
from meta_spliceai.splice_engine.meta_layer.models.hyenadna_validated_delta import (
    create_finetuned_model
)

# Create model
model = create_finetuned_model(unfreeze_last_n=2)
model = model.to('cuda')

# Optimizer with discriminative LR
optimizer = model.get_optimizer(base_lr=5e-5, encoder_lr_mult=0.1)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Mixed precision
scaler = torch.amp.GradScaler('cuda')

# Training loop
for epoch in range(50):
    model.train()
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            pred = model(batch['alt_seq'], batch['ref_base'], batch['alt_base'])
            loss = F.mse_loss(pred, batch['target_delta'])
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    scheduler.step()
    
    # Early stopping logic...
```

### Running in tmux

For long experiments, use tmux:

```bash
# Create session
tmux new-session -d -s hyenadna_train

# Run training
tmux send-keys -t hyenadna_train "cd /workspace/meta-spliceai && \
    source /workspace/miniforge3/etc/profile.d/conda.sh && \
    conda activate metaspliceai && \
    python -m meta_spliceai.splice_engine.meta_layer.tests.train_hyenadna_validated_delta \
        --mode finetune --unfreeze 2 2>&1 | tee logs/hyenadna_finetune.log" Enter

# Monitor
tmux attach -t hyenadna_train

# Or check logs
tail -f logs/hyenadna_finetune.log
```

---

## Hyperparameter Recommendations

### By Model Size

| Model | Batch | LR | Encoder LR | Grad Accum | VRAM |
|-------|-------|-----|------------|------------|------|
| small-32k | 32 | 1e-4 | 1e-5 | 2 | ~12GB |
| medium-160k | 16 | 5e-5 | 5e-6 | 4 | ~24GB |
| large-1m | 8 | 3e-5 | 3e-6 | 8 | ~40GB |

### By Fine-tuning Depth

| Unfreeze Layers | Encoder LR Mult | Weight Decay | Patience |
|-----------------|-----------------|--------------|----------|
| 0 (frozen) | N/A | 0.01 | 7 |
| 2 (shallow) | 0.10 | 0.01 | 10 |
| 4 (deep) | 0.05 | 0.02 | 10 |
| All (full) | 0.02 | 0.05 | 15 |

### General Guidelines

1. **Start with frozen encoder** to establish a baseline
2. **Use early stopping** - fine-tuning is prone to overfitting
3. **Lower LR for deeper fine-tuning** - more layers = more careful updates
4. **Monitor validation loss** - if it diverges from train loss, reduce LR
5. **Gradient accumulation** - use it to achieve larger effective batch sizes

---

## Experiments and Results

### Experiment Matrix (December 2025)

| Experiment | Model | Freeze | Unfreeze | Correlation | ROC-AUC | PR-AUC | Status |
|------------|-------|--------|----------|-------------|---------|--------|--------|
| **Baseline CNN** | Gated CNN | - | - | **0.609** | 0.585 | **0.702** | ✅ **BEST** |
| HyenaDNA Frozen | small-32k | Yes | 0 | 0.484 | 0.562 | 0.692 | ✅ Done |
| HyenaDNA FT-2 | medium-160k | No | 2 | 0.490 | 0.600 | 0.692 | ✅ Done |
| HyenaDNA FT-4 | medium-160k | No | 4 | - | - | - | ❌ Not tested |

### Key Findings (Updated with Actual Results)

1. **Frozen HyenaDNA underperformed CNN** - Pre-training alone isn't enough (r=0.484 vs r=0.609)
2. **Fine-tuning provided minimal improvement** - Only +1.2% correlation, +6.8% ROC-AUC
3. **Task-specific CNN wins** - Simple gated CNN outperforms 25M parameter pre-trained model
4. **Hypothesis validated**: For delta prediction, task-specific architectures beat foundation models

### Why Fine-tuning Didn't Help More

| Factor | Impact |
|--------|--------|
| **Task specificity** | Delta prediction requires learning what SpliceAI cares about, not general DNA patterns |
| **Insufficient adaptation** | 2 layers may not be enough for such a different task |
| **Data size** | 22K samples may be too small to effectively fine-tune a 25M parameter model |
| **Architecture mismatch** | Single-pass from pooled embeddings loses position information |

### Recommendations Based on Results

1. ✅ **Use task-specific CNN** for validated delta prediction (r=0.609)
2. ❌ **Don't use HyenaDNA** for this task without significant modifications
3. 🔄 Consider **multi-task pre-training** on splice prediction specifically
4. 🔄 Consider **meta-recalibration** approach to improve base model upstream

---

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
--batch-size 8

# Increase gradient accumulation
--grad-accum 8

# Use FP16 (default)
model = create_hyenadna_model(..., use_fp16=True)
```

### Training Instability

```python
# Lower learning rate
--lr 1e-5

# Lower encoder LR multiplier
--encoder-lr-mult 0.05

# Increase weight decay
--weight-decay 0.02
```

### Overfitting

```python
# Add more dropout
model = create_hyenadna_model(..., dropout=0.2)

# Reduce epochs / increase patience
--epochs 30 --patience 15

# Freeze more layers
--unfreeze 1
```

### HyenaDNA Not Loading

```python
# Check transformers installation
pip install transformers --upgrade

# Check internet connection (downloads from HuggingFace)

# Use fallback CNN
model = create_hyenadna_model(..., use_fp16=False)  # Sometimes helps
```

---

## References

### Papers

1. **HyenaDNA**: Nguyen et al. "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution" (2023)
   - https://arxiv.org/abs/2306.15794

2. **SpliceAI**: Jaganathan et al. "Predicting Splicing from Primary Sequence with Deep Learning" (2019)
   - https://doi.org/10.1016/j.cell.2018.12.015

3. **ULMFiT** (discriminative fine-tuning): Howard & Ruder (2018)
   - https://arxiv.org/abs/1801.06146

### Code

- **HyenaDNA**: https://github.com/HazyResearch/hyena-dna
- **Meta-SpliceAI**: Internal repository

### Related Methods

- `docs/methods/VALIDATED_DELTA_PREDICTION.md` - Validated delta prediction rationale
- `docs/experiments/004_validated_delta/` - Validated delta experiments

---

## Appendix: Full API Reference

### `HyenaDNAValidatedDelta`

```python
class HyenaDNAValidatedDelta(nn.Module):
    """HyenaDNA-based validated delta predictor."""
    
    def __init__(self, config: HyenaDNAConfig = None):
        """
        Parameters
        ----------
        config : HyenaDNAConfig
            Model configuration
        """
    
    def forward(self, alt_seq, ref_base, alt_base) -> torch.Tensor:
        """
        Predict delta scores.
        
        Parameters
        ----------
        alt_seq : torch.Tensor [B, 4, L]
            One-hot encoded alternate sequence
        ref_base : torch.Tensor [B, 4]
            Reference base one-hot
        alt_base : torch.Tensor [B, 4]
            Alternate base one-hot
        
        Returns
        -------
        torch.Tensor [B, 3]
            Delta scores [Δ_donor, Δ_acceptor, Δ_neither]
        """
    
    def get_param_groups(self, base_lr, encoder_lr_mult) -> List[Dict]:
        """Get parameter groups for discriminative LR."""
    
    def get_optimizer(self, base_lr, encoder_lr_mult, weight_decay) -> Optimizer:
        """Create optimizer with discriminative LR."""
```

### `create_hyenadna_model`

```python
def create_hyenadna_model(
    model_name: str = 'hyenadna-small-32k',
    hidden_dim: int = 256,
    freeze_encoder: bool = True,
    unfreeze_last_n: int = 0,
    dropout: float = 0.1,
    use_fp16: bool = True
) -> HyenaDNAValidatedDelta:
    """
    Factory function for creating HyenaDNA models.
    
    Parameters
    ----------
    model_name : str
        HyenaDNA variant
    hidden_dim : int
        Hidden dimension
    freeze_encoder : bool
        Freeze all encoder weights
    unfreeze_last_n : int
        Layers to unfreeze (if freeze_encoder=False)
    dropout : float
        Dropout probability
    use_fp16 : bool
        Use FP16 for encoder
    """
```

---

*This tutorial is part of the Meta-SpliceAI documentation. For questions, contact the development team.*

