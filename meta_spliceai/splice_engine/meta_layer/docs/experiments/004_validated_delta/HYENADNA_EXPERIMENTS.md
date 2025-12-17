# HyenaDNA Experiments for Validated Delta Prediction

**Date**: December 17, 2025  
**Platform**: RunPods (NVIDIA A40, 47.6 GB VRAM)  
**Status**: 🔄 In Progress

---

## Overview

This document tracks experiments using HyenaDNA pre-trained DNA language models for splice delta prediction. The goal is to determine whether pre-trained genomic representations improve upon task-specific CNNs.

---

## Experiments

### 1. HyenaDNA-small Frozen ✅ COMPLETE

**Hypothesis**: Pre-trained DNA representations should capture biological grammar that helps splice prediction.

**Config**:
| Parameter | Value |
|-----------|-------|
| Model | `hyenadna-small-32k` |
| Strategy | **Frozen encoder** (transfer learning) |
| Hidden dim | 256 |
| Training samples | 22,132 (balanced) |
| Test samples | 725 |
| Epochs | 18 (early stopped) |
| Batch size | 32 |

**Results**:
| Metric | HyenaDNA Frozen | CNN Baseline | Difference |
|--------|-----------------|--------------|------------|
| Correlation | r = 0.484 | r = 0.609 | **-26%** ❌ |
| ROC-AUC | 0.562 | 0.585 | -4% |
| PR-AUC | 0.692 | 0.702 | -1% |
| Time | 36 min | 48 min | Faster |

**Conclusion**: ❌ Frozen HyenaDNA underperformed simple CNN.

---

### 2. HyenaDNA-medium Fine-tuned (Last 2 Layers) 🔄 PENDING

**Hypothesis**: Unfreezing the last few layers allows the model to adapt pre-trained representations to splice-specific patterns.

**Fine-tuning Strategy**:
```
┌─────────────────────────────────────────────────────────────┐
│                    HyenaDNA-medium                          │
├─────────────────────────────────────────────────────────────┤
│ Embedding Layer           ← FROZEN                          │
│ Layer 1                   ← FROZEN                          │
│ Layer 2                   ← FROZEN                          │
│ ...                       ← FROZEN                          │
│ Layer N-2                 ← FROZEN                          │
│ Layer N-1                 ← TRAINABLE (lr = 5e-6)           │
│ Layer N                   ← TRAINABLE (lr = 5e-6)           │
│ Layer Norm                ← TRAINABLE (lr = 5e-6)           │
├─────────────────────────────────────────────────────────────┤
│ Projection (256 → hidden) ← TRAINABLE (lr = 5e-5)           │
│ Variant Embed             ← TRAINABLE (lr = 5e-5)           │
│ Delta Head                ← TRAINABLE (lr = 5e-5)           │
└─────────────────────────────────────────────────────────────┘
```

**Key Techniques**:
1. **Gradual Unfreezing**: Only last N layers trainable
2. **Discriminative Learning Rates**: Encoder gets 10x lower LR
3. **Early Stopping**: Prevent overfitting during fine-tuning
4. **Smaller Batch Size**: More gradient updates per epoch

**Config**:
| Parameter | Value |
|-----------|-------|
| Model | `hyenadna-medium-160k` |
| Strategy | **Unfreeze last 2 layers** |
| Encoder LR | 5e-6 (0.1x base) |
| Head LR | 5e-5 |
| Training samples | 25,000 |
| Batch size | 16 |
| Gradient accumulation | 4 |
| Patience | 10 epochs |

**Expected Results**:
| Metric | Target |
|--------|--------|
| Correlation | r > 0.55 (better than CNN's 0.609 would be ideal) |
| ROC-AUC | > 0.60 |
| PR-AUC | > 0.72 |

---

### 3. HyenaDNA-medium Fine-tuned (Deep - Last 4 Layers) 🔄 PENDING

**Hypothesis**: More layers allow deeper adaptation but risk overfitting.

**Config**:
| Parameter | Value |
|-----------|-------|
| Model | `hyenadna-medium-160k` |
| Strategy | **Unfreeze last 4 layers** |
| Encoder LR | 2.5e-6 (0.05x base) |
| Head LR | 3e-5 |
| Batch size | 8 |
| Gradient accumulation | 8 |
| Weight decay | 0.02 (more regularization) |

---

## Architecture Details

### HyenaDNA Models

| Model | Layers | Embed Dim | Context | Parameters |
|-------|--------|-----------|---------|------------|
| hyenadna-small-32k | 4 | 256 | 32K | ~7M |
| hyenadna-medium-160k | 8 | 256 | 160K | ~25M |
| hyenadna-large-1m | 12 | 512 | 1M | ~100M |

### Fine-tuning Architecture

```
Input: alt_seq [B, 4, 501] (one-hot encoded)
           ↓
    Convert to token IDs [B, 501]
           ↓
    ┌─────────────────────────────┐
    │      HyenaDNA Encoder       │
    │  (partial layers trainable) │
    └─────────────────────────────┘
           ↓
    Hidden states [B, 501, 256]
           ↓
    Global Average Pool → [B, 256]
           ↓
    Projection → [B, hidden_dim]
           ↓
    ┌─────────────────────────────┐
    │      Variant Embedding      │
    │  ref_base[4] + alt_base[4]  │
    │         → [B, hidden_dim]   │
    └─────────────────────────────┘
           ↓
    Concatenate → [B, 2 * hidden_dim]
           ↓
    ┌─────────────────────────────┐
    │       Delta Head            │
    │    3-layer MLP → [B, 3]     │
    └─────────────────────────────┘
           ↓
Output: [Δ_donor, Δ_acceptor, Δ_neither]
```

---

## Run Commands

```bash
# HyenaDNA fine-tuning (last 2 layers)
tmux new-session -d -s hyenadna_ft
tmux send-keys -t hyenadna_ft "cd /workspace/meta-spliceai && \
  source /workspace/miniforge3/etc/profile.d/conda.sh && \
  conda activate metaspliceai && \
  export META_SPLICEAI_ROOT=/workspace/meta-spliceai && \
  export SS_DATA_ROOT=/workspace/meta-spliceai/data && \
  export SS_FASTA_PATH=/workspace/meta-spliceai/data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa && \
  python -m meta_spliceai.splice_engine.meta_layer.tests.test_gpu_validated_delta_experiments \
    --exp hyenadna_finetune --device cuda 2>&1 | tee logs/gpu_exp_hyenadna_finetune.log" Enter

# Deep fine-tuning (last 4 layers)
python -m meta_spliceai.splice_engine.meta_layer.tests.test_gpu_validated_delta_experiments \
    --exp hyenadna_finetune_deep --device cuda 2>&1 | tee logs/gpu_exp_hyenadna_finetune_deep.log
```

---

## Monitoring

```bash
# Check progress
tail -f /workspace/meta-spliceai/logs/gpu_exp_hyenadna_finetune.log

# Attach to tmux
tmux attach -t hyenadna_ft

# GPU memory
nvidia-smi
```

---

## Results Summary

| Experiment | Unfreeze | Correlation | ROC-AUC | PR-AUC | Status |
|------------|----------|-------------|---------|--------|--------|
| HyenaDNA-small frozen | 0 | 0.484 | 0.562 | 0.692 | ✅ Done |
| **CNN baseline** | N/A | **0.609** | **0.585** | **0.702** | ✅ Best |
| HyenaDNA-medium ft-2 | 2 | - | - | - | 🔄 Pending |
| HyenaDNA-medium ft-4 | 4 | - | - | - | 🔄 Pending |

---

## Key Questions

1. **Does fine-tuning help?** - Can we beat the CNN baseline with fine-tuned HyenaDNA?
2. **How many layers to unfreeze?** - Is 2 enough or do we need more?
3. **Optimal learning rate ratio?** - Is 10x difference (0.1 mult) correct?
4. **Memory constraints?** - Can A40 handle larger batch with fine-tuning?

---

## Theoretical Considerations

### Why Frozen HyenaDNA Underperformed

1. **Feature mismatch**: HyenaDNA learns general DNA patterns, not splice-specific deltas
2. **Context length**: Trained on 32K but fed 501bp → may miss long-range patterns
3. **Task mismatch**: Next-token prediction ≠ delta regression
4. **Pooling limitation**: Global average pool loses positional information

### Why Fine-tuning Might Help

1. **Adapts representations**: Last layers can learn splice-specific features
2. **Preserves low-level patterns**: Early layers keep general DNA understanding
3. **Discriminative LR**: Prevents catastrophic forgetting
4. **More capacity**: Medium model has 8 layers vs 4 in small

### Why Fine-tuning Might Not Help

1. **Data size**: 25K samples may not be enough to fine-tune a 25M param model
2. **Overfitting risk**: More trainable params = more risk
3. **Task is too specific**: Delta prediction may need specialized architecture

---

*Results will be updated as experiments complete.*

