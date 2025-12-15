# Meta-Layer Architecture

**Last Updated**: December 2025  
**Status**: Design Document

---

## Overview

The Meta-Layer is a **multimodal deep learning system** that recalibrates base model splice site predictions by combining:

1. **Sequence embeddings** from DNA language models (HyenaDNA)
2. **Score embeddings** from base model features
3. **Cross-modal fusion** to leverage both modalities

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           META-LAYER SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        DATA LAYER                                      │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  Base Layer Artifacts          SpliceVarDB                             │ │
│  │  ─────────────────────         ───────────                             │ │
│  │  • analysis_sequences_*.tsv    • 50K validated variants                │ │
│  │  • 501nt context windows       • Splice-altering classifications      │ │
│  │  • 50+ derived features        • hg19/hg38 coordinates                 │ │
│  │  • GTF-based labels            • Sample weights                        │ │
│  │                                                                        │ │
│  │                    ↓                           ↓                       │ │
│  │                    └───────────┬───────────────┘                       │ │
│  │                                ↓                                       │ │
│  │                    ┌───────────────────────┐                           │ │
│  │                    │  ArtifactLoader       │                           │ │
│  │                    │  (base-model-agnostic)│                           │ │
│  │                    └───────────────────────┘                           │ │
│  │                                ↓                                       │ │
│  │                    ┌───────────────────────┐                           │ │
│  │                    │  MetaLayerDataset     │                           │ │
│  │                    │  (PyTorch Dataset)    │                           │ │
│  │                    └───────────────────────┘                           │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                   ↓                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        MODEL LAYER                                     │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  ┌─────────────────────┐     ┌─────────────────────┐                  │ │
│  │  │  Sequence Encoder   │     │  Score Encoder      │                  │ │
│  │  ├─────────────────────┤     ├─────────────────────┤                  │ │
│  │  │  Options:           │     │  MLP Network:       │                  │ │
│  │  │  • HyenaDNA (SSM)   │     │  [50+ feat] → [D]   │                  │ │
│  │  │  • DNABERT-2        │     │  LayerNorm + GELU   │                  │ │
│  │  │  • CNN (lightweight)│     │                     │                  │ │
│  │  │                     │     │                     │                  │ │
│  │  │  Output: [B, D]     │     │  Output: [B, D]     │                  │ │
│  │  └─────────────────────┘     └─────────────────────┘                  │ │
│  │           ↓                           ↓                               │ │
│  │           └───────────┬───────────────┘                               │ │
│  │                       ↓                                               │ │
│  │           ┌───────────────────────┐                                   │ │
│  │           │    Fusion Layer       │                                   │ │
│  │           ├───────────────────────┤                                   │ │
│  │           │  Options:             │                                   │ │
│  │           │  • Cross-attention    │                                   │ │
│  │           │  • Concatenation      │                                   │ │
│  │           │  • Gated fusion       │                                   │ │
│  │           │                       │                                   │ │
│  │           │  Output: [B, D*2]     │                                   │ │
│  │           └───────────────────────┘                                   │ │
│  │                       ↓                                               │ │
│  │           ┌───────────────────────┐                                   │ │
│  │           │   Classification Head │                                   │ │
│  │           ├───────────────────────┤                                   │ │
│  │           │  MLP + Softmax        │                                   │ │
│  │           │                       │                                   │ │
│  │           │  Output: [B, 3]       │                                   │ │
│  │           │  P(donor), P(acc),    │                                   │ │
│  │           │  P(neither)           │                                   │ │
│  │           └───────────────────────┘                                   │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                   ↓                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        INFERENCE LAYER                                 │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  Recalibrated Scores → Splice Sites → Donor-Acceptor Pairs → Exons    │ │
│  │                                                                        │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │ │
│  │  │ SpliceSiteCaller │→│ ExonPredictor   │→│ DecoyFilter     │        │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘        │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Principles

### 1. Base-Model-Agnostic

The meta-layer works with **any base model** through automatic artifact routing:

```python
# Works with any base model - just change the parameter
config = MetaLayerConfig(base_model='openspliceai')  # or 'spliceai', 'newmodel'

# Automatic path resolution
config.artifacts_dir  # → data/mane/GRCh38/openspliceai_eval/meta_models
config.genome_build   # → GRCh38
```

### 2. Multimodal Fusion

Combines two complementary modalities:

| Modality | Information | Encoder |
|----------|-------------|---------|
| **Sequence** | Long-range dependencies, motifs | HyenaDNA (SSM) |
| **Scores** | Base model knowledge, local patterns | MLP |

### 3. Consistent Output Format

Output matches base model format for seamless integration:

```python
# Base model output
base_output = {'donor': 0.7, 'acceptor': 0.2, 'neither': 0.1}

# Meta-layer output (same format!)
meta_output = {'donor': 0.85, 'acceptor': 0.1, 'neither': 0.05}
```

### 4. Scalable Compute

Supports both local development (M1 Mac) and cloud training (RunPods):

| Environment | Encoder | Batch Size | Training Time |
|-------------|---------|------------|---------------|
| M1 Mac (16GB) | CNN | 32 | ~2 hours (subset) |
| RunPods GPU | HyenaDNA | 128 | ~6 hours (full) |

---

## Component Details

### Sequence Encoder

Converts 501nt DNA sequences to embeddings:

```python
class SequenceEncoderFactory:
    ENCODERS = {
        'hyenadna': HyenaDNAEncoder,  # Best quality, requires GPU
        'dnabert2': DNABERT2Encoder,  # Good quality, moderate GPU
        'cnn': CNNEncoder,            # Lightweight, CPU-friendly
        'none': IdentityEncoder       # Skip sequence (baseline)
    }
```

### Score Encoder

Converts 50+ numeric features to embeddings:

```python
class ScoreEncoder(nn.Module):
    def __init__(self, num_features=50, hidden_dim=256):
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
```

### Fusion Layer

Combines sequence and score embeddings:

```python
class CrossAttentionFusion(nn.Module):
    """
    Sequence attends to score features.
    Allows the model to learn which score features are 
    relevant given the sequence context.
    """
    def __init__(self, hidden_dim=256, num_heads=8):
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads)
```

---

## Training Pipeline

```
1. Data Loading
   └── ArtifactLoader.load_analysis_sequences()
       └── Returns: sequences, features, labels

2. Batch Creation
   └── MetaLayerDataset + DataLoader
       └── Tokenizes sequences
       └── Normalizes features
       └── Applies sample weights

3. Forward Pass
   └── sequence → SequenceEncoder → seq_emb
   └── features → ScoreEncoder → score_emb
   └── (seq_emb, score_emb) → Fusion → combined_emb
   └── combined_emb → Classifier → logits

4. Loss Computation
   └── CrossEntropyLoss(logits, labels, weight=sample_weights)

5. Optimization
   └── AdamW with learning rate scheduling
   └── Gradient accumulation for large batches
```

---

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **PR-AUC** | Area under Precision-Recall curve | > 0.95 |
| **Top-k Accuracy** | % of top-k predictions that are correct | > 0.90 (k=100) |
| **AP** | Average Precision | > 0.90 |
| **ECE** | Expected Calibration Error | < 0.05 |

---

## Future Extensions

1. **Transcript-aware predictions**: Consider multiple transcripts per gene
2. **Tissue-specific splicing**: Incorporate tissue context
3. **Variant effect prediction**: Direct ΔScores for variants
4. **Ensemble meta-layer**: Combine multiple meta-layers

---

## Related Documentation

- [LABELING_STRATEGY.md](LABELING_STRATEGY.md) - Label creation
- [ALTERNATIVE_SPLICING_PIPELINE.md](ALTERNATIVE_SPLICING_PIPELINE.md) - Scores to exons
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Training instructions

---

*Last Updated: December 2025*

