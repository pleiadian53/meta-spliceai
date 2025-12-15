# Meta-Layer: Base-Model-Agnostic Multimodal Meta-Learning

**Status**: 🚧 In Development  
**Version**: 0.1.0  
**Last Updated**: December 2025

---

## Overview

The Meta-Layer is a **multimodal deep learning system** that recalibrates base model splice site predictions to:

1. **Correct FPs/FNs** - Reduce false positives and false negatives from base models
2. **Predict context-dependent splicing** - Account for variant-induced alternative splicing
3. **Maintain consistency** - Output same format as base layer (per-nucleotide probabilities)

### Key Design Principle: Base-Model-Agnostic

Just like the base layer supports any splice prediction model, the meta-layer works with **any base model** via a single parameter:

```python
from meta_spliceai.splice_engine.meta_layer import train_meta_model

# Works with SpliceAI
results = train_meta_model(base_model='spliceai', ...)

# Works with OpenSpliceAI
results = train_meta_model(base_model='openspliceai', ...)

# Works with future models
results = train_meta_model(base_model='newmodel', ...)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         META-LAYER ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  INPUT: Base Layer Artifacts (analysis_sequences_*.tsv)        │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  • 501nt contextual sequences                                   │   │
│  │  • Base model scores (donor, acceptor, neither)                 │   │
│  │  • 50+ derived features                                         │   │
│  │  • Labels (splice_type from GTF annotations)                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SEQUENCE ENCODER (Modality 1)                                  │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  Options: HyenaDNA, DNABERT-2, CNN (lightweight)                │   │
│  │  Output: [B, D] sequence embeddings                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SCORE ENCODER (Modality 2)                                     │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  MLP: [50+ features] → [D] score embeddings                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  FUSION LAYER                                                    │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  Cross-attention or concatenation                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT: Recalibrated probabilities                             │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  P(donor), P(acceptor), P(neither) per nucleotide               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Package Structure

```
meta_layer/
├── __init__.py                 # Package entry point
├── README.md                   # This file
│
├── core/
│   ├── __init__.py
│   ├── config.py               # MetaLayerConfig
│   ├── artifact_loader.py      # Load base layer artifacts
│   └── feature_schema.py       # Standardized feature definitions
│
├── models/
│   ├── __init__.py
│   ├── sequence_encoder.py     # DNA LM wrapper (HyenaDNA, etc.)
│   ├── score_encoder.py        # MLP for score features
│   ├── fusion.py               # Cross-modal fusion
│   ├── meta_splice_model.py    # Main model class
│   └── losses.py               # Custom losses
│
├── data/
│   ├── __init__.py
│   ├── artifact_reader.py      # Read analysis_sequences
│   ├── dataset.py              # PyTorch Dataset
│   ├── dataloader.py           # Efficient batching
│   └── variant_integrator.py   # SpliceVarDB integration
│
├── training/
│   ├── __init__.py
│   ├── trainer.py              # Training loop
│   ├── evaluator.py            # Metrics (PR-AUC, top-k, etc.)
│   └── callbacks.py            # Checkpointing, early stopping
│
├── inference/
│   ├── __init__.py
│   ├── predictor.py            # Inference engine
│   ├── splice_site_caller.py   # Peak detection + thresholding
│   └── exon_predictor.py       # Donor-acceptor pairing
│
├── workflows/
│   ├── __init__.py
│   ├── prepare_training_data.py
│   ├── train_meta_model.py
│   ├── evaluate_meta_model.py
│   └── predict_alternative_splicing.py
│
├── configs/
│   ├── default.yaml
│   ├── hyenadna.yaml
│   └── lightweight.yaml
│
├── cli/
│   └── run_meta_layer.py       # CLI entry point
│
├── docs/                       # Package documentation
│   ├── ARCHITECTURE.md
│   ├── LABELING_STRATEGY.md
│   ├── ALTERNATIVE_SPLICING_PIPELINE.md
│   └── TRAINING_GUIDE.md
│
└── examples/
    ├── train_simple.py
    └── predict_variants.py
```

---

## Quick Start

### 1. Prepare Training Data

```python
from meta_spliceai.splice_engine.meta_layer.workflows import prepare_training_data

# Prepare dataset from OpenSpliceAI artifacts
dataset = prepare_training_data(
    base_model='openspliceai',
    variant_source='splicevardb',
    output_dir='data/meta_training/openspliceai_v1'
)
```

### 2. Train Meta-Layer

```python
from meta_spliceai.splice_engine.meta_layer import train_meta_model

# Train with HyenaDNA (requires GPU)
results = train_meta_model(
    base_model='openspliceai',
    sequence_encoder='hyenadna',
    config='configs/hyenadna.yaml',
    output_dir='models/meta_layer_v1'
)

# Or lightweight version (CPU-friendly)
results = train_meta_model(
    base_model='openspliceai',
    sequence_encoder='cnn',
    config='configs/lightweight.yaml',
    output_dir='models/meta_layer_v1_lite'
)
```

### 3. Predict Alternative Splicing

```python
from meta_spliceai.splice_engine.meta_layer import MetaLayerPredictor

# Load trained model
predictor = MetaLayerPredictor(
    model_path='models/meta_layer_v1',
    base_model='openspliceai'
)

# Predict for a gene
results = predictor.predict_gene(
    gene_id='gene-BRCA1',
    return_exons=True
)

# Access predictions
print(f"Splice sites: {len(results['splice_sites'])}")
print(f"Predicted exons: {len(results['exons'])}")
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Detailed system architecture |
| [LABELING_STRATEGY.md](docs/LABELING_STRATEGY.md) | How labels are created from SpliceVarDB |
| [ALTERNATIVE_SPLICING_PIPELINE.md](docs/ALTERNATIVE_SPLICING_PIPELINE.md) | From scores to exon predictions |
| [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) | Step-by-step training instructions |

---

## Related Packages

| Package | Purpose | Relationship |
|---------|---------|--------------|
| `meta_models/` | Original tabular meta-learning | Predecessor (reference) |
| `openspliceai_recalibration/` | Early prototype | Deprecated (merged here) |
| `case_studies/` | Variant databases | Data source (SpliceVarDB) |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers (for HyenaDNA)
- polars, pandas
- scikit-learn

---

## Status

| Component | Status |
|-----------|--------|
| Core config | 🚧 In progress |
| Artifact loader | 🚧 In progress |
| Dataset preparation | 📋 Planned |
| Sequence encoder | 📋 Planned |
| Training pipeline | 📋 Planned |
| Evaluation | 📋 Planned |
| Inference | 📋 Planned |
| CLI | 📋 Planned |

---

*Last Updated: December 2025*

