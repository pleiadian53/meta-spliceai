# Meta Layer Documentation

**Central documentation hub for MetaSpliceAI's Multimodal Meta-Learning Layer**

---

## 📚 Quick Navigation

### Getting Started
- **[Package README](../../meta_spliceai/splice_engine/meta_layer/README.md)** - Main package overview
- **[Architecture](../../meta_spliceai/splice_engine/meta_layer/docs/ARCHITECTURE.md)** - System design
- **[Training Guide](../../meta_spliceai/splice_engine/meta_layer/docs/TRAINING_GUIDE.md)** - How to train models

### Core Concepts
- **[Labeling Strategy](../../meta_spliceai/splice_engine/meta_layer/docs/LABELING_STRATEGY.md)** - How training labels are created
- **[Data Format and Leakage](../../meta_spliceai/splice_engine/meta_layer/docs/DATA_FORMAT_AND_LEAKAGE.md)** - Preventing data leakage
- **[Training vs Inference](../../meta_spliceai/splice_engine/meta_layer/docs/TRAINING_VS_INFERENCE.md)** - Subsampled vs full coverage

### Methods
- **[Validated Delta Prediction](../../meta_spliceai/splice_engine/meta_layer/docs/methods/VALIDATED_DELTA_PREDICTION.md)** - Ground-truth validated approach
- **[Paired Delta Prediction](../../meta_spliceai/splice_engine/meta_layer/docs/methods/PAIRED_DELTA_PREDICTION.md)** - Reference-alternate comparison
- **[Multi-Step Framework](../../meta_spliceai/splice_engine/meta_layer/docs/methods/MULTI_STEP_FRAMEWORK.md)** - Staged training pipeline
- **[Meta Recalibration](../../meta_spliceai/splice_engine/meta_layer/docs/methods/META_RECALIBRATION.md)** - Score recalibration approach
- **[Roadmap](../../meta_spliceai/splice_engine/meta_layer/docs/methods/ROADMAP.md)** - Development roadmap

### Advanced
- **[HyenaDNA Fine-tuning](../../meta_spliceai/splice_engine/meta_layer/docs/methods/HYENADNA_FINETUNING_TUTORIAL.md)** - GPU-accelerated training
- **[GPU Requirements](../../meta_spliceai/splice_engine/meta_layer/docs/methods/GPU_REQUIREMENTS.md)** - Hardware specifications
- **[Alternative Splicing Pipeline](../../meta_spliceai/splice_engine/meta_layer/docs/ALTERNATIVE_SPLICING_PIPELINE.md)** - From scores to exons

### Data Sources
- **[SpliceVarDB](../../meta_spliceai/splice_engine/meta_layer/docs/data/SPLICEVARDB.md)** - Validated variant database
- **[MutSpliceDB](../../meta_spliceai/splice_engine/meta_layer/docs/data/MUTSPLICEDB.md)** - Cancer splice mutations
- **[HGVS Tutorial](../../meta_spliceai/splice_engine/meta_layer/docs/data/HGVS_TUTORIAL.md)** - Variant notation parsing
- **[Liftover Tutorial](../../meta_spliceai/splice_engine/meta_layer/docs/data/LIFTOVER_TUTORIAL.md)** - Coordinate conversion

### Experiments
- **[Experiment Overview](../../meta_spliceai/splice_engine/meta_layer/docs/experiments/README.md)** - All experiments
- **[Canonical Classification](../../meta_spliceai/splice_engine/meta_layer/docs/experiments/001_canonical_classification/)** - Experiment 001
- **[Delta Prediction](../../meta_spliceai/splice_engine/meta_layer/docs/experiments/002_delta_prediction/)** - Experiment 002
- **[Binary Classification](../../meta_spliceai/splice_engine/meta_layer/docs/experiments/003_binary_classification/)** - Experiment 003
- **[Validated Delta](../../meta_spliceai/splice_engine/meta_layer/docs/experiments/004_validated_delta/)** - Experiment 004 ⭐
- **[GPU Training Guide](../../meta_spliceai/splice_engine/meta_layer/docs/experiments/GPU_TRAINING_GUIDE.md)** - RunPods/Cloud setup

### Setup & Configuration
- **[RunPods Complete Setup](../../meta_spliceai/splice_engine/meta_layer/docs/setup/RUNPODS_COMPLETE_SETUP.md)** - Cloud GPU setup
- **[Model Transfer Guide](../../meta_spliceai/splice_engine/meta_layer/docs/setup/MODEL_TRANSFER_GUIDE.md)** - Moving trained models
- **[Path Resolution Issues](../../meta_spliceai/splice_engine/meta_layer/docs/setup/PATH_RESOLUTION_ISSUES.md)** - Troubleshooting paths

---

## 🎯 Overview

The **Meta Layer** is MetaSpliceAI's multimodal deep learning system for splice site recalibration. It learns from base model errors to:

1. **Reduce false positives and false negatives**
2. **Predict variant-induced alternative splicing**
3. **Maintain per-nucleotide probability format**

### Key Innovation

Unlike the older tabular meta-models (XGBoost-based), the meta layer uses:

- **End-to-end learning** from raw DNA sequences
- **Multimodal fusion** of sequence + base model scores
- **Foundation model adapters** (HyenaDNA, DNABERT-2)
- **Validated ground truth** from SpliceVarDB

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    META LAYER PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUT: Base Layer Artifacts                                │
│  ├── 501nt contextual sequences                             │
│  ├── Base model scores (donor, acceptor, neither)           │
│  ├── 50+ derived features                                   │
│  └── Labels from GTF + SpliceVarDB                          │
│                                                              │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │ SEQUENCE        │        │ SCORE           │            │
│  │ ENCODER         │        │ ENCODER         │            │
│  │ (HyenaDNA/CNN)  │        │ (MLP)           │            │
│  └────────┬────────┘        └────────┬────────┘            │
│           │                          │                      │
│           └──────────┬───────────────┘                      │
│                      │                                      │
│              ┌───────▼────────┐                             │
│              │ FUSION LAYER   │                             │
│              │ (Attention)    │                             │
│              └───────┬────────┘                             │
│                      │                                      │
│              ┌───────▼────────┐                             │
│              │ OUTPUT HEAD    │                             │
│              └───────┬────────┘                             │
│                      │                                      │
│  OUTPUT: P(donor), P(acceptor), P(neither)                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Verify Base Layer Artifacts

```bash
python meta_spliceai/splice_engine/meta_layer/examples/verify_artifacts.py \
    --artifacts-dir data/mane/GRCh38/openspliceai_eval/meta_models/
```

### 2. Train Simple Model (CPU)

```bash
python meta_spliceai/splice_engine/meta_layer/examples/train_simple.py \
    --base-model openspliceai \
    --sequence-encoder cnn \
    --output-dir models/meta_v1_lite
```

### 3. Train with HyenaDNA (GPU)

```bash
python meta_spliceai/splice_engine/meta_layer/tests/train_hyenadna_validated_delta.py \
    --base-model openspliceai \
    --output-dir models/meta_v1_hyenadna \
    --gpu
```

### 4. Test Trained Model

```bash
python meta_spliceai/splice_engine/meta_layer/examples/test_model.py \
    --model models/meta_v1 \
    --test-genes BRCA1 TP53 CFTR
```

---

## 📊 Available Models

### Production Models

| Model | Purpose | Best For | Correlation |
|-------|---------|----------|-------------|
| `ValidatedDeltaPredictor` ⭐ | Single-pass delta prediction | **Production use** | r=0.41 |
| `SimpleCNNDeltaPredictor` | Paired delta prediction | Fast experimentation | r=0.38 |
| `HyenaDNADeltaPredictor` | GPU-accelerated deltas | Large-scale training | GPU required |

### Experimental Models

| Model | Purpose | Status |
|-------|---------|--------|
| `SpliceInducingClassifier` | Binary classification | AUC=0.61 |
| `MetaSpliceModel` | Full recalibration | 🚧 In Progress |
| `PositionLocalizer` | Splice site localization | 🚧 Testing |

---

## 🔬 Experiment History

### Completed Experiments

1. **[Experiment 001](../../meta_spliceai/splice_engine/meta_layer/docs/experiments/001_canonical_classification/)** - Canonical Classification
   - Binary classification: Splice site vs neither
   - AUC: 0.XX
   - Lessons: Need more nuanced approach

2. **[Experiment 002](../../meta_spliceai/splice_engine/meta_layer/docs/experiments/002_delta_prediction/)** - Delta Prediction
   - Predict score changes for variants
   - Initial correlation: r=0.38
   - Lessons: Paired sequences helpful

3. **[Experiment 003](../../meta_spliceai/splice_engine/meta_layer/docs/experiments/003_binary_classification/)** - Binary Classification v2
   - Improved architecture
   - Enhanced features
   - Results: Marginal improvement

4. **[Experiment 004](../../meta_spliceai/splice_engine/meta_layer/docs/experiments/004_validated_delta/)** ⭐ **Current Best**
   - Validated delta prediction with SpliceVarDB ground truth
   - Single-pass inference (efficient)
   - Correlation: r=0.41
   - Status: Recommended for use

### GPU Experiments

- **[RunPods A40 50K](../../meta_spliceai/splice_engine/meta_layer/docs/experiments/EXP_2025_12_17_RUNPODS_A40_50K.md)** - Large-scale GPU training
- **[HyenaDNA Fine-tuning](../../meta_spliceai/splice_engine/meta_layer/docs/experiments/004_validated_delta/HYENADNA_EXPERIMENTS.md)** - Foundation model adaptation

---

## 📁 Package Structure

```
meta_layer/
├── core/                  # Core infrastructure
│   ├── config.py         # Configuration management
│   ├── artifact_loader.py # Load base layer artifacts
│   ├── feature_schema.py  # Feature definitions
│   └── path_manager.py    # Dynamic path resolution
│
├── models/                # Neural network models
│   ├── validated_delta_predictor.py  ⭐ Recommended
│   ├── hyenadna_delta_predictor.py   # GPU-accelerated
│   ├── sequence_encoder.py           # DNA encoders
│   ├── score_encoder.py              # Score MLP
│   └── meta_splice_model.py          # Full system
│
├── data/                  # Data loading and processing
│   ├── dataset.py        # PyTorch datasets
│   ├── splicevardb_loader.py  # Validated variants
│   └── variant_dataset.py     # Variant-specific
│
├── training/              # Training infrastructure
│   ├── trainer.py        # Training loops
│   ├── evaluator.py      # Metrics computation
│   └── variant_evaluator.py  # Variant-specific metrics
│
├── inference/             # Prediction and deployment
│   ├── predictor.py      # Main inference engine
│   └── full_coverage_predictor.py  # Genome-wide
│
├── examples/              # User-facing examples
│   ├── train_simple.py   ⭐ Start here
│   ├── test_model.py
│   ├── verify_artifacts.py
│   └── run_base_vs_meta_comparison.py
│
├── tests/                 # Training/test scripts
│   ├── test_validated_delta_experiments.py  # CPU version
│   ├── test_gpu_validated_delta_experiments.py  # GPU version
│   ├── train_hyenadna_validated_delta.py  ⭐ GPU training
│   └── test_gpu_multistep_experiments.py  # Multi-step
│
└── docs/                  # Comprehensive documentation
    ├── ARCHITECTURE.md
    ├── TRAINING_GUIDE.md
    ├── methods/           # Detailed method docs
    ├── experiments/       # Experiment results
    ├── data/              # Data source docs
    └── setup/             # Setup guides
```

---

## 🎓 Learning Path

### Beginner

1. Read [Package README](../../meta_spliceai/splice_engine/meta_layer/README.md)
2. Understand [Architecture](../../meta_spliceai/splice_engine/meta_layer/docs/ARCHITECTURE.md)
3. Run [verify_artifacts.py](../../meta_spliceai/splice_engine/meta_layer/examples/verify_artifacts.py)
4. Try [train_simple.py](../../meta_spliceai/splice_engine/meta_layer/examples/train_simple.py)

### Intermediate

1. Read [Training Guide](../../meta_spliceai/splice_engine/meta_layer/docs/TRAINING_GUIDE.md)
2. Understand [Labeling Strategy](../../meta_spliceai/splice_engine/meta_layer/docs/LABELING_STRATEGY.md)
3. Review [Validated Delta Method](../../meta_spliceai/splice_engine/meta_layer/docs/methods/VALIDATED_DELTA_PREDICTION.md)
4. Run [test_validated_delta_experiments.py](../../meta_spliceai/splice_engine/meta_layer/tests/test_validated_delta_experiments.py)

### Advanced

1. Study [Multi-Step Framework](../../meta_spliceai/splice_engine/meta_layer/docs/methods/MULTI_STEP_FRAMEWORK.md)
2. Setup [RunPods GPU](../../meta_spliceai/splice_engine/meta_layer/docs/setup/RUNPODS_COMPLETE_SETUP.md)
3. Train [HyenaDNA model](../../meta_spliceai/splice_engine/meta_layer/tests/train_hyenadna_validated_delta.py)
4. Review [Experiment Results](../../meta_spliceai/splice_engine/meta_layer/docs/experiments/)

---

## 🔧 Configuration

### Config Files

Located in `meta_spliceai/splice_engine/meta_layer/configs/`:

- **default.yaml** - Default settings
- **lightweight.yaml** - CPU-friendly configuration
- **hyenadna.yaml** - GPU configuration with HyenaDNA

### Example Config

```yaml
# configs/lightweight.yaml
model:
  sequence_encoder: 
    type: cnn
    hidden_dim: 128
    n_layers: 4
  
  score_encoder:
    hidden_dim: 64
    dropout: 0.2
  
  fusion:
    type: concat  # or 'attention'

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 50
  device: cpu
```

---

## 🚧 Development Status

### Completed ✅

- Core configuration system
- Artifact loading from base layer
- Feature schema standardization
- PyTorch dataset implementation
- Sequence encoders (CNN, HyenaDNA)
- Score encoder (MLP)
- Training infrastructure
- Validated delta predictor ⭐
- Basic evaluation metrics
- Example scripts

### In Progress 🚧

- Full meta-splice model
- CLI tool registration
- Inference optimization
- Alternative splicing pipeline
- Comprehensive testing

### Planned 📋

- Production CLI (`train_meta_model`, `predict_meta_splice`)
- Model zoo with pre-trained weights
- Ensemble methods
- Real-time inference API
- Integration with variant analysis

---

## 📊 Performance

### Current Best Model (ValidatedDeltaPredictor)

**Training Data**: SpliceVarDB validated variants  
**Architecture**: Gated CNN + MLP fusion  
**Performance**:
- Correlation: r=0.41 (delta scores)
- Classification AUC: 0.61 (splice-altering)
- Inference: Single-pass (efficient)

**Improvements over Base**:
- Better calibration for pathogenic variants
- Reduced false positives in cryptic sites
- Enhanced detection of deep intronic variants

---

## 🤝 Related Components

- **Base Layer**: Provides input artifacts (sequences, scores, features)
- **Meta Models** (old): Tabular XGBoost predecessor
- **Case Studies**: Validation data (SpliceVarDB, ClinVar)

---

## 📞 Support

- **Documentation**: This directory + package docs
- **Examples**: `meta_layer/examples/`
- **Tests**: `meta_layer/tests/`
- **Issues**: Report via project repository

---

**Last Updated**: January 30, 2026  
**Status**: 🚧 In Active Development  
**Recommended Entry Point**: [train_simple.py](../../meta_spliceai/splice_engine/meta_layer/examples/train_simple.py)
