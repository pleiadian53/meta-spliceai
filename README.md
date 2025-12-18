# Meta-SpliceAI

**A meta-learning framework for splice site prediction with base model integration**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🧬 Overview

Meta-SpliceAI is an advanced framework for splice site prediction that leverages meta-learning to improve upon existing base models (SpliceAI, OpenSpliceAI, etc.). By analyzing and learning from base model errors, Meta-SpliceAI provides:

- **Enhanced Accuracy**: Corrects false positives and false negatives from base models
- **Multi-Model Support**: Works with various splice site prediction models
- **Two Meta-Learning Approaches**: Tabular (XGBoost) and Multimodal (Deep Learning)
- **Interpretable Results**: SHAP-based feature importance and attention visualization
- **Scalable Architecture**: Efficient processing for whole-genome analysis
- **Production-Ready CLI**: Easy-to-use command-line tools

### Key Features

- 🎯 **Multimodal Meta-Layer** ⭐NEW: Deep learning fusion of DNA sequences + base model scores
- 🧩 **Model-Agnostic Base Layer**: Works with any model producing per-nucleotide splice scores
- 🔬 **Dual Approach**: Tabular (fast, interpretable) + Multimodal (powerful, scalable)
- 📊 **Comprehensive Evaluation**: PR-AUC, top-k accuracy, alternative splice site detection
- 💾 **Smart Checkpointing**: Chunk-level resumption with automatic recovery
- 🧠 **Memory-Efficient**: Mini-batch processing for stable memory usage
- 🚀 **GPU Acceleration**: Support for CUDA/MPS-enabled training and inference
- 🧰 **Flexible Pipeline**: Modular design for easy customization

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Base Layer Architecture](#-base-layer-architecture)
- [Meta-Learning Approaches](#-meta-learning-approaches) ⭐NEW
- [CLI Tools](#-cli-tools)
- [Documentation](#-documentation)
- [Examples](#-examples)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔧 Installation

### Prerequisites

- Python 3.10+
- mamba or conda (for environment management)
- 16GB+ RAM recommended
- CUDA-capable GPU (optional, for acceleration)

### Using Mamba (Recommended)

```bash
# Clone the repository
git clone https://github.com/pleiadian53/meta-spliceai.git
cd meta-spliceai

# Create and activate environment
mamba create -n metaspliceai python=3.10 -y
mamba activate metaspliceai

# Install in editable mode
pip install -e .

# Verify installation
meta-spliceai-run --help
```

### Core Dependencies

- **Data Processing**: `polars`, `pandas`, `pybedtools`
- **Tabular ML**: `tensorflow`, `xgboost`, `scikit-learn`
- **Deep Learning**: `torch` (PyTorch), `transformers` (optional, for HyenaDNA)
- **Genomics**: `pyfaidx`, `gffutils`, `biopython`
- **Visualization**: `matplotlib`, `seaborn`, `shap`

See [`pyproject.toml`](pyproject.toml) for the complete dependency list.

---

## 🚀 Quick Start

### 1. Run Base Model Predictions

```bash
# Run OpenSpliceAI on chromosome 21
run_base_model \
    --base-model openspliceai \
    --mode production \
    --coverage full_genome \
    --chromosomes 21 \
    --output-dir data/mane/GRCh38/openspliceai_eval/
```

### 2. Evaluate Model Performance

```bash
# Evaluate predictions and generate metrics
evaluate_predictions \
    --predictions-file data/openspliceai_eval/predictions.h5 \
    --truth-file data/mane/GRCh38/splice_sites.bed \
    --output-dir evaluation_results/
```

### 3. Build Training Dataset

```bash
# Generate meta-model training data from base model artifacts
python meta_spliceai/splice_engine/meta_models/builder/incremental_builder.py \
    --input-dir data/mane/GRCh38/openspliceai_eval/meta_models/ \
    --output-dir training_data/ \
    --feature-type kmers \
    --kmer-size 3
```

---

## 🧩 Base Layer Architecture

### Model-Agnostic Design

Meta-SpliceAI's base layer is designed to work with **any splice site prediction model** that follows the prediction protocol:

**Requirements**:
- Model produces **per-nucleotide splice scores**: `(donor_prob, acceptor_prob, neither_prob)`
- Model accepts DNA sequences (ACGT) as input
- Model is pre-trained (weights available for loading)

**Supported Models**:

| Model | Genome Build | Annotation | Framework | Documentation |
|-------|--------------|------------|-----------|---------------|
| **SpliceAI** | GRCh37 | Ensembl v87 | Keras | [SPLICEAI.md](meta_spliceai/splice_engine/base_models/docs/SPLICEAI.md) |
| **OpenSpliceAI** | GRCh38 | MANE v1.3 | PyTorch | [OPENSPLICEAI.md](meta_spliceai/splice_engine/base_models/docs/OPENSPLICEAI.md) |
| **Custom Models** | Any | Any | Any | See below |

> ⚠️ **Important**: Each model requires matching genome build data. See [GENOME_BUILD_COMPATIBILITY.md](docs/base_models/GENOME_BUILD_COMPATIBILITY.md)

### Adding Custom Models

To integrate your own splice prediction model:

1. **Implement a model wrapper** with a `.predict()` method:

```python
class MyCustomSpliceModel:
    def predict(self, sequence: str) -> Dict[str, List[float]]:
        """
        Predict splice sites for a DNA sequence.
        
        Returns
        -------
        dict with keys:
            - 'donor_prob': List[float] (one per nucleotide)
            - 'acceptor_prob': List[float]
            - 'neither_prob': List[float]
            - 'positions': List[int] (1-indexed)
        """
        # Your model inference code here
        return {
            'donor_prob': [...],
            'acceptor_prob': [...],
            'neither_prob': [...]
        }
```

2. **Register in `model_utils.py`**:

```python
# meta_spliceai/splice_engine/meta_models/utils/model_utils.py
def load_base_model_ensemble(base_model: str, ...):
    if base_model == 'my_custom_model':
        return [MyCustomSpliceModel(...)], metadata
    # ... existing models ...
```

3. **Use immediately**:

```bash
run_base_model --base-model my_custom_model --mode test
```

**That's it!** The rest of the pipeline (evaluation, feature extraction, checkpointing, artifact management) works automatically.

---

## 🧠 Meta-Learning Approaches

Meta-SpliceAI provides **two complementary approaches** for recalibrating base model predictions:

| Approach | Package | Best For | Key Advantages |
|----------|---------|----------|----------------|
| **Tabular** | `meta_models/` | Interpretability, quick prototyping | SHAP explanations, fast training |
| **Multimodal** ⭐ | `meta_layer/` | Maximum accuracy, alternative splicing | End-to-end learning, sequence context |

### 1. Tabular Meta-Model (`meta_models/`)

XGBoost-based approach using engineered features:

```python
from meta_spliceai.splice_engine.meta_models.training import run_gene_cv_sigmoid

# Train with gene-wise cross-validation
results = run_gene_cv_sigmoid.main([
    '--dataset', 'training_data/',
    '--out-dir', 'models/tabular/',
    '--n-folds', '5'
])
```

**Features**:
- ✅ Fast training (CPU-friendly)
- ✅ SHAP-based interpretability
- ✅ Leakage detection built-in
- ✅ Gene-wise cross-validation

**Documentation**: [`meta_spliceai/splice_engine/meta_models/`](meta_spliceai/splice_engine/meta_models/)

### 2. Multimodal Meta-Layer (`meta_layer/`) ⭐NEW

Deep learning approach for variant effect prediction and splice site recalibration.

#### Available Models

| Model | Purpose | Best For | Correlation |
|-------|---------|----------|-------------|
| `ValidatedDeltaPredictor` ⭐ | Single-pass delta prediction | **Production use** | r=0.41 |
| `SimpleCNNDeltaPredictor` | Paired delta prediction | Fast experimentation | r=0.38 |
| `SpliceInducingClassifier` | Binary classification | "Is splice-altering?" | AUC=0.61 |
| `HyenaDNADeltaPredictor` | GPU-accelerated deltas | Large-scale training | (GPU required) |

#### Quick Start: Validated Delta Prediction

```python
from meta_spliceai.splice_engine.meta_layer.models import ValidatedDeltaPredictor
from meta_spliceai.splice_engine.meta_layer.data.splicevardb_loader import load_splicevardb

# Load SpliceVarDB variants (ground truth for variant effects)
loader = load_splicevardb(genome_build='GRCh38')
train_variants, test_variants = loader.get_train_test_split(test_chromosomes=['21', '22'])

# Create model (works on M1 Mac or GPU)
model = ValidatedDeltaPredictor(hidden_dim=128, n_layers=6)

# Train with validated targets (SpliceVarDB provides ground truth filtering)
# See: meta_layer/tests/test_validated_delta_experiments.py for full example
```

#### How Validated Delta Works

```
Input: alt_sequence + variant_info (ref_base, alt_base)
       ↓
   [Gated CNN Encoder]
       ↓
Target: SpliceVarDB-validated delta
  - If "Splice-altering": target = base_model(alt) - base_model(ref)
  - If "Normal": target = [0, 0, 0] (no effect!)
       ↓
Output: Δ = [Δ_donor, Δ_acceptor, Δ_neither]
```

**Key Innovation**: Uses SpliceVarDB classifications as ground truth to filter training targets, avoiding learning from inaccurate base model predictions.

**Features**:
- ✅ **Validated targets**: Ground truth from SpliceVarDB experimental data
- ✅ **Single-pass inference**: Efficient (no paired sequences needed)
- ✅ **Gated CNN architecture**: Dilated convolutions for large receptive field
- ✅ **Base-model-agnostic**: Works with SpliceAI, OpenSpliceAI, etc.
- ✅ **GPU acceleration**: MPS (M1 Mac) and CUDA support
- ✅ **HyenaDNA ready**: Pre-trained DNA encoder for GPU environments

**GPU-Accelerated Training (RunPods)**:
```python
from meta_spliceai.splice_engine.meta_layer.models import HyenaDNADeltaPredictor

# For RunPods with RTX 4090 or A40
model = HyenaDNADeltaPredictor(
    model_name='hyenadna-small-32k',
    freeze_encoder=True,
    hidden_dim=256
).to('cuda')
```

**Documentation**: 
- [`meta_layer/docs/methods/ROADMAP.md`](meta_spliceai/splice_engine/meta_layer/docs/methods/ROADMAP.md) - Development roadmap
- [`meta_layer/docs/methods/GPU_REQUIREMENTS.md`](meta_spliceai/splice_engine/meta_layer/docs/methods/GPU_REQUIREMENTS.md) - Compute requirements
- [`meta_layer/docs/experiments/`](meta_spliceai/splice_engine/meta_layer/docs/experiments/) - Experiment results

### Why Multimodal? 🎯

The multimodal approach addresses limitations of tabular features:

| Challenge | Tabular Solution | Multimodal Solution |
|-----------|-----------------|---------------------|
| **Feature engineering** | Manual k-mer extraction | End-to-end learning from raw sequence |
| **Long-range context** | Fixed window features | CNN/HyenaDNA captures variable patterns |
| **Scalability** | Feature matrix grows with gene length | Fixed-size embeddings |
| **Alternative splicing** | Limited pattern recognition | Learns splice motifs directly |

### Entry Points for Integration

The base layer can be accessed through multiple entry points depending on your use case:

| Entry Point | Best For | Documentation |
|-------------|----------|---------------|
| **CLI** (`run_base_model`) | Production runs, automation | Below |
| **Python API** (`run_base_model_predictions()`) | Custom workflows, research | [`UNIVERSAL_BASE_MODEL_SUPPORT.md`](docs/base_models/UNIVERSAL_BASE_MODEL_SUPPORT.md) |
| **Shell Scripts** | Sequential processing, logging | [`scripts/training/`](scripts/training/) |

**Documentation**:
- **User Guides**: [`docs/base_models/`](docs/base_models/) - Setup, usage, examples
- **Technical Details**: [`meta_spliceai/splice_engine/base_models/docs/`](meta_spliceai/splice_engine/base_models/docs/) - Implementation, porting, data mappings

---

## 🔨 CLI Tools

Meta-SpliceAI provides production-ready command-line tools:

### `run_base_model`
Run base model predictions on genomic data with **memory-efficient processing** and **automatic checkpointing**.

```bash
run_base_model --help
```

**Key Features**:
- 🧠 **Memory-Efficient**: Mini-batch processing (50 genes/batch) for stable memory usage
- 💾 **Smart Checkpointing**: Chunk-level resumption (500-gene chunks)
- 🔄 **Auto-Recovery**: Automatically resumes from interruptions
- 📊 **Progress Monitoring**: Real-time progress bars and logging

**Basic Usage**:
```bash
# Full genome pass with OpenSpliceAI
run_base_model \
    --base-model openspliceai \
    --mode production \
    --coverage full_genome

# Specific chromosomes
run_base_model \
    --base-model spliceai \
    --chromosomes 1,2,21 \
    --mode production

# Test mode (small subset)
run_base_model \
    --base-model openspliceai \
    --mode test \
    --coverage sample
```

**Options**:
- `--base-model`: Model to use (`openspliceai`, `spliceai`, or custom)
- `--mode`: Run mode (`production`, `test`) - controls artifact overwriting
- `--coverage`: Analysis coverage (`full_genome`, `sample`)
- `--chromosomes`: Comma-separated list of chromosomes to process
- `--verbosity`: Output verbosity (0=minimal, 1=normal, 2=detailed)
- `--mini-batch-size`: Genes per mini-batch (default: 50, reduce if OOM)
- `--no-final-aggregate`: Skip memory-heavy final aggregation

### `evaluate_predictions`
Evaluate model predictions and generate metrics.

```bash
evaluate_predictions --help
```

**Options**:
- `--predictions-file`: Predictions to evaluate
- `--truth-file`: Ground truth annotations
- `--metrics`: Metrics to calculate (precision, recall, f1, roc_auc)
- `--output-dir`: Output directory for evaluation results

### `annotate_splice_sites`
Generate and validate splice site annotations from GTF files with enhanced metadata (14 columns).

```bash
annotate_splice_sites --help
```

**Quick Examples**:
```bash
# Generate with validation for MANE/GRCh38
annotate_splice_sites --build mane-grch38 --validate

# List available builds
annotate_splice_sites --list-builds

# Custom GTF file
annotate_splice_sites --gtf annotations.gtf --output sites.tsv --validate
```

**Options**:
- `--build`: Use predefined build (mane-grch38, ensembl-grch37, ensembl-grch38)
- `--gtf`: Path to custom GTF file
- `--output`: Output TSV file path
- `--consensus-window`: Consensus window size (default: 2)
- `--validate`: Validate consistency with GTF (5 validation tests)
- `--list-builds`: Show available genomic builds

**See**: [`docs/CLI_SPLICE_SITES.md`](docs/CLI_SPLICE_SITES.md) for complete documentation

---

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Installation Guide](docs/installation/)** - Detailed setup instructions
- **[Base Models](docs/base_models/)** - Base model integration and comparison
- **[Training](docs/training/)** - Meta-model training workflows
- **[Testing](docs/testing/)** - Testing procedures and validation
- **[Development](docs/development/)** - Development guidelines and setup

### Quick Links

#### Base Layer Guides
- **[Universal Base Model Support](docs/base_models/UNIVERSAL_BASE_MODEL_SUPPORT.md)** - Multi-model architecture and usage
- **[Base Model Comparison Guide](docs/base_models/BASE_MODEL_COMPARISON_GUIDE.md)** - Comparing SpliceAI vs OpenSpliceAI
- **[Genome Build Compatibility](docs/base_models/GENOME_BUILD_COMPATIBILITY.md)** - Critical build requirements
- **[Package-Level Docs](meta_spliceai/splice_engine/base_models/docs/)** - Technical implementation details

#### Data & Organization
- **[Data Layout Master Guide](docs/data/DATA_LAYOUT_MASTER_GUIDE.md)** - Complete data organization reference
- [Documentation Structure](DOCUMENTATION.md)

#### Multimodal Meta-Layer (NEW)
- **[Architecture](meta_spliceai/splice_engine/meta_layer/docs/ARCHITECTURE.md)** - Multimodal system design
- **[Training vs Inference](meta_spliceai/splice_engine/meta_layer/docs/TRAINING_VS_INFERENCE.md)** - Subsampled vs full coverage
- **[Labeling Strategy](meta_spliceai/splice_engine/meta_layer/docs/LABELING_STRATEGY.md)** - How labels are derived
- **[Data Leakage Prevention](meta_spliceai/splice_engine/meta_layer/docs/DATA_FORMAT_AND_LEAKAGE.md)** - Feature exclusion rules
- **[Genomic Resources Integration](meta_spliceai/splice_engine/meta_layer/docs/GENOMIC_RESOURCES_INTEGRATION.md)** - Dynamic path resolution

#### Technical References
- [TN Sampling Fix Implementation](docs/training/TN_SAMPLING_FIX_IMPLEMENTATION.md)
- [Base Model Artifacts Verification](docs/training/BASE_MODEL_ARTIFACTS_VERIFICATION.md)

---

## 💡 Examples

### Using the Base Layer Python API

```python
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)

# Run predictions on specific genes
results = run_enhanced_splice_prediction_workflow(
    base_model='openspliceai',
    target_genes=['UNC13A', 'STMN2', 'TARDBP'],  # ALS-related genes
    verbosity=1
)

# Access results
print(f"Processed genes: {results['manifest_summary']['processed_genes']}")
print(f"Total positions: {results['positions'].height}")

# Get high-confidence predictions
import polars as pl
high_conf = results['positions'].filter(
    pl.col('donor_score') > 0.9
)
print(f"High-confidence donors: {high_conf.height}")

# Save results
results['positions'].write_csv('predictions.csv')
```

### Basic Usage (Legacy)

```python
from meta_spliceai.splice_engine import enhanced_evaluation
from meta_spliceai.base_models import OpenSpliceAIRunner

# Initialize base model
runner = OpenSpliceAIRunner(
    genome_fasta="genome.fa",
    annotation_gtf="annotation.gtf"
)

# Run predictions
predictions = runner.predict_splice_sites(
    chromosome="chr1",
    start=1000000,
    end=2000000
)

# Evaluate with enhanced metrics
evaluator = enhanced_evaluation.EnhancedEvaluator(
    predictions=predictions,
    truth_bed="truth.bed"
)

metrics = evaluator.calculate_metrics()
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")
```

### Feature Extraction for Meta-Learning

```python
from meta_spliceai.splice_engine.meta_models.builder import incremental_builder

# Initialize dataset builder
builder = incremental_builder.IncrementalDatasetBuilder(
    input_dir="artifacts/",
    output_dir="training_data/",
    feature_config={
        "kmer_size": 3,
        "include_positional": True,
        "include_base_scores": True
    }
)

# Build training dataset
builder.build_dataset()
print(f"Training samples: {builder.n_samples}")
print(f"Features: {builder.feature_names}")
```

### Error Analysis

```python
from meta_spliceai.splice_engine.meta_models.core import enhanced_evaluation

# Analyze prediction errors
analyzer = enhanced_evaluation.ErrorAnalyzer(
    predictions_file="predictions.h5",
    truth_file="truth.bed"
)

# Get false positives and false negatives
fps = analyzer.get_false_positives()
fns = analyzer.get_false_negatives()

print(f"False Positives: {len(fps)}")
print(f"False Negatives: {len(fns)}")

# Visualize error patterns
analyzer.plot_error_distribution()
analyzer.plot_score_distributions()
```

More examples available in the [documentation](docs/).

---

## 📁 Project Structure

```
meta-spliceai/
├── meta_spliceai/                    # Main package
│   │
│   ├── cli/                         # Command-line interface tools
│   │   ├── run_base_model_cli.py   # meta-spliceai-run entry point
│   │   └── evaluate_cli.py         # meta-spliceai-eval entry point
│   │
│   ├── splice_engine/              # Core splice analysis engine
│   │   ├── base_models/           # Base model integrations
│   │   │   ├── docs/              # Technical docs (SPLICEAI.md, OPENSPLICEAI.md, etc.)
│   │   │   └── ...                # Model adapters and utilities
│   │   │
│   │   ├── meta_models/           # Tabular meta-learning (XGBoost)
│   │   │   ├── core/              # Core evaluation & error analysis
│   │   │   │   ├── enhanced_evaluation.py  # Enhanced metrics
│   │   │   │   └── error_analyzer.py       # Error classification
│   │   │   ├── training/          # Training scripts (gene-wise CV)
│   │   │   └── builder/           # Training dataset builder
│   │   │       ├── incremental_builder.py  # Scalable dataset assembly
│   │   │       └── feature_extractor.py    # K-mer & context features
│   │   │
│   │   ├── meta_layer/            # Multimodal meta-learning ⭐NEW
│   │   │   ├── core/              # Config, artifact loading, feature schema
│   │   │   ├── models/            # Sequence encoder, score encoder, fusion
│   │   │   ├── data/              # PyTorch Dataset and DataLoader
│   │   │   ├── training/          # Trainer, evaluator (PR-AUC, top-k)
│   │   │   ├── inference/         # Predictor, full coverage inference
│   │   │   ├── docs/              # Architecture, labeling, leakage docs
│   │   │   └── examples/          # Training and verification scripts
│   │   │
│   │   └── docs/                  # Splice engine documentation
│   │
│   ├── system/                    # System configuration
│   │   ├── genomic_resources/     # Genomic data management
│   │   └── config/                # Configuration files
│   │
│   └── utils/                     # Common utilities
│       ├── file_utils.py          # File I/O helpers
│       └── data_utils.py          # Data processing utilities
│
├── scripts/                        # Helper scripts
│   ├── training/                  # Training workflows
│   │   ├── run_full_genome_base_model_pass.py  # Genome-wide predictions
│   │   ├── run_chromosomes_sequential.sh       # Batch processing
│   │   └── monitor_chromosomes.sh              # Progress monitoring
│   │
│   ├── setup/                     # Setup and installation
│   │   ├── download_grch38_mane_data.sh       # Data download
│   │   └── extract_grch38_mane_sequences.sh   # Sequence extraction
│   │
│   └── testing/                   # Testing scripts
│       └── run_openspliceai_test.sh           # OpenSpliceAI tests
│
├── docs/                           # Documentation (user-facing)
│   ├── README.md                  # Documentation index
│   ├── installation/              # Installation guides
│   ├── base_models/               # Base model user guides
│   │   ├── UNIVERSAL_BASE_MODEL_SUPPORT.md   # Multi-model architecture
│   │   ├── GENOME_BUILD_COMPATIBILITY.md     # Build requirements
│   │   ├── BASE_MODEL_COMPARISON_GUIDE.md    # Model comparison
│   │   └── ...                               # Setup guides, examples
│   ├── training/                  # Training documentation
│   ├── testing/                   # Testing procedures
│   └── development/               # Development guidelines
│
├── data/                           # Data directory (gitignored)
│   └── mane/                      # MANE transcripts
│       └── GRCh38/                # Human genome build 38
│           ├── openspliceai_eval/ # OpenSpliceAI outputs
│           └── spliceai_eval/     # SpliceAI outputs
│
├── pyproject.toml                  # Package configuration (Poetry)
├── README.md                       # This file
├── LICENSE                         # MIT License
├── CONTRIBUTING.md                 # Contribution guidelines
└── DOCUMENTATION.md                # Documentation structure guide
```

### Key Components

#### 🔧 Core Packages

- **`meta_spliceai/cli/`** - Production-ready command-line tools
- **`meta_spliceai/splice_engine/`** - Core analysis engine with base model integration
- **`meta_spliceai/splice_engine/meta_models/`** - Tabular meta-learning (XGBoost, SHAP)
- **`meta_spliceai/splice_engine/meta_layer/`** - Multimodal meta-learning (Deep Learning) ⭐NEW

#### 📝 Scripts

- **`scripts/training/`** - Genome-wide base model prediction workflows
- **`scripts/setup/`** - Data download and preparation scripts
- **`scripts/testing/`** - Validation and testing utilities

#### 📚 Documentation

- **`docs/`** - Comprehensive user and developer documentation
- **Package-level docs** - Technical documentation within module directories

---

## 👩‍💻 Development

### Setting Up Development Environment

```bash
# Create development environment
mamba create -n metaspliceai python=3.10 -y
mamba activate metaspliceai

# Install in editable mode
pip install -e .

# Verify CLI commands are registered
which meta-spliceai-run
meta-spliceai-run --help
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_base_models.py

# Run with coverage
pytest --cov=meta_spliceai tests/
```

### Code Quality

```bash
# Format code
black meta_spliceai/

# Lint code
flake8 meta_spliceai/

# Type checking
mypy meta_spliceai/
```

See [Development Guide](docs/development/) for more details.

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code of Conduct
- Development workflow
- Pull request process
- Testing requirements
- Documentation standards

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Meta-SpliceAI builds upon several excellent open-source projects:

- **[SpliceAI](https://github.com/Illumina/SpliceAI)** - Base splice site prediction model
- **[OpenSpliceAI](https://github.com/opengenomics/OpenSpliceAI)** - Alternative splice site predictor
- **[Polars](https://pola.rs/)** - High-performance DataFrames
- **[TensorFlow](https://www.tensorflow.org/)** - Machine learning framework
- **[SHAP](https://github.com/slundberg/shap)** - Model interpretation

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/pleiadian53/meta-spliceai/issues)
- **Documentation**: [docs/](docs/)
- **Discussions**: [GitHub Discussions](https://github.com/pleiadian53/meta-spliceai/discussions)

---

## 🔬 Citation

If you use Meta-SpliceAI in your research, please cite:

```bibtex
@software{metaspliceai2025,
  author = {Chiu, Barnett},
  title = {Meta-SpliceAI: A meta-learning framework for splice site prediction},
  year = {2025},
  url = {https://github.com/pleiadian53/meta-spliceai}
}
```

---

**Meta-SpliceAI** - Advancing splice site prediction and isoform discovery through meta-learning

*Version 0.4.0 | Last Updated: December 2025*

---

### Changelog (v0.4.0)

- ⭐ **NEW**: `ValidatedDeltaPredictor` - Single-pass delta prediction with SpliceVarDB ground truth (r=0.41)
- ⭐ **NEW**: `SpliceInducingClassifier` - Binary classification for splice-altering variants
- ⭐ **NEW**: `HyenaDNADeltaPredictor` - GPU-accelerated with pre-trained DNA foundation model
- ⭐ **NEW**: Comprehensive experiment documentation and GPU requirements guide
- 🔧 Gated CNN architecture with dilated convolutions for improved delta prediction
- 🔧 SpliceVarDB integration for validated training targets
- 📚 New docs: `ROADMAP.md`, `GPU_REQUIREMENTS.md`, experiment results

### Changelog (v0.3.0)

- ⭐ Multimodal meta-layer (`meta_layer/`) with deep learning fusion
- ⭐ Full coverage inference for all nucleotides in a gene
- ⭐ Automatic data leakage detection and prevention
- 🔧 Integration with `genomic_resources` for base-model-agnostic design
- 📚 Comprehensive documentation for multimodal approach
