# SpliceAI Base Model Documentation

**Last Updated:** December 2025  
**Status:** ✅ Fully Supported

---

## Overview

SpliceAI is a deep learning model for predicting splice sites from pre-mRNA sequence. It was developed by Illumina and published in Cell (2019).

| Property | Value |
|----------|-------|
| **Developer** | Illumina |
| **Year** | 2019 |
| **Framework** | Keras/TensorFlow |
| **Genome Build** | GRCh37 (hg19) |
| **Annotation Source** | Ensembl (GENCODE V24lift37) |

---

## Critical: Genome Build Compatibility

### ⚠️ SpliceAI was trained on GRCh37/hg19, NOT GRCh38!

This is critical for correct evaluation. Using mismatched genome builds causes severe performance degradation.

| Build | PR-AUC | Status |
|-------|--------|--------|
| GRCh37 (correct) | ~0.85-0.95 | ✅ Matches training |
| GRCh38 (incorrect) | ~0.54 | ❌ 44% performance drop |

### Solution

MetaSpliceAI automatically routes SpliceAI to GRCh37 data:

```python
from meta_spliceai import run_base_model_predictions

# SpliceAI automatically uses GRCh37/Ensembl
results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1', 'TP53']
)
```

---

## Model Details

### Architecture

- **Type**: Deep residual neural network
- **Input**: 10,000 bp sequence context (5,000 bp upstream + 5,000 bp downstream)
- **Output**: 3 scores per position (donor, acceptor, neither)
- **Layers**: 32 residual blocks with dilated convolutions
- **Parameters**: ~10M parameters

### Model Variants

| Model | Context | Use Case |
|-------|---------|----------|
| SpliceAI-80nt | 80 nt | Fast, local splicing |
| SpliceAI-400nt | 400 nt | Balanced |
| SpliceAI-2k | 2,000 nt | Medium-range |
| SpliceAI-10k | 10,000 nt | Long-range splicing (default) |

**MetaSpliceAI uses**: SpliceAI-10k (5 ensemble models)

### Model Files

```
data/models/spliceai/
├── spliceai1.h5
├── spliceai2.h5
├── spliceai3.h5
├── spliceai4.h5
└── spliceai5.h5
```

---

## Training Data

### Genome and Annotations

- **Genome Build**: GRCh37/hg19
- **Annotations**: GENCODE V24lift37 (2016)
- **Genes**: 20,287 protein-coding genes
- **Splice Junctions**: ~130,000 donor-acceptor pairs

### Train/Test Split

**Training Set** (13,384 genes):
- Chromosomes: 2, 4, 6, 8, 10-22, X, Y

**Test Set** (1,652 genes):
- Chromosomes: 1, 3, 5, 7, 9
- No paralogs from training set

---

## Performance

### Reported (Paper, GRCh37)

- **PR-AUC**: 0.97
- **Top-k Accuracy**: 95%
- **On lincRNAs**: 84% (lower due to incomplete annotations)

### Our Evaluation (GRCh37)

Validated on December 2025 with 5 genes (HTT, MYC, BRCA1, APOB, KRAS):

| Metric | Value |
|--------|-------|
| Positions compared | 345,397 |
| Max score difference vs agentic-spliceai | 0.00 |
| Result | ✅ Identical outputs |

---

## Usage in MetaSpliceAI

### CLI

```bash
# Basic usage
run_base_model --base-model spliceai --mode test

# With specific genes
run_base_model --base-model spliceai --genes BRCA1,TP53

# Full coverage with nucleotide scores
run_base_model --base-model spliceai --save-nucleotide-scores
```

### Python API

```python
from meta_spliceai import run_base_model_predictions

# SpliceAI automatically uses GRCh37/Ensembl
results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1', 'TP53'],
    save_nucleotide_scores=True
)

# Access results
positions_df = results['positions']
nucleotide_df = results['nucleotide_scores']
```

### Configuration

SpliceAI is configured via `SpliceAIConfig`:

```python
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig

config = SpliceAIConfig(base_model='spliceai')
# Automatically sets:
#   - gtf_file: data/ensembl/GRCh37/annotations.gtf
#   - genome_fasta: data/ensembl/GRCh37/genome.fa
#   - eval_dir: data/ensembl/GRCh37/spliceai_eval/
```

---

## Data Paths

SpliceAI uses GRCh37/Ensembl data:

```
data/ensembl/GRCh37/
├── genome.fa                    # Reference genome
├── genome.fa.fai                # FASTA index
├── annotations.gtf              # Ensembl v87 annotations
├── gene_features.tsv            # Derived gene features
├── splice_sites_enhanced.tsv    # Derived splice sites
└── spliceai_eval/               # Prediction outputs
    └── meta_models/
        ├── full_splice_positions_enhanced.tsv
        ├── nucleotide_scores.tsv
        └── gene_manifest.tsv
```

---

## Known Limitations

### 1. Genome Build Dependency

- **Critical**: Must use GRCh37 for optimal performance
- MetaSpliceAI handles this automatically

### 2. Annotation Age

- Trained on GENCODE V24 (2016)
- Missing 8+ years of novel isoforms
- May have lower recall on newer annotations

### 3. Non-Coding RNAs

- Lower performance on lincRNAs (84% vs 95%)
- Trained primarily on protein-coding genes

---

## Comparison to Other Models

| Model | Build | Framework | PR-AUC | Notes |
|-------|-------|-----------|--------|-------|
| **SpliceAI** | GRCh37 | Keras | 0.97 | Original, well-validated |
| **OpenSpliceAI** | GRCh38 | PyTorch | ~0.95 | Open-source, modern |

See [OPENSPLICEAI.md](OPENSPLICEAI.md) for OpenSpliceAI details.

---

## References

### Primary Paper

Jaganathan, K., et al. (2019). "Predicting Splicing from Primary Sequence with Deep Learning." *Cell*, 176(3), 535-548.

### Key Points from Paper

1. **Training**: GRCh37/hg19, GENCODE V24lift37
2. **Performance**: PR-AUC 0.97, Top-k 95%
3. **Architecture**: 32 residual blocks, dilated convolutions
4. **Context**: 10,000 bp (5kb upstream + 5kb downstream)

---

## Related Documentation

- [OPENSPLICEAI.md](OPENSPLICEAI.md) - OpenSpliceAI model documentation
- [POSITION_COORDINATE_SYSTEMS.md](POSITION_COORDINATE_SYSTEMS.md) - Coordinate handling
- [GENOME_BUILD_COMPATIBILITY.md](../../../docs/base_models/GENOME_BUILD_COMPATIBILITY.md) - Build compatibility guide
- [UNIVERSAL_BASE_MODEL_SUPPORT.md](../../../docs/base_models/UNIVERSAL_BASE_MODEL_SUPPORT.md) - Multi-model support
