# OpenSpliceAI Base Model Documentation

**Last Updated:** December 2025  
**Status:** ✅ Fully Supported

---

## Overview

OpenSpliceAI is an open-source reimplementation and retraining of SpliceAI using PyTorch and modern genomic annotations (MANE). It provides splice site predictions compatible with GRCh38.

| Property | Value |
|----------|-------|
| **Developer** | Open-source community |
| **Framework** | PyTorch |
| **Genome Build** | GRCh38 |
| **Annotation Source** | MANE (Matched Annotation from NCBI and EBI) |
| **Release Used** | MANE v1.3 |

---

## Why OpenSpliceAI?

### Advantages over SpliceAI

| Aspect | SpliceAI | OpenSpliceAI |
|--------|----------|--------------|
| **Genome Build** | GRCh37 (legacy) | GRCh38 (current) |
| **Framework** | Keras/TensorFlow | PyTorch |
| **Annotations** | GENCODE V24 (2016) | MANE v1.3 (2023+) |
| **Open Source** | Weights only | Full training code |
| **Reproducibility** | Limited | Full |

### When to Use OpenSpliceAI

- Working with GRCh38 coordinates
- Need modern annotation compatibility
- Prefer PyTorch ecosystem
- Require reproducible training

---

## Model Details

### Architecture

- **Type**: Deep residual neural network (same as SpliceAI)
- **Input**: 10,000 bp sequence context
- **Output**: 3 scores per position (donor, acceptor, neither)
- **Framework**: PyTorch

### Model Files

```
data/models/openspliceai/
├── model_1.pt
├── model_2.pt
├── model_3.pt
├── model_4.pt
└── model_5.pt
```

---

## Training Data

### Genome and Annotations

- **Genome Build**: GRCh38
- **Annotations**: MANE v1.3
- **Annotation Type**: Canonical transcripts (matched NCBI/EBI)

### MANE Annotation Benefits

- **Curated**: Manually reviewed canonical transcripts
- **Consistent**: Same transcript IDs in RefSeq and Ensembl
- **Current**: Regularly updated
- **Clinical**: Used in clinical variant interpretation

---

## Performance

### Reported Performance

Similar to SpliceAI when evaluated on matched build:
- **PR-AUC**: ~0.95
- **Top-k Accuracy**: ~0.93

### Our Evaluation (GRCh38/MANE)

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
run_base_model --base-model openspliceai --mode test

# With specific genes
run_base_model --base-model openspliceai --genes BRCA1,TP53

# Full coverage with nucleotide scores
run_base_model --base-model openspliceai --save-nucleotide-scores
```

### Python API

```python
from meta_spliceai import run_base_model_predictions

# OpenSpliceAI automatically uses GRCh38/MANE
results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1', 'TP53'],
    save_nucleotide_scores=True
)

# Access results
positions_df = results['positions']
nucleotide_df = results['nucleotide_scores']
```

### Configuration

OpenSpliceAI is configured via `SpliceAIConfig`:

```python
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig

config = SpliceAIConfig(base_model='openspliceai')
# Automatically sets:
#   - gtf_file: data/mane/GRCh38/annotations.gtf
#   - genome_fasta: data/mane/GRCh38/genome.fa
#   - eval_dir: data/mane/GRCh38/openspliceai_eval/
```

---

## Data Paths

OpenSpliceAI uses GRCh38/MANE data:

```
data/mane/GRCh38/
├── genome.fa                    # Reference genome
├── genome.fa.fai                # FASTA index
├── annotations.gtf              # MANE v1.3 annotations
├── gene_features.tsv            # Derived gene features
├── splice_sites_enhanced.tsv    # Derived splice sites
└── openspliceai_eval/           # Prediction outputs
    └── meta_models/
        ├── full_splice_positions_enhanced.tsv
        ├── nucleotide_scores.tsv
        └── gene_manifest.tsv
```

---

## Comparison to SpliceAI

| Aspect | SpliceAI | OpenSpliceAI |
|--------|----------|--------------|
| **Genome Build** | GRCh37 | GRCh38 |
| **Annotations** | Ensembl v87 | MANE v1.3 |
| **Data Directory** | `data/ensembl/GRCh37/` | `data/mane/GRCh38/` |
| **Eval Directory** | `spliceai_eval/` | `openspliceai_eval/` |
| **Model Format** | Keras HDF5 (.h5) | PyTorch (.pt) |
| **Framework** | TensorFlow/Keras | PyTorch |

### Choosing Between Models

Use **SpliceAI** when:
- Working with GRCh37 coordinates
- Comparing to legacy analyses
- Need exact replication of original SpliceAI

Use **OpenSpliceAI** when:
- Working with GRCh38 coordinates
- Need modern annotation compatibility
- Prefer PyTorch ecosystem
- Integrating with clinical workflows (MANE)

---

## Known Considerations

### 1. Build-Specific

- Must use GRCh38 data for correct performance
- MetaSpliceAI handles this automatically

### 2. MANE Annotations

- Uses canonical transcripts only
- May have fewer isoforms than Ensembl
- Better for clinical applications

### 3. PyTorch Dependency

- Requires PyTorch installation
- GPU acceleration available

---

## Adding New Base Models

OpenSpliceAI demonstrates the extensibility pattern. To add a new model:

1. **Create model files** in `data/models/<model_name>/`
2. **Add configuration** to `model_config.py`
3. **Specify build and annotation source**
4. **Update Registry** if using non-standard paths

See [UNIVERSAL_BASE_MODEL_SUPPORT.md](../../../docs/base_models/UNIVERSAL_BASE_MODEL_SUPPORT.md) for details.

---

## Related Documentation

- [SPLICEAI.md](SPLICEAI.md) - SpliceAI model documentation
- [POSITION_COORDINATE_SYSTEMS.md](POSITION_COORDINATE_SYSTEMS.md) - Coordinate handling
- [GENOME_BUILD_COMPATIBILITY.md](../../../docs/base_models/GENOME_BUILD_COMPATIBILITY.md) - Build compatibility guide
- [UNIVERSAL_BASE_MODEL_SUPPORT.md](../../../docs/base_models/UNIVERSAL_BASE_MODEL_SUPPORT.md) - Multi-model support

