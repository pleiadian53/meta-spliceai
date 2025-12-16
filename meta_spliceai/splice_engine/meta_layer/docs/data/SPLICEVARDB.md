# SpliceVarDB: Dataset Documentation

**Location**: `data/splicevardb/`  
**Purpose**: Ground truth labels for variant effects on splicing  
**Last Updated**: December 2025

---

## Overview

SpliceVarDB is a curated database of genetic variants with experimentally validated 
or computationally predicted effects on RNA splicing. We use it to:

1. **Validate delta prediction targets** (filter which base model deltas to trust)
2. **Evaluate model performance** on real variant effects
3. **Weight training samples** by clinical significance

---

## Classification Categories

SpliceVarDB provides a `classification` field with the following values:

| Classification | Description | How We Use It |
|----------------|-------------|---------------|
| **Splice-altering** | Variant has confirmed effect on splicing | Trust base model delta, use as target |
| **Normal** | Variant has NO effect on splicing | Target = [0,0,0] (override base model) |
| **Low-frequency** | Rare variant, effect uncertain | Skip (don't train on it) |
| **Conflicting** | Evidence is contradictory | Skip (don't train on it) |

### Classification Source

These classifications are derived from:
1. **Experimental validation** (RNA-seq, minigene assays)
2. **Clinical annotations** (ClinVar pathogenicity)
3. **Computational predictions** (multiple tools agreeing)

**Note**: The exact classification logic varies by variant. Some are experimentally 
validated, others are computational predictions. For model training, we treat 
"Splice-altering" and "Normal" as ground truth.

---

## Data Schema

The SpliceVarDB loader returns `VariantRecord` objects with these fields:

```python
@dataclass
class VariantRecord:
    # Genomic coordinates
    chrom: str           # Chromosome (e.g., "1", "22", "X")
    position: int        # 1-based position
    ref_allele: str      # Reference allele (e.g., "G")
    alt_allele: str      # Alternate allele (e.g., "A")
    
    # Variant info
    variant_id: str      # Unique identifier (e.g., "rs123456")
    gene: str            # Gene symbol (e.g., "BRCA1")
    
    # Classification (THE KEY FIELD)
    classification: str  # "Splice-altering", "Normal", "Low-frequency", "Conflicting"
    
    # Additional context
    method: str          # How classification was determined
    location: str        # "Exonic", "Intronic", etc.
```

---

## How We Derive Training Labels

### For Validated Delta Prediction

```python
def get_training_target(variant, base_model, fasta):
    """
    Derive training target from SpliceVarDB classification.
    
    This is the KEY INSIGHT: Use SpliceVarDB to validate base model predictions.
    """
    
    if variant.classification == 'Splice-altering':
        # SpliceVarDB confirms this variant affects splicing
        # TRUST the base model's delta prediction
        ref_seq = get_sequence(variant, fasta)
        alt_seq = apply_variant(ref_seq, variant)
        target = base_model(alt_seq) - base_model(ref_seq)
        return target  # e.g., [+0.35, -0.02, -0.33]
    
    elif variant.classification == 'Normal':
        # SpliceVarDB confirms this variant does NOT affect splicing
        # Target should be ZERO, even if base model predicts otherwise
        return np.array([0.0, 0.0, 0.0])
    
    else:  # Low-frequency, Conflicting
        # Uncertain - don't use for training
        return None
```

### For Binary Classification (Multi-Step Framework)

```python
def get_binary_label(variant):
    """
    Simple binary label: Is this variant splice-altering?
    """
    if variant.classification == 'Splice-altering':
        return 1  # Positive class
    elif variant.classification == 'Normal':
        return 0  # Negative class
    else:
        return None  # Exclude from training
```

---

## Target Format Summary

| Task | Target Type | Shape | Example |
|------|-------------|-------|---------|
| **Validated Delta** | Continuous | [3] | [+0.35, -0.02, -0.33] |
| **Binary Classification** | Integer | scalar | 0 or 1 |
| **Effect Type** | Integer | scalar | 0-5 (class index) |

---

## Data Statistics

### Train/Test Split

We use chromosome-based splitting to prevent data leakage:

```python
loader = load_splicevardb(genome_build='GRCh38')
train, test = loader.get_train_test_split(test_chromosomes=['21', '22'])

# Typical counts:
# Train: ~13,000 Splice-altering, ~11,000 Normal
# Test:  ~400 Splice-altering, ~300 Normal
```

### Class Balance

For training, we typically balance classes:

```python
# Balanced training set
n_each = min(len(splice_altering), len(normal))
balanced = splice_altering[:n_each] + normal[:n_each]
```

---

## Loading Data

### Basic Usage

```python
from meta_spliceai.splice_engine.meta_layer.data.splicevardb_loader import (
    load_splicevardb
)

# Load for GRCh38 (OpenSpliceAI)
loader = load_splicevardb(genome_build='GRCh38')

# Get train/test split
train_variants, test_variants = loader.get_train_test_split(
    test_chromosomes=['21', '22']
)

# Filter by classification
splice_altering = [v for v in train_variants 
                   if v.classification == 'Splice-altering']
normal = [v for v in train_variants 
          if v.classification == 'Normal']
```

### With Sequence Context

```python
from pyfaidx import Fasta
from meta_spliceai.system.genomic_resources import Registry

registry = Registry(build='GRCh38')
fasta = Fasta(str(registry.get_fasta_path()), sequence_always_upper=True)

for variant in splice_altering[:10]:
    # Get sequence around variant
    start = variant.position - 250
    end = variant.position + 250
    chrom = variant.chrom.replace('chr', '')
    
    ref_seq = str(fasta[chrom][start:end].seq)
    print(f"{variant.gene}: {variant.ref_allele}>{variant.alt_allele}")
```

---

## Important Notes

### Genome Build Compatibility

- **GRCh38**: Used with OpenSpliceAI, MANE annotations
- **GRCh37**: Used with SpliceAI, Ensembl annotations

Ensure the SpliceVarDB genome build matches your base model!

### Classification is NOT Position Label

SpliceVarDB classification describes **variant effect**, not splice site type:

```
SpliceVarDB classification: "Splice-altering"
  → Means: "This variant affects splicing"
  → Does NOT mean: "This position is a splice site"

GTF annotation (splice_type): "donor"
  → Means: "This position IS a donor splice site"
```

### Why We Use SpliceVarDB for Validation

Base models (SpliceAI, OpenSpliceAI) may predict incorrect delta scores,
especially for variants that don't actually affect splicing.

SpliceVarDB tells us the GROUND TRUTH:
- If variant is "Splice-altering" → base model delta is probably right
- If variant is "Normal" → base model delta should be zero (even if it's not!)

---

## Related Files

| File | Description |
|------|-------------|
| `data/splicevardb_loader.py` | Loader implementation |
| `data/variant_dataset.py` | PyTorch dataset for variants |
| `docs/methods/ROADMAP.md` | How SpliceVarDB fits in the pipeline |
| `models/validated_delta_predictor.py` | Model using validated targets |

---

## References

- SpliceVarDB: https://splicevardb.org/ (if applicable)
- Related papers on splicing variant databases

---

*This document is part of the meta_layer data documentation.*

