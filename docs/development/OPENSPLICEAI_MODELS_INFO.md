# OpenSpliceAI Pre-trained Models

**Source**: [OpenSpliceAI GitHub Repository](https://github.com/Kuanhao-Chao/OpenSpliceAI)

---

## Available Models

### OpenSpliceAI-MANE (Recommended)

**Location**: [openspliceai-mane](https://github.com/Kuanhao-Chao/OpenSpliceAI/tree/main/models/openspliceai-mane)

**Training Data**:
- **Annotations**: MANE (Matched Annotation from NCBI and EMBL-EBI)
- **Genome Build**: GRCh38
- **Quality**: High-quality, clinically relevant transcripts

### Context Window Options

| Context | URL | Performance | Use Case |
|---------|-----|-------------|----------|
| **10,000nt** | [10000nt](https://github.com/Kuanhao-Chao/OpenSpliceAI/tree/main/models/openspliceai-mane/10000nt) | **Best** ✅ | **Recommended** - Long-range splicing |
| 2,000nt | [2000nt](https://github.com/Kuanhao-Chao/OpenSpliceAI/tree/main/models/openspliceai-mane/2000nt) | Good | Medium-range splicing |
| 400nt | [400nt](https://github.com/Kuanhao-Chao/OpenSpliceAI/tree/main/models/openspliceai-mane/400nt) | Moderate | Local splicing |
| 80nt | [80nt](https://github.com/Kuanhao-Chao/OpenSpliceAI/tree/main/models/openspliceai-mane/80nt) | Fast | Quick predictions |

**Our Choice**: **10,000nt context** (best performance, matches SpliceAI's context)

---

## Model Files (10,000nt Context)

**Ensemble of 5 Models**:
```
model_10000nt_rs10.pt
model_10000nt_rs11.pt
model_10000nt_rs12.pt
model_10000nt_rs13.pt
model_10000nt_rs14.pt
```

**Total Size**: ~13MB (estimated)

**Download URL Pattern**:
```
https://github.com/Kuanhao-Chao/OpenSpliceAI/raw/main/models/openspliceai-mane/10000nt/model_10000nt_rs{10-14}.pt
```

---

## Training Details

### MANE Annotations

**What is MANE?**
- Matched Annotation from NCBI and EMBL-EBI
- High-confidence transcript set
- One representative transcript per gene
- Clinically relevant
- GRCh38-based

**Advantages over GENCODE**:
- Higher quality (curated)
- Clinical focus
- Reduced redundancy
- Better for variant interpretation

### Comparison with SpliceAI

| Aspect | SpliceAI | OpenSpliceAI-MANE |
|--------|----------|-------------------|
| **Annotations** | GENCODE V24lift37 | MANE |
| **Genome Build** | GRCh37 | GRCh38 |
| **Context** | 10,000nt | 10,000nt |
| **Format** | Keras (.h5) | PyTorch (.pt) |
| **Release Year** | 2019 | 2023+ |
| **Clinical Focus** | General | High |

---

## Download Instructions

### Quick Download

```bash
# Run our download script
./scripts/base_model/download_openspliceai_models.sh
```

### Manual Download

```bash
# Create directory
mkdir -p data/models/openspliceai

# Download each model
cd data/models/openspliceai

for i in {10..14}; do
    curl -L "https://github.com/Kuanhao-Chao/OpenSpliceAI/raw/main/models/openspliceai-mane/10000nt/model_10000nt_rs${i}.pt" \
         -o "model_10000nt_rs${i}.pt"
done

# Verify
ls -lh *.pt
```

---

## Usage in MetaSpliceAI

### Loading Models

```python
from meta_spliceai.splice_engine.meta_models.utils.model_utils import (
    load_openspliceai_ensemble
)

# Load 10,000nt context models
models = load_openspliceai_ensemble(context=10000)
print(f"Loaded {len(models)} models")
```

### Running Predictions

```python
from meta_spliceai import run_base_model_predictions

# Use OpenSpliceAI
results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1', 'TP53'],
    mode='test'
)
```

---

## Model Architecture

**Based on**: SpliceAI architecture
- Deep residual neural network
- Dilated convolutions
- 32 residual blocks
- ~10M parameters per model

**Ensemble**: 5 models (rs10-rs14)
- Different random seeds
- Averaged predictions
- Improved robustness

---

## Performance Expectations

### On GRCh38 (Expected)

Based on MANE training:
- **PR-AUC**: ~0.90-0.95 (expected)
- **Top-k Accuracy**: ~0.85-0.90 (expected)
- **F1 Score**: ~0.85-0.90 (expected)

**Note**: Performance to be validated after download

### Comparison with SpliceAI

| Metric | SpliceAI (GRCh37) | OpenSpliceAI (GRCh38) |
|--------|-------------------|----------------------|
| PR-AUC | 0.97 | ~0.90-0.95 (est.) |
| Build Match | ✅ (on GRCh37) | ✅ (on GRCh38) |
| Clinical Focus | Moderate | High |

---

## References

### Primary Source

- **GitHub**: [Kuanhao-Chao/OpenSpliceAI](https://github.com/Kuanhao-Chao/OpenSpliceAI)
- **Models**: [openspliceai-mane/10000nt](https://github.com/Kuanhao-Chao/OpenSpliceAI/tree/main/models/openspliceai-mane/10000nt)

### Related Papers

- Original SpliceAI: Jaganathan et al., Cell 2019
- OpenSpliceAI: Check repository for publications

### MANE Project

- **Website**: [NCBI MANE](https://www.ncbi.nlm.nih.gov/refseq/MANE/)
- **Description**: Matched Annotation from NCBI and EMBL-EBI

---

## Integration Status

- [x] Models identified
- [x] Download script created
- [ ] Models downloaded
- [ ] Loader function implemented
- [ ] Interface updated
- [ ] Testing complete

---

## Next Steps

1. **Download models** (30 min)
   ```bash
   ./scripts/base_model/download_openspliceai_models.sh
   ```

2. **Implement loader** (1 hour)
   - Add `load_openspliceai_ensemble()` to `model_utils.py`

3. **Update interface** (30 min)
   - Remove `NotImplementedError`
   - Add model selection

4. **Test** (1 hour)
   - Validate loading
   - Test predictions
   - Compare with SpliceAI

---

**Last Updated**: 2025-11-05  
**Status**: Ready to download


