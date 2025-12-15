# OpenSpliceAI Integration - Quick Start Guide

**Ready to implement OpenSpliceAI support? Start here!**

---

## 🎯 Goal

Add OpenSpliceAI as a second base model alongside SpliceAI, enabling:
- Multi-model predictions
- Model comparison
- Build-specific validation (GRCh37 vs GRCh38)

---

## ⚡ Quick Implementation (3 Steps)

### Step 1: Download Models (30 minutes)

```bash
# Create download script
cat > scripts/base_model/download_openspliceai_models.sh << 'EOF'
#!/bin/bash
set -e

MODEL_DIR="data/models/openspliceai"
GITHUB_BASE="https://github.com/OpenOmics/OpenSpliceAI/raw/main/models"

echo "📥 Downloading OpenSpliceAI models..."
mkdir -p "$MODEL_DIR"

for i in {10..14}; do
    MODEL_FILE="model_10000nt_rs${i}.pt"
    echo "  ⏳ $MODEL_FILE..."
    curl -L "$GITHUB_BASE/$MODEL_FILE" -o "$MODEL_DIR/$MODEL_FILE"
done

echo "✅ Download complete!"
ls -lh "$MODEL_DIR"
EOF

# Make executable and run
chmod +x scripts/base_model/download_openspliceai_models.sh
./scripts/base_model/download_openspliceai_models.sh
```

### Step 2: Implement Loader (2 hours)

**Add to** `meta_spliceai/splice_engine/meta_models/utils/model_utils.py`:

```python
def load_openspliceai_ensemble(context: int = 10000) -> List:
    """Load OpenSpliceAI ensemble models."""
    import torch
    import os
    
    model_dir = "data/models/openspliceai/"
    models = []
    
    for i in range(10, 15):
        model_file = f"model_{context}nt_rs{i}.pt"
        model_path = os.path.join(model_dir, model_file)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"OpenSpliceAI model not found: {model_path}\n"
                f"Run: ./scripts/base_model/download_openspliceai_models.sh"
            )
        
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        models.append(model)
    
    return models
```

**Test it**:
```python
from meta_spliceai.splice_engine.meta_models.utils.model_utils import load_openspliceai_ensemble
models = load_openspliceai_ensemble()
print(f"✅ Loaded {len(models)} models")
```

### Step 3: Enable in Interface (1 hour)

**Update** `meta_spliceai/run_base_model.py`:

```python
# Remove the NotImplementedError
if base_model_lower == 'openspliceai':
    # NOW SUPPORTED!
    pass  # Will use workflow's model loading

# In workflow, add model selection
if base_model == 'openspliceai':
    from meta_spliceai.splice_engine.meta_models.utils.model_utils import (
        load_openspliceai_ensemble
    )
    models = load_openspliceai_ensemble(context=10_000)
```

**Test it**:
```python
from meta_spliceai import run_base_model_predictions

results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1'],
    mode='test'
)
print(f"✅ OpenSpliceAI predictions: {results['positions'].height} positions")
```

---

## 📋 Full Implementation Checklist

### Phase 1: Models ⏳
- [ ] Create download script
- [ ] Download 5 OpenSpliceAI models (~13MB)
- [ ] Verify models load correctly
- [ ] Add model metadata

### Phase 2: Integration ⏳
- [ ] Implement `load_openspliceai_ensemble()`
- [ ] Update `run_base_model.py`
- [ ] Update `splice_prediction_workflow.py`
- [ ] Add `base_model` parameter to config

### Phase 3: Testing ⏳
- [ ] Test model loading
- [ ] Test predictions on sample genes
- [ ] Compare SpliceAI vs OpenSpliceAI
- [ ] Validate performance metrics

### Phase 4: Documentation ⏳
- [ ] Update user guide
- [ ] Add examples
- [ ] Document build differences
- [ ] Create comparison tutorial

---

## 🔑 Key Differences: SpliceAI vs OpenSpliceAI

| Aspect | SpliceAI | OpenSpliceAI |
|--------|----------|--------------|
| **Training Build** | GRCh37 | GRCh38 (likely) |
| **Annotations** | GENCODE V24lift37 | MANE (likely) |
| **Model Format** | Keras (.h5) | PyTorch (.pt) |
| **Models** | 5 files (~13MB) | 5 files (~13MB) |
| **Context** | 10,000 bp | 10,000 bp |
| **Our Setup** | ✅ Ready | ⏳ In Progress |

---

## 💡 Usage Examples

### Basic Usage

```python
from meta_spliceai import run_base_model_predictions

# SpliceAI (GRCh37)
spliceai_results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1', 'TP53']
)

# OpenSpliceAI (GRCh38)
openspliceai_results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1', 'TP53']
)
```

### Model Comparison

```python
from meta_spliceai import predict_splice_sites

# Quick predictions
spliceai_pos = predict_splice_sites('BRCA1', base_model='spliceai')
openspliceai_pos = predict_splice_sites('BRCA1', base_model='openspliceai')

print(f"SpliceAI:     {spliceai_pos.height} positions")
print(f"OpenSpliceAI: {openspliceai_pos.height} positions")
```

### Configuration

```python
from meta_spliceai import BaseModelConfig, run_base_model_predictions

# OpenSpliceAI with custom config
config = BaseModelConfig(
    base_model='openspliceai',
    gtf_file='data/mane/GRCh38/MANE.v1.0.gtf',
    genome_fasta='data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa',
    threshold=0.5,
    mode='test'
)

results = run_base_model_predictions(
    config=config,
    target_genes=['BRCA1']
)
```

---

## 🚨 Important Notes

### Genomic Build Compatibility

**SpliceAI**: Trained on GRCh37
- ✅ Use GRCh37 annotations
- ✅ Ensembl release 87
- ✅ Performance validated

**OpenSpliceAI**: Likely trained on GRCh38
- ⏳ Use GRCh38/MANE annotations
- ⏳ Performance to be validated
- ⚠️ Different coordinates from SpliceAI

### Direct Comparison Challenges

Because models use different builds:
- ❌ Can't directly compare coordinates
- ✅ Can compare F1 scores on same genes
- ✅ Can compare relative performance
- ⚠️ Need liftOver for coordinate mapping

---

## 📚 Related Documentation

- [Readiness Assessment](OPENSPLICEAI_READINESS_ASSESSMENT.md) - Detailed analysis
- [Base Model Guide](../tutorials/BASE_MODEL_PREDICTION_GUIDE.md) - User tutorial
- [OpenSpliceAI Adapter](../../meta_spliceai/splice_engine/meta_models/openspliceai_adapter/README.md) - Technical details

---

## ✅ Success Criteria

**Minimum**: OpenSpliceAI predictions work
- [ ] Models load
- [ ] Predictions run
- [ ] Results generated

**Complete**: Full integration
- [ ] All tests pass
- [ ] Performance validated
- [ ] Documentation complete

**Production**: Ready for users
- [ ] Examples provided
- [ ] Comparison tools available
- [ ] Best practices documented

---

**Ready to start?** Begin with Step 1: Download Models! 🚀



