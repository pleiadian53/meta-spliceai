# GPU Experiment Wishlist

**Purpose**: Track expensive experiments that require GPU resources (RunPods)  
**Created**: December 15, 2025  
**Status**: Pending GPU execution

---

## 🎯 Experiments Requiring GPU

### 1. HyenaDNA Encoder Integration

| Experiment | Model | VRAM | Time Est. | Priority |
|------------|-------|------|-----------|----------|
| HyenaDNA-small + ValidatedDelta | hyenadna-small-32k | 12GB | 2-4h | ⭐ HIGH |
| HyenaDNA-small + Quantile Loss | hyenadna-small-32k | 12GB | 2-4h | HIGH |
| HyenaDNA-medium + ValidatedDelta | hyenadna-medium-160k | 24GB | 4-8h | MEDIUM |
| HyenaDNA-large (fine-tuning) | hyenadna-large-1m | 48GB+ | 12-24h | LOW |

**Expected Improvement**: r=0.41 → r>0.55

**Command**:
```python
from meta_spliceai.splice_engine.meta_layer.models import HyenaDNADeltaPredictor

model = HyenaDNADeltaPredictor(
    model_name='hyenadna-small-32k',
    hidden_dim=256,
    freeze_encoder=True
).to('cuda')
```

---

### 2. Full SpliceVarDB Dataset

| Experiment | Samples | VRAM | Time Est. | Priority |
|------------|---------|------|-----------|----------|
| ValidatedDelta 10K | 10,000 | 8GB | 1-2h | ⭐ HIGH |
| ValidatedDelta 25K | 25,000 | 12GB | 3-5h | HIGH |
| ValidatedDelta 50K (full) | ~50,000 | 16GB | 6-10h | MEDIUM |
| SimpleCNN 50K | ~50,000 | 12GB | 4-6h | MEDIUM |

**Current**: Only tested on 2,000 samples on M1 Mac

**Expected Improvement**: More data should improve generalization

---

### 3. Longer Context Windows

| Experiment | Context | VRAM | Time Est. | Priority |
|------------|---------|------|-----------|----------|
| SimpleCNN 501nt | 501nt | 12GB | 2-3h | HIGH |
| SimpleCNN 1001nt | 1001nt | 16GB | 4-6h | MEDIUM |
| HyenaDNA 10Knt | 10,000nt | 24GB | 8-12h | LOW |

**Current**: 101nt context may miss distant splice effects

---

### 4. Cross-Validation

| Experiment | Folds | VRAM | Time Est. | Priority |
|------------|-------|------|-----------|----------|
| 5-fold CV ValidatedDelta | 5 | 8GB | 5-8h | HIGH |
| 5-fold CV HyenaDNA | 5 | 16GB | 15-20h | MEDIUM |
| Leave-one-chromosome-out | 22 | 8GB | 20-40h | LOW |

**Purpose**: Robust performance estimates

---

### 5. Multi-Step Framework (Steps 2-4)

| Experiment | Task | VRAM | Time Est. | Priority |
|------------|------|------|-----------|----------|
| Effect Type Classification | 4-class | 8GB | 1-2h | MEDIUM |
| Position Localization | Regression | 8GB | 2-3h | MEDIUM |
| Delta Magnitude | Regression | 8GB | 2-3h | LOW |
| Unified Multi-Task | All tasks | 16GB | 6-10h | MEDIUM |

**Current**: Only Step 1 (Binary Classification) tested, AUC=0.61

---

### 6. Ensemble Methods

| Experiment | Components | VRAM | Time Est. | Priority |
|------------|------------|------|-----------|----------|
| Quantile Ensemble (50, 75, 90) | 3 models | 8GB | 3-4h | MEDIUM |
| CNN + HyenaDNA Ensemble | 2 models | 20GB | 6-8h | LOW |

---

## 📊 GPU Recommendations

| GPU | VRAM | Suitable Experiments |
|-----|------|----------------------|
| RTX 3090/4090 | 24GB | HyenaDNA-small, full dataset, CV |
| A40 | 48GB | HyenaDNA-medium, long context |
| A100 | 80GB | HyenaDNA-large fine-tuning |

### RunPods Pricing Reference (as of Dec 2025)

| GPU | $/hr | Recommended For |
|-----|------|-----------------|
| RTX 4090 | ~$0.44 | Most experiments |
| A40 | ~$0.79 | HyenaDNA-medium |
| A100 (80GB) | ~$1.89 | Fine-tuning only |

---

## ✅ Completed Experiments

Move experiments here after completion:

| Experiment | Date | Result | Notes |
|------------|------|--------|-------|
| *None yet* | - | - | - |

---

## 📝 How to Add New Experiments

1. Add to appropriate section above
2. Specify: Model, VRAM, Time estimate, Priority
3. After completion, move to "Completed" section with results

---

*This wishlist is tracked in the public repo for GPU experiment planning.*

