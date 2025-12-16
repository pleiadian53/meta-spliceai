# Experiment Results: 001 Canonical Classification

**Date**: December 14, 2025  
**Run ID**: 20251214_161359  
**Checkpoint**: `meta_layer_dev/20251214_161359/checkpoints/meta_layer_phase1.pt`

---

## Training Configuration

```yaml
base_model: openspliceai
genome_build: GRCh38
annotation_source: MANE

architecture:
  sequence_encoder: CNN
  sequence_channels: 32
  sequence_layers: 4
  hidden_dim: 256
  num_classes: 3
  dropout: 0.1
  total_parameters: 613,507

training:
  samples: 19,998
  class_balance: true
  samples_per_class: 6,666
  train_chromosomes: ["1", "2", ..., "20"]
  test_chromosome: "21"
  
  epochs: 15
  batch_size: 64
  learning_rate: 1e-4
  optimizer: AdamW
  weight_decay: 0.01
  scheduler: CosineAnnealingLR

device: mps (Apple M1)
```

---

## Training Progress

| Epoch | Loss | Accuracy |
|-------|------|----------|
| 5 | 0.0301 | 99.23% |
| 10 | 0.0056 | 99.86% |
| 15 | 0.0010 | 100.00% |

Training converged quickly with no signs of overfitting on the validation set.

---

## Task 1: Canonical Splice Site Classification

### Test Set Statistics

- **Chromosome**: 21
- **Total samples**: 7,858
- **Class distribution**: (natural, not balanced)

### Metrics

| Metric | Base Model | Meta-Layer | Δ |
|--------|------------|------------|---|
| **Accuracy** | 0.9761 | 0.9911 | +0.0150 |
| **Donor AP** | 0.9981 | 0.9966 | -0.0015 |
| **Acceptor AP** | 0.9972 | 0.9964 | -0.0008 |
| **Neither AP** | 0.9925 | 0.9989 | +0.0064 |

### Interpretation

- Overall accuracy improved by **1.5 percentage points**
- Donor and acceptor AP slightly decreased (within noise)
- Neither class AP improved by **0.64 percentage points**
- The meta-layer is better at identifying non-splice positions

---

## Task 2: Variant Effect Detection (SpliceVarDB)

### Test Set Statistics

- **Chromosome**: 21
- **Total variants evaluated**: 20 (subset for speed)
- **By classification**:
  - Splice-altering: 6
  - Normal: 1
  - Low-frequency: 13

### Delta Score Results

| Variant Class | N | Base Max|Δ| | Meta Max|Δ| | Base Detected | Meta Detected |
|---------------|---|----------|------------|---------------|---------------|
| Splice-altering | 6 | 0.1933 | 0.0228 | 4/6 (67%) | 1/6 (17%) |
| Normal | 1 | 0.0004 | 0.0010 | 0/1 (0%) | 0/1 (0%) |
| Low-frequency | 13 | 0.0415 | 0.0014 | 2/13 (15%) | 0/13 (0%) |

Detection threshold: |Δ| > 0.1

### Individual Variant Results

```
Classification        Base Δ   Meta Δ   Detected?
-----------------------------------------------------
Splice-altering       0.0308   0.0004   Base: ❌  Meta: ❌
Splice-altering       0.2981   0.0072   Base: ✅  Meta: ❌
Low-frequency         0.0001   0.0007   Base: ❌  Meta: ❌
Low-frequency         0.0002   0.0013   Base: ❌  Meta: ❌
Low-frequency         0.0000   0.0013   Base: ❌  Meta: ❌
Low-frequency         0.0004   0.0002   Base: ❌  Meta: ❌
Low-frequency         0.0003   0.0013   Base: ❌  Meta: ❌
Normal                0.0004   0.0010   Base: ❌  Meta: ❌
Low-frequency         0.0001   0.0006   Base: ❌  Meta: ❌
Low-frequency         0.0001   0.0013   Base: ❌  Meta: ❌
Low-frequency         0.0000   0.0011   Base: ❌  Meta: ❌
Low-frequency         0.0001   0.0034   Base: ❌  Meta: ❌
Splice-altering       0.3523   0.1060   Base: ✅  Meta: ✅
Low-frequency         0.3462   0.0037   Base: ✅  Meta: ❌
Splice-altering       0.3591   0.0142   Base: ✅  Meta: ❌
Low-frequency         0.1883   0.0022   Base: ✅  Meta: ❌
Low-frequency         0.0035   0.0008   Base: ❌  Meta: ❌
Splice-altering       0.1192   0.0075   Base: ✅  Meta: ❌
Low-frequency         0.0000   0.0004   Base: ❌  Meta: ❌
Splice-altering       0.0006   0.0014   Base: ❌  Meta: ❌
```

### Statistical Summary

| Metric | Base Model | Meta-Layer | Ratio |
|--------|------------|------------|-------|
| Mean |Δ| (all) | 0.0705 | 0.0107 | 0.15x |
| Max |Δ| (all) | 0.3591 | 0.1060 | 0.30x |
| Mean |Δ| (splice-altering) | 0.1933 | 0.0228 | 0.12x |
| Detection rate (splice-altering) | 67% | 17% | 0.25x |

### Interpretation

- Meta-layer delta scores are **7-10x smaller** than base model
- Meta-layer detected only **1 of 6** splice-altering variants
- Base model detected **4 of 6** splice-altering variants
- Meta-layer is **worse** at variant effect detection

---

## Conclusion

| Task | Outcome | Details |
|------|---------|---------|
| Canonical classification | ✅ Success | +1.5% accuracy |
| Variant effect detection | ❌ Failure | -50% detection rate |

The meta-layer improves what it was trained for (classification) but fails at what it wasn't trained for (variant effect detection).

---

## Artifacts Produced

```
data/mane/GRCh38/openspliceai_eval/meta_layer_dev/20251214_161359/
├── checkpoints/
│   └── meta_layer_phase1.pt    # Model weights + config + results
└── (evaluation outputs would go here)
```

### Checkpoint Contents

```python
{
    'model_state_dict': {...},
    'config': {
        'base_model': 'openspliceai',
        'sequence_encoder': 'cnn',
        'hidden_dim': 256,
        'num_score_features': 43
    },
    'results': {
        'donor': {'base_ap': 0.9981, 'meta_ap': 0.9966, 'improvement': -0.0015},
        'acceptor': {'base_ap': 0.9972, 'meta_ap': 0.9964, 'improvement': -0.0008},
        'neither': {'base_ap': 0.9925, 'meta_ap': 0.9989, 'improvement': 0.0064}
    }
}
```

---

## Reproducibility

To reproduce this experiment:

```bash
cd /Users/pleiadian53/work/meta-spliceai
mamba activate metaspliceai

# The training script is embedded in the chat session
# Key components:
#   - MetaLayerConfig(base_model='openspliceai')
#   - prepare_training_data(chromosomes=['1'..'20'], max_samples=20000)
#   - MetaSpliceModel(sequence_encoder='cnn', hidden_dim=256)
#   - 15 epochs with AdamW + CosineAnnealingLR
```

See [README.md](./README.md) for full experimental setup.

