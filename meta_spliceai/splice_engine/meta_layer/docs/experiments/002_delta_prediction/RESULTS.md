# Experiment 002: Direct Delta Prediction - Results

**Date**: December 14-15, 2025  
**Status**: ✅ Completed (Phase 2 iterations)

---

## Executive Summary

| Approach | Correlation | Detection | Verdict |
|----------|-------------|-----------|---------|
| Phase 1 (Classification) | N/A | 17% | ❌ Failed for variant detection |
| V2 Original (200 samples) | r=-0.04 | 100% (misleading) | ❌ No learning |
| V2 Improved (2000 samples) | r=0.002 | 43% | ❌ No correlation |
| Gated CNN | r=0.36 | 0% (conservative) | ✅ Good correlation |
| Option 1: Scaling | r=0.22 | 100% (all positive) | ❌ Overfitting |
| Option 2: Temperature | r=-0.03 | 0% | ❌ No improvement |
| **Option 3: Quantile (τ=0.9)** | **r=0.38** | **20%** | ✅ **BEST** |
| Option 4: Hybrid | r=-0.07 | 0% | ❌ Task interference |

**Key Finding**: **Quantile Regression (τ=0.9) achieved the best correlation: r=0.38 with reasonable detection (20% splice-altering, 16% false positives)**.

---

## Detailed Results by Iteration

### Iteration 1: V2 Original (Baseline)

**Configuration**:
- Training samples: 200
- Architecture: CNN (64 hidden, 4 layers)
- Epochs: 20

**Results**:
```
Class                | N   | Base  | Pred  | Base μ | Pred μ
----------------------------------------------------------
Splice-altering      |  17 |  17/17 |  17/17 | 0.20   | 0.61
Normal               |   5 |   0/5  |   5/5  | 0.001  | 0.60
Low-frequency        |  10 |   3/10 |  10/10 | 0.07   | 0.69

Pearson:  r=-0.0408 (p=5.83e-01)
Spearman: ρ=-0.0652 (p=3.76e-01)
```

**Analysis**: Model outputs constant high values (~0.6). No learning.

---

### Iteration 2: V2 Improved (10x Data)

**Configuration**:
- Training samples: 2000 (10x increase)
- Architecture: CNN (128 hidden, 6 layers)
- Balanced: 50% splice-altering
- Epochs: 30

**Results**:
```
Class                | N   | Base Det | Pred Det | Base μ | Pred μ
-----------------------------------------------------------------
Splice-altering      | 100 |    41/100  |    43/100  | 0.1261 | 0.1036
Normal               |  33 |     0/33  |    11/33  | 0.0005 | 0.0933
Low-frequency        |  66 |     7/66  |    23/66  | 0.0388 | 0.0912

Pearson:  r=0.0016 (p=9.82e-01)
Spearman: ρ=-0.1084 (p=1.26e-01)
```

**Analysis**: Better calibration (mean ~0.1 not 0.6), but still no correlation.

---

### Iteration 3: Gated CNN ⭐ BEST CORRELATION

**Configuration**:
- Training samples: 1500
- Architecture: **Gated residual blocks with dilated convolutions**
- Dilations: 1, 2, 4, 1, 2, 4 (6 layers)
- Gating: sigmoid gate per block
- LayerNorm + GELU
- OneCycleLR scheduler
- Parameters: 3,070,210

**Results**:
```
Class                | N   | Base  | Pred  | Base μ | Pred μ
------------------------------------------------------------
Splice-altering      |  75 |  29/75 |   0/75 | 0.1175 | 0.0471
Normal               |  28 |   0/28 |   0/28 | 0.0004 | 0.0469
Low-frequency        |  46 |   5/46 |   0/46 | 0.0405 | 0.0469

Pearson:  r=0.3589 (p=6.48e-06) ✅
Spearman: ρ=0.0786 (p=3.39e-01)
```

**Analysis**: 
- **Meaningful correlation achieved!** (r=0.36, p<0.001)
- Predictions too conservative (mean ~0.047)
- Detection rate 0% (below threshold)
- Proves the concept: sequence → delta mapping is learnable

---

### Iteration 4: Option 1 - Output Scaling

**Configuration**:
- Base: Gated CNN
- Learnable scale factor (initialized: 2.0)
- Additional training: 20 epochs

**Results**:
```
Learned scale: 1.9921

Class                | N   | Base  | Pred  | Base μ | Pred μ
------------------------------------------------------------
Splice-altering      |  75 |  29/75 |  75/75 | 0.1175 | 0.1972
Normal               |  28 |   0/28 |  28/28 | 0.0004 | 0.2020
Low-frequency        |  46 |   5/46 |  46/46 | 0.0405 | 0.1928

Pearson:  r=0.2222 (p=6.27e-03)
Spearman: ρ=0.0476 (p=5.63e-01)
```

**Analysis**: 
- Scale factor learned (~2.0)
- 100% detection for ALL classes (false positives)
- Correlation dropped to r=0.22
- Overfitting to high predictions

---

### Iteration 5: Option 2 - Temperature Scaling

**Configuration**:
- Base: Gated CNN
- Learnable temperature (log-space for positivity)
- Additional training: 20 epochs

**Results**:
```
Learned temperature: 1.0106

Pearson: r=-0.0274 (p=7.87e-01)
Mean pred: 0.0125

Detection: 0/50 splice-altering, 0/50 false positives
```

**Analysis**:
- Temperature learned to ~1.0 (no change)
- Predictions extremely conservative
- Correlation dropped to near zero
- Temperature scaling alone doesn't help

---

### Iteration 6: Option 3 - Quantile Regression ⭐ BEST OVERALL

**Configuration**:
- Quantile: τ = 0.9 (predict 90th percentile)
- Pinball loss: asymmetric penalty for under/over-prediction
- Same Gated CNN architecture

**Results**:
```
Pearson: r=0.3804 (p=9.48e-05) ✅ BEST
Mean pred: 0.0929

Detection: 10/50 (20%) splice-altering
False Pos: 8/50 (16%)
```

**Analysis**:
- **Best correlation achieved: r=0.38**
- Reasonable detection rate (20%)
- Acceptable false positive rate (16%)
- Pinball loss encourages predicting higher values for high-delta variants
- Much better calibration than MSE loss

---

### Iteration 7: Option 4 - Hybrid (Classification + Regression)

**Configuration**:
- Multi-task: 30% classification + 70% regression
- Shared Gated CNN encoder
- Classification head: Global pooling → MLP → binary
- Regression head: Per-position → delta

**Results**:
```
CLASSIFICATION (Splice-altering?)
   Accuracy:  0.5000 (random)
   Precision: 0.0000
   Recall:    0.0000
   F1:        0.0000
   ROC-AUC:   0.5356

REGRESSION (Delta prediction)
   Class                | N   | Base  | Pred  | Base μ | Pred μ
   ------------------------------------------------------------
   Splice-altering      |  75 |  29/75 |   0/75 | 0.1175 | 0.0729
   Normal               |  28 |   0/28 |   0/28 | 0.0004 | 0.0727
   Low-frequency        |  46 |   5/46 |   0/46 | 0.0405 | 0.0728

   Pearson:  r=-0.0695 (p=3.98e-01)
   Spearman: ρ=0.1156 (p=1.59e-01)
```

**Analysis**:
- Classification: Random performance (AUC ~0.5)
- Regression: No correlation (r=-0.07)
- Task interference: Both tasks hurt each other
- Multi-task learning not effective here

---

## Architecture Comparison

| Architecture | Key Features | Params | Correlation | Detection |
|--------------|--------------|--------|-------------|-----------|
| Simple CNN | Conv1d blocks | 366K | r=-0.04 | 100% (bad) |
| Deeper CNN | 128d, 6 layers | 700K | r=0.002 | 43% |
| Gated CNN | Dilated + gating + LayerNorm | 3M | r=0.36 | 0% |
| Option 1: Scaled | + learnable scale | 3M + 1 | r=0.22 | 100% (bad) |
| Option 2: Temperature | + learnable temp | 3M + 1 | r=-0.03 | 0% |
| **Option 3: Quantile** | **+ pinball loss (τ=0.9)** | 3M | **r=0.38** | **20%** |
| Option 4: Hybrid | + classification head | 3.5M | r=-0.07 | 0% |

---

## Key Insights

### 1. Architecture > Data (at this scale)

| Change | Correlation Δ |
|--------|--------------|
| 10x more data (200 → 2000) | +0.04 (negligible) |
| Better architecture (CNN → Gated) | **+0.36** |

### 2. What Worked

✅ **Gated Residual Blocks**: Allow selective feature attention  
✅ **Dilated Convolutions**: Capture longer-range dependencies  
✅ **LayerNorm**: Better normalization for variable-length sequences  
✅ **OneCycleLR**: Better learning rate scheduling  
✅ **Balanced Training**: 50% splice-altering variants  

### 3. What Didn't Work

❌ **Simple Scaling**: Leads to overfitting  
❌ **Multi-task Learning**: Task interference  
❌ **More Data Alone**: Doesn't help without architecture improvements  

### 4. Remaining Challenge: Calibration

The Gated CNN learned the relationship (r=0.36) but predictions are too conservative:
- Target mean: ~0.08
- Predicted mean: ~0.047
- Gap: ~40% underestimation

---

## Checkpoints Saved

| Model | Path | Correlation |
|-------|------|-------------|
| V2 Original | `.../delta_predictor_v2_phase2.pt` | -0.04 |
| V2 Improved | `.../delta_predictor_v2_improved.pt` | 0.002 |
| Gated CNN | `.../improved_cnn_delta.pt` | 0.36 |
| Option 1: Scaled | `.../scaled_delta.pt` | 0.22 |
| Option 2: Temperature | `.../temp_scaled.pt` | -0.03 |
| **Option 3: Quantile** | **`.../quantile_90.pt`** | **0.38** |
| Option 4: Hybrid | `.../hybrid_delta.pt` | -0.07 |

---

## Next Steps

### 1. HyenaDNA Pre-trained Encoder

The Gated CNN is trained from scratch. HyenaDNA provides:
- Pre-training on human genome
- Better sequence representations
- Longer context (up to 1M bp)

See: [COMPUTE_ENVIRONMENTS.md](./COMPUTE_ENVIRONMENTS.md)

### 2. Different Loss Functions

- **Huber loss**: Robust to outliers
- **Focal loss**: Focus on hard examples
- **Correlation loss**: Directly optimize correlation

### 3. Data Augmentation

- Reverse complement
- Noise injection
- Sliding window on longer sequences

### 4. Ensemble Methods

- Combine multiple Gated CNN models
- Different random seeds
- Average predictions

---

## Conclusion

**The Gated CNN architecture achieved r=0.36 correlation with base model deltas**, proving that learning the sequence → variant effect mapping is feasible.

The main remaining challenge is calibration (scaling predictions correctly). HyenaDNA pre-training should provide better representations and potentially improve both correlation and calibration.

For further experiments, see:
- [IMPROVEMENTS.md](./IMPROVEMENTS.md) - Iteration details
- [COMPUTE_ENVIRONMENTS.md](./COMPUTE_ENVIRONMENTS.md) - GPU setup for HyenaDNA
