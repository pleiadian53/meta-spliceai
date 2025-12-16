# Experiment 002: Improvement Iterations

**Date**: December 14, 2025

---

## Summary of Iterations

| Version | Train N | Model | Pearson r | Detection |
|---------|---------|-------|-----------|-----------|
| V2 Original | 200 | CNN (64d, 4L) | -0.04 | 100% (misleading) |
| V2 Improved | 2000 | CNN (128d, 6L) | 0.002 | 43% (vs 41% base) |
| **Improved CNN** | 1500 | **Gated CNN** | **0.36** ✅ | 0% (too conservative) |

---

## Iteration 1: Original V2 (Baseline)

**Configuration**:
- 200 training samples
- CNN encoder: 64 hidden dim, 4 layers
- 20 epochs

**Results**:
- Pearson r = -0.04 (no correlation)
- All predictions uniformly high (~0.6)
- 100% "detection" was misleading

**Problem**: Insufficient data, model outputs constant high values.

---

## Iteration 2: More Data (10x)

**Changes**:
- 2000 training samples (10x increase)
- Balanced: 50% splice-altering + 50% other
- Larger model: 128 hidden dim, 6 layers
- 30 epochs

**Results**:
- Pearson r = 0.002 (still no correlation)
- Mean predictions now reasonable (~0.1)
- Detection rate 43% (slightly better than base 41%)

**Problem**: More data helped calibration but not correlation.

---

## Iteration 3: Improved Architecture ⭐

**Changes**:
- **Gated residual blocks**: Learning to selectively attend
- **Dilated convolutions**: Larger receptive field (1, 2, 4)
- **GELU activation**: Smoother gradients
- **LayerNorm**: Better normalization for small batches
- **OneCycleLR**: Better learning rate scheduling
- 3M parameters (vs 366K before)

**Architecture**:
```python
class SimpleCNNDeltaPredictor:
    - Initial embedding (Conv1d)
    - 6x DilatedResidualBlock (with gating)
    - Per-position delta head
    
class DilatedResidualBlock:
    - Dilated Conv1d (dilation=1,2,4,1,2,4)
    - LayerNorm
    - GELU activation
    - Gating mechanism (sigmoid)
    - Residual connection
```

**Results**:
- **Pearson r = 0.36 (p < 0.001)** ✅
- Mean predictions ~0.047 (too low)
- Detection rate 0% (predictions below threshold)

**Analysis**:
- Model IS learning the relationship
- But predictions are too conservative
- Need to scale outputs or adjust training

---

## Key Insights

### 1. Architecture Matters More Than Data (At This Scale)

| Factor | Correlation Improvement |
|--------|------------------------|
| 10x more data | 0.04 → 0.002 (none) |
| Better architecture | 0.002 → **0.36** |

### 2. Gating + Dilated Convolutions Are Key

The combination of:
- **Gating**: Allows model to learn which features matter
- **Dilated convolutions**: Captures longer-range patterns
- **Residual connections**: Easier gradient flow

### 3. Calibration vs Correlation Trade-off

- V2 Original: Poor calibration, no correlation
- V2 Improved: Good calibration, no correlation
- Improved CNN: Poor calibration, **good correlation** ✅

---

## Recommendations for Further Improvement

### Option 1: Output Scaling

```python
# Scale predictions to match target distribution
delta_pred = delta_pred * 2.5  # Scale factor from training stats
```

### Option 2: Temperature Scaling

```python
# Learn a temperature parameter
delta_pred = delta_pred / self.temperature
```

### Option 3: Quantile Regression

```python
# Predict quantiles instead of mean
loss = quantile_loss(pred, target, tau=0.9)
```

### Option 4: Combined Classification + Regression

```python
# Multi-task: Classify + Regress
class_pred = self.classifier(diff)  # Binary: splice-altering?
delta_pred = self.regressor(diff)   # Continuous: how much?
```

---

## Files Created

| File | Description |
|------|-------------|
| `delta_predictor_v2.py` | Original V2 architecture |
| `hyenadna_delta_predictor.py` | Improved CNN + HyenaDNA wrapper |
| `delta_predictor_v2_phase2.pt` | Original checkpoint |
| `delta_predictor_v2_improved.pt` | 10x data checkpoint |
| `improved_cnn_delta.pt` | Best correlation checkpoint |

---

## Conclusion

**The improved CNN architecture with gating achieved meaningful correlation (r=0.36) with base model deltas**, proving that the sequence → delta mapping CAN be learned.

The remaining challenge is calibration: scaling predictions to match the target magnitude.

**Next steps**:
1. Apply output scaling
2. Try HyenaDNA pre-training (if available)
3. Consider classification formulation

