# Paired Delta Prediction (Siamese Architecture)

**Status**: ⚠️ DEPRECATED for variant delta prediction - See [Validated Delta](VALIDATED_DELTA_PREDICTION.md)  
**Best Result**: r=0.38 correlation (insufficient for variant effect detection)  
**Implementation**: `DeltaPredictorV2` in `models/delta_predictor_v2.py`  
**Last Updated**: December 17, 2025

---

## Executive Summary

Paired Delta Prediction uses a **Siamese (twin) architecture** to predict splice probability changes by comparing reference and alternate sequences. For **variant delta prediction specifically**, this approach achieved only r=0.38 correlation due to fundamental limitations in target quality.

**Why it was superseded for variant detection**: The targets come directly from base model predictions without validation, meaning we train on potentially incorrect labels.

### Important Distinction

The underlying concept of **per-position prediction** (`[L, 3]` output shape) remains valuable for a different task: **Meta-Recalibration** of base model splice site scores. See [META_RECALIBRATION.md](META_RECALIBRATION.md) for this approach.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TWO RELATED BUT DISTINCT TASKS                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  THIS DOCUMENT: Paired Delta for Variant Detection                      │
│  ────────────────────────────────────────────────                       │
│    Task:   Predict variant-induced splice changes                       │
│    Output: [L, 3] delta (or [3] summary)                               │
│    Status: ⚠️ Deprecated - use Validated Delta instead                 │
│                                                                         │
│  RELATED: Meta-Recalibration (see META_RECALIBRATION.md)               │
│  ────────────────────────────────────────────────────                   │
│    Task:   Improve per-position splice site predictions                 │
│    Output: [L, 3] recalibrated scores                                  │
│    Status: 🔬 Proposed - potentially valuable as upstream step          │
│                                                                         │
│  The Siamese architecture idea is NOT dead - it just serves a          │
│  different purpose (recalibration) than direct variant detection.      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Method Overview](#method-overview)
2. [Architecture](#architecture)
3. [Target Computation](#target-computation)
4. [Biological Interpretation](#biological-interpretation)
5. [Limitations](#limitations)
6. [Experimental Results](#experimental-results)
7. [Lessons Learned](#lessons-learned)
8. [Relationship to Other Methods](#relationship-to-other-methods)

---

## Method Overview

### Core Idea

The Siamese approach mirrors how biologists think about variant effects:

```
"What's different between the reference and variant sequence?"

    Reference:  ...ACGTACGT[G]ACGTACGT...
    Variant:    ...ACGTACGT[A]ACGTACGT...
                            ↑
                       Single nucleotide change
                            
    Question: How do splice site probabilities change?
```

### Architecture Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PAIRED (SIAMESE) DELTA PREDICTION                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ref_seq ──→ [Shared Encoder] ──→ ref_features ─┐                     │
│                                                   ├──→ diff ──→ [Head] │
│   alt_seq ──→ [Shared Encoder] ──→ alt_features ─┘                     │
│                                                                         │
│   Output: Δ = [Δ_donor, Δ_acceptor] per position                       │
│                                                                         │
│   Key Property: Same encoder weights for both sequences                │
│                 → Learns to detect DIFFERENCES, not absolute values    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Siamese?

| Design Choice | Rationale |
|--------------|-----------|
| **Shared encoder** | Forces model to learn comparable representations |
| **Difference operation** | Explicitly captures what changed |
| **Per-position output** | Localizes where splice effects occur |

---

## Architecture

### Implementation Variants Tested

#### V1: Global Pooling (Failed)
```python
# Output: [B, 2] single delta per variant
class DeltaPredictorV1:
    encoder: SimpleCNN → global_avg_pool → [B, H]
    diff: alt_embed - ref_embed
    delta_head: MLP → [B, 2]
```
**Result**: No learning (r ≈ 0)  
**Problem**: Lost positional information

#### V2: Per-Position (Marginal)
```python
# Output: [B, L, 2] delta per position
class DeltaPredictorV2:
    encoder: SimpleCNN → [B, H, L]
    diff: alt_features - ref_features
    delta_head: Conv1d → [B, 2, L]
```
**Result**: r ≈ 0.1  
**Problem**: Insufficient receptive field

#### V3: Gated CNN with Dilated Convolutions (Best)
```python
# Output: [B, L, 2] with better receptive field
class SimpleCNNDeltaPredictor:
    encoder: GatedCNN with dilations [1, 2, 4, 8, 16, 32]
    diff: alt_features - ref_features
    delta_head: Gated conv → [B, 2, L]
```
**Result**: r = 0.38  
**Improvement**: Larger receptive field captures splice context

---

## Target Computation

### How Training Targets Are Derived

This is where the fundamental limitation lies. Targets come from the base model:

```python
def compute_paired_delta_target(variant, fasta, base_model):
    """
    Compute delta target using base model predictions.
    
    ⚠️ LIMITATION: Base model predictions are used directly as targets.
    If base model is wrong, we train on incorrect labels!
    """
    # 1. Extract extended sequences (10K context for base model)
    center_pos = variant.position
    ref_seq = fasta[variant.chrom][center_pos - 5000 : center_pos + 5000]
    
    # 2. Create alt sequence
    var_offset = 5000  # Variant at center
    alt_seq = ref_seq[:var_offset] + variant.alt_allele + ref_seq[var_offset + 1:]
    
    # 3. Run base model on BOTH sequences
    ref_probs = base_model(ref_seq)  # [L_out, 3] - probabilities per position
    alt_probs = base_model(alt_seq)  # [L_out, 3]
    
    # 4. Compute delta at each position
    delta = alt_probs - ref_probs  # [L_out, 3]
    
    # 5. Extract window around variant (±50bp)
    center = len(delta) // 2
    target_window = delta[center - 50 : center + 51]  # [101, 3]
    
    return target_window[:, :2]  # [101, 2] - donor and acceptor only
```

### Target Format

| Component | Shape | Meaning |
|-----------|-------|---------|
| `target[i, 0]` | scalar | Δ_donor at position i |
| `target[i, 1]` | scalar | Δ_acceptor at position i |

### The Fundamental Problem

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WHY PAIRED TARGETS ARE UNRELIABLE                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CASE 1: Base model correctly identifies splice-altering variant        │
│    SpliceVarDB: "Splice-altering"                                       │
│    Base model:  Δ = [+0.4, 0]  ← Correct signal                        │
│    Target:      [+0.4, 0]      ← Good training example                 │
│                                                                         │
│  CASE 2: Base model misses a real effect                                │
│    SpliceVarDB: "Splice-altering"                                       │
│    Base model:  Δ = [0, 0]     ← FALSE NEGATIVE                        │
│    Target:      [0, 0]         ← Training on WRONG label!              │
│                                                                         │
│  CASE 3: Base model hallucinates an effect                              │
│    SpliceVarDB: "Normal"                                                │
│    Base model:  Δ = [+0.3, 0]  ← FALSE POSITIVE                        │
│    Target:      [+0.3, 0]      ← Training on WRONG label!              │
│                                                                         │
│  CONCLUSION: ~40% of base model predictions may be wrong                │
│              → We're training on noisy/incorrect labels                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Biological Interpretation

### What the Model Learns

Despite limitations, the Siamese approach does learn meaningful patterns:

| Learned Pattern | Biological Meaning |
|-----------------|-------------------|
| GT-AG motif changes | Core splice site disruption |
| Branch point alterations | Pre-mRNA splicing mechanism |
| ESE/ESS changes | Exonic splicing enhancers/silencers |
| Local sequence context | Position-dependent effects |

### Why Partial Success (r=0.38)?

1. **Motif recognition**: CNN learns canonical splice motifs (GT donor, AG acceptor)
2. **Context sensitivity**: Gated CNN captures surrounding sequence features
3. **Relative comparison**: Siamese architecture naturally computes differences

### Why Not Better?

1. **No explicit variant encoding**: Model doesn't know what base changed to what
2. **Target noise**: Training on base model's mistakes
3. **No ground truth filtering**: All variants treated equally regardless of evidence

---

## Limitations

### 1. Target Quality (Critical)

| Issue | Impact | Severity |
|-------|--------|----------|
| Base model false positives | Train to predict effects where none exist | **HIGH** |
| Base model false negatives | Train to predict no effect for real variants | **HIGH** |
| Magnitude errors | Learn wrong effect sizes | MEDIUM |
| Position errors | Learn wrong affected positions | MEDIUM |

### 2. Inference Efficiency

```python
# Two forward passes required at inference
ref_embed = encoder(ref_seq)  # Pass 1
alt_embed = encoder(alt_seq)  # Pass 2
delta = delta_head(alt_embed - ref_embed)

# Compare to Validated Delta: single pass
delta = model(alt_seq, ref_base, alt_base)  # Pass 1 only
```

### 3. Missing Variant Information

The model only sees sequences, not explicit variant info:

```
What the model sees:     ref_seq vs alt_seq
What it doesn't know:    Which base changed? (A→G? C→T?)
                         Where exactly is the variant?
                         Is this a known pathogenic position?
```

### 4. Biological Limitations

| Limitation | Biological Impact |
|------------|------------------|
| 501nt context only | May miss distant regulatory elements |
| SNVs only (current) | Doesn't handle indels well |
| No tissue context | Same prediction for all tissues |
| No evolutionary info | Doesn't leverage conservation |

---

## Experimental Results

### Performance Summary

| Configuration | Correlation | ROC-AUC | Notes |
|--------------|-------------|---------|-------|
| V1 Global Pool | r ≈ 0 | ~0.50 | No learning |
| V2 Per-Position | r ≈ 0.1 | ~0.52 | Minimal learning |
| V3 Gated CNN | r = 0.36 | ~0.56 | Reasonable |
| V3 + Quantile Calibration | **r = 0.38** | ~0.58 | Best paired result |

### Training Details

```python
# Dataset
train_size: ~2000 variants (balanced SA/Normal)
test_size: ~500 variants (chr21, chr22)

# Hyperparameters
optimizer: AdamW(lr=5e-5, weight_decay=0.02)
scheduler: OneCycleLR(max_lr=1e-3)
epochs: 20-30
batch_size: 32
context_window: 101nt (±50bp)

# Loss
loss = F.mse_loss(pred, target)
sample_weights: 2.0 for SA, 1.0 for Normal
```

---

## Lessons Learned

### Key Insights

1. **Target quality matters more than architecture**: Switching from paired to validated targets improved correlation from 0.38 → 0.507 (+33%)

2. **Explicit variant encoding helps**: Knowing ref→alt change provides useful signal

3. **Single pass is sufficient**: Reference sequence isn't needed if variant info is encoded

4. **Ground truth validation is essential**: SpliceVarDB classifications filter out noisy targets

### What This Approach Got Right

- ✅ Siamese architecture is intuitive and principled
- ✅ Gated CNN is effective for sequence encoding
- ✅ Per-position output captures local effects

### What This Approach Got Wrong

- ❌ Trusting base model predictions unconditionally
- ❌ Not leveraging ground truth labels for target filtering
- ❌ Missing explicit variant encoding

---

## Relationship to Other Methods

### Compared to Validated Delta Prediction

| Aspect | Paired Delta | Validated Delta |
|--------|-------------|-----------------|
| Target source | Base model (noisy) | SpliceVarDB-validated |
| Correlation | r = 0.38 | **r = 0.609** (+60%) |
| Forward passes | 2 | 1 |
| Variant encoding | Implicit (difference) | Explicit (one-hot) |
| Status | Deprecated | **Recommended** |

### Connection to Multi-Step Framework

Paired Delta predicts **continuous deltas** but doesn't answer discrete questions:

| Question | Paired Delta | Multi-Step |
|----------|-------------|------------|
| "Is this splice-altering?" | Requires threshold on Δ | Direct yes/no |
| "What type of effect?" | Infer from sign of Δ | Direct classification |
| "Where is the effect?" | Position with max Δ | Attention/localization |

**Recommendation**: For interpretable decisions, use Multi-Step Framework. For continuous score adjustment, use Validated Delta Prediction.

---

## Files

| File | Description | Status |
|------|-------------|--------|
| `models/delta_predictor_v2.py` | V2 implementation | Deprecated |
| `models/hyenadna_delta_predictor.py` | V3 Gated CNN | Deprecated |
| `docs/experiments/002_delta_prediction/` | Experiment logs | Historical |

---

## Conclusion

Paired Delta Prediction was an important stepping stone that revealed:

1. **Architecture is not the bottleneck** - Gated CNN works well
2. **Target quality is critical** - Base model predictions are too noisy
3. **Validation is essential** - SpliceVarDB provides the ground truth needed

This led directly to the development of **Validated Delta Prediction**, which addresses these limitations by using SpliceVarDB classifications to filter and validate training targets.

---

*This method is documented for historical reference. For new work, use Validated Delta Prediction or Multi-Step Framework.*

