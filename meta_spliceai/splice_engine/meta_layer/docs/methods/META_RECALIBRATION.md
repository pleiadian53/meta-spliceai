# Meta-Recalibration: Per-Position Splice Score Refinement

**Status**: 🔬 PROPOSED - High potential, not yet fully implemented  
**Output Shape**: `[L, 3]` (same as base model)  
**Purpose**: Improve per-position splice site probability estimates  
**Last Updated**: December 17, 2025

---

## Executive Summary

Meta-Recalibration is a **per-position refinement layer** that takes base model outputs and additional context to produce improved splice site probability estimates. Unlike variant delta prediction (which outputs a summary `[3]` vector), meta-recalibration preserves the full spatial resolution of the base model (`[L, 3]`).

**Key Insight**: This approach can serve as an **upstream improvement** to the base model, which then produces better delta targets for variant effect detection.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Architecture Options](#architecture-options)
3. [Relationship to Other Methods](#relationship-to-other-methods)
4. [Two-Stage Pipeline](#two-stage-pipeline)
5. [Training Strategy](#training-strategy)
6. [Expected Benefits](#expected-benefits)
7. [Implementation Plan](#implementation-plan)

---

## Motivation

### The Base Model Limitation

Base models (SpliceAI, OpenSpliceAI) are trained on annotated splice sites from gene annotations:

```
Base Model Training:
  Input:  DNA sequence
  Target: Known splice sites from RefSeq/GENCODE annotations
  Output: P(donor), P(acceptor), P(neither) at each position
```

**Limitations of base models:**
- Trained only on canonical splice sites in well-characterized genes
- May miss tissue-specific or condition-specific splice sites
- Calibration may be off (probabilities don't reflect true uncertainty)
- Don't incorporate additional evidence (conservation, RNA-seq, etc.)

### The Meta-Recalibration Opportunity

```
Meta-Recalibration:
  Input:  DNA sequence + base_model_scores [L, 3] + context features
  Target: Refined splice site annotations (or better labels)
  Output: Recalibrated P(donor), P(acceptor), P(neither) at each position
          Shape: [L, 3] - SAME as base model
```

**What we can add:**
- Base model outputs as features (not just targets)
- Evolutionary conservation scores
- RNA-seq splice junction evidence
- Flanking sequence context beyond base model's receptive field
- Chromatin accessibility (ATAC-seq)
- Known regulatory elements (ESE/ESS motifs)

---

## Architecture Options

### Option A: Residual Refinement

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RESIDUAL META-RECALIBRATION                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    sequence ──→ [Base Model] ──→ base_scores [L, 3]                    │
│         │                              │                                │
│         │                              │                                │
│         ▼                              ▼                                │
│    [Context Encoder]            [Score Encoder]                        │
│         │                              │                                │
│         └──────────┬───────────────────┘                               │
│                    │                                                    │
│                    ▼                                                    │
│              [Fusion Layer]                                            │
│                    │                                                    │
│                    ▼                                                    │
│              [Refinement Head]                                         │
│                    │                                                    │
│                    ▼                                                    │
│              correction [L, 3]                                         │
│                    │                                                    │
│                    ▼                                                    │
│    recalibrated = base_scores + α * correction   ← Residual connection │
│                                                                         │
│    Output: [L, 3] recalibrated probabilities                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Stable training (starts from base model performance)
- Learn corrections, not absolute values
- α can be learned or scheduled

### Option B: Full Replacement

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FULL META-RECALIBRATION                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    sequence ──→ [Base Model] ──→ base_scores [L, 3]                    │
│         │                              │                                │
│         ▼                              ▼                                │
│    [Sequence      ──────────────  [Base Score                          │
│     Encoder]                       Features]                           │
│         │                              │                                │
│         └──────────┬───────────────────┘                               │
│                    │                                                    │
│              [Meta Encoder]                                            │
│              (Transformer/CNN)                                         │
│                    │                                                    │
│                    ▼                                                    │
│    recalibrated_scores [L, 3]  ← Direct prediction                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Can learn to completely override base model when wrong
- More flexible
- May achieve better final performance

### Option C: Siamese with Score Fusion

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SIAMESE META-RECALIBRATION                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│    For variant analysis:                                                │
│                                                                         │
│    ref_seq ──→ [Base Model] ──→ ref_base_scores [L, 3]                 │
│       │                              │                                  │
│       ▼                              ▼                                  │
│    [Shared Meta Encoder] ──────→ ref_recalibrated [L, 3]               │
│                                                                         │
│    alt_seq ──→ [Base Model] ──→ alt_base_scores [L, 3]                 │
│       │                              │                                  │
│       ▼                              ▼                                  │
│    [Shared Meta Encoder] ──────→ alt_recalibrated [L, 3]               │
│                                                                         │
│    delta = alt_recalibrated - ref_recalibrated  [L, 3]                 │
│                                                                         │
│    → Use this improved delta for variant effect analysis               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Directly applicable to variant analysis
- Improved base → improved deltas → better variant effect targets

---

## Relationship to Other Methods

### Method Comparison

| Method | Input | Output | Task |
|--------|-------|--------|------|
| **Base Model** | sequence | `[L, 3]` | Splice site prediction |
| **Meta-Recalibration** | sequence + base_scores + context | `[L, 3]` | Improved splice site prediction |
| **Validated Delta** | alt_seq + variant_info | `[3]` | Variant effect magnitude |
| **Multi-Step** | alt_seq + variant_info | classification | Variant effect detection |

### How They Fit Together

```
                        COMPLETE META-LAYER ECOSYSTEM
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  LAYER 0: BASE MODEL                                                    │
│  ────────────────────                                                   │
│    SpliceAI / OpenSpliceAI                                              │
│    Output: base_scores [L, 3]                                           │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│  LAYER 1: META-RECALIBRATION (this method)                              │
│  ─────────────────────────────────────────                              │
│    Input: sequence + base_scores + context                              │
│    Output: recalibrated_scores [L, 3]                                   │
│    Benefit: Better per-position predictions                             │
│                                                                         │
│                              ↓                                          │
│                                                                         │
│  LAYER 2: VARIANT ANALYSIS (downstream methods)                         │
│  ──────────────────────────────────────────────                         │
│                                                                         │
│    ┌─────────────────────┐    ┌─────────────────────┐                  │
│    │ VALIDATED DELTA     │    │ MULTI-STEP          │                  │
│    │                     │    │                     │                  │
│    │ Using recalibrated  │    │ Step 1: Is it SA?   │                  │
│    │ scores for delta    │    │ Step 2: What type?  │                  │
│    │ computation         │    │ Step 3: Where?      │                  │
│    │                     │    │                     │                  │
│    │ → Better targets!   │    │ → Decisions         │                  │
│    └─────────────────────┘    └─────────────────────┘                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Two-Stage Pipeline

### Why Meta-Recalibration Improves Validated Delta

The current validated delta approach relies on base model for computing targets:

```python
# Current approach
if variant.classification == 'Splice-altering':
    target_delta = base_model(alt) - base_model(ref)  # Limited by base model quality
```

With meta-recalibration:

```python
# Improved approach
if variant.classification == 'Splice-altering':
    ref_recalibrated = meta_model(ref_seq, base_model(ref_seq))
    alt_recalibrated = meta_model(alt_seq, base_model(alt_seq))
    target_delta = alt_recalibrated - ref_recalibrated  # Better quality targets!
```

### Expected Improvement

| Aspect | Base Model Only | With Meta-Recalibration |
|--------|-----------------|------------------------|
| Splice site detection | ~95% accuracy | Potentially 97-98%+ |
| Delta target quality | Limited by base errors | Corrected errors |
| Validated Delta performance | r=0.609 | Potentially higher |
| False positive rate | ~5% | Potentially 2-3% |

---

## Training Strategy

### Option 1: Supervised (Requires Better Labels)

```python
# Train on improved annotations
target = improved_splice_annotations  # From RNA-seq, manual curation, etc.
loss = cross_entropy(recalibrated_scores, target)
```

**Challenge**: Where do better labels come from?
- RNA-seq splice junction evidence
- Cross-species conservation
- Manual curation
- Consensus of multiple tools

### Option 2: Self-Supervised / Consistency

```python
# Train on consistency objectives
# Example: Forward and reverse complement should agree
fwd_scores = meta_model(sequence)
rev_scores = meta_model(reverse_complement(sequence))

loss = consistency_loss(fwd_scores, rev_scores)
```

### Option 3: Weakly Supervised (SpliceVarDB-derived)

```python
# Use SpliceVarDB to derive better labels
# Where SpliceVarDB says "Normal", suppress any base model peaks
# Where SpliceVarDB says "Splice-altering", trust/enhance base model peaks

def get_meta_target(sequence, base_scores, splicevardb_label):
    if splicevardb_label == 'Normal':
        # Variant region should have NO splice effect
        # Target: suppress peaks near variant
        return suppress_local_peaks(base_scores, variant_position)
    elif splicevardb_label == 'Splice-altering':
        # Trust base model peaks
        return base_scores
```

### Option 4: Multi-Task Learning

```python
# Joint training: recalibration + classification
class MultiTaskMetaModel(nn.Module):
    def forward(self, seq, base_scores):
        features = self.encoder(seq, base_scores)
        
        # Task 1: Recalibrated scores
        recalibrated = self.score_head(features)  # [L, 3]
        
        # Task 2: Variant classification (at center)
        is_splice_altering = self.classifier(features[:, L//2, :])  # Binary
        
        return recalibrated, is_splice_altering

# Joint loss
loss = recalibration_loss + λ * classification_loss
```

---

## Expected Benefits

### 1. Better Base Predictions

| Scenario | Base Model | Meta-Recalibrated |
|----------|-----------|-------------------|
| Weak splice sites | Often missed | Enhanced detection |
| Cryptic sites | False positives | Suppressed if no evidence |
| Context-dependent | Ignored | Incorporated |

### 2. Better Delta Targets for Validated Delta

```
Current:
  Base model wrong → Delta target wrong → Meta model learns noise
  
With Meta-Recalibration:
  Base model wrong → Meta-recalibration corrects → Better delta target → Better variant detection
```

### 3. Interpretable Improvements

Unlike end-to-end black boxes, we can:
- Compare base vs recalibrated scores
- Identify where meta-layer disagrees with base
- Understand what features drive corrections

---

## Implementation Plan

### Phase 1: Baseline Meta-Recalibration

```python
class MetaRecalibrationModel(nn.Module):
    """
    Residual meta-recalibration layer.
    
    Takes base model scores and sequence, outputs refined scores.
    """
    def __init__(self, hidden_dim=128, n_layers=6):
        super().__init__()
        
        # Sequence encoder
        self.seq_encoder = GatedCNNEncoder(hidden_dim, n_layers)
        
        # Base score encoder
        self.score_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Correction head (outputs residual)
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        # Learnable residual weight
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, sequence, base_scores):
        """
        Parameters
        ----------
        sequence : torch.Tensor [B, 4, L]
            One-hot encoded DNA sequence
        base_scores : torch.Tensor [B, L, 3]
            Base model probability predictions
        
        Returns
        -------
        torch.Tensor [B, L, 3]
            Recalibrated probability predictions
        """
        # Encode sequence
        seq_features = self.seq_encoder.encode_per_position(sequence)  # [B, L, H]
        
        # Encode base scores
        score_features = self.score_encoder(base_scores)  # [B, L, H]
        
        # Fuse
        combined = torch.cat([seq_features, score_features], dim=-1)
        fused = self.fusion(combined)  # [B, L, H]
        
        # Predict correction
        correction = self.correction_head(fused)  # [B, L, 3]
        
        # Apply residual correction
        recalibrated = base_scores + self.alpha * correction
        
        # Re-normalize to valid probabilities
        recalibrated = F.softmax(recalibrated, dim=-1)
        
        return recalibrated
```

### Phase 2: Integration with Validated Delta

```python
def compute_recalibrated_delta_target(variant, fasta, base_models, meta_model, device):
    """
    Compute delta target using meta-recalibrated scores instead of base model.
    """
    # Get sequences
    ref_seq = get_sequence(variant, fasta)
    alt_seq = apply_variant(ref_seq, variant)
    
    # Get base model scores
    ref_base = base_model_predict(ref_seq, base_models, device)
    alt_base = base_model_predict(alt_seq, base_models, device)
    
    # Apply meta-recalibration
    ref_recalibrated = meta_model(ref_seq, ref_base)
    alt_recalibrated = meta_model(alt_seq, alt_base)
    
    # Compute delta from recalibrated scores
    delta = alt_recalibrated - ref_recalibrated
    
    # Apply validated delta logic
    if variant.classification == 'Splice-altering':
        return extract_max_delta(delta)  # Use recalibrated delta
    elif variant.classification == 'Normal':
        return np.zeros(3)  # Still override to zero
    else:
        return None
```

### Phase 3: Joint Training

Train meta-recalibration and validated delta together in a multi-task setup.

---

## Open Questions

1. **What's the best training signal for meta-recalibration?**
   - RNA-seq evidence?
   - Cross-species conservation?
   - SpliceVarDB-derived weak labels?

2. **Should we use the same architecture as base model or different?**
   - Similar: can leverage transfer learning
   - Different: may capture complementary patterns

3. **How much context is needed?**
   - Base model uses 10K flanking
   - Meta-layer could use shorter (501nt) or longer (HyenaDNA 32K)

4. **Can we use meta-recalibration without running base model at inference?**
   - Distillation: train meta-model to predict recalibrated scores directly from sequence

---

## Files

| File | Description | Status |
|------|-------------|--------|
| `models/meta_recalibration.py` | Model implementation | ⏳ TODO |
| `tests/test_meta_recalibration.py` | Training experiments | ⏳ TODO |

---

## Summary

Meta-Recalibration addresses a different task than Validated Delta:

| Aspect | Meta-Recalibration | Validated Delta |
|--------|-------------------|-----------------|
| **Output** | `[L, 3]` per-position | `[3]` per variant |
| **Task** | Improve splice site detection | Predict variant effect |
| **Relationship** | Upstream (can improve delta targets) | Downstream (uses base/meta scores) |

**Key insight from user**: The value of per-position prediction (`[L, 3]`) is that it can serve as an improved base for computing deltas. Rather than relying on potentially flawed base model outputs for delta targets, we can use a meta-recalibrated model to get better delta targets, which then improves validated delta training.

---

*This document describes a proposed direction. See `VALIDATED_DELTA_PREDICTION.md` for the current recommended approach.*

