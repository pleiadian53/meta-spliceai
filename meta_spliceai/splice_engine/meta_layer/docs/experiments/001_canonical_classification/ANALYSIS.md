# Analysis: Why Canonical Classification Doesn't Translate to Variant Detection

**Experiment**: 001_canonical_classification  
**Date**: December 14, 2025

---

## The Core Problem

The meta-layer was trained to answer:
> "Is this position a donor site, acceptor site, or neither?"

But variant effect detection asks:
> "How does this variant CHANGE the probability of splice sites?"

These are fundamentally different questions.

---

## Detailed Analysis

### 1. Training Objective Mismatch

**What we trained**:
```
Input:  [501nt context, 43 features]
Output: P(donor), P(acceptor), P(neither)
Loss:   CrossEntropyLoss(predicted_class, GTF_label)
```

**What variant detection needs**:
```
Input:  [ref_context, alt_context]
Output: Δ_donor, Δ_acceptor (change in probability)
Target: Large |Δ| for splice-altering variants
```

The model learned to classify positions, not to detect changes. A position that's "definitely not a splice site" (P(neither) ≈ 1.0) will still output the same prediction whether we give it the reference or alternate sequence.

### 2. Context Insensitivity to Single Variants

Consider a 501nt context window:
```
...ACGTACGTACGTACGT[G→A]ACGTACGTACGT...
     ↑                ↑                ↑
   -250             center           +250
```

A single nucleotide change represents **0.2%** of the input. The model sees:
- Reference: 501nt with 'G' at center
- Alternate: 501nt with 'A' at center

With 500 identical nucleotides, the model's internal representations are nearly identical. The output probabilities barely change.

**Evidence**:
```
Variant: chr21:BRCA1 splice site
Base model delta:  0.3523  (correctly detects change)
Meta-layer delta:  0.1060  (70% smaller signal)
```

### 3. Position-Centric vs. Sequence-Level Architecture

**Base Model (OpenSpliceAI)**:
```
Input:  [L nucleotides]
Output: [L, 3] predictions for ALL positions
Delta:  alt_scores - ref_scores = [L, 3] deltas
Result: Can find MAX |delta| anywhere in window
```

**Meta-Layer (Position-Centric)**:
```
Input:  [501nt centered on position i]
Output: [3] prediction for position i only
Delta:  alt_prob - ref_prob = [3] single delta
Result: Only measures change at ONE position
```

The base model's architecture is inherently suited for variant effect detection because it can detect splice site creation/destruction anywhere in the sequence. The meta-layer only measures changes at the exact center position.

### 4. No Variant-Aware Training Signal

The training data consists of:
- Canonical splice sites from GTF annotations
- Subsampled non-splice positions

SpliceVarDB variants and their classifications were NOT used during training. The model has never seen:
- What a "splice-altering" variant looks like
- How reference and alternate sequences differ
- What delta score pattern indicates a splicing effect

---

## Quantitative Analysis

### Delta Score Distribution

| Variant Class | N | Base Mean |Δ| | Meta Mean |Δ| | Ratio |
|---------------|---|-----------|---|--------------|-------|
| Splice-altering | 6 | 0.1933 | 0.0228 | 0.12x |
| Normal | 1 | 0.0004 | 0.0010 | 2.5x |
| Low-frequency | 13 | 0.0415 | 0.0014 | 0.03x |

The meta-layer's delta scores are an **order of magnitude smaller** for splice-altering variants.

### Why Base Model Works Better

The base model was specifically designed for variant effect prediction:

1. **Output all positions**: Enables finding MAX delta in a window
2. **Trained on variant data**: SpliceAI was trained with variant-aware objectives
3. **Long context**: 10,000nt context captures distant effects
4. **Direct probability output**: No intermediate classification step

---

## Visualizing the Problem

```
                    CANONICAL CLASSIFICATION
                    ========================
                    
Training:           Test (Same Task):
┌─────────────────┐ ┌─────────────────┐
│ Position + GTF  │ │ Position + GTF  │
│     label       │ │     label       │
└────────┬────────┘ └────────┬────────┘
         │                   │
         ▼                   ▼
    [Meta-Layer]        [Meta-Layer]
         │                   │
         ▼                   ▼
    Classification      Classification
    Accuracy: 99.11%    ← SAME TASK ✅


                    VARIANT EFFECT DETECTION
                    ========================
                    
Training:               Test (Different Task):
┌─────────────────┐     ┌──────────────────────┐
│ Position + GTF  │     │ ref_seq vs alt_seq   │
│     label       │     │ SpliceVarDB label    │
└────────┬────────┘     └──────────┬───────────┘
         │                         │
         ▼                         ▼
    [Meta-Layer]              [Meta-Layer]
         │                    (ref) → P_ref
         ▼                    (alt) → P_alt
    Classification                  │
    (learned)                       ▼
                             Delta = P_alt - P_ref
                             Detection: 17%
                             ← DIFFERENT TASK ❌
```

---

## Implications for Phase 2

To improve variant effect detection, we need to:

### Option A: Explicit Delta Training

Train the meta-layer to predict delta scores directly:
```python
# Training objective
loss = MSE(
    predicted_delta,
    base_model_delta  # or SpliceVarDB-derived target
)
```

### Option B: Contrastive Learning

Train on (ref, alt) pairs with a contrastive objective:
```python
# Push apart ref and alt embeddings for splice-altering variants
# Keep similar for normal variants
loss = ContrastiveLoss(
    ref_embedding,
    alt_embedding,
    label=is_splice_altering
)
```

### Option C: Sequence-to-Sequence Architecture

Use an architecture that outputs predictions for all positions:
```python
# Model outputs [L, 3] like base model
meta_output = meta_layer(sequence)  # [L, 3]
meta_delta = meta_layer(alt_seq) - meta_layer(ref_seq)  # [L, 3]
max_delta = meta_delta.abs().max()
```

See `002_delta_prediction/` for implementation of these approaches.

---

## Conclusion

The canonical classification approach successfully improves base model accuracy on the classification task it was trained for. However, this does not translate to improved variant effect detection because:

1. **Different objective**: Classification ≠ change detection
2. **Insensitive architecture**: Single-position output cannot capture window-level effects
3. **Missing training signal**: No exposure to variant-specific patterns

The key insight is that **training objective must match evaluation objective**. For variant effect detection, we need variant-aware training.

