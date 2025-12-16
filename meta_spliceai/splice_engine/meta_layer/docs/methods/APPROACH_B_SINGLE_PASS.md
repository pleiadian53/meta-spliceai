# Approach B: Single-Pass Delta Prediction

**Status**: ✅ IMPLEMENTED & TESTED (Best Result: r=0.507) ⭐  
**Implementation**: `ValidatedDeltaPredictor`  
**Last Updated**: December 16, 2025

---

## Overview

Approach B predicts splice site delta scores in a **single forward pass**, using only the alternate sequence and variant information as input. Unlike Approach A (which requires both ref and alt sequences), Approach B encodes the variant change as input features.

### Key Distinction from Approach A

| Aspect | Approach A (Paired) | Approach B (Single-Pass) |
|--------|---------------------|--------------------------|
| **Input** | ref_seq + alt_seq | alt_seq + variant_info |
| **Target** | base_model(alt) - base_model(ref) | SpliceVarDB-derived delta |
| **Inference** | Two encoder passes | Single encoder pass |
| **Target source** | Base model (potentially wrong) | Ground truth (SpliceVarDB) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     APPROACH B: SINGLE-PASS DELTA                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ INPUT                                                               │  │
│  │                                                                     │  │
│  │   alt_sequence: "ACGT...A...ACGT" (501nt with variant embedded)    │  │
│  │   ref_base:     "G" → [0, 0, 1, 0]  (one-hot)                      │  │
│  │   alt_base:     "A" → [1, 0, 0, 0]  (one-hot)                      │  │
│  │   (optional) variant_position: center (250)                        │  │
│  │   (optional) base_scores: [donor, acceptor, neither] from base     │  │
│  │                                                                     │  │
│  └─────────────────────────────┬──────────────────────────────────────┘  │
│                                │                                         │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ ENCODER                                                             │  │
│  │                                                                     │  │
│  │   alt_seq ──→ [Gated CNN / HyenaDNA] ──→ seq_features [B, H]       │  │
│  │                                                                     │  │
│  │   variant_info ──→ [MLP] ──→ var_features [B, H]                   │  │
│  │       • ref_base [4]                                                │  │
│  │       • alt_base [4]                                                │  │
│  │       • (optional) base_scores [3]                                  │  │
│  │                                                                     │  │
│  └─────────────────────────────┬──────────────────────────────────────┘  │
│                                │                                         │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ FUSION + DELTA HEAD                                                 │  │
│  │                                                                     │  │
│  │   combined = concat(seq_features, var_features)                    │  │
│  │   Δ = delta_head(combined)  →  [Δ_donor, Δ_acceptor, Δ_neither]   │  │
│  │                                                                     │  │
│  └─────────────────────────────┬──────────────────────────────────────┘  │
│                                │                                         │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ OUTPUT                                                              │  │
│  │                                                                     │  │
│  │   Δ = [+0.33, -0.02, -0.31]                                        │  │
│  │                                                                     │  │
│  │   Interpretation:                                                   │  │
│  │     • Δ_donor = +0.33   → Donor GAIN                               │  │
│  │     • Δ_acceptor = -0.02 → Acceptor unchanged                      │  │
│  │     • Δ_neither = -0.31  → Less likely to be neither               │  │
│  │                                                                     │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## The Key Challenge: Deriving Delta Targets

SpliceVarDB provides **categorical labels** (Splice-altering, Normal, etc.), not numeric delta values. We need to define how to convert these to training targets.

### Option 1: Binary Delta

```python
# Simple: effect or no effect
if classification == 'Splice-altering':
    if effect_type == 'Donor_gain':
        delta = [+1.0, 0.0, -1.0]  # donor up, neither down
    elif effect_type == 'Donor_loss':
        delta = [-1.0, 0.0, +1.0]  # donor down, neither up
    # ... etc
else:
    delta = [0.0, 0.0, 0.0]  # no change
```

**Pros**: Simple, clear signal  
**Cons**: Loses magnitude information

### Option 2: Literature-Based Magnitudes

```python
# Use biologically-informed magnitudes
EFFECT_MAGNITUDES = {
    'Donor_gain': [+0.5, 0.0, -0.5],
    'Donor_loss': [-0.5, 0.0, +0.5],
    'Acceptor_gain': [0.0, +0.5, -0.5],
    'Acceptor_loss': [0.0, -0.5, +0.5],
    'Complex': [+0.3, +0.3, -0.6],  # both effects
}
```

**Pros**: More realistic targets  
**Cons**: Magnitudes are arbitrary

### Option 3: Use Base Model Deltas WHERE ACCURATE

```python
# Only use base model delta when SpliceVarDB confirms the effect
if splicevardb_confirms_effect(variant):
    delta = base_model(alt) - base_model(ref)  # trusted
else:
    delta = [0.0, 0.0, 0.0]  # base model was wrong
```

**Pros**: Uses real deltas, validated by ground truth  
**Cons**: Still dependent on base model for positive cases

### Option 4: Soft Labels from Classification

```python
# Convert categorical to soft distribution
# P(effect_type) from classification + confidence
if classification == 'Splice-altering':
    delta_donor = +0.5 * confidence
    delta_acceptor = 0.0
else:
    delta = [0.0, 0.0, 0.0]
```

**Pros**: Incorporates uncertainty  
**Cons**: Need confidence scores

### Recommended: Option 3 (Validated Base Model Deltas)

Use base model deltas **only when SpliceVarDB confirms** the variant has an effect. This filters out cases where the base model is wrong.

```python
def get_approach_b_target(variant, base_model, fasta):
    """
    Get delta target for Approach B.
    
    Only trust base model delta when SpliceVarDB confirms effect.
    """
    # Get sequences
    ref_seq = get_sequence(variant, fasta)
    alt_seq = apply_variant(ref_seq, variant)
    
    # Get base model delta
    base_delta = base_model(alt_seq) - base_model(ref_seq)
    
    if variant.classification == 'Splice-altering':
        # Base model delta is trusted (confirmed by SpliceVarDB)
        return base_delta
    elif variant.classification == 'Normal':
        # Should have no effect (even if base model says otherwise)
        return np.zeros(3)
    else:
        # Low-frequency, Conflicting - uncertain
        return None  # Exclude from training
```

---

## Inference Workflow

```python
# At inference time (single pass!)
def predict_variant_effect(model, variant, fasta, base_model):
    """
    Predict delta for a new variant.
    """
    # 1. Get alt sequence
    alt_seq = get_alt_sequence(variant, fasta)
    
    # 2. Encode variant info
    ref_base_onehot = one_hot(variant.ref_allele[0])
    alt_base_onehot = one_hot(variant.alt_allele[0])
    
    # 3. Single forward pass
    delta = model(alt_seq, ref_base_onehot, alt_base_onehot)
    
    # 4. Get base scores (optional, for final calculation)
    base_scores = base_model(alt_seq)
    
    # 5. Final prediction
    final_scores = base_scores + delta
    
    return {
        'delta': delta,
        'base_scores': base_scores,
        'final_scores': final_scores,
        'interpretation': interpret_delta(delta)
    }

def interpret_delta(delta, threshold=0.1):
    """Interpret delta values."""
    effects = []
    if delta[0] > threshold:
        effects.append('Donor gain')
    if delta[0] < -threshold:
        effects.append('Donor loss')
    if delta[1] > threshold:
        effects.append('Acceptor gain')
    if delta[1] < -threshold:
        effects.append('Acceptor loss')
    return effects if effects else ['No significant effect']
```

---

## Comparison to Multi-Step Framework

| Aspect | Approach B | Multi-Step Framework |
|--------|------------|---------------------|
| **Output** | Delta values [3] | Classification + localization |
| **Interpretability** | Requires threshold | Directly interpretable |
| **Training** | Single model | Multiple models (or multi-task) |
| **Use case** | Score adjustment | Decision making |

**They are complementary!**

- Use **Approach B** when you need to adjust splice site scores
- Use **Multi-Step** when you need binary decisions (is this variant problematic?)

---

## Implementation Plan

### Step 1: Data Preparation

```python
class ApproachBDataset(Dataset):
    """Dataset for Approach B training."""
    
    def __init__(self, variants, fasta, base_model):
        self.samples = []
        
        for v in variants:
            if v.classification in ['Splice-altering', 'Normal']:
                target = self._get_target(v, fasta, base_model)
                if target is not None:
                    self.samples.append({
                        'alt_seq': get_alt_sequence(v, fasta),
                        'ref_base': v.ref_allele[0],
                        'alt_base': v.alt_allele[0],
                        'target': target
                    })
    
    def _get_target(self, variant, fasta, base_model):
        if variant.classification == 'Normal':
            return np.zeros(3)
        else:
            # Get validated base model delta
            ref_seq = get_sequence(variant, fasta)
            alt_seq = apply_variant(ref_seq, variant)
            return base_model(alt_seq) - base_model(ref_seq)
```

### Step 2: Model Implementation

```python
class ApproachBDeltaPredictor(nn.Module):
    """
    Single-pass delta predictor (Approach B).
    """
    
    def __init__(self, hidden_dim=128, n_layers=6):
        super().__init__()
        
        # Sequence encoder
        self.encoder = GatedCNNEncoder(hidden_dim, n_layers)
        
        # Variant info encoder
        self.var_embed = nn.Sequential(
            nn.Linear(8, hidden_dim),  # ref_base[4] + alt_base[4]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Delta head
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [Δ_donor, Δ_acceptor, Δ_neither]
        )
    
    def forward(self, alt_seq, ref_base, alt_base):
        seq_features = self.encoder(alt_seq)
        var_features = self.var_embed(torch.cat([ref_base, alt_base], dim=-1))
        combined = torch.cat([seq_features, var_features], dim=-1)
        delta = self.delta_head(combined)
        return delta
```

### Step 3: Training

```python
def train_approach_b(model, train_loader, epochs=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        for batch in train_loader:
            alt_seq = batch['alt_seq']
            ref_base = batch['ref_base']
            alt_base = batch['alt_base']
            target = batch['target']
            
            pred = model(alt_seq, ref_base, alt_base)
            loss = F.mse_loss(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Step 4: Evaluation

Key metrics:
- **Per-position correlation** with true delta
- **Effect detection rate** (Δ > threshold matches SpliceVarDB)
- **PR-AUC** for binary effect detection

---

## Why Approach B May Work Better

1. **Ground-truth targets**: Uses SpliceVarDB to validate/filter base model deltas
2. **Efficient inference**: Single forward pass
3. **Explicit variant encoding**: Model knows what change occurred
4. **Not learning from errors**: Doesn't blindly copy base model mistakes

---

## Implementation Status ✅

**Implemented as**: `ValidatedDeltaPredictor` in `models/validated_delta_predictor.py`

### Results by Dataset Size

| Samples | Correlation | ROC-AUC | PR-AUC |
|---------|-------------|---------|--------|
| 2,000 | r=0.41 | 0.58 | 0.62 |
| **8,000** | **r=0.507** | **0.589** | **0.633** |

**Key Finding**: Scaling from 2000 → 8000 samples improved correlation by **+24%**.

### Comparison to Approach A

| Aspect | Approach A (Paired) | Approach B (ValidatedDelta) |
|--------|---------------------|----------------------------|
| Correlation | r=0.38 | **r=0.507** (+33%) |
| Target source | Base model (noisy) | SpliceVarDB-validated |
| Forward passes | 2 | 1 |
| Inference speed | Slower | Faster |

---

## Files

| File | Description |
|------|-------------|
| `models/validated_delta_predictor.py` | Model implementation |
| `tests/test_validated_delta_experiments.py` | Experiment runner |
| `docs/experiments/004_validated_delta/` | Results documentation |

---

## Next Steps

1. ✅ ~~Implement model~~ → `ValidatedDeltaPredictor`
2. ✅ ~~Test with 2000 samples~~ → r=0.41
3. ✅ ~~Scale to 8000 samples~~ → r=0.507
4. **Recommended**: Scale to full SpliceVarDB (~50K samples) on GPU
   - Expected improvement: r=0.55-0.60 (+10-15% from 8K)
   - Enables proper cross-validation
   - Better statistical power for evaluation
5. **Recommended**: Biological evaluation (see `docs/experiments/BIOLOGICAL_EVALUATION_PLAN.md`)
   - Evaluate splice site peak detection
   - Test variant-induced gain/loss detection
   - More interpretable than correlation alone
6. **Pending**: Test with HyenaDNA encoder (GPU required)
7. **Pending**: Cross-validation for robust estimates

---

*This approach is now the recommended method for delta prediction. See `docs/experiments/004_validated_delta/` for detailed results.*

