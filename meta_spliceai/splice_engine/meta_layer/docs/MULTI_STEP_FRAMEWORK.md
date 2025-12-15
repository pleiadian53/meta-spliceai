# Multi-Step Framework for Alternative Splice Site Prediction

**Date**: December 15, 2025  
**Status**: Proposed Architecture

---

## Problem Statement

**Goal**: Predict if/how a genetic variant induces changes in splicing patterns.

**Challenge**: Directly predicting delta scores (continuous, per-position) is hard:
- Current best: r=0.38 correlation (not great)
- Regression on sparse signals is difficult
- Most variants don't affect splicing at all

---

## Proposed Multi-Step Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-STEP ALTERNATIVE SPLICE PREDICTION                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 1: SPLICE-INDUCING CLASSIFICATION (Binary)                      │    │
│  │                                                                      │    │
│  │ Question: "Does this variant affect splicing at all?"                │    │
│  │                                                                      │    │
│  │ Input:  variant context (alt_seq + variant_info)                     │    │
│  │ Output: P(splice-altering) ∈ [0, 1]                                  │    │
│  │                                                                      │    │
│  │ Training data: SpliceVarDB classifications                           │    │
│  │   - Positive: "Splice-altering" variants                            │    │
│  │   - Negative: "Normal" variants                                      │    │
│  │                                                                      │    │
│  │ Advantages:                                                          │    │
│  │   ✅ Binary classification is easier than regression                 │    │
│  │   ✅ Well-balanced with SpliceVarDB data                             │    │
│  │   ✅ Directly answers: "Should I investigate this variant?"          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                               │
│                    If P > threshold (e.g., 0.5)                             │
│                              ↓                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 2: EFFECT TYPE CLASSIFICATION (Multi-class)                     │    │
│  │                                                                      │    │
│  │ Question: "What TYPE of splicing change does this variant cause?"    │    │
│  │                                                                      │    │
│  │ Input:  variant context (same as Step 1)                             │    │
│  │ Output: Distribution over effect types                               │    │
│  │   - Donor gain (new donor created)                                   │    │
│  │   - Donor loss (existing donor disrupted)                            │    │
│  │   - Acceptor gain (new acceptor created)                             │    │
│  │   - Acceptor loss (existing acceptor disrupted)                      │    │
│  │   - Complex (multiple effects)                                       │    │
│  │                                                                      │    │
│  │ Training data: SpliceVarDB effect annotations                        │    │
│  │                                                                      │    │
│  │ Advantages:                                                          │    │
│  │   ✅ Directly interpretable for biologists                           │    │
│  │   ✅ Multi-class classification is tractable                         │    │
│  │   ✅ Guides downstream analysis                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                               │
│                    For splice-altering variants                             │
│                              ↓                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 3: POSITION LOCALIZATION (Regression/Attention)                 │    │
│  │                                                                      │    │
│  │ Question: "WHERE in the window does the effect occur?"               │    │
│  │                                                                      │    │
│  │ Input:  variant context + predicted effect type                      │    │
│  │ Output: Position-wise attention/probability                          │    │
│  │   - P(affected position) for each position in window                 │    │
│  │   - Or: Top-K most affected positions                                │    │
│  │                                                                      │    │
│  │ Architecture options:                                                │    │
│  │   a) Attention-based: Output attention weights over positions        │    │
│  │   b) Segmentation: Binary mask of affected positions                 │    │
│  │   c) Regression: Delta magnitude at each position                    │    │
│  │                                                                      │    │
│  │ Advantages:                                                          │    │
│  │   ✅ Conditioned on positive classification (cleaner signal)         │    │
│  │   ✅ Can use effect type as additional context                       │    │
│  │   ✅ Smaller output space (just positions, not full deltas)          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              ↓                                               │
│                    Optional refinement                                      │
│                              ↓                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 4: DELTA MAGNITUDE (Optional Regression)                        │    │
│  │                                                                      │    │
│  │ Question: "How STRONG is the effect at the identified position?"     │    │
│  │                                                                      │    │
│  │ Input:  variant context + position + effect type                     │    │
│  │ Output: Δ_donor, Δ_acceptor at specific position                     │    │
│  │                                                                      │    │
│  │ Advantages:                                                          │    │
│  │   ✅ Focused on known-affected positions only                        │    │
│  │   ✅ Much smaller problem than full [L, 2] regression                │    │
│  │   ✅ Can use cross-attention with position embedding                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Details

### Step 1: Splice-Inducing Classification

**Task**: Binary classification - Is this variant splice-altering?

```python
class SpliceInducingClassifier(nn.Module):
    """
    Binary classifier: P(variant affects splicing)
    
    Input: alt_sequence [B, 4, L] + variant_info [B, 8] (ref_base + alt_base one-hot)
    Output: P(splice-altering) [B, 1]
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = GatedCNNEncoder(hidden_dim)
        self.variant_embed = nn.Linear(8, hidden_dim)  # ref_base[4] + alt_base[4]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, alt_seq, ref_base, alt_base):
        seq_features = self.encoder(alt_seq)  # [B, H]
        var_features = self.variant_embed(torch.cat([ref_base, alt_base], dim=-1))
        combined = torch.cat([seq_features, var_features], dim=-1)
        return self.classifier(combined)
```

**Training Data**:
- Positive: SpliceVarDB "Splice-altering" variants
- Negative: SpliceVarDB "Normal" variants
- Balanced: ~50/50 split

**Evaluation Metrics**:
- ROC-AUC
- PR-AUC (better for imbalanced)
- F1-score

---

### Step 2: Effect Type Classification

**Task**: Multi-class classification - What type of effect?

```python
class EffectTypeClassifier(nn.Module):
    """
    Multi-class classifier: What type of splicing effect?
    
    Classes:
      0: Donor gain (new donor site created)
      1: Donor loss (existing donor disrupted)
      2: Acceptor gain (new acceptor site created)
      3: Acceptor loss (existing acceptor disrupted)
      4: Complex (multiple effects)
    """
    def __init__(self, hidden_dim=128, num_classes=5):
        super().__init__()
        self.encoder = GatedCNNEncoder(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, alt_seq, variant_info):
        features = self.encoder(alt_seq)
        return self.classifier(features)  # [B, 5]
```

**Training Data**:
- SpliceVarDB effect annotations
- May need to derive from delta scores if not directly available

---

### Step 3: Position Localization

**Task**: Identify which positions are affected

**Option A: Attention-based**
```python
class PositionLocalizer(nn.Module):
    """
    Output attention weights over positions.
    High attention = likely affected position.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = GatedCNNEncoder(hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, alt_seq):
        features = self.encoder.encode_per_position(alt_seq)  # [B, L, H]
        attention = F.softmax(self.attention(features).squeeze(-1), dim=-1)  # [B, L]
        return attention
```

**Option B: Binary segmentation**
```python
class PositionSegmenter(nn.Module):
    """
    Binary mask: Which positions are affected?
    """
    def forward(self, alt_seq):
        features = self.encoder(alt_seq)  # [B, L, H]
        mask_logits = self.head(features)  # [B, L, 1]
        return torch.sigmoid(mask_logits)
```

---

### Step 4: Delta Magnitude (Optional)

**Task**: Predict actual delta values at identified positions

```python
class DeltaMagnitudePredictor(nn.Module):
    """
    Given a position known to be affected, predict the delta magnitude.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = GatedCNNEncoder(hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, 2)  # [Δ_donor, Δ_acceptor]
    
    def forward(self, alt_seq, position_idx):
        features = self.encoder.encode_per_position(alt_seq)  # [B, L, H]
        position_features = features[:, position_idx, :]  # [B, H]
        return self.delta_head(position_features)  # [B, 2]
```

---

## Comparison: Current vs Proposed

| Aspect | Current (Single-Step) | Proposed (Multi-Step) |
|--------|----------------------|----------------------|
| **Task** | Predict [L, 2] deltas directly | Sequential classification → localization |
| **Difficulty** | Hard (sparse regression) | Easier (decomposed problems) |
| **Interpretability** | Low (just numbers) | High (yes/no → type → where) |
| **Training signal** | Weak (r=0.38 correlation) | Stronger (binary/multi-class labels) |
| **Inference** | One model | Multiple models (can be combined) |
| **Biology alignment** | ❌ Indirect | ✅ Matches how biologists think |

---

## Implementation Priority

### Phase 2A: Splice-Inducing Classification (Recommended First)

**Why start here?**
1. Binary classification is well-understood
2. Direct alignment with SpliceVarDB labels
3. Answers the most important question first
4. Can achieve high accuracy with current architecture
5. Results are immediately useful

**Training data**:
```python
# Directly from SpliceVarDB
positive = variants[variants['classification'] == 'Splice-altering']
negative = variants[variants['classification'] == 'Normal']
```

### Phase 2B: Effect Type Classification

**After successful 2A, add effect type prediction**
- Can be trained jointly with 2A (multi-task)
- Or as a separate model conditioned on P(splice-altering)

### Phase 2C: Position Localization

**Most challenging, save for later**
- Requires positional labels (where exactly is the effect?)
- May need to derive from base model delta analysis
- Could use attention interpretation from Steps 1/2

---

## Proposed Model: Unified Multi-Task

```python
class UnifiedSplicePredictor(nn.Module):
    """
    Multi-task model for splice variant analysis.
    
    Outputs:
      1. P(splice-altering): Binary probability
      2. Effect type distribution: [5] logits
      3. Position attention: [L] weights (interpretable)
    """
    def __init__(self, hidden_dim=128, context_length=101):
        super().__init__()
        
        # Shared encoder
        self.encoder = GatedCNNEncoder(hidden_dim, n_layers=6)
        
        # Variant embedding (ref_base + alt_base)
        self.variant_embed = nn.Linear(8, hidden_dim)
        
        # Task heads
        self.is_splice_altering = nn.Linear(hidden_dim, 1)  # Binary
        self.effect_type = nn.Linear(hidden_dim, 5)  # Multi-class
        self.position_attention = nn.Linear(hidden_dim, 1)  # Per-position
    
    def forward(self, alt_seq, ref_base_onehot, alt_base_onehot):
        # Encode sequence
        seq_features = self.encoder.encode_per_position(alt_seq)  # [B, L, H]
        
        # Global features (pooled)
        global_features = seq_features.mean(dim=1)  # [B, H]
        
        # Add variant info
        var_info = torch.cat([ref_base_onehot, alt_base_onehot], dim=-1)
        var_features = self.variant_embed(var_info)  # [B, H]
        combined = global_features + var_features
        
        # Task outputs
        p_splice_altering = torch.sigmoid(self.is_splice_altering(combined))  # [B, 1]
        effect_logits = self.effect_type(combined)  # [B, 5]
        position_attn = F.softmax(self.position_attention(seq_features).squeeze(-1), dim=-1)  # [B, L]
        
        return {
            'p_splice_altering': p_splice_altering,
            'effect_type_logits': effect_logits,
            'position_attention': position_attn
        }
```

---

## Summary

**Current approach** (Approach A - Paired Prediction):
- Takes ref_seq AND alt_seq
- Outputs [L, 2] deltas
- Achieves r=0.38 correlation
- Single-step, hard regression task

**Proposed multi-step framework**:
1. **Step 1**: Is this variant splice-altering? (Binary) ← **Start here**
2. **Step 2**: What type of effect? (Multi-class)
3. **Step 3**: Where is the effect? (Localization)
4. **Step 4**: How strong? (Regression, optional)

**Key insight**: Breaking down the problem into smaller, more tractable questions will likely yield better overall performance and more interpretable results.

---

## Alignment with Approach B

The proposed Step 1 (SpliceInducingClassifier) is closer to Approach B:
- Single forward pass (no ref_seq needed)
- Uses variant info (ref_base, alt_base) as features
- More efficient at inference

This can be extended to full Approach B by conditioning on effect type.

