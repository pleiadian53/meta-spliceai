# Multi-Step Framework for Alternative Splice Site Prediction

**Date**: December 15, 2025  
**Updated**: December 16, 2025  
**Status**: Partially Implemented (Step 1 tested, Steps 2-4 pending)

---

## 🎯 Why Multi-Step May Be Best for Drug Discovery

| Capability | ValidatedDelta | Multi-Step | Why Multi-Step Wins |
|------------|---------------|------------|---------------------|
| **Variant Triage** | ⚠️ Needs threshold | ✅ Direct yes/no | Fast screening of 1000s of variants |
| **Effect Interpretation** | Requires analysis | ✅ Direct output | "Donor gain" not "Δ=0.35" |
| **ASO Target Design** | ⚠️ Indirect | ✅ Position + type | Know WHERE to target |
| **Regulatory Approval** | ⚠️ Black box | ✅ Explainable | Each step is interpretable |

**Key Insight**: For drug discovery, you need to EXPLAIN predictions to stakeholders. Multi-Step provides a decision trail.

---

## Problem Statement

**Goal**: Predict if/how a genetic variant induces changes in splicing patterns.

**Challenge**: Directly predicting delta scores (continuous, per-position) is hard:
- ValidatedDelta achieves r=0.507 (good but not perfect)
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

**Task**: Identify which positions are affected by the variant

**The Key Challenge**: SpliceVarDB tells us WHERE the mutation is, but NOT where the affected splice site is.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   VARIANT vs AFFECTED SPLICE SITE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Example: Variant at position 1000 might:                               │
│    - Destroy existing donor at position 1000 (variant position)         │
│    - Create new cryptic donor at position 1005                          │
│    - Disrupt existing acceptor at position 990                          │
│                                                                         │
│  SpliceVarDB provides:                                                  │
│    ✅ Variant position (1000)                                           │
│    ✅ Classification (Splice-altering)                                  │
│    ⚠️ HGVS hints (c.670-1G>T → near acceptor)                          │
│    ❌ Exact affected splice site position                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Deriving Position Labels

We have TWO approaches for creating training labels:

**Approach 1: HGVS Parsing (Weak Labels)**

HGVS notation contains position hints relative to exon boundaries:

```python
from meta_spliceai.splice_engine.meta_layer.data.position_labels import (
    derive_position_labels_from_hgvs
)

# HGVS notation patterns:
# c.670-1G>T   → 1bp before exon = ACCEPTOR site (canonical)
# c.1092+2T>A  → 2bp after exon  = DONOR site (canonical)
# c.1018-550A>G → 550bp in intron = deep intronic (cryptic?)

hint = variant.get_position_hint()
print(f"Site type: {hint.site_type}")           # 'acceptor'
print(f"Distance: {hint.distance_from_boundary}")  # 1
print(f"Canonical: {hint.is_canonical_region}")    # True
print(f"Confidence: {hint.confidence}")            # 'high'
```

| HGVS Pattern | Interpretation | Likely Effect |
|--------------|----------------|---------------|
| `c.XXX-1` or `c.XXX-2` | Canonical acceptor (AG) | Acceptor loss |
| `c.XXX+1` or `c.XXX+2` | Canonical donor (GT) | Donor loss |
| `c.XXX-N` (N > 25) | Deep intronic | Cryptic site? |
| `c.XXX+N` (N > 10) | Deep intronic | Cryptic site? |
| `c.XXX` (no +/-) | Exonic | ESE/ESS disruption |

**Approach 2: Base Model Delta Analysis (Recommended)**

Use the base model to find where delta peaks occur:

```python
from meta_spliceai.splice_engine.meta_layer.data.position_labels import (
    derive_position_labels_from_delta,
    create_position_attention_target
)

# Find positions with significant delta
affected_positions = derive_position_labels_from_delta(
    ref_seq, alt_seq, base_model, 
    threshold=0.1  # Minimum delta to consider
)

# Each position has:
# - position: int (0-indexed in sequence)
# - effect_type: 'donor_gain', 'donor_loss', 'acceptor_gain', 'acceptor_loss'
# - delta_value: float (signed, positive=gain, negative=loss)
# - confidence: 'high', 'medium', 'low'

for pos in affected_positions:
    print(f"Position {pos.position}: {pos.effect_type} (Δ={pos.delta_value:.3f})")

# Create attention target for training
attention_target = create_position_attention_target(
    affected_positions, 
    sequence_length=501,
    sigma=3.0  # Gaussian spread around peak
)
# attention_target is [501] with soft peaks at affected positions
```

#### Architecture Options

**Option A: Attention-based (Recommended)**
```python
class PositionLocalizer(nn.Module):
    """
    Output attention weights over positions.
    High attention = likely affected position.
    
    Training target: Soft attention distribution from base model delta peaks
    Loss: KL divergence or cross-entropy with soft targets
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = GatedCNNEncoder(hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, alt_seq, effect_type_embedding=None):
        features = self.encoder.encode_per_position(alt_seq)  # [B, L, H]
        
        # Optionally condition on effect type from Step 2
        if effect_type_embedding is not None:
            features = features + effect_type_embedding.unsqueeze(1)
        
        attention = F.softmax(self.attention(features).squeeze(-1), dim=-1)  # [B, L]
        return attention
```

**Option B: Binary segmentation**
```python
class PositionSegmenter(nn.Module):
    """
    Binary mask: Which positions are affected?
    
    Training target: Binary mask from base model delta peaks (expanded ±2bp)
    Loss: Binary cross-entropy
    """
    def forward(self, alt_seq):
        features = self.encoder(alt_seq)  # [B, L, H]
        mask_logits = self.head(features)  # [B, L, 1]
        return torch.sigmoid(mask_logits)
```

**Option C: Regression (point prediction)**
```python
class PositionRegressor(nn.Module):
    """
    Directly predict the position offset from variant.
    
    Output: Single integer offset (e.g., -5 means 5bp before variant)
    """
    def forward(self, alt_seq):
        features = self.encoder(alt_seq)  # [B, H]
        offset = self.head(features)  # [B, 1] - continuous offset
        return offset
```

#### Implementation Status

| Component | File | Status |
|-----------|------|--------|
| HGVS Parsing | `data/splicevardb_loader.py` | ✅ Implemented |
| Position Label Derivation | `data/position_labels.py` | ✅ Implemented |
| PositionLocalizer Model | `models/position_localizer.py` | ⏳ TODO |
| Training Script | `tests/test_position_localization.py` | ⏳ TODO |

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

## Comparison: ValidatedDelta vs Multi-Step

| Aspect | ValidatedDelta (r=0.507) | Multi-Step Framework |
|--------|-------------------------|----------------------|
| **Task** | Predict [Δ_donor, Δ_acceptor] directly | Sequential: classify → type → locate |
| **Difficulty** | Medium (regression w/ validated targets) | Easier (decomposed classification) |
| **Interpretability** | Low (requires threshold for decisions) | **High** (explicit yes/no → type → where) |
| **Output** | Continuous delta scores | Categorical decisions + attention |
| **Inference** | Single pass | Multiple models (or multi-task) |
| **Biology alignment** | ⚠️ Indirect | ✅ Matches clinical workflow |
| **Current status** | r=0.507 (8K samples) | Step 1: AUC=0.61 (needs >0.7) |

### When to Use Which

| Use Case | Best Method | Why |
|----------|-------------|-----|
| **Ranking variants by severity** | ValidatedDelta | Continuous scores enable ranking |
| **"Is this pathogenic?"** | **Multi-Step (Step 1)** | Direct yes/no answer |
| **"What kind of effect?"** | **Multi-Step (Step 2)** | Direct effect type output |
| **"Where should I target ASO?"** | **Multi-Step (Step 3)** | Position localization |
| **Recalibrating base model** | ValidatedDelta | Additive correction scores |

### Complementary Approaches

**ValidatedDelta** and **Multi-Step** are complementary, not competing:

```
Multi-Step for DECISIONS:             ValidatedDelta for REFINEMENT:
─────────────────────────             ─────────────────────────────
"Is it splice-altering?" → YES        "By how much?"
"What type?" → Donor gain             → Δ_donor = +0.42
"Where?" → Position 127               → Recalibrated score = 0.87
```

**Recommended Pipeline for Drug Target Discovery**:
1. **Triage** (Multi-Step Step 1): Filter 10,000 variants → 500 candidates
2. **Classify** (Multi-Step Step 2): 500 variants → 200 donor-gain, 150 acceptor-loss, etc.
3. **Quantify** (ValidatedDelta): Get precise delta scores for top candidates
4. **Locate** (Multi-Step Step 3): Find exact target positions for ASO design

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

## Current Implementation Status

### Step 1: Splice-Inducing Classification ✅ IMPLEMENTED

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| ROC-AUC | 0.61 | >0.75 | ⚠️ Needs improvement |
| F1 Score | 0.53 | >0.70 | ⚠️ Needs improvement |
| PR-AUC | 0.58 | >0.65 | ⚠️ Needs improvement |

**Implementation**: `SpliceInducingClassifier` in `models/splice_classifier.py`  
**Experiment**: `docs/experiments/003_binary_classification/`

**Why current performance is limited**:
1. Small training data (2000 samples)
2. Simple CNN encoder (HyenaDNA may help)
3. No data augmentation (reverse complement, etc.)

### Step 2: Effect Type Classification ✅ IMPLEMENTED (not tested)

**Implementation**: `EffectTypeClassifier` in `models/splice_classifier.py`  
**Status**: Code ready, needs training/evaluation

### Step 3: Position Localization ❌ NOT IMPLEMENTED

**Priority**: MEDIUM (after Steps 1-2 are solid)  
**Challenge**: Need positional labels from SpliceVarDB or delta score analysis

### Step 4: Delta Magnitude ❌ NOT IMPLEMENTED

**Priority**: LOW (ValidatedDelta handles this case)  
**May not be needed**: ValidatedDelta (r=0.507) may be sufficient

---

## 🧬 RNA Therapeutics Application

### Why Multi-Step is Critical for ASO/Splice-Switching Oligonucleotide Design

```
VARIANT SCREENING PIPELINE FOR RNA THERAPEUTICS
───────────────────────────────────────────────

1. TRIAGE (Multi-Step Step 1)
   Input:  10,000 candidate variants
   Filter: P(splice-altering) > 0.5
   Output: 1,000 high-priority variants
   Time:   ~1 minute (batch inference)
   
2. EFFECT CLASSIFICATION (Multi-Step Step 2)
   Input:  1,000 high-priority variants
   Filter: Effect type = "Exon inclusion" or "Cryptic activation"
   Output: 300 therapeutically-relevant variants
   
3. LOCALIZATION (Multi-Step Step 3)
   Input:  300 therapeutically-relevant variants
   Output: Exact splice site positions ± 5nt
   Use:    Design 18-25mer ASO targeting this position
   
4. QUANTIFICATION (ValidatedDelta)
   Input:  300 variants
   Output: Expected delta scores
   Use:    Predict ASO efficacy (larger Δ = stronger modulation)
```

### Effect Types → Therapeutic Strategies

| Effect Type | Therapeutic Implication | ASO Strategy |
|-------------|------------------------|--------------|
| **Donor gain** | Cryptic exon inclusion | Block new donor |
| **Donor loss** | Exon skipping | Restore donor or induce skip |
| **Acceptor gain** | Cryptic exon inclusion | Block new acceptor |
| **Acceptor loss** | Exon skipping | Restore acceptor or induce skip |
| **Complex** | Multiple effects | Careful target selection |

### Why This Matters for Clinical Development

1. **Regulatory**: FDA requires mechanistic understanding. "It's splice-altering because..." is better than "Δ=0.35"
2. **Target Selection**: Effect type guides ASO design (block vs. restore)
3. **Safety**: Position localization helps avoid off-target effects
4. **Efficacy Prediction**: Magnitude estimation predicts clinical response

---

## Summary

**Current best delta predictor** (ValidatedDelta):
- Single forward pass, r=0.507 correlation
- Best for ranking and quantification
- Outputs continuous delta scores

**Multi-step framework** (complementary):
1. **Step 1**: Is this variant splice-altering? (Binary) → AUC=0.61 ⚠️
2. **Step 2**: What type of effect? (Multi-class) → Implemented, not tested
3. **Step 3**: Where is the effect? (Localization) → Not implemented
4. **Step 4**: How strong? (Regression) → Use ValidatedDelta instead

**Key insight**: Multi-Step provides **interpretable decisions** that ValidatedDelta cannot:
- "Yes this is pathogenic" vs "Δ=0.35"
- "It's a donor gain" vs "Δ_donor=0.35"
- "Position 127 is affected" vs "max at position 127"

**Recommended**: Use BOTH approaches together for comprehensive analysis.

---

## Files

| File | Description | Status |
|------|-------------|--------|
| `models/splice_classifier.py` | Step 1, 2, Unified classifiers | ✅ Implemented |
| `docs/experiments/003_binary_classification/` | Step 1 results | ✅ Complete |
| `models/validated_delta_predictor.py` | Alternative to Step 4 | ✅ Best results |

---

## Relationship to ValidatedDelta (Approach B)

Multi-Step and ValidatedDelta are complementary:

| Framework | Question Answered | Output Type |
|-----------|------------------|-------------|
| Multi-Step Step 1 | "Splice-altering?" | Binary (yes/no) |
| Multi-Step Step 2 | "What type?" | Categorical (5 classes) |
| Multi-Step Step 3 | "Where?" | Position (attention/mask) |
| **ValidatedDelta** | "How much?" | Continuous (Δ scores) |

ValidatedDelta subsumes Step 4 (delta magnitude) with better performance (r=0.507).

