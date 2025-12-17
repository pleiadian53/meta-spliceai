# Labeling Strategy for Meta-Layer Training

**Last Updated**: December 2025  
**Status**: Design Document

---

## Overview

This document explains how labels are created for training the multimodal meta-layer, including:

1. How base layer labels (from GTF annotations) are used
2. How SpliceVarDB enhances the training data
3. The relationship between meta-layer and base model labels
4. Handling variant-induced alternative splicing

---

## The Core Question: What Do Labels Represent?

### Answer: Labels are **consistent with base model training data**

The meta-layer outputs **per-nucleotide splice site probabilities** in the same format as the base model:

```
P(donor):    Probability this nucleotide is a donor splice site
P(acceptor): Probability this nucleotide is an acceptor splice site
P(neither):  Probability this is not a splice site
```

**Key principle**: The label at each position represents the **ground truth splice site status** from genomic annotations (GTF/GFF).

---

## Label Source: GTF/GFF Annotations

### How Base Models Get Their Labels

Both SpliceAI and OpenSpliceAI are trained on **canonical splice sites** from gene annotations:

```
GTF File → Extract Splice Sites → Labels
                    ↓
   Donor sites:    5' end of introns (exon-intron junction)
   Acceptor sites: 3' end of introns (intron-exon junction)
   Neither:        All other positions
```

### The Meta-Layer Uses the Same Labels

```
Base Layer Artifacts (analysis_sequences_*.tsv)
├── sequence:     501nt context window
├── features:     Base model scores + derived features
└── splice_type:  Ground truth label ('donor', 'acceptor', '')
                       ↑
                  From GTF annotations!
```

**This is already in the artifacts!** The `splice_type` column in `analysis_sequences_*.tsv` contains the labels.

---

## Label Encoding

```python
# In meta_spliceai/splice_engine/meta_layer/core/feature_schema.py

LABEL_ENCODING = {
    'donor': 0,
    'acceptor': 1,
    'neither': 2,
    '': 2  # Empty string also maps to 'neither'
}

LABEL_DECODING = {
    0: 'donor',
    1: 'acceptor',
    2: 'neither'
}
```

### Label Distribution (Example from OpenSpliceAI artifacts)

```
Position Type      | Count      | Percentage
------------------ | ---------- | ----------
donor              | ~185K      | ~23%
acceptor           | ~185K      | ~23%
neither (sampled)  | ~440K      | ~54%
------------------ | ---------- | ----------
Total              | ~810K      | 100%
```

Note: "Neither" positions are sampled from the full genome to keep the dataset balanced.

---

## How SpliceVarDB Enhances Training

### SpliceVarDB Does NOT Replace Labels

SpliceVarDB provides **additional context** about positions, not new labels:

```
SpliceVarDB Contribution:
├── Identifies positions where splicing is experimentally validated
├── Provides classification (splice-altering vs non-splice-altering)
└── Enables weighted training (higher weight for validated positions)
```

### Training Enhancement Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                   TRAINING DATA CONSTRUCTION                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Base Layer Artifacts                                            │
│  ──────────────────────────────────────────────────────────────  │
│  • ~810K positions with features and labels                      │
│  • Labels from GTF annotations                                   │
│  • Default weight: 1.0                                           │
│                                                                  │
│                        ↓ ANNOTATE                                │
│                                                                  │
│  SpliceVarDB Matching                                            │
│  ──────────────────────────────────────────────────────────────  │
│  • Match variant positions to base layer positions               │
│  • Add classification (splice-altering, low-frequency, etc.)     │
│  • Adjust sample weights based on classification                 │
│                                                                  │
│                        ↓ RESULT                                  │
│                                                                  │
│  Enhanced Training Dataset                                       │
│  ──────────────────────────────────────────────────────────────  │
│  • Same labels (donor, acceptor, neither)                        │
│  • Same features (sequence + scores)                             │
│  • Additional: variant_classification, sample_weight             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Weight Assignment by Classification

```python
def get_sample_weight(variant_classification: str) -> float:
    """
    Assign sample weight based on SpliceVarDB classification.
    
    Higher weight = more important during training.
    """
    weights = {
        # Experimentally validated splice-altering
        'Splice-altering': 2.0,  # Boost importance
        
        # Experimentally validated non-splice-altering
        'Non-splice-altering': 1.5,  # Important negative examples
        
        # Uncertain/low-frequency effect
        'Low-frequency': 0.5,  # Reduced weight (uncertain)
        
        # No variant at this position
        None: 1.0  # Default weight
    }
    return weights.get(variant_classification, 1.0)
```

---

## Understanding Variant Effects on Labels

### Scenario 1: Splice-Altering Variant at a Splice Site

```
Position: chr1:12345 (canonical donor site)
GTF Label: 'donor' (this IS a splice site)
Variant: Causes loss of this donor site

Training interpretation:
├── The label is still 'donor' (the canonical annotation)
├── But the base model might have trouble here
├── Meta-layer learns: "positions with certain features need correction"
└── The model learns patterns that indicate unreliable base predictions
```

### Scenario 2: Splice-Altering Variant Creates New Splice Site

```
Position: chr1:12350 (not a canonical splice site)
GTF Label: 'neither'
Variant: Creates a new donor site (cryptic splicing)

Training interpretation:
├── The label remains 'neither' (canonical annotation)
├── But this is a position where splicing CAN occur
├── Meta-layer learns: "certain sequence patterns can activate splicing"
└── Useful for predicting context-dependent alternative splicing
```

### Scenario 3: Non-Splice-Altering Variant

```
Position: chr1:12340 (canonical donor site)
GTF Label: 'donor'
Variant: Does NOT affect splicing

Training interpretation:
├── Label is 'donor', and it SHOULD be predicted as donor
├── This is a confirmed positive example
├── Meta-layer reinforcement: "this pattern = reliable splice site"
```

---

## Evaluation Strategy

### The Core Evaluation Questions

1. **Does the meta-layer degrade canonical splice site prediction?**
2. **Does the meta-layer better capture alternative/cryptic splice sites?**
3. **How well does SpliceVarDB weighting help?**

### Evaluation Datasets

```
┌─────────────────────────────────────────────────────────────────────┐
│ DATASET 1: Held-Out Canonical Sites (from artifacts)               │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose: Ensure no degradation on standard splice site prediction  │
│                                                                     │
│ Split:                                                              │
│   - Train: Chromosomes 1-17 (80%)                                   │
│   - Validation: Chromosomes 18-20 (10%)                             │
│   - Test: Chromosomes 21-22, X (10%)                                │
│                                                                     │
│ Metrics:                                                            │
│   - PR-AUC (primary)                                                │
│   - ROC-AUC                                                         │
│   - Top-k accuracy (SpliceAI-style)                                 │
│   - Per-class precision/recall                                      │
│                                                                     │
│ Comparison:                                                         │
│   - Meta-layer PR-AUC vs Base model PR-AUC                          │
│   - Should be EQUAL or BETTER                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ DATASET 2: SpliceVarDB-Annotated Positions                         │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose: Evaluate alternative splice site detection                 │
│                                                                     │
│ Stratification:                                                     │
│   - Splice-altering variants (cryptic sites created/destroyed)     │
│   - Non-splice-altering variants (should maintain prediction)      │
│   - Low-frequency variants (uncertain ground truth)                │
│                                                                     │
│ Key Metrics for Splice-Altering:                                    │
│   - Does meta-layer predict HIGHER donor/acceptor at cryptic sites?│
│   - Does meta-layer predict LOWER scores at disrupted sites?       │
│   - Delta improvement over base model                               │
│                                                                     │
│ Comparison:                                                         │
│   |meta_score - expected| vs |base_score - expected|               │
│   Should show: meta_layer closer to biological reality              │
└─────────────────────────────────────────────────────────────────────┘
```

### Evaluation Metrics

```python
def evaluate_meta_layer(
    meta_predictions: np.ndarray,  # [N, 3]
    base_predictions: np.ndarray,  # [N, 3]
    labels: np.ndarray,            # [N]
    variant_annotations: pd.Series # Optional: SpliceVarDB classifications
) -> Dict[str, float]:
    """
    Comprehensive evaluation of meta-layer performance.
    
    Returns metrics for:
    1. Overall performance (PR-AUC, ROC-AUC)
    2. Per-class performance (donor, acceptor, neither)
    3. Comparison to base model (improvement/degradation)
    4. SpliceVarDB-stratified performance (if annotations provided)
    """
    
    results = {}
    
    # === 1. Overall Performance ===
    results['meta_pr_auc'] = compute_pr_auc(labels, meta_predictions)
    results['base_pr_auc'] = compute_pr_auc(labels, base_predictions)
    results['pr_auc_delta'] = results['meta_pr_auc'] - results['base_pr_auc']
    
    # === 2. Degradation Check ===
    # On canonical positions (no variant annotation)
    canonical_mask = variant_annotations.isna() if variant_annotations is not None else np.ones(len(labels), dtype=bool)
    results['canonical_meta_pr_auc'] = compute_pr_auc(labels[canonical_mask], meta_predictions[canonical_mask])
    results['canonical_base_pr_auc'] = compute_pr_auc(labels[canonical_mask], base_predictions[canonical_mask])
    results['canonical_degradation'] = results['canonical_meta_pr_auc'] < results['canonical_base_pr_auc']
    
    # === 3. SpliceVarDB-Stratified Performance ===
    if variant_annotations is not None:
        for classification in ['Splice-altering', 'Non-splice-altering', 'Low-frequency']:
            mask = variant_annotations == classification
            if mask.sum() > 0:
                results[f'{classification}_meta_pr_auc'] = compute_pr_auc(labels[mask], meta_predictions[mask])
                results[f'{classification}_base_pr_auc'] = compute_pr_auc(labels[mask], base_predictions[mask])
                results[f'{classification}_improvement'] = (
                    results[f'{classification}_meta_pr_auc'] - results[f'{classification}_base_pr_auc']
                )
    
    # === 4. Score Calibration ===
    # For splice-altering variants that CREATE sites:
    # Meta-layer should predict HIGHER donor/acceptor scores
    # (This requires effect-type annotation from SpliceVarDB)
    
    return results
```

### Success Criteria

| Metric | Minimum Threshold | Target |
|--------|------------------|--------|
| Canonical PR-AUC | ≥ Base model | +0.5% improvement |
| Canonical degradation | Must be False | - |
| Splice-altering improvement | > 0 | +5% over base |
| Non-splice-altering performance | ≥ Base model | Maintain |

### Evaluation Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Train on weighted data (SpliceVarDB weights applied)       │
│                                                                     │
│ STEP 2: Evaluate on canonical test set                              │
│         → Check: No degradation?                                    │
│         → If degradation: Reduce weight differential or tune loss  │
│                                                                     │
│ STEP 3: Evaluate on SpliceVarDB-annotated positions                 │
│         → Check: Improvement on splice-altering variants?           │
│         → Analyze: Which variant types benefit most?                │
│                                                                     │
│ STEP 4: Ablation study                                              │
│         → Compare: With vs without SpliceVarDB weighting            │
│         → Identify: How much does weighting contribute?             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Alternative: Delta-Learning Formulation

An alternative approach (not currently implemented) is to learn the **delta** between base model and ground truth:

```
Traditional:  Input → Meta-Layer → P(splice site)
Delta:        Input → Meta-Layer → Δ to add to base prediction
```

### Delta Labels

```python
# For delta learning (not implemented)
delta_label = ground_truth_prob - base_model_prediction

# Example:
# Ground truth: donor (encoded as [1, 0, 0])
# Base model:   [0.7, 0.2, 0.1]
# Delta:        [0.3, -0.2, -0.1]  ← "boost donor by 0.3, reduce acceptor by 0.2"
```

### Variant-Effect Interpretation

The delta formulation enables clinically meaningful statements:

```
"Given variant rs123456 at chr1:12345,
 the probability of position chr1:12350 being a donor site
 INCREASES by 0.45 (from 0.12 to 0.57)"
```

This is analogous to SpliceAI's ΔScore, but learned through the meta-layer.

#### Variant-Conditioned Prediction Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ REFERENCE SEQUENCE                                                  │
│ ...ACGT[G]TAAGTC...   (G = reference allele at variant position)   │
│         ↓                                                           │
│ Base Model → [0.12, 0.05, 0.83] at position +5                     │
│              (12% donor, 5% acceptor, 83% neither)                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ VARIANT SEQUENCE                                                    │
│ ...ACGT[A]TAAGTC...   (A = alternate allele)                       │
│         ↓                                                           │
│ Base Model → [0.45, 0.03, 0.52] at position +5                     │
│              (45% donor, 3% acceptor, 52% neither)                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ DELTA (Variant Effect)                                              │
│                                                                     │
│ ΔDonor    = 0.45 - 0.12 = +0.33  → "Cryptic donor ACTIVATED"       │
│ ΔAcceptor = 0.03 - 0.05 = -0.02  → "Minimal change"                │
│ ΔNeither  = 0.52 - 0.83 = -0.31  → "Less likely to be non-site"    │
│                                                                     │
│ Interpretation: "This G>A variant at chr1:12345 creates a          │
│                  cryptic donor site at position chr1:12350"        │
└─────────────────────────────────────────────────────────────────────┘
```

#### Two Approaches for Variant-Effect Learning

**Approach A: Paired Prediction (Reference vs Variant)**

```python
class VariantEffectPredictor(nn.Module):
    """
    Predict variant effect by comparing reference vs variant predictions.
    
    This explicitly models the delta as: Δ = P(variant) - P(reference)
    """
    
    def __init__(self, base_model: MetaSpliceModel):
        super().__init__()
        self.model = base_model  # Shared model for both ref and alt
    
    def forward(
        self,
        ref_sequence: torch.Tensor,   # Reference sequence tokens
        alt_sequence: torch.Tensor,   # Variant sequence tokens
        score_features: torch.Tensor  # Base model features (shared)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            ref_probs: [B, 3] probabilities for reference
            alt_probs: [B, 3] probabilities for variant
            delta: [B, 3] effect of variant (alt - ref)
        """
        ref_probs = self.model(ref_sequence, score_features)
        alt_probs = self.model(alt_sequence, score_features)
        delta = alt_probs - ref_probs
        
        return ref_probs, alt_probs, delta
    
    def interpret(self, delta: torch.Tensor) -> List[str]:
        """Generate human-readable interpretation."""
        interpretations = []
        for d in delta:
            effects = []
            if d[0] > 0.1:  # Donor increase
                effects.append(f"Donor gain (+{d[0]:.2f})")
            elif d[0] < -0.1:
                effects.append(f"Donor loss ({d[0]:.2f})")
            if d[1] > 0.1:  # Acceptor increase
                effects.append(f"Acceptor gain (+{d[1]:.2f})")
            elif d[1] < -0.1:
                effects.append(f"Acceptor loss ({d[1]:.2f})")
            
            if not effects:
                effects.append("No significant splice effect")
            
            interpretations.append("; ".join(effects))
        
        return interpretations
```

**Approach B: Direct Delta Prediction**

```python
class DirectDeltaPredictor(nn.Module):
    """
    Directly predict the delta without running reference.
    
    More efficient for inference: single forward pass.
    """
    
    def __init__(
        self,
        num_score_features: int,
        hidden_dim: int = 256,
        **kwargs
    ):
        super().__init__()
        self.seq_encoder = SequenceEncoderFactory.create("cnn", output_dim=hidden_dim)
        self.score_encoder = ScoreEncoder(num_score_features, hidden_dim=hidden_dim)
        
        # Variant-aware encoding
        self.variant_encoder = nn.Sequential(
            nn.Linear(4 + 4, hidden_dim // 4),  # ref_base + alt_base one-hot
            nn.GELU()
        )
        
        # Delta prediction head
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3),
            nn.Tanh()  # Delta bounded to [-1, 1]
        )
    
    def forward(
        self,
        alt_sequence: torch.Tensor,    # Variant sequence
        score_features: torch.Tensor,  # Base model features
        ref_base: torch.Tensor,        # [B, 4] one-hot reference base
        alt_base: torch.Tensor,        # [B, 4] one-hot alternate base
        base_scores: torch.Tensor      # [B, 3] base model predictions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            delta: [B, 3] predicted effect of variant
            adjusted_scores: [B, 3] base_scores + delta
        """
        seq_emb = self.seq_encoder(alt_sequence)
        score_emb = self.score_encoder(score_features)
        variant_emb = self.variant_encoder(torch.cat([ref_base, alt_base], dim=-1))
        
        combined = torch.cat([seq_emb, score_emb, variant_emb], dim=-1)
        delta = self.delta_head(combined)
        
        adjusted_scores = F.softmax(base_scores + delta, dim=-1)
        
        return delta, adjusted_scores
```

#### Clinical Interpretation Output

```python
@dataclass
class VariantSpliceEffect:
    """Clinical-ready variant effect report."""
    
    variant_id: str
    chrom: str
    position: int
    ref_allele: str
    alt_allele: str
    
    # Effects at different positions relative to variant
    effects: List[Dict[str, Any]]  # position -> {delta_donor, delta_acceptor, interpretation}
    
    # Summary
    max_delta_donor: float
    max_delta_acceptor: float
    primary_effect: str  # "Donor gain", "Donor loss", "Acceptor gain", "Acceptor loss", "None"
    
    def to_vcf_annotation(self) -> str:
        """Format for VCF INFO field, similar to SpliceAI."""
        # Format: SYMBOL|ΔDonor|ΔAcceptor|Position
        return f"{self.primary_effect}|{self.max_delta_donor:.2f}|{self.max_delta_acceptor:.2f}"
    
    def to_clinical_report(self) -> str:
        """Generate clinical interpretation."""
        report = f"Variant: {self.chrom}:{self.position} {self.ref_allele}>{self.alt_allele}\n"
        report += f"Primary Effect: {self.primary_effect}\n"
        
        if self.max_delta_donor > 0.2:
            report += f"⚠️ CRYPTIC DONOR: +{self.max_delta_donor:.2f} probability\n"
        if self.max_delta_donor < -0.2:
            report += f"⚠️ DONOR LOSS: {self.max_delta_donor:.2f} probability\n"
        if self.max_delta_acceptor > 0.2:
            report += f"⚠️ CRYPTIC ACCEPTOR: +{self.max_delta_acceptor:.2f} probability\n"
        if self.max_delta_acceptor < -0.2:
            report += f"⚠️ ACCEPTOR LOSS: {self.max_delta_acceptor:.2f} probability\n"
        
        return report


# Example usage
effect = VariantSpliceEffect(
    variant_id="rs123456",
    chrom="chr1",
    position=12345,
    ref_allele="G",
    alt_allele="A",
    effects=[
        {"position": 12350, "delta_donor": 0.33, "delta_acceptor": -0.02, 
         "interpretation": "Cryptic donor activated"},
    ],
    max_delta_donor=0.33,
    max_delta_acceptor=0.0,
    primary_effect="Donor gain"
)

print(effect.to_clinical_report())
# Output:
# Variant: chr1:12345 G>A
# Primary Effect: Donor gain
# ⚠️ CRYPTIC DONOR: +0.33 probability
```

---

## Training Data Preparation for Direct Delta Prediction

### The Core Challenge

For Direct Delta Prediction (Approach B), we need **target delta values** for training:

```
Input: variant sequence + features + ref_base + alt_base
Output: Δ = [Δ_donor, Δ_acceptor, Δ_neither]

Question: What is the GROUND TRUTH delta for training?
```

### Four Options for Target Delta

#### Option 1: Base Model Delta (Weak Supervision)

```
Run base model on BOTH reference and variant sequences:
  ref_scores = base_model(ref_sequence)
  alt_scores = base_model(alt_sequence)
  target_delta = alt_scores - ref_scores

Problem: Just teaches the model to replicate base model behavior!
         No improvement over running base model twice.
```

#### Option 2: SpliceVarDB Classification → Soft Targets

```python
def classification_to_delta(
    classification: str,
    effect_type: str,  # From SpliceVarDB: "Donor gain", "Acceptor loss", etc.
    position_type: str,  # "donor", "acceptor", "neither" from GTF
    confidence: float = 0.5  # How much to push the delta
) -> np.ndarray:
    """
    Convert SpliceVarDB classification to target delta.
    
    This is the KEY function for training data preparation.
    """
    
    delta = np.zeros(3)  # [donor, acceptor, neither]
    
    if classification == "Splice-altering":
        # Variant DOES affect splicing
        if effect_type == "Donor gain":
            delta[0] = +confidence   # Increase donor probability
            delta[2] = -confidence   # Decrease neither
        elif effect_type == "Donor loss":
            delta[0] = -confidence   # Decrease donor probability
            delta[2] = +confidence   # Increase neither
        elif effect_type == "Acceptor gain":
            delta[1] = +confidence   # Increase acceptor
            delta[2] = -confidence
        elif effect_type == "Acceptor loss":
            delta[1] = -confidence
            delta[2] = +confidence
        elif effect_type == "Exon skipping":
            # Both donor and acceptor at exon boundaries affected
            if position_type == "donor":
                delta[0] = -confidence
            elif position_type == "acceptor":
                delta[1] = -confidence
            delta[2] = +confidence
    
    elif classification == "Non-splice-altering":
        # Variant does NOT affect splicing → delta should be ~0
        delta = np.zeros(3)
    
    elif classification == "Low-frequency":
        # Uncertain → smaller delta magnitude
        # Could also be handled via sample weights
        delta = np.zeros(3)  # Or use reduced confidence
    
    return delta
```

#### Option 3: Hybrid - Base Model Delta + SpliceVarDB Correction

```python
def compute_hybrid_target(
    ref_scores: np.ndarray,      # Base model on reference
    alt_scores: np.ndarray,      # Base model on variant
    classification: str,         # SpliceVarDB classification
    effect_type: str,            # SpliceVarDB effect type
    correction_strength: float = 0.3
) -> np.ndarray:
    """
    Hybrid approach: Start with base model delta, adjust based on SpliceVarDB.
    
    This is the RECOMMENDED approach for training.
    
    Logic:
    - If SpliceVarDB says "Splice-altering" but base model delta is small:
      → Boost the delta (base model missed it)
    - If SpliceVarDB says "Non-splice-altering" but base model delta is large:
      → Shrink the delta (base model false positive)
    """
    
    base_delta = alt_scores - ref_scores
    
    if classification == "Splice-altering":
        # Variant should have effect
        expected_direction = get_expected_direction(effect_type)  # +1 or -1
        
        # Check if base model captured the effect
        for i, expected in enumerate(expected_direction):
            if expected != 0:
                if np.sign(base_delta[i]) != expected:
                    # Base model got direction WRONG → strong correction
                    base_delta[i] = expected * correction_strength
                elif np.abs(base_delta[i]) < 0.1:
                    # Base model got direction right but too weak → boost
                    base_delta[i] = expected * max(np.abs(base_delta[i]), correction_strength)
    
    elif classification == "Non-splice-altering":
        # Variant should have NO effect → shrink delta toward 0
        base_delta = base_delta * (1 - correction_strength)
    
    return base_delta


def get_expected_direction(effect_type: str) -> np.ndarray:
    """Map effect type to expected delta direction."""
    directions = {
        "Donor gain":     [+1,  0, -1],
        "Donor loss":     [-1,  0, +1],
        "Acceptor gain":  [ 0, +1, -1],
        "Acceptor loss":  [ 0, -1, +1],
        "Exon skipping":  [-1, -1, +1],  # Both splice sites weakened
    }
    return np.array(directions.get(effect_type, [0, 0, 0]))
```

#### Option 4: Multi-Position Delta from Experimental Data

```python
def extract_experimental_deltas(
    variant_record: Dict,
    window_size: int = 50  # Positions around variant
) -> List[Tuple[int, np.ndarray]]:
    """
    For variants with experimental validation, extract deltas at MULTIPLE positions.
    
    SpliceVarDB variants with RNA-seq evidence can tell us:
    - Which position became a new splice site (cryptic)
    - Which position lost splice site function
    
    This creates richer training signal.
    """
    
    deltas = []
    variant_pos = variant_record['position']
    
    # If we know the exact cryptic site position from experimental data
    if 'cryptic_site_position' in variant_record:
        cryptic_pos = variant_record['cryptic_site_position']
        site_type = variant_record['cryptic_site_type']  # "donor" or "acceptor"
        
        # At the cryptic site: strong positive delta
        if site_type == "donor":
            deltas.append((cryptic_pos, np.array([+0.7, 0, -0.7])))
        else:
            deltas.append((cryptic_pos, np.array([0, +0.7, -0.7])))
    
    # If we know the original site that was disrupted
    if 'disrupted_site_position' in variant_record:
        disrupted_pos = variant_record['disrupted_site_position']
        site_type = variant_record['disrupted_site_type']
        
        # At the disrupted site: strong negative delta
        if site_type == "donor":
            deltas.append((disrupted_pos, np.array([-0.7, 0, +0.7])))
        else:
            deltas.append((disrupted_pos, np.array([0, -0.7, +0.7])))
    
    # For other positions in window: delta should be ~0
    for offset in range(-window_size, window_size + 1):
        pos = variant_pos + offset
        if pos not in [d[0] for d in deltas]:
            deltas.append((pos, np.array([0, 0, 0])))
    
    return deltas
```

### Recommended Training Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: Collect Variant-Position Pairs                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ From SpliceVarDB:                                                   │
│   - variant_id, chrom, position, ref, alt                           │
│   - classification: "Splice-altering" / "Non-splice-altering"       │
│   - effect_type: "Donor gain" / "Acceptor loss" / etc.              │
│                                                                     │
│ For each variant, consider positions in window [-250, +250]         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: Generate Reference and Variant Sequences                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ For each position in window:                                        │
│   ref_seq = fasta[chrom][pos-250:pos+251]  # 501nt context          │
│   alt_seq = ref_seq with variant substituted                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: Run Base Model on Both Sequences                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ ref_scores = base_model.predict(ref_seq)  # [donor, acceptor, none] │
│ alt_scores = base_model.predict(alt_seq)                            │
│ base_delta = alt_scores - ref_scores                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: Compute Target Delta (Hybrid Approach)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ target_delta = compute_hybrid_target(                               │
│     ref_scores,                                                     │
│     alt_scores,                                                     │
│     classification,  # From SpliceVarDB                             │
│     effect_type,     # From SpliceVarDB                             │
│     correction_strength=0.3                                         │
│ )                                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 5: Create Training Sample                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ sample = {                                                          │
│     'alt_sequence': alt_seq,                                        │
│     'score_features': extract_features(ref_scores, alt_scores),     │
│     'ref_base': one_hot(ref),                                       │
│     'alt_base': one_hot(alt),                                       │
│     'base_scores': ref_scores,  # Reference prediction              │
│     'target_delta': target_delta,                                   │
│     'sample_weight': get_weight(classification)                     │
│ }                                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation: Training Data Generator

```python
from dataclasses import dataclass
from typing import Iterator, Optional
import polars as pl
import numpy as np

@dataclass
class DeltaTrainingSample:
    """Single training sample for Direct Delta Prediction."""
    
    variant_id: str
    chrom: str
    position: int  # Position being predicted (not variant position)
    variant_position: int
    
    # Sequences
    ref_sequence: str  # 501nt centered on position
    alt_sequence: str  # 501nt with variant substituted
    
    # Base encodings
    ref_base: np.ndarray  # [4] one-hot
    alt_base: np.ndarray  # [4] one-hot
    
    # Base model outputs
    ref_scores: np.ndarray  # [3] base model on reference
    alt_scores: np.ndarray  # [3] base model on variant
    base_delta: np.ndarray  # [3] alt_scores - ref_scores
    
    # Target for training
    target_delta: np.ndarray  # [3] corrected delta
    
    # Metadata
    classification: str  # SpliceVarDB classification
    effect_type: Optional[str]
    sample_weight: float


class DeltaTrainingDataGenerator:
    """
    Generate training data for Direct Delta Prediction.
    
    Uses SpliceVarDB variants + base model predictions to create
    (input, target_delta) pairs.
    """
    
    def __init__(
        self,
        config: MetaLayerConfig,
        splicevardb_path: Path,
        window_size: int = 50,  # Positions around variant
        correction_strength: float = 0.3
    ):
        self.config = config
        self.window_size = window_size
        self.correction_strength = correction_strength
        
        # Load resources
        self.registry = Registry(build=config.genome_build)
        self.fasta = self.registry.get_fasta()
        self.base_model = load_base_model(config.base_model)
        
        # Load SpliceVarDB
        self.variants = self._load_splicevardb(splicevardb_path)
    
    def _load_splicevardb(self, path: Path) -> pl.DataFrame:
        """Load and parse SpliceVarDB."""
        df = pl.read_csv(path, separator='\t')
        
        # Parse coordinates for the correct genome build
        coord_col = 'hg38' if 'GRCh38' in self.config.genome_build else 'hg19'
        
        df = df.with_columns([
            pl.col(coord_col).str.strip('"').str.split('-').list.get(0).alias('chrom'),
            pl.col(coord_col).str.strip('"').str.split('-').list.get(1).cast(pl.Int64).alias('var_pos'),
            pl.col(coord_col).str.strip('"').str.split('-').list.get(2).alias('ref'),
            pl.col(coord_col).str.strip('"').str.split('-').list.get(3).alias('alt'),
        ])
        
        return df
    
    def generate_samples(
        self,
        chromosomes: Optional[List[str]] = None
    ) -> Iterator[DeltaTrainingSample]:
        """
        Generate training samples for all variants.
        
        For each variant, yields samples for positions in [-window_size, +window_size].
        """
        
        variants = self.variants
        if chromosomes:
            variants = variants.filter(pl.col('chrom').is_in(chromosomes))
        
        for row in variants.iter_rows(named=True):
            yield from self._generate_samples_for_variant(row)
    
    def _generate_samples_for_variant(
        self,
        variant: Dict
    ) -> Iterator[DeltaTrainingSample]:
        """Generate samples for a single variant."""
        
        chrom = variant['chrom']
        var_pos = variant['var_pos']
        ref_allele = variant['ref']
        alt_allele = variant['alt']
        classification = variant.get('classification', 'Unknown')
        effect_type = variant.get('effect_type')
        
        # Get extended sequence (variant + window on each side + context for 501nt windows)
        start = var_pos - self.window_size - 250
        end = var_pos + self.window_size + 251
        
        ref_extended = self.fasta.fetch(chrom, start, end)
        
        # Create variant sequence
        var_offset = var_pos - start
        alt_extended = (
            ref_extended[:var_offset] + 
            alt_allele + 
            ref_extended[var_offset + len(ref_allele):]
        )
        
        # Generate samples for each position in window
        for offset in range(-self.window_size, self.window_size + 1):
            pos = var_pos + offset
            
            # Extract 501nt windows centered on this position
            local_offset = 250 + offset
            ref_seq = ref_extended[local_offset:local_offset + 501]
            alt_seq = alt_extended[local_offset:local_offset + 501]
            
            if len(ref_seq) != 501 or len(alt_seq) != 501:
                continue  # Skip edge cases
            
            # Run base model
            ref_scores = self.base_model.predict_single(ref_seq)
            alt_scores = self.base_model.predict_single(alt_seq)
            base_delta = alt_scores - ref_scores
            
            # Compute target delta
            target_delta = self._compute_target_delta(
                ref_scores, alt_scores, classification, effect_type
            )
            
            # Sample weight
            weight = self._get_sample_weight(classification, offset)
            
            yield DeltaTrainingSample(
                variant_id=variant.get('variant_id', f"{chrom}:{var_pos}"),
                chrom=chrom,
                position=pos,
                variant_position=var_pos,
                ref_sequence=ref_seq,
                alt_sequence=alt_seq,
                ref_base=self._one_hot(ref_allele[0] if ref_allele else 'N'),
                alt_base=self._one_hot(alt_allele[0] if alt_allele else 'N'),
                ref_scores=ref_scores,
                alt_scores=alt_scores,
                base_delta=base_delta,
                target_delta=target_delta,
                classification=classification,
                effect_type=effect_type,
                sample_weight=weight
            )
    
    def _compute_target_delta(
        self,
        ref_scores: np.ndarray,
        alt_scores: np.ndarray,
        classification: str,
        effect_type: Optional[str]
    ) -> np.ndarray:
        """Compute target delta using hybrid approach."""
        return compute_hybrid_target(
            ref_scores, alt_scores, classification, 
            effect_type or "Unknown",
            self.correction_strength
        )
    
    def _get_sample_weight(self, classification: str, offset: int) -> float:
        """
        Compute sample weight.
        
        Higher weight for:
        - Splice-altering variants
        - Positions close to variant
        """
        base_weight = {
            'Splice-altering': 2.0,
            'Non-splice-altering': 1.5,
            'Low-frequency': 0.5,
        }.get(classification, 1.0)
        
        # Distance decay: positions closer to variant are more important
        distance_factor = 1.0 / (1.0 + 0.1 * abs(offset))
        
        return base_weight * distance_factor
    
    @staticmethod
    def _one_hot(base: str) -> np.ndarray:
        """One-hot encode a nucleotide."""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        vec = np.zeros(4)
        idx = mapping.get(base.upper(), 4)
        if idx < 4:
            vec[idx] = 1.0
        return vec
```

### Comparison: Training Data for Both Approaches

| Aspect | Approach A (Paired) | Approach B (Direct Delta) |
|--------|---------------------|---------------------------|
| **Training input** | ref_seq, alt_seq, features | alt_seq, features, ref_base, alt_base, base_scores |
| **Training target** | ref_labels, alt_labels (from GTF) | target_delta (from hybrid) |
| **Data source** | Base layer artifacts + SpliceVarDB | SpliceVarDB variants only |
| **Samples per variant** | 2 (ref + alt) | ~100 (positions in window) |
| **What model learns** | Predict splice sites for any sequence | Predict correction given variant context |

---

#### Connection to SpliceAI Delta Scores

The meta-layer delta formulation is conceptually similar to SpliceAI's ΔScore:

| SpliceAI ΔScore | Meta-Layer Delta |
|-----------------|------------------|
| ΔDonor_loss | delta[0] < 0 when position is annotated donor |
| ΔDonor_gain | delta[0] > 0 when position is NOT annotated donor |
| ΔAcceptor_loss | delta[1] < 0 when position is annotated acceptor |
| ΔAcceptor_gain | delta[1] > 0 when position is NOT annotated acceptor |

**Key difference**: Meta-layer can learn context-dependent effects that SpliceAI might miss, because:
1. It incorporates base model uncertainty (score features)
2. It's trained with SpliceVarDB-weighted samples
3. It can leverage longer-range context through HyenaDNA

### Comparison: Direct Prediction vs Delta-Learning

| Aspect | Direct Prediction | Delta-Learning |
|--------|-------------------|----------------|
| **Output** | [P(donor), P(acceptor), P(neither)] | [Δ_donor, Δ_acceptor, Δ_neither] |
| **Final score** | Directly from model | base_score + Δ |
| **Loss function** | Cross-entropy | MSE on deltas |
| **Canonical sites** | May forget base model knowledge | Inherently preserves base model |
| **Training signal** | Full probability | Only the correction |
| **Interpretability** | "This is a donor site" | "Boost donor by 0.3" |

### Why Delta-Learning May Be Better for Recalibration

```
Problem with Direct Prediction:
─────────────────────────────────
If base model says [0.95, 0.03, 0.02] for a true donor site,
meta-layer must ALSO learn to output [~1.0, 0, 0].

This is wasteful - the base model already got it right!
The meta-layer might even make it worse.

Delta-Learning Solution:
─────────────────────────────────
If base model says [0.95, 0.03, 0.02] for a true donor site,
delta-layer can output [0.0, 0.0, 0.0] (no correction needed).

For a WRONG prediction [0.3, 0.6, 0.1] where truth is donor,
delta-layer outputs [0.7, -0.6, 0.0] → Final: [1.0, 0.0, 0.1]
```

### Delta-Learning Implementation

```python
class DeltaMetaSpliceModel(nn.Module):
    """
    Meta-layer that predicts corrections to base model scores.
    
    Final prediction = base_score + delta
    """
    
    def __init__(
        self,
        num_score_features: int,
        seq_encoder_type: str = "cnn",
        hidden_dim: int = 256,
        num_classes: int = 3,
        **kwargs
    ):
        super().__init__()
        self.seq_encoder = SequenceEncoderFactory.create(
            seq_encoder_type, output_dim=hidden_dim, **kwargs
        )
        self.score_encoder = ScoreEncoder(
            num_score_features, hidden_dim=hidden_dim, output_dim=hidden_dim
        )
        
        # Output: delta values (can be positive or negative)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
            nn.Tanh()  # Bound deltas to [-1, 1]
        )
        
        # Learnable scaling factor for deltas
        self.delta_scale = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self, 
        sequence_tokens: torch.Tensor,
        score_features: torch.Tensor,
        base_scores: torch.Tensor  # [B, 3] - the base model predictions
    ) -> torch.Tensor:
        # Encode inputs
        seq_emb = self.seq_encoder(sequence_tokens)
        score_emb = self.score_encoder(score_features)
        
        # Fuse embeddings
        combined = torch.cat([seq_emb, score_emb], dim=-1)
        
        # Predict delta
        delta = self.delta_head(combined) * self.delta_scale
        
        # Apply delta to base scores
        adjusted_scores = base_scores + delta
        
        # Normalize to valid probabilities
        adjusted_probs = F.softmax(adjusted_scores, dim=-1)
        
        return adjusted_probs, delta  # Return both for analysis


class DeltaLoss(nn.Module):
    """
    Loss function for delta-learning.
    
    Components:
    1. MSE on delta (match ground truth - base prediction)
    2. Regularization to keep deltas small (prefer minimal correction)
    3. Cross-entropy on final prediction (ensure valid output)
    """
    
    def __init__(self, lambda_reg: float = 0.1, lambda_ce: float = 0.5):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_ce = lambda_ce
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(
        self,
        predicted_probs: torch.Tensor,  # [B, 3] final probabilities
        predicted_delta: torch.Tensor,  # [B, 3] delta values
        base_scores: torch.Tensor,      # [B, 3] base model predictions
        labels: torch.Tensor            # [B] ground truth class
    ) -> torch.Tensor:
        # Target delta: what correction is needed?
        target_probs = F.one_hot(labels, num_classes=3).float()
        target_delta = target_probs - base_scores
        
        # 1. Delta MSE loss
        delta_loss = self.mse(predicted_delta, target_delta)
        
        # 2. Regularization: prefer small deltas
        reg_loss = torch.mean(predicted_delta ** 2)
        
        # 3. Cross-entropy on final prediction
        ce_loss = self.ce(predicted_probs, labels)
        
        total_loss = delta_loss + self.lambda_reg * reg_loss + self.lambda_ce * ce_loss
        
        return total_loss, {
            'delta_mse': delta_loss.item(),
            'regularization': reg_loss.item(),
            'cross_entropy': ce_loss.item()
        }
```

### When to Use Delta-Learning

| Use Case | Recommended Approach | Reason |
|----------|---------------------|--------|
| Base model is already good | Delta-Learning | Preserve existing accuracy |
| Many false positives/negatives | Delta-Learning | Focus on corrections |
| Want interpretable corrections | Delta-Learning | Can inspect Δ values |
| Base model is poor | Direct Prediction | Start fresh |
| Simple deployment | Direct Prediction | Single model output |

### Evaluation for Delta-Learning

Additional metrics specific to delta-learning:

```python
def evaluate_delta_learning(
    deltas: np.ndarray,          # [N, 3] predicted deltas
    base_scores: np.ndarray,     # [N, 3] base model scores
    labels: np.ndarray,          # [N] ground truth
    variant_annotations: pd.Series
) -> Dict[str, float]:
    """Evaluate delta-learning specific metrics."""
    
    results = {}
    
    # Average delta magnitude (should be small for correct predictions)
    correct_mask = np.argmax(base_scores, axis=1) == labels
    results['avg_delta_correct'] = np.abs(deltas[correct_mask]).mean()
    results['avg_delta_incorrect'] = np.abs(deltas[~correct_mask]).mean()
    
    # Delta should be larger for incorrect predictions
    results['delta_discrimination'] = (
        results['avg_delta_incorrect'] / (results['avg_delta_correct'] + 1e-8)
    )
    
    # For splice-altering variants: are deltas meaningful?
    if variant_annotations is not None:
        splice_altering = variant_annotations == 'Splice-altering'
        results['avg_delta_splice_altering'] = np.abs(deltas[splice_altering]).mean()
        results['avg_delta_non_splice_altering'] = np.abs(
            deltas[variant_annotations == 'Non-splice-altering']
        ).mean()
    
    return results
```

**Why we use direct prediction instead of delta (currently):**
1. Simpler loss function (cross-entropy vs MSE on deltas)
2. No need for calibrated base model outputs
3. Consistent with base model training objective
4. Easier to interpret predictions

**When to switch to delta-learning:**
1. If direct prediction degrades canonical performance
2. If we want more interpretable corrections
3. If base model is already very good and we just need fine-tuning

---

## Data Preparation Pipeline

### Step-by-Step Label Processing

```python
from meta_spliceai.splice_engine.meta_layer.data import prepare_training_dataset

def prepare_training_dataset(
    config: MetaLayerConfig,
    variant_source: str = 'splicevardb'
) -> pl.DataFrame:
    """
    Prepare training dataset with labels and optional variant annotations.
    
    Steps:
    1. Load base layer artifacts (includes labels from GTF)
    2. Encode labels to integers
    3. Optionally annotate with variant classifications
    4. Assign sample weights
    5. Return training-ready DataFrame
    """
    
    # Step 1: Load artifacts
    loader = ArtifactLoader(config)
    df = loader.load_analysis_sequences()
    
    # Step 2: Encode labels
    df = df.with_columns(
        pl.col('splice_type')
        .fill_null('')
        .replace(LABEL_ENCODING)
        .alias('label')
    )
    
    # Step 3: Add variant annotations (if requested)
    if variant_source == 'splicevardb':
        df = annotate_with_splicevardb(df, config)
    
    # Step 4: Assign sample weights
    df = df.with_columns(
        pl.col('variant_classification')
        .apply(get_sample_weight)
        .alias('sample_weight')
    )
    
    return df
```

### Label Verification

```python
def verify_labels(df: pl.DataFrame) -> None:
    """Verify label distribution and quality."""
    
    # Check label distribution
    label_counts = df.group_by('label').count()
    print("Label distribution:")
    for row in label_counts.iter_rows(named=True):
        print(f"  {LABEL_DECODING[row['label']]}: {row['count']:,}")
    
    # Verify no null labels
    null_count = df.filter(pl.col('label').is_null()).height
    assert null_count == 0, f"Found {null_count} null labels!"
    
    # Check label consistency with base model predictions
    # (donor labels should have high donor_score, etc.)
    for label_idx, label_name in LABEL_DECODING.items():
        subset = df.filter(pl.col('label') == label_idx)
        
        if label_name == 'donor':
            avg_score = subset['donor_score'].mean()
        elif label_name == 'acceptor':
            avg_score = subset['acceptor_score'].mean()
        else:
            avg_score = subset['neither_score'].mean()
        
        print(f"  {label_name}: avg matching score = {avg_score:.3f}")
```

---

## SpliceVarDB Integration Details

### Matching Variants to Positions

```python
def annotate_with_splicevardb(
    df: pl.DataFrame,
    config: MetaLayerConfig
) -> pl.DataFrame:
    """
    Annotate positions with SpliceVarDB variant classifications.
    
    Parameters
    ----------
    df : pl.DataFrame
        Base layer positions with columns: chrom, position
    config : MetaLayerConfig
        Configuration with genome build info
    
    Returns
    -------
    pl.DataFrame
        Original dataframe with added columns:
        - variant_id: SpliceVarDB variant ID (if any)
        - variant_classification: 'Splice-altering', 'Low-frequency', etc.
        - variant_method: Experimental validation method
    """
    
    # Load SpliceVarDB
    splicevardb = load_splicevardb()
    
    # Parse coordinates based on genome build
    coord_col = config.coordinate_column  # 'hg19' or 'hg38'
    
    # Create position key in SpliceVarDB
    # Format: "1-12345-A-G" → chrom="1", pos=12345
    variants = splicevardb.with_columns([
        pl.col(coord_col).str.strip('"').str.split('-').list.get(0).alias('var_chrom'),
        pl.col(coord_col).str.strip('"').str.split('-').list.get(1).cast(pl.Int64).alias('var_pos'),
    ])
    
    # Create position key in base layer data
    df = df.with_columns([
        pl.concat_str([
            pl.col('chrom'),
            pl.lit(':'),
            pl.col('position').cast(pl.Utf8)
        ]).alias('pos_key')
    ])
    
    # Create position key in variants
    variants = variants.with_columns([
        pl.concat_str([
            pl.col('var_chrom'),
            pl.lit(':'),
            pl.col('var_pos').cast(pl.Utf8)
        ]).alias('pos_key')
    ])
    
    # Left join
    annotated = df.join(
        variants.select(['pos_key', 'variant_id', 'classification', 'method']),
        on='pos_key',
        how='left'
    )
    
    # Rename columns
    annotated = annotated.rename({
        'classification': 'variant_classification',
        'method': 'variant_method'
    })
    
    # Log statistics
    n_annotated = annotated.filter(~pl.col('variant_classification').is_null()).height
    print(f"Annotated {n_annotated} positions with SpliceVarDB variants")
    
    return annotated
```

---

## Important Clarification: SpliceVarDB Classifications vs. Labels

### SpliceVarDB Classification ≠ Meta-Layer Label

```
SpliceVarDB data:
┌──────────────────────────────────────────────────────────────────┐
│ variant_id: "rs123456"                                           │
│ position: chr1:12345                                             │
│ classification: "Splice-altering"  ← About the VARIANT effect   │
│ effect: "Cryptic donor activation"                               │
└──────────────────────────────────────────────────────────────────┘

Meta-layer training data:
┌──────────────────────────────────────────────────────────────────┐
│ position: chr1:12345                                             │
│ splice_type: "neither"             ← From GTF (no canonical site)│
│ label: 2 (neither)                 ← UNCHANGED by variant!       │
│ sample_weight: 2.0                 ← INCREASED due to variant    │
│ variant_classification: "Splice-altering"  ← For context         │
└──────────────────────────────────────────────────────────────────┘
```

### Why We Don't Change Labels Based on Variants

1. **Training objective consistency**: Meta-layer should predict like base model (canonical splice sites)
2. **Variant effects are conditional**: A variant might create a splice site, but only when the variant is present
3. **Weighting is safer**: Emphasize these positions without changing the learning target

### Alternative Approach: Multi-Task Learning (Future)

For explicitly predicting alternative splice sites, consider:

```python
# Future: Multi-task output
output = {
    'canonical': [P(donor), P(acceptor), P(neither)],  # Standard task
    'alt_splice_potential': P(could become splice site given variant)  # New task
}
```

---

## Summary: Label Strategy

| Question | Answer |
|----------|--------|
| **Where do labels come from?** | GTF/GFF annotations (same as base model training) |
| **Are labels in base layer artifacts?** | Yes! The `splice_type` column |
| **Does SpliceVarDB provide new labels?** | No, it provides context/weights |
| **What do labels represent?** | Canonical splice site status at each position |
| **How are variants used?** | To identify important positions and adjust training weights |
| **Is the output format the same as base model?** | Yes! P(donor), P(acceptor), P(neither) |
| **Can SpliceVarDB change a label?** | No - labels are always from GTF |
| **What if a variant creates a new splice site?** | Label stays "neither", but weight increases |

---

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system architecture
- [ALTERNATIVE_SPLICING_PIPELINE.md](ALTERNATIVE_SPLICING_PIPELINE.md) - From scores to exon predictions
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Step-by-step training instructions
- [methods/ROADMAP.md](methods/ROADMAP.md) - **Methodology development roadmap**
- [methods/PAIRED_DELTA_PREDICTION.md](methods/PAIRED_DELTA_PREDICTION.md) - Paired delta prediction (deprecated)
- [methods/VALIDATED_DELTA_PREDICTION.md](methods/VALIDATED_DELTA_PREDICTION.md) - **Validated delta prediction** (recommended)
- [MULTI_STEP_FRAMEWORK.md](MULTI_STEP_FRAMEWORK.md) - Decomposed problem approach

---

## Update: Methodology Evolution (December 15, 2025)

### Clarification: Approach A vs Approach B vs Multi-Step Framework

This document describes two approaches for variant effect prediction:

1. **Approach A (Section above)**: Paired Prediction
   - Input: ref_seq + alt_seq
   - Target: base_model(alt) - base_model(ref)
   - **Status**: Tested, r=0.38 correlation (insufficient)
   - **Limitation**: Targets may be inaccurate if base model is wrong

2. **Approach B (Section above)**: Single-Pass Delta Prediction
   - Input: alt_seq + variant_info (ref_base, alt_base)
   - Target: SpliceVarDB-validated delta
   - **Status**: Proposed, not yet implemented
   - **Advantage**: Uses ground truth labels

### What Was Incorrectly Labeled "Approach B"

The binary classifier (`SpliceInducingClassifier`) tested on December 15, 2025 is **NOT Approach B**. It is actually **Step 1 of the Multi-Step Framework**:

| What Was Tested | Correct Name | Description |
|-----------------|--------------|-------------|
| `SpliceInducingClassifier` | Multi-Step Step 1 | Binary: "Is this variant splice-altering?" |

### Multi-Step Framework (NEW)

A decomposed approach with multiple sub-tasks:

1. **Step 1**: Binary Classification - "Is this variant splice-altering?"
   - Tested: AUC=0.61, F1=0.53 (needs F1 > 0.7)
   
2. **Step 2**: Effect Type Classification (not yet implemented)
   
3. **Step 3**: Position Localization (not yet implemented)
   
4. **Step 4**: Delta Magnitude (not yet implemented)

See [methods/ROADMAP.md](methods/ROADMAP.md) for the complete methodology development plan.

---

*Last Updated: December 15, 2025*

