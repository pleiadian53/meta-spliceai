# Delta Score Implementation

**Last Updated**: December 2025  
**Status**: Implementation Guide

---

## Overview

Delta scores quantify how a genetic variant affects splice site predictions. This document describes:

1. What delta scores represent
2. How they are computed (base layer vs meta-layer)
3. Format specification for model comparison
4. Integration with SpliceVarDB evaluation

---

## What Are Delta Scores?

Delta scores measure the **change in splice site probability** caused by a variant:

```
Δ = P(splice site | variant sequence) - P(splice site | reference sequence)

Positive Δ → Variant INCREASES splice probability (gain)
Negative Δ → Variant DECREASES splice probability (loss)
```

### Delta Score Types

| Delta Type | Formula | Interpretation |
|------------|---------|----------------|
| Δ_donor | P(donor|alt) - P(donor|ref) | Donor site gain (+) or loss (-) |
| Δ_acceptor | P(acceptor|alt) - P(acceptor|ref) | Acceptor site gain (+) or loss (-) |
| Δ_neither | P(neither|alt) - P(neither|ref) | Change in non-splice probability |

### SpliceAI-Style Delta Scores

The original SpliceAI uses four summary scores:

| Score | Description | Our Equivalent |
|-------|-------------|----------------|
| DS_AG (Acceptor Gain) | Max positive Δ_acceptor | `max(Δ_acceptor, 0)` |
| DS_AL (Acceptor Loss) | Max negative Δ_acceptor (as positive) | `max(-Δ_acceptor, 0)` |
| DS_DG (Donor Gain) | Max positive Δ_donor | `max(Δ_donor, 0)` |
| DS_DL (Donor Loss) | Max negative Δ_donor (as positive) | `max(-Δ_donor, 0)` |

---

## Computation Pipeline

### Step 1: Sequence Generation

```python
# Given a variant at position P with ref=G, alt=A
ref_sequence = fasta[chrom][start:end]  # e.g., 501nt centered on P
alt_sequence = ref_sequence[:250] + 'A' + ref_sequence[251:]  # Substitute at center
```

### Step 2: Model Predictions

```python
# Base model predictions
ref_probs_base = base_model(ref_sequence)  # [L, 3]
alt_probs_base = base_model(alt_sequence)  # [L, 3]

# Meta-layer predictions (uses base model features)
ref_features = extract_features(ref_sequence, ref_probs_base)
alt_features = extract_features(alt_sequence, alt_probs_base)

ref_probs_meta = meta_model(ref_sequence, ref_features)  # [L, 3]
alt_probs_meta = meta_model(alt_sequence, alt_features)  # [L, 3]
```

### Step 3: Delta Computation

```python
# Base layer delta
base_delta = alt_probs_base - ref_probs_base  # [L, 3]

# Meta-layer delta
meta_delta = alt_probs_meta - ref_probs_meta  # [L, 3]

# Extract at variant position
base_delta_at_variant = base_delta[variant_pos]  # [3]
meta_delta_at_variant = meta_delta[variant_pos]  # [3]
```

---

## Output Format Specification

### Per-Position Delta (for detailed analysis)

```python
@dataclass
class DeltaScoreResult:
    """Delta scores between reference and variant sequences."""
    
    # Full position-level delta [L, 3]
    delta_probabilities: np.ndarray
    
    # Delta at variant position [3]
    delta_donor: float      # Δ_donor at variant position
    delta_acceptor: float   # Δ_acceptor at variant position  
    delta_neither: float    # Δ_neither at variant position
    
    # Max deltas across sequence (SpliceAI-style)
    max_donor_gain: float     # DS_DG: max positive Δ_donor
    max_donor_loss: float     # DS_DL: max negative Δ_donor (as positive)
    max_acceptor_gain: float  # DS_AG: max positive Δ_acceptor
    max_acceptor_loss: float  # DS_AL: max negative Δ_acceptor (as positive)
    
    # Positions of max deltas
    pos_max_donor_gain: int
    pos_max_donor_loss: int
    pos_max_acceptor_gain: int
    pos_max_acceptor_loss: int
```

### Comparison Format (for evaluation)

```python
@dataclass
class DeltaComparison:
    """Comparison between base and meta-layer delta scores."""
    
    variant_id: int
    chrom: str
    position: int
    classification: str  # SpliceVarDB classification
    
    # Base model deltas
    base_delta_donor: float
    base_delta_acceptor: float
    base_delta_neither: float
    
    # Meta-layer deltas
    meta_delta_donor: float
    meta_delta_acceptor: float
    meta_delta_neither: float
    
    # Comparison metrics
    @property
    def base_detects_effect(self) -> bool:
        """Whether base model detects significant splice effect."""
        return max(abs(self.base_delta_donor), abs(self.base_delta_acceptor)) > 0.1
    
    @property
    def meta_detects_effect(self) -> bool:
        """Whether meta-layer detects significant splice effect."""
        return max(abs(self.meta_delta_donor), abs(self.meta_delta_acceptor)) > 0.1
    
    @property
    def meta_improves(self) -> bool:
        """Whether meta-layer detects effect that base missed."""
        return self.meta_detects_effect and not self.base_detects_effect
```

---

## Implementation Details

### Context Trimming

The base model (OpenSpliceAI/SpliceAI) trims context from both ends of the output:

```
Input sequence:   |-------- L nucleotides --------|
Model output:     |--C/2--|-- L-C positions --|--C/2--|
                  trimmed     usable output    trimmed

C = context length (e.g., 10000 for 10000nt flanking model)
```

When computing delta at a specific variant position:

```python
# Convert input position to output position
output_position = variant_position - (context_length // 2)
output_position = max(0, min(output_position, output_length - 1))
```

### Handling Indels

For insertions/deletions, the ref and alt sequences have different lengths:

```python
# Deletion: ref has more bases than alt
ref_seq = "ACGTACGT"  # 8nt
alt_seq = "ACGT"      # 4nt (deleted ACGT)

# Insertion: alt has more bases than ref
ref_seq = "ACGT"      # 4nt
alt_seq = "ACGTACGT"  # 8nt (inserted ACGT)

# Solution: Use minimum length and adjust positions
min_len = min(len(ref_probs), len(alt_probs))
delta = alt_probs[:min_len] - ref_probs[:min_len]
```

### Strand Handling

For genes on the negative strand:

```python
if strand == '-':
    # Reverse complement the sequences before prediction
    ref_seq_rc = reverse_complement(ref_seq)
    alt_seq_rc = reverse_complement(alt_seq)
    
    # Run predictions
    ref_probs = model(ref_seq_rc)
    alt_probs = model(alt_seq_rc)
    
    # Reverse the output
    ref_probs = ref_probs[::-1]
    alt_probs = alt_probs[::-1]
```

---

## Comparison: Base vs Meta-Layer

### What We're Measuring

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EVALUATION FRAMEWORK                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  For each SpliceVarDB variant:                                      │
│                                                                     │
│  1. Generate ref and alt sequences                                  │
│  2. Compute base_delta (base model only)                            │
│  3. Compute meta_delta (base model + meta-layer)                    │
│  4. Compare with SpliceVarDB classification:                        │
│                                                                     │
│     If classification == "Splice-altering":                         │
│       Expected: Large |delta|                                       │
│       Success: Model detects the effect                             │
│                                                                     │
│     If classification == "Non-splice-altering":                     │
│       Expected: Small |delta| (near 0)                              │
│       Success: Model does NOT falsely detect effect                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Detection Rate** | TP / (TP + FN) | Sensitivity to splice-altering variants |
| **False Positive Rate** | FP / (FP + TN) | Specificity for non-splice-altering variants |
| **Improvement Count** | Cases where meta detects, base doesn't | Added value of meta-layer |
| **Degradation Count** | Cases where base detects, meta doesn't | Potential overfitting |

### Expected Outcome

```
Goal: meta_delta captures splice effects that base_delta misses

┌────────────────────────────────────────────────────────────┐
│ Splice-altering variant at cryptic donor site             │
│                                                            │
│ Base model:  Δ_donor = 0.05 (weak signal, below threshold)│
│ Meta-layer:  Δ_donor = 0.35 (strong signal, detected!)    │
│                                                            │
│ → Meta-layer IMPROVES detection                            │
└────────────────────────────────────────────────────────────┘
```

---

## Code Location

| Component | File | Description |
|-----------|------|-------------|
| Base Model Predictor | `inference/base_model_predictor.py` | `BaseModelPredictor.compute_delta()` |
| Variant Evaluator | `training/variant_evaluator.py` | `VariantEffectEvaluator.evaluate()` |
| Delta Result | `inference/base_model_predictor.py` | `DeltaScoreResult` dataclass |
| Comparison Result | `training/variant_evaluator.py` | `DeltaResult` dataclass |

---

## Usage Example

```python
from meta_spliceai.splice_engine.meta_layer.inference import (
    BaseModelPredictor,
    get_base_model_predictor
)
from meta_spliceai.splice_engine.meta_layer.training import (
    VariantEffectEvaluator,
    evaluate_variant_effects
)
from meta_spliceai.splice_engine.meta_layer.data import load_splicevardb

# Load base model predictor
base_predictor = get_base_model_predictor('openspliceai', flanking_size=10000)

# Compute delta for a single variant
ref_seq = "ACGT..." * 1000  # Reference sequence
alt_seq = "ACGA..." * 1000  # With G>A variant

delta = base_predictor.compute_delta(ref_seq, alt_seq, variant_position=500)
print(f"Donor delta: {delta.delta_donor:.4f}")
print(f"Max donor gain: {delta.max_donor_gain:.4f}")

# Evaluate meta-layer on SpliceVarDB
from meta_spliceai.splice_engine.meta_layer.models import MetaSpliceModel
from meta_spliceai.splice_engine.meta_layer.core.config import MetaLayerConfig

config = MetaLayerConfig(base_model='openspliceai')
meta_model = MetaSpliceModel(...)  # Load trained model

result = evaluate_variant_effects(
    meta_model=meta_model,
    config=config,
    test_chromosomes=['21', '22']
)

print(f"Base detection rate: {result.base_detection_rate:.1%}")
print(f"Meta detection rate: {result.meta_detection_rate:.1%}")
print(f"Improvement: {result.improvement_count} variants")
```

---

## Related Documentation

- [LABELING_STRATEGY.md](LABELING_STRATEGY.md) - Training labels and SpliceVarDB integration
- [ARCHITECTURE.md](ARCHITECTURE.md) - Meta-layer model architecture
- [TRAINING_VS_INFERENCE.md](TRAINING_VS_INFERENCE.md) - Subsampled training vs full coverage inference

---

*Last Updated: December 2025*

