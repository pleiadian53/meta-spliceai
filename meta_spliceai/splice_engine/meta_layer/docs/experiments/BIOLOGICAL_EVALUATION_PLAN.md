# Biological Evaluation Plan for Approach B

**Purpose**: Evaluate delta scores in a biologically interpretable way  
**Status**: Design Document  
**Last Updated**: December 16, 2025

---

## Overview

Current evaluation uses correlation between predicted and target deltas. While useful, this doesn't directly test the biological hypothesis: **"Do delta-adjusted scores correctly identify variant-induced splice site changes?"**

This document outlines a more biologically interpretable evaluation strategy.

---

## Question 1: Is 50K Variants Worth It?

### Current Results

| Samples | Correlation | ROC-AUC | PR-AUC | Improvement |
|---------|-------------|---------|--------|-------------|
| 2,000 | r=0.41 | 0.58 | 0.62 | Baseline |
| 8,000 | **r=0.507** | **0.589** | **0.633** | **+24% correlation** |

### Analysis

**Pros of scaling to 50K:**
1. **Statistical power**: 50K samples would provide much better estimates of generalization
2. **Diminishing returns expected**: 2K→8K showed +24% improvement. 8K→50K likely +10-15% more
3. **Better class balance**: More examples of rare splice-altering patterns
4. **Cross-validation**: With 50K, can do proper 5-fold CV for robust estimates
5. **GPU efficiency**: Training on 50K is still fast on GPU (~1-2 hours)

**Cons:**
1. **Diminishing returns**: Each additional sample provides less marginal benefit
2. **Data quality**: Need to ensure all 50K variants are high-quality
3. **Training time**: Still manageable but longer

### Recommendation: **YES, scale to 50K**

**Rationale:**
- Current model likely still underfitting (only 8K samples)
- 50K would give proper statistical power for evaluation
- Can do proper train/val/test splits and cross-validation
- GPU training makes it feasible
- Expected improvement: r=0.55-0.60 (modest but meaningful)

**Implementation:**
```python
# In test_validated_delta_experiments.py
config_50k = ExperimentConfig(
    name="Full SpliceVarDB (50K)",
    max_train=50000,  # Use all available
    max_test=5000,
    epochs=50,
    batch_size=128,  # Larger batch for GPU
    description="Full dataset for robust evaluation"
)
```

---

## Question 2: Biologically Interpretable Evaluation

### Current Evaluation (Limitations)

Current metrics:
- **Correlation**: How well predicted delta matches target delta
- **ROC-AUC**: Can we detect splice-altering vs normal variants?
- **PR-AUC**: Precision-recall for effect detection

**Problem**: These don't directly test whether delta-adjusted scores correctly identify **actual splice sites**.

### Proposed: Splice Site Peak Detection Evaluation

**Idea**: Use delta-adjusted scores to predict splice sites, then compare to annotations.

#### Step 1: Apply Delta to Base Scores

```python
def get_adjusted_scores(variant, base_model, delta_model, fasta):
    """
    Get splice site scores after applying delta correction.
    
    Returns:
    - base_scores: [L, 3] original base model scores
    - delta: [3] predicted delta
    - adjusted_scores: [L, 3] base_scores + delta (broadcast)
    """
    # Get alt sequence
    alt_seq = get_alt_sequence(variant, fasta)
    
    # Base model prediction
    base_scores = base_model(alt_seq)  # [L, 3]
    
    # Meta-layer delta prediction
    delta = delta_model(alt_seq, variant_info)  # [3]
    
    # Apply delta (broadcast to all positions)
    adjusted_scores = base_scores + delta.unsqueeze(0)  # [L, 3]
    
    return {
        'base_scores': base_scores,
        'delta': delta,
        'adjusted_scores': adjusted_scores
    }
```

#### Step 2: Find Splice Site Peaks

```python
def find_splice_site_peaks(scores, threshold=0.5, window_size=5):
    """
    Find donor/acceptor peaks in probability scores.
    
    Parameters:
    - scores: [L, 3] probability scores
    - threshold: Minimum probability to consider
    - window_size: Local maximum window
    
    Returns:
    - donor_peaks: List of (position, probability) tuples
    - acceptor_peaks: List of (position, probability) tuples
    """
    donor_scores = scores[:, 0]  # Donor probabilities
    acceptor_scores = scores[:, 1]  # Acceptor probabilities
    
    # Find local maxima above threshold
    donor_peaks = find_local_maxima(donor_scores, threshold, window_size)
    acceptor_peaks = find_local_maxima(acceptor_scores, threshold, window_size)
    
    return donor_peaks, acceptor_peaks

def find_local_maxima(scores, threshold, window):
    """Find local maxima in 1D array."""
    peaks = []
    for i in range(window, len(scores) - window):
        if scores[i] >= threshold:
            if scores[i] == max(scores[i-window:i+window+1]):
                peaks.append((i, float(scores[i])))
    return peaks
```

#### Step 3: Compare to Canonical Annotations

```python
def evaluate_splice_site_predictions(
    predicted_peaks,
    annotated_sites,
    variant_position,
    tolerance=2
):
    """
    Evaluate whether predicted peaks match annotated sites.
    
    For splice-altering variants:
    - Check if NEW peaks appear (gain)
    - Check if existing peaks disappear (loss)
    
    For normal variants:
    - Check if no spurious peaks appear
    """
    # Get annotated sites near variant
    nearby_annotated = get_nearby_annotated_sites(
        annotated_sites, variant_position, window=500
    )
    
    # Match predicted to annotated
    matches = match_peaks_to_annotations(
        predicted_peaks, nearby_annotated, tolerance
    )
    
    return {
        'true_positives': matches['tp'],
        'false_positives': matches['fp'],
        'false_negatives': matches['fn'],
        'precision': len(matches['tp']) / max(1, len(predicted_peaks)),
        'recall': len(matches['tp']) / max(1, len(nearby_annotated))
    }
```

#### Step 4: Variant-Specific Evaluation

```python
def evaluate_variant_effect(
    variant,
    base_scores,
    adjusted_scores,
    annotated_sites
):
    """
    Evaluate whether delta correctly captures variant effect.
    
    For Splice-altering variants:
    - Adjusted scores should show NEW or STRONGER peaks
    - These should align with known variant-induced sites
    
    For Normal variants:
    - Adjusted scores should NOT create spurious peaks
    - Should maintain canonical site predictions
    """
    # Find peaks in base vs adjusted
    base_peaks = find_splice_site_peaks(base_scores)
    adjusted_peaks = find_splice_site_peaks(adjusted_scores)
    
    # Compare
    if variant.classification == 'Splice-altering':
        # Should see gain or loss
        new_peaks = set(adjusted_peaks) - set(base_peaks)
        lost_peaks = set(base_peaks) - set(adjusted_peaks)
        
        # Check if changes align with annotations
        # (This requires SpliceVarDB to specify affected sites)
        return {
            'has_gain': len(new_peaks) > 0,
            'has_loss': len(lost_peaks) > 0,
            'new_peaks': new_peaks,
            'lost_peaks': lost_peaks
        }
    else:  # Normal
        # Should NOT create spurious peaks
        spurious = set(adjusted_peaks) - set(base_peaks)
        return {
            'spurious_peaks': len(spurious),
            'maintains_canonical': len(set(base_peaks) & set(adjusted_peaks)) / max(1, len(base_peaks))
        }
```

---

## Implementation Plan

### Phase 1: Basic Peak Detection

1. Implement `find_splice_site_peaks()` function
2. Test on known canonical sites (should find them)
3. Compare base vs adjusted peaks

### Phase 2: Variant Effect Evaluation

1. For each variant, compute base and adjusted scores
2. Find peaks in both
3. Compare to nearby annotated sites
4. Compute precision/recall

### Phase 3: SpliceVarDB Alignment

1. For splice-altering variants, check if new peaks align with known effects
2. For normal variants, check if no spurious peaks appear
3. Aggregate metrics across all variants

---

## Expected Metrics

### Splice Site Detection Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Peak Precision** | % of predicted peaks that match annotations | >0.7 |
| **Peak Recall** | % of annotated sites found | >0.6 |
| **Gain Detection** | % of splice-altering variants with detected gains | >0.5 |
| **False Gain Rate** | % of normal variants with spurious peaks | <0.1 |

### Comparison to Current Metrics

| Current Metric | Biological Equivalent |
|----------------|----------------------|
| Correlation (r=0.507) | How well delta magnitude matches target |
| ROC-AUC (0.589) | Can we detect splice-altering variants? |
| **NEW: Peak Precision** | Do adjusted scores find correct splice sites? |
| **NEW: Gain Detection** | Do we detect variant-induced splice sites? |

---

## Files to Create

1. `evaluation/biological_evaluation.py` - Main evaluation functions
2. `evaluation/peak_detection.py` - Splice site peak finding
3. `tests/test_biological_evaluation.py` - Test script

---

## Next Steps

1. ✅ Design evaluation strategy (this document)
2. ⏳ Implement peak detection functions
3. ⏳ Integrate with existing test script
4. ⏳ Run on 8K test set first
5. ⏳ Scale to 50K with biological evaluation

---

*This evaluation directly tests the biological hypothesis: "Does the delta correction improve splice site prediction accuracy?"*

