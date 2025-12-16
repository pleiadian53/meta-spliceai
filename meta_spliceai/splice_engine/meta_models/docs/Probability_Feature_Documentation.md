# Probability Feature Documentation

**Version:** 1.0  
**Date:** July 10, 2025  
**Purpose:** Comprehensive guide to probability-based and context-based features used in SpliceAI meta-models

## Table of Contents

1. [Overview](#overview)
2. [Basic Probability Features](#basic-probability-features)
3. [Context Score Features](#context-score-features)
4. [Signal Processing Features](#signal-processing-features)
5. [Cross-Type Comparison Features](#cross-type-comparison-features)
6. [Feature Interpretation Guide](#feature-interpretation-guide)
7. [Use Cases and Applications](#use-cases-and-applications)

---

## Overview

The SpliceAI meta-model uses three types of derived features to improve splice site prediction accuracy:

1. **Raw Probability Scores**: Direct outputs from SpliceAI (donor_score, acceptor_score, neither_score)
2. **Context Scores**: Probabilities at nearby positions (±1, ±2 nucleotides)
3. **Derived Features**: Mathematical transformations and signal processing features

These features help the meta-model distinguish between true positives and false positives, and identify missed splice sites (false negatives).

---

## Basic Probability Features

### Raw Probability Scores

| Feature | Description | Range | Interpretation |
|---------|-------------|-------|----------------|
| `donor_score` | SpliceAI probability for donor splice site | [0, 1] | Higher values indicate stronger donor signal |
| `acceptor_score` | SpliceAI probability for acceptor splice site | [0, 1] | Higher values indicate stronger acceptor signal |
| `neither_score` | SpliceAI probability for non-splice position | [0, 1] | Higher values indicate non-splice regions |

### Normalized Probability Features

| Feature | Description | Formula | Use Case |
|---------|-------------|---------|----------|
| `relative_donor_probability` | Relative strength of donor vs acceptor | `donor / (donor + acceptor)` | Distinguishing donor from acceptor sites |
| `splice_probability` | Combined splice site probability | `(donor + acceptor) / (donor + acceptor + neither)` | Overall splice site confidence |
| `probability_entropy` | Uncertainty in probability distribution | `-Σ(p × log(p))` | Measuring prediction confidence |

### Differential Features

| Feature | Description | Formula | Biological Meaning |
|---------|-------------|---------|-------------------|
| `donor_acceptor_diff` | Relative difference between donor/acceptor | `(donor - acceptor) / max(donor, acceptor)` | Type specificity strength |
| `splice_neither_diff` | Splice vs non-splice strength | `max(donor, acceptor) - neither) / max(all)` | Overall splice signal strength |
| `donor_acceptor_logodds` | Log-odds ratio of donor vs acceptor | `log(donor) - log(acceptor)` | Statistical significance of type preference |
| `splice_neither_logodds` | Log-odds ratio of splice vs non-splice | `log(donor + acceptor) - log(neither)` | Statistical significance of splice prediction |

---

## Context Score Features

### Context Positions

Context scores represent SpliceAI probabilities at nearby positions:

- `context_score_m2`: Score at position -2 (2 nucleotides upstream)
- `context_score_m1`: Score at position -1 (1 nucleotide upstream)  
- `context_score_p1`: Score at position +1 (1 nucleotide downstream)
- `context_score_p2`: Score at position +2 (2 nucleotides downstream)

### Context-Agnostic Features

| Feature | Description | Formula | Purpose |
|---------|-------------|---------|---------|
| `context_neighbor_mean` | Average of neighboring scores | `(m2 + m1 + p1 + p2) / 4` | Local background level |
| `context_asymmetry` | Upstream vs downstream bias | `(m1 + m2) - (p1 + p2)` | Directional signal pattern |
| `context_max` | Maximum neighboring score | `max(m2, m1, p1, p2)` | Nearby peak detection |

---

## Signal Processing Features

These features apply signal processing concepts to identify true splice sites based on their characteristic patterns.

### Peak Detection Features

| Feature | Description | Biological Rationale |
|---------|-------------|---------------------|
| `donor_is_local_peak` | Boolean: Is position a local maximum? | True splice sites show sharp peaks |
| `acceptor_is_local_peak` | Boolean: Is position a local maximum? | True splice sites show sharp peaks |

**Formula**: `(score > m1) & (score > p1) & (score > 0.001)`

### Peak Height Analysis

| Feature | Description | Formula | Interpretation |
|---------|-------------|---------|----------------|
| `donor_peak_height_ratio` | How many times higher than neighbors | `donor / (neighbor_mean + ε)` | Peak prominence measure |
| `acceptor_peak_height_ratio` | How many times higher than neighbors | `acceptor / (neighbor_mean + ε)` | Peak prominence measure |

**High values** (>2.0): Strong, isolated peaks → likely true positives  
**Low values** (<1.5): Weak or broad signals → likely false positives

### Signal Derivatives

| Feature | Description | Formula | Signal Processing Concept |
|---------|-------------|---------|--------------------------|
| `donor_second_derivative` | Rate of change of rate of change | `(donor - m1) - (p1 - donor)` | Curvature at peak |
| `acceptor_second_derivative` | Rate of change of rate of change | `(acceptor - m1) - (p1 - acceptor)` | Curvature at peak |

**Positive values**: Concave up (sharp peak) → true splice sites  
**Negative values**: Concave down (broad signal) → false positives

### Signal Strength Features

| Feature | Description | Formula | Use Case |
|---------|-------------|---------|----------|
| `donor_signal_strength` | Signal above background | `donor - neighbor_mean` | Background subtraction |
| `acceptor_signal_strength` | Signal above background | `acceptor - neighbor_mean` | Background subtraction |

### Context Differential Features

| Feature | Description | Purpose |
|---------|-------------|---------|
| `donor_diff_m1`, `donor_diff_m2` | Rise from upstream positions | Detecting signal onset |
| `donor_diff_p1`, `donor_diff_p2` | Fall to downstream positions | Detecting signal offset |
| `acceptor_diff_m1`, etc. | Same for acceptor positions | Acceptor signal analysis |

### Advanced Ratios

| Feature | Description | Formula | Interpretation |
|---------|-------------|---------|----------------|
| `donor_surge_ratio` | Peak vs immediate neighbors | `donor / (m1 + p1 + ε)` | Local prominence |
| `donor_context_diff_ratio` | Peak vs highest neighbor | `donor / max(neighbors) + ε)` | Relative peak strength |

---

## Cross-Type Comparison Features

These features compare donor and acceptor patterns to distinguish between splice site types.

| Feature | Description | Formula | Use Case |
|---------|-------------|---------|----------|
| `donor_acceptor_peak_ratio` | Ratio of peak heights | `donor_peak_height / acceptor_peak_height` | Type discrimination |
| `type_signal_difference` | Difference in signal strengths | `donor_signal_strength - acceptor_signal_strength` | Type preference |
| `score_difference_ratio` | Normalized score difference | `(donor - acceptor) / (donor + acceptor)` | Type confidence |
| `signal_strength_ratio` | Ratio of signal strengths | `donor_signal_strength / acceptor_signal_strength` | Relative type strength |

---

## Feature Interpretation Guide

### For False Positive Reduction

**High FP Risk Indicators:**
- `donor_peak_height_ratio < 1.5` (weak peak)
- `donor_second_derivative < 0` (broad signal)
- `donor_is_local_peak = False` (not a local maximum)
- `splice_neither_diff < 0.1` (weak splice signal)

**Low FP Risk Indicators:**
- `donor_peak_height_ratio > 3.0` (sharp peak)
- `donor_second_derivative > 0.05` (strong curvature)
- `splice_probability > 0.8` (high confidence)

### For False Negative Rescue

**Missed Splice Site Indicators:**
- `splice_probability > 0.3` but `prediction = neither` (below threshold)
- `donor_signal_strength > 0.1` with `donor_is_local_peak = True`
- `type_signal_difference` strongly favors one type
- High `acceptor_peak_height_ratio` in "neither" predictions

### Type Discrimination

**Strong Donor Indicators:**
- `type_signal_difference > 0.1` (donor signal stronger)
- `donor_acceptor_peak_ratio > 2.0` (donor peak much higher)
- `relative_donor_probability > 0.8`

**Strong Acceptor Indicators:**
- `type_signal_difference < -0.1` (acceptor signal stronger)
- `donor_acceptor_peak_ratio < 0.5` (acceptor peak much higher)
- `relative_donor_probability < 0.2`

---

## Use Cases and Applications

### 1. Meta-Model Training

These features serve as input to machine learning models (XGBoost, Random Forest) that learn to:
- Distinguish true positives from false positives
- Identify false negatives that should be rescued
- Improve splice site type classification

### 2. Quality Control

Features can be used for:
- **Confidence scoring**: Higher `probability_entropy` → lower confidence
- **Peak quality assessment**: `peak_height_ratio` and `second_derivative`
- **Context consistency**: `context_asymmetry` for unusual patterns

### 3. Biological Discovery

- **Weak splice sites**: Low `signal_strength` but high `is_local_peak`
- **Alternative splicing**: Multiple peaks with similar `peak_height_ratio`
- **Regulatory elements**: Unusual `context_asymmetry` patterns

### 4. Model Interpretation

- **Feature importance**: Which signal processing features matter most?
- **Error analysis**: What patterns distinguish errors from correct predictions?
- **Threshold optimization**: How do features vary with prediction thresholds?

---

## Mathematical Formulations

### Epsilon Handling

All ratio calculations include a small epsilon (ε = 1e-10) to prevent division by zero:

```python
# Safe ratio calculation
ratio = numerator / (denominator + epsilon)

# Safe log calculation  
log_value = (value + epsilon).log()
```

### Context Position Mapping

For a splice site at position `pos`:
- `m2`: position `pos - 2`
- `m1`: position `pos - 1`  
- `pos`: target position
- `p1`: position `pos + 1`
- `p2`: position `pos + 2`

### Peak Detection Logic

A position is considered a local peak if:
1. `score > score_m1` (higher than upstream)
2. `score > score_p1` (higher than downstream)  
3. `score > threshold` (above minimum threshold, typically 0.001)

---

## Notes on Feature Engineering

### Design Principles

1. **Biological relevance**: Features should reflect known splice site properties
2. **Signal processing**: Apply techniques for peak detection and noise reduction
3. **Numerical stability**: Handle edge cases and prevent mathematical errors
4. **Interpretability**: Features should have clear biological or mathematical meaning

### Computational Considerations

- **Memory efficiency**: Features computed using Polars expressions
- **Batch processing**: Applied to entire DataFrames at once
- **Missing value handling**: Proper handling of edge positions and missing context

### Validation

Features are validated through:
- **SHAP analysis**: Understanding feature importance in trained models
- **Correlation analysis**: Ensuring features capture different aspects
- **Biological validation**: Checking against known splice site properties 