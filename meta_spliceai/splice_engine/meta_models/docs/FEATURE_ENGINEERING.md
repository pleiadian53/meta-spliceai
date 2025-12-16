# Context-Aware Splice Site Feature Engineering

This document describes the advanced feature engineering approach used in the splice site meta-modeling system, with a focus on context-aware features derived from SpliceAI probability scores.

## Feature Generation Workflow

Our feature engineering follows a comprehensive four-phase approach:

1. **Raw Probability Scores**: All three probability scores (donor, acceptor, neither) preserved for every position
2. **Basic Probability Features**: Derived directly from these three probability scores
3. **Context-Based Features**: Generated consistently for both donor and acceptor sites using window-based context analysis
4. **Cross-Type Features**: Compare patterns between donor and acceptor sites, building on both previous phases

## Enhanced Processing Pipeline

The updated workflow, implemented in `enhanced_process_predictions_with_all_scores`, ensures that feature generation is systematic and consistent across both donor and acceptor sites, eliminating null values that previously occurred when certain features were only calculated for one splice site type.

## Phase 1: Raw Probability Scores

The foundation of our feature engineering process starts with the preservation of all three probability scores for every position:

| Feature | Description | Source |
|---------|-------------|--------|
| `donor_score` | Probability of position being a donor site | SpliceAI |
| `acceptor_score` | Probability of position being an acceptor site | SpliceAI |
| `neither_score` | Probability of position not being a splice site | SpliceAI |

These raw scores are now consistently preserved for all positions, regardless of whether they're being evaluated as potential donor or acceptor sites, ensuring no information is lost in subsequent analysis.

## Phase 2: Basic Probability Features

These fundamental features are derived from the three probability scores and don't require context information:

| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `relative_donor_probability` | Normalized donor probability relative to acceptor | Values near 1 strongly favor donor over acceptor |
| `splice_probability` | Combined probability of being any splice site | Higher values indicate stronger splice signals overall |
| `donor_acceptor_diff` | Normalized difference between donor and acceptor | Measures relative strength of donor vs acceptor signals |
| `splice_neither_diff` | Normalized difference between splice and non-splice | Measures how distinctly a position is recognized as a splice site |
| `donor_acceptor_logodds` | Log-odds ratio of donor vs acceptor | Better captures extreme probability differences |
| `splice_neither_logodds` | Log-odds ratio of splice vs non-splice | Better captures extreme probability differences |
| `probability_entropy` | Entropy measure of the probability distribution | Lower values indicate more confident predictions |

## Phase 3: Context Scores and Derived Features

### Context Score Extraction

Context scores represent the probability values at positions surrounding a candidate splice site:

1. For each position (TP, FP, FN, TN), we extract a window of scores (±2 positions by default)
2. At transcript boundaries, symmetric zero-padding is applied to maintain consistent feature dimensions
3. These raw context scores are available as features with names like:
   - `context_donor_score_m2` (2 positions upstream)
   - `context_donor_score_m1` (1 position upstream)
   - `donor_score` (center position)
   - `context_donor_score_p1` (1 position downstream)
   - `context_donor_score_p2` (2 positions downstream)

With the enhanced workflow in `enhanced_process_predictions_with_all_scores`, we now systematically extract these context scores for both donor and acceptor sites, ensuring complete feature consistency. Context features are always generated for both donor and acceptor sites by default, eliminating null values that previously occurred when certain features were only calculated for one splice site type.

### Context-Agnostic Features

These features are calculated for all positions regardless of splice type (donor or acceptor) and provide consistent context analysis across all sites:

| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `context_neighbor_mean` | Average of all surrounding positions | Baseline background probability level |
| `context_asymmetry` | Difference between upstream and downstream | Captures directionality of probability pattern |
| `context_max` | Maximum score in surrounding positions | Highest competing score in neighborhood |

### Basic Differential Features

These features capture the "peak" characteristics of probability distributions around splice sites:

| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `donor_diff_m1` | Difference between center and -1 position | Higher values indicate sharper rise |
| `donor_diff_m2` | Difference between center and -2 position | Captures wider upstream pattern |
| `donor_diff_p1` | Difference between center and +1 position | Higher values indicate sharper fall |
| `donor_diff_p2` | Difference between center and +2 position | Captures wider downstream pattern |
| `donor_surge_ratio` | Ratio of center to immediate neighbors | Measures how much the center "stands out" |
| `donor_is_local_peak` | Binary indicator of local maximum | 1 if center is higher than both neighbors |
| `donor_weighted_context` | Weighted average with higher weight for center | Smoothed score that values central position |

### Statistical Features

These features capture statistical properties of the context window:

| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `donor_peak_height_ratio` | Ratio of center to neighbor average | How many times higher than background |
| `donor_signal_strength` | Difference between center and neighbor average | Absolute elevation above background |
| `donor_context_diff_ratio` | Ratio of center to highest neighbor | Distinguishes from nearby high scores |

### Signal Processing Inspired Features

These features are inspired by signal processing concepts:

| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `donor_second_derivative` | Approximation of curvature | Higher values indicate sharper peaks |
| `donor_weighted_context` | Weighted average with higher weight for center | Smoothed score that values central position |

## Phase 4: Cross-Type Features

These features compare patterns between donor and acceptor sites and are computed after both donor and acceptor context features are available:

| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `donor_acceptor_peak_ratio` | Ratio of donor peak height to acceptor peak height | Higher values suggest stronger donor character |
| `type_signal_difference` | Difference between donor and acceptor signal strength | Positive values favor donor, negative favor acceptor |
| `score_difference_ratio` | Normalized difference between donor and acceptor scores | Values close to +1 favor donor, close to -1 favor acceptor |
| `signal_strength_ratio` | Ratio of donor signal strength to acceptor signal strength | Higher values indicate stronger donor signal relative to acceptor |

With the enhanced workflow, these cross-type features are now consistently calculated for all positions, as all the prerequisite features are guaranteed to be available.

## Use Cases and Benefits

Context-based features provide several advantages for meta-modeling:

1. **Rescue False Negatives**: Identify true splice sites with scores below threshold but characteristic context patterns
2. **Filter False Positives**: Remove high-scoring positions that lack the typical context pattern of true sites
3. **Improve Classification**: Provide richer feature space for machine learning models
4. **Capture Biological Context**: Better represent the spatial nature of splice site recognition
5. **Feature Consistency**: With the enhanced workflow, features are consistently calculated for all positions, eliminating null values in downstream analyses

## Implementation

Context-based features are automatically detected and generated in the enhanced workflow. The implementation supports both Pandas and Polars dataframes through our cross-framework utilities.

### Enhanced Processing Pipeline

The primary entry point for feature generation is now `enhanced_process_predictions_with_all_scores`, which uses the comprehensive `enhanced_evaluate_splice_site_errors` function:

```python
# Main entry point for feature generation
error_df, positions_df = enhanced_process_predictions_with_all_scores(
    predictions=predictions,
    ss_annotations_df=annotations_df,
    threshold=0.5,
    add_derived_features=True,  # Generate derived features from probability scores
    fill_missing_values=False,  # Option to fill missing values if needed
    fill_value=0.0,           # Value to use when filling missing values
    verbose=1
)
```

### Handling Missing Values

While the enhanced workflow is designed to generate complete feature sets without missing values, there might still be edge cases where some values are missing. The workflow includes explicit support for handling these cases:

```python
# Fill missing values in derived features
if fill_missing_values:
    # Identify all derived feature columns (excluding basic scores)
    # Context-agnostic, donor-specific, acceptor-specific, and cross-type features
    # Fill identified columns with the specified fill_value (default: 0.0)
```

This functionality ensures that downstream machine learning models receive complete feature vectors, even in rare edge cases.

### Context Score Extraction

```python
# Example code showing context score extraction
def get_context_scores(probabilities: np.ndarray, position: int, window_size: int = 2) -> List[float]:
    """
    Extract context scores for a position with proper handling of boundaries.
    
    Args:
        probabilities: Array of probability scores
        position: Center position to extract context around
        window_size: Number of positions to include on each side
        
    Returns:
        List of length 2*window_size+1 containing the context window with zero-padding as needed
    """
    # Implementation details...
```

## Visual Examples of Feature Patterns

Below are text-based visualizations of typical probability patterns around splice sites and how our derived features help identify them.

### Pattern 1: Strong True Positive (TP) Donor Site

```
Positions:      -2    -1     0     +1    +2
                 |     |     |     |     |
Probabilities:  0.10  0.25  0.85  0.20  0.05
                 |     |     ^     |     |
                 |     |   CENTER   |     |
```

**Key Features:**
- `donor_score` = 0.85 (high center score)
- `donor_diff_m1` = 0.60 (strong rise from -1)
- `donor_diff_p1` = 0.65 (strong fall to +1)
- `donor_is_local_peak` = 1 (definitely a peak)
- `donor_surge_ratio` = 1.89 (stands out from neighbors)
- `donor_peak_height_ratio` = 5.67 (much higher than background)

### Pattern 2: Weak True Positive (Borderline/FN case)

```
Positions:      -2    -1     0     +1    +2
                 |     |     |     |     |
Probabilities:  0.05  0.15  0.45  0.10  0.02
                 |     |     ^     |     |
                 |     |   CENTER   |     |
```

**Key Features:**
- `donor_score` = 0.45 (might fall below threshold)
- `donor_diff_m1` = 0.30 (still significant rise)
- `donor_is_local_peak` = 1 (still a peak)
- `donor_peak_height_ratio` = 5.62 (similar to strong TP!)
- `donor_second_derivative` = 0.35 (sharp curvature)

This case shows why context features are valuable - the absolute score is below threshold, but the *pattern* matches a true site.

### Pattern 3: False Positive (High score but wrong pattern)

```
Positions:      -2    -1     0     +1    +2
                 |     |     |     |     |
Probabilities:  0.40  0.55  0.60  0.65  0.50
                 |     |     ^     |     |
                 |     |   CENTER   |     |
```

**Key Features:**
- `donor_score` = 0.60 (above threshold)
- `donor_diff_m1` = 0.05 (minimal rise)
- `donor_diff_p1` = -0.05 (negative fall - continues rising after site)
- `donor_is_local_peak` = 0 (not a peak!)
- `donor_signal_strength` = 0.0875 (minimal elevation above background)
- `donor_peak_height_ratio` = 1.14 (barely above background)
- `donor_context_diff_ratio` = 0.92 (center is actually lower than max neighbor)
- `donor_second_derivative` = -0.10 (wrong curvature)

This site would be incorrectly classified by score alone, but context features reveal it's unlikely to be a true site.

### Pattern 4: Donor vs Acceptor Disambiguation

```
DONOR PATTERN
Positions:      -2    -1     0     +1    +2
                 |     |     |     |     |
Probabilities:  0.10  0.30  0.70  0.15  0.05
                 |     |     ^     |     |
                 
ACCEPTOR PATTERN (same position)
Positions:      -2    -1     0     +1    +2
                 |     |     |     |     |
Probabilities:  0.05  0.15  0.60  0.25  0.10
                 |     |     ^     |     |
```

**Cross-Type Features:**
- `donor_acceptor_peak_ratio` = 1.17 (donor pattern slightly stronger)
- `type_signal_difference` = 0.10 (favors donor interpretation)
- `score_difference_ratio` = 0.08 (normalized difference between donor/acceptor scores)
- `signal_strength_ratio` = 1.67 (donor signal strength relative to acceptor signal)

**Individual Pattern Features:**
- `donor_signal_strength` = 0.28 (donor's elevation above background)
- `acceptor_signal_strength` = 0.17 (acceptor's elevation above background)
- `donor_diff_m1` = 0.40 (sharp rise before donor site)
- `donor_diff_p1` = 0.55 (sharp fall after donor site)
- `acceptor_diff_m1` = 0.45 (sharp rise before acceptor site)
- `acceptor_diff_p1` = 0.35 (moderate fall after acceptor site)

This example shows how cross-type features help disambiguate positions with high scores for both donor and acceptor.

### Pattern 5: Edge Case - Plateau Pattern

```
Positions:      -2    -1     0     +1    +2
                 |     |     |     |     |
Probabilities:  0.20  0.60  0.60  0.55  0.15
                 |     |     ^     |     |
                 |     |___CENTER___|     |
```

**Key Features:**
- `donor_score` = 0.60 (high enough to pass threshold)
- `donor_is_local_peak` = 0 (not actually a peak)
- `donor_diff_m1` = 0.00 (no rise from -1)
- `donor_context_max` = 0.60 (tie with neighbor)
- `donor_signal_strength` = 0.23 (modest elevation above average)

This pattern shows a case where the position is part of a plateau rather than a distinctive peak, suggesting possible cryptic or complex sites.

### Pattern 6: Strong Context Asymmetry (Indicates Direction)

```
Positions:      -2    -1     0     +1    +2
                 |     |     |     |     |
Probabilities:  0.05  0.40  0.75  0.55  0.25
                 |     |     ^     |     |
                 |     |   CENTER   |     |
```

**Key Features:**
- `donor_context_asymmetry` = -0.35 (strongly negative - downstream bias)
- `donor_weighted_context` = 0.525 (downweighted due to imbalance)
- `donor_second_derivative` = -0.25 (indicates gradual decline after peak)

The asymmetry in this pattern may indicate a complex splicing mechanism or alternative site usage.


## Future Directions

Potential enhancements to context-based feature engineering:

1. **Adaptive Window Sizes**: Dynamically adjust window size based on transcript structure
2. **Wavelet-Based Features**: Apply wavelet transforms to capture multi-scale patterns
3. **Cross-Sample Normalization**: Normalize features across different genes/transcripts
4. **Sequence-Probability Integration**: Combine context probability patterns with sequence motifs

## References

- SpliceAI: Algorithm paper describing the original probability scores
- Multiple sources on signal processing for peak detection
- Literature on splice site recognition patterns
