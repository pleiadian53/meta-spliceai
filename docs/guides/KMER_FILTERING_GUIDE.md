# K-mer Filtering Guide for Splice Site Prediction

This guide describes the domain-specific k-mer filtering strategies implemented for splice site prediction, which provide more appropriate alternatives to conventional variance/correlation filtering for genomic sequence data.

## 🎯 Overview

The k-mer filtering system addresses the challenge of handling large feature sets (e.g., 4096+ features for 6-mers) when sample sizes are limited. Unlike conventional methods that may remove biologically important rare motifs, our domain-specific approaches preserve features that are critical for splice site recognition.

## 🧬 Domain-Specific Filtering Strategies

### 1. Biological Motif-Based Filtering (`motif`)

**Purpose**: Preserves k-mers containing known splice site consensus motifs.

**How it works**:
- Identifies k-mers containing donor site motifs (GT, GC, AT)
- Identifies k-mers containing acceptor site motifs (AG, CG)
- Position-aware filtering for 6-mers (critical positions 1-2 for donors, 4-5 for acceptors)

**Best for**: Preserving biologically relevant features, even if rare.

```python
from meta_spliceai.splice_engine.meta_models.features import KmerFilterConfig

config = KmerFilterConfig(
    enabled=True,
    strategy='motif',
    splice_type='both'  # 'donor', 'acceptor', or 'both'
)
```

### 2. Mutual Information Filtering (`mi`)

**Purpose**: Captures non-linear relationships between k-mers and splice site labels.

**Advantages over correlation**:
- Handles sparse features well (many k-mers are rare)
- Captures non-linear relationships
- Preserves informative rare motifs

**Best for**: Feature selection based on information content.

```python
config = KmerFilterConfig(
    enabled=True,
    strategy='mi',
    threshold=0.01  # Minimum mutual information score
)
```

### 3. Sparsity-Aware Filtering (`sparsity`)

**Purpose**: Filters based on occurrence patterns rather than variance.

**Logic**:
- Removes overly rare features (not informative when present)
- Removes overly common features (not discriminative)
- Preserves features with appropriate sparsity

**Best for**: Handling the sparse nature of k-mer features.

```python
config = KmerFilterConfig(
    enabled=True,
    strategy='sparsity',
    min_occurrence_rate=0.001,  # Minimum 0.1% occurrence
    max_occurrence_rate=0.95    # Maximum 95% occurrence
)
```

### 4. Ensemble Filtering (`ensemble`)

**Purpose**: Combines multiple strategies for robust feature selection.

**Approach**:
- Applies motif-based, MI-based, and sparsity-based filtering
- Uses union (conservative) or intersection (aggressive) combination
- Provides most robust feature selection

**Best for**: Production use with balanced feature reduction.

```python
config = KmerFilterConfig(
    enabled=True,
    strategy='ensemble',
    combination_method='union'  # 'union' or 'intersection'
)
```

## 🎛️ Preset Configurations

### Conservative (`conservative`)
- **Strategy**: Motif-based only
- **Use case**: Preserve all biologically relevant features
- **Expected reduction**: 10-30%

### Balanced (`balanced`)
- **Strategy**: Ensemble with moderate thresholds
- **Use case**: Good balance between reduction and preservation
- **Expected reduction**: 40-60%

### Aggressive (`aggressive`)
- **Strategy**: Ensemble with strict thresholds
- **Use case**: Maximum feature reduction
- **Expected reduction**: 70-90%

### Motif-Only (`motif_only`)
- **Strategy**: Motif-based filtering
- **Use case**: Preserve only known splice site motifs
- **Expected reduction**: 20-40%

### MI-Only (`mi_only`)
- **Strategy**: Mutual information filtering
- **Use case**: Information-theoretic feature selection
- **Expected reduction**: 50-80%

## 🔧 Integration with CV Scripts

### Basic Integration

```python
from meta_spliceai.splice_engine.meta_models.features import (
    add_kmer_filtering_args,
    KmerFilterConfig,
    integrate_kmer_filtering_in_cv
)

# Add arguments to parser
add_kmer_filtering_args(parser)

# Create configuration from arguments
config = KmerFilterConfig.from_args(args)

# Apply filtering during CV
X_filtered, filtered_features = integrate_kmer_filtering_in_cv(
    X, y, feature_names, config
)
```

### Command-Line Usage

```bash
# Enable k-mer filtering with ensemble strategy
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000/master \
    --out-dir models/gene_cv_with_filtering \
    --filter-kmers \
    --kmer-filter-strategy ensemble \
    --n-folds 5

# Use preset configuration
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000/master \
    --out-dir models/gene_cv_conservative \
    --filter-kmers \
    --kmer-filter-strategy motif \
    --kmer-splice-type both

# Custom MI threshold
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000/master \
    --out-dir models/gene_cv_mi \
    --filter-kmers \
    --kmer-filter-strategy mi \
    --kmer-mi-threshold 0.02
```

## 📊 Usage Examples

### Example 1: Conservative Filtering for Small Datasets

```python
from meta_spliceai.splice_engine.meta_models.features import get_preset_config

# Use conservative preset for small datasets
config = get_preset_config('conservative')

# Apply to dataset
X_filtered, filtered_features = integrate_kmer_filtering_in_cv(
    X, y, feature_names, config
)

print(f"Features reduced from {len(feature_names)} to {len(filtered_features)}")
```

### Example 2: Aggressive Filtering for Large Datasets

```python
# Use aggressive preset for large datasets
config = get_preset_config('aggressive')

# Apply filtering
X_filtered, filtered_features = integrate_kmer_filtering_in_cv(
    X, y, feature_names, config
)

print(f"Features reduced from {len(feature_names)} to {len(filtered_features)}")
```

### Example 3: Custom Configuration

```python
# Create custom configuration
config = KmerFilterConfig(
    enabled=True,
    strategy='ensemble',
    threshold=0.005,  # Lower MI threshold
    min_occurrence_rate=0.0005,  # Allow rarer features
    max_occurrence_rate=0.98  # Allow more common features
)

# Apply filtering
X_filtered, filtered_features = integrate_kmer_filtering_in_cv(
    X, y, feature_names, config
)
```

## 🔍 Monitoring and Diagnostics

### Filtering Statistics

```python
from meta_spliceai.splice_engine.meta_models.features import print_filtering_stats

# Print detailed statistics
print_filtering_stats(X, feature_names, config)
```

Output:
```
📊 K-mer Filtering Statistics:
   Total features: 4100
   K-mer features: 4096 (99.9%)
   Non-k-mer features: 4
   K-mer size distribution:
     6mer: 4096 features
   Filtering strategy: ensemble
   Strategy parameters: {'threshold': 0.01, 'min_occurrence_rate': 0.001, 'max_occurrence_rate': 0.95}
```

### Configuration Validation

```python
from meta_spliceai.splice_engine.meta_models.features import validate_filtering_config

# Validate configuration before use
if not validate_filtering_config(config):
    print("Invalid configuration detected")
```

## 🎯 Best Practices

### 1. Strategy Selection

| Scenario | Recommended Strategy | Reasoning |
|----------|---------------------|-----------|
| Small dataset (<1000 samples) | `conservative` or `motif` | Preserve biological relevance |
| Medium dataset (1000-10000 samples) | `balanced` or `ensemble` | Good balance of reduction/preservation |
| Large dataset (>10000 samples) | `aggressive` or `mi` | Maximum feature reduction |
| Research/exploration | `motif_only` | Focus on known biology |
| Production/deployment | `ensemble` | Robust, comprehensive approach |

### 2. Parameter Tuning

**MI Threshold**:
- `0.001-0.005`: Very conservative (preserves most features)
- `0.01-0.02`: Balanced (moderate reduction)
- `0.05+`: Aggressive (significant reduction)

**Sparsity Range**:
- `[0.0001, 0.99]`: Very permissive
- `[0.001, 0.95]`: Balanced (default)
- `[0.01, 0.9]`: Restrictive

### 3. Validation

Always validate filtering results:
- Check that biologically important motifs are preserved
- Verify that feature reduction is reasonable for your dataset size
- Monitor model performance with and without filtering

## 🚀 Advanced Usage

### Custom Filtering Strategies

```python
from meta_spliceai.splice_engine.meta_models.features import KmerFilteringStrategy

class CustomFilter(KmerFilteringStrategy):
    def __init__(self, custom_threshold: float = 0.5):
        super().__init__("custom", "Custom filtering strategy")
        self.custom_threshold = custom_threshold
    
    def filter(self, X: np.ndarray, y: np.ndarray, kmer_features: List[str], **kwargs) -> List[str]:
        # Implement custom filtering logic
        selected_features = []
        for i, feature in enumerate(kmer_features):
            if not _is_kmer(feature):
                continue
            
            # Custom filtering logic here
            if self._custom_criterion(X[:, i], y):
                selected_features.append(feature)
        
        return selected_features
    
    def _custom_criterion(self, feature_values: np.ndarray, y: np.ndarray) -> bool:
        # Implement your custom criterion
        return np.mean(feature_values) > self.custom_threshold
```

### Integration with Existing Pipelines

```python
# In your CV script
def apply_kmer_filtering_in_fold(X_train, y_train, feature_names, config):
    """Apply k-mer filtering within a CV fold."""
    if not config.enabled:
        return X_train, feature_names
    
    # Apply filtering to training data
    X_train_filtered, filtered_features = integrate_kmer_filtering_in_cv(
        X_train, y_train, feature_names, config
    )
    
    # Get feature indices for consistent application to test data
    feature_indices = [feature_names.index(f) for f in filtered_features]
    
    return X_train_filtered, filtered_features, feature_indices
```

## 📈 Performance Expectations

### Typical Feature Reduction

| Strategy | Small Dataset | Medium Dataset | Large Dataset |
|----------|---------------|----------------|---------------|
| `motif` | 10-30% | 20-40% | 30-50% |
| `mi` | 30-60% | 50-80% | 70-90% |
| `sparsity` | 20-50% | 40-70% | 60-85% |
| `ensemble` | 40-70% | 60-85% | 80-95% |

### Memory and Speed Benefits

- **Memory reduction**: 50-90% depending on strategy
- **Training speed**: 2-10x faster with aggressive filtering
- **Inference speed**: 2-5x faster with reduced features

## 🔧 Troubleshooting

### Common Issues

1. **No features selected**: Lower thresholds or use more permissive settings
2. **Too many features removed**: Use conservative strategy or adjust parameters
3. **Biological motifs lost**: Use motif-based filtering or lower MI threshold
4. **Memory still high**: Use aggressive filtering or reduce k-mer size

### Debugging

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check filtering statistics
stats = get_kmer_filtering_stats(X, feature_names)
print(f"K-mer ratio: {stats['kmer_ratio']:.1%}")

# Validate configuration
if not validate_filtering_config(config):
    print("Configuration validation failed")
```

## 📚 References

- **Splice site motifs**: GT (donor), AG (acceptor) consensus sequences
- **K-mer sparsity**: Most k-mers are rare in genomic sequences
- **Mutual information**: Better than correlation for sparse, non-linear features
- **Ensemble methods**: Combine multiple strategies for robust selection

This filtering system provides domain-specific approaches that are more appropriate than conventional methods for genomic sequence data and splice site prediction tasks. 