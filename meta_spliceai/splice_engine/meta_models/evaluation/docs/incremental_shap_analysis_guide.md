# Incremental SHAP Analysis: Scalable Feature Importance for Large Genomic Datasets

**Module:** `shap_incremental.py`  
**Purpose:** Memory-efficient SHAP feature importance analysis that scales to large genomic datasets  
**Innovation:** Incremental processing to solve Out-of-Memory (OOM) problems in traditional SHAP analysis  
**Created:** January 2025  

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Incremental SHAP Innovation](#incremental-shap-innovation)
3. [Technical Architecture](#technical-architecture)
4. [Memory Efficiency Analysis](#memory-efficiency-analysis)
5. [Usage Guide](#usage-guide)
6. [Performance Optimization](#performance-optimization)
7. [Integration Examples](#integration-examples)
8. [Troubleshooting](#troubleshooting)

---

## Problem Statement

### Traditional SHAP Limitations

Traditional SHAP analysis faces significant scalability challenges when applied to large genomic datasets:

**Memory Requirements:**
- **Full Dataset Loading:** Traditional SHAP requires loading the entire feature matrix into memory
- **SHAP Value Storage:** For a dataset with N samples and F features, SHAP generates an N×F matrix of explanations
- **Memory Complexity:** O(N × F × sizeof(float)) which can easily exceed available RAM

**Real-World Impact:**
```python
# Example: Typical genomic dataset
N_samples = 1,000,000    # 1M splice sites
N_features = 50,000      # 50K features (k-mers, positional, etc.)
memory_required = N_samples * N_features * 4 bytes  # ~200GB for SHAP values alone
```

**OOM Problems Encountered:**
- Datasets larger than ~100K samples with >10K features consistently fail
- GPU memory limitations even more restrictive
- Traditional solutions (dataset subsampling) sacrifice statistical power

---

## Incremental SHAP Innovation

### Core Concept

The incremental SHAP approach fundamentally changes how SHAP values are computed and aggregated:

**Key Innovation: Online Aggregation**
Instead of materializing the full SHAP matrix, we compute feature importance as:
```
importance[feature_i] = mean(|SHAP_values[:, feature_i]|)
```

**Streaming Processing:**
```python
# Traditional approach (memory intensive)
shap_values = explainer.shap_values(X)  # Stores N×F matrix
importance = np.abs(shap_values).mean(axis=0)

# Incremental approach (memory efficient)  
importance = np.zeros(n_features)
for batch in batches(X):
    batch_shap = explainer.shap_values(batch)  # Only batch_size×F
    importance += np.abs(batch_shap).sum(axis=0)
importance /= total_samples  # Online mean calculation
```

### Mathematical Foundation

**Incremental Mean Calculation:**
The mean absolute SHAP importance can be computed incrementally:

```
μ_n = (1/n) × Σ|SHAP_i|  where i = 1 to n

Incremental update:
μ_{n+k} = (n×μ_n + Σ|SHAP_{n+1 to n+k}|) / (n+k)
```

**Statistical Equivalence:**
The incremental approach produces statistically identical results to traditional SHAP analysis, but with dramatically reduced memory requirements.

---

## Technical Architecture

### Core Functions

#### 1. `incremental_shap_importance()`
**Purpose:** Main function for computing global SHAP feature importance

```python
def incremental_shap_importance(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    *,
    background_size: int = 1000,    # Small representative sample
    batch_size: int = 512,          # Configurable memory usage
    class_idx: int | None = 1,      # Multi-class support
    approximate: bool = False,      # Speed vs accuracy trade-off
    dtype: str = "float32",         # Memory optimization
    agg: Literal["mean_abs", "sum_abs"] = "mean_abs",
    random_state: int = 42,
    verbose: bool = True,
) -> pd.Series
```

**Key Parameters:**
- **`background_size`:** Size of background dataset for explainer (affects accuracy vs speed)
- **`batch_size`:** Number of samples processed per iteration (affects memory usage)
- **`class_idx`:** Which output class to analyze for multi-class models
- **`approximate`:** Use SHAP's fast approximate algorithm (2-3x faster)

#### 2. `create_memory_efficient_beeswarm_plot()`
**Purpose:** Generate SHAP beeswarm plots with controlled memory usage

```python
def create_memory_efficient_beeswarm_plot(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    *,
    background_size: int = 100,     # Smaller background for visualization
    sample_size: int = 500,         # Subsample for plotting
    top_n_features: int = 20,       # Focus on top features
    approximate: bool = True,       # Speed optimization for viz
    # ... other parameters
) -> plt.Figure
```

### Memory Management Strategy

#### Three-Tier Memory Optimization

**1. Background Dataset Sampling**
```python
# Use small representative sample for explainer initialization
background = shap_sample(X, background_size=1000, random_state=42)
explainer = shap.TreeExplainer(model, data=background)
```

**2. Batch Processing**
```python
# Process data in manageable chunks
for start in range(0, len(X), batch_size):
    end = min(start + batch_size, len(X))
    batch_X = X[start:end]
    batch_shap = explainer.shap_values(batch_X)
    # Accumulate without storing full matrix
    abs_sum += np.abs(batch_shap).sum(axis=0)
```

**3. Data Type Optimization**
```python
# Use float32 instead of float64 to halve memory usage
X_batch = X[start:end].astype("float32")
```

---

## Memory Efficiency Analysis

### Memory Complexity Comparison

| Approach | Memory Complexity | Example (1M samples, 50K features) |
|----------|-------------------|-------------------------------------|
| Traditional SHAP | O(N × F) | ~200GB |
| Incremental SHAP | O(batch_size × F) | ~100MB (batch_size=512) |
| **Reduction Factor** | **N / batch_size** | **~2000x improvement** |

### Practical Memory Usage

**Configuration Examples:**
```python
# Memory-constrained environment (8GB RAM)
incremental_shap_importance(model, X, batch_size=128, background_size=500)

# Standard workstation (32GB RAM)  
incremental_shap_importance(model, X, batch_size=512, background_size=1000)

# High-memory server (128GB RAM)
incremental_shap_importance(model, X, batch_size=2048, background_size=2000)
```

### Performance Benchmarks

**Dataset Scalability:**
- **100K samples, 10K features:** Traditional SHAP ✓, Incremental SHAP ✓
- **500K samples, 25K features:** Traditional SHAP ❌ (OOM), Incremental SHAP ✓
- **1M+ samples, 50K+ features:** Traditional SHAP ❌ (OOM), Incremental SHAP ✓

---

## Usage Guide

### Basic Usage

#### 1. Standard Feature Importance Analysis
```python
from meta_spliceai.splice_engine.meta_models.evaluation.shap_incremental import incremental_shap_importance

# Load your model and data
model = load_trained_model("model.pkl")
X = load_feature_matrix("features.parquet")

# Compute importance with automatic memory management
importance = incremental_shap_importance(
    model=model,
    X=X,
    batch_size=512,        # Adjust based on available memory
    background_size=1000,  # Representative sample size
    verbose=True
)

# Results as pandas Series sorted by importance
print(f"Top 10 features:\n{importance.head(10)}")
```

#### 2. Full Analysis Pipeline
```python
from meta_spliceai.splice_engine.meta_models.evaluation.shap_incremental import run_incremental_shap_analysis

# Complete SHAP analysis with organized output
shap_analysis_dir = run_incremental_shap_analysis(
    dataset_path="train_pc_1000_3mers/master",
    out_dir="results/my_analysis",
    sample=None,           # Use full dataset
    batch_size=512,
    background_size=1000,
    top_n=30
)

print(f"SHAP analysis saved to: {shap_analysis_dir}")
```

### Advanced Configuration

#### Memory-Constrained Environments
```python
# For systems with limited memory
importance = incremental_shap_importance(
    model=model,
    X=X,
    batch_size=128,        # Smaller batches
    background_size=500,   # Smaller background
    dtype="float32",       # Reduced precision
    approximate=True,      # Faster computation
    verbose=True
)
```

#### High-Accuracy Analysis
```python
# For maximum accuracy (requires more memory/time)
importance = incremental_shap_importance(
    model=model,
    X=X,
    batch_size=2048,       # Larger batches (if memory allows)
    background_size=2000,  # Larger background sample
    dtype="float64",       # Full precision
    approximate=False,     # Exact computation
    verbose=True
)
```

#### Ensemble Model Analysis
```python
# For MetaSpliceAI's ensemble models
from meta_spliceai.splice_engine.meta_models.evaluation.shap_incremental import create_ensemble_beeswarm_plots

# Analyze each binary classifier in the ensemble
plot_paths = create_ensemble_beeswarm_plots(
    model=ensemble_model,  # CalibratedSigmoidEnsemble
    X=X,
    sample_size=1000,      # Subsample for visualization
    save_dir="plots/shap_analysis",
    plot_format="pdf"
)

print(f"Ensemble plots saved: {plot_paths}")
```

---

## Performance Optimization

### Tuning Parameters for Your System

#### Memory vs Speed Trade-offs

**Batch Size Selection:**
```python
import psutil

# Automatic batch size based on available memory
available_gb = psutil.virtual_memory().available / (1024**3)
n_features = X.shape[1]

if available_gb > 32:
    batch_size = 2048
elif available_gb > 16:
    batch_size = 1024
elif available_gb > 8:
    batch_size = 512
else:
    batch_size = 256

print(f"Using batch_size={batch_size} for {available_gb:.1f}GB available memory")
```

**Background Size Optimization:**
```python
# Balance between accuracy and speed
dataset_size = len(X)

if dataset_size > 500000:
    background_size = 2000    # Large datasets need larger background
elif dataset_size > 100000:
    background_size = 1000    # Standard setting
else:
    background_size = 500     # Small datasets can use smaller background
```

### GPU vs CPU Considerations

**GPU Memory Management:**
- SHAP TreeExplainer primarily uses CPU
- GPU memory is typically more limited than system RAM
- Focus CPU optimization for SHAP analysis

**Hybrid Approach:**
```python
# Model inference on GPU, SHAP computation on CPU
model_cpu = move_model_to_cpu(model)
importance = incremental_shap_importance(model_cpu, X, ...)
```

---

## Integration Examples

### Integration with CV Pipeline

```python
# In run_gene_cv_sigmoid.py
def main():
    # ... existing CV code ...
    
    # Add SHAP analysis after model training
    if args.run_shap_analysis:
        print("[CV] Running incremental SHAP analysis...")
        shap_dir = run_incremental_shap_analysis(
            dataset_path=args.dataset,
            out_dir=out_dir,
            batch_size=args.shap_batch_size or 512,
            background_size=args.shap_background_size or 1000
        )
        print(f"[CV] SHAP analysis completed: {shap_dir}")
```

### Integration with Feature Importance Pipeline

```python
from meta_spliceai.splice_engine.meta_models.evaluation.feature_importance_integration import FeatureImportanceAnalyzer

class FeatureImportanceAnalyzer:
    def __init__(self):
        self.methods = {
            'shap_incremental': self._run_incremental_shap,
            'permutation': self._run_permutation_importance,
            'statistical': self._run_statistical_tests
        }
    
    def _run_incremental_shap(self, model, X, **kwargs):
        return incremental_shap_importance(
            model=model,
            X=X,
            batch_size=kwargs.get('batch_size', 512),
            background_size=kwargs.get('background_size', 1000)
        )
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Memory Still Insufficient
**Symptoms:** OOM errors even with incremental processing
**Solutions:**
```python
# Reduce batch size further
incremental_shap_importance(model, X, batch_size=64)

# Use smaller background
incremental_shap_importance(model, X, background_size=100)

# Enable approximate mode
incremental_shap_importance(model, X, approximate=True)

# Use float32 precision
incremental_shap_importance(model, X, dtype="float32")
```

#### 2. Slow Performance
**Symptoms:** Analysis takes too long to complete
**Solutions:**
```python
# Enable approximate mode (2-3x faster)
incremental_shap_importance(model, X, approximate=True)

# Increase batch size (if memory allows)
incremental_shap_importance(model, X, batch_size=1024)

# Reduce background size
incremental_shap_importance(model, X, background_size=500)
```

#### 3. Ensemble Model Compatibility
**Symptoms:** "Model type not yet supported by TreeExplainer"
**Solutions:**
- The module automatically detects and handles MetaSpliceAI ensemble models
- Uses individual binary classifiers from the ensemble
- Provides per-class SHAP analysis

#### 4. SHAP Import Errors
**Symptoms:** Keras/TensorFlow compatibility issues
**Solutions:**
- The module includes comprehensive compatibility fixes
- Automatically patches problematic imports
- Creates necessary mock modules for Keras 3.x compatibility

### Performance Monitoring

```python
import time
import psutil

def monitor_shap_analysis():
    start_time = time.time()
    start_memory = psutil.virtual_memory().used
    
    # Run analysis
    importance = incremental_shap_importance(model, X, verbose=True)
    
    end_time = time.time()
    end_memory = psutil.virtual_memory().used
    
    print(f"Analysis completed in {end_time - start_time:.1f} seconds")
    print(f"Memory usage: {(end_memory - start_memory) / 1024**3:.2f} GB")
    
    return importance
```

---

## Best Practices

### 1. Parameter Selection Guidelines

**For Production Analysis:**
```python
# Balanced performance and accuracy
incremental_shap_importance(
    model=model,
    X=X,
    batch_size=512,
    background_size=1000,
    approximate=False,
    dtype="float32"
)
```

**For Exploratory Analysis:**
```python
# Fast iteration for development
incremental_shap_importance(
    model=model,
    X=X,
    batch_size=256,
    background_size=500,
    approximate=True,
    dtype="float32"
)
```

### 2. Integration Workflow

1. **Start with exploratory analysis** using approximate mode
2. **Validate results** with a subset using exact computation  
3. **Scale to full dataset** with optimized parameters
4. **Generate visualizations** using memory-efficient plotting
5. **Document results** with parameter settings for reproducibility

### 3. Quality Assurance

**Validate Incremental Results:**
```python
# Compare incremental vs traditional on small subset
subset_size = 10000
X_subset = X.sample(subset_size)

# Traditional approach (for validation only)
traditional_importance = traditional_shap_analysis(model, X_subset)

# Incremental approach
incremental_importance = incremental_shap_importance(model, X_subset)

# Verify correlation
correlation = np.corrcoef(traditional_importance, incremental_importance)[0,1]
print(f"Incremental vs Traditional correlation: {correlation:.4f}")
```

---

This incremental SHAP approach represents a significant advancement in making feature importance analysis scalable for large genomic datasets, enabling comprehensive analysis that was previously computationally infeasible while maintaining statistical rigor and accuracy.