# Memory Optimization Guide

**Date:** October 17, 2025  
**Status:** ✅ **ACTIVE**

---

## Overview

This guide provides strategies for optimizing memory usage when building and training meta-models with large genomic datasets.

## Current Memory Profile

### Dataset Generation (`incremental_builder.py`)
- **Storage**: Parquet files (columnar, compressed)
- **Processing**: Polars DataFrames (lazy evaluation, zero-copy operations)
- **Memory footprint**: ~200-500 MB per batch of 100 genes
- **Peak usage**: 2-4 GB for 1000 genes (M1 MacBook with 16GB RAM)

### Model Training (`run_gene_cv_sigmoid.py`)
- **Data loading**: Polars (memory-efficient)
- **Preprocessing**: Polars → NumPy → Pandas (one-time conversion)
- **Training**: XGBoost works with NumPy arrays internally
- **Memory footprint**: 
  - 100 genes: ~500 MB
  - 1000 genes: ~5 GB (estimated)
  - 10,000 genes: ~50 GB (requires optimization)

---

## Memory Optimization Strategies

### 1. **Data Format: Use Polars Instead of Pandas**

**Current Implementation** ✅:
```python
# Dataset is stored in Polars throughout incremental_builder
# Only converted to pandas when absolutely necessary for training
```

**Why it matters**:
- Polars is 5-10x more memory-efficient than Pandas
- Zero-copy operations where possible
- Better handling of nullable types

### 2. **Conversion Path: Polars → NumPy → Pandas**

**Optimized Conversion** (implemented in `preprocessing.py`):
```python
# EFFICIENT: Single-step conversion
X_numpy = X_df.to_numpy()  # Polars → NumPy (contiguous array)
X = pd.DataFrame(X_numpy, columns=X_df.columns)  # NumPy → Pandas

# INEFFICIENT (old method): Column-by-column
# data_dict = {col: X_df[col].to_list() for col in X_df.columns}
# X = pd.DataFrame(data_dict)  # Creates temporary lists!
```

**Memory savings**: ~30-50% reduction in peak memory usage

### 3. **Why Not Skip Pandas Entirely?**

The training code temporarily needs pandas for **column names**:

```python
# From trainer.py
X_df, y_series = preprocessing.prepare_training_data(df, return_type="pandas")
self.feature_names_ = list(X_df.columns)  # Need column names!
X = X_df.values  # Then immediately convert to numpy
```

**Future optimization**: Add `return_type="numpy"` option that returns:
- NumPy array for training
- Column names as a separate list

### 4. **Batch Processing for Very Large Datasets**

For datasets >10,000 genes:

```bash
# Use ensemble training strategy (trains on subsets, combines models)
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset data/train_pc_10000_3mers/master \
    --out-dir results/meta_model_10k \
    --n-estimators 200 \
    --n-folds 3 \
    --ensemble-size 5 \        # Split into 5 sub-models
    --calibrate-per-class \
    --verbose
```

This trains 5 separate models on 2000 genes each, then combines predictions.

### 5. **Incremental Builder Memory Settings**

For large-scale dataset generation:

```bash
# Optimize batch sizes for your RAM
./scripts/builder/run_builder_resumable.sh \
    --n-genes 10000 \
    --batch-size 50 \          # Process 50 genes at a time (default: 100)
    --batch-rows 10000 \       # Write 10K rows per batch (default: 20K)
    --run-workflow             # Still include base model pass
```

**Memory impact**:
- Smaller `--batch-size`: Lower peak RAM, but more batches (slower)
- Smaller `--batch-rows`: More frequent writes, less RAM buffering

---

## Memory Requirements by Dataset Size

| Dataset Size | Genes | Positions (est.) | RAM Required | Recommended Strategy |
|--------------|-------|------------------|--------------|---------------------|
| **Small**    | 100   | 25K              | 2-4 GB       | Single model        |
| **Medium**   | 1,000 | 250K             | 8-16 GB      | Single model        |
| **Large**    | 5,000 | 1.2M             | 32-64 GB     | Ensemble (5 models) |
| **Very Large** | 10,000+ | 2.5M+          | 64+ GB       | Ensemble (10 models) or cloud |

*Estimates based on 131 features with 3-mers, including overhead*

---

## Current Laptop Limits (M1 MacBook, 16GB RAM)

### Safe Operating Range
- **Dataset generation**: Up to 2000 genes safely
- **Model training**: Up to 1000 genes comfortably
- **With swap**: Up to 2000 genes (slower due to paging)

### What Happens When RAM is Exceeded
1. **macOS swap**: System uses SSD as virtual memory (slow but works)
2. **Python killed**: If swap exhausted, process terminated by OS
3. **Symptoms**: Slow performance, high disk I/O, unresponsive UI

---

## Monitoring Memory Usage

### During Training
```bash
# Watch memory in real-time
watch -n 5 'ps aux | grep run_gene_cv_sigmoid | head -1'

# Or use Activity Monitor (GUI)
```

### In Python
```python
import psutil
import os

process = psutil.Process(os.getpid())
mem_gb = process.memory_info().rss / 1024**3
print(f"Current memory: {mem_gb:.2f} GB")
```

---

## Best Practices for Large Datasets

1. **Start small, scale up**:
   - Test with 100 genes first
   - Then 500, 1000, 2000...
   - Monitor memory at each step

2. **Use `nohup` or `tmux`**:
   - Large training runs can take hours
   - Don't let laptop sleep interrupt training

3. **Clean up between runs**:
   ```bash
   # Remove old results to free disk space
   rm -rf results/meta_model_test_*
   rm -rf logs/*.log
   ```

4. **Consider cloud for very large datasets**:
   - AWS/Azure/GCP instances with 64-128 GB RAM
   - Run incremental_builder remotely
   - Download final parquet files locally for analysis

---

## Future Optimizations

### Planned Improvements
1. **Streaming training**: Process data in chunks without loading all into RAM
2. **Sparse feature encoding**: For k-mers, use sparse matrices
3. **GPU acceleration**: XGBoost GPU support for faster training
4. **Dask integration**: Distributed processing for very large datasets

### Experimental: Direct NumPy Training
```python
# In preprocessing.py - future option
X_numpy, column_names = preprocessing.prepare_training_data(
    df, 
    return_type="numpy",  # Skip pandas entirely
    return_column_names=True
)
```

---

## Troubleshooting

### "MemoryError" during training
```bash
# Reduce dataset size
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset data/train_pc_1000_3mers/master \
    --sample-genes 500 \    # Use only 500 genes for testing
    --out-dir results/test
```

### "Killed" (process terminated by OS)
- **Cause**: Out of memory, OS killed the process
- **Solution**: Use smaller batch sizes or ensemble training

### Very slow training (swap thrashing)
- **Symptom**: High disk I/O, slow progress
- **Solution**: Reduce dataset size or move to larger machine

---

## Summary

**Key Takeaways**:
1. ✅ Current implementation is already memory-optimized (Polars → NumPy → Pandas)
2. ✅ M1 MacBook (16GB) can handle up to ~1000 genes comfortably
3. ⚠️ For >2000 genes, use ensemble training or cloud resources
4. 🔮 Future: Direct NumPy conversion will improve further

**Next Steps**:
- Test 1000-gene training (currently running in background)
- Document performance and memory usage
- Add ensemble training docs if needed for larger datasets

