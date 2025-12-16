# Large-Scale Meta-Model Training Guide

**⚠️ DEPRECATED:** This documentation is outdated. Please use **[Multi-Instance Ensemble Training](MULTI_INSTANCE_ENSEMBLE_TRAINING.md)** instead.

**Last Updated:** Pre-January 2025 (OUTDATED)  
**Status:** ❌ **SUPERSEDED**  
**Replaced By:** [Multi-Instance Ensemble Training](MULTI_INSTANCE_ENSEMBLE_TRAINING.md) and [Comprehensive Training Guide](COMPREHENSIVE_TRAINING_GUIDE.md)

This document consolidates **best practices** for training the multiclass splice-site meta-model when the feature matrix no longer fits comfortably in RAM, along with comprehensive analysis capabilities available in both gene-aware and chromosome-aware CV approaches.

> If the default `--row-cap` (100,000 rows) already works for your workstation, you do **not** need the memory optimization features here. Come back when you want to train on **millions** of splice-site examples with comprehensive analysis.

---

## 1. Enhanced CV Scripts Overview

### Modern Implementation Options

| Script | CV Strategy | Memory Optimization | Analysis Features | Best For |
|--------|-------------|---------------------|-------------------|----------|
| **`run_gene_cv_sigmoid.py`** | Gene-wise splitting | Standard | Complete pipeline | In-domain evaluation, <10K genes |
| **`run_loco_cv_multiclass_scalable.py`** | Chromosome-wise splitting | Advanced | Complete pipeline | Out-of-distribution, 20K+ genes |

Both scripts now provide **complete feature parity** with identical analysis capabilities:

### Comprehensive Analysis Pipeline (Both Scripts)
1. **CV Metrics Visualization**: 7 publication-ready plots
2. **Enhanced ROC/PR Curves**: Multiclass and binary analysis
3. **Feature Importance Analysis**: 4 complementary methods
4. **Enhanced SHAP Analysis**: Memory-efficient with comprehensive visualizations
5. **Feature Quality Control**: Leakage detection and automated filtering
6. **Diagnostic Analysis**: 8 comprehensive diagnostic functions
7. **Transcript-Level Analysis**: Optional top-k accuracy evaluation

---

## 2. Execution Modes and Memory Strategies

### 2.1 Gene-Aware CV (`run_gene_cv_sigmoid.py`)

| Mode | Typical dataset size | RAM needed | Speed | Configuration |
|------|----------------------|-----------:|-------|---------------|
| **Standard** | ≤ 1K genes | 8-16 GB | Fast | Default settings |
| **Large Dataset** | 1K-10K genes | 16-32 GB | Medium | `--row-cap 0 --tree-method hist` |
| **GPU Accelerated** | 0.5K-2K genes | GPU VRAM 8-16 GB | Fast | `--tree-method gpu_hist --gpu` |
| **Memory Constrained** | Any size | 4-8 GB | Slow | `--row-cap 50000 --filter-features` |

### 2.2 Chromosome-Aware CV (`run_loco_cv_multiclass_scalable.py`)

| Mode | Typical dataset size | RAM needed | Speed | Configuration |
|------|----------------------|-----------:|-------|---------------|
| **Standard** | 1K-10K genes | 16-32 GB | Medium | Default settings |
| **Memory Optimized** | 10K-50K genes | 8-16 GB | Medium | `--use-chunked-loading --memory-optimize` |
| **Large Scale** | 50K+ genes | 4-8 GB | Slow | `--use-sparse-kmers --chunk-size 5000` |
| **GPU Accelerated** | 1K-5K genes | GPU VRAM 12-24 GB | Fast | `--tree-method gpu_hist --gpu` |

---

## 3. Quick Reference Commands

### 3.1 Gene-Aware CV Examples

```bash
# 1. Standard comprehensive analysis
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000/master \
    --out-dir models/comprehensive_gene_cv \
    --calibrate-per-class --plot-curves --check-leakage \
    --n-folds 5 --verbose

# 2. Memory-optimized for large datasets
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_large/master \
    --out-dir models/large_gene_cv \
    --row-cap 0 --tree-method hist --filter-features \
    --diag-sample 5000 --calibrate-per-class

# 3. GPU-accelerated training
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000/master \
    --out-dir models/gpu_gene_cv \
    --tree-method gpu_hist --gpu --n-estimators 200 \
    --calibrate-per-class --plot-curves
```

### 3.2 Chromosome-Aware CV Examples

```bash
# 1. Standard comprehensive analysis
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_pc_1000/master \
    --out-dir models/comprehensive_loco_cv \
    --calibrate-per-class --plot-curves --check-leakage \
    --transcript-topk --verbose 2

# 2. Memory-optimized for very large datasets
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_full/master \
    --out-dir models/large_scale_loco_cv \
    --use-chunked-loading --chunk-size 10000 \
    --use-sparse-kmers --memory-optimize \
    --feature-selection --max-features 1000 \
    --calibrate-per-class --row-cap 0

# 3. Fixed holdout with specific chromosomes
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_pc_1000/master \
    --out-dir models/holdout_analysis \
    --heldout-chroms "21,22,X" --calibrate-per-class \
    --plot-curves --transcript-topk
```

---

## 4. Key Parameters and Optimizations

### 4.1 Memory Management

#### Row Capping
* **`SS_MAX_ROWS`** (environment variable) – hard upper bound applied *during dataset load*
* **`--row-cap`** (CLI) – convenience wrapper; ignored if `SS_MAX_ROWS` is already set
* Set either to `0` to disable capping

#### Tree Methods
* `auto` – let XGBoost decide (may choose exact method → OOM on wide data)
* `hist` – CPU histogram algorithm, memory-efficient
* `gpu_hist` – GPU histogram; fast if the dataset fits GPU VRAM

#### Histogram Bin Configuration
* `--max-bin 256` – Default sweet spot (1-byte bins, ~99% of exact accuracy)
* Raise to 512/1024 for extremely non-linear features at cost of RAM/VRAM

### 4.2 Feature Management

#### Quality Control
* `--check-leakage` – Enable correlation analysis
* `--leakage-threshold 0.95` – Correlation threshold
* `--auto-exclude-leaky` – Automatically remove problematic features
* `--exclude-features configs/exclude_features.txt` – Custom exclusion list

#### Feature Selection
* `--filter-features` – Basic importance-based filtering
* `--feature-selection` – Advanced feature selection (LOCO-CV only)
* `--max-features N` – Limit total features (LOCO-CV only)

### 4.3 Analysis Configuration

#### Visualization
* `--plot-curves` – Enable ROC/PR curve generation (default: enabled)
* `--plot-format pdf` – Output format (pdf/png/svg)
* `--n-roc-points 101` – Resolution for ROC curves
* `--dpi 300` – Plot resolution

#### Diagnostic Analysis
* `--diag-sample 25000` – Sample size for diagnostics
* `--neigh-sample 1000` – Neighborhood analysis sample size
* `--neigh-window 50` – Neighborhood window size
* `--leakage-probe` – Advanced leakage detection

#### Calibration
* `--calibrate-per-class` – Per-class calibration (recommended)
* `--calib-method platt` – Calibration method (platt/isotonic)
* `--meta-calibrate` – Additional meta-calibration

---

## 5. Advanced Memory Optimization (LOCO-CV Only)

### 5.1 Chunked Data Loading
```bash
# Enable chunked loading for datasets that don't fit in RAM
--use-chunked-loading --chunk-size 10000
```

### 5.2 Sparse Matrix Support
```bash
# Use sparse matrices for high-dimensional k-mer features
--use-sparse-kmers --memory-optimize
```

### 5.3 Combined Memory Optimization
```bash
# Maximum memory efficiency for very large datasets
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --dataset train_full/master \
    --out-dir models/ultra_large_scale \
    --use-chunked-loading --chunk-size 5000 \
    --use-sparse-kmers --memory-optimize \
    --feature-selection --max-features 500 \
    --row-cap 0 --tree-method hist --max-bin 128 \
    --diag-sample 1000 --verbose 2
```

---

## 6. Comprehensive Output Structure

Both scripts generate identical output categories with complete analysis results:

### 6.1 Core Results
```
<out_dir>/
├── {gene_cv_metrics.csv | loco_metrics.csv}  # Main CV results
├── metrics_aggregate.json                    # Summary statistics
├── feature_manifest.csv                      # Feature list
├── model_multiclass.pkl                      # Final trained model
```

### 6.2 Enhanced Analysis
```
├── cv_metrics_visualization/                 # 7 visualization plots
├── roc_pr_curves_*.pdf                       # ROC/PR curve analysis
├── multiclass_*_curves.pdf                   # Multiclass curve analysis
├── shap_analysis/                            # SHAP reports
├── feature_importance_analysis/              # 4-method feature analysis
├── feature_label_correlations.csv            # Leakage analysis
```

### 6.3 Diagnostic Reports
```
├── [8 comprehensive diagnostic files]        # Performance diagnostics
├── fold_*/                                   # Per-fold detailed results
└── [calibration and neighborhood analysis]   # Optional analyses
```

---

## 7. Performance Expectations and Guidelines

### 7.1 Typical Results

| CV Type | Dataset Size | F1 Improvement | Error Reduction | Processing Time |
|---------|--------------|----------------|-----------------|-----------------|
| **Gene CV** | 1K genes | 30-40% | 300-800/fold | 1-3 hours |
| **LOCO CV** | 1K genes | 25-35% | 200-600/fold | 2-5 hours |
| **Gene CV** | 10K genes | 25-35% | 500-1000/fold | 5-12 hours |
| **LOCO CV** | 10K genes | 20-30% | 300-800/fold | 8-20 hours |

### 7.2 Resource Requirements

| Configuration | Memory | Disk Space | CPU Cores | GPU Memory |
|---------------|--------|------------|-----------|------------|
| **Standard Gene CV** | 8-32 GB | 2-5 GB | 8+ | Optional |
| **Standard LOCO CV** | 16-64 GB | 3-8 GB | 12+ | Optional |
| **Memory-Optimized** | 4-16 GB | 5-15 GB | 8+ | Optional |
| **GPU Accelerated** | 8-16 GB | 2-8 GB | 8+ | 8-24 GB |

### 7.3 Scalability Guidelines

| Dataset Size | Recommended Approach | Key Optimizations |
|--------------|---------------------|-------------------|
| **<1K genes** | Gene CV standard | Default settings |
| **1K-5K genes** | Gene CV or LOCO CV | Consider memory opts |
| **5K-20K genes** | LOCO CV standard | Memory optimization |
| **20K+ genes** | LOCO CV optimized | All memory features |

---

## 8. How External-Memory Algorithm Works (Advanced)

### 8.1 Histogram-Based Split Finding

A *histogram* in gradient-boosted decision trees stores *aggregated gradients* for each feature-bucket:

1. **Binning**: Every feature discretized into `max_bin` quantile buckets (256 by default)
2. **Gradient Accumulation**: At each tree node, accumulate (∑g, ∑h) for every (feature, bucket) pair
3. **Split Scanning**: Walk buckets left→right for optimal split in O(features × max_bin) time

### 8.2 External-Memory Processing

1. **Binning**: Same as regular hist - features → buckets
2. **Block Streaming**: Data memory-mapped from disk in ~4 MB blocks
3. **Per-block Histograms**: Each block contributes to histogram accumulation
4. **Reduction**: Block histograms merged exactly for full-dataset accuracy

> **Note**: External-memory works only with CPU `hist` method, not GPU

---

## 9. Hardware & Performance Tips

### 9.1 Storage
* **Use fast NVMe SSDs** - spinning disks will be painfully slow for large datasets
* **Local storage preferred** - avoid network drives for large-scale training

### 9.2 Memory
* **RAM vs VRAM**: GPU training requires dataset to fit in GPU memory
* **Memory monitoring**: Use `htop`, `nvidia-smi` to monitor resource usage
* **Swap avoidance**: Ensure sufficient RAM to avoid swapping

### 9.3 CPU/GPU
* **Multi-core scaling**: XGBoost uses all available cores by default
* **GPU acceleration**: Effective for medium-sized datasets that fit VRAM
* **Hybrid approach**: Use GPU for training, CPU for analysis

---

## 10. Troubleshooting Common Issues

### 10.1 Memory Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| **OOM during load** | Dataset > RAM | Use `--row-cap` or `--use-chunked-loading` |
| **OOM during training** | Features > RAM | Use `--filter-features` or `--feature-selection` |
| **OOM during analysis** | Large SHAP/viz | Reduce `--diag-sample` |
| **GPU OOM** | Dataset > VRAM | Use CPU `hist` or reduce dataset size |

### 10.2 Performance Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| **Very slow training** | HDD or network drive | Use local NVMe SSD |
| **Slow analysis** | Large diagnostic sample | Reduce `--diag-sample` |
| **Memory leaks** | Large dataset processing | Use memory-optimized LOCO CV |

### 10.3 Analysis Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| **Missing plots** | Plotting disabled | Ensure `--plot-curves` enabled |
| **SHAP failures** | Memory constraints | Reduce sample size or use fallback |
| **Leakage warnings** | Correlated features | Use `--auto-exclude-leaky` |

---

## 11. Best Practices Summary

### 11.1 Dataset Size Strategy
1. **Small datasets (<1K genes)**: Use gene CV with full analysis
2. **Medium datasets (1K-10K genes)**: Consider both CV approaches
3. **Large datasets (10K-50K genes)**: Use LOCO CV with memory optimization
4. **Very large datasets (50K+ genes)**: Use all memory optimization features

### 11.2 Analysis Configuration
1. **Always enable calibration**: `--calibrate-per-class` for better probability estimates
2. **Use leakage detection**: `--check-leakage` for quality control
3. **Enable comprehensive analysis**: Default visualization and diagnostic settings
4. **Optimize for resources**: Adjust sample sizes based on available memory

### 11.3 Evaluation Strategy
1. **Use both CV approaches** when possible for comprehensive evaluation
2. **Gene CV for in-domain** assessment and model selection
3. **LOCO CV for out-of-distribution** testing and robustness evaluation
4. **Compare results** between approaches to understand model generalization

---

*Last updated: 2025-01-15*
