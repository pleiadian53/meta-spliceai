# Evaluation Package Module Overview

**Package:** `meta_spliceai.splice_engine.meta_models.evaluation`  
**Purpose:** Comprehensive evaluation, analysis, and visualization tools for meta-learning splice site prediction models  
**Last Updated:** January 2025  

## Table of Contents

1. [Package Overview](#package-overview)
2. [Core Evaluation Modules](#core-evaluation-modules)
3. [Feature Importance & SHAP Analysis](#feature-importance--shap-analysis)
4. [Cross-Validation & Metrics](#cross-validation--metrics)
5. [Specialized Analysis Tools](#specialized-analysis-tools)
6. [Visualization Utilities](#visualization-utilities)
7. [Integration & Utilities](#integration--utilities)
8. [Usage Guidelines](#usage-guidelines)

---

## Package Overview

The evaluation package provides a comprehensive suite of tools for analyzing, evaluating, and visualizing meta-learning models for splice site prediction. It focuses on:

- **Rigorous cross-validation** with gene-aware splitting to prevent data leakage
- **Feature importance analysis** using both statistical tests and SHAP explanations
- **Model diagnostics** including overfitting detection and calibration analysis
- **Performance visualization** with publication-ready plots and metrics
- **Scalable analysis** designed to handle large genomic datasets efficiently

---

## Core Evaluation Modules

### 1. **Cross-Validation & Metrics**

#### `cv_metrics_viz.py` *(31KB, 773 lines)*
**Purpose:** Comprehensive cross-validation metrics visualization and reporting  
**Key Features:**
- Dynamic baseline error calculation from training datasets
- Organized CV output with automatic directory structure
- Publication-ready performance comparison plots
- Integration with dynamic baseline error calculator

**Key Functions:**
```python
generate_cv_metrics_report(csv_path, out_dir, dataset_path, plot_format, dpi)
create_base_vs_meta_comparison_plots(df, out_dir, plot_format, dpi)
load_cv_metrics(csv_path)
```

**Recent Updates:** 
- Removed all hardcoded baseline values
- Added purely dynamic baseline calculation
- Enhanced error reduction reporting with percentages

#### `baseline_error_calculator.py` *(11KB, 330 lines)*
**Purpose:** Dynamic calculation of baseline model errors for any dataset  
**Key Features:**
- Dataset-agnostic baseline error analysis
- Efficient sampling for large datasets (configurable sample sizes)
- Automatic scaling to CV dataset proportions
- Compatible with any k-mer configuration and dataset size

**Key Functions:**
```python
calculate_baseline_errors_from_dataset(dataset_path, sample_size, verbose)
calculate_cv_error_reductions(cv_df, dataset_path, sample_size, verbose)
```

**Use Case:** Eliminates hardcoded baseline values, works with any training dataset (train_pc_1000, train_pc_20000, etc.)

### 2. **Specialized Cross-Validation**

#### `cv_evaluation.py` *(19KB, 557 lines)*
**Purpose:** Core cross-validation evaluation framework  
**Key Features:**
- Gene-aware fold splitting to prevent data leakage
- Performance metric calculation across folds
- Integration with meta-learning ensemble models

#### `transcript_level_cv.py` *(16KB, 509 lines)*
**Purpose:** Transcript-level cross-validation analysis  
**Key Features:**
- Transcript-aware evaluation metrics
- Hierarchical performance analysis
- Integration with transcript mapping utilities

---

## Feature Importance & SHAP Analysis

### 3. **Scalable SHAP Analysis**

#### `shap_incremental.py` *(61KB, 1533 lines)*
**Purpose:** Memory-efficient, scalable SHAP feature importance analysis  
**Key Features:**
- **Incremental processing** to avoid OOM (Out of Memory) issues
- **Streaming SHAP computation** without loading entire datasets
- **Online aggregation** of SHAP values using configurable batch sizes
- **Custom ensemble support** for MetaSpliceAI's multi-class architectures
- **Enhanced compatibility** with Keras 3.x, TensorFlow, and transformers

**Core Innovation - Incremental Processing:**
```python
def incremental_shap_importance(model, X, *, batch_size=512, background_size=1000):
    """
    Compute global SHAP feature importance without large memory use.
    
    - Processes data in configurable batches to fit GPU/CPU RAM
    - Uses small background dataset for explainer efficiency  
    - Online accumulation of |SHAP| values without materializing full tensor
    - Memory usage: O(batch_size * features) instead of O(dataset_size * features)
    """
```

**Key Functions:**
```python
incremental_shap_importance(model, X, batch_size, background_size, class_idx)
run_incremental_shap_analysis(dataset_path, out_dir, sample, batch_size)
create_memory_efficient_beeswarm_plot(model, X, sample_size, top_n_features)
create_ensemble_beeswarm_plots(model, X, save_dir, plot_format)
```

**Scalability Benefits:**
- **Memory Efficiency:** Handles datasets too large for traditional SHAP analysis
- **Configurable Batching:** Adjust batch_size based on available memory
- **Background Sampling:** Small representative sample for explainer initialization  
- **Incremental Aggregation:** Computes mean(|SHAP|) without storing full arrays

#### `shap_viz.py` *(36KB, 902 lines)*
**Purpose:** Comprehensive SHAP visualization suite  
**Key Features:**
- **Multiple visualization types:** bar charts, beeswarm plots, heatmaps
- **Per-class analysis** for multi-class ensemble models
- **Publication-ready plots** with customizable formats
- **Memory-efficient rendering** with sampling for large datasets

**Key Functions:**
```python
generate_comprehensive_shap_report(importance_csv, model_path, dataset_path, out_dir)
create_feature_importance_barcharts(importance_csv, out_dir, top_n, plot_format)
create_shap_beeswarm_plots(model_path, dataset_path, out_dir, sample_size)
create_feature_importance_heatmap(importance_csv, out_dir, top_n, plot_format)
```

### 4. **Traditional Feature Importance**

#### `feature_importance.py` *(67KB, 1547 lines)*
**Purpose:** Statistical feature importance analysis and testing  
**Key Features:**
- Permutation importance with statistical significance testing
- Feature correlation analysis and multicollinearity detection
- Integration with meta-learning ensemble architectures

#### `feature_importance_integration.py` *(38KB, 946 lines)*
**Purpose:** Integration layer for feature importance workflows  
**Key Features:**
- Unified interface for multiple importance methods
- Batch processing for large feature sets
- Integration with cross-validation pipelines

---

## Specialized Analysis Tools

### 5. **Model Diagnostics**

#### `overfitting_monitor.py` *(25KB, 600 lines)*
**Purpose:** Comprehensive overfitting detection and early stopping  
**Key Features:**
- Performance gap analysis between training and validation
- Convergence monitoring with configurable thresholds
- Early stopping recommendations for optimal model size

#### `calibration_diagnostics.py` *(30KB, 826 lines)*
**Purpose:** Model calibration analysis and diagnostics  
**Key Features:**
- Reliability diagrams and calibration curves
- Expected Calibration Error (ECE) computation
- Temperature scaling analysis

#### `calibration_integration.py` *(12KB, 323 lines)*
**Purpose:** Integration utilities for calibration analysis  
**Key Features:**
- Platt scaling integration
- Temperature scaling workflows
- Calibration method comparison

### 6. **Data Quality & Leakage Analysis**

#### `leakage_analysis.py` *(32KB, 797 lines)*
**Purpose:** Data leakage detection in cross-validation setups  
**Key Features:**
- Gene-level leakage detection
- Transcript overlap analysis
- Statistical tests for fold independence

#### `transcript_mapping.py` *(18KB, 435 lines)*
**Purpose:** Transcript-level analysis and mapping utilities  
**Key Features:**
- Gene-to-transcript mapping validation
- Cross-validation fold transcript distribution
- Hierarchical analysis support

---

## Visualization Utilities

### 7. **Performance Visualization**

#### `multiclass_roc_pr.py` *(17KB, 439 lines)*
**Purpose:** Multi-class ROC and Precision-Recall curve analysis  
**Key Features:**
- One-vs-rest and micro/macro averaging
- Area under curve calculations
- Confidence intervals for performance metrics

#### `top_k_metrics.py` *(8.0KB, 215 lines)*
**Purpose:** Top-k accuracy and ranking metrics  
**Key Features:**
- Top-k accuracy calculation for multi-class problems
- Ranking-based performance evaluation
- Integration with ensemble prediction workflows

#### `viz_utils.py` *(14KB, 411 lines)*
**Purpose:** Common visualization utilities and styling  
**Key Features:**
- Consistent plot styling and color schemes
- Publication-ready figure formatting
- Common plotting functions and utilities

### 8. **Evaluation Scripts**

#### `scripts/` *(Directory)*
**Purpose:** Utility scripts for performance analysis and visualization  
**Key Features:**
- Standalone scripts for specific evaluation tasks
- F1-based PR curve generation
- Performance comparison utilities
- Modular design for easy integration

**Key Scripts:**
```bash
# Generate F1-based PR curves from existing CV results
python scripts/generate_f1_pr_curves.py <results_dir>

# Example usage
python scripts/generate_f1_pr_curves.py results/gene_cv_pc_1000_3mers_run_2_more_genes
```

**Features:**
- Loads existing PR curve data from CSV files
- Calculates F1 scores for each precision-recall point
- Identifies maximum F1 operating points
- Adds F1 contour lines for performance visualization
- Generates publication-ready PDF plots
- Saves detailed F1 metrics to JSON

**Design Philosophy:**
- **Modular design**: Each script has a single, focused purpose
- **Reusable functions**: Core logic can be imported by other modules
- **Clear documentation**: Comprehensive docstrings and usage examples
- **Error handling**: Robust input validation and error messages

---

## Integration & Utilities

### 9. **Support Modules**

#### `feature_utils.py` *(6.7KB, 182 lines)*
**Purpose:** Feature processing and manipulation utilities  
**Key Features:**
- Feature name standardization
- K-mer feature handling
- Feature set validation and harmonization

#### `plot_feature_correlations.py` *(7.3KB, 236 lines)*
**Purpose:** Feature correlation analysis and visualization  
**Key Features:**
- Correlation matrix computation and visualization
- Feature clustering based on correlation
- Multicollinearity detection

#### `standalone_shap_analysis.py` *(38KB, 937 lines)*
**Purpose:** Standalone SHAP analysis workflows  
**Key Features:**
- Independent SHAP analysis pipelines
- Batch processing for multiple models
- Integration with external analysis tools

---

## Usage Guidelines

### Memory Management for Large Datasets

**For SHAP Analysis:**
```python
# Use incremental SHAP for large datasets
from meta_spliceai.splice_engine.meta_models.evaluation.shap_incremental import incremental_shap_importance

# Configure for your memory constraints
importance = incremental_shap_importance(
    model=trained_model,
    X=feature_matrix,
    batch_size=512,        # Adjust based on available memory
    background_size=1000,  # Small representative sample
    verbose=True
)
```

**For CV Metrics:**
```python
# Use dynamic baseline calculation
from meta_spliceai.splice_engine.meta_models.evaluation.cv_metrics_viz import generate_cv_metrics_report

result = generate_cv_metrics_report(
    csv_path="cv_metrics.csv",
    out_dir="results/analysis",
    dataset_path="train_pc_1000_3mers/master",  # Any dataset, no hardcoding
    plot_format="pdf"
)
```

### Best Practices

1. **Memory Efficiency**: Use incremental processing for SHAP analysis on large datasets
2. **Dynamic Baselines**: Always provide dataset_path for accurate baseline calculations
3. **Organized Output**: Use the automatic directory organization features
4. **Publication Quality**: Leverage the built-in styling for publication-ready figures
5. **Gene-Aware CV**: Always use gene-aware splitting to prevent data leakage

### Common Workflows

**Complete Model Evaluation:**
```python
# 1. Run gene-aware CV with metrics
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000_3mers/master \
    --out-dir results/eval_run_1 \
    --n-folds 5

# 2. Generate comprehensive SHAP analysis
from meta_spliceai.splice_engine.meta_models.evaluation.shap_incremental import run_incremental_shap_analysis
shap_dir = run_incremental_shap_analysis("train_pc_1000_3mers/master", "results/eval_run_1")

# 3. Create publication-ready visualizations
from meta_spliceai.splice_engine.meta_models.evaluation.shap_viz import generate_comprehensive_shap_report
shap_report = generate_comprehensive_shap_report(
    importance_csv=f"{shap_dir}/importance/shap_importance_incremental.csv",
    model_path="results/eval_run_1/model_multiclass.pkl",
    dataset_path="train_pc_1000_3mers/master",
    out_dir="results/eval_run_1"
)
```

---

## Recent Updates & Improvements

### January 2025
- **Fixed SHAP Analysis:** Resolved all indentation and syntax errors in shap_viz.py and shap_incremental.py
- **Enhanced CV Metrics:** Added purely dynamic baseline calculation, removed all hardcoded values
- **Improved Memory Management:** Enhanced incremental SHAP processing for better scalability
- **Better Documentation:** Comprehensive module overview and usage guidelines

### Key Innovations
- **Incremental SHAP Processing:** Breakthrough in memory-efficient feature importance analysis
- **Dynamic Baseline Calculation:** Eliminates dataset-specific hardcoding
- **Enhanced Ensemble Support:** Full compatibility with MetaSpliceAI's meta-learning architectures
- **Publication-Ready Output:** Organized, high-quality visualizations and reports

---

This evaluation package represents a comprehensive, scalable, and robust framework for analyzing meta-learning models in splice site prediction, with particular emphasis on memory efficiency and publication-quality results.