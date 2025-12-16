# Cross-Validation Evaluation Extension

This document describes the CV evaluation extension modules that provide enhanced metrics and visualization capabilities for MetaSpliceAI meta-models.

## Overview

The CV evaluation extension modules provide tools for:

1. Per-fold performance metrics calculation and storage (CSV/TSV output)
2. Cross-validation aggregation with statistical measures (mean, std, confidence intervals)
3. Calculation of relative error reduction metrics (FP/FN reduction percentages)
4. Visualization of performance across folds (box plots, paired bar charts)
5. Transcript-level top-k accuracy metrics across folds

These modules are designed as **non-invasive extensions** to the existing CV pipeline and will not affect the behavior of standard gene-aware and chromosome-aware CV scripts.

## Module Structure

The extension consists of three main components:

1. **cv_evaluation.py** - Core evaluation utilities for per-fold metrics and CV aggregation
2. **transcript_level_cv.py** - Transcript-level top-k accuracy evaluation across CV folds
3. **run_cv_evaluation.py** - Runner script to process fold directories and generate reports

## Usage

### Basic Usage

To run a complete evaluation on an existing cross-validation output directory:

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_cv_evaluation \
    --cv-output-dir models/meta_model_per_class_calibrated \
    --out-dir evaluation_reports
```

### Advanced Options

Additional parameters for customization:

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_cv_evaluation \
    --cv-output-dir models/meta_model_per_class_calibrated \
    --base-model-name "SpliceAI" \
    --meta-model-name "Meta-Model with Calibration" \
    --fold-pattern "fold_*" \
    --confidence-level 0.95 \
    --out-dir custom_evaluation_reports
```

### Integrating with Existing CV Scripts

To use these evaluation modules within existing CV pipelines:

```python
from meta_spliceai.splice_engine.meta_models.evaluation.cv_evaluation import (
    calculate_fold_metrics,
    save_fold_metrics,
    run_full_cv_evaluation
)

# At the end of your CV run, after all folds are processed:
fold_results = []
for fold_id, (base_metrics, meta_metrics) in enumerate(all_fold_metrics):
    fold_results.append({
        "base": base_metrics,
        "meta": meta_metrics
    })

# Generate comprehensive evaluation report
run_full_cv_evaluation(
    fold_results=fold_results,
    output_dir="your_output_dir",
    base_model_name="Your Base Model",
    meta_model_name="Your Meta Model"
)
```

## Output Files

The evaluation modules generate several output files:

### Per-fold Metrics
- `fold_X_metrics.csv` - Detailed metrics for each fold

### Aggregate Metrics
- `all_folds_metrics.csv` - Combined metrics from all folds
- `cv_metrics_summary.csv` - Statistical summary (mean, std, CI) across folds
- `cv_metrics_summary.json` - JSON format of summary statistics
- `cv_headline_metrics.json` - Simplified headline metrics

### Visualizations
- `cv_performance_summary.png` - Overall performance summary plot
- `f1_comparison_by_fold.png` - F1 score comparison across folds
- `pct_fp_reduction_boxplot.png` - Box plot of false positive reductions
- `pct_fn_reduction_boxplot.png` - Box plot of false negative reductions

### Transcript-Level Metrics (when using transcript_level_cv)
- `fold_X_transcript_metrics.csv` - Transcript-level metrics per fold
- `all_transcript_metrics.csv` - Combined transcript metrics
- `transcript_summary.json` - Summary of transcript-level performance
- `transcript_top_k_accuracy_boxplot.png` - Box plot of top-k accuracy
- `transcript_top_k_accuracy_line.png` - Line plot of accuracy by k value

## Headline Metrics

The evaluation modules provide several key headline metrics:

1. **F1 Score Improvement** - Percentage improvement in F1 score 
2. **False Positive Reduction** - Percentage reduction in false positives
3. **False Negative Reduction** - Percentage reduction in false negatives

These metrics are calculated by:
1. Computing per-fold relative improvements
2. Aggregating across folds (mean, std, confidence intervals)

## Integration with Existing Data

The evaluation modules automatically detect and work with existing performance files:
- Base model performance: `base_performance.json`, `base_metrics.json`, `full_splice_performance.tsv`
- Meta model performance: `meta_performance.json`, `meta_metrics.json`, `full_splice_performance_meta.tsv`

## Backward Compatibility

This extension is designed to be non-invasive and will not interfere with existing CV scripts or evaluation methods. All functionality is implemented as additions rather than modifications to ensure full backward compatibility.
