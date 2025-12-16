# Post-Training Diagnostics Tool

This document describes how to use the `run_diagnostics.py` tool for testing post-training diagnostics independently from the full meta-model training pipeline.

## Overview

The diagnostics tool allows you to run individual post-training analysis routines on pre-trained meta-models. This is especially useful for:

1. Debugging issues in the post-training phase without rerunning the entire training pipeline
2. Testing enhancements to diagnostic routines on existing models
3. Generating missing performance metrics or visualizations for already trained models

## Usage

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_diagnostics \
  --model-dir models/meta_model_debug_run \
  --dataset train_pc_1000/master \
  --diag-sample 5000 \
  --verbose
```

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--model-dir` | Directory containing the trained model and metadata (required) |
| `--dataset` | Path to dataset for testing (required) |
| `--diag-sample` | Number of samples to use for diagnostics (default: 5000) |
| `--neigh-sample` | Sample size for neighborhood diagnostics (default: 3) |
| `--neigh-window` | Window size for neighborhood diagnostics (default: 12) |
| `--plot-curves` | Generate ROC and PR curves |
| `--n-roc-points` | Number of points in ROC curve (default: 201) |
| `--plot-format` | Format for output plots: pdf, png, or svg (default: pdf) |
| `--include-tns` | Include true negatives in performance metrics |
| `--check-leakage` | Run leakage checks on features |
| `--leakage-probe` | Run detailed leakage probe analysis |
| `--verbose` | Enable detailed logging output |
| `--skip-module` | Skip specific diagnostic modules (options: meta_splice_performance, base_vs_meta, leakage_probe, neighbour_diagnostics, probability_diagnostics) |

## Available Diagnostic Modules

1. **Meta Splice Performance (`meta_splice_performance`)**
   - Evaluates model performance on splice site prediction
   - Generates detailed metrics on accuracy, precision, recall, etc.
   - Outputs `full_splice_performance_meta.tsv`

2. **Base vs Meta Comparison (`base_vs_meta`)**
   - Compares performance between base and meta models
   - Requires both `full_splice_performance_meta.tsv` and `full_splice_performance.tsv`
   - Generates comparative visualizations and metrics

3. **Leakage Probe (`leakage_probe`)**
   - Analyzes feature correlations to detect potential data leakage
   - Only runs if `--leakage-probe` is specified

4. **Neighbor Window Diagnostics (`neighbour_diagnostics`)**
   - Analyzes spatial effects in the data
   - Runs if `--neigh-sample` > 0

5. **Probability Diagnostics (`probability_diagnostics`)**
   - Analyzes probability distributions and calibration
   - Suggests optimal threshold values

## Troubleshooting

### Common Issues and Solutions

1. **Non-numeric data error**
   - Error: `could not convert string to float: 'X'`
   - Solution: The tool implements a fallback approach that handles mixed data types

2. **Missing performance files**
   - Error: `Missing performance files for comparison`
   - Solution: Run the `meta_splice_performance` module first to generate the required TSV files

3. **Neighbor window diagnostics failure**
   - Error: `No such file or directory`
   - Solution: Ensure the model directory has all required metadata and model files

## Examples

### Basic Run with All Diagnostics

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_diagnostics \
  --model-dir models/meta_model_run1 \
  --dataset train_pc_1000/master \
  --verbose \
  --plot-curves \
  --include-tns
```

### Run Only Base vs Meta Comparison

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_diagnostics \
  --model-dir models/meta_model_run1 \
  --dataset train_pc_1000/master \
  --skip-module meta_splice_performance leakage_probe neighbour_diagnostics probability_diagnostics
```

### Memory-Efficient Run

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_diagnostics \
  --model-dir models/meta_model_run1 \
  --dataset train_pc_1000/master \
  --diag-sample 2000 \
  --verbose
```
