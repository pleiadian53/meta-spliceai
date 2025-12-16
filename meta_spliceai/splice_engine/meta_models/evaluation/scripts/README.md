# Evaluation Scripts

This directory contains utility scripts for performance analysis and visualization in the MetaSpliceAI evaluation package.

## Scripts Overview

### `generate_f1_pr_curves.py`
Generates F1-based Precision-Recall curves from existing CV results.

**Usage:**
```bash
python generate_f1_pr_curves.py <results_dir>
```

**Example:**
```bash
python generate_f1_pr_curves.py results/gene_cv_pc_1000_3mers_run_2_more_genes
```

**Features:**
- Loads existing PR curve data from CSV files
- Calculates F1 scores for each precision-recall point
- Identifies maximum F1 operating points
- Adds F1 contour lines (0.5, 0.6, 0.7, 0.8, 0.9)
- Generates publication-ready PDF plots
- Saves F1 summary metrics to JSON

**Output:**
- `pr_curves_f1_optimized.pdf` - F1-based PR curves
- `f1_pr_curve_summary.json` - Detailed F1 metrics

## Integration with Evaluation Package

These scripts are designed to work with the main evaluation modules:

- **`viz_utils.py`** - Core visualization functions
- **`cv_metrics_viz.py`** - CV metrics generation
- **`shap_viz.py`** - Feature importance analysis

## Design Philosophy

### Why F1-Based PR Curves?
- **Imbalanced data**: Splice site prediction has few positive examples
- **Balanced metric**: F1 score provides harmonic mean of precision/recall
- **Practical interpretation**: Single number for model comparison
- **Operating point selection**: Maximum F1 shows optimal threshold

### Script Organization
- **Modular design**: Each script has a single, focused purpose
- **Reusable functions**: Core logic can be imported by other modules
- **Clear documentation**: Comprehensive docstrings and usage examples
- **Error handling**: Robust input validation and error messages

## Future Scripts

Potential additions to this directory:
- `compare_ap_vs_f1.py` - Side-by-side AP vs F1 comparison
- `performance_analysis.py` - Comprehensive performance metrics
- `threshold_optimization.py` - Optimal threshold selection
- `model_comparison.py` - Multi-model performance comparison 