# CV Metrics Visualization Module

This module provides comprehensive visualization capabilities for comparing base model (SpliceAI) vs meta model performance across cross-validation folds.

## Overview

The CV metrics visualization module creates publication-ready plots that illustrate the performance improvements achieved by the meta model compared to the base model. It analyzes the `gene_cv_metrics.csv` file generated during cross-validation and produces multiple visualization perspectives.

## Key Features

### 📊 **Performance Comparison Plots**
- **F1 Score Comparison**: Fold-by-fold and statistical summary comparison
- **ROC AUC Comparison**: Area under ROC curve analysis with confidence intervals
- **Average Precision Comparison**: Precision-recall performance across folds

### 📉 **Error Reduction Analysis**
- **False Positive Reduction**: Shows `delta_fp` (Base FP - Meta FP) across folds
- **False Negative Reduction**: Shows `delta_fn` (Base FN - Meta FN) across folds
- **Combined Error Analysis**: Total error reduction visualization
- **Quadrant Analysis**: Scatter plot showing improvement patterns

### 📈 **Advanced Analytics**
- **Performance Overview**: Multi-metric comparison dashboard
- **Improvement Summary**: Percentage improvements with statistical significance
- **Top-k Accuracy Analysis**: Gene-level and splice-type-specific accuracy
- **Method Comparison**: Base vs meta model effectiveness

## Metric Interpretations

### Core CV Metrics
- **`test_macro_f1`**: Macro-averaged F1 score across all three classes (neither, donor, acceptor) on test set
- **`splice_accuracy`**: Accuracy calculated only on splice sites (donor + acceptor), excluding "neither" class
- **`splice_macro_f1`**: Macro-averaged F1 score for splice sites only (donor + acceptor)

### Base vs Meta Comparison
- **`base_f1` / `meta_f1`**: F1 scores for splice site detection (binary classification)
- **`auc_base` / `auc_meta`**: ROC AUC scores for splice site detection
- **`ap_base` / `ap_meta`**: Average precision scores for splice site detection

### Error Reduction
- **`delta_fp`**: False positive reduction (Base FP - Meta FP). Positive = Meta better
- **`delta_fn`**: False negative reduction (Base FN - Meta FN). Positive = Meta better

## Usage

### 1. Automatic Integration (Recommended)

Run the integration script to automatically add visualization to your CV workflow:

```bash
python scripts/integrate_cv_metrics_viz.py
```

This modifies `run_gene_cv_sigmoid.py` to automatically generate visualization reports after CV completion.

### 2. Standalone Usage

For existing CSV files, use the standalone script:

```bash
python scripts/generate_cv_metrics_viz.py path/to/gene_cv_metrics.csv --out-dir output_directory
```

**Options:**
- `--plot-format`: Format for plots (`png`, `pdf`, `svg`). Default: `png`
- `--dpi`: Resolution for plots. Default: `300`

### 3. Python API

```python
from meta_spliceai.splice_engine.meta_models.evaluation.cv_metrics_viz import generate_cv_metrics_report

# Generate comprehensive report
result = generate_cv_metrics_report(
    csv_path="models/meta_model_test/gene_cv_metrics.csv",
    out_dir="visualization_output",
    plot_format='png',
    dpi=300
)

print(f"Generated {len(result['plot_files'])} plots")
print(f"Summary report: {result['report_path']}")
```

### 4. Demo Script

Try the functionality with the demo script:

```bash
python scripts/demo_cv_metrics_viz.py
```

## Generated Outputs

### 📁 **Directory Structure**
```
output_directory/
├── cv_metrics_visualization/
│   ├── cv_f1_comparison.png           # F1 score comparison
│   ├── cv_auc_comparison.png          # ROC AUC comparison  
│   ├── cv_ap_comparison.png           # Average precision comparison
│   ├── cv_error_reduction.png         # Error reduction analysis
│   ├── cv_performance_overview.png    # Multi-metric overview
│   ├── cv_improvement_summary.png     # Improvement summary
│   ├── cv_topk_analysis.png          # Top-k accuracy analysis
│   └── cv_metrics_summary.txt        # Text summary report
```

### 📊 **Plot Descriptions**

#### 1. **F1 Comparison Plot** (`cv_f1_comparison.png`)
- **Left panel**: F1 scores across CV folds for base and meta models
- **Right panel**: Statistical summary with mean ± std and improvement quantification

#### 2. **ROC AUC Comparison Plot** (`cv_auc_comparison.png`)
- **Left panel**: ROC AUC scores across CV folds
- **Right panel**: Statistical summary showing AUC improvements

#### 3. **Average Precision Comparison Plot** (`cv_ap_comparison.png`)
- **Left panel**: Average precision scores across CV folds
- **Right panel**: Statistical summary of precision-recall performance

#### 4. **Error Reduction Analysis Plot** (`cv_error_reduction.png`)
- **Top-left**: False positive reduction across folds
- **Top-right**: False negative reduction across folds
- **Bottom-left**: Average error reduction summary
- **Bottom-right**: Total error reduction per fold

#### 5. **Performance Overview Plot** (`cv_performance_overview.png`)
- **Top-left**: Multi-metric comparison (F1, AUC, AP)
- **Top-right**: Accuracy metrics breakdown
- **Bottom-left**: F1 score metrics breakdown
- **Bottom-right**: Top-k accuracy distribution

#### 6. **Improvement Summary Plot** (`cv_improvement_summary.png`)
- **Left panel**: Percentage improvements for each metric
- **Right panel**: Error reduction effectiveness scatter plot with quadrant analysis

#### 7. **Top-k Analysis Plot** (`cv_topk_analysis.png`) *(if available)*
- **Top-left**: Top-k accuracy across folds
- **Top-right**: Donor vs Acceptor top-k comparison
- **Bottom-left**: Top-k accuracy distribution
- **Bottom-right**: Gene count analysis per fold

### 📄 **Summary Report** (`cv_metrics_summary.txt`)

Text-based summary containing:
- Performance comparison statistics
- Error reduction analysis
- Top-k accuracy metrics
- Key improvement highlights

## Interpretation Guide

### 🎯 **What to Look For**

#### **Positive Indicators**
- **Meta F1 > Base F1**: Improved splice site detection
- **Meta AUC > Base AUC**: Better ranking/discrimination
- **Meta AP > Base AP**: Better precision-recall balance
- **Positive ΔFP and ΔFN**: Fewer errors in meta model

#### **Performance Patterns**
- **Consistent Improvement**: Meta model better across all folds
- **Stable Performance**: Low variance across folds indicates robust model
- **Error Reduction**: Both FP and FN reduction indicates balanced improvement

#### **Quadrant Analysis** (Error Reduction Scatter Plot)
- **Green quadrant** (top-right): Both FP and FN improved ✅
- **Blue quadrants**: One error type improved 🔄
- **Red quadrant** (bottom-left): Both error types worse ❌

### 📊 **Statistical Significance**
- Error bars show standard deviation across CV folds
- Improvement percentages indicate relative gains
- Consistent patterns across folds suggest genuine improvement

## Example Output Interpretation

```
F1 Score Improvement: 0.045 ± 0.012 (4.8% improvement)
ROC AUC Improvement: 0.023 ± 0.008 (2.7% improvement)
Average Precision Improvement: 0.067 ± 0.019 (8.2% improvement)
False Positive Reduction: 145.2 ± 23.4 fewer FPs per fold
False Negative Reduction: 89.6 ± 15.7 fewer FNs per fold
```

**Interpretation**: The meta model shows consistent improvements across all metrics, with particularly strong gains in precision-recall performance (8.2% AP improvement) and substantial error reduction (234.8 fewer total errors per fold).

## Integration with Existing Workflow

### Automatic Generation
When integrated, the visualization report is generated automatically after CV completion:

```
[Gene-CV-Sigmoid] ... (normal CV output)

Generating CV metrics visualization report...
[INFO] CV metrics visualization completed successfully
[INFO] Visualization directory: models/output/cv_metrics_visualization
[INFO] Generated 7 plots:
  - F1 Comparison: cv_f1_comparison.png
  - Auc Comparison: cv_auc_comparison.png
  - Ap Comparison: cv_ap_comparison.png
  - Error Reduction: cv_error_reduction.png
  - Performance Overview: cv_performance_overview.png
  - Improvement Summary: cv_improvement_summary.png
  - Topk Analysis: cv_topk_analysis.png
```

### Error Handling
- **Graceful Degradation**: Visualization failures don't break main CV workflow
- **Missing Data**: Handles missing columns gracefully
- **Verbose Mode**: Use `--verbose` flag for detailed error reporting

## Dependencies

- `pandas` - Data manipulation
- `numpy` - Numerical computations  
- `matplotlib` - Plotting backend
- `seaborn` - Statistical visualization
- `pathlib` - Path handling

## Troubleshooting

### Common Issues

#### **Missing CSV File**
```
Error: CSV file not found at path/to/gene_cv_metrics.csv
```
**Solution**: Ensure the CV workflow completed successfully and generated the metrics file.

#### **Missing Columns**
```
Warning: Column 'base_f1' not found in dataset
```
**Solution**: Check that your CV workflow includes base vs meta comparison metrics.

#### **Import Errors**
```
ModuleNotFoundError: No module named 'cv_metrics_viz'
```
**Solution**: Ensure you're running from the project root directory.

### **Performance Tips**
- Use `--plot-format pdf` for publication-quality vector graphics
- Increase `--dpi` for higher resolution raster images
- Run demo script first to verify functionality

## Contributing

To extend the visualization module:

1. **Add new plot functions** following the `_create_*_plot()` pattern
2. **Register new plots** in `create_base_vs_meta_comparison_plots()`
3. **Update summary statistics** in `_calculate_summary_statistics()`
4. **Add documentation** for new metrics and interpretations

## References

- [Matplotlib Documentation](https://matplotlib.org/stable/tutorials/index.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Cross-validation Best Practices](https://scikit-learn.org/stable/modules/cross_validation.html)

---

**Note**: This module is designed to work seamlessly with the existing splice site prediction workflow. For questions or feature requests, refer to the main project documentation. 