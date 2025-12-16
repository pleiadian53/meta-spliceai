# Evaluation Documentation

This directory contains technical documentation related to the evaluation components of the splice site prediction system.

## Documents

### Package Overview

- **[QUICK_START_GUIDE.md](./QUICK_START_GUIDE.md)** - 🚀 **Start here!** Quick setup and common workflows
  - Essential commands and code examples
  - Key innovations and benefits
  - Troubleshooting tips and common workflows

- **[evaluation_modules_overview.md](./evaluation_modules_overview.md)** - Comprehensive overview of all evaluation package modules  
  - Complete module descriptions and key features
  - Usage guidelines and best practices  
  - Recent updates and innovations
  - Integration examples and workflows

### Performance Analysis

- **[confusion_matrix_interpretation_guide.md](./confusion_matrix_interpretation_guide.md)** - Complete guide to interpreting confusion matrices from CV scripts
  - **Standard convention**: Rows = True labels, Columns = Predicted labels
  - **Cell-by-cell interpretation** with real examples from gene-aware and chromosome-aware CV
  - **Performance metrics calculation** (Precision, Recall, F1-Score, Accuracy)
  - **Memory aids** and troubleshooting guidelines
  - **Common questions** and performance indicators

- **[thresholds_and_fmax_analysis.md](./thresholds_and_fmax_analysis.md)** - Comprehensive guide to thresholds and Fmax analysis in multiclass classification
  - **Threshold generalization** to multiclass scenarios (confidence-based, per-class, one-vs-rest)
  - **Fmax calculation** (binary, macro, micro, weighted variants)
  - **Relationship analysis** between Fmax and argmax-based F1 scores
  - **Practical implementation** with code examples and integration guidelines
  - **Model calibration insights** and threshold optimization strategies

### SHAP Analysis

- **[incremental_shap_analysis_guide.md](./incremental_shap_analysis_guide.md)** - Complete guide to scalable SHAP feature importance analysis
  - **Innovation:** Incremental processing to solve Out-of-Memory (OOM) problems
  - Memory-efficient algorithms for large genomic datasets
  - Performance optimization and parameter tuning
  - Integration with meta-learning pipelines
  - Troubleshooting and best practices

- **[SHAP_Compatibility_Resolution.md](./SHAP_Compatibility_Resolution.md)** - Comprehensive guide to resolving SHAP/Keras/TensorFlow compatibility issues in the gene-aware cross-validation pipeline
  - Problem analysis and root cause identification
  - Solution implementation and library version management
  - Testing procedures and verification steps
  - Future maintenance guidelines
  - Technical implementation details

### Overfitting Analysis

- **[overfitting_analysis_guide.md](./overfitting_analysis_guide.md)** - Complete guide to understanding and interpreting overfitting analysis in meta-learning cross-validation
  - Comprehensive metric definitions (performance gap, overfitting score, convergence analysis)
  - Detailed visualization interpretation guide
  - Model instance naming convention for independent sigmoid ensemble
  - Best practices and troubleshooting guidelines
  - Integration examples and usage patterns

## Related Modules

The documentation in this directory relates to the following evaluation modules:

### Core Analysis
- `shap_incremental.py` - **Scalable SHAP analysis** with incremental processing for large datasets
- `shap_viz.py` - SHAP visualization utilities with publication-ready plots
- `baseline_error_calculator.py` - **Dynamic baseline error calculation** for any dataset
- `cv_metrics_viz.py` - Cross-validation metrics visualization with organized output

### Feature Importance  
- `feature_importance.py` - Statistical feature importance analysis and testing
- `feature_importance_integration.py` - Integration layer for feature importance workflows

### Model Diagnostics
- `overfitting_monitor.py` - Comprehensive overfitting detection and analysis system
- `calibration_diagnostics.py` - Model calibration analysis and diagnostics
- `leakage_analysis.py` - Data leakage detection in cross-validation setups

### Specialized Analysis
- `multiclass_roc_pr.py` - Multi-class ROC and Precision-Recall curve analysis
- `transcript_level_cv.py` - Transcript-level cross-validation analysis
- `top_k_metrics.py` - Top-k accuracy and ranking metrics

## Contributing

When adding new evaluation components or fixing compatibility issues:

1. **Document the problem** - Create clear problem descriptions with error messages
2. **Analyze root causes** - Investigate the technical reasons behind issues
3. **Document solutions** - Provide step-by-step resolution procedures
4. **Include testing** - Add verification steps and test cases
5. **Plan maintenance** - Include future monitoring and maintenance guidelines

## Maintenance Schedule

- **Quarterly**: Review library compatibility and update documentation
- **Before major updates**: Test all documented procedures
- **After fixes**: Update documentation with lessons learned 