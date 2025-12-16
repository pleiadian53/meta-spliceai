# Meta-Models Analysis Documentation

This directory contains documentation and examples for various analysis components of the meta-learning system.

## Documentation Structure

### 📊 **Splice Site Visualization** (`splice_site_visualization/`)
Complete visualization system for comparing base model (SpliceAI) vs meta-learning predictions.

**Contents:**
- `splice_visualization_solution.md` - Technical documentation
- `SOLUTION_SUMMARY.md` - Executive summary
- `demo_splice_visualization.py` - Demo application
- `test_splice_visualization_comprehensive.py` - Test suite
- `README.md` - Quick start guide

**Status:** ✅ Complete and fully functional

### 📈 **Meta-Learning Correlation Analysis**
- `meta_learning_correlation_analysis_narrative.md` - Analysis of feature correlations and meta-learning improvements

## Quick Access

### Run Splice Site Visualization Demo
```bash
conda activate surveyor
python splice_site_visualization/demo_splice_visualization.py
```

### Run Comprehensive Tests
```bash
conda activate surveyor
python splice_site_visualization/test_splice_visualization_comprehensive.py
```

## Related Components

### Core Implementation Files
- `../splice_site_comparison_visualizer.py` - Main visualization system
- `../` - Other analysis modules

### Generated Outputs
- `../../../../results/splice_comparison_demo/` - Demo visualizations
- `../../../../results/test_splice_visualization/` - Test outputs

## System Requirements

- **Environment**: `conda activate surveyor`
- **Dependencies**: pandas, numpy, matplotlib, seaborn, polars
- **Data Requirements**: Training data and CV results

## Contributing

When adding new analysis documentation:

1. Create a dedicated subdirectory (e.g., `new_analysis_type/`)
2. Include comprehensive documentation and examples
3. Add test scripts and demo applications
4. Update this main README
5. Follow the established documentation patterns

## Status Overview

| Component | Status | Files | Description |
|-----------|--------|-------|-------------|
| Splice Site Visualization | ✅ Complete | 5 files | Publication-quality visualizations |
| Correlation Analysis | ✅ Complete | 1 file | Feature correlation analysis |

---

**Last Updated:** January 2025  
**Maintainer:** Splice Surveyor Development Team 