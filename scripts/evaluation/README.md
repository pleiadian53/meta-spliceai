# Evaluation Scripts Directory

This directory contains organized scripts and tools for evaluating meta-learning models, analyzing performance, and generating comprehensive reports.

## Directory Structure

```
scripts/evaluation/
├── README.md                    # This file
├── overfitting/                 # Overfitting analysis and monitoring
├── cv_metrics/                  # Cross-validation metrics visualization
└── feature_importance/          # Feature importance analysis tools
```

## 📊 Cross-Validation Metrics (`cv_metrics/`)

Tools for visualizing and analyzing cross-validation performance metrics.

### Available Scripts:
- **`generate_cv_metrics_viz.py`** - Standalone CV metrics visualization for existing results
- **`demo_cv_metrics_viz.py`** - Demo script showing CV metrics visualization capabilities
- **`integrate_cv_metrics_viz.py`** - Integration helper for adding CV metrics to existing workflows

### Usage Examples:
```bash
# Generate visualization for existing CV results
python scripts/evaluation/cv_metrics/generate_cv_metrics_viz.py results/gene_cv_1000_run_15/gene_cv_metrics.csv

# Demo CV metrics visualization
python scripts/evaluation/cv_metrics/demo_cv_metrics_viz.py
```

### Generated Outputs:
- Performance comparison plots (base vs meta models)
- Error reduction analysis
- Top-k accuracy trends
- Statistical significance tests

## 🚨 Overfitting Analysis (`overfitting/`)

Comprehensive overfitting detection and monitoring system for meta-learning training.

### Available Scripts:
- **`integration_guide.md`** - Complete integration guide for overfitting monitoring
- **`enhanced_cv_example.py`** - Example implementation with overfitting monitoring
- **`test_overfitting_integration.py`** - Test script for overfitting integration

### Key Features:
- **Real-time overfitting detection** during CV training
- **Performance gap analysis** (training vs validation)
- **Convergence monitoring** and early stopping recommendations
- **Comprehensive visualizations** (learning curves, summary plots)

### Usage Examples:
```bash
# Run enhanced CV with overfitting monitoring
python scripts/evaluation/overfitting/enhanced_cv_example.py

# Test overfitting integration
python scripts/evaluation/overfitting/test_overfitting_integration.py
```

### Generated Outputs:
- `overfitting_analysis.json` - Comprehensive metrics
- `overfitting_summary.pdf` - Four-panel summary visualization
- `learning_curves_by_fold.pdf` - Individual model learning curves
- `aggregated_learning_curves.pdf` - Mean curves with confidence bands

## 🔍 Feature Importance Analysis (`feature_importance/`)

Multi-method feature importance analysis and SHAP visualizations.

### Available Scripts:
- **`demo_feature_importance.py`** - Demo of comprehensive feature importance analysis
- **`test_feature_importance_integration.py`** - Test script for feature importance integration
- **`integrate_feature_importance_cv.py`** - Integration helper for CV workflows
- **`generate_shap_visualizations.py`** - SHAP visualization generation

### Analysis Methods:
1. **XGBoost Feature Importance** (gain, weight, cover)
2. **Permutation Importance** with statistical significance
3. **SHAP Analysis** (TreeExplainer with incremental processing)
4. **Correlation Analysis** and data leakage detection

### Usage Examples:
```bash
# Demo feature importance analysis
python scripts/evaluation/feature_importance/demo_feature_importance.py

# Test integration with CV workflow
python scripts/evaluation/feature_importance/test_feature_importance_integration.py train_pc_1000/master

# Generate SHAP visualizations
python scripts/evaluation/feature_importance/generate_shap_visualizations.py --model-path results/gene_cv_1000_run_15/model_multiclass.pkl
```

### Generated Outputs:
- `gene_cv_comprehensive_results.xlsx` - Multi-method analysis results
- `integrated_summary.json` - Summary statistics
- SHAP importance plots and dependency plots
- Feature correlation heatmaps

## 🚀 Quick Start Guide

### 1. For New Users
Start with the demo scripts to understand capabilities:
```bash
# Try CV metrics visualization
python scripts/evaluation/cv_metrics/demo_cv_metrics_viz.py

# Try feature importance analysis
python scripts/evaluation/feature_importance/demo_feature_importance.py

# Try overfitting monitoring
python scripts/evaluation/overfitting/enhanced_cv_example.py
```

### 2. For Existing CV Results
Analyze your existing results:
```bash
# Visualize CV metrics
python scripts/evaluation/cv_metrics/generate_cv_metrics_viz.py path/to/gene_cv_metrics.csv

# Generate SHAP analysis
python scripts/evaluation/feature_importance/generate_shap_visualizations.py --model-path path/to/model.pkl
```

### 3. For New CV Workflows
Integrate evaluation tools into your training:
```bash
# Add overfitting monitoring to gene CV
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset data/train_pc_1000/master \
    --out-dir results/monitored_run \
    --monitor-overfitting \
    --overfitting-threshold 0.05

# The evaluation tools will be automatically integrated
```

## 📖 Documentation

### Comprehensive Guides:
- **[Overfitting Analysis Guide](../../meta_spliceai/splice_engine/meta_models/evaluation/docs/overfitting_analysis_guide.md)** - Complete overfitting analysis documentation
- **[Integration Guide](./overfitting/integration_guide.md)** - How to integrate overfitting monitoring
- **[SHAP Compatibility Guide](../../meta_spliceai/splice_engine/meta_models/evaluation/docs/SHAP_Compatibility_Resolution.md)** - SHAP analysis troubleshooting

### Key Concepts:
- **Performance Gap**: Difference between training and validation performance
- **Overfitting Score**: Composite metric measuring overfitting severity
- **Convergence Analysis**: When model improvement stops
- **Feature Importance**: Multi-method analysis of feature contributions

## 🔧 Integration with CV Scripts

All evaluation tools are designed to integrate seamlessly with existing CV scripts:

### Gene-Aware CV:
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --monitor-overfitting \
    --overfitting-threshold 0.05
```

### Chromosome-Aware CV:
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_loco_cv_multiclass_scalable \
    --monitor-overfitting \
    --overfitting-threshold 0.05
```

## 🛠️ Troubleshooting

### Common Issues:

1. **Missing Dependencies**
   ```bash
   pip install matplotlib seaborn shap openpyxl
   ```

2. **SHAP Compatibility Issues**
   - See [SHAP Compatibility Guide](../../meta_spliceai/splice_engine/meta_models/evaluation/docs/SHAP_Compatibility_Resolution.md)

3. **Memory Issues with Large Datasets**
   - Use `--memory-optimize` flag in CV scripts
   - Reduce sample sizes in analysis scripts

4. **Visualization Errors**
   - Check plot format compatibility (`pdf`, `png`, `svg`)
   - Verify output directory permissions

### Getting Help:
- Check individual script documentation (`python script.py --help`)
- Review integration guides in each subdirectory
- Check the main evaluation module documentation

## 📊 Output Organization

All evaluation tools follow consistent output organization:

```
results/your_experiment/
├── gene_cv_metrics.csv              # CV fold metrics
├── model_multiclass.pkl             # Trained model
├── cv_metrics_visualization/        # CV metrics plots
├── overfitting_analysis/           # Overfitting monitoring results
├── feature_importance_analysis/    # Multi-method feature analysis
└── leakage_analysis/              # Data leakage detection
```

This organization makes it easy to find and compare results across different experiments and methods.

## 🎯 Best Practices

1. **Always enable overfitting monitoring** for production runs
2. **Review visualizations** before finalizing models
3. **Check for data leakage** in feature importance analysis
4. **Compare results** across different CV strategies
5. **Document findings** for reproducibility

## 🔄 Maintenance

- **Update scripts** when new evaluation methods are added
- **Test integration** with new CV implementations
- **Review documentation** for accuracy and completeness
- **Monitor dependencies** for compatibility issues 

## 🎉 **Documentation Organization Complete!**

### **✅ What I've Accomplished:**

## 📁 **1. Organized Scattered Evaluation Scripts**

**Before**: Scripts scattered in main `scripts/` directory
**After**: Well-organized structure in `scripts/evaluation/`

### **Files Moved:**
- `demo_feature_importance.py` → `scripts/evaluation/feature_importance/`
- `test_feature_importance_integration.py` → `scripts/evaluation/feature_importance/`
- `integrate_feature_importance_cv.py` → `scripts/evaluation/feature_importance/`
- `generate_shap_visualizations.py` → `scripts/evaluation/feature_importance/`
- `generate_cv_metrics_viz.py` → `scripts/evaluation/cv_metrics/`

## 📖 **2. Created Comprehensive Documentation**

### **New Documentation Files:**

1. **`meta_spliceai/splice_engine/meta_models/evaluation/docs/overfitting_analysis_guide.md`**
   - Complete overfitting analysis documentation
   - Detailed metric explanations
   - Visualization interpretation guide
   - Best practices and troubleshooting

2. **`scripts/evaluation/README.md`**
   - Comprehensive evaluation tools guide
   - Quick start examples
   - Integration instructions
   - Troubleshooting section

### **Updated Documentation:**
- **`scripts/README.md`** - Added evaluation tools section
- **`meta_spliceai/splice_engine/meta_models/evaluation/docs/README.md`** - Added overfitting guide reference

## 📊 **3. Final Organization Structure:**

```
scripts/evaluation/
├── README.md                                    # Comprehensive evaluation guide
├── overfitting/                                # Overfitting analysis tools
│   ├── integration_guide.md                   # Integration documentation
│   ├── enhanced_cv_example.py                 # Example implementation
│   └── test_overfitting_integration.py        # Test script
├── cv_metrics/                                 # CV metrics visualization
│   ├── generate_cv_metrics_viz.py             # Standalone visualization
│   ├── demo_cv_metrics_viz.py                 # Demo script
│   └── integrate_cv_metrics_viz.py            # Integration helper
└── feature_importance/                         # Feature importance analysis
    ├── demo_feature_importance.py             # Demo script
    ├── test_feature_importance_integration.py # Test script
    ├── integrate_feature_importance_cv.py     # Integration helper
    └── generate_shap_visualizations.py        # SHAP visualization
```

## 🚀 **4. Benefits for Users:**

### **Easy Discovery:**
- All evaluation tools in one organized location
- Clear categorization by functionality
- Comprehensive README guides

### **Better Navigation:**
- Quick access links in main scripts README
- Cross-references between documentation
- Consistent naming and organization

### **Improved Usability:**
- Example commands for common tasks
- Integration instructions for CV workflows
- Troubleshooting guides for common issues

### **Professional Organization:**
- Follows best practices for project structure
- Separates concerns logically
- Makes maintenance easier

## 🎯 **5. Quick Access Commands:**

```bash
# View all evaluation tools
ls scripts/evaluation/

# Get comprehensive evaluation guide
cat scripts/evaluation/README.md

# Access overfitting analysis documentation
cat meta_spliceai/splice_engine/meta_models/evaluation/docs/overfitting_analysis_guide.md

# Try demo scripts
python scripts/evaluation/feature_importance/demo_feature_importance.py
python scripts/evaluation/cv_metrics/demo_cv_metrics_viz.py
python scripts/evaluation/overfitting/enhanced_cv_example.py
```

## 📈 **6. Documentation Cross-References:**

The documentation now forms a cohesive network:
- **Scripts README** → **Evaluation README** → **Specific tool docs**
- **Module docs** ← → **Scripts docs** (bidirectional references)
- **Integration guides** → **Technical documentation**

This organization makes it **much easier** for users to:
1. **Find the right tool** for their evaluation needs
2. **Understand how to use** the tools effectively
3. **Integrate tools** into their workflows
4. **Troubleshoot issues** when they arise
5. **Contribute new tools** following established patterns

The evaluation documentation is now **professional, comprehensive, and user-friendly**! 🎉 