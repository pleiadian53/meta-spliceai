# Post-Training Analysis Scripts

This directory contains **follow-up analysis scripts** that can be run after completing CV training runs to perform additional diagnostics, comparisons, and insights.

## 🎯 **Purpose**

These scripts are designed to be run **after** the main training scripts (like `run_gene_cv_sigmoid.py` or `run_loco_cv_multiclass_scalable.py`) to:

- **Compare multiple CV runs** for reproducibility assessment
- **Analyze performance trends** across different experiments
- **Generate comprehensive reports** for publication or documentation
- **Validate model stability** and consistency
- **Provide insights** for model improvement

## 📋 **Available Scripts**

### 1. **CV Run Comparison** (`compare_cv_runs.py`)

**Purpose**: Compare results from multiple CV runs to assess reproducibility and identify trends.

**Usage**:
```bash
# Compare two CV runs
python -m meta_spliceai.splice_engine.meta_models.analysis.compare_cv_runs \
    --run1 results/gene_cv_pc_1000_3mers_run_2_more_genes \
    --run2 results/gene_cv_pc_1000_3mers_run_3 \
    --output cv_comparison_results
```

**Features**:
- ✅ **Statistical significance testing** (t-tests)
- ✅ **Performance trend analysis**
- ✅ **Reproducibility assessment** (similarity scoring)
- ✅ **Detailed metric comparisons** (15+ metrics)
- ✅ **HTML report generation** with visualizations
- ✅ **JSON data export** for further analysis
- ✅ **Visualization plots** (bar charts, pie charts, etc.)

**Outputs**:
- `cv_comparison_report.html` - Comprehensive HTML report
- `cv_comparison_visualization.png` - Summary plots
- `comparison_results.json` - Raw data for further analysis

**Example Use Cases**:
- Compare different random seeds
- Validate model reproducibility
- Assess parameter sensitivity
- Document experimental results

## 🔄 **Workflow Integration**

### **Typical Workflow**:

1. **Run CV Training**:
   ```bash
   python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
       --dataset data/train_pc_1000_3mers \
       --out-dir results/gene_cv_run_1
   ```

2. **Run Additional CV** (if needed):
   ```bash
   python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
       --dataset data/train_pc_1000_3mers \
       --out-dir results/gene_cv_run_2
   ```

3. **Compare Results**:
   ```bash
   python -m meta_spliceai.splice_engine.meta_models.analysis.compare_cv_runs \
       --run1 results/gene_cv_run_1 \
       --run2 results/gene_cv_run_2 \
       --output analysis/cv_comparison
   ```

### **When to Use**:

- ✅ **After completing CV training** to validate results
- ✅ **Before publishing** to ensure reproducibility
- ✅ **During model development** to assess stability
- ✅ **For parameter tuning** to compare different configurations
- ✅ **For documentation** to generate comprehensive reports

## 📊 **Analysis Capabilities**

### **Statistical Analysis**:
- **T-tests** for significance testing
- **Confidence intervals** for metric differences
- **Effect size calculations**
- **Reproducibility scoring**

### **Visualization**:
- **Bar charts** for metric comparison
- **Percent change analysis**
- **Statistical significance distribution**
- **Similarity score visualization**

### **Reporting**:
- **HTML reports** with embedded visualizations
- **Executive summaries** with key insights
- **Detailed metric tables**
- **Recommendations** based on results

## 🎯 **Best Practices**

### **1. Consistent Naming**:
```bash
# Use descriptive run names
results/gene_cv_pc_1000_3mers_run_1_seed42
results/gene_cv_pc_1000_3mers_run_2_seed123
```

### **2. Organized Output**:
```bash
# Create dedicated analysis directories
analysis/cv_comparison_seed_variation/
analysis/cv_comparison_parameter_tuning/
analysis/cv_comparison_final_validation/
```

### **3. Documentation**:
```bash
# Document your comparisons
echo "Comparing seed 42 vs seed 123 for reproducibility validation" > analysis/README.md
```

## 🔧 **Extending the Framework**

### **Adding New Analysis Scripts**:

1. **Create new script** in this directory
2. **Follow naming convention**: `analyze_*.py` or `compare_*.py`
3. **Add to this README** with usage examples
4. **Include proper documentation** and help text

### **Example Template**:
```python
#!/usr/bin/env python3
"""
[Script Name] - Post-Training Analysis Tool

Purpose: [Brief description]

Usage:
    python -m meta_spliceai.splice_engine.meta_models.analysis.[script_name] \
        --input [input_path] \
        --output [output_path]

Features:
    - [Feature 1]
    - [Feature 2]
    - [Feature 3]
"""

# Implementation here...
```

## 📈 **Future Enhancements**

### **Planned Features**:
- **Multi-run comparison** (3+ runs)
- **Trend analysis** across multiple experiments
- **Automated report generation** for CI/CD
- **Integration with experiment tracking** (MLflow, Weights & Biases)
- **Performance regression detection**
- **Automated alerting** for significant changes

### **Integration Opportunities**:
- **Jupyter notebooks** for interactive analysis
- **Dashboard generation** (Streamlit, Plotly)
- **Email reporting** for automated notifications
- **Database storage** for historical tracking

## 🎉 **Benefits**

### **For Researchers**:
- **Validate experimental results**
- **Ensure reproducibility**
- **Generate publication-ready reports**
- **Track model improvements**

### **For Developers**:
- **Monitor model stability**
- **Detect performance regressions**
- **Automate quality checks**
- **Streamline documentation**

### **For Teams**:
- **Standardize analysis workflows**
- **Share results easily**
- **Maintain quality standards**
- **Accelerate model development**

---

**Remember**: These scripts are designed to be **easy to use** and **comprehensive in analysis**. They should help you gain deeper insights into your model performance and ensure the quality of your results! 🚀 