# Analysis Scripts Directory

This directory contains organized scripts and tools for analyzing meta-learning models, visualizing predictions, and understanding model improvements.

## Directory Structure

```
scripts/analysis/
├── README.md                    # This file
└── splice_site_visualization/   # Splice site prediction visualization tools
```

## 🧬 Splice Site Visualization (`splice_site_visualization/`)

Tools for visualizing predicted splice sites and comparing predictions between base models (like SpliceAI) and meta-learning models.

### **Purpose**
Demonstrate the improvements achieved by meta-learning approaches:
- **Rescued False Negatives (FNs)**: True splice sites that the base model missed but the meta model correctly identified
- **Eliminated False Positives (FPs)**: Incorrect predictions made by the base model that the meta model correctly rejected
- **Nucleotide Sequence Context**: Understanding the sequence patterns where these corrections occurred

### **Available Scripts**
- **`run_splice_site_visualization_examples.sh`** - Shell script with concrete gene examples from training data
- **`splice_site_visualization_demo.py`** - Python demonstration script showing programmatic usage
- **Related Module**: `meta_spliceai/splice_engine/meta_models/analysis/splice_site_comparison_visualizer.py`

### **Features**
- Publication-quality genomic browser-style plots
- Side-by-side comparison of base vs meta models
- Separate panels for donor (GT) and acceptor (AG) sites
- Highlighted rescued FNs and eliminated FPs
- Support for single-gene and multi-gene analysis

### **Usage Examples**
```bash
# Quick start with pre-configured examples
cd scripts/analysis/splice_site_visualization/
./run_specific_gene_example.sh

# Python programmatic usage
python simple_usage_example.py

# Custom analysis
python -m meta_spliceai.splice_engine.meta_models.analysis.splice_site_comparison_visualizer \
    --dataset train_pc_1000/master \
    --genes ENSG00000205592,ENSG00000142798 \
    --cv-results results/gene_cv_1000_run_15/position_level_classification_results.tsv \
    --output-dir results/custom_analysis \
    --threshold 0.5
```

### **Output Files**
- `splice_comparison_{gene_id}.png`: Individual gene plots
- `multi_gene_splice_comparison.png`: Multi-gene comparison
- `meta_learning_improvements_summary.txt`: Detailed improvement report

### **Visualization Layout**
```
┌─────────────────────────────────────────────────────────────┐
│                    5' Splice Sites "GT"                    │
├─────────────────────┬───────────────────────────────────────┤
│   Base Model        │        Meta Model                     │
│   (predictions)     │        (predictions)                  │
├─────────────────────┼───────────────────────────────────────┤
│   Observed Sites    │        Observed Sites                 │
│   (ground truth)    │        (ground truth)                 │
├─────────────────────┴───────────────────────────────────────┤
│                    3' Splice Sites "AG"                    │
├─────────────────────┬───────────────────────────────────────┤
│   Base Model        │        Meta Model                     │
│   (predictions)     │        (predictions)                  │
├─────────────────────┼───────────────────────────────────────┤
│   Observed Sites    │        Observed Sites                 │
│   (ground truth)    │        (ground truth)                 │
└─────────────────────┴───────────────────────────────────────┘
```

### **Color Coding**
- **Red**: Donor sites (5' GT)
- **Blue**: Acceptor sites (3' AG)
- **Green**: Rescued sites (FN → TP)
- **Orange**: Eliminated false positives (FP → TN)
- **Dark gray**: True sites (ground truth)

## 📋 Prerequisites

### **Environment Setup**
```bash
# Always activate the surveyor environment first
conda activate surveyor
```

### **Required Data**
- **Training dataset**: `train_pc_1000/master` (or your dataset path)
- **CV results**: `results/gene_cv_1000_run_15/position_level_classification_results.tsv`
- **Gene features**: `data/ensembl/spliceai_analysis/gene_features.tsv`

### **Dependencies**
All required packages should be available in the `surveyor` environment:
- pandas, numpy
- matplotlib, seaborn
- meta_spliceai modules

## 🔗 Integration with Other Tools

### **Related Directories**
- **Training**: `meta_spliceai/splice_engine/meta_models/training/`
- **Core Analysis**: `meta_spliceai/splice_engine/meta_models/analysis/`
- **Evaluation**: `scripts/evaluation/`

### **Workflow Integration**
1. **Train model**: Run gene-aware cross-validation
2. **Generate predictions**: Get position-level results
3. **Visualize**: Use analysis scripts to create publication-quality plots
4. **Evaluate**: Use evaluation scripts for performance metrics

## 🚀 Quick Start Guide

### **Step 1: Prepare Data**
```bash
# Ensure you have the required data files
ls train_pc_1000/master/
ls results/gene_cv_1000_run_15/position_level_classification_results.tsv
ls data/ensembl/spliceai_analysis/gene_features.tsv
```

### **Step 2: Run Analysis**
```bash
# Activate environment
conda activate surveyor

# Navigate to analysis directory
cd scripts/analysis/splice_site_visualization/

# Run examples
./run_splice_site_visualization_examples.sh
```

### **Step 3: View Results**
```bash
# Check generated plots
ls results/*/splice_comparison_*.png
ls results/*/multi_gene_splice_comparison.png
```

## 🛠️ Development Status

### **Current Features**
- ✅ **Core visualization**: Basic functionality implemented
- ✅ **Data loading**: Supports multiple input formats
- ✅ **Gene comparison**: Single and multi-gene analysis
- ✅ **Organized structure**: Proper directory organization

### **Known Issues**
1. **Title positioning**: "3' splice site AG" text may overlap with plots
2. **Memory efficiency**: Large datasets may cause memory issues
3. **Sequence context**: Not yet implemented for understanding correction patterns
4. **Scalability**: Multi-gene plots may become cluttered

### **Planned Improvements**
1. **Better layout**: Fix title positioning and subplot spacing
2. **Sequence analysis**: Add nucleotide context visualization
3. **Interactive plots**: Consider plotly/bokeh for interactive exploration
4. **Performance optimization**: Better memory management for large datasets
5. **Statistical significance**: Add significance testing for improvements

## 📚 Documentation

### **Detailed Guides**
- **Splice Site Visualization**: `splice_site_visualization/README.md`
- **Core Module Documentation**: `meta_spliceai/splice_engine/meta_models/analysis/docs/`

### **Related Documentation**
- **Evaluation Tools**: `scripts/evaluation/README.md`
- **Training Scripts**: `meta_spliceai/splice_engine/meta_models/training/docs/`

---

*This directory provides essential tools for analyzing and visualizing the improvements achieved by meta-learning approaches in splice site prediction.* 