# Splice Site Visualization Scripts

This directory contains scripts for visualizing predicted splice sites and comparing predictions between base models (like SpliceAI) and meta-learning models.

## 🎯 Purpose

The main goal of these visualization tools is to demonstrate the improvements achieved by meta-learning approaches:

- **Rescued False Negatives (FNs)**: True splice sites that the base model missed but the meta model correctly identified
- **Eliminated False Positives (FPs)**: Incorrect predictions made by the base model that the meta model correctly rejected
- **Nucleotide Sequence Context**: Understanding the sequence patterns where these corrections occurred

## 📁 Directory Structure

```
scripts/analysis/splice_site_visualization/
├── README.md                                      # This file
├── run_splice_site_visualization_examples.sh     # Shell script with concrete gene examples
├── splice_site_visualization_demo.py             # Python demonstration script
├── splice_site_gene_discovery.py                 # Helper script to discover suitable genes
├── splice_site_enhanced_data_loader.py           # Enhanced data loader for specific genes
└── [future scripts and examples]
```

## 🔧 Available Scripts

### **1. Shell Script Examples (`run_splice_site_visualization_examples.sh`)**

**Purpose**: Provides concrete examples using real genes from training data

**Features**:
- Pre-selected genes with good splice site coverage
- Multiple analysis scenarios (single gene, multi-gene, different thresholds)
- Concrete gene IDs that are confirmed to exist in `train_pc_1000/master`
- Command-line examples ready to run

**Usage**:
```bash
# Make executable
chmod +x run_splice_site_visualization_examples.sh

# Run all examples
./run_splice_site_visualization_examples.sh

# Or run individual commands from the script
```

**Example genes included**:
- **MUC19** (ENSG00000205592): 170 donors, 173 acceptors
- **HSPG2** (ENSG00000142798): 102 donors, 104 acceptors  
- **DMD** (ENSG00000198947): 100 donors, 94 acceptors
- **HERC2** (ENSG00000128731): 93 donors, 95 acceptors

### **2. Python Demonstration Script (`splice_site_visualization_demo.py`)**

**Purpose**: Demonstrates programmatic usage of the visualization module

**Features**:
- Shows how to use the `SpliceSiteComparisonVisualizer` class
- Hierarchical data sampling for memory efficiency
- Automated gene discovery and filtering
- Both single-gene and multi-gene visualization examples

**Usage**:
```bash
# Run the demonstration
python splice_site_visualization_demo.py

# Or adapt the code for your own analysis
```

### **3. Gene Discovery Helper (`splice_site_gene_discovery.py`)**

**Purpose**: Discovers and ranks genes suitable for splice site visualization testing

**Features**:
- Analyzes splice site counts per gene
- Cross-references gene names with IDs
- Calculates FP/FN rates from CV results
- Generates ranked lists with suitability scores
- Supports both gene discovery and specific gene analysis

**Usage**:
```bash
# Discover top genes for testing
python splice_site_gene_discovery.py \
    --dataset train_pc_1000/master \
    --cv-results results/gene_cv_1000_run_15/position_level_classification_results.tsv \
    --output genes_for_splice_site_testing.tsv \
    --top-n 20

# Analyze specific target genes
python splice_site_gene_discovery.py \
    --dataset train_pc_1000/master \
    --target-genes ENSG00000205592,MUC19,HSPG2 \
    --output specific_genes_analysis.tsv
```

### **4. Enhanced Data Loader (`splice_site_enhanced_data_loader.py`)**

**Purpose**: Efficient data loading for specific genes without loading entire datasets

**Features**:
- Gene name to ID resolution
- Memory-efficient targeted gene loading
- Cross-referencing with metadata files
- Support for gene/transcript/splice site annotations
- Fallback loading strategies

**Usage**:
```python
from splice_site_enhanced_data_loader import SpliceSiteDataLoader

# Load specific genes by name or ID
loader = SpliceSiteDataLoader()
loader.load_metadata()
data = loader.load_genes(['MUC19', 'ENSG00000142798'], 'train_pc_1000/master')

# Or use convenience function
from splice_site_enhanced_data_loader import load_genes_for_visualization
data = load_genes_for_visualization(['MUC19', 'HSPG2'], 'train_pc_1000/master')
```

## 🔗 Related Module

### **Core Visualization Module**
**Location**: `meta_spliceai/splice_engine/meta_models/analysis/splice_site_comparison_visualizer.py`

**Key Features**:
- Publication-quality genomic browser-style plots
- Side-by-side comparison of base vs meta models
- Separate panels for donor (GT) and acceptor (AG) sites
- Highlighted rescued FNs and eliminated FPs
- Support for multiple output formats (PNG, PDF, SVG)

## 📊 Visualization Outputs

### **Single Gene Analysis**
- **Donor sites (5' GT)**: Base model vs Meta model predictions
- **Acceptor sites (3' AG)**: Base model vs Meta model predictions
- **True sites**: Observed splice sites from annotations
- **Improvements**: Highlighted rescued sites and eliminated false positives

### **Multi-Gene Comparison**
- Side-by-side comparison of multiple genes
- Consistent scaling and formatting
- Summary statistics for each gene

### **Output Files**
- `splice_comparison_{gene_id}.png`: Individual gene plots
- `multi_gene_splice_comparison.png`: Multi-gene comparison
- `meta_learning_improvements_summary.txt`: Detailed improvement report

### **Output Directory Examples**
- `results/splice_site_viz_single_gene_MUC19/`: Single gene analysis results
- `results/splice_site_viz_multi_gene_comparison/`: Multi-gene comparison results
- `results/splice_site_viz_threshold_analysis_HERC2/`: Threshold sensitivity analysis
- `results/splice_site_viz_demo/`: Python demonstration results

## 🚀 Quick Start

### **Option 1: Use Pre-configured Examples**
```bash
cd scripts/analysis/splice_site_visualization/
./run_splice_site_visualization_examples.sh
```

### **Option 2: Python Programmatic Usage**
```bash
cd scripts/analysis/splice_site_visualization/
python splice_site_visualization_demo.py
```

### **Option 3: Custom Analysis**
```bash
python -m meta_spliceai.splice_engine.meta_models.analysis.splice_site_comparison_visualizer \
    --dataset train_pc_1000/master \
    --genes ENSG00000205592,ENSG00000142798 \
    --cv-results results/gene_cv_1000_run_15/position_level_classification_results.tsv \
    --output-dir results/custom_analysis \
    --threshold 0.5
```

## 📋 Prerequisites

### **Required Data**
- **Training dataset**: `train_pc_1000/master` (or your dataset path)
- **CV results**: `results/gene_cv_1000_run_15/position_level_classification_results.tsv`
- **Gene features**: `data/ensembl/spliceai_analysis/gene_features.tsv`

### **Environment**
```bash
# Always activate the surveyor environment
conda activate surveyor
```

### **Dependencies**
All required packages should be available in the `surveyor` environment:
- pandas, numpy
- matplotlib, seaborn
- meta_spliceai modules

## 🔍 Understanding the Visualizations

### **Panel Layout**
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

### **Interpretation**
- **Height of lines**: Prediction probability/confidence
- **Green highlights**: Sites rescued by meta-learning
- **Missing predictions**: False positives eliminated
- **Consistency check**: True sites should be identical between panels

## 🛠️ Known Issues and Improvements Needed

### **Current Issues**
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

## 📈 Development Status

- ✅ **Core visualization**: Basic functionality implemented
- ✅ **Data loading**: Supports multiple input formats
- ✅ **Gene comparison**: Single and multi-gene analysis
- 🔄 **Layout improvements**: In progress
- 🔄 **Sequence context**: Planned
- 🔄 **Interactive features**: Future development

## 🔗 Integration with Other Tools

### **Related Scripts**
- **Training**: `meta_spliceai/splice_engine/meta_models/training/run_gene_cv_sigmoid.py`
- **Analysis**: `meta_spliceai/splice_engine/meta_models/analysis/create_comprehensive_feature_analysis.py`
- **Evaluation**: `scripts/evaluation/` tools

### **Workflow Integration**
1. **Train model**: Run gene-aware cross-validation
2. **Generate predictions**: Get position-level results
3. **Visualize**: Use these scripts to create publication-quality plots
4. **Analyze**: Examine sequence context and patterns

---

*This directory provides essential tools for visualizing and understanding the improvements achieved by meta-learning approaches in splice site prediction.* 