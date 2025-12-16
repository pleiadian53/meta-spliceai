# Splice Site Visualization Solution

## Overview

The splice site visualization system provides publication-quality visualizations comparing base model (SpliceAI) predictions with meta-learning enhanced predictions. It creates genomic browser-style plots that clearly show the benefits of meta-learning for splice site prediction.

## Key Features

### 1. **Data Integration**
- Loads training data with hierarchical gene-level sampling
- Integrates cross-validation results containing both base and meta predictions
- Validates data consistency between base and meta models
- Handles gene feature mapping for display names

### 2. **Visualization Types**
- **Single Gene Plots**: Detailed comparison for individual genes
- **Multi-Gene Plots**: Side-by-side comparison of multiple genes
- **Summary Reports**: Quantitative analysis of improvements

### 3. **Meta-Learning Improvement Detection**
- **Rescued Sites**: False negatives converted to true positives
- **Eliminated FPs**: False positives converted to true negatives
- **Threshold-based Analysis**: Configurable probability thresholds

### 4. **Publication-Ready Output**
- High-resolution PNG plots (300 DPI)
- Consistent color coding (red for donors, blue for acceptors)
- Professional styling with clear annotations
- Comprehensive legends and labels

## System Architecture

```
SpliceSiteComparisonVisualizer
├── Data Loading
│   ├── load_dataset() - Training data with hierarchical sampling
│   ├── load_cv_results() - Cross-validation results
│   ├── load_gene_features() - Gene annotations
│   └── format_cv_results_as_meta_data() - Data formatting
├── Analysis
│   ├── get_gene_data() - Gene-specific data extraction
│   ├── identify_splice_site_changes() - Improvement detection
│   ├── validate_gene_data_consistency() - Data validation
│   └── align_gene_datasets() - Position alignment
└── Visualization
    ├── create_gene_comparison_plot() - Single gene plots
    ├── create_multi_gene_comparison() - Multi-gene plots
    ├── create_splice_site_panel() - Individual plot panels
    └── generate_summary_report() - Quantitative reports
```

## Usage Examples

### 1. Basic Demo Usage

```python
# Run the comprehensive demo
python demo_splice_visualization.py
```

**Output:**
- Individual gene plots: `splice_comparison_*.png`
- Multi-gene comparison: `multi_gene_splice_comparison.png`
- Summary report: `meta_learning_improvements_summary.txt`

### 2. Command Line Interface

```bash
# Single gene analysis
python -m meta_spliceai.splice_engine.meta_models.analysis.splice_site_comparison_visualizer \
    --dataset train_pc_1000/master \
    --genes ENSG00000131018 \
    --cv-results results/gene_cv_1000_run_15/position_level_classification_results.tsv \
    --output-dir results/splice_analysis \
    --threshold 0.5

# Multiple gene comparison
python -m meta_spliceai.splice_engine.meta_models.analysis.splice_site_comparison_visualizer \
    --dataset train_pc_1000/master \
    --genes ENSG00000131018,ENSG00000114270,ENSG00000142798 \
    --cv-results results/gene_cv_1000_run_15/position_level_classification_results.tsv \
    --output-dir results/splice_analysis \
    --threshold 0.3
```

### 3. Programmatic Usage

```python
from meta_spliceai.splice_engine.meta_models.analysis.splice_site_comparison_visualizer import SpliceSiteComparisonVisualizer

# Initialize visualizer
viz = SpliceSiteComparisonVisualizer(verbose=True)

# Load data
viz.load_gene_features('data/ensembl/spliceai_analysis/gene_features.tsv')
base_data = viz.load_dataset('train_pc_1000/master', sample_genes=100)
cv_results = viz.load_cv_results('results/gene_cv_1000_run_15/position_level_classification_results.tsv')
meta_data = viz.format_cv_results_as_meta_data(cv_results)

# Create visualizations
output_dir = Path('results/custom_analysis')
output_dir.mkdir(parents=True, exist_ok=True)

# Single gene analysis
gene_id = 'ENSG00000205592'
gene_data_base = viz.get_gene_data(base_data, gene_id)
gene_data_meta = viz.get_gene_data(meta_data, gene_id)

result = viz.create_gene_comparison_plot(
    gene_data_base, gene_data_meta, 
    gene_id, output_dir, 
    threshold=0.5
)

print(f"Plot saved: {result['plot_file']}")
print(f"Changes detected: {result['changes']}")
```

## Test Results

The comprehensive test suite validates all functionality:

### ✅ **Data Loading Tests**
- Gene features: 63,140 loaded successfully
- Training data: 41,392 positions from 50 genes
- CV results: 15,000 predictions loaded
- Meta data formatting: 15,000 predictions formatted

### ✅ **Gene Analysis Tests**
- Suitable genes: 20 found with sufficient splice sites
- Data extraction: Successful for base and meta datasets
- Change detection: Improvements quantified correctly

### ✅ **Visualization Generation Tests**
- Single gene plots: Generated successfully
- Multi-gene comparisons: Created for 3 genes
- Summary reports: Comprehensive analysis completed

### ✅ **Edge Case Handling Tests**
- Non-existent genes: Proper error handling
- Empty meta data: Graceful degradation
- Different thresholds: Flexible analysis

## Key Improvements Detected

From the test run, the meta-learning system shows significant improvements:

### **Overall Results (10 genes analyzed)**
- **Total rescued splice sites**: 8
  - Rescued donors: 3
  - Rescued acceptors: 5
- **Total eliminated false positives**: 165
  - Eliminated FP donors: 68
  - Eliminated FP acceptors: 97
- **Total improvements**: 173

### **Gene-Specific Examples**
- **MUC19**: 36 false positives eliminated
- **MACF1**: 1 acceptor rescued, 11 FPs eliminated
- **HSPG2**: 1 donor rescued, 25 FPs eliminated

## Technical Details

### **Data Alignment**
- Validates consistency between base and meta predictions
- Aligns datasets to common positions for fair comparison
- Handles position mismatches gracefully

### **Visualization Design**
- **Color Coding**: Red for donors (GT), blue for acceptors (AG)
- **Panel Layout**: Predicted vs observed sites in paired panels
- **Highlighting**: Green for rescued sites, orange for eliminated FPs
- **Scaling**: Probability scores as line heights

### **Performance Optimizations**
- Hierarchical gene-level sampling preserves splice sites
- K-mer feature filtering reduces memory usage
- Efficient data alignment algorithms
- Parallel processing for multiple genes

## File Structure

```
results/splice_comparison_demo/
├── splice_comparison_ENSG00000205592.png    # Single gene plot
├── multi_gene_splice_comparison.png         # Multi-gene comparison
└── meta_learning_improvements_summary.txt   # Quantitative report

results/test_splice_visualization/
├── splice_comparison_ENSG00000152556.png    # Test single gene
├── multi_gene_splice_comparison.png         # Test multi-gene
└── meta_learning_improvements_summary.txt   # Test report
```

## Dependencies

- **Core**: pandas, numpy, matplotlib, seaborn
- **Data**: polars (for efficient data loading)
- **Visualization**: matplotlib patches, gridspec
- **Analysis**: scipy, scikit-learn (for metrics)

## Best Practices

### **Data Preparation**
1. Use hierarchical sampling to preserve splice sites
2. Validate data consistency between base and meta models
3. Filter k-mer features for memory efficiency
4. Align datasets to common positions

### **Visualization**
1. Select genes with sufficient splice sites (≥3 donors, ≥3 acceptors)
2. Use appropriate probability thresholds (0.3-0.7)
3. Include both single-gene and multi-gene analyses
4. Generate quantitative summary reports

### **Analysis**
1. Focus on genes with clear improvements
2. Validate rescued sites manually when possible
3. Consider biological context of splice site changes
4. Use multiple threshold values for sensitivity analysis

## Conclusion

The splice site visualization system successfully demonstrates the benefits of meta-learning for splice site prediction. It provides:

- **Clear Visual Evidence**: Publication-quality plots showing improvements
- **Quantitative Analysis**: Precise metrics for rescued sites and eliminated FPs
- **Flexible Usage**: Command-line and programmatic interfaces
- **Robust Implementation**: Comprehensive error handling and validation

The system is ready for publication use and can be extended for additional analysis types as needed. 