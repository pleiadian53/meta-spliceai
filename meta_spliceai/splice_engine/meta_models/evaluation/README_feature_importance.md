# Feature Importance Analysis Module

This module provides comprehensive feature importance analysis using multiple statistical and machine learning methods, with publication-ready visualizations.

## Overview

The feature importance analysis system includes:

1. **Multiple Analysis Methods**:
   - XGBoost internal feature importance (multiple types)
   - Statistical hypothesis testing with FDR correction
   - Effect size measurement (Cohen's d, Cramer's V)
   - Mutual information analysis
   - Integration with SHAP analysis

2. **Publication-Ready Visualizations**:
   - High-quality plots with customizable styling
   - Method comparison and overlap analysis
   - Unified feature ranking across methods
   - Comprehensive summary reports

3. **Scalable Implementation**:
   - Handles large datasets efficiently
   - Incremental SHAP analysis for memory management
   - Batch processing capabilities
   - Integration with existing gene-wise CV workflow

## Quick Start

### Basic Usage

```python
from meta_spliceai.splice_engine.meta_models.evaluation.feature_importance import analyze_feature_importance
import pandas as pd
import joblib

# Load your data and model
X = pd.read_csv("your_features.csv")
y = pd.read_csv("your_target.csv")
model = joblib.load("your_model.pkl")

# Run comprehensive analysis
results = analyze_feature_importance(
    model=model,
    X=X,
    y=y,
    output_dir="feature_analysis_results",
    subject="my_analysis",
    top_k=25,
    methods=['xgboost', 'hypothesis_testing', 'effect_sizes', 'mutual_info'],
    verbose=1
)
```

### Integration with Gene-wise CV Workflow

```python
from meta_spliceai.splice_engine.meta_models.evaluation.feature_importance_integration import integrate_with_gene_cv_workflow

# Analyze results from gene-wise CV
results = integrate_with_gene_cv_workflow(
    cv_results_dir="path/to/cv/results",
    output_dir="feature_importance_analysis",
    subject="gene_cv_analysis",
    top_k=25,
    include_shap=True,
    verbose=1
)
```

### Command Line Usage

```bash
# Run analysis from command line
python meta_spliceai/splice_engine/meta_models/evaluation/feature_importance_integration.py \
    /path/to/cv/results \
    --output-dir /path/to/output \
    --subject my_analysis \
    --top-k 25 \
    --methods xgboost hypothesis_testing effect_sizes \
    --verbose
```

## Analysis Methods

### 1. XGBoost Feature Importance

Analyzes multiple types of XGBoost internal importance:
- **Weight**: Number of times feature is used for splits
- **Gain**: Average gain when feature is used
- **Cover**: Average coverage of feature
- **Total Gain**: Total gain across all trees
- **Total Cover**: Total coverage across all trees

### 2. Statistical Hypothesis Testing

Performs appropriate statistical tests based on feature type:
- **Numerical features**: t-test (if normal) or Mann-Whitney U test
- **Categorical features**: Chi-square test or Fisher's exact test
- **FDR correction**: Benjamini-Hochberg procedure for multiple testing
- **Effect size guidelines**: Small (0.2), Medium (0.5), Large (0.8)

### 3. Effect Size Measurement

Quantifies practical significance:
- **Cohen's d**: For numerical features (standardized mean difference)
- **Cramer's V**: For categorical features (association strength)
- **Rank-biserial correlation**: For ordinal features

### 4. Mutual Information

Measures information-theoretic dependence:
- **Works for both**: Numerical and categorical features
- **Non-linear relationships**: Captures complex dependencies
- **Unified scale**: Comparable across feature types

## Output Files

### Generated Plots

1. **XGBoost Importance Plots**:
   - `{subject}_xgboost_importance_{type}.png`
   - Separate plots for each importance type
   - Color-coded bars with value labels

2. **Hypothesis Testing Results**:
   - `{subject}_hypothesis_testing_results.png`
   - Dual plots: significance levels and test statistics
   - FDR-corrected p-values with significance threshold

3. **Effect Size Analysis**:
   - `{subject}_effect_sizes.png`
   - Directional effects and magnitude comparison
   - Effect size guidelines overlaid

4. **Mutual Information**:
   - `{subject}_mutual_information.png`
   - Color-coded by feature type
   - Information content visualization

5. **Method Comparison**:
   - `{subject}_method_overlap.png`: Overlap heatmap
   - `{subject}_ranking_comparison.png`: Side-by-side rankings

### Data Files

1. **Comprehensive Results**: `{subject}_comprehensive_results.xlsx`
   - Separate sheets for each method
   - Full feature rankings and statistics

2. **Integrated Summary**: `{subject}_integrated_summary.json`
   - Top features by method
   - Consensus features across methods
   - Method agreement metrics
   - Unified feature ranking

## Advanced Usage

### Custom Feature Categories

```python
feature_categories = {
    'numerical_features': ['score', 'num_exons', 'transcript_length'],
    'categorical_features': ['gene_type', 'strand'],
    'motif_features': ['3mer_GT', '4mer_AG', '5mer_GTA'],
    'derived_categorical_features': ['chrom_type']
}

results = analyze_feature_importance(
    model=model,
    X=X,
    y=y,
    feature_categories=feature_categories,
    # ... other parameters
)
```

### Selective Method Analysis

```python
# Run only specific methods
results = analyze_feature_importance(
    model=model,
    X=X,
    y=y,
    methods=['hypothesis_testing', 'effect_sizes'],  # Skip XGBoost and mutual info
    verbose=1
)
```

### Large Dataset Handling

```python
# For large datasets, use integration module with SHAP subsampling
results = run_comprehensive_feature_importance_analysis(
    model_path="model.pkl",
    data_path="large_dataset.csv",
    output_dir="results",
    shap_max_samples=5000,  # Limit SHAP analysis
    include_shap=True,
    verbose=1
)
```

## Class-Based API

### FeatureImportanceAnalyzer

```python
from meta_spliceai.splice_engine.meta_models.evaluation.feature_importance import FeatureImportanceAnalyzer

# Initialize analyzer
analyzer = FeatureImportanceAnalyzer(
    output_dir="my_analysis",
    subject="splice_sites"
)

# Run individual analyses
xgb_results = analyzer.analyze_xgboost_importance(model, X, top_k=20)
stat_results = analyzer.analyze_hypothesis_testing(X, y, top_k=20)
effect_results = analyzer.analyze_effect_sizes(X, y, top_k=20)
mi_results = analyzer.analyze_mutual_information(X, y, top_k=20)

# Create comparison plots
analyzer.create_comparison_plots(
    {'xgboost': xgb_results, 'stats': stat_results}, 
    top_k=20
)

# Save all results
analyzer.save_results("comprehensive_analysis.xlsx")
```

## Integration Examples

### With Existing Gene-wise CV

```python
# After running gene-wise CV
cv_output_dir = "gene_cv_results"

# Add feature importance analysis
feature_results = integrate_with_gene_cv_workflow(
    cv_results_dir=cv_output_dir,
    subject="my_genes_analysis",
    top_k=30,
    methods=['xgboost', 'hypothesis_testing', 'effect_sizes'],
    include_shap=True
)

# Access different results
standard_results = feature_results['standard_results']
shap_results = feature_results['shap_results']
summary = feature_results['summary']

# Get consensus features
consensus_features = summary['consensus_features']
print(f"Top consensus features: {consensus_features[:10]}")
```

### Custom Workflow Integration

```python
# Custom integration in your existing pipeline
def my_analysis_pipeline(data_path, model_path):
    # Your existing analysis...
    
    # Add feature importance
    feature_results = run_comprehensive_feature_importance_analysis(
        model_path=model_path,
        data_path=data_path,
        output_dir="feature_analysis",
        methods=['all'],  # Run all methods
        top_k=25
    )
    
    # Extract key findings
    important_features = feature_results['summary']['consensus_features']
    
    return important_features
```

## Interpreting Results

### Feature Rankings

- **Lower rank = more important** (rank 1 is most important)
- **Consensus features**: Appear in multiple methods' top lists
- **Method agreement**: Jaccard similarity between method rankings
- **Average rank**: Weighted average across methods where feature appears

### Statistical Significance

- **FDR-corrected p-values**: Control false discovery rate
- **Effect sizes**: Practical significance beyond statistical significance
- **Test selection**: Automatic based on feature type and distribution

### Visualization Guide

1. **Bar plots**: Direct comparison of importance scores
2. **Overlap heatmaps**: Method agreement visualization
3. **Dual plots**: Statistical significance + effect magnitude
4. **Color coding**: Feature types, significance levels, effect directions

## Troubleshooting

### Common Issues

1. **Memory errors with SHAP**:
   ```python
   # Reduce sample size
   results = analyze_feature_importance(..., shap_max_samples=1000)
   ```

2. **Missing XGBoost model methods**:
   ```python
   # Ensure model is XGBoost type
   import xgboost as xgb
   assert isinstance(model, xgb.XGBClassifier)
   ```

3. **Feature type classification**:
   ```python
   # Provide explicit feature categories
   feature_categories = {...}  # Define explicitly
   ```

### Performance Tips

1. **Large datasets**: Use integration module with sampling
2. **Many features**: Focus on top-k from each method first
3. **Multiple models**: Run analysis separately and compare
4. **Memory management**: Process in batches for very large datasets

## Dependencies

Required packages:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- statsmodels
- xgboost (for XGBoost analysis)
- shap (for SHAP integration)
- openpyxl (for Excel output)

Install with:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels xgboost shap openpyxl
```

## Examples and Demos

Run the demo script to see the module in action:
```bash
python scripts/demo_feature_importance.py
```

This will:
1. Generate sample data with realistic features
2. Train a model and run all analysis methods
3. Create publication-ready plots
4. Save comprehensive results
5. Demonstrate integration capabilities

The demo creates realistic splice site data with:
- Numerical features (scores, counts, lengths)
- Categorical features (gene types, chromosomes)
- K-mer motif features
- Correlated target variable

## Contributing

When adding new analysis methods:

1. **Follow the pattern**: Implement `analyze_{method_name}` function
2. **Create visualization**: Add corresponding `_plot_{method_name}` function
3. **Update integration**: Add to method list and comparison functions
4. **Document thoroughly**: Update this README and add docstrings
5. **Test extensively**: Ensure compatibility with existing workflow

## Citation

If you use this feature importance analysis module in your research, please cite:

```bibtex
@software{meta_spliceai_feature_importance,
  title={Comprehensive Feature Importance Analysis for Splice Site Classification},
  author={Splice Surveyor Development Team},
  year={2024},
  url={https://github.com/your-repo/meta-spliceai}
}
``` 