# Visualizing Error Patterns

## Overview

Visualization is a critical component of error analysis in the MetaSpliceAI project. Effective visualizations help researchers understand:

1. Which features contribute most to prediction errors
2. How different error types (FP/FN) relate to genomic contexts
3. Patterns in sequence motifs that lead to errors
4. Distribution of errors across genes and transcripts

This document outlines the key visualization techniques available in the error_model package.

## Feature Importance Visualization

### Global Feature Importance

```python
from meta_spliceai.splice_engine.error_model.utils import safely_save_figure
import matplotlib.pyplot as plt
import shap

# Using SHAP for global feature importance
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, plot_type="bar", max_display=20)
safely_save_figure("global_feature_importance.pdf")

# Alternative: Using built-in visualization functions
from meta_spliceai.splice_engine.error_model import plot_feature_importance

plot_feature_importance(
    model=trained_model,
    feature_names=feature_names,
    output_path="feature_importance_barplot.pdf",
    max_features=20,
    show_values=True
)
```

### Local Feature Importance

```python
from meta_spliceai.splice_engine.error_model.utils import (
    select_samples_for_analysis,
    get_sample_shap_files
)

# Select samples for detailed analysis
sample_indices = select_samples_for_analysis(
    model=model,
    X=X_test, 
    y=y_test,
    n_samples=5,
    sample_selection="misclassified"
)

# Generate waterfall plots
for i, idx in enumerate(sample_indices):
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], max_display=15)
    safely_save_figure(f"sample_{i}_waterfall.pdf")
```

## Sequence Pattern Visualization

### Sequence Motif Logos

```python
from meta_spliceai.splice_engine.error_model import plot_sequence_motifs

# Visualize sequence motifs associated with errors
plot_sequence_motifs(
    sequences=error_sequences,
    output_path="error_motifs.pdf",
    window_size=20,
    center_position=10
)
```

### Attention Heatmaps

For transformer models, visualize attention weights:

```python
from meta_spliceai.splice_engine.error_sequence_model import visualize_attention_heatmap

# Create attention heatmap for a sequence
visualize_attention_heatmap(
    model=transformer_model,
    tokenizer=tokenizer,
    sequence=test_sequence,
    layer_indices=[11, 12],  # Typically later layers are more task-specific
    output_path="attention_heatmap.pdf"
)
```

## Error Distribution Analysis

```python
from meta_spliceai.splice_engine.error_model import plot_error_distribution

# Visualize error distribution across different features
plot_error_distribution(
    data=labeled_data,
    feature_column="gene_type",
    error_type="FP",  # Can be "FP", "FN", or "all"
    normalize=True,
    output_path="fp_by_gene_type.pdf"
)
```

## Sample Selection Visualization

The package includes functionality to visualize different sample selection strategies:

```python
from meta_spliceai.splice_engine.error_model import visualize_sample_selection

# Compare different selection strategies
visualize_sample_selection(
    model=model,
    X=X_test,
    y=y_test,
    strategies=["random", "high_confidence", "border", "misclassified"],
    n_samples=10,
    feature_names=["feature1", "feature2"],  # For 2D projection
    output_path="sample_selection_comparison.pdf"
)
```

## Decision Boundary Visualization

For simplified feature spaces, you can visualize the decision boundary:

```python
from meta_spliceai.splice_engine.error_model import plot_decision_boundary

# Plot decision boundary for two selected features
plot_decision_boundary(
    model=model,
    X=X_test,
    y=y_test,
    feature1_idx=5,  # Index of first feature to plot
    feature2_idx=10, # Index of second feature to plot
    feature_names=feature_names,
    output_path="decision_boundary.pdf"
)
```

## Confusion Matrix and Performance Metrics

```python
from meta_spliceai.splice_engine.error_model import plot_confusion_matrix, plot_roc_curve

# Plot confusion matrix
plot_confusion_matrix(
    y_true=y_test,
    y_pred=y_pred,
    labels=["Correct", "Error"],
    normalize=True,
    output_path="confusion_matrix.pdf"
)

# Plot ROC curve
plot_roc_curve(
    model=model,
    X_test=X_test,
    y_test=y_test,
    output_path="roc_curve.pdf"
)
```

## Custom Visualization Examples

The error_model package supports custom visualizations through its utilities:

```python
from meta_spliceai.splice_engine.error_model.utils import safely_save_figure
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create custom visualization
plt.figure(figsize=(12, 8))

# Example: Compare feature importance across error types
fp_importance = pd.read_csv("fp_importance.csv")
fn_importance = pd.read_csv("fn_importance.csv")

merged_df = pd.merge(
    fp_importance.rename(columns={"importance": "FP_importance"}),
    fn_importance.rename(columns={"importance": "FN_importance"}),
    on="feature"
)

# Plot top 15 features by average importance
top_features = merged_df.assign(
    avg_importance=(merged_df["FP_importance"] + merged_df["FN_importance"]) / 2
).nlargest(15, "avg_importance")

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=top_features, 
    x="FP_importance", 
    y="FN_importance",
    s=100
)

for i, row in top_features.iterrows():
    plt.annotate(
        row["feature"], 
        (row["FP_importance"], row["FN_importance"]),
        xytext=(5, 5),
        textcoords='offset points'
    )

plt.title("Feature Importance Comparison: FP vs FN")
plt.xlabel("Importance for False Positives")
plt.ylabel("Importance for False Negatives")
plt.plot([0, max(top_features["FP_importance"])], 
         [0, max(top_features["FP_importance"])], 
         'k--', alpha=0.3)

safely_save_figure("fp_fn_importance_comparison.pdf")
```

## Best Practices

1. **Consistency**: Use consistent color schemes across visualizations
2. **Accessibility**: Ensure visualizations are readable (font sizes, color choices)
3. **Saving**: Always use the `safely_save_figure` utility to prevent memory leaks
4. **Documentation**: Include clear titles, labels, and legends
5. **Memory Management**: Close figures after saving to prevent memory leaks
