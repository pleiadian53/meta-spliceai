# Classical ML Approach for Error Analysis

## Overview

The classical machine learning approach in MetaSpliceAI's error analysis framework uses established ML algorithms, primarily XGBoost, to classify splice site prediction errors. This approach is particularly effective for:

1. Interpretable feature importance analysis
2. Handling both numerical and categorical features
3. Efficient training on moderately sized datasets
4. Producing reliable error classification models

## Key Components

### Error Classifier

The `ErrorClassifier` class provides a unified interface for training and evaluating error models using the classical ML approach. It supports:

- Multiple ML algorithms (XGBoost is the default)
- Cross-validation evaluation
- Feature importance analysis
- Hyperparameter tuning
- Model persistence and loading

### XGBoost Trainer

The XGBoost trainer module provides specialized functions for training XGBoost models optimized for error classification, including:

- Appropriate hyperparameter defaults for genomic data
- Early stopping to prevent overfitting
- Class weight balancing for imbalanced datasets
- Effective feature handling

## Feature Engineering

Feature engineering is critical for the classical ML approach. The framework supports several feature types:

### Sequence-Based Features
- k-mer frequencies
- Position-specific nucleotide encoding
- Sequence complexity measures
- Conservation scores
- Local sequence context

### Gene-Level Features
- Gene type (protein-coding, lncRNA, etc.)
- Gene length
- Exon count
- GC content
- Evolutionary conservation

### Transcript-Level Features
- Transcript type
- Exon position (first, internal, last)
- Distance to neighboring exons
- Splice site strength scores
- Reading frame position

## SHAP Analysis

SHAP (SHapley Additive exPlanations) analysis is a key technique used to interpret the trained models:

1. **Global Feature Importance**: Identifies which features most influence error classification across the entire dataset
2. **Local Feature Importance**: Examines specific instances to understand why particular splice sites were misclassified
3. **Feature Interaction**: Reveals how features interact to influence predictions
4. **Dependence Plots**: Shows how a feature's impact changes based on its value

The error model package includes specialized visualization functions for SHAP analysis that can:

- Generate summary plots of feature importance
- Create detailed waterfall plots for individual predictions
- Compare feature importance across different error types
- Identify patterns in feature contributions

## Sample Selection for Analysis

The package provides utilities for selecting informative samples for detailed analysis:

- **Random**: Standard random sampling
- **High Confidence**: Samples where the model is most confident
- **Low Confidence**: Samples where the model is least confident
- **Border**: Samples near the decision boundary
- **Misclassified**: Focuses on samples the model predicted incorrectly

This targeted sample selection helps identify the most informative examples for understanding model behavior and error patterns.

## Integration with Error Correction

The classical ML approach integrates with error correction strategies by:

1. Identifying the most discriminative features for different error types
2. Guiding the development of targeted corrections for specific error patterns
3. Providing interpretable insights that can be translated into correction rules
4. Supporting the meta-model training with feature importance knowledge

## Usage Example

```python
from meta_spliceai.splice_engine.error_model import ErrorClassifier
from meta_spliceai.splice_engine.error_model.utils import (
    select_samples_for_analysis,
    safely_save_figure
)
import matplotlib.pyplot as plt
import shap

# Initialize and train the classifier
classifier = ErrorClassifier(model_type='xgboost')
classifier.train(
    X_train=features_df,
    y_train=labels_series,
    feature_columns=feature_columns
)

# Evaluate performance
metrics = classifier.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

# Generate SHAP values
explainer = classifier.get_shap_explainer()
shap_values = explainer(X_test)

# Plot global feature importance
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar")
safely_save_figure("global_importance.pdf")

# Analyze misclassified samples
misclassified = select_samples_for_analysis(
    classifier.model, X_test, y_test, 
    n_samples=5, 
    selection="misclassified"
)
for i, idx in enumerate(misclassified):
    shap.plots.waterfall(shap_values[idx], max_display=10)
    safely_save_figure(f"misclassified_sample_{i}.pdf")
```
