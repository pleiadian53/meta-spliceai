# Error Model Training

This module provides components for training models that analyze splice site prediction errors in the MetaSpliceAI project.

## Overview

Error analysis is a critical stage in the MetaSpliceAI workflow that aims to identify and understand patterns in splice site prediction errors. This training module provides the machine learning components needed to build models that can classify and analyze these errors.

## Key Components

### XGBoost Trainer

The XGBoost trainer provides functionality for training gradient boosting models to classify splice site prediction errors. It includes:

- Model training with appropriate hyperparameters
- Cross-validation evaluation
- Feature importance analysis
- Model persistence and loading

### Error Classifier

The error classifier provides a high-level interface for training and evaluating error models, with features such as:

- Multiple model type support
- Unified data preparation
- Consistent feature handling
- Comprehensive evaluation metrics

### Visualization Utilities

The visualization module offers tools for interpreting and visualizing model results, including:

- Feature importance plots
- SHAP analysis visualizations
- Performance metric visualization
- Custom plot utilities

## Usage

For typical use cases, import the high-level functions from the package:

```python
from meta_spliceai.splice_engine.error_model.training import (
    train_xgboost_model,
    evaluate_xgboost_model,
    xgboost_pipeline
)

# Train a model
model, train_metrics = train_xgboost_model(
    X_train, y_train, 
    feature_names=feature_columns
)

# Evaluate the model
eval_metrics = evaluate_xgboost_model(
    model, X_test, y_test
)

# Or use the complete pipeline
results = xgboost_pipeline(
    X, y, 
    feature_names=feature_columns,
    test_size=0.2,
    random_state=42
)
```

For more comprehensive examples, refer to the documentation in the `docs` directory.

## Integration

This module is designed to work seamlessly with the broader error model package. Typically, you'll use these components through the higher-level interfaces provided by the error_model package:

```python
from meta_spliceai.splice_engine.error_model import process_error_model

results = process_error_model(
    data=data,
    output_dir=output_dir,
    feature_columns=feature_columns,
    target_column="is_error"
)
```
