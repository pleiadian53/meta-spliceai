# Error Analysis Workflow

This document describes the end-to-end workflow for error analysis in the MetaSpliceAI project.

## Prerequisites

Before starting the error analysis workflow, ensure you have:

1. Completed base model predictions (typically using SpliceAI)
2. Generated labeled splice sites with TP, FP, FN, and TN designations
3. Extracted genomic sequences for these sites
4. Prepared feature data including gene and transcript annotations

## Workflow Steps

### 1. Data Preparation

```python
from meta_spliceai.splice_engine.error_model import load_and_subsample_dataset

# Load dataset with appropriate subsampling
data = load_and_subsample_dataset(
    input_file="/path/to/labeled_sites.tsv",
    sample_size=100000,  # Adjust based on available resources
    stratify=True,       # Maintain class distribution
    random_state=42      # For reproducibility
)
```

### 2. Model Training

#### Classical ML Model (XGBoost)

```python
from meta_spliceai.splice_engine.error_model import process_error_model

# Train and analyze an error classification model
results = process_error_model(
    data=data,
    output_dir="/path/to/output",
    feature_columns=["seq_features", "gene_features", "transcript_features"],
    model_type="xgboost",
    target_column="is_error",  # 1 for errors (FP/FN), 0 for correct predictions (TP/TN)
    generate_shap=True,        # Generate SHAP explanations
    cv_folds=5                 # Cross-validation folds
)
```

#### Transformer Model

```python
from meta_spliceai.splice_engine.error_sequence_model import train_error_sequence_model

# Fine-tune a transformer model for error classification
model = train_error_sequence_model(
    train_data=train_data,
    val_data=val_data,
    model_name="DNABERT-2-117M",  # Base model
    output_dir="/path/to/transformer_output",
    batch_size=32,
    learning_rate=2e-5,
    epochs=5
)
```

### 3. Model Evaluation

```python
from meta_spliceai.splice_engine.error_model import verify_error_model_outputs, get_model_status

# Verify all expected outputs were generated
verification = verify_error_model_outputs("/path/to/output")
print(f"Model output verification: {verification}")

# Check model status and performance
status = get_model_status("/path/to/output")
print(f"Model accuracy: {status['accuracy']}")
print(f"Model F1 score: {status['f1']}")
```

### 4. Feature Importance Analysis

```python
import matplotlib.pyplot as plt
from meta_spliceai.splice_engine.error_model.utils import safely_save_figure

# Load feature importance from model outputs
importance_df = pd.read_csv("/path/to/output/feature_importance.csv")

# Plot top features
plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df.head(20), x='importance', y='feature')
plt.title("Top 20 Features for Error Classification")
safely_save_figure("/path/to/output/custom_importance.pdf")
```

### 5. Advanced Analysis

For specific error types, you can run dedicated analyses:

```python
# Filter data for false positives only
fp_data = data[data['prediction_type'] == 'FP']

# Process FP-specific error model
fp_results = process_error_model(
    data=fp_data,
    output_dir="/path/to/fp_output",
    feature_columns=["seq_features", "gene_features", "transcript_features"],
    model_type="xgboost",
    target_column="is_fp_error",
    generate_shap=True
)
```

## Integration with Meta-Model Stage

The error analysis insights are used to:

1. Develop targeted meta-models for specific error types
2. Improve feature extraction for the meta-model stage
3. Guide architectural decisions in meta-model development

## Common Issues and Solutions

- **Class Imbalance**: Use `apply_stratified_sampling` to maintain proper class distributions
- **Memory Usage**: For large datasets, enable the `memory_efficient` option in `process_error_model`
- **Feature Interpretation**: Use the sample selection utilities to analyze specific types of errors:
  ```python
  from meta_spliceai.splice_engine.error_model.utils import select_samples_for_analysis
  
  # Get samples near the decision boundary
  border_samples = select_samples_for_analysis(
      model, X_test, y_test, 
      n_samples=10, 
      selection="border"
  )
  ```
