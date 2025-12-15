# MetaSpliceAI - Splice Engine

## Overview

The `splice_engine` is the core computational framework of MetaSpliceAI, providing a comprehensive suite of tools for splice site prediction, error analysis, and novel isoform discovery. This framework integrates state-of-the-art deep learning models with classical machine learning approaches to deliver accurate splice site predictions and detailed error analysis.

## Architecture

The `splice_engine` is organized into several interconnected modules, each responsible for a specific part of the analysis pipeline:

```
splice_engine/
├── SpliceAI Integration - Base splice site prediction
├── Error Analysis       - Error classification and pattern identification
├── Model Explanation    - Feature importance and model interpretability
├── Evaluation Framework - Performance metrics and benchmarking
├── Utilities            - Support functions and helpers
└── Data Processing      - Dataset preparation and transformation
```

## Component Documentation

### SpliceAI Integration

* **Purpose**: Provides the base splice site prediction functionality
* **Key Modules**:
  * [`run_spliceai_workflow.py`](./run_spliceai_workflow.py) - Main SpliceAI prediction pipeline
  * [`demo_spliceai.py`](./demo_spliceai.py) - Example usage and demonstration
  * [`extract_gene_sequences.py`](./extract_gene_sequences.py) - Sequence extraction utilities

### Error Analysis Framework

* **Purpose**: Identifies and characterizes errors made by the base prediction models
* **Key Modules**:
  * [`splice_error_analyzer.py`](./splice_error_analyzer.py) - Core error analysis functionality
  * [`label_splice_sites.py`](./label_splice_sites.py) - Truth labeling for evaluation
  * [`performance_analyzer.py`](./performance_analyzer.py) - Performance metrics and visualization
* **Detailed Documentation**: See [model_training/README.md](./model_training/README.md) for the error analysis workflow

### Model Explanation System

#### Classical ML Pipeline

* **Purpose**: Uses XGBoost and other classical ML approaches for error classification
* **Key Modules**:
  * [`model_training/`](./model_training/) - Refactored error classifier and model training
  * [`feature_importance/`](./feature_importance/) - Feature importance calculation methods
* **Detailed Documentation**: See [model_training/README.md](./model_training/README.md) for XGBoost pipeline

#### Transformer Pipeline

* **Purpose**: Uses transformer models for complex sequence pattern detection
* **Key Modules**:
  * [`error_sequence_model.py`](./error_sequence_model.py) - Transformer model architecture
  * [`train_error_sequence_model.py`](./train_error_sequence_model.py) - Training pipeline
  * [`hybrid_error_sequence_model.py`](./hybrid_error_sequence_model.py) - Combined approach

### Evaluation Framework

* **Purpose**: Provides standardized evaluation metrics and benchmarking
* **Key Modules**:
  * [`evaluate_models.py`](./evaluate_models.py) - Evaluation framework
  * [`model_evaluator.py`](./model_evaluator.py) - Metrics and evaluation functions
  * [`model_utils.py`](./model_utils.py) - Utility functions for model evaluation

### Utilities

* **Purpose**: Support functions used across the framework
* **Key Modules**:
  * [`utils_bio.py`](./utils_bio.py) - Biological sequence utilities
  * [`utils_fs.py`](./utils_fs.py) - File system operations
  * [`utils_df.py`](./utils_df.py) - DataFrame handling
  * [`utils_plot.py`](./utils_plot.py) - Visualization utilities
  * [`utils_doc.py`](./utils_doc.py) - Documentation helpers

### Data Processing

* **Purpose**: Dataset preparation and transformation
* **Key Modules**:
  * [`prepare_splice_site_dataset.py`](./prepare_splice_site_dataset.py) - Dataset creation
  * [`overlapping_gene_mapper.py`](./overlapping_gene_mapper.py) - Gene overlap handling
  * [`sequence_featurizer.py`](./sequence_featurizer.py) - Feature extraction from sequences

## Subpackage Documentation

The following subpackages have detailed documentation:

* [**model_training/**](./model_training/README.md) - Error classifier framework and XGBoost pipeline
  * Includes comprehensive documentation on error model types, outputs, and interpretation

* [**feature_importance/**](./feature_importance/) - Multiple methods for feature importance calculation
  * XGBoost built-in importance metrics
  * SHAP (SHapley Additive exPlanations) analysis
  * Statistical hypothesis testing
  * Effect size measurements
  * Mutual information

## Specialized Documentation

The following documents provide in-depth information on specific topics:

* [**ERROR_ANALYSIS.md**](./ERROR_ANALYSIS.md) - Comprehensive guide to error analysis models (FP vs TP, FN vs TP, FN vs TN)
  * Detailed biological interpretations of error types
  * Feature importance analysis techniques
  * Meta-model development insights
  * Visualization outputs and their significance

## Workflows

### Error Analysis Workflow

The error analysis workflow uses a combination of module components to:
1. Identify and classify prediction errors
2. Extract features that differentiate error classes
3. Train XGBoost models to predict errors
4. Calculate feature importance using multiple methods
5. Generate visualizations and reports

See [model_training/README.md](./model_training/README.md) for detailed documentation.

### Transformer-based Sequence Analysis

The transformer workflow uses neural networks to:
1. Process raw genomic sequences
2. Identify complex patterns beyond k-mer representations
3. Capture long-range dependencies in splice sites
4. Provide attention-based explanations of predictions

## Getting Started

```python
# Example: Running the error analysis workflow
from meta_spliceai.splice_engine.model_training.error_classifier import workflow_train_error_classifier

# Run the complete error analysis for all splice types and error models
workflow_train_error_classifier(
    experiment='my_experiment',  # Experiment name for organization
    n_splits=5,                  # Cross-validation folds
    top_k=20                     # Number of top features to analyze
)
```

## References

* SpliceAI: [SpliceAI paper](https://www.cell.com/cell/fulltext/S0092-8674(18)31629-5)
* SHAP: [A Unified Approach to Interpreting Model Predictions](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
* XGBoost: [XGBoost Documentation](https://xgboost.readthedocs.io/)
