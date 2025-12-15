# Error Model Package

The Error Model package provides a comprehensive framework for analyzing and modeling errors in splice site predictions. It is designed to work with the existing MetaSpliceAI modules while providing a clean, modular interface for both traditional ML models and sequence-based neural approaches.

## Components

The package consists of the following core modules:

### 1. Workflow Module (`workflow.py`)

This module provides high-level workflows for training, evaluating, and analyzing error models across different splice types and error categories.

Key functions:
- `process_error_model()`: Process a single error model for a specific splice type and error category
- `process_all_error_models()`: Process all error models across specified splice types
- `apply_stratified_sampling()`: Apply stratified subsampling to reduce dataset size while maintaining class distribution

### 2. Classifier Module (`classifier.py`)

This module provides a clean interface to the existing error classifier functionality, bridging the old implementation with the new package structure.

Key components:
- `ErrorClassifier`: Class for training and analyzing error classifiers with an object-oriented interface
- Feature importance analysis methods 
- Model evaluation metrics

### 3. Sequence Module (`sequence.py`)

This module provides a clean interface to the existing error sequence model functionality, serving as a bridge between the new package structure and the existing implementation.

Key components:
- `ErrorSequenceModel`: Interface for sequence-based error models
- `DistributedErrorSequenceModel`: Distributed training interface for sequence-based models
- Methods for analyzing and explaining sequence model errors

### 4. Utilities Module (`utils.py`)

This module provides helper functions for the error model package, such as output verification and result processing utilities.

Key functions:
- `verify_error_model_outputs()`: Verify that expected output files exist with correct naming conventions
- `get_model_status()`: Get the status of error models across all combinations
- `get_model_file_paths()`: Get paths to output files for a specific error model
- `safely_save_figure()`: Safely save matplotlib figures and prevent memory leaks
- `select_samples_for_analysis()`: Select samples for targeted feature importance visualization using multiple strategies
- `get_sample_shap_files()`: Get all sample-specific SHAP plot files for a given error model
- `get_analysis_summary()`: Get a comprehensive summary of all analysis outputs for the given experiment

## Output File Validation

The error model module includes robust file validation logic to track and verify all outputs generated during error analysis workflows. This includes:

1. **Status Tracking**: The completion status of error models is tracked via special marker files (`donor_FP_vs_TP_done.txt`).
2. **Comprehensive File Detection**: The module can identify all generated output files across different directories, including:
   - Feature distribution plots
   - Global importance plots
   - Local SHAP analysis plots
   - ROC and PR curves (including CV-based variants with variable fold counts)
   - Sample-specific plots (with variable sample IDs)
   - Importance metric files (TSV)
3. **Flexible Pattern Matching**: The file validation logic uses wildcard patterns to handle varying parameters:
   - `*folds` for different cross-validation fold counts
   - `sample*` for different sample IDs in local SHAP plots
   - `top*` for different top-k values in feature rankings
4. **Analysis Summary**: The `get_analysis_summary()` function provides a comprehensive overview of all analysis outputs across different splice types and error models.

## CLI Usage

The package includes a command-line interface (CLI) tool for training error classifiers. The CLI provides flexible options for model selection, data sampling, and visualization parameters.

### Running the CLI

```bash
# Run from project root directory
python -m meta_spliceai.splice_engine.error_model.cli.train_classifier [options]
```

### Key CLI Options

#### 1. Model Selection

The CLI supports three ways to specify which models to train:

**Option A: Default Parameter Specification (Preferred)**
```bash
python -m meta_spliceai.splice_engine.error_model.cli.train_classifier \
  --splice-type donor \
  --error-label FP \
  --correct-label TP
```
Sensible defaults are provided:
- `splice_type` defaults to "any" (combining donor and acceptor sites)
- `error_label` defaults to "FP"
- `correct_label` defaults to "TP"

**Option B: Using Shorthand Format**
```bash
python -m meta_spliceai.splice_engine.error_model.cli.train_classifier --model donor:fp_vs_tp
```

**Option C: Process All Combinations**
```bash
python -m meta_spliceai.splice_engine.error_model.cli.train_classifier --all
```

#### 2. Experiment Configuration

```bash
python -m meta_spliceai.splice_engine.error_model.cli.train_classifier \
  --all \
  --experiment my_experiment_name
```

#### 3. Data Sampling for Faster Development

```bash
python -m meta_spliceai.splice_engine.error_model.cli.train_classifier \
  --splice-type donor \
  --error-label FP \
  --correct-label TP \
  --sample-ratio 0.1 \
  --max-samples 1000
```

#### 4. Feature Importance Methods and Visualization

```bash
python -m meta_spliceai.splice_engine.error_model.cli.train_classifier \
  --model donor:fp_vs_tp \
  --importance-methods shap,xgboost,mutual_info \
  --top-k 20 \
  --shap-local-top-k 20 \
  --shap-global-top-k 50 \
  --shap-plot-top-k 25 \
  --feature-plot-type violin
```

#### 5. Model Parameters

```bash
python -m meta_spliceai.splice_engine.error_model.cli.train_classifier \
  --model donor:fp_vs_tp \
  --n-estimators 100 \
  --max-depth 6 \
  --learning-rate 0.1 \
  --subsample 0.8 \
  --colsample-bytree 0.8
```

### Full Example

```bash
python -m meta_spliceai.splice_engine.error_model.cli.train_classifier \
  --experiment myproject \
  --splice-type donor \
  --error-label FP \
  --correct-label TP \
  --sample-ratio 0.1 \
  --max-samples 1000 \
  --importance-methods shap,xgboost,mutual_info \
  --top-k 20 \
  --feature-plot-type violin \
  --n-estimators 100 \
  --verbose 2
```

## Usage Examples (Python API)

### Training a Single Error Model

```python
from meta_spliceai.splice_engine.error_model import process_error_model

# Process a single error model
result = process_error_model(
    error_label="FP",
    correct_label="TP",
    splice_type="donor",
    experiment="my_experiment",
    # Use 10% of data for faster processing
    sample_ratio=0.1,
    max_samples=1000,
    # Feature importance methods to use
    importance_methods=["shap", "xgboost", "mutual_info"],
    # Visualization parameters
    top_k=20,
    use_advanced_feature_plots=True,
    feature_plot_type="box",
    # Sample selection for targeted feature importance
    sample_selection="misclassified",  # Options: random, high_confidence, low_confidence, border, misclassified, custom
    n_samples=5  # Number of samples to visualize for local feature importance
)
```

### Processing All Error Models

```python
from meta_spliceai.splice_engine.error_model import process_all_error_models

# Process all combinations of splice types and error models
results = process_all_error_models(
    experiment="my_experiment",
    model_type="xgboost",  # Ensure this matches your data preparation
    sample_ratio=0.1,
    max_samples=1000,
    # Feature importance methods to use
    importance_methods=["shap", "xgboost", "mutual_info"],
    # Visualization parameters
    top_k=20,
    use_advanced_feature_plots=True,
    feature_plot_type="box",
    # SHAP parameters
    shap_local_top_k=20,
    shap_global_top_k=50,
    shap_plot_top_k=25,
    # Verify outputs after processing
    verify_outputs=True,
    splice_types=["donor", "acceptor", "any"],
    error_models=["fp_vs_tp", "fn_vs_tp", "fn_vs_tn"],
    skip_biologically_meaningless=True,
    enable_check_existing=True  # Skip models that already have outputs
)
```

### Using the OOP Interface with ErrorClassifier

```python
from meta_spliceai.splice_engine.error_model import ErrorClassifier

# Initialize the classifier
classifier = ErrorClassifier(
    error_label="FP",
    correct_label="TP", 
    splice_type="donor",
    experiment="my_experiment",
    output_dir="path/to/output"
)

# Load and subsample data
classifier.load_data(
    sample_ratio=0.1,
    max_samples=1000,
    verbose=1
)

# Train the model
model, results = classifier.train(
    n_splits=5,
    n_estimators=100,
    top_k=20,
    use_advanced_feature_plots=True,
    feature_plot_type="box",
    max_depth=6,
    importance_methods=["shap", "xgboost", "mutual_info"],
    verbose=1
)

# Generate feature importance visualizations with targeted sample selection
classifier.visualize_feature_importance(
    sample_selection="misclassified",  # Focus on samples the model got wrong
    n_samples=10,
    shap_plot_top_k=25
)
```

## Advanced Features

### Parameter Namespace Pattern

The error model package implements a robust parameter passing pattern using Python's `inspect` module, which automatically adapts to API changes and prevents passing invalid parameters. This pattern is used throughout the codebase to simplify function calls and improve maintainability.

### Feature Tracking

The error model ensures consistency between model training and explanation by:
- Explicitly tracking feature columns used during model training
- Storing feature names in model objects for future reference
- Performing feature consistency checks during SHAP analysis
- Handling dimensional mismatches between model features and analysis data

### Sample Selection for Feature Importance Analysis

The error model package supports systematic sample selection for more targeted feature importance visualization:

- **random**: Selects random samples (default method)
- **high_confidence**: Focuses on samples where the model is most confident
- **low_confidence**: Focuses on samples where the model is least confident
- **border**: Analyzes samples near the decision boundary (most uncertain)
- **misclassified**: Focuses on samples the model got wrong
- **custom**: Allows providing specific sample indices

This targeted analysis helps understand model behavior on specific types of data points, particularly useful for debugging model weaknesses.

## CLI Tools

The package also includes command-line interface tools in the `cli` directory for common tasks.

## Testing

The error_model package includes a comprehensive test suite in the `tests` directory. These tests verify the functionality of data sampling, model training, and result verification components.

### Running Tests

Tests can be run in several ways:

#### Method 1: Directly Running a Test File

```bash
# From the project root directory
python -m meta_spliceai.splice_engine.error_model.tests.test_workflow
```

#### Method 2: Using unittest Module Discovery

```bash
# From the project root directory
python -m unittest discover -s meta_spliceai/splice_engine/error_model/tests
```

#### Method 3: Using pytest (if installed)

```bash
# From the project root directory
pytest meta_spliceai/splice_engine/error_model/tests
```

#### Method 4: Running a Specific Test Class or Method

```bash
# Run just the TestSampling class
python -m unittest meta_spliceai.splice_engine.error_model.tests.test_workflow.TestSampling

# Run a specific test method
python -m unittest meta_spliceai.splice_engine.error_model.tests.test_workflow.TestSampling.test_apply_stratified_sampling_ratio
```

For more details on tests, see the README in the [tests directory](tests/README.md).
