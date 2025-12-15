# Splice Site Error Analysis Framework

## Overview

The error analysis framework provides tools for training machine learning models to understand, classify, and explain errors made by splice site prediction models (such as SpliceAI). This enables deep insights into the patterns and features that contribute to prediction errors, ultimately leading to improved models.

## Workflow

1. **Problem Formulation**: The workflow begins by defining an error classification problem:

   - **FP vs TP** (False Positives vs True Positives):
     - Identifies features that distinguish falsely predicted splice sites from correctly predicted ones
     - Targets the problem of decoy or spurious "near-canonical" signals that confuse base models
     - Reveals negative signals or partial motifs that should be penalized in improved models
     - Helps reduce false discovery rate in downstream applications
   
   - **FN vs TP** (False Negatives vs True Positives):
     - Reveals which real splice signals the base model undervalues or misses
     - Identifies new or nonstandard motifs that deserve higher weight in predictions
     - Helps improve sensitivity for non-canonical but functional splice sites
     - Particularly valuable for discovering novel exons with atypical boundaries
   
   - **FN vs TN** (False Negatives vs True Negatives):
     - Distinguishes missed real sites from truly negative regions that lack splicing signals
     - Identifies partial signals or borderline cases that are functional but go undetected
     - Helps characterize "cryptic" splice sites that may be conditionally active
     - Critical for understanding splice site strength spectrum and context dependency

   Each error model highlights different important sequence motifs and features:
   - From FP vs TP: A set of decoy motifs that lead to false predictions
   - From FN vs TP: A set of under-recognized legitimate splicing signals
   - From FN vs TN: A set of partial or context-dependent signals in cryptic sites

2. **Feature Extraction**: Features are derived from:
   - Sequence features (k-mers)
   - Gene/transcript-level features
   - Prediction scores and metrics

3. **Model Training**: XGBoost models are trained to classify errors with cross-validation

4. **Feature Importance Analysis**: Multiple methods analyze feature importance:
   - XGBoost built-in importance metrics
   - SHAP (SHapley Additive exPlanations) values
   - Statistical hypothesis testing
   - Effect size measurements
   - Mutual information

5. **Visualization**: Comprehensive visualizations of model performance and feature importance

The ultimate goal is to identify features (including the base model's prediction scores) that best explain the differences between error classes and correct predictions, particularly the characteristics of the contextual sequences surrounding predicted splice sites. These insights can then inform the development of improved models or post-processing steps to enhance prediction accuracy.

## Directory Structure

Output files are organized in a hierarchical structure:

```
<experiment_dir>/
    ├── donor/
    │   ├── fp_vs_tp/
    │   ├── fn_vs_tp/
    │   └── fn_vs_tn/
    ├── acceptor/
    │   ├── fp_vs_tp/
    │   ├── fn_vs_tp/
    │   └── fn_vs_tn/
    └── any/
        ├── fp_vs_tp/
        ├── fn_vs_tp/
        └── fn_vs_tn/
```

- `experiment_dir`: The main experiment identifier (e.g., "hard_genes")
- `donor`, `acceptor`, `any`: Splice site type
- `fp_vs_tp`, `fn_vs_tp`, `fn_vs_tn`: Error classification task

## Output Files

The error analysis pipeline generates the following outputs for each experiment/splice-type/error-model combination:

For all file patterns below, `{prefix}` follows the format `{splice_type}_{error_model}`. For example:
- `donor_fp_vs_tp` - For donor splice sites, comparing False Positives vs True Positives
- `acceptor_fn_vs_tp` - For acceptor splice sites, comparing False Negatives vs True Positives
- `any_fn_vs_tn` - For both splice site types, comparing False Negatives vs True Negatives

Note that while the directory structure uses lowercase (e.g., `fp_vs_tp`), some filenames might use uppercase for error types (e.g., `donor_FP_vs_TP_done.txt`). The patterns below use lowercase for consistency, but actual filenames might vary.

All output files for a given combination are stored in: `<experiment_dir>/<splice_type>/<error_model>/xgboost/`

### 1. Model Performance Metrics

| File Pattern | Description |
|--------------|-------------|
| `{prefix}-xgboost-roc.pdf` | Standard ROC curve for model evaluation |
| `{prefix}-xgboost-ROC-CV-{n_splits}folds.pdf` | Cross-validated ROC curve showing model robustness across folds |
| `{prefix}-xgboost-prc.pdf` | Standard Precision-Recall curve |
| `{prefix}-xgboost-PRC-CV-{n_splits}folds.pdf` | Cross-validated Precision-Recall curve showing model robustness |

### 2. Feature Importance Analysis

#### 2.1 XGBoost Native Importance Metrics

| File Pattern | Description |
|--------------|-------------|
| `{prefix}-xgboost-importance-xgboost-weight.tsv` | Top-K features by weight importance |
| `{prefix}-xgboost-importance-xgboost-weight-full.tsv` | All features by weight importance |
| `{prefix}-xgboost-importance-weight-barplot.pdf` | Visualization of weight importance |
| `{prefix}-xgboost-importance-xgboost-total_gain.tsv` | Top-K features by total gain importance |
| `{prefix}-xgboost-importance-xgboost-total_gain-full.tsv` | All features by total gain importance |
| `{prefix}-xgboost-importance-total_gain-barplot.pdf` | Visualization of total gain importance |

#### 2.2 SHAP Analysis

| File Pattern | Description |
|--------------|-------------|
| `{prefix}-xgboost-importance-shap.tsv` | Top-K features by SHAP importance |
| `{prefix}-xgboost-importance-shap-full.tsv` | All features by SHAP importance |
| `{prefix}-shap_summary_bar-meta.pdf` | Bar plot of SHAP feature importance |
| `{prefix}-shap_beeswarm-meta.pdf` | SHAP beeswarm plot showing feature impact and direction |
| `{prefix}-shap_summary_with_margin.pdf` | SHAP summary with extra margin to prevent label truncation |
| `{prefix}-global_shap_importance-meta.csv` | Global SHAP importance scores |

#### 2.3 Statistical Analysis

| File Pattern | Description |
|--------------|-------------|
| `{prefix}-xgboost-hypo-testing-results.tsv` | Results from statistical hypothesis testing |
| `{prefix}-xgboost-importance-hypo-testing.tsv` | Top-K features by hypothesis testing importance |
| `{prefix}-xgboost-importance-hypo-testing-full.tsv` | All features by hypothesis testing importance |
| `{prefix}-xgboost-hypo-testing-barplot.pdf` | Visualization of hypothesis testing results |
| `{prefix}-xgboost-effect-sizes-results.tsv` | Calculated effect sizes for features |
| `{prefix}-xgboost-importance-effect-sizes-full.tsv` | All features by effect size importance |
| `{prefix}-xgboost-effect-sizes-barplot.pdf` | Visualization of effect sizes |

#### 2.4 Motif-Specific Analysis

| File Pattern | Description |
|--------------|-------------|
| `{prefix}-xgboost-motif-importance-shap.tsv` | Top-K motif features by SHAP importance |
| `{prefix}-xgboost-motif-importance-shap-full.tsv` | All motif features by SHAP importance |
| `{prefix}-motif_importance-barplot.pdf` | Visualization of motif importance |
| `{prefix}-nonmotif_importance-barplot.pdf` | Visualization of non-motif feature importance |

### 3. Feature Comparison and Distribution Analysis

| File Pattern | Description |
|--------------|-------------|
| `{prefix}-feature-importance-comparison.pdf` | Comparison of importance rankings across methods |
| `{prefix}-feature-distributions.pdf` | Distribution plots of top features across classes |
| `{prefix}-local-shap-frequency-comparison-meta.pdf` | Comparison of local SHAP values frequency |
| `{prefix}-local_top25_freq-meta.csv` | Frequency analysis of top local feature importance |
| `{prefix}-global_importance-barplot.pdf` | Global feature importance across all methods |

### 4. Completion Marker

| File Pattern | Description |
|--------------|-------------|
| `{prefix}_done.txt` | Marker file indicating successful completion |

## Usage

The error analysis workflow can be executed through:

1. The `train_error_classifier` function in `error_classifier.py` for training a single error type model
2. The `xgboost_pipeline` function in `xgboost_trainer.py` for direct access to the training pipeline
3. The `workflow_train_error_classifier` function in `error_classifier.py` for analyzing all combinations of splice types and error models

### Single Error Type Analysis

```python
from meta_spliceai.splice_engine.model_training.error_classifier import train_error_classifier

# Train an error classifier for False Positives vs True Positives on donor sites
model, results = train_error_classifier(
    pred_type='FP',           # Error type to analyze (FP, FN)
    splice_type='donor',      # Splice site type (donor, acceptor, or None for both)
    output_dir='results',     # Output directory
    n_splits=5,               # Number of cross-validation folds
    top_k=20                  # Number of top features to analyze
)
```

### Complete Error Analysis Workflow

To run the complete error analysis across all possible combinations of splice types and error models:

```python
from meta_spliceai.splice_engine.model_training.error_classifier import workflow_train_error_classifier

# Run the complete error analysis workflow
workflow_train_error_classifier(
    experiment='hard_genes',  # Experiment name for directory organization
    output_dir='results',     # Base output directory
    n_splits=5,               # Number of cross-validation folds
    top_k=20,                 # Number of top features to analyze
    enable_check_existing=False  # Skip combinations that have already been processed
)
```

This function systematically processes all combinations of:
- Splice types: donor, acceptor, any
- Error models: fp_vs_tp, fn_vs_tp, fn_vs_tn

The results are organized in the directory structure described above, with each combination having its own complete set of analysis outputs.

## Interpreting Results

### Feature Importance Interpretation

1. **SHAP Values**: Show both magnitude and direction of feature impact
   - Higher absolute SHAP value = stronger influence on prediction
   - Positive SHAP = feature pushes prediction toward the positive class (error)
   - Negative SHAP = feature pushes prediction toward the negative class (correct)

2. **XGBoost Importance**:
   - Weight: Number of times a feature appears in trees
   - Gain: Improvement in accuracy contributed by a feature
   - Cover: Relative number of observations affected by a feature

3. **Statistical Measures**:
   - Hypothesis testing: Statistical significance of feature difference between classes
   - Effect sizes: Magnitude of the difference between classes (Cohen's d, etc.)
   - Mutual information: Amount of information gained about class by knowing the feature

### Key Visualizations

1. **ROC & PR Curves**: Model performance on distinguishing between classes
2. **SHAP Beeswarm**: Shows distribution of SHAP values across features
3. **Feature Distributions**: Shows how feature values differ between classes
4. **Feature Importance Comparison**: Shows agreement between different importance measures

### Most Informative Outputs

When examining the results of an error analysis run, these files are particularly valuable for understanding why errors occur:

1. `{prefix}-shap_beeswarm-meta.pdf` - Shows the most influential features and their effects
2. `{prefix}-feature-distributions.pdf` - Shows how feature values are distributed across error vs. correct predictions
3. `{prefix}-global_importance-barplot.pdf` - Summarizes importance across all methods
4. `{prefix}-xgboost-ROC-CV-{n_splits}folds.pdf` - Shows model performance robustness across cross-validation folds
5. `{prefix}-local_top25_freq-meta.csv` - Identifies features that are consistently important for individual samples

Start with these outputs to gain the most insights about error patterns, then explore other outputs for more detailed analysis.

## References

- [SHAP: A Unified Approach to Interpreting Model Predictions](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Effect Size Calculations](https://en.wikipedia.org/wiki/Effect_size)
