# Error Analysis and Model Explanation in MetaSpliceAI

## Overview

The error analysis framework in MetaSpliceAI investigates false positives (FPs), false negatives (FNs), and other prediction errors from splice site predictors (e.g., SpliceAI). This framework aims to understand key patterns, features, and biological signals driving these errors. Insights from this process guide the interpretation of model predictions, elucidate error mechanisms, and inform the development of improved, more accurate models.

## Core Error Analysis Models

MetaSpliceAI implements three distinct error analysis models, each targeting specific prediction error types:

### 1. FP vs TP (False Positives vs. True Positives)

**Objective**: Identify genomic and sequence features that differentiate incorrectly predicted splice sites (FPs) from correctly predicted splice sites (TPs).

**Biological Interpretation**:
- Targets spurious, near-canonical, or cryptic splice signals that resemble genuine splice consensus sequences (e.g., partial GT-AG motifs, weak branch points or polypyrimidine tracts)
- Reveals "decoy" motifs or repetitive sequence elements that mislead models into falsely identifying splice sites

**Significance for Meta-Model**:
- Guides meta-models toward penalizing predictions when such decoy or cryptic signals are detected
- Reduces false discovery rates and enhances precision
- Enables filtering of spurious predictions in novel junction discovery

**Key Analysis Methods**:
- **Global Aggregation**: Aggregates feature importance across all samples, capturing universal patterns linked to false positives
  - SHAP values and summary plots
  - XGBoost intrinsic feature importance metrics (weight, total gain)
  - Hypothesis testing (e.g., Wilcoxon test) to confirm statistical significance
- **Local Aggregation**: Examines per-sample feature importance for context-specific signals
  - Top-K feature selection per sample to reveal subtle patterns
  - Frequency differential analysis for motifs disproportionately associated with FPs
  - Context-specific visualizations (local SHAP explanations, attention maps)

### 2. FN vs TP (False Negatives vs. True Positives)

**Objective**: Identify which genuine splice signals are missed or undervalued by the base model, causing false negatives (FNs).

**Biological Interpretation**:
- Highlights non-canonical but biologically valid splice sites (e.g., GC-AG introns, non-standard branch points, weaker polypyrimidine tracts)
- Reveals regulatory sequences such as exonic/intronic splicing enhancers (ESE/ISE) or silencers (ESS/ISS) that significantly influence splicing but may be overlooked

**Significance for Meta-Model**:
- Enables meta-models to upweight subtle or alternative splicing signals
- Increases sensitivity for atypical yet functional splice sites
- Enhances discovery of novel exons or isoforms

**Key Analysis Methods**:
- **Global Aggregation**: Highlights features frequently associated with missed splice sites across the dataset
  - Identifies systemic model shortcomings
  - Reveals patterns of non-canonical sites consistently missed
- **Local Aggregation**: Emphasizes individual predictions to uncover nuanced features
  - Reveals subtle or context-dependent biological signals overlooked by the model
  - Identifies tissue-specific or condition-specific factors influencing splicing

### 3. FN vs TN (False Negatives vs True Negatives)

**Objective**: Distinguish between genuine splice sites missed by the model (FNs) and true negatives (TNs) lacking any functional splice site signal.

**Biological Interpretation**:
- Identifies borderline or cryptic splice sites that are conditionally active or subtly regulated
- Characterizes signals in contexts that do not strictly follow canonical splice-site rules yet possess partial or context-dependent splicing potential

**Significance for Meta-Model**:
- Refines sensitivity to subtle or cryptic splice sites
- Crucial for understanding the full spectrum of splice site strength
- Improves conditional or context-dependent splice site recognition

**Key Analysis Methods**:
- **Global Aggregation**: Identifies broad differences between missed splice sites and genuine non-splice sites
  - Reveals globally relevant weak signals
  - Accounts for class imbalance through subsampling
- **Local Aggregation**: Reveals context-dependent features
  - Pinpoints subtle local signals (e.g., weak or non-canonical motifs)
  - Frequency differential analysis for motifs distinctly associated with FNs versus TNs

## Splice Types and Model Coverage

MetaSpliceAI systematically builds error-specific analysis models covering multiple splice site and error types:

**Splice Types (M)**:
- **Any**: Both donor and acceptor sites
- **Donor-specific**: Only donor splice site features
- **Acceptor-specific**: Only acceptor splice site features

**Model Types (N)**:
- **FP vs TP**: Reveals misleading motifs causing false positives
- **FN vs TP**: Highlights overlooked or underestimated motifs
- **FN vs TN**: Differentiates weak splice sites from true negatives

**Total Models**:
- With 3 splice types and 3 error model types, MetaSpliceAI produces 3 × 3 = 9 unique SHAP-based interpretability analyses
- Each model provides distinct biological insights into splicing mechanisms

## Feature Importance Analysis Techniques

MetaSpliceAI employs multiple complementary methods to ensure robust feature importance analysis:

### 1. SHAP (SHapley Additive exPlanations)
- Provides consistent, theoretically sound feature attribution
- Enables both global (dataset-level) and local (sample-level) explanations
- Visualized through summary plots, beeswarm plots, and force plots

### 2. XGBoost Native Metrics
- Weight: Number of times a feature is used in all trees
- Gain: Improvement in accuracy brought by a feature
- Coverage: Relative number of observations affected by a feature

### 3. Statistical Hypothesis Testing
- Wilcoxon rank-sum test for non-parametric comparison
- Identifies statistically significant feature differences between classes
- Calculated p-values and effect sizes quantify importance

### 4. Effect Size Measurements
- Cohen's d and other effect size metrics
- Quantifies the magnitude of difference between classes
- Robust to sample size variations

### 5. Mutual Information
- Measures information gain from features
- Captures non-linear relationships
- Complements other importance metrics

## Implementation in MetaSpliceAI

The error analysis framework is implemented through several key components:

- **`model_training/error_classifier.py`**: Core workflow for training error classifiers
- **`feature_importance/`**: Multiple methods for calculating feature importance
- **`xgboost_trainer.py`**: XGBoost model training and evaluation
- **`splice_error_analyzer.py`**: Core error analysis functionality

## Feature Highlights for Meta-Model Development

Each error model type provides distinctive insights critical for meta-model development:

- **FP vs. TP**: Decoy or spurious motifs → target for penalization
- **FN vs. TP**: Legitimate but undervalued signals → target for enhancement
- **FN vs. TN**: Context-dependent cryptic signals → target for subtle refinement

## Consistency and Biological Relevance

- These error analysis models align precisely with established biological principles of RNA splicing regulation and splice site recognition
- They guide meta-model development by identifying precise biological features to either penalize or reward
- This clear biological grounding ensures robustness, interpretability, and effectiveness of downstream model improvements

## Outputs and Visualizations

The error analysis workflow produces numerous outputs to facilitate interpretation:

### Visualization Files
- Feature distribution plots
- Feature importance comparison plots
- SHAP summary plots (global and local)
- ROC and PR curves for model evaluation

### Data Files
- Feature importance rankings (SHAP, XGBoost, hypothesis testing)
- Statistical test results
- Effect size measurements
- Model performance metrics

## Usage in MetaSpliceAI

The error analysis framework can be used through the `workflow_train_error_classifier()` function:

```python
from meta_spliceai.splice_engine.model_training.error_classifier import workflow_train_error_classifier

# Run error analysis workflow for all splice types and error models
workflow_train_error_classifier(
    experiment='my_experiment',
    n_splits=5,
    top_k=20
)
```

For testing and development, synthetic data can be used:

```python
from meta_spliceai.splice_engine.model_training.examples.test_workflow import create_synthetic_dataset

# Create synthetic data for testing
X, y, feature_names, output_dir, test_df = create_synthetic_dataset()

# Run workflow with synthetic data
workflow_train_error_classifier(
    experiment='synthetic_test',
    test_mode=True,
    test_data=test_df
)
```

## References

- SpliceAI: [A deep neural network that accurately predicts splice junctions from genomic data](https://www.cell.com/cell/fulltext/S0092-8674(18)31629-5)
- SHAP: [A Unified Approach to Interpreting Model Predictions](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
- XGBoost: [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
