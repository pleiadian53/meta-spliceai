# Feature Leakage Probe

## Overview

The leakage probe is a diagnostic tool designed to identify potential feature leakage in the MetaSpliceAI meta-model. Feature leakage occurs when a feature is too strongly correlated with the target variable (splice type), which may indicate data contamination or features that are effectively proxies for the label itself.

## Usage

The leakage probe can be activated by passing the `--leakage-probe` flag to the gene CV script:

```bash
conda run -n surveyor python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset train_pc_1000/master \
    --out-dir models/meta_model_per_class_calibrated \
    --calibrate-per-class \
    --leakage-probe \
    [other options...]
```

## Implementation

The probe calculates correlation coefficients (Pearson by default) between each feature and the target variable (splice type). Features with correlation values exceeding the threshold (default: 0.95) are flagged as potentially "leaky" features.

### Process:

1. Load the feature manifest from the model directory
2. Calculate correlations between each feature and the splice type
3. Flag features with correlations above the threshold
4. Generate a comprehensive report of all features and their correlations

### Output Files:

- **feature_correlations.csv**: Complete report of all features and their correlations with the target
- **leakage_probe.tsv**: Only features with correlations above the threshold (legacy output)

### Output Format:

Each row in the reports contains:
- **feature**: Name of the feature
- **correlation**: Correlation coefficient with the target variable
- **is_leaky**: Boolean flag indicating if the correlation exceeds the threshold

## Interpreting Results

### What to Look For:

- **Very high correlations (>0.99)**: May indicate actual leakage or duplicated information
- **High correlations (0.9-0.99)**: Expected for some derived features, especially those based on base model scores
- **Moderate correlations (0.7-0.9)**: Typically indicate strong predictive features

### Common Patterns:

Features derived from base model outputs (e.g., `donor_score`, `acceptor_score`, `neither_score`) will naturally have higher correlations with the target, as they are designed to predict splice sites. This is expected and is not problematic in a meta-model context.

## Meta-Learning and Stack Generalization

### Legitimacy of Using Base Model Outputs

The MetaSpliceAI meta-model uses an approach called **stack generalization** (or "stacking"), which is a well-established ensemble learning technique. In stacking:

1. **Base learners** (like SpliceAI) make predictions
2. A **meta-learner** is trained using the base learners' predictions as features

This approach is entirely legitimate and follows standard practice in machine learning. The meta-model learns to:
- Identify when base models are likely correct or incorrect
- Combine their predictions optimally for different contexts
- Improve overall accuracy by addressing base model weaknesses

### Empirical Evidence for Stacking Effectiveness

Stack generalization has been empirically validated across numerous domains:

1. **Machine Learning Competitions**: Stacking is widely used by winning solutions in Kaggle competitions and other ML challenges

2. **Academic Research**:
   - Wolpert (1992) introduced stacking and demonstrated improved performance over individual models
   - Breiman (1996) showed that stacking outperforms both individual models and simple averaging
   - Džeroski and Ženko (2004) demonstrated that stacking with meta-decision trees achieved better performance than bagging or boosting

3. **Industry Applications**:
   - Recommendation systems (Netflix, Amazon) use stacked models to improve prediction quality
   - Medical diagnostics systems employ stacking to combine different types of diagnostic models
   - Computer vision systems frequently use stacking to combine CNN architectures

4. **MetaSpliceAI Results**:
   - The meta-model consistently outperforms base models in cross-validation
   - Performance improvements are observed across different gene sets and chromosomes
   - Error reduction is consistent across different splice site types

### Features Used in Meta-Models

The features in MetaSpliceAI's meta-model fall into several categories:

1. **Base model outputs**:
   - Direct scores (e.g., `donor_score`, `acceptor_score`, `neither_score`)
   - Valid meta-features for stack generalization

2. **Derived metrics from base outputs**:
   - Differences, ratios, and other combinations (e.g., `donor_acceptor_diff`)
   - Provide the meta-model with relationships between base predictions

3. **Contextual genomic information**:
   - Gene/transcript annotations, position information
   - Help the meta-model understand when base models may struggle

4. **Sequence features**:
   - k-mers, GC content, sequence complexity
   - Capture sequence patterns that may influence splice site prediction

All these features are legitimate in a meta-model context, and high correlation with the target for some features (especially those derived from base model outputs) is expected and not problematic.

## Conclusion

The leakage probe serves as a diagnostic tool to ensure data quality and feature integrity, but it's important to interpret its results in the context of stack generalization. High correlations between certain features and the target are expected in a meta-model that uses base model outputs as features. This is not "leakage" in the problematic sense but rather the intended design of stack generalization.
