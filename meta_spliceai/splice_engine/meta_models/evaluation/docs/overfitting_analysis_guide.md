# Overfitting Analysis Guide

## Overview

The overfitting analysis system provides comprehensive monitoring and visualization of overfitting patterns during meta-learning cross-validation training. This system is particularly important for the **independent sigmoid ensemble** approach used in `run_gene_cv_sigmoid.py`, where multiple binary classifiers are trained for each CV fold.

## Key Components

### 1. OverfittingMonitor Class
- **Location**: `meta_spliceai/splice_engine/meta_models/evaluation/overfitting_monitor.py`
- **Purpose**: Monitors training progress and detects overfitting patterns
- **Integration**: Automatically integrated into gene-aware and chromosome-aware CV scripts

### 2. Generated Outputs
```
results/gene_cv_1000_run_15/overfitting_analysis/
├── overfitting_analysis.json          # Comprehensive metrics
├── overfitting_summary.txt            # Human-readable summary
├── learning_curves_by_fold.pdf        # Individual model learning curves
├── aggregated_learning_curves.pdf     # Mean curves with confidence bands
└── overfitting_summary.pdf            # Four-panel summary visualization
```

## Understanding the Metrics

### 1. Performance Gap

**Definition:**
```python
# For loss metrics (like logloss):
performance_gap = final_validation_score - final_training_score

# For performance metrics (like accuracy):
performance_gap = final_training_score - final_validation_score
```

**Interpretation:**
- **Positive values**: Model performs better on training than validation (overfitting)
- **Values > 0.05**: Significant overfitting (default threshold)
- **Near zero**: Good generalization
- **Negative values**: Unusual, possible data issues

**Default Threshold: 0.05**
- Configurable via `--overfitting-threshold` parameter
- Represents acceptable difference between training and validation performance
- Orange dashed line in plots shows this threshold

### 2. Overfitting Score

**Definition:**
```python
def _calculate_overfitting_score(self, train_scores, val_scores, is_loss_metric):
    # Calculate gap at each iteration
    if is_loss_metric:
        gaps = [val - train for train, val in zip(train_scores, val_scores)]
    else:
        gaps = [train - val for train, val in zip(train_scores, val_scores)]
    
    # Average positive gaps (area between curves)
    overfitting_score = np.mean([max(0, gap) for gap in gaps])
    
    # Penalty for large final gap
    final_gap = gaps[-1]
    if final_gap > self.gap_threshold:
        overfitting_score += final_gap * 2  # 2x penalty multiplier
    
    return overfitting_score
```

**Components:**
1. **Average area between training/validation curves** (persistent overfitting)
2. **Penalty for large final gap** (severe final overfitting)

**Interpretation:**
- **Higher scores**: More severe overfitting
- **Lower scores**: Better generalization
- **Composite metric**: Captures both sustained and final overfitting

### 3. Best Iteration

**Definition:**
```python
# For loss metrics (lower is better):
best_iteration = np.argmin(validation_scores)

# For performance metrics (higher is better):
best_iteration = np.argmax(validation_scores)
```

**Purpose:**
- **Optimal stopping point** where validation performance peaked
- **Recommended n_estimators**: `mean(best_iterations) + std(best_iterations)`
- **Early stopping guidance**: Helps set optimal patience parameters

### 4. Convergence Point

**Definition:**
```python
def _detect_convergence(self, scores, is_loss_metric):
    for i in range(patience, len(scores)):
        recent_window = scores[i-patience:i]
        current_score = scores[i]
        
        if is_loss_metric:
            # No improvement in reducing loss
            if min(recent_window) - current_score < min_improvement:
                return i
        else:
            # No improvement in increasing performance
            if current_score - max(recent_window) < min_improvement:
                return i
```

**Parameters:**
- **Patience**: Number of iterations to wait for improvement (default: 10)
- **Min improvement**: Minimum improvement threshold (default: 0.001)

**Interpretation:**
- **When improvement stops** (< `min_improvement` for `patience` iterations)
- **Convergence before best iteration**: Good (stopped improving before peak)
- **Convergence after best iteration**: Overfitting (continued training past optimal point)

## Visualization Guide

### 1. Training-Validation Performance Gap (Upper-Left)

**X-axis**: Model instances (e.g., 0_0, 0_1, 0_2, 1_0, 1_1, 1_2, ...)
- **Format**: `{fold_idx}_{class_idx}`
- **fold_idx**: CV fold number (0, 1, 2, 3, 4 for 5-fold CV)
- **class_idx**: Splice site class (0=Neither, 1=Donor, 2=Acceptor)

**Y-axis**: Performance gap (training - validation performance)
**Orange line**: Overfitting threshold (default: 0.05)

**Interpretation:**
- **Bars above threshold**: Significant overfitting
- **Consistent patterns across folds**: Systematic issues
- **Class-specific patterns**: Some splice types more prone to overfitting

### 2. Overfitting Score Distribution (Upper-Right)

**Histogram**: Distribution of overfitting scores across all binary models
**Red line**: Mean overfitting score

**Interpretation:**
- **Right-skewed distribution**: Most models generalize well
- **High mean**: Systematic overfitting issues
- **Wide distribution**: Inconsistent training behavior

### 3. Best Iteration Across Folds (Lower-Left)

**X-axis**: Model instances
**Y-axis**: Optimal boosting iteration number
**Red line**: Mean best iteration

**Interpretation:**
- **Consistent values**: Stable training requirements
- **High variance**: Some models need more/fewer iterations
- **Outliers**: Problematic model instances

### 4. Convergence Analysis (Lower-Right)

**Two lines:**
- **Circles**: Best iteration (when validation performance peaked)
- **Squares**: Convergence point (when improvement stopped)

**Interpretation:**
- **Lines close together**: Training stopped near optimal point
- **Convergence before best**: Good (stopped improving before peak)
- **Convergence after best**: Overfitting (continued past optimal)
- **Large gaps**: Significant overfitting occurred

## Model Instance Naming Convention

### Independent Sigmoid Ensemble Approach

The meta-learning system trains **3 binary classifiers per CV fold**:

```
For 5-fold CV:
- 0_0, 0_1, 0_2: Fold 0 → Neither, Donor, Acceptor classifiers
- 1_0, 1_1, 1_2: Fold 1 → Neither, Donor, Acceptor classifiers
- 2_0, 2_1, 2_2: Fold 2 → Neither, Donor, Acceptor classifiers
- 3_0, 3_1, 3_2: Fold 3 → Neither, Donor, Acceptor classifiers
- 4_0, 4_1, 4_2: Fold 4 → Neither, Donor, Acceptor classifiers
```

**Total**: 15 binary models for 5-fold CV

### Analysis Patterns

1. **Fold consistency**: Compare across fold numbers (0_1, 1_1, 2_1, 3_1, 4_1)
2. **Class-specific behavior**: Compare within folds (0_0 vs 0_1 vs 0_2)
3. **Overall stability**: Distribution across all 15 models

## Usage in CV Scripts

### Enabling Overfitting Analysis

```bash
# Gene-aware CV with overfitting monitoring
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset data/train_pc_1000/master \
    --out-dir results/gene_cv_1000_run_15 \
    --monitor-overfitting \
    --overfitting-threshold 0.05 \
    --early-stopping-patience 20 \
    --convergence-improvement 0.001
```

### Integration Code

```python
from meta_spliceai.splice_engine.meta_models.evaluation.overfitting_monitor import (
    OverfittingMonitor
)

# Initialize monitor
monitor = OverfittingMonitor(
    primary_metric="logloss",
    gap_threshold=0.05,
    patience=20,
    min_improvement=0.001
)

# During training (for each binary classifier)
for cls in (0, 1, 2):
    model, evals_result = _train_binary_model(X_train, y_train, X_val, y_val, args)
    
    # Add evaluation results to monitor
    monitor.add_fold_metrics(evals_result, f"{fold_idx}_{cls}")

# Generate analysis after CV loop
overfitting_report = monitor.generate_overfitting_report(out_dir)
monitor.plot_learning_curves(out_dir, plot_format="pdf")
```

## Interpretation Guidelines

### 1. Healthy Training Patterns

**Good signs:**
- Performance gaps < 0.05 for most models
- Consistent best iterations across folds
- Convergence points close to best iterations
- Low overfitting scores (< 0.02)

### 2. Problematic Patterns

**Warning signs:**
- Performance gaps > 0.05 for many models
- High variance in best iterations
- Convergence well after best iteration
- High overfitting scores (> 0.05)

### 3. Class-Specific Issues

**Donor classifiers (x_1):**
- Often more challenging due to sequence complexity
- May require different regularization

**Acceptor classifiers (x_2):**
- Usually more stable
- Consistent performance across folds

**Neither classifiers (x_0):**
- Largest class, often most stable
- Can be affected by class imbalance

## Recommended Actions

### 1. High Overfitting Scores

**Solutions:**
- Increase regularization parameters
- Reduce model complexity
- Implement early stopping
- Increase validation set size

### 2. Inconsistent Best Iterations

**Solutions:**
- Fold-specific hyperparameter tuning
- Increase cross-validation folds
- Check for data quality issues
- Consider ensemble methods

### 3. Late Convergence

**Solutions:**
- Increase patience for early stopping
- Reduce learning rate
- Implement learning rate scheduling
- Check for gradient issues

## Advanced Analysis

### 1. Learning Curves by Fold

**File**: `learning_curves_by_fold.pdf`
- Individual learning curves for each binary classifier
- Shows training/validation progression
- Marks best iteration and convergence points

### 2. Aggregated Learning Curves

**File**: `aggregated_learning_curves.pdf`
- Mean curves with confidence bands
- Overall training behavior
- Identifies systematic patterns

### 3. JSON Report

**File**: `overfitting_analysis.json`
```json
{
  "summary": {
    "total_folds": 15,
    "folds_with_overfitting": 3,
    "mean_performance_gap": 0.0234,
    "mean_overfitting_score": 0.0187,
    "recommended_n_estimators": 245
  },
  "fold_details": [
    {
      "fold_idx": "0_0",
      "performance_gap": 0.0123,
      "overfitting_score": 0.0089,
      "best_iteration": 156,
      "convergence_iteration": 178,
      "overfitting_detected": false
    }
  ]
}
```

## Best Practices

### 1. Regular Monitoring

- **Always enable** overfitting analysis for production runs
- **Review patterns** across different datasets
- **Track changes** over time and experiments

### 2. Threshold Tuning

- **Start with 0.05** as default threshold
- **Adjust based** on dataset characteristics
- **Consider class-specific** thresholds

### 3. Integration Workflow

1. **Enable monitoring** in CV scripts
2. **Review visualizations** after training
3. **Adjust hyperparameters** based on findings
4. **Document patterns** for future reference

## Troubleshooting

### Common Issues

1. **No overfitting data collected**
   - Check if `--monitor-overfitting` flag is enabled
   - Verify XGBoost evaluation results are captured
   - Ensure sufficient training iterations

2. **Inconsistent results**
   - Check random seed consistency
   - Verify data preprocessing
   - Review fold creation process

3. **Visualization errors**
   - Check matplotlib/seaborn versions
   - Verify output directory permissions
   - Review plot format compatibility

### Performance Considerations

- **Memory usage**: Monitor increases with number of folds
- **Computation time**: Minimal overhead during training
- **Storage**: Visualizations can be large for many folds

## Conclusion

The overfitting analysis system provides comprehensive insights into meta-learning training behavior, enabling:

1. **Early detection** of overfitting issues
2. **Systematic optimization** of hyperparameters
3. **Quality assurance** for production models
4. **Publication-ready** visualizations

This analysis is particularly valuable for the independent sigmoid ensemble approach, where multiple binary classifiers must be trained reliably across different CV folds and splice site classes. 