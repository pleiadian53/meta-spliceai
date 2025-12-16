# Thresholds and Fmax Analysis in Multiclass Classification

## Overview

This document explains the role of thresholds in multiclass classification, the concept of Fmax (maximum F1 score), and their relationship to argmax-based evaluation methods used in MetaSpliceAI's cross-validation workflows.

## Thresholds in Classification

### Binary vs Multiclass Thresholds

#### **Binary Classification (Simple)**
```python
# Single threshold determines positive vs negative
y_pred = (y_prob >= threshold).astype(int)
```

#### **Multiclass Classification (Complex)**
```python
# Multiple approaches possible:

# 1. Argmax (most common) - no threshold needed
y_pred = y_proba.argmax(axis=1)

# 2. Confidence threshold - reject low-confidence predictions
confidence = y_proba.max(axis=1)
y_pred = np.where(confidence >= threshold, y_proba.argmax(axis=1), -1)  # -1 = reject

# 3. Per-class thresholds - different threshold for each class
y_pred = np.zeros(len(y_proba))
for i in range(n_classes):
    mask = y_proba[:, i] >= class_thresholds[i]
    y_pred[mask] = i
```

### Generalization to Multiclass Scenarios

Thresholds can be generalized to multiclass scenarios in several ways:

#### **1. Confidence-Based Rejection**
```python
def confidence_threshold_prediction(y_proba, confidence_threshold=0.8):
    """Reject predictions below confidence threshold."""
    max_probs = y_proba.max(axis=1)
    y_pred = np.where(max_probs >= confidence_threshold, 
                      y_proba.argmax(axis=1), 
                      -1)  # -1 = "uncertain" class
    return y_pred
```

#### **2. Per-Class Thresholds**
```python
def per_class_threshold_prediction(y_proba, class_thresholds):
    """Apply different threshold for each class."""
    n_classes = len(class_thresholds)
    y_pred = np.zeros(len(y_proba))
    
    for i, threshold in enumerate(class_thresholds):
        mask = y_proba[:, i] >= threshold
        y_pred[mask] = i
    
    # Handle conflicts (multiple classes above threshold)
    conflicts = (y_proba >= class_thresholds).sum(axis=1) > 1
    y_pred[conflicts] = y_proba[conflicts].argmax(axis=1)
    
    return y_pred
```

#### **3. One-vs-Rest Thresholds**
```python
def one_vs_rest_threshold_prediction(y_proba, class_thresholds):
    """Treat each class as binary (one-vs-rest)."""
    n_classes = y_proba.shape[1]
    y_pred = np.zeros(len(y_proba))
    
    for i in range(n_classes):
        # Class i vs all others
        y_prob_binary = y_proba[:, i]
        threshold = class_thresholds[i]
        
        # Predict class i if probability >= threshold
        mask = y_prob_binary >= threshold
        y_pred[mask] = i
    
    # Handle conflicts by using argmax
    conflicts = (y_proba >= class_thresholds).sum(axis=1) > 1
    y_pred[conflicts] = y_proba[conflicts].argmax(axis=1)
    
    return y_pred
```

## Fmax: Maximum F1 Score Across Thresholds

### Definition and Concept

**Fmax** is the maximum F1 score achievable by varying the threshold across all possible values. It represents the optimal performance a model can achieve with perfect threshold tuning.

### Binary Fmax Calculation

```python
def calculate_fmax(y_true, y_prob, n_thresholds=100):
    """
    Calculate Fmax - maximum F1 score across all possible thresholds.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    n_thresholds : int, default=100
        Number of threshold values to test
        
    Returns
    -------
    fmax : float
        Maximum F1 score achieved
    optimal_threshold : float
        Threshold that achieves the maximum F1 score
    """
    from sklearn.metrics import f1_score
    
    thresholds = np.linspace(0, 1, n_thresholds)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)
    
    fmax = np.max(f1_scores)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    
    return fmax, optimal_threshold
```

### Multiclass Fmax Variants

#### **1. Macro Fmax (Average across classes)**
```python
def calculate_macro_fmax(y_true, y_proba, n_thresholds=100):
    """
    Calculate macro Fmax across all classes.
    
    Parameters
    ----------
    y_true : array-like
        True multiclass labels
    y_proba : array-like
        Predicted class probabilities (n_samples, n_classes)
    n_thresholds : int, default=100
        Number of threshold values to test per class
        
    Returns
    -------
    macro_fmax : float
        Average Fmax across all classes
    class_fmax_scores : list
        Fmax score for each class
    optimal_thresholds : list
        Optimal threshold for each class
    """
    from sklearn.metrics import f1_score
    
    n_classes = y_proba.shape[1]
    class_fmax_scores = []
    optimal_thresholds = []
    
    for i in range(n_classes):
        # One-vs-rest for each class
        y_binary = (y_true == i).astype(int)
        y_prob_binary = y_proba[:, i]
        
        fmax, optimal_threshold = calculate_fmax(y_binary, y_prob_binary, n_thresholds)
        class_fmax_scores.append(fmax)
        optimal_thresholds.append(optimal_threshold)
    
    macro_fmax = np.mean(class_fmax_scores)
    return macro_fmax, class_fmax_scores, optimal_thresholds
```

#### **2. Micro Fmax (Overall Fmax)**
```python
def calculate_micro_fmax(y_true, y_proba, n_thresholds=100):
    """
    Calculate micro Fmax using combined probabilities.
    
    Parameters
    ----------
    y_true : array-like
        True multiclass labels
    y_proba : array-like
        Predicted class probabilities (n_samples, n_classes)
    n_thresholds : int, default=100
        Number of threshold values to test
        
    Returns
    -------
    micro_fmax : float
        Overall Fmax score
    optimal_threshold : float
        Optimal threshold for combined prediction
    """
    # Convert to binary: any splice site vs none
    # Assuming labels: 0=donor, 1=acceptor, 2=neither
    y_binary = (y_true != 2).astype(int)  # 1=splice, 0=neither
    y_prob_binary = y_proba[:, 0] + y_proba[:, 1]  # Combined splice probability
    
    fmax, optimal_threshold = calculate_fmax(y_binary, y_prob_binary, n_thresholds)
    return fmax, optimal_threshold
```

#### **3. Weighted Fmax**
```python
def calculate_weighted_fmax(y_true, y_proba, n_thresholds=100):
    """
    Calculate weighted Fmax based on class frequencies.
    
    Parameters
    ----------
    y_true : array-like
        True multiclass labels
    y_proba : array-like
        Predicted class probabilities (n_samples, n_classes)
    n_thresholds : int, default=100
        Number of threshold values to test per class
        
    Returns
    -------
    weighted_fmax : float
        Weighted average Fmax across classes
    class_weights : list
        Weight of each class based on frequency
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Calculate class weights
    classes = np.unique(y_true)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_true)
    
    # Calculate Fmax for each class
    macro_fmax, class_fmax_scores, _ = calculate_macro_fmax(y_true, y_proba, n_thresholds)
    
    # Weighted average
    weighted_fmax = np.average(class_fmax_scores, weights=class_weights)
    
    return weighted_fmax, class_weights
```

## Relationship: Fmax vs Argmax F1

### Theoretical Relationship

The relationship between Fmax and argmax-based F1 depends on the model's probability calibration:

#### **1. Perfect Calibration**
```python
# If model probabilities are perfectly calibrated:
# Fmax ≈ Argmax F1 (they should be very close)
# Optimal threshold ≈ 0.5 (for balanced datasets)
```

#### **2. Overconfident Model**
```python
# If model is overconfident (probabilities too extreme):
# Fmax > Argmax F1 (threshold optimization helps)
# Optimal threshold < 0.5 (lower threshold needed)
```

#### **3. Underconfident Model**
```python
# If model is underconfident (probabilities too conservative):
# Fmax > Argmax F1 (threshold optimization helps)
# Optimal threshold > 0.5 (higher threshold needed)
```

### Practical Example

```python
import numpy as np
from sklearn.metrics import f1_score

def demonstrate_fmax_vs_argmax():
    """Demonstrate the relationship between Fmax and Argmax F1."""
    
    # Example: Overconfident model
    y_true = np.array([0, 0, 1, 1, 1])  # Binary labels
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5])  # Probabilities
    
    # Argmax approach (threshold = 0.5)
    y_pred_argmax = (y_prob >= 0.5).astype(int)
    f1_argmax = f1_score(y_true, y_pred_argmax)
    
    # Fmax approach (find optimal threshold)
    fmax, optimal_threshold = calculate_fmax(y_true, y_prob)
    
    print(f"Argmax F1 (threshold=0.5): {f1_argmax:.3f}")
    print(f"Fmax: {fmax:.3f}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Improvement: {fmax - f1_argmax:.3f}")
    
    return f1_argmax, fmax, optimal_threshold
```

### Expected Relationships

#### **For Well-Calibrated Models**
- **Fmax ≈ Argmax F1**: The model's natural confidence aligns with optimal thresholds
- **Optimal threshold ≈ 0.5**: For balanced datasets
- **Calibration gap ≈ 0**: Minimal benefit from threshold optimization

#### **For Poorly Calibrated Models**
- **Fmax > Argmax F1**: Threshold optimization provides significant improvement
- **Optimal threshold ≠ 0.5**: Model needs calibration
- **Large calibration gap**: Significant benefit from threshold optimization

#### **For Splice Site Prediction (Class Imbalanced)**
Given the class imbalance (many "neither" vs few splice sites):
- **Fmax > Argmax F1**: Likely due to class imbalance
- **Optimal threshold < 0.5**: Lower threshold needed to catch minority classes
- **Per-class thresholds differ**: Each class may need different optimal threshold

## Practical Implementation

### Integration with CV Workflow

```python
def enhanced_cv_metrics(y_true, y_proba, y_pred_argmax):
    """
    Calculate comprehensive metrics including Fmax analysis.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    y_pred_argmax : array-like
        Argmax predictions
        
    Returns
    -------
    dict
        Comprehensive metrics including Fmax analysis
    """
    from sklearn.metrics import f1_score, accuracy_score
    
    # Basic metrics (argmax-based)
    acc = accuracy_score(y_true, y_pred_argmax)
    macro_f1 = f1_score(y_true, y_pred_argmax, average="macro")
    
    # Fmax analysis
    macro_fmax, class_fmax_scores, optimal_thresholds = calculate_macro_fmax(y_true, y_proba)
    micro_fmax, micro_optimal_threshold = calculate_micro_fmax(y_true, y_proba)
    
    # Calibration analysis
    calibration_gap = macro_fmax - macro_f1
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_fmax": macro_fmax,
        "micro_fmax": micro_fmax,
        "calibration_gap": calibration_gap,
        "class_fmax_scores": class_fmax_scores,
        "optimal_thresholds": optimal_thresholds,
        "micro_optimal_threshold": micro_optimal_threshold,
        "well_calibrated": calibration_gap < 0.05,  # Threshold for "well calibrated"
    }
```

### Threshold Analysis for Model Comparison

```python
def compare_models_with_fmax(base_proba, meta_proba, y_true):
    """
    Compare base and meta models using both argmax and Fmax approaches.
    
    Parameters
    ----------
    base_proba : array-like
        Base model probabilities
    meta_proba : array-like
        Meta model probabilities
    y_true : array-like
        True labels
        
    Returns
    -------
    dict
        Comparison results
    """
    # Argmax predictions
    base_pred = base_proba.argmax(axis=1)
    meta_pred = meta_proba.argmax(axis=1)
    
    # Argmax-based comparison
    base_f1 = f1_score(y_true, base_pred, average="macro")
    meta_f1 = f1_score(y_true, meta_pred, average="macro")
    
    # Fmax-based comparison
    base_fmax, base_thresholds, _ = calculate_macro_fmax(y_true, base_proba)
    meta_fmax, meta_thresholds, _ = calculate_macro_fmax(y_true, meta_proba)
    
    return {
        "argmax_comparison": {
            "base_f1": base_f1,
            "meta_f1": meta_f1,
            "improvement": meta_f1 - base_f1
        },
        "fmax_comparison": {
            "base_fmax": base_fmax,
            "meta_fmax": meta_fmax,
            "improvement": meta_fmax - base_fmax
        },
        "calibration_analysis": {
            "base_calibration_gap": base_fmax - base_f1,
            "meta_calibration_gap": meta_fmax - meta_f1,
            "base_thresholds": base_thresholds,
            "meta_thresholds": meta_thresholds
        }
    }
```

## Recommendations and Best Practices

### When to Use Each Approach

#### **Argmax-Based Evaluation**
- **Use for**: Direct model capability comparison
- **Advantages**: Simple, no threshold tuning, fair comparison
- **Best for**: Research papers, model selection, capability assessment

#### **Fmax-Based Evaluation**
- **Use for**: Optimal performance assessment, calibration analysis
- **Advantages**: Shows best possible performance, identifies calibration issues
- **Best for**: Model optimization, deployment decisions, calibration analysis

#### **Threshold-Based Evaluation**
- **Use for**: Practical deployment, specific operating points
- **Advantages**: Realistic performance at chosen thresholds
- **Best for**: Production systems, specific use cases

### Reporting Recommendations

#### **Comprehensive Model Report**
```python
def generate_comprehensive_report(y_true, y_proba, model_name="Model"):
    """Generate comprehensive model evaluation report."""
    
    # Calculate all metrics
    y_pred = y_proba.argmax(axis=1)
    metrics = enhanced_cv_metrics(y_true, y_proba, y_pred)
    
    print(f"=== {model_name} Evaluation Report ===")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1 (Argmax): {metrics['macro_f1']:.3f}")
    print(f"Macro Fmax: {metrics['macro_fmax']:.3f}")
    print(f"Calibration Gap: {metrics['calibration_gap']:.3f}")
    print(f"Well Calibrated: {metrics['well_calibrated']}")
    
    if metrics['calibration_gap'] > 0.05:
        print("⚠️  Model may benefit from calibration")
        print(f"Optimal thresholds: {metrics['optimal_thresholds']}")
    
    return metrics
```

### Future Enhancements

#### **1. Automatic Calibration**
```python
def auto_calibrate_model(model, X_val, y_val):
    """Automatically calibrate model using validation data."""
    from sklearn.calibration import CalibratedClassifierCV
    
    calibrated_model = CalibratedClassifierCV(
        model, 
        cv='prefit', 
        method='isotonic'
    )
    calibrated_model.fit(X_val, y_val)
    return calibrated_model
```

#### **2. Threshold Optimization**
```python
def optimize_thresholds(y_true, y_proba, metric='f1'):
    """Find optimal thresholds for each class."""
    n_classes = y_proba.shape[1]
    optimal_thresholds = []
    
    for i in range(n_classes):
        y_binary = (y_true == i).astype(int)
        y_prob_binary = y_proba[:, i]
        
        # Grid search for optimal threshold
        thresholds = np.linspace(0, 1, 100)
        scores = []
        
        for thresh in thresholds:
            y_pred = (y_prob_binary >= thresh).astype(int)
            if metric == 'f1':
                score = f1_score(y_binary, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_binary, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_binary, y_pred, zero_division=0)
            scores.append(score)
        
        optimal_threshold = thresholds[np.argmax(scores)]
        optimal_thresholds.append(optimal_threshold)
    
    return optimal_thresholds
```

## Related Documentation

- [Confusion Matrix Interpretation Guide](confusion_matrix_interpretation_guide.md)
- [Evaluation Modules Overview](evaluation_modules_overview.md)
- [Model Calibration Guide](../calibration_diagnostics.py)

## References

- [Scikit-learn Threshold Optimization](https://scikit-learn.org/stable/modules/model_evaluation.html#threshold-optimization)
- [Model Calibration in Machine Learning](https://scikit-learn.org/stable/modules/calibration.html)
- [F1 Score and Threshold Optimization](https://en.wikipedia.org/wiki/F-score) 