# Deep Learning Model Incompatibility with Multi-Instance Training

**Last Updated:** September 2025  
**Status:** ✅ **DOCUMENTED & RESOLVED**

---

## 🎯 Executive Summary

TabNet and TensorFlow models (`tf_mlp`) are **fundamentally incompatible** with the Multi-Instance Ensemble Training architecture used by `run_gene_cv_sigmoid.py` due to architectural differences between 3-binary-classifier and multi-class classification paradigms.

**Solution:** Created separate `run_gene_cv_deep_learning.py` module for deep learning models.

---

## 🔍 The Core Problem

### Architecture Mismatch

**`run_gene_cv_sigmoid.py` uses 3-binary-classifier approach:**
```python
# Each instance trains 3 separate binary models
for cls in (0, 1, 2):  # neither, donor, acceptor
    y_bin = (y == cls).astype(int)  # Convert to binary
    model = _train_binary_model(X, y_bin, ...)  # Binary classifier
    models_cls.append(model)

# Wraps in SigmoidEnsemble
ensemble = SigmoidEnsemble(models_cls, feature_names)
```

**Deep learning models are single multi-class classifiers:**
```python
# TabNet: Single multi-class model
model = TabNetClassifier(n_d=64, n_a=64, n_steps=5)
model.fit(X, y)  # y has 3 classes: [0, 1, 2]
proba = model.predict_proba(X)  # Shape: (n_samples, 3)

# TensorFlow: Multi-class neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")  # 3 classes
])
model.compile(loss="categorical_crossentropy")
```

---

## ⚠️ Specific Incompatibilities

### 1. Training Loop Architecture
```python
# Multi-Instance expects 3 binary models per instance
for instance in instances:
    binary_models = []
    for class_idx in [0, 1, 2]:
        binary_model = train_binary_classifier(X, y_binary)
        binary_models.append(binary_model)
    instance_ensemble = SigmoidEnsemble(binary_models)

# TabNet/TensorFlow are single models
# Cannot be split into 3 binary classifiers
# Cannot be trained in binary classification loop
```

### 2. SigmoidEnsemble Wrapper Incompatibility
```python
class SigmoidEnsemble:
    def __init__(self, models_cls, feature_names):
        # Expects 3 separate binary models
        self.models_cls = models_cls  # [neither_model, donor_model, acceptor_model]
    
    def predict_proba(self, X):
        # Combines 3 binary predictions
        proba_parts = [m.predict_proba(X)[:, 1] for m in self.models_cls]
        return np.column_stack(proba_parts)

# TabNet/TensorFlow already output 3-class probabilities
# Cannot be wrapped in SigmoidEnsemble
# Would create double-wrapping: SigmoidEnsemble(TabNet) → incorrect
```

### 3. Multi-Instance Training Incompatibility
```python
# Multi-Instance Training expects SigmoidEnsemble-compatible models
def _train_instance_model(self, gene_subset):
    # This calls the same 3-binary-classifier training loop
    models_cls = []
    for cls in (0, 1, 2):
        model = _train_binary_model(X, y_bin, ...)  # Binary training
        models_cls.append(model)
    
    # Creates SigmoidEnsemble for each instance
    ensemble = SigmoidEnsemble(models_cls, feature_names)
    return ensemble

# TabNet/TensorFlow cannot be trained this way
# They need direct multi-class training, not 3-binary-classifier approach
```

### 4. Model Consolidation Incompatibility
```python
# Multi-Instance consolidation expects SigmoidEnsemble instances
def _consolidate_models(self, trained_instances):
    consolidated_models = []
    for instance in trained_instances:
        # Each instance is a SigmoidEnsemble
        ensemble = instance['model']  # SigmoidEnsemble object
        consolidated_models.append(ensemble)
    
    # Create ConsolidatedMetaModel from SigmoidEnsemble instances
    return ConsolidatedMetaModel(consolidated_models)

# TabNet/TensorFlow instances would be single models
# Cannot be consolidated using SigmoidEnsemble logic
# Would break the entire consolidation pipeline
```

### 5. SHAP Analysis Incompatibility
```python
# Multi-Instance SHAP expects 3-binary-model structure
def _compute_ensemble_shap(self, X):
    for instance in self.instance_models:
        # Each instance has 3 binary models
        for class_name in ['neither', 'donor', 'acceptor']:
            binary_model = instance.get_model(class_name)
            shap_values = TreeExplainer(binary_model).shap_values(X)
            # Process binary SHAP values...

# TabNet/TensorFlow are single multi-class models
# Cannot be analyzed using 3-binary-model SHAP logic
# Would require completely different SHAP analysis approach
```

---

## 🚫 Why This Architecture Cannot Be Modified

**The Multi-Instance system is deeply integrated with 3-binary-classifier paradigm:**

1. **Instance Training**: Each instance trains 3 binary models
2. **Model Storage**: Each instance stores SigmoidEnsemble
3. **Consolidation Logic**: Combines SigmoidEnsemble instances
4. **SHAP Analysis**: Analyzes 3-binary-model structure
5. **Inference Interface**: Expects SigmoidEnsemble-compatible models

**Modifying for deep learning would require:**
- Complete rewrite of instance training pipeline
- New model storage and consolidation logic
- New SHAP analysis framework
- New inference interface
- Breaking compatibility with existing workflows

**This would essentially create a completely different system, not a modification of the existing one.**

---

## ✅ The Solution: Separate Deep Learning CV Module

**This is why we created `run_gene_cv_deep_learning.py`:**

```python
# Deep learning CV uses multi-class classification directly
class DeepLearningGeneCV:
    def _train_model(self, X_train, y_train, X_val, y_val):
        # Direct multi-class training - no binary classifier loop
        if algorithm == 'tabnet':
            model.fit(X_train, y_train)  # Multi-class training
        elif algorithm == 'tf_mlp_multiclass':
            model.fit(X_train, y_train)  # Multi-class training
        
        return model  # Single model, no SigmoidEnsemble wrapper
    
    def _predict(self, X):
        # Direct predictions from single model
        y_pred = self.model.predict(X)      # Class predictions
        y_prob = self.model.predict_proba(X)  # Probability predictions
        return y_pred, y_prob
```

**Benefits of Separate Module:**
- ✅ **Proper Architecture**: Multi-class models trained correctly
- ✅ **Gene-Aware CV**: Maintains gene boundary integrity
- ✅ **No Breaking Changes**: Preserves existing Multi-Instance system
- ✅ **Model Compatibility**: Works with any sklearn-compatible multi-class model
- ✅ **Future-Proof**: Easy to add new deep learning models

---

## 📊 System Comparison

| Aspect | `run_gene_cv_sigmoid.py` | `run_gene_cv_deep_learning.py` |
|--------|-------------------------|--------------------------------|
| **Classification** | 3 separate binary models | 1 multi-class model |
| **Models** | Tree-based, linear | Deep learning (TabNet, TF, etc.) |
| **Training** | Binary classification loop | Direct multi-class training |
| **Ensemble** | SigmoidEnsemble wrapper | Direct model usage |
| **Multi-Instance** | ✅ Supported | ❌ Not applicable |
| **Use Case** | Traditional ML | Deep learning research |

---

## 🎯 When to Use Each System

### Use `run_gene_cv_sigmoid.py` for:
- ✅ Traditional ML algorithms (XGBoost, CatBoost, LightGBM, Random Forest)
- ✅ Multi-Instance Training with large datasets
- ✅ 3-binary-classifier approach
- ✅ Production workflows requiring proven stability

### Use `run_gene_cv_deep_learning.py` for:
- ✅ Deep learning models (TabNet, TensorFlow, transformers)
- ✅ Multi-class classification research
- ✅ Multi-modal approaches (sequence + tabular)
- ✅ State-of-the-art model experimentation

---

## 📚 Detailed Documentation

**For comprehensive technical details, see:**

1. **[COMPREHENSIVE_TRAINING_GUIDE.md](meta_spliceai/splice_engine/meta_models/training/docs/COMPREHENSIVE_TRAINING_GUIDE.md#deep-learning-model-limitations-in-multi-instance-training)**
   - Complete algorithm selection guide
   - Deep learning limitations section
   - When to use each system

2. **[MULTI_INSTANCE_ENSEMBLE_TRAINING.md](meta_spliceai/splice_engine/meta_models/training/docs/MULTI_INSTANCE_ENSEMBLE_TRAINING.md#deep-learning-model-architecture-incompatibility)**
   - Technical architecture details
   - Why modification isn't feasible
   - Multi-Instance system requirements

3. **[DEEP_LEARNING_CV_README.md](DEEP_LEARNING_CV_README.md)**
   - Complete usage guide for deep learning CV
   - Working examples and commands
   - Installation requirements

---

## 🎉 Conclusion

The incompatibility between deep learning models and Multi-Instance Training is **fundamental and architectural**, not a simple implementation detail. The solution of creating a separate `run_gene_cv_deep_learning.py` module:

- ✅ **Preserves existing functionality** (no breaking changes)
- ✅ **Enables deep learning research** (proper multi-class training)
- ✅ **Maintains gene-aware CV** (no data leakage)
- ✅ **Future-proofs the system** (easy to add new models)

**Both systems work together to provide comprehensive coverage of all machine learning approaches for splice site prediction.**

