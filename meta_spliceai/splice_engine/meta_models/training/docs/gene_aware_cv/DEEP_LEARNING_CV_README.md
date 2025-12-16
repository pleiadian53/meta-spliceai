# Deep Learning Gene-Aware Cross-Validation

This module provides gene-aware cross-validation specifically designed for deep learning models that use multi-class classification instead of the 3-binary-classifier approach used in `run_gene_cv_sigmoid.py`.

## 🎯 Key Features

- **Gene-aware CV**: No data leakage between genes (proper train/test separation)
- **Multi-class classification**: Direct 3-class prediction (neither/donor/acceptor)
- **Model-agnostic design**: Works with any sklearn-compatible multi-class classifier
- **Comprehensive evaluation**: ROC/PR curves, per-class metrics, confusion matrices
- **Deep learning support**: TabNet, TensorFlow, multi-modal transformers

## 🚀 Quick Start

### Basic Usage

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_deep_learning \
    --dataset data/ensembl/spliceai_analysis \
    --out-dir results/tabnet_cv \
    --algorithm tabnet \
    --n-folds 5
```

### With Custom Parameters

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_deep_learning \
    --dataset data/ensembl/spliceai_analysis \
    --out-dir results/tf_mlp_cv \
    --algorithm tf_mlp_multiclass \
    --n-folds 5 \
    --algorithm-params '{"hidden_units": [512, 256, 128], "dropout_rate": 0.4, "epochs": 100}'
```

## 📋 Supported Models

### 1. TabNet (Recommended)
```bash
--algorithm tabnet
```
- **Type**: Attention-based deep learning for tabular data
- **Best for**: High-dimensional k-mer features with built-in feature selection
- **Parameters**: `n_d`, `n_a`, `n_steps`, `gamma`, `lambda_sparse`

### 2. TensorFlow MLP Multi-Class
```bash
--algorithm tf_mlp_multiclass
```
- **Type**: Multi-class neural network
- **Best for**: Standard tabular features with deep learning
- **Parameters**: `hidden_units`, `dropout_rate`, `epochs`, `lr`

### 3. Multi-Modal Transformer
```bash
--algorithm multimodal_transformer
```
- **Type**: Sequence + tabular features
- **Best for**: Raw DNA sequences combined with numerical features
- **Parameters**: `sequence_length`, `embedding_dim`, `num_heads`, `num_layers`

## 🔧 Command Line Options

### Required Arguments
- `--dataset`: Path to dataset directory
- `--out-dir`: Output directory for results
- `--algorithm`: Algorithm to use (see supported models above)

### CV Configuration
- `--n-folds`: Number of CV folds (default: 5)
- `--valid-size`: Validation set size (default: 0.1)
- `--gene-col`: Gene column name (default: "gene_id")

### Data Configuration
- `--max-variants`: Maximum variants to process
- `--row-cap`: Row cap for dataset (default: 100,000)

### Model Configuration
- `--algorithm-params`: JSON string of algorithm-specific parameters

### Evaluation Configuration
- `--plot-curves`: Generate ROC/PR curves (default: True)
- `--plot-format`: Plot file format (pdf/png/svg, default: pdf)

## 📊 Output Artifacts

Each run generates:

### Metrics Files
- `fold_metrics.csv`: Per-fold performance metrics
- `aggregate_metrics.json`: Aggregate statistics across folds
- `cv_report.txt`: Comprehensive text report

### Model Files
- `trained_model.pkl`: Trained model (last fold)

### Visualizations
- `multiclass_roc_curves.pdf`: ROC curves for each class
- `multiclass_pr_curves.pdf`: Precision-Recall curves for each class

## 🧬 Gene-Aware CV Process

1. **Data Loading**: Load dataset and prepare features
2. **Gene Grouping**: Group samples by gene to prevent data leakage
3. **CV Splits**: Create train/validation/test splits preserving gene boundaries
4. **Model Training**: Train model on each fold with appropriate method
5. **Evaluation**: Comprehensive metrics and visualizations
6. **Results**: Save all artifacts and generate reports

## 🔍 Key Differences from `run_gene_cv_sigmoid.py`

| Aspect | `run_gene_cv_sigmoid.py` | `run_gene_cv_deep_learning.py` |
|--------|-------------------------|--------------------------------|
| **Classification** | 3 separate binary models | 1 multi-class model |
| **Models** | Tree-based, linear | Deep learning (TabNet, TF, etc.) |
| **Training** | Binary classification loop | Direct multi-class training |
| **Ensemble** | SigmoidEnsemble wrapper | Direct model usage |
| **Use Case** | Traditional ML | Deep learning research |

## ⚠️ Why Deep Learning Models Don't Work with `run_gene_cv_sigmoid.py`

**Technical Limitation:** TabNet and TensorFlow models are **fundamentally incompatible** with the Multi-Instance Ensemble Training architecture used by `run_gene_cv_sigmoid.py` due to architectural differences.

### The Core Problem: 3-Binary-Classifier vs Multi-Class Architecture

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

### Specific Incompatibilities

1. **Training Loop**: Deep learning models cannot be split into 3 binary classifiers
2. **SigmoidEnsemble Wrapper**: TabNet/TensorFlow already output 3-class probabilities
3. **Multi-Instance Training**: Each instance expects 3 binary models, not 1 multi-class model
4. **Model Consolidation**: Cannot combine single models using SigmoidEnsemble logic
5. **SHAP Analysis**: Expects 3-binary-model structure, not single multi-class model

### Why We Created a Separate Module

**Modifying `run_gene_cv_sigmoid.py` would require:**
- Complete rewrite of the 3-binary-classifier training loop
- New model storage and consolidation logic
- New SHAP analysis framework
- Breaking compatibility with existing Multi-Instance Training
- Essentially creating a completely different system

**Our solution: `run_gene_cv_deep_learning.py`**
- ✅ **Proper Architecture**: Multi-class models trained correctly
- ✅ **Gene-Aware CV**: Maintains gene boundary integrity
- ✅ **No Breaking Changes**: Preserves existing system
- ✅ **Model Compatibility**: Works with any sklearn-compatible multi-class model

**For detailed technical explanation, see:**
- [COMPREHENSIVE_TRAINING_GUIDE.md](meta_spliceai/splice_engine/meta_models/training/docs/COMPREHENSIVE_TRAINING_GUIDE.md#deep-learning-model-limitations-in-multi-instance-training)
- [MULTI_INSTANCE_ENSEMBLE_TRAINING.md](meta_spliceai/splice_engine/meta_models/training/docs/MULTI_INSTANCE_ENSEMBLE_TRAINING.md#deep-learning-model-architecture-incompatibility)

## 🛠️ Installation Requirements

### For TabNet
```bash
pip install pytorch-tabnet torch
```

### For TensorFlow Models
```bash
pip install tensorflow scikeras
```

### For Multi-Modal Transformer
```bash
pip install tensorflow scikeras
```

## 📈 Performance Tips

### Memory Optimization
- Use `--max-variants` to limit dataset size for testing
- Reduce `--row-cap` for memory-constrained systems
- Use smaller batch sizes for large models

### Model-Specific Tips

#### TabNet
- Start with default parameters
- Increase `n_steps` for more complex patterns
- Adjust `lambda_sparse` for feature selection strength

#### TensorFlow MLP
- Use deeper networks for complex patterns
- Adjust `dropout_rate` for regularization
- Monitor training with `--verbose`

#### Multi-Modal Transformer
- Use smaller `sequence_length` for memory efficiency
- Reduce `num_layers` and `num_heads` for faster training
- Requires both sequence and tabular features

## 🐛 Troubleshooting

### Common Issues

1. **ImportError: pytorch-tabnet not installed**
   ```bash
   pip install pytorch-tabnet
   ```

2. **ImportError: scikeras / tensorflow not installed**
   ```bash
   pip install tensorflow scikeras
   ```

3. **CUDA out of memory**
   - Reduce `--max-variants`
   - Use smaller batch sizes
   - Use CPU-only mode

4. **Dataset not found**
   - Check dataset path
   - Ensure dataset has required columns

### Debug Mode
```bash
--verbose  # Enable detailed logging
```

## 🔬 Advanced Usage

### Custom Model Parameters
```bash
--algorithm-params '{"n_d": 128, "n_a": 128, "n_steps": 8, "gamma": 2.0}'
```

### Multi-Modal with Sequence Data
```bash
--algorithm multimodal_transformer \
--algorithm-params '{"sequence_length": 2000, "embedding_dim": 128, "num_heads": 8}'
```

### Large Dataset Processing
```bash
--max-variants 50000 \
--row-cap 0  # No row cap
```

## 📚 Examples

See `../examples/deep_learning_cv_examples.py` for complete working examples with different models and configurations.

## 🤝 Contributing

To add new deep learning models:

1. Add model factory function to `models.py`
2. Register with `@_register("model_name")`
3. Ensure sklearn-compatible interface
4. Test with the CV system

## 📄 License

Same as MetaSpliceAI project.
