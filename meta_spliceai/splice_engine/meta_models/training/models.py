"""Model registry for meta-model training.

This module maps a *model spec* (dict or simple string) → a ready-to-fit
estimator that follows the scikit-learn API (``fit`` / ``predict_proba`` /
``predict``).

Goals
-----
1. Provide a sane default (XGBoost) with hyper-parameters tuned for splice-site
   classification.
2. Be completely *model-agnostic* – users can register their own factory
   functions or override parameters without touching the training loop.
3. Leave the door open for deep-learning models.  For example, when the feature
   matrix includes raw *sequence* (instead of k-mers) one can request the
   ``"tf_mlp"`` spec, which returns a small Keras Sequential wrapped in
   ``scikeras`` so that it behaves like a classifier.

API
---
``get_model(spec: str | dict | None) -> Estimator``
    *spec* may be:
    • ``None`` – return the **default** model (``xgboost`` with defaults).
    • ``"xgboost"`` – shortcut for the default.
    • Any other *registered* key.
    • A *dict*  with a mandatory ``"name"`` field and optional hyper-params that
      override the registry default.

``register_model(name, factory)``
    Runtime extension hook: add custom models programmatically.

Example
-------
>>> spec = {"name": "xgboost", "n_estimators": 500, "learning_rate": 0.03}
>>> clf = get_model(spec)
>>> clf.fit(X_train, y_train)
"""
from __future__ import annotations

from typing import Any, Callable, Dict

# ---------------------------------------------------------------------------
#  3rd-party imports guarded – heavy deps are optional -----------------------
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier  # type: ignore
except ImportError:  # pragma: no cover – XGBoost optional for some envs
    XGBClassifier = None  # type: ignore

try:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – unlikely in full env
    RandomForestClassifier = LogisticRegression = None  # type: ignore

try:
    from catboost import CatBoostClassifier  # type: ignore
except ImportError:  # pragma: no cover – CatBoost optional
    CatBoostClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier  # type: ignore
except ImportError:  # pragma: no cover – LightGBM optional
    LGBMClassifier = None  # type: ignore

try:
    # Lightweight wrapper that turns Keras models into sklearn estimators
    from scikeras.wrappers import KerasClassifier  # type: ignore
    import tensorflow as tf  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    KerasClassifier = tf = None  # type: ignore

try:
    # TabNet: Attention-based deep learning for tabular data
    from pytorch_tabnet.tab_model import TabNetClassifier  # type: ignore
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    TabNetClassifier = torch = None  # type: ignore

__all__ = [
    "get_model",
    "register_model",
    "AVAILABLE_MODELS",
]

# ---------------------------------------------------------------------------
#  Registry implementation ---------------------------------------------------
# ---------------------------------------------------------------------------
Factory = Callable[[dict[str, Any]], Any]
_AVAILABLE_MODELS: dict[str, Factory] = {}


def _register(name: str):
    """Decorator to register a factory function under *name*."""

    def decorator(func: Factory):
        if name in _AVAILABLE_MODELS:
            raise ValueError(f"Model '{name}' already registered.")
        _AVAILABLE_MODELS[name] = func
        return func

    return decorator


# ------------------- Built-in model factories ------------------------------


@_register("xgboost")
def _make_xgb(params: dict[str, Any] | None = None):  # noqa: D401
    """Return a tuned **XGBClassifier** instance."""

    if XGBClassifier is None:
        raise ImportError("xgboost is not installed. Please 'pip install xgboost'.")

    default: dict[str, Any] = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )
    if params:
        default.update(params)
    return XGBClassifier(**default)


@_register("random_forest")
def _make_rf(params: dict[str, Any] | None = None):
    if RandomForestClassifier is None:
        raise ImportError("scikit-learn is not installed.")
    default = dict(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    if params:
        default.update(params)
    return RandomForestClassifier(**default)


@_register("log_reg")
def _make_logreg(params: dict[str, Any] | None = None):
    if LogisticRegression is None:
        raise ImportError("scikit-learn is not installed.")
    default = dict(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )
    if params:
        default.update(params)
    return LogisticRegression(**default)


@_register("catboost")
def _make_catboost(params: dict[str, Any] | None = None):
    """Return a tuned CatBoostClassifier instance optimized for genomic data."""
    
    if CatBoostClassifier is None:
        raise ImportError("catboost is not installed. Please 'pip install catboost'.")
    
    default = dict(
        iterations=800,
        learning_rate=0.1,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=False,
        allow_writing_files=False,  # Prevent temp file creation
        thread_count=-1,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        colsample_bylevel=0.8,
        reg_lambda=1.0,
        # Genomic-specific optimizations
        auto_class_weights='Balanced',  # Handle class imbalance
        border_count=254,  # Similar to XGBoost max_bin
    )
    if params:
        default.update(params)
    return CatBoostClassifier(**default)


@_register("lightgbm")  
def _make_lightgbm(params: dict[str, Any] | None = None):
    """Return a tuned LGBMClassifier instance optimized for genomic data."""
    
    if LGBMClassifier is None:
        raise ImportError("lightgbm is not installed. Please 'pip install lightgbm'.")
    
    default = dict(
        n_estimators=800,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary',
        metric='auc',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        # Genomic-specific optimizations
        class_weight='balanced',
        max_bin=255,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
    )
    if params:
        default.update(params)
    return LGBMClassifier(**default)


@_register("tf_mlp")
def _make_tf_mlp(params: dict[str, Any] | None = None):
    """Simple dense network wrapped for sklearn-compatible training.

    Expected input: numeric feature matrix (e.g. k-mers or dense embedding).
    For raw *sequence* models you would implement a separate factory (e.g.
    ``"tf_cnn"``) that builds a convolutional model on sequence tokens.
    """

    if KerasClassifier is None or tf is None:
        raise ImportError("scikeras / tensorflow not installed.")

    def build_model(input_dim: int, **hp):  # pylint: disable=unused-argument
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp.get("lr", 1e-3)),
            loss="binary_crossentropy",
            metrics=["AUC"],
        )
        return model

    default = dict(
        model=build_model,
        epochs=30,
        batch_size=256,
        verbose=0,
    )
    if params:
        default.update(params)
    return KerasClassifier(**default)


@_register("tf_mlp_multiclass")
def _make_tf_mlp_multiclass(params: dict[str, Any] | None = None):
    """Multi-class dense network for 3-class classification (neither/donor/acceptor).

    This model is designed for the deep learning CV pipeline and supports
    proper multi-class classification instead of the 3-binary-classifier approach.
    """

    if KerasClassifier is None or tf is None:
        raise ImportError("scikeras / tensorflow not installed.")

    def build_model(input_dim: int, **hp):  # pylint: disable=unused-argument
        n_classes = hp.get("n_classes", 3)
        hidden_units = hp.get("hidden_units", [256, 128])
        dropout_rate = hp.get("dropout_rate", 0.3)
        learning_rate = hp.get("lr", 1e-3)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
        ])
        
        # Add hidden layers
        for units in hidden_units:
            model.add(tf.keras.layers.Dense(units, activation="relu"))
            model.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Output layer for multi-class classification
        model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
        
        # Compile with appropriate loss and metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy", "AUC"]
        )
        return model

    default = dict(
        model=build_model,
        epochs=50,
        batch_size=256,
        verbose=0,
        n_classes=3,
        hidden_units=[256, 128],
        dropout_rate=0.3,
        lr=1e-3
    )
    if params:
        default.update(params)
    return KerasClassifier(**default)


@_register("multimodal_transformer")
def _make_multimodal_transformer(params: dict[str, Any] | None = None):
    """Multi-modal transformer model for sequence + tabular features.
    
    This model combines sequence data (DNA sequences) with tabular features
    (k-mers, positional features, etc.) for enhanced splice site prediction.
    
    Architecture:
    - Sequence branch: Transformer encoder for DNA sequences
    - Tabular branch: Dense layers for numerical features  
    - Fusion: Concatenation + final classification layers
    """
    
    if KerasClassifier is None or tf is None:
        raise ImportError("scikeras / tensorflow not installed.")
    
    def build_model(input_dim: int, **hp):  # pylint: disable=unused-argument
        n_classes = hp.get("n_classes", 3)
        sequence_length = hp.get("sequence_length", 1000)
        embedding_dim = hp.get("embedding_dim", 128)
        num_heads = hp.get("num_heads", 8)
        num_layers = hp.get("num_layers", 4)
        tabular_units = hp.get("tabular_units", [256, 128])
        learning_rate = hp.get("lr", 1e-4)
        
        # Input layers
        sequence_input = tf.keras.layers.Input(shape=(sequence_length,), name="sequence")
        tabular_input = tf.keras.layers.Input(shape=(input_dim - sequence_length,), name="tabular")
        
        # Sequence branch: Transformer encoder
        # Embedding layer for DNA tokens (A, C, G, T, N)
        vocab_size = 5  # A, C, G, T, N
        sequence_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=sequence_length,
            name="sequence_embedding"
        )(sequence_input)
        
        # Positional encoding
        position_encoding = tf.keras.layers.Embedding(
            input_dim=sequence_length,
            output_dim=embedding_dim,
            name="position_encoding"
        )(tf.range(sequence_length))
        
        sequence_encoded = sequence_embedding + position_encoding
        
        # Transformer layers
        for i in range(num_layers):
            # Multi-head attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embedding_dim // num_heads,
                name=f"attention_{i}"
            )(sequence_encoded, sequence_encoded)
            
            # Add & Norm
            attention_output = tf.keras.layers.Dropout(0.1)(attention_output)
            sequence_encoded = tf.keras.layers.LayerNormalization(
                name=f"norm1_{i}"
            )(sequence_encoded + attention_output)
            
            # Feed forward
            ffn_output = tf.keras.layers.Dense(
                embedding_dim * 4,
                activation="relu",
                name=f"ffn1_{i}"
            )(sequence_encoded)
            ffn_output = tf.keras.layers.Dense(
                embedding_dim,
                name=f"ffn2_{i}"
            )(ffn_output)
            
            # Add & Norm
            ffn_output = tf.keras.layers.Dropout(0.1)(ffn_output)
            sequence_encoded = tf.keras.layers.LayerNormalization(
                name=f"norm2_{i}"
            )(sequence_encoded + ffn_output)
        
        # Global average pooling for sequence
        sequence_pooled = tf.keras.layers.GlobalAveragePooling1D(name="sequence_pooling")(sequence_encoded)
        
        # Tabular branch: Dense layers
        tabular_output = tabular_input
        for units in tabular_units:
            tabular_output = tf.keras.layers.Dense(
                units,
                activation="relu",
                name=f"tabular_dense_{units}"
            )(tabular_output)
            tabular_output = tf.keras.layers.Dropout(0.3)(tabular_output)
        
        # Fusion: Concatenate both branches
        fused = tf.keras.layers.Concatenate(name="fusion")([sequence_pooled, tabular_output])
        
        # Final classification layers
        fused = tf.keras.layers.Dense(256, activation="relu", name="fusion_dense1")(fused)
        fused = tf.keras.layers.Dropout(0.3)(fused)
        fused = tf.keras.layers.Dense(128, activation="relu", name="fusion_dense2")(fused)
        fused = tf.keras.layers.Dropout(0.3)(fused)
        
        # Output layer
        output = tf.keras.layers.Dense(n_classes, activation="softmax", name="output")(fused)
        
        # Create model
        model = tf.keras.Model(
            inputs=[sequence_input, tabular_input],
            outputs=output,
            name="multimodal_transformer"
        )
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy", "AUC"]
        )
        
        return model
    
    default = dict(
        model=build_model,
        epochs=100,
        batch_size=32,
        verbose=0,
        n_classes=3,
        sequence_length=1000,
        embedding_dim=128,
        num_heads=8,
        num_layers=4,
        tabular_units=[256, 128],
        lr=1e-4
    )
    if params:
        default.update(params)
    return KerasClassifier(**default)


@_register("tabnet")
def _make_tabnet(params: dict[str, Any] | None = None):
    """TabNet: Attention-based deep learning for tabular data.
    
    Particularly effective for high-dimensional genomic features with
    built-in feature selection via attention mechanisms.
    
    Reference:
    Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning.
    AAAI 2021.
    """
    
    if TabNetClassifier is None or torch is None:
        raise ImportError(
            "pytorch-tabnet not installed. Please 'pip install pytorch-tabnet'."
        )
    
    # Separate initialization parameters from training parameters
    init_params = dict(
        # Architecture parameters
        n_d=64,               # Width of decision prediction layer
        n_a=64,               # Width of attention embedding  
        n_steps=5,            # Number of decision steps
        gamma=1.5,            # Relaxation parameter for feature reuse
        n_independent=2,      # Number of independent GLU layers
        n_shared=2,           # Number of shared GLU layers
        
        # Regularization
        lambda_sparse=1e-4,   # Sparsity regularization
        
        # Optimizer configuration
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.9),
        
        # GPU configuration - auto-detect CUDA
        device_name='auto',   # 'auto', 'cpu', or 'cuda'
        
        # Other parameters
        seed=42,
        verbose=0,
        mask_type='sparsemax',  # Better for high-dimensional k-mers
        cat_idxs=[],            # Will be set based on categorical features
        cat_dims=[],            # Will be set based on categorical features
        cat_emb_dim=1,          # Embedding dimension for categoricals
    )
    
    # Training parameters (not passed to __init__)
    training_params = {
        'max_epochs': 100,
        'patience': 15,
        'batch_size': 1024,
        'virtual_batch_size': 128,
        'num_workers': 0,
    }
    
    # Merge user parameters, separating init from training params
    if params:
        for key, value in params.items():
            if key in training_params:
                training_params[key] = value
            else:
                init_params[key] = value
    
    # Handle device selection for GPU utilization
    if init_params.get('device_name') == 'auto':
        if torch.cuda.is_available():
            init_params['device_name'] = 'cuda'
            print(f"🚀 TabNet: Auto-detected CUDA, using GPU acceleration")
        else:
            init_params['device_name'] = 'cpu'
            print(f"⚠️  TabNet: No CUDA detected, using CPU")
    elif init_params.get('device_name') == 'cuda':
        print(f"🚀 TabNet: Using explicit CUDA device for GPU acceleration")
    
    # Create TabNet model with only initialization parameters
    model = TabNetClassifier(**init_params)
    
    # Store training parameters as attributes for use during fit()
    for key, value in training_params.items():
        setattr(model, f'_tabnet_{key}', value)
    
    return model


# ---------------------------------------------------------------------------
#  Public helpers ------------------------------------------------------------
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = tuple(_AVAILABLE_MODELS.keys())


def register_model(name: str, factory: Factory) -> None:
    """Register *factory* under *name* at runtime."""
    if name in _AVAILABLE_MODELS:
        raise ValueError(f"Model '{name}' already registered.")
    _AVAILABLE_MODELS[name] = factory  # noqa: D401


def get_model(spec: str | dict[str, Any] | None = None):  # noqa: D401
    """Return a ready-to-fit estimator according to *spec*.

    Parameters
    ----------
    spec
        • ``None``           → default XGBoost.
        • ``"name"``         → lookup registry with default hyper-params.
        • ``{"name": str, ...params}``  → merge *params* into defaults.
    """
    if spec is None:
        name = "xgboost"
        params = None
    elif isinstance(spec, str):
        name, params = spec, None
    elif isinstance(spec, dict):
        if "name" not in spec:
            raise KeyError("Model spec dict must include a 'name' key.")
        name = spec["name"]
        params = {k: v for k, v in spec.items() if k != "name"}
    else:
        raise TypeError("spec must be None, str, or dict")

    if name not in _AVAILABLE_MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(_AVAILABLE_MODELS)}")

    factory = _AVAILABLE_MODELS[name]
    return factory(params)
