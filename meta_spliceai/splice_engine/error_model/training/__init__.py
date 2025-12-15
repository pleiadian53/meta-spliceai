"""
Training module for error analysis models in MetaSpliceAI.

This package provides modular components for training models to analyze splice site 
prediction errors and extract insights about error mechanisms.

Key modules:
- xgboost_trainer: XGBoost-based model training and evaluation
- error_classifier: Error classification model implementation
- visualization: Visualization utilities for model analysis

For most use cases, import the high-level functions from the package level:
- train_xgboost_model: Train an XGBoost error classifier
- evaluate_xgboost_model: Evaluate model performance
- xgboost_pipeline: End-to-end training and evaluation workflow
- train_error_classifier: High-level interface for error model training
"""

# Import key components for convenient access
from meta_spliceai.splice_engine.model_training.xgboost_trainer import (
    train_xgboost_model,
    evaluate_xgboost_model,
    xgboost_pipeline
)

from meta_spliceai.splice_engine.model_training.error_classifier import (
    train_error_classifier
)

# Re-export for direct import from training module
__all__ = [
    'train_xgboost_model',
    'evaluate_xgboost_model',
    'xgboost_pipeline',
    'train_error_classifier'
]
