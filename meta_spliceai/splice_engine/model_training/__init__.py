"""
Model training package for MetaSpliceAI.

This package provides modular components for training models to analyze splice site 
prediction errors and extract insights about error mechanisms.
"""

from .xgboost_trainer import (
    train_xgboost_model,
    evaluate_xgboost_model,
    xgboost_pipeline
)

from .error_classifier import (
    train_error_classifier
)

__all__ = [
    'train_xgboost_model',
    'evaluate_xgboost_model',
    'xgboost_pipeline',
    'train_error_classifier'
]
