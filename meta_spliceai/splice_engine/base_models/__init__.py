"""
Base Models Package

This package provides shared utilities for loading and using the underlying
splice site prediction models (SpliceAI, OpenSpliceAI).

Both `meta_models` (tabular approach) and `meta_layer` (multimodal approach)
depend on this shared package for base model functionality.

Key Components
--------------
- loaders: Functions to load SpliceAI and OpenSpliceAI model ensembles
- predictors: BaseModelPredictor for computing predictions and delta scores

Supported Models
----------------
- SpliceAI: Original Keras models (GRCh37/Ensembl)
- OpenSpliceAI: PyTorch reimplementation (GRCh38/MANE)

Usage
-----
>>> from meta_spliceai.splice_engine.base_models import load_base_model_ensemble
>>> models, metadata = load_base_model_ensemble('openspliceai')
>>> print(f"Loaded {metadata['num_models']} models for {metadata['genome_build']}")
"""

from .loaders import (
    load_base_model_ensemble,
    load_openspliceai_ensemble,
    load_spliceai_ensemble,
)

__all__ = [
    'load_base_model_ensemble',
    'load_openspliceai_ensemble', 
    'load_spliceai_ensemble',
]

