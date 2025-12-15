"""
Base model loaders for SpliceAI and OpenSpliceAI.

This module provides the unified interface for loading base splice site
prediction models, abstracting away framework differences (Keras vs PyTorch)
and model-specific details.

Functions
---------
load_base_model_ensemble : Unified loader for any supported base model
load_openspliceai_ensemble : Load OpenSpliceAI PyTorch models
load_spliceai_ensemble : Load original SpliceAI Keras models
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def load_base_model_ensemble(
    base_model: str = 'spliceai',
    context: int = 10000,
    device: Optional[str] = None,
    verbosity: int = 1
) -> Tuple[List, Dict[str, Any]]:
    """
    Load base model ensemble with automatic model selection.
    
    This is the unified entry point for loading any supported base model.
    It automatically routes to the appropriate loader based on the model type.
    
    Parameters
    ----------
    base_model : str, default='spliceai'
        Base model to load:
        - 'spliceai': SpliceAI (GRCh37, Keras)
        - 'openspliceai': OpenSpliceAI (GRCh38, PyTorch)
    
    context : int, default=10000
        Context window size in nucleotides (80, 400, 2000, or 10000)
    
    device : str, optional
        Device for PyTorch models ('cpu', 'cuda', 'mps').
        If None, auto-detects best available device.
    
    verbosity : int, default=1
        Output verbosity level
    
    Returns
    -------
    Tuple[List, Dict[str, Any]]
        - models: List of loaded model objects
        - metadata: Dictionary with model information:
            - 'base_model': Model name
            - 'genome_build': Genomic build (GRCh37 or GRCh38)
            - 'context': Context window size
            - 'framework': Deep learning framework (keras or pytorch)
            - 'num_models': Number of models in ensemble
            - 'device': Device (for PyTorch models)
    
    Raises
    ------
    ValueError
        If unsupported base model is specified
    
    Examples
    --------
    Load SpliceAI:
    
    >>> models, metadata = load_base_model_ensemble('spliceai')
    >>> print(f"Loaded {metadata['num_models']} models for {metadata['genome_build']}")
    
    Load OpenSpliceAI:
    
    >>> models, metadata = load_base_model_ensemble('openspliceai', device='cpu')
    >>> print(f"Using {metadata['framework']} on {metadata['device']}")
    """
    base_model_lower = base_model.lower()
    
    if base_model_lower == 'spliceai':
        models = load_spliceai_ensemble(context=context)
        metadata = {
            'base_model': 'spliceai',
            'genome_build': 'GRCh37',
            'context': context,
            'framework': 'keras',
            'num_models': len(models),
            'device': 'cpu'  # Keras/TensorFlow uses CPU by default
        }
        
    elif base_model_lower == 'openspliceai':
        models, device_used = load_openspliceai_ensemble(
            context=context,
            device=device,
            verbosity=verbosity
        )
        metadata = {
            'base_model': 'openspliceai',
            'genome_build': 'GRCh38',
            'context': context,
            'framework': 'pytorch',
            'num_models': len(models) if isinstance(models, list) else 1,
            'device': device_used
        }
        
    else:
        raise ValueError(
            f"Unsupported base model: '{base_model}'. "
            f"Supported models: ['spliceai', 'openspliceai']"
        )
    
    if verbosity >= 1:
        logger.info(
            f"Loaded {metadata['num_models']} {metadata['base_model']} "
            f"models ({metadata['genome_build']}, {metadata['framework']})"
        )
    
    return models, metadata


def load_openspliceai_ensemble(
    context: int = 10000,
    device: Optional[str] = None,
    verbosity: int = 1
) -> Tuple[List, str]:
    """
    Load OpenSpliceAI ensemble models with specified context.
    
    OpenSpliceAI models are PyTorch-based and trained on GRCh38 with MANE annotations.
    
    Parameters
    ----------
    context : int, default=10000
        Context window size. Must match available model (10000nt recommended)
    
    device : str, optional
        PyTorch device ('cpu', 'cuda', 'mps'). Auto-detects if None.
    
    verbosity : int, default=1
        Output verbosity
    
    Returns
    -------
    Tuple[List, str]
        - models: List of loaded PyTorch models
        - device: Device used for loading
    
    Notes
    -----
    OpenSpliceAI uses 5 ensemble models (rs10-rs14) trained with different
    random seeds for robust predictions.
    """
    import torch
    from meta_spliceai.openspliceai.predict.predict import load_pytorch_models
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    if verbosity >= 1:
        logger.info(f"Loading OpenSpliceAI models with {context}nt context on {device}")
    
    # Load models from systematic path
    model_path = "data/models/openspliceai/"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"OpenSpliceAI models not found at {model_path}. "
            f"Please run: ./scripts/base_model/download_openspliceai_models.sh"
        )
    
    # OpenSpliceAI's load_pytorch_models returns (models_list, params_dict)
    # Parameters: model_path, device, SL (sequence length), CL (context length)
    models = load_pytorch_models(
        model_path,
        device,
        SL=5000,  # Output sequence length
        CL=context
    )
    
    # Extract just the models list (first element of returned tuple)
    if isinstance(models, tuple):
        models = models[0]
    
    if verbosity >= 1:
        model_count = len(models) if isinstance(models, list) else 1
        logger.info(f"Loaded {model_count} OpenSpliceAI models successfully")
    
    return models, device


def load_spliceai_ensemble(context: int = 10000) -> List:
    """
    Load the SpliceAI ensemble models.
    
    SpliceAI models are Keras-based and trained on GRCh37 with Ensembl annotations.
    
    Parameters
    ----------
    context : int, optional
        Context size for SpliceAI models, by default 10000
        (Note: Keras models don't use this directly, included for API consistency)
        
    Returns
    -------
    List
        List of loaded SpliceAI Keras models
        
    Notes
    -----
    SpliceAI uses an ensemble of 5 models for prediction, which are averaged
    to produce final splice site probabilities.
    """
    from keras.models import load_model
    
    # Load SpliceAI models from systematic path
    model_dir = "data/models/spliceai/"
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"SpliceAI models not found at {model_dir}. "
            f"Please ensure models are installed."
        )
    
    paths = [os.path.join(model_dir, f"spliceai{i}.h5") for i in range(1, 6)]
    models = [load_model(path) for path in paths]
    
    return models


def get_model_metadata(models: List) -> Dict[str, Any]:
    """
    Extract metadata from the loaded models.
    
    Parameters
    ----------
    models : List
        List of loaded models
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing model metadata
    """
    metadata = {
        'num_models': len(models),
        'model_types': []
    }
    
    for i, model in enumerate(models):
        try:
            metadata['model_types'].append({
                'index': i,
                'name': f'model_{i+1}',
                'input_shape': str(getattr(model, 'input_shape', 'unknown')),
                'output_shape': str(getattr(model, 'output_shape', 'unknown'))
            })
        except Exception as e:
            metadata['model_types'].append({
                'index': i,
                'name': f'model_{i+1}',
                'error': str(e)
            })
    
    return metadata

