"""
Model utilities for splice site prediction.

This module re-exports base model utilities from the shared base_models package.
For new code, prefer importing directly from:
    meta_spliceai.splice_engine.base_models

Supported base models:
- SpliceAI (GRCh37, Keras)
- OpenSpliceAI (GRCh38, PyTorch)

Note: This module is maintained for backward compatibility.
New code should use the shared base_models package directly.
"""

import os
from typing import List, Dict, Any, Optional, Union, Tuple

# Re-export from shared base_models package for backward compatibility
from meta_spliceai.splice_engine.base_models import (
    load_base_model_ensemble as _load_base_model_ensemble,
    load_openspliceai_ensemble as _load_openspliceai_ensemble,
    load_spliceai_ensemble as _load_spliceai_ensemble,
)


def load_base_model_ensemble(
    base_model: str = 'spliceai',
    context: int = 10000,
    device: Optional[str] = None,
    verbosity: int = 1
) -> Tuple[List, Dict[str, Any]]:
    """
    Load base model ensemble with automatic model selection.
    
    .. deprecated::
        This function is maintained for backward compatibility.
        New code should import directly from:
        ``meta_spliceai.splice_engine.base_models.load_base_model_ensemble``
    
    Parameters
    ----------
    base_model : str, default='spliceai'
        Base model to load ('spliceai' or 'openspliceai')
    context : int, default=10000
        Context window size in nucleotides
    device : str, optional
        Device for PyTorch models
    verbosity : int, default=1
        Output verbosity level
    
    Returns
    -------
    Tuple[List, Dict[str, Any]]
        (models, metadata) tuple
    """
    # Delegate to shared implementation
    return _load_base_model_ensemble(
        base_model=base_model,
        context=context,
        device=device,
        verbosity=verbosity
    )


def load_openspliceai_ensemble(
    context: int = 10000,
    device: Optional[str] = None,
    verbosity: int = 1
) -> Tuple[List, str]:
    """
    Load OpenSpliceAI ensemble models.
    
    .. deprecated::
        Use ``meta_spliceai.splice_engine.base_models.load_openspliceai_ensemble``
    """
    return _load_openspliceai_ensemble(
        context=context,
        device=device,
        verbosity=verbosity
    )


def load_spliceai_ensemble(context: int = 10000) -> List:
    """
    Load SpliceAI ensemble models.
    
    .. deprecated::
        Use ``meta_spliceai.splice_engine.base_models.load_spliceai_ensemble``
    """
    return _load_spliceai_ensemble(context=context)


def verify_model_compatibility(models: List, min_version: str = '1.3.1') -> bool:
    """
    Verify that the loaded models are compatible with the expected version.
    
    Parameters
    ----------
    models : List
        List of loaded models to verify
    min_version : str, optional
        Minimum required SpliceAI version, by default '1.3.1'
        
    Returns
    -------
    bool
        True if models are compatible, False otherwise
    """
    try:
        import spliceai
        from packaging import version
        
        # Check SpliceAI package version
        current_version = spliceai.__version__
        if version.parse(current_version) < version.parse(min_version):
            print(f"Warning: SpliceAI version {current_version} is older than the recommended {min_version}")
            return False
            
        # Basic model verification
        if not models or len(models) != 5:
            print(f"Warning: Expected 5 SpliceAI models, found {len(models) if models else 0}")
            return False
            
        # All checks passed
        return True
    except Exception as e:
        print(f"Error verifying model compatibility: {e}")
        return False


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
                'name': f'spliceai{i+1}',
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape)
            })
        except Exception as e:
            metadata['model_types'].append({
                'index': i,
                'name': f'spliceai{i+1}',
                'error': str(e)
            })
    
    return metadata
