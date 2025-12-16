#!/usr/bin/env python3
"""
Enhanced Model Loader for Multi-Batch Ensemble Models

This module extends the standard model loading functionality to handle
both single models and multi-batch ensemble models created by the
automated all-genes trainer.

Key Features:
- Transparent loading of single models vs. ensemble models
- Voting-based ensemble predictions
- Weighted ensemble predictions (future enhancement)
- Compatible with existing inference workflow
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class EnsembleModelWrapper:
    """
    Wrapper that makes multi-batch ensemble models compatible with
    the standard model interface expected by the inference workflow.
    
    This class provides a `predict_proba` method that combines predictions
    from multiple batch models using ensemble voting.
    """
    
    def __init__(self, ensemble_data: Dict[str, Any]):
        """
        Initialize the ensemble wrapper.
        
        Parameters
        ----------
        ensemble_data : Dict[str, Any]
            Dictionary containing ensemble information:
            - 'batch_models': List of batch model dictionaries
            - 'combination_method': Strategy for combining predictions
            - 'feature_names': List of feature names
        """
        self.ensemble_data = ensemble_data
        self.batch_models = ensemble_data['batch_models']
        self.combination_method = ensemble_data.get('combination_method', 'voting')
        self.feature_names = ensemble_data.get('feature_names', [])
        
        # Extract individual models for prediction
        self.models = [batch['model'] for batch in self.batch_models]
        
        # Validate models
        if not self.models:
            raise ValueError("No batch models found in ensemble")
        
        logger.info(f"Initialized ensemble with {len(self.models)} batch models")
        logger.info(f"Combination method: {self.combination_method}")
        logger.info(f"Total genes covered: {ensemble_data.get('total_genes', 'unknown')}")
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions by combining batch model predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Probability predictions (n_samples, n_classes)
        """
        if len(X) == 0:
            return np.empty((0, 3))  # Return empty array with correct shape
        
        # Collect predictions from all batch models
        batch_predictions = []
        
        for i, model in enumerate(self.models):
            try:
                pred = model.predict_proba(X)
                batch_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Batch model {i} failed to predict: {e}")
                continue
        
        if not batch_predictions:
            raise RuntimeError("All batch models failed to generate predictions")
        
        # Combine predictions based on strategy
        if self.combination_method == 'voting':
            # Simple average of probabilities
            ensemble_pred = np.mean(batch_predictions, axis=0)
        elif self.combination_method == 'weighted':
            # Weighted average (future enhancement)
            # For now, fall back to simple average
            ensemble_pred = np.mean(batch_predictions, axis=0)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        return ensemble_pred
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate class predictions (argmax of probabilities)."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    @property
    def n_classes_(self) -> int:
        """Number of classes (compatible with sklearn interface)."""
        return 3  # Always 3 for splice site classification
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the first batch model."""
        if self.models and hasattr(self.models[0], name):
            return getattr(self.models[0], name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def load_model_with_ensemble_support(
    model_path: Union[str, Path], 
    use_calibration: bool = True,
    verbose: bool = True
) -> Any:
    """
    Enhanced model loader that handles both single models and multi-batch ensembles.
    
    This function extends the standard model loading to automatically detect
    and handle ensemble models created by the automated all-genes trainer.
    
    Parameters
    ----------
    model_path : str or Path
        Path to model file. Can be:
        - Single model: model_multiclass.pkl
        - Ensemble model: model_multiclass_all_genes.pkl
        - Directory: Will search for appropriate model
    use_calibration : bool, default=True
        Whether to use calibration when available
    verbose : bool, default=True
        Whether to print loading information
        
    Returns
    -------
    Any
        Model object compatible with predict_proba interface
    """
    model_path = Path(model_path)
    
    # If directory provided, resolve to specific model file
    if model_path.is_dir():
        model_path = resolve_ensemble_model_path(model_path, verbose=verbose)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if verbose:
        print(f"[EnsembleLoader] Loading model: {model_path}")
    
    # Load the model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Check if this is an ensemble model
    if isinstance(model_data, dict) and model_data.get('type') == 'AllGenesBatchEnsemble':
        if verbose:
            print(f"[EnsembleLoader] Detected multi-batch ensemble model")
            print(f"  Batch models: {model_data.get('batch_count', 0)}")
            print(f"  Total genes: {model_data.get('total_genes', 0)}")
            print(f"  Combination method: {model_data.get('combination_method', 'voting')}")
        
        # Wrap ensemble in compatibility layer
        ensemble_wrapper = EnsembleModelWrapper(model_data)
        return ensemble_wrapper
    
    else:
        # Standard single model - use existing loading logic
        if verbose:
            print(f"[EnsembleLoader] Detected standard single model")
        
        # Import and use the original loading function
        from ..inference_workflow_utils import load_model_with_calibration
        model = load_model_with_calibration(model_path, use_calibration=use_calibration)
        
        # Ensure the model has a predict method for compatibility
        if not hasattr(model, 'predict'):
            # Add predict method if missing
            def predict_method(X):
                probas = model.predict_proba(X)
                return np.argmax(probas, axis=1)
            model.predict = predict_method
        
        return model


def resolve_ensemble_model_path(model_dir: Path, verbose: bool = True) -> Path:
    """
    Resolve model path from directory, preferring ensemble models.
    
    Search priority:
    1. model_multiclass_all_genes.pkl (multi-batch ensemble)
    2. model_multiclass.pkl (standard single model)
    3. Other model files (*.pkl, *.joblib, *.sav)
    
    Parameters
    ----------
    model_dir : Path
        Directory containing model files
    verbose : bool, default=True
        Whether to print resolution information
        
    Returns
    -------
    Path
        Resolved model file path
    """
    if verbose:
        print(f"[EnsembleLoader] Resolving model in directory: {model_dir}")
    
    # Priority 1: Multi-batch ensemble model
    ensemble_path = model_dir / "model_multiclass_all_genes.pkl"
    if ensemble_path.exists():
        if verbose:
            print(f"[EnsembleLoader] Found multi-batch ensemble: {ensemble_path.name}")
        return ensemble_path
    
    # Priority 2: Standard single model
    single_path = model_dir / "model_multiclass.pkl"
    if single_path.exists():
        if verbose:
            print(f"[EnsembleLoader] Found standard model: {single_path.name}")
        return single_path
    
    # Priority 3: Search for any model files
    model_patterns = ["*.pkl", "*.joblib", "*.sav"]
    candidates = []
    
    for pattern in model_patterns:
        candidates.extend(model_dir.glob(pattern))
    
    if not candidates:
        raise FileNotFoundError(
            f"No model files found in directory: {model_dir}. "
            f"Expected: model_multiclass_all_genes.pkl or model_multiclass.pkl"
        )
    
    # Sort by preference and modification time
    def score_candidate(path: Path) -> tuple:
        name = path.name.lower()
        score = 0
        
        # Prefer ensemble models
        if "all_genes" in name:
            score += 100
        elif "multiclass" in name:
            score += 50
        elif "model" in name:
            score += 25
        
        # Prefer .pkl files
        if path.suffix == ".pkl":
            score += 10
        elif path.suffix == ".joblib":
            score += 5
        
        # Use modification time as tiebreaker
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
            
        return (score, mtime)
    
    best_candidate = max(candidates, key=score_candidate)
    
    if verbose:
        print(f"[EnsembleLoader] Selected best candidate: {best_candidate.name}")
    
    return best_candidate


def create_ensemble_info(model_path: Path) -> Dict[str, Any]:
    """
    Extract ensemble information for logging and debugging.
    
    Parameters
    ----------
    model_path : Path
        Path to model file
        
    Returns
    -------
    Dict[str, Any]
        Ensemble information dictionary
    """
    if not model_path.exists():
        return {"error": "Model file not found"}
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict) and model_data.get('type') == 'AllGenesBatchEnsemble':
            return {
                "type": "ensemble",
                "batch_count": model_data.get('batch_count', 0),
                "total_genes": model_data.get('total_genes', 0),
                "unique_genes": model_data.get('unique_genes', 0),
                "total_positions": model_data.get('total_positions', 0),
                "combination_method": model_data.get('combination_method', 'voting'),
                "feature_count": len(model_data.get('feature_names', []))
            }
        else:
            # Try to get info from single model
            model_type = type(model_data).__name__
            feature_count = len(getattr(model_data, 'feature_names', []))
            
            return {
                "type": "single",
                "model_class": model_type,
                "feature_count": feature_count,
                "has_calibration": "Calibrated" in model_type
            }
            
    except Exception as e:
        return {"error": str(e)}


# Convenience function for backward compatibility
def load_ensemble_model(model_path: Union[str, Path], **kwargs) -> Any:
    """Convenience wrapper for load_model_with_ensemble_support."""
    return load_model_with_ensemble_support(model_path, **kwargs)
