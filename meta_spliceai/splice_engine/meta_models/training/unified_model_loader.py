#!/usr/bin/env python3
"""
Unified Model Loader

This module provides a consistent interface for loading different types of meta-models,
ensuring that downstream inference workflows remain compatible regardless of how
the model was trained.

Supported Model Types:
1. Standard single meta-model (SigmoidEnsemble)
2. Calibrated meta-models (CalibratedSigmoidEnsemble, PerClassCalibratedSigmoidEnsemble)
3. Multi-instance consolidated models (ConsolidatedMetaModel)
4. Batch ensemble models (AllGenesBatchEnsemble)
5. Future model types (extensible architecture)

Key Design Principle:
All loaded models expose the same interface:
- predict_proba(X) -> np.ndarray of shape (n_samples, 3)
- predict(X) -> np.ndarray of shape (n_samples,)
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod


class UnifiedModelInterface(ABC):
    """
    Abstract base class defining the unified interface for all meta-models.
    
    This ensures that all model types provide consistent methods for inference,
    regardless of their internal implementation or training methodology.
    """
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities. Returns shape (n_samples, 3)."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels. Returns shape (n_samples,)."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names expected by this model."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get metadata about this model."""
        pass


class StandardModelWrapper(UnifiedModelInterface):
    """Wrapper for standard single meta-models."""
    
    def __init__(self, model: Any):
        self.model = model
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def get_feature_names(self) -> List[str]:
        if hasattr(self.model, 'feature_names'):
            return self.model.feature_names
        elif hasattr(self.model, 'get_feature_names'):
            return self.model.get_feature_names()
        else:
            return [f"feature_{i}" for i in range(getattr(self.model, 'n_features_', 0))]
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': 'StandardModel',
            'class_name': type(self.model).__name__,
            'n_features': len(self.get_feature_names()),
            'has_calibration': hasattr(self.model, 'calibrators'),
            'has_multiple_models': hasattr(self.model, 'models')
        }


class ConsolidatedModelWrapper(UnifiedModelInterface):
    """Wrapper for multi-instance consolidated models."""
    
    def __init__(self, model: Any):
        self.model = model
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def get_feature_names(self) -> List[str]:
        return self.model.feature_names
    
    def get_model_info(self) -> Dict[str, Any]:
        base_info = {
            'model_type': 'ConsolidatedMetaModel',
            'class_name': type(self.model).__name__,
            'n_features': len(self.get_feature_names())
        }
        
        if hasattr(self.model, 'get_instance_info'):
            base_info.update(self.model.get_instance_info())
            
        return base_info


class BatchEnsembleWrapper(UnifiedModelInterface):
    """Wrapper for batch ensemble models."""
    
    def __init__(self, model: Any):
        self.model = model
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Handle batch ensemble model prediction
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif isinstance(self.model, dict) and self.model.get('type') == 'AllGenesBatchEnsemble':
            # Use ensemble model wrapper
            from meta_spliceai.splice_engine.meta_models.workflows.inference.ensemble_model_loader import EnsembleModelWrapper
            wrapper = EnsembleModelWrapper(self.model)
            return wrapper.predict_proba(X)
        else:
            raise ValueError(f"Unknown batch ensemble model format: {type(self.model)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def get_feature_names(self) -> List[str]:
        if hasattr(self.model, 'feature_names'):
            return self.model.feature_names
        elif isinstance(self.model, dict):
            return self.model.get('feature_names', [])
        else:
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_type': 'BatchEnsemble',
            'class_name': type(self.model).__name__,
            'n_features': len(self.get_feature_names()),
            'is_batch_ensemble': True
        }


def load_unified_model(model_path: Union[str, Path]) -> UnifiedModelInterface:
    """
    Load any type of meta-model and return a unified interface.
    
    This function automatically detects the model type and wraps it
    in the appropriate interface, ensuring consistent behavior for
    downstream inference workflows.
    
    Parameters
    ----------
    model_path : str or Path
        Path to the saved model file (.pkl)
        
    Returns
    -------
    UnifiedModelInterface
        Wrapped model with consistent interface
        
    Examples
    --------
    >>> model = load_unified_model("results/run1/model_multiclass.pkl")
    >>> proba = model.predict_proba(X_test)  # Always returns (n_samples, 3)
    >>> pred = model.predict(X_test)         # Always returns (n_samples,)
    >>> features = model.get_feature_names() # Always returns List[str]
    >>> info = model.get_model_info()        # Always returns Dict[str, Any]
    """
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Detect model type and wrap appropriately
    model_type = type(model).__name__
    
    if model_type == "ConsolidatedMetaModel":
        # Multi-instance consolidated model
        return ConsolidatedModelWrapper(model)
    
    elif model_type in ["SigmoidEnsemble", "CalibratedSigmoidEnsemble", "PerClassCalibratedSigmoidEnsemble"]:
        # Standard single meta-models
        return StandardModelWrapper(model)
    
    elif isinstance(model, dict) and model.get('type') == 'AllGenesBatchEnsemble':
        # Batch ensemble model
        return BatchEnsembleWrapper(model)
    
    elif hasattr(model, 'predict_proba') and hasattr(model, 'predict'):
        # Generic model with prediction interface
        return StandardModelWrapper(model)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Cannot create unified interface.")


def get_model_compatibility_info(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get compatibility information about a model without fully loading it.
    
    This is useful for checking model compatibility before loading in
    inference workflows.
    """
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        return {"error": f"Model file not found: {model_path}"}
    
    try:
        # Try to load and analyze the model
        unified_model = load_unified_model(model_path)
        
        info = unified_model.get_model_info()
        info.update({
            'model_path': str(model_path),
            'file_size_mb': model_path.stat().st_size / (1024*1024),
            'is_compatible': True,
            'interface_version': '1.0'
        })
        
        return info
        
    except Exception as e:
        return {
            'model_path': str(model_path),
            'is_compatible': False,
            'error': str(e),
            'interface_version': '1.0'
        }


def create_model_registry(results_dir: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Create a registry of all available models in a results directory.
    
    This helps with model management and selection for inference.
    """
    
    results_dir = Path(results_dir)
    registry = {}
    
    # Find all model files
    model_files = list(results_dir.glob("**/model_multiclass.pkl"))
    
    for model_file in model_files:
        run_name = model_file.parent.name
        
        # Get model compatibility info
        compat_info = get_model_compatibility_info(model_file)
        
        # Add run-specific information
        run_info = {
            'run_directory': str(model_file.parent),
            'model_file': str(model_file),
            'run_name': run_name,
            **compat_info
        }
        
        # Check for additional metadata files
        metadata_files = {
            'training_metadata': model_file.parent / "complete_training_results.json",
            'cv_metrics': model_file.parent / "gene_cv_metrics.csv",
            'feature_manifest': model_file.parent / "feature_manifest.csv",
            'consolidation_info': model_file.parent / "consolidation_info.json"
        }
        
        for meta_name, meta_path in metadata_files.items():
            run_info[f'has_{meta_name}'] = meta_path.exists()
            if meta_path.exists() and meta_name == 'consolidation_info':
                # This indicates a multi-instance model
                run_info['is_multi_instance'] = True
        
        registry[run_name] = run_info
    
    return registry


if __name__ == "__main__":
    # Test the unified model loader
    import argparse
    
    parser = argparse.ArgumentParser(description="Test unified model loader")
    parser.add_argument("--model-path", help="Path to model file")
    parser.add_argument("--test-data", help="Optional test data for prediction test")
    parser.add_argument("--registry-dir", help="Create model registry for directory")
    
    args = parser.parse_args()
    
    if not args.model_path and not args.registry_dir:
        parser.error("Either --model-path or --registry-dir must be specified")
    
    if args.registry_dir:
        print(f"Creating model registry for: {args.registry_dir}")
        registry = create_model_registry(args.registry_dir)
        
        print(f"\nFound {len(registry)} models:")
        for run_name, info in registry.items():
            print(f"  {run_name}:")
            print(f"    Type: {info.get('model_type', 'Unknown')}")
            print(f"    Features: {info.get('n_features', 'Unknown')}")
            print(f"    Compatible: {info.get('is_compatible', False)}")
            if info.get('is_multi_instance', False):
                print(f"    Multi-instance: Yes")
    
    else:
        print(f"Testing model loading: {args.model_path}")
        
        # Test compatibility check
        compat_info = get_model_compatibility_info(args.model_path)
        print(f"Compatibility info: {compat_info}")
        
        if compat_info.get('is_compatible', False):
            # Test model loading
            model = load_unified_model(args.model_path)
            print(f"Model loaded successfully!")
            print(f"Model info: {model.get_model_info()}")
            print(f"Feature names: {len(model.get_feature_names())} features")
            
            # Test prediction if test data provided
            if args.test_data:
                test_df = pd.read_csv(args.test_data)
                if len(test_df) > 0:
                    # Take first few rows for testing
                    X_test = test_df.iloc[:5].values
                    
                    try:
                        proba = model.predict_proba(X_test)
                        pred = model.predict(X_test)
                        
                        print(f"Prediction test successful!")
                        print(f"  Probabilities shape: {proba.shape}")
                        print(f"  Predictions shape: {pred.shape}")
                        print(f"  Sample probabilities: {proba[0]}")
                        print(f"  Sample prediction: {pred[0]}")
                        
                    except Exception as e:
                        print(f"Prediction test failed: {e}")
        else:
            print(f"Model is not compatible: {compat_info.get('error', 'Unknown error')}")
