#!/usr/bin/env python3
"""
Model Resource Manager for Inference Workflow

This module provides systematic management of model-related resources including:
- Feature manifests
- Model files (trained meta-models)
- Training dataset manifests
- Model metadata and configurations

This complements the data_resource_manager.py which handles genomic data resources.

DIRECTORY STRUCTURE FOR MODEL RESOURCES (FLEXIBLE):
1. Model directory: results/{model_run_name}/
   - model_multiclass.pkl (trained meta-model)
   - feature_manifest.csv (feature schema)
   - model_metadata.json (model configuration)
   - training_config.json (training parameters)

2. Training dataset directory: train_{dataset_name}/
   - master/ (training data in parquet format)
   - gene_manifest.csv (genes used in training)
   - training_metadata.json (dataset configuration)

3. Inference artifacts: test_{dataset_name}/
   - master/ (test data in parquet format)
   - gene_manifest.csv (genes processed for inference)

EXAMPLES:
- Model: results/gene_cv_pc_1000_3mers_run_4/ (1000 genes, high error counts)
- Model: results/gene_cv_pc_5000_3mers_run_2/ (5000 genes, better coverage)
- Model: results/gene_cv_comprehensive_run_1/ (comprehensive training set)
- Training: train_pc_1000_3mers/, train_pc_5000_3mers/, train_comprehensive/
- Test: test_pc_1000_3mers/, test_pc_5000_3mers/, test_comprehensive/
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import json

logger = logging.getLogger(__name__)


class ModelResourceManager:
    """
    Systematic manager for model-related resources in the inference workflow.
    
    This manager provides a centralized way to locate:
    - Trained meta-models and their metadata
    - Feature manifests and training schemas
    - Training and test dataset manifests
    - Model configurations and parameters
    """
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """
        Initialize the model resource manager.
        
        Parameters
        ----------
        project_root : str or Path, optional
            Project root directory. If None, will attempt to auto-detect.
        """
        if project_root is None:
            project_root = self._find_project_root()
        
        self.project_root = Path(project_root).resolve()
        
        # Define expected model file patterns
        self.model_file_patterns = {
            "multiclass": ["model_multiclass.pkl", "model_multiclass.joblib"],
            "binary": ["model_binary.pkl", "model_binary.joblib"],
            "best": ["best_model.pkl", "best_model.joblib"],
            "final": ["final_model.pkl", "final_model.joblib"],
            "calibrated": ["calibrated_model.pkl", "calibrated_model.joblib"]
        }
        
        # Define expected manifest patterns
        self.manifest_patterns = {
            "feature_manifest": "feature_manifest.csv",
            "gene_manifest": "gene_manifest.csv",
            "training_manifest": "training_manifest.json",
            "model_metadata": "model_metadata.json"
        }
        
        logger.debug(f"Initialized ModelResourceManager with root: {self.project_root}")
    
    def _find_project_root(self) -> Path:
        """Auto-detect project root by looking for characteristic files."""
        current_dir = Path(__file__).resolve()
        
        # Look for characteristic project files
        key_files = [
            "meta_spliceai/__init__.py",
            "results/",
            "train_pc_1000_3mers/"
        ]
        
        # Start from current directory and go up
        for parent in [current_dir] + list(current_dir.parents):
            for key_file in key_files:
                if (parent / key_file).exists():
                    logger.debug(f"Found project root: {parent}")
                    return parent
        
        # Fallback: assume current working directory
        cwd = Path.cwd()
        logger.warning(f"Could not auto-detect project root, using: {cwd}")
        return cwd
    
    def locate_model_directory(self, model_path: Union[str, Path]) -> Path:
        """
        Locate the model directory from a model path.
        
        Parameters
        ----------
        model_path : str or Path
            Path to model file or directory
            
        Returns
        -------
        Path
            Directory containing the model and its resources
        """
        model_path = Path(model_path)
        
        if model_path.is_file():
            return model_path.parent
        elif model_path.is_dir():
            return model_path
        else:
            # Try to resolve relative to project root
            resolved_path = self.project_root / model_path
            if resolved_path.is_file():
                return resolved_path.parent
            elif resolved_path.is_dir():
                return resolved_path
            else:
                raise FileNotFoundError(f"Model path not found: {model_path}")
    
    def get_feature_manifest_path(self, model_path: Union[str, Path]) -> Optional[Path]:
        """
        Get the path to the feature manifest for a given model.
        
        Parameters
        ----------
        model_path : str or Path
            Path to model file or directory
            
        Returns
        -------
        Path or None
            Path to feature manifest if found
        """
        try:
            model_dir = self.locate_model_directory(model_path)
            manifest_path = model_dir / self.manifest_patterns["feature_manifest"]
            
            if manifest_path.exists():
                logger.debug(f"Found feature manifest: {manifest_path}")
                return manifest_path
            else:
                logger.warning(f"Feature manifest not found: {manifest_path}")
                return None
        except Exception as e:
            logger.error(f"Error locating feature manifest: {e}")
            return None
    
    def load_feature_schema(self, model_path: Union[str, Path]) -> Optional[List[str]]:
        """
        Load feature schema from model's feature manifest.
        
        Parameters
        ----------
        model_path : str or Path
            Path to model file or directory
            
        Returns
        -------
        List[str] or None
            List of feature names in training order, or None if not found
        """
        manifest_path = self.get_feature_manifest_path(model_path)
        if not manifest_path:
            return None
        
        try:
            # Try different column name formats
            manifest_df = pd.read_csv(manifest_path)
            
            # Check for different possible column names
            feature_columns = ["feature", "feature_name", "column_name", "name"]
            
            for col in feature_columns:
                if col in manifest_df.columns:
                    features = manifest_df[col].tolist()
                    logger.info(f"✅ Loaded {len(features)} features from {manifest_path}")
                    return features
            
            # If no standard column found, use first column
            if len(manifest_df.columns) > 0:
                features = manifest_df.iloc[:, 0].tolist()
                logger.warning(f"Using first column as features: {len(features)} features from {manifest_path}")
                return features
            
            logger.error(f"No feature columns found in {manifest_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading feature schema: {e}")
            return None
    
    def get_model_metadata_path(self, model_path: Union[str, Path]) -> Optional[Path]:
        """Get the path to model metadata."""
        try:
            model_dir = self.locate_model_directory(model_path)
            metadata_path = model_dir / self.manifest_patterns["model_metadata"]
            return metadata_path if metadata_path.exists() else None
        except Exception:
            return None
    
    def get_training_dataset_info(self, training_dataset_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a training dataset.
        
        Parameters
        ----------
        training_dataset_path : str or Path
            Path to training dataset directory
            
        Returns
        -------
        Dict[str, Any]
            Information about the training dataset
        """
        dataset_path = Path(training_dataset_path)
        if not dataset_path.is_absolute():
            dataset_path = self.project_root / dataset_path
        
        info = {
            "path": str(dataset_path),
            "exists": dataset_path.exists(),
            "master_dir": None,
            "gene_manifest": None,
            "parquet_files": [],
            "total_genes": 0
        }
        
        if not dataset_path.exists():
            return info
        
        # Check for master directory
        master_dir = dataset_path / "master"
        if master_dir.exists():
            info["master_dir"] = str(master_dir)
            
            # Find parquet files
            parquet_files = list(master_dir.glob("*.parquet"))
            info["parquet_files"] = [str(f) for f in parquet_files]
            
            # Check for gene manifest
            gene_manifest = master_dir / "gene_manifest.csv"
            if gene_manifest.exists():
                info["gene_manifest"] = str(gene_manifest)
                try:
                    manifest_df = pd.read_csv(gene_manifest)
                    if "gene_id" in manifest_df.columns:
                        info["total_genes"] = len(manifest_df["gene_id"].unique())
                except Exception as e:
                    logger.warning(f"Could not read gene manifest: {e}")
        
        return info
    
    def get_test_dataset_directory(self, training_dataset_path: Union[str, Path]) -> Path:
        """
        Get the corresponding test dataset directory for a training dataset.
        
        Follows the convention: train_pc_1000_3mers -> test_pc_1000_3mers
        
        Parameters
        ----------
        training_dataset_path : str or Path
            Path to training dataset
            
        Returns
        -------
        Path
            Path to corresponding test dataset directory
        """
        dataset_path = Path(training_dataset_path)
        dataset_name = dataset_path.name
        
        # Convert train_* to test_*
        if dataset_name.startswith("train_"):
            test_name = dataset_name.replace("train_", "test_", 1)
        else:
            test_name = f"test_{dataset_name}"
        
        test_dir = self.project_root / test_name
        return test_dir
    
    def validate_model_resources(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate that all expected model resources are available.
        
        Parameters
        ----------
        model_path : str or Path
            Path to model file or directory
            
        Returns
        -------
        Dict[str, Any]
            Validation results for model resources
        """
        validation = {
            "model_directory_found": False,
            "model_file_found": False,
            "feature_manifest_found": False,
            "model_metadata_found": False,
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Check model directory
            model_dir = self.locate_model_directory(model_path)
            validation["model_directory_found"] = True
            validation["model_directory"] = str(model_dir)
            
            # Check for model file
            if Path(model_path).is_file():
                validation["model_file_found"] = True
                validation["model_file"] = str(model_path)
            else:
                # Look for model files in directory
                model_files = []
                for pattern_list in self.model_file_patterns.values():
                    for pattern in pattern_list:
                        model_file = model_dir / pattern
                        if model_file.exists():
                            model_files.append(model_file)
                
                if model_files:
                    validation["model_file_found"] = True
                    validation["model_file"] = str(model_files[0])  # Use first found
                    if len(model_files) > 1:
                        validation["recommendations"].append(f"Multiple model files found: {[str(f) for f in model_files]}")
                else:
                    validation["issues"].append("No model files found in directory")
            
            # Check for feature manifest
            feature_manifest = self.get_feature_manifest_path(model_path)
            if feature_manifest:
                validation["feature_manifest_found"] = True
                validation["feature_manifest"] = str(feature_manifest)
                
                # Validate manifest content
                try:
                    features = self.load_feature_schema(model_path)
                    if features:
                        validation["feature_count"] = len(features)
                        validation["sample_features"] = features[:5]  # First 5 features
                    else:
                        validation["issues"].append("Feature manifest exists but could not load features")
                except Exception as e:
                    validation["issues"].append(f"Error reading feature manifest: {e}")
            else:
                validation["issues"].append("Feature manifest not found")
            
            # Check for model metadata
            metadata_path = self.get_model_metadata_path(model_path)
            if metadata_path:
                validation["model_metadata_found"] = True
                validation["model_metadata"] = str(metadata_path)
            
        except Exception as e:
            validation["issues"].append(f"Error validating model resources: {e}")
        
        # Generate summary
        critical_resources = ["model_directory_found", "model_file_found", "feature_manifest_found"]
        validation["all_critical_found"] = all(validation.get(key, False) for key in critical_resources)
        
        if validation["all_critical_found"]:
            validation["summary"] = "✅ All critical model resources found"
        else:
            missing = [key for key in critical_resources if not validation.get(key, False)]
            validation["summary"] = f"❌ Missing critical resources: {missing}"
        
        return validation
    
    def create_model_config(self, model_path: Union[str, Path], 
                           training_dataset_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Create a comprehensive model configuration for inference.
        
        Parameters
        ----------
        model_path : str or Path
            Path to model file or directory
        training_dataset_path : str or Path, optional
            Path to training dataset directory
            
        Returns
        -------
        Dict[str, Any]
            Complete model configuration for inference
        """
        config = {
            "model_resources": {},
            "training_resources": {},
            "paths": {},
            "validation": {}
        }
        
        # Validate model resources
        model_validation = self.validate_model_resources(model_path)
        config["validation"]["model"] = model_validation
        
        if model_validation["all_critical_found"]:
            config["model_resources"] = {
                "model_directory": model_validation["model_directory"],
                "model_file": model_validation["model_file"],
                "feature_manifest": model_validation["feature_manifest"],
                "feature_schema": self.load_feature_schema(model_path)
            }
            
            # Add model metadata if available
            if model_validation.get("model_metadata_found"):
                config["model_resources"]["model_metadata"] = model_validation["model_metadata"]
        
        # Validate training dataset if provided
        if training_dataset_path:
            training_info = self.get_training_dataset_info(training_dataset_path)
            config["training_resources"] = training_info
            config["validation"]["training"] = {
                "dataset_found": training_info["exists"],
                "master_dir_found": training_info["master_dir"] is not None,
                "gene_manifest_found": training_info["gene_manifest"] is not None,
                "total_genes": training_info["total_genes"]
            }
            
            # Get corresponding test dataset directory
            test_dir = self.get_test_dataset_directory(training_dataset_path)
            config["paths"]["test_dataset_directory"] = str(test_dir)
        
        # Add systematic paths
        config["paths"].update({
            "project_root": str(self.project_root),
            "model_directory": config["model_resources"].get("model_directory"),
            "training_dataset_directory": str(training_dataset_path) if training_dataset_path else None
        })
        
        return config
    
    def get_training_schema_path(self, model_path: Union[str, Path]) -> Optional[Path]:
        """
        Get the training schema path (feature manifest) for a model.
        
        This is the systematic way to find feature ordering for inference.
        
        Parameters
        ----------
        model_path : str or Path
            Path to model file or directory
            
        Returns
        -------
        Path or None
            Path to training schema (feature manifest)
        """
        return self.get_feature_manifest_path(model_path)
    
    def get_test_dataset_directory(self, training_dataset_path: Union[str, Path]) -> Path:
        """Get the test dataset directory corresponding to a training dataset."""
        dataset_path = Path(training_dataset_path)
        dataset_name = dataset_path.name
        
        # Convert train_* to test_*
        if dataset_name.startswith("train_"):
            test_name = dataset_name.replace("train_", "test_", 1)
        else:
            test_name = f"test_{dataset_name}"
        
        test_dir = self.project_root / test_name
        return test_dir


def create_model_resource_manager(project_root: Optional[Union[str, Path]] = None) -> ModelResourceManager:
    """
    Factory function to create a model resource manager.
    
    Parameters
    ----------
    project_root : str or Path, optional
        Project root directory. If None, will auto-detect.
        
    Returns
    -------
    ModelResourceManager
        Configured model resource manager
    """
    return ModelResourceManager(project_root=project_root)


def resolve_model_and_schema_paths(model_path: Union[str, Path], 
                                 training_dataset_path: Optional[Union[str, Path]] = None) -> Dict[str, Optional[str]]:
    """
    Convenience function to resolve all model and schema paths systematically.
    
    This is the main entry point for other workflows to get all necessary paths.
    
    Parameters
    ----------
    model_path : str or Path
        Path to model file or directory
    training_dataset_path : str or Path, optional
        Path to training dataset directory
        
    Returns
    -------
    Dict[str, Optional[str]]
        Dictionary with resolved paths:
        - model_file: Path to actual model file
        - model_directory: Directory containing model
        - feature_manifest: Path to feature manifest
        - training_schema: Path to training schema (same as feature_manifest)
        - training_dataset: Path to training dataset
        - test_dataset: Path to corresponding test dataset
    """
    manager = create_model_resource_manager()
    
    try:
        # Get model directory and validate resources
        model_dir = manager.locate_model_directory(model_path)
        validation = manager.validate_model_resources(model_path)
        
        paths = {
            "model_file": validation.get("model_file"),
            "model_directory": str(model_dir),
            "feature_manifest": validation.get("feature_manifest"),
            "training_schema": validation.get("feature_manifest"),  # Same as feature_manifest
            "training_dataset": str(training_dataset_path) if training_dataset_path else None,
            "test_dataset": str(manager.get_test_dataset_directory(training_dataset_path)) if training_dataset_path else None
        }
        
        # Log the resolved paths
        logger.info(f"📋 Model Resource Resolution:")
        for key, value in paths.items():
            if value:
                logger.info(f"   {key}: {value}")
            else:
                logger.warning(f"   {key}: Not found")
        
        return paths
        
    except Exception as e:
        logger.error(f"Error resolving model and schema paths: {e}")
        return {
            "model_file": None,
            "model_directory": None,
            "feature_manifest": None,
            "training_schema": None,
            "training_dataset": None,
            "test_dataset": None,
            "error": str(e)
        }


def main():
    """Command-line interface for model resource management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Resource Manager")
    parser.add_argument("model_path", help="Path to model file or directory")
    parser.add_argument("--training-dataset", help="Path to training dataset directory")
    parser.add_argument("--validate", action="store_true", help="Validate model resources")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    manager = create_model_resource_manager()
    
    if args.validate:
        # Validate model resources
        validation = manager.validate_model_resources(args.model_path)
        
        print(f"\n🔍 Model Resource Validation")
        print(f"Model path: {args.model_path}")
        print(f"Summary: {validation['summary']}")
        
        if validation.get("issues"):
            print(f"\nIssues:")
            for issue in validation["issues"]:
                print(f"  ❌ {issue}")
        
        if validation.get("recommendations"):
            print(f"\nRecommendations:")
            for rec in validation["recommendations"]:
                print(f"  💡 {rec}")
        
        # Show feature schema if available
        features = manager.load_feature_schema(args.model_path)
        if features:
            print(f"\nFeature Schema ({len(features)} features):")
            for i, feature in enumerate(features[:10]):
                print(f"  {i+1:3d}. {feature}")
            if len(features) > 10:
                print(f"  ... and {len(features)-10} more features")
    
    else:
        # Resolve paths
        paths = resolve_model_and_schema_paths(args.model_path, args.training_dataset)
        
        print(f"\n📋 Resolved Model and Schema Paths")
        for key, value in paths.items():
            if value:
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: Not found")


if __name__ == "__main__":
    main()
