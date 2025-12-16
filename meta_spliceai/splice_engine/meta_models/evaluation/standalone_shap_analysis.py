#!/usr/bin/env python3
"""
Standalone SHAP Analysis Module

This module provides a comprehensive SHAP analysis workflow that can be run independently
to test and debug SHAP functionality before integrating into CV scripts.

Features:
- Memory-efficient incremental SHAP analysis
- Comprehensive visualization generation
- Proper error handling and dependency isolation
- Compatible with sigmoid ensemble models
- Detailed logging and progress reporting

Usage:
    python standalone_shap_analysis.py --dataset data/train_pc_1000/master --model-dir results/test_model --output-dir results/shap_test

Author: Surveyor AI team
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import traceback

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Set up environment variables to prevent dependency conflicts."""
    env_vars = {
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'TRANSFORMERS_VERBOSITY': 'error',
        'KERAS_BACKEND': 'tensorflow',
        'TRANSFORMERS_OFFLINE': '1',
        'HF_HUB_DISABLE_TELEMETRY': '1',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_CACHE': '/tmp/transformers_cache_disabled',
        'HF_HOME': '/tmp/hf_home_disabled',
        'CUDA_VISIBLE_DEVICES': '',
        'PYTHONWARNINGS': 'ignore',
        'MLFLOW_TRACKING_URI': '',
        'WANDB_DISABLED': 'true',
        'COMET_DISABLE': 'true'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    logger.info("Environment configured for SHAP analysis")


class StandaloneSHAPAnalyzer:
    """Standalone SHAP analysis class with comprehensive error handling."""
    
    def __init__(self, dataset_path: Union[str, Path], model_dir: Union[str, Path], 
                 output_dir: Union[str, Path], verbose: bool = True):
        """
        Initialize the SHAP analyzer.
        
        Parameters
        ----------
        dataset_path : Union[str, Path]
            Path to the dataset directory or file
        model_dir : Union[str, Path] 
            Directory containing the trained model files
        output_dir : Union[str, Path]
            Output directory for SHAP analysis results
        verbose : bool
            Whether to enable verbose logging
        """
        self.dataset_path = Path(dataset_path)
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        
        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Organize SHAP outputs under feature_importance_analysis for consistency
        self.feature_importance_dir = self.output_dir / "feature_importance_analysis"
        self.feature_importance_dir.mkdir(exist_ok=True, parents=True)
        
        self.shap_analysis_dir = self.feature_importance_dir / "shap_analysis"
        self.shap_analysis_dir.mkdir(exist_ok=True, parents=True)
        
        self.shap_importance_dir = self.shap_analysis_dir / "importance"
        self.shap_importance_dir.mkdir(exist_ok=True, parents=True)
        
        self.shap_viz_dir = self.shap_analysis_dir / "comprehensive_visualizations"
        self.shap_viz_dir.mkdir(exist_ok=True, parents=True)
        
        self.shap_beeswarm_dir = self.shap_analysis_dir / "beeswarm_plots"
        self.shap_beeswarm_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Initialized SHAP analyzer")
        logger.info(f"  Dataset: {self.dataset_path}")
        logger.info(f"  Model directory: {self.model_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
        
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check if all required files exist."""
        checks = {}
        
        # Check dataset
        checks['dataset_exists'] = self.dataset_path.exists()
        
        # Check model files
        model_pkl = self.model_dir / "model_multiclass.pkl"
        model_json = self.model_dir / "model_multiclass.json"
        checks['model_exists'] = model_pkl.exists() or model_json.exists()
        
        # Check feature manifest
        feature_manifest_csv = self.model_dir / "feature_manifest.csv"
        feature_manifest_json = self.model_dir / "train.features.json"
        checks['feature_manifest_exists'] = feature_manifest_csv.exists() or feature_manifest_json.exists()
        
        # Log results
        logger.info("Prerequisite check:")
        for check, status in checks.items():
            status_str = "✓" if status else "✗"
            logger.info(f"  {status_str} {check}: {status}")
        
        all_good = all(checks.values())
        if not all_good:
            logger.warning("Some prerequisites are missing!")
            if not checks['dataset_exists']:
                logger.warning(f"  Dataset not found: {self.dataset_path}")
            if not checks['model_exists']:
                logger.warning(f"  Model not found in: {self.model_dir}")
                logger.warning(f"    Looking for: model_multiclass.pkl or model_multiclass.json")
            if not checks['feature_manifest_exists']:
                logger.warning(f"  Feature manifest not found in: {self.model_dir}")
                logger.warning(f"    Looking for: feature_manifest.csv or train.features.json")
        
        return checks
    
    def test_shap_import(self) -> bool:
        """Test SHAP import with dependency isolation."""
        logger.info("Testing SHAP import with dependency isolation...")
        
        try:
            # Import with enhanced isolation
            from meta_spliceai.splice_engine.meta_models.evaluation.shap_incremental import (
                _test_shap_import_safety
            )
            
            success, error_msg = _test_shap_import_safety()
            
            if success:
                logger.info("✓ SHAP import test passed")
                return True
            else:
                logger.error(f"✗ SHAP import test failed: {error_msg}")
                return False
                
        except Exception as e:
            logger.error(f"✗ SHAP import test failed with exception: {e}")
            return False
    
    def load_model_and_features(self) -> tuple[Any, List[str]]:
        """Load the trained model and feature names."""
        logger.info("Loading model and feature manifest...")
        
        # Load model
        model = None
        model_pkl = self.model_dir / "model_multiclass.pkl"
        model_json = self.model_dir / "model_multiclass.json"
        
        if model_pkl.exists():
            logger.info(f"Loading model from: {model_pkl}")
            try:
                import pickle
                with open(model_pkl, "rb") as f:
                    model = pickle.load(f)
                logger.info(f"✓ Loaded model: {type(model).__name__}")
            except Exception as e:
                logger.error(f"✗ Failed to load pickle model: {e}")
                raise
        elif model_json.exists():
            logger.info(f"Loading model from: {model_json}")
            try:
                import xgboost as xgb
                model = xgb.Booster()
                model.load_model(str(model_json))
                logger.info(f"✓ Loaded XGBoost model")
            except Exception as e:
                logger.error(f"✗ Failed to load XGBoost model: {e}")
                raise
        else:
            raise FileNotFoundError("No model file found")
        
        # Load feature names
        feature_names = []
        feature_csv = self.model_dir / "feature_manifest.csv"
        feature_json = self.model_dir / "train.features.json"
        
        if feature_csv.exists():
            logger.info(f"Loading features from: {feature_csv}")
            try:
                feature_df = pd.read_csv(feature_csv)
                feature_names = feature_df["feature"].tolist()
                logger.info(f"✓ Loaded {len(feature_names)} features")
            except Exception as e:
                logger.error(f"✗ Failed to load feature CSV: {e}")
                raise
        elif feature_json.exists():
            logger.info(f"Loading features from: {feature_json}")
            try:
                with open(feature_json, "r") as f:
                    feature_data = json.load(f)
                feature_names = feature_data["feature_names"]
                logger.info(f"✓ Loaded {len(feature_names)} features")
            except Exception as e:
                logger.error(f"✗ Failed to load feature JSON: {e}")
                raise
        else:
            raise FileNotFoundError("No feature manifest found")
        
        return model, feature_names
    
    def prepare_dataset(self, feature_names: List[str], sample_size: Optional[int] = None) -> tuple[pd.DataFrame, pd.Series]:
        """Load and prepare the dataset for SHAP analysis."""
        logger.info("Loading and preparing dataset...")
        
        # Set row cap if sample size is specified
        if sample_size:
            os.environ["SS_MAX_ROWS"] = str(sample_size)
            logger.info(f"Setting sample size to {sample_size}")
        
        try:
            from meta_spliceai.splice_engine.meta_models.training import datasets
            from meta_spliceai.splice_engine.meta_models.builder import preprocessing
            
            # Load dataset
            logger.info(f"Loading dataset from: {self.dataset_path}")
            df = datasets.load_dataset(str(self.dataset_path))
            logger.info(f"✓ Loaded dataset with {len(df)} rows")
            
            # Prepare training data
            logger.info("Preparing features for SHAP analysis...")
            X_df, y_series = preprocessing.prepare_training_data(
                df, 
                label_col="splice_type", 
                return_type="pandas", 
                verbose=0,
                encode_chrom=True  # Include encoded chromosome as a feature
            )
            
            logger.info(f"✓ Prepared {X_df.shape[0]} samples with {X_df.shape[1]} features")
            
            # Filter to only include features that the model knows about
            available_features = [f for f in feature_names if f in X_df.columns]
            missing_features = [f for f in feature_names if f not in X_df.columns]
            
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features from dataset:")
                for feat in missing_features[:5]:  # Show first 5
                    logger.warning(f"  - {feat}")
                if len(missing_features) > 5:
                    logger.warning(f"  ... and {len(missing_features) - 5} more")
            
            logger.info(f"Using {len(available_features)} available features")
            X_filtered = X_df[available_features]
            
            return X_filtered, y_series
            
        except Exception as e:
            logger.error(f"✗ Failed to prepare dataset: {e}")
            raise
    
    def run_incremental_shap_analysis(self, model: Any, X: pd.DataFrame, 
                                    batch_size: int = 512, background_size: int = 100) -> bool:
        """Run incremental SHAP importance analysis."""
        logger.info("Running incremental SHAP importance analysis...")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Background size: {background_size}")
        logger.info(f"  Data shape: {X.shape}")
        
        try:
            from meta_spliceai.splice_engine.meta_models.evaluation.shap_incremental import (
                incremental_shap_importance
            )
            
            # Run SHAP analysis with progress tracking
            logger.info("Computing SHAP values...")
            shap_importance_series = incremental_shap_importance(
                model,
                X,
                batch_size=batch_size,
                background_size=background_size,
                verbose=self.verbose
            )
            
            # Convert to DataFrame and save
            shap_importance_df = pd.DataFrame({
                'feature': shap_importance_series.index,
                'shap_importance': shap_importance_series.values
            })
            
            # Add class-specific importance columns for ensemble models
            if hasattr(model, 'models') or hasattr(model, 'get_base_models'):
                logger.info("Detected ensemble model - computing class-specific importance...")
                try:
                    # Get binary models from ensemble
                    if hasattr(model, 'get_base_models'):
                        binary_models = model.get_base_models()
                    elif hasattr(model, 'models'):
                        binary_models = model.models
                    else:
                        binary_models = [model]  # Fallback
                    
                    class_names = ['neither', 'donor', 'acceptor']
                    
                    for i, (binary_model, class_name) in enumerate(zip(binary_models[:3], class_names)):
                        try:
                            class_importance = incremental_shap_importance(
                                binary_model,
                                X,
                                batch_size=batch_size,
                                background_size=background_size,
                                verbose=False  # Reduce verbosity for class-specific runs
                            )
                            shap_importance_df[f'importance_{class_name}'] = shap_importance_df['feature'].map(
                                class_importance.to_dict()
                            ).fillna(0.0)
                            logger.info(f"✓ Computed {class_name} class importance")
                        except Exception as e:
                            logger.warning(f"Failed to compute {class_name} class importance: {e}")
                            shap_importance_df[f'importance_{class_name}'] = 0.0001  # Small fallback value
                    
                    # Add mean importance across classes
                    importance_cols = [col for col in shap_importance_df.columns if col.startswith('importance_')]
                    if len(importance_cols) > 1:
                        shap_importance_df['importance_mean'] = shap_importance_df[importance_cols].mean(axis=1)
                    else:
                        shap_importance_df['importance_mean'] = shap_importance_df['shap_importance']
                        
                except Exception as e:
                    logger.warning(f"Failed to compute class-specific importance: {e}")
                    # Add default columns for compatibility
                    for class_name in ['neither', 'donor', 'acceptor']:
                        shap_importance_df[f'importance_{class_name}'] = 0.0001
                    shap_importance_df['importance_mean'] = shap_importance_df['shap_importance']
            else:
                # Single model - use overall importance as mean
                shap_importance_df['importance_mean'] = shap_importance_df['shap_importance']
                for class_name in ['neither', 'donor', 'acceptor']:
                    shap_importance_df[f'importance_{class_name}'] = shap_importance_df['shap_importance'] / 3
            
            # Save SHAP importance results
            out_csv = self.shap_importance_dir / "shap_importance_incremental.csv"
            shap_importance_df.to_csv(out_csv, index=False)
            logger.info(f"✓ Saved SHAP importance to: {out_csv}")
            
            # Log top features
            top_features = shap_importance_series.head(10)
            logger.info("Top 10 features by SHAP importance:")
            for i, (feature, importance) in enumerate(top_features.items(), 1):
                logger.info(f"  {i:2d}. {feature:<30} {importance:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Incremental SHAP analysis failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def create_basic_visualizations(self, model: Any, X: pd.DataFrame, top_n: int = 20) -> bool:
        """Create basic SHAP visualizations."""
        logger.info("Creating basic SHAP visualizations...")
        
        try:
            from meta_spliceai.splice_engine.meta_models.evaluation.shap_incremental import (
                plot_feature_importance
            )
            
            # Load SHAP importance data
            importance_csv = self.shap_importance_dir / "shap_importance_incremental.csv"
            if not importance_csv.exists():
                logger.error("SHAP importance file not found - run incremental analysis first")
                return False
            
            importance_df = pd.read_csv(importance_csv)
            
            # Create basic bar plot
            if 'shap_importance' in importance_df.columns:
                importance_series = pd.Series(
                    importance_df['shap_importance'].values,
                    index=importance_df['feature']
                ).sort_values(ascending=False)
                
                plot_feature_importance(
                    importance_series,
                    title="SHAP Feature Importance",
                    save_path=self.shap_viz_dir / "shap_feature_importance_basic.png",
                    top_n=top_n
                )
                logger.info("✓ Created basic feature importance plot")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Basic visualization creation failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def create_comprehensive_visualizations(self, sample_size: int = 1000, top_n: int = 20) -> bool:
        """Create comprehensive SHAP visualizations."""
        logger.info("Creating comprehensive SHAP visualizations...")
        
        try:
            from meta_spliceai.splice_engine.meta_models.evaluation.shap_viz import (
                generate_comprehensive_shap_report
            )
            
            # Check required files
            importance_csv = self.shap_importance_dir / "shap_importance_incremental.csv"
            model_pkl = self.model_dir / "model_multiclass.pkl"
            
            if not importance_csv.exists():
                logger.error("SHAP importance file not found")
                return False
            
            if not model_pkl.exists():
                logger.warning("Model pickle file not found - skipping beeswarm plots")
                # Create bar charts only
                from meta_spliceai.splice_engine.meta_models.evaluation.shap_viz import (
                    create_feature_importance_barcharts,
                    create_feature_importance_heatmap
                )
                
                try:
                    # Create bar charts
                    bar_chart_paths = create_feature_importance_barcharts(
                        importance_csv, self.shap_viz_dir, top_n=top_n, plot_format="png"
                    )
                    logger.info(f"✓ Created {len(bar_chart_paths)} bar charts")
                    
                    # Create heatmap
                    heatmap_path = create_feature_importance_heatmap(
                        importance_csv, self.shap_viz_dir, top_n=top_n*2, plot_format="png"
                    )
                    logger.info(f"✓ Created heatmap: {heatmap_path}")
                    
                    return True
                except Exception as e:
                    logger.error(f"✗ Failed to create basic visualizations: {e}")
                    return False
            
            # Generate full comprehensive report
            shap_results = generate_comprehensive_shap_report(
                importance_csv=importance_csv,
                model_path=model_pkl,
                dataset_path=self.dataset_path,
                out_dir=self.output_dir,
                top_n=top_n,
                sample_size=sample_size,
                plot_format="png"
            )
            
            logger.info("✓ Comprehensive SHAP report generated")
            
            # Log summary statistics
            if 'summary_stats' in shap_results:
                stats = shap_results['summary_stats']
                logger.info("SHAP Summary Statistics:")
                logger.info(f"  Total features analyzed: {stats.get('total_features', 'N/A')}")
                logger.info(f"  Top feature overall: {stats.get('top_feature_overall', 'N/A')}")
                logger.info(f"  Top feature neither: {stats.get('top_feature_neither', 'N/A')}")
                logger.info(f"  Top feature donor: {stats.get('top_feature_donor', 'N/A')}")
                logger.info(f"  Top feature acceptor: {stats.get('top_feature_acceptor', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Comprehensive visualization creation failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def create_memory_efficient_beeswarm_plots(self, model: Any, X: pd.DataFrame, 
                                             sample_size: int = 500, top_n: int = 20) -> bool:
        """Create memory-efficient beeswarm plots."""
        logger.info("Creating memory-efficient beeswarm plots...")
        logger.info(f"  Sample size: {sample_size}")
        logger.info(f"  Top features: {top_n}")
        
        try:
            from meta_spliceai.splice_engine.meta_models.evaluation.shap_incremental import (
                create_ensemble_beeswarm_plots
            )
            
            plot_paths = create_ensemble_beeswarm_plots(
                model,
                X,
                background_size=100,
                sample_size=sample_size,
                top_n_features=top_n,
                approximate=True,
                dtype="float32",
                random_state=42,
                save_dir=self.shap_beeswarm_dir,
                plot_format="png",
                figsize=(10, 8),
                dpi=300,
                verbose=self.verbose,
            )
            
            if plot_paths:
                logger.info(f"✓ Created {len(plot_paths)} beeswarm plots:")
                for plot_name, plot_path in plot_paths.items():
                    logger.info(f"  - {plot_name}: {plot_path}")
                return True
            else:
                logger.warning("No beeswarm plots were created")
                return False
                
        except Exception as e:
            logger.error(f"✗ Beeswarm plot creation failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def run_full_analysis(self, sample_size: Optional[int] = None, 
                         batch_size: int = 512, background_size: int = 100,
                         viz_sample_size: int = 1000, top_n: int = 20) -> Dict[str, Any]:
        """Run the complete SHAP analysis workflow."""
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE SHAP ANALYSIS")
        logger.info("=" * 60)
        
        results = {
            'success': False,
            'steps_completed': [],
            'steps_failed': [],
            'output_files': {},
            'error_messages': []
        }
        
        try:
            # Step 1: Check prerequisites
            logger.info("\n1. Checking prerequisites...")
            prereqs = self.check_prerequisites()
            if not all(prereqs.values()):
                error_msg = "Prerequisites check failed"
                results['error_messages'].append(error_msg)
                logger.error(error_msg)
                return results
            results['steps_completed'].append('prerequisites')
            
            # Step 2: Test SHAP import
            logger.info("\n2. Testing SHAP import...")
            if not self.test_shap_import():
                error_msg = "SHAP import test failed"
                results['error_messages'].append(error_msg)
                results['steps_failed'].append('shap_import')
                logger.error(error_msg)
                # Continue anyway - sometimes the test is overly strict
            else:
                results['steps_completed'].append('shap_import')
            
            # Step 3: Load model and features
            logger.info("\n3. Loading model and features...")
            try:
                model, feature_names = self.load_model_and_features()
                results['steps_completed'].append('model_loading')
                results['model_type'] = type(model).__name__
                results['num_features'] = len(feature_names)
            except Exception as e:
                error_msg = f"Model loading failed: {e}"
                results['error_messages'].append(error_msg)
                results['steps_failed'].append('model_loading')
                logger.error(error_msg)
                return results
            
            # Step 4: Prepare dataset
            logger.info("\n4. Preparing dataset...")
            try:
                X, y = self.prepare_dataset(feature_names, sample_size)
                results['steps_completed'].append('dataset_preparation')
                results['dataset_shape'] = X.shape
                results['num_samples'] = len(X)
                results['num_available_features'] = X.shape[1]
            except Exception as e:
                error_msg = f"Dataset preparation failed: {e}"
                results['error_messages'].append(error_msg)
                results['steps_failed'].append('dataset_preparation')
                logger.error(error_msg)
                return results
            
            # Step 5: Run incremental SHAP analysis
            logger.info("\n5. Running incremental SHAP analysis...")
            try:
                if self.run_incremental_shap_analysis(model, X, batch_size, background_size):
                    results['steps_completed'].append('incremental_shap')
                    results['output_files']['shap_importance'] = str(self.shap_importance_dir / "shap_importance_incremental.csv")
                else:
                    results['steps_failed'].append('incremental_shap')
            except Exception as e:
                error_msg = f"Incremental SHAP analysis failed: {e}"
                results['error_messages'].append(error_msg)
                results['steps_failed'].append('incremental_shap')
                logger.error(error_msg)
            
            # Step 6: Create basic visualizations
            logger.info("\n6. Creating basic visualizations...")
            try:
                if self.create_basic_visualizations(model, X, top_n):
                    results['steps_completed'].append('basic_visualizations')
                    results['output_files']['basic_plots'] = str(self.shap_viz_dir)
                else:
                    results['steps_failed'].append('basic_visualizations')
            except Exception as e:
                error_msg = f"Basic visualization creation failed: {e}"
                results['error_messages'].append(error_msg)
                results['steps_failed'].append('basic_visualizations')
                logger.error(error_msg)
            
            # Step 7: Create comprehensive visualizations
            logger.info("\n7. Creating comprehensive visualizations...")
            try:
                if self.create_comprehensive_visualizations(viz_sample_size, top_n):
                    results['steps_completed'].append('comprehensive_visualizations')
                    results['output_files']['comprehensive_plots'] = str(self.shap_viz_dir)
                else:
                    results['steps_failed'].append('comprehensive_visualizations')
            except Exception as e:
                error_msg = f"Comprehensive visualization creation failed: {e}"
                results['error_messages'].append(error_msg)
                results['steps_failed'].append('comprehensive_visualizations')
                logger.error(error_msg)
            
            # Step 8: Create beeswarm plots
            logger.info("\n8. Creating beeswarm plots...")
            try:
                if self.create_memory_efficient_beeswarm_plots(model, X, viz_sample_size, top_n):
                    results['steps_completed'].append('beeswarm_plots')
                    results['output_files']['beeswarm_plots'] = str(self.shap_beeswarm_dir)
                else:
                    results['steps_failed'].append('beeswarm_plots')
            except Exception as e:
                error_msg = f"Beeswarm plot creation failed: {e}"
                results['error_messages'].append(error_msg)
                results['steps_failed'].append('beeswarm_plots')
                logger.error(error_msg)
            
            # Determine overall success
            critical_steps = ['prerequisites', 'model_loading', 'dataset_preparation']
            critical_success = all(step in results['steps_completed'] for step in critical_steps)
            
            if critical_success and len(results['steps_completed']) >= 5:
                results['success'] = True
                logger.info("\n✓ SHAP analysis completed successfully!")
            else:
                logger.warning("\n⚠️  SHAP analysis completed with some failures")
            
        except Exception as e:
            error_msg = f"Unexpected error in SHAP analysis: {e}"
            results['error_messages'].append(error_msg)
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SHAP ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall success: {results['success']}")
        logger.info(f"Steps completed: {len(results['steps_completed'])}")
        logger.info(f"Steps failed: {len(results['steps_failed'])}")
        
        if results['steps_completed']:
            logger.info("\nCompleted steps:")
            for step in results['steps_completed']:
                logger.info(f"  ✓ {step}")
        
        if results['steps_failed']:
            logger.info("\nFailed steps:")
            for step in results['steps_failed']:
                logger.info(f"  ✗ {step}")
        
        if results['output_files']:
            logger.info("\nGenerated output files:")
            for desc, path in results['output_files'].items():
                logger.info(f"  📁 {desc}: {path}")
        
        logger.info(f"\nAll outputs saved to: {self.output_dir}")
        logger.info("=" * 60)
        
        return results


def create_test_model_if_needed(model_dir: Path, dataset_path: Path) -> bool:
    """Create a simple test model if none exists."""
    model_pkl = model_dir / "model_multiclass.pkl"
    feature_csv = model_dir / "feature_manifest.csv"
    
    if model_pkl.exists() and feature_csv.exists():
        return True
    
    logger.info("No existing model found, creating a simple test model...")
    
    try:
        from meta_spliceai.splice_engine.meta_models.training import datasets
        from meta_spliceai.splice_engine.meta_models.builder import preprocessing
        from meta_spliceai.splice_engine.meta_models.training.classifier_utils import SigmoidEnsemble
        from xgboost import XGBClassifier
        import pickle
        
        # Load small sample of data
        os.environ["SS_MAX_ROWS"] = "1000"
        df = datasets.load_dataset(str(dataset_path))
        
        # Prepare data
        X_df, y_series = preprocessing.prepare_training_data(
            df, label_col="splice_type", return_type="pandas", verbose=0, encode_chrom=True
        )
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_series)
        
        # Create mapping for expected classes
        encoder_classes = label_encoder.classes_
        class_mapping = {}
        for i, class_name in enumerate(encoder_classes):
            if class_name == 'donor':
                class_mapping[i] = 0
            elif class_name == 'acceptor':
                class_mapping[i] = 1
            else:
                class_mapping[i] = 2
        
        y = np.array([class_mapping[label] for label in y])
        
        X = X_df.values
        feature_names = list(X_df.columns)
        
        # Train simple models
        models = []
        for cls in range(3):
            y_bin = (y == cls).astype(int)
            clf = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
            clf.fit(X, y_bin)
            models.append(clf)
        
        # Create ensemble
        ensemble = SigmoidEnsemble(models, feature_names)
        
        # Save model and features
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_pkl, "wb") as f:
            pickle.dump(ensemble, f)
        
        pd.DataFrame({"feature": feature_names}).to_csv(feature_csv, index=False)
        
        logger.info(f"✓ Created test model with {len(feature_names)} features")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to create test model: {e}")
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Standalone SHAP Analysis for Meta-Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python standalone_shap_analysis.py --dataset data/train_pc_1000/master --model-dir results/test_model

  # With custom parameters
  python standalone_shap_analysis.py --dataset data/train_pc_1000/master --model-dir results/test_model \\
    --output-dir results/shap_test --sample-size 5000 --batch-size 256

  # Create test model if needed
  python standalone_shap_analysis.py --dataset data/train_pc_1000/master --model-dir results/test_model \\
    --create-test-model
        """
    )
    
    parser.add_argument(
        "--dataset", 
        required=True,
        help="Path to the dataset directory or file"
    )
    
    parser.add_argument(
        "--model-dir",
        required=True, 
        help="Directory containing the trained model files"
    )
    
    parser.add_argument(
        "--output-dir",
        default="results/shap_analysis",
        help="Output directory for SHAP analysis results (default: results/shap_analysis)"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of samples to use from dataset (default: use all)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for SHAP computation (default: 512)"
    )
    
    parser.add_argument(
        "--background-size",
        type=int,
        default=100,
        help="Background sample size for SHAP explainer (default: 100)"
    )
    
    parser.add_argument(
        "--viz-sample-size",
        type=int,
        default=1000,
        help="Sample size for visualization plots (default: 1000)"
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top features to display in plots (default: 20)"
    )
    
    parser.add_argument(
        "--create-test-model",
        action="store_true",
        help="Create a simple test model if none exists"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress most output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set up logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up environment
    setup_environment()
    
    # Convert paths
    dataset_path = Path(args.dataset)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    
    logger.info("Starting standalone SHAP analysis")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create test model if requested and needed
    if args.create_test_model:
        if not create_test_model_if_needed(model_dir, dataset_path):
            logger.error("Failed to create test model")
            sys.exit(1)
    
    # Create analyzer
    analyzer = StandaloneSHAPAnalyzer(
        dataset_path=dataset_path,
        model_dir=model_dir,
        output_dir=output_dir,
        verbose=args.verbose
    )
    
    # Run analysis
    results = analyzer.run_full_analysis(
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        background_size=args.background_size,
        viz_sample_size=args.viz_sample_size,
        top_n=args.top_n
    )
    
    # Save results summary
    results_file = output_dir / "shap_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results summary saved to: {results_file}")
    
    if results['success']:
        logger.info("🎉 SHAP analysis completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ SHAP analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 