#!/usr/bin/env python3
"""Common utilities for cross-validation training scripts.

This module provides shared functionality for CV training scripts including:
- Smart dataset path resolution
- Training summary generation
- Common CV utilities and helpers
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
import platform
import psutil

logger = logging.getLogger(__name__)


def resolve_dataset_path(dataset_path: str) -> tuple[str, str, int]:
    """
    Smart dataset path resolution that automatically detects master/ subdirectory.
    
    Parameters
    ----------
    dataset_path : str
        Original dataset path provided by user
        
    Returns
    -------
    tuple[str, str, int]
        (original_path, actual_path, parquet_count)
        - original_path: The path as provided by user
        - actual_path: The path that will be used (with master/ if detected)
        - parquet_count: Number of parquet files found
    """
    original_path = dataset_path
    dataset_path_obj = Path(dataset_path)
    
    # Check if master subdirectory exists and contains parquet files
    master_path = dataset_path_obj / "master"
    if master_path.exists() and master_path.is_dir():
        parquet_files = list(master_path.glob("*.parquet"))
        if parquet_files:
            actual_path = str(master_path)
            parquet_count = len(parquet_files)
            logger.info(f"📁 Auto-detected master subdirectory: {master_path}")
            logger.info(f"   Found {parquet_count} parquet files in master/")
            return original_path, actual_path, parquet_count
        else:
            logger.warning(f"⚠️  Master subdirectory exists but contains no parquet files: {master_path}")
    
    # Check if the original path contains parquet files
    parquet_files = list(dataset_path_obj.glob("*.parquet"))
    if parquet_files:
        actual_path = original_path
        parquet_count = len(parquet_files)
        logger.info(f"📁 Using dataset directly: {dataset_path_obj}")
        logger.info(f"   Found {parquet_count} parquet files")
        return original_path, actual_path, parquet_count
    
    # No parquet files found
    logger.warning(f"⚠️  No parquet files found in {dataset_path_obj}")
    logger.warning(f"   Expected location: {master_path}")
    return original_path, original_path, 0


def collect_dataset_info(dataset_path: str) -> Dict[str, Any]:
    """
    Collect comprehensive dataset information.
    
    Parameters
    ----------
    dataset_path : str
        Path to the dataset directory
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing dataset metadata
    """
    dataset_info = {}
    
    try:
        # Try to find gene manifest
        gene_manifest_path = Path(dataset_path).parent / "gene_manifest.csv"
        if gene_manifest_path.exists():
            dataset_info["gene_manifest_path"] = str(gene_manifest_path)
            
            # Read basic gene manifest info
            gene_manifest = pd.read_csv(gene_manifest_path)
            dataset_info["total_genes"] = len(gene_manifest)
            
            if "gene_type" in gene_manifest.columns:
                gene_types = gene_manifest["gene_type"].value_counts().to_dict()
                dataset_info["gene_type_distribution"] = gene_types
                
            # Add additional manifest statistics if available
            if "gene_length" in gene_manifest.columns:
                dataset_info["gene_length_stats"] = {
                    "mean": float(gene_manifest["gene_length"].mean()),
                    "median": float(gene_manifest["gene_length"].median()),
                    "min": float(gene_manifest["gene_length"].min()),
                    "max": float(gene_manifest["gene_length"].max())
                }
                
            if "splice_density_per_kb" in gene_manifest.columns:
                dataset_info["splice_density_stats"] = {
                    "mean": float(gene_manifest["splice_density_per_kb"].mean()),
                    "median": float(gene_manifest["splice_density_per_kb"].median()),
                    "min": float(gene_manifest["splice_density_per_kb"].min()),
                    "max": float(gene_manifest["splice_density_per_kb"].max())
                }
                
    except Exception as e:
        dataset_info["gene_manifest_error"] = str(e)
    
    return dataset_info


def collect_system_info() -> Dict[str, Any]:
    """
    Collect system information for training summary.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing system metadata
    """
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 1)
    }


def collect_training_parameters(args: argparse.Namespace, original_dataset_path: str) -> Dict[str, Any]:
    """
    Collect training parameters from argparse namespace.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
    original_dataset_path : str
        Original dataset path provided by user
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing training parameters
    """
    # Core training parameters
    training_params = {
        "dataset_path": original_dataset_path,
        "actual_dataset_path": getattr(args, 'dataset', original_dataset_path),
        "output_directory": str(Path(args.out_dir)),
        "script_name": Path(sys.argv[0]).name,
        "command_line": " ".join(sys.argv),
    }
    
    # CV parameters
    cv_params = [
        'n_folds', 'n_splits', 'valid_size', 'row_cap', 'min_rows_test',
        'heldout_chroms', 'group_col', 'gene_col'
    ]
    
    # Model parameters
    model_params = [
        'n_estimators', 'tree_method', 'max_bin', 'device', 'learning_rate',
        'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree'
    ]
    
    # Feature parameters
    feature_params = [
        'feature_selection', 'max_features', 'selection_method',
        'exclude_features', 'force_features', 'use_sparse_kmers'
    ]
    
    # Analysis parameters
    analysis_params = [
        'calibrate', 'calibrate_per_class', 'calib_method',
        'monitor_overfitting', 'overfitting_threshold', 'early_stopping_patience',
        'convergence_improvement', 'check_leakage', 'leakage_threshold',
        'auto_exclude_leaky', 'plot_curves', 'plot_format', 'verbose'
    ]
    
    # Collect all parameters
    all_params = cv_params + model_params + feature_params + analysis_params
    
    for param in all_params:
        if hasattr(args, param):
            value = getattr(args, param)
            # Convert numpy types to native Python types for JSON serialization
            if isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            training_params[param] = value
    
    return training_params


def collect_performance_summary(fold_rows: List[Dict]) -> Dict[str, Any]:
    """
    Collect performance summary from CV fold results.
    
    Parameters
    ----------
    fold_rows : List[Dict]
        List of fold result dictionaries
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing performance metrics
    """
    if not fold_rows:
        return {}
    
    fold_df = pd.DataFrame(fold_rows)
    performance_summary = {}
    
    # Basic metrics (excluding accuracy due to class imbalance issues)
    basic_metrics = [
        'test_macro_f1', 'test_macro_avg_precision', 'splice_macro_f1',
        'top_k_accuracy', 'donor_f1', 'acceptor_f1', 'donor_ap', 'acceptor_ap',
        'auc_meta', 'ap_meta'
    ]
    
    for metric in basic_metrics:
        if metric in fold_df.columns:
            values = fold_df[metric].dropna()
            if len(values) > 0:
                performance_summary[f"mean_{metric}"] = float(values.mean())
                performance_summary[f"std_{metric}"] = float(values.std())
                performance_summary[f"min_{metric}"] = float(values.min())
                performance_summary[f"max_{metric}"] = float(values.max())
    
    # Fold count
    performance_summary["total_folds"] = len(fold_rows)
    
    # Sample counts
    if 'test_samples' in fold_df.columns:
        performance_summary["total_test_samples"] = int(fold_df['test_samples'].sum())
        performance_summary["mean_test_samples_per_fold"] = float(fold_df['test_samples'].mean())
    
    return performance_summary


def generate_training_summary(
    out_dir: Path,
    args: argparse.Namespace,
    original_dataset_path: str,
    fold_rows: List[Dict],
    script_name: str = "cv_training"
) -> None:
    """
    Generate comprehensive training summary report.
    
    Parameters
    ----------
    out_dir : Path
        Output directory for training results
    args : argparse.Namespace
        Parsed command line arguments
    original_dataset_path : str
        Original dataset path provided by user
    fold_rows : List[Dict]
        List of fold result dictionaries
    script_name : str
        Name of the training script
    """
    logger.info("\n" + "="*80)
    logger.info("📋 GENERATING TRAINING SUMMARY")
    logger.info("="*80)
    
    # Collect all information
    system_info = collect_system_info()
    training_params = collect_training_parameters(args, original_dataset_path)
    dataset_info = collect_dataset_info(training_params["actual_dataset_path"])
    performance_summary = collect_performance_summary(fold_rows)
    
    # Create comprehensive summary
    summary = {
        "training_summary": {
            "training_date": datetime.now().isoformat(),
            "script": script_name,
            "version": "1.0",
            "system_info": system_info,
            "training_parameters": training_params,
            "dataset_info": dataset_info,
            "performance_summary": performance_summary
        }
    }
    
    # Save summary to JSON
    summary_json_path = out_dir / "training_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save human-readable summary
    summary_txt_path = out_dir / "training_summary.txt"
    with open(summary_txt_path, "w") as f:
        f.write("META-MODEL TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Training Date: {summary['training_summary']['training_date']}\n")
        f.write(f"Script: {summary['training_summary']['script']}\n\n")
        
        f.write("DATASET INFORMATION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Original Path: {training_params['dataset_path']}\n")
        f.write(f"Actual Path: {training_params['actual_dataset_path']}\n")
        f.write(f"Output Directory: {training_params['output_directory']}\n")
        if "gene_manifest_path" in dataset_info:
            f.write(f"Gene Manifest: {dataset_info['gene_manifest_path']}\n")
        if "total_genes" in dataset_info:
            f.write(f"Total Genes: {dataset_info['total_genes']}\n")
        if "gene_type_distribution" in dataset_info:
            f.write(f"Gene Types: {dataset_info['gene_type_distribution']}\n")
        f.write("\n")
        
        f.write("TRAINING PARAMETERS:\n")
        f.write("-" * 20 + "\n")
        
        # CV parameters
        cv_params = ['n_folds', 'n_splits', 'valid_size', 'row_cap', 'min_rows_test']
        for param in cv_params:
            if param in training_params:
                f.write(f"{param.replace('_', ' ').title()}: {training_params[param]}\n")
        
        # Model parameters
        model_params = ['n_estimators', 'tree_method', 'max_bin', 'device']
        for param in model_params:
            if param in training_params:
                f.write(f"{param.replace('_', ' ').title()}: {training_params[param]}\n")
        
        # Analysis parameters
        analysis_params = ['calibrate_per_class', 'auto_exclude_leaky', 'monitor_overfitting']
        for param in analysis_params:
            if param in training_params:
                f.write(f"{param.replace('_', ' ').title()}: {training_params[param]}\n")
        f.write("\n")
        
        f.write("PERFORMANCE SUMMARY:\n")
        f.write("-" * 20 + "\n")
        if performance_summary:
            # Primary performance metrics (class-imbalance aware)
            if "mean_test_macro_f1" in performance_summary:
                f.write(f"CV F1 Macro: {performance_summary['mean_test_macro_f1']:.3f} ± {performance_summary['std_test_macro_f1']:.3f}\n")
            if "mean_test_macro_avg_precision" in performance_summary:
                f.write(f"CV Average Precision: {performance_summary['mean_test_macro_avg_precision']:.3f} ± {performance_summary['std_test_macro_avg_precision']:.3f}\n")
            if "mean_top_k_accuracy" in performance_summary:
                f.write(f"CV Top-k Accuracy: {performance_summary['mean_top_k_accuracy']:.3f} ± {performance_summary['std_top_k_accuracy']:.3f}\n")
            
            # Per-class performance
            if "mean_donor_f1" in performance_summary:
                f.write(f"Donor F1: {performance_summary['mean_donor_f1']:.3f} ± {performance_summary['std_donor_f1']:.3f}\n")
            if "mean_acceptor_f1" in performance_summary:
                f.write(f"Acceptor F1: {performance_summary['mean_acceptor_f1']:.3f} ± {performance_summary['std_acceptor_f1']:.3f}\n")
            if "mean_donor_ap" in performance_summary:
                f.write(f"Donor AP: {performance_summary['mean_donor_ap']:.3f} ± {performance_summary['std_donor_ap']:.3f}\n")
            if "mean_acceptor_ap" in performance_summary:
                f.write(f"Acceptor AP: {performance_summary['mean_acceptor_ap']:.3f} ± {performance_summary['std_acceptor_ap']:.3f}\n")
                
            if "total_folds" in performance_summary:
                f.write(f"Total Folds: {performance_summary['total_folds']}\n")
        f.write("\n")
        
        f.write("SYSTEM INFORMATION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Platform: {system_info['platform']}\n")
        f.write(f"Python Version: {system_info['python_version']}\n")
        f.write(f"CPU Cores: {system_info['cpu_count']}\n")
        f.write(f"Memory: {system_info['memory_gb']} GB\n")
        f.write(f"Available Memory: {system_info['memory_available_gb']} GB\n\n")
        
        f.write("KEY FILES:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Trained Model: {out_dir}/model_multiclass.pkl\n")
        f.write(f"Feature Manifest: {out_dir}/feature_manifest.csv\n")
        f.write(f"CV Metrics: {out_dir}/gene_cv_metrics.csv\n")
        f.write(f"Training Summary (JSON): {out_dir}/training_summary.json\n")
        f.write(f"Training Summary (TXT): {out_dir}/training_summary.txt\n")
        if "gene_manifest_path" in dataset_info:
            f.write(f"Dataset Gene Manifest: {dataset_info['gene_manifest_path']}\n")
    
    logger.info(f"✅ Training summary saved to:")
    logger.info(f"   JSON: {summary_json_path}")
    logger.info(f"   TXT:  {summary_txt_path}")
    
    # Display key information
    logger.info(f"\n📊 TRAINING SUMMARY:")
    logger.info(f"   Dataset: {training_params['dataset_path']} → {training_params['actual_dataset_path']}")
    logger.info(f"   Output: {training_params['output_directory']}")
    logger.info(f"   Model: {out_dir}/model_multiclass.pkl")
    logger.info(f"   Features: {out_dir}/feature_manifest.csv")
    if "gene_manifest_path" in dataset_info:
        logger.info(f"   Gene Manifest: {dataset_info['gene_manifest_path']}")
    if performance_summary and "mean_test_macro_f1" in performance_summary:
        logger.info(f"   Performance: F1 = {performance_summary['mean_test_macro_f1']:.3f} ± {performance_summary['std_test_macro_f1']:.3f}")


def setup_cv_logging(args: argparse.Namespace) -> None:
    """
    Set up logging for CV training scripts.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
    """
    # Set verbosity level - handle both --verbose flag and --verbosity level
    verbosity_level = getattr(args, 'verbosity', 1)
    if getattr(args, 'verbose', False):  # --verbose flag overrides to debug level
        verbosity_level = 2
    
    if verbosity_level == 0:
        logging.getLogger().setLevel(logging.WARNING)
    elif verbosity_level == 2:
        logging.getLogger().setLevel(logging.DEBUG)
    else:  # verbosity_level == 1 (default)
        logging.getLogger().setLevel(logging.INFO)
    
    # Configure logging format
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )


def validate_cv_arguments(args: argparse.Namespace) -> None:
    """
    Validate CV training arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments
        
    Raises
    ------
    ValueError
        If arguments are invalid
    """
    # Validate row cap
    if hasattr(args, 'row_cap') and args.row_cap < 0:
        raise ValueError("--row-cap must be >= 0")
    
    # Validate CV folds
    if hasattr(args, 'n_folds') and args.n_folds < 2:
        raise ValueError("--n-folds must be >= 2")
    
    # Validate validation size
    if hasattr(args, 'valid_size') and (args.valid_size <= 0 or args.valid_size >= 1):
        raise ValueError("--valid-size must be between 0 and 1")
    
    # Validate min rows test
    if hasattr(args, 'min_rows_test') and args.min_rows_test < 1:
        raise ValueError("--min-rows-test must be >= 1")
    
    # Validate n estimators
    if hasattr(args, 'n_estimators') and args.n_estimators < 1:
        raise ValueError("--n-estimators must be >= 1")
    
    # Validate leakage threshold
    if hasattr(args, 'leakage_threshold') and (args.leakage_threshold < 0 or args.leakage_threshold > 1):
        raise ValueError("--leakage-threshold must be between 0 and 1")
    
    # Validate overfitting threshold
    if hasattr(args, 'overfitting_threshold') and args.overfitting_threshold < 0:
        raise ValueError("--overfitting-threshold must be >= 0")


def create_output_directory(out_dir: str) -> Path:
    """
    Create output directory and return Path object.
    
    Parameters
    ----------
    out_dir : str
        Output directory path
        
    Returns
    -------
    Path
        Path object for the output directory
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path
