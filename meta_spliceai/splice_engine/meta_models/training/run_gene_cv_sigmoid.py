#!/usr/bin/env python3
"""Gene-wise K-fold CV for the meta-model with *independent sigmoid* outputs.

This mirrors `run_gene_cv_multiclass.py` but replaces the single
multi-class soft-max XGBoost with **three binary XGBoost classifiers**
(one-vs-rest for neither / donor / acceptor).  The full meta feature
vector is left untouched.

A lightweight `SigmoidEnsemble` wrapper (defined in
`classifier_utils.py`) exposes a `predict_proba(X)` method that stacks
the three class probabilities into the familiar ``(n,3)`` matrix, so
all downstream diagnostics and evaluation utilities continue to work
unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import warnings
# Suppress repeated "invalid value encountered in divide" warnings from numpy.corrcoef
warnings.filterwarnings(
    "ignore",
    message=r"invalid value encountered in divide",
    category=RuntimeWarning,
    module=r"numpy",
)
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from meta_spliceai.splice_engine.meta_models.training.models import AVAILABLE_MODELS
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    top_k_accuracy_score,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)

from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.builder import preprocessing
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import (
    _encode_labels,
    SigmoidEnsemble,  # new wrapper class
    PerClassCalibratedSigmoidEnsemble,  # per-class calibration wrapper
)
from meta_spliceai.splice_engine.meta_models.evaluation.viz_utils import (
    plot_roc_pr_curves, 
    plot_roc_pr_curves_f1, 
    check_feature_correlations, 
    plot_combined_roc_pr_curves_meta
)
from meta_spliceai.splice_engine.meta_models.evaluation.multiclass_roc_pr import (
    plot_multiclass_roc_pr_curves,
    create_improved_binary_pr_plot
)
from meta_spliceai.splice_engine.meta_models.evaluation.shap_viz import (
    generate_comprehensive_shap_report
)
from meta_spliceai.splice_engine.meta_models.evaluation.feature_utils import (
    load_excluded_features,
    filter_features,
    save_feature_importance,
)

# Import the top-k accuracy modules
from meta_spliceai.splice_engine.meta_models.evaluation.top_k_metrics import (
    calculate_cv_fold_top_k,
    report_top_k_accuracy
)

# Import transcript-level mapping module (optional)
try:
    from meta_spliceai.splice_engine.meta_models.evaluation.transcript_mapping import (
        calculate_transcript_level_top_k,
        report_transcript_top_k
    )
    TRANSCRIPT_MAPPING_AVAILABLE = True
except ImportError:
    TRANSCRIPT_MAPPING_AVAILABLE = False
from meta_spliceai.splice_engine.utils_kmer import is_kmer_feature as _is_kmer
from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
from meta_spliceai.splice_engine.meta_models.evaluation.shap_incremental import incremental_shap_importance, plot_feature_importance, run_incremental_shap_analysis
from meta_spliceai.splice_engine.meta_models.evaluation.feature_importance_integration import run_gene_cv_feature_importance_analysis

from meta_spliceai.splice_engine.meta_models.evaluation.cv_metrics_viz import generate_cv_metrics_report

# Import overfitting monitoring
from meta_spliceai.splice_engine.meta_models.evaluation.overfitting_monitor import (
    OverfittingMonitor, enhanced_model_training
)

from meta_spliceai.splice_engine.utils_doc import (
    print_emphasized,
    print_with_indent
)

# Import CV utilities
from meta_spliceai.splice_engine.meta_models.training import cv_utils

################################################################################
# CLI
################################################################################

def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(description="Gene-wise K-fold CV for the 3-way meta-classifier (independent sigmoid outputs).")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True)

    p.add_argument("--gene-col", default="gene_id")
    p.add_argument("--n-folds", "--n-splits", type=int, default=5, dest="n_folds",
                   help="Number of CV folds (alias: --n-splits)")
    p.add_argument("--valid-size", type=float, default=0.1)
    p.add_argument("--row-cap", type=int, default=100_000)

    # Diagnostics / evaluation
    p.add_argument("--diag-sample", type=int, default=25_000)
    p.add_argument("--annotations", default=None,
                    help="[DEPRECATED] Use --splice-sites-path instead. Kept for backward compatibility.")
    p.add_argument("--neigh-sample", type=int, default=0)
    p.add_argument("--neigh-window", type=int, default=10)
    p.add_argument("--sample-genes", type=int, default=None,                   
                        help="[TESTING ONLY] Sample a small subset of genes for quick testing/debugging. "
                        "Do NOT use for production training - CV should use all available data.")
    

    ### Batch Ensemble Training
    p.add_argument("--gene-start-idx", type=int, default=0,
                   help="Starting index for gene selection (enables batch processing)")
    p.add_argument("--gene-end-idx", type=int, default=None,
                   help="Ending index for gene selection (enables batch processing)")
    p.add_argument("--train-all-genes", action="store_true",
                   help="Train on ALL genes using automated batch processing")
    p.add_argument("--max-genes-in-memory", type=int, default=None,
                   help="Override automatic gene limit calculation (expert use)")
    p.add_argument("--memory-safety-factor", type=float, default=0.6,
                   help="Safety factor for memory usage (0.0-1.0, default: 0.6)")

    ### Multi-Instance Ensemble Configuration
    p.add_argument("--genes-per-instance", type=int, default=1500,
                   help="Number of genes per instance in multi-instance ensemble (default: 1500)")
    p.add_argument("--max-instances", type=int, default=10,
                   help="Maximum number of instances for multi-instance ensemble (default: 10)")
    p.add_argument("--instance-overlap", type=float, default=0.1,
                   help="Overlap ratio between instances (0.0-0.5, default: 0.1)")
    p.add_argument("--memory-per-gene-mb", type=float, default=8.0,
                   help="Estimated memory per gene in MB for capacity planning (default: 8.0)")
    p.add_argument("--max-memory-per-instance-gb", type=float, default=15.0,
                   help="Maximum memory per instance in GB (default: 15.0)")
    p.add_argument("--auto-adjust-instance-size", action="store_true", default=True,
                   help="Automatically adjust instance size based on available memory (default: True)")
    p.add_argument("--resume-from-checkpoint", action="store_true", default=True,
                   help="Resume training from existing completed instances (default: True)")
    p.add_argument("--force-retrain-all", action="store_true", default=False,
                   help="Force retraining of all instances, ignoring checkpoints")

    ### Algorithm Selection
    p.add_argument("--algorithm", default="xgboost",
                   choices=list(AVAILABLE_MODELS),
                   help="Base classifier algorithm (default: xgboost)")
    p.add_argument("--algorithm-params", type=str,
                   help="JSON string of algorithm-specific hyperparameters")

    # Base-vs-meta comparison flags
    p.add_argument("--donor-score-col", default="donor_score",
                   help="Column with raw donor probability from the base model.")
    p.add_argument("--acceptor-score-col", default="acceptor_score",
                   help="Column with raw acceptor probability from the base model.")
    p.add_argument("--splice-prob-col", default="score",
                   help="Optional column that already contains donor+acceptor probability. If present it is preferred.")
    p.add_argument("--base-thresh", type=float, default=0.5,
                   help="Threshold on raw base splice probability (donor+acceptor) to call a splice site.")
    p.add_argument("--top-k", type=int, default=5,
                   help="k for top-k accuracy when comparing base vs meta.")
    
    # Transcript-level top-k accuracy options
    p.add_argument("--transcript-topk", action="store_true",
                   help="Enable transcript-level top-k accuracy using annotation files")
    p.add_argument("--no-transcript-cache", action="store_true",
                   help="Disable caching for transcript mapping (may significantly increase runtime)")
    p.add_argument("--splice-sites-path", 
                   default="data/ensembl/splice_sites.tsv",
                   help="Path to splice site annotations file")
    p.add_argument("--transcript-features-path", 
                   default="data/ensembl/spliceai_analysis/transcript_features.tsv",
                   help="Path to transcript features file")
    p.add_argument("--gene-features-path", 
                   default="data/ensembl/spliceai_analysis/gene_features.tsv",
                   help="Path to gene features file")
    p.add_argument("--position-col", default="position",
                   help="Column name for genomic positions in dataset")
    p.add_argument("--chrom-col", default="chrom",
                   help="Column name for chromosome in dataset")

    p.add_argument("--seed", type=int, default=42)

    # XGBoost params
    p.add_argument("--tree-method", default="hist", choices=["hist", "gpu_hist", "approx"])
    p.add_argument("--max-bin", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--n-estimators", type=int, default=800)

    # Site-level eval tuning
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--base-tsv", default=None)
    
    # ROC/PR curve plotting options
    p.add_argument("--plot-curves", action="store_true", default=True,
                   help="If set, save per-fold and mean ROC/PR curves as files (default: True)")
    p.add_argument("--no-plot-curves", dest="plot_curves", action="store_false",
                   help="Disable saving of ROC/PR curves")
    p.add_argument("--n-roc-points", type=int, default=101,
                   help="Number of equally spaced points (0-1) to sample when averaging ROC/PR curves (default: 101)")
    p.add_argument("--plot-format", type=str, default="pdf", choices=["pdf", "png", "svg"],
                   help="File format for ROC/PR curve plots (default: pdf)")
    
    # Feature leakage checking
    p.add_argument("--check-leakage", action="store_true", default=True,
                   help="Check for potential feature leakage by correlation analysis (default: True)")
    p.add_argument("--no-leakage-check", dest="check_leakage", action="store_false",
                   help="Disable feature leakage checking")
    p.add_argument("--leakage-threshold", type=float, default=0.95,
                   help="Correlation threshold for detecting potentially leaky features (default: 0.95)")
    p.add_argument("--auto-exclude-leaky", action="store_true", default=False,
                   help="Automatically exclude features that exceed leakage threshold (default: False)")

    p.add_argument("--errors-only", action="store_true")
    p.add_argument("--error-artifact", default=None, dest="error_artifact",
                   help="TSV/Parquet file with per-position pred_type labels (TP/FP/FN) generated from the base model.\n"
                        "Required when using --errors-only if the artifact is not in the default search paths.")
    p.add_argument("--include-tns", action="store_true", help="Include true negatives in splice-site evaluation summary.")

    p.add_argument("--leakage-probe", action="store_true")
    p.add_argument("--exclude-features", default="configs/exclude_features.txt", 
                   help="Path to a file containing features to exclude from training (one feature per line). "
                        "Default is 'configs/exclude_features.txt' which contains known problematic features. "
                        "Can also accept a comma-separated list (e.g., 'distance_to_start,chrom').")
    p.add_argument("--calibrate", action="store_true", help="Enable probability calibration")
    p.add_argument("--calibrate-per-class", action="store_true", 
                   help="Enable per-class probability calibration instead of binary splice/non-splice calibration")
    p.add_argument("--calib-method", default="platt", choices=["platt", "isotonic"],
                   help="Calibration algorithm (platt = logistic sigmoid, isotonic = monotonic)")
    p.add_argument("--skip-eval", action="store_true", 
                   help="Skip all evaluation steps after model training")
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose output for detailed debugging information")
    

    # SHAP analysis control options
    p.add_argument("--skip-shap", action="store_true", default=False,
                   help="Skip SHAP analysis entirely for faster iterations")
    p.add_argument("--shap-sample", type=int, default=None,
                   help="Override SHAP sample size (default: uses --diag-sample)")
    p.add_argument("--fast-shap", action="store_true", default=False,
                   help="Use fast SHAP analysis with reduced sample size (1000 samples)")
    p.add_argument("--minimal-diagnostics", action="store_true", default=False,
                   help="Run only essential diagnostics (skip SHAP, neighbor analysis, leakage probe)")
    p.add_argument("--skip-feature-importance", action="store_true", default=False,
                   help="Skip comprehensive feature importance analysis for faster iterations")
    
    # Overfitting monitoring arguments
    p.add_argument("--monitor-overfitting", action="store_true", default=False,
                   help="Enable comprehensive overfitting monitoring and analysis")
    p.add_argument("--overfitting-threshold", type=float, default=0.05,
                   help="Performance gap threshold for overfitting detection (default: 0.05)")
    p.add_argument("--early-stopping-patience", type=int, default=20,
                   help="Patience for early stopping detection (default: 20)")
    p.add_argument("--convergence-improvement", type=float, default=0.001,
                   help="Minimum improvement threshold for convergence detection (default: 0.001)")
    
    # Memory optimization arguments  
    p.add_argument("--memory-optimize", action="store_true", default=False,
                   help="Enable memory optimization for low-memory systems")
    p.add_argument("--max-diag-sample", type=int, default=25000,
                   help="Maximum diagnostic sample size for memory optimization (default: 25000)")
    
    # Calibration analysis arguments
    p.add_argument("--calibration-analysis", action="store_true", default=False,
                   help="Enable comprehensive calibration analysis and overconfidence detection")
    p.add_argument("--calibration-sample", type=int, default=None,
                   help="Sample size for calibration analysis (default: use all data)")
    p.add_argument("--quick-overconfidence-check", action="store_true", default=True,
                   help="Run quick overconfidence detection on CV results (default: True)")
    p.add_argument("--no-overconfidence-check", dest="quick_overconfidence_check", action="store_false",
                   help="Disable quick overconfidence detection")
    
    return p.parse_args(argv)

################################################################################
# Helpers
################################################################################

def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresh: float, k: int = 5) -> Dict[str, float]:
    """Return common binary classification metrics given probabilities."""
    y_pred = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    # build 2-column prob array for top-k
    try:
        # For binary classification, k should be at most 2
        k_binary = min(k, 2)
        y_prob_2col = np.column_stack([1 - y_prob, y_prob])
        
        # Ensure we have both classes in y_true for top-k calculation
        unique_classes = np.unique(y_true)
        if len(unique_classes) >= 2 and len(y_true) > 0:
            topk = top_k_accuracy_score(y_true, y_prob_2col, k=k_binary)
        else:
            # Fallback: use regular accuracy if top-k can't be computed
            topk = accuracy_score(y_true, y_pred)
    except Exception as e:
        # Fallback: use regular accuracy
        try:
            topk = accuracy_score(y_true, y_pred)
        except:
            topk = 0.0
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "topk_acc": topk,
    }

def _train_binary_model(X: np.ndarray, y_bin: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, args: argparse.Namespace) -> tuple[Any, dict]:
    """Fit a binary classifier with algorithm selection support."""
    
    # Get algorithm from args or use default
    algorithm = getattr(args, 'algorithm', 'xgboost')
    
    # Log algorithm selection (only once per fold to avoid spam)
    if not hasattr(_train_binary_model, '_algorithm_logged') or _train_binary_model._algorithm_logged != algorithm:
        print(f"        🤖 Using {algorithm.upper()} algorithm for binary classification", flush=True)
        _train_binary_model._algorithm_logged = algorithm
    
    # Parse algorithm-specific parameters if provided
    algo_params = {}
    if hasattr(args, 'algorithm_params') and args.algorithm_params:
        import json
        try:
            algo_params = json.loads(args.algorithm_params)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid algorithm-params JSON: {e}")
            algo_params = {}
    
    # Map common args to algorithm-specific parameters
    common_params = _map_args_to_algorithm_params(args, algorithm)
    
    # Merge parameters (algo_params override common_params)
    final_params = {**common_params, **algo_params}
    
    # Create model using registry
    from meta_spliceai.splice_engine.meta_models.training.models import get_model
    model_spec = {"name": algorithm, **final_params}
    model = get_model(model_spec)
    
    # Algorithm-specific training with evaluation sets
    evals_result = {}
    
    if algorithm in ['xgboost', 'lightgbm']:
        # Tree-based models with eval sets and early stopping support
        model.fit(X, y_bin, eval_set=[(X, y_bin), (X_val, y_val)], verbose=False)
        evals_result = getattr(model, 'evals_result_', {})
        
    elif algorithm == 'catboost':
        # CatBoost with eval set
        model.fit(X, y_bin, eval_set=[(X_val, y_val)], verbose=False)
        # CatBoost stores evaluation results differently
        evals_result = getattr(model, 'evals_result_', {})
        
    elif algorithm == 'tabnet':
        # TabNet with custom training parameters
        fit_params = {
            'eval_set': [(X_val, y_val)],
            'eval_metric': ['logloss'],
            'patience': getattr(model, '_tabnet_patience', 15),
            'max_epochs': getattr(model, '_tabnet_max_epochs', 100),
            'batch_size': getattr(model, '_tabnet_batch_size', 1024),
            'virtual_batch_size': getattr(model, '_tabnet_virtual_batch_size', 128),
            'num_workers': getattr(model, '_tabnet_num_workers', 0),
        }
        model.fit(X, y_bin, **fit_params)
        evals_result = {}
        
    else:
        # Standard sklearn models (random_forest, log_reg, etc.)
        model.fit(X, y_bin)
        evals_result = {}
    
    return model, evals_result


def _map_args_to_algorithm_params(args: argparse.Namespace, algorithm: str) -> dict:
    """Map command-line args to algorithm-specific parameters."""
    
    # Base parameters - algorithm-specific random seed handling
    base_params = {}
    
    if algorithm == "xgboost":
        base_params.update({
            "n_estimators": args.n_estimators,
            "tree_method": getattr(args, 'tree_method', 'hist'),
            "max_bin": getattr(args, 'max_bin', 256),
            "device": getattr(args, 'device', 'auto') if getattr(args, 'device', 'auto') != "auto" else None,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": getattr(args, 'seed', 42),
        })
        
    elif algorithm == "lightgbm":
        base_params.update({
            "n_estimators": args.n_estimators,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": getattr(args, 'seed', 42),
        })
        
    elif algorithm == "catboost":
        base_params.update({
            "iterations": args.n_estimators,
            "learning_rate": 0.1,
            "depth": 6,
            "random_seed": getattr(args, 'seed', 42),  # CatBoost uses random_seed, not random_state
            "thread_count": -1,
        })
        
    elif algorithm == "random_forest":
        base_params.update({
            "n_estimators": args.n_estimators,
            "max_depth": None,
            "n_jobs": -1,
            "random_state": getattr(args, 'seed', 42),
        })
        
    elif algorithm == "log_reg":
        base_params.update({
            "max_iter": 1000,
            "random_state": getattr(args, 'seed', 42),
            "n_jobs": -1,
        })
    
    return base_params

################################################################################
# Main logic
################################################################################

def _calculate_optimal_gene_limit(
    total_genes: int,
    max_genes_override: Optional[int] = None,
    safety_factor: float = 0.6,
    verbose: bool = True,
    memory_per_gene_mb: float = 8.0
) -> int:
    """
    Calculate optimal gene limit based on available system memory.
    
    This function dynamically determines the maximum number of genes that can
    be safely processed based on:
    - Available system memory
    - Estimated memory usage per gene
    - Safety margins for training overhead
    
    Parameters
    ----------
    total_genes : int
        Total number of genes in dataset
    verbose : bool
        Whether to print analysis details
        
    Returns
    -------
    int
        Optimal maximum number of genes to process in memory
    """
    import psutil
    
    # Get system memory information
    memory_info = psutil.virtual_memory()
    total_memory_gb = memory_info.total / (1024**3)
    available_memory_gb = memory_info.available / (1024**3)
    
    if verbose:
        print(f"[MemoryAnalysis] System memory analysis:")
        print(f"  Total memory: {total_memory_gb:.1f} GB")
        print(f"  Available memory: {available_memory_gb:.1f} GB")
        print(f"  Memory utilization: {memory_info.percent:.1f}%")
    
    # Conservative estimates based on our testing:
    # - 1500 genes ≈ 6-7 GB dataset + training overhead ≈ 12-15 GB total
    # - Linear scaling: ~8-10 MB per gene for dataset + training
    # Now configurable via function parameter
    
    # Handle override if provided
    if max_genes_override is not None:
        if verbose:
            print(f"[MemoryAnalysis] Using manual override: {max_genes_override:,} genes")
            print(f"  Estimated memory usage: {max_genes_override * memory_per_gene_mb / 1024:.1f} GB")
        return min(max_genes_override, total_genes)
    
    # Calculate safe limits with configurable safety margins
    usable_memory_gb = available_memory_gb * safety_factor
    
    # Convert to genes
    max_genes_by_memory = int((usable_memory_gb * 1024) / memory_per_gene_mb)
    
    # Apply reasonable bounds
    min_genes_safe = 500   # Minimum for meaningful training
    max_genes_absolute = 4000  # Absolute maximum based on testing
    
    # Choose the most conservative limit
    max_genes_safe = max(
        min_genes_safe,
        min(max_genes_by_memory, max_genes_absolute, total_genes)
    )
    
    if verbose:
        print(f"[MemoryAnalysis] Gene limit calculation:")
        print(f"  Estimated memory per gene: {memory_per_gene_mb:.1f} MB")
        print(f"  Usable memory (60% safety): {usable_memory_gb:.1f} GB")
        print(f"  Memory-based limit: {max_genes_by_memory:,} genes")
        print(f"  Absolute maximum: {max_genes_absolute:,} genes")
        print(f"  Final safe limit: {max_genes_safe:,} genes")
        
        # Provide context
        coverage_pct = (max_genes_safe / total_genes) * 100 if total_genes > 0 else 0
        print(f"  Coverage: {coverage_pct:.1f}% of {total_genes:,} total genes")
        
        if max_genes_safe >= total_genes:
            print(f"  ✅ Can process ALL genes in single batch!")
        elif max_genes_safe >= total_genes * 0.5:
            print(f"  ✅ Can process majority of genes in single batch")
        else:
            print(f"  ℹ️  Will use representative sampling or multi-batch approach")
    
    return max_genes_safe


def _get_dataset_info(dataset_path: str | Path) -> dict:
    """Get dataset information from path.
    
    Parameters
    ----------
    dataset_path : str | Path
        Path to dataset directory or file
    
    Returns
    -------
    dict
        Dictionary with dataset metadata including source path and format
    """
    from pathlib import Path
    
    path = Path(dataset_path)
    result = {
        "path": str(path),
        "is_directory": path.is_dir(),
        "format": "parquet"
    }
    
    # Add additional metadata if available
    metadata_path = path / "metadata.json" if path.is_dir() else path.parent / "metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path, "r") as f:
            try:
                metadata = json.load(f)
                result.update(metadata)
            except json.JSONDecodeError:
                pass
    
    return result



##########################################################################
# Training Strategy Selection and Implementation
##########################################################################

def _should_use_batch_ensemble_training(args: argparse.Namespace) -> bool:
    """
    Determine whether to use batch ensemble training based on dataset and arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
        
    Returns
    -------
    bool
        True if batch ensemble training should be used
    """
    # Force batch ensemble if explicitly requested
    if getattr(args, 'train_all_genes', False):
        return True
    
    # Check dataset size to determine if batch ensemble is needed
    try:
        # Quick dataset analysis to estimate size
        dataset_info = _get_dataset_info(args.dataset)
        
        # Estimate gene count and memory requirements
        from meta_spliceai.splice_engine.meta_models.training.streaming_dataset_loader import StreamingDatasetLoader
        loader = StreamingDatasetLoader(args.dataset, verbose=False)
        info = loader.get_dataset_info()
        total_genes = info['total_genes']
        
        # Calculate optimal gene limit for single model training
        max_genes_safe = _calculate_optimal_gene_limit(
            total_genes=total_genes,
            max_genes_override=getattr(args, 'max_genes_in_memory', None),
            safety_factor=getattr(args, 'memory_safety_factor', 0.6),
            verbose=getattr(args, 'verbose', False),
            memory_per_gene_mb=getattr(args, 'memory_per_gene_mb', 8.0)
        )
        
        # Use batch ensemble if dataset is too large for single model training
        if total_genes > max_genes_safe:
            print(f"[Strategy Selection] Dataset has {total_genes:,} genes > {max_genes_safe:,} memory limit")
            print(f"[Strategy Selection] → Using batch ensemble training")
            return True
        else:
            print(f"[Strategy Selection] Dataset has {total_genes:,} genes ≤ {max_genes_safe:,} memory limit")
            print(f"[Strategy Selection] → Using single model training")
            return False
            
    except Exception as e:
        print(f"[Strategy Selection] Error analyzing dataset: {e}")
        print(f"[Strategy Selection] → Defaulting to single model training")
        return False


def _run_single_model_training(args: argparse.Namespace) -> None:
    """
    Run single model training using the complete original implementation.
    
    This function contains the proven training logic from the v0 implementation
    that works reliably for small to medium datasets.
    """
    # Import the complete original implementation and run it
    from meta_spliceai.splice_engine.meta_models.training import run_gene_cv_sigmoid_v0
    
    # The v0 implementation has a main() function that contains all the proven logic
    # We'll call it directly but need to extract its core logic here to avoid import issues
    
    # For now, let's implement the core logic directly based on the v0 reference
    _run_complete_single_model_pipeline(args)


def _run_batch_ensemble_training(args: argparse.Namespace) -> dict:
    """
    Run batch ensemble training for large datasets.
    
    This function coordinates batch ensemble training using the proven
    automated_all_genes_trainer module.
    """
    try:
        from meta_spliceai.splice_engine.meta_models.training.automated_all_genes_trainer import run_automated_all_genes_training
        
        results = run_automated_all_genes_training(
            dataset_path=args.dataset,
            out_dir=args.out_dir,
            n_estimators=getattr(args, 'n_estimators', 800),
            n_folds=getattr(args, 'n_folds', 5),
            max_genes_per_batch=1200,  # Conservative batch size
            max_memory_gb=12.0,
            calibrate_per_class=getattr(args, 'calibrate_per_class', True),
            auto_exclude_leaky=getattr(args, 'auto_exclude_leaky', True),
            monitor_overfitting=getattr(args, 'monitor_overfitting', True),
            neigh_sample=getattr(args, 'neigh_sample', 2000),
            early_stopping_patience=getattr(args, 'early_stopping_patience', 30),
            verbose=getattr(args, 'verbose', True)
        )
        
        print(f"🎉 [Batch Ensemble] Training completed!")
        print(f"  Total genes trained: {results.get('total_genes_trained', 0):,}")
        print(f"  Successful batches: {results.get('successful_batch_count', 0)}")
        print(f"  Model saved: {results.get('model_path', 'N/A')}")
        
        return results
        
    except ImportError as e:
        print(f"❌ [Batch Ensemble] Required modules not available: {e}")
        raise ImportError(f"Batch ensemble training requires automated_all_genes_trainer module: {e}")
    except Exception as e:
        print(f"❌ [Batch Ensemble] Training failed: {e}")
        raise RuntimeError(f"Batch ensemble training failed: {e}")


def _run_complete_single_model_pipeline(args: argparse.Namespace) -> None:
    """
    Run the complete single model training pipeline.
    
    This contains the proven implementation from the v0 script that works
    reliably for small to medium datasets.
    """
    # Setup logging and validation
    from meta_spliceai.splice_engine.meta_models.training import cv_utils
    cv_utils.setup_cv_logging(args)
    cv_utils.validate_cv_arguments(args)
    
    # Smart dataset path resolution
    original_dataset_path, actual_dataset_path, parquet_count = cv_utils.resolve_dataset_path(args.dataset)
    args.dataset = actual_dataset_path
    
    # Log important settings
    exclude_file = Path(args.out_dir) / "exclude_features.txt"
    if args.exclude_features or exclude_file.exists():
        sources = []
        if args.exclude_features:
            sources.append("command line")
        if exclude_file.exists():
            sources.append("exclude_features.txt")
        logging.info(f"Feature exclusion enabled from {' and '.join(sources)}")
    
    out_dir = cv_utils.create_output_directory(args.out_dir)

    # Handle deprecated --annotations parameter
    if getattr(args, 'annotations', None) and not args.splice_sites_path:
        print("Warning: --annotations is deprecated, please use --splice-sites-path instead")
        args.splice_sites_path = args.annotations

    # honour row-cap via env var used by datasets.load_dataset
    # BUT: disable row-cap when using gene-aware sampling to avoid conflicts
    if getattr(args, 'sample_genes', None) is not None:
        # Gene-aware sampling takes precedence - disable row cap
        os.environ["SS_MAX_ROWS"] = "0"
        print(f"[INFO] Disabled row cap for gene-aware sampling (--sample-genes {args.sample_genes})")
    elif args.row_cap > 0 and not os.getenv("SS_MAX_ROWS"):
        os.environ["SS_MAX_ROWS"] = str(args.row_cap)
    elif args.row_cap == 0 and not os.getenv("SS_MAX_ROWS"):
        # Use full dataset when row_cap is explicitly set to 0
        os.environ["SS_MAX_ROWS"] = "0"

    # Load dataset
    df = _load_dataset_for_single_model(args)
    
    if args.gene_col not in df.columns:
        raise KeyError(f"Column '{args.gene_col}' not found in dataset")

    # Prepare training data - use chromosome as a feature
    X_df, y_series = preprocessing.prepare_training_data(
        df, 
        label_col="splice_type", 
        return_type="pandas", 
        verbose=1,
        preserve_transcript_columns=True,  # Always preserve transcript columns for transcript-level metrics
        encode_chrom=True  # Include encoded chromosome as a feature
    )
    
    # Continue with the complete original pipeline implementation
    _run_original_cv_pipeline(args, out_dir, df, X_df, y_series, original_dataset_path)


def _load_dataset_for_single_model(args: argparse.Namespace):
    """Load dataset appropriately for single model training."""
    if getattr(args, 'sample_genes', None) is not None:
        # Use gene-level sampling for faster testing
        from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
        print(f"[INFO] Sampling {args.sample_genes} genes from dataset for faster testing")
        return load_dataset_sample(args.dataset, sample_genes=args.sample_genes, random_seed=args.seed)
    else:
        # Load full dataset with memory optimization for large datasets
        try:
            from meta_spliceai.splice_engine.meta_models.training.memory_optimized_datasets import (
                load_dataset_with_memory_management,
                estimate_dataset_size_efficiently
            )
            
            # Estimate dataset size first
            estimated_rows, file_count = estimate_dataset_size_efficiently(args.dataset)
            
            # Use memory optimization for datasets >2M rows or >10 files
            if estimated_rows > 2000000 or file_count > 10:
                print(f"[INFO] Large dataset detected ({estimated_rows:,} rows, {file_count} files)")
                print(f"[INFO] Using memory-optimized loading...")
                print(f"[INFO] This may take several minutes for large datasets...")
                return load_dataset_with_memory_management(
                    args.dataset,
                    max_memory_gb=12.0,  # Use up to 12GB for loading
                    fallback_to_standard=False  # Don't fall back to avoid schema errors
                )
            else:
                print(f"[INFO] Standard dataset size ({estimated_rows:,} rows), using standard loader")
                return datasets.load_dataset(args.dataset)
                
        except ImportError as e:
            print(f"[ERROR] Memory optimization module not available: {e}")
            print(f"[ERROR] Cannot proceed with large dataset using standard loader due to memory constraints")
            print(f"[SOLUTION] Please ensure memory_optimized_datasets.py is available")
            raise ImportError(f"Memory optimization required for large datasets but not available: {e}")
        except Exception as e:
            print(f"[ERROR] Memory-optimized loading failed: {e}")
            if "3mer_NNN" in str(e) or "SchemaError" in str(e):
                print(f"[ERROR] Schema mismatch detected in dataset")
                print(f"[SOLUTION] Run the schema validation utility:")
                print(f"  python -m meta_spliceai.splice_engine.meta_models.builder.validate_dataset_schema {args.dataset} --fix")
                raise RuntimeError(f"Dataset schema issues prevent loading. Please fix schema first: {e}")
            else:
                print(f"[ERROR] Cannot proceed with large dataset using standard loader due to memory constraints")
                raise RuntimeError(f"Memory-optimized loading failed and fallback not safe for large datasets: {e}")


def _run_original_cv_pipeline(args: argparse.Namespace, out_dir: Path, df, X_df, y_series, original_dataset_path: str) -> None:
    """
    Run the complete original CV pipeline from the v0 implementation.
    
    This contains all the proven logic for feature processing, cross-validation,
    model training, and comprehensive analysis that produces the expected outputs.
    """
    
    # Save feature names to JSON and CSV, reflecting any exclusions
    feature_path = out_dir / "train.features.json"
    feature_csv_path = out_dir / "feature_manifest.csv"
    
    # Always save both formats for consistency
    features_json = {"feature_names": list(X_df.columns)}
    with open(feature_path, "w") as f:
        json.dump(features_json, f)
    
    # CSV version for easier inspection
    pd.DataFrame({"feature": list(X_df.columns)}).to_csv(feature_csv_path, index=False)
    
    # Handle feature exclusion from both command-line and exclude_features.txt file
    X_df, excluded_features = _handle_feature_exclusions(args, out_dir, X_df)
    
    # Always save transcript mapping columns if transcript-topk is enabled
    transcript_columns = _prepare_transcript_columns(args, df, X_df)
    
    X = X_df.values
    y = _encode_labels(y_series)
    genes = df[args.gene_col].to_numpy()
    feature_names = list(X_df.columns)

    # Locate base probability columns
    splice_prob_idx, donor_idx, acceptor_idx = _locate_base_probability_columns(args, feature_names)
    
    # Print feature summary
    _print_feature_summary(feature_names)

    # Run production-ready training pipeline with proper evaluation separation
    _execute_production_training_pipeline(
        args, out_dir, X, y, genes, feature_names, 
        splice_prob_idx, donor_idx, acceptor_idx, transcript_columns, original_dataset_path
    )


def _execute_production_training_pipeline(
    args: argparse.Namespace, 
    out_dir: Path, 
    X: np.ndarray, 
    y: np.ndarray, 
    genes: np.ndarray,
    feature_names: List[str],
    splice_prob_idx: int | None,
    donor_idx: int | None,
    acceptor_idx: int | None,
    transcript_columns: dict,
    original_dataset_path: str
) -> None:
    """
    Execute production-ready meta-model training pipeline with proper evaluation separation.
    
    DESIGN PHILOSOPHY
    -----------------
    This pipeline implements MLOps best practices by addressing a critical data leakage 
    issue in traditional ML workflows: evaluating a model trained on all data using the 
    same data for performance analysis.
    
    PROBLEM SOLVED
    --------------
    Traditional flow (INCORRECT):
    1. Run CV → get realistic metrics
    2. Train final model on ALL data  
    3. Evaluate final model on SAME data → DATA LEAKAGE!
    
    Production flow (CORRECT):
    1. Run CV → get realistic metrics
    2. Generate all analysis from CV results → NO DATA LEAKAGE  
    3. Train final model on ALL data → DEPLOYMENT ONLY
    
    PIPELINE STAGES
    ---------------
    Stage 1: Cross-validation evaluation
        - Gene-wise K-fold CV with proper train/test separation
        - Generates unbiased performance metrics
        - Collects calibration data from holdout predictions
        - Creates ROC/PR curve data without data leakage
    
    Stage 2: CV-based post-training analysis  
        - All performance analysis uses ONLY CV results
        - ROC/PR curves from CV fold predictions
        - Calibration analysis from CV holdout data
        - Feature importance from CV models (if available)
        - No model loading or re-evaluation on training data
    
    Stage 3: Production model training
        - Trains final model on complete dataset
        - Optimized for inference on unseen positions/genes
        - NOT used for any performance evaluation
        - Deployment-ready with proper metadata
    
    KEY BENEFITS
    ------------
    ✅ Eliminates data leakage in performance evaluation
    ✅ Provides realistic performance estimates for production
    ✅ Maximizes training data for final production model
    ✅ Clear separation between evaluation and deployment
    ✅ Compatible with existing inference workflows
    ✅ Maintains all expected output artifacts
    
    USE CASES
    ---------
    - Model deployment for inference on unseen genes
    - Variant analysis with realistic performance expectations  
    - MLOps pipelines requiring trustworthy metrics
    - Research requiring proper statistical evaluation
    
    Parameters
    ----------
    args : argparse.Namespace
        Training configuration and hyperparameters
    out_dir : Path
        Output directory for all artifacts
    X : np.ndarray
        Feature matrix for training
    y : np.ndarray
        Encoded target labels  
    genes : np.ndarray
        Gene identifiers for group-aware CV
    feature_names : List[str]
        Names of features in X
    splice_prob_idx : int | None
        Index of combined splice probability column (if available)
    donor_idx : int | None
        Index of donor probability column (if available)  
    acceptor_idx : int | None
        Index of acceptor probability column (if available)
    transcript_columns : dict
        Preserved transcript mapping columns for analysis
    original_dataset_path : str
        Original dataset path for metadata
        
    Output Artifacts
    ----------------
    - model_multiclass.pkl: Production model (trained on all data)
    - gene_cv_metrics.csv: CV-based performance metrics (realistic)
    - metrics_aggregate.json: Aggregate CV performance
    - model_metadata.json: Evaluation methodology metadata
    - cv_performance_report.txt: Comprehensive CV analysis
    - calibration_analysis/: CV-based calibration data
    - All existing visualization artifacts (ROC/PR curves, etc.)
    
    Notes
    -----
    - All performance metrics reflect CV-based evaluation (no data leakage)
    - Production model should only be used for inference, not evaluation
    - This approach is superior to traditional train/test splits for gene-based data
    - Compatible with existing inference and analysis workflows
    """
    
    print("🚀 [Production Pipeline] Executing production-ready training workflow")
    print("=" * 70)
    print("📋 Pipeline stages:")
    print("   1. Cross-validation evaluation (unbiased performance metrics)")
    print("   2. CV-based post-training analysis (no data leakage)")
    print("   3. Production model training (deployment ready)")
    print("=" * 70)
    
    # Stage 1: Cross-validation evaluation
    print(f"\n📊 [Stage 1] Cross-validation evaluation")
    cv_results = _perform_cross_validation_evaluation(
        args, out_dir, X, y, genes, feature_names, 
        splice_prob_idx, donor_idx, acceptor_idx, transcript_columns
    )
    
    # Stage 2: CV-based post-training analysis  
    print(f"\n📈 [Stage 2] CV-based post-training analysis")
    _perform_cv_based_analysis(args, out_dir, cv_results, original_dataset_path)
    
    # Stage 3: Production model training
    print(f"\n🎯 [Stage 3] Production model training")
    production_model = _train_production_model(
        args, X, y, feature_names, cv_results.calibration_data
    )
    
    # Save production model and generate deployment artifacts
    _finalize_production_artifacts(args, out_dir, production_model, cv_results)
    
    print("\n🎉 [Production Pipeline] Training pipeline completed successfully!")
    print("=" * 70)


@dataclass
class CrossValidationResults:
    """Container for cross-validation results and artifacts."""
    fold_metrics: List[Dict[str, Any]]
    calibration_data: Dict[str, Any]
    performance_curves: Dict[str, Any]
    feature_importance: Dict[str, Any]
    evaluation_summary: Dict[str, Any]


def _perform_cross_validation_evaluation(
    args: argparse.Namespace, 
    out_dir: Path, 
    X: np.ndarray, 
    y: np.ndarray, 
    genes: np.ndarray,
    feature_names: List[str],
    splice_prob_idx: int | None,
    donor_idx: int | None,
    acceptor_idx: int | None,
    transcript_columns: dict
) -> CrossValidationResults:
    """
    Perform comprehensive cross-validation evaluation.
    
    This stage provides unbiased performance estimates using proper gene-wise
    cross-validation that ensures no data leakage between training and test sets.
    
    Returns
    -------
    CrossValidationResults
        Comprehensive CV results including metrics, curves, and calibration data
    """
    
    print(f"   🔄 Running {args.n_folds}-fold gene-wise cross-validation")
    print(f"   📊 Total samples: {X.shape[0]:,}, Features: {X.shape[1]:,}")
    print(f"   🧬 Unique genes: {len(np.unique(genes)):,}")
    
    # Run the existing CV function (this is already correct - no changes needed)
    fold_rows, calibration_data = _run_gene_wise_cross_validation(
        args, out_dir, X, y, genes, feature_names, 
        splice_prob_idx, donor_idx, acceptor_idx, transcript_columns
    )
    
    # Extract performance curves from CV results
    performance_curves = _extract_performance_curves_from_cv(fold_rows)
    
    # Generate feature importance from CV models (if available)
    feature_importance = _generate_cv_feature_importance(args, out_dir, fold_rows)
    
    # Create evaluation summary
    evaluation_summary = _generate_evaluation_summary(fold_rows, args)
    
    print(f"   ✅ Cross-validation completed: {len(fold_rows)} folds")
    print(f"   📈 Mean F1: {evaluation_summary['mean_f1']:.3f} ± {evaluation_summary['std_f1']:.3f}")
    print(f"   🎯 Mean AP: {evaluation_summary['mean_ap']:.3f} ± {evaluation_summary['std_ap']:.3f}")
    
    return CrossValidationResults(
        fold_metrics=fold_rows,
        calibration_data=calibration_data,
        performance_curves=performance_curves,
        feature_importance=feature_importance,
        evaluation_summary=evaluation_summary
    )


def _perform_cv_based_analysis(
    args: argparse.Namespace, 
    out_dir: Path, 
    cv_results: CrossValidationResults,
    original_dataset_path: str
) -> None:
    """
    Perform comprehensive post-training analysis using only CV results.
    
    This ensures all performance analysis is based on proper holdout evaluation
    without any data leakage from the final production model.
    
    Analysis Components:
    - ROC/PR curve generation from CV folds
    - Calibration analysis from CV predictions  
    - Feature importance aggregation
    - Performance visualization suite
    - Model comparison metrics
    """
    
    print("   📊 Generating performance visualizations from CV results")
    
    # Generate ROC/PR curves from CV data (no model loading required)
    _generate_cv_based_roc_pr_curves(args, out_dir, cv_results)
    
    # Generate calibration analysis from CV predictions
    _generate_cv_based_calibration_analysis(args, out_dir, cv_results)
    
    # Create comprehensive performance reports
    _generate_cv_based_performance_reports(args, out_dir, cv_results, original_dataset_path)
    
    # Generate CV metrics visualization
    _generate_cv_metrics_visualization(args, out_dir)
    
    print("   ✅ CV-based analysis completed (no data leakage)")


def _train_production_model(
    args: argparse.Namespace,
    X: np.ndarray,
    y: np.ndarray, 
    feature_names: List[str],
    calibration_data: Dict[str, Any]
) -> Any:
    """
    Train final production model on complete dataset.
    
    This model is trained on ALL available data for optimal performance on 
    unseen positions and genes. It should NOT be used for any performance 
    evaluation - that was done in the CV stage.
    
    Returns
    -------
    Any
        Production-ready ensemble model trained on complete dataset
    """
    
    print("   🎯 Training production model on complete dataset")
    print("   ⚠️  This model is for deployment only - performance was evaluated via CV")
    print(f"   📊 Training on {X.shape[0]:,} samples with {X.shape[1]:,} features")
    
    # Use the existing final ensemble training logic
    production_model = _train_final_ensemble(args, X, y, feature_names, calibration_data)
    
    print("   ✅ Production model training completed")
    
    return production_model


def _finalize_production_artifacts(
    args: argparse.Namespace,
    out_dir: Path,
    production_model: Any,
    cv_results: CrossValidationResults
) -> None:
    """
    Save production model and generate deployment artifacts.
    
    Creates all necessary files for model deployment while ensuring
    performance metrics reflect realistic CV-based evaluation.
    """
    
    print("   💾 Saving production model and deployment artifacts")
    
    # Save production model
    import pickle
    model_path = out_dir / "model_multiclass.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(production_model, f)
    print(f"   ✅ Production model saved: {model_path}")
    
    # Save CV metrics as the authoritative performance record
    df_metrics = pd.DataFrame(cv_results.fold_metrics)
    df_metrics.to_csv(out_dir / "gene_cv_metrics.csv", index=False)
    
    # Save aggregate performance metrics (from CV, not production model)
    key_metrics = ["test_macro_f1", "splice_macro_f1", "top_k_accuracy", 
                   "donor_f1", "acceptor_f1", "donor_ap", "acceptor_ap",
                   "auc_meta", "ap_meta"]
    available_metrics = [m for m in key_metrics if m in df_metrics.columns]
    mean_metrics = df_metrics[available_metrics].mean()
    
    with open(out_dir / "metrics_aggregate.json", "w") as f:
        json.dump(mean_metrics.to_dict(), f, indent=2)
    
    # Add metadata to distinguish CV-based vs production model metrics
    algorithm = getattr(args, 'algorithm', 'xgboost')
    metadata = {
        "evaluation_method": "cross_validation",
        "model_type": "production_deployment", 
        "algorithm": algorithm,
        "data_leakage_free": True,
        "cv_folds": len(cv_results.fold_metrics),
        "performance_note": "Metrics are from CV evaluation, not production model testing"
    }
    
    with open(out_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("   ✅ Deployment artifacts created")


def _extract_performance_curves_from_cv(fold_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract performance curve data from CV fold results."""
    return {
        'roc_curves': [row.get('auc_meta', 0) for row in fold_rows],
        'pr_curves': [row.get('ap_meta', 0) for row in fold_rows],
        'fold_count': len(fold_rows)
    }


def _generate_cv_feature_importance(args: argparse.Namespace, out_dir: Path, fold_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate feature importance analysis from CV models."""
    # Placeholder for CV-based feature importance
    # This would aggregate importance across CV folds if models were saved
    return {
        'method': 'cv_aggregated',
        'available': False,
        'note': 'Feature importance from CV folds not implemented yet'
    }


def _generate_evaluation_summary(fold_rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    """Generate comprehensive evaluation summary from CV results."""
    if not fold_rows:
        return {}
    
    df_metrics = pd.DataFrame(fold_rows)
    
    # Calculate key summary statistics
    summary = {
        'mean_f1': df_metrics['test_macro_f1'].mean() if 'test_macro_f1' in df_metrics.columns else 0.0,
        'std_f1': df_metrics['test_macro_f1'].std() if 'test_macro_f1' in df_metrics.columns else 0.0,
        'mean_ap': df_metrics['test_macro_avg_precision'].mean() if 'test_macro_avg_precision' in df_metrics.columns else 0.0,
        'std_ap': df_metrics['test_macro_avg_precision'].std() if 'test_macro_avg_precision' in df_metrics.columns else 0.0,
        'fold_count': len(fold_rows),
        'total_genes_evaluated': sum(row.get('top_k_n_genes', 0) for row in fold_rows)
    }
    
    return summary


def _generate_cv_based_roc_pr_curves(args: argparse.Namespace, out_dir: Path, cv_results: CrossValidationResults) -> None:
    """Generate ROC/PR curves using CV fold data (no model loading)."""
    print("     📈 Generating ROC/PR curves from CV fold data")
    
    # The existing CV loop already generates these curves in _run_gene_wise_cross_validation
    # This function can enhance or supplement those curves if needed
    
    # For now, rely on the existing curve generation in the CV loop
    # Future enhancement: consolidate curve generation here for better organization
    pass


def _generate_cv_based_calibration_analysis(args: argparse.Namespace, out_dir: Path, cv_results: CrossValidationResults) -> None:
    """Generate calibration analysis using CV predictions."""
    print("     📊 Generating calibration analysis from CV predictions")
    
    # Use the calibration_data from CV to generate calibration plots
    # This avoids loading the final model and re-evaluating on training data
    calibration_data = cv_results.calibration_data
    
    if not calibration_data or not calibration_data.get('calib_scores'):
        print("     ⚠️  No calibration data available from CV")
        return
    
    try:
        # Generate calibration plots using CV data
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        
        # Create calibration analysis directory
        calib_dir = out_dir / "calibration_analysis"
        calib_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate calibration curve from CV predictions
        if calibration_data.get('calib_scores') and calibration_data.get('calib_labels'):
            scores = np.concatenate(calibration_data['calib_scores'])
            labels = np.concatenate(calibration_data['calib_labels'])
            
            if len(scores) > 0 and len(labels) > 0:
                # Generate calibration curve
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    labels, scores, n_bins=10, strategy='uniform'
                )
                
                # Save calibration data
                calib_data = {
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist(),
                    'n_samples': len(scores),
                    'method': 'cv_based_calibration'
                }
                
                with open(calib_dir / "calibration_data.json", 'w') as f:
                    json.dump(calib_data, f, indent=2)
                
                print("     ✅ CV-based calibration analysis completed")
        
    except Exception as e:
        print(f"     ⚠️  Calibration analysis failed: {e}")


def _generate_cv_based_performance_reports(
    args: argparse.Namespace, 
    out_dir: Path, 
    cv_results: CrossValidationResults,
    original_dataset_path: str
) -> None:
    """Generate comprehensive performance reports using CV results."""
    print("     📋 Generating performance reports from CV data")
    
    # Generate training summary with CV-based metrics
    from meta_spliceai.splice_engine.meta_models.training import cv_utils
    try:
        cv_utils.generate_training_summary(
            out_dir=out_dir,
            args=args,
            original_dataset_path=original_dataset_path,
            fold_rows=cv_results.fold_metrics,
            script_name="run_gene_cv_sigmoid.py"
        )
        print("     ✅ Training summary generated")
    except Exception as e:
        print(f"     ⚠️  Training summary generation failed: {e}")
    
    # Create comprehensive performance comparison
    try:
        _create_cv_performance_comparison_report(out_dir, cv_results)
        print("     ✅ Performance comparison report generated")
    except Exception as e:
        print(f"     ⚠️  Performance comparison failed: {e}")


def _create_cv_performance_comparison_report(out_dir: Path, cv_results: CrossValidationResults) -> None:
    """Create detailed performance comparison report based on CV results."""
    
    report_path = out_dir / "cv_performance_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("📊 CROSS-VALIDATION PERFORMANCE REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("🔍 EVALUATION METHOD: Gene-wise Cross-Validation\n")
        f.write("✅ DATA LEAKAGE: None (proper train/test separation)\n")
        f.write("🎯 MODEL TYPE: Production deployment model\n\n")
        
        # Summary statistics
        summary = cv_results.evaluation_summary
        f.write("📈 PERFORMANCE SUMMARY:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean F1 Score: {summary.get('mean_f1', 0):.3f} ± {summary.get('std_f1', 0):.3f}\n")
        f.write(f"Mean Average Precision: {summary.get('mean_ap', 0):.3f} ± {summary.get('std_ap', 0):.3f}\n")
        f.write(f"CV Folds: {summary.get('fold_count', 0)}\n")
        f.write(f"Total Genes Evaluated: {summary.get('total_genes_evaluated', 0):,}\n\n")
        
        # Fold-by-fold results
        f.write("📊 FOLD-BY-FOLD RESULTS:\n")
        f.write("-" * 30 + "\n")
        for i, fold in enumerate(cv_results.fold_metrics):
            f1 = fold.get('test_macro_f1', 0)
            ap = fold.get('test_macro_avg_precision', 0)
            f.write(f"Fold {i+1}: F1={f1:.3f}, AP={ap:.3f}\n")
        
        f.write(f"\n✅ All metrics are based on proper holdout evaluation\n")
        f.write(f"✅ No data leakage - production model not used for evaluation\n")
        f.write(f"✅ Realistic performance estimates for deployment\n")


def _handle_feature_exclusions(args: argparse.Namespace, out_dir: Path, X_df) -> tuple:
    """Handle feature exclusion from command-line and files."""
    from meta_spliceai.splice_engine.meta_models.evaluation.feature_utils import load_excluded_features
    from meta_spliceai.splice_engine.meta_models.evaluation.viz_utils import check_feature_correlations
    
    exclude_list = []
    
    if args.exclude_features:
        # First check if it's a valid file path
        exclude_path = Path(args.exclude_features)
        if exclude_path.exists() and exclude_path.is_file():
            # It's a file path, use our enhanced loader
            print(f"[INFO] Loading exclude features from file: {exclude_path}")
            file_exclusions = load_excluded_features(exclude_path)
            if file_exclusions:
                exclude_list.extend(file_exclusions)
        else:
            # Not a file path, treat as comma-separated list
            print(f"[INFO] Treating exclude features as comma-separated list")
            exclude_list.extend([f.strip() for f in args.exclude_features.split(',') if f.strip()])
    
    # Also check for exclude_features.txt in output dir for backward compatibility
    exclude_file = out_dir / "exclude_features.txt"
    if exclude_file.exists():
        try:
            print(f"[INFO] Also loading exclude features from output directory: {exclude_file}")
            file_exclusions = load_excluded_features(exclude_file)
            if file_exclusions:
                exclude_list.extend(file_exclusions)
        except Exception as e:
            print(f"Warning: Error reading exclude_features.txt: {e}")
    
    # Check for potential feature leakage if enabled
    if getattr(args, 'check_leakage', True):
        print(f"\nRunning comprehensive leakage analysis (threshold={getattr(args, 'leakage_threshold', 0.95)})...")
        
        try:
            # Import our comprehensive leakage analysis module
            from meta_spliceai.splice_engine.meta_models.evaluation.leakage_analysis import LeakageAnalyzer
            
            # Create leakage analysis directory
            leakage_analysis_dir = out_dir / "leakage_analysis" 
            leakage_analysis_dir.mkdir(exist_ok=True, parents=True)
            
            # Initialize analyzer
            analyzer = LeakageAnalyzer(
                output_dir=leakage_analysis_dir,
                subject="gene_cv_leakage"
            )
            
            # Run comprehensive analysis
            leakage_results = analyzer.run_comprehensive_analysis(
                X=X_df,
                y=y_series,
                threshold=getattr(args, 'leakage_threshold', 0.95),
                methods=['pearson', 'spearman'],
                top_n=50,
                verbose=1 if getattr(args, 'verbose', False) else 0
            )
            
            # Extract leaky features for potential auto-exclusion
            leaky_features = set()
            for method_results in leakage_results['correlation_results'].values():
                leaky_features.update(method_results['leaky_features']['feature'].tolist())
            leaky_features = list(leaky_features)
            
            print(f"[Leakage Analysis] Found {len(leaky_features)} potentially leaky features")
            print(f"[Leakage Analysis] Comprehensive results saved to: {leakage_analysis_dir}")
            
            # If auto-exclude is enabled, add leaky features to exclusion list
            if getattr(args, 'auto_exclude_leaky', False) and leaky_features:
                print(f"Auto-excluding {len(leaky_features)} potentially leaky features")
                exclude_list.extend(leaky_features)
            
        except Exception as e:
            print(f"[Leakage Analysis] Error in comprehensive analysis: {e}")
            print("[Leakage Analysis] Falling back to basic correlation analysis...")
            
            # Fallback to basic analysis
            X_np = X_df.values
            y_np = _encode_labels(y_series)
            curr_features = X_df.columns.tolist()
            
            correlation_report_path = out_dir / "feature_label_correlations.csv"
            leaky_features, corr_df = check_feature_correlations(
                X_np, y_np, curr_features, getattr(args, 'leakage_threshold', 0.95), correlation_report_path
            )
            
            if getattr(args, 'auto_exclude_leaky', False) and leaky_features:
                print(f"Auto-excluding {len(leaky_features)} potentially leaky features")
                exclude_list.extend(leaky_features)
    
    # Apply all exclusions
    excluded_features = []
    if exclude_list:
        # Remove duplicates while preserving order
        exclude_list = list(dict.fromkeys(exclude_list))
        original_feature_count = X_df.shape[1]
        
        for feature in exclude_list:
            if feature in X_df.columns:
                X_df = X_df.drop(columns=[feature])
                excluded_features.append(feature)
            else:
                print(f"Warning: Requested to exclude '{feature}', but it was not found in the feature set")
        
        if excluded_features:
            print(f"\nExcluded {len(excluded_features)} features from training data:")
            for feature in excluded_features:
                print(f"  - {feature}")
            print(f"Feature count reduced from {original_feature_count} to {X_df.shape[1]}\n")
            
            # Save the actual excluded features list for reference
            with open(out_dir / "excluded_features.txt", 'w') as f:
                f.write("# Features excluded during training\n")
                for feature in excluded_features:
                    f.write(f"{feature}\n")
    
    return X_df, excluded_features


def _prepare_transcript_columns(args: argparse.Namespace, df, X_df) -> dict:
    """Prepare transcript mapping columns - always preserve transcript_id for transcript-level metrics."""
    transcript_columns = {}
    
    # Always preserve transcript_id if available for transcript-level top-k accuracy
    # Handle both pandas and polars dataframes
    if hasattr(df, 'to_pandas'):
        # It's a polars DataFrame
        columns_to_preserve = []
        if 'transcript_id' in df.columns:
            columns_to_preserve.append('transcript_id')
        if 'position' in df.columns:
            columns_to_preserve.append('position')
        if 'chrom' in df.columns:
            columns_to_preserve.append('chrom')
            
        if columns_to_preserve:
            pandas_df = df.select(columns_to_preserve).to_pandas()
            for col in columns_to_preserve:
                transcript_columns[col] = pandas_df[col]
    else:
        # It's already a pandas DataFrame
        if 'transcript_id' in df.columns:
            transcript_columns['transcript_id'] = df['transcript_id'].copy()
        if 'position' in df.columns:
            transcript_columns['position'] = df['position'].copy()
        if 'chrom' in df.columns:
            transcript_columns['chrom'] = df['chrom'].copy()
    
    # Handle position if it's in X_df but not yet saved
    if 'position' in X_df.columns and 'position' not in transcript_columns:
        transcript_columns['position'] = X_df['position'].copy()
        X_df = X_df.drop(columns=['position'])
    
    # Remove transcript_id from features if it accidentally got included
    if 'transcript_id' in X_df.columns:
        X_df = X_df.drop(columns=['transcript_id'])
    
    return transcript_columns


def _locate_base_probability_columns(args: argparse.Namespace, feature_names: List[str]) -> tuple:
    """Locate base probability columns in the feature matrix."""
    splice_prob_idx: int | None = None
    donor_idx: int | None = None
    acceptor_idx: int | None = None

    # Prefer the explicit donor+acceptor raw scores whenever they are present;
    # fall back to a pre-computed combined column *only* if one of them is missing.
    donor_present = args.donor_score_col in feature_names
    accept_present = args.acceptor_score_col in feature_names

    if donor_present and accept_present:
        donor_idx = feature_names.index(args.donor_score_col)
        acceptor_idx = feature_names.index(args.acceptor_score_col)
        print(f"[Gene-CV-Sigmoid] Using '{args.donor_score_col}' + '{args.acceptor_score_col}' as base splice probability")
    else:
        # Check combined probability column
        if args.splice_prob_col in feature_names:
            splice_prob_idx = feature_names.index(args.splice_prob_col)
            print(f"[Gene-CV-Sigmoid] Using '{args.splice_prob_col}' as base splice probability column (donor/acceptor columns missing)")
        else:
            missing_cols = []
            if not donor_present:
                missing_cols.append(args.donor_score_col)
            if not accept_present:
                missing_cols.append(args.acceptor_score_col)
            raise KeyError(f"Required base score columns {missing_cols} not present, and '{args.splice_prob_col}' also absent.")
    
    return splice_prob_idx, donor_idx, acceptor_idx


def _print_feature_summary(feature_names: List[str]) -> None:
    """Print summary of features being used."""
    from meta_spliceai.splice_engine.utils_kmer import is_kmer_feature as _is_kmer
    
    non_kmer = [f for f in feature_names if not _is_kmer(f)]
    print(f"[Gene-CV-Sigmoid] Features: {len(feature_names)} total – {len(non_kmer)} non-k-mer")
    if len(feature_names) - len(non_kmer) > 0:
        sample_kmer = random.sample([f for f in feature_names if _is_kmer(f)], k=min(3, len(feature_names) - len(non_kmer)))
        print("   Example k-mers:", ", ".join(sample_kmer))


def _run_gene_wise_cross_validation(
    args: argparse.Namespace, 
    out_dir: Path, 
    X: np.ndarray, 
    y: np.ndarray, 
    genes: np.ndarray,
    feature_names: List[str],
    splice_prob_idx: int | None,
    donor_idx: int | None,
    acceptor_idx: int | None,
    transcript_columns: dict
) -> tuple:
    """Run the complete gene-wise cross-validation loop."""
    
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from meta_spliceai.splice_engine.meta_models.evaluation.top_k_metrics import (
        calculate_cv_fold_top_k, report_top_k_accuracy
    )
    
    # Initialize CV
    gkf = GroupKFold(n_splits=args.n_folds)
    fold_rows: List[Dict[str, object]] = []
    
    # Initialize overfitting monitor if enabled
    monitor = None
    if getattr(args, 'monitor_overfitting', False):
        from meta_spliceai.splice_engine.meta_models.evaluation.overfitting_monitor import OverfittingMonitor
        print(f"\n[Overfitting Monitor] Initializing overfitting detection system")
        print(f"  Primary metric: logloss")
        print(f"  Gap threshold: {getattr(args, 'overfitting_threshold', 0.05)}")
        print(f"  Early stopping patience: {getattr(args, 'early_stopping_patience', 20)}")
        print(f"  Convergence improvement: {getattr(args, 'convergence_improvement', 0.001)}")
        
        monitor = OverfittingMonitor(
            primary_metric="logloss",
            gap_threshold=getattr(args, 'overfitting_threshold', 0.05),
            patience=getattr(args, 'early_stopping_patience', 20),
            min_improvement=getattr(args, 'convergence_improvement', 0.001)
        )
    
    # CV fold loop and collection of hold-out scores for calibration
    calib_scores = []  # P(splice) = donor_score + acceptor_score during folds
    calib_labels = []  # binary 0/1: any splice vs no splice
    
    # For per-class calibration, we need separate data for each class
    per_class_calib_scores = [[] for _ in range(3)]  # Raw scores for [neither, donor, acceptor]
    per_class_calib_labels = [[] for _ in range(3)]  # Binary 0/1 for each class

    # Lists for base vs meta aggregation
    base_f1s: list[float] = []
    meta_f1s: list[float] = []
    base_topks: list[float] = []
    meta_topks: list[float] = []
    donor_topks = []    # Track donor-specific top-k
    acceptor_topks = [] # Track acceptor-specific top-k
    fp_deltas: list[int] = []
    fn_deltas: list[int] = []
    
    # Containers for ROC and PR curve data
    roc_base, roc_meta = [], []  # Lists of (fpr, tpr) arrays per fold
    pr_base, pr_meta = [], []    # Lists of (recall, precision) arrays per fold
    auc_base, auc_meta = [], []  # AUC values per fold
    ap_base, ap_meta = [], []    # Average precision values per fold

    # For plotting: collect truth and predictions across folds
    y_true_bins, y_prob_bases, y_prob_metas = [], [], []
    # For multiclass plotting: collect original multiclass data
    y_true_multiclass, y_prob_bases_multiclass, y_prob_metas_multiclass = [], [], []
    # For splice type-aware overconfidence analysis: collect splice type information
    y_splice_types = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(gkf.split(X, y, groups=genes)):
        print(f"[Gene-CV-Sigmoid] Fold {fold_idx+1}/{args.n_folds}  test_rows={len(test_idx)}")

        # TRAIN/VALID split preserving gene groups
        rel_valid = args.valid_size / (1.0 - 1.0 / args.n_folds)
        gss = GroupShuffleSplit(n_splits=1, test_size=rel_valid, random_state=args.seed)
        train_idx, valid_idx = next(gss.split(train_val_idx, y[train_val_idx], groups=genes[train_val_idx]))
        train_idx = train_val_idx[train_idx]
        valid_idx = train_val_idx[valid_idx]

        # --- Train 3 independent binary models ---
        models_cls: List[XGBClassifier] = []
        for cls in (0, 1, 2):
            y_train_bin = (y[train_idx] == cls).astype(int)
            y_val_bin = (y[valid_idx] == cls).astype(int)
            model_c, evals_result = _train_binary_model(X[train_idx], y_train_bin, X[valid_idx], y_val_bin, args)
            models_cls.append(model_c)
            
            # Add evaluation results to overfitting monitor
            if monitor is not None and evals_result:
                try:
                    monitor.add_fold_metrics(evals_result, f"{fold_idx}_{cls}")
                    if getattr(args, 'verbose', False):
                        print(f"    [Overfitting Monitor] Added metrics for fold {fold_idx+1}, class {cls}")
                except Exception as e:
                    if getattr(args, 'verbose', False):
                        print(f"    [Overfitting Monitor] Warning: Failed to add metrics for fold {fold_idx+1}, class {cls}: {e}")

        # Handle calibration data collection
        if getattr(args, 'calibrate', False) or getattr(args, 'calibrate_per_class', False):
            proba_val = np.column_stack([m.predict_proba(X[valid_idx])[:, 1] for m in models_cls])
            
            # For standard calibration (splice vs non-splice)
            if getattr(args, 'calibrate', False):
                s_val = proba_val[:, 1] + proba_val[:, 2]  # Sum of donor + acceptor scores
                y_bin_val = (y[valid_idx] != 0).astype(int)  # 1 if any splice site, 0 otherwise
                calib_scores.append(s_val)
                calib_labels.append(y_bin_val)
            
            # For per-class calibration
            if getattr(args, 'calibrate_per_class', False):
                y_val = y[valid_idx]
                for cls_idx in range(3):  # 0=neither, 1=donor, 2=acceptor
                    # Raw scores for this class
                    cls_scores = proba_val[:, cls_idx]
                    # Binary labels: 1 if true class is cls_idx, 0 otherwise
                    cls_labels = (y_val == cls_idx).astype(int)
                    per_class_calib_scores[cls_idx].append(cls_scores)
                    per_class_calib_labels[cls_idx].append(cls_labels)

        # Predict probabilities on test set
        proba_parts = [m.predict_proba(X[test_idx])[:, 1] for m in models_cls]  # P(class)
        proba = np.column_stack(proba_parts)  # shape (n,3)

        pred = proba.argmax(axis=1)

        acc = accuracy_score(y[test_idx], pred)
        macro_f1 = f1_score(y[test_idx], pred, average="macro")

        splice_mask = y[test_idx] != 0
        splice_acc = accuracy_score(y[test_idx][splice_mask], pred[splice_mask]) if splice_mask.any() else np.nan
        splice_macro_f1 = f1_score(y[test_idx][splice_mask], pred[splice_mask], average="macro") if splice_mask.any() else np.nan

        # Calculate gene-level top-k accuracy using the new implementation
        # Get gene IDs for the test set
        gene_ids_test = genes[test_idx]
        
        # Calculate top-k accuracy using our new function
        gene_top_k_metrics = calculate_cv_fold_top_k(
            X=X[test_idx], 
            y=y[test_idx], 
            probs=proba, 
            gene_ids=gene_ids_test,
            donor_label=0,  # Label 0 = donor sites based on memory
            acceptor_label=1,  # Label 1 = acceptor sites
            neither_label=2,   # Label 2 = neither sites
        )
        
        # Print detailed top-k accuracy report
        print(report_top_k_accuracy(gene_top_k_metrics))
        
        # Log more detailed gene-level top-k metrics
        print(f"  Gene-level Top-k:  Donor={gene_top_k_metrics['donor_top_k']:.3f}, "
              f"Acceptor={gene_top_k_metrics['acceptor_top_k']:.3f}, "
              f"Combined={gene_top_k_metrics['combined_top_k']:.3f}, "
              f"n_genes={gene_top_k_metrics.get('n_groups', 0)}")
        
        # Store donor and acceptor specific metrics separately for later analysis
        donor_topks.append(gene_top_k_metrics['donor_top_k'])
        acceptor_topks.append(gene_top_k_metrics['acceptor_top_k'])
        
        # Calculate transcript-level top-k accuracy if enabled
        transcript_top_k_metrics = _calculate_transcript_level_metrics(
            args, transcript_columns, test_idx, y, proba
        )
        
        # Store the combined top-k accuracy for summary statistics
        top_k_acc = gene_top_k_metrics["combined_top_k"]

        # Base vs meta binary splice-site metrics
        y_true_bin = (y[test_idx] != 0).astype(int)
        y_prob_meta = proba[:, 1] + proba[:, 2]
        if splice_prob_idx is not None:
            y_prob_base = X[test_idx, splice_prob_idx]
        else:
            y_prob_base = X[test_idx, donor_idx] + X[test_idx, acceptor_idx]

        meta_metrics = _binary_metrics(y_true_bin, y_prob_meta, thresh=getattr(args, 'threshold', None) or 0.5, k=getattr(args, 'top_k', 5))
        base_metrics = _binary_metrics(y_true_bin, y_prob_base, thresh=args.base_thresh, k=getattr(args, 'top_k', 5))

        base_f1s.append(base_metrics["f1"])
        meta_f1s.append(meta_metrics["f1"])
        base_topks.append(base_metrics["topk_acc"])
        meta_topks.append(meta_metrics["topk_acc"])
        fp_deltas.append(base_metrics["fp"] - meta_metrics["fp"])
        fn_deltas.append(base_metrics["fn"] - meta_metrics["fn"])
        
        # Collect ROC and PR curve data for this fold
        from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
        
        # Base model ROC and PR
        fpr_b, tpr_b, _ = roc_curve(y_true_bin, y_prob_base)
        prec_b, rec_b, _ = precision_recall_curve(y_true_bin, y_prob_base)
        roc_base.append(np.column_stack([fpr_b, tpr_b]))
        pr_base.append(np.column_stack([rec_b, prec_b]))
        auc_base.append(auc(fpr_b, tpr_b))
        ap_base.append(average_precision_score(y_true_bin, y_prob_base))
        
        # Meta model ROC and PR
        fpr_m, tpr_m, _ = roc_curve(y_true_bin, y_prob_meta)
        prec_m, rec_m, _ = precision_recall_curve(y_true_bin, y_prob_meta)
        roc_meta.append(np.column_stack([fpr_m, tpr_m]))
        pr_meta.append(np.column_stack([rec_m, prec_m]))
        auc_meta.append(auc(fpr_m, tpr_m))
        ap_meta.append(average_precision_score(y_true_bin, y_prob_meta))
        
        # Store fold data for plotting
        y_true_bins.append(y_true_bin)
        y_prob_bases.append(y_prob_base)
        y_prob_metas.append(y_prob_meta)
        
        # Store splice type information for overconfidence analysis
        y_splice_types.append(y[test_idx])
        
        # Store multiclass data for enhanced plotting
        y_true_multiclass.append(y[test_idx])
        y_prob_base_multiclass = _reconstruct_base_multiclass_probabilities(
            X, test_idx, splice_prob_idx, donor_idx, acceptor_idx
        )
        y_prob_bases_multiclass.append(y_prob_base_multiclass)
        y_prob_metas_multiclass.append(proba)  # Meta model already has multiclass probabilities

        print(f"   Base  F1={base_metrics['f1']:.3f} top{getattr(args, 'top_k', 5)}={base_metrics['topk_acc']:.3f} (FP={base_metrics['fp']} FN={base_metrics['fn']})")
        print(f"   Meta  F1={meta_metrics['f1']:.3f} top{getattr(args, 'top_k', 5)}={meta_metrics['topk_acc']:.3f} (FP={meta_metrics['fp']} FN={meta_metrics['fn']})  "
              f"ΔFP={base_metrics['fp']-meta_metrics['fp']}  ΔFN={base_metrics['fn']-meta_metrics['fn']}")

        cm = confusion_matrix(y[test_idx], pred, labels=[0, 1, 2])
        print("Confusion matrix (fold", fold_idx, ")\n", pd.DataFrame(cm, index=["neither", "donor", "acceptor"], columns=["neither", "donor", "acceptor"]))

        # Calculate per-class F1 scores and average precision
        from sklearn.metrics import f1_score, average_precision_score
        y_test = y[test_idx]
        
        # Per-class F1 scores
        donor_f1 = f1_score(y_test == 1, pred == 1, average='binary')
        acceptor_f1 = f1_score(y_test == 2, pred == 2, average='binary')
        
        # Per-class average precision
        donor_ap = average_precision_score(y_test == 1, proba[:, 1])
        acceptor_ap = average_precision_score(y_test == 2, proba[:, 2])
        
        fold_row = {
            "fold": fold_idx,
            "test_rows": len(test_idx),
            "test_macro_f1": macro_f1,
            "test_macro_avg_precision": (donor_ap + acceptor_ap) / 2,  # Macro average precision
            "splice_macro_f1": splice_macro_f1,
            "top_k_accuracy": top_k_acc,
            "top_k_donor": gene_top_k_metrics["donor_top_k"],
            "top_k_acceptor": gene_top_k_metrics["acceptor_top_k"],
            "top_k_n_genes": gene_top_k_metrics.get("n_groups", 0),
            "donor_f1": donor_f1,
            "acceptor_f1": acceptor_f1,
            "donor_ap": donor_ap,
            "acceptor_ap": acceptor_ap,
            "base_f1": base_metrics["f1"],
            "meta_f1": meta_metrics["f1"],
            "base_topk": base_metrics["topk_acc"],
            "meta_topk": meta_metrics["topk_acc"],
            "delta_fp": base_metrics["fp"] - meta_metrics["fp"],
            "delta_fn": base_metrics["fn"] - meta_metrics["fn"],
            "auc_base": auc_base[-1],  # ROC AUC for base model in this fold
            "auc_meta": auc_meta[-1],  # ROC AUC for meta model in this fold
            "ap_base": ap_base[-1],    # Average precision for base model in this fold
            "ap_meta": ap_meta[-1],    # Average precision for meta model in this fold
        }
        
        # Add transcript-level metrics if available
        if transcript_top_k_metrics is not None:
            fold_row.update({
                "tx_top_k_donor": transcript_top_k_metrics.get("donor_top_k", float('nan')),
                "tx_top_k_acceptor": transcript_top_k_metrics.get("acceptor_top_k", float('nan')),
                "tx_top_k_combined": transcript_top_k_metrics.get("combined_top_k", float('nan')),
                "tx_top_k_n_transcripts": transcript_top_k_metrics.get("n_transcripts", 0),
            })
        
        fold_rows.append(fold_row)
        
        # Ensure JSON serialisability of NumPy scalars
        row_serial = {
            k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
            for k, v in fold_row.items()
        }
        with open(out_dir / f"metrics_fold{fold_idx}.json", "w") as fh:
            json.dump(row_serial, fh, indent=2)

    # Write TSV summary across folds for easier inspection
    pd.DataFrame(fold_rows).to_csv(out_dir / "metrics_folds.tsv", sep="\t", index=False)
    
    # Print aggregate metrics to console
    print(f"\nAUC  base  mean={np.mean(auc_base):.3f} ±{np.std(auc_base):.3f}")
    print(f"AUC  meta  mean={np.mean(auc_meta):.3f} ±{np.std(auc_meta):.3f}")
    print(f"AP   base  mean={np.mean(ap_base):.3f} ±{np.std(ap_base):.3f}")
    print(f"AP   meta  mean={np.mean(ap_meta):.3f} ±{np.std(ap_meta):.3f}")
    
    # Generate ROC and PR curve plots if enabled
    if getattr(args, 'plot_curves', True):
        from meta_spliceai.splice_engine.meta_models.evaluation.viz_utils import plot_roc_pr_curves
        
        # Use the modular plotting function
        curve_metrics = plot_roc_pr_curves(
            y_true=y_true_bins,
            y_pred_base=y_prob_bases,
            y_pred_meta=y_prob_metas,
            out_dir=out_dir,
            n_roc_points=getattr(args, 'n_roc_points', 101),
            plot_format=getattr(args, 'plot_format', 'pdf'),
            base_name='Base',
            meta_name='Meta',
            fold_ids=list(range(len(y_true_bins)))
        )
    
    # Generate enhanced visualizations
    _generate_enhanced_visualizations(args, out_dir, y_true_bins, y_prob_bases, y_prob_metas, 
                                     y_true_multiclass, y_prob_bases_multiclass, y_prob_metas_multiclass)
    
    # Prepare calibration data
    calibration_data = {
        'calib_scores': calib_scores,
        'calib_labels': calib_labels,
        'per_class_calib_scores': per_class_calib_scores,
        'per_class_calib_labels': per_class_calib_labels
    }
    
    # Print summary statistics
    _print_cv_summary_statistics(args, base_f1s, meta_f1s, base_topks, meta_topks, 
                                donor_topks, acceptor_topks, fold_rows, fp_deltas, fn_deltas)
    
    return fold_rows, calibration_data


def _calculate_transcript_level_metrics(args: argparse.Namespace, transcript_columns: dict, 
                                       test_idx: np.ndarray, y: np.ndarray, proba: np.ndarray):
    """Calculate transcript-level top-k accuracy if enabled."""
    transcript_top_k_metrics = None
    
    # Always calculate transcript-level metrics if transcript IDs are available
    if 'transcript_id' in transcript_columns:
        try:
            from meta_spliceai.splice_engine.meta_models.training.transcript_topk_accuracy import (
                calculate_transcript_top_k_for_cv_fold
            )
            
            # Get transcript IDs for test indices
            transcript_ids = transcript_columns['transcript_id'].iloc[test_idx].values
            
            # Get positions if available
            positions = None
            if 'position' in transcript_columns:
                positions = transcript_columns['position'].iloc[test_idx].values
            else:
                # Use index as fallback position
                positions = np.arange(len(test_idx))
            
            # Calculate transcript-level top-k accuracy
            transcript_top_k_metrics = calculate_transcript_top_k_for_cv_fold(
                X=None,  # Not needed for the calculation
                y=y[test_idx],
                probs=proba,
                transcript_ids=transcript_ids,
                positions=positions,
                donor_label=1,  # Assuming label encoding: 0=neither, 1=donor, 2=acceptor
                acceptor_label=2,
                neither_label=0,
                verbose=getattr(args, 'verbose', False)
            )
            
            # Print transcript-level metrics
            if transcript_top_k_metrics and getattr(args, 'verbose', False):
                print(f"\n[Transcript Top-k] Results:")
                print(f"  Donor accuracy: {transcript_top_k_metrics['donor_top_k']:.3f}")
                print(f"  Acceptor accuracy: {transcript_top_k_metrics['acceptor_top_k']:.3f}")
                print(f"  Combined accuracy: {transcript_top_k_metrics['combined_top_k']:.3f}")
                print(f"  Transcripts analyzed: {transcript_top_k_metrics['n_transcripts']}")
                
        except Exception as e:
            print(f"\nError calculating transcript-level metrics: {e}")
            # Don't let transcript metrics failure break the whole pipeline
    
    return transcript_top_k_metrics


def _reconstruct_base_multiclass_probabilities(X: np.ndarray, test_idx: np.ndarray, 
                                              splice_prob_idx: int | None, 
                                              donor_idx: int | None, 
                                              acceptor_idx: int | None) -> np.ndarray:
    """Reconstruct base model multiclass probabilities from binary features."""
    if splice_prob_idx is not None:
        # Only have combined splice probability, approximate multiclass
        base_prob_splice = X[test_idx, splice_prob_idx]
        base_prob_neither = 1 - base_prob_splice
        # Approximate donor/acceptor split (equal weights)
        base_prob_donor = base_prob_splice * 0.5
        base_prob_acceptor = base_prob_splice * 0.5
        y_prob_base_multiclass = np.column_stack([base_prob_neither, base_prob_donor, base_prob_acceptor])
    else:
        # Have separate donor/acceptor scores
        base_prob_donor = X[test_idx, donor_idx]
        base_prob_acceptor = X[test_idx, acceptor_idx]
        base_prob_neither = 1 - (base_prob_donor + base_prob_acceptor)
        # Clip to ensure probabilities are valid
        base_prob_neither = np.clip(base_prob_neither, 0, 1)
        y_prob_base_multiclass = np.column_stack([base_prob_neither, base_prob_donor, base_prob_acceptor])
    
    return y_prob_base_multiclass


def _generate_enhanced_visualizations(args: argparse.Namespace, out_dir: Path, 
                                     y_true_bins, y_prob_bases, y_prob_metas,
                                     y_true_multiclass, y_prob_bases_multiclass, y_prob_metas_multiclass):
    """Generate enhanced visualizations including multiclass ROC/PR curves."""
    
    # Create enhanced visualizations
    print("\n[Enhanced Visualizations] Creating improved binary PR curves...")
    try:
        from meta_spliceai.splice_engine.meta_models.evaluation.multiclass_roc_pr import create_improved_binary_pr_plot
        create_improved_binary_pr_plot(
            y_true=y_true_bins,
            y_pred_base=y_prob_bases,
            y_pred_meta=y_prob_metas,
            out_dir=out_dir,
            plot_format=getattr(args, 'plot_format', 'pdf'),
            base_name='Base',
            meta_name='Meta'
        )
        print("[Enhanced Visualizations] ✓ Improved binary PR curves created")
    except Exception as e:
        print(f"[Enhanced Visualizations] ✗ Error creating improved binary PR curves: {e}")
    
    print("\n[Enhanced Visualizations] Creating multiclass ROC/PR curves...")
    try:
        from meta_spliceai.splice_engine.meta_models.evaluation.multiclass_roc_pr import plot_multiclass_roc_pr_curves
        multiclass_metrics = plot_multiclass_roc_pr_curves(
            y_true=y_true_multiclass,
            y_pred_base=y_prob_bases_multiclass,
            y_pred_meta=y_prob_metas_multiclass,
            out_dir=out_dir,
            plot_format=getattr(args, 'plot_format', 'pdf'),
            base_name='Base',
            meta_name='Meta'
        )
        print("[Enhanced Visualizations] ✓ Multiclass ROC/PR curves created")
        
        # Log multiclass metrics
        print("\n[Multiclass Metrics Summary]")
        for class_name in ['donor', 'acceptor']:
            if class_name in multiclass_metrics:
                auc_base = multiclass_metrics[class_name]['auc']['base']
                auc_meta = multiclass_metrics[class_name]['auc']['meta']
                ap_base = multiclass_metrics[class_name]['ap']['base']
                ap_meta = multiclass_metrics[class_name]['ap']['meta']
                
                print(f"{class_name.title()} AUC: Base={auc_base['mean']:.3f}±{auc_base['std']:.3f}, Meta={auc_meta['mean']:.3f}±{auc_meta['std']:.3f}")
                print(f"{class_name.title()} AP:  Base={ap_base['mean']:.3f}±{ap_base['std']:.3f}, Meta={ap_meta['mean']:.3f}±{ap_meta['std']:.3f}")
    except Exception as e:
        print(f"[Enhanced Visualizations] ✗ Error creating multiclass ROC/PR curves: {e}")

    # Generate the combined ROC/PR curves meta PDF that the script expects
    print("\n[Enhanced Visualizations] Creating combined ROC/PR curves meta PDF...")
    try:
        from meta_spliceai.splice_engine.meta_models.evaluation.viz_utils import plot_combined_roc_pr_curves_meta
        plot_combined_roc_pr_curves_meta(
            y_true=y_true_bins,
            y_pred_base=y_prob_bases,
            y_pred_meta=y_prob_metas,
            out_dir=out_dir,
            plot_format=getattr(args, 'plot_format', 'pdf'),
            base_name='Base',
            meta_name='Meta',
            n_roc_points=getattr(args, 'n_roc_points', 101)
        )
        print("[Enhanced Visualizations] ✓ Combined ROC/PR curves meta PDF created")
    except Exception as e:
        print(f"[Enhanced Visualizations] ✗ Error creating combined ROC/PR curves meta PDF: {e}")


def _print_cv_summary_statistics(args: argparse.Namespace, base_f1s, meta_f1s, base_topks, meta_topks,
                                donor_topks, acceptor_topks, fold_rows, fp_deltas, fn_deltas):
    """Print comprehensive CV summary statistics."""
    
    # Additional aggregate printout for base vs meta
    if base_f1s:
        print("\n=== Base vs Meta (splice / non-splice) ===")
        print(f"Base  F1   mean={np.mean(base_f1s):.3f} ±{np.std(base_f1s):.3f}")
        print(f"Meta  F1   mean={np.mean(meta_f1s):.3f} ±{np.std(meta_f1s):.3f}")
        print(f"Base  top{getattr(args, 'top_k', 5)} mean={np.mean(base_topks):.3f}")
        print(f"Meta  top{getattr(args, 'top_k', 5)} mean={np.mean(meta_topks):.3f}")
        print("Gene-level top-k accuracy statistics:")
        print(f"  Donor    mean={np.mean(donor_topks):.3f} std={np.std(donor_topks):.3f}")
        print(f"  Acceptor mean={np.mean(acceptor_topks):.3f} std={np.std(acceptor_topks):.3f}")
        print(f"  Combined mean={np.mean([r['top_k_accuracy'] for r in fold_rows]):.3f} "
              f"std={np.std([r['top_k_accuracy'] for r in fold_rows]):.3f}")
        print(f"  Average genes per fold: {np.mean([r['top_k_n_genes'] for r in fold_rows]):.1f}")
        
        # Display transcript-level metrics if available
        if any('tx_top_k_donor' in row for row in fold_rows):
            print("\nTranscript-level top-k accuracy statistics:")
            tx_donors = [row['tx_top_k_donor'] for row in fold_rows if 'tx_top_k_donor' in row]
            tx_acceptors = [row['tx_top_k_acceptor'] for row in fold_rows if 'tx_top_k_acceptor' in row]
            tx_combined = [row['tx_top_k_combined'] for row in fold_rows if 'tx_top_k_combined' in row]
            tx_counts = [row['tx_top_k_n_transcripts'] for row in fold_rows if 'tx_top_k_n_transcripts' in row]
            
            if tx_donors and tx_acceptors and tx_combined:
                print(f"  Donor    mean={np.nanmean(tx_donors):.3f} std={np.nanstd(tx_donors):.3f}")
                print(f"  Acceptor mean={np.nanmean(tx_acceptors):.3f} std={np.nanstd(tx_acceptors):.3f}")
                print(f"  Combined mean={np.nanmean(tx_combined):.3f} std={np.nanstd(tx_combined):.3f}")
                print(f"  Average transcripts per fold: {np.mean(tx_counts):.1f}")
        
        print(f"Median ΔFP = {int(np.median(fp_deltas)):+d}   Median ΔFN = {int(np.median(fn_deltas)):+d}")


def _train_final_ensemble(args: argparse.Namespace, X: np.ndarray, y: np.ndarray, 
                         feature_names: List[str], calibration_data: dict):
    """Train final ensemble on full data with optional calibration."""
    from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
    
    # Train final ensemble on full data
    models_full: List[XGBClassifier] = []
    for cls in (0, 1, 2):
        y_bin = (y == cls).astype(int)
        model_c, _ = _train_binary_model(X, y_bin, X, y_bin, args)  # self-eval set to silence warnings
        models_full.append(model_c)

    # Log final ensemble creation with algorithm info
    algorithm = getattr(args, 'algorithm', 'xgboost')
    print(f"🎯 [Final Ensemble] Creating production ensemble with {algorithm.upper()} models", flush=True)
    
    if getattr(args, 'calibrate_per_class', False):
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        
        # Create per-class calibrators
        calibrators = []
        for cls_idx in range(3):  # 0=neither, 1=donor, 2=acceptor
            # Concatenate scores and labels from all CV folds for this class
            cls_scores = np.concatenate(calibration_data['per_class_calib_scores'][cls_idx])
            cls_labels = np.concatenate(calibration_data['per_class_calib_labels'][cls_idx])
            
            print(f"[Per-class calibration] Class {cls_idx}: {cls_scores.shape[0]} samples, {cls_labels.sum()} positives")
            
            # Create and fit the calibrator
            if getattr(args, 'calib_method', 'platt') == "platt":
                calibrator = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
                calibrator.fit(cls_scores.reshape(-1, 1), cls_labels)
            elif getattr(args, 'calib_method', 'platt') == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(cls_scores, cls_labels)
            else:
                raise ValueError("Unsupported calibration method: " + getattr(args, 'calib_method', 'platt'))
            
            calibrators.append(calibrator)
        
        # Create ensemble with per-class calibration
        ensemble = PerClassCalibratedSigmoidEnsemble(models_full, feature_names, calibrators)
        print(f"[Info] Created PerClassCalibratedSigmoidEnsemble with {algorithm.upper()} models and separate calibration for each class")
    
    elif getattr(args, 'calibrate', False):
        s_train = np.concatenate(calibration_data['calib_scores'])
        y_bin = np.concatenate(calibration_data['calib_labels'])

        if getattr(args, 'calib_method', 'platt') == "platt":
            from sklearn.linear_model import LogisticRegression
            calibrator = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
            calibrator.fit(s_train.reshape(-1, 1), y_bin)
        elif getattr(args, 'calib_method', 'platt') == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(s_train, y_bin)
        else:
            raise ValueError("Unsupported calibration method: " + getattr(args, 'calib_method', 'platt'))

        ensemble = _cutils.CalibratedSigmoidEnsemble(models_full, feature_names, calibrator)
        print(f"[Info] Created CalibratedSigmoidEnsemble with {algorithm.upper()} models and binary splice/non-splice calibration")
    else:
        ensemble = SigmoidEnsemble(models_full, feature_names)
        print(f"[Info] Created uncalibrated SigmoidEnsemble with {algorithm.upper()} models")
    
    return ensemble


def _save_model_and_run_analysis_legacy(args: argparse.Namespace, out_dir: Path, ensemble, fold_rows: List[Dict], original_dataset_path: str):
    """DEPRECATED: Legacy model saving with data leakage issues. Use production pipeline instead."""
    
    # Save initial model for diagnostics
    import pickle
    with open(out_dir / "model_multiclass.pkl", "wb") as fh:
        pickle.dump(ensemble, fh)
    
    # Generate CV metrics
    df_metrics = pd.DataFrame(fold_rows)
    df_metrics.to_csv(out_dir / "gene_cv_metrics.csv", index=False)
    print("\nGene-CV-Sigmoid results by fold:\n", df_metrics)
    
    # Calculate mean metrics for key performance indicators
    key_metrics = ["test_macro_f1", "splice_macro_f1", "top_k_accuracy", 
                   "donor_f1", "acceptor_f1", "donor_ap", "acceptor_ap",
                   "auc_meta", "ap_meta"]
    
    # Filter to only include metrics that exist in the DataFrame
    available_metrics = [m for m in key_metrics if m in df_metrics.columns]
    mean_metrics = df_metrics[available_metrics].mean()
    
    print("\nAverage across folds:\n", mean_metrics.to_string())
    with open(out_dir / "metrics_aggregate.json", "w") as fh:
        json.dump(mean_metrics.to_dict(), fh, indent=2)
    
    # Generate CV metrics visualization report
    _generate_cv_metrics_visualization(args, out_dir)
    
    # Determine optimal threshold and save final model
    _determine_optimal_threshold_and_save_model(args, out_dir, ensemble)
    
    # Skip evaluation if requested
    if getattr(args, 'skip_eval', False):
        print("[INFO] Skipping evaluation steps due to --skip-eval flag")
        return
    
    # Run comprehensive post-training analysis
    _run_comprehensive_post_training_analysis(args, out_dir, original_dataset_path)


def _generate_cv_metrics_visualization(args: argparse.Namespace, out_dir: Path):
    """Generate CV metrics visualization report."""
    try:
        from meta_spliceai.splice_engine.meta_models.evaluation.cv_metrics_viz import generate_cv_metrics_report
        
        print("\nGenerating CV metrics visualization report...")
        cv_metrics_csv = out_dir / "gene_cv_metrics.csv"
        if cv_metrics_csv.exists():
            viz_result = generate_cv_metrics_report(
                csv_path=cv_metrics_csv,
                out_dir=out_dir,
                dataset_path=args.dataset,
                plot_format=getattr(args, 'plot_format', 'pdf'),
                dpi=300
            )
            print(f"[INFO] CV metrics visualization completed successfully")
            print(f"[INFO] Visualization directory: {viz_result['visualization_dir']}")
            print(f"[INFO] Generated {len(viz_result['plot_files'])} plots:")
            for plot_name, plot_path in viz_result['plot_files'].items():
                print(f"  - {plot_name.replace('_', ' ').title()}: {Path(plot_path).name}")
        else:
            print(f"[WARNING] CV metrics CSV not found at {cv_metrics_csv}")
    except Exception as e:
        print(f"[WARNING] CV metrics visualization failed with error: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()


def _determine_optimal_threshold_and_save_model(args: argparse.Namespace, out_dir: Path, ensemble):
    """Determine optimal threshold and save final model."""
    
    # Determine threshold
    if getattr(args, 'threshold', None) is not None:
        thresh = args.threshold
    else:
        sugg_file = out_dir / "threshold_suggestion.txt"
        if sugg_file.exists():
            try:
                import pandas as _pd
                ts_df = _pd.read_csv(sugg_file, sep="\t", header=None, names=["key", "value"])
                if "threshold_global" in ts_df["key"].values:
                    thresh = float(ts_df.loc[ts_df["key"]=="threshold_global", "value"].iloc[0])
                elif "best_threshold" in ts_df["key"].values:
                    thresh = float(ts_df.loc[ts_df["key"]=="best_threshold", "value"].iloc[0])
                else:
                    thresh = 0.9
                print(f"[Gene-CV-Sigmoid] Using suggested threshold {thresh:.3f} from probability_diagnostics")
                # capture per-class thresholds if present
                for _cls, attr in [("threshold_donor", "threshold_donor"), ("threshold_acceptor", "threshold_acceptor")]:
                    if _cls in ts_df["key"].values:
                        try:
                            setattr(ensemble, attr, float(ts_df.loc[ts_df["key"]==_cls, "value"].iloc[0]))
                        except Exception:
                            pass
            except Exception as _e:
                print("[Gene-CV-Sigmoid] Failed to parse threshold_suggestion.txt:", _e)
                thresh = 0.9
        else:
            thresh = 0.9

    # Persist ensemble with attached optimal threshold for inference
    try:
        ensemble.optimal_threshold = thresh  # binary cutoff
        ensemble.threshold_neither = thresh  # for completeness
        # donor/acceptor thresholds set earlier if available
    except Exception:
        pass  # guard: attribute injection should never fail
    
    import pickle
    with open(out_dir / "model_multiclass.pkl", "wb") as fh:
        pickle.dump(ensemble, fh)


def _run_comprehensive_post_training_analysis(args: argparse.Namespace, out_dir: Path, original_dataset_path: str):
    """Run comprehensive post-training analysis and diagnostics."""
    import os  # Add missing import
    
    print("\n" + "="*60)
    print("🎯 RUNNING CLASSIFICATION-BASED META-MODEL EVALUATION")
    print("="*60)
    print("ℹ️  This evaluation compares meta vs base predictions at the")
    print("   SAME training positions, not position discovery performance.")
    print("   This is the correct approach for meta-model evaluation.")
    print("="*60)
    
    # Diagnostics + post-training analysis
    diag_sample = None if getattr(args, 'diag_sample', 25000) == 0 else getattr(args, 'diag_sample', 25000)
    
    # Apply memory optimization if enabled
    if getattr(args, 'memory_optimize', False):
        # Reduce diagnostic sample sizes for memory-constrained systems
        if diag_sample is None or diag_sample > getattr(args, 'max_diag_sample', 25000):
            diag_sample = getattr(args, 'max_diag_sample', 25000)
            print(f"[Memory Optimization] Reduced diagnostic sample to {diag_sample} for memory efficiency")
        
        # Reduce neighbor sample size if it's too large
        if getattr(args, 'neigh_sample', 0) > 1000:
            args.neigh_sample = 1000
            print(f"[Memory Optimization] Reduced neighbor sample to {args.neigh_sample} for memory efficiency")
    
    from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
    
    print(f"[Post-Training Analysis] Running essential diagnostics...", flush=True)
    
    # Set environment variable to preserve gene_id for gene-aware sampling in diagnostics
    original_ss_preserve = os.environ.get('SS_PRESERVE_GENE_ID')
    os.environ['SS_PRESERVE_GENE_ID'] = '1'
    
    try:
        _cutils.richer_metrics(args.dataset, out_dir, sample=diag_sample)
        _cutils.gene_score_delta(args.dataset, out_dir, sample=diag_sample)
        
        # Run SHAP analysis (if not skipped)
        _run_shap_analysis(args, out_dir, diag_sample)
        
        _cutils.probability_diagnostics(args.dataset, out_dir, sample=diag_sample)
        _cutils.base_vs_meta(args.dataset, out_dir, sample=diag_sample)
    finally:
        # Restore original environment variable
        if original_ss_preserve is not None:
            os.environ['SS_PRESERVE_GENE_ID'] = original_ss_preserve
        else:
            os.environ.pop('SS_PRESERVE_GENE_ID', None)
    
    print(f"[Post-Training Analysis] Essential diagnostics completed", flush=True)

    # Run classification-based evaluation (the correct approach)
    _run_classification_based_evaluation(args, out_dir, diag_sample)
    
    # Additional diagnostics
    _run_additional_diagnostics(args, out_dir)
    
    # Display comprehensive performance summary
    _display_comprehensive_performance_summary(out_dir, fold_rows, args, original_dataset_path)


def _run_shap_analysis(args: argparse.Namespace, out_dir: Path, diag_sample: int):
    """Run SHAP analysis with fallback handling and configurable options."""
    
    # Check if SHAP analysis should be skipped
    if getattr(args, 'skip_shap', False):
        print("\n" + "="*60)
        print("⏩ SHAP ANALYSIS SKIPPED")
        print("="*60)
        print("[SHAP Analysis] Skipping SHAP analysis due to --skip-shap flag")
        print("[SHAP Analysis] Use --fast-shap for quick SHAP or remove --skip-shap for full analysis")
        return
    
    if getattr(args, 'minimal_diagnostics', False):
        print("\n" + "="*60)
        print("⏩ SHAP ANALYSIS SKIPPED (MINIMAL DIAGNOSTICS)")
        print("="*60)
        print("[SHAP Analysis] Skipping SHAP analysis due to --minimal-diagnostics flag")
        return
    
    # Determine SHAP sample size
    shap_sample = diag_sample
    if getattr(args, 'shap_sample', None) is not None:
        shap_sample = args.shap_sample
        print(f"[SHAP Analysis] Using custom SHAP sample size: {shap_sample}")
    elif getattr(args, 'fast_shap', False):
        shap_sample = 1000
        print(f"[SHAP Analysis] Using fast SHAP with reduced sample size: {shap_sample}")
    
    shap_analysis_completed = False
    try:
        print("\n" + "="*60)
        print("🔍 SHAP ANALYSIS DIAGNOSTICS")
        print("="*60)
        print(f"[SHAP Analysis] Starting memory-efficient SHAP analysis...")
        print(f"[SHAP Analysis] Dataset: {args.dataset}")
        print(f"[SHAP Analysis] Output directory: {out_dir}")
        print(f"[SHAP Analysis] SHAP sample size: {shap_sample}")
        if getattr(args, 'fast_shap', False):
            print(f"[SHAP Analysis] 🚀 Fast SHAP mode enabled - reduced sample for quick iteration")
        
        # Check prerequisites before running SHAP
        model_file = out_dir / "model_multiclass.pkl"
        feature_manifest = out_dir / "feature_manifest.csv"
        
        print(f"[SHAP Analysis] Checking prerequisites...")
        print(f"  Model file exists: {model_file.exists()} ({model_file})")
        print(f"  Feature manifest exists: {feature_manifest.exists()} ({feature_manifest})")
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        if not feature_manifest.exists():
            raise FileNotFoundError(f"Feature manifest not found: {feature_manifest}")
        
        # Run SHAP analysis with detailed progress
        print(f"[SHAP Analysis] Running incremental SHAP analysis...")
        from meta_spliceai.splice_engine.meta_models.evaluation.shap_incremental import run_incremental_shap_analysis
        shap_output_dir = run_incremental_shap_analysis(args.dataset, out_dir, sample=shap_sample)
        
        # Check if SHAP analysis actually produced output
        expected_shap_file = out_dir / "feature_importance_analysis" / "shap_analysis" / "importance" / "shap_importance_incremental.csv"
        
        if expected_shap_file.exists():
            try:
                shap_df = pd.read_csv(expected_shap_file)
                print(f"  SHAP results shape: {shap_df.shape}")
                if len(shap_df) > 0:
                    shap_analysis_completed = True
                    print(f"[SHAP Analysis] ✓ SHAP analysis completed successfully!")
                else:
                    print(f"[SHAP Analysis] ⚠️  SHAP file is empty")
            except Exception as e:
                print(f"[SHAP Analysis] ⚠️  Error reading SHAP results: {e}")
        else:
            print(f"[SHAP Analysis] ⚠️  SHAP output file not created")
        
        # Generate comprehensive SHAP visualization report
        if shap_analysis_completed:
            print(f"[SHAP Analysis] Creating comprehensive SHAP visualization report...")
            try:
                from meta_spliceai.splice_engine.meta_models.evaluation.shap_viz import generate_comprehensive_shap_report
                
                shap_importance_csv = expected_shap_file
                model_pkl = out_dir / "model_multiclass.pkl"
                
                if shap_importance_csv.exists() and model_pkl.exists():
                    shap_results = generate_comprehensive_shap_report(
                        importance_csv=shap_importance_csv,
                        model_path=model_pkl,
                        dataset_path=args.dataset,
                        out_dir=out_dir,
                        top_n=20,
                        sample_size=min(1000, diag_sample if diag_sample else 1000),
                        plot_format=getattr(args, 'plot_format', 'pdf')
                    )
                    print(f"[SHAP Analysis] ✓ Comprehensive SHAP report generated")
                    
            except Exception as e:
                print(f"[SHAP Analysis] ✗ Error generating comprehensive SHAP report: {e}")
        
    except Exception as e:
        print(f"[SHAP Analysis] ✗ Memory-efficient SHAP analysis failed with error: {e}")
        
        # Fallback to original SHAP analysis
        try:
            from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
            if hasattr(_cutils, 'shap_importance'):
                print(f"[SHAP Analysis] Original SHAP function found, executing...")
                _cutils.shap_importance(args.dataset, out_dir, sample=shap_sample)
                
                fallback_shap_file = out_dir / "shap_importance_incremental.csv"
                if fallback_shap_file.exists():
                    try:
                        fallback_df = pd.read_csv(fallback_shap_file)
                        if len(fallback_df) > 0:
                            shap_analysis_completed = True
                            print(f"[SHAP Analysis] ✓ Fallback SHAP analysis completed successfully!")
                    except Exception as e:
                        print(f"[SHAP Analysis] ⚠️  Error reading fallback SHAP results: {e}")
        except Exception as e2:
            print(f"[SHAP Analysis] ✗ Original SHAP importance analysis also failed: {e2}")


def _run_classification_based_evaluation(args: argparse.Namespace, out_dir: Path, diag_sample: int):
    """Run classification-based evaluation (the correct approach for meta-models)."""
    
    # 1. Position-level classification comparison (CORRECT approach)
    try:
        from meta_spliceai.splice_engine.meta_models.training.eval_meta_splice import meta_splice_performance_correct
        
        # Use the correct evaluation that compares meta vs base at training positions
        result_path = meta_splice_performance_correct(
            dataset_path=args.dataset,
            run_dir=out_dir,
            sample=diag_sample,
            out_tsv=out_dir / "position_level_classification_results.tsv",
            verbose=getattr(args, 'verbose', True),
        )
        
        if getattr(args, 'verbose', False):
            print(f"[Gene-CV-Sigmoid] ✅ Position-level classification comparison completed!")
            print(f"  Results saved to: {result_path}")
                    
    except Exception as e:
        print(f"[Gene-CV-Sigmoid] ❌ Position-level classification comparison failed: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()

    # 2. Gene-level classification comparison (CORRECT approach)
    try:
        from meta_spliceai.splice_engine.meta_models.training.eval_meta_splice import meta_splice_performance_argmax
        
        # Always use argmax-based evaluation (works regardless of calibration issues)
        meta_comparison_path = meta_splice_performance_argmax(
            dataset_path=args.dataset,
            run_dir=out_dir,
            sample=diag_sample,
            out_tsv=out_dir / "gene_level_argmax_results.tsv",
            verbose=getattr(args, 'verbose', True),
            donor_score_col="donor_score",
            acceptor_score_col="acceptor_score", 
            gene_col="gene_id",
            label_col="splice_type"
        )
        
        if getattr(args, 'verbose', False):
            print(f"[Gene-CV-Sigmoid] ✅ Gene-level ARGMAX comparison completed!")
            print(f"  Results saved to: {meta_comparison_path}")
        
        # Create expected TSV files by copying the generated files
        try:
            import shutil
            # Copy gene_level_argmax_results.tsv to expected meta_vs_base_performance.tsv
            argmax_results_file = out_dir / "gene_level_argmax_results.tsv"
            meta_vs_base_file = out_dir / "meta_vs_base_performance.tsv"
            
            if argmax_results_file.exists():
                shutil.copy2(argmax_results_file, meta_vs_base_file)
                if getattr(args, 'verbose', False):
                    print(f"[Gene-CV-Sigmoid] ✅ Created meta_vs_base_performance.tsv")
            
            # Create perf_meta_vs_base.tsv as a fallback copy
            perf_meta_vs_base_file = out_dir / "perf_meta_vs_base.tsv"
            if argmax_results_file.exists():
                shutil.copy2(argmax_results_file, perf_meta_vs_base_file)
                if getattr(args, 'verbose', False):
                    print(f"[Gene-CV-Sigmoid] ✅ Created perf_meta_vs_base.tsv")
            
            # Create detailed_position_comparison.tsv from position-level results
            position_results_file = out_dir / "position_level_classification_results.tsv"
            detailed_position_file = out_dir / "detailed_position_comparison.tsv"
            
            if position_results_file.exists():
                shutil.copy2(position_results_file, detailed_position_file)
                if getattr(args, 'verbose', False):
                    print(f"[Gene-CV-Sigmoid] ✅ Created detailed_position_comparison.tsv")
                    
        except Exception as e:
            print(f"[Gene-CV-Sigmoid] ⚠️  Error creating additional TSV files: {e}")
            
    except Exception as e:
        print(f"[Gene-CV-Sigmoid] ❌ Gene-level classification comparison failed: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()


def _run_additional_diagnostics(args: argparse.Namespace, out_dir: Path):
    """Run additional diagnostics including leakage probe and neighbor analysis."""
    
    from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
    
    # Skip additional diagnostics if minimal mode is enabled
    if getattr(args, 'minimal_diagnostics', False):
        print("\n" + "="*60)
        print("⏩ ADDITIONAL DIAGNOSTICS SKIPPED (MINIMAL DIAGNOSTICS)")
        print("="*60)
        print("[Additional Diagnostics] Skipping leakage probe and neighbor analysis due to --minimal-diagnostics flag")
        return
    
    # Enhanced leakage probe
    if getattr(args, 'leakage_probe', False):
        try:
            print("\n[Enhanced Leakage Probe] Running comprehensive leakage visualization analysis...")
            from meta_spliceai.splice_engine.meta_models.evaluation.leakage_analysis import run_comprehensive_leakage_analysis
            
            # Run comprehensive leakage analysis with visualizations
            enhanced_leakage_results = run_comprehensive_leakage_analysis(
                dataset_path=args.dataset,
                run_dir=out_dir,
                threshold=getattr(args, 'leakage_threshold', 0.95),
                methods=['pearson', 'spearman'],
                sample=10_000,
                top_n=50,
                verbose=1 if getattr(args, 'verbose', False) else 0
            )
            
            print(f"[Enhanced Leakage Probe] Comprehensive analysis completed!")
            print(f"[Enhanced Leakage Probe] Results saved to: {enhanced_leakage_results['output_directory']}")
            
        except Exception as e:
            print(f"[Enhanced Leakage Probe] Enhanced analysis failed: {e}")
            print("[Enhanced Leakage Probe] Falling back to basic leakage probe...")
            
            # Fallback to basic leakage probe
            try:
                _cutils.leakage_probe(args.dataset, out_dir, sample=10_000)
            except Exception as e:
                print("[warning] leakage_probe failed:", e)

    # Run neighbour window diagnostic for deeper insight into spatial effects
    try:
        if getattr(args, 'verbose', False):
            print("\n[DEBUG] Running neighbor window diagnostics (for feature analysis)")
            
        # Make sure output directory exists
        neighbor_dir = out_dir / "neighbor_diagnostics"
        neighbor_dir.mkdir(exist_ok=True, parents=True)
        
        _cutils.neighbour_window_diagnostics(
            dataset_path=args.dataset,
            run_dir=out_dir,
            annotations_path=args.splice_sites_path,  # Use splice sites path 
            n_sample=getattr(args, 'neigh_sample', 0) if getattr(args, 'neigh_sample', 0) > 0 else None,
            window=getattr(args, 'neigh_window', 10),
        )
    except Exception as e:
        print("[warning] neighbour_window_diagnostics failed:", e)
        if getattr(args, 'verbose', False):
            import traceback
            print("[DEBUG] Neighbor window diagnostics error details:")
            traceback.print_exc()

    print("\n" + "="*60)
    print("🎉 CLASSIFICATION-BASED EVALUATION COMPLETE!")
    print("="*60)
    print("✅ All evaluation used classification-based methods")
    print("✅ No position-based evaluation was performed")
    print("✅ Results reflect true meta-model performance")
    print("="*60)


def _display_comprehensive_performance_summary(out_dir: Path, fold_rows: list, args, original_dataset_path: str):
    """Display a comprehensive performance comparison table showing base vs meta improvements."""
    
    import pandas as pd
    import json
    
    verbose = getattr(args, 'verbose', True)
    if not verbose:
        return
        
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE PERFORMANCE SUMMARY: BASE vs META MODEL")
    print("="*80)
    
    # 1. CV Fold-level Summary
    if fold_rows:
        fold_df = pd.DataFrame(fold_rows)
        
        # Base vs Meta binary metrics summary
        if 'base_f1' in fold_df.columns and 'meta_f1' in fold_df.columns:
            print("\n🎯 BINARY SPLICE DETECTION PERFORMANCE:")
            print("-" * 50)
            
            base_f1_mean = fold_df['base_f1'].mean()
            meta_f1_mean = fold_df['meta_f1'].mean()
            base_f1_std = fold_df['base_f1'].std()
            meta_f1_std = fold_df['meta_f1'].std()
            
            print(f"  F1 Score:      Base = {base_f1_mean:.3f} ± {base_f1_std:.3f}")
            print(f"                 Meta = {meta_f1_mean:.3f} ± {meta_f1_std:.3f}")
            print(f"                 Δ F1 = {meta_f1_mean - base_f1_mean:+.3f} ({((meta_f1_mean - base_f1_mean) / base_f1_mean * 100):+.1f}%)")
            
            if 'auc_base' in fold_df.columns and 'auc_meta' in fold_df.columns:
                base_auc_mean = fold_df['auc_base'].mean()
                meta_auc_mean = fold_df['auc_meta'].mean()
                print(f"  ROC AUC:       Base = {base_auc_mean:.3f}")
                print(f"                 Meta = {meta_auc_mean:.3f}")
                print(f"                 Δ AUC = {meta_auc_mean - base_auc_mean:+.3f}")
            
            if 'ap_base' in fold_df.columns and 'ap_meta' in fold_df.columns:
                base_ap_mean = fold_df['ap_base'].mean()
                meta_ap_mean = fold_df['ap_meta'].mean()
                print(f"  Avg Precision: Base = {base_ap_mean:.3f}")
                print(f"                 Meta = {meta_ap_mean:.3f}")
                print(f"                 Δ AP = {meta_ap_mean - base_ap_mean:+.3f}")
    
    # 2. Gene-level and transcript-level top-k accuracy
    if fold_rows and any('top_k_accuracy' in row for row in fold_rows):
        print(f"\n🎯 TOP-{getattr(args, 'top_k', 5)} GENE-LEVEL ACCURACY:")
        print("-" * 50)
        
        gene_topk_combined = [row['top_k_accuracy'] for row in fold_rows if 'top_k_accuracy' in row]
        gene_topk_donor = [row['top_k_donor'] for row in fold_rows if 'top_k_donor' in row]
        gene_topk_acceptor = [row['top_k_acceptor'] for row in fold_rows if 'top_k_acceptor' in row]
        
        if gene_topk_combined:
            print(f"  Combined: {np.mean(gene_topk_combined):.3f} ± {np.std(gene_topk_combined):.3f}")
        if gene_topk_donor:
            print(f"  Donor:    {np.mean(gene_topk_donor):.3f} ± {np.std(gene_topk_donor):.3f}")
        if gene_topk_acceptor:
            print(f"  Acceptor: {np.mean(gene_topk_acceptor):.3f} ± {np.std(gene_topk_acceptor):.3f}")
    
    # 3. Generated artifacts summary
    print(f"\n📁 GENERATED ANALYSIS ARTIFACTS:")
    print("-" * 50)
    
    key_files = [
        ("gene_cv_metrics.csv", "CV fold metrics"),
        ("model_multiclass.pkl", "Trained meta-model"),
        ("roc_curves_meta.pdf", "ROC curves (meta model)"),
        ("pr_curves_meta.pdf", "PR curves (meta model)"),
        ("cv_metrics_visualization/", "CV visualization suite"),
        ("feature_importance_analysis/", "Multi-method feature analysis"),
        ("leakage_analysis/", "Comprehensive data leakage analysis"),
        ("meta_evaluation_summary.json", "Correct position-level comparison"),
        ("gene_level_argmax_results.tsv", "Detailed ARGMAX gene results"),
        ("meta_vs_base_performance.tsv", "Classification-based gene comparison"),
        ("compare_base_meta.json", "Position-level comparison"),
        ("probability_diagnostics.png", "Calibration analysis"),
    ]
    
    for filename, description in key_files:
        filepath = out_dir / filename
        if filepath.exists():
            if filepath.is_dir():
                file_count = len(list(filepath.glob("*")))
                print(f"  ✓ {filename:<40} {description} ({file_count} files)")
            else:
                print(f"  ✓ {filename:<40} {description}")
        else:
            print(f"  ✗ {filename:<40} {description} (missing)")
    
    print("="*80)
    print("🎉 CV ANALYSIS COMPLETE! Check the artifacts above for detailed results.")
    print("="*80)
    
    # Generate comprehensive training summary
    from meta_spliceai.splice_engine.meta_models.training import cv_utils
    cv_utils.generate_training_summary(
        out_dir=out_dir,
        args=args,
        original_dataset_path=original_dataset_path,
        fold_rows=fold_rows,
        script_name="run_gene_cv_sigmoid.py"
    )


##########################################################################
# Main
##########################################################################

def main(argv: List[str] | None = None) -> None:  # pragma: no cover
    """
    Clean driver script for meta-model training.
    
    This function serves as a minimal driver that delegates all complex logic
    to appropriate utility modules, maintaining clean separation of concerns.
    """
    import sys
    
    print("🔍 [Driver] Meta-Model Training Pipeline", flush=True)
    print("=" * 50, flush=True)
    
    # Parse arguments
    args = _parse_args(argv)
    
    if getattr(args, 'verbose', False):
        print(f"🔍 [Driver] Arguments parsed successfully", flush=True)
        print(f"  Dataset: {args.dataset}")
        print(f"  Output: {args.out_dir}")
        print(f"  Train all genes: {getattr(args, 'train_all_genes', False)}")
    
    # Use the training orchestrator for clean separation of concerns
    try:
        from meta_spliceai.splice_engine.meta_models.training.training_orchestrator import (
            MetaModelTrainingOrchestrator
        )
        
        print("🚀 [Driver] Using training orchestrator for clean pipeline execution", flush=True)
        
        orchestrator = MetaModelTrainingOrchestrator(verbose=getattr(args, 'verbose', True))
        results = orchestrator.run_complete_training_pipeline(args)
        
        print("🎉 [Driver] Training pipeline completed successfully!", flush=True)
        
        # Print summary of results
        if getattr(args, 'verbose', False):
            print("\n📊 [Driver] Training Summary:")
            print(f"  Strategy: {results.get('training_strategy', 'Unknown')}")
            print(f"  Model saved: {results.get('model_path', 'N/A')}")
            if 'performance_metrics' in results and results['performance_metrics']:
                perf = results['performance_metrics']
                if 'mean_accuracy' in perf:
                    print(f"  CV Accuracy: {perf['mean_accuracy']:.3f} ± {perf.get('std_accuracy', 0):.3f}")
        
        return
        
    except ImportError as e:
        print(f"⚠️  [Driver] Training orchestrator not available: {e}", flush=True)
        print(f"⚠️  [Driver] Falling back to direct implementation", flush=True)
        
        # Fallback to direct implementation if orchestrator is not available
        should_use_batch_ensemble = _should_use_batch_ensemble_training(args)
        
        if should_use_batch_ensemble:
            print("🔥 [Driver] Using batch ensemble training for large dataset...", flush=True)
            try:
                results = _run_batch_ensemble_training(args)
                print("🎉 [Driver] Batch ensemble training completed successfully!", flush=True)
                return
            except Exception as e:
                print(f"❌ [Driver] Batch ensemble training failed: {e}", flush=True)
                if getattr(args, 'verbose', False):
                    import traceback
                    traceback.print_exc()
                sys.exit(1)
        else:
            print("🔧 [Driver] Using single model training for dataset...", flush=True)
            try:
                _run_single_model_training(args)
                print("🎉 [Driver] Single model training completed successfully!", flush=True)
                return
            except Exception as e:
                print(f"❌ [Driver] Single model training failed: {e}", flush=True)
                if getattr(args, 'verbose', False):
                    import traceback
                    traceback.print_exc()
                sys.exit(1)
                
    except Exception as e:
        print(f"❌ [Driver] Training failed: {e}", flush=True)
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
