#!/usr/bin/env python3
"""Enhanced ablation study driver for the 3-class meta-model.

This script supports both gene-aware and chromosome-aware cross-validation
approaches and integrates with the full diagnostic and evaluation pipeline
from the main CV scripts.

For a set of *feature subset modes* ("full", "no_spliceai", "no_probs", "no_kmer",
"only_kmer", "raw_scores", …) this script:

1. Loads the dataset using the same preprocessing pipeline as the CV scripts
2. Filters the feature matrix according to the selected mode
3. Runs either gene-aware or chromosome-aware CV with full diagnostics
4. Saves comprehensive results and generates visualization reports
5. Produces an aggregate comparison across all ablation modes

Example Usage
-------------

Gene-aware ablation:
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_ablation_multiclass \
  --dataset train_pc_1000/master \
  --out-dir results/ablation_gene_aware \
  --cv-strategy gene \
  --modes full,no_spliceai,no_kmer,only_kmer,raw_scores \
  --n-folds 5 \
  --verbose
```

Chromosome-aware ablation:
```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_ablation_multiclass \
  --dataset train_pc_1000/master \
  --out-dir results/ablation_chromosome_aware \
  --cv-strategy chromosome \
  --modes full,no_probs,only_kmer \
  --verbose
```
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import random
from pathlib import Path
from typing import Dict, Callable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier

from meta_spliceai.splice_engine.meta_models.builder import preprocessing
from meta_spliceai.splice_engine.meta_models.training import datasets, chromosome_split as csplit
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import (
    _encode_labels, SigmoidEnsemble
)
from meta_spliceai.splice_engine.meta_models.evaluation.feature_utils import (
    load_excluded_features, filter_features
)
from meta_spliceai.splice_engine.meta_models.evaluation.viz_utils import check_feature_correlations
from meta_spliceai.splice_engine.utils_kmer import is_kmer_feature as _is_kmer
from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
from meta_spliceai.splice_engine.meta_models.evaluation.feature_importance_integration import (
    run_gene_cv_feature_importance_analysis
)
from meta_spliceai.splice_engine.meta_models.evaluation.cv_metrics_viz import generate_cv_metrics_report

# Import unified training system for batch ensemble support
from meta_spliceai.splice_engine.meta_models.training.training_orchestrator import MetaModelTrainingOrchestrator
from meta_spliceai.splice_engine.meta_models.training.training_strategies import (
    select_optimal_training_strategy, TrainingResult
)
from meta_spliceai.splice_engine.meta_models.training.unified_dataset_utils import (
    load_and_prepare_training_dataset
)

# Label mapping for interpretability
LABEL_NAMES = {0: "neither", 1: "donor", 2: "acceptor"}

# --------------------------------------------------------------------------------------
#  Feature subset predicates - Enhanced from original
# --------------------------------------------------------------------------------------

def _pred_full(col: str) -> bool:
    """Keep all features."""
    return True

def _pred_raw_scores(col: str) -> bool:
    """Only keep the three raw SpliceAI probability columns."""
    return re.fullmatch(r"(donor|acceptor|neither)_score", col) is not None

def _pred_no_spliceai(col: str) -> bool:
    """Exclude ALL SpliceAI-derived features (raw scores + derived features)."""
    # Load the comprehensive list of SpliceAI-derived features
    spliceai_features = {
        # Raw scores
        "donor_score", "acceptor_score", "neither_score", "splice_probability",
        # Basic probability features
        "relative_donor_probability", "donor_acceptor_diff", "splice_neither_diff",
        "donor_acceptor_logodds", "splice_neither_logodds", "probability_entropy",
        # Context-agnostic features
        "context_neighbor_mean", "context_asymmetry", "context_max",
        # Donor-specific derived features
        "donor_diff_m1", "donor_diff_m2", "donor_diff_p1", "donor_diff_p2",
        "donor_surge_ratio", "donor_is_local_peak", "donor_weighted_context",
        "donor_peak_height_ratio", "donor_second_derivative", "donor_signal_strength",
        "donor_context_diff_ratio",
        # Acceptor-specific derived features
        "acceptor_diff_m1", "acceptor_diff_m2", "acceptor_diff_p1", "acceptor_diff_p2",
        "acceptor_surge_ratio", "acceptor_is_local_peak", "acceptor_weighted_context",
        "acceptor_peak_height_ratio", "acceptor_second_derivative", "acceptor_signal_strength",
        "acceptor_context_diff_ratio",
        # Cross-type features
        "donor_acceptor_peak_ratio", "type_signal_difference", "score_difference_ratio",
        "signal_strength_ratio"
    }
    
    return col not in spliceai_features

def _pred_no_probs(col: str) -> bool:
    """Exclude probability-derived features."""
    if col in _PROB_FEATURES:
        return False
    # Heuristic fallback: drop columns that look like *_score or contain "prob"/"logodds"/"ratio"
    if re.search(r"_score$", col) or re.search(r"prob|logodds|ratio", col):
        return False
    return True

def _pred_no_kmer(col: str) -> bool:
    """Exclude k-mer features."""
    return not _is_kmer(col)

def _pred_only_kmer(col: str) -> bool:
    """Keep only k-mer features."""
    return _is_kmer(col)

def _pred_positional_only(col: str) -> bool:
    """Keep only positional features (distance, position-based)."""
    positional_keywords = ['distance', 'position', 'relative', 'offset', 'coord']
    return any(keyword in col.lower() for keyword in positional_keywords)

def _pred_no_positional(col: str) -> bool:
    """Exclude positional features."""
    return not _pred_positional_only(col)

def _pred_context_only(col: str) -> bool:
    """Keep only context window features."""
    context_keywords = ['context', 'neighbor', 'window', '_m1', '_m2', '_p1', '_p2']
    return any(keyword in col.lower() for keyword in context_keywords)

# Extended probability-related columns
_PROB_FEATURES = {
    # Raw SpliceAI scores
    "donor_score", "acceptor_score", "neither_score",
    # Context window scores
    "context_score_m2", "context_score_m1", "context_score_p1", "context_score_p2",
    # Direct probabilities / entropy
    "relative_donor_probability", "splice_probability", "probability_entropy",
    # Pairwise differences / log-odds
    "donor_acceptor_diff", "splice_neither_diff", "donor_acceptor_logodds", "splice_neither_logodds",
    # Context neighbour aggregations
    "context_neighbor_mean", "context_asymmetry", "context_max",
    # Donor dynamics
    "donor_diff_m1", "donor_diff_m2", "donor_diff_p1", "donor_diff_p2",
    "donor_surge_ratio", "donor_is_local_peak", "donor_weighted_context", "donor_peak_height_ratio",
    "donor_second_derivative", "donor_signal_strength", "donor_context_diff_ratio",
    # Acceptor dynamics
    "acceptor_diff_m1", "acceptor_diff_m2", "acceptor_diff_p1", "acceptor_diff_p2",
    "acceptor_surge_ratio", "acceptor_is_local_peak", "acceptor_weighted_context", "acceptor_peak_height_ratio",
    "acceptor_second_derivative", "acceptor_signal_strength", "acceptor_context_diff_ratio",
    # Combined
    "donor_acceptor_peak_ratio", "type_signal_difference", "score_difference_ratio", "signal_strength_ratio",
}

# Map mode → predicate function
_MODE_MAP: Dict[str, Callable[[str], bool]] = {
    "full": _pred_full,
    "raw_scores": _pred_raw_scores,
    "no_probs": lambda c: _pred_no_probs(c) and _pred_no_spliceai(c),
    "no_kmer": _pred_no_kmer,
    "only_kmer": _pred_only_kmer,
    "positional_only": _pred_positional_only,
    "no_positional": _pred_no_positional,
    "context_only": _pred_context_only,
    # Deprecated alias
    "no_spliceai": _pred_no_spliceai,
}

# --------------------------------------------------------------------------------------
#  CLI
# --------------------------------------------------------------------------------------

def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enhanced ablation study for the 3-way meta-classifier with full diagnostic integration.")
    
    # Dataset and output
    p.add_argument("--dataset", required=True, help="Dataset directory or Parquet file")
    p.add_argument("--out-dir", required=True, help="Where to save results")
    
    # Ablation configuration
    p.add_argument("--modes", default="full,raw_scores,no_probs,no_kmer,only_kmer",
                   help="Comma-separated list of ablation modes to run")
    p.add_argument("--cv-strategy", default="gene", choices=["gene", "chromosome"],
                   help="Cross-validation strategy: gene-aware or chromosome-aware")
    
    # CV and data options
    p.add_argument("--gene-col", default="gene_id")
    p.add_argument("--n-folds", type=int, default=5, help="Number of CV folds (for gene-aware CV)")
    p.add_argument("--valid-size", type=float, default=0.15)
    p.add_argument("--min-rows-test", type=int, default=1_000, help="Minimum rows per test set (for chromosome-aware CV)")
    p.add_argument("--row-cap", type=int, default=200_000)
    p.add_argument("--sample-genes", type=int, default=None,
                   help="Sample only a subset of genes for faster testing")

    # Model parameters - Memory-optimized defaults
    p.add_argument("--tree-method", default="hist", choices=["hist", "gpu_hist", "approx"])
    p.add_argument("--max-bin", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--n-estimators", type=int, default=200)  # Reduced from 400 for memory efficiency

    # Feature handling
    p.add_argument("--exclude-features", default="configs/exclude_features.txt", 
                   help="Path to file with features to exclude or comma-separated list")
    p.add_argument("--check-leakage", action="store_true", default=True)
    p.add_argument("--leakage-threshold", type=float, default=0.95)
    p.add_argument("--auto-exclude-leaky", action="store_true", default=False)
    
    # Memory optimization options
    p.add_argument("--memory-optimize", action="store_true", default=False,
                   help="Enable aggressive memory optimization for low-memory systems")
    p.add_argument("--max-features", type=int, default=None,
                   help="Maximum number of features to keep (enables feature selection)")
    p.add_argument("--feature-selection-method", default="model", choices=["model", "mutual_info"],
                   help="Method for feature selection when max-features is specified")
    p.add_argument("--reduced-folds", action="store_true", default=False,
                   help="Use fewer CV folds (3 instead of 5) to reduce memory usage")
    p.add_argument("--max-ablation-sample", type=int, default=100_000,
                   help="Maximum sample size for ablation study (further reduced if memory-optimize enabled)")
    
    # Calibration options (required by training strategies)
    p.add_argument("--calibrate", action="store_true", default=False,
                   help="Enable binary splice/non-splice calibration")
    p.add_argument("--calibrate-per-class", action="store_true", default=True,
                   help="Enable per-class calibration (default for ablation)")
    p.add_argument("--calib-method", default="platt", choices=["platt", "isotonic"],
                   help="Calibration method")
    
    # Evaluation and diagnostics
    p.add_argument("--plot-format", default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--run-full-diagnostics", action="store_true", default=False,
                   help="Run full post-training diagnostics for each mode (time-intensive)")
    p.add_argument("--diag-sample", type=int, default=5000,  # Reduced from 10000
                   help="Sample size for diagnostic analyses")
    
    # Batch ensemble training support for large datasets
    p.add_argument("--train-all-genes", action="store_true",
                   help="Enable batch ensemble training for large datasets (10K+ genes)")
    p.add_argument("--max-genes-in-memory", type=int, default=None,
                   help="Maximum genes to process in memory (auto-detected if not specified)")
    p.add_argument("--memory-safety-factor", type=float, default=0.6,
                   help="Safety factor for memory calculations (0.6 = use 60% of available memory)")
    
    # Additional arguments required by training strategies
    p.add_argument("--base-thresh", type=float, default=0.5,
                   help="Threshold for base model evaluation")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Threshold for meta model evaluation")
    p.add_argument("--top-k", type=int, default=5,
                   help="Top-k parameter for evaluation")
    p.add_argument("--donor-score-col", default="donor_score",
                   help="Base model donor score column")
    p.add_argument("--acceptor-score-col", default="acceptor_score",
                   help="Base model acceptor score column")
    p.add_argument("--splice-prob-col", default="splice_probability",
                   help="Base model splice probability column")
    
    # General
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    
    return p.parse_args(argv)

# --------------------------------------------------------------------------------------
#  Core helpers
# --------------------------------------------------------------------------------------

def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive multiclass metrics focused on class-imbalance-aware measures."""
    metrics = {
        # Primary metrics for splice site prediction (class-imbalance aware)
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        
        # Per-class F1 scores (critical for splice site evaluation)
        "f1_neither": f1_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0] if 0 in y_true else 0.0,
        "f1_donor": f1_score(y_true, y_pred, labels=[1], average=None, zero_division=0)[0] if 1 in y_true else 0.0,
        "f1_acceptor": f1_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0] if 2 in y_true else 0.0,
    }
    
    # Binary splice vs non-splice metrics (class-imbalance aware)
    y_true_bin = (y_true != 0).astype(int)
    y_prob_bin = y_prob[:, 1] + y_prob[:, 2]  # donor + acceptor probabilities
    
    if len(np.unique(y_true_bin)) == 2:
        metrics.update({
            "binary_roc_auc": roc_auc_score(y_true_bin, y_prob_bin),
            "binary_average_precision": average_precision_score(y_true_bin, y_prob_bin),
            "binary_f1": f1_score(y_true_bin, (y_prob_bin >= 0.5).astype(int), zero_division=0),
        })
    else:
        metrics.update({
            "binary_roc_auc": np.nan,
            "binary_average_precision": np.nan,
            "binary_f1": np.nan,
        })
    
    # Per-class Average Precision (better than accuracy for imbalanced classes)
    try:
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import average_precision_score
        
        # Binarize labels for per-class AP calculation
        y_true_binarized = label_binarize(y_true, classes=[0, 1, 2])
        
        if y_true_binarized.shape[1] == 3:
            metrics.update({
                "ap_neither": average_precision_score(y_true_binarized[:, 0], y_prob[:, 0]),
                "ap_donor": average_precision_score(y_true_binarized[:, 1], y_prob[:, 1]),
                "ap_acceptor": average_precision_score(y_true_binarized[:, 2], y_prob[:, 2]),
                "ap_macro": (average_precision_score(y_true_binarized[:, 0], y_prob[:, 0]) +
                           average_precision_score(y_true_binarized[:, 1], y_prob[:, 1]) +
                           average_precision_score(y_true_binarized[:, 2], y_prob[:, 2])) / 3
            })
    except Exception:
        metrics.update({
            "ap_neither": np.nan,
            "ap_donor": np.nan, 
            "ap_acceptor": np.nan,
            "ap_macro": np.nan
        })
    
    # Top-k accuracy (proper implementation for splice sites)
    splice_mask = y_true != 0
    if splice_mask.any():
        k = int(splice_mask.sum())
        if k > 0:
            splice_prob = y_prob[:, 1] + y_prob[:, 2]
            top_idx = np.argsort(-splice_prob)[:k]
            top_k_correct = (y_true[top_idx] != 0).sum()
            metrics["top_k_accuracy"] = top_k_correct / k
        else:
            metrics["top_k_accuracy"] = np.nan
    else:
        metrics["top_k_accuracy"] = np.nan
    
    return metrics


def _train_multiclass_models(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, args: argparse.Namespace) -> List[XGBClassifier]:
    """Train the three binary models for the multiclass ensemble."""
    models = []
    
    # Memory-optimized training parameters
    training_params = {
        "n_estimators": args.n_estimators,
        "tree_method": args.tree_method,
        "max_bin": args.max_bin,
        "device": args.device if args.device != "auto" else None,
        "random_state": args.seed,
        "n_jobs": 1 if args.memory_optimize else -1,  # Reduce parallelism for memory optimization
        "max_depth": 4 if args.memory_optimize else 6,  # Reduce depth for memory
        "subsample": 0.7 if args.memory_optimize else 0.8,  # Reduce subsample
        "colsample_bytree": 0.7 if args.memory_optimize else 0.8,  # Reduce column sampling
    }
    
    if args.memory_optimize:
        print(f"    Using memory-optimized training parameters")
    
    for cls_name, cls_idx in [("donor", 1), ("acceptor", 2), ("neither", 0)]:
        print(f"    Training binary model for class: {cls_name}")
        
        # Create binary labels (current class vs. rest)
        y_train_binary = (y_train == cls_idx).astype(int)
        y_val_binary = (y_val == cls_idx).astype(int)
        
        model = XGBClassifier(**training_params)
        model.fit(X_train, y_train_binary, eval_set=[(X_val, y_val_binary)], verbose=False)
        models.append(model)
        
        # Memory cleanup after each model
        if args.memory_optimize:
            del y_train_binary, y_val_binary
            gc.collect()
    
    return models


def _run_gene_aware_cv(X: np.ndarray, y: np.ndarray, genes: np.ndarray, feature_names: List[str], 
                      mode: str, args: argparse.Namespace, out_dir: Path) -> Dict[str, float]:
    """Run gene-aware cross-validation for ablation analysis."""
    from sklearn.model_selection import GroupShuffleSplit
    
    gkf = GroupKFold(n_splits=args.n_folds)
    fold_rows: List[Dict] = []
    all_y_true, all_y_pred, all_y_prob = [], [], []
    
    print(f"    Running {args.n_folds}-fold gene-aware CV...")
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(gkf.split(X, y, groups=genes)):
        if args.verbose:
            print(f"      Fold {fold_idx+1}/{args.n_folds} (test_rows={len(test_idx)})")
        
        # Train/valid split preserving gene groups
        rel_valid = args.valid_size / (1.0 - 1.0 / args.n_folds)
        gss = GroupShuffleSplit(n_splits=1, test_size=rel_valid, random_state=args.seed)
        train_idx, valid_idx = next(gss.split(train_val_idx, y[train_val_idx], groups=genes[train_val_idx]))
        train_idx = train_val_idx[train_idx]
        valid_idx = train_val_idx[valid_idx]
        
        # Train models
        models = _train_multiclass_models(X[train_idx], y[train_idx], 
                                         X[valid_idx], y[valid_idx], args)
        
        # Predict on test set
        proba_parts = [m.predict_proba(X[test_idx])[:, 1] for m in models]
        proba = np.column_stack(proba_parts)  # shape (n,3)
        pred = proba.argmax(axis=1)
        y_true = y[test_idx]
        
        # Calculate metrics
        metrics = _multiclass_metrics(y_true, pred, proba)
        
        # Store for aggregation
        all_y_true.append(y_true)
        all_y_pred.append(pred)
        all_y_prob.append(proba)
        
        # Store fold results
        fold_row = {
            "fold": fold_idx,
            "test_rows": len(test_idx),
            **metrics
        }
        fold_rows.append(fold_row)
        
        # Save fold-specific results
        with open(out_dir / f"fold_{fold_idx}_metrics.json", "w") as f:
            json.dump({k: float(v) if isinstance(v, np.floating) else v for k, v in fold_row.items()}, f, indent=2)
    
    # Aggregate results
    df_metrics = pd.DataFrame(fold_rows)
    df_metrics.to_csv(out_dir / "cv_metrics_folds.csv", index=False)
    
    # Calculate mean metrics (focusing on class-imbalance-aware measures)
    mean_metrics = {}
    key_metrics = [
        "macro_f1", "weighted_f1", "binary_average_precision", "top_k_accuracy",
        "f1_donor", "f1_acceptor", "ap_macro", "ap_donor", "ap_acceptor"
    ]
    
    for metric in key_metrics:
        if metric in df_metrics.columns:
            values = df_metrics[metric].dropna()
            if len(values) > 0:
                mean_metrics[metric] = values.mean()
                mean_metrics[f"{metric}_std"] = values.std()
            else:
                mean_metrics[metric] = np.nan
                mean_metrics[f"{metric}_std"] = np.nan
    
    # Save aggregate metrics
    with open(out_dir / "cv_metrics_aggregate.json", "w") as f:
        json.dump(mean_metrics, f, indent=2)
    
    return mean_metrics


def _run_chromosome_aware_cv(X: np.ndarray, y: np.ndarray, chrom: np.ndarray, genes: np.ndarray, 
                           feature_names: List[str], mode: str, args: argparse.Namespace, out_dir: Path) -> Dict[str, float]:
    """Run chromosome-aware (LOCO) cross-validation for ablation analysis."""
    fold_rows: List[Dict] = []
    all_y_true, all_y_pred, all_y_prob = [], [], []
    
    print(f"    Running chromosome-aware (LOCO) CV...")
    
    for fold_idx, (held_out, tr_idx, val_idx, te_idx) in enumerate(csplit.loco_cv_splits(
        X, y,
        chrom_array=chrom,
        gene_array=genes,
        valid_size=args.valid_size,
        min_rows=args.min_rows_test,
        seed=args.seed,
    )):
        if args.verbose:
            print(f"      Fold {fold_idx+1} - Held out chromosome: {held_out} (test_rows={len(te_idx)})")
        
        # Train models
        models = _train_multiclass_models(X[tr_idx], y[tr_idx], 
                                         X[val_idx], y[val_idx], args)
        
        # Predict on test set
        proba_parts = [m.predict_proba(X[te_idx])[:, 1] for m in models]
        proba = np.column_stack(proba_parts)  # shape (n,3)
        pred = proba.argmax(axis=1)
        y_true = y[te_idx]
        
        # Calculate metrics
        metrics = _multiclass_metrics(y_true, pred, proba)
        
        # Store for aggregation
        all_y_true.append(y_true)
        all_y_pred.append(pred)
        all_y_prob.append(proba)
        
        # Store fold results
        fold_row = {
            "fold": fold_idx,
            "held_out_chromosome": held_out,
            "test_rows": len(te_idx),
            **metrics
        }
        fold_rows.append(fold_row)
        
        # Save fold-specific results
        with open(out_dir / f"fold_{fold_idx}_chr_{held_out}_metrics.json", "w") as f:
            json.dump({k: float(v) if isinstance(v, np.floating) else v for k, v in fold_row.items()}, f, indent=2)
    
    # Aggregate results
    df_metrics = pd.DataFrame(fold_rows)
    df_metrics.to_csv(out_dir / "loco_metrics_folds.csv", index=False)
    
    # Calculate mean metrics (focusing on class-imbalance-aware measures)
    mean_metrics = {}
    key_metrics = [
        "macro_f1", "weighted_f1", "binary_average_precision", "top_k_accuracy",
        "f1_donor", "f1_acceptor", "ap_macro", "ap_donor", "ap_acceptor"
    ]
    
    for metric in key_metrics:
        if metric in df_metrics.columns:
            values = df_metrics[metric].dropna()
            if len(values) > 0:
                mean_metrics[metric] = values.mean()
                mean_metrics[f"{metric}_std"] = values.std()
            else:
                mean_metrics[metric] = np.nan
                mean_metrics[f"{metric}_std"] = np.nan
    
    # Save aggregate metrics
    with open(out_dir / "loco_metrics_aggregate.json", "w") as f:
        json.dump(mean_metrics, f, indent=2)
    
    return mean_metrics


def _run_single_mode(X: np.ndarray, y: np.ndarray, chrom: np.ndarray | None, genes: np.ndarray,
                     feature_names: List[str], mode: str, pred_fn: Callable[[str], bool], args: argparse.Namespace, out_dir: Path) -> Dict[str, float]:
    """Run ablation analysis for a single feature subset mode."""
    
    sub_dir = out_dir / mode
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  [Mode: {mode}] Starting ablation analysis...")
    
    # Filter features based on mode
    keep_indices = [i for i, c in enumerate(feature_names) if pred_fn(c)]
    keep_cols = [feature_names[i] for i in keep_indices]
    
    if not keep_cols:
        raise ValueError(f"Mode '{mode}' removes all features – aborting.")
    
    # Subset feature matrix using boolean indexing
    X_sub = X[:, keep_indices]
    
    print(f"  [Mode: {mode}] Feature count: {X.shape[1]} → {X_sub.shape[1]}")
    
    # Save feature manifest for this mode
    pd.DataFrame({"feature": keep_cols}).to_csv(sub_dir / "feature_manifest.csv", index=False)
    
    # Log some example features
    if args.verbose:
        print(f"  [Mode: {mode}] Example features: {keep_cols[:5]}")
        if _is_kmer(keep_cols[0]) if keep_cols else False:
            print(f"  [Mode: {mode}] K-mer features detected")
    
    # Run cross-validation based on strategy
    if args.cv_strategy == "gene":
        mean_metrics = _run_gene_aware_cv(X_sub, y, genes, keep_cols, mode, args, sub_dir)
    elif args.cv_strategy == "chromosome":
        if chrom is None:
            raise ValueError("Chromosome data not available for chromosome-aware CV")
        mean_metrics = _run_chromosome_aware_cv(X_sub, y, chrom, genes, keep_cols, mode, args, sub_dir)
    else:
        raise ValueError(f"Unknown CV strategy: {args.cv_strategy}")
    
    # Save final model for this mode (trained on full data)
    final_models = _train_multiclass_models(X_sub, y, X_sub, y, args)
    ensemble = SigmoidEnsemble(final_models, keep_cols)
    
    import pickle
    with open(sub_dir / f"model_{mode}.pkl", "wb") as f:
        pickle.dump(ensemble, f)
    
    # Optional: Run full diagnostics for this mode
    if args.run_full_diagnostics:
        print(f"  [Mode: {mode}] Running full diagnostics...")
        try:
            # Run basic diagnostic analyses
            _cutils.richer_metrics(args.dataset, sub_dir, sample=args.diag_sample)
            _cutils.probability_diagnostics(args.dataset, sub_dir, sample=args.diag_sample)
            
            # Run feature importance analysis
            run_gene_cv_feature_importance_analysis(args.dataset, sub_dir, sample=args.diag_sample)
            
            print(f"  [Mode: {mode}] ✓ Full diagnostics completed")
        except Exception as e:
            print(f"  [Mode: {mode}] ⚠️ Diagnostics failed: {e}")
    
    # Generate visualization report if we have CV metrics
    try:
        cv_metrics_file = sub_dir / ("cv_metrics_folds.csv" if args.cv_strategy == "gene" else "loco_metrics_folds.csv")
        if cv_metrics_file.exists():
            generate_cv_metrics_report(
                csv_path=cv_metrics_file,
                out_dir=sub_dir,
                plot_format=args.plot_format,
                dpi=200
            )
            print(f"  [Mode: {mode}] ✓ Visualization report generated")
    except Exception as e:
        if args.verbose:
            print(f"  [Mode: {mode}] ⚠️ Visualization generation failed: {e}")
    
    # Memory cleanup for this mode
    del X_sub, final_models, ensemble
    gc.collect()
    
    print(f"  [Mode: {mode}] Completed - Accuracy: {mean_metrics.get('accuracy', 0):.3f}, "
          f"Macro F1: {mean_metrics.get('macro_f1', 0):.3f}")

    return {"mode": mode, "n_features": len(keep_cols), **mean_metrics}


def _run_unified_ablation_mode(
    dataset_path: str,
    out_dir: Path,
    mode: str,
    pred_fn: Callable[[str], bool],
    args: argparse.Namespace
) -> Dict[str, float]:
    """Run ablation mode using unified training system with batch ensemble support."""
    
    sub_dir = out_dir / mode
    sub_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  [Mode: {mode}] Starting unified ablation analysis...")
    
    # Create modified args for this mode
    mode_args = argparse.Namespace(**vars(args))
    mode_args.out_dir = str(sub_dir)
    mode_args.verbose = False  # Reduce verbosity for cleaner output
    
    try:
        # 1. Load and prepare dataset using unified system
        raw_df, X_df, y_series, genes = load_and_prepare_training_dataset(
            dataset_path, mode_args, verbose=False
        )
        
        # 2. Apply feature filtering for this ablation mode
        feature_names = list(X_df.columns)
        keep_features = [f for f in feature_names if pred_fn(f)]
        
        if not keep_features:
            raise ValueError(f"Mode '{mode}' removes all features")
        
        X_df_filtered = X_df[keep_features]
        
        print(f"  [Mode: {mode}] Features: {X_df.shape[1]} → {X_df_filtered.shape[1]}")
        
        # 3. Select appropriate training strategy (single model or batch ensemble)
        strategy = select_optimal_training_strategy(dataset_path, mode_args, verbose=False)
        
        if args.verbose:
            print(f"  [Mode: {mode}] Using strategy: {strategy.get_strategy_name()}")
        
        # 4. Run training using the selected strategy
        training_result = strategy.train_model(
            dataset_path, sub_dir, mode_args, X_df_filtered, y_series, genes
        )
        
        # 5. Extract performance metrics
        performance = training_result.performance_metrics or {}
        cv_results = training_result.cv_results or []
        
        # 6. Calculate comprehensive metrics
        result_metrics = {
            'mode': mode,
            'n_features': len(keep_features),
            'feature_names': keep_features,
            'training_strategy': strategy.get_strategy_name(),
            'model_path': str(training_result.model_path)
        }
        
        # Add performance metrics (class-imbalance-aware)
        if performance:
            result_metrics.update({
                'macro_f1': performance.get('mean_f1_macro', 0),
                'macro_f1_std': performance.get('std_f1_macro', 0),
                'cv_folds': len(cv_results)
            })
        
        # Add CV-specific metrics if available
        if cv_results:
            cv_df = pd.DataFrame(cv_results)
            if 'base_f1' in cv_df.columns and 'meta_f1' in cv_df.columns:
                result_metrics.update({
                    'base_f1_mean': cv_df['base_f1'].mean(),
                    'meta_f1_mean': cv_df['meta_f1'].mean(),
                    'f1_improvement': cv_df['meta_f1'].mean() - cv_df['base_f1'].mean()
                })
            
            if 'top_k_accuracy' in cv_df.columns:
                result_metrics['top_k_accuracy'] = cv_df['top_k_accuracy'].mean()
        
        # 7. Save detailed results for this mode
        mode_summary = {
            'ablation_mode': mode,
            'performance_metrics': result_metrics,
            'training_metadata': training_result.training_metadata,
            'excluded_features': training_result.excluded_features
        }
        
        with open(sub_dir / f"ablation_mode_{mode}_summary.json", "w") as f:
            json.dump(mode_summary, f, indent=2, default=str)
        
        print(f"  [Mode: {mode}] ✅ Completed - Acc={result_metrics.get('accuracy', 0):.3f}, "
              f"F1={result_metrics.get('macro_f1', 0):.3f}")
        
        return result_metrics
        
    except Exception as e:
        print(f"  [Mode: {mode}] ❌ FAILED: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        return {
            'mode': mode,
            'n_features': 0,
            'macro_f1': 0.0,
            'binary_average_precision': 0.0,
            'top_k_accuracy': 0.0,
            'error': str(e)
        }


# --------------------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------------------

import gc

def _apply_splice_aware_subsampling(df: pd.DataFrame, target_size: int, random_seed: int = 42) -> pd.DataFrame:
    """
    Apply intelligent subsampling that preserves splice sites while reducing non-splice sites.
    
    This function:
    1. Identifies splice sites (donor/acceptor) vs non-splice sites
    2. Keeps ALL splice sites (they're rare and crucial for training)
    3. Subsamples non-splice sites to meet the target size
    4. Maintains gene-level structure where possible
    
    Args:
        df: Input DataFrame with splice_type column
        target_size: Target number of rows after subsampling
        random_seed: Random seed for reproducibility
    
    Returns:
        Subsampled DataFrame with preserved splice sites
    """
    import numpy as np
    import pandas as pd
    
    np.random.seed(random_seed)
    
    # Identify splice sites vs non-splice sites
    if 'splice_type' not in df.columns:
        print(f"[WARNING] splice_type column not found. Using random sampling instead.")
        return df.sample(n=min(target_size, len(df)), random_state=random_seed)
    
    # Separate splice sites from non-splice sites
    splice_sites = df[df['splice_type'] != 'neither'].copy()
    non_splice_sites = df[df['splice_type'] == 'neither'].copy()
    
    print(f"[Ablation Analysis] Original data breakdown:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Splice sites (donor/acceptor): {len(splice_sites):,}")
    print(f"  Non-splice sites: {len(non_splice_sites):,}")
    print(f"  Target size: {target_size:,}")
    
    # Always keep ALL splice sites (they're rare and crucial)
    splice_sites_count = len(splice_sites)
    
    if splice_sites_count >= target_size:
        print(f"[Ablation Analysis] WARNING: {splice_sites_count} splice sites >= target size {target_size}")
        print(f"[Ablation Analysis] Keeping all splice sites and sampling minimal non-splice sites")
        
        # Keep all splice sites and add a few non-splice sites if possible
        remaining_budget = max(0, target_size - splice_sites_count)
        if remaining_budget > 0 and len(non_splice_sites) > 0:
            sampled_non_splice = non_splice_sites.sample(n=min(remaining_budget, len(non_splice_sites)), 
                                                        random_state=random_seed)
            result_df = pd.concat([splice_sites, sampled_non_splice], ignore_index=True)
        else:
            result_df = splice_sites.copy()
    else:
        # Normal case: keep all splice sites and sample non-splice sites
        remaining_budget = target_size - splice_sites_count
        
        if remaining_budget >= len(non_splice_sites):
            # We can keep everything
            print(f"[Ablation Analysis] Keeping all data ({len(df):,} rows)")
            result_df = df.copy()
        else:
            # Sample non-splice sites to fit budget
            sampled_non_splice = non_splice_sites.sample(n=remaining_budget, random_state=random_seed)
            result_df = pd.concat([splice_sites, sampled_non_splice], ignore_index=True)
            
            print(f"[Ablation Analysis] Intelligent subsampling completed:")
            print(f"  Kept all {splice_sites_count:,} splice sites")
            print(f"  Sampled {remaining_budget:,} non-splice sites from {len(non_splice_sites):,}")
            print(f"  Final size: {len(result_df):,} rows")
    
    # Shuffle the final result to avoid any ordering bias
    result_df = result_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    return result_df


# --------------------------------------------------------------------------------------
#  Ablation modes - Feature filtering functions
# --------------------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    args = _parse_args(argv)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Apply memory optimization settings
    if args.memory_optimize:
        print(f"[Ablation Analysis] Memory optimization enabled - applying aggressive memory settings")
        # Reduce sample size further for memory optimization
        if args.max_ablation_sample > 50_000:
            args.max_ablation_sample = 50_000
            print(f"[Ablation Analysis] Reduced max sample size to {args.max_ablation_sample} for memory optimization")
        
        # Enable feature selection if not already specified
        if args.max_features is None:
            args.max_features = 1000
            print(f"[Ablation Analysis] Auto-enabled feature selection with max_features={args.max_features}")
        
        # Reduce model complexity
        if args.n_estimators > 100:
            args.n_estimators = 100
            print(f"[Ablation Analysis] Reduced n_estimators to {args.n_estimators} for memory optimization")
        
        # Reduce diagnostic sample size
        if args.diag_sample > 2000:
            args.diag_sample = 2000
            print(f"[Ablation Analysis] Reduced diag_sample to {args.diag_sample} for memory optimization")
        
        # Use fewer folds if not already reduced
        if not args.reduced_folds:
            args.reduced_folds = True
            print(f"[Ablation Analysis] Auto-enabled reduced folds for memory optimization")

    # Apply reduced folds setting
    if args.reduced_folds:
        args.n_folds = 3
        print(f"[Ablation Analysis] Using {args.n_folds} folds instead of 5 for memory efficiency")

    # Calculate effective row cap for hierarchical sampling
    effective_row_cap = min(args.row_cap, args.max_ablation_sample) if args.row_cap > 0 else args.max_ablation_sample
    print(f"[Ablation Analysis] Effective row cap for sampling: {effective_row_cap}")

    print(f"[Ablation Analysis] Enhanced multiclass ablation study")
    print(f"[Ablation Analysis] CV Strategy: {args.cv_strategy}")
    print(f"[Ablation Analysis] Output directory: {out_dir}")
    print(f"[Ablation Analysis] Modes: {args.modes}")
    print(f"[Ablation Analysis] Memory optimization: {args.memory_optimize}")
    print(f"[Ablation Analysis] Hierarchical sampling enabled: preserves splice sites")
    
    # Force garbage collection before starting
    gc.collect()
    
    # 1. Load dataset with hierarchical gene-aware sampling
    print(f"[Ablation Analysis] Loading dataset with hierarchical sampling...")
    
    if args.sample_genes is not None:
        # Use explicit gene sampling if specified
        print(f"[Ablation Analysis] Gene-level sampling: {args.sample_genes} genes")
        from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
        df = load_dataset_sample(args.dataset, sample_genes=args.sample_genes, random_seed=args.seed)
    else:
        # Use hierarchical sampling based on effective row cap
        print(f"[Ablation Analysis] Hierarchical sampling with target size: {effective_row_cap}")
        from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
        
        # For ablation studies, we want to preserve splice site representation
        # Use gene-level sampling first, then row-level if needed
        df = load_dataset_sample(
            args.dataset, 
            sample_size=effective_row_cap,
            random_seed=args.seed
        )
        
        # If we still have too many rows, apply intelligent subsampling
        # that preserves splice sites while reducing non-splice sites
        if len(df) > effective_row_cap:
            print(f"[Ablation Analysis] Applying splice-site-aware subsampling...")
            df = _apply_splice_aware_subsampling(df, target_size=effective_row_cap, random_seed=args.seed)
    
    if args.gene_col not in df.columns:
        raise KeyError(f"Column '{args.gene_col}' not found in dataset")
    
    # 2. Prepare training data (same preprocessing as main CV scripts)
    X_df, y_series = preprocessing.prepare_training_data(
        df, 
        label_col="splice_type", 
        return_type="pandas", 
        verbose=1 if args.verbose else 0,
        encode_chrom=True  # Include chromosome encoding
    )
    
    # Memory optimization: clear original DataFrame early
    if args.memory_optimize:
        print(f"[Ablation Analysis] Clearing original DataFrame to free memory")
        del df
        gc.collect()
    
    # Save original feature manifest
    original_features = list(X_df.columns)
    with open(out_dir / "original_features.json", "w") as f:
        json.dump({"feature_names": original_features}, f)
    
    # 3. Handle feature exclusion (enhanced logic from main CV scripts)
    exclude_list = []
    
    if args.exclude_features:
        # First check if it's a valid file path
        exclude_path = Path(args.exclude_features)
        if exclude_path.exists() and exclude_path.is_file():
            # It's a file path, use our enhanced loader
            print(f"[Ablation Analysis] Loading exclude features from file: {exclude_path}")
            file_exclusions = load_excluded_features(exclude_path)
            if file_exclusions:
                exclude_list.extend(file_exclusions)
        else:
            # Not a file path, treat as comma-separated list
            print(f"[Ablation Analysis] Treating exclude features as comma-separated list")
            exclude_list.extend([f.strip() for f in args.exclude_features.split(',') if f.strip()])
    
    # Also check for exclude_features.txt in output dir for backward compatibility
    exclude_file = out_dir / "exclude_features.txt"
    if exclude_file.exists():
        try:
            print(f"[Ablation Analysis] Also loading exclude features from output directory: {exclude_file}")
            file_exclusions = load_excluded_features(exclude_file)
            if file_exclusions:
                exclude_list.extend(file_exclusions)
        except Exception as e:
            print(f"Warning: Error reading exclude_features.txt: {e}")
    
    # Check for leakage if enabled
    if args.check_leakage:
        print(f"\n[Ablation Analysis] Running comprehensive leakage analysis (threshold={args.leakage_threshold})...")
        
        try:
            # Import our comprehensive leakage analysis module
            from meta_spliceai.splice_engine.meta_models.evaluation.leakage_analysis import LeakageAnalyzer
            
            # Create leakage analysis directory
            leakage_analysis_dir = out_dir / "leakage_analysis" 
            leakage_analysis_dir.mkdir(exist_ok=True, parents=True)
            
            # Initialize analyzer
            analyzer = LeakageAnalyzer(
                output_dir=leakage_analysis_dir,
                subject="ablation_leakage"
            )
            
            # Run comprehensive analysis
            leakage_results = analyzer.run_comprehensive_analysis(
                X=X_df,
                y=y_series,
                threshold=args.leakage_threshold,
                methods=['pearson', 'spearman'],
                top_n=50,
                verbose=1 if args.verbose else 0
            )
            
            # Extract leaky features for potential auto-exclusion
            leaky_features = set()
            for method_results in leakage_results['correlation_results'].values():
                leaky_features.update(method_results['leaky_features']['feature'].tolist())
            leaky_features = list(leaky_features)
            
            print(f"[Ablation Analysis] Found {len(leaky_features)} potentially leaky features")
            print(f"[Ablation Analysis] Comprehensive results saved to: {leakage_analysis_dir}")
            
            # If auto-exclude is enabled, add leaky features to exclusion list
            if args.auto_exclude_leaky and leaky_features:
                print(f"[Ablation Analysis] Auto-excluding {len(leaky_features)} potentially leaky features")
                exclude_list.extend(leaky_features)
            
        except Exception as e:
            print(f"[Ablation Analysis] Error in comprehensive analysis: {e}")
            print("[Ablation Analysis] Falling back to basic correlation analysis...")
            
            # Fallback to basic analysis
        X_np = X_df.values
        y_np = _encode_labels(y_series)
        curr_features = X_df.columns.tolist()
        
        correlation_report_path = out_dir / "feature_label_correlations.csv"
        leaky_features, corr_df = check_feature_correlations(
            X_np, y_np, curr_features, args.leakage_threshold, correlation_report_path
        )
        
        if args.auto_exclude_leaky and leaky_features:
            print(f"[Ablation Analysis] Auto-excluding {len(leaky_features)} potentially leaky features")
            exclude_list.extend(leaky_features)
    
    # 4. Apply feature selection if requested (before exclusions)
    if args.max_features is not None and args.max_features < X_df.shape[1]:
        print(f"\n[Ablation Analysis] Running feature selection: {X_df.shape[1]} → {args.max_features} features")
        
        try:
            from sklearn.feature_selection import SelectFromModel, mutual_info_classif
            from sklearn.ensemble import RandomForestClassifier
            
            # Convert to numpy for feature selection
            X_temp = X_df.values
            y_temp = _encode_labels(y_series)
            
            if args.feature_selection_method == "model":
                # Use RandomForestClassifier for model-based selection (faster than XGBoost)
                rf = RandomForestClassifier(n_estimators=50, random_state=args.seed, n_jobs=-1)
                selector = SelectFromModel(rf, max_features=args.max_features)
                selector.fit(X_temp, y_temp)
                selected_features = X_df.columns[selector.get_support()].tolist()
            else:  # mutual_info
                # Use mutual information for selection
                mi_scores = mutual_info_classif(X_temp, y_temp, random_state=args.seed)
                selected_indices = np.argsort(mi_scores)[-args.max_features:]
                selected_features = X_df.columns[selected_indices].tolist()
            
            # Apply feature selection
            X_df = X_df[selected_features]
            print(f"[Ablation Analysis] Feature selection completed: {len(selected_features)} features selected")
            
            # Clean up temporary variables
            del X_temp, y_temp
            if 'selector' in locals():
                del selector
            gc.collect()
            
        except Exception as e:
            print(f"[Ablation Analysis] Warning: Feature selection failed: {e}")
            print(f"[Ablation Analysis] Continuing with all features")
    
    # Apply exclusions
    if exclude_list:
        exclude_list = list(dict.fromkeys(exclude_list))  # Remove duplicates
        original_feature_count = X_df.shape[1]
        excluded_features = []
        
        for feature in exclude_list:
            if feature in X_df.columns:
                X_df = X_df.drop(columns=[feature])
                excluded_features.append(feature)
        
        if excluded_features:
            print(f"[Ablation Analysis] Excluded {len(excluded_features)} features")
            print(f"[Ablation Analysis] Feature count: {original_feature_count} → {X_df.shape[1]}")
            
            # Save excluded features
            with open(out_dir / "excluded_features.txt", 'w') as f:
                f.write("# Features excluded during ablation analysis\n")
                for feature in excluded_features:
                    f.write(f"{feature}\n")
    
    # Convert to numpy arrays
    y = _encode_labels(y_series)
    chrom = df["chrom"].to_numpy() if not args.memory_optimize else None
    genes = df[args.gene_col].to_numpy()
    
    # Save feature names before potential memory cleanup
    final_feature_names = list(X_df.columns)
    
    # Memory optimization: Get chrom data before clearing df if needed
    if args.memory_optimize and chrom is None and args.cv_strategy == "chromosome":
        print(f"[Ablation Analysis] Warning: chromosome CV requested but df already cleared. Loading chrom data separately.")
        # We need chromosome data for chromosome CV, reload just that column
        temp_df = datasets.load_dataset(args.dataset)
        chrom = temp_df["chrom"].to_numpy()[:len(genes)]
        del temp_df
        gc.collect()
    
    # Convert to numpy for processing (before clearing DataFrame)
    X = X_df.values
    
    # Memory optimization: Clear DataFrame and get final memory state
    if args.memory_optimize:
        del X_df, y_series
        if 'df' in locals():
            del df
        gc.collect()
        print(f"[Ablation Analysis] Memory cleanup completed")
    
    print(f"[Ablation Analysis] Dataset shape: {X.shape}")
    print(f"[Ablation Analysis] Class distribution: {np.bincount(y)}")
    for cls_idx, count in enumerate(np.bincount(y)):
        print(f"  {LABEL_NAMES[cls_idx]}: {count} ({count/len(y)*100:.1f}%)")
    
    # 4. Determine ablation approach - ALWAYS use unified for reliability
    total_genes = len(np.unique(genes)) if 'genes' in locals() else 0
    use_unified_approach = True  # Always use unified approach for better reliability
    
    if use_unified_approach:
        print(f"[Ablation Analysis] Using UNIFIED TRAINING SYSTEM for large dataset ({total_genes:,} genes)")
        print(f"[Ablation Analysis] This supports batch ensemble training automatically")
    else:
        print(f"[Ablation Analysis] Using LEGACY APPROACH for medium dataset ({total_genes:,} genes)")
    
    # 5. Run ablation analysis for each mode
    results: List[Dict[str, float]] = []
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    
    for mode_idx, mode in enumerate(modes):
        print(f"\n[Ablation Analysis] === Mode {mode_idx+1}/{len(modes)}: {mode} ===")
        
        # Force garbage collection before each mode
        gc.collect()
        
        pred_fn = _MODE_MAP.get(mode)
        if pred_fn is None:
            available_modes = ', '.join(_MODE_MAP.keys())
            raise KeyError(f"Unknown mode '{mode}'. Available modes: {available_modes}")
        
        try:
            if use_unified_approach:
                # Use unified training system (supports batch ensemble)
                res = _run_unified_ablation_mode(
                    args.dataset, out_dir, mode, pred_fn, args
                )
            else:
                # Use legacy approach (for smaller datasets)
                res = _run_single_mode(X, y, chrom, genes, final_feature_names, 
                                     mode, pred_fn, args, out_dir)
            
            results.append(res)
            
            # Aggressive memory cleanup after each mode
            if args.memory_optimize:
                gc.collect()
                print(f"  [Mode: {mode}] Memory cleanup completed")
                
        except Exception as e:
            print(f"  [Mode: {mode}] ❌ FAILED: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            
            # Add failed result
            results.append({
                "mode": mode,
                "n_features": 0,
                "macro_f1": 0.0,
                "binary_average_precision": 0.0,
                "top_k_accuracy": 0.0,
                "error": str(e)
            })
            
            # Clean up even after failure
            if args.memory_optimize:
                gc.collect()

    # Final memory cleanup before summary generation
    gc.collect()

    # 5. Generate summary table and visualizations
    print(f"\n[Ablation Analysis] Generating summary report...")
    
    df_summary = pd.DataFrame(results)
    df_summary.to_csv(out_dir / "ablation_summary.csv", index=False)
    
    # Create summary report
    summary_report = {
        "ablation_config": {
            "cv_strategy": args.cv_strategy,
            "n_folds": args.n_folds if args.cv_strategy == "gene" else "variable (chromosome-dependent)",
            "modes_tested": len(modes),
            "dataset": args.dataset,
            "total_features_original": len(original_features),
            "total_features_after_exclusion": X_df.shape[1],
            "excluded_features": len(exclude_list) if exclude_list else 0
        },
        "results_summary": {
            "best_mode_f1": df_summary.loc[df_summary['macro_f1'].idxmax()]['mode'] if not df_summary['macro_f1'].isna().all() else None,
            "best_mode_ap": df_summary.loc[df_summary['binary_average_precision'].idxmax()]['mode'] if 'binary_average_precision' in df_summary.columns and not df_summary['binary_average_precision'].isna().all() else None,
            "best_mode_topk": df_summary.loc[df_summary['top_k_accuracy'].idxmax()]['mode'] if 'top_k_accuracy' in df_summary.columns and not df_summary['top_k_accuracy'].isna().all() else None,
            "mean_f1": df_summary['macro_f1'].mean(),
            "std_f1": df_summary['macro_f1'].std(),
            "mean_ap": df_summary['binary_average_precision'].mean() if 'binary_average_precision' in df_summary.columns else np.nan,
            "std_ap": df_summary['binary_average_precision'].std() if 'binary_average_precision' in df_summary.columns else np.nan,
        }
    }
    
    with open(out_dir / "ablation_report.json", "w") as f:
        json.dump(summary_report, f, indent=2)
    
    # Generate comparison plots
    if args.plot_format and len(results) > 1:
        try:
            plt.figure(figsize=(15, 5))
            
            # Accuracy comparison
            plt.subplot(1, 3, 1)
            valid_results = [r for r in results if not np.isnan(r.get('accuracy', np.nan))]
            if valid_results:
                modes_plot = [r['mode'] for r in valid_results]
                accuracies = [r['accuracy'] for r in valid_results]
                colors = plt.cm.viridis(np.linspace(0, 1, len(valid_results)))
                
                bars = plt.bar(range(len(modes_plot)), accuracies, color=colors, alpha=0.7)
                plt.xlabel('Ablation Mode')
                plt.ylabel('Accuracy')
                plt.title('Accuracy by Ablation Mode')
                plt.xticks(range(len(modes_plot)), modes_plot, rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, acc in zip(bars, accuracies):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
            
            # F1 comparison
            plt.subplot(1, 3, 2)
            if valid_results:
                f1_scores = [r['macro_f1'] for r in valid_results]
                bars = plt.bar(range(len(modes_plot)), f1_scores, color=colors, alpha=0.7)
                plt.xlabel('Ablation Mode')
                plt.ylabel('Macro F1')
                plt.title('Macro F1 by Ablation Mode')
                plt.xticks(range(len(modes_plot)), modes_plot, rotation=45, ha='right')
                plt.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, f1 in zip(bars, f1_scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Feature count comparison
            plt.subplot(1, 3, 3)
            n_features = [r['n_features'] for r in results]
            modes_all = [r['mode'] for r in results]
            bars = plt.bar(range(len(modes_all)), n_features, alpha=0.7, color='orange')
            plt.xlabel('Ablation Mode')
            plt.ylabel('Number of Features')
            plt.title('Feature Count by Mode')
            plt.xticks(range(len(modes_all)), modes_all, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, n_feat in zip(bars, n_features):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(n_features)*0.01,
                        f'{n_feat}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(out_dir / f"ablation_comparison.{args.plot_format}", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[Ablation Analysis] ✓ Comparison plots generated")
        except Exception as e:
            print(f"[Ablation Analysis] ⚠️ Plot generation failed: {e}")
    
    # Print final summary (focusing on class-imbalance-aware metrics)
    print(f"\n[Ablation Analysis] === FINAL SUMMARY (Class-Imbalance-Aware Metrics) ===")
    if not df_summary.empty:
        print("Ablation results:")
        display_cols = ["mode", "n_features", "macro_f1", "binary_average_precision", "top_k_accuracy"]
        available_cols = [c for c in display_cols if c in df_summary.columns]
        print(df_summary[available_cols].round(3).to_string(index=False))
        
        print(f"\n🎯 BEST PERFORMING MODES:")
        if summary_report["results_summary"]["best_mode_f1"]:
            print(f"  Best Macro F1: {summary_report['results_summary']['best_mode_f1']}")
        if summary_report["results_summary"]["best_mode_ap"]:
            print(f"  Best Average Precision: {summary_report['results_summary']['best_mode_ap']}")
        if summary_report["results_summary"]["best_mode_topk"]:
            print(f"  Best Top-k Accuracy: {summary_report['results_summary']['best_mode_topk']}")
        
        # Show the key ablation finding
        full_mode = df_summary[df_summary['mode'] == 'full']
        no_spliceai_mode = df_summary[df_summary['mode'] == 'no_spliceai']
        
        if not full_mode.empty and not no_spliceai_mode.empty:
            full_f1 = full_mode['macro_f1'].iloc[0]
            no_spliceai_f1 = no_spliceai_mode['macro_f1'].iloc[0]
            f1_drop = full_f1 - no_spliceai_f1
            relative_drop = (f1_drop / full_f1) * 100 if full_f1 > 0 else 0
            
            print(f"\n🔬 KEY FINDING - SpliceAI Contribution:")
            print(f"  Full features F1: {full_f1:.3f}")
            print(f"  No SpliceAI F1: {no_spliceai_f1:.3f}")
            print(f"  Impact: {f1_drop:.3f} F1 drop ({relative_drop:.1f}% relative)")
            
            if f1_drop > 0.2:
                print(f"  🚨 SpliceAI features are CRITICAL for performance")
            elif f1_drop > 0.1:
                print(f"  ✅ SpliceAI features provide SIGNIFICANT benefit")
            else:
                print(f"  ⚠️  SpliceAI features provide MODERATE benefit")
    
    print(f"\nResults saved to: {out_dir}")
    print(f"Summary report: ablation_report.json")
    print(f"Detailed results: ablation_summary.csv")


if __name__ == "__main__":  # pragma: no cover
    main()
