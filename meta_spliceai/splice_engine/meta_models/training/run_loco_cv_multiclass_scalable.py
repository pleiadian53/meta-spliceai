#!/usr/bin/env python3
"""Memory-efficient Leave-One-Chromosome-Out (LOCO-CV) driver for the meta-model.

This script provides a scalable implementation of chromosome-aware cross-validation
that can handle large datasets (20K+ genes) through:
1. Chunked/streaming data loading
2. Feature selection to reduce dimensionality
3. Sparse matrix representation for k-mers
4. Memory-efficient processing of folds

Usage is identical to run_loco_cv_multiclass.py with additional scaling options.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

# Suppress repeated "invalid value encountered in divide" warnings from numpy.corrcoef
warnings.filterwarnings(
    "ignore",
    message=r"invalid value encountered in divide",
    category=RuntimeWarning,
    module=r"numpy",
)

from meta_spliceai.splice_engine.meta_models.builder import preprocessing
from meta_spliceai.splice_engine.meta_models.training import chromosome_split as csplit
from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import _encode_labels, _preprocess_features_for_model

# Enhanced analysis and visualization imports
from meta_spliceai.splice_engine.meta_models.evaluation.viz_utils import plot_roc_pr_curves, check_feature_correlations
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
from meta_spliceai.splice_engine.meta_models.evaluation.shap_incremental import (
    run_incremental_shap_analysis
)
from meta_spliceai.splice_engine.meta_models.evaluation.feature_importance_integration import (
    run_gene_cv_feature_importance_analysis
)
from meta_spliceai.splice_engine.meta_models.evaluation.cv_metrics_viz import (
    generate_cv_metrics_report
)

# Import overfitting monitoring
from meta_spliceai.splice_engine.meta_models.evaluation.overfitting_monitor import (
    OverfittingMonitor, enhanced_model_training
)

# Import scalability modules
try:
    from meta_spliceai.splice_engine.meta_models.training import scalability_utils, chunked_datasets
    SCALABILITY_UTILS_AVAILABLE = True
except ImportError:
    SCALABILITY_UTILS_AVAILABLE = False
    print("Warning: scalability_utils module not available. Running in compatibility mode.")

# Import CV utilities
from meta_spliceai.splice_engine.meta_models.training import cv_utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    p = argparse.ArgumentParser(description="Memory-efficient LOCO-CV for the 3-way meta-classifier.")
    p.add_argument("--dataset", required=True, help="Dataset directory or single Parquet file")
    p.add_argument("--out-dir", required=True, help="Output directory for metrics and models")

    p.add_argument(
        "--group-col", default="chrom", help="Column defining the CV groups (default: 'chrom')"
    )
    p.add_argument(
        "--gene-col", default="gene_id", help="Column used for gene-aware CV (default: 'gene_id')"
    )
    p.add_argument(
        "--base-tsv", help="Include base model columns from this TSV in output metrics"
    )
    p.add_argument(
        "--errors-only",
        action="store_true",
        help="Evaluate only on rows where the base model made an error",
    )

    p.add_argument(
        "--row-cap",
        type=int,
        default=0,
        help="Cap on the number of rows to use (0 = no cap)",
    )
    p.add_argument(
        "--valid-size",
        type=float,
        default=0.15,
        help="Fraction of training data to use for validation (default: 0.15)",
    )
    p.add_argument(
        "--min-rows-test",
        type=int,
        default=1000,
        help="Minimum number of rows in each test fold",
    )
    p.add_argument(
        "--heldout-chroms",
        type=str,
        help="Fixed test chromosomes (comma-separated), e.g. '21,22,X'. If not provided, use LOCO-CV.",
    )

    # XGBoost options
    p.add_argument(
        "--tree-method",
        type=str,
        default="hist",
        choices=["hist", "gpu_hist", "approx"],
        help="XGBoost tree construction algorithm (default: hist)",
    )
    p.add_argument(
        "--max-bin",
        type=int,
        default=256,
        help="Maximum number of bins for histogram-based algorithms",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for XGBoost ('auto', 'cpu', 'cuda', or specific GPU index)",
    )
    p.add_argument(
        "--n-estimators",
        type=int,
        default=1200,
        help="Number of boosting rounds (default: 1200)",
    )
    
    p.add_argument(
        "--leakage-probe",
        action="store_true",
        help="Run leakage correlation probe after training",
    )

    # Feature selection options
    p.add_argument(
        "--feature-selection",
        action="store_true",
        help="Enable feature selection to reduce dimensionality",
    )
    p.add_argument(
        "--max-features",
        type=int,
        default=1000,
        help="Maximum number of features to select when feature selection is enabled",
    )
    p.add_argument(
        "--selection-method",
        type=str,
        default="model",
        choices=["model", "mutual_info"],
        help="Method to use for feature selection",
    )
    p.add_argument(
        "--exclude-features",
        type=str,
        help="Path to a file with features to exclude, one per line",
    )
    p.add_argument(
        "--force-features",
        type=str,
        help="Path to a file with features to always include, one per line",
    )
    
    # Chunked processing options
    p.add_argument(
        "--use-chunked-loading",
        action="store_true",
        help="Use memory-efficient chunked data loading",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=10000,
        help="Number of rows to process per chunk",
    )
    p.add_argument(
        "--use-sparse-kmers",
        action="store_true",
        help="Use sparse matrix representation for k-mer features",
    )
    p.add_argument(
        "--memory-optimize",
        action="store_true",
        help="Apply memory optimization to numerical data",
    )

    # Transcript evaluation options
    p.add_argument(
        "--transcript-topk",
        action="store_true",
        help="Enable transcript-level evaluation by preserving transcript columns",
    )
    p.add_argument(
        "--no-transcript-cache",
        action="store_true",
        help="Disable transcript mapping cache",
    )
    
    # Calibration options
    p.add_argument(
        "--calibrate", 
        action="store_true",
        help="Enable probability calibration for binary splice/non-splice"
    )
    p.add_argument(
        "--calibrate-per-class", 
        action="store_true",
        help="Enable per-class probability calibration instead of binary splice/non-splice calibration"
    )
    p.add_argument(
        "--calib-method", 
        default="platt", 
        choices=["platt", "isotonic"],
        help="Calibration algorithm (platt = logistic sigmoid, isotonic = monotonic)"
    )
    
    # Neighborhood analysis options
    p.add_argument(
        "--neigh-sample", 
        type=int, 
        default=0,
        help="Number of positions to sample for neighborhood analysis"
    )
    p.add_argument(
        "--neigh-window", 
        type=int, 
        default=10,
        help="Window size around sampled positions for neighborhood analysis"
    )
    p.add_argument(
        "--diag-sample", 
        type=int, 
        default=0,
        help="Number of samples to use for diagnostic plots and evaluations"
    )
    
    # ROC/PR curve plotting options
    p.add_argument(
        "--plot-curves", 
        action="store_true", 
        default=True,
        help="If set, save per-fold and mean ROC/PR curves as files (default: True)"
    )
    p.add_argument(
        "--no-plot-curves", 
        dest="plot_curves", 
        action="store_false",
        help="Disable saving of ROC/PR curves"
    )
    p.add_argument(
        "--n-roc-points", 
        type=int, 
        default=101,
        help="Number of equally spaced points (0-1) to sample when averaging ROC/PR curves (default: 101)"
    )
    p.add_argument(
        "--plot-format", 
        type=str, 
        default="pdf", 
        choices=["pdf", "png", "svg"],
        help="File format for ROC/PR curve plots (default: pdf)"
    )
    
    # Feature leakage checking
    p.add_argument(
        "--check-leakage", 
        action="store_true", 
        default=True,
        help="Check for potential feature leakage by correlation analysis (default: True)"
    )
    p.add_argument(
        "--no-leakage-check", 
        dest="check_leakage", 
        action="store_false",
        help="Disable feature leakage checking"
    )
    p.add_argument(
        "--leakage-threshold", 
        type=float, 
        default=0.95,
        help="Correlation threshold for detecting potentially leaky features (default: 0.95)"
    )
    p.add_argument(
        "--auto-exclude-leaky", 
        action="store_true", 
        default=False,
        help="Automatically exclude features that exceed leakage threshold (default: False)"
    )
    
    # Other options
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (equivalent to verbosity level 2)",
    )
    p.add_argument(
        "--verbosity", 
        type=int,
        default=1,
        help="Verbosity level (0=quiet, 1=normal, 2=debug). Use --verbose flag for debug level.",
    )
    
    # Overfitting monitoring arguments
    p.add_argument(
        "--monitor-overfitting", 
        action="store_true", 
        default=False,
        help="Enable comprehensive overfitting monitoring and analysis"
    )
    p.add_argument(
        "--overfitting-threshold", 
        type=float, 
        default=0.05,
        help="Performance gap threshold for overfitting detection (default: 0.05)"
    )
    p.add_argument(
        "--early-stopping-patience", 
        type=int, 
        default=20,
        help="Patience for early stopping detection (default: 20)"
    )
    p.add_argument(
        "--convergence-improvement", 
        type=float, 
        default=0.001,
        help="Minimum improvement threshold for convergence detection (default: 0.001)"
    )

    return p.parse_args(argv)


def create_xgb_classifier(args) -> XGBClassifier:
    """Create an XGBoost classifier with the specified parameters."""
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "tree_method": args.tree_method,
        "max_bin": args.max_bin,
        "n_estimators": args.n_estimators,
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0,
        "random_state": args.seed,
    }
    
    # Add device parameter if not auto
    if args.device != "auto":
        params["device"] = args.device
    
    return XGBClassifier(**params)


def perform_neighborhood_analysis(
    model, 
    X, 
    df, 
    chrom_col, 
    pos_col, 
    sample_count, 
    window_size, 
    out_dir, 
    plot_title="Neighborhood Analysis"
) -> dict:
    """Analyze model predictions in neighborhoods around sampled positions.
    
    Parameters
    ----------
    model : classifier
        Trained model to generate predictions
    X : array-like
        Feature matrix for prediction
    df : pd.DataFrame
        Original data frame with position information
    chrom_col : str
        Column name for chromosome
    pos_col : str
        Column name for position
    sample_count : int
        Number of positions to sample
    window_size : int
        Window size around sampled positions
    out_dir : Path
        Output directory for analysis results
        
    Returns
    -------
    dict
        Analysis results
    """
    import matplotlib.pyplot as plt
    from collections import defaultdict
    
    if sample_count <= 0 or window_size <= 0:
        return None
        
    logger.info(f"Performing neighborhood analysis with {sample_count} samples and window size {window_size}")
    
    # Sample positions to analyze
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(len(df), min(sample_count, len(df)), replace=False)
    
    # Extract chromosome and position for each sample
    sample_chroms = df[chrom_col].iloc[sample_indices].values
    sample_positions = df[pos_col].iloc[sample_indices].values
    
    # Group samples by chromosome
    chrom_samples = defaultdict(list)
    for idx, (chrom, pos) in enumerate(zip(sample_chroms, sample_positions)):
        chrom_samples[chrom].append((idx, pos))
    
    # Results container
    neighborhood_results = []
    
    # Process each chromosome
    for chrom, samples in chrom_samples.items():
        # Get all positions for this chromosome
        chrom_mask = df[chrom_col] == chrom
        chrom_df = df[chrom_mask]
        chrom_X = X[chrom_mask] if isinstance(X, np.ndarray) else X.iloc[chrom_mask].values
        
        # Process each sampled position
        for sample_idx, center_pos in samples:
            # Find positions within the window
            window_mask = (chrom_df[pos_col] >= center_pos - window_size) & \
                          (chrom_df[pos_col] <= center_pos + window_size)
            window_df = chrom_df[window_mask]
            window_X = chrom_X[window_mask] if isinstance(chrom_X, np.ndarray) else chrom_X.iloc[window_mask].values
            
            if len(window_df) == 0:
                continue
                
            # Get predictions for positions in the window
            window_probs = model.predict_proba(window_X)
            
            # Record results
            for i, (pos, probs) in enumerate(zip(window_df[pos_col], window_probs)):
                result = {
                    "sample_idx": sample_idx,
                    "chrom": chrom,
                    "center_pos": center_pos,
                    "pos": pos,
                    "rel_pos": pos - center_pos,
                    "prob_neither": probs[0],
                    "prob_donor": probs[1],
                    "prob_acceptor": probs[2],
                    "true_label": window_df["splice_type"].iloc[i] if "splice_type" in window_df else None
                }
                neighborhood_results.append(result)
    
    if not neighborhood_results:
        logger.warning("No neighborhood results generated. Check window size and sample count.")
        return None
        
    # Convert to DataFrame
    neigh_df = pd.DataFrame(neighborhood_results)
    
    # Save results
    neigh_path = out_dir / "neighborhood_analysis.csv"
    neigh_df.to_csv(neigh_path, index=False)
    logger.info(f"Saved neighborhood analysis to {neigh_path}")
    
    # Create visualization
    try:
        plt.figure(figsize=(12, 8))
        
        # Group by relative position and calculate average probabilities
        pos_probs = neigh_df.groupby("rel_pos").agg({
            "prob_donor": "mean",
            "prob_acceptor": "mean",
            "prob_neither": "mean"
        }).reset_index()
        
        plt.plot(pos_probs["rel_pos"], pos_probs["prob_donor"], label="Donor")
        plt.plot(pos_probs["rel_pos"], pos_probs["prob_acceptor"], label="Acceptor")
        plt.plot(pos_probs["rel_pos"], pos_probs["prob_neither"], label="Neither")
        
        plt.xlabel("Distance from sampled position")
        plt.ylabel("Average probability")
        plt.title("Neighborhood probability patterns")
        plt.axvline(x=0, color="gray", linestyle="--")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.savefig(out_dir / "neighborhood_analysis.png")
        plt.close()
        logger.info(f"Created neighborhood visualization at {out_dir / 'neighborhood_analysis.png'}")
        
    except Exception as e:
        logger.warning(f"Failed to create neighborhood visualization: {e}")
        
    return {"neighborhood_samples": len(neigh_df)}


def evaluate_fold(
    model, 
    X_test, 
    y_test, 
    fold_name: str,
    splice_mask: np.ndarray = None,
    topk_data: dict = None,
) -> Dict:
    """Evaluate model performance on a test fold."""
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {
        "fold": fold_name,  # Use 'fold' to match gene CV format
        "test_samples": len(y_test),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_macro_f1": f1_score(y_test, y_pred, average="macro"),
    }
    
    # Calculate splice site accuracy (excluding "neither" class)
    if splice_mask is not None and np.any(splice_mask):
        y_test_splice = y_test[splice_mask]
        y_pred_splice = y_pred[splice_mask]
        metrics["splice_samples"] = len(y_test_splice)
        metrics["splice_accuracy"] = accuracy_score(y_test_splice, y_pred_splice)
        metrics["splice_macro_f1"] = f1_score(y_test_splice, y_pred_splice, average="macro")
    else:
        # Add default values if no splice sites
        metrics["splice_samples"] = 0
        metrics["splice_accuracy"] = np.nan
        metrics["splice_macro_f1"] = np.nan
    
    # Calculate ROC AUC and Average Precision for meta-only comparison
    y_true_bin = (y_test != 0).astype(int)  # Binary: splice vs non-splice
    y_prob_meta_bin = y_proba[:, 1] + y_proba[:, 2]  # Combined splice probability
    
    if len(np.unique(y_true_bin)) > 1:  # Ensure both classes are present
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            metrics["auc_meta"] = roc_auc_score(y_true_bin, y_prob_meta_bin)
            metrics["ap_meta"] = average_precision_score(y_true_bin, y_prob_meta_bin)
        except Exception:
            metrics["auc_meta"] = np.nan
            metrics["ap_meta"] = np.nan
    else:
        metrics["auc_meta"] = np.nan
        metrics["ap_meta"] = np.nan
    
    # Calculate top-k accuracy
    if topk_data and hasattr(_cutils, 'evaluate_top_k_accuracy'):
        try:
            topk_acc = _cutils.evaluate_top_k_accuracy(
                y_proba, 
                topk_data["tx_ids"], 
                topk_data["tx_labels"], 
                k=3
            )
            metrics["top_k_accuracy"] = topk_acc
        except Exception:
            metrics["top_k_accuracy"] = np.nan
    else:
        metrics["top_k_accuracy"] = np.nan
    
    return metrics


def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Setup logging and validation
    cv_utils.setup_cv_logging(args)
    cv_utils.validate_cv_arguments(args)
    
    # Smart dataset path resolution
    original_dataset_path, actual_dataset_path, parquet_count = cv_utils.resolve_dataset_path(args.dataset)
    args.dataset = actual_dataset_path
    
    out_dir = cv_utils.create_output_directory(args.out_dir)
    
    # Initialize sparse_data variable
    sparse_data = {"use_sparse": False}
    
    # Determine if we're using sparse representation
    # Note: For now, we disable sparse representation since use_sparse argument doesn't exist
    # This can be enabled later by adding the argument to the parser
    use_sparse_kmers = getattr(args, 'use_sparse_kmers', False)
    
    if use_sparse_kmers and SCALABILITY_UTILS_AVAILABLE:
        logger.info("Using sparse representation for k-mer features")
        # Load with sparse k-mer representation
        logger.info("Loading dataset with sparse k-mer features")
        sparse_data = load_sparse_dataset(
            args.dataset, 
            row_cap=args.row_cap,
            chunksize=args.chunksize,
            random_state=args.seed
        )
    
    # ---------------------------------------------------------------------
    # 1. Data loading using memory-efficient methods
    # ---------------------------------------------------------------------
    use_chunked_loading = getattr(args, 'use_chunked_loading', False)
    
    if use_chunked_loading and SCALABILITY_UTILS_AVAILABLE:
        logger.info("Using memory-efficient chunked data loading")
        
        # Load exclude/include feature lists if provided
        exclude_features = []
        force_include_features = []
        
        if args.exclude_features and os.path.exists(args.exclude_features):
            with open(args.exclude_features, 'r') as f:
                exclude_features = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(exclude_features)} features to exclude")
        
        if args.force_features and os.path.exists(args.force_features):
            with open(args.force_features, 'r') as f:
                force_include_features = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(force_include_features)} features to force include")
            
        # Required columns for CV
        required_cols = ['splice_type', args.group_col]
        if args.gene_col != args.group_col:
            required_cols.append(args.gene_col)
        if args.transcript_topk:
            required_cols.extend(['transcript_id'])
        
        # Note: chunked loading implementation would go here
        # For now, fall back to standard loading since the args don't exist
        logger.info("Chunked loading not fully implemented, falling back to standard loading")
        
        # Use standard dataset loading
        logger.info("Using standard dataset loading")
        from meta_spliceai.splice_engine.meta_models.training import datasets
        
        df = datasets.load_dataset(args.dataset)
        
        # Extract feature names from the DataFrame
        feature_names = [col for col in df.columns if col not in ['splice_type', 'gene_id', 'chrom', 'position']]
        
        # Assign columns to X, y, and other arrays
        y_raw = df["splice_type"].to_numpy()
        chrom_array = df[args.group_col].to_numpy()
        gene_array = df[args.gene_col].to_numpy() if args.gene_col in df.columns else None
        
        # Encode string labels to numeric values using sklearn's LabelEncoder
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        
        # Map to expected class indices (donor: 0, acceptor: 1, neither: 2)
        encoder_classes = label_encoder.classes_
        logger.info(f"Original label encoder classes: {encoder_classes}")
        
        # Define expected class mapping based on our schema
        class_mapping = {}
        for i, class_name in enumerate(encoder_classes):
            if class_name == 'donor':
                class_mapping[i] = 0
            elif class_name == 'acceptor':
                class_mapping[i] = 1
            else:  # '0' or other non-splice values
                class_mapping[i] = 2
        
        # Apply the mapping to get final classes
        y = np.array([class_mapping[label] for label in y])
        
        # Log the encoded label distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        logger.info(f"Final encoded labels: {dict(zip(unique_labels, counts))}"
                    f" (0=donor, 1=acceptor, 2=neither)")
        
        # Extract features
        X = df[feature_names]
        sparse_data = {"use_sparse": False}
    
    else:
        # Use standard dataset loading
        logger.info("Using standard dataset loading")
        from meta_spliceai.splice_engine.meta_models.training import datasets
        
        df = datasets.load_dataset(args.dataset)
        
        # Extract feature names from the DataFrame
        feature_names = [col for col in df.columns if col not in ['splice_type', 'gene_id', 'chrom', 'position']]
        
        # Assign columns to X, y, and other arrays
        y_raw = df["splice_type"].to_numpy()
        chrom_array = df[args.group_col].to_numpy()
        gene_array = df[args.gene_col].to_numpy() if args.gene_col in df.columns else None
        
        # Encode string labels to numeric values using sklearn's LabelEncoder
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        
        # Map to expected class indices (donor: 0, acceptor: 1, neither: 2)
        encoder_classes = label_encoder.classes_
        logger.info(f"Original label encoder classes: {encoder_classes}")
        
        # Define expected class mapping based on our schema
        class_mapping = {}
        for i, class_name in enumerate(encoder_classes):
            if class_name == 'donor':
                class_mapping[i] = 0
            elif class_name == 'acceptor':
                class_mapping[i] = 1
            else:  # '0' or other non-splice values
                class_mapping[i] = 2
        
        # Apply the mapping to get final classes
        y = np.array([class_mapping[label] for label in y])
        
        # Log the encoded label distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        logger.info(f"Final encoded labels: {dict(zip(unique_labels, counts))}"
                    f" (0=donor, 1=acceptor, 2=neither)")
        
        # Extract features
        X = df[feature_names]
        sparse_data = {"use_sparse": False}
    
    # ---------------------------------------------------------------------
    # 2. Feature selection (if enabled)
    # ---------------------------------------------------------------------
    feature_selection_info = {}
    
    if args.feature_selection and SCALABILITY_UTILS_AVAILABLE and X is not None:
        # Skip if we're already using sparse representation (handled during loading)
        logger.info(f"Running feature selection using '{args.selection_method}' method")
        logger.info(f"Selecting up to {args.max_features} features from {len(feature_names)}")
        
        start_time = pd.Timestamp.now()
        X_selected, selected_features, feature_info = scalability_utils.select_features(
            X, y,
            max_features=args.max_features,
            method=args.selection_method,
            random_state=args.seed,
            exclude_features=exclude_features,
            force_include_features=force_include_features
        )
        duration = (pd.Timestamp.now() - start_time).total_seconds()
        
        # Update feature information
        logger.info(f"Feature selection completed in {duration:.2f} seconds")
        logger.info(f"Selected {len(selected_features)} features out of {len(feature_names)}")
        logger.info(f"  - K-mer features: {feature_info['selected_kmer_features']} (was {feature_info['original_kmer_features']})")
        logger.info(f"  - Non-k-mer features: {feature_info['selected_non_kmer_features']} (was {feature_info['original_non_kmer_features']})")
        
        # Update X and feature_names
        X = X_selected
        feature_names = selected_features
        feature_selection_info = feature_info
        
        # Save feature selection information
        with open(out_dir / "feature_selection_info.json", "w") as f:
            json.dump(feature_info, f, indent=2)
        
        # Save selected feature list
        with open(out_dir / "selected_features.txt", "w") as f:
            for feature in selected_features:
                f.write(f"{feature}\n")
    
    # ---------------------------------------------------------------------
    # 3. Set up cross-validation splits
    # ---------------------------------------------------------------------
    # Always save transcript mapping columns if transcript-topk is enabled
    transcript_columns = {}
    if args.transcript_topk:
        if sparse_data["use_sparse"]:
            # Extract transcript data from metadata
            transcript_columns = {
                "transcript_id": metadata.get("transcript_id"),
                "chrom": metadata.get("chrom"),
                "position": metadata.get("position")
            }
        else:
            # Extract transcript columns from DataFrame - preserve position and chrom for mapping
            # Always get the original string version of chrom from the input dataframe
            # This ensures we have it for transcript mapping regardless of encoding
            # Handle both pandas and polars dataframes
            if hasattr(df, 'to_pandas'):
                # It's a polars DataFrame
                pandas_df = df.select(['chrom', 'position']).to_pandas()
                transcript_columns['chrom'] = pandas_df['chrom']
                if 'position' in pandas_df.columns:
                    transcript_columns['position'] = pandas_df['position']
            else:
                # It's already a pandas DataFrame
                transcript_columns['chrom'] = df['chrom'].copy()
                if 'position' in df.columns:
                    transcript_columns['position'] = df['position'].copy()
            
            # Also get transcript_id if available
            for col in ["transcript_id"]:
                if col in df.columns:
                    transcript_columns[col] = df[col].values
    
    # Save feature manifest
    if not sparse_data["use_sparse"]:
        pd.DataFrame({"feature": feature_names}).to_csv(out_dir / "feature_manifest.csv", index=False)

    # Check for potential feature leakage if enabled
    if args.check_leakage and not sparse_data["use_sparse"] and X is not None:
        # Need X and y in numpy format for correlation analysis
        X_np = X.values if hasattr(X, 'values') else X
        y_np = y
        curr_features = feature_names
        
        # Save a correlation report in the output directory
        correlation_report_path = out_dir / "feature_label_correlations.csv"
        logger.info(f"Checking for potential feature leakage (threshold={args.leakage_threshold})...")
        
        try:
            leaky_features, corr_df = check_feature_correlations(
                X_np, y_np, curr_features, args.leakage_threshold, correlation_report_path
            )
            
            if leaky_features:
                logger.warning(f"Found {len(leaky_features)} potentially leaky features:")
                for feat in leaky_features:
                    logger.warning(f"  - {feat}")
                
                # If auto-exclude is enabled, remove leaky features
                if args.auto_exclude_leaky:
                    logger.info(f"Auto-excluding {len(leaky_features)} potentially leaky features")
                    if hasattr(X, 'drop'):
                        X = X.drop(columns=leaky_features)
                    else:
                        # Remove by index for numpy arrays
                        leaky_indices = [i for i, f in enumerate(feature_names) if f in leaky_features]
                        X = np.delete(X, leaky_indices, axis=1)
                    
                    # Update feature names
                    feature_names = [f for f in feature_names if f not in leaky_features]
                    logger.info(f"Feature count reduced to {len(feature_names)} after removing leaky features")
            else:
                logger.info("No potentially leaky features detected")
        except Exception as e:
            logger.warning(f"Feature leakage check failed: {e}")
            if args.verbose >= 2:
                import traceback
                traceback.print_exc()
    
    # Set up chromosome splits
    fold_rows = []
    
    if args.heldout_chroms:
        # Fixed chromosome split
        heldout = args.heldout_chroms.split(",")
        logger.info(f"Using fixed heldout chromosomes: {heldout}")
        
        # Create train/test split
        test_mask = np.isin(chrom_array, heldout)
        train_val_mask = ~test_mask
        
        # Create train/valid split
        inner_genes = gene_array[train_val_mask] if gene_array is not None else None
        train_idx_rel, valid_idx_rel = csplit._train_valid_split(
            range(train_val_mask.sum()),
            y[train_val_mask],
            valid_size=args.valid_size,
            gene_groups=inner_genes,
            seed=args.seed,
        )
        
        # Convert to absolute indices
        train_val_indices = np.where(train_val_mask)[0]
        train_idx = train_val_indices[train_idx_rel]
        valid_idx = train_val_indices[valid_idx_rel]
        test_idx = np.where(test_mask)[0]
        
        # Add single fold
        folds = [("heldout", train_idx, valid_idx, test_idx)]
        
    else:
        # Leave-one-out cross-validation
        logger.info(f"Using Leave-One-Chromosome-Out CV with {args.group_col}")
        folds = []
        
        # Ensure we have numpy arrays for CV splitting
        y_array = y.values if hasattr(y, 'values') else np.array(y)
        chrom_array_np = chrom_array.values if hasattr(chrom_array, 'values') else np.array(chrom_array)
        gene_array_np = gene_array.values if gene_array is not None and hasattr(gene_array, 'values') else gene_array
        
        # Create dummy X for splitting (only used for shape)
        dummy_X = np.arange(len(y_array)).reshape(-1, 1)
        
        # Get CV folds
        for fold_name, train_idx, valid_idx, test_idx in csplit.loco_cv_splits(
            dummy_X,  # Use dummy array with correct shape
            y_array,
            chrom_array_np,
            valid_size=args.valid_size,
            gene_array=gene_array_np,
            min_rows=args.min_rows_test,
            seed=args.seed,
        ):
            folds.append((fold_name, train_idx, valid_idx, test_idx))
        
        logger.info(f"Created {len(folds)} CV folds")
    
    # ---------------------------------------------------------------------
    # 4. Run CV folds
    # ---------------------------------------------------------------------
    logger.info("Starting cross-validation")
    fold_rows = []
    
    # Initialize overfitting monitor if enabled
    monitor = None
    if args.monitor_overfitting:
        logger.info("Initializing overfitting detection system")
        logger.info(f"  Primary metric: mlogloss")
        logger.info(f"  Gap threshold: {args.overfitting_threshold}")
        logger.info(f"  Early stopping patience: {args.early_stopping_patience}")
        logger.info(f"  Convergence improvement: {args.convergence_improvement}")
        
        monitor = OverfittingMonitor(
            primary_metric="mlogloss",  # Use multiclass log loss
            gap_threshold=args.overfitting_threshold,
            patience=args.early_stopping_patience,
            min_improvement=args.convergence_improvement
        )
    
    # For calibration, we need to collect validation probabilities and labels
    calib_data = None
    if args.calibrate or args.calibrate_per_class:
        logger.info(f"Collecting validation data for {'per-class' if args.calibrate_per_class else 'binary'} calibration")
        calib_data = {
            # For binary splice/non-splice calibration
            'binary': {
                'probs': [],    # Combined donor+acceptor probabilities
                'labels': []   # Binary 0/1 (neither/splice)
            },
            # For per-class calibration
            'per_class': {
                0: {'probs': [], 'labels': []},  # Neither vs rest
                1: {'probs': [], 'labels': []},  # Donor vs rest
                2: {'probs': [], 'labels': []}   # Acceptor vs rest
            }
        }
    
    # Data collection for enhanced ROC/PR curve plotting
    if args.plot_curves:
        # Containers for ROC and PR curve data
        roc_curves = []  # List of (fpr, tpr) arrays per fold
        pr_curves = []   # List of (recall, precision) arrays per fold
        auc_values = []  # AUC values per fold
        ap_values = []   # Average precision values per fold
        
        # For plotting: collect truth and predictions across folds
        y_true_bins = []   # Binary labels for each fold
        y_prob_metas = []  # Meta model probabilities for each fold
        y_true_multiclass = []  # Multiclass labels for each fold
        y_prob_metas_multiclass = []  # Meta model multiclass probabilities for each fold
    
    for fold_idx, (fold_name, train_idx, valid_idx, test_idx) in enumerate(folds):
        logger.info(f"Processing fold {fold_idx+1}/{len(folds)}: {fold_name}")
        
        # Create train/valid/test sets
        if sparse_data["use_sparse"]:
            # Special handling for sparse representation
            X_train_kmer = sparse_data["X_kmer"][train_idx]
            X_train_non_kmer = sparse_data["X_non_kmer"].iloc[train_idx].reset_index(drop=True)
            
            X_valid_kmer = sparse_data["X_kmer"][valid_idx] if len(valid_idx) > 0 else None
            X_valid_non_kmer = sparse_data["X_non_kmer"].iloc[valid_idx].reset_index(drop=True) if len(valid_idx) > 0 else None
            
            X_test_kmer = sparse_data["X_kmer"][test_idx]
            X_test_non_kmer = sparse_data["X_non_kmer"].iloc[test_idx].reset_index(drop=True)
            
            # Create DMatrix objects for XGBoost
            dtrain = _create_combined_dmatrix(
                X_train_kmer, X_train_non_kmer, y[train_idx]
            )
            
            dvalid = _create_combined_dmatrix(
                X_valid_kmer, X_valid_non_kmer, y[valid_idx]
            ) if len(valid_idx) > 0 else None
            
            # For prediction, we'll construct test data later
            
            clf = create_xgb_classifier(args)
            
            # Set up eval set if validation data exists
            eval_set = [(dtrain, 'train')]
            if dvalid is not None:
                eval_set.append((dvalid, 'val'))
            
            # Train model
            clf.fit(
                dtrain,
                y=None,  # Labels are included in DMatrix
                eval_set=eval_set,
                verbose=True if args.verbose > 0 else False,
            )
            
            # Add evaluation results to overfitting monitor
            if monitor is not None:
                try:
                    # Get evaluation results for overfitting analysis
                    evals_result = clf.evals_result() if hasattr(clf, 'evals_result') else {}
                    if evals_result:
                        monitor.add_fold_metrics(evals_result, fold_name)
                        if args.verbose >= 2:
                            logger.debug(f"Added overfitting metrics for fold {fold_name}")
                except Exception as e:
                    if args.verbose >= 1:
                        logger.warning(f"Failed to add overfitting metrics for fold {fold_name}: {e}")
            
            # Save model
            clf.save_model(str(fold_model_path))
            
            # Collect validation probabilities for calibration
            if (args.calibrate or args.calibrate_per_class) and dvalid is not None:
                val_probs = clf.predict_proba(dvalid)
                val_labels = y[valid_idx]
                
                # Store for binary calibration
                if args.calibrate:
                    # Calculate binary labels (splice site vs non-splice site)
                    binary_labels = (val_labels != 0).astype(int)  # 0=neither, 1/2=splice
                    # Combined donor+acceptor probability
                    binary_probs = val_probs[:, 1] + val_probs[:, 2]
                    calib_data['binary']['probs'].append(binary_probs)
                    calib_data['binary']['labels'].append(binary_labels)
                
                # Store for per-class calibration
                if args.calibrate_per_class:
                    for cls_idx in range(3):  # 0=neither, 1=donor, 2=acceptor
                        cls_labels = (val_labels == cls_idx).astype(int)
                        cls_probs = val_probs[:, cls_idx]
                        calib_data['per_class'][cls_idx]['probs'].append(cls_probs)
                        calib_data['per_class'][cls_idx]['labels'].append(cls_labels)
            
            # For prediction, create combined test data
            X_test_combined = _combine_sparse_and_dense(X_test_kmer, X_test_non_kmer)
            
        else:
            # Standard NumPy arrays
            X_train = X.iloc[train_idx].values if hasattr(X, 'iloc') else X[train_idx]
            y_train = y[train_idx]
            
            X_valid = X.iloc[valid_idx].values if hasattr(X, 'iloc') else X[valid_idx] if len(valid_idx) > 0 else None
            y_valid = y[valid_idx] if len(valid_idx) > 0 else None
            
            X_test = X.iloc[test_idx].values if hasattr(X, 'iloc') else X[test_idx]
            y_test = y[test_idx]

            # Log label distributions to help debug
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            logger.info(f"Train label distribution: {dict(zip(unique_train, counts_train))}")
            if y_valid is not None and len(y_valid) > 0:
                unique_valid, counts_valid = np.unique(y_valid, return_counts=True)
                logger.info(f"Valid label distribution: {dict(zip(unique_valid, counts_valid))}")
            
            # Create and train model
            fold_model_path = out_dir / f"fold_{fold_name}_model.json"
            
            # Train using standard arrays
            clf = create_xgb_classifier(args)
            
            # Set up eval set
            eval_set = [(X_train, y_train)]
            if len(valid_idx) > 0 and y_valid is not None:
                eval_set.append((X_valid, y_valid))
            
            # Train model
            clf.fit(
                X_train, 
                y_train,
                eval_set=eval_set,
                verbose=True if args.verbose > 0 else False,
            )
            
            # Add evaluation results to overfitting monitor
            if monitor is not None:
                try:
                    # Get evaluation results for overfitting analysis
                    evals_result = clf.evals_result() if hasattr(clf, 'evals_result') else {}
                    if evals_result:
                        monitor.add_fold_metrics(evals_result, fold_name)
                        if args.verbose >= 2:
                            logger.debug(f"Added overfitting metrics for fold {fold_name}")
                except Exception as e:
                    if args.verbose >= 1:
                        logger.warning(f"Failed to add overfitting metrics for fold {fold_name}: {e}")
            
            # Save model
            clf.save_model(str(fold_model_path))
            
            # Collect validation probabilities for calibration
            if (args.calibrate or args.calibrate_per_class) and len(valid_idx) > 0 and y_valid is not None:
                val_probs = clf.predict_proba(X_valid)
                val_labels = y_valid
                
                # Store for binary calibration
                if args.calibrate:
                    # Calculate binary labels (splice site vs non-splice site)
                    binary_labels = (val_labels != 0).astype(int)  # 0=neither, 1/2=splice
                    # Combined donor+acceptor probability
                    binary_probs = val_probs[:, 1] + val_probs[:, 2]
                    calib_data['binary']['probs'].append(binary_probs)
                    calib_data['binary']['labels'].append(binary_labels)
                
                # Store for per-class calibration
                if args.calibrate_per_class:
                    for cls_idx in range(3):  # 0=neither, 1=donor, 2=acceptor
                        cls_labels = (val_labels == cls_idx).astype(int)
                        cls_probs = val_probs[:, cls_idx]
                        calib_data['per_class'][cls_idx]['probs'].append(cls_probs)
                        calib_data['per_class'][cls_idx]['labels'].append(cls_labels)
            
            # For prediction, use X_test directly
            X_test_combined = X_test
        
        # Evaluate on test set
        y_test = y[test_idx]
        
        # Create topk data if needed
        topk_data = None
        if args.transcript_topk and transcript_columns:
            topk_data = {
                "tx_ids": transcript_columns["transcript_id"][test_idx],
                "tx_labels": y_test,
            }
            
        # Apply diagnostic sampling if requested
        if args.diag_sample > 0 and args.diag_sample < len(y_test):
            # Sample a subset of the test data for detailed diagnostics
            np.random.seed(42)  # For reproducibility
            diag_indices = np.random.choice(len(y_test), args.diag_sample, replace=False)
            
            # Log the sampling information
            logger.info(f"Using {args.diag_sample} diagnostic samples for detailed evaluation")
            
            # Extract sampled data - ensure consistent indexing
            # Handle numpy arrays vs DataFrames properly
            if isinstance(X_test_combined, np.ndarray):
                X_test_diag = X_test_combined[diag_indices]
            else:
                # For DataFrames, ensure we have consecutive indices first
                X_test_combined = X_test_combined.reset_index(drop=True)
                X_test_diag = X_test_combined.iloc[diag_indices]
                
            y_test_diag = y_test[diag_indices]
            
            # Update topk data if available
            if topk_data:
                topk_data = {
                    "tx_ids": topk_data["tx_ids"][diag_indices],
                    "tx_labels": topk_data["tx_labels"][diag_indices],
                }
                
            # Update splice mask if available
            if splice_mask is not None:
                splice_mask = splice_mask[diag_indices]
                
            # Use the sampled data for evaluation
            X_test_combined = X_test_diag
            y_test = y_test_diag
        
        # Calculate splice mask (non-neither class)
        splice_mask = y_test != 2  # Assuming label 2 is "neither"
        
        # Evaluate model
        metrics = evaluate_fold(
            clf, X_test_combined, y_test,
            fold_name=fold_name,
            splice_mask=splice_mask,
            topk_data=topk_data,
        )
        
        # Collect ROC/PR curve data if plotting is enabled
        if args.plot_curves:
            # Get model predictions for curve plotting
            y_proba_test = clf.predict_proba(X_test_combined)
            
            # Binary classification: splice vs non-splice
            y_true_bin = (y_test != 0).astype(int)  # 0=neither, 1/2=splice
            y_prob_meta_bin = y_proba_test[:, 1] + y_proba_test[:, 2]  # Combined splice probability
            
            # Calculate ROC and PR curves
            fpr, tpr, _ = roc_curve(y_true_bin, y_prob_meta_bin)
            precision, recall, _ = precision_recall_curve(y_true_bin, y_prob_meta_bin)
            
            # Store curve data
            roc_curves.append(np.column_stack([fpr, tpr]))
            pr_curves.append(np.column_stack([recall, precision]))
            auc_values.append(auc(fpr, tpr))
            ap_values.append(average_precision_score(y_true_bin, y_prob_meta_bin))
            
            # Store data for aggregated plotting
            y_true_bins.append(y_true_bin)
            y_prob_metas.append(y_prob_meta_bin)
            y_true_multiclass.append(y_test)
            y_prob_metas_multiclass.append(y_proba_test)
        
        # Add metrics to fold results
        fold_rows.append(metrics)
        
        # Save fold-specific metrics
        with open(out_dir / f"fold_{fold_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Clean up to free memory
        del clf
        if sparse_data["use_sparse"]:
            del dtrain, X_test_combined
            if dvalid is not None:
                del dvalid
        else:
            del X_train, X_valid, X_test
        gc.collect()
    
    # ---------------------------------------------------------------------
    # 5. Apply calibration if requested
    # ---------------------------------------------------------------------
    if (args.calibrate or args.calibrate_per_class) and calib_data is not None:
        logger.info("Fitting calibration models based on validation data")
        
        # Train a final model on all data
        logger.info("Training final model on all data for calibration")
        final_model = create_xgb_classifier(args)
        final_model.fit(X, y, verbose=args.verbose > 0)
        
        # Save the uncalibrated model
        final_model_path = out_dir / "final_model_uncalibrated.json"
        final_model.save_model(str(final_model_path))
        logger.info(f"Saved uncalibrated final model to {final_model_path}")
        
        # Per-class calibration
        if args.calibrate_per_class:
            logger.info("Fitting per-class calibrators")
            calibrators = []
            
            for cls_idx in range(3):  # 0=neither, 1=donor, 2=acceptor
                # Combine all validation data for this class
                cls_probs = np.concatenate(calib_data['per_class'][cls_idx]['probs'])
                cls_labels = np.concatenate(calib_data['per_class'][cls_idx]['labels'])
                
                logger.info(f"[Per-class calibration] Class {cls_idx}: {cls_probs.shape[0]} samples, "
                          f"{cls_labels.sum()} positives")
                
                # Create and fit calibrator based on method
                if args.calib_method == "platt":
                    calibrator = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
                    calibrator.fit(cls_probs.reshape(-1, 1), cls_labels)
                elif args.calib_method == "isotonic":
                    calibrator = IsotonicRegression(out_of_bounds="clip")
                    calibrator.fit(cls_probs, cls_labels)
                else:
                    raise ValueError(f"Unsupported calibration method: {args.calib_method}")
                
                calibrators.append(calibrator)
            
            # Create ensemble with per-class calibration
            ensemble = _cutils.PerClassCalibratedSigmoidEnsemble(
                [final_model] * 3,  # Use the same model for all three classes
                feature_names,
                calibrators
            )
            
            # Save the calibrated ensemble model
            calibrated_model_path = out_dir / "final_model_calibrated_per_class.pkl"
            with open(calibrated_model_path, "wb") as f:
                import pickle
                pickle.dump(ensemble, f)
            logger.info(f"Saved per-class calibrated model to {calibrated_model_path}")
            
        # Binary (splice/non-splice) calibration
        elif args.calibrate:
            logger.info("Fitting binary calibrator for splice/non-splice")
            
            # Combine all validation data
            s_train = np.concatenate(calib_data['binary']['probs'])
            y_bin = np.concatenate(calib_data['binary']['labels'])
            
            logger.info(f"[Binary calibration] {s_train.shape[0]} samples, {y_bin.sum()} positives")
            
            # Create and fit calibrator
            if args.calib_method == "platt":
                calibrator = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
                calibrator.fit(s_train.reshape(-1, 1), y_bin)
            elif args.calib_method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(s_train, y_bin)
            else:
                raise ValueError(f"Unsupported calibration method: {args.calib_method}")
            
            # Create calibrated ensemble
            ensemble = _cutils.CalibratedSigmoidEnsemble(
                [final_model] * 3,  # Use the same model for all three classes
                feature_names,
                calibrator
            )
            
            # Save the calibrated ensemble model
            calibrated_model_path = out_dir / "final_model_calibrated_binary.pkl"
            with open(calibrated_model_path, "wb") as f:
                import pickle
                pickle.dump(ensemble, f)
            logger.info(f"Saved binary calibrated model to {calibrated_model_path}")
        
        # Use the calibrated model for neighborhood analysis if available
        if args.calibrate or args.calibrate_per_class:
            model_for_analysis = ensemble
        else:
            model_for_analysis = final_model
            
        # Perform neighborhood analysis if requested
        if args.neigh_sample > 0 and args.neigh_window > 0 and "chrom" in meta_data and "pos" in meta_data:
            logger.info("Performing neighborhood analysis")
            
            # Create a DataFrame with position information
            analysis_df = pd.DataFrame({
                "chrom": meta_data["chrom"],
                "pos": meta_data["pos"],
                "label": y
            })
            
            # Run analysis
            neigh_results = perform_neighborhood_analysis(
                model=model_for_analysis,
                X=X,  # Use all data
                df=analysis_df,
                chrom_col="chrom",
                pos_col="pos",
                sample_count=args.neigh_sample,
                window_size=args.neigh_window,
                out_dir=out_dir
            )
            
            if neigh_results:
                logger.info(f"Neighborhood analysis complete with {neigh_results['neighborhood_samples']} samples")
    
    # ---------------------------------------------------------------------
    # 6. Output aggregated results
    # ---------------------------------------------------------------------
    df_metrics = pd.DataFrame(fold_rows)
    
    if len(fold_rows) == 0:
        logger.warning("No valid folds were found with the current sample size.")
        logger.warning("Please try increasing the row-cap or reducing min-rows-test.")
        return
        
    with open(out_dir / "loco_metrics.csv", "w") as fh:
        df_metrics.to_csv(fh, index=False)

    logger.info("\nLOCO-CV results by held-out chromosome group:")
    logger.info(df_metrics)

    metric_columns = [
        "test_accuracy", "test_macro_f1", "splice_accuracy", "splice_macro_f1"
    ]
    if "top_k_accuracy" in df_metrics.columns:
        metric_columns.append("top_k_accuracy")
    
    # Ensure all required columns exist in the dataframe
    mean_metrics = df_metrics[metric_columns].mean()
    logger.info("\nAverage across folds:")
    for name, val in mean_metrics.items():
        logger.info(f"{name:>16s} {val:.6f}")

    # Save overall summary
    with open(out_dir / "metrics_aggregate.json", "w") as fh:
        json.dump({k: float(v) for k, v in mean_metrics.items()}, fh, indent=2)
    
    # Generate CV metrics visualization report
    try:
        logger.info("Generating CV metrics visualization report...")
        cv_metrics_csv = out_dir / "loco_metrics.csv"
        if cv_metrics_csv.exists():
            viz_result = generate_cv_metrics_report(
                csv_path=cv_metrics_csv,
                out_dir=out_dir,
                plot_format=args.plot_format,
                dpi=300
            )
            logger.info(f"CV metrics visualization completed successfully")
            logger.info(f"Visualization directory: {viz_result['visualization_dir']}")
            logger.info(f"Generated {len(viz_result['plot_files'])} plots:")
            for plot_name, plot_path in viz_result['plot_files'].items():
                logger.info(f"  - {plot_name.replace('_', ' ').title()}: {Path(plot_path).name}")
        else:
            logger.warning(f"CV metrics CSV not found at {cv_metrics_csv}")
    except Exception as e:
        logger.warning(f"CV metrics visualization failed with error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
    
    # Generate overfitting analysis if monitoring was enabled
    if monitor is not None and len(monitor.fold_metrics) > 0:
        try:
            logger.info("Generating comprehensive overfitting analysis...")
            
            # Create overfitting analysis subdirectory
            overfitting_dir = out_dir / "overfitting_analysis"
            overfitting_dir.mkdir(exist_ok=True)
            
            # Generate overfitting report
            overfitting_report = monitor.generate_overfitting_report(overfitting_dir)
            
            # Generate learning curves and visualizations
            monitor.plot_learning_curves(overfitting_dir, plot_format=args.plot_format)
            
            # Print summary
            summary = overfitting_report['summary']
            logger.info("Overfitting Analysis Summary:")
            logger.info(f"  Total CV folds analyzed: {summary['total_folds']}")
            logger.info(f"  Folds with overfitting: {summary['folds_with_overfitting']}")
            logger.info(f"  Early stopped folds: {summary['early_stopped_folds']}")
            logger.info(f"  Mean performance gap: {summary['mean_performance_gap']:.4f} ± {summary['std_performance_gap']:.4f}")
            logger.info(f"  Mean overfitting score: {summary['mean_overfitting_score']:.4f}")
            logger.info(f"  Recommended n_estimators: {summary['recommended_n_estimators']}")
            
            # Create additional summary report
            summary_path = overfitting_dir / "overfitting_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("Chromosome-Aware CV Overfitting Analysis Summary\n")
                f.write("=" * 55 + "\n\n")
                f.write(f"Analysis Parameters:\n")
                f.write(f"  Primary metric: mlogloss\n")
                f.write(f"  Gap threshold: {args.overfitting_threshold}\n")
                f.write(f"  Early stopping patience: {args.early_stopping_patience}\n")
                f.write(f"  Convergence improvement: {args.convergence_improvement}\n\n")
                f.write(f"Results:\n")
                f.write(f"  Total CV folds: {summary['total_folds']}\n")
                f.write(f"  Folds with overfitting: {summary['folds_with_overfitting']}\n")
                f.write(f"  Early stopped folds: {summary['early_stopped_folds']}\n")
                f.write(f"  Mean performance gap: {summary['mean_performance_gap']:.4f} ± {summary['std_performance_gap']:.4f}\n")
                f.write(f"  Recommended n_estimators: {summary['recommended_n_estimators']}\n\n")
                f.write(f"Generated Files:\n")
                f.write(f"  - overfitting_analysis.json: Detailed metrics\n")
                f.write(f"  - learning_curves_by_fold.{args.plot_format}: Individual fold curves\n")
                f.write(f"  - aggregated_learning_curves.{args.plot_format}: Mean curves with confidence bands\n")
                f.write(f"  - overfitting_summary.{args.plot_format}: Summary visualizations\n")
            
            logger.info(f"Overfitting analysis saved to: {overfitting_dir}")
            logger.info(f"Summary report: {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate overfitting analysis: {e}")
            if args.verbose >= 2:
                import traceback
                traceback.print_exc()
    elif args.monitor_overfitting:
        logger.warning("No overfitting data collected - this may indicate an issue with evaluation result capture")
    
    # Generate ROC and PR curve plots if enabled
    if args.plot_curves and len(y_true_bins) > 0:
        try:
            logger.info("Generating ROC and PR curve plots...")
            
            # Use the modular plotting function from gene CV script
            curve_metrics = plot_roc_pr_curves(
                y_true=y_true_bins,
                y_pred_base=None,  # No base model for LOCO-CV
                y_pred_meta=y_prob_metas,
                out_dir=out_dir,
                n_roc_points=args.n_roc_points,
                plot_format=args.plot_format,
                base_name=None,  # No base model
                meta_name='Meta',
                fold_ids=[f"fold_{i}" for i in range(len(y_true_bins))]
            )
            
            # Log curve metrics
            auc_m_mean = np.mean(auc_values)
            auc_m_std = np.std(auc_values)
            ap_m_mean = np.mean(ap_values)
            ap_m_std = np.std(ap_values)
            
            logger.info(f"ROC AUC  mean={auc_m_mean:.3f} ±{auc_m_std:.3f}")
            logger.info(f"PR AP    mean={ap_m_mean:.3f} ±{ap_m_std:.3f}")
            
            # Create enhanced multiclass visualizations
            logger.info("Creating multiclass ROC/PR curves...")
            try:
                multiclass_metrics = plot_multiclass_roc_pr_curves(
                    y_true=y_true_multiclass,
                    y_pred_base=None,  # No base model for LOCO-CV
                    y_pred_meta=y_prob_metas_multiclass,
                    out_dir=out_dir,
                    plot_format=args.plot_format,
                    base_name=None,
                    meta_name='Meta'
                )
                logger.info("✓ Multiclass ROC/PR curves created")
                
                # Log multiclass metrics
                for class_name in ['donor', 'acceptor']:
                    if class_name in multiclass_metrics:
                        auc_meta = multiclass_metrics[class_name]['auc']['meta']
                        ap_meta = multiclass_metrics[class_name]['ap']['meta']
                        logger.info(f"{class_name.title()} AUC: Meta={auc_meta['mean']:.3f}±{auc_meta['std']:.3f}")
                        logger.info(f"{class_name.title()} AP:  Meta={ap_meta['mean']:.3f}±{ap_meta['std']:.3f}")
            except Exception as e:
                logger.warning(f"Error creating multiclass ROC/PR curves: {e}")
                if args.verbose >= 2:
                    import traceback
                    traceback.print_exc()
            
        except Exception as e:
            logger.warning(f"ROC/PR curve plotting failed with error: {e}")
            if args.verbose >= 2:
                import traceback
                traceback.print_exc()
    
    # ---------------------------------------------------------------------
    # 7. Enhanced post-training analysis and diagnostics
    # ---------------------------------------------------------------------
    
    # Determine sample size for diagnostics
    diag_sample = None if args.diag_sample == 0 else args.diag_sample
    
    # Run enhanced SHAP analysis
    try:
        logger.info("Running enhanced SHAP importance analysis...")
        
        # Create a model for SHAP analysis (use the final model if available)
        if 'final_model' in locals():
            import pickle
            with open(out_dir / "model_multiclass.pkl", "wb") as fh:
                pickle.dump(final_model, fh)
        
        # Try memory-efficient SHAP analysis first
        shap_analysis_completed = False
        try:
            run_incremental_shap_analysis(args.dataset, out_dir, sample=diag_sample)
            shap_analysis_completed = True
            logger.info("Memory-efficient SHAP analysis completed successfully")
            
            # Generate comprehensive SHAP visualization report
            logger.info("Creating comprehensive SHAP analysis report...")
            try:
                shap_importance_csv = out_dir / "shap_importance_incremental.csv"
                model_pkl = out_dir / "model_multiclass.pkl"
                
                if shap_importance_csv.exists() and model_pkl.exists():
                    shap_results = generate_comprehensive_shap_report(
                        importance_csv=shap_importance_csv,
                        model_path=model_pkl,
                        dataset_path=args.dataset,
                        out_dir=out_dir,
                        top_n=20,
                        sample_size=min(1000, diag_sample if diag_sample else 1000),
                        plot_format=args.plot_format
                    )
                    logger.info("✓ Comprehensive SHAP report generated")
                    
                    # Log top features for each class
                    if 'summary_stats' in shap_results:
                        stats = shap_results['summary_stats']
                        logger.info(f"SHAP Summary - Top features by class:")
                        logger.info(f"  Overall: {stats.get('top_feature_overall', 'N/A')}")
                        logger.info(f"  Neither: {stats.get('top_feature_neither', 'N/A')}")
                        logger.info(f"  Donor:   {stats.get('top_feature_donor', 'N/A')}")
                        logger.info(f"  Acceptor: {stats.get('top_feature_acceptor', 'N/A')}")
                else:
                    logger.warning("Required files missing for comprehensive SHAP analysis")
            except Exception as e:
                logger.warning(f"Error generating comprehensive SHAP report: {e}")
                if args.verbose >= 2:
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            logger.warning(f"Memory-efficient SHAP analysis failed: {e}")
            # Fall back to standard SHAP analysis
            try:
                _cutils.shap_importance(args.dataset, out_dir, sample=diag_sample)
                shap_analysis_completed = True
                logger.info("Standard SHAP importance analysis completed successfully")
            except Exception as e2:
                logger.warning(f"Standard SHAP importance analysis also failed: {e2}")
        
        if not shap_analysis_completed:
            logger.info("Both SHAP analysis methods failed. This is often due to:")
            logger.info("  - Non-numeric data in the feature matrix")
            logger.info("  - Memory constraints with large datasets")
            logger.info("  - Model compatibility issues with SHAP TreeExplainer")
            logger.info("  - The analysis will continue with other diagnostic methods")
    
    except Exception as e:
        logger.warning(f"SHAP analysis failed with error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
    
    # Run comprehensive feature importance analysis
    try:
        logger.info("Running comprehensive feature importance analysis...")
        feature_importance_dir = run_gene_cv_feature_importance_analysis(
            args.dataset, out_dir, sample=diag_sample
        )
        if feature_importance_dir:
            logger.info("Comprehensive feature importance analysis completed successfully")
        else:
            logger.warning("Feature importance analysis failed")
    except Exception as e:
        logger.warning(f"Feature importance analysis failed with error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
    
    # Run additional diagnostics
    try:
        logger.info("Running additional diagnostic analyses...")
        _cutils.richer_metrics(args.dataset, out_dir, sample=diag_sample)
        _cutils.gene_score_delta(args.dataset, out_dir, sample=diag_sample)
        _cutils.probability_diagnostics(args.dataset, out_dir, sample=diag_sample)
        _cutils.base_vs_meta(args.dataset, out_dir, sample=diag_sample)
        
        # Run meta splice performance evaluation
        try:
            logger.info("Running meta splice performance evaluation...")
            _cutils.meta_splice_performance(
                dataset_path=args.dataset,
                run_dir=out_dir,
                threshold=0.9,  # Default threshold, will be overridden by threshold_suggestion.txt if available
                sample=diag_sample,
            )
        except Exception as e:
            logger.warning(f"Meta splice performance evaluation failed: {e}")
            if args.verbose >= 2:
                import traceback
                traceback.print_exc()
        
        # Run leakage probe if requested
        if args.leakage_probe:
            try:
                # Use smaller sample size to avoid OOM
                _cutils.leakage_probe(args.dataset, out_dir, sample=min(10_000, diag_sample if diag_sample else 10_000))
            except Exception as e:
                logger.warning(f"Leakage probe failed: {e}")
        
        # Run neighbour window diagnostics if requested
        if args.neigh_sample > 0:
            try:
                logger.info("Running neighbor window diagnostics...")
                neighbor_dir = out_dir / "neighbor_diagnostics"
                neighbor_dir.mkdir(exist_ok=True, parents=True)
                
                _cutils.neighbour_window_diagnostics(
                    dataset_path=args.dataset,
                    run_dir=out_dir,
                    annotations_path=None,  # Will use default annotations
                    n_sample=args.neigh_sample,
                    window=args.neigh_window,
                )
            except Exception as e:
                logger.warning(f"Neighbor window diagnostics failed: {e}")
                if args.verbose >= 2:
                    import traceback
                    traceback.print_exc()
    except Exception as e:
        logger.warning(f"Additional diagnostics failed with error: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
    
    # Generate comprehensive training summary
    cv_utils.generate_training_summary(
        out_dir=out_dir,
        args=args,
        original_dataset_path=original_dataset_path,
        fold_rows=fold_rows,
        script_name="run_loco_cv_multiclass_scalable.py"
    )
    
    logger.info("LOCO-CV analysis completed successfully")


def _create_combined_dmatrix(X_kmer, X_non_kmer, y=None):
    """Create an XGBoost DMatrix from sparse k-mer and dense non-k-mer features."""
    import xgboost as xgb
    
    # Convert non-k-mer dataframe to numpy array
    X_non_kmer_arr = X_non_kmer.values if hasattr(X_non_kmer, 'values') else X_non_kmer
    
    # Create combined feature matrix
    if X_non_kmer_arr.shape[1] > 0:
        # We have both k-mer and non-k-mer features
        X_combined = sparse.hstack([X_non_kmer_arr, X_kmer]).tocsr()
    else:
        # Only k-mer features
        X_combined = X_kmer
    
    # Create DMatrix
    if y is not None:
        dmatrix = xgb.DMatrix(X_combined, label=y)
    else:
        dmatrix = xgb.DMatrix(X_combined)
    
    return dmatrix


def analyze_unseen_positions_with_inference_workflow(
    model_path: Path,
    test_positions: Dict[str, List[Dict]],
    window_size: int = 50,
    output_dir: Optional[Path] = None,
    covered_pos: Optional[Dict[str, Set[int]]] = None,
    use_calibration: bool = True,
    verbosity: int = 1
) -> Dict[str, Any]:
    """
    Analyze unseen positions using the enhanced inference workflow.
    
    This function leverages the enhanced inference workflow to generate features for positions
    not seen during training, then performs prediction and neighborhood analysis.
    
    Parameters
    ----------
    model_path : Path
        Path to the saved model file
    test_positions : Dict[str, List[Dict]]
        Dictionary mapping chromosome/gene_id to list of positions with metadata
        Each position entry should have at least 'position' key
    window_size : int, default=50
        Size of window around each position for neighborhood analysis
    output_dir : Path, optional
        Directory for output files
    covered_pos : Dict[str, Set[int]], optional
        Positions already covered in the training set
    use_calibration : bool, default=True
        Whether to use model calibration if available
    verbosity : int, default=1
        Verbosity level of logging
    
    Returns
    -------
    Dict[str, Any]
        Results of unseen position analysis
    """
    from meta_spliceai.splice_engine.meta_models.workflows.inference_workflow_utils import analyze_unseen_neighborhood
    
    # Extract center positions from test_positions
    center_positions = {}
    for chrom, positions in test_positions.items():
        center_positions[chrom] = [p["position"] for p in positions]
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = Path(f"unseen_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Set default thresholds for ambiguous zone
    t_low = 0.02
    t_high = 0.80
    
    # Run analysis through the enhanced inference workflow
    results = analyze_unseen_neighborhood(
        model_path=model_path,
        center_positions=center_positions,
        window_size=window_size,
        output_dir=output_dir,
        use_calibration=use_calibration,
        covered_pos=covered_pos,
        t_low=t_low,
        t_high=t_high,
        verbosity=verbosity
    )
    
    # Add original position metadata if analysis succeeded
    if results.get("success", False):
        try:
            # Read results dataframe
            results_path = results.get("results_path")
            if results_path and Path(results_path).exists():
                results_df = pd.read_csv(results_path)
                
                # Create a mapping of position metadata for easier lookup
                position_meta = {}
                for chrom, positions in test_positions.items():
                    for pos_info in positions:
                        position_meta[(chrom, pos_info["position"])] = pos_info
                
                # Add metadata columns to results_df
                for meta_key in ["label", "type", "score", "category"]:
                    if any(meta_key in pos_info for chrom_pos in position_meta.values() for pos_info in [chrom_pos]):
                        results_df[f"meta_{meta_key}"] = results_df.apply(
                            lambda row: position_meta.get((row["id"], row["center_pos"]), {}).get(meta_key, None),
                            axis=1
                        )
                
                # Save enhanced results
                results_df.to_csv(results_path, index=False)
                results["enhanced_results"] = True
        except Exception as e:
            if verbosity >= 1:
                print(f"[warning] Failed to enhance results with metadata: {e}")
    
    return results


def _combine_sparse_and_dense(X_sparse, X_dense):
    """Combine sparse and dense features for prediction."""
    # For sparse matrices, convert dense to numpy array if needed
    X_dense_arr = X_dense.values if hasattr(X_dense, 'values') else X_dense
    
    # Handle case of empty dense or sparse matrix
    if X_dense_arr.shape[1] == 0:
        return X_sparse
    elif X_sparse.shape[1] == 0:
        return X_dense_arr
    
    # Combine sparse and dense matrices
    X_combined = sparse.hstack([X_sparse, X_dense_arr]).tocsr()
    return X_combined


def display_comprehensive_performance_summary(out_dir: Path, fold_rows: list, args, verbose: bool = True):
    """Display a comprehensive performance comparison table showing base vs meta improvements."""
    
    if not verbose:
        return
        
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE PERFORMANCE SUMMARY: BASE vs META MODEL (CHROMOSOME-AWARE CV)")
    print("="*80)
    
    # 1. CV Fold-level Summary
    if fold_rows:
        fold_df = pd.DataFrame(fold_rows)
        
        # Basic accuracy metrics
        if 'test_accuracy' in fold_df.columns:
            print("\n🎯 MULTICLASS CLASSIFICATION PERFORMANCE:")
            print("-" * 50)
            
            test_acc_mean = fold_df['test_accuracy'].mean()
            test_acc_std = fold_df['test_accuracy'].std()
            
            print(f"  Test Accuracy:     {test_acc_mean:.3f} ± {test_acc_std:.3f}")
            
            if 'test_macro_f1' in fold_df.columns:
                test_f1_mean = fold_df['test_macro_f1'].mean()
                test_f1_std = fold_df['test_macro_f1'].std()
                print(f"  Test Macro F1:     {test_f1_mean:.3f} ± {test_f1_std:.3f}")
            
            if 'splice_accuracy' in fold_df.columns:
                splice_acc_mean = fold_df['splice_accuracy'].mean()
                splice_acc_std = fold_df['splice_accuracy'].std()
                print(f"  Splice Accuracy:   {splice_acc_mean:.3f} ± {splice_acc_std:.3f}")
            
            if 'splice_macro_f1' in fold_df.columns:
                splice_f1_mean = fold_df['splice_macro_f1'].mean()
                splice_f1_std = fold_df['splice_macro_f1'].std()
                print(f"  Splice Macro F1:   {splice_f1_mean:.3f} ± {splice_f1_std:.3f}")
        
        # Per-fold chromosome information
        if 'fold_name' in fold_df.columns:
            print(f"\n🧬 CHROMOSOME-SPECIFIC PERFORMANCE:")
            print("-" * 50)
            
            # Group by chromosome and show performance
            for _, row in fold_df.iterrows():
                fold_name = row.get('fold_name', 'Unknown')
                test_acc = row.get('test_accuracy', 0)
                splice_acc = row.get('splice_accuracy', 0)
                print(f"  {fold_name:<12} Test Acc: {test_acc:.3f}, Splice Acc: {splice_acc:.3f}")
        
        # Top-k accuracy if available
        if 'top_k_accuracy' in fold_df.columns:
            print(f"\n🎯 TOP-K ACCURACY PERFORMANCE:")
            print("-" * 50)
            
            topk_mean = fold_df['top_k_accuracy'].mean()
            topk_std = fold_df['top_k_accuracy'].std()
            print(f"  Top-K Accuracy:    {topk_mean:.3f} ± {topk_std:.3f}")
    
    # 2. Try to load and display detailed comparison if available
    comparison_files = [
        out_dir / "perf_meta_vs_base.tsv",
        out_dir / "compare_base_meta.json",
        out_dir / "detailed_position_comparison.tsv"
    ]
    
    comparison_displayed = False
    
    # Check for base_vs_meta results
    base_meta_json = out_dir / "compare_base_meta.json"
    if base_meta_json.exists():
        try:
            with open(base_meta_json, 'r') as f:
                base_meta_stats = json.load(f)
            
            print(f"\n🧬 POSITION-LEVEL CLASSIFICATION PERFORMANCE:")
            print("-" * 50)
            print(f"  Base Model Accuracy: {base_meta_stats.get('base_accuracy', 0):.3f}")
            print(f"  Meta Model Accuracy: {base_meta_stats.get('meta_accuracy', 0):.3f}")
            print(f"  Accuracy Improvement: {base_meta_stats.get('meta_accuracy', 0) - base_meta_stats.get('base_accuracy', 0):+.3f}")
            print(f"  Base Macro F1: {base_meta_stats.get('base_macro_f1', 0):.3f}")
            print(f"  Meta Macro F1: {base_meta_stats.get('meta_macro_f1', 0):.3f}")
            print(f"  F1 Improvement: {base_meta_stats.get('meta_macro_f1', 0) - base_meta_stats.get('base_macro_f1', 0):+.3f}")
            
            # Error counts
            base_fp = base_meta_stats.get('base_fp', 0)
            meta_fp = base_meta_stats.get('meta_fp', 0)
            base_fn = base_meta_stats.get('base_fn', 0)
            meta_fn = base_meta_stats.get('meta_fn', 0)
            
            print(f"\n📈 POSITION-LEVEL ERROR REDUCTION:")
            print(f"  Base FP: {base_fp:,} → Meta FP: {meta_fp:,} (Δ: {base_fp - meta_fp:+,})")
            print(f"  Base FN: {base_fn:,} → Meta FN: {meta_fn:,} (Δ: {base_fn - meta_fn:+,})")
            print(f"  Net Error Reduction: {(base_fp - meta_fp) + (base_fn - meta_fn):+,} positions")
            
            comparison_displayed = True
            
        except Exception as e:
            logger.warning(f"Could not load base_vs_meta.json: {e}")
    
    # Check for splice performance comparison
    perf_comparison = out_dir / "perf_meta_vs_base.tsv"
    if perf_comparison.exists() and not comparison_displayed:
        try:
            comp_df = pd.read_csv(perf_comparison, sep='\t')
            
            if 'f1_score_delta' in comp_df.columns:
                print(f"\n🧬 GENE-LEVEL PERFORMANCE COMPARISON:")
                print("-" * 50)
                
                # Summary statistics
                mean_f1_delta = comp_df['f1_score_delta'].mean()
                genes_improved = (comp_df['f1_score_delta'] > 0).sum()
                genes_worsened = (comp_df['f1_score_delta'] < 0).sum()
                genes_unchanged = (comp_df['f1_score_delta'] == 0).sum()
                
                print(f"  Mean F1 improvement per gene: {mean_f1_delta:+.3f}")
                print(f"  Genes improved: {genes_improved}")
                print(f"  Genes worsened: {genes_worsened}")
                print(f"  Genes unchanged: {genes_unchanged}")
                
                if 'TP_delta' in comp_df.columns:
                    tp_delta_total = comp_df['TP_delta'].sum()
                    fp_delta_total = comp_df['FP_delta'].sum()
                    fn_delta_total = comp_df['FN_delta'].sum()
                    
                    print(f"  Total TP gained: {tp_delta_total:+,}")
                    print(f"  Total FP reduced: {fp_delta_total:+,}")
                    print(f"  Total FN reduced: {fn_delta_total:+,}")
                    
                comparison_displayed = True
                
        except Exception as e:
            logger.warning(f"Could not load perf_meta_vs_base.tsv: {e}")
    
    # 3. Generated artifacts summary
    print(f"\n📁 GENERATED ANALYSIS ARTIFACTS:")
    print("-" * 50)
    
    key_files = [
        ("loco_metrics.csv", "CV fold metrics"),
        ("final_model_calibrated_binary.pkl", "Calibrated meta-model (binary)"),
        ("final_model_calibrated_per_class.pkl", "Calibrated meta-model (per-class)"),
        ("final_model_uncalibrated.json", "Uncalibrated base model"),
        ("roc_pr_curves_meta.pdf", "ROC/PR curves"),
        ("cv_metrics_visualization/", "CV visualization suite"),
        ("shap_analysis/", "SHAP importance analysis"),
        ("feature_importance_analysis/", "Multi-method feature analysis"),
        ("compare_base_meta.json", "Position-level comparison"),
        ("perf_meta_vs_base.tsv", "Gene-level comparison"),
        ("probability_diagnostics.png", "Calibration analysis"),
        ("neighborhood_analysis.csv", "Neighborhood analysis"),
        ("overfitting_analysis/", "Overfitting monitoring"),
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
    print("🎉 CHROMOSOME-AWARE CV ANALYSIS COMPLETE! Check the artifacts above for detailed results.")
    print("="*80)


if __name__ == "__main__":  # pragma: no cover
    main()
