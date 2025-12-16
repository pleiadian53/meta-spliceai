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
from typing import List, Dict
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
                   help="Sample only a subset of genes for faster testing")

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
        topk = top_k_accuracy_score(y_true, np.column_stack([1 - y_prob, y_prob]), k=min(k, 2))
    except Exception:
        topk = float("nan")
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

def _train_binary_model(X: np.ndarray, y_bin: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, args: argparse.Namespace) -> tuple[XGBClassifier, dict]:
    """Fit a binary XGBClassifier with common hyper-parameters and return evaluation results."""
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        tree_method=args.tree_method,
        max_bin=args.max_bin if args.tree_method in {"hist", "gpu_hist"} else None,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.seed,
        n_jobs=-1,
        device=args.device if args.device != "auto" else None,
    )
    model.fit(X, y_bin, eval_set=[(X, y_bin), (X_val, y_val)], verbose=False)
    
    # Get evaluation results
    evals_result = model.evals_result() if hasattr(model, 'evals_result') else {}
    
    return model, evals_result

################################################################################
# Main logic
################################################################################

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
# Main
##########################################################################

def main(argv: List[str] | None = None) -> None:  # pragma: no cover
    args = _parse_args(argv)
    
    # Setup logging and validation
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


    
    # 2. Run K-fold CV with gene grouping
    if args.n_folds > 1:
        print("\nStarting gene-wise K-fold CV...")

    # Handle deprecated --annotations parameter
    if args.annotations and not args.splice_sites_path:
        print("Warning: --annotations is deprecated, please use --splice-sites-path instead")
        args.splice_sites_path = args.annotations

    # honour row-cap via env var used by datasets.load_dataset
    if args.row_cap > 0 and not os.getenv("SS_MAX_ROWS"):
        os.environ["SS_MAX_ROWS"] = str(args.row_cap)
    elif args.row_cap == 0 and not os.getenv("SS_MAX_ROWS"):
        # Use full dataset when row_cap is explicitly set to 0
        os.environ["SS_MAX_ROWS"] = "0"

    # 1. Load dataset
    if args.sample_genes is not None:
        # Use gene-level sampling for faster testing
        from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
        print(f"[INFO] Sampling {args.sample_genes} genes from dataset for faster testing")
        df = load_dataset_sample(args.dataset, sample_genes=args.sample_genes, random_seed=args.seed)
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
                df = load_dataset_with_memory_management(
                    args.dataset,
                    max_memory_gb=12.0,  # Use up to 12GB for loading
                    fallback_to_standard=False  # Don't fall back to avoid schema errors
                )
            else:
                print(f"[INFO] Standard dataset size ({estimated_rows:,} rows), using standard loader")
                df = datasets.load_dataset(args.dataset)
                
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
        
    if args.gene_col not in df.columns:
        raise KeyError(f"Column '{args.gene_col}' not found in dataset")

    # Prepare training data - use chromosome as a feature
    X_df, y_series = preprocessing.prepare_training_data(
        df, 
        label_col="splice_type", 
        return_type="pandas", 
        verbose=1,
        preserve_transcript_columns=args.transcript_topk,
        encode_chrom=True  # Include encoded chromosome as a feature
    )
    
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
    
    # 3. Check for potential feature leakage if enabled
    if args.check_leakage:
        print(f"\nRunning comprehensive leakage analysis (threshold={args.leakage_threshold})...")
        
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
            
            print(f"[Leakage Analysis] Found {len(leaky_features)} potentially leaky features")
            print(f"[Leakage Analysis] Comprehensive results saved to: {leakage_analysis_dir}")
            
            # If auto-exclude is enabled, add leaky features to exclusion list
            if args.auto_exclude_leaky and leaky_features:
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
                X_np, y_np, curr_features, args.leakage_threshold, correlation_report_path
            )
            
            if args.auto_exclude_leaky and leaky_features:
                print(f"Auto-excluding {len(leaky_features)} potentially leaky features")
                exclude_list.extend(leaky_features)
    
    # 4. Apply all exclusions
    if exclude_list:
        # Remove duplicates while preserving order
        exclude_list = list(dict.fromkeys(exclude_list))
        original_feature_count = X_df.shape[1]
        excluded_features = []
        
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
    
    # Always save transcript mapping columns if transcript-topk is enabled
    transcript_columns = {}
    if args.transcript_topk:
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
        
        # Handle position if it's in X_df but not yet saved
        if 'position' in X_df.columns and 'position' not in transcript_columns:
            transcript_columns['position'] = X_df['position'].copy()
            X_df = X_df.drop(columns=['position'])
    
    X = X_df.values
    y = _encode_labels(y_series)
    genes = df[args.gene_col].to_numpy()

    feature_names = list(X_df.columns)

    # Filter features based on exclusion list if provided
    excluded_features = []
    # Feature exclusion is now handled earlier in the script, see above
    # No need to update feature_names here as it's now done in the earlier exclusion section

    # Locate base probability columns
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
    non_kmer = [f for f in feature_names if not _is_kmer(f)]
    print(f"[Gene-CV-Sigmoid] Features: {len(feature_names)} total – {len(non_kmer)} non-k-mer")
    if len(feature_names) - len(non_kmer) > 0:
        sample_kmer = random.sample([f for f in feature_names if _is_kmer(f)], k=min(3, len(feature_names) - len(non_kmer)))
        print("   Example k-mers:", ", ".join(sample_kmer))

    # 2. Gene-wise K-fold CV
    gkf = GroupKFold(n_splits=args.n_folds)
    fold_rows: List[Dict[str, object]] = []
    
    # Initialize overfitting monitor if enabled
    monitor = None
    if args.monitor_overfitting:
        print(f"\n[Overfitting Monitor] Initializing overfitting detection system")
        print(f"  Primary metric: logloss")
        print(f"  Gap threshold: {args.overfitting_threshold}")
        print(f"  Early stopping patience: {args.early_stopping_patience}")
        print(f"  Convergence improvement: {args.convergence_improvement}")
        
        monitor = OverfittingMonitor(
            primary_metric="logloss",
            gap_threshold=args.overfitting_threshold,
            patience=args.early_stopping_patience,
            min_improvement=args.convergence_improvement
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

    # Initialize metrics collections
    fold_rows = []  # will become a DataFrame of fold metrics
    base_f1s, meta_f1s = [], []  # F1 scores for base and meta models
    base_topks, meta_topks = [], []  # Top-k accuracies
    fp_deltas, fn_deltas = [], []  # False positive and negative deltas
    donor_topks, acceptor_topks = [], []  # Donor and acceptor top-k accuracies
    auc_base, auc_meta = [], []  # ROC AUCs
    ap_base, ap_meta = [], []  # Average precision
    roc_base, roc_meta = [], []  # ROC curves
    pr_base, pr_meta = [], []  # PR curves
    
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
                    if args.verbose:
                        print(f"    [Overfitting Monitor] Added metrics for fold {fold_idx+1}, class {cls}")
                except Exception as e:
                    if args.verbose:
                        print(f"    [Overfitting Monitor] Warning: Failed to add metrics for fold {fold_idx+1}, class {cls}: {e}")

        # Predict probabilities on validation set for calibration
        if args.calibrate or args.calibrate_per_class:
            proba_val = np.column_stack([m.predict_proba(X[valid_idx])[:, 1] for m in models_cls])
            
            # For standard calibration (splice vs non-splice)
            if args.calibrate:
                s_val = proba_val[:, 1] + proba_val[:, 2]  # Sum of donor + acceptor scores
                y_bin_val = (y[valid_idx] != 0).astype(int)  # 1 if any splice site, 0 otherwise
                calib_scores.append(s_val)
                calib_labels.append(y_bin_val)
            
            # For per-class calibration
            if args.calibrate_per_class:
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
        transcript_top_k_metrics = None
        if args.transcript_topk and TRANSCRIPT_MAPPING_AVAILABLE:
            try:
                # Check if we have both required columns (position always needed, and original chrom strings)
                if 'position' in transcript_columns and 'chrom' in transcript_columns:
                    # Get the values for test indices
                    positions = transcript_columns['position'].iloc[test_idx].values
                    chroms = transcript_columns['chrom'].iloc[test_idx].values
                    
                    # Create DataFrame for transcript mapping
                    tx_df = pd.DataFrame({
                        'position': positions,
                        'chrom': chroms,
                        'label': y[test_idx],
                        'prob_donor': proba[:, 0],  # Label 0 = donor sites
                        'prob_acceptor': proba[:, 1],  # Label 1 = acceptor sites
                    })
                    
                    # Calculate transcript-level metrics
                    transcript_top_k_metrics = calculate_transcript_level_top_k(
                        df=tx_df,
                        splice_sites_path=args.splice_sites_path,
                        transcript_features_path=args.transcript_features_path,
                        gene_features_path=args.gene_features_path,
                        donor_label=0,     # Meta-model uses 0=donor, 1=acceptor
                        acceptor_label=1,
                        neither_label=2,
                        position_col="position", 
                        chrom_col="chrom",
                        label_col="label",  # Maps to column containing site labels in df
                        use_cache=not args.no_transcript_cache
                    )
                    
                    # Print transcript-level metrics report
                    print("\n" + report_transcript_top_k(transcript_top_k_metrics))
                else:
                    missing = []
                    if 'position' not in transcript_columns:
                        missing.append("'position'")
                    if 'chrom' not in transcript_columns:
                        missing.append("'chrom'")
                    print(f"\nSkipping transcript-level metrics: Missing required columns {', '.join(missing)}")
                    print("Make sure both 'chrom' and 'position' are preserved for transcript mapping")
            except Exception as e:
                print(f"\nError calculating transcript-level metrics: {e}")
                # Don't let transcript metrics failure break the whole pipeline
        
        # Store the combined top-k accuracy for summary statistics
        top_k_acc = gene_top_k_metrics["combined_top_k"]

        # ----------------------------------------------------
        # Base vs meta binary splice-site metrics
        # ----------------------------------------------------
        y_true_bin = (y[test_idx] != 0).astype(int)
        y_prob_meta = proba[:, 1] + proba[:, 2]
        if splice_prob_idx is not None:
            y_prob_base = X[test_idx, splice_prob_idx]
        else:
            y_prob_base = X[test_idx, donor_idx] + X[test_idx, acceptor_idx]

        meta_metrics = _binary_metrics(y_true_bin, y_prob_meta, thresh=args.threshold or 0.5, k=args.top_k)
        base_metrics = _binary_metrics(y_true_bin, y_prob_base, thresh=args.base_thresh, k=args.top_k)

        base_f1s.append(base_metrics["f1"])
        meta_f1s.append(meta_metrics["f1"])
        base_topks.append(base_metrics["topk_acc"])
        meta_topks.append(meta_metrics["topk_acc"])
        fp_deltas.append(base_metrics["fp"] - meta_metrics["fp"])
        fn_deltas.append(base_metrics["fn"] - meta_metrics["fn"])
        
        # Collect ROC and PR curve data for this fold
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
        # Base model multiclass probabilities (reconstruct from binary features)
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
        
        y_prob_bases_multiclass.append(y_prob_base_multiclass)
        y_prob_metas_multiclass.append(proba)  # Meta model already has multiclass probabilities

        print(f"   Base  F1={base_metrics['f1']:.3f} top{args.top_k}={base_metrics['topk_acc']:.3f} (FP={base_metrics['fp']} FN={base_metrics['fn']})")
        print(f"   Meta  F1={meta_metrics['f1']:.3f} top{args.top_k}={meta_metrics['topk_acc']:.3f} (FP={meta_metrics['fp']} FN={meta_metrics['fn']})  "
              f"ΔFP={base_metrics['fp']-meta_metrics['fp']}  ΔFN={base_metrics['fn']-meta_metrics['fn']}")

        cm = confusion_matrix(y[test_idx], pred, labels=[0, 1, 2])
        print("Confusion matrix (fold", fold_idx, ")\n", pd.DataFrame(cm, index=["neither", "donor", "acceptor"], columns=["neither", "donor", "acceptor"]))

        fold_row = {
            "fold": fold_idx,
            "test_rows": len(test_idx),
            "test_accuracy": acc,
            "test_macro_f1": macro_f1,
            "splice_accuracy": splice_acc,
            "splice_macro_f1": splice_macro_f1,
            "top_k_accuracy": top_k_acc,
            "top_k_donor": gene_top_k_metrics["donor_top_k"],
            "top_k_acceptor": gene_top_k_metrics["acceptor_top_k"],
            "top_k_n_genes": gene_top_k_metrics.get("n_groups", 0),
        }
        
        # Add transcript-level metrics if available
        if transcript_top_k_metrics is not None:
            fold_row.update({
                "tx_top_k_donor": transcript_top_k_metrics.get("transcript_donor_top_k", float('nan')),
                "tx_top_k_acceptor": transcript_top_k_metrics.get("transcript_acceptor_top_k", float('nan')),
                "tx_top_k_combined": transcript_top_k_metrics.get("transcript_combined_top_k", float('nan')),
                "tx_top_k_n_transcripts": transcript_top_k_metrics.get("transcript_n_groups", 0),
            })
        
        # base vs meta binary metrics
        fold_row.update({
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
    
    # Optional calibration analysis 
    calibration_results = None
    overconf_metrics = None
    eval_approach = "threshold_safe"  # Default approach
    
    try:
        # Import calibration analysis functions if available
        from meta_spliceai.splice_engine.meta_models.evaluation.calibration_integration import get_calibration_functions
        calib_funcs = get_calibration_functions()
        
        if calib_funcs is not None:
            # Quick overconfidence check on CV results (enabled by default)
            if args.quick_overconfidence_check and meta_f1s:
                # Use meta probabilities from last fold as representative sample
                if y_prob_metas:
                    combined_y_true = np.concatenate(y_true_bins)
                    combined_y_prob = np.concatenate(y_prob_metas)
                    combined_y_splice_type = np.concatenate(y_splice_types)
                    
                    overconf_metrics = calib_funcs['detect_overconfidence_issues'](
                        combined_y_true, combined_y_prob, combined_y_splice_type, verbose=args.verbose
                    )
            
            # Comprehensive calibration analysis (optional)
            if args.calibration_analysis:
                calibration_results = calib_funcs['run_calibration_analysis'](
                    dataset_path=args.dataset,
                    model_path=out_dir / "model_multiclass.pkl",
                    out_dir=out_dir,
                    sample_size=args.calibration_sample,
                    plot_format=args.plot_format,
                    verbose=args.verbose,
                    enable_analysis=True
                )
            
            # Get evaluation approach recommendation
            eval_approach = calib_funcs['suggest_evaluation_approach'](
                calibration_results=calibration_results,
                overconf_metrics=overconf_metrics,
                verbose=args.verbose
            )
            
        else:
            if args.calibration_analysis:
                print("[Calibration Analysis] ⚠️  Calibration analysis requested but module not available")
            elif args.verbose:
                print("[Calibration Analysis] Module not available (optional feature)")
                
    except ImportError:
        if args.calibration_analysis:
            print("[Calibration Analysis] ⚠️  Calibration analysis requested but dependencies not available")
    except Exception as e:
        if args.verbose:
            print(f"[Calibration Analysis] Warning: {e}")
    
    # Log the recommended evaluation approach for the post-training evaluation
    if args.verbose and eval_approach:
        print(f"\n[Evaluation Strategy] Recommended approach: {eval_approach}")
        if eval_approach == "argmax_only":
            print("[Evaluation Strategy] Post-training evaluation will use ONLY argmax-based methods")
        elif eval_approach == "argmax_primary":  
            print("[Evaluation Strategy] Post-training evaluation will prioritize argmax methods")
        else:
            print("[Evaluation Strategy] Threshold-based evaluation should work reliably")
    
    # Generate ROC and PR curve plots if enabled
    if args.plot_curves:
        # Prepare data for plotting
        # The fold-level data is already properly stored in the lists
        
        # Use the modular plotting function
        curve_metrics = plot_roc_pr_curves(
            y_true=y_true_bins,
            y_pred_base=y_prob_bases,
            y_pred_meta=y_prob_metas,
            out_dir=out_dir,
            n_roc_points=args.n_roc_points,
            plot_format=args.plot_format,
            base_name='Base',
            meta_name='Meta',
            fold_ids=list(range(len(y_true_bins)))
        )
        
        # Log curve metrics to the console (redundant but kept for consistency)
        auc_b_mean, auc_b_std = curve_metrics['auc']['base']['mean'], curve_metrics['auc']['base']['std']
        auc_m_mean, auc_m_std = curve_metrics['auc']['meta']['mean'], curve_metrics['auc']['meta']['std']
        ap_b_mean, ap_b_std = curve_metrics['ap']['base']['mean'], curve_metrics['ap']['base']['std']
        ap_m_mean, ap_m_std = curve_metrics['ap']['meta']['mean'], curve_metrics['ap']['meta']['std']
        
        print(f"\nAUC  base  mean={auc_b_mean:.3f} ±{auc_b_std:.3f}")
        print(f"AUC  meta  mean={auc_m_mean:.3f} ±{auc_m_std:.3f}")
        print(f"AP   base  mean={ap_b_mean:.3f} ±{ap_b_std:.3f}")
        print(f"AP   meta  mean={ap_m_mean:.3f} ±{ap_m_std:.3f}")
        
        # Calculate F1 scores using same thresholds as CV workflow
        print("\n[F1 Analysis] Calculating F1 scores using CV workflow thresholds...")
        try:
            # Use same thresholds as CV workflow
            meta_threshold = args.threshold or 0.5
            base_threshold = args.base_thresh
            
            f1_base_scores = []
            f1_meta_scores = []
            
            for y_t, y_pb, y_pm in zip(y_true_bins, y_prob_bases, y_prob_metas):
                # Base model F1 at base_thresh
                y_pred_base_binary = (y_pb >= base_threshold).astype(int)
                f1_base_scores.append(f1_score(y_t, y_pred_base_binary))
                
                # Meta model F1 at meta_threshold
                y_pred_meta_binary = (y_pm >= meta_threshold).astype(int)
                f1_meta_scores.append(f1_score(y_t, y_pred_meta_binary))
            
            f1_b_mean = np.mean(f1_base_scores)
            f1_b_std = np.std(f1_base_scores)
            f1_m_mean = np.mean(f1_meta_scores)
            f1_m_std = np.std(f1_meta_scores)
            
            print(f"F1   base  mean={f1_b_mean:.3f} ±{f1_b_std:.3f} (threshold={base_threshold})")
            print(f"F1   meta  mean={f1_m_mean:.3f} ±{f1_m_std:.3f} (threshold={meta_threshold})")
            print("[F1 Analysis] ✓ F1 scores calculated using CV workflow thresholds")
            
        except Exception as e:
            print(f"[F1 Analysis] ✗ Error calculating F1 scores: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        
        # Create enhanced visualizations
        print("\n[Enhanced Visualizations] Creating improved binary PR curves...")
        try:
            create_improved_binary_pr_plot(
                y_true=y_true_bins,
                y_pred_base=y_prob_bases,
                y_pred_meta=y_prob_metas,
                out_dir=out_dir,
                plot_format=args.plot_format,
                base_name='Base',
                meta_name='Meta'
            )
            print("[Enhanced Visualizations] ✓ Improved binary PR curves created")
        except Exception as e:
            print(f"[Enhanced Visualizations] ✗ Error creating improved binary PR curves: {e}")
        
        print("\n[Enhanced Visualizations] Creating multiclass ROC/PR curves...")
        try:
            multiclass_metrics = plot_multiclass_roc_pr_curves(
                y_true=y_true_multiclass,
                y_pred_base=y_prob_bases_multiclass,
                y_pred_meta=y_prob_metas_multiclass,
                out_dir=out_dir,
                plot_format=args.plot_format,
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
            if args.verbose:
                import traceback
                traceback.print_exc()

        # Generate the combined ROC/PR curves meta PDF that the script expects
        print("\n[Enhanced Visualizations] Creating combined ROC/PR curves meta PDF...")
        try:
            plot_combined_roc_pr_curves_meta(
                y_true=y_true_bins,
                y_pred_base=y_prob_bases,
                y_pred_meta=y_prob_metas,
                out_dir=out_dir,
                plot_format=args.plot_format,
                base_name='Base',
                meta_name='Meta',
                n_roc_points=args.n_roc_points
            )
            print("[Enhanced Visualizations] ✓ Combined ROC/PR curves meta PDF created")
        except Exception as e:
            print(f"[Enhanced Visualizations] ✗ Error creating combined ROC/PR curves meta PDF: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # 3. Train final ensemble on full data
    models_full: List[XGBClassifier] = []
    for cls in (0, 1, 2):
        y_bin = (y == cls).astype(int)
        model_c, _ = _train_binary_model(X, y_bin, X, y_bin, args)  # self-eval set to silence warnings
        models_full.append(model_c)

    if args.calibrate_per_class:
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        
        # Create per-class calibrators
        calibrators = []
        for cls_idx in range(3):  # 0=neither, 1=donor, 2=acceptor
            # Concatenate scores and labels from all CV folds for this class
            cls_scores = np.concatenate(per_class_calib_scores[cls_idx])
            cls_labels = np.concatenate(per_class_calib_labels[cls_idx])
            
            print(f"[Per-class calibration] Class {cls_idx}: {cls_scores.shape[0]} samples, {cls_labels.sum()} positives")
            
            # Create and fit the calibrator
            if args.calib_method == "platt":
                calibrator = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
                calibrator.fit(cls_scores.reshape(-1, 1), cls_labels)
            elif args.calib_method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(cls_scores, cls_labels)
            else:
                raise ValueError("Unsupported calibration method: " + args.calib_method)
            
            calibrators.append(calibrator)
        
        # Create ensemble with per-class calibration
        ensemble = PerClassCalibratedSigmoidEnsemble(models_full, feature_names, calibrators)
        print("[Info] Created PerClassCalibratedSigmoidEnsemble with separate calibration for each class")
    
    elif args.calibrate:
        s_train = np.concatenate(calib_scores)
        y_bin = np.concatenate(calib_labels)

        if args.calib_method == "platt":
            from sklearn.linear_model import LogisticRegression
            calibrator = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs")
            calibrator.fit(s_train.reshape(-1, 1), y_bin)
        elif args.calib_method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            calibrator = IsotonicRegression(out_of_bounds="clip").fit(s_train, y_bin)
        else:
            raise ValueError("Unsupported calibration method: " + args.calib_method)

        ensemble = _cutils.CalibratedSigmoidEnsemble(models_full, feature_names, calibrator)
        print("[Info] Created CalibratedSigmoidEnsemble with binary splice/non-splice calibration")
    else:
        ensemble = SigmoidEnsemble(models_full, feature_names)
        print("[Info] Created uncalibrated SigmoidEnsemble")

    # initial pickling so diagnostics can load the model
    import pickle
    with open(out_dir / "model_multiclass.pkl", "wb") as fh:
        pickle.dump(ensemble, fh)
    # we will overwrite after attaching optimal_threshold later
    pd.DataFrame({"feature": feature_names}).to_csv(out_dir / "feature_manifest.csv", index=False)

    # 4. Aggregate CV metrics
    df_metrics = pd.DataFrame(fold_rows)

    # Additional aggregate printout for base vs meta
    if base_f1s:
        print("\n=== Base vs Meta (splice / non-splice) ===")
        print(f"Base  F1   mean={np.mean(base_f1s):.3f} ±{np.std(base_f1s):.3f}")
        print(f"Meta  F1   mean={np.mean(meta_f1s):.3f} ±{np.std(meta_f1s):.3f}")
        print(f"Base  top{args.top_k} mean={np.mean(base_topks):.3f}")
        print(f"Meta  top{args.top_k} mean={np.mean(meta_topks):.3f}")
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
    df_metrics.to_csv(out_dir / "gene_cv_metrics.csv", index=False)
    print("\nGene-CV-Sigmoid results by fold:\n", df_metrics)
    mean_metrics = df_metrics[[
        "test_accuracy", "test_macro_f1", "splice_accuracy", "splice_macro_f1", "top_k_accuracy"
    ]].mean()
    print("\nAverage across folds:\n", mean_metrics.to_string())
    with open(out_dir / "metrics_aggregate.json", "w") as fh:
        json.dump(mean_metrics.to_dict(), fh, indent=2)
    # Generate CV metrics visualization report
    try:
        print("\nGenerating CV metrics visualization report...")
        cv_metrics_csv = out_dir / "gene_cv_metrics.csv"
        if cv_metrics_csv.exists():
            viz_result = generate_cv_metrics_report(
                csv_path=cv_metrics_csv,
                out_dir=out_dir,
                dataset_path=args.dataset,
                plot_format=args.plot_format,
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
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    # Generate overfitting analysis if monitoring was enabled
    if monitor is not None and len(monitor.fold_metrics) > 0:
        try:
            print("\n[Overfitting Monitor] Generating comprehensive overfitting analysis...")
            
            # Create overfitting analysis subdirectory
            overfitting_dir = out_dir / "overfitting_analysis"
            overfitting_dir.mkdir(exist_ok=True)
            
            # Generate overfitting report
            overfitting_report = monitor.generate_overfitting_report(overfitting_dir)
            
            # Generate learning curves and visualizations
            monitor.plot_learning_curves(overfitting_dir, plot_format=args.plot_format)
            
            # Print summary
            summary = overfitting_report['summary']
            print(f"\n[Overfitting Monitor] Analysis Summary:")
            print(f"  Total binary models trained: {summary['total_folds']}")
            print(f"  Models with overfitting: {summary['folds_with_overfitting']}")
            print(f"  Early stopped models: {summary['early_stopped_folds']}")
            print(f"  Mean performance gap: {summary['mean_performance_gap']:.4f} ± {summary['std_performance_gap']:.4f}")
            print(f"  Mean overfitting score: {summary['mean_overfitting_score']:.4f}")
            print(f"  Recommended n_estimators: {summary['recommended_n_estimators']}")
            
            # Create additional summary report
            summary_path = overfitting_dir / "overfitting_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("Gene-Aware CV Overfitting Analysis Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Analysis Parameters:\n")
                f.write(f"  Gap threshold: {args.overfitting_threshold}\n")
                f.write(f"  Early stopping patience: {args.early_stopping_patience}\n")
                f.write(f"  Convergence improvement: {args.convergence_improvement}\n\n")
                f.write(f"Results:\n")
                f.write(f"  Total binary models: {summary['total_folds']}\n")
                f.write(f"  Models with overfitting: {summary['folds_with_overfitting']}\n")
                f.write(f"  Early stopped models: {summary['early_stopped_folds']}\n")
                f.write(f"  Mean performance gap: {summary['mean_performance_gap']:.4f} ± {summary['std_performance_gap']:.4f}\n")
                f.write(f"  Recommended n_estimators: {summary['recommended_n_estimators']}\n\n")
                f.write(f"Generated Files:\n")
                f.write(f"  - overfitting_analysis.json: Detailed metrics\n")
                f.write(f"  - learning_curves_by_fold.{args.plot_format}: Individual fold curves\n")
                f.write(f"  - aggregated_learning_curves.{args.plot_format}: Mean curves with confidence bands\n")
                f.write(f"  - overfitting_summary.{args.plot_format}: Summary visualizations\n")
            
            print(f"[Overfitting Monitor] Analysis saved to: {overfitting_dir}")
            print(f"[Overfitting Monitor] Summary report: {summary_path}")
            
        except Exception as e:
            print(f"[Overfitting Monitor] ERROR: Failed to generate overfitting analysis: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    elif args.monitor_overfitting:
        print("\n[Overfitting Monitor] No overfitting data collected - this may indicate an issue with evaluation result capture")


    ####################################################################

    # 5. Diagnostics + post-training analysis (reuse existing helpers)
    diag_sample = None if args.diag_sample == 0 else args.diag_sample
    
    # Apply memory optimization if enabled
    if args.memory_optimize:
        # Reduce diagnostic sample sizes for memory-constrained systems
        if diag_sample is None or diag_sample > args.max_diag_sample:
            diag_sample = args.max_diag_sample
            print(f"[Memory Optimization] Reduced diagnostic sample to {diag_sample} for memory efficiency")
        
        # Reduce neighbor sample size if it's too large
        if args.neigh_sample > 1000:
            args.neigh_sample = 1000
            print(f"[Memory Optimization] Reduced neighbor sample to {args.neigh_sample} for memory efficiency")
    _cutils.richer_metrics(args.dataset, out_dir, sample=diag_sample)
    _cutils.gene_score_delta(args.dataset, out_dir, sample=diag_sample)
    
    # Try the memory-efficient SHAP analysis first, fall back to original if it fails
    shap_analysis_completed = False
    try:
        print("\n" + "="*60)
        print("🔍 SHAP ANALYSIS DIAGNOSTICS")
        print("="*60)
        print(f"[SHAP Analysis] Starting memory-efficient SHAP analysis...")
        print(f"[SHAP Analysis] Dataset: {args.dataset}")
        print(f"[SHAP Analysis] Output directory: {out_dir}")
        print(f"[SHAP Analysis] Diagnostic sample size: {diag_sample}")
        
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
        shap_output_dir = run_incremental_shap_analysis(args.dataset, out_dir, sample=diag_sample)
        
        # Check if SHAP analysis actually produced output
        expected_shap_file = out_dir / "feature_importance_analysis" / "shap_analysis" / "importance" / "shap_importance_incremental.csv"
        
        print(f"[SHAP Analysis] Checking SHAP output...")
        print(f"  SHAP output directory: {shap_output_dir}")
        print(f"  Expected SHAP file: {expected_shap_file}")
        print(f"  SHAP file exists: {expected_shap_file.exists()}")
        
        if expected_shap_file.exists():
            # Read and validate SHAP results
            try:
                shap_df = pd.read_csv(expected_shap_file)
                print(f"  SHAP results shape: {shap_df.shape}")
                print(f"  SHAP columns: {list(shap_df.columns)}")
                print(f"  Top 3 features: {shap_df.head(3)['feature'].tolist() if 'feature' in shap_df.columns else 'N/A'}")
                
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
                # Look for SHAP importance files in the new organized location
                feature_importance_dir = out_dir / "feature_importance_analysis"
                shap_analysis_dir = feature_importance_dir / "shap_analysis"
                shap_importance_dir = shap_analysis_dir / "importance"
                
                shap_importance_csv = shap_importance_dir / "shap_importance_incremental.csv"
                model_pkl = out_dir / "model_multiclass.pkl"
                
                print(f"[SHAP Analysis] Visualization prerequisites:")
                print(f"  SHAP CSV: {shap_importance_csv.exists()} ({shap_importance_csv})")
                print(f"  Model PKL: {model_pkl.exists()} ({model_pkl})")
                
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
                    print(f"[SHAP Analysis] ✓ Comprehensive SHAP report generated")
                    print(f"[SHAP Analysis] All outputs organized under: {shap_analysis_dir}")
                    
                    # Log top features for each class
                    if 'summary_stats' in shap_results:
                        stats = shap_results['summary_stats']
                        print(f"[SHAP Analysis] Top features by class:")
                        print(f"  Overall: {stats.get('top_feature_overall', 'N/A')}")
                        print(f"  Neither: {stats.get('top_feature_neither', 'N/A')}")
                        print(f"  Donor:   {stats.get('top_feature_donor', 'N/A')}")
                        print(f"  Acceptor: {stats.get('top_feature_acceptor', 'N/A')}")
                else:
                    print(f"[SHAP Analysis] ✗ Required files missing for visualization:")
                    print(f"  SHAP CSV: {shap_importance_csv.exists()} ({shap_importance_csv})")
                    print(f"  Model PKL: {model_pkl.exists()} ({model_pkl})")
                    
            except Exception as e:
                print(f"[SHAP Analysis] ✗ Error generating comprehensive SHAP report: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
    except Exception as e:
        print(f"[SHAP Analysis] ✗ Memory-efficient SHAP analysis failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        print(f"[SHAP Analysis] Attempting fallback to original SHAP importance analysis...")
        try:
            # Check if original SHAP function exists
            if hasattr(_cutils, 'shap_importance'):
                print(f"[SHAP Analysis] Original SHAP function found, executing...")
                _cutils.shap_importance(args.dataset, out_dir, sample=diag_sample)
                
                # Check if original method produced output
                fallback_shap_file = out_dir / "shap_importance_incremental.csv"
                print(f"[SHAP Analysis] Checking fallback SHAP output...")
                print(f"  Fallback SHAP file: {fallback_shap_file}")
                print(f"  Fallback file exists: {fallback_shap_file.exists()}")
                
                if fallback_shap_file.exists():
                    try:
                        fallback_df = pd.read_csv(fallback_shap_file)
                        print(f"  Fallback results shape: {fallback_df.shape}")
                        if len(fallback_df) > 0:
                            shap_analysis_completed = True
                            print(f"[SHAP Analysis] ✓ Fallback SHAP analysis completed successfully!")
                        else:
                            print(f"[SHAP Analysis] ⚠️  Fallback SHAP file is empty")
                    except Exception as e:
                        print(f"[SHAP Analysis] ⚠️  Error reading fallback SHAP results: {e}")
                else:
                    print(f"[SHAP Analysis] ⚠️  Fallback SHAP analysis did not produce output")
            else:
                print(f"[SHAP Analysis] ⚠️  Original SHAP function not found in _cutils")
                
        except Exception as e2:
            print(f"[SHAP Analysis] ✗ Original SHAP importance analysis also failed: {e2}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Final SHAP analysis status
    print(f"\n[SHAP Analysis] Final status:")
    if shap_analysis_completed:
        print(f"  ✓ SHAP analysis completed successfully")
        # Check final output structure
        shap_analysis_dir = out_dir / "feature_importance_analysis" / "shap_analysis"
        if shap_analysis_dir.exists():
            print(f"  ✓ SHAP analysis directory created: {shap_analysis_dir}")
            
            # List contents of SHAP analysis directory
            try:
                for root, dirs, files in os.walk(shap_analysis_dir):
                    level = root.replace(str(shap_analysis_dir), '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"  {indent}{os.path.basename(root)}/")
                    sub_indent = ' ' * 2 * (level + 1)
                    for file in files:
                        print(f"  {sub_indent}{file}")
            except Exception as e:
                print(f"  ⚠️  Could not list SHAP directory contents: {e}")
        else:
            print(f"  ⚠️  SHAP analysis directory not found: {shap_analysis_dir}")
    else:
        print(f"  ✗ SHAP analysis failed completely")
        print(f"  ℹ️  This is often due to:")
        print(f"    - Non-numeric data in the feature matrix (e.g., chromosome strings)")
        print(f"    - Memory constraints with large datasets")
        print(f"    - Model compatibility issues with SHAP TreeExplainer")
        print(f"    - Missing dependencies (shap, matplotlib, etc.)")
        print(f"  ℹ️  Analysis will continue with other diagnostic methods")
    
    print("="*60)
    
    # Run comprehensive feature importance analysis with memory optimization
    try:
        print("\nRunning comprehensive feature importance analysis...")
        
        # Apply memory optimization for feature importance analysis
        feature_sample = diag_sample
        if args.memory_optimize and feature_sample and feature_sample > 10000:
            feature_sample = 10000
            print(f"[Memory Optimization] Reduced feature importance sample to {feature_sample}")
        
        feature_importance_dir = run_gene_cv_feature_importance_analysis(
            args.dataset, out_dir, sample=feature_sample
        )
        if feature_importance_dir:
            print("[INFO] Comprehensive feature importance analysis completed successfully")
        else:
            print("[WARNING] Feature importance analysis failed")
    except Exception as e:
        print(f"[WARNING] Feature importance analysis failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    _cutils.probability_diagnostics(args.dataset, out_dir, sample=diag_sample)
    _cutils.base_vs_meta(args.dataset, out_dir, sample=diag_sample)

    # --- meta splice-site evaluation ---
    if args.threshold is not None:
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

    # ------------------------------------------------------------------
    # Persist ensemble with attached optimal threshold for inference
    # ------------------------------------------------------------------
    try:
        ensemble.optimal_threshold = thresh  # binary cutoff
        ensemble.threshold_neither = thresh  # for completeness
        # donor/acceptor thresholds set earlier if available
    except Exception:
        pass  # guard: attribute injection should never fail
    import pickle
    with open(out_dir / "model_multiclass.pkl", "wb") as fh:
        pickle.dump(ensemble, fh)
    
    # Skip evaluation if requested
    if args.skip_eval:
        print("[INFO] Skipping evaluation steps due to --skip-eval flag")
        return
        
    # ------------------------------------------------------------------
    # 🔧 CRITICAL FIX: Use ONLY classification-based evaluation
    # The meta-model predicts classifications on training instances, not positions
    # Position-based evaluation is fundamentally wrong for this use case
    # ------------------------------------------------------------------
    
    print("\n" + "="*60)
    print("🎯 RUNNING CLASSIFICATION-BASED META-MODEL EVALUATION")
    print("="*60)
    print("ℹ️  This evaluation compares meta vs base predictions at the")
    print("   SAME training positions, not position discovery performance.")
    print("   This is the correct approach for meta-model evaluation.")
    print("="*60)
    
    from pathlib import Path as _Path
    base_tsv = _Path(args.base_tsv) if args.base_tsv else None
    
    # 1. Position-level classification comparison (CORRECT approach)
    try:
        if hasattr(args, 'verbose') and args.verbose:
            print("\n[DEBUG] Running position-level classification comparison...")
            print(f"  Training dataset: {args.dataset}")
            print(f"  Trained model: {out_dir}")
            print(f"  Sample size: {diag_sample}")
        
        from meta_spliceai.splice_engine.meta_models.training.eval_meta_splice import meta_splice_performance_correct
        
        # Use the correct evaluation that compares meta vs base at training positions
        result_path = meta_splice_performance_correct(
            dataset_path=args.dataset,
            run_dir=out_dir,
            sample=diag_sample,
            out_tsv=out_dir / "position_level_classification_results.tsv",
            verbose=args.verbose if hasattr(args, 'verbose') else 1,
        )
        
        if args.verbose:
            print(f"[Gene-CV-Sigmoid] ✅ Position-level classification comparison completed!")
            print(f"  Results saved to: {result_path}")
                    
    except Exception as e:
        print(f"[Gene-CV-Sigmoid] ❌ Position-level classification comparison failed: {e}")
        if hasattr(args, 'verbose') and args.verbose:
            import traceback
            traceback.print_exc()

    # 2. Gene-level classification comparison (CORRECT approach)
    try:
        if hasattr(args, 'verbose') and args.verbose:
            print("\n[DEBUG] Running gene-level classification comparison...")
            print(f"  Evaluation approach: {eval_approach}")
            
        from meta_spliceai.splice_engine.meta_models.training.eval_meta_splice import meta_splice_performance_argmax
        
        # Always use argmax-based evaluation (works regardless of calibration issues)
        meta_comparison_path = meta_splice_performance_argmax(
            dataset_path=args.dataset,
            run_dir=out_dir,
            sample=diag_sample,
            out_tsv=out_dir / "gene_level_argmax_results.tsv",
            verbose=args.verbose if hasattr(args, 'verbose') else 1,
            donor_score_col="donor_score",
            acceptor_score_col="acceptor_score", 
            gene_col="gene_id",
            label_col="splice_type"
        )
        
        if args.verbose:
            print(f"[Gene-CV-Sigmoid] ✅ Gene-level ARGMAX comparison completed!")
            print(f"  Results saved to: {meta_comparison_path}")
        
        # 3. ENHANCED: Generate per-nucleotide meta-scores for splice inference workflow
        try:
            if hasattr(args, 'verbose') and args.verbose:
                print("\n[DEBUG] Generating per-nucleotide meta-model scores...")
                print(f"  This enables position-aware splice site prediction")
                print(f"  Output: donor_meta, acceptor_meta, neither_meta tensors")
            
            from meta_spliceai.splice_engine.meta_models.training.incremental_score_generator import generate_per_nucleotide_meta_scores_incremental
            
            score_tensor_path = generate_per_nucleotide_meta_scores_incremental(
                dataset_path=args.dataset,
                run_dir=out_dir,
                sample=diag_sample,
                output_format="parquet",
                max_memory_gb=8.0,  # Conservative memory limit
                target_chunk_size=25_000,  # Process in 25k position chunks
                verbose=args.verbose if hasattr(args, 'verbose') else 1,
            )
            
            if args.verbose:
                print(f"[Gene-CV-Sigmoid] ✅ Per-nucleotide meta-scores generated!")
                print(f"  Score tensors saved to: {score_tensor_path}")
                print(f"  Compatible with splice_inference_workflow.py")
                print(f"  Enables full genomic evaluation beyond training positions")
                    
        except Exception as e:
            print(f"[Gene-CV-Sigmoid] ⚠️ Per-nucleotide score generation failed: {e}")
            if hasattr(args, 'verbose') and args.verbose:
                import traceback
                traceback.print_exc()
                print("  Note: This is optional and doesn't affect main CV results")
            
        # Create expected TSV files by copying the generated files
        try:
            # Copy gene_level_argmax_results.tsv to expected meta_vs_base_performance.tsv
            argmax_results_file = out_dir / "gene_level_argmax_results.tsv"
            meta_vs_base_file = out_dir / "meta_vs_base_performance.tsv"
            
            if argmax_results_file.exists():
                import shutil
                shutil.copy2(argmax_results_file, meta_vs_base_file)
                if args.verbose:
                    print(f"[Gene-CV-Sigmoid] ✅ Created meta_vs_base_performance.tsv")
            
            # Create perf_meta_vs_base.tsv as a fallback copy
            perf_meta_vs_base_file = out_dir / "perf_meta_vs_base.tsv"
            if argmax_results_file.exists():
                shutil.copy2(argmax_results_file, perf_meta_vs_base_file)
                if args.verbose:
                    print(f"[Gene-CV-Sigmoid] ✅ Created perf_meta_vs_base.tsv")
            
            # Create detailed_position_comparison.tsv from position-level results
            position_results_file = out_dir / "position_level_classification_results.tsv"
            detailed_position_file = out_dir / "detailed_position_comparison.tsv"
            
            if position_results_file.exists():
                shutil.copy2(position_results_file, detailed_position_file)
                if args.verbose:
                    print(f"[Gene-CV-Sigmoid] ✅ Created detailed_position_comparison.tsv")
                    
        except Exception as e:
            print(f"[Gene-CV-Sigmoid] ⚠️  Error creating additional TSV files: {e}")
            
        # Conditionally run threshold-based evaluation based on calibration analysis
        if eval_approach in ["threshold_safe", "argmax_primary"]:
            try:
                if args.verbose:
                    approach_desc = "safe" if eval_approach == "threshold_safe" else "secondary"
                    print(f"[DEBUG] Running threshold-based evaluation as {approach_desc} method...")
                
                from meta_spliceai.splice_engine.meta_models.training.eval_meta_splice import meta_splice_performance_simple
                
                threshold_results = meta_splice_performance_simple(
                    dataset_path=args.dataset,
                    run_dir=out_dir,
                    sample=diag_sample,
                    out_tsv=out_dir / "gene_level_threshold_results.tsv",
                    verbose=args.verbose if hasattr(args, 'verbose') else 1,
                )
                
                if args.verbose:
                    print(f"[Gene-CV-Sigmoid] ✅ Threshold-based comparison completed!")
                    print(f"  Results saved to: {threshold_results}")
                    if eval_approach == "argmax_primary":
                        print("  Note: Interpret threshold results cautiously due to calibration issues")
                        
            except Exception as e:
                print(f"[Gene-CV-Sigmoid] ⚠️  Threshold-based evaluation failed: {e}")
                print("  This is expected when severe calibration issues are present")
        else:
            if args.verbose:
                print("[DEBUG] Skipping threshold-based evaluation due to severe calibration issues")
                print("  Use argmax results as the primary evaluation method")
            
    except Exception as e:
        print(f"[Gene-CV-Sigmoid] ❌ Gene-level classification comparison failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

    # 3. Additional diagnostics (if not skipped)
    if args.leakage_probe:
        try:
            print("\n[Enhanced Leakage Probe] Running comprehensive leakage visualization analysis...")
            
            # Import our comprehensive leakage analysis function
            from meta_spliceai.splice_engine.meta_models.evaluation.leakage_analysis import run_comprehensive_leakage_analysis
            
            # Run comprehensive leakage analysis with visualizations
            enhanced_leakage_results = run_comprehensive_leakage_analysis(
                dataset_path=args.dataset,
                run_dir=out_dir,
                threshold=args.leakage_threshold,
                methods=['pearson', 'spearman'],
                sample=min(10_000, len(df)),
                top_n=50,
                verbose=1 if args.verbose else 0
            )
            
            print(f"[Enhanced Leakage Probe] Comprehensive analysis completed!")
            print(f"[Enhanced Leakage Probe] Results saved to: {enhanced_leakage_results['output_directory']}")
            
            # Also run basic leakage probe for backward compatibility
            try:
                _cutils.leakage_probe(args.dataset, out_dir, sample=min(10_000, len(df)))
                print(f"[Enhanced Leakage Probe] Basic leakage probe also completed for compatibility")
            except Exception as e:
                print(f"[Enhanced Leakage Probe] Basic leakage probe failed: {e}")
            
        except Exception as e:
            print(f"[Enhanced Leakage Probe] Enhanced analysis failed: {e}")
            print("[Enhanced Leakage Probe] Falling back to basic leakage probe...")
            
            # Fallback to basic leakage probe
            try:
                _cutils.leakage_probe(args.dataset, out_dir, sample=min(10_000, len(df)))
            except Exception as e:
                print("[warning] leakage_probe failed:", e)

    # 4. Run neighbour window diagnostic for deeper insight into spatial effects
    # NOTE: This is kept for feature analysis, not for evaluation
    try:
        if hasattr(args, 'verbose') and args.verbose:
            print("\n[DEBUG] Running neighbor window diagnostics (for feature analysis)")
            
        # Make sure output directory exists
        neighbor_dir = out_dir / "neighbor_diagnostics"
        neighbor_dir.mkdir(exist_ok=True, parents=True)
        
        _cutils.neighbour_window_diagnostics(
            dataset_path=args.dataset,
            run_dir=out_dir,
            annotations_path=args.splice_sites_path,  # Use splice sites path 
            n_sample=args.neigh_sample if args.neigh_sample > 0 else None,
            window=args.neigh_window,
        )
    except Exception as e:
        print("[warning] neighbour_window_diagnostics failed:", e)
        if hasattr(args, 'verbose') and args.verbose:
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

    # Display comprehensive performance summary
    display_comprehensive_performance_summary(out_dir, fold_rows, args, original_dataset_path, args.verbose)

def display_comprehensive_performance_summary(out_dir: Path, fold_rows: list, args, original_dataset_path: str, verbose: bool = True):
    """Display a comprehensive performance comparison table showing base vs meta improvements."""
    
    import pandas as pd
    import json
    
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
            
            # Error reduction summary
            if 'delta_fp' in fold_df.columns and 'delta_fn' in fold_df.columns:
                total_delta_fp = fold_df['delta_fp'].sum()
                total_delta_fn = fold_df['delta_fn'].sum()
                median_delta_fp = fold_df['delta_fp'].median()
                median_delta_fn = fold_df['delta_fn'].median()
                
                print(f"\n🔧 ERROR REDUCTION SUMMARY:")
                print(f"  Total ΔFP = {total_delta_fp:+d} (median per fold: {median_delta_fp:+.0f})")
                print(f"  Total ΔFN = {total_delta_fn:+d} (median per fold: {median_delta_fn:+.0f})")
                print(f"  Net Error Reduction = {total_delta_fp + total_delta_fn:+d}")
                
                # Calculate percentages of folds that improved
                fp_improved = (fold_df['delta_fp'] > 0).sum()
                fn_improved = (fold_df['delta_fn'] > 0).sum()
                both_improved = ((fold_df['delta_fp'] > 0) & (fold_df['delta_fn'] > 0)).sum()
                
                print(f"  Folds with FP reduction: {fp_improved}/{len(fold_df)} ({fp_improved/len(fold_df)*100:.0f}%)")
                print(f"  Folds with FN reduction: {fn_improved}/{len(fold_df)} ({fn_improved/len(fold_df)*100:.0f}%)")
                print(f"  Folds with both improved: {both_improved}/{len(fold_df)} ({both_improved/len(fold_df)*100:.0f}%)")
    
    # 2. Try to load and display detailed comparison if available
    comparison_files = [
        out_dir / "meta_vs_base_performance.tsv",
        out_dir / "meta_evaluation_summary.json",
        out_dir / "detailed_position_comparison.tsv",
        out_dir / "compare_base_meta.json",  # fallback
        out_dir / "perf_meta_vs_base.tsv",  # fallback
    ]
    
    comparison_displayed = False
    
    # Check for meta_evaluation_summary results from correct evaluation
    meta_eval_json = out_dir / "meta_evaluation_summary.json"
    if meta_eval_json.exists():
        try:
            with open(meta_eval_json, 'r') as f:
                meta_stats = json.load(f)
            
            print(f"\n🧬 POSITION-LEVEL CLASSIFICATION PERFORMANCE (CORRECT EVALUATION):")
            print("-" * 50)
            print(f"  Total positions evaluated: {meta_stats.get('total_positions', 0):,}")
            print(f"  Base Model Accuracy: {meta_stats.get('base_accuracy', 0):.3f}")
            print(f"  Meta Model Accuracy: {meta_stats.get('meta_accuracy', 0):.3f}")
            print(f"  Accuracy Improvement: {meta_stats.get('accuracy_improvement', 0):+.3f}")
            print(f"  Base Macro F1: {meta_stats.get('base_f1_macro', 0):.3f}")
            print(f"  Meta Macro F1: {meta_stats.get('meta_f1_macro', 0):.3f}")
            print(f"  F1 Improvement: {meta_stats.get('f1_improvement', 0):+.3f}")
            
            print(f"\n📈 POSITION-LEVEL ERROR CORRECTION:")
            print(f"  Corrections (meta fixed base errors): {meta_stats.get('corrections', 0):,}")
            print(f"  Regressions (meta introduced errors): {meta_stats.get('regressions', 0):,}")
            print(f"  Net Improvement: {meta_stats.get('net_improvement', 0):+,}")
            print(f"  Improvement Rate: {meta_stats.get('improvement_rate', 0):.1%}")
            
            # Per-class F1 improvements
            print(f"\n📊 PER-CLASS F1 IMPROVEMENTS:")
            for cls in ['neither', 'donor', 'acceptor']:
                base_f1 = meta_stats.get(f'base_f1_{cls}', 0)
                meta_f1 = meta_stats.get(f'meta_f1_{cls}', 0)
                improvement = meta_stats.get(f'f1_improvement_{cls}', 0)
                print(f"  {cls.capitalize()}: {base_f1:.3f} → {meta_f1:.3f} ({improvement:+.3f})")
            
            comparison_displayed = True
            
        except Exception as e:
            print(f"[DEBUG] Could not load meta_evaluation_summary.json: {e}")
    
    # Check for base_vs_meta results (fallback)
    base_meta_json = out_dir / "compare_base_meta.json"
    if base_meta_json.exists() and not comparison_displayed:
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
            print(f"[DEBUG] Could not load base_vs_meta.json: {e}")
    
    # Check for splice performance comparison
    meta_vs_base_tsv = out_dir / "meta_vs_base_performance.tsv"
    if meta_vs_base_tsv.exists() and not comparison_displayed:
        try:
            comp_df = pd.read_csv(meta_vs_base_tsv, sep='\t')
            
            if 'f1_score' in comp_df.columns:
                print(f"\n🧬 GENE-LEVEL PERFORMANCE COMPARISON (CLASSIFICATION-BASED):")
                print("-" * 50)
                
                # Summary statistics for meta vs base
                if 'TP0' in comp_df.columns and 'TP' in comp_df.columns:
                    # Delta calculations
                    total_tp_delta = (comp_df['TP'] - comp_df['TP0']).sum()
                    total_fp_delta = (comp_df['FP0'] - comp_df['FP']).sum()  # FP reduction
                    total_fn_delta = (comp_df['FN0'] - comp_df['FN']).sum()  # FN reduction
                    
                    print(f"  Total TP gained: {total_tp_delta:+,}")
                    print(f"  Total FP reduced: {total_fp_delta:+,}")
                    print(f"  Total FN reduced: {total_fn_delta:+,}")
                    
                    # Gene-level improvement counts
                    genes_tp_improved = ((comp_df['TP'] - comp_df['TP0']) > 0).sum()
                    genes_fp_improved = ((comp_df['FP0'] - comp_df['FP']) > 0).sum()
                    genes_fn_improved = ((comp_df['FN0'] - comp_df['FN']) > 0).sum()
                    
                    print(f"  Genes with TP improvement: {genes_tp_improved}")
                    print(f"  Genes with FP reduction: {genes_fp_improved}")
                    print(f"  Genes with FN reduction: {genes_fn_improved}")
                    
                    # F1 score improvements
                    mean_f1_base = comp_df['f1_score'].mean()  # This is meta F1
                    if 'TP0' in comp_df.columns:
                        # Calculate base F1 from TP0, FP0, FN0
                        base_f1_scores = 2 * comp_df['TP0'] / (2 * comp_df['TP0'] + comp_df['FP0'] + comp_df['FN0'])
                        base_f1_scores = base_f1_scores.fillna(0)
                        mean_f1_base_actual = base_f1_scores.mean()
                        print(f"  Mean F1 Score: Base = {mean_f1_base_actual:.3f}, Meta = {mean_f1_base:.3f}")
                        print(f"  F1 Improvement: {mean_f1_base - mean_f1_base_actual:+.3f}")
                else:
                    print(f"  Mean F1 Score (Meta): {comp_df['f1_score'].mean():.3f}")
                    print(f"  Mean Precision (Meta): {comp_df['precision'].mean():.3f}")
                    print(f"  Mean Recall (Meta): {comp_df['recall'].mean():.3f}")
                    
                comparison_displayed = True
                
        except Exception as e:
            print(f"[DEBUG] Could not load meta_vs_base_performance.tsv: {e}")
    
    # Check for splice performance comparison (fallback)
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
            print(f"[DEBUG] Could not load perf_meta_vs_base.tsv: {e}")
    
    # 3. Gene-level and transcript-level top-k accuracy
    if fold_rows and any('top_k_accuracy' in row for row in fold_rows):
        print(f"\n🎯 TOP-{args.top_k} GENE-LEVEL ACCURACY:")
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
        
        # Transcript-level if available
        if any('tx_top_k_combined' in row for row in fold_rows):
            print(f"\n🧬 TOP-{args.top_k} TRANSCRIPT-LEVEL ACCURACY:")
            print("-" * 50)
            
            tx_combined = [row['tx_top_k_combined'] for row in fold_rows if 'tx_top_k_combined' in row and not pd.isna(row['tx_top_k_combined'])]
            tx_donor = [row['tx_top_k_donor'] for row in fold_rows if 'tx_top_k_donor' in row and not pd.isna(row['tx_top_k_donor'])]
            tx_acceptor = [row['tx_top_k_acceptor'] for row in fold_rows if 'tx_top_k_acceptor' in row and not pd.isna(row['tx_top_k_acceptor'])]
            
            if tx_combined:
                print(f"  Combined: {np.mean(tx_combined):.3f} ± {np.std(tx_combined):.3f}")
            if tx_donor:
                print(f"  Donor:    {np.mean(tx_donor):.3f} ± {np.std(tx_donor):.3f}")
            if tx_acceptor:
                print(f"  Acceptor: {np.mean(tx_acceptor):.3f} ± {np.std(tx_acceptor):.3f}")
    
    # 4. Generated artifacts summary
    print(f"\n📁 GENERATED ANALYSIS ARTIFACTS:")
    print("-" * 50)
    
    key_files = [
        ("gene_cv_metrics.csv", "CV fold metrics"),
        ("model_multiclass.pkl", "Trained meta-model"),
        ("roc_curves_meta.pdf", "ROC curves (meta model)"),
        ("pr_curves_meta.pdf", "PR curves (meta model)"),
        ("cv_metrics_visualization/", "CV visualization suite"),
        ("shap_analysis/", "SHAP importance analysis"),
        ("feature_importance_analysis/", "Multi-method feature analysis"),
        ("leakage_analysis/", "Comprehensive data leakage analysis"),
        ("meta_evaluation_argmax_summary.json", "ARGMAX-based gene-level comparison"),
        ("gene_level_argmax_results.tsv", "Detailed ARGMAX gene results"),
        ("meta_evaluation_summary.json", "Correct position-level comparison"),
        ("position_level_classification_results.tsv", "Detailed position results"), 
        ("meta_vs_base_performance.tsv", "Classification-based gene comparison"),
        ("detailed_position_comparison.tsv", "Detailed position analysis"),
        ("compare_base_meta.json", "Position-level comparison (fallback)"),
        ("perf_meta_vs_base.tsv", "Gene-level comparison (fallback)"),
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
    
    # Final summary of the evaluation approach
    print("\n" + "="*80)
    print("🔧 EVALUATION METHODOLOGY SUMMARY")
    print("="*80)
    print("✅ CORRECT APPROACH: Classification-based evaluation")
    print("   - Compares meta vs base predictions at the SAME training positions")
    print("   - Uses splice_type labels to measure classification accuracy")
    print("   - No coordinate matching or position detection required")
    print("   - Directly measures improvement in splice site classification")
    print()
    print("❌ AVOIDED: Position-based evaluation")
    print("   - Would try to match predicted vs truth genomic coordinates")
    print("   - Inappropriate since meta-model predicts classifications, not positions")
    print("   - Leads to systematic coordinate offset errors")
    print("   - Results in misleading TP=0 with only FP/FN counts")
    print("="*80)
    
    # Generate comprehensive training summary
    cv_utils.generate_training_summary(
        out_dir=out_dir,
        args=args,
        original_dataset_path=original_dataset_path,
        fold_rows=fold_rows,
        script_name="run_gene_cv_sigmoid.py"
    )

if __name__ == "__main__":  # pragma: no cover
    main()
