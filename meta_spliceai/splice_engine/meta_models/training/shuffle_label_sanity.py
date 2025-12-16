#!/usr/bin/env python3
"""Train the meta-model with **shuffled labels** as a leakage sanity check.

If the pipeline is free from label leakage, performance metrics on the test
split must collapse to chance levels (AUC ≈ 0.5, accuracy/F1 near the class
priors).

This script now mirrors the preprocessing, feature handling, and evaluation
pipeline of `run_gene_cv_sigmoid.py` but shuffles the label column *before* 
any train/valid/test split. All other settings (feature preprocessing, 
group-aware splitting, model hyper-params, diagnostics) remain identical to 
the normal training run so that results are comparable.

Example
-------
$ python -m meta_spliceai.splice_engine.meta_models.training.shuffle_label_sanity \
    --dataset train_pc_1000/master \
    --out-dir results/shuffle_sanity_binary \
    --n-folds 5 \
    --n-estimators 500 \
    --verbose --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier

from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.builder import preprocessing
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import (
    _encode_labels, SigmoidEnsemble, CalibratedSigmoidEnsemble
)
from meta_spliceai.splice_engine.meta_models.evaluation.feature_utils import (
    load_excluded_features, filter_features
)
from meta_spliceai.splice_engine.meta_models.evaluation.viz_utils import check_feature_correlations
from meta_spliceai.splice_engine.utils_kmer import is_kmer_feature as _is_kmer
from meta_spliceai.splice_engine.meta_models.training import classifier_utils as _cutils


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shuffle-label sanity check for binary splice detection.")
    
    # Dataset and output
    p.add_argument("--dataset", required=True, help="Dataset directory or Parquet file")
    p.add_argument("--out-dir", required=True, help="Where to save results")
    
    # CV and data options
    p.add_argument("--gene-col", default="gene_id")
    p.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    p.add_argument("--valid-size", type=float, default=0.1)
    p.add_argument("--row-cap", type=int, default=100_000)
    p.add_argument("--sample-genes", type=int, default=None,
                   help="Sample only a subset of genes for faster testing")
    
    # Model parameters (should match gene CV defaults)
    p.add_argument("--tree-method", default="hist", choices=["hist", "gpu_hist", "approx"])
    p.add_argument("--max-bin", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--n-estimators", type=int, default=800)
    
    # Feature handling
    p.add_argument("--exclude-features", default="configs/exclude_features.txt", 
                   help="Path to file with features to exclude or comma-separated list")
    p.add_argument("--check-leakage", action="store_true", default=True)
    p.add_argument("--leakage-threshold", type=float, default=0.95)
    p.add_argument("--auto-exclude-leaky", action="store_true", default=False)
    
    # Evaluation options
    p.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    p.add_argument("--plot-format", default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--run-diagnostics", action="store_true", default=False,
                   help="Run post-training diagnostics (should show chance performance)")
    
    # General
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    
    return p.parse_args(argv)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresh: float) -> Dict[str, float]:
    """Calculate binary classification metrics."""
    y_pred = (y_prob >= thresh).astype(int)
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "average_precision": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
    }


def _train_binary_model(X: np.ndarray, y_bin: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, args: argparse.Namespace) -> XGBClassifier:
    """Train a binary XGBoost model (matches gene CV parameters)."""
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
    return model


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    
    # Honor row cap
    if args.row_cap > 0 and not os.getenv("SS_MAX_ROWS"):
        os.environ["SS_MAX_ROWS"] = str(args.row_cap)
    
    print(f"[Shuffle Label Sanity] Binary splice detection with shuffled labels")
    print(f"[Shuffle Label Sanity] Output directory: {out_dir}")
    print(f"[Shuffle Label Sanity] Random seed: {args.seed}")
    
    # 1. Load dataset (same as gene CV)
    if args.sample_genes is not None:
        from meta_spliceai.splice_engine.meta_models.training.label_utils import load_dataset_sample
        print(f"[Shuffle Label Sanity] Sampling {args.sample_genes} genes for faster testing")
        df = load_dataset_sample(args.dataset, sample_genes=args.sample_genes, random_seed=args.seed)
    else:
        df = datasets.load_dataset(args.dataset)
    
    if args.gene_col not in df.columns:
        raise KeyError(f"Column '{args.gene_col}' not found in dataset")
    
    # 2. Prepare training data (same preprocessing as gene CV)
    X_df, y_series = preprocessing.prepare_training_data(
        df, 
        label_col="splice_type", 
        return_type="pandas", 
        verbose=1 if args.verbose else 0,
        encode_chrom=True  # Include chromosome encoding
    )
    
    # Save feature names before exclusion
    original_features = list(X_df.columns)
    feature_path = out_dir / "original_features.json"
    with open(feature_path, "w") as f:
        json.dump({"feature_names": original_features}, f)
    
    # 3. Handle feature exclusion (same logic as gene CV)
    exclude_list = []
    
    if args.exclude_features:
        exclude_path = Path(args.exclude_features)
        if exclude_path.exists() and exclude_path.is_file():
            print(f"[Shuffle Label Sanity] Loading exclude features from file: {exclude_path}")
            file_exclusions = load_excluded_features(exclude_path)
            if file_exclusions:
                exclude_list.extend(file_exclusions)
        else:
            print(f"[Shuffle Label Sanity] Treating exclude features as comma-separated list")
            exclude_list.extend([f.strip() for f in args.exclude_features.split(',') if f.strip()])
    
    # Check for leakage if enabled
    if args.check_leakage:
        X_np = X_df.values
        y_np = _encode_labels(y_series)
        curr_features = X_df.columns.tolist()
        
        correlation_report_path = out_dir / "feature_label_correlations.csv"
        print(f"[Shuffle Label Sanity] Checking for potential feature leakage (threshold={args.leakage_threshold})...")
        
        leaky_features, corr_df = check_feature_correlations(
            X_np, y_np, curr_features, args.leakage_threshold, correlation_report_path
        )
        
        if args.auto_exclude_leaky and leaky_features:
            print(f"[Shuffle Label Sanity] Auto-excluding {len(leaky_features)} potentially leaky features")
            exclude_list.extend(leaky_features)
    
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
            print(f"[Shuffle Label Sanity] Excluded {len(excluded_features)} features")
            print(f"[Shuffle Label Sanity] Feature count: {original_feature_count} → {X_df.shape[1]}")
            
            # Save excluded features
            with open(out_dir / "excluded_features.txt", 'w') as f:
                f.write("# Features excluded during shuffle label sanity check\n")
                for feature in excluded_features:
                    f.write(f"{feature}\n")
    
    # Convert to numpy arrays
    X = X_df.values
    y_multiclass = _encode_labels(y_series)
    y_binary = (y_multiclass != 0).astype(int)  # Convert to binary: splice vs non-splice
    genes = df[args.gene_col].to_numpy()
    feature_names = list(X_df.columns)
    
    # Save final feature manifest
    pd.DataFrame({"feature": feature_names}).to_csv(out_dir / "feature_manifest.csv", index=False)
    
    print(f"[Shuffle Label Sanity] Dataset shape: {X.shape}")
    print(f"[Shuffle Label Sanity] Binary class distribution: {np.bincount(y_binary)}")
    print(f"[Shuffle Label Sanity] Features: {len(feature_names)} total")
    
    # 4. **CRITICAL**: Shuffle the binary labels
    print(f"[Shuffle Label Sanity] *** SHUFFLING LABELS *** (sanity check)")
    y_binary_shuffled = rng.permutation(y_binary)
    
    # Verify shuffling worked
    original_positive_rate = y_binary.mean()
    shuffled_positive_rate = y_binary_shuffled.mean()
    print(f"[Shuffle Label Sanity] Original positive rate: {original_positive_rate:.3f}")
    print(f"[Shuffle Label Sanity] Shuffled positive rate: {shuffled_positive_rate:.3f}")
    
    # 5. Gene-wise K-fold CV (same structure as gene CV)
    gkf = GroupKFold(n_splits=args.n_folds)
    fold_rows: List[Dict] = []
    
    # Collect metrics across folds
    fold_metrics = []
    all_y_true, all_y_prob = [], []
    
    print(f"\n[Shuffle Label Sanity] Starting {args.n_folds}-fold CV with shuffled labels...")
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(gkf.split(X, y_binary_shuffled, groups=genes)):
        print(f"[Shuffle Label Sanity] Fold {fold_idx+1}/{args.n_folds} (test_rows={len(test_idx)})")
        
        # Train/valid split preserving gene groups
        from sklearn.model_selection import GroupShuffleSplit
        rel_valid = args.valid_size / (1.0 - 1.0 / args.n_folds)
        gss = GroupShuffleSplit(n_splits=1, test_size=rel_valid, random_state=args.seed)
        train_idx, valid_idx = next(gss.split(train_val_idx, y_binary_shuffled[train_val_idx], groups=genes[train_val_idx]))
        train_idx = train_val_idx[train_idx]
        valid_idx = train_val_idx[valid_idx]
        
        # Train binary model
        model = _train_binary_model(X[train_idx], y_binary_shuffled[train_idx], 
                                   X[valid_idx], y_binary_shuffled[valid_idx], args)
        
        # Evaluate on test set
        y_prob = model.predict_proba(X[test_idx])[:, 1]
        y_true = y_binary_shuffled[test_idx]
        
        metrics = _binary_metrics(y_true, y_prob, args.threshold)
        fold_metrics.append(metrics)
        
        # Store for aggregation
        all_y_true.append(y_true)
        all_y_prob.append(y_prob)
        
        print(f"   Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}, "
              f"ROC-AUC: {metrics['roc_auc']:.3f}, AP: {metrics['average_precision']:.3f}")
        
        # Store fold results
        fold_row = {
            "fold": fold_idx,
            "test_rows": len(test_idx),
            **{f"test_{k}": v for k, v in metrics.items()}
        }
        fold_rows.append(fold_row)
        
        # Save fold-specific results
        with open(out_dir / f"metrics_fold{fold_idx}.json", "w") as f:
            json.dump({k: float(v) if isinstance(v, np.floating) else v for k, v in fold_row.items()}, f, indent=2)
    
    # 6. Aggregate results
    df_metrics = pd.DataFrame(fold_rows)
    df_metrics.to_csv(out_dir / "shuffle_metrics_folds.csv", index=False)
    
    # Calculate aggregate metrics
    mean_metrics = {}
    std_metrics = {}
    for metric in ["test_accuracy", "test_f1", "test_roc_auc", "test_average_precision"]:
        values = df_metrics[metric].dropna()
        mean_metrics[metric] = values.mean()
        std_metrics[metric] = values.std()
    
    print(f"\n[Shuffle Label Sanity] === RESULTS (should be near chance levels) ===")
    print(f"Accuracy:          {mean_metrics['test_accuracy']:.3f} ± {std_metrics['test_accuracy']:.3f}")
    print(f"F1 Score:          {mean_metrics['test_f1']:.3f} ± {std_metrics['test_f1']:.3f}")
    print(f"ROC AUC:           {mean_metrics['test_roc_auc']:.3f} ± {std_metrics['test_roc_auc']:.3f}")
    print(f"Average Precision: {mean_metrics['test_average_precision']:.3f} ± {std_metrics['test_average_precision']:.3f}")
    
    # Expected chance levels
    positive_rate = y_binary.mean()
    expected_accuracy = max(positive_rate, 1 - positive_rate)  # Majority class accuracy
    expected_auc = 0.5
    expected_ap = positive_rate  # Random classifier AP = positive class frequency
    
    print(f"\n[Shuffle Label Sanity] === EXPECTED CHANCE LEVELS ===")
    print(f"Expected Accuracy (majority class): {expected_accuracy:.3f}")
    print(f"Expected ROC AUC (random):          {expected_auc:.3f}")
    print(f"Expected Average Precision:         {expected_ap:.3f}")
    
    # Sanity check: flag if performance is significantly above chance
    auc_above_chance = mean_metrics['test_roc_auc'] > 0.6
    acc_above_chance = mean_metrics['test_accuracy'] > expected_accuracy + 0.1
    
    if auc_above_chance or acc_above_chance:
        print(f"\n[Shuffle Label Sanity] ⚠️  WARNING: Performance significantly above chance!")
        print(f"This may indicate label leakage or other issues in the pipeline.")
    else:
        print(f"\n[Shuffle Label Sanity] ✅ Performance at chance levels - pipeline appears healthy")
    
    # Save aggregate results
    aggregate_results = {
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "expected_chance_levels": {
            "accuracy": expected_accuracy,
            "roc_auc": expected_auc,
            "average_precision": expected_ap
        },
        "sanity_check": {
            "auc_above_chance": bool(auc_above_chance),
            "accuracy_above_chance": bool(acc_above_chance),
            "appears_healthy": not (auc_above_chance or acc_above_chance)
        },
        "dataset_info": {
            "n_samples": len(X),
            "n_features": len(feature_names),
            "n_folds": args.n_folds,
            "positive_rate": float(positive_rate),
            "excluded_features": len(exclude_list) if exclude_list else 0
        }
    }
    
    with open(out_dir / "shuffle_sanity_summary.json", "w") as f:
        json.dump(aggregate_results, f, indent=2)
    
    # Generate ROC and PR curve plots
    if args.plot_format:
        print(f"[Shuffle Label Sanity] Generating performance plots...")
        
        # Combine all folds for plotting
        y_true_combined = np.concatenate(all_y_true)
        y_prob_combined = np.concatenate(all_y_prob)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true_combined, y_prob_combined)
        
        plt.figure(figsize=(12, 5))
        
        # ROC plot
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, 'b-', label=f'Shuffled Labels (AUC = {mean_metrics["test_roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.5)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Shuffled Labels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # PR curve
        precision, recall, _ = precision_recall_curve(y_true_combined, y_prob_combined)
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, 'r-', label=f'Shuffled Labels (AP = {mean_metrics["test_average_precision"]:.3f})')
        plt.axhline(y=positive_rate, color='k', linestyle='--', alpha=0.5, 
                   label=f'Random (AP = {positive_rate:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Shuffled Labels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"shuffle_sanity_curves.{args.plot_format}", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Optional: Run basic diagnostics (should show chance performance)
    if args.run_diagnostics:
        print(f"\n[Shuffle Label Sanity] Running basic diagnostics...")
        
        # Train final model on full data for diagnostics
        final_model = _train_binary_model(X, y_binary_shuffled, X, y_binary_shuffled, args)
        
        # Create ensemble wrapper for compatibility
        ensemble = SigmoidEnsemble([final_model], feature_names)
        
        # Save model
        import pickle
        with open(out_dir / "model_shuffled.pkl", "wb") as f:
            pickle.dump(ensemble, f)
        
        # Run basic diagnostics (these should show chance performance)
        try:
            sample = min(10000, len(X))  # Use smaller sample for faster diagnostics
            _cutils.richer_metrics(args.dataset, out_dir, sample=sample)
            print(f"[Shuffle Label Sanity] ✓ Basic diagnostics completed")
        except Exception as e:
            print(f"[Shuffle Label Sanity] ⚠️  Diagnostics failed: {e}")
    
    print(f"\n[Shuffle Label Sanity] Analysis complete. Results saved to: {out_dir}")
    print(f"[Shuffle Label Sanity] Summary: shuffle_sanity_summary.json")


if __name__ == "__main__":
    main()
