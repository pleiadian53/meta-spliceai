#!/usr/bin/env python3
"""Shuffle-label sanity check for the *multiclass* (donor / acceptor / neither)
meta-model.

If everything is wired correctly and no label leakage exists, shuffling the
labels before the train/valid/test split should drive evaluation metrics to
chance levels (≈ 0.33 accuracy, macro-F1 near the class prior) regardless of
model capacity.

This script now mirrors the preprocessing, feature handling, and evaluation
pipeline of `run_gene_cv_multiclass.py` but performs **label shuffling only** – 
all other hyper-parameters stay identical so you can compare numbers directly.

Example
-------
$ python -m meta_spliceai.splice_engine.meta_models.training.shuffle_label_sanity_multiclass \
    --dataset train_pc_1000/master \
    --out-dir results/shuffle_sanity_multiclass \
    --n-folds 5 \
    --n-estimators 800 \
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
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)
from xgboost import XGBClassifier

from meta_spliceai.splice_engine.meta_models.training import datasets
from meta_spliceai.splice_engine.meta_models.builder import preprocessing
from meta_spliceai.splice_engine.meta_models.training.classifier_utils import (
    _encode_labels
)
from meta_spliceai.splice_engine.meta_models.evaluation.feature_utils import (
    load_excluded_features
)
from meta_spliceai.splice_engine.meta_models.evaluation.viz_utils import check_feature_correlations
from meta_spliceai.splice_engine.utils_kmer import is_kmer_feature as _is_kmer

# Label mapping for interpretability
LABEL_NAMES = {0: "neither", 1: "donor", 2: "acceptor"}


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shuffle-label sanity check for multiclass splice detection.")
    
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
    p.add_argument("--exclude-features", default=None, 
                   help="Path to file with features to exclude or comma-separated list")
    p.add_argument("--check-leakage", action="store_true", default=False)
    p.add_argument("--leakage-threshold", type=float, default=0.95)
    p.add_argument("--auto-exclude-leaky", action="store_true", default=False)
    
    # Evaluation options
    p.add_argument("--plot-format", default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--run-diagnostics", action="store_true", default=False,
                   help="Run post-training diagnostics (should show chance performance)")
    
    # General
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    
    return p.parse_args(argv)


def _multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate multiclass classification metrics."""
    # Splice-site-only metrics (exclude class 0="neither")
    splice_mask = y_true != 0
    if splice_mask.any():
        splice_acc = accuracy_score(y_true[splice_mask], y_pred[splice_mask])
        splice_macro_f1 = f1_score(y_true[splice_mask], y_pred[splice_mask], average="macro")
    else:
        splice_acc = np.nan
        splice_macro_f1 = np.nan

    # Top-K accuracy over splice sites (donor+acceptor)
    k = int(splice_mask.sum())
    if k > 0:
        # For shuffled labels, this becomes meaningless but we calculate it for consistency
        top_k_acc = np.nan  # Would need probabilities to calculate properly
    else:
        top_k_acc = np.nan

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "splice_accuracy": splice_acc,
        "splice_macro_f1": splice_macro_f1,
        "top_k_accuracy": top_k_acc,
        "top_k": k,
    }


def _calculate_roc_pr_metrics(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int = 3) -> Dict[str, float]:
    """Calculate ROC and PR metrics for multiclass classification."""
    # Binarize the labels for multiclass ROC/PR
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Calculate metrics for each class
    class_metrics = {}
    for i in range(n_classes):
        class_name = LABEL_NAMES[i]
        
        # ROC metrics
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        # PR metrics
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
        ap_score = average_precision_score(y_true_bin[:, i], y_proba[:, i])
        
        class_metrics[f"{class_name}_roc_auc"] = roc_auc
        class_metrics[f"{class_name}_ap"] = ap_score
    
    # Micro averages (aggregate across all classes)
    try:
        # Micro-average ROC (only valid for binary or with special handling for multiclass)
        micro_roc_auc = roc_auc_score(y_true_bin, y_proba, average="micro", multi_class="ovr")
        micro_ap = average_precision_score(y_true_bin, y_proba, average="micro")
    except Exception:
        micro_roc_auc = np.nan
        micro_ap = np.nan
    
    # Macro averages
    macro_roc_auc = np.mean([class_metrics[f"{LABEL_NAMES[i]}_roc_auc"] for i in range(n_classes)])
    macro_ap = np.mean([class_metrics[f"{LABEL_NAMES[i]}_ap"] for i in range(n_classes)])
    
    class_metrics.update({
        "micro_roc_auc": micro_roc_auc,
        "macro_roc_auc": macro_roc_auc,
        "micro_ap": micro_ap,
        "macro_ap": macro_ap,
    })
    
    return class_metrics


def _plot_roc_pr_curves(y_true_list: List[np.ndarray], y_proba_list: List[np.ndarray], 
                       out_dir: Path, plot_format: str = "pdf") -> None:
    """Plot ROC and PR curves aggregated across all folds."""
    n_classes = 3
    n_folds = len(y_true_list)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot ROC curves
    for i in range(n_classes):
        class_name = LABEL_NAMES[i]
        ax_roc = axes[0, i]
        
        # Aggregate ROC curves across folds
        all_fpr = []
        all_tpr = []
        all_auc = []
        
        for fold_idx in range(n_folds):
            y_true_fold = y_true_list[fold_idx]
            y_proba_fold = y_proba_list[fold_idx]
            
            # Binarize for this class
            y_true_bin = (y_true_fold == i).astype(int)
            y_scores = y_proba_fold[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
            roc_auc = auc(fpr, tpr)
            
            all_fpr.append(fpr)
            all_tpr.append(tpr)
            all_auc.append(roc_auc)
            
            # Plot individual fold (light lines)
            ax_roc.plot(fpr, tpr, alpha=0.3, linewidth=1)
        
        # Plot chance line (diagonal)
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.8, linewidth=2, label='Chance (AUC=0.5)')
        
        # Calculate and display mean AUC
        mean_auc = np.mean(all_auc)
        std_auc = np.std(all_auc)
        
        ax_roc.set_xlabel('False Positive Rate', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate', fontsize=12)
        ax_roc.set_title(f'ROC Curve - {class_name.title()}\nMean AUC = {mean_auc:.3f} ± {std_auc:.3f}', 
                        fontsize=14, fontweight='bold')
        ax_roc.legend(fontsize=10)
        ax_roc.grid(True, alpha=0.3)
        ax_roc.set_xlim([0, 1])
        ax_roc.set_ylim([0, 1])
    
    # Plot PR curves
    for i in range(n_classes):
        class_name = LABEL_NAMES[i]
        ax_pr = axes[1, i]
        
        # Aggregate PR curves across folds
        all_ap = []
        class_prevalence = []
        
        for fold_idx in range(n_folds):
            y_true_fold = y_true_list[fold_idx]
            y_proba_fold = y_proba_list[fold_idx]
            
            # Binarize for this class
            y_true_bin = (y_true_fold == i).astype(int)
            y_scores = y_proba_fold[:, i]
            
            precision, recall, _ = precision_recall_curve(y_true_bin, y_scores)
            ap_score = average_precision_score(y_true_bin, y_scores)
            prevalence = y_true_bin.mean()
            
            all_ap.append(ap_score)
            class_prevalence.append(prevalence)
            
            # Plot individual fold (light lines)
            ax_pr.plot(recall, precision, alpha=0.3, linewidth=1)
        
        # Plot chance line (horizontal at class prevalence)
        mean_prevalence = np.mean(class_prevalence)
        ax_pr.axhline(y=mean_prevalence, color='k', linestyle='--', alpha=0.8, linewidth=2, 
                     label=f'Chance (AP={mean_prevalence:.3f})')
        
        # Calculate and display mean AP
        mean_ap = np.mean(all_ap)
        std_ap = np.std(all_ap)
        
        ax_pr.set_xlabel('Recall', fontsize=12)
        ax_pr.set_ylabel('Precision', fontsize=12)
        ax_pr.set_title(f'PR Curve - {class_name.title()}\nMean AP = {mean_ap:.3f} ± {std_ap:.3f}', 
                       fontsize=14, fontweight='bold')
        ax_pr.legend(fontsize=10)
        ax_pr.grid(True, alpha=0.3)
        ax_pr.set_xlim([0, 1])
        ax_pr.set_ylim([0, 1])
    
    plt.suptitle('ROC and PR Curves - Shuffled Labels (Should Show Chance Performance)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(out_dir / f"roc_pr_curves_shuffled.{plot_format}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Shuffle Label Sanity] ✓ ROC and PR curves saved to: roc_pr_curves_shuffled.{plot_format}")


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
    
    print(f"[Shuffle Label Sanity] Multiclass splice detection with shuffled labels")
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
        verbose=1 if args.verbose else 0
    )
    
    # Convert to numpy arrays and encode labels exactly like gene CV
    X = X_df.values
    y = _encode_labels(y_series)
    genes = df[args.gene_col].to_numpy()
    
    # Quick feature overview (same as gene CV)
    feature_names = list(X_df.columns)
    non_kmer_feats = [f for f in feature_names if not _is_kmer(f)]
    kmer_feats = [f for f in feature_names if _is_kmer(f)]

    print(f"[Shuffle Label Sanity] Feature matrix: {len(feature_names)} columns – {len(non_kmer_feats)} non-k-mer features")
    if kmer_feats:
        sample_kmer = random.sample(kmer_feats, k=min(3, len(kmer_feats)))
        print("       Example k-mer feats:", ", ".join(sample_kmer))
    
    # 3. Handle feature exclusion (optional, like gene CV)
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
        curr_features = X_df.columns.tolist()
        
        correlation_report_path = out_dir / "feature_label_correlations.csv"
        print(f"[Shuffle Label Sanity] Checking for potential feature leakage (threshold={args.leakage_threshold})...")
        
        leaky_features, corr_df = check_feature_correlations(
            X, y, curr_features, args.leakage_threshold, correlation_report_path
        )
        
        if args.auto_exclude_leaky and leaky_features:
            print(f"[Shuffle Label Sanity] Auto-excluding {len(leaky_features)} potentially leaky features")
            exclude_list.extend(leaky_features)
    
    # Apply exclusions
    if exclude_list:
        exclude_list = list(dict.fromkeys(exclude_list))  # Remove duplicates
        original_feature_count = X.shape[1]
        excluded_indices = []
        
        for feature in exclude_list:
            if feature in feature_names:
                excluded_indices.append(feature_names.index(feature))
        
        if excluded_indices:
            # Remove excluded features from X and feature_names
            keep_indices = [i for i in range(len(feature_names)) if i not in excluded_indices]
            X = X[:, keep_indices]
            feature_names = [feature_names[i] for i in keep_indices]
            
            print(f"[Shuffle Label Sanity] Excluded {len(excluded_indices)} features")
            print(f"[Shuffle Label Sanity] Feature count: {original_feature_count} → {X.shape[1]}")
            
            # Save excluded features
            with open(out_dir / "excluded_features.txt", 'w') as f:
                f.write("# Features excluded during shuffle label sanity check\n")
                for idx in excluded_indices:
                    f.write(f"{list(X_df.columns)[idx]}\n")
    
    # Save final feature manifest
    pd.DataFrame({"feature": feature_names}).to_csv(out_dir / "feature_manifest.csv", index=False)
    
    print(f"[Shuffle Label Sanity] Dataset shape: {X.shape}")
    print(f"[Shuffle Label Sanity] Class distribution (original): {np.bincount(y)}")
    for cls_idx, count in enumerate(np.bincount(y)):
        print(f"  {LABEL_NAMES[cls_idx]}: {count} ({count/len(y)*100:.1f}%)")
    print(f"[Shuffle Label Sanity] Features: {len(feature_names)} total")
    
    # 4. **CRITICAL**: Shuffle the multiclass labels
    print(f"[Shuffle Label Sanity] *** SHUFFLING LABELS *** (sanity check)")
    y_shuffled = rng.permutation(y)

    # Verify shuffling worked
    original_dist = np.bincount(y) / len(y)
    shuffled_dist = np.bincount(y_shuffled) / len(y_shuffled)
    print(f"[Shuffle Label Sanity] Original distribution: {[f'{d:.3f}' for d in original_dist]}")
    print(f"[Shuffle Label Sanity] Shuffled distribution: {[f'{d:.3f}' for d in shuffled_dist]}")
    
    # 5. Gene-wise K-fold CV (exactly like gene CV)
    gkf = GroupKFold(n_splits=args.n_folds)
    fold_rows: List[Dict] = []
    
    # Collect predictions and probabilities for ROC/PR curves
    all_y_true: List[np.ndarray] = []
    all_y_proba: List[np.ndarray] = []
    
    print(f"\n[Shuffle Label Sanity] Starting {args.n_folds}-fold CV with shuffled labels...")
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(gkf.split(X, y_shuffled, groups=genes)):
        print(f"[Shuffle Label Sanity] Fold {fold_idx+1}/{args.n_folds} (test_rows={len(test_idx)})")
        
        # Split TRAIN into TRAIN/VALID with groups again (exactly like gene CV)
        rel_valid = args.valid_size / (1.0 - 1.0 / args.n_folds)
        gss = GroupShuffleSplit(n_splits=1, test_size=rel_valid, random_state=args.seed)
        (train_idx, valid_idx) = next(gss.split(train_val_idx, y_shuffled[train_val_idx], groups=genes[train_val_idx]))
        train_idx = train_val_idx[train_idx]
        valid_idx = train_val_idx[valid_idx]
        
        # Model (exactly matching gene CV parameters)
        model = XGBClassifier(
            n_estimators=args.n_estimators,
            tree_method=args.tree_method,
            max_bin=args.max_bin if args.tree_method in {"hist", "gpu_hist"} else None,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=args.seed,
            n_jobs=-1,
            device=args.device if args.device != "auto" else None,
        )

        model.fit(
            X[train_idx],
            y_shuffled[train_idx],
            eval_set=[(X[valid_idx], y_shuffled[valid_idx])],
            verbose=False,
        )

        proba = model.predict_proba(X[test_idx])
        pred = proba.argmax(axis=1)
        y_true_test = y_shuffled[test_idx]
        
        # Store predictions for ROC/PR curves
        all_y_true.append(y_true_test)
        all_y_proba.append(proba)
        
        # Calculate standard metrics (same as gene CV)
        metrics = _multiclass_metrics(y_true_test, pred)
        
        # Calculate ROC and PR metrics for this fold
        roc_pr_metrics = _calculate_roc_pr_metrics(y_true_test, proba)
        
        print(f"   Accuracy: {metrics['accuracy']:.3f}, Macro F1: {metrics['macro_f1']:.3f}")
        print(f"   Splice Acc: {metrics['splice_accuracy']:.3f}, Splice F1: {metrics['splice_macro_f1']:.3f}")
        print(f"   Macro ROC-AUC: {roc_pr_metrics['macro_roc_auc']:.3f}, Macro AP: {roc_pr_metrics['macro_ap']:.3f}")
        
        # Confusion matrix (same as gene CV)
        cm = confusion_matrix(y_true_test, pred, labels=[0, 1, 2])
        label_names = ["neither", "donor", "acceptor"]
        print("Class distribution (true labels):", {
            name: int((y_true_test == i).sum()) for i, name in enumerate(label_names)
        })
        print(pd.DataFrame(cm, index=label_names, columns=label_names))
        
        # Store fold results (including ROC/PR metrics)
        fold_row = {
            "fold": fold_idx,
            "test_rows": len(test_idx),
            **{f"test_{k}": v for k, v in metrics.items()},
            **{f"test_{k}": v for k, v in roc_pr_metrics.items()}
        }
        fold_rows.append(fold_row)
        
        # Save fold-specific results
        with open(out_dir / f"metrics_fold{fold_idx}.json", "w") as f:
            json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in fold_row.items()}, f, indent=2)
    
    # 6. Aggregate results
    df_metrics = pd.DataFrame(fold_rows)
    df_metrics.to_csv(out_dir / "shuffle_metrics_folds.csv", index=False)
    
    # Calculate aggregate metrics including ROC/PR
    mean_metrics = {}
    std_metrics = {}
    for metric in ["test_accuracy", "test_macro_f1", "test_splice_accuracy", "test_splice_macro_f1", 
                   "test_macro_roc_auc", "test_macro_ap"]:
        values = df_metrics[metric].dropna()
        if len(values) > 0:
            mean_metrics[metric] = values.mean()
            std_metrics[metric] = values.std()
        else:
            mean_metrics[metric] = np.nan
            std_metrics[metric] = np.nan
    
    print(f"\n[Shuffle Label Sanity] === RESULTS (should be near chance levels) ===")
    print(f"Overall Accuracy:      {mean_metrics['test_accuracy']:.3f} ± {std_metrics['test_accuracy']:.3f}")
    print(f"Overall Macro F1:      {mean_metrics['test_macro_f1']:.3f} ± {std_metrics['test_macro_f1']:.3f}")
    print(f"Splice-only Accuracy:  {mean_metrics['test_splice_accuracy']:.3f} ± {std_metrics['test_splice_accuracy']:.3f}")
    print(f"Splice-only Macro F1:  {mean_metrics['test_splice_macro_f1']:.3f} ± {std_metrics['test_splice_macro_f1']:.3f}")
    print(f"Macro ROC-AUC:         {mean_metrics['test_macro_roc_auc']:.3f} ± {std_metrics['test_macro_roc_auc']:.3f}")
    print(f"Macro AP:              {mean_metrics['test_macro_ap']:.3f} ± {std_metrics['test_macro_ap']:.3f}")
    
    # Expected chance levels
    class_distribution = np.bincount(y) / len(y)
    expected_accuracy = class_distribution.max()  # Majority class accuracy
    expected_macro_f1 = 1.0 / 3.0  # Random classifier macro F1 for 3 classes
    expected_splice_accuracy = 0.5  # Random binary accuracy for splice vs non-splice
    expected_splice_macro_f1 = 0.5  # Random binary F1 for donor vs acceptor
    expected_roc_auc = 0.5  # Random classifier ROC-AUC
    expected_ap_macro = np.mean(class_distribution)  # Random AP is class prevalence
    
    print(f"\n[Shuffle Label Sanity] === EXPECTED CHANCE LEVELS ===")
    print(f"Expected Overall Accuracy (majority class): {expected_accuracy:.3f}")
    print(f"Expected Overall Macro F1 (random):         {expected_macro_f1:.3f}")
    print(f"Expected Splice Accuracy (random binary):   {expected_splice_accuracy:.3f}")
    print(f"Expected Splice Macro F1 (random binary):   {expected_splice_macro_f1:.3f}")
    print(f"Expected ROC-AUC (random):                  {expected_roc_auc:.3f}")
    print(f"Expected AP (class prevalence):             {expected_ap_macro:.3f}")
    
    # Class distribution info
    print(f"\n[Shuffle Label Sanity] === CLASS DISTRIBUTION ===")
    for cls_idx, prob in enumerate(class_distribution):
        print(f"{LABEL_NAMES[cls_idx]}: {prob:.3f} ({prob*100:.1f}%)")
    
    # Sanity check: flag if performance is significantly above chance
    acc_above_chance = mean_metrics['test_accuracy'] > expected_accuracy + 0.1
    f1_above_chance = mean_metrics['test_macro_f1'] > expected_macro_f1 + 0.1
    auc_above_chance = mean_metrics['test_macro_roc_auc'] > expected_roc_auc + 0.1
    ap_above_chance = mean_metrics['test_macro_ap'] > expected_ap_macro + 0.1
    
    if acc_above_chance or f1_above_chance or auc_above_chance or ap_above_chance:
        print(f"\n[Shuffle Label Sanity] ⚠️  WARNING: Performance significantly above chance!")
        print(f"This may indicate label leakage or other issues in the pipeline.")
        if auc_above_chance:
            print(f"  ROC-AUC above chance: {mean_metrics['test_macro_roc_auc']:.3f} > {expected_roc_auc + 0.1:.3f}")
        if ap_above_chance:
            print(f"  AP above chance: {mean_metrics['test_macro_ap']:.3f} > {expected_ap_macro + 0.1:.3f}")
    else:
        print(f"\n[Shuffle Label Sanity] ✅ Performance at chance levels - pipeline appears healthy")
    
    # Generate ROC and PR curve plots
    print(f"\n[Shuffle Label Sanity] Generating ROC and PR curves...")
    _plot_roc_pr_curves(all_y_true, all_y_proba, out_dir, args.plot_format)
    
    # Save aggregate results (including ROC/PR metrics)
    aggregate_results = {
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "expected_chance_levels": {
            "overall_accuracy": expected_accuracy,
            "overall_macro_f1": expected_macro_f1,
            "splice_accuracy": expected_splice_accuracy,
            "splice_macro_f1": expected_splice_macro_f1,
            "roc_auc": expected_roc_auc,
            "ap": expected_ap_macro
        },
        "class_distribution": {
            LABEL_NAMES[i]: float(class_distribution[i]) for i in range(3)
        },
        "sanity_check": {
            "accuracy_above_chance": bool(acc_above_chance),
            "f1_above_chance": bool(f1_above_chance),
            "auc_above_chance": bool(auc_above_chance),
            "ap_above_chance": bool(ap_above_chance),
            "appears_healthy": not (acc_above_chance or f1_above_chance or auc_above_chance or ap_above_chance)
        },
        "dataset_info": {
            "n_samples": len(X),
            "n_features": len(feature_names),
            "n_folds": args.n_folds,
            "excluded_features": len(exclude_list) if exclude_list else 0
        }
    }
    
    with open(out_dir / "shuffle_sanity_multiclass_summary.json", "w") as f:
        json.dump(aggregate_results, f, indent=2)
    
    # Generate performance plots (existing bar charts)
    if args.plot_format:
        print(f"[Shuffle Label Sanity] Generating performance plots...")
        
        plt.figure(figsize=(20, 10))
        
        # Per-fold accuracy plot
        plt.subplot(2, 3, 1)
        fold_accuracies = [row['test_accuracy'] for row in fold_rows]
        plt.bar(range(1, args.n_folds + 1), fold_accuracies, alpha=0.7)
        plt.axhline(y=expected_accuracy, color='r', linestyle='--', 
                   label=f'Expected (majority): {expected_accuracy:.3f}')
        plt.axhline(y=mean_metrics['test_accuracy'], color='g', linestyle='-', 
                   label=f'Mean: {mean_metrics["test_accuracy"]:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.title('Per-Fold Accuracy (Shuffled Labels)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Per-fold macro F1 plot
        plt.subplot(2, 3, 2)
        fold_f1s = [row['test_macro_f1'] for row in fold_rows]
        plt.bar(range(1, args.n_folds + 1), fold_f1s, alpha=0.7, color='orange')
        plt.axhline(y=expected_macro_f1, color='r', linestyle='--', 
                   label=f'Expected (random): {expected_macro_f1:.3f}')
        plt.axhline(y=mean_metrics['test_macro_f1'], color='g', linestyle='-', 
                   label=f'Mean: {mean_metrics["test_macro_f1"]:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Macro F1')
        plt.title('Per-Fold Macro F1 (Shuffled Labels)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Per-fold ROC-AUC plot
        plt.subplot(2, 3, 3)
        fold_aucs = [row['test_macro_roc_auc'] for row in fold_rows]
        plt.bar(range(1, args.n_folds + 1), fold_aucs, alpha=0.7, color='blue')
        plt.axhline(y=expected_roc_auc, color='r', linestyle='--', 
                   label=f'Expected (random): {expected_roc_auc:.3f}')
        plt.axhline(y=mean_metrics['test_macro_roc_auc'], color='g', linestyle='-', 
                   label=f'Mean: {mean_metrics["test_macro_roc_auc"]:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Macro ROC-AUC')
        plt.title('Per-Fold Macro ROC-AUC (Shuffled Labels)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Per-fold AP plot
        plt.subplot(2, 3, 4)
        fold_aps = [row['test_macro_ap'] for row in fold_rows]
        plt.bar(range(1, args.n_folds + 1), fold_aps, alpha=0.7, color='purple')
        plt.axhline(y=expected_ap_macro, color='r', linestyle='--', 
                   label=f'Expected (prevalence): {expected_ap_macro:.3f}')
        plt.axhline(y=mean_metrics['test_macro_ap'], color='g', linestyle='-', 
                   label=f'Mean: {mean_metrics["test_macro_ap"]:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Macro AP')
        plt.title('Per-Fold Macro AP (Shuffled Labels)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Splice-only metrics
        plt.subplot(2, 3, 5)
        fold_splice_f1s = [row.get('test_splice_macro_f1', np.nan) for row in fold_rows]
        valid_splice_f1s = [f1 for f1 in fold_splice_f1s if not np.isnan(f1)]
        if valid_splice_f1s:
            plt.bar(range(1, len(valid_splice_f1s) + 1), valid_splice_f1s, alpha=0.7, color='green')
            plt.axhline(y=expected_splice_macro_f1, color='r', linestyle='--', 
                       label=f'Expected: {expected_splice_macro_f1:.3f}')
            if not np.isnan(mean_metrics['test_splice_macro_f1']):
                plt.axhline(y=mean_metrics['test_splice_macro_f1'], color='g', linestyle='-', 
                           label=f'Mean: {mean_metrics["test_splice_macro_f1"]:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Splice F1')
        plt.title('Per-Fold Splice F1 (Shuffled Labels)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Summary comparison
        plt.subplot(2, 3, 6)
        metrics_comparison = ['Accuracy', 'Macro F1', 'ROC-AUC', 'AP']
        observed_values = [mean_metrics['test_accuracy'], mean_metrics['test_macro_f1'], 
                          mean_metrics['test_macro_roc_auc'], mean_metrics['test_macro_ap']]
        expected_values = [expected_accuracy, expected_macro_f1, expected_roc_auc, expected_ap_macro]
        
        x_pos = np.arange(len(metrics_comparison))
        width = 0.35
        
        plt.bar(x_pos - width/2, observed_values, width, label='Observed (Shuffled)', alpha=0.7)
        plt.bar(x_pos + width/2, expected_values, width, label='Expected (Chance)', alpha=0.7)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Observed vs Expected Performance\n(Shuffled Labels)')
        plt.xticks(x_pos, metrics_comparison, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.suptitle('Shuffle Label Sanity Check - Performance Should Be at Chance Levels', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_dir / f"shuffle_sanity_multiclass_plots.{args.plot_format}", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Train final model for optional diagnostics
    if args.run_diagnostics:
        print(f"\n[Shuffle Label Sanity] Training final model for diagnostics...")
        
        final_model = XGBClassifier(
            n_estimators=args.n_estimators,
            tree_method=args.tree_method,
            max_bin=args.max_bin if args.tree_method in {"hist", "gpu_hist"} else None,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=args.seed,
            n_jobs=-1,
            device=args.device if args.device != "auto" else None,
        )
        final_model.fit(X, y_shuffled)
        
        # Save model
        import pickle
        with open(out_dir / "model_shuffled_multiclass.pkl", "wb") as f:
            pickle.dump(final_model, f)
        
        print(f"[Shuffle Label Sanity] ✓ Model saved for further diagnostics")
    
    print(f"\n[Shuffle Label Sanity] Analysis complete. Results saved to: {out_dir}")
    print(f"[Shuffle Label Sanity] Summary: shuffle_sanity_multiclass_summary.json")
    print(f"[Shuffle Label Sanity] ROC/PR Curves: roc_pr_curves_shuffled.{args.plot_format}")
    print(f"[Shuffle Label Sanity] Performance Plots: shuffle_sanity_multiclass_plots.{args.plot_format}")


if __name__ == "__main__":
    main()
