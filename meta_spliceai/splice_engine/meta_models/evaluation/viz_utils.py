#!/usr/bin/env python3
"""
Visualization utilities for model evaluation.

This module provides reusable plotting functions for model evaluation,
including ROC curves, precision-recall curves, and other diagnostic
visualizations.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    auc, average_precision_score, f1_score
)


def plot_roc_pr_curves(
    y_true: List[np.ndarray],
    y_pred_base: List[np.ndarray],
    y_pred_meta: List[np.ndarray],
    out_dir: Union[str, Path],
    n_roc_points: int = 101,
    plot_format: str = "pdf",
    base_name: str = "Base",
    meta_name: str = "Meta",
    fold_ids: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Plot ROC and PR curves for base and meta models across CV folds.
    
    Args:
        y_true: List of true binary labels for each fold
        y_pred_base: List of base model predictions for each fold
        y_pred_meta: List of meta model predictions for each fold
        out_dir: Output directory for plots
        n_roc_points: Number of points to sample for curve interpolation
        plot_format: File format for plots (pdf, png, svg)
        base_name: Name to use for base model in legends
        meta_name: Name to use for meta model in legends
        fold_ids: Optional list of fold IDs for detailed logging
        
    Returns:
        Dictionary of metrics containing AUC and AP values for base and meta models
    """
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)
    
    # Ensure we have lists of arrays
    if not fold_ids:
        fold_ids = list(range(len(y_true)))
        
    # Calculate and collect ROC and PR curve data for all folds
    roc_base, roc_meta = [], []  # Lists of (fpr, tpr) arrays per fold
    pr_base, pr_meta = [], []    # Lists of (recall, precision) arrays per fold
    auc_base, auc_meta = [], []  # AUC values per fold
    ap_base, ap_meta = [], []    # Average precision values per fold
    
    # Process each fold
    for i, (y_t, y_pb, y_pm) in enumerate(zip(y_true, y_pred_base, y_pred_meta)):
        fold = fold_ids[i] if fold_ids else i
        
        # Base model ROC and PR
        fpr_b, tpr_b, _ = roc_curve(y_t, y_pb)
        prec_b, rec_b, _ = precision_recall_curve(y_t, y_pb)
        roc_base.append(np.column_stack([fpr_b, tpr_b]))
        pr_base.append(np.column_stack([rec_b, prec_b]))
        auc_base.append(auc(fpr_b, tpr_b))
        ap_base.append(average_precision_score(y_t, y_pb))
        
        # Meta model ROC and PR
        fpr_m, tpr_m, _ = roc_curve(y_t, y_pm)
        prec_m, rec_m, _ = precision_recall_curve(y_t, y_pm)
        roc_meta.append(np.column_stack([fpr_m, tpr_m]))
        pr_meta.append(np.column_stack([rec_m, prec_m]))
        auc_meta.append(auc(fpr_m, tpr_m))
        ap_meta.append(average_precision_score(y_t, y_pm))
    
    # Generate plots
    x_roc = np.linspace(0, 1, n_roc_points)
    x_pr = np.linspace(0, 1, n_roc_points)
    
    # Interpolate curves to common x-axis points
    tpr_b = interp_curves(roc_base, x_roc)
    tpr_m = interp_curves(roc_meta, x_roc)
    pr_b = interp_curves(pr_base, x_pr)   # y = precision, x = recall
    pr_m = interp_curves(pr_meta, x_pr)
    
    # ROC curves
    plt.figure(figsize=(8, 6))
    for xy in roc_base:
        plt.plot(xy[:,0], xy[:,1], color='grey', alpha=0.3, lw=0.8)
    plot_mean_band(x_roc, tpr_b, f'{base_name} (mean AUC={np.mean(auc_base):.3f})', 'tab:blue')
    plot_mean_band(x_roc, tpr_m, f'{meta_name} (mean AUC={np.mean(auc_meta):.3f})', 'tab:orange')
    plt.plot([0,1], [0,1], 'k--', lw=0.8)
    plt.xlabel('False-Positive Rate')
    plt.ylabel('True-Positive Rate')
    plt.title('ROC – per fold & mean ±1 SD')
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_dir / f"roc_base_vs_meta.{plot_format}", dpi=300)
    plt.close()
    
    # PR curves
    plt.figure(figsize=(8, 6))
    for xy in pr_base:
        plt.plot(xy[:,0], xy[:,1], color='grey', alpha=0.3, lw=0.8)
    plot_mean_band(x_pr, pr_b, f'{base_name} (mean AP={np.mean(ap_base):.3f})', 'tab:blue')
    plot_mean_band(x_pr, pr_m, f'{meta_name} (mean AP={np.mean(ap_meta):.3f})', 'tab:orange')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall – per fold & mean ±1 SD')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_dir / f"pr_base_vs_meta.{plot_format}", dpi=300)
    plt.close()
    
    # Create metrics summary
    metrics = {
        "auc": {
            "base": {"values": auc_base, "mean": float(np.mean(auc_base)), "std": float(np.std(auc_base))},
            "meta": {"values": auc_meta, "mean": float(np.mean(auc_meta)), "std": float(np.std(auc_meta))},
        },
        "ap": {
            "base": {"values": ap_base, "mean": float(np.mean(ap_base)), "std": float(np.std(ap_base))},
            "meta": {"values": ap_meta, "mean": float(np.mean(ap_meta)), "std": float(np.std(ap_meta))},
        }
    }
    
    return metrics


def plot_roc_pr_curves_f1(
    y_true: List[np.ndarray],
    y_pred_base: List[np.ndarray],
    y_pred_meta: List[np.ndarray],
    out_dir: Union[str, Path],
    n_roc_points: int = 101,
    plot_format: str = "pdf",
    base_name: str = "Base",
    meta_name: str = "Meta",
    fold_ids: Optional[List[int]] = None,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Plot ROC and PR curves for base and meta models across CV folds with F1 scores.
    
    This function is similar to plot_roc_pr_curves but uses F1 scores instead of AP
    for the PR curve summary metrics.
    
    Args:
        y_true: List of true binary labels for each fold
        y_pred_base: List of base model predictions for each fold
        y_pred_meta: List of meta model predictions for each fold
        out_dir: Output directory for plots
        n_roc_points: Number of points to sample for curve interpolation
        plot_format: File format for plots (pdf, png, svg)
        base_name: Name to use for base model in legends
        meta_name: Name to use for meta model in legends
        fold_ids: Optional list of fold IDs for detailed logging
        threshold: Threshold for calculating F1 scores (default: 0.5)
        
    Returns:
        Dictionary of metrics containing AUC and F1 values for base and meta models
    """
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)
    
    # Ensure we have lists of arrays
    if not fold_ids:
        fold_ids = list(range(len(y_true)))
        
    # Calculate and collect ROC and PR curve data for all folds
    roc_base, roc_meta = [], []  # Lists of (fpr, tpr) arrays per fold
    pr_base, pr_meta = [], []    # Lists of (recall, precision) arrays per fold
    auc_base, auc_meta = [], []  # AUC values per fold
    f1_base, f1_meta = [], []    # F1 values per fold
    
    # Process each fold
    for i, (y_t, y_pb, y_pm) in enumerate(zip(y_true, y_pred_base, y_pred_meta)):
        fold = fold_ids[i] if fold_ids else i
        
        # Base model ROC and PR
        fpr_b, tpr_b, _ = roc_curve(y_t, y_pb)
        prec_b, rec_b, _ = precision_recall_curve(y_t, y_pb)
        roc_base.append(np.column_stack([fpr_b, tpr_b]))
        pr_base.append(np.column_stack([rec_b, prec_b]))
        auc_base.append(auc(fpr_b, tpr_b))
        
        # Calculate F1 score for base model at fixed threshold
        y_pred_binary_b = (y_pb >= threshold).astype(int)
        f1_base.append(f1_score(y_t, y_pred_binary_b))
        
        # Meta model ROC and PR
        fpr_m, tpr_m, _ = roc_curve(y_t, y_pm)
        prec_m, rec_m, _ = precision_recall_curve(y_t, y_pm)
        roc_meta.append(np.column_stack([fpr_m, tpr_m]))
        pr_meta.append(np.column_stack([rec_m, prec_m]))
        auc_meta.append(auc(fpr_m, tpr_m))
        
        # Calculate F1 score for meta model at fixed threshold
        y_pred_binary_m = (y_pm >= threshold).astype(int)
        f1_meta.append(f1_score(y_t, y_pred_binary_m))
    
    # Generate plots
    x_roc = np.linspace(0, 1, n_roc_points)
    x_pr = np.linspace(0, 1, n_roc_points)
    
    # Interpolate curves to common x-axis points
    tpr_b = interp_curves(roc_base, x_roc)
    tpr_m = interp_curves(roc_meta, x_roc)
    pr_b = interp_curves(pr_base, x_pr)   # y = precision, x = recall
    pr_m = interp_curves(pr_meta, x_pr)
    
    # ROC curves
    plt.figure(figsize=(8, 6))
    for xy in roc_base:
        plt.plot(xy[:,0], xy[:,1], color='grey', alpha=0.3, lw=0.8)
    plot_mean_band(x_roc, tpr_b, f'{base_name} (mean AUC={np.mean(auc_base):.3f})', 'tab:blue')
    plot_mean_band(x_roc, tpr_m, f'{meta_name} (mean AUC={np.mean(auc_meta):.3f})', 'tab:orange')
    plt.plot([0,1], [0,1], 'k--', lw=0.8)
    plt.xlabel('False-Positive Rate')
    plt.ylabel('True-Positive Rate')
    plt.title('ROC – per fold & mean ±1 SD')
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_dir / f"roc_base_vs_meta_f1.{plot_format}", dpi=300)
    plt.close()
    
    # PR curves with F1 scores
    plt.figure(figsize=(8, 6))
    for xy in pr_base:
        plt.plot(xy[:,0], xy[:,1], color='grey', alpha=0.3, lw=0.8)
    plot_mean_band(x_pr, pr_b, f'{base_name} (mean F1={np.mean(f1_base):.3f})', 'tab:blue')
    plot_mean_band(x_pr, pr_m, f'{meta_name} (mean F1={np.mean(f1_meta):.3f})', 'tab:orange')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall – per fold & mean ±1 SD (F1 @ threshold={threshold})')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(out_dir / f"pr_base_vs_meta_f1.{plot_format}", dpi=300)
    plt.close()
    
    # Create metrics summary
    metrics = {
        "auc": {
            "base": {"values": auc_base, "mean": float(np.mean(auc_base)), "std": float(np.std(auc_base))},
            "meta": {"values": auc_meta, "mean": float(np.mean(auc_meta)), "std": float(np.std(auc_meta))},
        },
        "f1": {
            "base": {"values": f1_base, "mean": float(np.mean(f1_base)), "std": float(np.std(f1_base))},
            "meta": {"values": f1_meta, "mean": float(np.mean(f1_meta)), "std": float(np.std(f1_meta))},
        }
    }
    
    return metrics


def plot_meta_roc_curves(
    y_true: List[np.ndarray],
    y_pred_base: List[np.ndarray],
    y_pred_meta: List[np.ndarray],
    out_dir: Union[str, Path],
    plot_format: str = "pdf",
    base_name: str = "Base",
    meta_name: str = "Meta",
    n_roc_points: int = 101,
) -> None:
    """
    Create ROC curves plot for meta model evaluation.
    
    Parameters
    ----------
    y_true : List[np.ndarray]
        List of true binary labels for each fold
    y_pred_base : List[np.ndarray]
        List of base model predictions for each fold
    y_pred_meta : List[np.ndarray]
        List of meta model predictions for each fold
    out_dir : Union[str, Path]
        Output directory for the plot
    plot_format : str
        File format for the plot (pdf, png, svg)
    base_name : str
        Name for base model in legend
    meta_name : str
        Name for meta model in legend  
    n_roc_points : int
        Number of points for curve interpolation
    """
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)
    
    # Calculate metrics for all folds
    auc_base, auc_meta = [], []
    roc_base, roc_meta = [], []
    
    for y_t, y_pb, y_pm in zip(y_true, y_pred_base, y_pred_meta):
        # Base model ROC
        fpr_b, tpr_b, _ = roc_curve(y_t, y_pb)
        roc_base.append(np.column_stack([fpr_b, tpr_b]))
        auc_base.append(auc(fpr_b, tpr_b))
        
        # Meta model ROC  
        fpr_m, tpr_m, _ = roc_curve(y_t, y_pm)
        roc_meta.append(np.column_stack([fpr_m, tpr_m]))
        auc_meta.append(auc(fpr_m, tpr_m))
    
    # Create ROC plot
    plt.figure(figsize=(8, 6))
    
    x_roc = np.linspace(0, 1, n_roc_points)
    
    # Plot individual fold curves (light)
    for xy in roc_base:
        plt.plot(xy[:,0], xy[:,1], color='lightblue', alpha=0.3, lw=0.8)
    for xy in roc_meta:
        plt.plot(xy[:,0], xy[:,1], color='lightcoral', alpha=0.3, lw=0.8)
    
    # Interpolate and plot mean curves
    tpr_b = interp_curves(roc_base, x_roc)
    tpr_m = interp_curves(roc_meta, x_roc)
    
    plot_mean_band(x_roc, tpr_b, f'{base_name} (AUC={np.mean(auc_base):.3f}±{np.std(auc_base):.3f})', 'tab:blue')
    plot_mean_band(x_roc, tpr_m, f'{meta_name} (AUC={np.mean(auc_meta):.3f}±{np.std(auc_meta):.3f})', 'tab:orange')
    
    plt.plot([0,1], [0,1], 'k--', lw=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Meta Model Performance')
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    plt.savefig(out_dir / f"roc_curves_meta.{plot_format}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Enhanced Visualizations] ROC curves saved to: roc_curves_meta.{plot_format}")


def plot_meta_pr_curves(
    y_true: List[np.ndarray],
    y_pred_base: List[np.ndarray],
    y_pred_meta: List[np.ndarray],
    out_dir: Union[str, Path],
    plot_format: str = "pdf",
    base_name: str = "Base",
    meta_name: str = "Meta",
    n_roc_points: int = 101,
) -> None:
    """
    Create PR curves plot for meta model evaluation.
    
    Parameters
    ----------
    y_true : List[np.ndarray]
        List of true binary labels for each fold
    y_pred_base : List[np.ndarray]
        List of base model predictions for each fold
    y_pred_meta : List[np.ndarray]
        List of meta model predictions for each fold
    out_dir : Union[str, Path]
        Output directory for the plot
    plot_format : str
        File format for the plot (pdf, png, svg)
    base_name : str
        Name for base model in legend
    meta_name : str
        Name for meta model in legend  
    n_roc_points : int
        Number of points for curve interpolation
    """
    if not isinstance(out_dir, Path):
        out_dir = Path(out_dir)
    
    # Calculate metrics for all folds
    ap_base, ap_meta = [], []
    pr_base, pr_meta = [], []
    
    for y_t, y_pb, y_pm in zip(y_true, y_pred_base, y_pred_meta):
        # Base model PR
        prec_b, rec_b, _ = precision_recall_curve(y_t, y_pb)
        pr_base.append(np.column_stack([rec_b, prec_b]))
        ap_base.append(average_precision_score(y_t, y_pb))
        
        # Meta model PR  
        prec_m, rec_m, _ = precision_recall_curve(y_t, y_pm)
        pr_meta.append(np.column_stack([rec_m, prec_m]))
        ap_meta.append(average_precision_score(y_t, y_pm))
    
    # Create PR plot
    plt.figure(figsize=(8, 6))
    
    x_pr = np.linspace(0, 1, n_roc_points)
    
    # Plot individual fold curves (light)
    for xy in pr_base:
        plt.plot(xy[:,0], xy[:,1], color='lightblue', alpha=0.3, lw=0.8)
    for xy in pr_meta:
        plt.plot(xy[:,0], xy[:,1], color='lightcoral', alpha=0.3, lw=0.8)
    
    # Interpolate and plot mean curves
    pr_b = interp_curves(pr_base, x_pr)
    pr_m = interp_curves(pr_meta, x_pr)
    
    plot_mean_band(x_pr, pr_b, f'{base_name} (AP={np.mean(ap_base):.3f}±{np.std(ap_base):.3f})', 'tab:blue')
    plot_mean_band(x_pr, pr_m, f'{meta_name} (AP={np.mean(ap_meta):.3f}±{np.std(ap_meta):.3f})', 'tab:orange')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Meta Model Performance')
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plt.savefig(out_dir / f"pr_curves_meta.{plot_format}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Enhanced Visualizations] PR curves saved to: pr_curves_meta.{plot_format}")


# Keep the combined function for backward compatibility, but rename the output
def plot_combined_roc_pr_curves_meta(
    y_true: List[np.ndarray],
    y_pred_base: List[np.ndarray],
    y_pred_meta: List[np.ndarray],
    out_dir: Union[str, Path],
    plot_format: str = "pdf",
    base_name: str = "Base",
    meta_name: str = "Meta",
    n_roc_points: int = 101,
) -> None:
    """
    Create separate ROC and PR curves plots for meta model evaluation.
    
    This function now creates two separate files instead of one combined file.
    """
    # Create separate ROC and PR curves
    plot_meta_roc_curves(
        y_true, y_pred_base, y_pred_meta, out_dir, plot_format, 
        base_name, meta_name, n_roc_points
    )
    
    plot_meta_pr_curves(
        y_true, y_pred_base, y_pred_meta, out_dir, plot_format,
        base_name, meta_name, n_roc_points
    )


def interp_curves(curves: List[np.ndarray], x_new: np.ndarray) -> List[np.ndarray]:
    """Interpolate curves to common x-axis points."""
    interp_y = []
    for curve in curves:
        x_orig = curve[:, 0]
        y_orig = curve[:, 1]
        # Handle duplicate x values by using the last occurrence
        _, unique_idx = np.unique(x_orig, return_index=True)
        x_unique = x_orig[unique_idx]
        y_unique = y_orig[unique_idx]
        y_interp = np.interp(x_new, x_unique, y_unique)
        interp_y.append(y_interp)
    return interp_y


def plot_mean_band(x: np.ndarray, y_values: List[np.ndarray], label: str, color: str, ax=None) -> None:
    """Plot mean curve with confidence band."""
    if ax is None:
        ax = plt.gca()
        
    y_mean = np.mean(y_values, axis=0)
    y_std = np.std(y_values, axis=0)
    
    ax.plot(x, y_mean, color=color, lw=2.5, label=label)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)


def check_feature_correlations(
    X: np.ndarray,
    y: np.ndarray, 
    feature_names: List[str],
    threshold: float = 0.95,
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[List[str], pd.DataFrame]:
    """
    Check for potentially leaky features by calculating correlations with labels.
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: Names of features
        threshold: Correlation threshold for detecting leaky features
        output_path: Optional path to save the correlation report
        
    Returns:
        Tuple of (leaky_feature_names, correlation_df)
    """
    # Calculate correlations between features and target
    correlations = []
    for i, feature_name in enumerate(feature_names):
        if len(np.unique(X[:, i])) <= 1:
            # Skip constant features
            continue
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append((feature_name, abs(corr), corr))
    
    # Create and sort correlation dataframe
    corr_df = pd.DataFrame(correlations, columns=['feature', 'abs_correlation', 'correlation'])
    corr_df = corr_df.sort_values('abs_correlation', ascending=False).reset_index(drop=True)
    
    # Identify potentially leaky features
    leaky_features = corr_df[corr_df['abs_correlation'] >= threshold]
    leaky_feature_names = leaky_features['feature'].tolist()
    
    # Print warnings
    if len(leaky_features) > 0:
        print(f"[WARNING] Found {len(leaky_features)} potentially leaky features with correlation >= {threshold}:")
        for _, row in leaky_features.iterrows():
            print(f"  - {row['feature']}: correlation = {row['correlation']:.4f}")
    
    # Save report if path provided
    if output_path:
        if not isinstance(output_path, Path):
            output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        corr_df.to_csv(output_path, index=False)
        print(f"Saved correlation report to {output_path}")
    
    return leaky_feature_names, corr_df
