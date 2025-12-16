#!/usr/bin/env python3
"""
Multi-class ROC and PR curve utilities for 3-way splice site classification.

This module provides enhanced visualization and evaluation for the donor/acceptor/neither
classification problem, going beyond the standard binary splice/non-splice evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, average_precision_score,
    roc_auc_score
)


def plot_multiclass_roc_pr_curves(
    y_true: List[np.ndarray],
    y_pred_base: List[np.ndarray], 
    y_pred_meta: List[np.ndarray],
    out_dir: Union[str, Path],
    class_names: List[str] = ["Neither", "Donor", "Acceptor"],
    plot_format: str = "pdf",
    n_roc_points: int = 101,
    base_name: str = "Base",
    meta_name: str = "Meta"
) -> Dict[str, Dict[str, float]]:
    """
    Plot separate ROC and PR curves for each class in the 3-way classification.
    
    Parameters
    ----------
    y_true : List[np.ndarray]
        List of true labels for each fold (values 0, 1, 2)
    y_pred_base : List[np.ndarray] 
        List of base model probability matrices, shape (n_samples, 3) per fold
    y_pred_meta : List[np.ndarray]
        List of meta model probability matrices, shape (n_samples, 3) per fold
    out_dir : str or Path
        Output directory for plots
    class_names : List[str]
        Names for the three classes [class_0, class_1, class_2]
    plot_format : str
        File format for plots (pdf, png, svg)
    n_roc_points : int
        Number of points for curve interpolation
        
    Returns
    -------
    Dict
        Nested dictionary with AUC and AP metrics for each class and model
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Colors for the three classes
    colors = ['tab:green', 'tab:red', 'tab:blue']  # Neither, Donor, Acceptor
    
    metrics = {
        "auc": {"base": {}, "meta": {}},
        "ap": {"base": {}, "meta": {}}
    }
    
    # Create separate plots for each class
    for class_idx, (class_name, color) in enumerate(zip(class_names, colors)):
        print(f"Processing class {class_idx}: {class_name}")
        
        # Collect metrics for this class across all folds
        auc_base_folds, auc_meta_folds = [], []
        ap_base_folds, ap_meta_folds = [], []
        roc_base_folds, roc_meta_folds = [], []
        pr_base_folds, pr_meta_folds = [], []
        
        for fold_idx, (y_t, y_pb, y_pm) in enumerate(zip(y_true, y_pred_base, y_pred_meta)):
            # Create binary labels for this class (one-vs-rest)
            y_binary = (y_t == class_idx).astype(int)
            
            # Get probabilities for this class
            prob_base = y_pb[:, class_idx]
            prob_meta = y_pm[:, class_idx]
            
            # Calculate ROC curves
            fpr_b, tpr_b, _ = roc_curve(y_binary, prob_base)
            fpr_m, tpr_m, _ = roc_curve(y_binary, prob_meta)
            roc_base_folds.append(np.column_stack([fpr_b, tpr_b]))
            roc_meta_folds.append(np.column_stack([fpr_m, tpr_m]))
            
            # Calculate PR curves
            prec_b, rec_b, _ = precision_recall_curve(y_binary, prob_base)
            prec_m, rec_m, _ = precision_recall_curve(y_binary, prob_meta)
            pr_base_folds.append(np.column_stack([rec_b, prec_b]))
            pr_meta_folds.append(np.column_stack([rec_m, prec_m]))
            
            # Calculate AUC and AP
            auc_base_folds.append(auc(fpr_b, tpr_b))
            auc_meta_folds.append(auc(fpr_m, tpr_m))
            ap_base_folds.append(average_precision_score(y_binary, prob_base))
            ap_meta_folds.append(average_precision_score(y_binary, prob_meta))
        
        # Store metrics
        metrics["auc"]["base"][class_name.lower()] = {
            "values": auc_base_folds,
            "mean": float(np.mean(auc_base_folds)),
            "std": float(np.std(auc_base_folds))
        }
        metrics["auc"]["meta"][class_name.lower()] = {
            "values": auc_meta_folds, 
            "mean": float(np.mean(auc_meta_folds)),
            "std": float(np.std(auc_meta_folds))
        }
        metrics["ap"]["base"][class_name.lower()] = {
            "values": ap_base_folds,
            "mean": float(np.mean(ap_base_folds)),
            "std": float(np.std(ap_base_folds))
        }
        metrics["ap"]["meta"][class_name.lower()] = {
            "values": ap_meta_folds,
            "mean": float(np.mean(ap_meta_folds)), 
            "std": float(np.std(ap_meta_folds))
        }
        
        # Create ROC plot for this class
        plt.figure(figsize=(8, 6))
        
        # Plot individual fold curves in light colors
        for roc_curve_fold in roc_base_folds:
            plt.plot(roc_curve_fold[:,0], roc_curve_fold[:,1], 
                    color='lightblue', alpha=0.3, lw=0.8)
        for roc_curve_fold in roc_meta_folds:
            plt.plot(roc_curve_fold[:,0], roc_curve_fold[:,1], 
                    color='lightcoral', alpha=0.3, lw=0.8)
        
        # Plot mean curves with improved visibility
        x_interp = np.linspace(0, 1, n_roc_points)
        
        # Interpolate base model curves
        tpr_base_interp = []
        for roc_curve_fold in roc_base_folds:
            tpr_base_interp.append(np.interp(x_interp, roc_curve_fold[:,0], roc_curve_fold[:,1]))
        tpr_base_mean = np.mean(tpr_base_interp, axis=0)
        tpr_base_std = np.std(tpr_base_interp, axis=0)
        
        # Interpolate meta model curves  
        tpr_meta_interp = []
        for roc_curve_fold in roc_meta_folds:
            tpr_meta_interp.append(np.interp(x_interp, roc_curve_fold[:,0], roc_curve_fold[:,1]))
        tpr_meta_mean = np.mean(tpr_meta_interp, axis=0)
        tpr_meta_std = np.std(tpr_meta_interp, axis=0)
        
        # Plot mean curves with error bands
        plt.plot(x_interp, tpr_base_mean, 'b-', lw=2.5, 
                label=f'{base_name} (AUC = {np.mean(auc_base_folds):.3f} ± {np.std(auc_base_folds):.3f})')
        plt.fill_between(x_interp, tpr_base_mean - tpr_base_std, tpr_base_mean + tpr_base_std, 
                        color='blue', alpha=0.2)
        
        plt.plot(x_interp, tpr_meta_mean, 'r-', lw=2.5,
                label=f'{meta_name} (AUC = {np.mean(auc_meta_folds):.3f} ± {np.std(auc_meta_folds):.3f})')
        plt.fill_between(x_interp, tpr_meta_mean - tpr_meta_std, tpr_meta_mean + tpr_meta_std,
                        color='red', alpha=0.2)
        
        # Diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.8)
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves: {class_name} Classification')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save ROC plot
        roc_path = out_dir / f"roc_{class_name.lower()}_class.{plot_format}"
        plt.savefig(roc_path, dpi=300)
        plt.close()
        
        # Create PR plot for this class
        plt.figure(figsize=(8, 6))
        
        # Plot individual fold curves
        for pr_curve_fold in pr_base_folds:
            plt.plot(pr_curve_fold[:,0], pr_curve_fold[:,1], 
                    color='lightblue', alpha=0.3, lw=0.8)
        for pr_curve_fold in pr_meta_folds:
            plt.plot(pr_curve_fold[:,0], pr_curve_fold[:,1], 
                    color='lightcoral', alpha=0.3, lw=0.8)
        
        # Interpolate PR curves (more complex due to precision-recall relationship)
        recall_interp = np.linspace(0, 1, n_roc_points)
        
        # Interpolate base model PR curves
        prec_base_interp = []
        for pr_curve_fold in pr_base_folds:
            # Reverse order for proper interpolation (recall should be increasing)
            recall_fold = pr_curve_fold[:,0][::-1] 
            prec_fold = pr_curve_fold[:,1][::-1]
            prec_base_interp.append(np.interp(recall_interp, recall_fold, prec_fold))
        prec_base_mean = np.mean(prec_base_interp, axis=0)
        prec_base_std = np.std(prec_base_interp, axis=0)
        
        # Interpolate meta model PR curves
        prec_meta_interp = []
        for pr_curve_fold in pr_meta_folds:
            recall_fold = pr_curve_fold[:,0][::-1]
            prec_fold = pr_curve_fold[:,1][::-1] 
            prec_meta_interp.append(np.interp(recall_interp, recall_fold, prec_fold))
        prec_meta_mean = np.mean(prec_meta_interp, axis=0)
        prec_meta_std = np.std(prec_meta_interp, axis=0)
        
        # Plot mean PR curves with error bands
        plt.plot(recall_interp, prec_base_mean, 'b-', lw=2.5,
                label=f'{base_name} (AP = {np.mean(ap_base_folds):.3f} ± {np.std(ap_base_folds):.3f})')
        plt.fill_between(recall_interp, prec_base_mean - prec_base_std, prec_base_mean + prec_base_std,
                        color='blue', alpha=0.2)
        
        plt.plot(recall_interp, prec_meta_mean, 'r-', lw=2.5,
                label=f'{meta_name} (AP = {np.mean(ap_meta_folds):.3f} ± {np.std(ap_meta_folds):.3f})')
        plt.fill_between(recall_interp, prec_meta_mean - prec_meta_std, prec_meta_mean + prec_meta_std,
                        color='red', alpha=0.2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves: {class_name} Classification')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        
        # Save PR plot
        pr_path = out_dir / f"pr_{class_name.lower()}_class.{plot_format}"
        plt.savefig(pr_path, dpi=300)
        plt.close()
    
    # Create summary plot comparing all classes
    create_multiclass_summary_plot(metrics, out_dir, class_names, plot_format)
    
    return metrics


def create_multiclass_summary_plot(
    metrics: Dict,
    out_dir: Path, 
    class_names: List[str],
    plot_format: str = "pdf"
) -> None:
    """Create a summary plot comparing AUC and AP across all classes."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multi-Class Performance Summary', fontsize=16)
    
    classes = [name.lower() for name in class_names]
    x_pos = np.arange(len(classes))
    width = 0.35
    
    # AUC comparison
    base_aucs = [metrics["auc"]["base"][cls]["mean"] for cls in classes]
    meta_aucs = [metrics["auc"]["meta"][cls]["mean"] for cls in classes]
    base_auc_stds = [metrics["auc"]["base"][cls]["std"] for cls in classes]
    meta_auc_stds = [metrics["auc"]["meta"][cls]["std"] for cls in classes]
    
    ax1.bar(x_pos - width/2, base_aucs, width, yerr=base_auc_stds, 
           label='Base Model', alpha=0.8, capsize=5)
    ax1.bar(x_pos + width/2, meta_aucs, width, yerr=meta_auc_stds,
           label='Meta Model', alpha=0.8, capsize=5)
    ax1.set_ylabel('ROC AUC')
    ax1.set_title('ROC AUC by Class')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([name.capitalize() for name in classes])
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # AP comparison
    base_aps = [metrics["ap"]["base"][cls]["mean"] for cls in classes]
    meta_aps = [metrics["ap"]["meta"][cls]["mean"] for cls in classes]
    base_ap_stds = [metrics["ap"]["base"][cls]["std"] for cls in classes]
    meta_ap_stds = [metrics["ap"]["meta"][cls]["std"] for cls in classes]
    
    ax2.bar(x_pos - width/2, base_aps, width, yerr=base_ap_stds,
           label='Base Model', alpha=0.8, capsize=5)
    ax2.bar(x_pos + width/2, meta_aps, width, yerr=meta_ap_stds,
           label='Meta Model', alpha=0.8, capsize=5)
    ax2.set_ylabel('Average Precision')
    ax2.set_title('Average Precision by Class')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.capitalize() for name in classes])
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Improvement ratios
    auc_improvements = [(meta_aucs[i] - base_aucs[i]) / base_aucs[i] * 100 
                       for i in range(len(classes))]
    ap_improvements = [(meta_aps[i] - base_aps[i]) / base_aps[i] * 100 
                      for i in range(len(classes))]
    
    ax3.bar(x_pos, auc_improvements, alpha=0.8, color='skyblue')
    ax3.set_ylabel('AUC Improvement (%)')
    ax3.set_title('ROC AUC Improvement')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([name.capitalize() for name in classes])
    ax3.grid(alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    
    ax4.bar(x_pos, ap_improvements, alpha=0.8, color='lightcoral')
    ax4.set_ylabel('AP Improvement (%)')
    ax4.set_title('Average Precision Improvement')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([name.capitalize() for name in classes])
    ax4.grid(alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    
    plt.tight_layout()
    
    # Save summary plot
    summary_path = out_dir / f"multiclass_summary.{plot_format}"
    plt.savefig(summary_path, dpi=300)
    plt.close()


def create_improved_binary_pr_plot(
    y_true: List[np.ndarray],
    y_pred_base: List[np.ndarray], 
    y_pred_meta: List[np.ndarray],
    out_dir: Union[str, Path],
    plot_format: str = "pdf",
    force_separation: bool = True,
    base_name: str = "Base",
    meta_name: str = "Meta"
) -> None:
    """
    Create an improved binary PR plot with better visibility of both curves.
    
    Parameters
    ----------
    y_true : List[np.ndarray]
        True binary labels for each fold
    y_pred_base : List[np.ndarray]
        Base model predictions for each fold
    y_pred_meta : List[np.ndarray] 
        Meta model predictions for each fold
    out_dir : str or Path
        Output directory
    plot_format : str
        File format for plots
    force_separation : bool
        Whether to use techniques to make both curves visible even when very close
    """
    out_dir = Path(out_dir)
    
    # Calculate metrics for all folds
    ap_base_folds, ap_meta_folds = [], []
    pr_base_folds, pr_meta_folds = [], []
    
    for y_t, y_pb, y_pm in zip(y_true, y_pred_base, y_pred_meta):
        # Calculate PR curves
        prec_b, rec_b, _ = precision_recall_curve(y_t, y_pb)
        prec_m, rec_m, _ = precision_recall_curve(y_t, y_pm)
        
        pr_base_folds.append(np.column_stack([rec_b, prec_b]))
        pr_meta_folds.append(np.column_stack([rec_m, prec_m]))
        
        ap_base_folds.append(average_precision_score(y_t, y_pb))
        ap_meta_folds.append(average_precision_score(y_t, y_pm))
    
    plt.figure(figsize=(10, 8))
    
    # Plot individual fold curves with different styles
    for i, pr_curve in enumerate(pr_base_folds):
        plt.plot(pr_curve[:,0], pr_curve[:,1], 
                color='lightblue', alpha=0.4, lw=1.5, 
                linestyle='--' if force_separation else '-',
                label=f'{base_name} (individual folds)' if i == 0 else "")
    
    for i, pr_curve in enumerate(pr_meta_folds):
        plt.plot(pr_curve[:,0], pr_curve[:,1], 
                color='lightcoral', alpha=0.4, lw=1.5,
                label=f'{meta_name} (individual folds)' if i == 0 else "")
    
    # Calculate and plot mean curves
    recall_interp = np.linspace(0, 1, 101)
    
    # Base model mean
    prec_base_interp = []
    for pr_curve in pr_base_folds:
        prec_base_interp.append(np.interp(recall_interp, pr_curve[:,0][::-1], pr_curve[:,1][::-1]))
    prec_base_mean = np.mean(prec_base_interp, axis=0)
    prec_base_std = np.std(prec_base_interp, axis=0)
    
    # Meta model mean  
    prec_meta_interp = []
    for pr_curve in pr_meta_folds:
        prec_meta_interp.append(np.interp(recall_interp, pr_curve[:,0][::-1], pr_curve[:,1][::-1]))
    prec_meta_mean = np.mean(prec_meta_interp, axis=0)
    prec_meta_std = np.std(prec_meta_interp, axis=0)
    
    # Plot mean curves with enhanced visibility
    plt.plot(recall_interp, prec_base_mean, 'b-', lw=4, 
            label=f'{base_name} Mean (AP = {np.mean(ap_base_folds):.3f} ± {np.std(ap_base_folds):.3f})')
    
    if force_separation:
        # Slightly offset the meta curve for visibility
        offset = 0.002
        plt.plot(recall_interp, prec_meta_mean + offset, 'r-', lw=3,
                label=f'{meta_name} Mean (AP = {np.mean(ap_meta_folds):.3f} ± {np.std(ap_meta_folds):.3f})')
    else:
        plt.plot(recall_interp, prec_meta_mean, 'r-', lw=3,
                label=f'{meta_name} Mean (AP = {np.mean(ap_meta_folds):.3f} ± {np.std(ap_meta_folds):.3f})')
    
    # Add confidence bands
    plt.fill_between(recall_interp, prec_base_mean - prec_base_std, prec_base_mean + prec_base_std,
                    color='blue', alpha=0.2)
    plt.fill_between(recall_interp, prec_meta_mean - prec_meta_std, prec_meta_mean + prec_meta_std,
                    color='red', alpha=0.2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves: Splice vs Non-Splice\n(Enhanced Visibility)', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Add text annotations about curve proximity
    if np.mean(ap_meta_folds) - np.mean(ap_base_folds) < 0.1:
        plt.text(0.05, 0.15, f'Note: Curves are very close\n(ΔAP = {np.mean(ap_meta_folds) - np.mean(ap_base_folds):.4f})', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    # Save improved PR plot
    improved_pr_path = out_dir / f"pr_binary_improved.{plot_format}"
    plt.savefig(improved_pr_path, dpi=300)
    plt.close() 