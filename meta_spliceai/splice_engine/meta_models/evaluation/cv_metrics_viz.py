#!/usr/bin/env python3
"""
Visualization module for gene CV metrics comparison between base and meta models.

This module creates comprehensive plots comparing base model (SpliceAI) performance
with meta model performance across cross-validation folds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

def load_cv_metrics(csv_path: str | Path) -> pd.DataFrame:
    """Load CV metrics from CSV file."""
    return pd.read_csv(csv_path)

def create_base_vs_meta_comparison_plots(
    df: pd.DataFrame,
    out_dir: str | Path,
    plot_format: str = 'png',
    dpi: int = 300
) -> Dict[str, str]:
    """
    Create comprehensive comparison plots between base and meta models.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing CV metrics
    out_dir : str | Path
        Output directory for plots
    plot_format : str
        Plot format ('png', 'pdf', 'svg')
    dpi : int
        Resolution for plots
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping plot names to file paths
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_files = {}
    
    # 1. F1 Score Comparison
    plot_files['f1_comparison'] = _create_f1_comparison_plot(
        df, out_dir, plot_format, dpi
    )
    
    # 2. AUC Comparison
    plot_files['auc_comparison'] = _create_auc_comparison_plot(
        df, out_dir, plot_format, dpi
    )
    
    # 3. Average Precision Comparison
    plot_files['ap_comparison'] = _create_ap_comparison_plot(
        df, out_dir, plot_format, dpi
    )
    
    # 4. Error Reduction Analysis
    plot_files['error_reduction'] = _create_error_reduction_plot(
        df, out_dir, plot_format, dpi
    )
    
    # 5. Combined Performance Overview
    plot_files['performance_overview'] = _create_performance_overview_plot(
        df, out_dir, plot_format, dpi
    )
    
    # 6. Performance Improvement Summary
    plot_files['improvement_summary'] = _create_improvement_summary_plot(
        df, out_dir, plot_format, dpi
    )
    
    # 7. Top-k Accuracy Analysis
    if 'top_k_accuracy' in df.columns:
        plot_files['topk_analysis'] = _create_topk_analysis_plot(
            df, out_dir, plot_format, dpi
        )
    
    return plot_files

def _create_f1_comparison_plot(
    df: pd.DataFrame,
    out_dir: Path,
    plot_format: str,
    dpi: int
) -> str:
    """Create F1 score comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Fold-by-fold comparison
    folds = df['fold'].values
    base_f1 = df['base_f1'].values
    meta_f1 = df['meta_f1'].values
    
    ax1.plot(folds, base_f1, 'o-', label='Base Model', linewidth=2, markersize=8)
    ax1.plot(folds, meta_f1, 's-', label='Meta Model', linewidth=2, markersize=8)
    ax1.set_xlabel('CV Fold')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score Comparison Across CV Folds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(folds)
    
    # Statistical summary
    improvement = meta_f1 - base_f1
    ax2.bar(['Base Model', 'Meta Model'], 
            [base_f1.mean(), meta_f1.mean()], 
            yerr=[base_f1.std(), meta_f1.std()],
            capsize=5, alpha=0.7)
    ax2.set_ylabel('F1 Score')
    ax2.set_title(f'F1 Score Summary\n(Mean Improvement: {improvement.mean():.3f}±{improvement.std():.3f})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    file_path = out_dir / f'cv_f1_comparison.{plot_format}'
    plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return str(file_path)

def _create_auc_comparison_plot(
    df: pd.DataFrame,
    out_dir: Path,
    plot_format: str,
    dpi: int
) -> str:
    """Create AUC comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Fold-by-fold comparison
    folds = df['fold'].values
    base_auc = df['auc_base'].values
    meta_auc = df['auc_meta'].values
    
    ax1.plot(folds, base_auc, 'o-', label='Base Model', linewidth=2, markersize=8)
    ax1.plot(folds, meta_auc, 's-', label='Meta Model', linewidth=2, markersize=8)
    ax1.set_xlabel('CV Fold')
    ax1.set_ylabel('ROC AUC')
    ax1.set_title('ROC AUC Comparison Across CV Folds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(folds)
    ax1.set_ylim([0.5, 1.0])
    
    # Statistical summary
    improvement = meta_auc - base_auc
    ax2.bar(['Base Model', 'Meta Model'], 
            [base_auc.mean(), meta_auc.mean()], 
            yerr=[base_auc.std(), meta_auc.std()],
            capsize=5, alpha=0.7)
    ax2.set_ylabel('ROC AUC')
    ax2.set_title(f'ROC AUC Summary\n(Mean Improvement: {improvement.mean():.3f}±{improvement.std():.3f})')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    file_path = out_dir / f'cv_auc_comparison.{plot_format}'
    plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return str(file_path)

def _create_ap_comparison_plot(
    df: pd.DataFrame,
    out_dir: Path,
    plot_format: str,
    dpi: int
) -> str:
    """Create Average Precision comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Fold-by-fold comparison
    folds = df['fold'].values
    base_ap = df['ap_base'].values
    meta_ap = df['ap_meta'].values
    
    ax1.plot(folds, base_ap, 'o-', label='Base Model', linewidth=2, markersize=8)
    ax1.plot(folds, meta_ap, 's-', label='Meta Model', linewidth=2, markersize=8)
    ax1.set_xlabel('CV Fold')
    ax1.set_ylabel('Average Precision')
    ax1.set_title('Average Precision Comparison Across CV Folds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(folds)
    
    # Statistical summary
    improvement = meta_ap - base_ap
    ax2.bar(['Base Model', 'Meta Model'], 
            [base_ap.mean(), meta_ap.mean()], 
            yerr=[base_ap.std(), meta_ap.std()],
            capsize=5, alpha=0.7)
    ax2.set_ylabel('Average Precision')
    ax2.set_title(f'Average Precision Summary\n(Mean Improvement: {improvement.mean():.3f}±{improvement.std():.3f})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    file_path = out_dir / f'cv_ap_comparison.{plot_format}'
    plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return str(file_path)

def _create_error_reduction_plot(
    df: pd.DataFrame,
    out_dir: Path,
    plot_format: str,
    dpi: int
) -> str:
    """Create error reduction analysis plot."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    folds = df['fold'].values
    delta_fp = df['delta_fp'].values
    delta_fn = df['delta_fn'].values
    
    # Delta FP across folds
    ax1.plot(folds, delta_fp, 'o-', color='red', linewidth=2, markersize=8)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('CV Fold')
    ax1.set_ylabel('ΔFP (Base - Meta)')
    ax1.set_title('False Positive Reduction Across CV Folds')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(folds)
    
    # Delta FN across folds
    ax2.plot(folds, delta_fn, 'o-', color='blue', linewidth=2, markersize=8)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('CV Fold')
    ax2.set_ylabel('ΔFN (Base - Meta)')
    ax2.set_title('False Negative Reduction Across CV Folds')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(folds)
    
    # Error reduction summary
    ax3.bar(['ΔFP', 'ΔFN'], 
            [delta_fp.mean(), delta_fn.mean()], 
            yerr=[delta_fp.std(), delta_fn.std()],
            capsize=5, alpha=0.7, color=['red', 'blue'])
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Mean Error Reduction')
    ax3.set_title('Average Error Reduction\n(Positive = Meta Better)')
    ax3.grid(True, alpha=0.3)
    
    # Combined error reduction
    total_error_reduction = delta_fp + delta_fn
    ax4.bar(folds, total_error_reduction, alpha=0.7, color='green')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('CV Fold')
    ax4.set_ylabel('Total Error Reduction')
    ax4.set_title(f'Total Error Reduction Per Fold\n(Mean: {total_error_reduction.mean():.1f}±{total_error_reduction.std():.1f})')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(folds)
    
    plt.tight_layout()
    file_path = out_dir / f'cv_error_reduction.{plot_format}'
    plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return str(file_path)

def _create_performance_overview_plot(
    df: pd.DataFrame,
    out_dir: Path,
    plot_format: str,
    dpi: int
) -> str:
    """Create comprehensive performance overview plot."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Performance metrics radar-style comparison
    metrics = ['base_f1', 'meta_f1', 'auc_base', 'auc_meta', 'ap_base', 'ap_meta']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if len(available_metrics) >= 6:
        # Multi-metric comparison
        base_means = [df['base_f1'].mean(), df['auc_base'].mean(), df['ap_base'].mean()]
        meta_means = [df['meta_f1'].mean(), df['auc_meta'].mean(), df['ap_meta'].mean()]
        metric_names = ['F1 Score', 'ROC AUC', 'Average Precision']
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        ax1.bar(x - width/2, base_means, width, label='Base Model', alpha=0.7)
        ax1.bar(x + width/2, meta_means, width, label='Meta Model', alpha=0.7)
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Performance Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Accuracy metrics
    if 'test_accuracy' in df.columns and 'splice_accuracy' in df.columns:
        acc_metrics = ['test_accuracy', 'splice_accuracy']
        acc_means = [df[m].mean() for m in acc_metrics]
        acc_stds = [df[m].std() for m in acc_metrics]
        
        ax2.bar(['Overall Accuracy', 'Splice Accuracy'], acc_means, 
                yerr=acc_stds, capsize=5, alpha=0.7)
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Metrics')
        ax2.grid(True, alpha=0.3)
    
    # F1 metrics breakdown
    if 'test_macro_f1' in df.columns and 'splice_macro_f1' in df.columns:
        f1_metrics = ['test_macro_f1', 'splice_macro_f1']
        f1_means = [df[m].mean() for m in f1_metrics]
        f1_stds = [df[m].std() for m in f1_metrics]
        
        ax3.bar(['Overall F1', 'Splice F1'], f1_means, 
                yerr=f1_stds, capsize=5, alpha=0.7)
        ax3.set_ylabel('F1 Score')
        ax3.set_title('F1 Score Metrics')
        ax3.grid(True, alpha=0.3)
    
    # Top-k accuracy if available
    if 'top_k_accuracy' in df.columns:
        ax4.boxplot([df['top_k_accuracy'].values], labels=['Top-k Accuracy'])
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Top-k Accuracy Distribution')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    file_path = out_dir / f'cv_performance_overview.{plot_format}'
    plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return str(file_path)

def _create_improvement_summary_plot(
    df: pd.DataFrame,
    out_dir: Path,
    plot_format: str,
    dpi: int
) -> str:
    """Create improvement summary plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate improvements
    improvements = {}
    if 'base_f1' in df.columns and 'meta_f1' in df.columns:
        improvements['F1 Score'] = ((df['meta_f1'] - df['base_f1']) / df['base_f1'] * 100).mean()
    if 'auc_base' in df.columns and 'auc_meta' in df.columns:
        improvements['ROC AUC'] = ((df['auc_meta'] - df['auc_base']) / df['auc_base'] * 100).mean()
    if 'ap_base' in df.columns and 'ap_meta' in df.columns:
        improvements['Average Precision'] = ((df['ap_meta'] - df['ap_base']) / df['ap_base'] * 100).mean()
    
    # Percentage improvements
    if improvements:
        metrics = list(improvements.keys())
        values = list(improvements.values())
        colors = ['green' if v > 0 else 'red' for v in values]
        
        ax1.bar(metrics, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Improvement (%)')
        ax1.set_title('Relative Performance Improvement')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Error reduction effectiveness
    if 'delta_fp' in df.columns and 'delta_fn' in df.columns:
        fold_colors = ['green' if (df.loc[i, 'delta_fp'] > 0 and df.loc[i, 'delta_fn'] > 0) else 'orange' 
                      for i in range(len(df))]
        
        ax2.scatter(df['delta_fp'], df['delta_fn'], c=fold_colors, s=100, alpha=0.7)
        ax2.set_xlabel('ΔFP (Base - Meta)')
        ax2.set_ylabel('ΔFN (Base - Meta)')
        ax2.set_title('Error Reduction Effectiveness')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add quadrant labels - CORRECTED POSITIONS
        # X-axis: delta_fp (Base - Meta FP) - positive means meta better
        # Y-axis: delta_fn (Base - Meta FN) - positive means meta better
        ax2.text(0.75, 0.95, 'Both Improved', transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax2.text(0.05, 0.95, 'FN Improved', transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax2.text(0.75, 0.05, 'FP Improved', transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax2.text(0.05, 0.05, 'Both Worse', transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    file_path = out_dir / f'cv_improvement_summary.{plot_format}'
    plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return str(file_path)

def _create_topk_analysis_plot(
    df: pd.DataFrame,
    out_dir: Path,
    plot_format: str,
    dpi: int
) -> str:
    """Create top-k accuracy analysis plot."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Top-k accuracy across folds
    if 'top_k_accuracy' in df.columns:
        folds = df['fold'].values
        topk_acc = df['top_k_accuracy'].values
        
        ax1.plot(folds, topk_acc, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('CV Fold')
        ax1.set_ylabel('Top-k Accuracy')
        ax1.set_title('Top-k Accuracy Across CV Folds')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(folds)
    
    # Donor vs Acceptor top-k if available
    if 'top_k_donor' in df.columns and 'top_k_acceptor' in df.columns:
        donor_topk = df['top_k_donor'].values
        acceptor_topk = df['top_k_acceptor'].values
        
        ax2.plot(folds, donor_topk, 'o-', label='Donor', linewidth=2, markersize=8)
        ax2.plot(folds, acceptor_topk, 's-', label='Acceptor', linewidth=2, markersize=8)
        ax2.set_xlabel('CV Fold')
        ax2.set_ylabel('Top-k Accuracy')
        ax2.set_title('Top-k Accuracy by Splice Type')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(folds)
    
    # Summary statistics
    if 'top_k_accuracy' in df.columns:
        # Filter out NaN values for histogram
        topk_acc_clean = topk_acc[~np.isnan(topk_acc)]
        if len(topk_acc_clean) > 0:
            ax3.hist(topk_acc_clean, bins=10, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Top-k Accuracy')
            ax3.set_ylabel('Frequency')
            ax3.set_title(f'Top-k Accuracy Distribution\n(Valid: {len(topk_acc_clean)}/{len(topk_acc)})')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No valid top-k\naccuracy data', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Top-k Accuracy Distribution\n(No valid data)')
            ax3.set_xlabel('Top-k Accuracy')
            ax3.set_ylabel('Frequency')
    
    # Gene count analysis if available
    if 'top_k_n_genes' in df.columns:
        gene_counts = df['top_k_n_genes'].values
        ax4.bar(folds, gene_counts, alpha=0.7)
        ax4.set_xlabel('CV Fold')
        ax4.set_ylabel('Number of Genes')
        ax4.set_title('Number of Genes per CV Fold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(folds)
    
    plt.tight_layout()
    file_path = out_dir / f'cv_topk_analysis.{plot_format}'
    plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return str(file_path)

def generate_cv_metrics_report(
    csv_path: str | Path,
    out_dir: str | Path,
    dataset_path: str | Path | None = None,
    plot_format: str = 'png',
    dpi: int = 300
) -> Dict[str, any]:
    """
    Generate comprehensive CV metrics visualization report.
    
    Parameters
    ----------
    csv_path : str | Path
        Path to gene_cv_metrics.csv file
    out_dir : str | Path
        Output directory for plots and report
    dataset_path : str | Path, optional
        Path to training dataset for dynamic baseline error calculation
    plot_format : str
        Plot format ('png', 'pdf', 'svg')
    dpi : int
        Resolution for plots
        
    Returns
    -------
    Dict[str, any]
        Dictionary containing summary statistics and plot paths
    """
    # Load data
    df = load_cv_metrics(csv_path)
    
    # Create organized output directory structure
    out_dir = Path(out_dir)
    viz_dir = out_dir / "cv_metrics_visualization"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    plot_files = create_base_vs_meta_comparison_plots(
        df, viz_dir, plot_format, dpi
    )
    
    # Calculate summary statistics
    summary_stats = _calculate_summary_statistics(df)
    
    # Try to get detailed baseline error analysis
    baseline_stats = _try_calculate_baseline_errors(viz_dir, df, dataset_path)
    summary_stats.update(baseline_stats)
    
    # Create summary report
    report_path = viz_dir / "cv_metrics_summary.txt"
    _create_summary_report(df, summary_stats, report_path)
    
    return {
        'summary_stats': summary_stats,
        'plot_files': plot_files,
        'report_path': str(report_path),
        'visualization_dir': str(viz_dir)
    }

def _try_calculate_baseline_errors(out_dir: Path, cv_df: pd.DataFrame, dataset_path: str | Path | None = None) -> Dict[str, any]:
    """Calculate baseline error counts using dynamic analysis of the training dataset."""
    baseline_stats = {}
    
    # Use dynamic baseline calculation - dataset path is required
    if dataset_path is None:
        print(f"[CV Metrics] Warning: No dataset path provided - baseline error analysis unavailable")
        return {'calculation_source': 'no_dataset_path'}
    
    try:
        from .baseline_error_calculator import calculate_cv_error_reductions
        
        print(f"[CV Metrics] Calculating baseline errors from dataset: {dataset_path}")
        baseline_stats = calculate_cv_error_reductions(
            cv_df=cv_df,
            dataset_path=dataset_path,
            sample_size=50000,  # Use sampling for efficiency
            verbose=True
        )
        
        if 'error' not in baseline_stats:
            # Successfully calculated dynamic baseline
            # Map keys to match expected format
            mapped_stats = baseline_stats.copy()
            
            # Map new keys to legacy keys for compatibility
            if 'cv_baseline_fp' in baseline_stats:
                mapped_stats['baseline_fp_total'] = baseline_stats['cv_baseline_fp']
            elif 'baseline_fp' in baseline_stats:
                mapped_stats['baseline_fp_total'] = baseline_stats['baseline_fp']
                
            if 'cv_baseline_fn' in baseline_stats:
                mapped_stats['baseline_fn_total'] = baseline_stats['cv_baseline_fn']
            elif 'baseline_fn' in baseline_stats:
                mapped_stats['baseline_fn_total'] = baseline_stats['baseline_fn']
            
            mapped_stats['calculation_source'] = 'dynamic_dataset_analysis'
            mapped_stats['is_dynamic_calculation'] = True
            return mapped_stats
        else:
            print(f"[CV Metrics] Dynamic calculation failed: {baseline_stats['error']}")
            return {'calculation_source': 'dynamic_calculation_failed', 'error': baseline_stats['error']}
            
    except Exception as e:
        print(f"[CV Metrics] Dynamic baseline calculation failed: {e}")
        return {'calculation_source': 'dynamic_calculation_error', 'error': str(e)}
    
    # If dynamic calculation failed and no dataset path provided, return basic context
    total_fp_reduction = cv_df['delta_fp'].sum()
    total_fn_reduction = cv_df['delta_fn'].sum()
    total_cv_rows = cv_df['test_rows'].sum()
    
    return {
        'estimated_error_context': f"~{total_cv_rows:,} test samples per fold",
        'total_delta_fp': total_fp_reduction,
        'total_delta_fn': total_fn_reduction,
        'total_errors_reduced': total_fp_reduction + total_fn_reduction,
        'has_detailed_analysis': False,
        'calculation_source': 'basic_context_only'
    }

def _calculate_summary_statistics(df: pd.DataFrame) -> Dict[str, any]:
    """Calculate summary statistics from CV metrics."""
    stats = {}
    
    # Basic performance metrics
    for metric in ['base_f1', 'meta_f1', 'auc_base', 'auc_meta', 'ap_base', 'ap_meta']:
        if metric in df.columns:
            stats[f'{metric}_mean'] = df[metric].mean()
            stats[f'{metric}_std'] = df[metric].std()
    
    # Improvements
    if 'base_f1' in df.columns and 'meta_f1' in df.columns:
        f1_improvement = df['meta_f1'] - df['base_f1']
        stats['f1_improvement_mean'] = f1_improvement.mean()
        stats['f1_improvement_std'] = f1_improvement.std()
        stats['f1_improvement_pct'] = ((df['meta_f1'] - df['base_f1']) / df['base_f1'] * 100).mean()
    
    if 'auc_base' in df.columns and 'auc_meta' in df.columns:
        auc_improvement = df['auc_meta'] - df['auc_base']
        stats['auc_improvement_mean'] = auc_improvement.mean()
        stats['auc_improvement_std'] = auc_improvement.std()
        stats['auc_improvement_pct'] = ((df['auc_meta'] - df['auc_base']) / df['auc_base'] * 100).mean()
    
    if 'ap_base' in df.columns and 'ap_meta' in df.columns:
        ap_improvement = df['ap_meta'] - df['ap_base']
        stats['ap_improvement_mean'] = ap_improvement.mean()
        stats['ap_improvement_std'] = ap_improvement.std()
        stats['ap_improvement_pct'] = ((df['ap_meta'] - df['ap_base']) / df['ap_base'] * 100).mean()
    
    # Error reduction
    if 'delta_fp' in df.columns and 'delta_fn' in df.columns:
        stats['delta_fp_mean'] = df['delta_fp'].mean()
        stats['delta_fp_std'] = df['delta_fp'].std()
        stats['delta_fn_mean'] = df['delta_fn'].mean()
        stats['delta_fn_std'] = df['delta_fn'].std()
        stats['total_error_reduction'] = (df['delta_fp'] + df['delta_fn']).mean()
        
        # Calculate total error reductions across all folds
        stats['total_delta_fp'] = df['delta_fp'].sum()
        stats['total_delta_fn'] = df['delta_fn'].sum()
        stats['total_errors_reduced'] = stats['total_delta_fp'] + stats['total_delta_fn']
        
        # Estimate baseline error rates from F1 scores and test set sizes
        # This provides context for the magnitude of error reductions
        if 'base_f1' in df.columns and 'test_rows' in df.columns:
            # Estimate baseline errors using F1 relationship to precision/recall
            # F1 = 2PR/(P+R), and errors relate to precision/recall gaps
            avg_test_rows = df['test_rows'].mean()
            avg_base_f1 = df['base_f1'].mean()
            
            # Rough estimate: lower F1 indicates more errors relative to true positives
            # This gives context for the error reduction magnitude
            stats['avg_test_rows'] = avg_test_rows
            stats['estimated_error_context'] = f"~{avg_test_rows:.0f} test samples per fold"
    
    # Top-k metrics
    if 'top_k_accuracy' in df.columns:
        stats['top_k_accuracy_mean'] = df['top_k_accuracy'].mean()
        stats['top_k_accuracy_std'] = df['top_k_accuracy'].std()
    
    if 'top_k_donor' in df.columns:
        stats['top_k_donor_mean'] = df['top_k_donor'].mean()
        stats['top_k_donor_std'] = df['top_k_donor'].std()
    
    if 'top_k_acceptor' in df.columns:
        stats['top_k_acceptor_mean'] = df['top_k_acceptor'].mean()
        stats['top_k_acceptor_std'] = df['top_k_acceptor'].std()
    
    return stats

def _create_summary_report(df: pd.DataFrame, stats: Dict[str, any], report_path: Path) -> None:
    """Create a text summary report."""
    with open(report_path, 'w') as f:
        f.write("Gene CV Metrics Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Number of CV folds: {len(df)}\n\n")
        
        # Absolute Performance Metrics
        f.write("Absolute Performance Metrics (Average across CV folds)\n")
        f.write("-" * 60 + "\n")
        
        if 'base_f1_mean' in stats and 'meta_f1_mean' in stats:
            f.write(f"Base Model F1 Score: {stats['base_f1_mean']:.4f} ± {stats['base_f1_std']:.4f}\n")
            f.write(f"Meta Model F1 Score: {stats['meta_f1_mean']:.4f} ± {stats['meta_f1_std']:.4f}\n\n")
        
        if 'auc_base_mean' in stats and 'auc_meta_mean' in stats:
            f.write(f"Base Model ROC AUC: {stats['auc_base_mean']:.4f} ± {stats['auc_base_std']:.4f}\n")
            f.write(f"Meta Model ROC AUC: {stats['auc_meta_mean']:.4f} ± {stats['auc_meta_std']:.4f}\n\n")
        
        if 'ap_base_mean' in stats and 'ap_meta_mean' in stats:
            f.write(f"Base Model Average Precision: {stats['ap_base_mean']:.4f} ± {stats['ap_base_std']:.4f}\n")
            f.write(f"Meta Model Average Precision: {stats['ap_meta_mean']:.4f} ± {stats['ap_meta_std']:.4f}\n\n")
        
        # Performance comparison (improvements)
        f.write("Performance Improvements (Meta vs Base)\n")
        f.write("-" * 50 + "\n")
        
        if 'f1_improvement_mean' in stats:
            f.write(f"F1 Score Improvement: {stats['f1_improvement_mean']:.4f} ± {stats['f1_improvement_std']:.4f}\n")
            f.write(f"F1 Score Improvement (%): {stats['f1_improvement_pct']:.2f}%\n")
        
        if 'auc_improvement_mean' in stats:
            f.write(f"ROC AUC Improvement: {stats['auc_improvement_mean']:.4f} ± {stats['auc_improvement_std']:.4f}\n")
            f.write(f"ROC AUC Improvement (%): {stats['auc_improvement_pct']:.2f}%\n")
        
        if 'ap_improvement_mean' in stats:
            f.write(f"Average Precision Improvement: {stats['ap_improvement_mean']:.4f} ± {stats['ap_improvement_std']:.4f}\n")
            f.write(f"Average Precision Improvement (%): {stats['ap_improvement_pct']:.2f}%\n")
        
        # Error reduction
        if 'delta_fp_mean' in stats:
            f.write("\nError Reduction Analysis\n")
            f.write("-" * 40 + "\n")
            
            # Per-fold statistics
            f.write(f"False Positive Reduction (per fold): {stats['delta_fp_mean']:.1f} ± {stats['delta_fp_std']:.1f}\n")
            f.write(f"False Negative Reduction (per fold): {stats['delta_fn_mean']:.1f} ± {stats['delta_fn_std']:.1f}\n")
            
            # Total reductions across all folds
            if 'total_delta_fp' in stats:
                f.write(f"Total FP Reduction (all folds): {stats['total_delta_fp']:.0f}\n")
                f.write(f"Total FN Reduction (all folds): {stats['total_delta_fn']:.0f}\n")
                f.write(f"Total Errors Reduced: {stats['total_errors_reduced']:.0f}\n")
            
            # Detailed percentage analysis if available
            if stats.get('has_detailed_analysis', False):
                f.write(f"\nBaseline Error Counts:\n")
                f.write(f"  Baseline False Positives: {stats['baseline_fp_total']:.0f}\n")
                f.write(f"  Baseline False Negatives: {stats['baseline_fn_total']:.0f}\n")
                f.write(f"\nPercentage Error Reductions:\n")
                f.write(f"  FP Reduction: {stats['fp_reduction_pct']:.1f}% ({stats['actual_fp_reduction']:.0f}/{stats['baseline_fp_total']:.0f})\n")
                f.write(f"  FN Reduction: {stats['fn_reduction_pct']:.1f}% ({stats['actual_fn_reduction']:.0f}/{stats['baseline_fn_total']:.0f})\n")
                
                if stats.get('is_known_baseline', False):
                    f.write(f"\nNote: Using proportional baseline error counts from {stats['dataset_name']} dataset\n")
                    if 'dataset_usage_pct' in stats:
                        f.write(f"  CV dataset: {stats['cv_dataset_rows']:,} rows ({stats['dataset_usage_pct']:.1f}% of full {stats['full_dataset_rows']:,} rows)\n")
                elif stats.get('is_subset_analysis', False):
                    f.write(f"\nNote: Baseline error counts estimated from detailed analysis subset\n")
                    f.write(f"  ({stats['subset_size']:.0f} samples analyzed from {stats['total_cv_size']:.0f} total CV samples)\n")
            
            # Context information
            if 'estimated_error_context' in stats:
                f.write(f"\nContext: {stats['estimated_error_context']}\n")
        
        if 'top_k_accuracy_mean' in stats:
            f.write("\nTop-k Accuracy Analysis\n")
            f.write("-" * 40 + "\n")
            f.write(f"Combined Top-k Accuracy: {stats['top_k_accuracy_mean']:.4f} ± {stats['top_k_accuracy_std']:.4f}\n")
            
            if 'top_k_donor_mean' in stats:
                f.write(f"Donor Top-k Accuracy: {stats['top_k_donor_mean']:.4f} ± {stats['top_k_donor_std']:.4f}\n")
            
            if 'top_k_acceptor_mean' in stats:
                f.write(f"Acceptor Top-k Accuracy: {stats['top_k_acceptor_mean']:.4f} ± {stats['top_k_acceptor_std']:.4f}\n")

# CLI interface
def main():
    """Command-line interface for CV metrics visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CV metrics visualization report")
    parser.add_argument("csv_path", help="Path to gene_cv_metrics.csv file")
    parser.add_argument("--out-dir", required=True, help="Output directory for plots and report")
    parser.add_argument("--plot-format", default="png", choices=["png", "pdf", "svg"], 
                       help="Plot format")
    parser.add_argument("--dpi", type=int, default=300, help="Plot resolution")
    
    args = parser.parse_args()
    
    # Generate report
    result = generate_cv_metrics_report(
        args.csv_path, 
        args.out_dir, 
        args.plot_format, 
        args.dpi
    )
    
    print(f"CV metrics visualization report generated!")
    print(f"Visualization directory: {result['visualization_dir']}")
    print(f"Summary report: {result['report_path']}")
    print(f"Generated {len(result['plot_files'])} plots:")
    for plot_name, plot_path in result['plot_files'].items():
        print(f"  - {plot_name}: {plot_path}")

if __name__ == "__main__":
    main() 