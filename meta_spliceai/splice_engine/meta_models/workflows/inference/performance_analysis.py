"""
Enhanced Performance Analysis for Inference Workflow

This module provides comprehensive performance analysis including:
- Per-gene performance reports
- ROC/PR curve generation
- Base vs Meta model comparisons
- Statistical metrics calculation
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import json
from datetime import datetime


class PerformanceAnalyzer:
    """Comprehensive performance analysis for inference results."""
    
    def __init__(self, output_dir: Path, verbose: bool = True):
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.performance_dir = self.output_dir / "performance_analysis"
        self.performance_dir.mkdir(exist_ok=True, parents=True)
        
    def generate_per_gene_report(self, gene_id: str, predictions_df: pl.DataFrame,
                                 statistics: Dict[str, Any]) -> Path:
        """Generate detailed performance report for a single gene."""
        report_path = self.performance_dir / f"{gene_id}_performance_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"PERFORMANCE REPORT - GENE {gene_id}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            
            # Basic statistics
            f.write("📊 BASIC STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total positions: {statistics['total_positions']:,}\n")
            f.write(f"Meta-model positions: {statistics['prediction_sources']['meta_model_positions']:,}\n")
            f.write(f"Base-model positions: {statistics['prediction_sources']['base_model_positions']:,}\n")
            f.write(f"Meta-model usage: {statistics['prediction_sources']['meta_model_percentage']:.2f}%\n\n")
            
            # Class distribution
            f.write("📈 CLASS DISTRIBUTION\n")
            f.write("-" * 20 + "\n")
            for class_name, count in statistics['class_distribution'].items():
                percentage = (count / statistics['total_positions'] * 100) if statistics['total_positions'] > 0 else 0
                f.write(f"{class_name:10s}: {count:7,} ({percentage:5.2f}%)\n")
            f.write("\n")
            
            # Base-only vs hybrid comparison if available
            if 'base_only_distribution' in statistics:
                f.write("🔄 BEFORE vs AFTER META-MODEL\n")
                f.write("-" * 40 + "\n")
                base_dist = statistics['base_only_distribution']
                current_dist = statistics['class_distribution']
                
                for key in ['TP', 'TN', 'FP', 'FN']:
                    base_val = base_dist.get(key, 0)
                    current_val = current_dist.get(key, 0)
                    delta = current_val - base_val
                    
                    base_pct = (base_val / statistics['total_positions'] * 100) if statistics['total_positions'] > 0 else 0
                    current_pct = (current_val / statistics['total_positions'] * 100) if statistics['total_positions'] > 0 else 0
                    
                    symbol = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
                    f.write(f"{key:3s}: {base_val:7,} ({base_pct:5.2f}%) → {current_val:7,} ({current_pct:5.2f}%) {symbol} {delta:+,}\n")
                
                # Calculate metrics
                if base_dist.get('TP', 0) + base_dist.get('FN', 0) > 0:
                    base_recall = base_dist.get('TP', 0) / (base_dist.get('TP', 0) + base_dist.get('FN', 0))
                else:
                    base_recall = 0
                    
                if base_dist.get('TP', 0) + base_dist.get('FP', 0) > 0:
                    base_precision = base_dist.get('TP', 0) / (base_dist.get('TP', 0) + base_dist.get('FP', 0))
                else:
                    base_precision = 0
                    
                if current_dist.get('TP', 0) + current_dist.get('FN', 0) > 0:
                    current_recall = current_dist.get('TP', 0) / (current_dist.get('TP', 0) + current_dist.get('FN', 0))
                else:
                    current_recall = 0
                    
                if current_dist.get('TP', 0) + current_dist.get('FP', 0) > 0:
                    current_precision = current_dist.get('TP', 0) / (current_dist.get('TP', 0) + current_dist.get('FP', 0))
                else:
                    current_precision = 0
                
                base_f1 = 2 * base_precision * base_recall / (base_precision + base_recall) if (base_precision + base_recall) > 0 else 0
                current_f1 = 2 * current_precision * current_recall / (current_precision + current_recall) if (current_precision + current_recall) > 0 else 0
                
                f.write(f"\n📊 PERFORMANCE METRICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Precision: {base_precision:.3f} → {current_precision:.3f} ({current_precision - base_precision:+.3f})\n")
                f.write(f"Recall:    {base_recall:.3f} → {current_recall:.3f} ({current_recall - base_recall:+.3f})\n")
                f.write(f"F1 Score:  {base_f1:.3f} → {current_f1:.3f} ({current_f1 - base_f1:+.3f})\n")
            
            # Confidence statistics
            if 'confidence_statistics' in statistics and statistics['confidence_statistics']:
                f.write(f"\n🎯 CONFIDENCE STATISTICS\n")
                f.write("-" * 20 + "\n")
                conf_stats = statistics['confidence_statistics']
                if 'mean' in conf_stats:
                    f.write(f"Mean confidence:   {conf_stats['mean']:.3f}\n")
                if 'std' in conf_stats:
                    f.write(f"Std confidence:    {conf_stats['std']:.3f}\n")
                if 'min' in conf_stats:
                    f.write(f"Min confidence:    {conf_stats['min']:.3f}\n")
                if 'max' in conf_stats:
                    f.write(f"Max confidence:    {conf_stats['max']:.3f}\n")
                if 'median' in conf_stats:
                    f.write(f"Median confidence: {conf_stats['median']:.3f}\n")
            
            # Processing efficiency
            if 'processing_time' in statistics:
                f.write(f"\n⚡ PROCESSING EFFICIENCY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Processing time: {statistics.get('processing_time', 0):.2f} seconds\n")
                if statistics['total_positions'] > 0:
                    positions_per_sec = statistics['total_positions'] / statistics.get('processing_time', 1)
                    f.write(f"Throughput: {positions_per_sec:,.0f} positions/second\n")
            
            f.write(f"\n{'='*60}\n")
        
        return report_path
    
    def generate_roc_pr_curves(self, all_predictions: Dict[str, pl.DataFrame],
                               comparison_modes: List[str] = ["base_only", "hybrid", "meta_only"]) -> Dict[str, Path]:
        """Generate ROC and PR curves comparing different prediction modes."""
        
        # Collect predictions across all genes
        y_true_all = []
        y_scores = {mode: [] for mode in comparison_modes}
        
        for gene_id, pred_df in all_predictions.items():
            # Extract true labels (binary: splice site or not)
            if 'splice_type' in pred_df.columns:
                # Binary classification: splice site (1) vs neither (0)
                # Handle potential null values
                splice_types = pred_df['splice_type'].fill_null('neither')
                y_true = (splice_types != 'neither').to_numpy().astype(int)
                y_true_all.extend(y_true)
                
                # Base-only scores
                if "base_only" in comparison_modes:
                    base_scores = (pred_df['donor_score'] + pred_df['acceptor_score']).to_numpy()
                    y_scores["base_only"].extend(base_scores)
                
                # Hybrid scores (what was actually used)
                if "hybrid" in comparison_modes:
                    hybrid_scores = []
                    for row in pred_df.to_dicts():
                        if row['prediction_source'] == 'meta_model':
                            score = row['donor_meta'] + row['acceptor_meta']
                        else:
                            score = row['donor_score'] + row['acceptor_score']
                        hybrid_scores.append(score)
                    y_scores["hybrid"].extend(hybrid_scores)
                
                # Meta-only scores (if available)
                if "meta_only" in comparison_modes and 'donor_meta' in pred_df.columns:
                    meta_scores = (pred_df['donor_meta'] + pred_df['acceptor_meta']).to_numpy()
                    y_scores["meta_only"].extend(meta_scores)
        
        # Convert to numpy arrays
        y_true_all = np.array(y_true_all)
        
        # Check if we have enough data points
        n_samples = len(y_true_all)
        n_positive = np.sum(y_true_all)
        n_negative = n_samples - n_positive
        
        if n_samples < 100:
            if self.verbose:
                print(f"⚠️ Limited data for curves: {n_samples} total positions, {n_positive} splice sites")
                print("   Consider running inference on more genes for more reliable curves")
        
        # Generate plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curves
        ax_roc = axes[0]
        roc_results = {}
        
        for mode in comparison_modes:
            if mode in y_scores and len(y_scores[mode]) > 0:
                y_score = np.array(y_scores[mode])
                fpr, tpr, _ = roc_curve(y_true_all, y_score)
                roc_auc = auc(fpr, tpr)
                
                ax_roc.plot(fpr, tpr, label=f'{mode.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
                roc_results[mode] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC Curves - Inference Results\n({n_samples:,} positions, {n_positive:,} splice sites)')
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, alpha=0.3)
        
        # PR Curves
        ax_pr = axes[1]
        pr_results = {}
        
        for mode in comparison_modes:
            if mode in y_scores and len(y_scores[mode]) > 0:
                y_score = np.array(y_scores[mode])
                precision, recall, _ = precision_recall_curve(y_true_all, y_score)
                ap = average_precision_score(y_true_all, y_score)
                
                ax_pr.plot(recall, precision, label=f'{mode.replace("_", " ").title()} (AP = {ap:.3f})')
                pr_results[mode] = {'precision': precision, 'recall': recall, 'ap': ap}
        
        # Add baseline (random classifier)
        baseline = n_positive / n_samples
        ax_pr.axhline(y=baseline, color='k', linestyle='--', alpha=0.3, label=f'Baseline ({baseline:.3f})')
        
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f'Precision-Recall Curves - Inference Results\n({n_samples:,} positions, {n_positive:,} splice sites)')
        ax_pr.legend(loc="lower left")
        ax_pr.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.performance_dir / "roc_pr_curves_inference.pdf"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.savefig(plot_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calculate F1 scores at optimal thresholds
        f1_results = {}
        for mode in comparison_modes:
            if mode in y_scores and len(y_scores[mode]) > 0:
                scores = np.array(y_scores[mode])
            else:
                continue
            
            # Find optimal threshold for F1 score
            thresholds = np.linspace(0.01, 0.99, 50)
            best_f1 = 0
            best_thresh = 0.5
            for thresh in thresholds:
                y_pred = (scores >= thresh).astype(int)
                f1 = f1_score(y_true_all, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
            
            # Calculate metrics at optimal threshold
            y_pred = (scores >= best_thresh).astype(int)
            precision = precision_score(y_true_all, y_pred, zero_division=0)
            recall = recall_score(y_true_all, y_pred, zero_division=0)
            
            f1_results[mode] = {
                'f1': best_f1,
                'threshold': best_thresh,
                'precision': precision,
                'recall': recall
            }
        
        # Save comprehensive numerical results
        results_path = self.performance_dir / "curve_metrics.json"
        comparison_path = self.performance_dir / "mode_comparison.json"
        
        metrics = {
            'n_samples': int(n_samples),
            'n_positive': int(n_positive),
            'n_negative': int(n_negative),
            'roc': {mode: {'auc': float(res['auc'])} for mode, res in roc_results.items()},
            'pr': {mode: {'ap': float(res['ap'])} for mode, res in pr_results.items()},
            'f1': {mode: {
                'f1': float(res['f1']),
                'threshold': float(res['threshold']),
                'precision': float(res['precision']),
                'recall': float(res['recall'])
            } for mode, res in f1_results.items()}
        }
        
        # Create comprehensive mode comparison
        mode_comparison = {
            'dataset_info': {
                'n_samples': int(n_samples),
                'n_positive': int(n_positive),
                'n_negative': int(n_negative),
                'class_imbalance_ratio': float(n_negative / n_positive) if n_positive > 0 else 0
            },
            'mode_metrics': {},
            'improvements_over_base': {},
            'best_performers': {}
        }
        
        # Compile all metrics for each mode
        for mode in comparison_modes:
            if mode in roc_results or mode in pr_results or mode in f1_results:
                mode_comparison['mode_metrics'][mode] = {
                    'roc_auc': float(roc_results.get(mode, {}).get('auc', 0)),
                    'pr_ap': float(pr_results.get(mode, {}).get('ap', 0)),
                    'f1_score': float(f1_results.get(mode, {}).get('f1', 0)),
                    'precision': float(f1_results.get(mode, {}).get('precision', 0)),
                    'recall': float(f1_results.get(mode, {}).get('recall', 0)),
                    'optimal_threshold': float(f1_results.get(mode, {}).get('threshold', 0.5))
                }
        
        # Calculate improvements over base_only
        if 'base_only' in mode_comparison['mode_metrics']:
            base_metrics = mode_comparison['mode_metrics']['base_only']
            for mode in ['hybrid', 'meta_only']:
                if mode in mode_comparison['mode_metrics']:
                    mode_metrics = mode_comparison['mode_metrics'][mode]
                    mode_comparison['improvements_over_base'][mode] = {
                        'auc': {
                            'absolute': mode_metrics['roc_auc'] - base_metrics['roc_auc'],
                            'relative_pct': ((mode_metrics['roc_auc'] - base_metrics['roc_auc']) / base_metrics['roc_auc'] * 100) if base_metrics['roc_auc'] > 0 else 0
                        },
                        'ap': {
                            'absolute': mode_metrics['pr_ap'] - base_metrics['pr_ap'],
                            'relative_pct': ((mode_metrics['pr_ap'] - base_metrics['pr_ap']) / base_metrics['pr_ap'] * 100) if base_metrics['pr_ap'] > 0 else 0
                        },
                        'f1': {
                            'absolute': mode_metrics['f1_score'] - base_metrics['f1_score'],
                            'relative_pct': ((mode_metrics['f1_score'] - base_metrics['f1_score']) / base_metrics['f1_score'] * 100) if base_metrics['f1_score'] > 0 else 0
                        }
                    }
        
        # Identify best performers
        if mode_comparison['mode_metrics']:
            mode_comparison['best_performers'] = {
                'by_auc': max(mode_comparison['mode_metrics'].items(), key=lambda x: x[1]['roc_auc'])[0],
                'by_ap': max(mode_comparison['mode_metrics'].items(), key=lambda x: x[1]['pr_ap'])[0],
                'by_f1': max(mode_comparison['mode_metrics'].items(), key=lambda x: x[1]['f1_score'])[0]
            }
        
        # Save both files
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        with open(comparison_path, 'w') as f:
            json.dump(mode_comparison, f, indent=2)
        
        if self.verbose:
            print(f"\n📊 Performance Metrics Summary:")
            print(f"   Total samples: {n_samples:,}")
            print(f"   Positive samples: {n_positive:,} ({n_positive/n_samples*100:.1f}%)")
            print(f"   Negative samples: {n_negative:,} ({n_negative/n_samples*100:.1f}%)")
            
            # Highlight AP scores (most relevant for imbalanced data)
            print(f"\n   📈 Average Precision (Higher is Better for Imbalanced Data):")
            # Only show modes that were actually compared
            ap_scores = {mode: pr_results[mode]['ap'] for mode in comparison_modes if mode in pr_results}
            best_mode = max(ap_scores, key=ap_scores.get) if ap_scores else None
            for mode, ap in ap_scores.items():
                marker = "🏆" if mode == best_mode else "  "
                improvement = ""
                if mode != "base_only" and "base_only" in ap_scores:
                    delta = ap - ap_scores["base_only"]
                    improvement = f" (+{delta:.3f})" if delta > 0 else f" ({delta:.3f})"
                print(f"     {marker} {mode:12s}: {ap:.3f}{improvement}")
            
            # Show F1 scores
            print(f"\n   🎯 F1 Scores (at optimal threshold):")
            # Only show modes that were actually compared
            f1_scores = {mode: f1_results[mode]['f1'] for mode in comparison_modes if mode in f1_results}
            best_f1_mode = max(f1_scores, key=f1_scores.get) if f1_scores else None
            for mode in comparison_modes:
                if mode in f1_results:
                    res = f1_results[mode]
                marker = "🏆" if mode == best_f1_mode else "  "
                improvement = ""
                if mode != "base_only" and "base_only" in f1_results:
                    delta = res['f1'] - f1_results["base_only"]['f1']
                    improvement = f" (+{delta:.3f})" if delta > 0 else f" ({delta:.3f})"
                print(f"     {marker} {mode:12s}: {res['f1']:.3f} (thresh={res['threshold']:.2f}){improvement}")
            
            # Still show ROC AUC for completeness
            print(f"\n   ROC AUC (Less Informative for Imbalanced Data):")
            # Only show modes that were actually compared
            for mode in comparison_modes:
                if mode in roc_results:
                    res = roc_results[mode]
                    print(f"     {mode:12s}: {res['auc']:.3f}")
        
        # Generate human-readable comparison report
        report_path = self.performance_dir / "mode_comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("INFERENCE MODE PERFORMANCE COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset info
            f.write(f"Dataset Statistics:\n")
            f.write(f"  Total samples: {n_samples:,}\n")
            f.write(f"  Positive (splice sites): {n_positive:,} ({n_positive/n_samples*100:.2f}%)\n")
            f.write(f"  Negative (non-splice): {n_negative:,} ({n_negative/n_samples*100:.2f}%)\n")
            f.write(f"  Class imbalance ratio: 1:{n_negative/n_positive:.1f}\n\n")
            
            # Performance table
            f.write("-" * 80 + "\n")
            f.write("Performance Metrics by Mode:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Mode':<12} {'ROC-AUC':<10} {'PR-AP':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}\n")
            f.write("-" * 80 + "\n")
            
            for mode in comparison_modes:
                if mode in mode_comparison['mode_metrics']:
                    m = mode_comparison['mode_metrics'][mode]
                    f.write(f"{mode:<12} {m['roc_auc']:<10.4f} {m['pr_ap']:<10.4f} ")
                    f.write(f"{m['f1_score']:<10.4f} {m['precision']:<10.4f} {m['recall']:<10.4f}\n")
            
            # Improvements over base
            if 'improvements_over_base' in mode_comparison and mode_comparison['improvements_over_base']:
                f.write("\n" + "-" * 80 + "\n")
                f.write("Improvements Over Base Model:\n")
                f.write("-" * 80 + "\n")
                for mode, improvements in mode_comparison['improvements_over_base'].items():
                    f.write(f"\n{mode.upper()}:\n")
                    for metric, values in improvements.items():
                        f.write(f"  {metric.upper()}: {values['absolute']:+.4f} ({values['relative_pct']:+.1f}%)\n")
            
            # Best performers
            if 'best_performers' in mode_comparison and mode_comparison['best_performers']:
                f.write("\n" + "-" * 80 + "\n")
                f.write("Best Performing Modes:\n")
                f.write("-" * 80 + "\n")
                for metric, best_mode in mode_comparison['best_performers'].items():
                    f.write(f"  {metric.replace('_', ' ').upper()}: {best_mode}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        return {
            'plot': plot_path,
            'metrics': results_path,
            'comparison': comparison_path,
            'report': report_path
        }
    
    def generate_consolidated_report(self, all_gene_stats: Dict[str, Dict[str, Any]],
                                    all_predictions: Dict[str, pl.DataFrame]) -> Path:
        """Generate a consolidated performance report across all genes."""
        
        report_path = self.performance_dir / "consolidated_performance_report.txt"
        
        # Aggregate statistics
        total_positions = sum(stats['total_positions'] for stats in all_gene_stats.values())
        total_meta_positions = sum(stats['prediction_sources']['meta_model_positions'] for stats in all_gene_stats.values())
        total_base_positions = sum(stats['prediction_sources']['base_model_positions'] for stats in all_gene_stats.values())
        
        # Aggregate class distributions
        aggregated_current = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        aggregated_base = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
        
        for stats in all_gene_stats.values():
            for key in aggregated_current:
                aggregated_current[key] += stats['class_distribution'].get(key, 0)
                if 'base_only_distribution' in stats:
                    aggregated_base[key] += stats['base_only_distribution'].get(key, 0)
        
        with open(report_path, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"CONSOLIDATED PERFORMANCE REPORT - {len(all_gene_stats)} GENES\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")
            
            # Gene summary
            f.write("🧬 GENE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total genes analyzed: {len(all_gene_stats)}\n")
            f.write(f"Genes: {', '.join(list(all_gene_stats.keys())[:10])}")
            if len(all_gene_stats) > 10:
                f.write(f" ... and {len(all_gene_stats) - 10} more")
            f.write("\n\n")
            
            # Overall statistics
            f.write("📊 OVERALL STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total positions: {total_positions:,}\n")
            f.write(f"Meta-model positions: {total_meta_positions:,}\n")
            f.write(f"Base-model positions: {total_base_positions:,}\n")
            meta_usage = (total_meta_positions / total_positions * 100) if total_positions > 0 else 0
            f.write(f"Meta-model usage: {meta_usage:.2f}%\n\n")
            
            # Performance comparison
            if any('base_only_distribution' in stats for stats in all_gene_stats.values()):
                f.write("📈 AGGREGATE PERFORMANCE: BEFORE vs AFTER META-MODEL\n")
                f.write("-" * 50 + "\n")
                f.write("BEFORE (Base Model Only) → AFTER (With Meta-Model):\n\n")
                
                for key in ['FN', 'FP', 'TN', 'TP']:
                    base_val = aggregated_base[key]
                    current_val = aggregated_current[key]
                    delta = current_val - base_val
                    
                    base_pct = (base_val / total_positions * 100) if total_positions > 0 else 0
                    current_pct = (current_val / total_positions * 100) if total_positions > 0 else 0
                    
                    symbol = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
                    f.write(f" {key}: {base_val:7,} ({base_pct:5.1f}%) → {current_val:7,} ({current_pct:5.1f}%) {symbol} {delta:+,}\n")
                
                # Calculate aggregate metrics
                f.write(f"\n📊 AGGREGATE METRICS\n")
                f.write("-" * 20 + "\n")
                
                # Base metrics
                if aggregated_base['TP'] + aggregated_base['FN'] > 0:
                    base_recall = aggregated_base['TP'] / (aggregated_base['TP'] + aggregated_base['FN'])
                else:
                    base_recall = 0
                    
                if aggregated_base['TP'] + aggregated_base['FP'] > 0:
                    base_precision = aggregated_base['TP'] / (aggregated_base['TP'] + aggregated_base['FP'])
                else:
                    base_precision = 0
                
                base_f1 = 2 * base_precision * base_recall / (base_precision + base_recall) if (base_precision + base_recall) > 0 else 0
                
                # Current metrics
                if aggregated_current['TP'] + aggregated_current['FN'] > 0:
                    current_recall = aggregated_current['TP'] / (aggregated_current['TP'] + aggregated_current['FN'])
                else:
                    current_recall = 0
                    
                if aggregated_current['TP'] + aggregated_current['FP'] > 0:
                    current_precision = aggregated_current['TP'] / (aggregated_current['TP'] + aggregated_current['FP'])
                else:
                    current_precision = 0
                
                current_f1 = 2 * current_precision * current_recall / (current_precision + current_recall) if (current_precision + current_recall) > 0 else 0
                
                # Matthews Correlation Coefficient
                base_mcc = self._calculate_mcc(aggregated_base)
                current_mcc = self._calculate_mcc(aggregated_current)
                
                f.write(f"Precision: {base_precision:.3f} → {current_precision:.3f} ({current_precision - base_precision:+.3f})\n")
                f.write(f"Recall:    {base_recall:.3f} → {current_recall:.3f} ({current_recall - base_recall:+.3f})\n")
                f.write(f"F1 Score:  {base_f1:.3f} → {current_f1:.3f} ({current_f1 - base_f1:+.3f})\n")
                f.write(f"MCC:       {base_mcc:.3f} → {current_mcc:.3f} ({current_mcc - base_mcc:+.3f})\n")
                
                # Improvement summary
                f.write(f"\n🎯 IMPROVEMENT SUMMARY\n")
                f.write("-" * 20 + "\n")
                fp_reduction = aggregated_base['FP'] - aggregated_current['FP']
                fn_reduction = aggregated_base['FN'] - aggregated_current['FN']
                total_error_reduction = fp_reduction + fn_reduction
                
                f.write(f"False Positives reduced: {fp_reduction:,}\n")
                f.write(f"False Negatives reduced: {fn_reduction:,}\n")
                f.write(f"Total errors reduced: {total_error_reduction:,}\n")
                
                if aggregated_base['FP'] > 0:
                    fp_reduction_pct = (fp_reduction / aggregated_base['FP']) * 100
                    f.write(f"FP reduction rate: {fp_reduction_pct:.1f}%\n")
                
                if aggregated_base['FN'] > 0:
                    fn_reduction_pct = (fn_reduction / aggregated_base['FN']) * 100
                    f.write(f"FN reduction rate: {fn_reduction_pct:.1f}%\n")
            
            # Per-gene summary
            f.write(f"\n📋 PER-GENE SUMMARY\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Gene ID':<20} {'Positions':>10} {'Meta Used':>10} {'Meta %':>8} {'Net Benefit':>12}\n")
            f.write("-" * 50 + "\n")
            
            for gene_id, stats in all_gene_stats.items():
                positions = stats['total_positions']
                meta_used = stats['prediction_sources']['meta_model_positions']
                meta_pct = stats['prediction_sources']['meta_model_percentage']
                
                # Calculate net benefit for this gene
                if 'base_only_distribution' in stats:
                    base = stats['base_only_distribution']
                    current = stats['class_distribution']
                    net_benefit = (current.get('TP', 0) - base.get('TP', 0)) + \
                                 (current.get('TN', 0) - base.get('TN', 0)) - \
                                 (current.get('FP', 0) - base.get('FP', 0)) - \
                                 (current.get('FN', 0) - base.get('FN', 0))
                else:
                    net_benefit = 0
                
                f.write(f"{gene_id:<20} {positions:>10,} {meta_used:>10,} {meta_pct:>7.1f}% {net_benefit:>+12,}\n")
            
            f.write(f"\n{'='*70}\n")
        
        return report_path
    
    def _calculate_mcc(self, confusion: Dict[str, int]) -> float:
        """Calculate Matthews Correlation Coefficient from confusion matrix."""
        tp = confusion.get('TP', 0)
        tn = confusion.get('TN', 0)
        fp = confusion.get('FP', 0)
        fn = confusion.get('FN', 0)
        
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
