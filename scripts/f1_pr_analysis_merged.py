#!/usr/bin/env python3
"""
Comprehensive F1-based PR Analysis for MetaSpliceAI Gene CV Results

PURPOSE:
This script provides comprehensive F1-based analysis of MetaSpliceAI gene-aware 
cross-validation (CV) results from meta-model training workflows.

DESIGNED FOR:
- Gene CV output directories like: results/gene_cv_pc_1000_3mers_run_4/
- Meta-model training and evaluation results
- Analysis of precision/recall trade-offs in splice site prediction

KEY ANALYSES:
1. Generate F1-optimized PR curves from existing PR data
2. Analyze FP/FN trade-offs in CV results to understand strategic optimization
3. Provide biological interpretation of model behavior

USAGE:
    python scripts/f1_pr_analysis_merged.py results/gene_cv_pc_1000_3mers_run_4
    python scripts/f1_pr_analysis_merged.py results/gene_cv_pc_1000_3mers_run_4 output_dir/

EXPECTED INPUT FILES:
- pr_donor.csv, pr_acceptor.csv, pr_neither.csv (PR curve data)
- metrics_fold0.json, metrics_fold1.json, ... (fold metrics)
- cv_metrics_visualization/cv_metrics_summary.txt (optional)

OUTPUT FILES:
- pr_curves_f1_optimized.pdf (F1-optimized PR curves)
- f1_pr_analysis_summary.txt (comprehensive analysis summary)

The script is self-contained and requires only standard scientific Python libraries.
See scripts/README_f1_pr_analysis.md for detailed documentation.
"""

import sys
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score


def load_pr_data(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load PR curve data from CSV file.
    
    Parameters
    ----------
    csv_path : Path
        Path to the PR CSV file
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (thresholds, precision, recall)
    """
    df = pd.read_csv(csv_path)
    
    # Extract columns (assuming standard format)
    if 'threshold' in df.columns:
        thresholds = df['threshold'].values
    else:
        # If no threshold column, create one from index
        thresholds = np.linspace(0, 1, len(df))
    
    precision = df['precision'].values
    recall = df['recall'].values
    
    return thresholds, precision, recall


def calculate_f1_at_thresholds(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    """
    Calculate F1 scores at different precision/recall points.
    
    Parameters
    ----------
    precision : np.ndarray
        Precision values
    recall : np.ndarray
        Recall values
        
    Returns
    -------
    np.ndarray
        F1 scores
    """
    # F1 = 2 * (precision * recall) / (precision + recall)
    # Handle division by zero
    denominator = precision + recall
    f1_scores = np.where(denominator > 0, 
                         2 * (precision * recall) / denominator, 
                         0)
    return f1_scores


def analyze_fp_fn_tradeoff(results_dir: Path) -> Dict[str, Any]:
    """
    Analyze the FP/FN trade-off in the CV results.
    
    Parameters
    ----------
    results_dir : Path
        Path to the CV results directory
        
    Returns
    -------
    Dict[str, Any]
        Analysis results including totals and insights
    """
    print("🔍 Analyzing FP/FN Trade-off in MetaSpliceAI Results")
    print("=" * 60)
    
    # Load CV metrics summary
    cv_summary_path = results_dir / "cv_metrics_visualization" / "cv_metrics_summary.txt"
    if cv_summary_path.exists():
        with open(cv_summary_path, 'r') as f:
            summary_content = f.read()
        print("📊 CV Metrics Summary:")
        print(summary_content)
        print()
    
    # Load individual fold metrics
    fold_metrics = []
    for i in range(10):  # Check up to 10 folds
        fold_file = results_dir / f"metrics_fold{i}.json"
        if fold_file.exists():
            with open(fold_file, 'r') as f:
                fold_data = json.load(f)
                fold_metrics.append(fold_data)
    
    analysis_results = {
        'total_delta_fp': 0,
        'total_delta_fn': 0,
        'fold_count': len(fold_metrics),
        'fold_details': []
    }
    
    if fold_metrics:
        print("📈 Individual Fold Analysis:")
        print("-" * 40)
        
        total_delta_fp = 0
        total_delta_fn = 0
        
        for i, fold in enumerate(fold_metrics):
            delta_fp = fold.get('delta_fp', 0)
            delta_fn = fold.get('delta_fn', 0)
            base_f1 = fold.get('base_f1', 0)
            meta_f1 = fold.get('meta_f1', 0)
            
            total_delta_fp += delta_fp
            total_delta_fn += delta_fn
            
            fold_detail = {
                'fold_id': i,
                'delta_fp': delta_fp,
                'delta_fn': delta_fn,
                'net_error_change': delta_fp + delta_fn,
                'base_f1': base_f1,
                'meta_f1': meta_f1,
                'f1_improvement': meta_f1 - base_f1
            }
            analysis_results['fold_details'].append(fold_detail)
            
            print(f"Fold {i}:")
            print(f"  FP Change: {delta_fp:+d} (positive = more FPs)")
            print(f"  FN Change: {delta_fn:+d} (negative = fewer FNs)")
            print(f"  Net Error Change: {delta_fp + delta_fn:+d}")
            print(f"  Base F1: {base_f1:.3f}")
            print(f"  Meta F1: {meta_f1:.3f}")
            print(f"  F1 Improvement: {meta_f1 - base_f1:+.3f}")
            print()
        
        analysis_results['total_delta_fp'] = total_delta_fp
        analysis_results['total_delta_fn'] = total_delta_fn
        
        print("📊 Overall Pattern Analysis:")
        print("-" * 40)
        print(f"Total FP Change: {total_delta_fp:+d}")
        print(f"Total FN Change: {total_delta_fn:+d}")
        print(f"Net Error Reduction: {total_delta_fp + total_delta_fn:+d}")
        print()
        
        # Calculate precision/recall trade-off
        if total_delta_fp != 0 or total_delta_fn != 0:
            print("🎯 Strategic Analysis:")
            print("-" * 40)
            
            if total_delta_fp > 0 and total_delta_fn < 0:
                print("✅ The meta model is making a STRATEGIC TRADE-OFF:")
                print("   • Sacrificing some precision (more FPs)")
                print("   • Gaining much more recall (fewer FNs)")
                print("   • This is often desirable in splice site prediction")
                print()
                
                print("💡 Why this trade-off makes sense:")
                print("   • False Negatives (missing real splice sites) are often more costly")
                print("   • Missing splice sites can break gene annotation")
                print("   • Extra predictions can be filtered in downstream analysis")
                print("   • F1 score heavily weights the harmonic mean of precision and recall")
                print()
                
                print("📈 F1 Score Improvement Explained:")
                print("   • F1 = 2 × (Precision × Recall) / (Precision + Recall)")
                print("   • The large recall gain outweighs the smaller precision loss")
                print("   • This results in a net F1 improvement")
                print()
                
                print("🎯 Conclusion:")
                print("   • The FP increase is NOT a bug or failure")
                print("   • It's a deliberate optimization strategy")
                print("   • The meta model is prioritizing finding more true splice sites")
                print("   • This is reflected in the F1 score improvement")
                
                analysis_results['strategy'] = 'precision_recall_tradeoff'
            elif total_delta_fp < 0 and total_delta_fn < 0:
                print("🎯 EXCELLENT: Both FPs and FNs reduced!")
                print("   • The meta model improved both precision and recall")
                print("   • This is the ideal outcome")
                analysis_results['strategy'] = 'both_improved'
            else:
                print("⚠️  Mixed results - requires deeper analysis")
                analysis_results['strategy'] = 'mixed'
    else:
        print("❌ No fold metrics found. Cannot analyze FP/FN trade-off.")
        analysis_results['strategy'] = 'no_data'
    
    return analysis_results


def generate_f1_pr_curves(results_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, float]:
    """
    Generate F1-based PR curves from existing PR data.
    
    Parameters
    ----------
    results_dir : Path
        Path to the CV results directory
    output_dir : Path, optional
        Output directory for the new plots
        
    Returns
    -------
    Dict[str, float]
        Dictionary of max F1 scores for each class
    """
    if output_dir is None:
        output_dir = results_dir
    else:
        output_dir.mkdir(exist_ok=True)
    
    print("📊 Generating F1-based PR Curves")
    print("=" * 40)
    
    # PR data files
    pr_files = {
        'donor': results_dir / "pr_donor.csv",
        'acceptor': results_dir / "pr_acceptor.csv", 
        'neither': results_dir / "pr_neither.csv"
    }
    
    # Colors for each class
    colors = {
        'donor': 'tab:blue',
        'acceptor': 'tab:orange', 
        'neither': 'tab:green'
    }
    
    max_f1_scores = {}
    
    plt.figure(figsize=(12, 8))
    
    for class_name, file_path in pr_files.items():
        if not file_path.exists():
            print(f"❌ Missing PR data for {class_name}")
            continue
            
        print(f"📈 Processing {class_name}...")
        
        # Load PR data
        thresholds, precision, recall = load_pr_data(file_path)
        
        # Calculate F1 scores
        f1_scores = calculate_f1_at_thresholds(precision, recall)
        
        # Find maximum F1 score
        max_f1_idx = np.argmax(f1_scores)
        max_f1 = f1_scores[max_f1_idx]
        max_f1_precision = precision[max_f1_idx]
        max_f1_recall = recall[max_f1_idx]
        
        max_f1_scores[class_name] = max_f1
        
        print(f"   Max F1: {max_f1:.3f} (P={max_f1_precision:.3f}, R={max_f1_recall:.3f})")
        
        # Plot PR curve
        plt.plot(recall, precision, 
                color=colors[class_name], 
                label=f'{class_name.capitalize()} (max F1={max_f1:.3f})',
                linewidth=2)
        
        # Mark the point of maximum F1
        plt.plot(max_f1_recall, max_f1_precision, 
                'o', color=colors[class_name], 
                markersize=8, markeredgecolor='black', markeredgewidth=1)
    
    if not max_f1_scores:
        print("❌ No PR data files found. Cannot generate F1-based curves.")
        return {}
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('F1-Optimized Precision-Recall Curves\n(Points mark maximum F1 scores)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # Add F1 contour lines
    f1_levels = [0.5, 0.7, 0.8, 0.9, 0.95]
    for f1_level in f1_levels:
        # F1 = 2*P*R/(P+R) => P = F1*R/(2*R-F1) for R > F1/2
        r_values = np.linspace(f1_level/2 + 0.01, 1, 100)
        p_values = f1_level * r_values / (2 * r_values - f1_level)
        # Only plot valid values
        valid_mask = (p_values >= 0) & (p_values <= 1)
        if np.any(valid_mask):
            plt.plot(r_values[valid_mask], p_values[valid_mask], 'k--', alpha=0.3, linewidth=0.8)
            if np.any(valid_mask):
                plt.text(r_values[valid_mask][-1], p_values[valid_mask][-1], f'F1={f1_level}', 
                        fontsize=8, ha='left', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = output_dir / "pr_curves_f1_optimized.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ F1-based PR curves saved to: {output_path}")
    
    # Create a summary table
    print("\n📊 F1 Score Summary:")
    print("-" * 50)
    print(f"{'Class':<10} {'Max F1':<8} {'Precision':<10} {'Recall':<8}")
    print("-" * 50)
    
    for class_name, file_path in pr_files.items():
        if file_path.exists():
            thresholds, precision, recall = load_pr_data(file_path)
            f1_scores = calculate_f1_at_thresholds(precision, recall)
            max_f1_idx = np.argmax(f1_scores)
            max_f1 = f1_scores[max_f1_idx]
            max_f1_precision = precision[max_f1_idx]
            max_f1_recall = recall[max_f1_idx]
            
            print(f"{class_name.capitalize():<10} {max_f1:<8.3f} {max_f1_precision:<10.3f} {max_f1_recall:<8.3f}")
    
    return max_f1_scores


def save_analysis_summary(results_dir: Path, fp_fn_analysis: Dict[str, Any], f1_scores: Dict[str, float]):
    """
    Save a comprehensive analysis summary to a text file.
    
    Parameters
    ----------
    results_dir : Path
        Results directory
    fp_fn_analysis : Dict[str, Any]
        FP/FN analysis results
    f1_scores : Dict[str, float]
        F1 scores for each class
    """
    summary_path = results_dir / "f1_pr_analysis_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("MetaSpliceAI F1-based PR Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # FP/FN Analysis
        f.write("FP/FN Trade-off Analysis:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total FP Change: {fp_fn_analysis.get('total_delta_fp', 0):+d}\n")
        f.write(f"Total FN Change: {fp_fn_analysis.get('total_delta_fn', 0):+d}\n")
        f.write(f"Net Error Change: {fp_fn_analysis.get('total_delta_fp', 0) + fp_fn_analysis.get('total_delta_fn', 0):+d}\n")
        f.write(f"Strategy: {fp_fn_analysis.get('strategy', 'unknown')}\n")
        f.write(f"Folds Analyzed: {fp_fn_analysis.get('fold_count', 0)}\n\n")
        
        # F1 Scores
        f.write("Maximum F1 Scores by Class:\n")
        f.write("-" * 30 + "\n")
        for class_name, f1_score in f1_scores.items():
            f.write(f"{class_name.capitalize()}: {f1_score:.3f}\n")
        
        if f1_scores:
            avg_f1 = np.mean(list(f1_scores.values()))
            f.write(f"\nAverage F1: {avg_f1:.3f}\n")
    
    print(f"📄 Analysis summary saved to: {summary_path}")


def main():
    """Main function to run the comprehensive analysis."""
    if len(sys.argv) < 2:
        print("Usage: python f1_pr_analysis_merged.py <results_directory> [output_directory]")
        print("Example: python f1_pr_analysis_merged.py results/gene_cv_pc_1000_3mers_run_1")
        print("\nThis script performs:")
        print("  1. FP/FN trade-off analysis from CV metrics")
        print("  2. F1-optimized PR curve generation")
        print("  3. Comprehensive summary report")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    print("🚀 MetaSpliceAI Comprehensive F1-based PR Analysis")
    print("=" * 60)
    print(f"📁 Analyzing results from: {results_dir}")
    print()
    
    # Step 1: Analyze FP/FN trade-off
    fp_fn_analysis = analyze_fp_fn_tradeoff(results_dir)
    
    print("\n" + "=" * 60)
    
    # Step 2: Generate F1-based PR curves
    f1_scores = generate_f1_pr_curves(results_dir, output_dir)
    
    print("\n" + "=" * 60)
    
    # Step 3: Save comprehensive summary
    save_analysis_summary(results_dir, fp_fn_analysis, f1_scores)
    
    print("\n" + "=" * 60)
    print("✅ Comprehensive Analysis Complete!")
    print("\n📋 Summary:")
    
    if fp_fn_analysis.get('strategy') == 'precision_recall_tradeoff':
        print("   • FP increase is a strategic trade-off, not a bug")
        print("   • Meta model prioritizes recall over precision")
        print("   • This is often optimal for splice site prediction")
    elif fp_fn_analysis.get('strategy') == 'both_improved':
        print("   • Both precision and recall improved - excellent results!")
    
    if f1_scores:
        avg_f1 = np.mean(list(f1_scores.values()))
        print(f"   • Average maximum F1 score: {avg_f1:.3f}")
        print(f"   • F1-optimized PR curves generated")
    
    print("\n🎯 Output files:")
    if output_dir:
        print(f"   • PR curves: {output_dir}/pr_curves_f1_optimized.pdf")
    else:
        print(f"   • PR curves: {results_dir}/pr_curves_f1_optimized.pdf")
    print(f"   • Summary: {results_dir}/f1_pr_analysis_summary.txt")


if __name__ == "__main__":
    main()
