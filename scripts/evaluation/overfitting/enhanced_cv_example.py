#!/usr/bin/env python3
"""
Enhanced CV Example with Overfitting Monitoring

This demonstrates how to integrate comprehensive overfitting detection
into the existing gene CV and LOCO CV scripts.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# Add the parent directory to the path so we can import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from meta_spliceai.splice_engine.meta_models.evaluation.overfitting_monitor import (
    OverfittingMonitor, enhanced_model_training
)


def enhanced_gene_cv_with_monitoring(X, y, genes, args, out_dir: Path):
    """
    Enhanced gene-wise CV with comprehensive overfitting monitoring.
    
    This shows how to integrate the overfitting monitor into the existing
    gene CV workflow.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve
    
    # Initialize overfitting monitor
    monitor = OverfittingMonitor(
        primary_metric="logloss",
        gap_threshold=0.05,  # Flag overfitting if val loss > train loss by 0.05
        patience=20,         # Early stopping patience
        min_improvement=0.001
    )
    
    # Standard gene-wise CV setup
    gkf = GroupKFold(n_splits=args.n_folds)
    fold_metrics = []
    
    print(f"\n{'='*60}")
    print(f"ENHANCED GENE-WISE CV WITH OVERFITTING MONITORING")
    print(f"{'='*60}")
    print(f"Primary metric: {monitor.primary_metric}")
    print(f"Gap threshold: {monitor.gap_threshold}")
    print(f"Patience: {monitor.patience}")
    print(f"{'='*60}\n")
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(gkf.split(X, y, groups=genes)):
        print(f"🔄 Processing Fold {fold_idx + 1}/{args.n_folds}")
        
        # Split train/validation preserving gene groups
        from sklearn.model_selection import GroupShuffleSplit
        rel_valid = 0.2  # 20% of training for validation
        gss = GroupShuffleSplit(n_splits=1, test_size=rel_valid, random_state=args.seed)
        train_idx, valid_idx = next(gss.split(train_val_idx, y[train_val_idx], groups=genes[train_val_idx]))
        train_idx = train_val_idx[train_idx]
        valid_idx = train_val_idx[valid_idx]
        
        print(f"  📊 Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")
        
        # Enhanced model training with monitoring
        model = enhanced_model_training(
            X[train_idx], y[train_idx],
            X[valid_idx], y[valid_idx],
            args, monitor, fold_idx
        )
        
        # Standard evaluation
        y_pred = model.predict(X[test_idx])
        y_prob = model.predict_proba(X[test_idx])[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y[test_idx], y_pred)
        f1 = f1_score(y[test_idx], y_pred, average='macro')
        auc = roc_auc_score(y[test_idx], y_prob)
        
        # Get overfitting metrics for this fold
        fold_overfitting = monitor.fold_metrics[-1]  # Most recent fold
        
        print(f"  📈 Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        print(f"  ⚠️  Performance Gap: {fold_overfitting.performance_gap:.4f}")
        print(f"  🎯 Best Iteration: {fold_overfitting.best_iteration}")
        print(f"  🔄 Convergence: {fold_overfitting.convergence_iteration}")
        
        if fold_overfitting.performance_gap > monitor.gap_threshold:
            print(f"  🚨 OVERFITTING DETECTED! (Gap: {fold_overfitting.performance_gap:.4f})")
        else:
            print(f"  ✅ No significant overfitting detected")
        
        fold_metrics.append({
            'fold': fold_idx,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'performance_gap': fold_overfitting.performance_gap,
            'overfitting_score': fold_overfitting.overfitting_score,
            'best_iteration': fold_overfitting.best_iteration,
            'convergence_iteration': fold_overfitting.convergence_iteration,
            'early_stopped': fold_overfitting.early_stopped
        })
        
        print(f"  {'✓' if not fold_overfitting.early_stopped else '⏹️'} Early stopping: {'No' if not fold_overfitting.early_stopped else 'Yes'}")
        print()
    
    # Generate comprehensive overfitting report
    print("📋 Generating comprehensive overfitting analysis...")
    overfitting_report = monitor.generate_overfitting_report(out_dir)
    
    # Create visualizations
    print("📊 Creating overfitting visualizations...")
    monitor.plot_learning_curves(out_dir, plot_format="pdf")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"OVERFITTING ANALYSIS SUMMARY")
    print(f"{'='*60}")
    summary = overfitting_report['summary']
    print(f"Total folds: {summary['total_folds']}")
    print(f"Folds with overfitting: {summary['folds_with_overfitting']}")
    print(f"Early stopped folds: {summary['early_stopped_folds']}")
    print(f"Mean performance gap: {summary['mean_performance_gap']:.4f} ± {summary['std_performance_gap']:.4f}")
    print(f"Mean overfitting score: {summary['mean_overfitting_score']:.4f}")
    print(f"Recommended n_estimators: {summary['recommended_n_estimators']}")
    print(f"{'='*60}")
    
    # Save fold metrics
    fold_df = pd.DataFrame(fold_metrics)
    fold_df.to_csv(out_dir / "enhanced_cv_metrics.csv", index=False)
    
    return fold_df, overfitting_report


def create_overfitting_dashboard(out_dir: Path, fold_df: pd.DataFrame, overfitting_report: Dict):
    """Create a comprehensive overfitting dashboard."""
    
    # Create dashboard figure
    fig = plt.figure(figsize=(20, 16))
    
    # Create a grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Performance Gap vs Fold
    ax1 = fig.add_subplot(gs[0, :2])
    bars = ax1.bar(fold_df['fold'], fold_df['performance_gap'], alpha=0.7, color='red')
    ax1.axhline(y=0.05, color='orange', linestyle='--', label='Overfitting Threshold')
    ax1.set_xlabel('CV Fold')
    ax1.set_ylabel('Performance Gap')
    ax1.set_title('Training-Validation Performance Gap by Fold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Color bars based on threshold
    for bar, gap in zip(bars, fold_df['performance_gap']):
        if gap > 0.05:
            bar.set_color('red')
        else:
            bar.set_color('green')
    
    # 2. Best Iteration Distribution
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.hist(fold_df['best_iteration'], bins=10, alpha=0.7, color='blue')
    ax2.axvline(x=fold_df['best_iteration'].mean(), color='red', linestyle='--', 
                label=f'Mean: {fold_df["best_iteration"].mean():.1f}')
    ax2.set_xlabel('Best Iteration')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Best Iteration Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance Metrics vs Overfitting
    ax3 = fig.add_subplot(gs[1, :2])
    scatter = ax3.scatter(fold_df['performance_gap'], fold_df['f1'], 
                         c=fold_df['overfitting_score'], cmap='viridis', s=100)
    ax3.set_xlabel('Performance Gap')
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score vs Performance Gap')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Overfitting Score')
    
    # 4. Convergence Analysis
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.plot(fold_df['fold'], fold_df['best_iteration'], 'o-', label='Best Iteration', linewidth=2)
    ax4.plot(fold_df['fold'], fold_df['convergence_iteration'], 's-', label='Convergence Point', linewidth=2)
    ax4.set_xlabel('CV Fold')
    ax4.set_ylabel('Iteration Number')
    ax4.set_title('Convergence Analysis')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Overfitting Score Heatmap
    ax5 = fig.add_subplot(gs[2, :])
    overfitting_matrix = fold_df[['fold', 'performance_gap', 'overfitting_score', 'f1', 'auc']].set_index('fold').T
    sns.heatmap(overfitting_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax5)
    ax5.set_title('Overfitting Metrics Heatmap')
    
    # 6. Summary Statistics
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    summary_text = f"""
    OVERFITTING ANALYSIS SUMMARY
    
    Total Folds: {overfitting_report['summary']['total_folds']}
    Folds with Overfitting: {overfitting_report['summary']['folds_with_overfitting']}
    Early Stopped Folds: {overfitting_report['summary']['early_stopped_folds']}
    
    Mean Performance Gap: {overfitting_report['summary']['mean_performance_gap']:.4f} ± {overfitting_report['summary']['std_performance_gap']:.4f}
    Mean Overfitting Score: {overfitting_report['summary']['mean_overfitting_score']:.4f}
    
    Recommended n_estimators: {overfitting_report['summary']['recommended_n_estimators']}
    Mean Best Iteration: {overfitting_report['summary']['mean_best_iteration']:.1f} ± {overfitting_report['summary']['std_best_iteration']:.1f}
    
    RECOMMENDATIONS:
    • {'Consider reducing n_estimators' if overfitting_report['summary']['folds_with_overfitting'] > 0 else 'Current model complexity appears appropriate'}
    • {'Enable early stopping' if overfitting_report['summary']['early_stopped_folds'] == 0 else 'Early stopping is working effectively'}
    • {'Consider regularization' if overfitting_report['summary']['mean_performance_gap'] > 0.05 else 'Generalization gap is acceptable'}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=12, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Comprehensive Overfitting Analysis Dashboard', fontsize=16, fontweight='bold')
    plt.savefig(out_dir / "overfitting_dashboard.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Overfitting dashboard saved to: {out_dir / 'overfitting_dashboard.pdf'}")


def generate_overfitting_recommendations(overfitting_report: Dict) -> List[str]:
    """Generate actionable recommendations based on overfitting analysis."""
    recommendations = []
    summary = overfitting_report['summary']
    
    # Check for overfitting
    if summary['folds_with_overfitting'] > 0:
        recommendations.append(
            f"🚨 OVERFITTING DETECTED: {summary['folds_with_overfitting']}/{summary['total_folds']} folds show overfitting. "
            f"Consider reducing n_estimators from current value to ~{summary['recommended_n_estimators']}."
        )
    
    # Check early stopping effectiveness
    if summary['early_stopped_folds'] == 0:
        recommendations.append(
            "⏹️ EARLY STOPPING: No folds triggered early stopping. "
            "Consider enabling early stopping with patience=20-50 to prevent overfitting."
        )
    
    # Check iteration consistency
    if summary['std_best_iteration'] > 50:
        recommendations.append(
            f"📊 ITERATION VARIANCE: High variance in best iterations (std={summary['std_best_iteration']:.1f}). "
            "This suggests unstable training. Consider increasing regularization or reducing learning rate."
        )
    
    # Check performance gap
    if summary['mean_performance_gap'] > 0.1:
        recommendations.append(
            f"📈 LARGE GAP: Mean performance gap ({summary['mean_performance_gap']:.4f}) is large. "
            "Consider: 1) Reducing model complexity, 2) Adding regularization, 3) Increasing training data."
        )
    
    # Positive recommendations
    if summary['folds_with_overfitting'] == 0:
        recommendations.append(
            "✅ GOOD GENERALIZATION: No significant overfitting detected. "
            "Current model complexity appears appropriate."
        )
    
    if summary['early_stopped_folds'] > 0:
        recommendations.append(
            f"✅ EFFECTIVE EARLY STOPPING: {summary['early_stopped_folds']} folds benefited from early stopping. "
            "Continue using early stopping to optimize training efficiency."
        )
    
    return recommendations


def main():
    """Example usage of enhanced CV with overfitting monitoring."""
    
    # Example with synthetic data
    print("🧪 Creating synthetic dataset for demonstration...")
    np.random.seed(42)
    
    # Create synthetic data
    n_samples = 1000
    n_features = 50
    n_genes = 20
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    genes = np.random.choice(range(n_genes), n_samples)
    
    # Simple args object
    class Args:
        n_folds = 5
        n_estimators = 100
        tree_method = "hist"
        max_bin = 256
        seed = 42
        device = "auto"
    
    args = Args()
    out_dir = Path("overfitting_demo_output")
    out_dir.mkdir(exist_ok=True)
    
    # Run enhanced CV
    fold_df, overfitting_report = enhanced_gene_cv_with_monitoring(X, y, genes, args, out_dir)
    
    # Create dashboard
    create_overfitting_dashboard(out_dir, fold_df, overfitting_report)
    
    # Generate recommendations
    recommendations = generate_overfitting_recommendations(overfitting_report)
    
    print(f"\n{'='*60}")
    print("📋 ACTIONABLE RECOMMENDATIONS")
    print(f"{'='*60}")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\n📁 All results saved to: {out_dir}")
    print("📊 Generated files:")
    print("   - overfitting_analysis.json")
    print("   - enhanced_cv_metrics.csv")
    print("   - learning_curves_by_fold.pdf")
    print("   - aggregated_learning_curves.pdf")
    print("   - overfitting_summary.pdf")
    print("   - overfitting_dashboard.pdf")


if __name__ == "__main__":
    main() 