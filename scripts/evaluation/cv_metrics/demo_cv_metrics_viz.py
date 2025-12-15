#!/usr/bin/env python3
"""
Demo script for CV metrics visualization.

This script demonstrates how to use the CV metrics visualization module
to create comprehensive plots comparing base vs meta model performance.
"""

import sys
from pathlib import Path
import pandas as pd

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_spliceai.splice_engine.meta_models.evaluation.cv_metrics_viz import (
    generate_cv_metrics_report,
    load_cv_metrics
)

def demo_cv_metrics_visualization():
    """Demo the CV metrics visualization functionality."""
    
    print("CV Metrics Visualization Demo")
    print("=" * 50)
    
    # Example paths - modify these to match your setup
    csv_path = "models/meta_model_test/gene_cv_metrics.csv"
    out_dir = "demo_cv_viz_output"
    
    # Check if the CSV file exists
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Please provide a valid path to gene_cv_metrics.csv")
        print("\nExample usage:")
        print("python scripts/demo_cv_metrics_viz.py")
        print("Or modify the csv_path variable in this script")
        return
    
    # Load and inspect the data
    print(f"Loading CV metrics from: {csv_path}")
    df = load_cv_metrics(csv_path)
    print(f"Loaded {len(df)} CV folds")
    print(f"Available columns: {list(df.columns)}")
    
    # Display basic statistics
    print("\n" + "="*50)
    print("Basic Statistics")
    print("="*50)
    
    if 'base_f1' in df.columns and 'meta_f1' in df.columns:
        print(f"F1 Score - Base: {df['base_f1'].mean():.3f} ± {df['base_f1'].std():.3f}")
        print(f"F1 Score - Meta: {df['meta_f1'].mean():.3f} ± {df['meta_f1'].std():.3f}")
        print(f"F1 Improvement: {(df['meta_f1'] - df['base_f1']).mean():.3f} ± {(df['meta_f1'] - df['base_f1']).std():.3f}")
    
    if 'auc_base' in df.columns and 'auc_meta' in df.columns:
        print(f"ROC AUC - Base: {df['auc_base'].mean():.3f} ± {df['auc_base'].std():.3f}")
        print(f"ROC AUC - Meta: {df['auc_meta'].mean():.3f} ± {df['auc_meta'].std():.3f}")
        print(f"AUC Improvement: {(df['auc_meta'] - df['auc_base']).mean():.3f} ± {(df['auc_meta'] - df['auc_base']).std():.3f}")
    
    if 'ap_base' in df.columns and 'ap_meta' in df.columns:
        print(f"Average Precision - Base: {df['ap_base'].mean():.3f} ± {df['ap_base'].std():.3f}")
        print(f"Average Precision - Meta: {df['ap_meta'].mean():.3f} ± {df['ap_meta'].std():.3f}")
        print(f"AP Improvement: {(df['ap_meta'] - df['ap_base']).mean():.3f} ± {(df['ap_meta'] - df['ap_base']).std():.3f}")
    
    if 'delta_fp' in df.columns and 'delta_fn' in df.columns:
        print(f"False Positive Reduction: {df['delta_fp'].mean():.1f} ± {df['delta_fp'].std():.1f}")
        print(f"False Negative Reduction: {df['delta_fn'].mean():.1f} ± {df['delta_fn'].std():.1f}")
        print(f"Total Error Reduction: {(df['delta_fp'] + df['delta_fn']).mean():.1f} ± {(df['delta_fp'] + df['delta_fn']).std():.1f}")
    
    # Generate comprehensive visualization report
    print("\n" + "="*50)
    print("Generating Visualization Report")
    print("="*50)
    
    try:
        result = generate_cv_metrics_report(
            csv_path=csv_path,
            out_dir=out_dir,
            plot_format='png',
            dpi=300
        )
        
        print(f"✓ Visualization report generated successfully!")
        print(f"✓ Output directory: {result['visualization_dir']}")
        print(f"✓ Summary report: {result['report_path']}")
        print(f"✓ Generated {len(result['plot_files'])} plots:")
        
        for plot_name, plot_path in result['plot_files'].items():
            print(f"  - {plot_name.replace('_', ' ').title()}: {Path(plot_path).name}")
        
        # Display key insights
        print("\n" + "="*50)
        print("Key Insights")
        print("="*50)
        
        stats = result['summary_stats']
        
        if 'f1_improvement_pct' in stats:
            print(f"• F1 Score improved by {stats['f1_improvement_pct']:.1f}% on average")
        
        if 'auc_improvement_pct' in stats:
            print(f"• ROC AUC improved by {stats['auc_improvement_pct']:.1f}% on average")
        
        if 'ap_improvement_pct' in stats:
            print(f"• Average Precision improved by {stats['ap_improvement_pct']:.1f}% on average")
        
        if 'total_error_reduction' in stats:
            total_error_reduction = stats['total_error_reduction']
            if total_error_reduction > 0:
                print(f"• Total error reduction: {total_error_reduction:.1f} fewer errors per fold")
            else:
                print(f"• Total error increase: {abs(total_error_reduction):.1f} more errors per fold")
        
        print("\n" + "="*50)
        print("Generated Plots Summary")
        print("="*50)
        
        print("1. F1 Comparison Plot:")
        print("   - Fold-by-fold F1 score comparison")
        print("   - Statistical summary with error bars")
        
        print("\n2. ROC AUC Comparison Plot:")
        print("   - Fold-by-fold ROC AUC comparison")
        print("   - Statistical summary with error bars")
        
        print("\n3. Average Precision Comparison Plot:")
        print("   - Fold-by-fold Average Precision comparison")
        print("   - Statistical summary with error bars")
        
        print("\n4. Error Reduction Analysis Plot:")
        print("   - False Positive reduction across folds")
        print("   - False Negative reduction across folds")
        print("   - Combined error reduction summary")
        print("   - Per-fold total error reduction")
        
        print("\n5. Performance Overview Plot:")
        print("   - Multi-metric comparison (F1, AUC, AP)")
        print("   - Accuracy metrics breakdown")
        print("   - F1 score metrics breakdown")
        print("   - Top-k accuracy distribution")
        
        print("\n6. Improvement Summary Plot:")
        print("   - Percentage improvements for each metric")
        print("   - Error reduction effectiveness scatter plot")
        print("   - Quadrant analysis showing improvement patterns")
        
        if 'topk_analysis' in result['plot_files']:
            print("\n7. Top-k Analysis Plot:")
            print("   - Top-k accuracy across folds")
            print("   - Donor vs Acceptor top-k comparison")
            print("   - Top-k accuracy distribution")
            print("   - Gene count analysis per fold")
        
        print(f"\n✓ All plots saved to: {result['visualization_dir']}")
        print(f"✓ Open the summary report: {result['report_path']}")
        
    except Exception as e:
        print(f"✗ Error generating visualization report: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point."""
    demo_cv_metrics_visualization()

if __name__ == "__main__":
    main() 