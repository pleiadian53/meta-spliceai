#!/usr/bin/env python3
"""
Generate F1-based PR Curve from Existing PR Curve Data

This script creates a PR curve that summarizes performance using F1 score
from the existing PR curve CSV files.

Part of the MetaSpliceAI evaluation package.
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

# Add the evaluation package to the path
sys.path.append(str(Path(__file__).parent.parent))

from viz_utils import plot_roc_pr_curves_f1

def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def load_pr_data(results_dir: str) -> Dict[str, pd.DataFrame]:
    """Load PR curve data from CSV files."""
    results_path = Path(results_dir)
    
    pr_data = {}
    for class_name in ['donor', 'acceptor', 'neither']:
        csv_file = results_path / f"pr_{class_name}.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            # Calculate F1 score for each point
            df['f1_score'] = df.apply(lambda row: calculate_f1_score(row['precision'], row['recall']), axis=1)
            pr_data[class_name] = df
            print(f"Loaded {len(df)} points for {class_name}")
        else:
            print(f"Warning: {csv_file} not found")
    
    return pr_data

def generate_f1_pr_curves(pr_data: Dict[str, pd.DataFrame], output_dir: Path):
    """Generate F1-based PR curves."""
    
    # Set up the plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Precision-Recall Curves with F1 Score Summary', fontsize=16, fontweight='bold')
    
    colors = {'donor': 'blue', 'acceptor': 'red', 'neither': 'green'}
    class_names = {'donor': 'Donor Sites', 'acceptor': 'Acceptor Sites', 'neither': 'Non-Splice Sites'}
    
    f1_summary = {}
    
    for i, (class_name, df) in enumerate(pr_data.items()):
        ax = axes[i]
        
        # Plot PR curve
        ax.plot(df['recall'], df['precision'], 
                color=colors[class_name], linewidth=2, alpha=0.8)
        
        # Find maximum F1 point
        max_f1_idx = df['f1_score'].idxmax()
        max_f1_precision = df.loc[max_f1_idx, 'precision']
        max_f1_recall = df.loc[max_f1_idx, 'recall']
        max_f1_score = df.loc[max_f1_idx, 'f1_score']
        
        # Mark maximum F1 point
        ax.scatter(max_f1_recall, max_f1_precision, 
                  color='red', s=100, zorder=5, 
                  label=f'Max F1: {max_f1_score:.3f}')
        
        # Calculate F1 at threshold=0.5 (more meaningful than averaging all points)
        # Find the point closest to threshold=0.5
        threshold = 0.5
        # For PR curves, we can use precision as a proxy for threshold
        # Find point with precision closest to 0.5
        precision_diff = np.abs(df['precision'] - threshold)
        threshold_idx = precision_diff.idxmin()
        threshold_f1 = df.loc[threshold_idx, 'f1_score']
        threshold_precision = df.loc[threshold_idx, 'precision']
        threshold_recall = df.loc[threshold_idx, 'recall']
        
        # Mark threshold point
        ax.scatter(threshold_recall, threshold_precision, 
                  color='orange', s=80, zorder=5, 
                  label=f'F1 at P=0.5: {threshold_f1:.3f}')
        
        # Add F1 contour lines
        f1_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
        for f1_level in f1_levels:
            # F1 = 2 * (P * R) / (P + R)
            # Solve for P given R and F1
            recalls = np.linspace(0.01, 0.99, 100)
            precisions = []
            for r in recalls:
                if f1_level == 0:
                    p = 0
                else:
                    p = (f1_level * r) / (2 * r - f1_level)
                if p > 0 and p <= 1:
                    precisions.append(p)
                else:
                    precisions.append(np.nan)
            
            ax.plot(recalls, precisions, '--', alpha=0.3, color='gray', linewidth=1)
            # Add F1 level label
            if len(precisions) > 0 and not np.isnan(precisions[50]):
                ax.text(0.5, precisions[50], f'F1={f1_level}', 
                       fontsize=8, alpha=0.7, ha='center')
        
        # Calculate more meaningful F1 metrics
        # 1. F1 at precision=0.5 (threshold point)
        # 2. F1 at recall=0.5 (balanced point)
        # 3. F1 at precision=0.8 (high precision point)
        
        # Find F1 at recall=0.5
        recall_diff = np.abs(df['recall'] - 0.5)
        recall_50_idx = recall_diff.idxmin()
        f1_at_recall_50 = df.loc[recall_50_idx, 'f1_score']
        
        # Find F1 at precision=0.8
        precision_80_diff = np.abs(df['precision'] - 0.8)
        precision_80_idx = precision_80_diff.idxmin()
        f1_at_precision_80 = df.loc[precision_80_idx, 'f1_score']
        
        # Store summary with more meaningful metrics
        f1_summary[class_name] = {
            'max_f1': max_f1_score,
            'f1_at_precision_50': threshold_f1,
            'f1_at_recall_50': f1_at_recall_50,
            'f1_at_precision_80': f1_at_precision_80,
            'max_f1_precision': max_f1_precision,
            'max_f1_recall': max_f1_recall,
            'threshold_precision': threshold_precision,
            'threshold_recall': threshold_recall
        }
        
        # Customize plot
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'{class_names[class_name]}\nMax F1: {max_f1_score:.3f}, F1 at P=0.5: {threshold_f1:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / "pr_curves_f1_optimized.pdf"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"F1-based PR curves saved to: {output_file}")
    
    # Save F1 summary
    summary_file = output_dir / "f1_pr_curve_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(f1_summary, f, indent=2)
    print(f"F1 summary saved to: {summary_file}")
    
    return f1_summary

def print_f1_summary(f1_summary: Dict[str, Dict]):
    """Print F1 score summary."""
    print("\n" + "="*60)
    print("F1 Score Summary for PR Curves")
    print("="*60)
    
    for class_name, metrics in f1_summary.items():
        print(f"\n{class_name.upper()} SITES:")
        print(f"  Maximum F1 Score: {metrics['max_f1']:.4f}")
        print(f"  F1 at Precision=0.5: {metrics['f1_at_precision_50']:.4f}")
        print(f"  F1 at Recall=0.5: {metrics['f1_at_recall_50']:.4f}")
        print(f"  F1 at Precision=0.8: {metrics['f1_at_precision_80']:.4f}")
        print(f"  At Max F1 - Precision: {metrics['max_f1_precision']:.4f}")
        print(f"  At Max F1 - Recall: {metrics['max_f1_recall']:.4f}")
        print(f"  At P=0.5 - Recall: {metrics['threshold_recall']:.4f}")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python generate_f1_pr_curves.py <results_dir>")
        print("\nExample:")
        print("  python generate_f1_pr_curves.py results/gene_cv_pc_1000_3mers_run_2_more_genes")
        print("\nThis script is part of the MetaSpliceAI evaluation package.")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' not found!")
        sys.exit(1)
    
    print(f"Loading PR curve data from: {results_path}")
    
    # Load PR curve data
    pr_data = load_pr_data(results_dir)
    
    if not pr_data:
        print("Error: No PR curve data found!")
        sys.exit(1)
    
    # Generate F1-based PR curves
    f1_summary = generate_f1_pr_curves(pr_data, results_path)
    
    # Print summary
    print_f1_summary(f1_summary)
    
    print("\n" + "="*60)
    print("✅ F1-based PR curve generation completed!")
    print("="*60)

if __name__ == "__main__":
    main() 