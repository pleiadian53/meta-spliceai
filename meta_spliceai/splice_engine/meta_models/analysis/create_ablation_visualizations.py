#!/usr/bin/env python3
"""
Create Publication-Ready Ablation Study Visualizations

This script creates high-quality plots that tell the story of the ablation study.

Usage:
    python -m meta_spliceai.splice_engine.meta_models.analysis.create_ablation_visualizations \
      --results-file results/ablation_study_xlarge/ablation_summary_corrected.json \
      --output-dir results/ablation_study_xlarge/plots
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any

# Set publication-ready style
plt.style.use('default')
sns.set_palette("husl")

# Publication settings
FIGURE_DPI = 300
FIGURE_SIZE_SINGLE = (10, 8)
FIGURE_SIZE_WIDE = (16, 6)
FIGURE_SIZE_TALL = (12, 10)

def load_and_process_results(results_file: str) -> Dict[str, Any]:
    """Load ablation results and compute averages across trials."""
    
    print(f"📊 Loading results from: {results_file}")
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Group by mode and compute statistics
    mode_stats = {}
    
    for mode in ['full', 'no_probs', 'no_kmer', 'only_probs', 'only_kmer']:
        mode_results = [r for r in results if r.get('mode') == mode and 'f1_macro' in r]
        
        if not mode_results:
            continue
        
        # Extract metrics
        f1_scores = [r['f1_macro'] for r in mode_results]
        ap_scores = [r['ap_macro'] for r in mode_results]
        roc_scores = [r['roc_auc_macro'] for r in mode_results if r['roc_auc_macro'] is not None]
        
        # Per-class F1 scores
        f1_per_class_all = [r['f1_per_class'] for r in mode_results]
        f1_neither = [f[0] for f in f1_per_class_all]  # Class 0: neither
        f1_donor = [f[1] for f in f1_per_class_all]    # Class 1: donor  
        f1_acceptor = [f[2] for f in f1_per_class_all] # Class 2: acceptor
        
        mode_stats[mode] = {
            'f1_macro_mean': np.mean(f1_scores),
            'f1_macro_std': np.std(f1_scores),
            'ap_macro_mean': np.mean(ap_scores),
            'ap_macro_std': np.std(ap_scores),
            'roc_auc_mean': np.mean(roc_scores) if roc_scores else np.nan,
            'roc_auc_std': np.std(roc_scores) if roc_scores else np.nan,
            'f1_neither_mean': np.mean(f1_neither),
            'f1_neither_std': np.std(f1_neither),
            'f1_donor_mean': np.mean(f1_donor),
            'f1_donor_std': np.std(f1_donor),
            'f1_acceptor_mean': np.mean(f1_acceptor),
            'f1_acceptor_std': np.std(f1_acceptor),
            'n_features': mode_results[0]['n_features'],
            'n_trials': len(mode_results)
        }
    
    print(f"Processed {len(mode_stats)} ablation modes")
    
    return {
        'mode_stats': mode_stats,
        'config': data['experiment_config']
    }

def create_main_comparison_plot(mode_stats: Dict[str, Any], output_dir: str):
    """Create the main comparison plot showing all metrics."""
    
    print("🎨 Creating main comparison plot...")
    
    # Prepare data
    modes = ['full', 'no_kmer', 'only_probs', 'no_probs', 'only_kmer']
    mode_labels = ['Full Model', 'No K-mers', 'Only SpliceAI', 'No SpliceAI', 'Only K-mers']
    
    f1_means = [mode_stats[mode]['f1_macro_mean'] for mode in modes]
    f1_stds = [mode_stats[mode]['f1_macro_std'] for mode in modes]
    
    ap_means = [mode_stats[mode]['ap_macro_mean'] for mode in modes]
    ap_stds = [mode_stats[mode]['ap_macro_std'] for mode in modes]
    
    roc_means = [mode_stats[mode]['roc_auc_mean'] if not np.isnan(mode_stats[mode]['roc_auc_mean']) else 0 
                 for mode in modes]
    roc_stds = [mode_stats[mode]['roc_auc_std'] if not np.isnan(mode_stats[mode]['roc_auc_std']) else 0 
                for mode in modes]
    
    # Colors for different feature types
    colors = ['#2E8B57', '#228B22', '#4169E1', '#DC143C', '#FF8C00']  # Green, Green, Blue, Red, Orange
    
    # Create the plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE, dpi=FIGURE_DPI)
    
    x = np.arange(len(modes))
    width = 0.25
    
    # Plot bars with error bars
    bars1 = ax.bar(x - width, f1_means, width, yerr=f1_stds, label='F1 Macro', 
                   color=colors, alpha=0.8, capsize=5)
    bars2 = ax.bar(x, ap_means, width, yerr=ap_stds, label='Average Precision', 
                   color=colors, alpha=0.6, capsize=5)
    bars3 = ax.bar(x + width, roc_means, width, yerr=roc_stds, label='ROC AUC', 
                   color=colors, alpha=0.4, capsize=5)
    
    # Customize plot
    ax.set_xlabel('Ablation Mode', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Score', fontsize=14, fontweight='bold')
    ax.set_title('Meta-Model Ablation Study Results\n(Averaged across 5 trials)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    # Add value labels on bars
    for i, (f1, ap, roc) in enumerate(zip(f1_means, ap_means, roc_means)):
        ax.text(i - width, f1 + f1_stds[i] + 0.01, f'{f1:.3f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(i, ap + ap_stds[i] + 0.01, f'{ap:.3f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        if roc > 0:
            ax.text(i + width, roc + roc_stds[i] + 0.01, f'{roc:.3f}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / "ablation_main_comparison.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pdf'), dpi=FIGURE_DPI, bbox_inches='tight')
    
    print(f"✅ Saved main comparison plot: {output_file}")
    plt.close()

def create_feature_contribution_plot(mode_stats: Dict[str, Any], output_dir: str):
    """Create a plot showing feature contribution analysis."""
    
    print("🎨 Creating feature contribution plot...")
    
    # Calculate feature contributions
    full_f1 = mode_stats['full']['f1_macro_mean']
    
    contributions = {
        'SpliceAI Features\n(Probability + Context)': full_f1 - mode_stats['no_probs']['f1_macro_mean'],
        'K-mer Features\n(Sequence Context)': full_f1 - mode_stats['no_kmer']['f1_macro_mean'],
        'Genomic Features\n(Position + Structure)': full_f1 - mode_stats['only_probs']['f1_macro_mean']
    }
    
    # Performance levels
    performance_levels = {
        'Full Model': mode_stats['full']['f1_macro_mean'],
        'SpliceAI Only': mode_stats['only_probs']['f1_macro_mean'],
        'K-mers Only': mode_stats['only_kmer']['f1_macro_mean'],
        'No SpliceAI': mode_stats['no_probs']['f1_macro_mean']
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE_WIDE, dpi=FIGURE_DPI)
    
    # Plot 1: Feature Contributions (how much performance drops without each feature type)
    feature_types = list(contributions.keys())
    contrib_values = [abs(v) for v in contributions.values()]
    colors1 = ['#DC143C', '#FF8C00', '#4169E1']
    
    bars1 = ax1.bar(feature_types, contrib_values, color=colors1, alpha=0.8)
    ax1.set_ylabel('F1 Macro Drop', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Type Importance\n(Performance Drop When Removed)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars1, contrib_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xticklabels(feature_types, rotation=45, ha='right')
    
    # Plot 2: Performance Hierarchy
    perf_labels = list(performance_levels.keys())
    perf_values = list(performance_levels.values())
    colors2 = ['#2E8B57', '#4169E1', '#FF8C00', '#DC143C']
    
    bars2 = ax2.bar(perf_labels, perf_values, color=colors2, alpha=0.8)
    ax2.set_ylabel('F1 Macro Score', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Hierarchy\n(Different Feature Combinations)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1.05)
    
    # Add value labels
    for bar, value in zip(bars2, perf_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xticklabels(perf_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / "ablation_feature_contribution.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pdf'), dpi=FIGURE_DPI, bbox_inches='tight')
    
    print(f"✅ Saved feature contribution plot: {output_file}")
    plt.close()

def create_per_class_performance_plot(mode_stats: Dict[str, Any], output_dir: str):
    """Create a plot showing per-class performance breakdown."""
    
    print("🎨 Creating per-class performance plot...")
    
    modes = ['full', 'only_probs', 'no_probs', 'only_kmer']
    mode_labels = ['Full Model', 'Only SpliceAI', 'No SpliceAI', 'Only K-mers']
    
    # Extract per-class F1 scores
    neither_scores = [mode_stats[mode]['f1_neither_mean'] for mode in modes]
    donor_scores = [mode_stats[mode]['f1_donor_mean'] for mode in modes]
    acceptor_scores = [mode_stats[mode]['f1_acceptor_mean'] for mode in modes]
    
    neither_stds = [mode_stats[mode]['f1_neither_std'] for mode in modes]
    donor_stds = [mode_stats[mode]['f1_donor_std'] for mode in modes]
    acceptor_stds = [mode_stats[mode]['f1_acceptor_std'] for mode in modes]
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE, dpi=FIGURE_DPI)
    
    x = np.arange(len(modes))
    width = 0.25
    
    bars1 = ax.bar(x - width, neither_scores, width, yerr=neither_stds, 
                   label='Neither (Non-splice)', color='#808080', alpha=0.8, capsize=5)
    bars2 = ax.bar(x, donor_scores, width, yerr=donor_stds, 
                   label='Donor Sites', color='#1E90FF', alpha=0.8, capsize=5)
    bars3 = ax.bar(x + width, acceptor_scores, width, yerr=acceptor_stds, 
                   label='Acceptor Sites', color='#32CD32', alpha=0.8, capsize=5)
    
    ax.set_xlabel('Ablation Mode', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    ax.set_title('Per-Class Performance Breakdown\n(F1 Scores by Splice Site Type)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(mode_labels, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    # Add value labels
    for i, (neither, donor, acceptor) in enumerate(zip(neither_scores, donor_scores, acceptor_scores)):
        ax.text(i - width, neither + neither_stds[i] + 0.01, f'{neither:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(i, donor + donor_stds[i] + 0.01, f'{donor:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(i + width, acceptor + acceptor_stds[i] + 0.01, f'{acceptor:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / "ablation_per_class_performance.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pdf'), dpi=FIGURE_DPI, bbox_inches='tight')
    
    print(f"✅ Saved per-class performance plot: {output_file}")
    plt.close()

def create_ablation_summary_plot(mode_stats: Dict[str, Any], output_dir: str):
    """Create a comprehensive summary visualization."""
    
    print("🎨 Creating ablation summary plot...")
    
    fig = plt.figure(figsize=(16, 12), dpi=FIGURE_DPI)
    
    # Create a 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: F1 Macro comparison
    ax1 = fig.add_subplot(gs[0, 0])
    modes = ['full', 'no_kmer', 'only_probs', 'no_probs', 'only_kmer']
    mode_labels = ['Full', 'No K-mers', 'SpliceAI Only', 'No SpliceAI', 'K-mers Only']
    f1_means = [mode_stats[mode]['f1_macro_mean'] for mode in modes]
    f1_stds = [mode_stats[mode]['f1_macro_std'] for mode in modes]
    
    bars = ax1.bar(mode_labels, f1_means, yerr=f1_stds, capsize=5,
                   color=['#2E8B57', '#228B22', '#4169E1', '#DC143C', '#FF8C00'], alpha=0.8)
    ax1.set_ylabel('F1 Macro', fontweight='bold')
    ax1.set_title('A) Overall Performance (F1 Macro)', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.05)
    
    for bar, f1, std in zip(bars, f1_means, f1_stds):
        ax1.text(bar.get_x() + bar.get_width()/2., f1 + std + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 2: Feature counts
    ax2 = fig.add_subplot(gs[0, 1])
    feature_counts = [mode_stats[mode]['n_features'] for mode in modes]
    bars2 = ax2.bar(mode_labels, feature_counts, 
                    color=['#2E8B57', '#228B22', '#4169E1', '#DC143C', '#FF8C00'], alpha=0.8)
    ax2.set_ylabel('Number of Features', fontweight='bold')
    ax2.set_title('B) Feature Count by Mode', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars2, feature_counts):
        ax2.text(bar.get_x() + bar.get_width()/2., count + 50,
                f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 3: SpliceAI importance
    ax3 = fig.add_subplot(gs[1, 0])
    spliceai_comparison = {
        'With SpliceAI\n(Full Model)': mode_stats['full']['f1_macro_mean'],
        'Without SpliceAI\n(No Probs)': mode_stats['no_probs']['f1_macro_mean']
    }
    
    bars3 = ax3.bar(spliceai_comparison.keys(), spliceai_comparison.values(),
                    color=['#2E8B57', '#DC143C'], alpha=0.8)
    ax3.set_ylabel('F1 Macro', fontweight='bold')
    ax3.set_title('C) SpliceAI Feature Importance', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1.05)
    
    # Add performance drop annotation
    drop = mode_stats['full']['f1_macro_mean'] - mode_stats['no_probs']['f1_macro_mean']
    ax3.annotate(f'Performance Drop:\n-{drop:.3f} F1 ({drop/mode_stats["full"]["f1_macro_mean"]*100:.1f}%)',
                xy=(1, mode_stats['no_probs']['f1_macro_mean']), 
                xytext=(0.5, 0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Plot 4: Biological interpretation
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Text summary
    summary_text = f"""
    Key Findings:
    
    ✅ SpliceAI features are CRITICAL
       • F1 drops from {mode_stats['full']['f1_macro_mean']:.3f} to {mode_stats['no_probs']['f1_macro_mean']:.3f} without them
       • {drop/mode_stats['full']['f1_macro_mean']*100:.1f}% relative performance loss
    
    ✅ K-mer features are REDUNDANT
       • No performance loss when removed
       • SpliceAI captures sequence patterns
    
    ✅ Meta-model enhances base model
       • Full model > SpliceAI only
       • Combines complementary information
    
    ✅ Sequence alone is INSUFFICIENT
       • K-mers only: {mode_stats['only_kmer']['f1_macro_mean']:.3f} F1
       • Need learned representations
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    ax4.set_title('D) Biological Insights', fontweight='bold', loc='left')
    
    plt.suptitle('Meta-Model Ablation Study: Complete Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    output_file = Path(output_dir) / "ablation_summary_comprehensive.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.pdf'), dpi=FIGURE_DPI, bbox_inches='tight')
    
    print(f"✅ Saved comprehensive summary plot: {output_file}")
    plt.close()

def create_results_table(mode_stats: Dict[str, Any], output_dir: str):
    """Create a publication-ready results table."""
    
    print("📊 Creating results table...")
    
    # Prepare data for table
    data = []
    for mode in ['full', 'no_kmer', 'only_probs', 'no_probs', 'only_kmer']:
        stats = mode_stats[mode]
        
        data.append({
            'Mode': mode.replace('_', ' ').title(),
            'Description': {
                'full': 'All features (SpliceAI + K-mers + Genomic)',
                'no_kmer': 'SpliceAI + Genomic features (no sequence)',
                'only_probs': 'SpliceAI features only',
                'no_probs': 'K-mers + Genomic (no SpliceAI)',
                'only_kmer': 'K-mer sequence features only'
            }[mode],
            'F1 Macro': f"{stats['f1_macro_mean']:.3f} ± {stats['f1_macro_std']:.3f}",
            'AP Macro': f"{stats['ap_macro_mean']:.3f} ± {stats['ap_macro_std']:.3f}",
            'ROC AUC': f"{stats['roc_auc_mean']:.3f} ± {stats['roc_auc_std']:.3f}" if not np.isnan(stats['roc_auc_mean']) else "N/A",
            'Features': f"{stats['n_features']:,}",
            'Trials': stats['n_trials']
        })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_file = Path(output_dir) / "ablation_results_table.csv"
    df.to_csv(csv_file, index=False)
    
    # Save as formatted text
    txt_file = Path(output_dir) / "ablation_results_table.txt"
    with open(txt_file, 'w') as f:
        f.write("Meta-Model Ablation Study Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\nNote: Values shown as mean ± standard deviation across trials\n")
    
    print(f"✅ Saved results table: {csv_file}")
    print(f"✅ Saved formatted table: {txt_file}")

def main():
    """Main function to create all ablation visualizations."""
    
    parser = argparse.ArgumentParser(description="Create publication-ready ablation study visualizations")
    parser.add_argument("--results-file", type=str, required=True,
                       help="Path to ablation_summary_corrected.json")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    print("🎨 Creating Publication-Ready Ablation Study Visualizations")
    print("=" * 80)
    print(f"Results file: {args.results_file}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load and process results
    data = load_and_process_results(args.results_file)
    mode_stats = data['mode_stats']
    
    # Create all visualizations
    create_main_comparison_plot(mode_stats, args.output_dir)
    create_feature_contribution_plot(mode_stats, args.output_dir)
    create_per_class_performance_plot(mode_stats, args.output_dir)
    create_ablation_summary_plot(mode_stats, args.output_dir)
    create_results_table(mode_stats, args.output_dir)
    
    print(f"\n✅ All visualizations created in: {args.output_dir}")
    print(f"\nPublication-ready files:")
    print(f"  📊 ablation_main_comparison.png/pdf - Main results")
    print(f"  📈 ablation_feature_contribution.png/pdf - Feature importance")
    print(f"  🎯 ablation_per_class_performance.png/pdf - Per-class breakdown")
    print(f"  📋 ablation_summary_comprehensive.png/pdf - Complete analysis")
    print(f"  📄 ablation_results_table.csv/txt - Numerical results")
    
    print(f"\n💡 Recommended for publication:")
    print(f"  1. Use 'ablation_summary_comprehensive.png' for main figure")
    print(f"  2. Include 'ablation_results_table.csv' as supplementary data")
    print(f"  3. Individual plots available for detailed analysis")

if __name__ == "__main__":
    main() 