#!/usr/bin/env python3
"""
Ablation study analysis utility.

This script analyzes the results of ablation studies to understand
feature importance and model behavior.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_ablation_results(ablation_dir: str, output_file: Optional[str] = None) -> Dict:
    """
    Analyze ablation study results and generate comprehensive analysis.
    
    Args:
        ablation_dir: Directory containing ablation study results
        output_file: Optional output file for detailed analysis
        
    Returns:
        Dictionary containing analysis results
    """
    ablation_dir = Path(ablation_dir)
    
    print("🔍 ABLATION STUDY ANALYSIS")
    print("=" * 50)
    print(f"Analyzing results from: {ablation_dir}")
    
    # Load ablation results
    ablation_file = ablation_dir / "ablation_summary.csv"
    if not ablation_file.exists():
        print(f"❌ Ablation results file not found: {ablation_file}")
        return {'error': 'Results file not found'}
    
    ablation_df = pd.read_csv(ablation_file)
    print(f"✅ Loaded results for {len(ablation_df)} ablation modes")
    
    # Basic statistics
    print(f"\n📊 ABLATION MODES TESTED:")
    print("-" * 30)
    for _, row in ablation_df.iterrows():
        mode = row['mode']
        n_features = row.get('n_features', 'N/A')
        accuracy = row.get('accuracy', 0)
        f1 = row.get('macro_f1', 0)
        print(f"  {mode:15s}: {n_features:4s} features, Acc={accuracy:.3f}, F1={f1:.3f}")
    
    # Find baseline (full features)
    baseline_row = ablation_df[ablation_df['mode'] == 'full']
    if baseline_row.empty:
        print("⚠️ No 'full' baseline found, using first row as baseline")
        baseline = ablation_df.iloc[0]
    else:
        baseline = baseline_row.iloc[0]
    
    print(f"\n🎯 BASELINE PERFORMANCE (Full Features):")
    print("-" * 40)
    print(f"  Mode: {baseline['mode']}")
    print(f"  Features: {baseline.get('n_features', 'N/A')}")
    print(f"  Accuracy: {baseline.get('accuracy', 0):.3f}")
    print(f"  Macro F1: {baseline.get('macro_f1', 0):.3f}")
    if 'weighted_f1' in baseline:
        print(f"  Weighted F1: {baseline.get('weighted_f1', 0):.3f}")
    
    # Feature contribution analysis
    print(f"\n📈 FEATURE CONTRIBUTION ANALYSIS:")
    print("-" * 40)
    
    contributions = []
    for _, row in ablation_df.iterrows():
        if row['mode'] != baseline['mode']:
            mode = row['mode']
            accuracy_drop = baseline.get('accuracy', 0) - row.get('accuracy', 0)
            f1_drop = baseline.get('macro_f1', 0) - row.get('macro_f1', 0)
            
            print(f"  {mode:15s}: Acc drop = {accuracy_drop:+.3f}, F1 drop = {f1_drop:+.3f}")
            
            contributions.append({
                'mode': mode,
                'accuracy_drop': accuracy_drop,
                'f1_drop': f1_drop,
                'n_features': row.get('n_features', 0)
            })
    
    # Load detailed analysis if available
    summary_file = ablation_dir / "ablation_report.json"
    statistical_tests = {}
    if summary_file.exists():
        try:
            with open(summary_file) as f:
                summary = json.load(f)
            
            print(f"\n📋 SUMMARY STATISTICS:")
            print("-" * 30)
            results_summary = summary.get('results_summary', {})
            if results_summary:
                best_acc_mode = results_summary.get('best_mode_accuracy', 'N/A')
                best_f1_mode = results_summary.get('best_mode_f1', 'N/A')
                mean_acc = results_summary.get('mean_accuracy', 0)
                mean_f1 = results_summary.get('mean_f1', 0)
                
                print(f"  Best accuracy mode: {best_acc_mode}")
                print(f"  Best F1 mode: {best_f1_mode}")
                print(f"  Mean accuracy: {mean_acc:.3f}")
                print(f"  Mean F1: {mean_f1:.3f}")
            
        except Exception as e:
            print(f"⚠️ Could not load detailed analysis: {e}")
    
    # Generate insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 20)
    
    # Find most impactful feature groups
    if contributions:
        # Sort by F1 drop (most negative = most important)
        contributions_sorted = sorted(contributions, key=lambda x: x['f1_drop'])
        
        most_important = contributions_sorted[0]  # Most negative drop
        least_important = contributions_sorted[-1]  # Least negative drop
        
        print(f"  🔥 Most critical features: {most_important['mode']} (F1 drop: {most_important['f1_drop']:+.3f})")
        print(f"  💡 Least critical features: {least_important['mode']} (F1 drop: {least_important['f1_drop']:+.3f})")
        
        # Feature efficiency analysis
        feature_efficiency = []
        for contrib in contributions:
            if contrib['n_features'] > 0:
                efficiency = abs(contrib['f1_drop']) / contrib['n_features']
                feature_efficiency.append((contrib['mode'], efficiency))
        
        if feature_efficiency:
            feature_efficiency.sort(key=lambda x: x[1], reverse=True)
            most_efficient = feature_efficiency[0]
            print(f"  ⚡ Most efficient features: {most_efficient[0]} ({most_efficient[1]:.4f} F1/feature)")
    
    # Prepare results
    results = {
        'ablation_dir': str(ablation_dir),
        'n_modes_tested': len(ablation_df),
        'baseline_performance': {
            'mode': baseline['mode'],
            'accuracy': baseline.get('accuracy', 0),
            'macro_f1': baseline.get('macro_f1', 0),
            'n_features': baseline.get('n_features', 0)
        },
        'feature_contributions': contributions,
        'summary_statistics': results_summary if 'results_summary' in locals() else {}
    }
    
    # Save detailed analysis if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed analysis saved to: {output_path}")
    
    return results


def create_ablation_plots(ablation_dir: str, output_dir: Optional[str] = None):
    """Create visualization plots for ablation study results."""
    ablation_dir = Path(ablation_dir)
    
    # Load data
    ablation_file = ablation_dir / "ablation_summary.csv"
    if not ablation_file.exists():
        print(f"❌ Cannot create plots: {ablation_file} not found")
        return
    
    ablation_df = pd.read_csv(ablation_file)
    
    # Set up output directory
    if output_dir:
        plot_dir = Path(output_dir)
    else:
        plot_dir = ablation_dir / "analysis_plots"
    
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy comparison
    modes = ablation_df['mode'].tolist()
    accuracies = ablation_df['accuracy'].tolist()
    
    axes[0].bar(range(len(modes)), accuracies, alpha=0.7, color='skyblue')
    axes[0].set_xlabel('Ablation Mode')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by Ablation Mode')
    axes[0].set_xticks(range(len(modes)))
    axes[0].set_xticklabels(modes, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # F1 comparison
    f1_scores = ablation_df['macro_f1'].tolist()
    axes[1].bar(range(len(modes)), f1_scores, alpha=0.7, color='lightcoral')
    axes[1].set_xlabel('Ablation Mode')
    axes[1].set_ylabel('Macro F1')
    axes[1].set_title('Macro F1 by Ablation Mode')
    axes[1].set_xticks(range(len(modes)))
    axes[1].set_xticklabels(modes, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    # Feature count comparison
    n_features = ablation_df['n_features'].tolist()
    axes[2].bar(range(len(modes)), n_features, alpha=0.7, color='lightgreen')
    axes[2].set_xlabel('Ablation Mode')
    axes[2].set_ylabel('Number of Features')
    axes[2].set_title('Feature Count by Mode')
    axes[2].set_xticks(range(len(modes)))
    axes[2].set_xticklabels(modes, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = plot_dir / "ablation_analysis.pdf"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Analysis plots saved to: {plot_file}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Ablation study analysis utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze ablation study results
  python ablation_analyzer.py \\
    --ablation-dir results/ablation_study_pc_5000_3mers_diverse \\
    --output ablation_analysis.json

  # Create plots only
  python ablation_analyzer.py \\
    --ablation-dir results/ablation_study_pc_5000_3mers_diverse \\
    --plots-only
        """
    )
    
    parser.add_argument("--ablation-dir", required=True,
                       help="Directory containing ablation study results")
    parser.add_argument("--output", default=None,
                       help="Output file for detailed analysis (JSON format)")
    parser.add_argument("--plot-dir", default=None,
                       help="Directory for output plots (default: ablation-dir/analysis_plots)")
    parser.add_argument("--plots-only", action="store_true",
                       help="Only create plots, skip analysis")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.plots_only:
        # Only create plots
        create_ablation_plots(args.ablation_dir, args.plot_dir)
    else:
        # Run full analysis
        results = analyze_ablation_results(args.ablation_dir, args.output)
        
        if 'error' in results:
            print(f"❌ Analysis failed: {results['error']}")
            sys.exit(1)
        
        # Also create plots
        try:
            create_ablation_plots(args.ablation_dir, args.plot_dir)
        except Exception as e:
            print(f"⚠️ Plot generation failed: {e}")
    
    print(f"\n🎉 Ablation analysis completed!")


if __name__ == "__main__":
    main()




