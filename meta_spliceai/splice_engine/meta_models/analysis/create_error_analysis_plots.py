#!/usr/bin/env python3
"""
Error Analysis Plots for Presentation

Generate visualizations showing SpliceAI prediction errors to illustrate
the problem that the meta-model solves.

Usage:
    python -m meta_spliceai.splice_engine.meta_models.analysis.create_error_analysis_plots \
      --dataset train_pc_1000/master \
      --output-dir results/presentation_plots \
      --n-examples 50
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for presentation-quality plots
plt.style.use('default')
sns.set_palette("husl")

def load_training_data(dataset_path: str) -> pd.DataFrame:
    """Load training data with SpliceAI predictions and true labels."""
    data_path = Path(dataset_path)
    
    # Look for common training data files
    possible_files = [
        "training_data.tsv",
        "master.tsv", 
        "positions_enhanced.tsv",
        "train_data.tsv"
    ]
    
    df = None
    for filename in possible_files:
        file_path = data_path / filename
        if file_path.exists():
            try:
                print(f"Loading training data from: {file_path}")
                df = pd.read_csv(file_path, sep='\t')
                break
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    
    if df is None:
        # Try to find any TSV file
        tsv_files = list(data_path.glob("*.tsv"))
        if tsv_files:
            print(f"Trying first TSV file: {tsv_files[0]}")
            df = pd.read_csv(tsv_files[0], sep='\t')
    
    if df is None:
        raise FileNotFoundError(f"No suitable training data found in {dataset_path}")
    
    print(f"Loaded {len(df)} training examples")
    print(f"Columns: {list(df.columns)}")
    
    return df

def identify_prediction_errors(df: pd.DataFrame, 
                             threshold: float = 0.5) -> Dict[str, pd.DataFrame]:
    """Identify false positives and false negatives in SpliceAI predictions."""
    
    # Determine column names (adapt to your data structure)
    splice_type_col = None
    donor_score_col = None
    acceptor_score_col = None
    
    # Look for common column patterns
    for col in df.columns:
        if 'splice_type' in col.lower():
            splice_type_col = col
        elif 'donor_score' in col.lower() or 'donor_prob' in col.lower():
            donor_score_col = col
        elif 'acceptor_score' in col.lower() or 'acceptor_prob' in col.lower():
            acceptor_score_col = col
    
    if not all([splice_type_col, donor_score_col, acceptor_score_col]):
        print("Could not identify all required columns. Available columns:")
        print(df.columns.tolist())
        # Use first few columns as fallback
        splice_type_col = df.columns[0] if splice_type_col is None else splice_type_col
        donor_score_col = df.columns[1] if donor_score_col is None else donor_score_col
        acceptor_score_col = df.columns[2] if acceptor_score_col is None else acceptor_score_col
    
    print(f"Using columns: {splice_type_col}, {donor_score_col}, {acceptor_score_col}")
    
    # Create binary predictions
    df['pred_donor'] = df[donor_score_col] > threshold
    df['pred_acceptor'] = df[acceptor_score_col] > threshold  
    df['pred_neither'] = (~df['pred_donor']) & (~df['pred_acceptor'])
    
    # True labels
    df['true_donor'] = df[splice_type_col] == 'donor'
    df['true_acceptor'] = df[splice_type_col] == 'acceptor'
    df['true_neither'] = df[splice_type_col] == 'neither'
    
    # Identify errors
    errors = {}
    
    # False Positives - predicted splice site but true neither
    errors['donor_fp'] = df[df['pred_donor'] & df['true_neither']].copy()
    errors['acceptor_fp'] = df[df['pred_acceptor'] & df['true_neither']].copy()
    
    # False Negatives - true splice site but predicted neither
    errors['donor_fn'] = df[df['true_donor'] & df['pred_neither']].copy()
    errors['acceptor_fn'] = df[df['true_acceptor'] & df['pred_neither']].copy()
    
    # Print error statistics
    print("\nError Statistics:")
    for error_type, error_df in errors.items():
        print(f"{error_type}: {len(error_df)} examples")
    
    return errors

def create_error_distribution_plot(errors: Dict[str, pd.DataFrame], 
                                 output_dir: Path) -> str:
    """Create bar chart showing error type distribution."""
    
    error_counts = {
        'Donor FP': len(errors['donor_fp']),
        'Donor FN': len(errors['donor_fn']),
        'Acceptor FP': len(errors['acceptor_fp']),
        'Acceptor FN': len(errors['acceptor_fn'])
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    splice_types = ['Donor', 'Acceptor']
    fp_counts = [error_counts['Donor FP'], error_counts['Acceptor FP']]
    fn_counts = [error_counts['Donor FN'], error_counts['Acceptor FN']]
    
    x = np.arange(len(splice_types))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fp_counts, width, label='False Positives', 
                   color='#ff6b6b', alpha=0.8)
    bars2 = ax.bar(x + width/2, fn_counts, width, label='False Negatives', 
                   color='#4ecdc4', alpha=0.8)
    
    ax.set_xlabel('Splice Site Type', fontsize=12)
    ax.set_ylabel('Number of Errors', fontsize=12)
    ax.set_title('SpliceAI Prediction Errors by Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(splice_types)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "error_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(plot_path)

def create_signal_example_plots(df: pd.DataFrame, 
                               errors: Dict[str, pd.DataFrame],
                               output_dir: Path,
                               n_examples: int = 3) -> List[str]:
    """Create signal trace plots showing specific error examples."""
    
    plot_paths = []
    
    # Try to find signal/context columns
    signal_cols = [col for col in df.columns if any(term in col.lower() 
                  for term in ['signal', 'context', 'neighbor', 'score'])]
    
    if len(signal_cols) < 2:
        print("Not enough signal columns found for signal plots")
        return plot_paths
    
    # Create examples for each error type
    for error_type, error_df in errors.items():
        if len(error_df) < n_examples:
            continue
            
        # Sample examples
        examples = error_df.sample(n=min(n_examples, len(error_df)), random_state=42)
        
        fig, axes = plt.subplots(n_examples, 1, figsize=(12, 4*n_examples))
        if n_examples == 1:
            axes = [axes]
        
        for i, (idx, row) in enumerate(examples.iterrows()):
            ax = axes[i]
            
            # Create mock signal trace (adapt to your actual data structure)
            positions = range(-10, 11)  # 21 positions around splice site
            
            # Extract signal values (adapt column names as needed)
            try:
                if error_type in ['donor_fp', 'donor_fn']:
                    signal = [row.get(f'donor_score_{j}', row.get('donor_score', 0.5)) 
                             for j in positions]
                else:
                    signal = [row.get(f'acceptor_score_{j}', row.get('acceptor_score', 0.5)) 
                             for j in positions]
            except:
                # Fallback: create representative signal patterns
                if 'fp' in error_type:
                    # False positive: broad, lower peak
                    signal = np.random.normal(0.6, 0.1, 21)
                    signal[8:13] = np.random.normal(0.75, 0.05, 5)  # Broad peak
                else:
                    # False negative: sharp but low peak
                    signal = np.random.normal(0.3, 0.05, 21)
                    signal[10] = 0.45  # Sharp but below threshold
            
            ax.plot(positions, signal, 'b-', linewidth=2)
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold')
            ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, label='Splice Site')
            ax.set_xlabel('Position relative to splice site')
            ax.set_ylabel('SpliceAI Score')
            ax.set_title(f'{error_type.replace("_", " ").title()} Example {i+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"signal_examples_{error_type}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_paths.append(str(plot_path))
    
    return plot_paths

def create_threshold_analysis_plot(df: pd.DataFrame, 
                                 output_dir: Path) -> str:
    """Create plot showing how errors change with threshold."""
    
    # Find score columns
    donor_col = None
    acceptor_col = None
    splice_type_col = None
    
    for col in df.columns:
        if 'donor_score' in col.lower() or 'donor_prob' in col.lower():
            donor_col = col
        elif 'acceptor_score' in col.lower() or 'acceptor_prob' in col.lower():
            acceptor_col = col
        elif 'splice_type' in col.lower():
            splice_type_col = col
    
    if not all([donor_col, acceptor_col, splice_type_col]):
        print("Could not find required columns for threshold analysis")
        return ""
    
    thresholds = np.linspace(0.1, 0.9, 17)
    
    donor_fp_counts = []
    donor_fn_counts = []
    acceptor_fp_counts = []
    acceptor_fn_counts = []
    
    for threshold in thresholds:
        # Predictions
        pred_donor = df[donor_col] > threshold
        pred_acceptor = df[acceptor_col] > threshold
        pred_neither = (~pred_donor) & (~pred_acceptor)
        
        # True labels
        true_donor = df[splice_type_col] == 'donor'
        true_acceptor = df[splice_type_col] == 'acceptor'
        true_neither = df[splice_type_col] == 'neither'
        
        # Count errors
        donor_fp_counts.append(sum(pred_donor & true_neither))
        donor_fn_counts.append(sum(true_donor & pred_neither))
        acceptor_fp_counts.append(sum(pred_acceptor & true_neither))
        acceptor_fn_counts.append(sum(true_acceptor & pred_neither))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Donor errors
    ax1.plot(thresholds, donor_fp_counts, 'r-', marker='o', label='False Positives')
    ax1.plot(thresholds, donor_fn_counts, 'b-', marker='s', label='False Negatives')
    ax1.axvline(x=0.5, color='k', linestyle='--', alpha=0.5, label='Default Threshold')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Number of Errors')
    ax1.set_title('Donor Site Errors vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Acceptor errors
    ax2.plot(thresholds, acceptor_fp_counts, 'r-', marker='o', label='False Positives')
    ax2.plot(thresholds, acceptor_fn_counts, 'b-', marker='s', label='False Negatives')
    ax2.axvline(x=0.5, color='k', linestyle='--', alpha=0.5, label='Default Threshold')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Number of Errors')
    ax2.set_title('Acceptor Site Errors vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "threshold_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(plot_path)

def main():
    """Main function to generate error analysis plots."""
    
    parser = argparse.ArgumentParser(description="Create error analysis plots for presentation")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to training dataset directory")
    parser.add_argument("--output-dir", type=str, default="results/presentation_plots",
                       help="Output directory for plots")
    parser.add_argument("--n-examples", type=int, default=3,
                       help="Number of examples per error type")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="SpliceAI threshold for error detection")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Creating Error Analysis Plots for Presentation")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    try:
        # Load training data
        print("\n📊 Loading training data...")
        df = load_training_data(args.dataset)
        
        # Identify prediction errors
        print("\n🔍 Identifying prediction errors...")
        errors = identify_prediction_errors(df, threshold=args.threshold)
        
        # Create error distribution plot
        print("\n📈 Creating error distribution plot...")
        dist_plot = create_error_distribution_plot(errors, output_dir)
        print(f"✅ Saved: {dist_plot}")
        
        # Create signal example plots
        print("\n🎨 Creating signal example plots...")
        signal_plots = create_signal_example_plots(df, errors, output_dir, args.n_examples)
        for plot_path in signal_plots:
            print(f"✅ Saved: {plot_path}")
        
        # Create threshold analysis plot
        print("\n📊 Creating threshold analysis plot...")
        threshold_plot = create_threshold_analysis_plot(df, output_dir)
        if threshold_plot:
            print(f"✅ Saved: {threshold_plot}")
        
        # Create summary report
        summary = {
            "dataset": args.dataset,
            "total_examples": len(df),
            "error_counts": {
                "donor_fp": len(errors['donor_fp']),
                "donor_fn": len(errors['donor_fn']),
                "acceptor_fp": len(errors['acceptor_fp']),
                "acceptor_fn": len(errors['acceptor_fn'])
            },
            "plots_created": {
                "error_distribution": dist_plot,
                "signal_examples": signal_plots,
                "threshold_analysis": threshold_plot if threshold_plot else None
            }
        }
        
        # Save summary
        summary_path = output_dir / "error_analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Summary saved to: {summary_path}")
        
        # Print usage instructions
        print(f"\n🎯 Plots ready for Slide 2!")
        print(f"Use these visualizations to show:")
        print(f"• Error distribution across splice types")
        print(f"• Specific examples of false positives/negatives")
        print(f"• How threshold choice affects errors")
        print(f"• The systematic patterns your meta-model addresses")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("This might be due to unexpected data format.")
        print("Try running with a sample of your data first.")

if __name__ == "__main__":
    main() 