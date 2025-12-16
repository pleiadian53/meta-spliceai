#!/usr/bin/env python3
"""
Simple Error Analysis Plots

Creates example plots for Slide 2 showing SpliceAI error patterns.
This version works with synthetic data if real data format is complex.

Usage:
    python -m meta_spliceai.splice_engine.meta_models.analysis.create_simple_error_plots \
      --output-dir results/presentation_plots
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set presentation style
plt.style.use('default')
sns.set_palette("husl")

def create_error_distribution_plot(output_dir: Path) -> str:
    """Create a representative error distribution plot."""
    
    # Example error counts (replace with your actual data)
    error_data = {
        'Donor': {'False Positives': 1247, 'False Negatives': 892},
        'Acceptor': {'False Positives': 1583, 'False Negatives': 1024}
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    splice_types = list(error_data.keys())
    fp_counts = [error_data[st]['False Positives'] for st in splice_types]
    fn_counts = [error_data[st]['False Negatives'] for st in splice_types]
    
    x = np.arange(len(splice_types))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, fp_counts, width, label='False Positives', 
                   color='#ff6b6b', alpha=0.8)
    bars2 = ax.bar(x + width/2, fn_counts, width, label='False Negatives', 
                   color='#4ecdc4', alpha=0.8)
    
    # Styling
    ax.set_xlabel('Splice Site Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Errors', fontsize=14, fontweight='bold')
    ax.set_title('SpliceAI Prediction Errors by Type', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(splice_types, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save with high quality
    plot_path = output_dir / "error_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(plot_path)

def create_false_positive_example(output_dir: Path) -> str:
    """Create example of false positive: broad signal vs sharp peak."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    positions = range(-10, 11)
    
    # TRUE POSITIVE: Sharp, high peak
    true_positive = np.random.normal(0.15, 0.05, 21)
    true_positive[9:12] = [0.7, 0.95, 0.75]  # Sharp peak
    
    ax1.plot(positions, true_positive, 'b-', linewidth=3, label='SpliceAI Score')
    ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.5, label='Splice Site')
    ax1.fill_between(positions, 0, true_positive, alpha=0.3, color='blue')
    ax1.set_xlabel('Position relative to splice site', fontsize=12)
    ax1.set_ylabel('SpliceAI Probability', fontsize=12)
    ax1.set_title('✅ True Positive: Sharp Peak', fontsize=14, fontweight='bold', color='green')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(-10, 10)
    
    # FALSE POSITIVE: Broad, moderate signal
    false_positive = np.random.normal(0.25, 0.08, 21)
    false_positive[6:15] = np.random.normal(0.65, 0.05, 9)  # Broad, moderate peak
    
    ax2.plot(positions, false_positive, 'r-', linewidth=3, label='SpliceAI Score')
    ax2.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.5, label='Splice Site')
    ax2.fill_between(positions, 0, false_positive, alpha=0.3, color='red')
    ax2.set_xlabel('Position relative to splice site', fontsize=12)
    ax2.set_ylabel('SpliceAI Probability', fontsize=12)
    ax2.set_title('❌ False Positive: Broad Signal', fontsize=14, fontweight='bold', color='red')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(-10, 10)
    
    plt.tight_layout()
    
    plot_path = output_dir / "false_positive_example.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(plot_path)

def create_false_negative_example(output_dir: Path) -> str:
    """Create example of false negative: below-threshold true site."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    positions = range(-10, 11)
    
    # TRUE NEGATIVE: Low, flat signal
    true_negative = np.random.normal(0.15, 0.03, 21)
    true_negative[10] = 0.25  # Slight bump but clearly below threshold
    
    ax1.plot(positions, true_negative, 'g-', linewidth=3, label='SpliceAI Score')
    ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.5, label='Splice Site')
    ax1.fill_between(positions, 0, true_negative, alpha=0.3, color='green')
    ax1.set_xlabel('Position relative to splice site', fontsize=12)
    ax1.set_ylabel('SpliceAI Probability', fontsize=12)
    ax1.set_title('✅ True Negative: Low Signal', fontsize=14, fontweight='bold', color='green')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(-10, 10)
    
    # FALSE NEGATIVE: Sharp but below threshold
    false_negative = np.random.normal(0.12, 0.03, 21)
    false_negative[10] = 0.42  # Sharp peak but just below threshold
    false_negative[9] = 0.35
    false_negative[11] = 0.35
    
    ax2.plot(positions, false_negative, 'orange', linewidth=3, label='SpliceAI Score')
    ax2.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.5, label='Splice Site')
    ax2.fill_between(positions, 0, false_negative, alpha=0.3, color='orange')
    ax2.set_xlabel('Position relative to splice site', fontsize=12)
    ax2.set_ylabel('SpliceAI Probability', fontsize=12)
    ax2.set_title('❌ False Negative: Below Threshold', fontsize=14, fontweight='bold', color='darkorange')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(-10, 10)
    
    plt.tight_layout()
    
    plot_path = output_dir / "false_negative_example.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(plot_path)

def create_meta_model_solution_plot(output_dir: Path) -> str:
    """Show how meta-model features help distinguish errors."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    positions = range(-10, 11)
    
    # Example 1: False Positive - Broad signal
    fp_signal = np.random.normal(0.25, 0.08, 21)
    fp_signal[6:15] = np.random.normal(0.65, 0.05, 9)
    
    ax1.plot(positions, fp_signal, 'r-', linewidth=3, label='SpliceAI Score')
    ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Threshold')
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.5)
    ax1.fill_between(positions, 0, fp_signal, alpha=0.3, color='red')
    ax1.set_title('❌ False Positive\n(Broad Signal)', fontweight='bold', color='red')
    ax1.set_ylabel('SpliceAI Probability')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Meta-model features for FP
    feature_names = ['Peak\nSharpness', 'Signal\nStrength', 'Context\nScore', 'Type\nDiscrimination']
    fp_features = [0.3, 0.6, 0.4, 0.2]  # Low sharpness = likely FP
    colors = ['red' if f < 0.5 else 'green' for f in fp_features]
    
    ax2.bar(feature_names, fp_features, color=colors, alpha=0.7)
    ax2.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    ax2.set_title('Meta-Model Features\n(Predict: NOT splice site)', fontweight='bold')
    ax2.set_ylabel('Feature Value')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Example 2: False Negative - Below threshold but sharp
    fn_signal = np.random.normal(0.12, 0.03, 21)
    fn_signal[10] = 0.42  # Sharp but below threshold
    fn_signal[9] = 0.35
    fn_signal[11] = 0.35
    
    ax3.plot(positions, fn_signal, 'orange', linewidth=3, label='SpliceAI Score')
    ax3.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Threshold')
    ax3.axvline(x=0, color='k', linestyle='-', alpha=0.5)
    ax3.fill_between(positions, 0, fn_signal, alpha=0.3, color='orange')
    ax3.set_title('❌ False Negative\n(Sharp but Low)', fontweight='bold', color='darkorange')
    ax3.set_xlabel('Position relative to splice site')
    ax3.set_ylabel('SpliceAI Probability')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Meta-model features for FN
    fn_features = [0.8, 0.4, 0.7, 0.6]  # High sharpness = likely true site
    colors = ['red' if f < 0.5 else 'green' for f in fn_features]
    
    ax4.bar(feature_names, fn_features, color=colors, alpha=0.7)
    ax4.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    ax4.set_title('Meta-Model Features\n(Predict: IS splice site)', fontweight='bold')
    ax4.set_xlabel('Feature Type')
    ax4.set_ylabel('Feature Value')
    ax4.set_ylim(0, 1)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    plot_path = output_dir / "meta_model_solution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return str(plot_path)

def create_slide_instructions(output_dir: Path) -> str:
    """Create instructions for using the plots in the presentation."""
    
    instructions = """
# 📊 Error Analysis Plots - Usage Guide for Slide 2

## Created Plots:

### 1. error_distribution.png
**Purpose**: Show the scale of the problem
**Use for**: Opening the slide to establish that SpliceAI makes systematic errors
**Key Message**: "SpliceAI makes thousands of errors across both donor and acceptor sites"

### 2. false_positive_example.png  
**Purpose**: Illustrate the difference between true and false positives
**Use for**: Explaining why broad signals are problematic
**Key Message**: "False positives have broad, moderate signals rather than sharp peaks"

### 3. false_negative_example.png
**Purpose**: Show how true splice sites can be missed
**Use for**: Demonstrating the threshold problem
**Key Message**: "False negatives are often sharp but below the threshold"

### 4. meta_model_solution.png
**Purpose**: Preview how meta-model features help
**Use for**: Transition to your solution
**Key Message**: "Meta-model features capture signal quality, not just magnitude"

## 🎯 Slide 2 Structure:

1. **Title**: "SpliceAI Prediction Errors: Patterns & Opportunities"

2. **Opening**: Show error_distribution.png
   - "SpliceAI makes systematic errors across splice types"
   - "These errors follow predictable patterns"

3. **Problem 1**: Show false_positive_example.png  
   - "False positives: broad signals that lack sharp peaks"
   - "Traditional thresholding can't distinguish signal quality"

4. **Problem 2**: Show false_negative_example.png
   - "False negatives: sharp signals below threshold"
   - "Some true sites have weaker but still valid signals"

5. **Solution Preview**: Show meta_model_solution.png
   - "Meta-model analyzes signal characteristics beyond magnitude"
   - "Features capture peak sharpness, context, and discrimination"

6. **Conclusion**: 
   - "Systematic error patterns → opportunity for meta-modeling"
   - "Next: How we engineered features to capture these patterns"

## 💡 Speaking Points:

- **Error Scale**: "In our training data, SpliceAI produces X false positives and Y false negatives"
- **Pattern Recognition**: "These aren't random errors - they follow systematic patterns"
- **Feature Engineering**: "Our meta-model captures what the eye can see: signal quality"
- **Transition**: "Let's see how we built features to detect these patterns automatically"

## 🎨 Visual Tips:

- Use the plots full-screen for impact
- Point out specific features in the signal traces
- Highlight the threshold line to show the decision boundary
- Use the feature bars to preview your solution approach

"""
    
    instructions_path = output_dir / "slide_instructions.md"
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    return str(instructions_path)

def main():
    """Main function to create simple error analysis plots."""
    
    parser = argparse.ArgumentParser(description="Create simple error analysis plots")
    parser.add_argument("--output-dir", type=str, default="results/presentation_plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎯 Creating Simple Error Analysis Plots for Slide 2")
    print("=" * 60)
    
    # Create all plots
    plots_created = []
    
    print("📊 Creating error distribution plot...")
    dist_plot = create_error_distribution_plot(output_dir)
    plots_created.append(dist_plot)
    print(f"✅ Saved: {dist_plot}")
    
    print("📈 Creating false positive example...")
    fp_plot = create_false_positive_example(output_dir)
    plots_created.append(fp_plot)
    print(f"✅ Saved: {fp_plot}")
    
    print("📉 Creating false negative example...")
    fn_plot = create_false_negative_example(output_dir)
    plots_created.append(fn_plot)
    print(f"✅ Saved: {fn_plot}")
    
    print("🎨 Creating meta-model solution preview...")
    solution_plot = create_meta_model_solution_plot(output_dir)
    plots_created.append(solution_plot)
    print(f"✅ Saved: {solution_plot}")
    
    print("📝 Creating slide instructions...")
    instructions = create_slide_instructions(output_dir)
    print(f"✅ Saved: {instructions}")
    
    print(f"\n🎉 All plots created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Plots created: {len(plots_created)}")
    print(f"📖 See slide_instructions.md for usage guide")

if __name__ == "__main__":
    main() 