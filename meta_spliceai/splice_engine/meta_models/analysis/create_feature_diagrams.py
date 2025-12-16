#!/usr/bin/env python3
"""
Feature Conceptual Diagram Generator

Creates clear diagrams explaining signal processing concepts used in 
probability-based features for splice site analysis.

Usage:
    python create_feature_diagrams.py --output-dir diagrams/
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Tuple, List
import seaborn as sns
import os
import sys

def find_project_root():
    """Find the project root directory (containing meta_spliceai/)."""
    current_dir = Path(__file__).resolve()
    
    # Walk up the directory tree to find meta_spliceai parent
    for parent in current_dir.parents:
        if (parent / "meta_spliceai").exists():
            return str(parent)
    
    # Fallback to current working directory
    return os.getcwd()

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_peak_detection_diagram(output_dir: Path) -> str:
    """Create diagram explaining peak detection concepts."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Peak Detection in Splice Site Prediction', fontsize=16, fontweight='bold')
    
    # Create position array
    x = np.linspace(-4, 4, 100)
    
    # True splice site - sharp peak
    true_peak = np.exp(-(x**2) / 0.3)
    
    # False positive - broad signal
    false_peak = 0.7 * np.exp(-(x**2) / 2.0) + 0.1
    
    # Plot true peak
    ax1.plot(x, true_peak, 'b-', linewidth=4, label='True Splice Site')
    ax1.fill_between(x, 0, true_peak, alpha=0.3, color='blue')
    
    # Mark context positions
    context_positions = [-2, -1, 1, 2]
    for pos in context_positions:
        ax1.axvline(pos, color='gray', linestyle=':', alpha=0.7)
        ax1.text(pos, -0.05, f'{pos:+d}', ha='center', va='top', fontsize=10)
    
    # Mark center position
    ax1.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(0, 1.1, 'Splice Site', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Calculate and show peak height ratio
    center_val = np.exp(0)  # Value at center
    neighbor_vals = [np.exp(-1), np.exp(-1)]  # Values at ±1
    neighbor_mean = np.mean(neighbor_vals)
    peak_ratio = center_val / neighbor_mean
    
    ax1.text(0.05, 0.95, f'Peak Height Ratio = {peak_ratio:.2f}', 
             transform=ax1.transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    ax1.set_title('TRUE SPLICE SITE\n(Sharp Peak)', fontweight='bold', color='blue')
    ax1.set_xlabel('Position relative to splice site')
    ax1.set_ylabel('Probability Score')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.2)
    
    # Plot false positive
    ax2.plot(x, false_peak, 'r-', linewidth=4, label='False Positive')
    ax2.fill_between(x, 0, false_peak, alpha=0.3, color='red')
    
    # Mark context positions
    for pos in context_positions:
        ax2.axvline(pos, color='gray', linestyle=':', alpha=0.7)
        ax2.text(pos, -0.05, f'{pos:+d}', ha='center', va='top', fontsize=10)
    
    # Mark center position
    ax2.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.text(0, 0.85, 'Predicted Site', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Calculate peak height ratio for false positive
    center_val_fp = 0.7 * np.exp(0) + 0.1
    neighbor_vals_fp = [0.7 * np.exp(-1) + 0.1, 0.7 * np.exp(-1) + 0.1]
    neighbor_mean_fp = np.mean(neighbor_vals_fp)
    peak_ratio_fp = center_val_fp / neighbor_mean_fp
    
    ax2.text(0.05, 0.95, f'Peak Height Ratio = {peak_ratio_fp:.2f}', 
             transform=ax2.transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    
    ax2.set_title('FALSE POSITIVE\n(Broad Signal)', fontweight='bold', color='red')
    ax2.set_xlabel('Position relative to splice site')
    ax2.set_ylabel('Probability Score')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.2)
    
    # Add interpretation box
    interpretation = (
        "Peak Height Ratio = center_score / neighbor_mean\n"
        "• > 2.0: Sharp peak (likely true site)\n"
        "• < 1.5: Broad signal (likely false positive)"
    )
    fig.text(0.5, 0.02, interpretation, ha='center', va='bottom', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    output_path = output_dir / "peak_detection_concept.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)

def create_second_derivative_diagram(output_dir: Path) -> str:
    """Create diagram explaining second derivative (curvature) concept."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Second Derivative (Curvature) Analysis', fontsize=16, fontweight='bold')
    
    # Create fine-grained position array for smooth curves
    x = np.linspace(-2, 2, 200)
    
    # Sharp peak (positive curvature)
    sharp_peak = np.exp(-(x**2) / 0.2)
    
    # Broad peak (negative curvature at center)
    broad_peak = 1 - 0.3 * x**2
    
    # Plot sharp peak
    ax1.plot(x, sharp_peak, 'g-', linewidth=4, label='Sharp Peak')
    ax1.fill_between(x, 0, sharp_peak, alpha=0.3, color='green')
    
    # Mark the three key positions for derivative calculation
    positions = [-1, 0, 1]
    pos_labels = ['m1', '0', 'p1']
    
    for pos, label in zip(positions, pos_labels):
        val = np.exp(-(pos**2) / 0.2)
        ax1.plot(pos, val, 'ro', markersize=8)
        ax1.text(pos, val + 0.1, f'{label}\n({val:.2f})', ha='center', va='bottom', fontsize=10)
    
    # Calculate second derivative
    val_m1 = np.exp(-1 / 0.2)
    val_0 = np.exp(0)
    val_p1 = np.exp(-1 / 0.2)
    
    second_deriv = (val_0 - val_m1) - (val_p1 - val_0)
    
    ax1.text(0.05, 0.95, f'2nd Derivative = {second_deriv:.3f}\n(Positive = Concave Up)', 
             transform=ax1.transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    ax1.set_title('SHARP PEAK\n(True Splice Site)', fontweight='bold', color='green')
    ax1.set_xlabel('Position relative to splice site')
    ax1.set_ylabel('Probability Score')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.2)
    
    # Plot broad peak
    ax2.plot(x, broad_peak, 'r-', linewidth=4, label='Broad Peak')
    ax2.fill_between(x, 0, broad_peak, alpha=0.3, color='red')
    
    # Mark the three key positions
    for pos, label in zip(positions, pos_labels):
        val = 1 - 0.3 * pos**2
        ax2.plot(pos, val, 'ro', markersize=8)
        ax2.text(pos, val + 0.1, f'{label}\n({val:.2f})', ha='center', va='bottom', fontsize=10)
    
    # Calculate second derivative for broad peak
    val_m1_broad = 1 - 0.3 * 1
    val_0_broad = 1 - 0.3 * 0
    val_p1_broad = 1 - 0.3 * 1
    
    second_deriv_broad = (val_0_broad - val_m1_broad) - (val_p1_broad - val_0_broad)
    
    ax2.text(0.05, 0.95, f'2nd Derivative = {second_deriv_broad:.3f}\n(Negative = Concave Down)', 
             transform=ax2.transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    
    ax2.set_title('BROAD PEAK\n(False Positive)', fontweight='bold', color='red')
    ax2.set_xlabel('Position relative to splice site')
    ax2.set_ylabel('Probability Score')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.2)
    
    # Add formula explanation
    formula = (
        "Second Derivative = (score₀ - score₋₁) - (score₊₁ - score₀)\n"
        "• Positive: Sharp peak (concave up) → True splice site\n"
        "• Negative: Broad signal (concave down) → False positive"
    )
    fig.text(0.5, 0.02, formula, ha='center', va='bottom', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    output_path = output_dir / "second_derivative_concept.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)

def create_type_signal_difference_diagram(output_dir: Path) -> str:
    """Create diagram explaining type signal difference concept."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Type Signal Difference for Splice Site Classification', fontsize=16, fontweight='bold')
    
    x = np.linspace(-3, 3, 100)
    
    # Scenario 1: Clear donor site
    donor_signal = np.exp(-(x**2) / 0.5)
    acceptor_signal = 0.3 * np.exp(-(x**2) / 1.5)
    
    # Background context
    context_bg = 0.1 * np.ones_like(x)
    
    ax1.plot(x, donor_signal, 'b-', linewidth=4, label='Donor Signal')
    ax1.plot(x, acceptor_signal, 'purple', linewidth=4, label='Acceptor Signal')
    ax1.plot(x, context_bg, 'gray', linewidth=2, linestyle=':', label='Background')
    
    # Fill areas to show signal strength
    ax1.fill_between(x, context_bg, donor_signal, alpha=0.3, color='blue', label='Donor Strength')
    ax1.fill_between(x, context_bg, acceptor_signal, alpha=0.3, color='purple', label='Acceptor Strength')
    
    # Calculate signal strengths at center
    donor_strength = 1.0 - 0.1
    acceptor_strength = 0.3 - 0.1
    type_diff = donor_strength - acceptor_strength
    
    ax1.text(0.05, 0.95, f'Type Signal Difference = {type_diff:.2f}\n(Positive = Donor Preferred)', 
             transform=ax1.transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    ax1.set_title('DONOR SPLICE SITE\n(Strong Donor Signal)', fontweight='bold', color='blue')
    ax1.set_xlabel('Position relative to splice site')
    ax1.set_ylabel('Probability Score')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.2)
    
    # Scenario 2: Clear acceptor site
    donor_signal_2 = 0.2 * np.exp(-(x**2) / 1.5)
    acceptor_signal_2 = 0.9 * np.exp(-(x**2) / 0.4)
    
    ax2.plot(x, donor_signal_2, 'b-', linewidth=4, label='Donor Signal')
    ax2.plot(x, acceptor_signal_2, 'purple', linewidth=4, label='Acceptor Signal')
    ax2.plot(x, context_bg, 'gray', linewidth=2, linestyle=':', label='Background')
    
    # Fill areas to show signal strength
    ax2.fill_between(x, context_bg, donor_signal_2, alpha=0.3, color='blue', label='Donor Strength')
    ax2.fill_between(x, context_bg, acceptor_signal_2, alpha=0.3, color='purple', label='Acceptor Strength')
    
    # Calculate signal strengths at center
    donor_strength_2 = 0.2 - 0.1
    acceptor_strength_2 = 0.9 - 0.1
    type_diff_2 = donor_strength_2 - acceptor_strength_2
    
    ax2.text(0.05, 0.95, f'Type Signal Difference = {type_diff_2:.2f}\n(Negative = Acceptor Preferred)', 
             transform=ax2.transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="plum", alpha=0.8))
    
    ax2.set_title('ACCEPTOR SPLICE SITE\n(Strong Acceptor Signal)', fontweight='bold', color='purple')
    ax2.set_xlabel('Position relative to splice site')
    ax2.set_ylabel('Probability Score')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.2)
    
    # Add interpretation
    interpretation = (
        "Type Signal Difference = donor_signal_strength - acceptor_signal_strength\n"
        "• > +0.1: Strong donor preference\n"
        "• < -0.1: Strong acceptor preference\n"
        "• ≈ 0: Ambiguous type (needs additional features)"
    )
    fig.text(0.5, 0.02, interpretation, ha='center', va='bottom', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    output_path = output_dir / "type_signal_difference_concept.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)

def create_context_pattern_diagram(output_dir: Path) -> str:
    """Create diagram showing context patterns around splice sites."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Context Pattern Analysis Around Splice Sites', fontsize=16, fontweight='bold')
    
    # Define positions
    positions = [-2, -1, 0, 1, 2]
    pos_labels = ['m2', 'm1', 'site', 'p1', 'p2']
    
    # 1. True positive pattern
    ax = axes[0, 0]
    tp_pattern = [0.1, 0.2, 0.9, 0.2, 0.1]  # Sharp peak at center
    
    bars = ax.bar(positions, tp_pattern, color='green', alpha=0.7, width=0.6)
    ax.plot(positions, tp_pattern, 'go-', linewidth=3, markersize=8)
    
    # Calculate features
    neighbor_mean = np.mean([0.1, 0.2, 0.2, 0.1])
    context_asymmetry = (0.2 + 0.1) - (0.2 + 0.1)
    
    ax.text(0.05, 0.95, f'Peak Height Ratio: {0.9/neighbor_mean:.2f}\nContext Asymmetry: {context_asymmetry:.2f}', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    ax.set_title('TRUE POSITIVE\n(Sharp Central Peak)', fontweight='bold', color='green')
    ax.set_xticks(positions)
    ax.set_xticklabels(pos_labels)
    ax.set_ylabel('Probability Score')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # 2. False positive pattern
    ax = axes[0, 1]
    fp_pattern = [0.3, 0.4, 0.5, 0.4, 0.3]  # Broad elevation
    
    bars = ax.bar(positions, fp_pattern, color='red', alpha=0.7, width=0.6)
    ax.plot(positions, fp_pattern, 'ro-', linewidth=3, markersize=8)
    
    neighbor_mean_fp = np.mean([0.3, 0.4, 0.4, 0.3])
    context_asymmetry_fp = (0.4 + 0.3) - (0.4 + 0.3)
    
    ax.text(0.05, 0.95, f'Peak Height Ratio: {0.5/neighbor_mean_fp:.2f}\nContext Asymmetry: {context_asymmetry_fp:.2f}', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    
    ax.set_title('FALSE POSITIVE\n(Broad Elevation)', fontweight='bold', color='red')
    ax.set_xticks(positions)
    ax.set_xticklabels(pos_labels)
    ax.set_ylabel('Probability Score')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # 3. Asymmetric pattern
    ax = axes[1, 0]
    asym_pattern = [0.05, 0.1, 0.8, 0.4, 0.3]  # Asymmetric decay
    
    bars = ax.bar(positions, asym_pattern, color='orange', alpha=0.7, width=0.6)
    ax.plot(positions, asym_pattern, 'o-', color='orange', linewidth=3, markersize=8)
    
    neighbor_mean_asym = np.mean([0.05, 0.1, 0.4, 0.3])
    context_asymmetry_asym = (0.1 + 0.05) - (0.4 + 0.3)
    
    ax.text(0.05, 0.95, f'Peak Height Ratio: {0.8/neighbor_mean_asym:.2f}\nContext Asymmetry: {context_asymmetry_asym:.2f}', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    
    ax.set_title('ASYMMETRIC PATTERN\n(Directional Bias)', fontweight='bold', color='orange')
    ax.set_xticks(positions)
    ax.set_xticklabels(pos_labels)
    ax.set_ylabel('Probability Score')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # 4. Context feature summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Create text summary
    summary_text = """
CONTEXT FEATURES SUMMARY

neighbor_mean = (m2 + m1 + p1 + p2) / 4
• Average background level

context_asymmetry = (m1 + m2) - (p1 + p2)  
• Upstream vs downstream bias
• Positive: upstream higher
• Negative: downstream higher

context_max = max(m2, m1, p1, p2)
• Highest neighboring score
• Indicates nearby competing sites

peak_height_ratio = center / neighbor_mean
• Peak prominence measure
• > 2.0: sharp peak (good)
• < 1.5: broad signal (poor)
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    output_path = output_dir / "context_pattern_concept.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)

def main():
    """Main function to create all conceptual diagrams."""
    
    parser = argparse.ArgumentParser(description="Create conceptual diagrams for probability features")
    parser.add_argument("--output-dir", type=str, default="results/probability_feature_analysis/diagrams",
                       help="Output directory for diagrams")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating conceptual diagrams...")
    print(f"Output directory: {output_dir}")
    
    # Create all diagrams
    diagrams = [
        ("Peak Detection", create_peak_detection_diagram),
        ("Second Derivative", create_second_derivative_diagram),
        ("Type Signal Difference", create_type_signal_difference_diagram),
        ("Context Patterns", create_context_pattern_diagram)
    ]
    
    created_files = []
    
    for name, func in diagrams:
        print(f"\nCreating {name} diagram...")
        try:
            output_path = func(output_dir)
            created_files.append(output_path)
            print(f"   Saved: {output_path}")
        except Exception as e:
            print(f"   Error creating {name}: {e}")
    
    print(f"\n✅ Diagram creation complete!")
    print(f"Created {len(created_files)} diagrams:")
    for file_path in created_files:
        print(f"  - {Path(file_path).name}")

if __name__ == "__main__":
    main() 