#!/usr/bin/env python3
"""
Probability Feature Visualization Tool

This script creates comprehensive visualizations to help understand the 
probability-based and context-based features used in SpliceAI meta-models.

Features visualized:
- Signal processing features (peak detection, derivatives, etc.)
- Context-based features (background, asymmetry, etc.)
- Cross-type comparison features
- Real examples from splice site data

Usage:
    python visualize_probability_features.py --data-file path/to/enhanced_positions.tsv --output-dir plots/
"""

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")

# Color palette for consistent visualization
COLORS = {
    'donor': '#2E86AB',      # Blue
    'acceptor': '#A23B72',   # Pink/Purple  
    'neither': '#F18F01',    # Orange
    'context': '#C73E1D',    # Red
    'background': '#CCCCCC', # Light gray
    'peak': '#FF6B6B',       # Light red
    'trough': '#4ECDC4'      # Teal
}

def load_enhanced_positions_data(file_path: str, sample_size: Optional[int] = None) -> pl.DataFrame:
    """Load enhanced positions data with probability features."""
    print(f"Loading data from: {file_path}")
    
    # Try different file formats
    if file_path.endswith('.parquet'):
        df = pl.read_parquet(file_path)
    elif file_path.endswith('.tsv'):
        df = pl.read_csv(file_path, separator='\t')
    else:
        # Try TSV first, then parquet
        try:
            df = pl.read_csv(file_path, separator='\t')
        except:
            df = pl.read_parquet(file_path)
    
    print(f"Loaded {df.height:,} positions with {len(df.columns)} features")
    
    # Sample data if requested
    if sample_size and df.height > sample_size:
        df = df.sample(sample_size, seed=42)
        print(f"Sampled {sample_size:,} positions for visualization")
    
    return df

def create_signal_processing_conceptual_plot(output_dir: str) -> str:
    """Create a conceptual diagram showing signal processing features."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Signal Processing Concepts for Splice Site Features', fontsize=16, fontweight='bold')
    
    # Generate synthetic signal for demonstration
    x = np.linspace(-5, 5, 101)
    
    # 1. Peak Detection
    ax = axes[0, 0]
    
    # True peak (sharp)
    true_peak = np.exp(-(x**2)/0.5)
    # False peak (broad)
    false_peak = 0.6 * np.exp(-(x**2)/2) + 0.1
    
    ax.plot(x, true_peak, 'b-', linewidth=3, label='True Splice Site (Sharp Peak)')
    ax.plot(x, false_peak, 'r--', linewidth=3, label='False Positive (Broad Signal)')
    
    # Mark peak positions
    ax.scatter([0], [1.0], color='blue', s=100, zorder=5)
    ax.scatter([0], [0.7], color='red', s=100, zorder=5)
    
    # Add context positions
    for pos in [-2, -1, 1, 2]:
        ax.axvline(pos, color='gray', alpha=0.3, linestyle=':')
        ax.text(pos, -0.1, f'{pos:+d}', ha='center', va='top', fontsize=10)
    
    ax.set_title('Peak Detection & Peak Height Ratio', fontweight='bold')
    ax.set_xlabel('Position relative to splice site')
    ax.set_ylabel('Probability Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    ax.annotate('peak_height_ratio = score / neighbor_mean', 
                xy=(0, 1.0), xytext=(2, 1.2),
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                fontsize=10, color='blue')
    
    # 2. Second Derivative (Curvature)
    ax = axes[0, 1]
    
    # Sharp peak - positive curvature
    sharp_x = x[40:61]  # Zoom in around peak
    sharp_y = true_peak[40:61]
    
    # Broad peak - negative curvature  
    broad_x = x[40:61]
    broad_y = false_peak[40:61]
    
    ax.plot(sharp_x, sharp_y, 'b-', linewidth=3, label='Sharp Peak (+ curvature)')
    ax.plot(broad_x, broad_y, 'r--', linewidth=3, label='Broad Peak (- curvature)')
    
    # Show curvature with arrows
    ax.annotate('', xy=(-0.5, 0.88), xytext=(-1, 0.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.annotate('', xy=(0.5, 0.88), xytext=(1, 0.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    ax.set_title('Second Derivative (Curvature Analysis)', fontweight='bold')
    ax.set_xlabel('Position relative to splice site')
    ax.set_ylabel('Probability Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add formula
    ax.text(0.05, 0.95, 'second_derivative = (score - m1) - (p1 - score)', 
            transform=ax.transAxes, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
    
    # 3. Signal Strength (Background Subtraction)
    ax = axes[1, 0]
    
    # Signal with varying background
    background = 0.1 + 0.05 * np.sin(x * 0.5)
    signal_with_bg = true_peak + background
    
    ax.plot(x, signal_with_bg, 'g-', linewidth=3, label='Raw Signal')
    ax.plot(x, background, 'gray', linewidth=2, label='Background Level')
    ax.fill_between(x, background, signal_with_bg, alpha=0.3, color='green', label='Signal Strength')
    
    ax.set_title('Signal Strength (Background Subtraction)', fontweight='bold')
    ax.set_xlabel('Position relative to splice site')
    ax.set_ylabel('Probability Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add formula
    ax.text(0.05, 0.95, 'signal_strength = score - neighbor_mean', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # 4. Cross-Type Features
    ax = axes[1, 1]
    
    # Donor vs Acceptor signals
    donor_signal = np.exp(-((x-0.5)**2)/0.3)
    acceptor_signal = 0.4 * np.exp(-((x+0.5)**2)/0.5)
    
    ax.plot(x, donor_signal, color=COLORS['donor'], linewidth=3, label='Donor Score')
    ax.plot(x, acceptor_signal, color=COLORS['acceptor'], linewidth=3, label='Acceptor Score')
    
    # Mark the difference
    ax.fill_between(x, donor_signal, acceptor_signal, alpha=0.3, color='purple', 
                   label='Type Signal Difference')
    
    ax.set_title('Cross-Type Comparison Features', fontweight='bold')
    ax.set_xlabel('Position relative to splice site')
    ax.set_ylabel('Probability Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add formula
    ax.text(0.05, 0.95, 'type_signal_difference = donor_strength - acceptor_strength', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="plum", alpha=0.7))
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "signal_processing_concepts.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)

def visualize_real_feature_examples(df: pl.DataFrame, output_dir: str, 
                                  max_examples: int = 20) -> List[str]:
    """Create visualizations showing real examples of features on actual data."""
    
    output_paths = []
    
    # Convert to pandas for easier plotting
    df_pd = df.to_pandas()
    
    # 1. Feature Distribution Analysis
    print("Creating feature distribution plots...")
    
    # Key features to visualize
    key_features = [
        'donor_peak_height_ratio', 'acceptor_peak_height_ratio',
        'donor_second_derivative', 'acceptor_second_derivative', 
        'type_signal_difference', 'splice_neither_diff',
        'donor_signal_strength', 'acceptor_signal_strength'
    ]
    
    # Filter to features that exist in the data
    available_features = [f for f in key_features if f in df_pd.columns]
    
    if not available_features:
        print("Warning: No key features found in data. Available columns:")
        print(df_pd.columns.tolist())
        return output_paths
    
    # Create distribution plots
    n_features = len(available_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Feature Distributions by Prediction Type', fontsize=16, fontweight='bold')
    
    for i, feature in enumerate(available_features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Plot distributions by prediction type
        if 'pred_type' in df_pd.columns:
            for pred_type in ['TP', 'FP', 'FN', 'TN']:
                if pred_type in df_pd['pred_type'].values:
                    data = df_pd[df_pd['pred_type'] == pred_type][feature].dropna()
                    if len(data) > 0:
                        ax.hist(data, alpha=0.6, bins=30, label=f'{pred_type} (n={len(data)})', density=True)
        else:
            # Fallback: plot overall distribution
            data = df_pd[feature].dropna()
            ax.hist(data, alpha=0.6, bins=30, density=True)
        
        ax.set_title(feature.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Density')
        if 'pred_type' in df_pd.columns:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "feature_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    output_paths.append(str(output_path))
    
    # 2. Feature Correlation Heatmap
    print("Creating feature correlation heatmap...")
    
    # Select numeric features for correlation analysis
    numeric_features = []
    for col in df_pd.columns:
        if df_pd[col].dtype in ['int64', 'float64'] and col not in ['position', 'window_start', 'window_end']:
            numeric_features.append(col)
    
    if len(numeric_features) > 2:
        # Limit to most interesting features to keep heatmap readable
        if len(numeric_features) > 20:
            # Prioritize key features
            priority_features = [f for f in available_features if f in numeric_features]
            other_features = [f for f in numeric_features if f not in priority_features]
            numeric_features = priority_features + other_features[:20-len(priority_features)]
        
        corr_matrix = df_pd[numeric_features].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        output_path = Path(output_dir) / "feature_correlation_heatmap.png" 
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths.append(str(output_path))
    
    # 3. Individual Feature Examples
    print("Creating individual feature example plots...")
    
    # Focus on most interpretable features
    example_features = [
        ('donor_peak_height_ratio', 'Peak Height Analysis'),
        ('type_signal_difference', 'Cross-Type Comparison'),
        ('splice_neither_diff', 'Splice vs Neither Discrimination')
    ]
    
    for feature, title in example_features:
        if feature not in df_pd.columns:
            continue
            
        # Create scatter plot showing feature vs outcome
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{title}: {feature}', fontsize=16, fontweight='bold')
        
        # Left plot: Feature distribution by prediction type
        if 'pred_type' in df_pd.columns:
            pred_types = ['TP', 'FP', 'FN', 'TN']
            colors = ['green', 'red', 'orange', 'blue']
            
            for pred_type, color in zip(pred_types, colors):
                if pred_type in df_pd['pred_type'].values:
                    data = df_pd[df_pd['pred_type'] == pred_type][feature].dropna()
                    if len(data) > 0:
                        ax1.hist(data, alpha=0.6, bins=20, label=f'{pred_type} (n={len(data)})', 
                               color=color, density=True)
            
            ax1.set_xlabel(f'{feature} Value')
            ax1.set_ylabel('Density')
            ax1.set_title('Distribution by Prediction Type')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Right plot: Feature vs another key feature (if available)
        if 'splice_probability' in df_pd.columns and feature != 'splice_probability':
            # Scatter plot
            if 'pred_type' in df_pd.columns:
                for pred_type, color in zip(['TP', 'FP'], ['green', 'red']):
                    if pred_type in df_pd['pred_type'].values:
                        subset = df_pd[df_pd['pred_type'] == pred_type]
                        if len(subset) > 0:
                            sample_size = min(1000, len(subset))  # Limit points for readability
                            sample = subset.sample(n=sample_size) if len(subset) > sample_size else subset
                            ax2.scatter(sample['splice_probability'], sample[feature], 
                                      alpha=0.6, s=20, label=pred_type, color=color)
            else:
                # Fallback: plot all points
                sample_size = min(2000, len(df_pd))
                sample = df_pd.sample(n=sample_size) if len(df_pd) > sample_size else df_pd
                ax2.scatter(sample['splice_probability'], sample[feature], alpha=0.6, s=20)
            
            ax2.set_xlabel('Splice Probability')
            ax2.set_ylabel(f'{feature} Value')
            ax2.set_title('Feature vs Splice Probability')
            if 'pred_type' in df_pd.columns:
                ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # Alternative: show summary statistics
            ax2.axis('off')
            if 'pred_type' in df_pd.columns:
                stats_text = []
                for pred_type in ['TP', 'FP', 'FN', 'TN']:
                    if pred_type in df_pd['pred_type'].values:
                        data = df_pd[df_pd['pred_type'] == pred_type][feature].dropna()
                        if len(data) > 0:
                            mean_val = data.mean()
                            std_val = data.std()
                            stats_text.append(f'{pred_type}: μ={mean_val:.3f}, σ={std_val:.3f}')
                
                ax2.text(0.1, 0.9, 'Summary Statistics:\n' + '\n'.join(stats_text),
                        transform=ax2.transAxes, fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        safe_feature_name = feature.replace('_', '-')
        output_path = Path(output_dir) / f"feature_example_{safe_feature_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        output_paths.append(str(output_path))
    
    return output_paths

def create_context_visualization(df: pl.DataFrame, output_dir: str) -> Optional[str]:
    """Create visualization showing context score patterns around splice sites."""
    
    # Check if context columns are available
    context_cols = ['context_score_m2', 'context_score_m1', 'context_score_p1', 'context_score_p2']
    available_context = [col for col in context_cols if col in df.columns]
    
    if len(available_context) < 4:
        print(f"Warning: Not enough context columns found. Available: {available_context}")
        return None
    
    print("Creating context pattern visualization...")
    
    # Convert to pandas for easier manipulation
    df_pd = df.to_pandas()
    
    # Filter to positions with all required scores
    required_cols = ['donor_score', 'acceptor_score'] + context_cols
    if 'pred_type' in df_pd.columns:
        required_cols.append('pred_type')
    
    # Remove rows with missing context data
    df_clean = df_pd.dropna(subset=required_cols)
    
    if len(df_clean) == 0:
        print("Warning: No complete context data found")
        return None
    
    # Sample data for visualization (to keep plots readable)
    sample_size = min(500, len(df_clean))
    df_sample = df_clean.sample(n=sample_size, random_state=42)
    
    # Create context pattern plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Context Score Patterns Around Splice Sites', fontsize=16, fontweight='bold')
    
    # Define positions for context visualization
    positions = [-2, -1, 0, 1, 2]
    
    # 1. Average context patterns by prediction type
    ax = axes[0, 0]
    
    if 'pred_type' in df_sample.columns:
        for pred_type, color in zip(['TP', 'FP'], ['green', 'red']):
            if pred_type in df_sample['pred_type'].values:
                subset = df_sample[df_sample['pred_type'] == pred_type]
                if len(subset) > 0:
                    # Calculate mean scores at each position
                    means = [
                        subset['context_score_m2'].mean(),
                        subset['context_score_m1'].mean(), 
                        subset['donor_score'].mean(),  # Position 0
                        subset['context_score_p1'].mean(),
                        subset['context_score_p2'].mean()
                    ]
                    ax.plot(positions, means, 'o-', linewidth=3, markersize=8, 
                           label=f'{pred_type} (n={len(subset)})', color=color)
    
    ax.set_title('Average Context Patterns', fontweight='bold')
    ax.set_xlabel('Position Relative to Splice Site')
    ax.set_ylabel('Average Probability Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='Splice Site')
    
    # 2. Individual examples (heatmap style)
    ax = axes[0, 1]
    
    # Select interesting examples (high and low peak height ratios)
    if 'donor_peak_height_ratio' in df_sample.columns:
        # Get examples with different peak characteristics
        high_peaks = df_sample.nlargest(10, 'donor_peak_height_ratio')
        low_peaks = df_sample.nsmallest(10, 'donor_peak_height_ratio')
        examples = pd.concat([high_peaks, low_peaks])
    else:
        examples = df_sample.head(20)
    
    # Create heatmap of context patterns
    context_matrix = []
    labels = []
    
    for idx, row in examples.iterrows():
        pattern = [
            row['context_score_m2'],
            row['context_score_m1'],
            row['donor_score'],
            row['context_score_p1'],
            row['context_score_p2']
        ]
        context_matrix.append(pattern)
        
        # Create label
        pred_type = row.get('pred_type', 'Unknown')
        peak_ratio = row.get('donor_peak_height_ratio', 0)
        labels.append(f'{pred_type} (r={peak_ratio:.1f})')
    
    context_matrix = np.array(context_matrix)
    
    im = ax.imshow(context_matrix, cmap='viridis', aspect='auto')
    ax.set_title('Individual Context Patterns', fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Examples')
    ax.set_xticks(range(5))
    ax.set_xticklabels(['-2', '-1', '0', '+1', '+2'])
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Probability Score')
    
    # 3. Context asymmetry analysis
    ax = axes[1, 0]
    
    if 'context_asymmetry' in df_sample.columns and 'pred_type' in df_sample.columns:
        # Plot asymmetry distributions
        for pred_type, color in zip(['TP', 'FP'], ['green', 'red']):
            if pred_type in df_sample['pred_type'].values:
                data = df_sample[df_sample['pred_type'] == pred_type]['context_asymmetry'].dropna()
                if len(data) > 0:
                    ax.hist(data, alpha=0.6, bins=20, label=f'{pred_type} (n={len(data)})', 
                           color=color, density=True)
        
        ax.set_title('Context Asymmetry Distribution', fontweight='bold')
        ax.set_xlabel('Context Asymmetry (upstream - downstream)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5, label='Symmetric')
    else:
        ax.text(0.5, 0.5, 'Context asymmetry\nfeature not available', 
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.axis('off')
    
    # 4. Peak detection visualization
    ax = axes[1, 1]
    
    if 'donor_is_local_peak' in df_sample.columns and 'pred_type' in df_sample.columns:
        # Create contingency table
        crosstab = pd.crosstab(df_sample['pred_type'], df_sample['donor_is_local_peak'], 
                              normalize='index') * 100
        
        # Plot as bar chart
        crosstab.plot(kind='bar', ax=ax, color=['lightcoral', 'lightgreen'])
        ax.set_title('Local Peak Detection by Prediction Type', fontweight='bold')
        ax.set_xlabel('Prediction Type')
        ax.set_ylabel('Percentage')
        ax.legend(['Not Local Peak', 'Is Local Peak'])
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, 'Peak detection\nfeature not available', 
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "context_patterns.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)

def create_feature_interpretation_guide(output_dir: str) -> str:
    """Create a visual guide for interpreting feature values."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature Interpretation Guide', fontsize=18, fontweight='bold')
    
    # 1. Peak Height Ratio Interpretation
    ax = axes[0, 0]
    
    ratios = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    interpretations = ['Very Weak', 'Weak', 'Moderate', 'Good', 'Strong', 'Very Strong']
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
    
    bars = ax.barh(range(len(ratios)), ratios, color=colors, alpha=0.7)
    ax.set_yticks(range(len(ratios)))
    ax.set_yticklabels([f'{r:.1f}' for r in ratios])
    ax.set_xlabel('Peak Height Ratio Value')
    ax.set_ylabel('Ratio Value')
    ax.set_title('Peak Height Ratio Interpretation', fontweight='bold')
    
    # Add interpretation labels
    for i, (bar, interp) in enumerate(zip(bars, interpretations)):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
               interp, ha='left', va='center', fontweight='bold')
    
    ax.set_xlim(0, 6)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 2. Second Derivative Interpretation  
    ax = axes[0, 1]
    
    x = np.linspace(-2, 2, 100)
    
    # Positive curvature (sharp peak)
    sharp_peak = 1 - 0.5 * x**2
    # Negative curvature (broad peak)
    broad_peak = 0.8 - 0.1 * x**2
    
    ax.plot(x, sharp_peak, 'g-', linewidth=3, label='Positive 2nd Derivative\n(Sharp Peak - True Site)')
    ax.plot(x, broad_peak, 'r--', linewidth=3, label='Negative 2nd Derivative\n(Broad Peak - False Positive)')
    
    ax.set_title('Second Derivative Interpretation', fontweight='bold')
    ax.set_xlabel('Position')
    ax.set_ylabel('Probability Score') 
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotations
    ax.text(-1.5, 0.3, 'Sharp curvature\n(2nd deriv > 0)', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    ax.text(0.5, 0.4, 'Flat/broad\n(2nd deriv < 0)', 
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    # 3. Type Signal Difference
    ax = axes[1, 0]
    
    differences = np.array([-0.3, -0.1, 0, 0.1, 0.3])
    type_prefs = ['Strong Acceptor', 'Weak Acceptor', 'Ambiguous', 'Weak Donor', 'Strong Donor']
    type_colors = ['purple', 'plum', 'gray', 'lightblue', 'blue']
    
    bars = ax.bar(range(len(differences)), differences, color=type_colors, alpha=0.7)
    ax.set_xticks(range(len(differences)))
    ax.set_xticklabels(type_prefs, rotation=45, ha='right')
    ax.set_ylabel('Type Signal Difference')
    ax.set_title('Type Signal Difference Interpretation', fontweight='bold')
    ax.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, differences):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + (0.01 if height >= 0 else -0.01),
               f'{val:+.1f}', ha='center', va='bottom' if height >= 0 else 'top', 
               fontweight='bold')
    
    # 4. Splice Probability Thresholds
    ax = axes[1, 1]
    
    # Create probability ranges and their interpretations
    prob_ranges = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
    range_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High', 'Extremely High']
    confidence_colors = ['red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
    
    # Create a visual representation
    y_pos = np.arange(len(prob_ranges))
    bar_heights = [end - start for start, end in prob_ranges]
    bar_bottoms = [start for start, end in prob_ranges]
    
    bars = ax.barh(y_pos, bar_heights, left=bar_bottoms, color=confidence_colors, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{start:.1f}-{end:.1f}' for start, end in prob_ranges])
    ax.set_xlabel('Splice Probability')
    ax.set_ylabel('Probability Range')
    ax.set_title('Splice Probability Confidence Levels', fontweight='bold')
    
    # Add interpretation labels
    for i, (bar, label) in enumerate(zip(bars, range_labels)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2,
               label, ha='center', va='center', fontweight='bold', 
               color='white' if i > 2 else 'black')
    
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "feature_interpretation_guide.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_path)

def main():
    """Main function to create all visualizations."""
    
    parser = argparse.ArgumentParser(description="Visualize probability-based features for splice site prediction")
    parser.add_argument("--data-file", type=str, required=True,
                       help="Path to enhanced positions data file (TSV or Parquet)")
    parser.add_argument("--output-dir", type=str, default="results/probability_feature_analysis/visualizations",
                       help="Output directory for plots")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Sample size for visualization (default: use all data)")
    parser.add_argument("--max-examples", type=int, default=20,
                       help="Maximum number of examples to show in detailed plots")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating probability feature visualizations...")
    print(f"Output directory: {output_dir}")
    
    # Create conceptual plots (no data required)
    print("\n1. Creating conceptual signal processing diagrams...")
    concept_path = create_signal_processing_conceptual_plot(str(output_dir))
    print(f"   Saved: {concept_path}")
    
    # Create interpretation guide
    print("\n2. Creating feature interpretation guide...")
    guide_path = create_feature_interpretation_guide(str(output_dir))
    print(f"   Saved: {guide_path}")
    
    # Load and analyze real data if provided
    if os.path.exists(args.data_file):
        print(f"\n3. Loading and analyzing real data from: {args.data_file}")
        
        try:
            df = load_enhanced_positions_data(args.data_file, args.sample_size)
            
            # Create real data visualizations
            print("\n4. Creating feature distribution and correlation plots...")
            real_data_paths = visualize_real_feature_examples(df, str(output_dir), args.max_examples)
            for path in real_data_paths:
                print(f"   Saved: {path}")
            
            # Create context visualization
            print("\n5. Creating context pattern visualization...")
            context_path = create_context_visualization(df, str(output_dir))
            if context_path:
                print(f"   Saved: {context_path}")
            else:
                print("   Skipped: insufficient context data")
                
        except Exception as e:
            print(f"Error processing data file: {e}")
            print("Continuing with conceptual plots only...")
    else:
        print(f"\nWarning: Data file not found: {args.data_file}")
        print("Creating conceptual plots only...")
    
    print(f"\n✅ All visualizations complete!")
    print(f"Check the output directory: {output_dir}")
    print(f"\nGenerated files:")
    for file_path in sorted(output_dir.glob("*.png")):
        print(f"  - {file_path.name}")

if __name__ == "__main__":
    main() 