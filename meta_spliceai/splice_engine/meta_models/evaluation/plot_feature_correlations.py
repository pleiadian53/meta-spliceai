#!/usr/bin/env python3
"""
Generate publication-ready feature correlation visualizations.

This script creates a horizontal bar chart showing feature correlations with labels,
excluding any features specified in an exclusion list.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def plot_feature_correlations(
    correlation_file: Path,
    exclude_file: Path = None,
    output_file: Path = None,
    top_n: int = 30,
    figsize: tuple = (10, 12),
    color_palette: str = "viridis",
    show_abs_corr: bool = True,
    title: str = "Feature-Label Correlation Strength",
):
    """
    Create a publication-ready horizontal bar chart of feature correlations.
    
    Parameters
    ----------
    correlation_file : Path
        Path to the feature_label_correlations.csv file
    exclude_file : Path, optional
        Path to the excluded_features.txt file
    output_file : Path, optional
        Path to save the output visualization
    top_n : int, default=30
        Number of top features to include in the visualization
    figsize : tuple, default=(10, 12)
        Figure size (width, height) in inches
    color_palette : str, default="viridis"
        Color palette to use for visualization
    show_abs_corr : bool, default=True
        If True, sort by absolute correlation; if False, sort by raw correlation
    title : str, default="Feature-Label Correlation Strength"
        Title for the plot
    """
    # Read correlation data
    if not correlation_file.exists():
        print(f"Error: Correlation file not found at {correlation_file}")
        return
    
    corr_df = pd.read_csv(correlation_file)
    print(f"Read {len(corr_df)} features from {correlation_file}")
    
    # Read excluded features if file provided
    excluded_features = []
    if exclude_file and exclude_file.exists():
        with open(exclude_file, 'r') as f:
            for line in f:
                # Skip comments and empty lines
                line = line.strip()
                if line and not line.startswith('#'):
                    excluded_features.append(line)
        print(f"Excluding {len(excluded_features)} features from visualization")
    
    # Filter out excluded features
    if excluded_features:
        corr_df = corr_df[~corr_df['feature'].isin(excluded_features)]
    
    # Sort by absolute or raw correlation
    sort_col = 'abs_correlation' if show_abs_corr else 'correlation'
    corr_df = corr_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    
    # Take top N features
    if len(corr_df) > top_n:
        corr_df = corr_df.iloc[:top_n]
        
    # Reverse the order so highest correlations appear at the top of the plot
    corr_df = corr_df.iloc[::-1].reset_index(drop=True)
    
    # Create the visualization
    plt.figure(figsize=figsize)
    
    # Set the style
    sns.set_style("whitegrid")
    
    # Create a categorical color palette based on feature groups
    # Group features by common prefixes for better visualization and semantic meaning
    def get_feature_group(feature_name):
        # Extract prefix from feature name for grouping
        if '_' in feature_name:
            prefix = feature_name.split('_')[0]
            # Handle special cases for better grouping
            if prefix in ['donor', 'acceptor']:
                if 'score' in feature_name:
                    return 'splice_score'
                elif 'diff' in feature_name:
                    return 'splice_diff'
                elif 'derivative' in feature_name:
                    return 'splice_derivative'
                else:
                    return prefix
            return prefix
        return feature_name
    
    feature_groups = corr_df['feature'].apply(get_feature_group)
    unique_groups = sorted(feature_groups.unique())
    color_map = dict(zip(unique_groups, 
                         sns.color_palette(color_palette, n_colors=len(unique_groups))))
    
    # Map each feature to its color based on group
    colors = [color_map[get_feature_group(feat)] for feat in corr_df['feature']]
    
    # Create the horizontal bar chart
    ax = plt.barh(
        y=corr_df['feature'],
        width=corr_df['correlation'],
        color=colors,
        height=0.7,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add correlation values as text labels
    for i, v in enumerate(corr_df['correlation']):
        plt.text(
            v + 0.01 if v >= 0 else v - 0.08, 
            i, 
            f"{v:.3f}", 
            va='center', 
            fontsize=9,
            fontweight='bold' if abs(v) > 0.7 else 'normal',
            color='black'
        )
    
    # Add a vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # Add grid lines
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Set plot labels and title
    plt.xlabel('Correlation with Target', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend for feature groups, positioned outside the plot
    legend_elements = [
        plt.Rectangle(
            (0, 0), 1, 1, 
            color=color_map[group], 
            label=group.capitalize()
        ) for group in unique_groups
    ]
    plt.legend(
        handles=legend_elements, 
        loc='upper center', 
        framealpha=0.9, 
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(4, len(unique_groups))
    )
    
    # Adjust layout to make room for the legend below
    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(bottom=0.22)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()
    
    return corr_df


def main():
    """Parse command-line arguments and run the visualization."""
    parser = argparse.ArgumentParser(
        description="Create publication-ready feature correlation visualizations"
    )
    parser.add_argument(
        "correlation_file",
        type=Path,
        help="Path to feature_label_correlations.csv file"
    )
    parser.add_argument(
        "--exclude-file",
        "-e",
        type=Path,
        help="Path to excluded_features.txt file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path for visualization (e.g., correlations.png)"
    )
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=30,
        help="Number of top features to include"
    )
    parser.add_argument(
        "--palette",
        "-p",
        default="viridis",
        choices=["viridis", "plasma", "inferno", "magma", "cividis", 
                 "Blues", "Greens", "Oranges", "Purples", "Reds"],
        help="Color palette to use"
    )
    parser.add_argument(
        "--raw-correlation",
        "-r",
        action="store_true",
        help="Sort by raw correlation instead of absolute value"
    )
    args = parser.parse_args()
    
    plot_feature_correlations(
        correlation_file=args.correlation_file,
        exclude_file=args.exclude_file,
        output_file=args.output,
        top_n=args.top,
        color_palette=args.palette,
        show_abs_corr=not args.raw_correlation
    )


if __name__ == "__main__":
    main()
