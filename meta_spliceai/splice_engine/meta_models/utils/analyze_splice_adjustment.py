from typing import List, Dict, Tuple, Optional, Union, Any

import os
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd  # For converting polars to pandas DataFrames

def create_adjustment_comparison_plot(
    positions_df_without_adj, 
    positions_df_with_adj, 
    output_path,
    site_types=None,
    pred_types=None,
    figsize=(20, 16)  # Taller to accommodate more plots
):
    """
    Create a comparison visualization showing the effect of splice site adjustments.
    
    Parameters:
    -----------
    positions_df_without_adj : pl.DataFrame
        DataFrame with positions without adjustments
    positions_df_with_adj : pl.DataFrame
        DataFrame with positions with adjustments
    output_path : str
        Path to save the comparison plot
    site_types : list, optional
        List of site types to include ("donor", "acceptor"). If None, include all.
    pred_types : list, optional
        List of prediction types to include ("TP", "FP", "FN", "TN"). If None, defaults to ["TP", "FP", "FN"].
    figsize : tuple, optional
        Figure size as (width, height)
    
    Returns:
    --------
    str
        Path to the saved plot
    """
    if positions_df_without_adj.height == 0 or positions_df_with_adj.height == 0:
        print("Warning: Empty positions DataFrame, cannot create comparison plot")
        return None
    
    # Default values
    if site_types is None:
        site_types = ["donor", "acceptor"]
    if pred_types is None:
        pred_types = ["TP", "FP", "FN"]
    
    # Create subplots for each site type with more spacing
    n_site_types = len(site_types)
    
    # First create the main comparison plots
    fig, axes = plt.subplots(n_site_types, 5, figsize=figsize)  # Added a fifth column
    
    # If only one site type, wrap axes in list for consistent indexing
    if n_site_types == 1:
        axes = [axes]
    
    # Add more space between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    
    # Color map for prediction types
    colors = {
        "TP": "green",
        "FP": "red",
        "FN": "orange",
        "TN": "blue"
    }
    
    # Create plots for each site type
    for i, site_type in enumerate(site_types):
        # Filter data for this site type
        df_without_adj = positions_df_without_adj.filter(
            (pl.col("pred_type").is_in(pred_types)) & 
            (pl.col("splice_type") == site_type)
        ).to_pandas()
        
        df_with_adj = positions_df_with_adj.filter(
            (pl.col("pred_type").is_in(pred_types)) & 
            (pl.col("splice_type") == site_type)
        ).to_pandas()
        
        # Skip if no data for this site type
        if len(df_without_adj) == 0 or len(df_with_adj) == 0:
            print(f"Warning: No data for site type '{site_type}', skipping")
            continue
        
        # Calculate probability sums for diagnostic purposes
        df_without_adj['prob_sum'] = df_without_adj['donor_score'] + df_without_adj['acceptor_score'] + df_without_adj['neither_score']
        df_with_adj['prob_sum'] = df_with_adj['donor_score'] + df_with_adj['acceptor_score'] + df_with_adj['neither_score']
        
        # Plot 1: Without adjustments - donor vs acceptor
        ax = axes[i][0]
        for pred_type in pred_types:
            subset = df_without_adj[df_without_adj["pred_type"] == pred_type]
            if len(subset) > 0:
                ax.scatter(
                    subset["donor_score"], 
                    subset["acceptor_score"], 
                    alpha=0.5, 
                    label=pred_type,
                    color=colors.get(pred_type)
                )
        
        ax.set_xlabel("Donor Probability")
        ax.set_ylabel("Acceptor Probability")
        ax.set_title(f"{site_type.capitalize()} - Original", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: With adjustments - donor vs acceptor
        ax = axes[i][1]
        for pred_type in pred_types:
            subset = df_with_adj[df_with_adj["pred_type"] == pred_type]
            if len(subset) > 0:
                ax.scatter(
                    subset["donor_score"], 
                    subset["acceptor_score"], 
                    alpha=0.5, 
                    label=pred_type,
                    color=colors.get(pred_type)
                )
        
        ax.set_xlabel("Donor Probability")
        ax.set_ylabel("Acceptor Probability")
        ax.set_title(f"{site_type.capitalize()} - Adjusted", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 3: Without adjustments - splice vs neither
        ax = axes[i][2]
        for pred_type in pred_types:
            subset = df_without_adj[df_without_adj["pred_type"] == pred_type]
            if len(subset) > 0:
                ax.scatter(
                    subset["donor_score"] + subset["acceptor_score"], 
                    subset["neither_score"], 
                    alpha=0.5, 
                    label=pred_type,
                    color=colors.get(pred_type)
                )
        
        # Add diagonal line where sum equals 1
        x = np.linspace(0, 1, 100)
        y = 1 - x
        ax.plot(x, y, 'k--', alpha=0.7, linewidth=1)
        
        ax.set_xlabel("Splice Prob (D+A)")
        ax.set_ylabel("Neither Prob")
        ax.set_title(f"{site_type.capitalize()} - Original Sum", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 4: With adjustments - splice vs neither
        ax = axes[i][3]
        for pred_type in pred_types:
            subset = df_with_adj[df_with_adj["pred_type"] == pred_type]
            if len(subset) > 0:
                ax.scatter(
                    subset["donor_score"] + subset["acceptor_score"], 
                    subset["neither_score"], 
                    alpha=0.5, 
                    label=pred_type,
                    color=colors.get(pred_type)
                )
        
        # Add diagonal line where sum equals 1
        ax.plot(x, y, 'k--', alpha=0.7, linewidth=1)
        
        ax.set_xlabel("Splice Prob (D+A)")
        ax.set_ylabel("Neither Prob")
        ax.set_title(f"{site_type.capitalize()} - Adjusted Sum", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 5: Probability Sum Histogram (new)
        ax = axes[i][4]
        # Fix the bins to be in a reasonable probability range (0 to 2 to handle any sums > 1)
        bins = np.linspace(0, 2, 50)
        ax.hist(df_without_adj['prob_sum'], bins=bins, alpha=0.5, label='Original', color='blue')
        ax.hist(df_with_adj['prob_sum'], bins=bins, alpha=0.5, label='Adjusted', color='orange')
        ax.axvline(x=1.0, color='k', linestyle='--', alpha=0.7)
        ax.set_xlim(0, 2)  # Set explicit limits to ensure we see the right range
        ax.set_xlabel("Total Probability Sum (D+A+N)")
        ax.set_ylabel("Count")
        ax.set_title(f"{site_type.capitalize()} - Probability Sums", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add statistics to the histogram plot
        stats_text = (
            f"Original: mean={df_without_adj['prob_sum'].mean():.3f}, std={df_without_adj['prob_sum'].std():.3f}\n"
            f"Adjusted: mean={df_with_adj['prob_sum'].mean():.3f}, std={df_with_adj['prob_sum'].std():.3f}"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round', alpha=0.1))
        
        # Add a note explaining why differences might not be immediately visible
        if i == 0:  # Only add this text to the first row
            fig.text(0.5, 0.01, 
                    "Note: The 'Original' and 'Adjusted' plots may look similar because the adjustments\n"
                    "primarily affect positions with high-confidence predictions, which are a small subset of all positions.",
                    ha='center', fontsize=10, style='italic')
    
    # Top-level title to explain plots better
    fig.suptitle("Splice Site Prediction Distributions: Original vs Position-Adjusted", 
                 fontsize=14, y=0.98)
    
    # Add row and column labels
    if n_site_types > 1:
        for i, site_type in enumerate(site_types):
            # Add a text label for each row
            fig.text(0.02, 0.5 + (0.5 - i)/(n_site_types), 
                    f"{site_type.upper()} SITES", 
                    fontsize=12, rotation=90, 
                    verticalalignment='center')
    
    # Add column explanations
    col_titles = ["Donor vs Acceptor", "Donor vs Acceptor", "Splice vs Neither", "Splice vs Neither", "Probability Sum"]
    col_subtitles = ["(Before Adjustment)", "(After Adjustment)", "(Before Adjustment)", "(After Adjustment)", "(Diagnostic)"]
    
    for i, (title, subtitle) in enumerate(zip(col_titles, col_subtitles)):
        fig.text(0.1 + i*0.2, 0.95, title, ha='center', fontsize=12)
        fig.text(0.1 + i*0.2, 0.93, subtitle, ha='center', fontsize=10, style='italic')
    
    # Add annotation to explain the diagonal line in plots 3 and 4
    fig.text(0.5, 0.02, 
             "Note: Dashed diagonal lines in 'Splice vs Neither' plots represent the theoretical constraint where D+A+N=1.\n"
             "Points above this line have probabilities that sum to >1, indicating a potential normalization issue.",
             ha='center', fontsize=10)
    
    # Tight layout with padding
    plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.90])  # Leave space for titles
    
    # Save the plot with higher DPI for better quality
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    # Now create a second figure focusing just on high-probability sites
    # This will make the adjustment effects much more visible
    high_prob_path = output_path.replace('.pdf', '_high_prob.pdf')
    
    fig2, axes2 = plt.subplots(n_site_types, 2, figsize=(12, 5 * n_site_types))
    if n_site_types == 1:
        axes2 = [axes2]
    
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for i, site_type in enumerate(site_types):
        # Filter data for this site type - only high probability predictions
        high_threshold = 0.5  # Consider scores >= 0.5 as high probability
        
        if site_type == 'donor':
            # For donor sites, filter by donor score
            df_without_adj_high = positions_df_without_adj.filter(
                (pl.col("pred_type").is_in(pred_types)) & 
                (pl.col("splice_type") == site_type) &
                (pl.col("donor_score") >= high_threshold)
            ).to_pandas()
            
            df_with_adj_high = positions_df_with_adj.filter(
                (pl.col("pred_type").is_in(pred_types)) & 
                (pl.col("splice_type") == site_type) &
                (pl.col("donor_score") >= high_threshold)
            ).to_pandas()
        else:
            # For acceptor sites, filter by acceptor score
            df_without_adj_high = positions_df_without_adj.filter(
                (pl.col("pred_type").is_in(pred_types)) & 
                (pl.col("splice_type") == site_type) &
                (pl.col("acceptor_score") >= high_threshold)
            ).to_pandas()
            
            df_with_adj_high = positions_df_with_adj.filter(
                (pl.col("pred_type").is_in(pred_types)) & 
                (pl.col("splice_type") == site_type) &
                (pl.col("acceptor_score") >= high_threshold)
            ).to_pandas()
        
        # Skip if no high-probability data for this site type
        if len(df_without_adj_high) == 0 and len(df_with_adj_high) == 0:
            print(f"Warning: No high-probability data for site type '{site_type}', skipping")
            continue
        
        # Calculate probability sums
        if len(df_without_adj_high) > 0:
            df_without_adj_high['prob_sum'] = df_without_adj_high['donor_score'] + df_without_adj_high['acceptor_score'] + df_without_adj_high['neither_score']
        if len(df_with_adj_high) > 0:
            df_with_adj_high['prob_sum'] = df_with_adj_high['donor_score'] + df_with_adj_high['acceptor_score'] + df_with_adj_high['neither_score']
        
        # Plot 1: Focused view of Splice vs Neither for high-probability sites - WITHOUT adjustment
        ax = axes2[i][0]
        for pred_type in pred_types:
            if len(df_without_adj_high) > 0:
                subset = df_without_adj_high[df_without_adj_high["pred_type"] == pred_type]
                if len(subset) > 0:
                    ax.scatter(
                        subset["donor_score"] + subset["acceptor_score"], 
                        subset["neither_score"], 
                        alpha=0.7, 
                        label=pred_type,
                        color=colors.get(pred_type),
                        edgecolor='k',
                        s=80  # Larger point size for better visibility
                    )
        
        # Add diagonal line where sum equals 1
        x = np.linspace(0, 1, 100)
        y = 1 - x
        ax.plot(x, y, 'k--', alpha=0.7, linewidth=1)
        
        ax.set_xlabel("Splice Probability (D+A)")
        ax.set_ylabel("Neither Probability")
        ax.set_title(f"{site_type.capitalize()} - High Probability Sites WITHOUT Adjustment", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add count of points
        if len(df_without_adj_high) > 0:
            ax.text(0.05, 0.95, f"n={len(df_without_adj_high)}", transform=ax.transAxes, 
                    verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', alpha=0.1))
        
        # Plot 2: Focused view of Splice vs Neither for high-probability sites - WITH adjustment
        ax = axes2[i][1]
        for pred_type in pred_types:
            if len(df_with_adj_high) > 0:
                subset = df_with_adj_high[df_with_adj_high["pred_type"] == pred_type]
                if len(subset) > 0:
                    ax.scatter(
                        subset["donor_score"] + subset["acceptor_score"], 
                        subset["neither_score"], 
                        alpha=0.7, 
                        label=pred_type,
                        color=colors.get(pred_type),
                        edgecolor='k',
                        s=80  # Larger point size for better visibility
                    )
        
        # Add diagonal line where sum equals 1
        ax.plot(x, y, 'k--', alpha=0.7, linewidth=1)
        
        ax.set_xlabel("Splice Probability (D+A)")
        ax.set_ylabel("Neither Probability")
        ax.set_title(f"{site_type.capitalize()} - High Probability Sites WITH Adjustment", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add count of points
        if len(df_with_adj_high) > 0:
            ax.text(0.05, 0.95, f"n={len(df_with_adj_high)}", transform=ax.transAxes, 
                    verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', alpha=0.1))
    
    fig2.suptitle("Effect of Adjustments on High-Probability Sites", fontsize=14, y=0.98)
    
    # Add a text explaining the purpose of this plot
    fig2.text(0.5, 0.01, 
               "This figure focuses only on high-probability sites (score >= 0.5) to better visualize the effect of adjustments.\n"
               "The diagonal line represents the theoretical constraint where D+A+N=1.",
               ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.95])
    plt.savefig(high_prob_path, dpi=150)
    plt.close()
    
    print(f"Additional high-probability focus plot saved to: {high_prob_path}")
    
    return output_path