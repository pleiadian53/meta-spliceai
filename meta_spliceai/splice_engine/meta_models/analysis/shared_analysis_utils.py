"""
Shared utility functions for splice site analysis.

This module contains utility functions that are used by both the FN rescue
and FP reduction analysis modules, to avoid code duplication and ensure
consistent implementations.
"""

import os
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# System config for file paths
from meta_spliceai.system.config import Config


def get_detailed_splice_site_counts(positions_df: pl.DataFrame) -> pd.DataFrame:
    """
    Get detailed counts of splice sites by gene, categorized by type and prediction.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing all splice positions
        
    Returns
    -------
    pd.DataFrame
        DataFrame with detailed counts for each gene, including:
        - total_positions: All positions evaluated for the gene
        - true_sites: Number of actual splice sites (label=1)
        - donor_sites: Number of donor splice sites
        - acceptor_sites: Number of acceptor splice sites
        - tp_count: True Positives count
        - fn_count: False Negatives count
        - tn_count: True Negatives count
        - fp_count: False Positives count
    """
    # Add label column by inferring from pred_type if it doesn't exist
    if 'label' not in positions_df.columns:
        # TP and FN are actual splice sites (label=1)
        # TN and FP are not splice sites (label=0)
        positions_df = positions_df.with_columns([
            pl.when(pl.col('pred_type').is_in(['TP', 'FN']))
            .then(1)
            .otherwise(0)
            .alias('label')
        ])
    
    # Get total positions by gene (all categories)
    total_positions = (
        positions_df
        .group_by("gene_id")
        .agg(pl.len().alias("total_positions"))
        .to_pandas()
    )
    
    # Count true splice sites by gene (labeled as real splice sites)
    true_sites = (
        positions_df
        .filter(pl.col("label") == 1)
        .group_by("gene_id")
        .agg(pl.len().alias("true_sites"))
        .to_pandas()
    )
    
    # Count by splice type
    donor_sites = (
        positions_df
        .filter(pl.col("splice_type") == "donor")
        .group_by("gene_id")
        .agg(pl.len().alias("donor_sites"))
        .to_pandas()
    )
    
    acceptor_sites = (
        positions_df
        .filter(pl.col("splice_type") == "acceptor")
        .group_by("gene_id")
        .agg(pl.len().alias("acceptor_sites"))
        .to_pandas()
    )
    
    # Count by prediction type
    tp_counts = (
        positions_df
        .filter(pl.col("pred_type") == "TP")
        .group_by("gene_id")
        .agg(pl.len().alias("tp_count"))
        .to_pandas()
    )
    
    fn_counts = (
        positions_df
        .filter(pl.col("pred_type") == "FN")
        .group_by("gene_id")
        .agg(pl.len().alias("fn_count"))
        .to_pandas()
    )
    
    tn_counts = (
        positions_df
        .filter(pl.col("pred_type") == "TN")
        .group_by("gene_id")
        .agg(pl.len().alias("tn_count"))
        .to_pandas()
    )
    
    fp_counts = (
        positions_df
        .filter(pl.col("pred_type") == "FP")
        .group_by("gene_id")
        .agg(pl.len().alias("fp_count"))
        .to_pandas()
    )
    
    # Merge all dataframes
    result = total_positions
    for df in [true_sites, donor_sites, acceptor_sites, tp_counts, fn_counts, tn_counts, fp_counts]:
        result = result.merge(df, on="gene_id", how="left")
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    # Calculate percentages for key metrics
    result["fn_percentage"] = (result["fn_count"] / result["true_sites"]) * 100
    result["fp_percentage"] = (result["fp_count"] / (result["fp_count"] + result["tn_count"])) * 100
    result["accuracy"] = ((result["tp_count"] + result["tn_count"]) / result["total_positions"]) * 100
    
    return result


def rank_features_by_importance(df, features, target_col, class_values, min_samples=5):
    """
    Calculate feature importance for distinguishing between two classes.
    Importance is measured using effect size (simplified Cohen's d).
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing features and target column
    features : list
        List of feature column names to evaluate
    target_col : str
        Name of the target column with class labels
    class_values : tuple or list
        Tuple or list containing the two class values to compare
    min_samples : int, default=5
        Minimum number of samples required in each class to calculate importance
        
    Returns
    -------
    list
        List of (feature, importance_score) tuples sorted by importance (descending)
    """
    if len(class_values) != 2:
        raise ValueError(f"Need exactly 2 class values, got {len(class_values)}")
        
    # Calculate feature importance scores
    feature_importance = {}
    for feature in features:
        if feature in [target_col]:
            continue
            
        try:
            # Extract data for each class
            class1_data = df[df[target_col] == class_values[0]][feature].dropna()
            class2_data = df[df[target_col] == class_values[1]][feature].dropna()
            
            if len(class1_data) < min_samples or len(class2_data) < min_samples:
                continue
                
            # Calculate effect size (simplified Cohen's d)
            mean1, mean2 = class1_data.mean(), class2_data.mean()
            std1, std2 = class1_data.std(), class2_data.std()
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            
            if pooled_std == 0:  # Avoid division by zero
                effect_size = 0
            else:
                effect_size = abs(mean1 - mean2) / pooled_std
                
            feature_importance[feature] = effect_size
        except Exception as e:
            print(f"Error calculating importance for {feature}: {e}")
    
    # Sort features by importance (descending)
    return sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)


def analyze_feature_distributions(positions_df: pl.DataFrame, output_dir: str, comparison_types=None, top_n_features=5, fast_mode=True, sample_size=1000):
    """
    Analyze and visualize feature distributions for different prediction types.
    Can be used for both FP vs TP comparison and FN vs TN comparison.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame containing all positions with prediction types
    output_dir : str
        Directory to save output visualizations
    comparison_types : tuple, optional
        Types to compare, e.g., ('FP', 'TP') or ('FN', 'TN'). 
        If None, determined automatically based on available data.
    top_n_features : int, default=5
        Number of top features to analyze based on their importance in distinguishing prediction types
    fast_mode : bool, default=True
        Whether to use performance optimizations (simpler plots, sampling)
    sample_size : int, default=1000
        Maximum number of samples to use for visualization when fast_mode=True
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to pandas for visualization
    # Apply sampling in fast mode to reduce data size
    if fast_mode and positions_df.height > sample_size:
        print(f"Sampling data from {positions_df.height} to {sample_size} rows for faster visualization")
        
        # Manual stratified sampling - polars GroupBy doesn't have sample method
        pred_types = positions_df.select("pred_type").unique().to_series().to_list()
        samples_per_type = min(sample_size // len(pred_types), 1000)
        
        # Sample from each prediction type separately
        sampled_dfs = []
        for pred_type in pred_types:
            type_df = positions_df.filter(pl.col("pred_type") == pred_type)
            if type_df.height > samples_per_type:
                # Use fraction-based sampling since polars supports that
                fraction = samples_per_type / type_df.height
                sampled_type = type_df.sample(fraction=fraction, seed=42)
                sampled_dfs.append(sampled_type)
            else:
                # Keep all rows if we have fewer than target
                sampled_dfs.append(type_df)
        
        # Concatenate the sampled dataframes
        if sampled_dfs:
            sampled_df = pl.concat(sampled_dfs)
            pdf = sampled_df.to_pandas()
        else:
            pdf = positions_df.to_pandas()
    else:
        pdf = positions_df.to_pandas()
    
    # Get counts by prediction type
    pred_counts = pdf['pred_type'].value_counts()
    print(f"Prediction type counts:\n{pred_counts}")
    
    # Determine comparison types if not specified
    if comparison_types is None:
        if 'FP' in pred_counts and 'TP' in pred_counts:
            comparison_types = ('FP', 'TP')
            print(f"Comparing False Positives vs True Positives")
        elif 'FN' in pred_counts and 'TN' in pred_counts:
            comparison_types = ('FN', 'TN')
            print(f"Comparing False Negatives vs True Negatives")
        else:
            available_types = list(pred_counts.index)
            if len(available_types) >= 2:
                comparison_types = (available_types[0], available_types[1])
                print(f"Using available types for comparison: {comparison_types}")
            else:
                raise ValueError(f"Insufficient prediction types for comparison: {available_types}")
    
    # Focus on features that might help distinguish between prediction types
    key_features = [
        # Basic probability features (derived from the three probability scores)
        'relative_donor_probability', 'splice_probability', 
        'donor_acceptor_diff', 'splice_neither_diff',
        'donor_acceptor_logodds', 'splice_neither_logodds',
        'probability_entropy',
        
        # Context-agnostic features
        'context_neighbor_mean', 'context_asymmetry', 'context_max',
        
        # Donor features
        'donor_score', 'donor_peak_height_ratio', 'donor_is_local_peak', 
        'donor_signal_strength', 'donor_surge_ratio', 'donor_second_derivative',
        'donor_diff_m1', 'donor_diff_m2', 'donor_diff_p1', 'donor_diff_p2',
        'donor_weighted_context', 'donor_context_diff_ratio',
        
        # Acceptor features
        'acceptor_score', 'acceptor_peak_height_ratio', 'acceptor_is_local_peak', 
        'acceptor_signal_strength', 'acceptor_surge_ratio', 'acceptor_second_derivative',
        'acceptor_diff_m1', 'acceptor_diff_m2', 'acceptor_diff_p1', 'acceptor_diff_p2',
        'acceptor_weighted_context', 'acceptor_context_diff_ratio',
        
        # Cross-type features
        'donor_acceptor_peak_ratio', 'type_signal_difference', 
        'score_difference_ratio', 'signal_strength_ratio'
    ]
    
    # Filter to include only features that exist in the dataframe
    available_features = [feature for feature in key_features if feature in pdf.columns]
    print(f"Found {len(available_features)} of {len(key_features)} analysis features in dataset")
    
    # Filter for numeric features only
    numeric_features = []
    for feature in available_features:
        try:
            # Skip features we know are categorical
            if feature in ['pred_type', 'splice_type', 'donor_is_local_peak', 'acceptor_is_local_peak']:
                numeric_features.append(feature)
                continue
                
            # Try to convert to numeric
            if pd.api.types.is_numeric_dtype(pdf[feature]):
                numeric_features.append(feature)
            elif pdf[feature].nunique() <= 2:  # Binary feature
                numeric_features.append(feature)
            else:
                print(f"Feature {feature} could not be converted to numeric, skipping...")
        except Exception as e:
            print(f"Error processing feature {feature}: {e}")
    
    print(f"Found {len(numeric_features)} numeric features")
    if len(numeric_features) == 0:
        print("No numeric features available for visualization. Skipping feature distribution analysis.")
        return
        
    # Rank features by their importance in distinguishing between prediction types
    if len(numeric_features) > top_n_features:
        print(f"Ranking features by importance for distinguishing {comparison_types[0]} from {comparison_types[1]}...")
        
        # Use our helper function to calculate feature importance
        sorted_features = rank_features_by_importance(
            df=pdf, 
            features=numeric_features,
            target_col='pred_type',
            class_values=comparison_types,
            min_samples=5
        )
        
        # Take top N features
        top_features = [f for f, _ in sorted_features[:top_n_features]]
        
        print(f"Selected top {len(top_features)} features by importance:")
        for i, (feature, score) in enumerate(sorted_features[:top_n_features]):
            print(f"  {i+1}. {feature}: {score:.4f}")
        
        numeric_features = top_features
    
    print(f"Analysis will visualize {len(numeric_features)} features")
    total_features = len(numeric_features)
    
    # Plot distributions of numeric features by prediction type
    for i, feature in enumerate(numeric_features):
        print(f"Processing feature {i+1}/{total_features}: {feature}...")
        try:
            # Filter to just the comparison types for clearer visualization
            comp_data = pdf[pdf['pred_type'].isin(comparison_types)]
            
            if comp_data.empty or len(comp_data['pred_type'].unique()) < 2:
                print(f"Skipping {feature} - insufficient data for comparison")
                continue
                
            # Check if this is a binary feature (only 0/1 or True/False values)
            is_binary = (comp_data[feature].nunique() <= 2) and all(val in [0, 1, True, False] for val in comp_data[feature].unique())
            
            # Skip histogram generation in fast mode for non-top features 
            # (we'll still do boxplots for all features as they're faster)
            if fast_mode and i >= 3:  # Only generate histograms for top 3 features in fast mode
                continue
                
            plt.figure(figsize=(10, 5))  # Smaller figure size for faster rendering
            
            if is_binary:
                # For binary features, use countplot with non-overlapping bars
                ax = sns.countplot(data=comp_data, x=feature, hue='pred_type', alpha=0.9)
                # Add space between groups for clarity
                plt.xticks([0, 1], ['False (0)', 'True (1)'])
                # Add count labels only if we have few enough bars for readability
                if len(comp_data['pred_type'].unique()) <= 3 and not fast_mode:  
                    for p in ax.patches:
                        ax.annotate(f'{p.get_height()}', 
                                  (p.get_x() + p.get_width() / 2., p.get_height()), 
                                  ha='center', va='bottom')
            else:
                # For continuous features, use multiple approach based on feature distribution
                if len(comp_data[feature].unique()) <= 10:  # Discrete with few values
                    # Use dodge to place bars side by side
                    ax = sns.histplot(data=comp_data, x=feature, hue='pred_type', 
                                multiple='dodge', alpha=0.9, discrete=True)
                else:  # Continuous with many values
                    # In fast mode, skip KDE which is computationally expensive
                    ax = sns.histplot(data=comp_data, x=feature, hue='pred_type', 
                                multiple='layer', alpha=0.6, 
                                kde=(not fast_mode),  # Only use KDE in non-fast mode
                                bins=20)  # Limit bins for faster rendering
            
            # Create a consistent and predictable color mapping for prediction types
            # Store this mapping at the function level to ensure consistency between plots
            if not hasattr(analyze_feature_distributions, 'color_mapping'):
                # Define a consistent color mapping based on prediction types
                # We'll always ensure the same prediction type gets the same color
                # Order: TN, FN, TP, FP (or subset of these depending on what's present)
                pred_types = comp_data['pred_type'].unique()
                print(f"Prediction types in data: {pred_types}")
                
                # Create a consistent ordering of prediction types
                ordered_types = []
                for t in ['TN', 'FN', 'TP', 'FP']:
                    if t in pred_types:
                        ordered_types.append(t)
                        
                # Get color cycle
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                
                # Create and store the mapping
                analyze_feature_distributions.color_mapping = {}
                for i, pred_type in enumerate(ordered_types):
                    if i < len(colors):
                        analyze_feature_distributions.color_mapping[pred_type] = colors[i]
                        
                print(f"Created color mapping: {analyze_feature_distributions.color_mapping}")
            
            # Use our consistent color mapping for the legend
            from matplotlib.patches import Patch
            legend_elements = []
            for pred_type, color in analyze_feature_distributions.color_mapping.items():
                if pred_type in comp_data['pred_type'].unique():
                    legend_elements.append(Patch(facecolor=color, label=pred_type))
            
            # Create a manual legend with explicit color-to-prediction-type mapping
            legend = plt.legend(handles=legend_elements, 
                          title='Prediction Type',
                          loc='upper right',
                          frameon=True,
                          framealpha=0.8)
            
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{feature}_{comparison_types[0].lower()}_{comparison_types[1].lower()}_distribution.png"), 
                        dpi=100)  # Lower DPI for faster saving
            plt.close()
        except Exception as e:
            print(f"Error creating histogram for {feature}: {e}")
            plt.close()
    
    # Create boxplots to compare distributions, but only for most important features in fast mode
    if fast_mode:
        # In fast mode, only create boxplots for top features
        boxplot_features = numeric_features[:min(5, len(numeric_features))]
    else:
        boxplot_features = numeric_features
        
    for i, feature in enumerate(boxplot_features):
        print(f"Creating boxplot {i+1}/{len(boxplot_features)}: {feature}...")
        try:
            # Skip categorical features
            if feature in ['pred_type', 'splice_type', 'donor_is_local_peak', 'acceptor_is_local_peak']:
                continue
                
            # Filter to just the comparison types
            comp_data = pdf[pdf['pred_type'].isin(comparison_types)]
            
            if comp_data.empty or len(comp_data['pred_type'].unique()) < 2:
                continue
                
            plt.figure(figsize=(8, 5))  # Smaller figure for faster rendering
            
            # Create boxplot with consistent colors using our stored color mapping
            # This ensures histograms and boxplots use exactly the same colors for each prediction type
            if hasattr(analyze_feature_distributions, 'color_mapping'):
                # Use our custom palette to ensure consistency with histograms
                custom_palette = {pred_type: analyze_feature_distributions.color_mapping[pred_type] 
                                 for pred_type in comp_data['pred_type'].unique() 
                                 if pred_type in analyze_feature_distributions.color_mapping}
                
                ax = sns.boxplot(data=comp_data, x='pred_type', y=feature, 
                             showfliers=not fast_mode,  # Skip outliers in fast mode
                             palette=custom_palette)  # Use our consistent palette
                             
                print(f"Using consistent color mapping for boxplot: {custom_palette}")
            else:
                # Fallback to default if color mapping hasn't been created yet
                ax = sns.boxplot(data=comp_data, x='pred_type', y=feature, 
                             showfliers=not fast_mode)
            
            # Add data points for better distribution visualization (if not in fast mode)
            if not fast_mode:
                sns.stripplot(data=comp_data, x='pred_type', y=feature, 
                           alpha=0.3, jitter=True, size=3)
            
            # Improve title and labels
            plt.title(f'Boxplot of {feature}', fontsize=12, fontweight='bold')
            plt.xlabel('Prediction Type', fontsize=10, fontweight='bold')
            plt.ylabel(feature, fontsize=10)
            
            # Add grid for easier value reading
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{feature}_{comparison_types[0].lower()}_{comparison_types[1].lower()}_boxplot.png"), dpi=100)
            plt.close()
        except Exception as e:
            print(f"Error creating boxplot for {feature}: {e}")
            plt.close()

    # Create feature correlation heatmaps (very expensive operation, so limit heavily in fast mode)
    try:
        # Skip correlation heatmaps entirely if we're in fast mode with a large dataset
        if fast_mode and pdf.shape[0] > 500:
            print("Skipping correlation heatmaps in fast mode for large dataset")
            return
        
        if len(numeric_features) >= 4:  # Only create grid if we have enough numeric features
            # First create overall correlation heatmap
            print("\nCreating overall feature correlation heatmap...")
            
            # Limit feature set even more aggressively in fast mode
            if fast_mode:
                max_features = min(5, len(numeric_features))
                selected_features = numeric_features[:max_features]
                plt.figure(figsize=(10, 8))
                annot = True  # Still use annotations since we have very few features
            else:
                max_features = min(10, len(numeric_features))
                selected_features = numeric_features[:max_features]
                plt.figure(figsize=(12, 10))
                # Only use annotations if we have few enough features
                annot = len(selected_features) <= 8
            
            # Create correlation matrix for these features
            corr_matrix = pdf[selected_features].corr()
            
            # Create the heatmap with simplified parameters in fast mode
            sns.heatmap(corr_matrix, 
                      annot=annot,  # Only annotate if we have few features
                      cmap='coolwarm', 
                      fmt='.1f' if fast_mode else '.2f',  # Less precision in fast mode
                      linewidths=0 if fast_mode else 0.5)  # No grid lines in fast mode
                      
            plt.title('Feature Correlation Overview')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "feature_correlation_heatmap.png"), dpi=100)
            plt.close()
            
            # In fast mode, skip the per-prediction-type heatmaps entirely
            if fast_mode:
                return
                
            # Then create per-prediction-type correlation heatmaps
            for pred_type in comparison_types:
                type_data = pdf[pdf['pred_type'] == pred_type]
                if type_data.empty:
                    print(f"No data for {pred_type}, skipping correlation heatmap")
                    continue
                    
                # Check if we have enough samples for this prediction type
                if type_data.shape[0] < 20:  # Need reasonable number of samples for correlation
                    print(f"Not enough samples for {pred_type} correlation heatmap (only {type_data.shape[0]})")
                    continue
                
                plt.figure(figsize=(12, 10))
                # Create correlation matrix for these features
                try:
                    corr_matrix = type_data[selected_features].corr()
                    sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm', fmt='.2f', linewidths=0.5)
                    plt.title(f'Feature Correlation for {pred_type} Predictions')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"feature_correlation_heatmap_{pred_type.lower()}.png"), dpi=100)
                except Exception as e:
                    print(f"Error creating correlation heatmap for {pred_type}: {e}")
                finally:
                    plt.close()
    except Exception as e:
        print(f"Error creating feature correlation heatmaps: {e}")
        plt.close()


def check_genomic_files_exist(mode='gene', seq_type='full'):
    """
    Check if the main genomic data files already exist to avoid extraction.
    
    Parameters
    ----------
    mode : str, default='gene'
        Mode for sequence extraction ('gene' or 'transcript')
    seq_type : str, default='full'
        Type of gene sequences to extract ('full' or 'minmax')
    
    Returns
    -------
    dict
        Dictionary with results of file existence checks
    """
    # Basic files that are always single files
    files_to_check = {
        "annotations": os.path.join(Config.PROJ_DIR, "data", "ensembl", "annotations_all_transcripts.tsv"),
        "splice_sites": os.path.join(Config.PROJ_DIR, "data", "ensembl", "splice_sites.tsv")
    }
    
    # For genomic sequences, we need to check the specific pattern based on mode and seq_type
    # Location where sequence files would be stored - use the global ensembl directory
    seq_dir = os.path.join(Config.PROJ_DIR, "data", "ensembl")
    assert os.path.exists(seq_dir), f"Sequence directory does not exist: {seq_dir}"
    
    # Standard chromosomes to check
    standard_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y']
    
    # Formats we might encounter
    formats = ['parquet', 'tsv', 'csv', 'pkl']
    
    # Define the three sequence patterns to check
    sequence_patterns = {
        "gene_regular": "gene_sequence_{chrom}.{fmt}",      # Regular gene sequences
        "gene_minmax": "gene_sequence_minmax_{chrom}.{fmt}", # Minmax gene sequences
        "transcript": "tx_sequence_{chrom}.{fmt}"           # Transcript sequences
    }
    
    # Select the appropriate pattern based on mode and seq_type
    if mode == 'transcript':
        # Transcript mode only uses transcript pattern
        patterns_to_check = {"transcript": sequence_patterns["transcript"]}
    else:  # mode == 'gene'
        if seq_type == 'minmax':
            # Gene mode with minmax type uses minmax pattern
            patterns_to_check = {"gene_minmax": sequence_patterns["gene_minmax"]}
        else:  # seq_type == 'full'
            # Gene mode with full type uses regular pattern
            patterns_to_check = {"gene_regular": sequence_patterns["gene_regular"]}
    
    # Check each selected pattern
    pattern_results = {}
    for pattern_name, pattern_template in patterns_to_check.items():
        pattern_results[pattern_name] = {}
        for fmt in formats:
            # Check if all chromosome files exist for this pattern and format
            all_exist = True
            file_paths = []
            
            for chrom in standard_chroms:
                file_name = pattern_template.format(chrom=chrom, fmt=fmt)
                file_path = os.path.join(seq_dir, file_name)
                file_paths.append(file_path)
                # if fmt == "parquet":
                #     print("[debug] checking", file_path)
                
                if not os.path.exists(file_path):
                    all_exist = False
                    break
            
            pattern_results[pattern_name][fmt] = {
                "complete": all_exist,
                "paths": file_paths if all_exist else []
            }
    
    # Determine if any sequence pattern is complete
    any_complete = False
    complete_info = {}
    
    for pattern_name, formats_data in pattern_results.items():
        for fmt, data in formats_data.items():
            if data["complete"]:
                any_complete = True
                if pattern_name not in complete_info:
                    complete_info[pattern_name] = []
                complete_info[pattern_name].append(fmt)
    
    # Set the main genomic_sequences result based on whether any pattern is complete
    files_to_check["genomic_sequences"] = any_complete
    
    # Build result dictionary
    results = {}
    for name, exists in files_to_check.items():
        results[name] = exists
    
    # Add detailed info about sequence patterns
    results["genomic_sequences_info"] = {
        "mode": mode,
        "seq_type": seq_type,
        "any_pattern_complete": any_complete,
        "complete_patterns": complete_info,
        "pattern_details": pattern_results
    }
        
    return results
