"""
Visualization Module for Error Analysis.

This module provides visualization functions for error analysis results, including
feature distributions, SHAP plots, and feature importance comparisons.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import re
from ..utils_doc import (
    print_emphasized, 
    print_with_indent
)

# Use the plot_feature_importance function instead of the basic barplot
from ..analysis_utils import plot_feature_importance

# Set default plot style
plt.style.use('seaborn-v0_8-whitegrid')

def get_sequence_feature_indices(feature_names):
    """
    Get indices of sequence-based features (k-mers) from feature names.
    
    Parameters
    ----------
    feature_names : list
        List of feature names
        
    Returns
    -------
    list
        Indices of sequence features
    """
    indices = []
    for i, feature in enumerate(feature_names):
        # Typical k-mer feature names: kmer_pos0_A, kmer_pos1_C, etc.
        if re.match(r'kmer_pos\d+_[ACGT]+', feature):
            indices.append(i)
    return indices

def get_genomic_feature_indices(feature_names):
    """
    Get indices of genomic features (non-k-mers) from feature names.
    
    Parameters
    ----------
    feature_names : list
        List of feature names
        
    Returns
    -------
    list
        Indices of genomic features
    """
    sequence_indices = get_sequence_feature_indices(feature_names)
    return [i for i in range(len(feature_names)) if i not in sequence_indices]

def save_figure(fig, output_path, dpi=300, bbox_inches='tight'):
    """
    Save a matplotlib figure to file.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    output_path : str
        Path to save the figure
    dpi : int, default=300
        DPI for the saved figure
    bbox_inches : str, default='tight'
        Bounding box in inches
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save figure
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)
    return output_path

def plot_feature_distributions(
    X_test,
    y_test, 
    features=None, 
    output_dir=None, 
    output_file=None,
    subject="feature",
    suffix="",
    max_features=30,  
    verbose=1
):
    """
    Plot distributions of features stratified by class label.
    
    Parameters
    ----------
    X_test : pandas.DataFrame
        Feature matrix
    y_test : array-like
        Target labels
    features : list, optional
        List of features to plot. If None, use all features
    output_dir : str, optional
        Output directory for saving plots
    output_file : str, optional
        Output filename for saving plots
    subject : str, default="feature"
        Subject name for output files
    suffix : str, default=""
        Suffix for output files
    max_features : int, default=30
        Maximum number of features to include in the visualization.
        If there are more features than this, select the top ones based on 
        mean difference between classes.
    verbose : int, default=1
        Verbosity level
    """
    if features is None:
        features = X_test.columns.tolist()
    
    # If there are too many features, select the top ones based on mean difference
    if len(features) > max_features:
        if verbose > 0:
            print(f"Too many features ({len(features)}) for visualization. Selecting top {max_features} based on mean difference.")
        
        # Calculate mean difference for each feature
        mean_diffs = {}
        for feature in features:
            mean_pos = X_test.loc[y_test == 1, feature].mean()
            mean_neg = X_test.loc[y_test == 0, feature].mean()
            mean_diffs[feature] = abs(mean_pos - mean_neg)
        
        # Select top features
        sorted_features = sorted(mean_diffs.items(), key=lambda x: x[1], reverse=True)
        features = [f[0] for f in sorted_features[:max_features]]
        
        if verbose > 0:
            print(f"Selected top {len(features)} features based on mean difference between classes.")
    
    if output_file is None:
        output_file = f"{subject}-feature-distributions{suffix}.pdf"
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
    else:
        output_path = output_file
    
    # Create a figure with subplots for each feature
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    
    # Flatten axes array for easier indexing
    if n_rows > 1:
        axes = axes.flatten()
    elif n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each feature
    for i, feature in enumerate(features):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get feature data for each class
        data_pos = X_test.loc[y_test == 1, feature]
        data_neg = X_test.loc[y_test == 0, feature]
        
        # Create density plots
        sns.kdeplot(data_pos, ax=ax, label='Positive (1)', color='red', fill=True, alpha=0.3)
        sns.kdeplot(data_neg, ax=ax, label='Negative (0)', color='blue', fill=True, alpha=0.3)
        
        # Add vertical lines for means
        ax.axvline(data_pos.mean(), color='red', linestyle='--', alpha=0.7)
        ax.axvline(data_neg.mean(), color='blue', linestyle='--', alpha=0.7)
        
        ax.set_title(feature)
        ax.legend()
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    save_figure(fig, output_path)
    
    if verbose > 0:
        print(f"Feature distributions saved to: {output_path}")
    
    return output_path

def shap_analysis_with_local_agg_plots(
    model,
    X_test,
    y_test,
    output_dir,
    class_labels=None,    # Dictionary mapping class names to numeric values
    error_class="Error",  # Error class name (e.g., "FP", "FN")
    correct_class="Correct", # Correct class name (e.g., "TP", "TN")
    local_top_k=10,
    global_top_k=20,
    plot_top_k=None,
    subject="shap-analysis",
    suffix="",
    colormap_global="viridis",       # Colormap for global importance plots
    colormap_motif="plasma",         # Colormap for motif feature plots
    colormap_nonmotif="viridis",     # Colormap for non-motif feature plots
    colormap_comparison="Set1",      # Colormap for class comparison plots
    verbose=1,
    return_all=False
):
    """
    Perform SHAP analysis, produce standard SHAP plots, and also build a 
    local top-K aggregator that compares error vs. correct samples.

    Parameters
    ----------
    model : trained model (e.g. XGBoost)
        The trained model to explain
    X_test : pd.DataFrame
        Test features (shape: [n_samples, n_features])
    y_test : array-like
        Test labels (shape: [n_samples])
    output_dir : str
        Directory to save output files
    class_labels : dict, default=None
        Dictionary mapping class names to numeric label values
        If None, defaults to {error_class: 1, correct_class: 0}
    error_class : str, default="Error"
        Name of the error class (e.g., "FP", "FN")
    correct_class : str, default="Correct"
        Name of the correct class (e.g., "TP", "TN")
    local_top_k : int, default=10
        Number of top features to consider per sample
    global_top_k : int, default=20
        Number of top features to consider globally
    plot_top_k : int, default=None
        Number of top features to include in plots. If None, use global_top_k.
    subject : str, default="shap-analysis"
        Subject name for output files
    suffix : str, default=""
        Suffix for output files
    colormap_global : str, default="viridis"
        Matplotlib colormap name for global feature importance plots
    colormap_motif : str, default="plasma"
        Matplotlib colormap name for motif feature importance plots
    colormap_nonmotif : str, default="viridis"
        Matplotlib colormap name for non-motif feature importance plots
    colormap_comparison : str, default="Set1"
        Matplotlib colormap name for class comparison plots
    verbose : int, default=1
        Verbosity level
    return_all : bool, default=False
        Whether to return all results

    Returns
    -------
    dict or None
        Dictionary with results if return_all=True, otherwise None
    """
    import matplotlib.cm as cm

    os.makedirs(output_dir, exist_ok=True)
    
    # Create default class labels dictionary if not provided
    if class_labels is None:
        class_labels = {
            error_class: 1,
            correct_class: 0
        }
    
    # Get numeric labels for error and correct classes
    error_label = class_labels.get(error_class, 1)   # Default to 1 if not found
    correct_label = class_labels.get(correct_class, 0)   # Default to 0 if not found
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values - use the newer Explanation object API
    shap_values = explainer(X_test)
    
    # Get feature names
    feature_names = X_test.columns.tolist()
    
    # Create meta results for feature importance
    meta_results = {}

    if plot_top_k is None:
        plot_top_k = global_top_k
    else:
        # plot_top_k should be less than or equal to global_top_k
        # if plot_top_k > global_top_k:
        #     # raise ValueError("plot_top_k should be smaller than global_top_k")
        #     print(f"Warning: plot_top_k ({plot_top_k}) is less than or equal to global_top_k ({global_top_k}). Using global_top_k instead.")
        #     plot_top_k = global_top_k
        pass
    
    # 1. Standard SHAP summary plot (bar)
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=plot_top_k, show=False)
    plt.title(f"SHAP Feature Importance (Top {plot_top_k})")
    plt.tight_layout()
    output_file = f"{subject}-shap_summary_bar-meta.pdf"
    output_path = os.path.join(output_dir, output_file)
    save_figure(plt.gcf(), output_path)
    
    # 2. Standard SHAP summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=plot_top_k, show=False)
    plt.title(f"SHAP Summary Plot (Top {plot_top_k})")
    plt.tight_layout()
    output_file = f"{subject}-shap_beeswarm-meta.pdf"
    output_path = os.path.join(output_dir, output_file)
    save_figure(plt.gcf(), output_path)
    
    # 3. SHAP summary with margin - use a simpler bar plot as fallback
    try:
        plt.figure(figsize=(12, 10))
        shap.plots.bar(shap_values, max_display=plot_top_k, show=False)
        plt.title(f"SHAP Feature Importance with Variance (Top {plot_top_k})")
        plt.tight_layout()
        output_file = f"{subject}-shap_summary_with_margin.pdf"
        output_path = os.path.join(output_dir, output_file)
        save_figure(plt.gcf(), output_path)
    except Exception as e:
        if verbose > 0:
            print(f"Warning: Could not create SHAP summary with margin: {e}")
        # Create a simpler bar plot as fallback
        plt.figure(figsize=(12, 10))
        shap.plots.bar(shap_values, max_display=plot_top_k, show=False)
        plt.title(f"SHAP Feature Importance (Top {plot_top_k})")
        plt.tight_layout()
        output_file = f"{subject}-shap_summary_with_margin.pdf"
        output_path = os.path.join(output_dir, output_file)
        save_figure(plt.gcf(), output_path)
    
    # 4. Calculate global feature importance based on SHAP values
    # Convert the Explanation object back to numpy arrays for further processing
    shap_values_array = shap_values.values
    
    # Calculate global importance
    importances = np.abs(shap_values_array).mean(axis=0)
    global_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    output_file = f"{subject}-global_shap_importance-meta.csv"
    output_path = os.path.join(output_dir, output_file)
    global_importance.to_csv(output_path, index=False)
    meta_results['global_importance'] = global_importance
    
    # 5. Plot global importance
    # Convert 'importance' column to 'importance_score' for compatibility with plot_feature_importance
    global_importance_for_plot = global_importance.rename(columns={'importance': 'importance_score'})
    
    output_file = f"{subject}-global_importance-barplot.pdf"
    output_path = os.path.join(output_dir, output_file)
    
    plot_feature_importance(
        df=global_importance_for_plot,
        title=f"Global SHAP Feature Importance (Top {plot_top_k})",
        output_path=output_path,
        top_k=plot_top_k,
        colormap=colormap_global,  # Use the colormap_global parameter
        use_continuous_color=True,  # Enable continuous color for smooth gradient based on importance scores
        figure_size=(12, 8),
        verbose=verbose
    )
    
    # 6. Local feature importance analysis - separate by class
    X_error = X_test[y_test == error_label]
    X_correct = X_test[y_test == correct_label]
    
    # Generate explanations for each subset
    shap_explanation_error = explainer(X_error)
    shap_explanation_correct = explainer(X_correct)
    
    # Convert to numpy arrays for processing
    shap_values_error = shap_explanation_error.values
    shap_values_correct = shap_explanation_correct.values
    
    # 7. Get top features per sample and count frequencies
    error_top_features = []
    correct_top_features = []
    
    # For each error sample, get top features
    for i in range(len(X_error)):
        sample_values = np.abs(shap_values_error[i])
        top_indices = np.argsort(sample_values)[-local_top_k:]
        for idx in top_indices:
            feature = feature_names[idx]
            error_top_features.append(feature)
    
    # For each correct sample, get top features
    for i in range(len(X_correct)):
        sample_values = np.abs(shap_values_correct[i])
        top_indices = np.argsort(sample_values)[-local_top_k:]
        for idx in top_indices:
            feature = feature_names[idx]
            correct_top_features.append(feature)
    
    # Count frequencies
    error_freq = pd.Series(error_top_features).value_counts().reset_index()
    error_freq.columns = ['feature', 'frequency']
    error_freq['normalized'] = error_freq['frequency'] / len(X_error)
    
    correct_freq = pd.Series(correct_top_features).value_counts().reset_index()
    correct_freq.columns = ['feature', 'frequency']
    correct_freq['normalized'] = correct_freq['frequency'] / len(X_correct)
    
    # Merge to compare
    freq_compare = pd.merge(
        error_freq, correct_freq,
        on='feature', how='outer',
        suffixes=('_error', '_correct')
    ).fillna(0)
    
    # Calculate difference in normalized frequency
    freq_compare['diff'] = freq_compare['normalized_error'] - freq_compare['normalized_correct']
    freq_compare['abs_diff'] = np.abs(freq_compare['diff'])
    
    # Sort by absolute difference
    freq_compare = freq_compare.sort_values('abs_diff', ascending=False)
    
    # Save to file
    output_file = f"{subject}-local_top{local_top_k}_freq-meta.csv"
    output_path = os.path.join(output_dir, output_file)

    # Save the top global_top_k features
    freq_compare.head(global_top_k).to_csv(output_path, index=False)
    
    # 8. Plot frequency comparison
    plt.figure(figsize=(14, 10))
    top_features = freq_compare.head(min(global_top_k, plot_top_k))['feature'].tolist()
    
    # Create plot data
    plot_data = []
    for feature in top_features:
        row = freq_compare[freq_compare['feature'] == feature].iloc[0]
        plot_data.append({
            'feature': feature,
            'frequency': row['normalized_error'],
            'class': error_class
        })
        plot_data.append({
            'feature': feature,
            'frequency': row['normalized_correct'],
            'class': correct_class
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create grouped bar chart with selected colormap for class comparison
    plt.figure(figsize=(14, 10))
    comparison_palette = sns.color_palette(colormap_comparison, n_colors=2)
    bar_plot = sns.barplot(x='feature', y='frequency', hue='class', data=plot_df, palette=comparison_palette)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Local Feature Importance Frequency Comparison (Top {plot_top_k})")
    plt.tight_layout()
    
    output_file = f"{subject}-local-shap-frequency-comparison-meta.pdf"
    output_path = os.path.join(output_dir, output_file)
    save_figure(plt.gcf(), output_path)
    
    # 9. Separate motif and non-motif features
    try:
        motif_indices = get_sequence_feature_indices(feature_names)
        genomic_indices = get_genomic_feature_indices(feature_names)
        
        # Get motif features
        motif_features = [feature_names[i] for i in motif_indices]
        motif_importance = global_importance[global_importance['feature'].isin(motif_features)]
        
        # Get non-motif features
        non_motif_features = [f for f in feature_names if f not in motif_features]
        non_motif_importance = global_importance[global_importance['feature'].isin(non_motif_features)]
        
        # Save to files
        output_file = f"{subject}-xgboost-motif-importance-shap-meta.tsv"
        output_path = os.path.join(output_dir, output_file)
        motif_importance.head(plot_top_k).to_csv(output_path, sep='\t', index=False)
        
        output_file = f"{subject}-xgboost-motif-importance-shap-full-meta.tsv"
        output_path = os.path.join(output_dir, output_file)
        motif_importance.to_csv(output_path, sep='\t', index=False)
        
        # Plot motif importance with continuous color mapping
        plt.figure(figsize=(12, 8))
        motif_plot = motif_importance.head(plot_top_k).copy().sort_values('importance', ascending=True)
        
        # Use the updated API for accessing colormaps
        cmap = plt.colormaps[colormap_motif]  # Use the user-specified colormap for motifs
        
        # Normalize importance values to [0,1] for color mapping
        min_val, max_val = motif_plot['importance'].min(), motif_plot['importance'].max()
        norm_values = (motif_plot['importance'] - min_val) / (max_val - min_val) if max_val > min_val else [0.5] * len(motif_plot)
        
        # Create bar colors based on normalized importance
        colors = [cmap(x) for x in norm_values]
        
        # Create the plot with custom colors
        bars = plt.barh(range(len(motif_plot)), motif_plot['importance'], color=colors)
        plt.yticks(range(len(motif_plot)), motif_plot['feature'])
        plt.title(f"Motif Feature Importance (Top {plot_top_k})")
        plt.xlabel('SHAP Importance')
        plt.ylabel('Feature')
        
        # Add a colorbar to show the mapping
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Importance Score')
        
        plt.tight_layout()
        
        output_file = f"{subject}-motif_importance-barplot-meta.pdf"
        output_path = os.path.join(output_dir, output_file)
        save_figure(plt.gcf(), output_path)
        
        # Plot non-motif importance with continuous color mapping
        plt.figure(figsize=(12, 8))
        non_motif_plot = non_motif_importance.head(plot_top_k).copy().sort_values('importance', ascending=True)
        
        # Use the updated API for accessing colormaps
        cmap = plt.colormaps[colormap_nonmotif]  # Use the user-specified colormap for non-motifs
        
        # Normalize importance values to [0,1] for color mapping
        min_val, max_val = non_motif_plot['importance'].min(), non_motif_plot['importance'].max()
        norm_values = (non_motif_plot['importance'] - min_val) / (max_val - min_val) if max_val > min_val else [0.5] * len(non_motif_plot)
        
        # Create bar colors based on normalized importance
        colors = [cmap(x) for x in norm_values]
        
        # Create the plot with custom colors
        bars = plt.barh(range(len(non_motif_plot)), non_motif_plot['importance'], color=colors)
        plt.yticks(range(len(non_motif_plot)), non_motif_plot['feature'])
        plt.title(f"Non-Motif Feature Importance (Top {plot_top_k})")
        plt.xlabel('SHAP Importance')
        plt.ylabel('Feature')
        
        # Add a colorbar to show the mapping
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Importance Score')
        
        plt.tight_layout()
        
        output_file = f"{subject}-nonmotif_importance-barplot-meta.pdf"
        output_path = os.path.join(output_dir, output_file)
        save_figure(plt.gcf(), output_path)
    except Exception as e:
        if verbose > 0:
            print(f"Warning: Could not separate motif and non-motif features: {e}")
    
    if return_all:
        return {
            'shap_values': shap_values,
            'global_importance': global_importance,
            'freq_compare': freq_compare
        }
    return None

def bar_chart_local_feature_importance(
    model, 
    X_test, 
    output_dir, 
    top_k=20, 
    subject="local-importance",
    suffix="",
    verbose=1,
    n_samples=5,
    sample_selection="random",
    custom_indices=None,
    y_test=None
):
    """
    Create bar charts of feature importance for specific samples.
    
    Parameters
    ----------
    model : trained model (e.g. XGBoost)
        The trained model to explain
    X_test : pd.DataFrame
        Test features
    output_dir : str
        Directory to save output files
    top_k : int, default=20
        Number of top features to display
    subject : str, default="local-importance"
        Subject name for output files
    suffix : str, default=""
        Suffix for output files
    verbose : int, default=1
        Verbosity level
    n_samples : int, default=5
        Number of samples to select for local explanation
    sample_selection : str, default="random"
        Method to select samples for visualization:
        - "random": Select random samples
        - "high_confidence": Select samples with highest prediction confidence
        - "low_confidence": Select samples with lowest prediction confidence
        - "border": Select samples near the decision boundary
        - "misclassified": Select samples that were misclassified
        - "custom": Use the custom_indices parameter
    custom_indices : list, default=None
        Custom list of sample indices to visualize when sample_selection="custom"
    y_test : array-like, default=None
        True labels for test data, required for some selection methods
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Get feature names
    feature_names = X_test.columns.tolist()
    
    # Select samples for local explanation based on selection method
    sample_indices = []
    
    if sample_selection == "custom" and custom_indices is not None:
        # Use custom indices provided by the user
        sample_indices = [idx for idx in custom_indices if idx < len(X_test)]
        if verbose > 0 and len(sample_indices) != len(custom_indices):
            print(f"Warning: Some custom indices were out of range and were ignored")
    else:
        if sample_selection == "random":
            # Random selection
            n_samples_to_select = min(n_samples, len(X_test))
            sample_indices = np.random.choice(len(X_test), n_samples_to_select, replace=False)
            
        elif sample_selection in ["high_confidence", "low_confidence", "border", "misclassified"]:
            # These methods need predictions
            if hasattr(model, "predict_proba"):
                try:
                    # Get prediction probabilities
                    y_proba = model.predict_proba(X_test)
                    
                    # For binary classification, get probability of positive class
                    if y_proba.shape[1] == 2:
                        confidence_scores = y_proba[:, 1]
                    else:
                        # For multiclass, use max probability as confidence
                        confidence_scores = np.max(y_proba, axis=1)
                    
                    if sample_selection == "high_confidence":
                        # Select highest confidence predictions
                        sample_indices = np.argsort(confidence_scores)[-n_samples:]
                        
                    elif sample_selection == "low_confidence":
                        # Select lowest confidence predictions
                        sample_indices = np.argsort(confidence_scores)[:n_samples]
                        
                    elif sample_selection == "border":
                        # For binary, samples closest to 0.5
                        # For multiclass, samples with smallest gap between top two classes
                        if y_proba.shape[1] == 2:
                            border_scores = np.abs(confidence_scores - 0.5)
                            sample_indices = np.argsort(border_scores)[:n_samples]
                        else:
                            # Get gap between top two prediction probabilities
                            sorted_probs = np.sort(y_proba, axis=1)
                            gaps = sorted_probs[:, -1] - sorted_probs[:, -2]  # Gap between top two
                            sample_indices = np.argsort(gaps)[:n_samples]
                            
                    elif sample_selection == "misclassified" and y_test is not None:
                        # Get predicted labels
                        y_pred = model.predict(X_test)
                        # Find misclassified samples
                        misclassified = np.where(y_pred != y_test)[0]
                        if len(misclassified) > 0:
                            # Select up to n_samples misclassified samples
                            n_samples_to_select = min(n_samples, len(misclassified))
                            sample_indices = np.random.choice(misclassified, n_samples_to_select, replace=False)
                        else:
                            if verbose > 0:
                                print("No misclassified samples found, using random selection instead")
                            n_samples_to_select = min(n_samples, len(X_test))
                            sample_indices = np.random.choice(len(X_test), n_samples_to_select, replace=False)
                
                except Exception as e:
                    if verbose > 0:
                        print(f"Error calculating prediction probabilities: {e}, using random selection instead")
                    n_samples_to_select = min(n_samples, len(X_test))
                    sample_indices = np.random.choice(len(X_test), n_samples_to_select, replace=False)
            else:
                if verbose > 0:
                    print(f"Model does not support predict_proba, using random selection instead")
                n_samples_to_select = min(n_samples, len(X_test))
                sample_indices = np.random.choice(len(X_test), n_samples_to_select, replace=False)
        else:
            # Default to random if invalid selection method
            if verbose > 0:
                print(f"Unknown sample selection method '{sample_selection}', using random selection instead")
            n_samples_to_select = min(n_samples, len(X_test))
            sample_indices = np.random.choice(len(X_test), n_samples_to_select, replace=False)
    
    # Limit the number of samples to visualize
    if len(sample_indices) > n_samples:
        sample_indices = sample_indices[:n_samples]
    
    # Create local feature importance plots for selected samples
    for i, idx in enumerate(sample_indices):
        # Get the sample
        sample = X_test.iloc[idx:idx+1]
        
        # Calculate SHAP values for this sample using the new SHAP API
        sample_explanation = explainer(sample)
        
        # Create DataFrame with feature names and SHAP values
        shap_df = pd.DataFrame({
            'feature': feature_names,
            'importance': sample_explanation.values[0]
        })
        
        # Sort by absolute importance
        shap_df['abs_importance'] = np.abs(shap_df['importance'])
        shap_df = shap_df.sort_values('abs_importance', ascending=False).head(top_k)
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        shap_df['color_group'] = ['negative' if x < 0 else 'positive' for x in shap_df['importance']]
        color_palette = {'negative': 'red', 'positive': 'blue'}
        sns.barplot(x='importance', y='feature', hue='color_group', palette=color_palette, data=shap_df, legend=False)
        plt.title(f"Local Feature Importance for Sample {idx}")
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_file = f"{subject}-sample{idx}{suffix}.pdf"
        output_path = os.path.join(output_dir, output_file)
        save_figure(plt.gcf(), output_path)
        
        if verbose > 0:
            print(f"Local importance plot for sample {idx} saved to: {output_path}")

def demo_local_feature_importance_strategies(
    model,
    X_test,
    y_test,
    output_dir,
    subject="feature-importance-demo",
    top_k=15,
    verbose=1
):
    """
    Demonstration function showcasing different sample selection strategies
    for local feature importance visualization.
    
    This demo creates multiple sets of visualizations using different selection methods
    to help users understand each strategy's purposes and outputs.
    
    Parameters
    ----------
    model : trained model (e.g. XGBoost)
        The trained model to explain
    X_test : pd.DataFrame
        Test features
    y_test : array-like
        Test labels
    output_dir : str
        Directory to save output files
    subject : str, default="feature-importance-demo"
        Subject name for output files
    top_k : int, default=15
        Number of top features to display
    verbose : int, default=1
        Verbosity level
    
    Returns
    -------
    None
        Function creates visualizations in the output directory
    """
    # Create a dedicated directory for demo outputs
    demo_dir = os.path.join(output_dir, f"{subject}-strategies")
    os.makedirs(demo_dir, exist_ok=True)
    
    if verbose > 0:
        print_emphasized("DEMO: Local Feature Importance Selection Strategies")
        print("This demo creates visualizations using different sample selection methods.")
        print(f"All outputs will be saved to: {demo_dir}")
    
    # 1. Random selection (baseline)
    if verbose > 0:
        print("\n1. Random Selection: Randomly selected samples")
    bar_chart_local_feature_importance(
        model=model,
        X_test=X_test,
        output_dir=demo_dir,
        top_k=top_k,
        subject=f"{subject}-random",
        verbose=verbose,
        n_samples=3,
        sample_selection="random"
    )
    
    # 2. High confidence predictions
    if verbose > 0:
        print("\n2. High Confidence: Samples where model is most confident")
    bar_chart_local_feature_importance(
        model=model,
        X_test=X_test,
        output_dir=demo_dir,
        top_k=top_k,
        subject=f"{subject}-high-confidence",
        verbose=verbose,
        n_samples=3,
        sample_selection="high_confidence"
    )
    
    # 3. Low confidence predictions
    if verbose > 0:
        print("\n3. Low Confidence: Samples where model is least confident")
    bar_chart_local_feature_importance(
        model=model,
        X_test=X_test,
        output_dir=demo_dir,
        top_k=top_k,
        subject=f"{subject}-low-confidence",
        verbose=verbose,
        n_samples=3,
        sample_selection="low_confidence"
    )
    
    # 4. Decision boundary cases
    if verbose > 0:
        print("\n4. Decision Boundary: Samples near the classification boundary")
    bar_chart_local_feature_importance(
        model=model,
        X_test=X_test,
        output_dir=demo_dir,
        top_k=top_k,
        subject=f"{subject}-boundary",
        verbose=verbose,
        n_samples=3,
        sample_selection="border"
    )
    
    # 5. Misclassified samples (most useful for error analysis)
    if verbose > 0:
        print("\n5. Misclassified: Samples the model predicted incorrectly")
    bar_chart_local_feature_importance(
        model=model,
        X_test=X_test,
        output_dir=demo_dir,
        top_k=top_k,
        subject=f"{subject}-misclassified",
        verbose=verbose,
        n_samples=3,
        sample_selection="misclassified",
        y_test=y_test
    )
    
    # 6. Custom indices example (here we just use a few specific indices)
    if verbose > 0:
        print("\n6. Custom Selection: Visualizing specific samples by index")
    # Select a few indices (for demo, we'll use first few samples)
    custom_indices = [0, 1, 2]  # In a real scenario, you would pick specific meaningful samples
    bar_chart_local_feature_importance(
        model=model,
        X_test=X_test,
        output_dir=demo_dir,
        top_k=top_k,
        subject=f"{subject}-custom",
        verbose=verbose,
        sample_selection="custom",
        custom_indices=custom_indices
    )
    
    if verbose > 0:
        print_emphasized("\nDemo Complete!")
        print(f"All visualization strategies have been created in: {demo_dir}")
        print("Compare these visualizations to understand the different selection strategies")
        print("and choose the most appropriate one for your analysis needs.")

def plot_feature_rankings_comparison(
    model,
    X_test,
    output_dir=None,
    output_file=None,
    feature_list=None,
    top_k=15,
    figsize=(12, 8),
    save=True,
    verbose=1
):
    """
    Plot a comparison of feature importance rankings from different methods,
    including model's feature_importances_ and SHAP values.
    
    Parameters
    ----------
    model : trained model (e.g. XGBoost)
        The trained model with feature_importances_ attribute
    X_test : pd.DataFrame
        Test features
    output_dir : str, default=None
        Directory to save output files
    output_file : str, default=None
        Output file name
    feature_list : list, default=None
        List of features to include in the plot
    top_k : int, default=15
        Number of top features to display
    figsize : tuple, default=(12, 8)
        Figure size
    save : bool, default=True
        Whether to save the plot
    verbose : int, default=1
        Verbosity level
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature importance rankings from different methods
    """
    # If output_dir provided, make sure it exists
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if output_file is None:
            output_file = "feature_importance_comparison.pdf"
        output_path = os.path.join(output_dir, output_file)
    
    # Get feature names
    feature_names = X_test.columns.tolist()
    if feature_list is not None:
        # Filter features
        feature_indices = [feature_names.index(f) for f in feature_list if f in feature_names]
        feature_names = [feature_names[i] for i in feature_indices]
        X_test = X_test[feature_names]
    
    # Initialize results dictionary
    rankings = {}
    
    # 1. Model's built-in feature importance
    if hasattr(model, 'feature_importances_'):
        model_importances = model.feature_importances_
        if feature_list is not None:
            model_importances = [model_importances[i] for i in feature_indices]
        
        # Convert to ranking
        model_ranking = pd.Series(model_importances, index=feature_names)
        model_ranking = model_ranking.sort_values(ascending=False)
        rankings['Model'] = model_ranking.index.tolist()
    
    # 2. SHAP feature importance - updated for the new SHAP API
    try:
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values with the new API
        shap_explanation = explainer(X_test)
        
        # Get SHAP values as numpy array
        shap_values = shap_explanation.values
        
        # Calculate mean absolute SHAP values for each feature
        shap_importances = np.abs(shap_values).mean(axis=0)
        
        # Convert to ranking
        shap_ranking = pd.Series(shap_importances, index=feature_names)
        shap_ranking = shap_ranking.sort_values(ascending=False)
        rankings['SHAP'] = shap_ranking.index.tolist()
    except Exception as e:
        if verbose > 0:
            print(f"Warning: Could not calculate SHAP feature importance: {e}")
    
    # Create summary DataFrame
    summary = pd.DataFrame(rankings)
    
    # Limit to top_k features
    summary = summary.iloc[:min(top_k, len(summary))]
    
    # Plot
    plt.figure(figsize=figsize)
    
    # Create positions for bars
    n_methods = len(rankings)
    n_features = len(summary)
    positions = np.arange(n_features)
    width = 0.8 / n_methods
    
    # Plot bars for each method
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (method, ranking) in enumerate(rankings.items()):
        # Map features to their positions
        y_pos = np.array([positions[summary.index[summary[method] == feat].tolist()[0]] 
                          if feat in summary[method].values else -1 
                          for feat in ranking[:top_k]])
        y_pos = y_pos[y_pos >= 0]  # Filter out features not in top_k
        
        # Plot
        plt.barh(positions + (i - n_methods/2 + 0.5) * width, 
                 np.arange(n_features, 0, -1)[:len(y_pos)], 
                 width, 
                 color=colors[i % len(colors)],
                 alpha=0.6,
                 label=method)
    
    # Add feature names
    plt.yticks(positions, summary.iloc[:, 0].values)
    plt.xlabel('Rank (lower is more important)')
    plt.ylabel('Feature')
    plt.title('Feature Importance Rankings Comparison')
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    if save and output_path is not None:
        save_figure(plt.gcf(), output_path)
        if verbose > 0:
            print(f"Feature importance comparison saved to: {output_path}")
    
    return summary
