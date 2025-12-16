#!/usr/bin/env python3
"""
Cross-Validation Evaluation Utilities for MetaSpliceAI Meta-Models

This module provides functions for:
1. Per-fold performance metrics calculation and storage
2. Cross-validation aggregation with statistical measures (mean, std, confidence intervals)
3. Calculation of relative error reduction metrics
4. Visualization of performance across folds
"""
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats

# Module version
__version__ = "0.1.0"

# Set default style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)


def calculate_fold_metrics(
    base_metrics: Dict[str, Any],
    meta_metrics: Dict[str, Any],
    fold_id: int,
) -> Dict[str, Any]:
    """
    Calculate performance metrics for a single fold comparing base model vs meta model.
    
    Parameters
    ----------
    base_metrics : Dict
        Dictionary containing base model metrics including TP, FP, FN, TN counts
    meta_metrics : Dict
        Dictionary containing meta model metrics including TP, FP, FN, TN counts
    fold_id : int
        Identifier for the current fold
        
    Returns
    -------
    Dict
        Dictionary with all comparative metrics for the fold
    """
    metrics = {
        "fold_id": fold_id,
        
        # Base model metrics
        "base_tp": base_metrics.get("TP", 0),
        "base_fp": base_metrics.get("FP", 0),
        "base_fn": base_metrics.get("FN", 0),
        "base_tn": base_metrics.get("TN", 0),
        
        # Meta model metrics
        "meta_tp": meta_metrics.get("TP", 0),
        "meta_fp": meta_metrics.get("FP", 0),
        "meta_fn": meta_metrics.get("FN", 0),
        "meta_tn": meta_metrics.get("TN", 0),
    }
    
    # Calculate deltas (improvements)
    metrics["delta_tp"] = metrics["meta_tp"] - metrics["base_tp"]
    metrics["delta_fp"] = metrics["base_fp"] - metrics["meta_fp"]  # Reduction in FP
    metrics["delta_fn"] = metrics["base_fn"] - metrics["meta_fn"]  # Reduction in FN
    metrics["delta_tn"] = metrics["meta_tn"] - metrics["base_tn"]
    
    # Calculate relative improvements (as proportions)
    # Avoid division by zero
    metrics["rel_fp_reduction"] = (metrics["delta_fp"] / metrics["base_fp"]) if metrics["base_fp"] > 0 else 0
    metrics["rel_fn_reduction"] = (metrics["delta_fn"] / metrics["base_fn"]) if metrics["base_fn"] > 0 else 0
    
    # Calculate percentages for easier interpretation
    metrics["pct_fp_reduction"] = metrics["rel_fp_reduction"] * 100
    metrics["pct_fn_reduction"] = metrics["rel_fn_reduction"] * 100
    
    # Calculate precision, recall, F1 for both models
    # Base model
    base_precision = metrics["base_tp"] / (metrics["base_tp"] + metrics["base_fp"]) if (metrics["base_tp"] + metrics["base_fp"]) > 0 else 0
    base_recall = metrics["base_tp"] / (metrics["base_tp"] + metrics["base_fn"]) if (metrics["base_tp"] + metrics["base_fn"]) > 0 else 0
    metrics["base_precision"] = base_precision
    metrics["base_recall"] = base_recall
    metrics["base_f1"] = 2 * (base_precision * base_recall) / (base_precision + base_recall) if (base_precision + base_recall) > 0 else 0
    
    # Meta model
    meta_precision = metrics["meta_tp"] / (metrics["meta_tp"] + metrics["meta_fp"]) if (metrics["meta_tp"] + metrics["meta_fp"]) > 0 else 0
    meta_recall = metrics["meta_tp"] / (metrics["meta_tp"] + metrics["meta_fn"]) if (metrics["meta_tp"] + metrics["meta_fn"]) > 0 else 0
    metrics["meta_precision"] = meta_precision
    metrics["meta_recall"] = meta_recall
    metrics["meta_f1"] = 2 * (meta_precision * meta_recall) / (meta_precision + meta_recall) if (meta_precision + meta_recall) > 0 else 0
    
    # Calculate F1 improvement
    metrics["delta_f1"] = metrics["meta_f1"] - metrics["base_f1"]
    metrics["rel_f1_improvement"] = (metrics["delta_f1"] / metrics["base_f1"]) if metrics["base_f1"] > 0 else 0
    metrics["pct_f1_improvement"] = metrics["rel_f1_improvement"] * 100
    
    return metrics


def save_fold_metrics(
    fold_metrics: Dict[str, Any], 
    output_dir: Union[str, Path],
    fold_id: int,
) -> Path:
    """
    Save per-fold metrics to a CSV file.
    
    Parameters
    ----------
    fold_metrics : Dict
        Dictionary with fold metrics
    output_dir : str or Path
        Directory where to save the output
    fold_id : int
        Identifier for the current fold
        
    Returns
    -------
    Path
        Path to the saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert metrics to DataFrame for easier serialization
    metrics_df = pd.DataFrame([fold_metrics])
    
    # Save to CSV
    output_file = output_dir / f"fold_{fold_id}_metrics.csv"
    metrics_df.to_csv(output_file, index=False)
    
    return output_file


def aggregate_fold_metrics(
    metrics_files: List[Union[str, Path]],
    output_dir: Union[str, Path],
    confidence_level: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate metrics across all folds and calculate summary statistics.
    
    Parameters
    ----------
    metrics_files : List
        List of paths to fold metrics files
    output_dir : str or Path
        Directory where to save the aggregate metrics
    confidence_level : float, default=0.95
        Confidence level for intervals (0.95 = 95% CI)
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (per_fold_df, summary_df) containing the combined fold metrics and summary stats
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all fold metrics
    dfs = []
    for file_path in metrics_files:
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    # Combine into a single DataFrame
    if dfs:
        per_fold_df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError("No fold metrics files were found")
    
    # Save combined fold metrics
    combined_path = output_dir / "all_folds_metrics.csv"
    per_fold_df.to_csv(combined_path, index=False)
    
    # Calculate summary statistics
    summary_metrics = {}
    
    # Calculate mean, std, and confidence intervals for important metrics
    key_metrics = [
        "base_precision", "base_recall", "base_f1",
        "meta_precision", "meta_recall", "meta_f1",
        "rel_fp_reduction", "rel_fn_reduction", "pct_fp_reduction", "pct_fn_reduction",
        "rel_f1_improvement", "pct_f1_improvement"
    ]
    
    for metric in key_metrics:
        values = per_fold_df[metric].values
        mean_val = np.mean(values)
        std_val = np.std(values)
        n = len(values)
        
        # Calculate confidence interval
        t_crit = stats.t.ppf((1 + confidence_level) / 2, n-1)
        ci_margin = t_crit * (std_val / np.sqrt(n))
        ci_low = mean_val - ci_margin
        ci_high = mean_val + ci_margin
        
        # Store in summary dict
        summary_metrics[f"{metric}_mean"] = mean_val
        summary_metrics[f"{metric}_std"] = std_val
        summary_metrics[f"{metric}_ci_low"] = ci_low
        summary_metrics[f"{metric}_ci_high"] = ci_high
    
    # Additional aggregate metrics
    summary_metrics["n_folds"] = n
    summary_metrics["confidence_level"] = confidence_level
    
    # Convert to DataFrame and save
    summary_df = pd.DataFrame([summary_metrics])
    summary_path = output_dir / "cv_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Also save a JSON version for easier programmatic access
    with open(output_dir / "cv_metrics_summary.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)
    
    # Create a simplified summary with headline metrics
    headline_metrics = {
        "n_folds": n,
        "base_f1_mean": summary_metrics["base_f1_mean"],
        "meta_f1_mean": summary_metrics["meta_f1_mean"],
        "pct_f1_improvement_mean": summary_metrics["pct_f1_improvement_mean"],
        "pct_fp_reduction_mean": summary_metrics["pct_fp_reduction_mean"],
        "pct_fn_reduction_mean": summary_metrics["pct_fn_reduction_mean"],
        "confidence_level": confidence_level
    }
    
    with open(output_dir / "cv_headline_metrics.json", "w") as f:
        json.dump(headline_metrics, f, indent=2)
    
    return per_fold_df, summary_df


def plot_fold_distributions(
    fold_metrics_df: pd.DataFrame,
    output_dir: Union[str, Path],
    metrics_to_plot: Optional[List[str]] = None,
) -> List[Path]:
    """
    Create box plots showing the distribution of metrics across folds.
    
    Parameters
    ----------
    fold_metrics_df : pd.DataFrame
        DataFrame containing metrics for all folds
    output_dir : str or Path
        Directory where to save the plots
    metrics_to_plot : List[str], optional
        List of metrics to plot. If None, defaults to key metrics
        
    Returns
    -------
    List[Path]
        Paths to the saved plot files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if metrics_to_plot is None:
        metrics_to_plot = [
            ("pct_fp_reduction", "FP Reduction (%)"),
            ("pct_fn_reduction", "FN Reduction (%)"),
            ("pct_f1_improvement", "F1 Score Improvement (%)"),
        ]
    
    plot_paths = []
    
    # Box plots for error reduction metrics
    for metric, title in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=fold_metrics_df[metric])
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title(f"{title} Across Folds")
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"{metric}_boxplot.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        plot_paths.append(plot_path)
    
    # Paired bar chart comparing base vs meta model F1 scores
    plt.figure(figsize=(12, 6))
    fold_metrics_df_melted = pd.melt(
        fold_metrics_df, 
        id_vars=['fold_id'], 
        value_vars=['base_f1', 'meta_f1'],
        var_name='Model', 
        value_name='F1 Score'
    )
    
    # Replace column names with friendly names
    fold_metrics_df_melted['Model'] = fold_metrics_df_melted['Model'].replace({
        'base_f1': 'Base Model', 
        'meta_f1': 'Meta Model'
    })
    
    sns.barplot(
        data=fold_metrics_df_melted, 
        x='fold_id', 
        y='F1 Score', 
        hue='Model',
        palette=['lightblue', 'darkblue']
    )
    plt.title('F1 Score Comparison Across Folds')
    plt.xlabel('Fold ID')
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "f1_comparison_by_fold.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    plot_paths.append(plot_path)
    
    # Create a summary visualization
    plt.figure(figsize=(12, 8))
    
    # Setup for 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cross-Validation Performance Summary', fontsize=16)
    
    # 1. F1 scores comparison
    axs[0, 0].bar(['Base Model', 'Meta Model'], 
                 [fold_metrics_df['base_f1'].mean(), fold_metrics_df['meta_f1'].mean()],
                 yerr=[fold_metrics_df['base_f1'].std(), fold_metrics_df['meta_f1'].std()],
                 capsize=10, color=['lightblue', 'darkblue'])
    axs[0, 0].set_title('Average F1 Score')
    axs[0, 0].set_ylim(0, 1.0)
    
    # 2. FP/FN Reduction
    reductions = [
        fold_metrics_df['pct_fp_reduction'].mean(), 
        fold_metrics_df['pct_fn_reduction'].mean()
    ]
    axs[0, 1].bar(['False Positive', 'False Negative'], 
                 reductions,
                 yerr=[fold_metrics_df['pct_fp_reduction'].std(), fold_metrics_df['pct_fn_reduction'].std()],
                 capsize=10, color=['green', 'purple'])
    axs[0, 1].set_title('Error Reduction (%)')
    
    # 3. Boxplot of FP/FN reductions
    reduction_data = pd.melt(
        fold_metrics_df, 
        id_vars=['fold_id'], 
        value_vars=['pct_fp_reduction', 'pct_fn_reduction'],
        var_name='Error Type', 
        value_name='Reduction (%)'
    )
    reduction_data['Error Type'] = reduction_data['Error Type'].replace({
        'pct_fp_reduction': 'False Positive', 
        'pct_fn_reduction': 'False Negative'
    })
    
    sns.boxplot(data=reduction_data, x='Error Type', y='Reduction (%)', ax=axs[1, 0])
    axs[1, 0].set_title('Error Reduction Distribution')
    
    # 4. F1 improvement distribution
    sns.histplot(fold_metrics_df['pct_f1_improvement'], kde=True, ax=axs[1, 1])
    axs[1, 1].set_title('F1 Score Improvement Distribution (%)')
    axs[1, 1].set_xlabel('F1 Improvement (%)')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save summary plot
    summary_path = output_dir / "cv_performance_summary.png"
    plt.savefig(summary_path, dpi=300)
    plt.close()
    plot_paths.append(summary_path)
    
    return plot_paths


def create_cv_evaluation_report(
    output_dir: Union[str, Path],
    base_model_name: str = "SpliceAI",
    meta_model_name: str = "Meta-Model",
) -> Path:
    """
    Create a Markdown report summarizing the cross-validation results.
    
    Parameters
    ----------
    output_dir : str or Path
        Directory with the metrics and plots
    base_model_name : str, default="SpliceAI"
        Name of the base model for the report
    meta_model_name : str, default="Meta-Model"
        Name of the meta model for the report
        
    Returns
    -------
    Path
        Path to the generated report file
    """
    output_dir = Path(output_dir)
    
    # Load summary metrics
    try:
        with open(output_dir / "cv_headline_metrics.json", "r") as f:
            headline = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Headline metrics file not found in {output_dir}")
    
    # Create the report content
    report_content = f"""# Meta-Model CV Evaluation Report

## Headline Metrics

### Relative Error Reductions
- **False Positive Reduction**: {headline['pct_fp_reduction_mean']:.2f}% ({headline['fp_delta_mean']:.1f} fewer FPs on average per fold)
- **False Negative Reduction**: {headline['pct_fn_reduction_mean']:.2f}% ({headline['fn_delta_mean']:.1f} fewer FNs on average per fold)
- **Total Error Reduction**: {(headline['pct_fp_reduction_mean'] + headline['pct_fn_reduction_mean'])/2:.2f}% combined

### Performance Metrics
- **F1 Score Improvement**: {headline['pct_f1_improvement_mean']:.2f}%

These metrics represent the average across all {headline['n_folds']} CV folds with {headline['confidence_level']*100:.0f}% confidence intervals available in the detailed metrics file.

## Visualizations

### Performance Summary
![Performance Summary](cv_performance_summary.png)

### F1 Score Comparison
![F1 Score Comparison](f1_comparison_by_fold.png)

### Error Reduction Distribution
![FP Reduction](pct_fp_reduction_boxplot.png)
![FN Reduction](pct_fn_reduction_boxplot.png)

## Detailed Metrics

Full detailed metrics are available in the following files:

- `all_folds_metrics.csv`: Raw metrics for each individual fold
- `cv_metrics_summary.csv`: Statistical summary across all folds
- `cv_metrics_summary.json`: JSON format of the summary statistics

## Interpretation

The {meta_model_name} demonstrates {'significant' if headline['pct_fp_reduction_mean'] > 10 else 'some'} improvement over the {base_model_name} baseline:

1. **False Positive Reduction**: The meta-model reduces false positives by {headline['pct_fp_reduction_mean']:.2f}% on average, improving precision.

2. **False Negative Reduction**: The meta-model reduces false negatives by {headline['pct_fn_reduction_mean']:.2f}% on average, improving recall.

3. **Overall Performance**: F1 score improves by {headline['pct_f1_improvement_mean']:.2f}% on average across folds.

"""
    
    # Write the report to a Markdown file
    report_path = output_dir / "cv_evaluation_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    
    return report_path


def run_full_cv_evaluation(
    fold_results: List[Dict[str, Dict[str, Any]]],
    output_dir: Union[str, Path],
    base_model_name: str = "SpliceAI",
    meta_model_name: str = "Meta-Model",
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Run a complete cross-validation evaluation workflow.
    
    Parameters
    ----------
    fold_results : List[Dict[str, Dict[str, Any]]]
        List of dictionaries containing base and meta model metrics for each fold
        Each dict should have 'base' and 'meta' keys with respective metrics
    output_dir : str or Path
        Directory where to save all evaluation outputs
    base_model_name : str, default="SpliceAI"
        Name of the base model
    meta_model_name : str, default="Meta-Model"  
        Name of the meta model
    confidence_level : float, default=0.95
        Confidence level for intervals
        
    Returns
    -------
    Dict[str, Any]
        Summary of the evaluation results
    """
    output_dir = Path(output_dir)
    cv_dir = output_dir / "cv_evaluation"
    cv_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each fold
    fold_metrics_files = []
    for i, fold_result in enumerate(fold_results):
        fold_id = i + 1
        
        # Calculate metrics for this fold
        fold_metrics = calculate_fold_metrics(
            base_metrics=fold_result['base'],
            meta_metrics=fold_result['meta'],
            fold_id=fold_id
        )
        
        # Save fold metrics
        fold_file = save_fold_metrics(fold_metrics, cv_dir, fold_id)
        fold_metrics_files.append(fold_file)
    
    # Aggregate metrics across all folds
    per_fold_df, summary_df = aggregate_fold_metrics(
        fold_metrics_files,
        cv_dir,
        confidence_level=confidence_level
    )
    
    # Generate visualizations
    plot_paths = plot_fold_distributions(per_fold_df, cv_dir)
    
    # Create evaluation report
    report_path = create_cv_evaluation_report(
        cv_dir,
        base_model_name=base_model_name,
        meta_model_name=meta_model_name
    )
    
    # Return a summary of results for the caller
    return {
        "headline_metrics_path": str(cv_dir / "cv_headline_metrics.json"),
        "detailed_metrics_path": str(cv_dir / "cv_metrics_summary.csv"),
        "report_path": str(report_path),
        "visualization_paths": [str(p) for p in plot_paths],
        "headline_metrics": {
            "f1_improvement": summary_df["pct_f1_improvement_mean"].iloc[0],
            "fp_reduction": summary_df["pct_fp_reduction_mean"].iloc[0],
            "fn_reduction": summary_df["pct_fn_reduction_mean"].iloc[0],
        }
    }


def get_version() -> str:
    """Return the version of this module."""
    return __version__


if __name__ == "__main__":
    # Example usage - can be used for testing
    print(f"MetaSpliceAI CV Evaluation Module v{get_version()}")
    print("This module provides utilities for cross-validation evaluation.")
    print("It should be imported by other scripts rather than run directly.")
