#!/usr/bin/env python3
"""
Transcript-Level Top-K Accuracy Evaluation for Cross-Validation

This module provides functions to:
1. Calculate transcript-level top-k accuracy for each fold
2. Aggregate and compare transcript-level metrics across CV folds
3. Generate visualizations of transcript-level performance improvement
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

# Module version
__version__ = "0.1.0"

from .top_k_metrics import calculate_transcript_level_top_k
from .cv_evaluation import aggregate_fold_metrics


def calculate_transcript_level_metrics_for_fold(
    predictions_file: Union[str, Path],
    annotations_file: Union[str, Path],
    k_values: List[int],
    fold_id: int,
    cache_dir: Optional[Union[str, Path]] = None,
    dynamic_k: bool = False,
    dynamic_k_percentiles: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Calculate transcript-level top-k accuracy metrics for a single fold
    
    Parameters
    ----------
    predictions_file : str or Path
        Path to the predictions file (CSV/TSV/Parquet)
    annotations_file : str or Path
        Path to the annotations file with transcript mappings
    k_values : List[int], default=[1, 3, 5, 10]
        List of k values for top-k accuracy calculation
    fold_id : int, default=1
        Identifier for the current fold
    cache_dir : str or Path, optional
        Directory to store/load transcript mapping cache
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with transcript-level metrics for the fold
    """
    # Import here to avoid circular imports
    from meta_spliceai.splice_engine.top_k_metrics import calculate_transcript_level_top_k

    # Store both standard and dynamic k-values to evaluate
    evaluation_k_values = list(k_values)
    dynamic_k_mapping = {}
    
    # Calculate dynamic k-values if enabled
    if dynamic_k:
        # First load and map transcripts to get statistics
        from collections import Counter
        import numpy as np
        import pandas as pd
        
        print(f"Calculating dynamic k-values based on transcript characteristics...")
        
        # We need to determine the distribution of splice sites per transcript
        # First, load annotations
        if str(annotations_file).endswith('.tsv') or str(annotations_file).endswith('.csv'):
            delimiter = '\t' if str(annotations_file).endswith('.tsv') else ','
            annotations = pd.read_csv(annotations_file, delimiter=delimiter)
        else:
            # Try as parquet
            try:
                annotations = pd.read_parquet(annotations_file)
            except Exception as e:
                raise ValueError(f"Unsupported annotation file format: {annotations_file}. Error: {e}")
        
        # Count splice sites per transcript
        if 'transcript_id' in annotations.columns:
            sites_per_transcript = Counter(annotations['transcript_id'].dropna())
            
            # Calculate percentile values
            sites_counts = np.array(list(sites_per_transcript.values()))
            
            if len(sites_counts) > 0:
                for percentile in dynamic_k_percentiles:
                    dynamic_k = max(1, int(np.percentile(sites_counts, percentile)))
                    dynamic_k_mapping[percentile] = dynamic_k
                    evaluation_k_values.append(dynamic_k)
                
                # Remove duplicates while preserving order
                evaluation_k_values = list(dict.fromkeys(evaluation_k_values))
                print(f"Dynamic k-values based on transcript percentiles: {dynamic_k_mapping}")

    # Perform transcript-level top-k accuracy evaluation
    metrics_dict = {}
    for k in evaluation_k_values:
        metrics = calculate_transcript_level_top_k(
            predictions_file, 
            annotations_file,
            k=k,
            cache_dir=cache_dir
        )
        
        # Store metrics in dictionary with k as key
        k_key = f"top_{k}"
        metrics_dict[k_key] = metrics["accuracy"]
        
        # Mark dynamic k-values in the metrics
        if dynamic_k:
            for percentile, dyn_k in dynamic_k_mapping.items():
                if k == dyn_k:
                    metrics_dict[f"dynamic_p{percentile}_k{k}"] = metrics["accuracy"]
        
        # Store additional metrics if available
        for key in ["n_transcripts", "n_correct", "donor_accuracy", "acceptor_accuracy"]:
            if key in metrics:
                metrics_dict[f"{k_key}_{key}"] = metrics[key]
    
    # Add fold identifier
    metrics_dict["fold_id"] = fold_id
    
    return metrics_dict


def save_transcript_metrics(
    metrics: Dict[str, Any],
    output_dir: Union[str, Path],
    fold_id: int,
) -> Path:
    """
    Save transcript-level metrics to a CSV file
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Dictionary with transcript metrics
    output_dir : str or Path
        Directory to save the metrics
    fold_id : int
        Fold identifier
        
    Returns
    -------
    Path
        Path to the saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame([metrics])
    
    # Save to file
    output_file = output_dir / f"fold_{fold_id}_transcript_metrics.csv"
    metrics_df.to_csv(output_file, index=False)
    
    return output_file


def generate_transcript_level_report(metrics_df: pd.DataFrame, output_dir: Path, dynamic_k: bool = False) -> Path:
    """
    Generate a report summarizing transcript-level metrics
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with transcript-level metrics
    output_dir : Path
        Directory to save the report
    dynamic_k : bool, default=False
        Whether dynamic k-values were used
    
    Returns
    -------
    Path
        Path to the generated report
    """
    output_dir = Path(output_dir)
    report_path = output_dir / "transcript_report.md"
    
    # Extract standard k values used in metrics
    k_values = sorted([int(col.split('_')[1]) for col in metrics_df.columns if col.startswith("top_") and not col.endswith("_n_transcripts") and not '_n_correct' in col and not '_donor_accuracy' in col and not '_acceptor_accuracy' in col])
    
    # Check for dynamic k values
    dynamic_k_cols = [col for col in metrics_df.columns if col.startswith("dynamic_p")]
    dynamic_info = """

## Dynamic K-Values

Dynamic k-values were calculated based on the distribution of splice sites per transcript:

| Percentile | k-value | Top-k Accuracy |
|------------|---------|----------------|
""" if dynamic_k_cols else ""
    
    for col in dynamic_k_cols:
        parts = col.replace('dynamic_p', '').split('_k')  # splits 'dynamic_p25_k3' into '25' and '3'
        percentile = parts[0]
        k = parts[1]
        accuracy = metrics_df[col].mean()
        dynamic_info += f"| {percentile}% | {k} | {accuracy:.4f} |\n"
    
    # Create report content
    report = f"""# Transcript-Level Top-K Accuracy Report

## Summary

This report summarizes transcript-level top-k accuracy metrics across {len(metrics_df)} CV folds.

## Top-K Accuracy

{dynamic_info if dynamic_k else ''}
"""
    
    # Save report
    with open(report_path, 'w') as f:
        f.write(report)
    
    return report_path


def plot_transcript_metrics(
    metrics_df: pd.DataFrame,
    output_dir: Union[str, Path],
    k_values: List[int] = [1, 3, 5, 10],
) -> List[Path]:
    """
    Create visualizations for transcript-level metrics
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with transcript metrics for all folds
    output_dir : str or Path
        Directory to save the plots
    k_values : List[int], default=[1, 3, 5, 10]
        List of k values used in the metrics
        
    Returns
    -------
    List[Path]
        List of saved plot paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = []
    
    # 1. Box plots for top-k accuracy by k
    plt.figure(figsize=(12, 8))
    
    # Prepare data for box plots
    accuracy_data = []
    for k in k_values:
        for _, row in metrics_df.iterrows():
            accuracy_data.append({
                "k": k,
                "accuracy": row[f"top_{k}_accuracy"],
                "fold": row["fold_id"]
            })
    
    accuracy_df = pd.DataFrame(accuracy_data)
    
    # Create box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=accuracy_df, x="k", y="accuracy")
    plt.title("Transcript-Level Top-K Accuracy Across Folds")
    plt.xlabel("k value")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    
    # Save plot
    box_plot_path = output_dir / "transcript_top_k_accuracy_boxplot.png"
    plt.savefig(box_plot_path, dpi=300)
    plt.close()
    plot_paths.append(box_plot_path)
    
    # 2. Line plot showing average accuracy by k
    plt.figure(figsize=(10, 6))
    
    # Calculate means and standard deviations
    mean_accuracies = []
    std_accuracies = []
    
    for k in k_values:
        col_name = f"top_{k}_accuracy"
        mean_accuracies.append(metrics_df[col_name].mean())
        std_accuracies.append(metrics_df[col_name].std())
    
    # Plot with error bands
    plt.errorbar(
        k_values, 
        mean_accuracies, 
        yerr=std_accuracies, 
        marker='o', 
        linestyle='-', 
        capsize=5
    )
    
    plt.title("Mean Transcript-Level Top-K Accuracy by K Value")
    plt.xlabel("k value")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    # Add specific values as annotations
    for i, (k, mean, std) in enumerate(zip(k_values, mean_accuracies, std_accuracies)):
        plt.annotate(
            f"{mean:.4f}±{std:.4f}", 
            (k, mean), 
            textcoords="offset points",
            xytext=(0, 10), 
            ha='center'
        )
    
    # Save plot
    line_plot_path = output_dir / "transcript_top_k_accuracy_line.png"
    plt.savefig(line_plot_path, dpi=300)
    plt.close()
    plot_paths.append(line_plot_path)
    
    # 3. Summary plot with comparison to baseline if available
    # Check if we have baseline metrics
    has_baseline = any("base" in col for col in metrics_df.columns)
    
    if has_baseline:
        plt.figure(figsize=(12, 8))
        
        base_means = []
        meta_means = []
        
        for k in k_values:
            base_col = f"base_top_{k}_accuracy"
            meta_col = f"top_{k}_accuracy"
            
            if base_col in metrics_df.columns:
                base_means.append(metrics_df[base_col].mean())
                meta_means.append(metrics_df[meta_col].mean())
        
        # Create bar chart
        x = np.arange(len(k_values))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - width/2, base_means, width, label='Base Model')
        ax.bar(x + width/2, meta_means, width, label='Meta Model')
        
        ax.set_xlabel('K Value')
        ax.set_ylabel('Accuracy')
        ax.set_title('Transcript-Level Top-K Accuracy Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(k_values)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save plot
        comparison_path = output_dir / "transcript_accuracy_comparison.png"
        plt.savefig(comparison_path, dpi=300)
        plt.close()
        plot_paths.append(comparison_path)
    
    return plot_paths


def run_transcript_level_cv_evaluation(
    fold_predictions: List[Tuple[Union[str, Path], int]],
    annotations_file: Union[str, Path],
    output_dir: Union[str, Path],
    k_values: List[int] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    dynamic_k: bool = False,
    dynamic_k_percentiles: List[float] = None
) -> Dict[str, Any]:
    """
    Run transcript-level top-k accuracy evaluation across CV folds
    
    Parameters
    ----------
    fold_predictions : List[Tuple[Union[str, Path], int]]
        List of tuples with (predictions_file, fold_id)
    annotations_file : str or Path
        Path to the annotations file
    output_dir : str or Path
        Directory to save evaluation results
    k_values : List[int], default=[1, 3, 5, 10]
        List of k values for top-k accuracy
    cache_dir : str or Path, optional
        Directory to store transcript mapping cache
    verbose : bool, default=True
        Whether to print verbose output
    dynamic_k : bool, default=False
        Whether to use dynamic k-values based on transcript characteristics
    dynamic_k_percentiles : List[float], optional
        Percentiles of splice sites per transcript to use as k-values (default: [10, 25, 50, 100])
        Directory for transcript mapping cache
        
    Returns
    -------
    Dict[str, Any]
        Summary of evaluation results
    """
    # Set default k values if not provided
    if k_values is None:
        k_values = [1, 3, 5, 10]
    
    # Set default dynamic k percentiles if enabled but not provided
    if dynamic_k and dynamic_k_percentiles is None:
        dynamic_k_percentiles = [10, 25, 50, 100]
    
    output_dir = Path(output_dir)
    transcript_dir = output_dir / "transcript_level"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each fold
    metrics_files = []
    
    for predictions_file, fold_id in fold_predictions:
        print(f"Processing fold {fold_id}...")
        
        # Calculate transcript metrics for this fold
        fold_metrics = calculate_transcript_level_metrics_for_fold(
            predictions_file=predictions_file,
            annotations_file=annotations_file,
            k_values=k_values,
            fold_id=fold_id,
            cache_dir=cache_dir,
            dynamic_k=dynamic_k,
            dynamic_k_percentiles=dynamic_k_percentiles
        )
        
        # Save metrics to file
        metrics_file = save_transcript_metrics(fold_metrics, transcript_dir, fold_id)
        metrics_files.append(metrics_file)
    
    # Aggregate metrics across folds
    all_metrics_df = pd.concat([pd.read_csv(f) for f in metrics_files], ignore_index=True)
    all_metrics_file = transcript_dir / "all_transcript_metrics.csv"
    all_metrics_df.to_csv(all_metrics_file, index=False)
    
    # Create summary dictionary
    summary = {
        "n_folds": len(fold_predictions),
    }
    
    # Add mean and std for each k value
    for k in k_values:
        k_key = f"top_{k}"
        if k_key in all_metrics_df.columns:
            summary[f"{k_key}_mean"] = all_metrics_df[k_key].mean()
            summary[f"{k_key}_std"] = all_metrics_df[k_key].std()
    
    # Add dynamic k summaries if enabled
    if dynamic_k:
        dynamic_k_cols = [col for col in all_metrics_df.columns if col.startswith("dynamic_p")]
        for col in dynamic_k_cols:
            parts = col.replace('dynamic_p', '').split('_k')  # splits 'dynamic_p25_k3' into '25' and '3'
            percentile = parts[0]
            k_val = parts[1]
            summary[f"dynamic_p{percentile}_k{k_val}_mean"] = all_metrics_df[col].mean()
            summary[f"dynamic_p{percentile}_k{k_val}_std"] = all_metrics_df[col].std()
            # Also store which k-value corresponds to which percentile
            summary[f"percentile_{percentile}_k"] = int(k_val)
    
    # Generate visualizations
    plot_paths = plot_transcript_metrics(all_metrics_df, transcript_dir, k_values)
    
    # Generate report
    report_path = generate_transcript_level_report(all_metrics_df, transcript_dir, dynamic_k=dynamic_k)
    
    # Save summary metrics
    with open(transcript_dir / "transcript_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Create visualizations
    plot_paths = plot_transcript_metrics(
        metrics_df=all_metrics_df,
        output_dir=transcript_dir,
        k_values=k_values
    )
    
    return {
        "metrics_path": str(all_metrics_file),
        "summary_path": str(transcript_dir / "transcript_summary.json"),
        "visualization_paths": [str(p) for p in plot_paths],
        "summary": summary
    }


def get_version() -> str:
    """Return the version of this module."""
    return __version__


if __name__ == "__main__":
    # Example usage
    print(f"MetaSpliceAI Transcript-Level CV Evaluation Module v{get_version()}")
    print("This module provides utilities for transcript-level evaluation across CV folds.")
    print("It should be imported by other scripts rather than run directly.")
