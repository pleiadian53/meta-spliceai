"""Comprehensive Model Evaluation Utilities

This module provides comprehensive evaluation metrics for splice site prediction models,
including F1 score, ROC-AUC, Average Precision (AP), and top-k accuracy.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_curve,
    precision_recall_curve
)


def calculate_comprehensive_metrics(
    positions_df: pl.DataFrame,
    pred_type_col: str = 'pred_type',
    score_col: str = 'score',
    splice_type_col: str = 'splice_type',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for splice site predictions.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        DataFrame with predictions and true labels.
        Required columns:
        - pred_type: 'TP', 'TN', 'FP', 'FN'
        - score: Prediction score (probability)
        - splice_type: 'donor', 'acceptor', or None
    pred_type_col : str, default='pred_type'
        Column name for prediction type (TP/TN/FP/FN)
    score_col : str, default='score'
        Column name for prediction scores
    splice_type_col : str, default='splice_type'
        Column name for splice type
    verbose : bool, default=True
        Print detailed metrics
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - tp, tn, fp, fn: Confusion matrix counts
        - precision, recall, f1: Classification metrics
        - accuracy: Overall accuracy
        - roc_auc: ROC-AUC score
        - average_precision: Average Precision (AP) score
        - top_k_accuracy: Top-k accuracy (if applicable)
        - donor_metrics: Per-splice-type metrics
        - acceptor_metrics: Per-splice-type metrics
    """
    if positions_df.height == 0:
        return {
            'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'accuracy': 0.0, 'roc_auc': 0.0, 'average_precision': 0.0,
            'top_k_accuracy': 0.0
        }
    
    # Extract confusion matrix counts
    tp = positions_df.filter(pl.col(pred_type_col) == 'TP').height
    tn = positions_df.filter(pl.col(pred_type_col) == 'TN').height
    fp = positions_df.filter(pl.col(pred_type_col) == 'FP').height
    fn = positions_df.filter(pl.col(pred_type_col) == 'FN').height
    
    # Calculate basic metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # Prepare binary labels and scores for ROC/PR curves
    # 1 = splice site (TP or FN), 0 = non-splice site (TN or FP)
    y_true = []
    y_scores = []
    
    for row in positions_df.iter_rows(named=True):
        pred_type = row[pred_type_col]
        score = row.get(score_col, 0.0)
        
        if pred_type in ['TP', 'FN']:
            y_true.append(1)  # True splice site
        else:  # TN or FP
            y_true.append(0)  # Non-splice site
        
        y_scores.append(float(score))
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Calculate ROC-AUC and Average Precision
    roc_auc = 0.0
    average_precision = 0.0
    
    if len(np.unique(y_true)) > 1:  # Need both classes
        try:
            roc_auc = float(roc_auc_score(y_true, y_scores))
        except Exception as e:
            if verbose:
                print(f"[warning] Could not calculate ROC-AUC: {e}")
        
        try:
            average_precision = float(average_precision_score(y_true, y_scores))
        except Exception as e:
            if verbose:
                print(f"[warning] Could not calculate Average Precision: {e}")
    
    # Calculate top-k accuracy
    top_k_accuracy = calculate_top_k_accuracy(
        positions_df,
        pred_type_col=pred_type_col,
        score_col=score_col,
        splice_type_col=splice_type_col
    )
    
    # Per-splice-type metrics
    donor_metrics = calculate_splice_type_metrics(
        positions_df.filter(pl.col(splice_type_col) == 'donor'),
        pred_type_col=pred_type_col,
        score_col=score_col
    )
    
    acceptor_metrics = calculate_splice_type_metrics(
        positions_df.filter(pl.col(splice_type_col) == 'acceptor'),
        pred_type_col=pred_type_col,
        score_col=score_col
    )
    
    results = {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'average_precision': average_precision,
        'top_k_accuracy': top_k_accuracy,
        'donor': donor_metrics,
        'acceptor': acceptor_metrics
    }
    
    if verbose:
        print("\n" + "=" * 80)
        print("COMPREHENSIVE EVALUATION METRICS")
        print("=" * 80)
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp:,}  TN: {tn:,}  FP: {fp:,}  FN: {fn:,}")
        print(f"\nClassification Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"\nRanking Metrics:")
        print(f"  ROC-AUC:          {roc_auc:.4f}")
        print(f"  Average Precision: {average_precision:.4f}")
        print(f"  Top-K Accuracy:   {top_k_accuracy:.4f}")
        print(f"\nPer-Splice-Type Metrics:")
        print(f"  Donor:")
        print(f"    F1: {donor_metrics.get('f1', 0.0):.4f}, "
              f"Precision: {donor_metrics.get('precision', 0.0):.4f}, "
              f"Recall: {donor_metrics.get('recall', 0.0):.4f}")
        print(f"  Acceptor:")
        print(f"    F1: {acceptor_metrics.get('f1', 0.0):.4f}, "
              f"Precision: {acceptor_metrics.get('precision', 0.0):.4f}, "
              f"Recall: {acceptor_metrics.get('recall', 0.0):.4f}")
        print("=" * 80)
    
    return results


def calculate_splice_type_metrics(
    positions_df: pl.DataFrame,
    pred_type_col: str = 'pred_type',
    score_col: str = 'score'
) -> Dict[str, float]:
    """Calculate metrics for a specific splice type."""
    if positions_df.height == 0:
        return {
            'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0
        }
    
    tp = positions_df.filter(pl.col(pred_type_col) == 'TP').height
    tn = positions_df.filter(pl.col(pred_type_col) == 'TN').height
    fp = positions_df.filter(pl.col(pred_type_col) == 'FP').height
    fn = positions_df.filter(pl.col(pred_type_col) == 'FN').height
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1
    }


def calculate_top_k_accuracy(
    positions_df: pl.DataFrame,
    pred_type_col: str = 'pred_type',
    score_col: str = 'score',
    splice_type_col: str = 'splice_type',
    k: Optional[int] = None
) -> float:
    """
    Calculate top-k accuracy for splice site prediction.
    
    For each gene, select the top-k highest-scoring positions and check
    if they match true splice sites (within consensus window).
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        Positions DataFrame
    pred_type_col : str
        Column name for prediction type
    score_col : str
        Column name for scores
    splice_type_col : str
        Column name for splice type
    k : Optional[int]
        Number of top predictions to consider. If None, uses the number
        of true splice sites per gene.
    
    Returns
    -------
    float
        Top-k accuracy (0.0 to 1.0)
    """
    if positions_df.height == 0:
        return 0.0
    
    # Group by gene and calculate per-gene top-k accuracy
    gene_accuracies = []
    
    for gene_id in positions_df['gene_id'].unique().to_list():
        gene_df = positions_df.filter(pl.col('gene_id') == gene_id)
        
        # Count true splice sites for this gene
        true_splice_sites = gene_df.filter(
            pl.col(pred_type_col).is_in(['TP', 'FN'])
        ).height
        
        if true_splice_sites == 0:
            continue  # Skip genes with no splice sites
        
        # Determine k (number of top predictions to consider)
        if k is None:
            k_gene = true_splice_sites
        else:
            k_gene = min(k, gene_df.height)
        
        if k_gene == 0:
            continue
        
        # Get top-k predictions by score
        top_k_df = gene_df.sort(score_col, descending=True).head(k_gene)
        
        # Count how many of top-k are true positives
        top_k_tp = top_k_df.filter(pl.col(pred_type_col) == 'TP').height
        
        # Calculate accuracy for this gene
        gene_accuracy = top_k_tp / k_gene
        gene_accuracies.append(gene_accuracy)
    
    # Return average across genes
    if len(gene_accuracies) == 0:
        return 0.0
    
    return float(np.mean(gene_accuracies))


def evaluate_full_genome_pass(
    positions_df: pl.DataFrame,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate a full-genome base model pass with comprehensive metrics.
    
    This function calculates all evaluation metrics and optionally saves
    them to a JSON file.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        Full positions DataFrame from the workflow
    output_path : Optional[str]
        Path to save evaluation results JSON
    verbose : bool
        Print detailed metrics
    
    Returns
    -------
    Dict[str, float]
        Comprehensive evaluation metrics
    """
    metrics = calculate_comprehensive_metrics(
        positions_df,
        verbose=verbose
    )
    
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        if verbose:
            print(f"\n✅ Saved evaluation metrics to: {output_path}")
    
    return metrics




