#!/usr/bin/env python3
"""
Transcript-level top-k accuracy metrics for splice site prediction.

This module implements the SpliceAI approach to top-k accuracy calculation:
1. For each transcript, rank every nucleotide position by the model's donor/acceptor probability
2. Pick the top k positions (where k = number of true splice sites in that transcript)
3. Compute the fraction of true sites that appear in that top-k set
4. Average this fraction across all transcripts

Reference: Jaganathan et al., 2019 (SpliceAI paper)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional


def calculate_site_top_k_accuracy(
    df: pd.DataFrame,
    *,
    prob_donor_col: str = "prob_donor",
    prob_acceptor_col: str = "prob_acceptor",
    label_col: str = "label",
    group_col: str = "gene_id",  # Gene ID as fallback if no transcript IDs
    donor_label: int = 0,  # Label for donor sites
    acceptor_label: int = 1,  # Label for acceptor sites
) -> Dict[str, float]:
    """
    Calculate per-gene or per-transcript top-k accuracy for donor and acceptor sites.
    
    This implementation follows the SpliceAI approach described in Jaganathan et al., 2019.
    For each gene/transcript, it ranks positions by probability and calculates how many true
    sites appear in the top-k predictions, where k equals the count of true sites in that gene.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing prediction probabilities and gene/transcript annotations
    prob_donor_col : str, default="prob_donor"
        Column name for donor site probability
    prob_acceptor_col : str, default="prob_acceptor"
        Column name for acceptor site probability
    label_col : str, default="label"
        Column name for the true label
    group_col : str, default="gene_id"
        Column name for gene or transcript identifier used for grouping
    donor_label : int, default=0
        Value in label_col that represents donor sites
    acceptor_label : int, default=1
        Value in label_col that represents acceptor sites

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy metrics for "donor_top_k", "acceptor_top_k", and "combined_top_k"
    """
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in the data")
        
    # For any empty group identifiers, generate a unique placeholder
    if df[group_col].isna().any() or (df[group_col] == "").any():
        df = df.copy()
        mask = df[group_col].isna() | (df[group_col] == "")
        if mask.any():
            print(f"Warning: Found {mask.sum()} rows with missing {group_col}, using row index as fallback")
            df.loc[mask, group_col] = [f"row_{i}" for i in df.index[mask]]
    
    results = {
        "donor_top_k": [],
        "acceptor_top_k": [],
    }
    
    # Process each gene/transcript separately
    for group_id, group_df in df.groupby(group_col):
        # Count true donor and acceptor sites in this group
        donor_sites = (group_df[label_col] == donor_label).sum()
        acceptor_sites = (group_df[label_col] == acceptor_label).sum()
        
        # If there are donor sites, calculate donor top-k accuracy
        if donor_sites > 0:
            # Get probabilities and true labels
            probs = group_df[prob_donor_col].values
            labels = group_df[label_col].values == donor_label
            
            # Find top k positions based on probability
            k = donor_sites
            top_k_indices = np.argsort(-probs)[:k]
            
            # Calculate accuracy: how many true sites are in the top-k predictions
            # This is the fraction of true sites that appear in the top-k predictions
            donor_acc = np.sum(labels[top_k_indices]) / donor_sites
            results["donor_top_k"].append(donor_acc)
        
        # If there are acceptor sites, calculate acceptor top-k accuracy
        if acceptor_sites > 0:
            # Get probabilities and true labels
            probs = group_df[prob_acceptor_col].values
            labels = group_df[label_col].values == acceptor_label
            
            # Find top k positions based on probability
            k = acceptor_sites
            top_k_indices = np.argsort(-probs)[:k]
            
            # Calculate accuracy: how many true sites are in the top-k predictions
            # This is the fraction of true sites that appear in the top-k predictions
            acceptor_acc = np.sum(labels[top_k_indices]) / acceptor_sites
            results["acceptor_top_k"].append(acceptor_acc)
    
    # Average across all genes/transcripts
    donor_top_k = np.mean(results["donor_top_k"]) if results["donor_top_k"] else np.nan
    acceptor_top_k = np.mean(results["acceptor_top_k"]) if results["acceptor_top_k"] else np.nan
    
    # Combined top-k is the average of donor and acceptor top-k
    combined_top_k = np.nanmean([donor_top_k, acceptor_top_k])
    
    return {
        "donor_top_k": donor_top_k,
        "acceptor_top_k": acceptor_top_k,
        "combined_top_k": combined_top_k,
        "n_groups": len(df[group_col].unique()),  # Number of genes/transcripts evaluated
    }


def calculate_cv_fold_top_k(
    X: np.ndarray,
    y: np.ndarray,
    probs: np.ndarray,
    gene_ids: np.ndarray,
    donor_label: int = 0,
    acceptor_label: int = 1,
    neither_label: int = 2,
) -> Dict[str, float]:
    """
    Calculate top-k accuracy for a cross-validation fold using gene grouping.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix for test data
    y : np.ndarray
        True labels for test data
    probs : np.ndarray
        Predicted probabilities, shape (n_samples, n_classes).
        Columns should be [donor, acceptor, neither] or [neither, donor, acceptor]
    gene_ids : np.ndarray
        Array of gene IDs corresponding to each sample
    donor_label : int, default=0
        Value in y that represents donor sites
    acceptor_label : int, default=1
        Value in y that represents acceptor sites
    neither_label : int, default=2
        Value in y that represents neither donor nor acceptor sites

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy metrics for "donor_top_k", "acceptor_top_k", and "combined_top_k"
    """
    # Determine the structure of probs based on its shape
    n_classes = probs.shape[1]
    if n_classes != 3:
        raise ValueError(f"Expected 3 classes in probabilities, got {n_classes}")
    
    # Map probabilities to the right columns
    # In our case, classes might be ordered as [donor, acceptor, neither]
    # or [neither, donor, acceptor]
    if y.max() <= 2:  # Standard ordering
        prob_donor = probs[:, donor_label]
        prob_acceptor = probs[:, acceptor_label]
    else:
        # If we need to determine the ordering differently, add logic here
        prob_donor = probs[:, 1]  # Assuming donor is column 1
        prob_acceptor = probs[:, 2]  # Assuming acceptor is column 2
    
    # Create a dataframe with all necessary data
    df = pd.DataFrame({
        "gene_id": gene_ids,
        "label": y,
        "prob_donor": prob_donor,
        "prob_acceptor": prob_acceptor,
    })
    
    # Calculate gene-level top-k accuracy
    return calculate_site_top_k_accuracy(
        df, 
        group_col="gene_id",
        donor_label=donor_label,
        acceptor_label=acceptor_label
    )


def report_top_k_accuracy(metrics: Dict[str, float]) -> str:
    """
    Generate a formatted string report of top-k accuracy metrics.

    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary with accuracy metrics from calculate_site_top_k_accuracy

    Returns
    -------
    str
        Formatted report string
    """
    group_type = "Gene"
    n_groups = metrics.get('n_groups', 0)
    return (
        f"Top-k Accuracy ({group_type}-level, {n_groups} {group_type.lower()}s):\n"
        f"  Donor:     {metrics['donor_top_k']:.4f}\n"
        f"  Acceptor:  {metrics['acceptor_top_k']:.4f}\n"
        f"  Combined:  {metrics['combined_top_k']:.4f}"
    )
