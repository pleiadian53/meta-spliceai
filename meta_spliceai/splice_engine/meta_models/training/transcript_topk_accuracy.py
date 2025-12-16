#!/usr/bin/env python3
"""
Transcript-Level Top-k Accuracy Implementation

This module implements the correct SpliceAI-style top-k accuracy calculation
at the transcript level, which is more accurate than gene-level calculation.

The approach:
1. For each transcript, rank all positions by splice probability (donor/acceptor)
2. Select top-k positions where k = actual number of splice sites in that transcript
3. Calculate fraction of true splice sites captured in top-k predictions
4. Average across all transcripts

This provides a more realistic evaluation of splice site prediction performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path


def calculate_transcript_level_top_k_accuracy(
    df: pd.DataFrame,
    *,
    transcript_col: str = "transcript_id",
    position_col: str = "position",
    label_col: str = "splice_type",
    prob_donor_col: str = "prob_donor", 
    prob_acceptor_col: str = "prob_acceptor",
    donor_label: str = "donor",
    acceptor_label: str = "acceptor",
    neither_label: str = "neither",
    verbose: bool = True
) -> Dict[str, float]:
    """
    Calculate transcript-level top-k accuracy following SpliceAI methodology.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with transcript annotations and predictions
    transcript_col : str
        Column containing transcript IDs
    position_col : str  
        Column containing genomic positions
    label_col : str
        Column containing true splice site labels
    prob_donor_col : str
        Column containing donor probabilities
    prob_acceptor_col : str
        Column containing acceptor probabilities
    donor_label : str
        Value representing donor sites in label_col
    acceptor_label : str
        Value representing acceptor sites in label_col
    neither_label : str
        Value representing non-splice sites in label_col
    verbose : bool
        Whether to print progress information
        
    Returns
    -------
    Dict[str, float]
        Dictionary with top-k accuracy metrics
    """
    
    # Validate required columns
    required_cols = [transcript_col, position_col, label_col, prob_donor_col, prob_acceptor_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter to transcripts with valid transcript IDs
    valid_transcripts = df[df[transcript_col].notna() & (df[transcript_col] != "")]
    
    if len(valid_transcripts) == 0:
        return {
            'donor_top_k': 0.0,
            'acceptor_top_k': 0.0, 
            'combined_top_k': 0.0,
            'n_transcripts': 0,
            'n_transcripts_with_donors': 0,
            'n_transcripts_with_acceptors': 0,
            'total_positions': 0
        }
    
    if verbose:
        print(f"[Transcript Top-k] Analyzing {len(valid_transcripts):,} positions across transcripts")
    
    # Group by transcript
    transcript_groups = valid_transcripts.groupby(transcript_col)
    
    donor_accuracies = []
    acceptor_accuracies = []
    combined_accuracies = []
    
    n_transcripts_with_donors = 0
    n_transcripts_with_acceptors = 0
    total_transcripts = 0
    
    for transcript_id, transcript_df in transcript_groups:
        total_transcripts += 1
        
        # Get true splice sites in this transcript
        true_donors = transcript_df[transcript_df[label_col] == donor_label]
        true_acceptors = transcript_df[transcript_df[label_col] == acceptor_label]
        
        n_true_donors = len(true_donors)
        n_true_acceptors = len(true_acceptors)
        n_true_splice = n_true_donors + n_true_acceptors
        
        # Skip transcripts with no splice sites
        if n_true_splice == 0:
            continue
            
        # Calculate donor top-k accuracy
        if n_true_donors > 0:
            n_transcripts_with_donors += 1
            
            # Rank all positions by donor probability
            donor_ranked = transcript_df.nlargest(n_true_donors, prob_donor_col)
            
            # Count how many true donors are in top-k
            donor_hits = len(donor_ranked[donor_ranked[label_col] == donor_label])
            donor_accuracy = donor_hits / n_true_donors
            donor_accuracies.append(donor_accuracy)
        
        # Calculate acceptor top-k accuracy  
        if n_true_acceptors > 0:
            n_transcripts_with_acceptors += 1
            
            # Rank all positions by acceptor probability
            acceptor_ranked = transcript_df.nlargest(n_true_acceptors, prob_acceptor_col)
            
            # Count how many true acceptors are in top-k
            acceptor_hits = len(acceptor_ranked[acceptor_ranked[label_col] == acceptor_label])
            acceptor_accuracy = acceptor_hits / n_true_acceptors
            acceptor_accuracies.append(acceptor_accuracy)
        
        # Calculate combined top-k accuracy
        if n_true_splice > 0:
            # Create combined splice probability (donor + acceptor)
            transcript_df_copy = transcript_df.copy()
            transcript_df_copy['splice_prob'] = (
                transcript_df_copy[prob_donor_col] + transcript_df_copy[prob_acceptor_col]
            )
            
            # Rank by combined splice probability
            splice_ranked = transcript_df_copy.nlargest(n_true_splice, 'splice_prob')
            
            # Count how many true splice sites are in top-k
            splice_hits = len(splice_ranked[
                splice_ranked[label_col].isin([donor_label, acceptor_label])
            ])
            combined_accuracy = splice_hits / n_true_splice
            combined_accuracies.append(combined_accuracy)
    
    # Calculate final metrics
    donor_top_k = np.mean(donor_accuracies) if donor_accuracies else 0.0
    acceptor_top_k = np.mean(acceptor_accuracies) if acceptor_accuracies else 0.0
    combined_top_k = np.mean(combined_accuracies) if combined_accuracies else 0.0
    
    if verbose:
        print(f"[Transcript Top-k] Processed {total_transcripts} transcripts")
        print(f"[Transcript Top-k] Transcripts with donors: {n_transcripts_with_donors}")
        print(f"[Transcript Top-k] Transcripts with acceptors: {n_transcripts_with_acceptors}")
        print(f"[Transcript Top-k] Donor accuracy: {donor_top_k:.3f}")
        print(f"[Transcript Top-k] Acceptor accuracy: {acceptor_top_k:.3f}")
        print(f"[Transcript Top-k] Combined accuracy: {combined_top_k:.3f}")
    
    return {
        'donor_top_k': donor_top_k,
        'acceptor_top_k': acceptor_top_k,
        'combined_top_k': combined_top_k,
        'n_transcripts': total_transcripts,
        'n_transcripts_with_donors': n_transcripts_with_donors,
        'n_transcripts_with_acceptors': n_transcripts_with_acceptors,
        'total_positions': len(valid_transcripts)
    }


def calculate_transcript_top_k_for_cv_fold(
    X: np.ndarray,
    y: np.ndarray, 
    probs: np.ndarray,
    transcript_ids: np.ndarray,
    positions: np.ndarray,
    donor_label: int = 1,
    acceptor_label: int = 2,
    neither_label: int = 0,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Calculate transcript-level top-k accuracy for a CV fold.
    
    This is the correct implementation that should replace the gene-level
    calculation for more accurate evaluation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (not used directly, but kept for interface compatibility)
    y : np.ndarray
        True labels (encoded as integers)
    probs : np.ndarray
        Predicted probabilities, shape (n_samples, 3)
        Assumes order: [neither, donor, acceptor]
    transcript_ids : np.ndarray
        Transcript IDs for each position
    positions : np.ndarray
        Genomic positions
    donor_label : int
        Integer label for donor sites
    acceptor_label : int  
        Integer label for acceptor sites
    neither_label : int
        Integer label for neither sites
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    Dict[str, float]
        Top-k accuracy metrics
    """
    
    # Create dataframe for transcript-level analysis
    df = pd.DataFrame({
        'transcript_id': transcript_ids,
        'position': positions,
        'splice_type': y,
        'prob_donor': probs[:, donor_label],
        'prob_acceptor': probs[:, acceptor_label],
        'prob_neither': probs[:, neither_label]
    })
    
    # Convert integer labels to string labels
    label_map = {neither_label: 'neither', donor_label: 'donor', acceptor_label: 'acceptor'}
    df['splice_type_str'] = df['splice_type'].map(label_map)
    
    # Filter out positions without transcript IDs
    df_with_transcripts = df[df['transcript_id'].notna() & (df['transcript_id'] != "")]
    
    if len(df_with_transcripts) == 0:
        if verbose:
            print("[Transcript Top-k] No positions with transcript IDs found")
        return {
            'donor_top_k': 0.0,
            'acceptor_top_k': 0.0,
            'combined_top_k': 0.0,
            'n_transcripts': 0,
            'n_transcripts_with_donors': 0,
            'n_transcripts_with_acceptors': 0,
            'total_positions': 0
        }
    
    # Calculate transcript-level top-k accuracy
    return calculate_transcript_level_top_k_accuracy(
        df_with_transcripts,
        transcript_col='transcript_id',
        position_col='position',
        label_col='splice_type_str',
        prob_donor_col='prob_donor',
        prob_acceptor_col='prob_acceptor',
        donor_label='donor',
        acceptor_label='acceptor', 
        neither_label='neither',
        verbose=verbose
    )


def create_transcript_context_for_cv(
    df: pd.DataFrame,
    test_idx: np.ndarray,
    transcript_col: str = "transcript_id",
    position_col: str = "position"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract transcript context for CV fold evaluation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with transcript information
    test_idx : np.ndarray
        Indices of test samples in the fold
    transcript_col : str
        Column containing transcript IDs
    position_col : str
        Column containing positions
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (transcript_ids, positions) for test samples
    """
    
    if transcript_col not in df.columns:
        raise ValueError(f"Transcript column '{transcript_col}' not found in dataset")
    
    if position_col not in df.columns:
        raise ValueError(f"Position column '{position_col}' not found in dataset")
    
    # Extract transcript info for test samples
    if hasattr(df, 'iloc'):
        # Pandas DataFrame
        transcript_ids = df.iloc[test_idx][transcript_col].values
        positions = df.iloc[test_idx][position_col].values
    else:
        # Polars DataFrame
        transcript_ids = df[test_idx][transcript_col].to_numpy()
        positions = df[test_idx][position_col].to_numpy()
    
    return transcript_ids, positions


if __name__ == "__main__":
    # Test the implementation
    print("Testing transcript-level top-k accuracy implementation...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'transcript_id': ['ENST001', 'ENST001', 'ENST001', 'ENST001', 'ENST002', 'ENST002'],
        'position': [100, 200, 300, 400, 500, 600],
        'splice_type': ['neither', 'donor', 'neither', 'acceptor', 'donor', 'neither'],
        'prob_donor': [0.1, 0.9, 0.2, 0.1, 0.8, 0.1],
        'prob_acceptor': [0.1, 0.1, 0.1, 0.9, 0.1, 0.1]
    })
    
    print("Sample data:")
    print(sample_data)
    print()
    
    # Calculate top-k accuracy
    metrics = calculate_transcript_level_top_k_accuracy(sample_data)
    print("Top-k accuracy metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")



