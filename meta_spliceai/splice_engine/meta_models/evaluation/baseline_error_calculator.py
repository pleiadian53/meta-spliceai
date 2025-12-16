#!/usr/bin/env python3
"""
Dynamic Baseline Error Calculator for CV Metrics Enhancement

This module provides dynamic calculation of baseline error counts from training datasets
to enhance CV metrics reporting with accurate baseline statistics and percentage reductions.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl


def calculate_baseline_errors_from_dataset(
    dataset_path: Union[str, Path],
    sample_size: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Union[int, float]]:
    """
    Calculate baseline error counts by analyzing the actual training dataset.
    
    This function reads the training dataset and calculates baseline SpliceAI error counts
    (false positives and false negatives) to provide accurate context for CV metrics.
    
    Parameters
    ----------
    dataset_path : str or Path
        Path to training dataset (file or directory containing parquet files)
    sample_size : int, optional
        If provided, sample this many rows for estimation (for large datasets)
    verbose : bool, default False
        Print progress information
        
    Returns
    -------
    Dict[str, Union[int, float]]
        Dictionary containing baseline error statistics:
        - 'baseline_fp': Total false positive count
        - 'baseline_fn': Total false negative count  
        - 'total_samples': Total number of samples analyzed
        - 'dataset_name': Name of the dataset
        - 'is_sampled': Whether the analysis used sampling
        - 'sample_size': Actual sample size used
    """
    dataset_path = Path(dataset_path)
    
    if verbose:
        print(f"[Baseline Calculator] Analyzing dataset: {dataset_path}")
    
    # Determine dataset name for reporting
    if dataset_path.is_file():
        dataset_name = dataset_path.stem
    else:
        dataset_name = dataset_path.name
    
    try:
        # Load dataset with Polars for efficiency
        if dataset_path.is_dir():
            # Directory of parquet files
            lf = pl.scan_parquet(str(dataset_path / "*.parquet"))
        else:
            # Single parquet file
            lf = pl.scan_parquet(str(dataset_path))
        
        # Select only columns we need
        required_cols = ["donor_score", "acceptor_score", "neither_score", "splice_type"]
        lf = lf.select(required_cols)
        
        # Apply sampling if requested
        if sample_size is not None:
            if verbose:
                print(f"[Baseline Calculator] Sampling {sample_size:,} rows for analysis")
            try:
                lf = lf.sample(n=sample_size, seed=42)
                is_sampled = True
                actual_sample_size = sample_size
            except AttributeError:
                # Fallback to limit if sample not available
                lf = lf.limit(sample_size)
                is_sampled = True
                actual_sample_size = sample_size
        else:
            is_sampled = False
            actual_sample_size = None
        
        # Collect data
        if verbose:
            print(f"[Baseline Calculator] Loading data...")
        
        df = lf.collect().to_pandas()
        total_samples = len(df)
        
        if verbose:
            print(f"[Baseline Calculator] Analyzing {total_samples:,} samples")
        
        # Calculate baseline errors
        baseline_errors = _calculate_spliceai_baseline_errors(df)
        
        # Prepare results
        results = {
            'baseline_fp': int(baseline_errors['fp']),
            'baseline_fn': int(baseline_errors['fn']),
            'total_samples': int(total_samples),
            'dataset_name': dataset_name,
            'is_sampled': is_sampled,
            'sample_size': actual_sample_size,
            'baseline_tp': int(baseline_errors['tp']),
            'baseline_tn': int(baseline_errors['tn']),
            'baseline_accuracy': float(baseline_errors['accuracy']),
            'baseline_precision': float(baseline_errors['precision']),
            'baseline_recall': float(baseline_errors['recall']),
            'baseline_f1': float(baseline_errors['f1'])
        }
        
        if verbose:
            print(f"[Baseline Calculator] Results:")
            print(f"  False Positives: {results['baseline_fp']:,}")
            print(f"  False Negatives: {results['baseline_fn']:,}")
            print(f"  Baseline Accuracy: {results['baseline_accuracy']:.3f}")
            print(f"  Baseline F1: {results['baseline_f1']:.3f}")
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"[Baseline Calculator] Error: {e}")
        return {
            'error': str(e),
            'dataset_name': dataset_name,
            'is_sampled': False,
            'sample_size': None
        }


def _calculate_spliceai_baseline_errors(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate SpliceAI baseline errors from dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with columns: donor_score, acceptor_score, neither_score, splice_type
        
    Returns
    -------
    Dict[str, float]
        Dictionary with error counts and metrics
    """
    # Get true labels (ground truth)
    true_splice = df['splice_type'].isin(['donor', 'acceptor'])
    true_neither = ~true_splice
    
    # Get SpliceAI predictions (argmax of scores)
    score_cols = ['donor_score', 'acceptor_score', 'neither_score']
    predicted_class = df[score_cols].idxmax(axis=1)
    
    # Convert predictions to binary (splice vs neither)
    pred_splice = predicted_class.isin(['donor_score', 'acceptor_score'])
    pred_neither = ~pred_splice
    
    # Calculate confusion matrix elements
    tp = (true_splice & pred_splice).sum()  # True splice, predicted splice
    tn = (true_neither & pred_neither).sum()  # True neither, predicted neither
    fp = (true_neither & pred_splice).sum()   # True neither, predicted splice (FALSE POSITIVE)
    fn = (true_splice & pred_neither).sum()   # True splice, predicted neither (FALSE NEGATIVE)
    
    # Calculate metrics
    total = len(df)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def estimate_cv_baseline_errors(
    dataset_path: Union[str, Path],
    cv_stats: Dict[str, Union[int, float]],
    sample_size: Optional[int] = 50000,
    verbose: bool = False
) -> Dict[str, Union[int, float]]:
    """
    Estimate baseline errors for CV dataset based on full dataset analysis.
    
    Parameters
    ----------
    dataset_path : str or Path
        Path to the training dataset used for CV
    cv_stats : Dict
        CV statistics containing information about the CV dataset size
    sample_size : int, optional
        Sample size for baseline error estimation (default: 50,000)
    verbose : bool, default False
        Print progress information
        
    Returns
    -------
    Dict[str, Union[int, float]]
        Enhanced baseline error statistics for CV reporting
    """
    # Calculate baseline errors from dataset
    baseline_stats = calculate_baseline_errors_from_dataset(
        dataset_path=dataset_path,
        sample_size=sample_size,
        verbose=verbose
    )
    
    if 'error' in baseline_stats:
        return baseline_stats
    
    # Estimate CV dataset baseline errors
    # If we have CV row information, scale the baseline errors appropriately
    if 'cv_total_test_rows' in cv_stats and 'total_samples' in baseline_stats:
        cv_rows = cv_stats['cv_total_test_rows']
        sample_rows = baseline_stats['total_samples']
        
        if baseline_stats['is_sampled']:
            # If we used sampling, estimate full dataset errors first
            # then scale to CV dataset size
            if verbose:
                print(f"[Baseline Calculator] Scaling from sample ({sample_rows:,}) to CV dataset ({cv_rows:,})")
            
            # Estimate scaling factor (this is approximate)
            scale_factor = cv_rows / sample_rows
            
        else:
            # If we used full dataset, scale directly to CV size
            scale_factor = cv_rows / sample_rows
        
        # Scale baseline errors to CV dataset size
        cv_baseline_fp = int(baseline_stats['baseline_fp'] * scale_factor)
        cv_baseline_fn = int(baseline_stats['baseline_fn'] * scale_factor)
        
    else:
        # Use the calculated values directly
        cv_baseline_fp = baseline_stats['baseline_fp']
        cv_baseline_fn = baseline_stats['baseline_fn']
    
    # Enhance with CV-specific information
    enhanced_stats = baseline_stats.copy()
    enhanced_stats.update({
        'cv_baseline_fp': cv_baseline_fp,
        'cv_baseline_fn': cv_baseline_fn,
        'has_detailed_analysis': True,
        'is_dynamic_calculation': True,
        'calculation_method': 'dataset_analysis'
    })
    
    return enhanced_stats


def calculate_cv_error_reductions(
    cv_df: pd.DataFrame,
    dataset_path: Union[str, Path],
    sample_size: Optional[int] = 50000,
    verbose: bool = False
) -> Dict[str, Union[int, float]]:
    """
    Calculate comprehensive error reduction statistics for CV results.
    
    This is the main function that should be called from CV metrics generation.
    
    Parameters
    ----------
    cv_df : pd.DataFrame
        CV results dataframe with columns: delta_fp, delta_fn, test_rows
    dataset_path : str or Path
        Path to training dataset
    sample_size : int, optional
        Sample size for baseline calculation (default: 50,000)
    verbose : bool, default False
        Print progress information
        
    Returns
    -------
    Dict[str, Union[int, float]]
        Complete baseline error statistics for CV reporting
    """
    # Calculate CV statistics
    cv_stats = {
        'cv_total_test_rows': cv_df['test_rows'].sum(),
        'total_fp_reduction': cv_df['delta_fp'].sum(),
        'total_fn_reduction': cv_df['delta_fn'].sum(),
    }
    
    # Get baseline error estimates
    baseline_stats = estimate_cv_baseline_errors(
        dataset_path=dataset_path,
        cv_stats=cv_stats,
        sample_size=sample_size,
        verbose=verbose
    )
    
    if 'error' in baseline_stats:
        return baseline_stats
    
    # Calculate reduction percentages
    cv_baseline_fp = baseline_stats.get('cv_baseline_fp', baseline_stats['baseline_fp'])
    cv_baseline_fn = baseline_stats.get('cv_baseline_fn', baseline_stats['baseline_fn'])
    
    fp_reduction_pct = (cv_stats['total_fp_reduction'] / cv_baseline_fp * 100) if cv_baseline_fp > 0 else 0
    fn_reduction_pct = (cv_stats['total_fn_reduction'] / cv_baseline_fn * 100) if cv_baseline_fn > 0 else 0
    
    # Combine all statistics
    final_stats = baseline_stats.copy()
    final_stats.update({
        'actual_fp_reduction': cv_stats['total_fp_reduction'],
        'actual_fn_reduction': cv_stats['total_fn_reduction'],
        'fp_reduction_pct': fp_reduction_pct,
        'fn_reduction_pct': fn_reduction_pct,
        'total_errors_reduced': cv_stats['total_fp_reduction'] + cv_stats['total_fn_reduction'],
        'cv_dataset_rows': cv_stats['cv_total_test_rows']
    })
    
    return final_stats