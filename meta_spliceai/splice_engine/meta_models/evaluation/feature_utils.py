#!/usr/bin/env python3
"""
Feature selection and filtering utilities for model training and evaluation.

This module provides functions for identifying and filtering potentially
problematic features in machine learning pipelines, particularly for
splice site prediction models.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Set

import numpy as np
import pandas as pd


def load_excluded_features(file_path: Union[str, Path]) -> List[str]:
    """
    Load list of features to exclude from a text file.
    
    Each line in the file is treated as a feature name.
    Comments starting with # are ignored.
    
    Args:
        file_path: Path to the file containing features to exclude
        
    Returns:
        List of feature names to exclude
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    # Search order: specified path, package configs, project root configs
    search_paths = [
        file_path,
        Path(__file__).parent.parent.parent / "training" / "configs" / file_path.name,
        Path.cwd() / "configs" / file_path.name
    ]
    
    # Try to find the file in any of the search paths
    found_path = None
    for path in search_paths:
        if path.exists():
            print(f"[INFO] Found feature exclusion file: {path}")
            found_path = path
            break
    else:
        print(f"[WARNING] Feature exclusion file not found in any search path: {file_path}")
        return []
    
    with open(found_path, 'r') as f:
        # Read lines, strip whitespace, filter out comments and empty lines
        excluded = [line.strip() for line in f.readlines()]
        excluded = [line for line in excluded if line and not line.startswith('#')]
    
    print(f"[INFO] Loaded {len(excluded)} features to exclude from {file_path}")
    return excluded


def filter_features(
    X: np.ndarray, 
    feature_names: List[str],
    excluded_features: Optional[Union[List[str], Set[str]]] = None,
    correlation_threshold: Optional[float] = None,
    y: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Filter features from a feature matrix based on exclusion criteria.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        feature_names: List of feature names corresponding to X columns
        excluded_features: Optional list of feature names to exclude
        correlation_threshold: Optional threshold for correlation with target
        y: Target values, required if correlation_threshold is provided
        
    Returns:
        Tuple of (filtered_X, filtered_feature_names)
    """
    if len(feature_names) != X.shape[1]:
        raise ValueError(f"Feature names length ({len(feature_names)}) doesn't match "
                         f"X shape[1] ({X.shape[1]})")
    
    # Start with all features
    keep_indices = list(range(len(feature_names)))
    filtered_names = feature_names.copy()
    
    # Filter by exclusion list
    if excluded_features:
        excluded_set = set(excluded_features)
        # Find features that match our exclusion list
        exclude_indices = []
        excluded_feature_names = []
        for i, name in enumerate(feature_names):
            if name in excluded_set:
                exclude_indices.append(i)
                excluded_feature_names.append(name)
        
        if exclude_indices:
            print(f"\n[INFO] Excluding {len(exclude_indices)} features from exclusion list:")
            # Show the first 10 excluded features, then a count if there are more
            for i, name in enumerate(excluded_feature_names[:10]):
                print(f"  - {name}")
            if len(excluded_feature_names) > 10:
                print(f"  - ... and {len(excluded_feature_names) - 10} more features")
                
            # Create mask of features to keep
            keep_mask = np.ones(len(feature_names), dtype=bool)
            keep_mask[exclude_indices] = False
            # Update indices and names
            keep_indices = [i for i in range(len(feature_names)) if keep_mask[i]]
            filtered_names = [name for i, name in enumerate(feature_names) if keep_mask[i]]
        else:
            print(f"\n[INFO] No features matched the exclusion list ({len(excluded_set)} patterns provided)")
    
    # Filter by correlation threshold
    if correlation_threshold is not None and y is not None:
        high_corr_features = []
        for i in keep_indices[:]:  # Copy to avoid modifying during iteration
            if len(np.unique(X[:, i])) <= 1:
                # Skip constant features
                continue
            corr = abs(np.corrcoef(X[:, i], y)[0, 1])
            if corr >= correlation_threshold:
                high_corr_features.append((feature_names[i], corr))
                keep_indices.remove(i)
        
        if high_corr_features:
            print(f"[INFO] Excluded {len(high_corr_features)} features with correlation >= {correlation_threshold}:")
            for name, corr in high_corr_features:
                print(f"  - {name}: correlation = {corr:.4f}")
            # Update filtered names
            filtered_names = [feature_names[i] for i in keep_indices]
    
    # Filter X matrix
    filtered_X = X[:, keep_indices]
    
    print(f"[INFO] Original features: {X.shape[1]}, Filtered features: {filtered_X.shape[1]}")
    return filtered_X, filtered_names


def save_feature_importance(
    importance_scores: np.ndarray,
    feature_names: List[str],
    output_path: Union[str, Path],
    n_top: int = 20,
) -> pd.DataFrame:
    """
    Save feature importance scores to CSV and return as DataFrame.
    
    Args:
        importance_scores: Array of feature importance scores
        feature_names: List of feature names corresponding to importance scores
        output_path: Path to save the feature importance CSV
        n_top: Number of top features to print
        
    Returns:
        DataFrame containing feature names and importance scores
    """
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    
    # Create DataFrame and sort by importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Save to CSV
    os.makedirs(output_path.parent, exist_ok=True)
    importance_df.to_csv(output_path, index=False)
    
    # Print top features
    print(f"\nTop {n_top} important features:")
    for i, (_, row) in enumerate(importance_df.head(n_top).iterrows()):
        print(f"{i+1:2d}. {row['feature']:30s} {row['importance']:.4f}")
    
    return importance_df
