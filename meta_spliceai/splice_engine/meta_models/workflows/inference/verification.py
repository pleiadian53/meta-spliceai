"""
Verification functions for selective meta-model inference.

This module contains all verification logic extracted from selective_meta_inference.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from meta_spliceai.splice_engine.meta_models.builder.preprocessing import (
    METADATA_COLUMNS, 
    LEAKAGE_COLUMNS, 
    SEQUENCE_COLUMNS, 
    REDUNDANT_COLUMNS
)
from .config import SelectiveInferenceConfig


def verify_selective_featurization(
    complete_base_pd: pd.DataFrame,
    uncertain_positions: pd.DataFrame,
    config: SelectiveInferenceConfig,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify that selective featurization is working correctly.
    
    Parameters
    ----------
    complete_base_pd : pd.DataFrame
        Complete base model predictions for all positions
    uncertain_positions : pd.DataFrame
        Positions selected for meta-model featurization
    config : SelectiveInferenceConfig
        Configuration used for inference
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    Dict[str, Any]
        Verification results and statistics
    """
    verification_results = {
        'total_positions': len(complete_base_pd),
        'uncertain_positions': len(uncertain_positions),
        'selective_efficiency': len(uncertain_positions) / len(complete_base_pd) if len(complete_base_pd) > 0 else 0,
        'verification_passed': True,
        'warnings': []
    }
    
    if verbose:
        print(f"\n🔍 VERIFYING SELECTIVE FEATURIZATION")
        print(f"=" * 50)
    
    # 1. Verify that uncertain positions are a subset of all positions
    all_positions = set(zip(complete_base_pd['gene_id'], complete_base_pd['position']))
    uncertain_positions_set = set(zip(uncertain_positions['gene_id'], uncertain_positions['position']))
    
    if not uncertain_positions_set.issubset(all_positions):
        verification_results['verification_passed'] = False
        verification_results['warnings'].append("Uncertain positions contain positions not in complete base predictions")
        if verbose:
            print(f"   ❌ ERROR: Uncertain positions contain positions not in complete base predictions")
    
    # 2. Verify uncertainty thresholds are applied correctly (mode-specific)
    max_scores = np.maximum(
        complete_base_pd['donor_score'].values,
        complete_base_pd['acceptor_score'].values
    )
    
    # Calculate expected uncertain positions based on inference mode
    if config.inference_mode == "base_only":
        # Base_only mode should have 0 uncertain positions (no meta-model recalibration)
        expected_uncertain_count = 0
    elif config.inference_mode == "meta_only":
        # Meta_only mode should have ALL positions as uncertain
        expected_uncertain_count = len(complete_base_pd)
    else:  # hybrid mode
        # Hybrid mode uses threshold-based uncertainty
        expected_uncertain_mask = (
            (max_scores >= config.uncertainty_threshold_low) & 
            (max_scores < config.uncertainty_threshold_high)
        )
        expected_uncertain_count = np.sum(expected_uncertain_mask)
    
    if len(uncertain_positions) != expected_uncertain_count:
        verification_results['verification_passed'] = False
        verification_results['warnings'].append(f"Uncertain position count mismatch: expected {expected_uncertain_count}, got {len(uncertain_positions)}")
        if verbose:
            print(f"   ❌ ERROR: Uncertain position count mismatch")
            print(f"      Expected: {expected_uncertain_count} (mode: {config.inference_mode})")
            print(f"      Actual: {len(uncertain_positions)}")
    
    # 3. Verify selective efficiency is reasonable
    efficiency = verification_results['selective_efficiency']
    if efficiency > 0.5:  # More than 50% of positions are uncertain
        verification_results['warnings'].append(f"High uncertainty rate: {efficiency:.1%} of positions are uncertain")
        if verbose:
            print(f"   ⚠️  WARNING: High uncertainty rate: {efficiency:.1%}")
    
    if verbose:
        print(f"   📊 Total positions: {verification_results['total_positions']:,}")
        print(f"   🎯 Uncertain positions: {verification_results['uncertain_positions']:,}")
        print(f"   ⚡ Selective efficiency: {efficiency:.1%}")
        print(f"   ✅ Verification: {'PASSED' if verification_results['verification_passed'] else 'FAILED'}")
        
        if verification_results['warnings']:
            print(f"   ⚠️  Warnings:")
            for warning in verification_results['warnings']:
                print(f"      - {warning}")
    
    return verification_results


def verify_no_label_leakage(
    feature_data: pd.DataFrame,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify that no label-related columns are present in the feature matrix.
    
    Parameters
    ----------
    feature_data : pd.DataFrame
        Feature matrix for meta-model inference
    verbose : bool, default=True
        Enable verbose output
        
    Returns
    -------
    Dict[str, Any]
        Verification results
    """
    verification_results = {
        'verification_passed': True,
        'leakage_columns': [],
        'warnings': []
    }
    
    if verbose:
        print(f"\n🔒 VERIFYING NO LABEL LEAKAGE")
        print(f"=" * 50)
    
    # Define columns that would constitute label leakage
    leakage_indicators = [
        'splice_type', 'pred_type', 'true_position', 'predicted_position',
        'label', 'is_adjusted', 'prediction_source'
    ]
    
    # Check for leakage columns
    for col in leakage_indicators:
        if col in feature_data.columns:
            verification_results['verification_passed'] = False
            verification_results['leakage_columns'].append(col)
    
    # Check for any columns that might contain label information
    suspicious_patterns = ['true_', 'predicted_', 'label', 'splice_type', 'pred_type']
    for col in feature_data.columns:
        for pattern in suspicious_patterns:
            if pattern in col.lower():
                verification_results['warnings'].append(f"Suspicious column name: {col}")
    
    if verbose:
        print(f"   📊 Feature matrix columns: {len(feature_data.columns)}")
        print(f"   🔍 Leakage columns found: {len(verification_results['leakage_columns'])}")
        
        if verification_results['leakage_columns']:
            print(f"   ❌ LEAKAGE DETECTED:")
            for col in verification_results['leakage_columns']:
                print(f"      - {col}")
        else:
            print(f"   ✅ No label leakage detected")
        
        if verification_results['warnings']:
            print(f"   ⚠️  Warnings:")
            for warning in verification_results['warnings']:
                print(f"      - {warning}")
        
        print(f"   ✅ Verification: {'PASSED' if verification_results['verification_passed'] else 'FAILED'}")
    
    return verification_results


def get_excluded_columns_for_inference(feature_manifest_features: Optional[List[str]] = None) -> List[str]:
    """
    Get columns that should be excluded from feature matrices during inference.
    
    This function uses the feature manifest as the source of truth. It only excludes
    columns that are both in the candidate exclusion lists AND not present in the
    feature manifest (meaning they weren't used in training).
    
    Parameters
    ----------
    feature_manifest_features : List[str], optional
        List of features from the feature manifest.
        If None, returns all candidate exclusion columns.
    
    Returns
    -------
    List[str]
        List of column names to exclude from feature matrices
    """
    # Use standardized column definitions from preprocessing module
    candidate_exclusions = []
    candidate_exclusions.extend(METADATA_COLUMNS)
    candidate_exclusions.extend(LEAKAGE_COLUMNS)
    candidate_exclusions.extend(SEQUENCE_COLUMNS)
    candidate_exclusions.extend(REDUNDANT_COLUMNS)
    
    # Add inference-specific candidate exclusions
    inference_specific_candidates = [
        "true_position",
        "predicted_position", 
        "label",
        "prediction_source",
        "chromosome",  # Alternative name for 'chrom'
        "donor_score",  # Base scores (not used as features during inference)
        "acceptor_score",
        "neither_score"
    ]
    candidate_exclusions.extend(inference_specific_candidates)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for col in candidate_exclusions:
        if col not in seen:
            seen.add(col)
            unique_candidates.append(col)
    
    # If no feature manifest provided, return all candidates (conservative approach)
    if feature_manifest_features is None:
        return unique_candidates
    
    # Otherwise, only exclude columns that are NOT in the feature manifest
    # (meaning they weren't used in training)
    excluded_columns = []
    for col in unique_candidates:
        if col not in feature_manifest_features:
            excluded_columns.append(col)
    
    return excluded_columns












