"""
Prediction combination logic for selective meta-model inference.

This module handles the combination of base model and meta-model predictions
with support for different inference modes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Literal


def combine_predictions_for_complete_coverage(
    base_predictions_df: pd.DataFrame,
    meta_predictions_df: pd.DataFrame,
    uncertainty_threshold_low: float = 0.02,
    uncertainty_threshold_high: float = 0.80,
    inference_mode: Literal["base_only", "hybrid", "meta_only"] = "hybrid"
) -> pd.DataFrame:
    """
    Combine base model and meta-model predictions for complete coverage.
    
    Inference modes:
    - 'base_only': Use only base model predictions (fastest)
    - 'hybrid': Use meta-model for uncertain positions, base for confident ones (default, balanced)
    - 'meta_only': Use meta-model for all positions where available (most comprehensive)
    
    Strategy:
    - Hybrid: Use meta-model predictions for uncertain positions, base for confident positions
    - Base only: Use only base model predictions  
    - Meta only: Use meta-model predictions wherever available
    - Ensure every position has a prediction
    
    Parameters
    ----------
    base_predictions_df : pd.DataFrame
        Base model predictions for all positions
    meta_predictions_df : pd.DataFrame  
        Meta-model predictions for uncertain positions only
    uncertainty_threshold_low : float
        Lower threshold for uncertainty
    uncertainty_threshold_high : float
        Upper threshold for uncertainty
    inference_mode : str
        One of 'base_only', 'hybrid', or 'meta_only'
        
    Returns
    -------
    pd.DataFrame
        Combined predictions with complete coverage
    """
        # Start with all base model predictions
    combined_df = base_predictions_df.copy()
    
    # Initialize prediction source
    combined_df['prediction_source'] = 'base_model'
    
    # Implement inference mode logic
    if inference_mode == "base_only":
        # Use only base model predictions - don't create meta columns at all
        # The meta columns will be the same as base columns for compatibility
        combined_df['donor_meta'] = combined_df['donor_score']
        combined_df['acceptor_meta'] = combined_df['acceptor_score']
        combined_df['neither_meta'] = combined_df['neither_score']
        uncertain_mask = pd.Series([False] * len(combined_df))  # No positions marked as uncertain
        
    elif inference_mode == "meta_only":
        # Initialize meta columns with base values first
        combined_df['donor_meta'] = combined_df['donor_score']
        combined_df['acceptor_meta'] = combined_df['acceptor_score']
        combined_df['neither_meta'] = combined_df['neither_score']
        # Use meta-model for all positions where available
        if not meta_predictions_df.empty:
            uncertain_mask = pd.Series([True] * len(combined_df))  # All positions marked as uncertain
            # CRITICAL FIX: In meta_only mode, we need to ensure ALL positions get meta-model predictions
            # Use the same lookup mechanism as hybrid mode but mark all positions as uncertain
            # The meta_predictions_df should contain predictions for ALL positions in meta_only mode
            # CRITICAL FIX: Always use lookup mechanism for meta_only mode
            # We cannot assume positions are in the same order, even if lengths match
            # The meta_predictions_df should contain predictions for ALL positions in meta_only mode
            if len(meta_predictions_df) != len(combined_df):
                # CRITICAL ERROR: In meta_only mode, we must have complete meta predictions
                raise RuntimeError(
                    f"Meta-only inference mode requires complete meta-model predictions for all positions. "
                    f"Expected {len(combined_df)} positions, but got {len(meta_predictions_df)} meta predictions. "
                    f"This indicates a failure in the meta-model prediction pipeline."
                )
        else:
            # CRITICAL ERROR: In meta_only mode, we cannot fall back to base model
            raise RuntimeError(
                f"Meta-only inference mode requires meta-model predictions, but meta_predictions_df is empty. "
                f"This indicates a complete failure in the meta-model prediction pipeline. "
                f"Cannot fall back to base model in meta_only mode."
            )
            
    else:  # inference_mode == "hybrid" (default)
        # Initialize meta columns with base values first
        combined_df['donor_meta'] = combined_df['donor_score']
        combined_df['acceptor_meta'] = combined_df['acceptor_score']
        combined_df['neither_meta'] = combined_df['neither_score']
        # Identify uncertain positions based on confidence thresholds
        max_score = combined_df[['donor_score', 'acceptor_score']].max(axis=1)
        uncertain_mask = (max_score >= uncertainty_threshold_low) & (max_score < uncertainty_threshold_high)
    
    # Apply meta-model predictions where needed
    if not meta_predictions_df.empty and uncertain_mask.any():
        # Create lookup for efficient updates
        meta_lookup = meta_predictions_df.set_index(['gene_id', 'position'])
        
        # Vectorized update for better performance
        # First, create a multi-index for the combined dataframe
        combined_index = pd.MultiIndex.from_frame(combined_df[['gene_id', 'position']])
        
        # Find positions that have meta predictions
        has_meta = combined_index.isin(meta_lookup.index)
        
        # Update positions that are uncertain AND have meta predictions
        update_mask = uncertain_mask & has_meta
        
        if update_mask.any():
            # Get the positions to update
            update_positions = combined_df.loc[update_mask, ['gene_id', 'position']]
            update_index = pd.MultiIndex.from_frame(update_positions)
            
            # Look up meta predictions
            meta_values = meta_lookup.loc[update_index]
            
            # Update the combined dataframe
            combined_df.loc[update_mask, 'donor_meta'] = meta_values['donor_meta'].values
            combined_df.loc[update_mask, 'acceptor_meta'] = meta_values['acceptor_meta'].values
            combined_df.loc[update_mask, 'neither_meta'] = meta_values['neither_meta'].values
            combined_df.loc[update_mask, 'prediction_source'] = 'meta_model'
            
            # CRITICAL FIX: Never update main score columns - keep original base predictions
            # The performance analysis will use the appropriate columns for each mode:
            # - base_only: uses donor_score + acceptor_score (base predictions)
            # - meta_only: uses donor_meta + acceptor_meta (meta predictions)  
            # - hybrid: uses donor_meta for meta positions, donor_score for base positions
            
            # Keep the original base predictions in main columns for comparison
            # Meta predictions go only to meta columns
    
    # CRITICAL CHECK: In meta_only mode, ensure ALL uncertain positions got meta predictions
    if inference_mode == "meta_only" and uncertain_mask.any():
        # Check if all uncertain positions got meta predictions
        uncertain_positions = uncertain_mask.sum()
        meta_positions = update_mask.sum() if 'update_mask' in locals() else 0
        
        if meta_positions < uncertain_positions:
            missing_positions = uncertain_positions - meta_positions
            raise RuntimeError(
                f"Meta-only inference mode failed: {missing_positions} out of {uncertain_positions} "
                f"uncertain positions did not receive meta-model predictions. "
                f"This indicates incomplete meta-model prediction coverage. "
                f"All positions must have meta predictions in meta_only mode."
            )
    
    # Add uncertainty classification
    combined_df['is_uncertain'] = uncertain_mask
    combined_df['confidence_category'] = 'uncertain'
    
    # Calculate confidence category based on max score for all inference modes
    max_score = combined_df[['donor_score', 'acceptor_score']].max(axis=1)
    combined_df.loc[max_score < uncertainty_threshold_low, 'confidence_category'] = 'confident_non_splice'
    combined_df.loc[max_score >= uncertainty_threshold_high, 'confidence_category'] = 'confident_splice'
    
    return combined_df


def calculate_prediction_statistics(
    combined_predictions: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate statistics from combined predictions.
    
    Parameters
    ----------
    combined_predictions : pd.DataFrame
        Combined predictions dataframe
        
    Returns
    -------
    Dict[str, Any]
        Statistics dictionary
    """
    total_positions = len(combined_predictions)
    meta_positions = (combined_predictions['prediction_source'] == 'meta_model').sum()
    base_positions = (combined_predictions['prediction_source'] == 'base_model').sum()
    
    # Calculate per-category statistics
    confidence_stats = combined_predictions['confidence_category'].value_counts().to_dict()
    
    # Calculate efficiency metrics
    selective_efficiency = base_positions / total_positions if total_positions > 0 else 0
    meta_coverage = meta_positions / total_positions if total_positions > 0 else 0
    
    return {
        'total_positions': total_positions,
        'meta_positions': meta_positions,
        'base_positions': base_positions,
        'selective_efficiency': selective_efficiency,
        'meta_coverage': meta_coverage,
        'confidence_distribution': confidence_stats,
        'uncertain_positions': combined_predictions['is_uncertain'].sum()
    }
