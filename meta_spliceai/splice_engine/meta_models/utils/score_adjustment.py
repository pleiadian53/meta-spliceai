"""
Score vector adjustment utilities - Version 2 with correlated probability vectors.

This version correctly handles the fact that (donor, acceptor, neither) scores at each
position are CORRELATED and must sum to 1.0. The adjustment shifts the ENTIRE probability
vector together, not individual score types independently.

Key Insight:
-----------
For each position, we have a probability distribution: (donor, acceptor, neither) that sums to 1.0.
When we adjust coordinates, we must shift this ENTIRE distribution as a unit to maintain the
probability constraint.

Different splice types have different adjustments, so we create separate "views":
- Donor view: All three scores shifted by donor adjustment
- Acceptor view: All three scores shifted by acceptor adjustment

When evaluating donors, use the donor view. When evaluating acceptors, use the acceptor view.
"""

import numpy as np
import polars as pl
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def shift_probability_vectors(
    donor_scores: np.ndarray,
    acceptor_scores: np.ndarray,
    neither_scores: np.ndarray,
    shift_amount: int,
    fill_value: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shift entire probability vectors together to maintain correlation.
    
    This function shifts ALL THREE score arrays by the same amount, ensuring
    that the probability constraint (sum = 1.0) is maintained at each position.
    
    Parameters
    ----------
    donor_scores : np.ndarray
        1D array of donor scores
    acceptor_scores : np.ndarray
        1D array of acceptor scores
    neither_scores : np.ndarray
        1D array of neither scores
    shift_amount : int
        Number of positions to shift:
        - Positive: shift toward 3' (position i gets values from i+shift)
        - Negative: shift toward 5' (position i gets values from i-shift)
    fill_value : float, optional
        Value for positions that shift out of bounds, by default 0.0
        Note: For probability vectors, we typically fill with (0, 0, 1) 
        meaning "neither" for out-of-bounds positions
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Shifted (donor, acceptor, neither) scores
    
    Examples
    --------
    >>> donor = np.array([0.1, 0.2, 0.9, 0.3, 0.1])
    >>> acceptor = np.array([0.2, 0.7, 0.05, 0.1, 0.2])
    >>> neither = np.array([0.7, 0.1, 0.05, 0.6, 0.7])
    >>> 
    >>> # Shift by +2: position i gets values from position i+2
    >>> d, a, n = shift_probability_vectors(donor, acceptor, neither, shift_amount=2)
    >>> # Position 0 now has the probability distribution from position 2
    >>> d[0], a[0], n[0]  # Should be (0.9, 0.05, 0.05)
    """
    n = len(donor_scores)
    
    if shift_amount == 0:
        return donor_scores.copy(), acceptor_scores.copy(), neither_scores.copy()
    
    # Initialize with fill values (default: neither=1.0, others=0.0 for out-of-bounds)
    shifted_donor = np.full(n, fill_value, dtype=donor_scores.dtype)
    shifted_acceptor = np.full(n, fill_value, dtype=acceptor_scores.dtype)
    shifted_neither = np.full(n, 1.0 - 2*fill_value, dtype=neither_scores.dtype)  # Ensure sum=1
    
    if shift_amount > 0:
        # Positive shift: position i gets values from position i+shift_amount
        if shift_amount < n:
            shifted_donor[:-shift_amount] = donor_scores[shift_amount:]
            shifted_acceptor[:-shift_amount] = acceptor_scores[shift_amount:]
            shifted_neither[:-shift_amount] = neither_scores[shift_amount:]
    else:
        # Negative shift: position i gets values from position i+shift_amount (shift_amount is negative)
        shift_amount = abs(shift_amount)
        if shift_amount < n:
            shifted_donor[shift_amount:] = donor_scores[:-shift_amount]
            shifted_acceptor[shift_amount:] = acceptor_scores[:-shift_amount]
            shifted_neither[shift_amount:] = neither_scores[:-shift_amount]
    
    return shifted_donor, shifted_acceptor, shifted_neither


def create_splice_type_views(
    donor_scores: np.ndarray,
    acceptor_scores: np.ndarray,
    neither_scores: np.ndarray,
    strand: str,
    adjustment_dict: Dict[str, Dict[str, int]],
    verbose: bool = False
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create separate coordinate-adjusted views for each splice type.
    
    This function creates three "views" of the data:
    1. Donor view: All scores shifted by donor adjustment
    2. Acceptor view: All scores shifted by acceptor adjustment
    3. Neither view: No shift (or minimal shift)
    
    When evaluating donor sites, use the donor view.
    When evaluating acceptor sites, use the acceptor view.
    
    Parameters
    ----------
    donor_scores : np.ndarray
        Original donor scores
    acceptor_scores : np.ndarray
        Original acceptor scores
    neither_scores : np.ndarray
        Original neither scores
    strand : str
        '+' or '-'
    adjustment_dict : Dict[str, Dict[str, int]]
        Adjustment values for each type and strand
    verbose : bool, optional
        Whether to print detailed logs
    
    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Dictionary with keys 'donor_view', 'acceptor_view', 'neither_view'
        Each value is a tuple of (donor_scores, acceptor_scores, neither_scores)
    
    Examples
    --------
    >>> adj_dict = {'donor': {'plus': 2, 'minus': 1}, 'acceptor': {'plus': 0, 'minus': -1}}
    >>> views = create_splice_type_views(donor, acceptor, neither, '+', adj_dict)
    >>> 
    >>> # When checking if position X is a donor, use:
    >>> donor_view_scores = views['donor_view'][0]  # donor scores from donor view
    >>> 
    >>> # When checking if position X is an acceptor, use:
    >>> acceptor_view_scores = views['acceptor_view'][1]  # acceptor scores from acceptor view
    """
    # Get adjustment values for this strand
    donor_adj = adjustment_dict['donor']['plus'] if strand == '+' else adjustment_dict['donor']['minus']
    acceptor_adj = adjustment_dict['acceptor']['plus'] if strand == '+' else adjustment_dict['acceptor']['minus']
    
    if verbose:
        logger.info(f"Creating splice type views for {strand} strand:")
        logger.info(f"  Donor adjustment: {donor_adj:+d} (model predicts {donor_adj}nt from true site)")
        logger.info(f"  Acceptor adjustment: {acceptor_adj:+d} (model predicts {acceptor_adj}nt from true site)")
    
    # CRITICAL: Negate adjustments (positive = upstream = need to shift backward)
    donor_shift = -donor_adj
    acceptor_shift = -acceptor_adj
    
    if verbose:
        logger.info(f"  → Donor view: shift by {donor_shift}")
        logger.info(f"  → Acceptor view: shift by {acceptor_shift}")
    
    # Create donor view: shift entire probability vector by donor adjustment
    donor_view = shift_probability_vectors(
        donor_scores, acceptor_scores, neither_scores,
        shift_amount=donor_shift,
        fill_value=0.0
    )
    
    # Create acceptor view: shift entire probability vector by acceptor adjustment
    acceptor_view = shift_probability_vectors(
        donor_scores, acceptor_scores, neither_scores,
        shift_amount=acceptor_shift,
        fill_value=0.0
    )
    
    # Neither view: typically no shift (use original scores)
    neither_view = (
        donor_scores.copy(),
        acceptor_scores.copy(),
        neither_scores.copy()
    )
    
    if verbose:
        # Verify probability constraints
        for view_name, view in [('donor', donor_view), ('acceptor', acceptor_view)]:
            sums = view[0] + view[1] + view[2]
            sum_ok = np.allclose(sums, 1.0, atol=0.01)
            logger.info(f"  {view_name.capitalize()} view: sum check {'✓' if sum_ok else '✗'} "
                       f"(min={sums.min():.3f}, max={sums.max():.3f}, mean={sums.mean():.3f})")
    
    return {
        'donor_view': donor_view,
        'acceptor_view': acceptor_view,
        'neither_view': neither_view
    }


def create_unified_adjusted_scores(
    donor_scores: np.ndarray,
    acceptor_scores: np.ndarray,
    neither_scores: np.ndarray,
    strand: str,
    adjustment_dict: Dict[str, Dict[str, int]],
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create unified adjusted scores by combining splice-type-specific views.
    
    This function creates a single set of adjusted scores where:
    - At positions predicted to be donors, use the donor view
    - At positions predicted to be acceptors, use the acceptor view
    - At other positions, use the original scores
    
    Parameters
    ----------
    donor_scores : np.ndarray
        Original donor scores
    acceptor_scores : np.ndarray
        Original acceptor scores  
    neither_scores : np.ndarray
        Original neither scores
    strand : str
        '+' or '-'
    adjustment_dict : Dict[str, Dict[str, int]]
        Adjustment values
    verbose : bool
        Whether to log details
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Unified adjusted (donor, acceptor, neither) scores
    
    Notes
    -----
    This approach determines the "dominant" splice type at each position
    and uses the appropriate view for that position.
    """
    # Create all views
    views = create_splice_type_views(
        donor_scores, acceptor_scores, neither_scores,
        strand, adjustment_dict, verbose
    )
    
    # Determine dominant type at each position (based on ORIGINAL scores)
    max_scores = np.maximum(np.maximum(donor_scores, acceptor_scores), neither_scores)
    is_donor = (donor_scores == max_scores) & (donor_scores > 0.1)  # Threshold to avoid noise
    is_acceptor = (acceptor_scores == max_scores) & (acceptor_scores > 0.1)
    
    # Initialize with neither view (default)
    adjusted_donor = views['neither_view'][0].copy()
    adjusted_acceptor = views['neither_view'][1].copy()
    adjusted_neither = views['neither_view'][2].copy()
    
    # Use donor view where donors are predicted
    donor_view = views['donor_view']
    adjusted_donor = np.where(is_donor, donor_view[0], adjusted_donor)
    adjusted_acceptor = np.where(is_donor, donor_view[1], adjusted_acceptor)
    adjusted_neither = np.where(is_donor, donor_view[2], adjusted_neither)
    
    # Use acceptor view where acceptors are predicted
    acceptor_view = views['acceptor_view']
    adjusted_donor = np.where(is_acceptor, acceptor_view[0], adjusted_donor)
    adjusted_acceptor = np.where(is_acceptor, acceptor_view[1], adjusted_acceptor)
    adjusted_neither = np.where(is_acceptor, acceptor_view[2], adjusted_neither)
    
    if verbose:
        n_donor_adjusted = np.sum(is_donor)
        n_acceptor_adjusted = np.sum(is_acceptor)
        logger.info(f"  Unified adjustment: {n_donor_adjusted} donor positions, "
                   f"{n_acceptor_adjusted} acceptor positions")
    
    return adjusted_donor, adjusted_acceptor, adjusted_neither


def adjust_predictions_dataframe_v2(
    predictions_df: pl.DataFrame,
    adjustment_dict: Dict[str, Dict[str, int]],
    score_columns: Tuple[str, str, str] = ('donor_prob', 'acceptor_prob', 'neither_prob'),
    position_column: str = 'position',
    strand_column: str = 'strand',
    method: str = 'multi_view',
    verbose: bool = False
) -> pl.DataFrame:
    """
    Apply coordinate adjustments with correlated probability vectors.
    
    CRITICAL APPROACH (2025-10-31):
    ================================
    Creates MULTIPLE VIEWS by adding separate columns for each splice type's adjusted scores.
    This allows evaluation code to use the appropriate view for each splice type.
    
    Output columns:
    - donor_prob, acceptor_prob, neither_prob (original, unadjusted)
    - donor_prob_donor_view, acceptor_prob_donor_view, neither_prob_donor_view
    - donor_prob_acceptor_view, acceptor_prob_acceptor_view, neither_prob_acceptor_view
    
    When evaluating:
    - For donors: use *_donor_view columns
    - For acceptors: use *_acceptor_view columns
    
    Parameters
    ----------
    predictions_df : pl.DataFrame
        DataFrame with predictions
    adjustment_dict : Dict[str, Dict[str, int]]
        Adjustment dictionary
    score_columns : Tuple[str, str, str]
        Names of (donor, acceptor, neither) columns
    position_column : str
        Position column name
    strand_column : str
        Strand column name
    method : str
        'multi_view' (default) - adds separate columns for each view
        'unified' - single set of adjusted scores (DEPRECATED, too conservative)
        'views' - returns dict with separate DataFrames (for debugging)
    verbose : bool
        Whether to log details
    
    Returns
    -------
    pl.DataFrame or Dict
        Adjusted predictions with multiple view columns
    """
    donor_col, acceptor_col, neither_col = score_columns
    
    # Validate
    required_cols = [position_column, strand_column, donor_col, acceptor_col, neither_col]
    missing = [col for col in required_cols if col not in predictions_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Verify single strand
    unique_strands = predictions_df[strand_column].unique().to_list()
    if len(unique_strands) != 1:
        raise ValueError(f"DataFrame must contain single strand. Found: {unique_strands}")
    
    strand = unique_strands[0]
    
    if verbose:
        logger.info(f"Adjusting {len(predictions_df):,} positions for {strand} strand (method={method})")
    
    # Sort by position
    predictions_df = predictions_df.sort(position_column)
    
    # Extract scores
    donor_scores = predictions_df[donor_col].to_numpy()
    acceptor_scores = predictions_df[acceptor_col].to_numpy()
    neither_scores = predictions_df[neither_col].to_numpy()
    
    # Create splice type views
    views = create_splice_type_views(
        donor_scores, acceptor_scores, neither_scores,
        strand, adjustment_dict, verbose
    )
    
    if method == 'multi_view':
        # Add separate columns for each view
        # This allows evaluation code to use the appropriate view
        donor_view = views['donor_view']
        acceptor_view = views['acceptor_view']
        
        predictions_df = predictions_df.with_columns([
            # Donor view columns
            pl.Series(name=f'{donor_col}_donor_view', values=donor_view[0]),
            pl.Series(name=f'{acceptor_col}_donor_view', values=donor_view[1]),
            pl.Series(name=f'{neither_col}_donor_view', values=donor_view[2]),
            
            # Acceptor view columns
            pl.Series(name=f'{donor_col}_acceptor_view', values=acceptor_view[0]),
            pl.Series(name=f'{acceptor_col}_acceptor_view', values=acceptor_view[1]),
            pl.Series(name=f'{neither_col}_acceptor_view', values=acceptor_view[2])
        ])
        
        if verbose:
            logger.info(f"✅ Multi-view adjustment complete (added 6 view columns)")
        
        return predictions_df
    
    elif method == 'unified':
        # DEPRECATED: Too conservative, only adjusts high-scoring positions
        logger.warning("⚠️  'unified' method is deprecated. Use 'multi_view' instead.")
        adj_donor, adj_acceptor, adj_neither = create_unified_adjusted_scores(
            donor_scores, acceptor_scores, neither_scores,
            strand, adjustment_dict, verbose
        )
        
        predictions_df = predictions_df.with_columns([
            pl.Series(name=donor_col, values=adj_donor),
            pl.Series(name=acceptor_col, values=adj_acceptor),
            pl.Series(name=neither_col, values=adj_neither)
        ])
        
        if verbose:
            logger.info(f"✅ Unified adjustment complete")
        
        return predictions_df
    
    elif method == 'views':
        # Return separate DataFrames (for debugging/analysis)
        result = {}
        for view_name, (d, a, n) in views.items():
            view_df = predictions_df.clone()
            view_df = view_df.with_columns([
                pl.Series(name=donor_col, values=d),
                pl.Series(name=acceptor_col, values=a),
                pl.Series(name=neither_col, values=n)
            ])
            result[view_name] = view_df
        
        if verbose:
            logger.info(f"✅ Created {len(result)} views")
        
        return result
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'multi_view', 'unified', or 'views'")


# Test function
def test_score_adjustment_v2():
    """Test the v2 score adjustment with correlated vectors."""
    print("="*80)
    print("TESTING SCORE ADJUSTMENT V2 - Correlated Probability Vectors")
    print("="*80)
    
    # Test data
    donor = np.array([0.1, 0.2, 0.9, 0.3, 0.1])
    acceptor = np.array([0.2, 0.7, 0.05, 0.1, 0.2])
    neither = np.array([0.7, 0.1, 0.05, 0.6, 0.7])
    
    print("\n1. Original Scores")
    print("-" * 40)
    print(f"Donor:    {donor}")
    print(f"Acceptor: {acceptor}")
    print(f"Neither:  {neither}")
    print(f"Sum:      {donor + acceptor + neither}")
    
    # Test shift
    print("\n2. Shift Probability Vectors by +2")
    print("-" * 40)
    d, a, n = shift_probability_vectors(donor, acceptor, neither, shift_amount=2)
    print(f"Donor:    {d}")
    print(f"Acceptor: {a}")
    print(f"Neither:  {n}")
    print(f"Sum:      {d + a + n}")
    print(f"Position 0 got distribution from position 2: ({d[0]:.2f}, {a[0]:.2f}, {n[0]:.2f})")
    print(f"Expected: (0.90, 0.05, 0.05) ✓" if np.allclose([d[0], a[0], n[0]], [0.9, 0.05, 0.05], atol=0.01) else "✗ FAILED")
    
    # Test views
    print("\n3. Create Splice Type Views")
    print("-" * 40)
    adj_dict = {'donor': {'plus': 2, 'minus': 1}, 'acceptor': {'plus': 0, 'minus': -1}}
    views = create_splice_type_views(donor, acceptor, neither, '+', adj_dict, verbose=True)
    
    print("\n4. Unified Adjustment")
    print("-" * 40)
    adj_d, adj_a, adj_n = create_unified_adjusted_scores(donor, acceptor, neither, '+', adj_dict, verbose=True)
    print(f"Adjusted donor:    {adj_d}")
    print(f"Adjusted acceptor: {adj_a}")
    print(f"Adjusted neither:  {adj_n}")
    print(f"Sum:               {adj_d + adj_a + adj_n}")
    
    print("\n" + "="*80)
    print("✅ V2 TESTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_score_adjustment_v2()

