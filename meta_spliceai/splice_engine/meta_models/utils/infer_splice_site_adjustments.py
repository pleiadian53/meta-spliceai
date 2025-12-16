"""
Automated adjustment detection for splice site annotation inconsistencies.

This module provides utilities to automatically detect and apply adjustments
needed to align predicted splice sites with annotation data.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Union, Any

from meta_spliceai.splice_engine.meta_models.core.enhanced_evaluation import normalize_strand

# Related files:
# infer_splice_site_adjustments.py - Core position adjustment logic
# verify_splice_adjustment.py - Validation utilities
# analyze_splice_adjustment.py - Visualization and analysis tools


# Port of the adjust_scores function with the same behavior
def adjust_scores(scores, strand, splice_type, is_neither_prob=False, normalize_edges=True):
    """
    Adjust splice site scores based on splice type- and strand-specific systematic discrepancies.

    Parameters:
    - scores (np.ndarray): The array of splice site probabilities (donor or acceptor).
    - strand (str): The strand of the gene ('+' or '-').
    - splice_type (str): The type of splice site ('donor' or 'acceptor').
    - is_neither_prob (bool): Whether this array represents 'neither' probabilities.
    - normalize_edges (bool): Whether to normalize edge positions (setting neither=1).

    Returns:
    - adjusted_scores (np.ndarray): The adjusted splice site probabilities.

    Memo: 
    - SpliceAI systematically predicts donor sites on + strand 2nt upstream of the true position
    - SpliceAI predicts donor sites on - strand 1nt upstream of the true position
    - SpliceAI predicts acceptor sites on + strand at the exact position (no offset)
    - SpliceAI predicts acceptor sites on - strand 1nt downstream of the true position
    """
    adjusted_scores = scores.copy()

    if splice_type == 'donor':
        if strand == '+':
            adjusted_scores = np.roll(adjusted_scores, 2)  # Shift forward by 2nt
            if normalize_edges:
                if is_neither_prob:
                    adjusted_scores[:2] = 1.0  # Set edge neither probabilities to 1.0
                else:
                    adjusted_scores[:2] = 0  # Set edge donor/acceptor probabilities to 0
            else:
                adjusted_scores[:2] = 0  # Reset wrapped-around values without normalization
        elif strand == '-':
            adjusted_scores = np.roll(adjusted_scores, 1)  # Shift forward by 1nt
            if normalize_edges:
                if is_neither_prob:
                    adjusted_scores[:1] = 1.0  # Set edge neither probabilities to 1.0
                else:
                    adjusted_scores[:1] = 0  # Set edge donor/acceptor probabilities to 0
            else:
                adjusted_scores[:1] = 0  # Reset wrapped-around values without normalization
        else:
            raise ValueError(f"Invalid strand value: {strand}")
    elif splice_type == 'acceptor':
        if strand == '+':
            # No need to roll for + strand acceptor sites in SpliceAI
            pass
        elif strand == '-':
            adjusted_scores = np.roll(adjusted_scores, -1)  # Shift backward by 1nt
            if normalize_edges:
                if is_neither_prob:
                    adjusted_scores[-1:] = 1.0  # Set edge neither probabilities to 1.0
                else:
                    adjusted_scores[-1:] = 0  # Set edge donor/acceptor probabilities to 0
            else:
                adjusted_scores[-1:] = 0  # Reset wrapped-around values without normalization
        else:
            raise ValueError(f"Invalid strand value: {strand}")
    else:
        raise ValueError(f"Invalid splice type: {splice_type}")

    return adjusted_scores


def apply_custom_splice_site_adjustments(scores, strand, splice_type, adjustment_dict, 
                                         is_neither_prob=False, normalize_edges=True):
    """
    Apply custom splice site adjustments based on the provided adjustment dictionary.
    
    Parameters:
    - scores (np.ndarray): The array of splice site probabilities (donor or acceptor).
    - strand (str): The strand of the gene ('+' or '-').
    - splice_type (str): The type of splice site ('donor' or 'acceptor').
    - adjustment_dict: Dictionary with adjustment values for each strand and site type.
                     Format: {'donor': {'plus': offset, 'minus': offset}, 
                              'acceptor': {'plus': offset, 'minus': offset}}
    - is_neither_prob (bool): Whether this array represents 'neither' probabilities.
    - normalize_edges (bool): Whether to normalize edge positions (setting neither=1).
    
    Returns:
    - adjusted_scores (np.ndarray): The adjusted splice site probabilities.
    """
    # Convert to numpy array if it's a list
    if isinstance(scores, list):
        scores = np.array(scores)
    
    # Make a copy to avoid modifying the original
    adjusted_scores = scores.copy()
    
    # No adjustments if adjustment_dict is not provided
    if adjustment_dict is None:
        return adjusted_scores
    
    # Normalize strand and get the corresponding key
    norm_strand = normalize_strand(strand)
    strand_key = 'plus' if norm_strand == '+' else 'minus'
    
    # Check if we have an adjustment for this site type and strand
    if splice_type in adjustment_dict and strand_key in adjustment_dict[splice_type]:
        offset = adjustment_dict[splice_type][strand_key]
        
        # Skip if no adjustment needed
        if offset == 0:
            return adjusted_scores
        
        # Apply the roll/shift
        adjusted_scores = np.roll(adjusted_scores, offset)
        
        # Zero out the wrapped-around values
        if offset > 0:
            if normalize_edges and is_neither_prob:
                # For neither probabilities at the edge, set to 1.0
                adjusted_scores[:offset] = 1.0
            else:
                # For donor or acceptor probabilities at the edge, set to 0
                adjusted_scores[:offset] = 0
        elif offset < 0:
            if normalize_edges and is_neither_prob:
                # For neither probabilities at the edge, set to 1.0
                adjusted_scores[offset:] = 1.0
            else:
                # For donor or acceptor probabilities at the edge, set to 0
                adjusted_scores[offset:] = 0
    
    return adjusted_scores


def empirical_infer_splice_site_adjustments(
    annotations_df, 
    pred_results, 
    search_range=(-5, 5),  # Range of offsets to search (inclusive)
    min_genes_per_category=3,  # Minimum number of genes required in each strand+type category
    consensus_window=2, 
    probability_threshold=0.4,
    min_tp_improvement=0.2,  # Minimum improvement in TP rate required to accept an adjustment
    verbose=False
):
    """
    Empirically infer the optimal position adjustments for splice sites by analyzing 
    prediction results across multiple genes, strands, and splice types.
    
    This function tests different position adjustments within the search range and
    identifies which adjustments maximize the match between predicted and true splice
    sites for each strand and splice type combination.
    
    Parameters:
    -----------
    annotations_df : pl.DataFrame
        DataFrame with true splice site annotations
    pred_results : dict
        Dictionary with predicted splice probabilities
    search_range : tuple(int, int)
        Range of offsets to search (min, max, inclusive)
    min_genes_per_category : int
        Minimum number of genes required for each strand+type combination
    consensus_window : int
        Size of window to consider predictions matching true positions
    probability_threshold : float
        Minimum probability threshold to consider (to filter out noise)
    min_tp_improvement : float
        Minimum improvement in TP rate required to accept an adjustment
    verbose : bool
        Whether to print verbose output
    
    Returns:
    --------
    tuple(dict, dict): 
        - Dictionary with inferred adjustments:
            {
                'donor': {'plus': offset_value, 'minus': offset_value},
                'acceptor': {'plus': offset_value, 'minus': offset_value}
            }
        - Dictionary with detailed statistics for each tested adjustment
    """
    import copy
    from meta_spliceai.splice_engine.meta_models.core.enhanced_evaluation import enhanced_evaluate_splice_site_errors
    
    if verbose:
        print(f"\n[info] Starting empirical splice site adjustment inference")
        print(f"[info] Analyzing {len(pred_results)} genes")
        print(f"[info] Testing offsets in range: {search_range[0]} to {search_range[1]}")
    
    # Initialize results
    gene_ids = sorted(pred_results.keys())
    
    # Extract data about which genes have which strands and splice types
    gene_metadata = {}
    for gene_id in gene_ids:
        # Get strand for this gene
        gene_annots = annotations_df.filter(pl.col("gene_id") == gene_id)
        if gene_annots.height == 0:
            continue
            
        strands = gene_annots["strand"].unique().to_list()
        if len(strands) > 1:
            if verbose:
                print(f"[warning] Gene {gene_id} has multiple strands: {strands}. Using first one.")
            strand = strands[0]
        else:
            strand = strands[0]
            
        # Get splice types for this gene
        splice_types = gene_annots["site_type"].unique().to_list()
        
        # Store metadata
        gene_metadata[gene_id] = {
            "strand": strand,
            "splice_types": splice_types
        }
    
    # Group genes by strand and check if we have enough for each category
    plus_strand_genes = [g for g, meta in gene_metadata.items() if meta["strand"] == "+"]
    minus_strand_genes = [g for g, meta in gene_metadata.items() if meta["strand"] == "-"]
    
    donor_genes = [g for g, meta in gene_metadata.items() if "donor" in meta["splice_types"]]
    acceptor_genes = [g for g, meta in gene_metadata.items() if "acceptor" in meta["splice_types"]]
    
    # Cross-reference to get genes in each category
    donor_plus_genes = [g for g in donor_genes if g in plus_strand_genes]
    donor_minus_genes = [g for g in donor_genes if g in minus_strand_genes]
    acceptor_plus_genes = [g for g in acceptor_genes if g in plus_strand_genes]
    acceptor_minus_genes = [g for g in acceptor_genes if g in minus_strand_genes]
    
    categories = {
        "donor_plus": donor_plus_genes,
        "donor_minus": donor_minus_genes,
        "acceptor_plus": acceptor_plus_genes,
        "acceptor_minus": acceptor_minus_genes
    }
    
    if verbose:
        print("\n[info] Gene counts by category:")
        for category, genes in categories.items():
            print(f"  {category}: {len(genes)} genes")
    
    # Check if we have enough genes in each category
    insufficient_categories = [cat for cat, genes in categories.items() if len(genes) < min_genes_per_category]
    if insufficient_categories:
        print(f"[warning] Insufficient genes in categories: {insufficient_categories}")
        print(f"[warning] Consider adding more target genes or reducing min_genes_per_category (currently {min_genes_per_category})")
    
    # Define strand and site combinations to test
    combinations = [
        {"splice_type": "donor", "strand": "+"},
        {"splice_type": "donor", "strand": "-"},
        {"splice_type": "acceptor", "strand": "+"},
        {"splice_type": "acceptor", "strand": "-"}
    ]
    
    # Initialize results tracking
    adjustment_stats = {
        "donor": {"plus": {}, "minus": {}},
        "acceptor": {"plus": {}, "minus": {}}
    }
    
    # Test each possible adjustment for each combination
    for combo in combinations:
        splice_type = combo["splice_type"]
        strand = combo["strand"]
        strand_key = "plus" if strand == "+" else "minus"
        
        category_genes = categories[f"{splice_type}_{strand_key}"]
        if len(category_genes) < min_genes_per_category:
            if verbose:
                print(f"[warning] Skipping {splice_type} sites on {strand} strand due to insufficient genes")
            continue
        
        if verbose:
            print(f"\n[info] Testing {splice_type} sites on {strand} strand with {len(category_genes)} genes")
        
        # Filter annotations to just this category
        category_annots = annotations_df.filter(
            (pl.col("site_type") == splice_type) & 
            (pl.col("strand") == strand) &
            (pl.col("gene_id").is_in(category_genes))
        )
        
        # Test each potential adjustment value
        for offset in range(search_range[0], search_range[1] + 1):
            if verbose:
                print(f"  Testing offset {offset}...", end="")
                
            # Create adjustment dict with just this one combination modified
            test_adjustment = {
                "donor": {"plus": 0, "minus": 0},
                "acceptor": {"plus": 0, "minus": 0}
            }
            test_adjustment[splice_type][strand_key] = offset
            
            # Create a deep copy of predictions for testing this adjustment
            test_predictions = copy.deepcopy({gene_id: pred_results[gene_id] for gene_id in category_genes})
            
            # Evaluate with this adjustment
            error_df, positions_df = enhanced_evaluate_splice_site_errors(
                category_annots,
                test_predictions,
                threshold=probability_threshold,
                consensus_window=consensus_window,
                splice_site_adjustments=test_adjustment,
                verbose=0  # Silence output
            )
            
            # Calculate metrics for this adjustment
            tp_count = positions_df.filter(pl.col("pred_type") == "TP").height
            fp_count = positions_df.filter(pl.col("pred_type") == "FP").height
            fn_count = positions_df.filter(pl.col("pred_type") == "FN").height
            
            total_relevant = tp_count + fn_count  # Total true splice sites
            if total_relevant == 0:
                precision = 0
                recall = 0
                f1 = 0
            else:
                recall = tp_count / total_relevant if total_relevant > 0 else 0
                precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store metrics
            adjustment_stats[splice_type][strand_key][offset] = {
                "tp": tp_count,
                "fp": fp_count,
                "fn": fn_count,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            
            if verbose:
                print(f" TP: {tp_count}, FP: {fp_count}, FN: {fn_count}, F1: {f1:.4f}")
    
    # Determine optimal adjustment for each category
    optimal_adjustments = {
        "donor": {"plus": 0, "minus": 0},
        "acceptor": {"plus": 0, "minus": 0}
    }
    
    for splice_type in ["donor", "acceptor"]:
        for strand_key in ["plus", "minus"]:
            # Check if we have stats for this category
            if not adjustment_stats[splice_type][strand_key]:
                if verbose:
                    print(f"[warning] No adjustment stats for {splice_type} on {strand_key} strand")
                continue
                
            # Find offset with highest F1 score
            best_offset = max(
                adjustment_stats[splice_type][strand_key].keys(),
                key=lambda k: adjustment_stats[splice_type][strand_key][k]["f1"]
            )
            
            # Compare to no adjustment (offset=0)
            baseline_f1 = adjustment_stats[splice_type][strand_key].get(0, {"f1": 0})["f1"]
            best_f1 = adjustment_stats[splice_type][strand_key][best_offset]["f1"]
            
            # Only apply adjustment if it provides significant improvement
            if best_f1 > baseline_f1 + min_tp_improvement:
                optimal_adjustments[splice_type][strand_key] = best_offset
                improvement = best_f1 - baseline_f1
                if verbose:
                    print(f"[info] Selected {best_offset} for {splice_type} on {strand_key} strand (F1 improvement: {improvement:.4f})")
            else:
                if verbose:
                    print(f"[info] No significant improvement for {splice_type} on {strand_key} strand, keeping offset 0")
    
    # Generate a summary report
    if verbose:
        print("\n====== ADJUSTMENT INFERENCE SUMMARY ======")
        print(f"Inferred adjustments:")
        print(f"  Donor sites:    +{optimal_adjustments['donor']['plus']} on plus strand, +{optimal_adjustments['donor']['minus']} on minus strand")
        print(f"  Acceptor sites: +{optimal_adjustments['acceptor']['plus']} on plus strand, {optimal_adjustments['acceptor']['minus']} on minus strand")
        print("========================================")
    
    return optimal_adjustments, adjustment_stats


def auto_detect_splice_site_adjustments(annotations_df, pred_results, consensus_window=2, threshold=0.1, verbose=False, use_empirical=False):
    """
    Automatically detect the optimal offset adjustment for aligning predicted splice sites 
    with annotation data.
    
    Parameters:
    -----------
    annotations_df : pl.DataFrame
        DataFrame with true splice site annotations
    pred_results : dict
        Dictionary with predicted splice probabilities
    consensus_window : int
        Size of window to search for optimal offset
    threshold : float
        Minimum probability threshold to consider (to filter out noise)
    verbose : bool
        Whether to print verbose output
    use_empirical : bool
        Whether to use empirical inference (data-driven) instead of hardcoded pattern
    
    Returns:
    --------
    dict: Dictionary with detected adjustments:
        {
            'donor': {'plus': offset_value, 'minus': offset_value},
            'acceptor': {'plus': offset_value, 'minus': offset_value}
        }
    """
    if use_empirical:
        # Use data-driven approach to infer adjustments
        if verbose:
            print("[info] Using empirical (data-driven) approach to infer adjustments")
        
        inferred_adjustments, _ = empirical_infer_splice_site_adjustments(
            annotations_df=annotations_df,
            pred_results=pred_results,
            consensus_window=consensus_window,
            probability_threshold=threshold,
            verbose=verbose
        )
        
        return inferred_adjustments
    else:
        # SpliceAI has a known systematic pattern - this is how SpliceAI scores must be
        # adjusted to align with true positions
        spliceai_pattern = {
            'donor': {'plus': 2, 'minus': 1},
            'acceptor': {'plus': 0, 'minus': -1}
        }
        
        if verbose:
            print("\nUsing SpliceAI's known adjustment pattern:")
            print(f"  Donor sites:    +{spliceai_pattern['donor']['plus']} on plus strand, +{spliceai_pattern['donor']['minus']} on minus strand")
            print(f"  Acceptor sites: +{spliceai_pattern['acceptor']['plus']} on plus strand, {spliceai_pattern['acceptor']['minus']} on minus strand")
        
        return spliceai_pattern


def apply_auto_detected_adjustments(scores, strand, splice_type, adjustment_dict):
    """
    Apply automatically detected adjustments to scores.
    
    Parameters:
    -----------
    scores : np.ndarray or list
        Array of splice site probabilities
    strand : str
        Strand of the gene ('+' or '-')
    splice_type : str
        Type of splice site ('donor' or 'acceptor')
    adjustment_dict : dict
        Dictionary with detected adjustments from auto_detect_splice_site_adjustments
    
    Returns:
    --------
    np.ndarray
        Adjusted scores
    """
    # Convert to numpy array if it's a list
    if isinstance(scores, list):
        scores = np.array(scores)
        
    norm_strand = normalize_strand(strand)
    strand_key = 'plus' if norm_strand == '+' else 'minus'
    offset = adjustment_dict[splice_type][strand_key]
    
    # Make a copy of the scores to avoid modifying the original
    adjusted_scores = scores.copy()
    
    # No adjustment needed
    if offset == 0:
        return adjusted_scores
    
    # Shift scores - this is implementing the same logic as adjust_scores
    # in enhanced_evaluation.py but using the detected offset values
    adjusted_scores = np.roll(adjusted_scores, offset)
    
    # Zero out the wrapped-around values
    if offset > 0:
        adjusted_scores[:offset] = 0
    elif offset < 0:
        adjusted_scores[offset:] = 0
    
    return adjusted_scores


def calculate_prediction_statistics(annotations_df, pred_results, threshold=0.1, 
                                   adjustment_dict=None, auto_adjust=False, verbose=False):
    """
    Calculate statistics for splice site predictions with and without adjustments.
    
    Parameters:
    -----------
    annotations_df : pl.DataFrame
        DataFrame with true splice site annotations
    pred_results : dict
        Dictionary with predicted splice probabilities
    threshold : float
        Threshold for classifying a prediction as a splice site
    adjustment_dict : dict, optional
        Dictionary with detected adjustments
    auto_adjust : bool
        Whether to auto-detect adjustments if not provided
    verbose : bool
        Whether to print verbose output
        
    Returns:
    --------
    dict
        Dictionary with prediction statistics
    """
    if auto_adjust and adjustment_dict is None:
        adjustment_dict = auto_detect_splice_site_adjustments(
            annotations_df, pred_results, threshold=threshold
        )
    
    # Initialize statistics
    stats = {
        'donor_total': 0,
        'donor_correct': 0,
        'acceptor_total': 0,
        'acceptor_correct': 0,
        'high_scores_found': 0  # Track scores above threshold
    }
    
    # Find maximum scores to help diagnose issues with threshold
    max_scores = {
        'donor': 0.0,
        'acceptor': 0.0
    }
    
    # For each gene in predictions, extract relevant annotations
    transcript_groups = {}
    
    # First, we need to group the annotations by gene and collect relative positions
    gene_positions = {}
    for row in annotations_df.iter_rows(named=True):
        gene_id = row['gene_id']
        site_type = row['site_type']
        
        if gene_id not in gene_positions:
            gene_positions[gene_id] = {
                'donor': [],
                'acceptor': [],
                'strand': normalize_strand(row['strand'])
            }
        
        # The key pattern is to use the *relative* positions, not absolute genomic positions
        # For testing purposes, we'll use positions 0-99 for our test data
        # In a real implementation, you would convert absolute to relative using gene start/end info
        rel_pos = row.get('rel_position', None)
        if rel_pos is not None:
            gene_positions[gene_id][site_type].append(rel_pos)
    
    # Calculate prediction accuracy
    for gene_id, pred_data in pred_results.items():
        # Skip if gene not in annotations
        if gene_id not in gene_positions:
            if verbose:
                print(f"Warning: Missing annotations for gene {gene_id}")
            continue
            
        gene_data = gene_positions[gene_id]
        strand = gene_data['strand']
        strand_key = 'plus' if strand == '+' else 'minus'
        
        for site_type in ['donor', 'acceptor']:
            true_sites = gene_data[site_type]
            if not true_sites:
                continue
                
            # Check if the prediction data has the required key
            if f'{site_type}_prob' not in pred_data:
                if verbose:
                    print(f"Warning: Missing {site_type}_prob in predictions for gene {gene_id}")
                continue
                
            orig_probs = pred_data[f'{site_type}_prob']
            
            # Convert to numpy array if it's a list
            if isinstance(orig_probs, list):
                orig_probs = np.array(orig_probs)
            
            # Track maximum scores
            max_score = np.max(orig_probs) if len(orig_probs) > 0 else 0
            max_scores[site_type] = max(max_scores[site_type], max_score)
            
            # Apply adjustments if provided
            if adjustment_dict is not None:
                adjusted_probs = apply_auto_detected_adjustments(
                    orig_probs, 
                    strand, 
                    site_type, 
                    adjustment_dict
                )
            else:
                adjusted_probs = orig_probs.copy()
            
            # Track high scores
            scores_above_threshold = np.sum(adjusted_probs >= threshold)
            stats['high_scores_found'] += scores_above_threshold
            
            # For each true site, check if it's correctly predicted
            for pos in true_sites:
                stats[f'{site_type}_total'] += 1
                
                # Skip if position is out of bounds
                if pos < 0 or pos >= len(adjusted_probs):
                    if verbose:
                        print(f"Warning: Position {pos} out of bounds for {gene_id} {site_type}")
                    continue
                
                # A true site is considered correctly predicted if its
                # adjusted probability exceeds the threshold
                if adjusted_probs[pos] >= threshold:
                    stats[f'{site_type}_correct'] += 1
                    if verbose:
                        print(f"✓ {gene_id} {site_type} at pos {pos}, adj_prob={adjusted_probs[pos]:.4f}")
                elif verbose:
                    window_start = max(0, pos - 2)
                    window_end = min(len(adjusted_probs) - 1, pos + 2)
                    window_probs = adjusted_probs[window_start:window_end+1]
                    probs_str = ", ".join([f"{p:.4f}" for p in window_probs])
                    print(f"✗ {gene_id} {site_type} at pos {pos}, probs[{window_start}:{window_end}]=[{probs_str}]")
                    
                    # Show original window too
                    orig_window_probs = orig_probs[window_start:window_end+1]
                    orig_probs_str = ", ".join([f"{p:.4f}" for p in orig_window_probs])
                    print(f"  Original probs[{window_start}:{window_end}]=[{orig_probs_str}]")
    
    # Calculate accuracies
    if stats['donor_total'] > 0:
        stats['donor_accuracy'] = stats['donor_correct'] / stats['donor_total']
    else:
        stats['donor_accuracy'] = 0
        
    if stats['acceptor_total'] > 0:
        stats['acceptor_accuracy'] = stats['acceptor_correct'] / stats['acceptor_total']
    else:
        stats['acceptor_accuracy'] = 0
    
    # Diagnostic information
    if verbose:
        print("\nDiagnostic information:")
        print(f"  Maximum donor score found: {max_scores['donor']:.4f}")
        print(f"  Maximum acceptor score found: {max_scores['acceptor']:.4f}")
        print(f"  Total sites with scores above threshold ({threshold}): {stats['high_scores_found']}")
        
    return stats