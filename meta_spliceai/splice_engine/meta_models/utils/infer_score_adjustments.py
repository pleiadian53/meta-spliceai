"""
Automated adjustment detection for splice site score alignment.

This module provides utilities to automatically detect the optimal score vector
adjustments needed to align predicted splice sites with annotation data.

KEY DIFFERENCE FROM infer_splice_site_adjustments.py:
- OLD: Adjusted POSITION COORDINATES (caused collisions, coverage loss)
- NEW: Adjusts SCORE VECTORS with correlated probability constraints

This approach:
1. Maintains 100% coverage (all N positions for N-bp gene)
2. Preserves probability constraints (donor + acceptor + neither = 1.0)
3. No position collisions
4. Uses splice-type-specific views (donor view, acceptor view)
"""

import numpy as np
import polars as pl
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def empirical_infer_score_adjustments(
    annotations_df: pl.DataFrame,
    pred_results: Dict[str, Dict[str, Any]],
    search_range: Tuple[int, int] = (-5, 5),
    min_genes_per_category: int = 3,
    probability_threshold: float = 0.5,
    min_f1_improvement: float = 0.05,
    verbose: bool = False
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Any]]:
    """
    Empirically infer optimal score vector adjustments by testing different
    shift amounts and measuring their impact on F1 score.
    
    This function implements the NEW score-shifting paradigm using correlated
    probability vectors from score_adjustment_v2.py.
    
    For each splice type (donor/acceptor) and strand (+/-):
    1. Extract true splice sites from annotations
    2. Extract predicted scores for all positions
    3. Test different shift amounts in search_range
    4. For each shift:
       - Apply correlated probability vector shift (all 3 scores move together)
       - Evaluate predictions against annotations
       - Calculate F1 score
    5. Select shift that maximizes F1 score
    
    Parameters
    ----------
    annotations_df : pl.DataFrame
        DataFrame with true splice site annotations (columns: gene_id, position, 
        splice_type/site_type, strand, chrom)
    pred_results : Dict[str, Dict[str, Any]]
        Dictionary of predictions per gene:
        {
            'gene_id': {
                'donor_prob': np.ndarray,
                'acceptor_prob': np.ndarray,
                'neither_prob': np.ndarray,
                'positions': np.ndarray or List[int],
                'strand': str,
                'gene_start': int,
                'gene_end': int
            }
        }
    search_range : Tuple[int, int]
        Range of shift amounts to test (min, max, inclusive)
    min_genes_per_category : int
        Minimum number of genes required for each strand+type combination
    probability_threshold : float
        Threshold for classifying a position as a splice site
    min_f1_improvement : float
        Minimum F1 improvement required to accept a non-zero adjustment
    verbose : bool
        Whether to print detailed progress
        
    Returns
    -------
    Tuple[Dict[str, Dict[str, int]], Dict[str, Any]]
        - Optimal adjustments: {'donor': {'plus': shift, 'minus': shift}, 
                                'acceptor': {'plus': shift, 'minus': shift}}
        - Detailed statistics for each tested adjustment
    """
    from meta_spliceai.splice_engine.meta_models.utils.score_adjustment import (
        adjust_predictions_dataframe_v2
    )
    
    if verbose:
        logger.info(f"\n{'='*80}")
        logger.info("EMPIRICAL SCORE ADJUSTMENT DETECTION")
        logger.info(f"{'='*80}")
        logger.info(f"Analyzing {len(pred_results)} genes")
        logger.info(f"Testing shift amounts: {search_range[0]} to {search_range[1]}")
        logger.info(f"Threshold: {probability_threshold}")
    
    # Categorize genes by strand and available splice types
    gene_categories = _categorize_genes(annotations_df, pred_results, verbose)
    
    # Check if we have enough genes per category
    for category, genes in gene_categories.items():
        if len(genes) < min_genes_per_category:
            if verbose:
                logger.warning(f"⚠️  Only {len(genes)} genes for {category} "
                             f"(min: {min_genes_per_category})")
    
    # Test each splice type + strand combination
    combinations = [
        {'splice_type': 'donor', 'strand': '+', 'strand_key': 'plus'},
        {'splice_type': 'donor', 'strand': '-', 'strand_key': 'minus'},
        {'splice_type': 'acceptor', 'strand': '+', 'strand_key': 'plus'},
        {'splice_type': 'acceptor', 'strand': '-', 'strand_key': 'minus'}
    ]
    
    # Store results for each combination
    adjustment_stats = {
        'donor': {'plus': {}, 'minus': {}},
        'acceptor': {'plus': {}, 'minus': {}}
    }
    
    optimal_adjustments = {
        'donor': {'plus': 0, 'minus': 0},
        'acceptor': {'plus': 0, 'minus': 0}
    }
    
    # Test each combination
    for combo in combinations:
        splice_type = combo['splice_type']
        strand = combo['strand']
        strand_key = combo['strand_key']
        category = f"{splice_type}_{strand_key}"
        
        category_genes = gene_categories.get(category, [])
        if len(category_genes) < min_genes_per_category:
            if verbose:
                logger.warning(f"Skipping {category}: insufficient genes")
            continue
        
        if verbose:
            logger.info(f"\n{'─'*80}")
            logger.info(f"Testing {splice_type.upper()} on {strand} strand ({len(category_genes)} genes)")
            logger.info(f"{'─'*80}")
        
        # Filter annotations for this category
        # Handle both 'site_type' and 'splice_type' column names
        splice_col = 'site_type' if 'site_type' in annotations_df.columns else 'splice_type'
        category_annots = annotations_df.filter(
            (pl.col(splice_col) == splice_type) &
            (pl.col('strand') == strand) &
            (pl.col('gene_id').is_in(category_genes))
        )
        
        # Test each shift amount
        for shift_amount in range(search_range[0], search_range[1] + 1):
            # Create test adjustment dict
            test_adjustment = {
                'donor': {'plus': 0, 'minus': 0},
                'acceptor': {'plus': 0, 'minus': 0}
            }
            test_adjustment[splice_type][strand_key] = shift_amount
            
            # Evaluate with this adjustment
            metrics = _evaluate_with_adjustment(
                category_genes,
                pred_results,
                category_annots,
                test_adjustment,
                splice_type,
                probability_threshold,
                verbose=(verbose and shift_amount % 2 == 0)  # Log every other shift
            )
            
            # Store results
            adjustment_stats[splice_type][strand_key][shift_amount] = metrics
            
            if verbose:
                logger.info(f"  Shift {shift_amount:+2d}: "
                          f"TP={metrics['tp']:3d}, "
                          f"FP={metrics['fp']:3d}, "
                          f"FN={metrics['fn']:3d}, "
                          f"F1={metrics['f1']:.3f}")
        
        # Find optimal shift (maximizes F1)
        if adjustment_stats[splice_type][strand_key]:
            best_shift = max(
                adjustment_stats[splice_type][strand_key].keys(),
                key=lambda k: adjustment_stats[splice_type][strand_key][k]['f1']
            )
            
            baseline_f1 = adjustment_stats[splice_type][strand_key][0]['f1']
            best_f1 = adjustment_stats[splice_type][strand_key][best_shift]['f1']
            improvement = best_f1 - baseline_f1
            
            # Only use non-zero adjustment if it provides significant improvement
            if improvement >= min_f1_improvement:
                optimal_adjustments[splice_type][strand_key] = best_shift
                if verbose:
                    logger.info(f"\n  ✅ Selected shift: {best_shift:+d} "
                              f"(F1: {baseline_f1:.3f} → {best_f1:.3f}, "
                              f"improvement: {improvement:+.3f})")
            else:
                optimal_adjustments[splice_type][strand_key] = 0
                if verbose:
                    logger.info(f"\n  ℹ️  Keeping shift: 0 "
                              f"(best shift {best_shift:+d} only improved F1 by {improvement:.3f}, "
                              f"< threshold {min_f1_improvement})")
    
    # Print summary
    if verbose:
        logger.info(f"\n{'='*80}")
        logger.info("OPTIMAL ADJUSTMENTS")
        logger.info(f"{'='*80}")
        logger.info(f"Donor sites:")
        logger.info(f"  + strand: {optimal_adjustments['donor']['plus']:+d}")
        logger.info(f"  - strand: {optimal_adjustments['donor']['minus']:+d}")
        logger.info(f"Acceptor sites:")
        logger.info(f"  + strand: {optimal_adjustments['acceptor']['plus']:+d}")
        logger.info(f"  - strand: {optimal_adjustments['acceptor']['minus']:+d}")
        logger.info(f"{'='*80}")
    
    return optimal_adjustments, adjustment_stats


def _categorize_genes(
    annotations_df: pl.DataFrame,
    pred_results: Dict[str, Dict[str, Any]],
    verbose: bool = False
) -> Dict[str, List[str]]:
    """Categorize genes by strand and available splice types."""
    gene_metadata = {}
    
    for gene_id in pred_results.keys():
        gene_annots = annotations_df.filter(pl.col('gene_id') == gene_id)
        if gene_annots.height == 0:
            continue
        
        strand = gene_annots['strand'].unique().to_list()[0]
        
        # Handle both 'site_type' and 'splice_type' column names
        if 'site_type' in gene_annots.columns:
            splice_types = gene_annots['site_type'].unique().to_list()
        elif 'splice_type' in gene_annots.columns:
            splice_types = gene_annots['splice_type'].unique().to_list()
        else:
            splice_types = []
        
        gene_metadata[gene_id] = {
            'strand': strand,
            'splice_types': splice_types
        }
    
    # Group by category
    categories = {
        'donor_plus': [],
        'donor_minus': [],
        'acceptor_plus': [],
        'acceptor_minus': []
    }
    
    for gene_id, meta in gene_metadata.items():
        strand_key = 'plus' if meta['strand'] == '+' else 'minus'
        for stype in meta['splice_types']:
            category = f"{stype}_{strand_key}"
            if category in categories:
                categories[category].append(gene_id)
    
    if verbose:
        logger.info("\nGene counts by category:")
        for category, genes in categories.items():
            logger.info(f"  {category}: {len(genes)} genes")
    
    return categories


def _evaluate_with_adjustment(
    gene_ids: List[str],
    pred_results: Dict[str, Dict[str, Any]],
    annotations_df: pl.DataFrame,
    adjustment_dict: Dict[str, Dict[str, int]],
    splice_type: str,
    threshold: float,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate predictions with a specific adjustment.
    
    Uses the score-shifting approach from score_adjustment_v2.py.
    """
    from meta_spliceai.splice_engine.meta_models.utils.score_adjustment import (
        create_splice_type_views
    )
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    for gene_id in gene_ids:
        if gene_id not in pred_results:
            continue
        
        pred_data = pred_results[gene_id]
        
        # Get predictions and strand
        donor_scores = np.array(pred_data.get('donor_prob', []))
        acceptor_scores = np.array(pred_data.get('acceptor_prob', []))
        neither_scores = np.array(pred_data.get('neither_prob', []))
        positions = np.array(pred_data.get('positions', []))
        strand = pred_data.get('strand', '+')
        
        if len(donor_scores) == 0 or len(positions) == 0:
            continue
        
        # Create splice-type-specific views with correlated probability vectors
        views = create_splice_type_views(
            donor_scores=donor_scores,
            acceptor_scores=acceptor_scores,
            neither_scores=neither_scores,
            strand=strand,
            adjustment_dict=adjustment_dict,
            verbose=False
        )
        
        # Use the appropriate view for this splice type
        if splice_type == 'donor':
            adjusted_scores = views['donor_view'][0]  # donor scores from donor view
        else:  # acceptor
            adjusted_scores = views['acceptor_view'][1]  # acceptor scores from acceptor view
        
        # Get true sites for this gene
        gene_annots = annotations_df.filter(pl.col('gene_id') == gene_id)
        true_positions = set(gene_annots['position'].to_list())
        
        # Get predicted sites (above threshold)
        pred_positions = set(positions[adjusted_scores > threshold].tolist())
        
        # Calculate TP, FP, FN
        tp = len(true_positions & pred_positions)
        fp = len(pred_positions - true_positions)
        fn = len(true_positions - pred_positions)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
    
    # Calculate metrics
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': tp_total,
        'fp': fp_total,
        'fn': fn_total,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def auto_detect_score_adjustments(
    annotations_df: pl.DataFrame,
    pred_results: Dict[str, Dict[str, Any]],
    use_empirical: bool = True,
    search_range: Tuple[int, int] = (-5, 5),
    threshold: float = 0.5,
    verbose: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    Automatically detect optimal score vector adjustments.
    
    This is the main entry point for adjustment detection.
    
    Parameters
    ----------
    annotations_df : pl.DataFrame
        Ground truth annotations
    pred_results : Dict[str, Dict[str, Any]]
        Predictions per gene
    use_empirical : bool
        If True, use data-driven detection. If False, return zero adjustments.
    search_range : Tuple[int, int]
        Range of shifts to test
    threshold : float
        Classification threshold
    verbose : bool
        Print progress
        
    Returns
    -------
    Dict[str, Dict[str, int]]
        Optimal adjustments
    """
    if use_empirical and pred_results is not None:
        if verbose:
            logger.info("Using empirical (data-driven) score adjustment detection")
        
        optimal_adjustments, _ = empirical_infer_score_adjustments(
            annotations_df=annotations_df,
            pred_results=pred_results,
            search_range=search_range,
            probability_threshold=threshold,
            verbose=verbose
        )
        
        return optimal_adjustments
    else:
        if verbose:
            logger.info("Using zero adjustments (no empirical detection)")
        
        # Default: no adjustments (base model assumed aligned)
        return {
            'donor': {'plus': 0, 'minus': 0},
            'acceptor': {'plus': 0, 'minus': 0}
        }


def save_adjustment_dict(
    adjustment_dict: Dict[str, Dict[str, int]],
    output_path: Path,
    verbose: bool = False
) -> None:
    """Save adjustment dictionary to JSON file."""
    import json
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(adjustment_dict, f, indent=4)
    
    if verbose:
        logger.info(f"Saved adjustments to {output_path}")


def load_adjustment_dict(
    input_path: Path,
    verbose: bool = False
) -> Optional[Dict[str, Dict[str, int]]]:
    """Load adjustment dictionary from JSON file."""
    import json
    
    input_path = Path(input_path)
    if not input_path.exists():
        if verbose:
            logger.warning(f"Adjustment file not found: {input_path}")
        return None
    
    with open(input_path, 'r') as f:
        adjustment_dict = json.load(f)
    
    if verbose:
        logger.info(f"Loaded adjustments from {input_path}")
        logger.info(f"  Donor: +{adjustment_dict['donor']['plus']}/{adjustment_dict['donor']['minus']}")
        logger.info(f"  Acceptor: +{adjustment_dict['acceptor']['plus']}/{adjustment_dict['acceptor']['minus']}")
    
    return adjustment_dict


# Test function
def test_score_adjustment_detection():
    """Test the score adjustment detection on sample data."""
    print("="*80)
    print("TESTING SCORE ADJUSTMENT DETECTION")
    print("="*80)
    
    # This would require actual prediction data and annotations
    # For now, just verify the imports work
    from meta_spliceai.splice_engine.meta_models.utils.score_adjustment_v2 import (
        create_splice_type_views,
        adjust_predictions_dataframe_v2
    )
    
    print("✅ Imports successful")
    print("✅ Module ready for use")
    
    print("\nUsage example:")
    print("""
    from meta_spliceai.splice_engine.meta_models.utils.infer_score_adjustments import (
        auto_detect_score_adjustments
    )
    
    # Detect optimal adjustments
    adjustments = auto_detect_score_adjustments(
        annotations_df=annotations,
        pred_results=predictions,
        use_empirical=True,
        verbose=True
    )
    
    print(f"Optimal adjustments: {adjustments}")
    """)


if __name__ == "__main__":
    test_score_adjustment_detection()

