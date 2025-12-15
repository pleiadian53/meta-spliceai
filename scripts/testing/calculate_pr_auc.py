#!/usr/bin/env python3
"""
Calculate PR-AUC (Precision-Recall Area Under Curve) for splice site predictions.

This is the metric SpliceAI reports (PR-AUC = 0.97), so we should use it
for proper comparison instead of just F1 at a fixed threshold.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import polars as pl
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_predictions_and_annotations(gene_id: str, gene_info: Dict):
    """Load predictions and GTF annotations for a gene."""
    from scripts.testing.generate_and_test_20_genes import load_gtf_splice_sites
    
    # Load predictions
    pred_file = Path(f"predictions/spliceai_eval/meta_models/complete_base_predictions/{gene_id}/complete_predictions_{gene_id}.parquet")
    if not pred_file.exists():
        return None, None
    
    predictions_df = pl.read_parquet(pred_file)
    
    # Load annotations
    annotations_df, gtf_info = load_gtf_splice_sites(gene_id)
    if annotations_df.height == 0:
        return None, None
    
    return predictions_df, annotations_df


def calculate_pr_auc_for_gene(predictions_df: pl.DataFrame, annotations_df: pl.DataFrame, 
                               splice_type: str, use_view: bool = True) -> Dict:
    """
    Calculate PR-AUC for a specific splice type.
    
    Parameters
    ----------
    predictions_df : pl.DataFrame
        Predictions with scores
    annotations_df : pl.DataFrame
        Ground truth annotations
    splice_type : str
        'donor' or 'acceptor'
    use_view : bool
        Whether to use view-specific columns (e.g., donor_score_donor_view)
    
    Returns
    -------
    Dict with PR-AUC, average precision, and other metrics
    """
    # Get true positions
    true_positions = set(annotations_df.filter(
        pl.col('splice_type') == splice_type
    )['position'].to_list())
    
    if len(true_positions) == 0:
        return {'pr_auc': None, 'ap': None, 'n_true': 0}
    
    # Get score column
    if use_view:
        score_col = f'{splice_type}_score_{splice_type}_view'
        if score_col not in predictions_df.columns:
            score_col = f'{splice_type}_score'
    else:
        score_col = f'{splice_type}_score'
    
    # Create binary labels (1 for true splice sites, 0 for others)
    positions = predictions_df['position'].to_numpy()
    scores = predictions_df[score_col].to_numpy()
    
    # Label each position
    labels = np.array([1 if pos in true_positions else 0 for pos in positions])
    
    # Calculate PR curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    
    # Average precision (alternative PR-AUC calculation)
    ap = average_precision_score(labels, scores)
    
    return {
        'pr_auc': pr_auc,
        'ap': ap,  # Average Precision (sklearn's PR-AUC)
        'n_true': len(true_positions),
        'n_total': len(positions),
        'n_positive': np.sum(labels),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'mean_score': np.mean(scores)
    }


def calculate_pr_auc_for_all_genes(gene_list: List[Dict]) -> Dict:
    """Calculate PR-AUC across all test genes."""
    results = {
        'donor': [],
        'acceptor': [],
        'overall': []
    }
    
    successful_genes = []
    
    for gene_info in gene_list:
        gene_id = gene_info['gene_id']
        gene_name = gene_info['gene_name']
        
        logger.info(f"\nProcessing {gene_name} ({gene_id})...")
        
        predictions_df, annotations_df = load_predictions_and_annotations(gene_id, gene_info)
        if predictions_df is None:
            logger.warning(f"  ⚠️  No predictions found")
            continue
        
        # Calculate for donors
        donor_metrics = calculate_pr_auc_for_gene(predictions_df, annotations_df, 'donor')
        if donor_metrics['pr_auc'] is not None:
            results['donor'].append(donor_metrics)
            logger.info(f"  Donor PR-AUC: {donor_metrics['ap']:.3f} (n={donor_metrics['n_true']})")
        
        # Calculate for acceptors
        acceptor_metrics = calculate_pr_auc_for_gene(predictions_df, annotations_df, 'acceptor')
        if acceptor_metrics['pr_auc'] is not None:
            results['acceptor'].append(acceptor_metrics)
            logger.info(f"  Acceptor PR-AUC: {acceptor_metrics['ap']:.3f} (n={acceptor_metrics['n_true']})")
        
        # Overall (combined)
        if donor_metrics['pr_auc'] is not None and acceptor_metrics['pr_auc'] is not None:
            # Weighted average by number of true sites
            total_sites = donor_metrics['n_true'] + acceptor_metrics['n_true']
            overall_ap = (
                donor_metrics['ap'] * donor_metrics['n_true'] +
                acceptor_metrics['ap'] * acceptor_metrics['n_true']
            ) / total_sites
            
            results['overall'].append({
                'ap': overall_ap,
                'gene_id': gene_id,
                'gene_name': gene_name
            })
            logger.info(f"  Overall PR-AUC: {overall_ap:.3f}")
            
            successful_genes.append({
                'gene_id': gene_id,
                'gene_name': gene_name,
                'donor_ap': donor_metrics['ap'],
                'acceptor_ap': acceptor_metrics['ap'],
                'overall_ap': overall_ap
            })
    
    return results, successful_genes


def print_summary(results: Dict, successful_genes: List[Dict]):
    """Print summary statistics."""
    logger.info("\n" + "="*80)
    logger.info("PR-AUC SUMMARY (Average Precision)")
    logger.info("="*80)
    
    if len(results['donor']) > 0:
        donor_aps = [r['ap'] for r in results['donor']]
        logger.info(f"\nDonor Sites:")
        logger.info(f"  Mean AP: {np.mean(donor_aps):.3f} ± {np.std(donor_aps):.3f}")
        logger.info(f"  Median AP: {np.median(donor_aps):.3f}")
        logger.info(f"  Min AP: {np.min(donor_aps):.3f}")
        logger.info(f"  Max AP: {np.max(donor_aps):.3f}")
        logger.info(f"  N genes: {len(donor_aps)}")
    
    if len(results['acceptor']) > 0:
        acceptor_aps = [r['ap'] for r in results['acceptor']]
        logger.info(f"\nAcceptor Sites:")
        logger.info(f"  Mean AP: {np.mean(acceptor_aps):.3f} ± {np.std(acceptor_aps):.3f}")
        logger.info(f"  Median AP: {np.median(acceptor_aps):.3f}")
        logger.info(f"  Min AP: {np.min(acceptor_aps):.3f}")
        logger.info(f"  Max AP: {np.max(acceptor_aps):.3f}")
        logger.info(f"  N genes: {len(acceptor_aps)}")
    
    if len(results['overall']) > 0:
        overall_aps = [r['ap'] for r in results['overall']]
        logger.info(f"\nOverall (Combined):")
        logger.info(f"  Mean AP: {np.mean(overall_aps):.3f} ± {np.std(overall_aps):.3f}")
        logger.info(f"  Median AP: {np.median(overall_aps):.3f}")
        logger.info(f"  Min AP: {np.min(overall_aps):.3f}")
        logger.info(f"  Max AP: {np.max(overall_aps):.3f}")
        logger.info(f"  N genes: {len(overall_aps)}")
    
    logger.info("\n" + "="*80)
    logger.info("COMPARISON TO SPLICEAI PAPER")
    logger.info("="*80)
    logger.info("SpliceAI-10k reported PR-AUC: 0.97")
    if len(overall_aps) > 0:
        our_mean = np.mean(overall_aps)
        logger.info(f"Our mean PR-AUC: {our_mean:.3f}")
        diff = our_mean - 0.97
        logger.info(f"Difference: {diff:+.3f} ({diff/0.97*100:+.1f}%)")
        
        if our_mean >= 0.95:
            logger.info("\n✅ EXCELLENT! Within expected range of SpliceAI performance")
        elif our_mean >= 0.90:
            logger.info("\n✅ GOOD! Close to SpliceAI performance")
        elif our_mean >= 0.80:
            logger.info("\n⚠️  MODERATE: Some gap from SpliceAI performance")
        else:
            logger.info("\n❌ NEEDS INVESTIGATION: Significant gap from SpliceAI performance")


def main():
    """Main entry point."""
    from scripts.testing.generate_and_test_20_genes import get_20_test_genes
    
    logger.info("="*80)
    logger.info("CALCULATING PR-AUC FOR TEST GENES")
    logger.info("="*80)
    
    test_genes = get_20_test_genes()
    logger.info(f"\nLoading predictions for {len(test_genes)} genes...")
    
    results, successful_genes = calculate_pr_auc_for_all_genes(test_genes)
    
    print_summary(results, successful_genes)
    
    # Print per-gene details
    logger.info("\n" + "="*80)
    logger.info("PER-GENE RESULTS")
    logger.info("="*80)
    
    # Sort by overall AP
    successful_genes.sort(key=lambda x: x['overall_ap'], reverse=True)
    
    logger.info(f"\n{'Gene':<15} {'Donor AP':>10} {'Acceptor AP':>12} {'Overall AP':>12}")
    logger.info("-" * 52)
    for g in successful_genes:
        logger.info(f"{g['gene_name']:<15} {g['donor_ap']:>10.3f} {g['acceptor_ap']:>12.3f} {g['overall_ap']:>12.3f}")


if __name__ == "__main__":
    main()

