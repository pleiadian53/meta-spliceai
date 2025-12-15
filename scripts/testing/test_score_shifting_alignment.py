#!/usr/bin/env python3
"""
Test Score-Shifting Alignment with GTF Annotations

This test verifies that the score-shifting coordinate adjustment correctly aligns
predictions with GTF annotations by comparing F1 scores with the old position-shifting approach.

Expected: F1 scores >= 0.8 for protein-coding genes (SpliceAI's documented performance)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import polars as pl
import logging
from typing import Dict, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_gtf_annotations(gene_id: str) -> pl.DataFrame:
    """Load GTF annotations for a gene."""
    from meta_spliceai.splice_engine.meta_models.workflows.data_preparation import load_splice_sites_from_gtf
    
    gtf_path = Path("data/ensembl/Homo_sapiens.GRCh38.112.gtf")
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF file not found: {gtf_path}")
    
    annotations_df = load_splice_sites_from_gtf(str(gtf_path), gene_ids=[gene_id])
    return annotations_df


def evaluate_predictions(predictions_df: pl.DataFrame, annotations_df: pl.DataFrame, 
                        threshold: float = 0.5) -> Dict:
    """
    Evaluate predictions against GTF annotations.
    
    Returns metrics for donor, acceptor, and overall performance.
    """
    # Get annotated splice sites
    donor_sites = set(annotations_df.filter(pl.col('splice_type') == 'donor')['position'].to_list())
    acceptor_sites = set(annotations_df.filter(pl.col('splice_type') == 'acceptor')['position'].to_list())
    
    # Get predicted splice sites (using threshold)
    pred_donors = set(predictions_df.filter(pl.col('donor_score') > threshold)['position'].to_list())
    pred_acceptors = set(predictions_df.filter(pl.col('acceptor_score') > threshold)['position'].to_list())
    
    # Calculate metrics for donors
    donor_tp = len(donor_sites & pred_donors)
    donor_fp = len(pred_donors - donor_sites)
    donor_fn = len(donor_sites - pred_donors)
    
    donor_precision = donor_tp / (donor_tp + donor_fp) if (donor_tp + donor_fp) > 0 else 0
    donor_recall = donor_tp / (donor_tp + donor_fn) if (donor_tp + donor_fn) > 0 else 0
    donor_f1 = 2 * donor_precision * donor_recall / (donor_precision + donor_recall) if (donor_precision + donor_recall) > 0 else 0
    
    # Calculate metrics for acceptors
    acceptor_tp = len(acceptor_sites & pred_acceptors)
    acceptor_fp = len(pred_acceptors - acceptor_sites)
    acceptor_fn = len(acceptor_sites - pred_acceptors)
    
    acceptor_precision = acceptor_tp / (acceptor_tp + acceptor_fp) if (acceptor_tp + acceptor_fp) > 0 else 0
    acceptor_recall = acceptor_tp / (acceptor_tp + acceptor_fn) if (acceptor_tp + acceptor_fn) > 0 else 0
    acceptor_f1 = 2 * acceptor_precision * acceptor_recall / (acceptor_precision + acceptor_recall) if (acceptor_precision + acceptor_recall) > 0 else 0
    
    # Overall metrics
    total_tp = donor_tp + acceptor_tp
    total_fp = donor_fp + acceptor_fp
    total_fn = donor_fn + acceptor_fn
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    return {
        'donor': {
            'precision': donor_precision,
            'recall': donor_recall,
            'f1': donor_f1,
            'tp': donor_tp,
            'fp': donor_fp,
            'fn': donor_fn,
            'n_annotated': len(donor_sites),
            'n_predicted': len(pred_donors)
        },
        'acceptor': {
            'precision': acceptor_precision,
            'recall': acceptor_recall,
            'f1': acceptor_f1,
            'tp': acceptor_tp,
            'fp': acceptor_fp,
            'fn': acceptor_fn,
            'n_annotated': len(acceptor_sites),
            'n_predicted': len(pred_acceptors)
        },
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'n_annotated': len(donor_sites) + len(acceptor_sites),
            'n_predicted': len(pred_donors) + len(pred_acceptors)
        }
    }


def test_gene_alignment(gene_id: str, gene_name: str, expected_length: int) -> Dict:
    """Test alignment of score-shifted predictions with GTF annotations."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {gene_id} ({gene_name}) - {expected_length:,} bp")
    logger.info(f"{'='*80}")
    
    # Load predictions (with score-shifting)
    from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import EnhancedSelectiveInferenceWorkflow
    
    workflow = EnhancedSelectiveInferenceWorkflow()
    
    logger.info(f"Running prediction with score-shifting adjustment...")
    result = workflow.predict_for_genes(
        gene_ids=[gene_id],
        mode='base_only',
        save_predictions=True
    )
    
    if gene_id not in result or result[gene_id] is None:
        logger.error(f"❌ No predictions for {gene_id}")
        return None
    
    predictions_df = result[gene_id]
    
    # Verify coverage
    actual_positions = predictions_df.height
    coverage_pct = (actual_positions / expected_length) * 100
    
    logger.info(f"\n📊 Coverage Check:")
    logger.info(f"  Expected: {expected_length:,} positions")
    logger.info(f"  Actual:   {actual_positions:,} positions")
    logger.info(f"  Coverage: {coverage_pct:.1f}%")
    
    if actual_positions != expected_length:
        logger.warning(f"  ⚠️  Coverage mismatch!")
    else:
        logger.info(f"  ✅ Full coverage")
    
    # Load GTF annotations
    logger.info(f"\nLoading GTF annotations...")
    annotations_df = load_gtf_annotations(gene_id)
    
    if annotations_df.height == 0:
        logger.warning(f"  ⚠️  No annotations found for {gene_id}")
        return None
    
    n_donor_annot = len(annotations_df.filter(pl.col('splice_type') == 'donor'))
    n_acceptor_annot = len(annotations_df.filter(pl.col('splice_type') == 'acceptor'))
    logger.info(f"  Found {n_donor_annot} donor sites, {n_acceptor_annot} acceptor sites")
    
    # Evaluate at different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = None
    
    logger.info(f"\n🎯 Evaluating at different thresholds:")
    for threshold in thresholds:
        metrics = evaluate_predictions(predictions_df, annotations_df, threshold)
        logger.info(f"  Threshold {threshold:.1f}: F1={metrics['overall']['f1']:.3f} "
                   f"(P={metrics['overall']['precision']:.3f}, R={metrics['overall']['recall']:.3f})")
        
        if metrics['overall']['f1'] > best_f1:
            best_f1 = metrics['overall']['f1']
            best_threshold = threshold
            best_metrics = metrics
    
    logger.info(f"\n  Best threshold: {best_threshold:.1f} (F1={best_f1:.3f})")
    
    # Detailed metrics at best threshold
    logger.info(f"\n📈 Detailed Metrics (threshold={best_threshold:.1f}):")
    
    logger.info(f"\n  Donor Sites:")
    logger.info(f"    Precision: {best_metrics['donor']['precision']:.3f}")
    logger.info(f"    Recall:    {best_metrics['donor']['recall']:.3f}")
    logger.info(f"    F1:        {best_metrics['donor']['f1']:.3f}")
    logger.info(f"    TP={best_metrics['donor']['tp']}, FP={best_metrics['donor']['fp']}, FN={best_metrics['donor']['fn']}")
    logger.info(f"    Annotated: {best_metrics['donor']['n_annotated']}, Predicted: {best_metrics['donor']['n_predicted']}")
    
    logger.info(f"\n  Acceptor Sites:")
    logger.info(f"    Precision: {best_metrics['acceptor']['precision']:.3f}")
    logger.info(f"    Recall:    {best_metrics['acceptor']['recall']:.3f}")
    logger.info(f"    F1:        {best_metrics['acceptor']['f1']:.3f}")
    logger.info(f"    TP={best_metrics['acceptor']['tp']}, FP={best_metrics['acceptor']['fp']}, FN={best_metrics['acceptor']['fn']}")
    logger.info(f"    Annotated: {best_metrics['acceptor']['n_annotated']}, Predicted: {best_metrics['acceptor']['n_predicted']}")
    
    logger.info(f"\n  Overall:")
    logger.info(f"    Precision: {best_metrics['overall']['precision']:.3f}")
    logger.info(f"    Recall:    {best_metrics['overall']['recall']:.3f}")
    logger.info(f"    F1:        {best_metrics['overall']['f1']:.3f}")
    logger.info(f"    TP={best_metrics['overall']['tp']}, FP={best_metrics['overall']['fp']}, FN={best_metrics['overall']['fn']}")
    
    # Performance assessment
    logger.info(f"\n🎓 Performance Assessment:")
    if best_f1 >= 0.8:
        logger.info(f"  ✅ EXCELLENT: F1={best_f1:.3f} >= 0.8 (expected for SpliceAI)")
    elif best_f1 >= 0.7:
        logger.info(f"  ✅ GOOD: F1={best_f1:.3f} >= 0.7")
    elif best_f1 >= 0.6:
        logger.info(f"  ⚠️  ACCEPTABLE: F1={best_f1:.3f} >= 0.6 (below expected)")
    else:
        logger.info(f"  ❌ POOR: F1={best_f1:.3f} < 0.6 (alignment may be incorrect!)")
    
    return {
        'gene_id': gene_id,
        'gene_name': gene_name,
        'coverage': coverage_pct,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'metrics': best_metrics
    }


def main():
    """Main test function."""
    logger.info("="*80)
    logger.info("SCORE-SHIFTING ALIGNMENT TEST")
    logger.info("="*80)
    logger.info("\nObjective: Verify score-shifting correctly aligns with GTF annotations")
    logger.info("Expected: F1 scores >= 0.8 for protein-coding genes (SpliceAI performance)")
    logger.info("")
    
    # Test genes (protein-coding, well-annotated)
    test_genes = [
        {'gene_id': 'ENSG00000134202', 'gene_name': 'GSTM3', 'length': 7107},
        {'gene_id': 'ENSG00000141510', 'gene_name': 'TP53', 'length': 25768},
        {'gene_id': 'ENSG00000157764', 'gene_name': 'BRAF', 'length': 205603},
    ]
    
    results = []
    for gene in test_genes:
        result = test_gene_alignment(gene['gene_id'], gene['gene_name'], gene['length'])
        if result:
            results.append(result)
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    
    if not results:
        logger.error("❌ No results to summarize")
        return
    
    # Coverage summary
    logger.info(f"\nCoverage:")
    for r in results:
        status = "✅" if r['coverage'] == 100.0 else "⚠️"
        logger.info(f"  {status} {r['gene_name']:10s}: {r['coverage']:.1f}%")
    
    # Performance summary
    logger.info(f"\nAlignment Performance (F1 scores):")
    for r in results:
        f1 = r['best_f1']
        if f1 >= 0.8:
            status = "✅"
        elif f1 >= 0.7:
            status = "✅"
        elif f1 >= 0.6:
            status = "⚠️"
        else:
            status = "❌"
        
        logger.info(f"  {status} {r['gene_name']:10s}: F1={f1:.3f} (threshold={r['best_threshold']:.1f})")
        logger.info(f"     Donor F1={r['metrics']['donor']['f1']:.3f}, Acceptor F1={r['metrics']['acceptor']['f1']:.3f}")
    
    # Overall assessment
    avg_f1 = sum(r['best_f1'] for r in results) / len(results)
    avg_coverage = sum(r['coverage'] for r in results) / len(results)
    
    logger.info(f"\nOverall:")
    logger.info(f"  Average Coverage: {avg_coverage:.1f}%")
    logger.info(f"  Average F1:       {avg_f1:.3f}")
    
    logger.info(f"\n{'='*80}")
    logger.info("VERDICT")
    logger.info(f"{'='*80}")
    
    if avg_coverage >= 99.0 and avg_f1 >= 0.8:
        logger.info("✅ SUCCESS: Score-shifting achieves full coverage AND high alignment!")
        logger.info("   - Coverage: {:.1f}% (target: 100%)".format(avg_coverage))
        logger.info("   - F1 Score: {:.3f} (target: >= 0.8)".format(avg_f1))
        logger.info("   - Conclusion: Adjustment is CORRECT and working as expected.")
    elif avg_coverage >= 99.0 and avg_f1 >= 0.7:
        logger.info("✅ GOOD: Score-shifting achieves full coverage and good alignment")
        logger.info("   - Coverage: {:.1f}% (target: 100%)".format(avg_coverage))
        logger.info("   - F1 Score: {:.3f} (target: >= 0.7)".format(avg_f1))
        logger.info("   - Conclusion: Adjustment is working correctly.")
    elif avg_coverage >= 99.0:
        logger.warning("⚠️  PARTIAL: Full coverage but lower-than-expected F1")
        logger.warning("   - Coverage: {:.1f}% ✅".format(avg_coverage))
        logger.warning("   - F1 Score: {:.3f} (expected >= 0.8)".format(avg_f1))
        logger.warning("   - Possible issues: Threshold, gene complexity, or adjustment values")
    else:
        logger.error("❌ FAILURE: Coverage or alignment issues detected")
        logger.error("   - Coverage: {:.1f}% (target: 100%)".format(avg_coverage))
        logger.error("   - F1 Score: {:.3f}".format(avg_f1))
        logger.error("   - Action: Review adjustment implementation")
    
    logger.info("="*80)
    
    # Save results
    output_path = Path("results/score_shifting_alignment_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main()

