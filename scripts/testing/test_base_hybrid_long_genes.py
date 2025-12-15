#!/usr/bin/env python3
"""
Test base-only and hybrid modes on longer protein-coding genes.

This test verifies that:
1. Base-only mode (SpliceAI) has high F1 scores (>0.8) on protein-coding genes
2. Hybrid mode performs similarly to base-only (should not degrade)
3. Predictions align with GTF annotations

We'll test on genes of varying lengths to ensure the full coverage logic works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import EnhancedSelectiveInferenceWorkflow
from meta_spliceai.system.genomic_resources import Registry
import polars as pl
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_test_genes():
    """Get a curated list of test genes (protein-coding, well-annotated, varying lengths)."""
    # Manually selected protein-coding genes with good annotation
    test_genes = [
        {'gene_id': 'ENSG00000141510', 'gene_name': 'TP53', 'gene_length': 25768, 'description': 'Tumor protein p53'},
        {'gene_id': 'ENSG00000157764', 'gene_name': 'BRAF', 'gene_length': 205603, 'description': 'B-Raf proto-oncogene'},
        {'gene_id': 'ENSG00000134202', 'gene_name': 'GSTM3', 'gene_length': 7107, 'description': 'Glutathione S-transferase mu 3'},
        {'gene_id': 'ENSG00000171862', 'gene_name': 'PTEN', 'gene_length': 105338, 'description': 'Phosphatase and tensin homolog'},
        {'gene_id': 'ENSG00000139618', 'gene_name': 'BRCA2', 'gene_length': 84195, 'description': 'Breast cancer type 2 susceptibility protein'},
    ]
    
    return test_genes


def evaluate_predictions(predictions_df, annotations_df, gene_id, mode):
    """Evaluate predictions against annotations."""
    
    # Get annotated splice sites
    donor_sites = set(annotations_df.filter(pl.col('splice_type') == 'donor')['position'].to_list())
    acceptor_sites = set(annotations_df.filter(pl.col('splice_type') == 'acceptor')['position'].to_list())
    
    # Use threshold of 0.5 for splice site detection
    threshold = 0.5
    
    # Predicted splice sites
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
        'gene_id': gene_id,
        'mode': mode,
        'donor_f1': donor_f1,
        'donor_precision': donor_precision,
        'donor_recall': donor_recall,
        'donor_tp': donor_tp,
        'donor_fp': donor_fp,
        'donor_fn': donor_fn,
        'acceptor_f1': acceptor_f1,
        'acceptor_precision': acceptor_precision,
        'acceptor_recall': acceptor_recall,
        'acceptor_tp': acceptor_tp,
        'acceptor_fp': acceptor_fp,
        'acceptor_fn': acceptor_fn,
        'overall_f1': overall_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'n_annotations': len(donor_sites) + len(acceptor_sites),
        'n_predictions': len(pred_donors) + len(pred_acceptors)
    }


def test_gene(gene_dict, workflow, mode='base_only'):
    """Test a single gene in specified mode."""
    gene_id = gene_dict['gene_id']
    gene_name = gene_dict['gene_name']
    gene_length = gene_dict['gene_length']
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {gene_id} ({gene_name}) - {gene_length:,} bp - Mode: {mode.upper()}")
    logger.info(f"{'='*80}")
    
    # Run inference
    result = workflow.predict_for_genes(
        gene_ids=[gene_id],
        mode=mode,
        save_predictions=True
    )
    
    if gene_id not in result or result[gene_id] is None:
        logger.error(f"❌ No predictions for {gene_id}")
        return None
    
    predictions_df = result[gene_id]
    
    # Load annotations for evaluation
    gtf_path = Path("data/ensembl/Homo_sapiens.GRCh38.112.gtf")
    
    if not gtf_path.exists():
        logger.error(f"GTF file not found: {gtf_path}")
        return None
    
    # Load splice sites from GTF
    from meta_spliceai.splice_engine.meta_models.workflows.data_preparation import load_splice_sites_from_gtf
    annotations_df = load_splice_sites_from_gtf(str(gtf_path), gene_ids=[gene_id])
    
    if annotations_df.height == 0:
        logger.warning(f"⚠️  No annotations found for {gene_id}")
        return None
    
    # Evaluate
    metrics = evaluate_predictions(predictions_df, annotations_df, gene_id, mode)
    
    # Report
    logger.info(f"\n📊 Results for {gene_id} ({mode}):")
    logger.info(f"  Gene length: {gene_length:,} bp")
    logger.info(f"  Predictions: {predictions_df.height:,} positions")
    logger.info(f"  Annotations: {metrics['n_annotations']} splice sites")
    logger.info(f"\n  Donor Sites:")
    logger.info(f"    Precision: {metrics['donor_precision']:.3f}")
    logger.info(f"    Recall:    {metrics['donor_recall']:.3f}")
    logger.info(f"    F1:        {metrics['donor_f1']:.3f}")
    logger.info(f"    TP: {metrics['donor_tp']}, FP: {metrics['donor_fp']}, FN: {metrics['donor_fn']}")
    logger.info(f"\n  Acceptor Sites:")
    logger.info(f"    Precision: {metrics['acceptor_precision']:.3f}")
    logger.info(f"    Recall:    {metrics['acceptor_recall']:.3f}")
    logger.info(f"    F1:        {metrics['acceptor_f1']:.3f}")
    logger.info(f"    TP: {metrics['acceptor_tp']}, FP: {metrics['acceptor_fp']}, FN: {metrics['acceptor_fn']}")
    logger.info(f"\n  Overall:")
    logger.info(f"    Precision: {metrics['overall_precision']:.3f}")
    logger.info(f"    Recall:    {metrics['overall_recall']:.3f}")
    logger.info(f"    F1:        {metrics['overall_f1']:.3f}")
    
    # Check if performance is acceptable
    if metrics['overall_f1'] < 0.7:
        logger.warning(f"  ⚠️  LOW F1 SCORE: {metrics['overall_f1']:.3f} (expected >0.7 for protein-coding)")
    else:
        logger.info(f"  ✅ Good performance: F1 = {metrics['overall_f1']:.3f}")
    
    return metrics


def main():
    """Main test function."""
    logger.info("="*80)
    logger.info("BASE-ONLY AND HYBRID MODE TEST: Long Protein-Coding Genes")
    logger.info("="*80)
    logger.info("\nObjective: Verify SpliceAI predictions align with GTF annotations")
    logger.info("Expected: F1 scores >0.7 (ideally >0.8) for protein-coding genes")
    logger.info("")
    
    # Load test genes
    logger.info("Loading test genes...")
    test_genes = get_test_genes()
    
    logger.info(f"\nSelected {len(test_genes)} genes:")
    for gene in test_genes:
        logger.info(f"  {gene['gene_id']:20s} {gene['gene_name']:15s} {gene['gene_length']:8,} bp  - {gene['description']}")
    
    # Initialize workflow
    workflow = EnhancedSelectiveInferenceWorkflow()
    
    # Test each gene in both modes
    all_results = []
    
    for gene in test_genes:
        # Test base-only mode
        base_metrics = test_gene(gene, workflow, mode='base_only')
        if base_metrics:
            all_results.append(base_metrics)
        
        # Test hybrid mode
        hybrid_metrics = test_gene(gene, workflow, mode='hybrid')
        if hybrid_metrics:
            all_results.append(hybrid_metrics)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    results_df = pl.DataFrame(all_results)
    
    # Group by mode
    for mode in ['base_only', 'hybrid']:
        mode_results = results_df.filter(pl.col('mode') == mode)
        
        if len(mode_results) == 0:
            continue
        
        avg_f1 = mode_results['overall_f1'].mean()
        avg_precision = mode_results['overall_precision'].mean()
        avg_recall = mode_results['overall_recall'].mean()
        
        logger.info(f"\n{mode.upper()} Mode:")
        logger.info(f"  Average F1:        {avg_f1:.3f}")
        logger.info(f"  Average Precision: {avg_precision:.3f}")
        logger.info(f"  Average Recall:    {avg_recall:.3f}")
        
        # Check if performance is acceptable
        if avg_f1 < 0.7:
            logger.warning(f"  ⚠️  BELOW EXPECTED PERFORMANCE (F1 < 0.7)")
        else:
            logger.info(f"  ✅ Good average performance")
        
        # Per-gene breakdown
        logger.info(f"\n  Per-gene F1 scores:")
        for row in mode_results.iter_rows(named=True):
            status = "✅" if row['overall_f1'] >= 0.7 else "⚠️"
            logger.info(f"    {status} {row['gene_id']}: {row['overall_f1']:.3f}")
    
    # Compare base vs hybrid
    logger.info(f"\n{'='*80}")
    logger.info("BASE vs HYBRID Comparison:")
    logger.info(f"{'='*80}")
    
    for gene_id in results_df['gene_id'].unique():
        gene_results = results_df.filter(pl.col('gene_id') == gene_id)
        
        if len(gene_results) != 2:
            continue
        
        base_f1 = gene_results.filter(pl.col('mode') == 'base_only')['overall_f1'][0]
        hybrid_f1 = gene_results.filter(pl.col('mode') == 'hybrid')['overall_f1'][0]
        
        diff = hybrid_f1 - base_f1
        diff_pct = (diff / base_f1 * 100) if base_f1 > 0 else 0
        
        if abs(diff_pct) > 5:
            status = "⚠️" if diff < 0 else "✅"
            logger.info(f"{status} {gene_id}: base={base_f1:.3f}, hybrid={hybrid_f1:.3f} ({diff_pct:+.1f}%)")
        else:
            logger.info(f"  {gene_id}: base={base_f1:.3f}, hybrid={hybrid_f1:.3f} (similar)")
    
    # Final verdict
    logger.info(f"\n{'='*80}")
    base_avg = results_df.filter(pl.col('mode') == 'base_only')['overall_f1'].mean()
    hybrid_avg = results_df.filter(pl.col('mode') == 'hybrid')['overall_f1'].mean()
    
    if base_avg >= 0.7 and hybrid_avg >= 0.7:
        logger.info("✅ VERDICT: Base and hybrid modes are working correctly!")
        logger.info("   Predictions are well-aligned with GTF annotations.")
        logger.info("   Ready to investigate meta-model issue.")
    else:
        logger.warning("⚠️  VERDICT: Performance below expected!")
        logger.warning(f"   Base F1: {base_avg:.3f}, Hybrid F1: {hybrid_avg:.3f}")
        logger.warning("   There may still be issues with the prediction pipeline.")
    
    logger.info("="*80)
    
    # Save results
    output_path = Path("results/base_hybrid_long_genes_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.write_json(output_path)
    logger.info(f"\n📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main()

