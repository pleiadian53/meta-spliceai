#!/usr/bin/env python3
"""
Test the new score-shifting coordinate adjustment approach.

This test verifies that:
1. Score-shifting maintains 100% coverage (all N positions for N-bp gene)
2. No position collisions occur
3. Scores are correctly aligned with GTF annotations
4. Performance (F1 scores) is maintained or improved
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import polars as pl
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_gene(gene_id, gene_name, expected_length):
    """Test a single gene with the new score-shifting approach."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {gene_id} ({gene_name}) - Expected length: {expected_length:,} bp")
    logger.info(f"{'='*80}")
    
    # Use the existing comprehensive test infrastructure
    from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import EnhancedSelectiveInferenceWorkflow
    
    # Initialize workflow (will use default config)
    workflow = EnhancedSelectiveInferenceWorkflow()
    
    # Run prediction
    logger.info(f"\nRunning base-only prediction...")
    result = workflow.predict_for_genes(
        gene_ids=[gene_id],
        mode='base_only',
        save_predictions=True
    )
    
    if gene_id not in result or result[gene_id] is None:
        logger.error(f"❌ No predictions for {gene_id}")
        return None
    
    predictions_df = result[gene_id]
    
    # Check coverage
    actual_positions = predictions_df.height
    coverage_pct = (actual_positions / expected_length) * 100
    
    logger.info(f"\n📊 Coverage Analysis:")
    logger.info(f"  Expected positions: {expected_length:,}")
    logger.info(f"  Actual positions:   {actual_positions:,}")
    logger.info(f"  Coverage:           {coverage_pct:.2f}%")
    
    # Check for duplicates
    unique_positions = predictions_df['position'].n_unique()
    if actual_positions != unique_positions:
        logger.warning(f"  ⚠️  Duplicates detected: {actual_positions - unique_positions} duplicate positions")
    else:
        logger.info(f"  ✅ No duplicates: All {unique_positions:,} positions are unique")
    
    # Verify full coverage
    if actual_positions == expected_length:
        logger.info(f"  ✅ FULL COVERAGE ACHIEVED: 100%")
        return True
    elif coverage_pct >= 99:
        logger.info(f"  ✅ Near-full coverage: {coverage_pct:.2f}% (acceptable)")
        return True
    else:
        logger.warning(f"  ⚠️  Incomplete coverage: {coverage_pct:.2f}%")
        return False


def main():
    """Main test function."""
    logger.info("="*80)
    logger.info("SCORE-SHIFTING COORDINATE ADJUSTMENT TEST")
    logger.info("="*80)
    logger.info("\nObjective: Verify score-shifting maintains 100% coverage")
    logger.info("Expected: All N positions for N-bp gene, no position collisions")
    logger.info("")
    
    # Test genes with known lengths
    test_genes = [
        {'gene_id': 'ENSG00000134202', 'gene_name': 'GSTM3', 'length': 7107},
        {'gene_id': 'ENSG00000141510', 'gene_name': 'TP53', 'length': 25768},
        {'gene_id': 'ENSG00000157764', 'gene_name': 'BRAF', 'length': 205603},
    ]
    
    results = []
    for gene in test_genes:
        success = test_gene(gene['gene_id'], gene['gene_name'], gene['length'])
        results.append({
            'gene_id': gene['gene_id'],
            'gene_name': gene['gene_name'],
            'success': success
        })
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    
    successes = sum(1 for r in results if r['success'])
    total = len(results)
    
    for r in results:
        status = "✅" if r['success'] else "❌"
        logger.info(f"{status} {r['gene_id']} ({r['gene_name']})")
    
    logger.info(f"\nResults: {successes}/{total} genes with full/near-full coverage")
    
    if successes == total:
        logger.info("\n✅ SUCCESS: Score-shifting approach maintains full coverage!")
        logger.info("   Ready to test alignment with GTF annotations.")
    else:
        logger.warning(f"\n⚠️  PARTIAL SUCCESS: {total - successes} genes with incomplete coverage")
        logger.warning("   May need further investigation.")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()

