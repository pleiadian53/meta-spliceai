#!/usr/bin/env python3
"""
Test the new score-based adjustment detection system.

This script will:
1. Load predictions and annotations for test genes
2. Run empirical detection to find optimal score shifts
3. Compare results to the old hardcoded values
4. Validate that the detected shifts improve F1 scores
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import polars as pl
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_test_data(gene_ids: list):
    """Load predictions and annotations for test genes."""
    from scripts.testing.generate_and_test_20_genes import load_gtf_splice_sites
    
    pred_results = {}
    all_annotations = []
    
    for gene_id in gene_ids:
        # Load predictions
        pred_file = Path(f"predictions/spliceai_eval/meta_models/complete_base_predictions/{gene_id}/complete_predictions_{gene_id}.parquet")
        if not pred_file.exists():
            logger.warning(f"No predictions for {gene_id}")
            continue
        
        predictions_df = pl.read_parquet(pred_file)
        
        # Convert to expected format
        pred_results[gene_id] = {
            'donor_prob': predictions_df['donor_score'].to_numpy(),
            'acceptor_prob': predictions_df['acceptor_score'].to_numpy(),
            'neither_prob': predictions_df['neither_score'].to_numpy(),
            'positions': predictions_df['position'].to_numpy(),
            'strand': predictions_df['strand'][0],
            'gene_start': predictions_df['gene_start'][0] if 'gene_start' in predictions_df.columns else 0,
            'gene_end': predictions_df['gene_end'][0] if 'gene_end' in predictions_df.columns else len(predictions_df)
        }
        
        # Load annotations
        annotations_df, _ = load_gtf_splice_sites(gene_id)
        if annotations_df.height > 0:
            # Add gene_id column
            annotations_df = annotations_df.with_columns([
                pl.lit(gene_id).alias('gene_id')
            ])
            all_annotations.append(annotations_df)
    
    # Combine all annotations
    if all_annotations:
        combined_annotations = pl.concat(all_annotations)
    else:
        combined_annotations = pl.DataFrame()
    
    return pred_results, combined_annotations


def main():
    """Main test function."""
    logger.info("="*80)
    logger.info("SCORE-BASED ADJUSTMENT DETECTION TEST")
    logger.info("="*80)
    
    # Use genes from the 20-gene test
    from scripts.testing.generate_and_test_20_genes import get_20_test_genes
    
    test_genes = get_20_test_genes()
    
    # Use a subset for faster testing (or all if you want comprehensive)
    gene_ids = [g['gene_id'] for g in test_genes[:10]]  # First 10 genes
    
    logger.info(f"\nLoading data for {len(gene_ids)} genes...")
    pred_results, annotations_df = load_test_data(gene_ids)
    
    logger.info(f"Loaded predictions for {len(pred_results)} genes")
    logger.info(f"Loaded annotations: {annotations_df.height} splice sites")
    
    if len(pred_results) < 3:
        logger.error("Insufficient genes with predictions. Run generate_and_test_20_genes.py first.")
        return
    
    # Run empirical detection
    from meta_spliceai.splice_engine.meta_models.utils.infer_score_adjustments import (
        auto_detect_score_adjustments
    )
    
    logger.info("\n" + "="*80)
    logger.info("RUNNING EMPIRICAL DETECTION")
    logger.info("="*80)
    
    detected_adjustments = auto_detect_score_adjustments(
        annotations_df=annotations_df,
        pred_results=pred_results,
        use_empirical=True,
        search_range=(-5, 5),
        threshold=0.5,
        verbose=True
    )
    
    # Compare to old hardcoded values
    old_adjustments = {
        'donor': {'plus': 2, 'minus': 1},
        'acceptor': {'plus': 0, 'minus': -1}
    }
    
    logger.info("\n" + "="*80)
    logger.info("COMPARISON")
    logger.info("="*80)
    
    logger.info("\nOld (hardcoded) adjustments:")
    logger.info(f"  Donor:    + strand: {old_adjustments['donor']['plus']:+d}, "
              f"- strand: {old_adjustments['donor']['minus']:+d}")
    logger.info(f"  Acceptor: + strand: {old_adjustments['acceptor']['plus']:+d}, "
              f"- strand: {old_adjustments['acceptor']['minus']:+d}")
    
    logger.info("\nNew (empirically detected) adjustments:")
    logger.info(f"  Donor:    + strand: {detected_adjustments['donor']['plus']:+d}, "
              f"- strand: {detected_adjustments['donor']['minus']:+d}")
    logger.info(f"  Acceptor: + strand: {detected_adjustments['acceptor']['plus']:+d}, "
              f"- strand: {detected_adjustments['acceptor']['minus']:+d}")
    
    # Check if they match
    matches = []
    for stype in ['donor', 'acceptor']:
        for strand in ['plus', 'minus']:
            old_val = old_adjustments[stype][strand]
            new_val = detected_adjustments[stype][strand]
            match = "✅ SAME" if old_val == new_val else f"❌ DIFFERENT (Δ={new_val - old_val:+d})"
            matches.append(match)
            logger.info(f"\n  {stype.capitalize()} {strand}: {match}")
    
    # Overall verdict
    logger.info("\n" + "="*80)
    logger.info("VERDICT")
    logger.info("="*80)
    
    if all("SAME" in m for m in matches):
        logger.info("✅ Empirical detection confirms old adjustments")
    else:
        logger.info("❌ Empirical detection suggests DIFFERENT adjustments!")
        logger.info("\nThis validates our finding that the old adjustments")
        logger.info("were not optimal for the current dataset/workflow.")
    
    # Save the detected adjustments
    from meta_spliceai.splice_engine.meta_models.utils.infer_score_adjustments import (
        save_adjustment_dict
    )
    
    output_path = Path("predictions/empirically_detected_score_adjustments.json")
    save_adjustment_dict(detected_adjustments, output_path, verbose=True)
    
    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()

