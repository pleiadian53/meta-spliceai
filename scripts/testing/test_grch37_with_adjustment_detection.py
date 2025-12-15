#!/usr/bin/env python3
"""
Test GRCh37 Workflow with Automatic Score Adjustment Detection

This script runs the workflow with automatic adjustment detection enabled
to determine if any score view adjustments are needed for GRCh37.

Usage:
    python scripts/testing/test_grch37_with_adjustment_detection.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import logging
from meta_spliceai.system.genomic_resources import Registry
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("GRCH37 SCORE ADJUSTMENT DETECTION TEST")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This test will:")
    logger.info("  1. Run predictions on chr21 and chr22")
    logger.info("  2. Enable automatic score adjustment detection")
    logger.info("  3. Determine optimal adjustments for GRCh37")
    logger.info("  4. Report if any adjustments are needed")
    logger.info("")
    
    # Get build-specific paths
    build = 'GRCh37'
    release = '87'
    registry = Registry(build=build, release=release)
    
    gtf_path = registry.resolve('gtf')
    fasta_path = registry.resolve('fasta')
    
    if not gtf_path or not fasta_path:
        raise FileNotFoundError(f"GTF or FASTA not found for build {build}")
    
    logger.info(f"Using GRCh37 references:")
    logger.info(f"  GTF:   {gtf_path}")
    logger.info(f"  FASTA: {fasta_path}")
    logger.info("")
    
    # Set up configuration
    data_root = Path("data/ensembl") / build
    eval_dir = str(data_root / "spliceai_eval")
    
    config = SpliceAIConfig(
        gtf_file=str(gtf_path),
        genome_fasta=str(fasta_path),
        eval_dir=eval_dir,
        output_subdir="adjustment_detection",
        
        # Use existing data (don't re-extract)
        do_extract_annotations=False,
        do_extract_splice_sites=False,
        do_extract_sequences=False,
        do_find_overlaping_genes=False,
        
        # Use build-specific local_dir
        local_dir=str(data_root),
        
        # CRITICAL: Enable automatic adjustment detection
        use_auto_position_adjustments=True,
        
        # Standard parameters
        threshold=0.5,
        consensus_window=2,
        error_window=500,
        test_mode=True,
        chromosomes=['21', '22'],
        
        # Format settings
        format='parquet',
        seq_format='parquet',
        separator='\t'
    )
    
    logger.info("🔍 Running workflow with automatic adjustment detection...")
    logger.info("")
    
    try:
        results = run_enhanced_splice_prediction_workflow(
            config=config,
            target_genes=None,
            target_chromosomes=['21', '22'],
            verbosity=2,  # Verbose to see adjustment detection details
            no_final_aggregate=False,
            no_tn_sampling=False,
            action="predict"
        )
        
        if results.get('success'):
            logger.info("")
            logger.info("=" * 80)
            logger.info("✅ ADJUSTMENT DETECTION COMPLETE")
            logger.info("=" * 80)
            logger.info("")
            
            # Check if adjustments were detected
            if 'adjustments' in results:
                adjustments = results['adjustments']
                logger.info("📊 Detected Adjustments:")
                logger.info(f"  Donor adjustments:    {adjustments.get('donor', 'N/A')}")
                logger.info(f"  Acceptor adjustments: {adjustments.get('acceptor', 'N/A')}")
                logger.info("")
                
                # Check if adjustments are zero
                donor_adj = adjustments.get('donor', [0, 0])
                acceptor_adj = adjustments.get('acceptor', [0, 0])
                
                if all(x == 0 for x in donor_adj + acceptor_adj):
                    logger.info("✅ RESULT: No adjustments needed!")
                    logger.info("")
                    logger.info("This confirms that GRCh37 coordinates are correctly aligned")
                    logger.info("with SpliceAI's training data (which was also GRCh37).")
                else:
                    logger.info("⚠️  RESULT: Adjustments detected!")
                    logger.info("")
                    logger.info("This suggests there may be coordinate differences between")
                    logger.info("our GRCh37 annotations and SpliceAI's training annotations.")
            else:
                logger.info("⚠️  No adjustment information in results")
                logger.info("This may indicate adjustment detection was not run")
            
            # Show performance metrics
            if 'positions' in results and results['positions'] is not None:
                import polars as pl
                positions = results['positions']
                
                # Calculate metrics
                tp = (positions['pred_type'] == 'TP').sum()
                fp = (positions['pred_type'] == 'FP').sum()
                fn = (positions['pred_type'] == 'FN').sum()
                tn = (positions['pred_type'] == 'TN').sum()
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                logger.info("")
                logger.info("📊 Performance Metrics:")
                logger.info(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
                logger.info(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
                logger.info(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
                logger.info(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
            
            logger.info("")
            logger.info("=" * 80)
            
        else:
            logger.error("❌ Workflow failed")
            if 'error' in results:
                logger.error(f"Error: {results['error']}")
    
    except Exception as e:
        logger.error(f"❌ Error running workflow: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



