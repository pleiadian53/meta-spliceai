#!/usr/bin/env python3
"""
Run Complete SpliceAI Workflow for GRCh37

This script runs the full splice prediction workflow for GRCh37, generating:
1. All derived genomic features (gene_features, transcript_features, exon_features)
2. Splice site predictions
3. Analysis sequences
4. Error analysis
5. All artifacts in GRCh37-specific directories

Usage:
    # Run on specific chromosomes (recommended for testing)
    python scripts/setup/run_grch37_full_workflow.py --chromosomes 21,22 --test-mode
    
    # Run on all chromosomes (full evaluation)
    python scripts/setup/run_grch37_full_workflow.py --chromosomes 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y
    
    # Run with specific target genes
    python scripts/setup/run_grch37_full_workflow.py --target-genes BRCA1,TP53,EGFR
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import argparse
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def run_grch37_workflow(
    build: str = 'GRCh37',
    release: str = '87',
    chromosomes: Optional[List[str]] = None,
    target_genes: Optional[List[str]] = None,
    test_mode: bool = False,
    verbosity: int = 1
):
    """
    Run the complete splice prediction workflow for GRCh37.
    
    Parameters
    ----------
    build : str
        Genome build (default: 'GRCh37')
    release : str
        Ensembl release (default: '87')
    chromosomes : List[str], optional
        List of chromosomes to process (e.g., ['21', '22'])
    target_genes : List[str], optional
        List of specific genes to process
    test_mode : bool
        If True, use smaller chunk sizes for testing
    verbosity : int
        Verbosity level (0=minimal, 1=normal, 2=detailed)
    """
    from meta_spliceai.system.genomic_resources import Registry
    from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
    from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
        run_enhanced_splice_prediction_workflow
    )
    
    logger.info("=" * 80)
    logger.info("GRCH37 FULL WORKFLOW - COMPLETE PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Build: {build}")
    logger.info(f"Release: {release}")
    logger.info(f"Chromosomes: {chromosomes or 'All'}")
    logger.info(f"Target genes: {target_genes or 'All'}")
    logger.info(f"Test mode: {test_mode}")
    logger.info("")
    
    # Get build-specific paths using Registry
    registry = Registry(build=build, release=release)
    gtf_path = registry.resolve('gtf')
    fasta_path = registry.resolve('fasta')
    
    if not gtf_path or not fasta_path:
        raise FileNotFoundError(f"GTF or FASTA not found for build {build}")
    
    logger.info(f"📁 Using build-specific references:")
    logger.info(f"   GTF:   {gtf_path}")
    logger.info(f"   FASTA: {fasta_path}")
    logger.info("")
    
    # Set up output directory structure
    # CRITICAL: Use build-specific eval_dir to isolate GRCh37 artifacts
    data_root = Path("data/ensembl") / build
    eval_dir = str(data_root / "spliceai_eval")
    
    logger.info(f"📂 Output directory structure:")
    logger.info(f"   Data root: {data_root}")
    logger.info(f"   Eval dir:  {eval_dir}")
    logger.info(f"   Artifacts will be saved to: {eval_dir}/meta_models/")
    logger.info("")
    
    # Create configuration
    config = SpliceAIConfig(
        gtf_file=str(gtf_path),
        genome_fasta=str(fasta_path),
        eval_dir=eval_dir,
        output_subdir="meta_models",
        
        # Enable all data preparation steps
        do_extract_annotations=True,
        do_extract_splice_sites=True,
        do_extract_sequences=True,
        do_find_overlaping_genes=True,
        
        # Use build-specific local_dir for intermediate files
        local_dir=str(data_root),
        
        # Standard parameters
        threshold=0.5,
        consensus_window=2,
        error_window=500,
        test_mode=test_mode,
        chromosomes=chromosomes,
        
        # Enable automatic position adjustments (verify coordinate alignment)
        # This samples ~20 genes to detect optimal adjustments between base model
        # predictions and genome build annotations. Critical for multi-build support.
        use_auto_position_adjustments=True,
        
        # Format settings
        format='parquet',
        seq_format='parquet',
        separator='\t'
    )
    
    logger.info("🚀 Starting workflow...")
    logger.info("")
    logger.info("This will:")
    logger.info("  1. Extract gene annotations from GRCh37 GTF")
    logger.info("  2. Extract splice site annotations")
    logger.info("  3. Extract genomic sequences")
    logger.info("  4. Identify overlapping genes")
    logger.info("  5. Run SpliceAI predictions")
    logger.info("  6. Generate error analysis")
    logger.info("  7. Extract analysis sequences (±250bp windows)")
    logger.info("  8. Save all artifacts to GRCh37-specific directories")
    logger.info("")
    
    if test_mode:
        logger.info("⚠️  TEST MODE: Using smaller chunk sizes")
        logger.info("")
    
    # Run the workflow
    try:
        results = run_enhanced_splice_prediction_workflow(
            config=config,
            target_genes=target_genes,
            target_chromosomes=chromosomes,
            verbosity=verbosity,
            no_final_aggregate=False,  # Enable aggregation
            no_tn_sampling=False,  # Enable TN sampling
            action="predict"
        )
        
        if results.get('success'):
            logger.info("")
            logger.info("=" * 80)
            logger.info("✅ WORKFLOW COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            # Summary
            if 'positions' in results and results['positions'] is not None:
                n_positions = len(results['positions'])
                logger.info(f"Total positions analyzed: {n_positions:,}")
            
            if 'error_analysis' in results and results['error_analysis'] is not None:
                n_errors = len(results['error_analysis'])
                logger.info(f"Total errors analyzed: {n_errors:,}")
            
            logger.info("")
            logger.info("📂 Artifacts saved to:")
            logger.info(f"   {eval_dir}/meta_models/")
            logger.info("")
            logger.info("Generated files include:")
            logger.info("  • gene_features.tsv")
            logger.info("  • transcript_features.tsv")
            logger.info("  • exon_features.tsv")
            logger.info("  • splice_sites_enhanced.tsv")
            logger.info("  • gene_sequence_*.parquet")
            logger.info("  • analysis_sequences_*_chunk_*.parquet")
            logger.info("  • error_analysis_*_chunk_*.parquet")
            logger.info("  • splice_positions_enhanced_*_chunk_*.parquet")
            logger.info("")
            
            return results
        else:
            logger.error("❌ Workflow failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error running workflow: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Run complete SpliceAI workflow for GRCh37',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on chromosomes 21 and 22
  python scripts/setup/run_grch37_full_workflow.py --chromosomes 21,22 --test-mode
  
  # Run on specific genes
  python scripts/setup/run_grch37_full_workflow.py --target-genes BRCA1,TP53,EGFR --chromosomes 17,11,7
  
  # Full run on all chromosomes (takes several hours)
  python scripts/setup/run_grch37_full_workflow.py --chromosomes 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y
        """
    )
    
    parser.add_argument('--build', type=str, default='GRCh37',
                       help='Genome build (default: GRCh37)')
    parser.add_argument('--release', type=str, default='87',
                       help='Ensembl release (default: 87)')
    parser.add_argument('--chromosomes', type=str,
                       help='Comma-separated list of chromosomes (e.g., "21,22,X")')
    parser.add_argument('--target-genes', type=str,
                       help='Comma-separated list of gene names or IDs')
    parser.add_argument('--test-mode', action='store_true',
                       help='Use smaller chunk sizes for testing')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                       help='Increase verbosity (can be repeated: -v, -vv)')
    
    args = parser.parse_args()
    
    # Parse chromosomes
    chromosomes = None
    if args.chromosomes:
        chromosomes = [c.strip() for c in args.chromosomes.split(',')]
    
    # Parse target genes
    target_genes = None
    if args.target_genes:
        target_genes = [g.strip() for g in args.target_genes.split(',')]
    
    # Run workflow
    results = run_grch37_workflow(
        build=args.build,
        release=args.release,
        chromosomes=chromosomes,
        target_genes=target_genes,
        test_mode=args.test_mode,
        verbosity=args.verbose
    )
    
    if results and results.get('success'):
        logger.info("🎉 All done!")
        sys.exit(0)
    else:
        logger.error("❌ Workflow failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

