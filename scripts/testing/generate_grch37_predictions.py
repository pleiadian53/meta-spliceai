#!/usr/bin/env python3
"""
Generate SpliceAI Predictions for GRCh37 Genes

This script generates fresh SpliceAI predictions using GRCh37 reference genome.
No caching, no fallbacks - everything from scratch.

Usage:
    # Generate for specific genes
    python scripts/testing/generate_grch37_predictions.py --genes ENSG00000141510,ENSG00000134202
    
    # Generate for 10 sampled genes
    python scripts/testing/generate_grch37_predictions.py --num-genes 10
    
    # Generate for all 50 evaluation genes
    python scripts/testing/generate_grch37_predictions.py --num-genes 50
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import argparse
import logging
from typing import List, Dict
import shutil

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def clear_existing_predictions(gene_ids: List[str]):
    """Remove existing prediction files for the specified genes."""
    pred_base = Path("predictions/spliceai_eval/meta_models/complete_base_predictions")
    
    if not pred_base.exists():
        return
    
    removed_count = 0
    for gene_id in gene_ids:
        gene_dir = pred_base / gene_id
        if gene_dir.exists():
            shutil.rmtree(gene_dir)
            removed_count += 1
            logger.info(f"  🗑️  Removed existing predictions for {gene_id}")
    
    if removed_count > 0:
        logger.info(f"✅ Cleared {removed_count} existing prediction directories")
    else:
        logger.info("ℹ️  No existing predictions to clear")


def generate_predictions_for_genes(
    gene_ids: List[str],
    build: str = 'GRCh37',
    release: str = '87',
    force_regenerate: bool = True
):
    """
    Generate SpliceAI predictions for genes using the specified genome build.
    
    Parameters
    ----------
    gene_ids : List[str]
        List of Ensembl gene IDs
    build : str
        Genome build (e.g., 'GRCh37', 'GRCh38')
    release : str
        Ensembl release (e.g., '87', '112')
    force_regenerate : bool
        If True, remove existing predictions before generating new ones
    """
    from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
        EnhancedSelectiveInferenceWorkflow,
        EnhancedSelectiveInferenceConfig
    )
    from meta_spliceai.system.genomic_resources import Registry
    
    logger.info(f"\n{'='*80}")
    logger.info(f"GENERATING SPLICEAI PREDICTIONS FOR {len(gene_ids)} GENES")
    logger.info(f"{'='*80}")
    logger.info(f"Genome Build: {build}")
    logger.info(f"Ensembl Release: {release}")
    logger.info(f"Mode: base_only (no meta-model)")
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
    
    # Clear existing predictions if requested
    if force_regenerate:
        logger.info("🗑️  Clearing existing predictions...")
        clear_existing_predictions(gene_ids)
        logger.info("")
    
    # Get model path (not used in base_only mode, but required by config)
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / 'results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl'
    
    # Process genes one at a time to track progress
    successful_genes = []
    failed_genes = []
    
    for i, gene_id in enumerate(gene_ids, 1):
        logger.info(f"[{i}/{len(gene_ids)}] Processing {gene_id}...")
        
        try:
            # Initialize workflow with config
            # CRITICAL: The workflow will use the GTF/FASTA from the Registry
            # based on the current environment or default config
            config = EnhancedSelectiveInferenceConfig(
                target_genes=[gene_id],
                model_path=model_path,
                inference_mode='base_only',  # Only SpliceAI, no meta-model
                output_name='complete_base_predictions',
                uncertainty_threshold_low=0.02,
                uncertainty_threshold_high=0.50,
                use_timestamped_output=False,
                verbose=1  # Show progress
            )
            
            # IMPORTANT: The workflow will automatically use the correct GTF/FASTA
            # from the genomic_resources Registry based on the config
            workflow = EnhancedSelectiveInferenceWorkflow(config)
            
            # Override the registry to use the specified build
            workflow.registry = registry
            
            results = workflow.run_incremental()
            
            if results.success:
                logger.info(f"  ✅ Success - predictions saved")
                successful_genes.append(gene_id)
            else:
                logger.warning(f"  ⚠️  Failed - no predictions generated")
                failed_genes.append(gene_id)
        
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            failed_genes.append(gene_id)
            import traceback
            if logger.level == logging.DEBUG:
                traceback.print_exc()
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"GENERATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"✅ Successful: {len(successful_genes)}/{len(gene_ids)}")
    if failed_genes:
        logger.info(f"❌ Failed: {len(failed_genes)}/{len(gene_ids)}")
        logger.info(f"   Failed genes: {', '.join(failed_genes[:5])}{'...' if len(failed_genes) > 5 else ''}")
    logger.info("")
    
    return successful_genes, failed_genes


def main():
    parser = argparse.ArgumentParser(description='Generate SpliceAI predictions for GRCh37 genes')
    parser.add_argument('--genes', type=str, help='Comma-separated list of gene IDs')
    parser.add_argument('--num-genes', type=int, default=10, help='Number of genes to sample (default: 10)')
    parser.add_argument('--build', type=str, default='GRCh37', help='Genome build (default: GRCh37)')
    parser.add_argument('--release', type=str, default='87', help='Ensembl release (default: 87)')
    parser.add_argument('--no-clear', action='store_true', help='Do not clear existing predictions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for gene sampling (default: 42)')
    args = parser.parse_args()
    
    # Get gene list
    if args.genes:
        gene_ids = [g.strip() for g in args.genes.split(',')]
        logger.info(f"Using {len(gene_ids)} specified genes")
    else:
        # Sample genes from GTF
        from scripts.testing.comprehensive_spliceai_evaluation import sample_protein_coding_genes_from_gtf
        
        logger.info(f"Sampling {args.num_genes} protein-coding genes from {args.build}...")
        genes = sample_protein_coding_genes_from_gtf(
            build=args.build,
            release=args.release,
            num_genes=args.num_genes,
            seed=args.seed
        )
        gene_ids = [g['gene_id'] for g in genes]
        
        logger.info(f"\nSampled genes:")
        for i, gene in enumerate(genes, 1):
            logger.info(f"  {i:2d}. {gene['gene_name']:20s} ({gene['gene_id']}) chr{gene['chr']:3s} {gene['strand']}")
        logger.info("")
    
    # Generate predictions
    successful, failed = generate_predictions_for_genes(
        gene_ids=gene_ids,
        build=args.build,
        release=args.release,
        force_regenerate=not args.no_clear
    )
    
    # Exit with error code if any failed
    if failed:
        sys.exit(1)
    else:
        logger.info("🎉 All predictions generated successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

