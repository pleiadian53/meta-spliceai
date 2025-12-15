#!/usr/bin/env python3
"""
Full Genome Base Model Pass

Run base model predictions on ALL genes for meta-learning training dataset generation.

This script:
1. Runs OpenSpliceAI (or SpliceAI) on all genes in the annotation
2. Generates all artifacts (analysis_sequences_*, positions, errors, manifest)
3. Calculates comprehensive evaluation metrics (F1, ROCAUC, AP, top-k)
4. Saves results for meta-learning training

Usage:
    python scripts/training/run_full_genome_base_model_pass.py --base-model openspliceai
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from meta_spliceai.run_base_model import run_base_model_predictions, BaseModelConfig
from meta_spliceai.splice_engine.meta_models.utils.comprehensive_evaluation import (
    evaluate_full_genome_pass
)


def main():
    parser = argparse.ArgumentParser(
        description="Run full genome base model pass for meta-learning training"
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='openspliceai',
        choices=['spliceai', 'openspliceai'],
        help='Base model to use (default: openspliceai)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='production',
        choices=['test', 'production'],
        help='Execution mode (default: production)'
    )
    parser.add_argument(
        '--coverage',
        type=str,
        default='full_genome',
        choices=['gene_subset', 'chromosome', 'full_genome'],
        help='Coverage mode (default: full_genome)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Splice site score threshold (default: 0.5)'
    )
    parser.add_argument(
        '--no-tn-sampling',
        action='store_true',
        help='Disable true negative sampling (keep all TN positions)'
    )
    parser.add_argument(
        '--save-nucleotide-scores',
        action='store_true',
        help='Save nucleotide-level scores (WARNING: large data volume)'
    )
    parser.add_argument(
        '--verbosity',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Output verbosity (0=minimal, 1=normal, 2=detailed)'
    )
    parser.add_argument(
        '--chromosomes',
        type=str,
        default=None,
        help='Comma-separated list of chromosomes to process (e.g., "1,2,X" or "21" for single chromosome)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: auto-generated based on base model and timestamp)'
    )
    
    args = parser.parse_args()
    
    # Parse chromosomes argument
    target_chromosomes = None
    if args.chromosomes:
        target_chromosomes = [chr.strip() for chr in args.chromosomes.split(',')]
    
    # Generate test name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if target_chromosomes:
        chrom_suffix = '_'.join(target_chromosomes) if len(target_chromosomes) <= 3 else f"{len(target_chromosomes)}chroms"
        test_name = f"{args.base_model}_chr{chrom_suffix}_{timestamp}"
    else:
        test_name = f"full_genome_{args.base_model}_{timestamp}"
    
    print("=" * 80)
    print("FULL GENOME BASE MODEL PASS")
    print("=" * 80)
    print()
    print(f"Base Model: {args.base_model.upper()}")
    print(f"Mode: {args.mode}")
    print(f"Coverage: {args.coverage}")
    print(f"Chromosomes: {', '.join(target_chromosomes) if target_chromosomes else 'ALL'}")
    print(f"Test Name: {test_name}")
    print(f"Threshold: {args.threshold}")
    print(f"TN Sampling: {'Disabled' if args.no_tn_sampling else 'Enabled'}")
    print(f"Nucleotide Scores: {'Enabled' if args.save_nucleotide_scores else 'Disabled'}")
    print()
    
    # Create configuration
    config = BaseModelConfig(
        base_model=args.base_model,
        mode=args.mode,
        coverage=args.coverage,
        test_name=test_name,
        threshold=args.threshold,
        consensus_window=2,
        error_window=500,
        use_auto_position_adjustments=True,
        save_nucleotide_scores=args.save_nucleotide_scores
    )
    
    # Run predictions on ALL genes (target_genes=None means all genes)
    print("=" * 80)
    print("STEP 1: Running Base Model Predictions")
    print("=" * 80)
    print()
    if target_chromosomes:
        print(f"⚠️  This will process ALL genes on chromosomes: {', '.join(target_chromosomes)}")
    else:
        print("⚠️  This will process ALL genes in the annotation.")
    print("    This may take several hours depending on the dataset size.")
    print()
    
    results = run_base_model_predictions(
        base_model=args.base_model,
        target_genes=None,  # None = all genes
        target_chromosomes=target_chromosomes,  # NEW: Chromosome filtering
        config=config,
        verbosity=args.verbosity,
        no_tn_sampling=args.no_tn_sampling,
        save_nucleotide_scores=args.save_nucleotide_scores
    )
    
    if not results.get('success', False):
        print("❌ Workflow failed!")
        sys.exit(1)
    
    positions_df = results.get('positions')
    if positions_df is None or positions_df.height == 0:
        print("❌ No positions generated!")
        sys.exit(1)
    
    print()
    print("=" * 80)
    print("STEP 2: Comprehensive Evaluation")
    print("=" * 80)
    print()
    
    # Calculate comprehensive metrics
    eval_output_path = Path(results['paths']['artifacts_dir']) / 'evaluation_metrics.json'
    metrics = evaluate_full_genome_pass(
        positions_df,
        output_path=str(eval_output_path),
        verbose=True
    )
    
    # Save summary
    summary = {
        'test_name': test_name,
        'base_model': args.base_model,
        'mode': args.mode,
        'coverage': args.coverage,
        'timestamp': timestamp,
        'total_positions': int(positions_df.height),
        'total_genes': int(positions_df['gene_id'].n_unique()),
        'metrics': metrics,
        'paths': results['paths'],
        'manifest_summary': results.get('manifest_summary', {})
    }
    
    summary_path = Path(results['paths']['artifacts_dir']) / 'full_genome_pass_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print("=" * 80)
    print("STEP 3: Artifact Verification")
    print("=" * 80)
    print()
    
    # Verify artifacts
    artifacts_dir = Path(results['paths']['artifacts_dir'])
    artifacts_to_check = [
        'full_splice_positions_enhanced.tsv',
        'full_splice_errors.tsv',
        'gene_manifest.tsv',
        'evaluation_metrics.json',
        'full_genome_pass_summary.json'
    ]
    
    if args.save_nucleotide_scores:
        artifacts_to_check.append('nucleotide_scores.tsv')
    
    print("Checking artifacts:")
    all_present = True
    for artifact in artifacts_to_check:
        artifact_path = artifacts_dir / artifact
        if artifact_path.exists():
            size_mb = artifact_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {artifact} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {artifact} (MISSING)")
            all_present = False
    
    # Check analysis_sequences files
    analysis_seq_files = list(artifacts_dir.glob('analysis_sequences_*.tsv'))
    if analysis_seq_files:
        total_size_mb = sum(f.stat().st_size for f in analysis_seq_files) / (1024 * 1024)
        print(f"  ✅ analysis_sequences_*.tsv ({len(analysis_seq_files)} files, {total_size_mb:.2f} MB total)")
    else:
        print(f"  ⚠️  analysis_sequences_*.tsv (No files found)")
    
    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()
    print(f"✅ Full genome pass completed successfully!")
    print()
    print(f"Base Model: {args.base_model.upper()}")
    print(f"Total Genes: {summary['total_genes']:,}")
    print(f"Total Positions: {summary['total_positions']:,}")
    print()
    print(f"Performance Metrics:")
    print(f"  F1 Score:          {metrics['f1']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")
    print(f"  Top-K Accuracy:    {metrics['top_k_accuracy']:.4f}")
    print()
    print(f"Artifacts Directory:")
    print(f"  {artifacts_dir}")
    print()
    print(f"Summary saved to:")
    print(f"  {summary_path}")
    print()
    print("=" * 80)
    print()
    print("🎉 Ready for meta-learning training!")
    print()
    print("Next steps:")
    print("  1. Review evaluation metrics in evaluation_metrics.json")
    print("  2. Verify artifacts are complete")
    print("  3. Proceed to meta-learning training pipeline")
    print()


if __name__ == '__main__':
    main()

