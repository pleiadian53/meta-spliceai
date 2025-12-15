#!/usr/bin/env python3
"""
CLI entry point for running base model predictions.

This provides a user-friendly command-line interface for splice site prediction
using base models like SpliceAI or OpenSpliceAI.

Usage:
    # Single gene
    meta-spliceai-run --genes BRCA1 TP53
    
    # Chromosome
    meta-spliceai-run --chromosomes 21 --base-model openspliceai
    
    # Full genome pass for meta-learning
    meta-spliceai-run --base-model openspliceai --mode production --coverage full_genome
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

from meta_spliceai.run_base_model import run_base_model_predictions, BaseModelConfig
from meta_spliceai.splice_engine.meta_models.utils.comprehensive_evaluation import (
    evaluate_full_genome_pass
)


def main():
    """Main CLI entry point for base model predictions."""
    parser = argparse.ArgumentParser(
        prog='meta-spliceai-run',
        description='Run splice site predictions using base models (SpliceAI, OpenSpliceAI)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single gene analysis
  meta-spliceai-run --genes BRCA1 TP53
  
  # Chromosome analysis
  meta-spliceai-run --chromosomes 21 --base-model openspliceai
  
  # Full genome pass (production mode)
  meta-spliceai-run --base-model openspliceai --mode production --coverage full_genome
  
  # Specific chromosomes for training data
  meta-spliceai-run --chromosomes 1,2,3 --mode production --verbosity 2
  
  # Test mode with custom name
  meta-spliceai-run --genes BRCA1 --mode test --test-name brca1_validation
"""
    )
    
    # Model selection
    parser.add_argument(
        '--base-model',
        type=str,
        default='openspliceai',
        choices=['spliceai', 'openspliceai'],
        help='Base model to use (default: openspliceai)'
    )
    
    # Target selection (mutually exclusive in practice, but both can be used)
    target_group = parser.add_argument_group('target selection')
    target_group.add_argument(
        '--genes',
        type=str,
        nargs='+',
        help='Gene symbols or Ensembl IDs to analyze (e.g., BRCA1 TP53)'
    )
    target_group.add_argument(
        '--chromosomes',
        type=str,
        help='Comma-separated list of chromosomes (e.g., "1,2,X" or "21")'
    )
    
    # Mode and coverage
    parser.add_argument(
        '--mode',
        type=str,
        default='test',
        choices=['test', 'production'],
        help='Execution mode: test (overwritable) or production (immutable) (default: test)'
    )
    parser.add_argument(
        '--coverage',
        type=str,
        default='gene_subset',
        choices=['gene_subset', 'chromosome', 'full_genome'],
        help='Data coverage mode (default: gene_subset)'
    )
    parser.add_argument(
        '--test-name',
        type=str,
        default=None,
        help='Custom test name for test mode artifacts'
    )
    
    # Prediction parameters
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
    
    # Output control
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: auto-generated based on base model)'
    )
    parser.add_argument(
        '--verbosity',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Output verbosity (0=minimal, 1=normal, 2=detailed) (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Parse chromosomes
    target_chromosomes = None
    if args.chromosomes:
        target_chromosomes = [chr.strip() for chr in args.chromosomes.split(',')]
    
    # Generate test name if needed
    test_name = args.test_name
    if args.mode == 'test' and not test_name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.genes:
            gene_str = '_'.join(args.genes[:3])
            test_name = f"{args.base_model}_{gene_str}_{timestamp}"
        elif target_chromosomes:
            chr_str = '_'.join(target_chromosomes[:3])
            test_name = f"{args.base_model}_chr{chr_str}_{timestamp}"
        else:
            test_name = f"{args.base_model}_{timestamp}"
    
    # Create configuration
    config = BaseModelConfig(
        base_model=args.base_model,
        mode=args.mode,
        coverage=args.coverage,
        test_name=test_name,
        threshold=args.threshold,
        save_nucleotide_scores=args.save_nucleotide_scores
    )
    
    # Print header
    print("=" * 80)
    print("META-SPLICEAI BASE MODEL PREDICTIONS")
    print("=" * 80)
    print(f"Base Model:  {args.base_model}")
    print(f"Mode:        {args.mode}")
    print(f"Coverage:    {args.coverage}")
    if args.genes:
        print(f"Genes:       {', '.join(args.genes)}")
    if target_chromosomes:
        print(f"Chromosomes: {', '.join(target_chromosomes)}")
    print(f"Threshold:   {args.threshold}")
    print(f"TN Sampling: {'Disabled' if args.no_tn_sampling else 'Enabled'}")
    print("=" * 80)
    print()
    
    # Run predictions
    try:
        results = run_base_model_predictions(
            base_model=args.base_model,
            target_genes=args.genes,
            target_chromosomes=target_chromosomes,
            config=config,
            verbosity=args.verbosity,
            no_tn_sampling=args.no_tn_sampling,
            save_nucleotide_scores=args.save_nucleotide_scores
        )
        
        # Print summary
        print()
        print("=" * 80)
        print("PREDICTION COMPLETE")
        print("=" * 80)
        
        if 'positions' in results and results['positions'] is not None:
            positions_df = results['positions']
            print(f"Total positions analyzed: {len(positions_df)}")
            
            # Count by prediction type
            if 'pred_type' in positions_df.columns:
                for pred_type in ['TP', 'FP', 'FN', 'TN']:
                    count = len(positions_df.filter(positions_df['pred_type'] == pred_type))
                    if count > 0:
                        print(f"  {pred_type}: {count}")
        
        if 'metrics' in results:
            metrics = results['metrics']
            print(f"\nPerformance Metrics:")
            print(f"  F1 Score:  {metrics.get('f1', 'N/A'):.4f}")
            print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
            print(f"  Recall:    {metrics.get('recall', 'N/A'):.4f}")
        
        if 'output_dir' in results:
            print(f"\nOutput directory: {results['output_dir']}")
        
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        if args.verbosity >= 2:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

