#!/usr/bin/env python3
"""
Base Model vs Meta-Layer Comparison

This script compares variant effect prediction between:
1. Base model only (OpenSpliceAI or SpliceAI)
2. Base model + Meta-layer (multimodal approach)

The comparison evaluates on SpliceVarDB variants to assess whether the 
meta-layer improves detection of splice-altering variants.

Usage:
    # Quick test (5 epochs, 5000 samples)
    python run_base_vs_meta_comparison.py --quick

    # Full training
    python run_base_vs_meta_comparison.py --epochs 30 --base-model openspliceai
    
    # With specific output directory
    python run_base_vs_meta_comparison.py --output-dir ./experiments/variant_comparison

Renamed from 'run_phase1_comparison.py' for clarity.
"""

import argparse
import logging
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_paths():
    """Add project root to path."""
    import sys
    project_root = Path(__file__).resolve().parents[5]  # meta-spliceai root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def run_comparison(
    base_model: str = 'openspliceai',
    epochs: int = 30,
    max_train_samples: int = None,
    eval_max_variants: int = 1000,
    output_dir: str = './phase1_comparison',
    sequence_encoder: str = 'cnn',  # 'cnn' for M1 Mac, 'hyenadna' for GPU
    quick_test: bool = False
):
    """
    Run Phase 1 comparison between base model and meta-layer.
    
    Parameters
    ----------
    base_model : str
        Base model to use ('openspliceai' or 'spliceai')
    epochs : int
        Number of training epochs
    max_train_samples : int, optional
        Maximum training samples (None for all)
    eval_max_variants : int
        Maximum variants for evaluation
    output_dir : str
        Output directory for results
    sequence_encoder : str
        Sequence encoder type ('cnn' or 'hyenadna')
    quick_test : bool
        If True, use reduced settings for quick testing
    
    Returns
    -------
    dict
        Comparison results
    """
    setup_paths()
    
    # Import after path setup
    # Using descriptive aliases (CanonicalSplice*) instead of Phase1*
    from meta_spliceai.splice_engine.meta_layer.workflows import (
        CanonicalSpliceConfig,
        CanonicalSpliceWorkflow,
        run_canonical_splice_training
    )
    
    # Configure for quick test if requested
    if quick_test:
        epochs = 5
        max_train_samples = 5000
        eval_max_variants = 100
        logger.info("Running in QUICK TEST mode (5 epochs, 5000 samples, 100 variants)")
    
    # Create config
    config = CanonicalSpliceConfig(
        base_model=base_model,
        epochs=epochs,
        output_dir=output_dir,
        sequence_encoder=sequence_encoder,
        max_train_samples=max_train_samples,
        eval_max_variants=eval_max_variants,
        test_chromosomes=['21', '22'],  # Held-out chromosomes
    )
    
    logger.info("=" * 70)
    logger.info("PHASE 1 COMPARISON: Base Model vs Meta-Layer")
    logger.info("=" * 70)
    logger.info(f"Base model: {base_model}")
    logger.info(f"Genome build: {config.genome_build}")
    logger.info(f"Sequence encoder: {sequence_encoder}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Max train samples: {max_train_samples or 'all'}")
    logger.info(f"Eval variants: {eval_max_variants}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 70)
    
    # Run workflow
    workflow = CanonicalSpliceWorkflow(config)
    result = workflow.run()
    
    # Generate comparison summary
    comparison = generate_comparison_summary(result)
    
    # Save comparison
    comparison_path = Path(output_dir) / 'comparison_summary.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print comparison
    print_comparison_summary(comparison)
    
    return comparison


def generate_comparison_summary(result) -> dict:
    """Generate a summary comparing base model vs meta-layer."""
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'base_model': result.config.base_model,
            'genome_build': result.config.genome_build,
            'epochs': result.config.epochs,
            'sequence_encoder': result.config.sequence_encoder,
        },
        'training': {
            'best_epoch': result.training_result.best_epoch,
            'training_time_seconds': result.training_result.total_time_seconds,
        },
        'canonical_performance': result.canonical_test_metrics,
    }
    
    # Add variant evaluation if available
    if result.variant_evaluation:
        var_eval = result.variant_evaluation
        summary['variant_evaluation'] = {
            'total_variants': var_eval.total_variants,
            
            # Detection accuracy
            'base_accuracy': var_eval.base_accuracy,
            'meta_accuracy': var_eval.meta_accuracy,
            'accuracy_improvement': var_eval.meta_accuracy - var_eval.base_accuracy,
            
            # Detection rates for splice-altering variants
            'base_detection_rate': var_eval.base_detection_rate,
            'meta_detection_rate': var_eval.meta_detection_rate,
            'detection_rate_improvement': var_eval.meta_detection_rate - var_eval.base_detection_rate,
            
            # False positive rates
            'base_false_positive_rate': var_eval.base_false_positive_rate,
            'meta_false_positive_rate': var_eval.meta_false_positive_rate,
            
            # Improvement counts
            'improvement_count': var_eval.improvement_count,
            'degradation_count': var_eval.degradation_count,
            'net_improvement': var_eval.improvement_count - var_eval.degradation_count,
            
            # Delta magnitudes
            'mean_base_delta_splice_altering': var_eval.mean_base_delta_splice_altering,
            'mean_meta_delta_splice_altering': var_eval.mean_meta_delta_splice_altering,
            'mean_base_delta_non_splice_altering': var_eval.mean_base_delta_non_splice_altering,
            'mean_meta_delta_non_splice_altering': var_eval.mean_meta_delta_non_splice_altering,
        }
        
        # Key metric: Does meta-layer help?
        summary['conclusion'] = {
            'meta_layer_helps_detection': var_eval.meta_detection_rate > var_eval.base_detection_rate,
            'meta_layer_reduces_false_positives': var_eval.meta_false_positive_rate < var_eval.base_false_positive_rate,
            'net_improvement_positive': (var_eval.improvement_count - var_eval.degradation_count) > 0,
        }
    
    return summary


def print_comparison_summary(summary: dict):
    """Print a formatted comparison summary."""
    
    print("\n")
    print("=" * 70)
    print("                 COMPARISON SUMMARY: BASE vs META-LAYER")
    print("=" * 70)
    
    config = summary.get('config', {})
    print(f"\n📋 Configuration:")
    print(f"   Base model:       {config.get('base_model', 'N/A')}")
    print(f"   Genome build:     {config.get('genome_build', 'N/A')}")
    print(f"   Sequence encoder: {config.get('sequence_encoder', 'N/A')}")
    
    training = summary.get('training', {})
    print(f"\n🏋️ Training:")
    print(f"   Best epoch:       {training.get('best_epoch', 'N/A')}")
    print(f"   Training time:    {training.get('training_time_seconds', 0):.1f}s")
    
    canonical = summary.get('canonical_performance', {})
    print(f"\n📊 Canonical Splice Site Performance (Test Set):")
    print(f"   Accuracy:         {canonical.get('accuracy', 0):.4f}")
    print(f"   PR-AUC (macro):   {canonical.get('pr_auc_macro', 0):.4f}")
    
    var_eval = summary.get('variant_evaluation', {})
    if var_eval:
        print(f"\n🧬 Variant Effect Detection (SpliceVarDB):")
        print(f"   Total variants:   {var_eval.get('total_variants', 0)}")
        
        print(f"\n   DETECTION ACCURACY:")
        base_acc = var_eval.get('base_accuracy', 0)
        meta_acc = var_eval.get('meta_accuracy', 0)
        acc_imp = var_eval.get('accuracy_improvement', 0)
        print(f"   ┌─────────────────────────────────────────────────────┐")
        print(f"   │  Base model:  {base_acc:>6.1%}                            │")
        print(f"   │  Meta-layer:  {meta_acc:>6.1%}  ({acc_imp:+.1%})                  │")
        print(f"   └─────────────────────────────────────────────────────┘")
        
        print(f"\n   SPLICE-ALTERING DETECTION RATE:")
        base_det = var_eval.get('base_detection_rate', 0)
        meta_det = var_eval.get('meta_detection_rate', 0)
        det_imp = var_eval.get('detection_rate_improvement', 0)
        print(f"   ┌─────────────────────────────────────────────────────┐")
        print(f"   │  Base model:  {base_det:>6.1%}                            │")
        print(f"   │  Meta-layer:  {meta_det:>6.1%}  ({det_imp:+.1%})                  │")
        print(f"   └─────────────────────────────────────────────────────┘")
        
        print(f"\n   FALSE POSITIVE RATE (non-splice-altering):")
        base_fp = var_eval.get('base_false_positive_rate', 0)
        meta_fp = var_eval.get('meta_false_positive_rate', 0)
        print(f"   ┌─────────────────────────────────────────────────────┐")
        print(f"   │  Base model:  {base_fp:>6.1%}                            │")
        print(f"   │  Meta-layer:  {meta_fp:>6.1%}                            │")
        print(f"   └─────────────────────────────────────────────────────┘")
        
        print(f"\n   IMPROVEMENT ANALYSIS:")
        imp = var_eval.get('improvement_count', 0)
        deg = var_eval.get('degradation_count', 0)
        net = var_eval.get('net_improvement', 0)
        print(f"   ┌─────────────────────────────────────────────────────┐")
        print(f"   │  Meta improved (base missed): {imp:>4} variants         │")
        print(f"   │  Meta degraded (base caught): {deg:>4} variants         │")
        print(f"   │  Net improvement:             {net:>+4} variants         │")
        print(f"   └─────────────────────────────────────────────────────┘")
        
        print(f"\n   MEAN DELTA MAGNITUDES:")
        print(f"   ┌─────────────────────────────────────────────────────┐")
        print(f"   │  Splice-altering (base):      {var_eval.get('mean_base_delta_splice_altering', 0):.4f}              │")
        print(f"   │  Splice-altering (meta):      {var_eval.get('mean_meta_delta_splice_altering', 0):.4f}              │")
        print(f"   │  Non-splice-altering (base):  {var_eval.get('mean_base_delta_non_splice_altering', 0):.4f}              │")
        print(f"   │  Non-splice-altering (meta):  {var_eval.get('mean_meta_delta_non_splice_altering', 0):.4f}              │")
        print(f"   └─────────────────────────────────────────────────────┘")
        
        # Conclusion
        conclusion = summary.get('conclusion', {})
        print(f"\n   📌 CONCLUSION:")
        if conclusion.get('meta_layer_helps_detection'):
            print(f"   ✅ Meta-layer IMPROVES splice-altering variant detection")
        else:
            print(f"   ❌ Meta-layer does NOT improve detection (needs more training)")
        
        if conclusion.get('meta_layer_reduces_false_positives'):
            print(f"   ✅ Meta-layer REDUCES false positive rate")
        else:
            print(f"   ⚠️  Meta-layer does NOT reduce false positives")
        
        if conclusion.get('net_improvement_positive'):
            print(f"   ✅ Net improvement is POSITIVE ({net} variants)")
        else:
            print(f"   ⚠️  Net improvement is NOT positive")
    
    print("\n" + "=" * 70)
    print("                          END OF COMPARISON")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1 Comparison: Base Model vs Meta-Layer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (5 epochs, limited data)
  python run_phase1_comparison.py --quick
  
  # Full training with OpenSpliceAI
  python run_phase1_comparison.py --epochs 30 --base-model openspliceai
  
  # Full training with SpliceAI (GRCh37)
  python run_phase1_comparison.py --epochs 30 --base-model spliceai
  
  # With specific output directory
  python run_phase1_comparison.py --output-dir ./experiments/phase1_v1
        """
    )
    
    parser.add_argument(
        '--base-model', 
        default='openspliceai',
        choices=['openspliceai', 'spliceai'],
        help='Base model to use (default: openspliceai)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=30,
        help='Number of training epochs (default: 30)'
    )
    parser.add_argument(
        '--max-train-samples', 
        type=int, 
        default=None,
        help='Maximum training samples (default: all)'
    )
    parser.add_argument(
        '--eval-max-variants', 
        type=int, 
        default=1000,
        help='Maximum variants for evaluation (default: 1000)'
    )
    parser.add_argument(
        '--output-dir', 
        default='./phase1_comparison',
        help='Output directory (default: ./phase1_comparison)'
    )
    parser.add_argument(
        '--sequence-encoder', 
        default='cnn',
        choices=['cnn', 'hyenadna'],
        help='Sequence encoder (default: cnn for M1 Mac compatibility)'
    )
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Quick test mode (5 epochs, 5000 samples, 100 variants)'
    )
    
    args = parser.parse_args()
    
    run_comparison(
        base_model=args.base_model,
        epochs=args.epochs,
        max_train_samples=args.max_train_samples,
        eval_max_variants=args.eval_max_variants,
        output_dir=args.output_dir,
        sequence_encoder=args.sequence_encoder,
        quick_test=args.quick
    )


if __name__ == '__main__':
    main()

