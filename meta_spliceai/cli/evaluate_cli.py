#!/usr/bin/env python3
"""
CLI entry point for evaluating base model predictions.

This provides tools for computing metrics and analyzing base model performance.

Usage:
    # Evaluate existing artifacts
    meta-spliceai-eval --artifacts-dir data/mane/GRCh38/openspliceai_eval/meta_models
    
    # Quick summary
    meta-spliceai-eval --artifacts-dir data/mane/GRCh38/openspliceai_eval/meta_models --summary
"""

import sys
import argparse
from pathlib import Path

from meta_spliceai.splice_engine.meta_models.utils.comprehensive_evaluation import (
    evaluate_full_genome_pass
)


def main():
    """Main CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        prog='meta-spliceai-eval',
        description='Evaluate base model prediction artifacts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate artifacts directory
  meta-spliceai-eval --artifacts-dir data/mane/GRCh38/openspliceai_eval/meta_models
  
  # Quick summary only
  meta-spliceai-eval --artifacts-dir data/mane/GRCh38/openspliceai_eval/meta_models --summary
"""
    )
    
    parser.add_argument(
        '--artifacts-dir',
        type=str,
        required=True,
        help='Directory containing base model prediction artifacts'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show summary only (no detailed metrics)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=None,
        help='Save metrics to JSON file'
    )
    parser.add_argument(
        '--verbosity',
        type=int,
        default=1,
        choices=[0, 1, 2],
        help='Output verbosity (default: 1)'
    )
    
    args = parser.parse_args()
    
    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        print(f"❌ Error: Artifacts directory not found: {artifacts_dir}", file=sys.stderr)
        return 1
    
    print("=" * 80)
    print("META-SPLICEAI EVALUATION")
    print("=" * 80)
    print(f"Artifacts: {artifacts_dir}")
    print("=" * 80)
    print()
    
    try:
        # Check for positions file
        positions_file = artifacts_dir / "full_splice_positions_enhanced.tsv"
        if not positions_file.exists():
            print(f"❌ Error: Positions file not found: {positions_file}", file=sys.stderr)
            return 1
        
        # Run evaluation
        metrics = evaluate_full_genome_pass(
            artifacts_dir=str(artifacts_dir),
            verbose=args.verbosity >= 1
        )
        
        # Print summary
        print()
        print("=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"F1 Score:          {metrics.get('f1', 'N/A'):.4f}")
        print(f"Precision:         {metrics.get('precision', 'N/A'):.4f}")
        print(f"Recall:            {metrics.get('recall', 'N/A'):.4f}")
        print(f"Accuracy:          {metrics.get('accuracy', 'N/A'):.4f}")
        
        if not args.summary and 'roc_auc' in metrics:
            print(f"\nRanking Metrics:")
            print(f"  ROC-AUC:          {metrics.get('roc_auc', 'N/A'):.4f}")
            print(f"  Average Precision: {metrics.get('average_precision', 'N/A'):.4f}")
        
        print("=" * 80)
        
        # Save to file if requested
        if args.output_file:
            import json
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\n✅ Metrics saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        if args.verbosity >= 2:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

