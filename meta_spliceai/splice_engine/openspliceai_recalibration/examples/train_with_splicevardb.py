#!/usr/bin/env python3
"""
Complete Example: Train OpenSpliceAI Recalibration Model with SpliceVarDB

This script demonstrates the complete workflow for training a recalibration
model using SpliceVarDB's experimentally validated splice variants.

Usage:
------
# Basic usage with demo data
python train_with_splicevardb.py

# With real data
python train_with_splicevardb.py \
    --data-dir ./data/splicevardb \
    --reference-genome /path/to/hg38.fa \
    --output-dir ./models/recalibration \
    --splicevardb-token YOUR_TOKEN

# Quick test with limited variants
python train_with_splicevardb.py --max-variants 100 --test-mode
"""

import os
import sys
from pathlib import Path
import argparse
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))

from openspliceai_recalibration.workflows import SpliceVarDBTrainingPipeline
from openspliceai_recalibration.workflows.splicevardb_pipeline import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Train OpenSpliceAI recalibration with SpliceVarDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        default="./data/splicevardb",
        help="Directory for SpliceVarDB data"
    )
    parser.add_argument(
        "--reference-genome",
        help="Path to reference genome FASTA (e.g., hg38.fa)"
    )
    parser.add_argument(
        "--splicevardb-token",
        default=os.getenv("SPLICEVARDB_TOKEN"),
        help="SpliceVarDB API token (or set SPLICEVARDB_TOKEN env var)"
    )
    
    # Model arguments
    parser.add_argument(
        "--openspliceai-models",
        help="Directory containing OpenSpliceAI models (default: auto-detect)"
    )
    parser.add_argument(
        "--recalibration-method",
        default="isotonic",
        choices=["isotonic", "platt", "xgboost"],
        help="Recalibration method"
    )
    
    # Training arguments
    parser.add_argument(
        "--cv-strategy",
        default="gene_holdout",
        choices=["gene_holdout", "region_stratified", "random"],
        help="Cross-validation strategy"
    )
    parser.add_argument(
        "--test-genes",
        nargs="+",
        default=["CFTR", "BRCA1", "BRCA2"],
        help="Genes to hold out for testing"
    )
    parser.add_argument(
        "--feature-set",
        default="delta_full",
        choices=["delta_basic", "delta_full", "delta_plus_context"],
        help="Feature set to use"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        default="./models/openspliceai_recalibration",
        help="Output directory for models and reports"
    )
    
    # Runtime arguments
    parser.add_argument(
        "--max-variants",
        type=int,
        help="Maximum variants to process (for testing)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with demo data"
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recompute predictions even if cached"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for OpenSpliceAI inference"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="Increase verbosity (-v, -vv, -vvv)"
    )
    
    args = parser.parse_args()
    
    # Set environment variable if token provided
    if args.splicevardb_token:
        os.environ["SPLICEVARDB_TOKEN"] = args.splicevardb_token
    
    # Print banner
    print("=" * 70)
    print("  OPENSPLICEAI RECALIBRATION TRAINING WITH SPLICEVARDB")
    print("=" * 70)
    print()
    
    # Create configuration
    config = PipelineConfig(
        data_dir=args.data_dir,
        genome_build="GRCh38",
        reference_genome=args.reference_genome,
        openspliceai_model_dir=args.openspliceai_models,
        recalibration_method=args.recalibration_method,
        cv_strategy=args.cv_strategy,
        test_genes=args.test_genes,
        feature_set=args.feature_set,
        output_dir=args.output_dir,
        verbose=args.verbose,
        device=args.device,
    )
    
    # Print configuration
    print("Configuration:")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Reference genome: {config.reference_genome or 'Demo mode (no real sequences)'}")
    print(f"  OpenSpliceAI models: {config.openspliceai_model_dir or 'Auto-detect'}")
    print(f"  Recalibration method: {config.recalibration_method}")
    print(f"  CV strategy: {config.cv_strategy}")
    print(f"  Feature set: {config.feature_set}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Device: {config.device}")
    if args.max_variants:
        print(f"  Max variants: {args.max_variants} (test mode)")
    print()
    
    # Test mode message
    if args.test_mode or not config.reference_genome:
        print("⚠️  RUNNING IN TEST MODE")
        print("   Using demo data and placeholder sequences.")
        print("   For real training, provide --reference-genome")
        print()
    
    try:
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = SpliceVarDBTrainingPipeline(config=config)
        
        # Run pipeline
        logger.info("Starting training pipeline...")
        results = pipeline.run(
            force_recompute=args.force_recompute,
            max_variants=args.max_variants
        )
        
        # Print results
        print()
        print("=" * 70)
        print("  TRAINING RESULTS")
        print("=" * 70)
        print()
        
        metrics = results["metrics"]
        print("Performance Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()
        
        report = results["report"]
        print("Data Statistics:")
        stats = report["data_statistics"]
        print(f"  Total variants: {stats['total_variants']}")
        print(f"  Splice-altering: {stats['splice_altering']}")
        print(f"  Not splice-altering: {stats['not_splice_altering']}")
        print(f"  Unique genes: {stats['unique_genes']}")
        print()
        
        print(f"✅ Training complete!")
        print(f"   Results saved to: {results['output_dir']}")
        print()
        
        # Next steps
        print("Next Steps:")
        print("  1. Review training report:")
        print(f"     cat {results['output_dir']}/training_report.json")
        print("  2. Inspect features:")
        print(f"     cat {results['output_dir']}/training_features.parquet")
        print("  3. Use trained model for prediction (see inference example)")
        print()
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print()
        print("❌ Error: Required files not found")
        print()
        print("Please ensure:")
        print("  1. OpenSpliceAI models are downloaded:")
        print("     ./scripts/base_model/download_openspliceai_models.sh")
        print("  2. SpliceVarDB data is accessible (or will download automatically)")
        print("  3. Reference genome is provided for real training:")
        print("     --reference-genome /path/to/hg38.fa")
        print()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print()
        print(f"❌ Pipeline failed: {e}")
        print()
        if args.verbose < 2:
            print("Run with -vv for detailed error information")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()





