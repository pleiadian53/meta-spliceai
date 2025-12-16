#!/usr/bin/env python3
"""
Test script for the Complete Coverage Inference Workflow

This script demonstrates the corrected inference approach that ensures:
1. Complete position coverage (no gaps) for target genes
2. Base model predictions for ALL positions  
3. Selective meta-model application based only on uncertainty
4. Proper output schema with all required columns

Usage:
    python test_complete_coverage.py --gene ENSG00000236172
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[4]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.inference.complete_coverage_workflow import (
    run_complete_coverage_inference
)


def main():
    """Test the complete coverage inference workflow."""
    parser = argparse.ArgumentParser(description="Test complete coverage inference workflow")
    parser.add_argument(
        "--gene", 
        default="ENSG00000236172",
        help="Gene ID to test (default: ENSG00000236172)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/test_complete_coverage",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Configuration
    target_genes = [args.gene]
    base_model_path = "results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl"
    meta_model_path = "results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl"  # Same for now
    training_dataset_path = "train_pc_1000_3mers"
    output_dir = args.output_dir
    
    print("🧬 Testing Complete Coverage Inference Workflow")
    print("=" * 60)
    print(f"Target gene: {args.gene}")
    print(f"Base model: {base_model_path}")
    print(f"Training dataset: {training_dataset_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Run the workflow
    result = run_complete_coverage_inference(
        target_genes=target_genes,
        base_model_path=base_model_path,
        meta_model_path=meta_model_path,
        training_dataset_path=training_dataset_path,
        output_dir=output_dir,
        uncertainty_threshold=0.5
    )
    
    # Display results
    if result['success']:
        print("\n✅ Complete Coverage Inference Successful!")
        print(f"📊 Total positions: {result['total_positions']}")
        print(f"🧠 Meta-model applied: {result['meta_model_applied']} ({result['meta_application_rate']:.1%})")
        print(f"✅ Continuous coverage: {result['continuous_coverage']}")
        print(f"⏱️  Runtime: {result['runtime_seconds']:.1f} seconds")
        print(f"📁 Results saved to: {result['output_file']}")
        
        # Quick validation
        import polars as pl
        predictions_df = pl.read_parquet(result['output_file'])
        
        print("\n📋 Quick Validation:")
        print(f"   Rows in output: {predictions_df.height}")
        print(f"   Columns: {len(predictions_df.columns)}")
        print(f"   Required columns present: {all(col in predictions_df.columns for col in ['gene_id', 'position', 'donor_score', 'acceptor_score', 'neither_score', 'donor_meta', 'acceptor_meta', 'neither_meta', 'splice_type', 'is_adjusted'])}")
        
        # Check position continuity
        gene_positions = predictions_df.filter(pl.col('gene_id') == args.gene)['position'].sort()
        positions_list = gene_positions.to_list()
        gaps = []
        for i in range(len(positions_list) - 1):
            if positions_list[i+1] - positions_list[i] > 1:
                gaps.append((positions_list[i], positions_list[i+1]))
        
        print(f"   Position range: {min(positions_list)} to {max(positions_list)}")
        print(f"   Position gaps: {len(gaps)}")
        print(f"   Continuous coverage: {len(gaps) == 0}")
        
        if gaps:
            print(f"   First few gaps: {gaps[:5]}")
        
    else:
        print(f"\n❌ Complete Coverage Inference Failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()