#!/usr/bin/env python3
"""
Test script for enriched inference workflow with genomic feature enrichment.

Directly tests the enhanced_selective_inference.py module with the integrated
GenomicFeatureEnricher to verify coordinate conversion and feature addition.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
    EnhancedSelectiveInferenceWorkflow,
    EnhancedSelectiveInferenceConfig
)

def main():
    # Test configuration
    model_path = project_root / "results/gene_cv_1000_run_15/ablation_study/full/model_full.pkl"
    
    # Test genes (same as Test 1a)
    test_genes = [
        "ENSG00000065413",  # NSD1 (chr2, minus strand, 344kb)
        "ENSG00000134202",  # GSTM1 (chr1, plus strand, ~23kb)
        "ENSG00000169239",  # TYRO3 (chr15, plus strand, ~29kb)
    ]
    
    print("=" * 80)
    print("🧬 TESTING ENRICHED INFERENCE WORKFLOW")
    print("=" * 80)
    print(f"Model: {model_path.name}")
    print(f"Genes: {', '.join(test_genes)}")
    print(f"Mode: base_only (no meta-model recalibration)")
    print()
    
    # Create config
    config = EnhancedSelectiveInferenceConfig(
        model_path=model_path,
        target_genes=test_genes,
        inference_mode='base_only',  # Base-only mode
        ensure_complete_coverage=True,
        enable_memory_monitoring=True,
        max_memory_gb=8.0,
        verbose=2
    )
    
    # Run workflow
    print("🚀 Starting inference workflow...")
    print()
    
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    results = workflow.run()
    
    print()
    print("=" * 80)
    print("✅ WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"Output directory: {workflow.output_dir}")
    print(f"Genes processed: {len(test_genes)}")
    print()
    
    # Validate output
    print("🔍 Validating enriched output...")
    import polars as pl
    
    # Check per-gene files
    per_gene_dir = workflow.output_dir / "per_gene"
    if per_gene_dir.exists():
        gene_files = list(per_gene_dir.glob("*.parquet"))
        print(f"   ✅ Found {len(gene_files)} per-gene prediction files")
        
        # Load first gene and check enrichment
        if gene_files:
            first_gene_file = gene_files[0]
            df = pl.read_parquet(first_gene_file)
            
            print(f"\n📊 Sample gene: {first_gene_file.stem}")
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {len(df.columns)}")
            
            # Check enrichment columns
            enriched_cols = [
                'absolute_position', 'gene_start', 'gene_end', 
                'gene_name', 'gene_type', 'distance_to_start', 'distance_to_end'
            ]
            
            print(f"\n   🧬 Genomic enrichment status:")
            for col in enriched_cols:
                if col in df.columns:
                    non_null = df[col].is_not_null().sum()
                    print(f"      ✅ {col}: {non_null}/{len(df)} ({100*non_null/len(df):.1f}%)")
                else:
                    print(f"      ❌ {col}: MISSING")
            
            # Show sample row
            print(f"\n   📍 Sample position (row 1000):")
            if len(df) > 1000:
                sample = df[1000]
                for col in ['gene_id', 'position', 'absolute_position', 'gene_start', 'gene_end', 'strand']:
                    if col in df.columns:
                        print(f"      {col}: {sample[col]}")
    else:
        print(f"   ❌ Per-gene directory not found: {per_gene_dir}")
    
    print()
    print("=" * 80)
    print("Test complete! Check output directory for full results.")
    print("=" * 80)

if __name__ == "__main__":
    main()


