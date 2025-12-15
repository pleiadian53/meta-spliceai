#!/usr/bin/env python3
"""
Quick test to verify meta-model is now being applied correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
    EnhancedSelectiveInferenceConfig,
    EnhancedSelectiveInferenceWorkflow
)

def main():
    print("=" * 80)
    print("🧪 TESTING META-MODEL APPLICATION FIX")
    print("=" * 80)
    print()
    
    # Test gene
    gene_id = "ENSG00000134202"  # GSTM3
    gene_name = "GSTM3"
    
    print(f"Test gene: {gene_name} ({gene_id})")
    print()
    
    # Test meta-only mode
    print("─" * 80)
    print("Testing META-ONLY Mode")
    print("─" * 80)
    print()
    
    config = EnhancedSelectiveInferenceConfig(
        model_path=Path("results/gene_cv_1000_run_15/ablation_study/full/model_full.pkl"),
        inference_mode="meta_only",
        target_genes=[gene_id],
        inference_base_dir=Path("data/ensembl/spliceai_eval/meta_models/inference"),
        uncertainty_threshold_high=0.50,  # Lower threshold to enable meta-model
        verbose=1,
    )
    
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    
    try:
        results = workflow.run()
        print()
        print("=" * 80)
        print("✅ META-ONLY MODE TEST COMPLETE")
        print("=" * 80)
        print()
        print(f"Predictions saved to:")
        print(f"  {results.output_path}")
        print()
        
        # Check if predictions were generated
        import polars as pl
        pred_file = Path(f"data/ensembl/spliceai_eval/meta_models/inference/predictions/meta_only/per_gene/{gene_id}_predictions.parquet")
        
        if pred_file.exists():
            df = pl.read_parquet(pred_file)
            print(f"Predictions generated: {len(df)} positions")
            
            # Check meta-model application
            if 'is_adjusted' in df.columns:
                n_adjusted = df['is_adjusted'].sum()
                pct_adjusted = n_adjusted / len(df) * 100
                print(f"Meta-model applied: {n_adjusted}/{len(df)} ({pct_adjusted:.1f}%)")
                
                if pct_adjusted >= 80:
                    print("  ✅ Expected: ~100% application in meta-only mode")
                elif pct_adjusted > 10:
                    print("  ⚠️  Partial application - check feature generation")
                else:
                    print("  ❌ FAILED: Meta-model barely applied")
            
            # Check if scores differ
            import numpy as np
            donor_base = df['donor_score'].to_numpy()
            donor_meta = df['donor_meta'].to_numpy()
            
            diff = np.abs(donor_base - donor_meta)
            n_diff = np.sum(diff > 1e-6)
            pct_diff = n_diff / len(df) * 100
            
            print(f"\nScore differences:")
            print(f"  Positions where donor_meta ≠ donor_score: {n_diff}/{len(df)} ({pct_diff:.1f}%)")
            if pct_diff >= 80:
                print("  ✅ Expected: Most positions recalibrated")
            elif pct_diff > 10:
                print("  ⚠️  Some positions recalibrated")
            else:
                print("  ❌ FAILED: No recalibration detected")
                
        else:
            print(f"❌ Prediction file not found: {pred_file}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

