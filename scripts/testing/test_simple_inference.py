#!/usr/bin/env python3
"""
Simple test to verify basic inference works.
Tests base-only mode on a single gene.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
    EnhancedSelectiveInferenceWorkflow,
    EnhancedSelectiveInferenceConfig
)

def main():
    print("\n" + "="*60)
    print("SIMPLE INFERENCE TEST - BASE-ONLY MODE")
    print("="*60)
    
    # Test a gene on chromosome 17 (should have sequences)
    test_gene = 'ENSG00000141736'  # ERBB2 on chr17
    output_dir = Path("predictions/simple_test")
    
    print(f"\nTest gene: {test_gene} (ERBB2)")
    print(f"Mode: base_only")
    print(f"Output: {output_dir}")
    
    # Create config
    # Note: model_path required even for base-only mode (not used)
    config = EnhancedSelectiveInferenceConfig(
        model_path=Path("results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl"),
        target_genes=[test_gene],
        inference_mode='base_only',
        inference_base_dir=output_dir
    )
    
    # Run inference
    print(f"\nRunning inference...")
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    
    try:
        results = workflow.run()
        
        if results.success:
            print(f"\n✅ Inference completed successfully!")
            print(f"   Predictions: {results.base_predictions_path}")
            return 0
        else:
            print(f"\n❌ Inference failed")
            return 1
            
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

