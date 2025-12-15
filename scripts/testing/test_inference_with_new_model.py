#!/usr/bin/env python3
"""
Test inference workflow with freshly trained model.

This script tests all 3 operational modes with the new model that includes k-mers.
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

def test_inference_modes():
    """Test all three inference modes with the new model."""
    
    # Test genes (mix of observed and unobserved)
    test_genes = {
        'observed': ['ENSG00000134202', 'ENSG00000169239', 'ENSG00000141510'],  # In training
        'unobserved': ['ENSG00000157764', 'ENSG00000141736', 'ENSG00000105810']  # Not in training
    }
    
    # New model path
    model_path = Path("results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl")
    
    print("=" * 80)
    print("🧪 TESTING INFERENCE WORKFLOW WITH NEW MODEL")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Features: 121 (including 64 k-mers)")
    print(f"Package: meta_spliceai (native)")
    print()
    
    for gene_type, genes in test_genes.items():
        print(f"\n📊 Testing {gene_type.upper()} genes:")
        print(f"   Genes: {', '.join(genes)}")
        print()
        
        for mode in ['base_only', 'hybrid', 'meta_only']:
            print(f"\n🔬 Mode: {mode}")
            print("-" * 60)
            
            try:
                # Configure workflow
                config = EnhancedSelectiveInferenceConfig(
                    target_genes=genes,
                    model_path=model_path,
                    inference_mode=mode,
                    inference_base_dir=Path("predictions"),
                    output_name=f"{mode}_{gene_type}",
                    uncertainty_threshold_low=0.10,
                    uncertainty_threshold_high=0.50,  # Lowered from 0.80
                    enable_memory_monitoring=True,
                    use_timestamped_output=False  # Overwrite previous runs
                )
                
                # Run inference
                workflow = EnhancedSelectiveInferenceWorkflow(config)
                results = workflow.run_incremental()
                
                print(f"✅ {mode} completed successfully!")
                print(f"   Output: {results.hybrid_predictions_path or results.base_predictions_path}")
                print(f"   Genes processed: {results.genes_processed}")
                print(f"   Total positions: {results.total_positions}")
                print(f"   Meta-model usage: {results.positions_recalibrated}/{results.total_positions} ({results.positions_recalibrated/results.total_positions*100:.1f}%)")
                
            except Exception as e:
                print(f"❌ {mode} failed: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("🎉 INFERENCE TESTING COMPLETE!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Compare prediction scores between modes")
    print("  2. Validate splice site detection accuracy")
    print("  3. Verify meta-model recalibration effects")
    print()

if __name__ == "__main__":
    test_inference_modes()

