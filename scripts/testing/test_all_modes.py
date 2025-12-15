#!/usr/bin/env python3
"""
Test all three inference modes to verify they work correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from meta_spliceai.splice_engine.meta_models.workflows.inference.enhanced_selective_inference import (
    EnhancedSelectiveInferenceWorkflow,
    EnhancedSelectiveInferenceConfig
)

def test_mode(mode: str, test_genes: list):
    """Test a specific inference mode."""
    print(f"\n{'='*60}")
    print(f"Testing {mode.upper()} mode")
    print(f"{'='*60}")
    
    config = EnhancedSelectiveInferenceConfig(
        model_path=Path('results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl'),
        target_genes=test_genes,
        inference_mode=mode,
        inference_base_dir=Path(f'predictions/test_{mode}')
    )
    
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    
    try:
        results = workflow.run()
        print(f"✅ {mode} mode completed successfully!")
        print(f"   Output: {results.combined_predictions_path}")
        print(f"   Genes processed: {len(test_genes)}")
        return True
    except Exception as e:
        print(f"❌ {mode} mode failed: {e}")
        return False

def main():
    """Test all three inference modes."""
    # Test genes (smaller set for quick testing)
    test_genes = ['ENSG00000141736']  # Just one gene for quick testing
    
    print("🧪 Testing all three inference modes...")
    
    modes = ['base_only', 'hybrid', 'meta_only']
    results = {}
    
    for mode in modes:
        results[mode] = test_mode(mode, test_genes)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for mode, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{mode:12}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All modes are working correctly!")
    else:
        print("\n⚠️  Some modes failed. Check the logs above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
