"""
Test OutputManager Integration in Inference Workflow

Tests that the new output_resources module is properly integrated into
the enhanced_selective_inference workflow.

Created: 2025-10-28
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


def test_output_manager_initialization():
    """Test that OutputManager is properly initialized."""
    print("="*80)
    print("TEST 1: OutputManager Initialization")
    print("="*80)
    
    # Create minimal config
    config = EnhancedSelectiveInferenceConfig(
        target_genes=['ENSG00000169239'],
        model_path=Path('results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl'),
        inference_mode='hybrid',
        verbose=2
    )
    
    # Initialize workflow
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    
    # Check OutputManager was created
    assert hasattr(workflow, 'output_manager'), "OutputManager not created"
    print("✅ OutputManager created")
    
    # Check output paths
    test_gene = 'ENSG00000169239'
    paths = workflow.output_manager.get_gene_output_paths(test_gene)
    
    print(f"\n📁 Output Paths for {test_gene}:")
    print(f"  Gene directory: {paths.gene_dir}")
    print(f"  Predictions file: {paths.predictions_file}")
    print(f"  Artifacts directory: {paths.artifacts_dir}")
    print(f"  Analysis sequences: {paths.analysis_sequences_dir}")
    print(f"  Base predictions: {paths.base_predictions_dir}")
    
    # Verify paths are under predictions/
    predictions_base = workflow.output_manager.registry.resolve('predictions_base')
    assert str(paths.artifacts_dir).startswith(str(predictions_base)), \
        "Artifacts not under predictions/"
    print("\n✅ All paths under predictions/ (single directory)")
    
    # Verify mode-based organization
    assert workflow.config.inference_mode in str(paths.gene_dir), \
        "Mode not in gene directory path"
    print(f"✅ Mode-based organization: {workflow.config.inference_mode}")
    
    return True


def test_combined_output_path():
    """Test combined output path."""
    print("\n" + "="*80)
    print("TEST 2: Combined Output Path")
    print("="*80)
    
    config = EnhancedSelectiveInferenceConfig(
        target_genes=['ENSG00000169239'],
        model_path=Path('results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl'),
        inference_mode='meta_only'
    )
    
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    
    combined_path = workflow.output_manager.get_combined_output_path()
    print(f"📄 Combined predictions path: {combined_path}")
    
    # Should be mode-specific
    assert 'meta_only' in str(combined_path), "Mode not in combined path"
    print("✅ Combined path is mode-specific")
    
    return True


def test_base_model_configuration():
    """Test that base_model_name is configurable."""
    print("\n" + "="*80)
    print("TEST 3: Base Model Configuration")
    print("="*80)
    
    config = EnhancedSelectiveInferenceConfig(
        target_genes=['ENSG00000169239'],
        model_path=Path('results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl'),
        inference_mode='hybrid'
    )
    
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    
    # Check default base model
    artifacts_dir = workflow.output_manager.registry.resolve('artifacts')
    print(f"📁 Artifacts directory: {artifacts_dir}")
    
    # Should contain spliceai_eval (default)
    assert 'spliceai_eval' in str(artifacts_dir), "Default base model not 'spliceai'"
    print("✅ Default base model: spliceai")
    
    # Verify it's under predictions/
    predictions_base = workflow.output_manager.registry.resolve('predictions_base')
    assert str(artifacts_dir).startswith(str(predictions_base)), \
        "Artifacts not under predictions/"
    print("✅ Artifacts under predictions/spliceai_eval/meta_models/")
    
    return True


def test_backward_compatibility():
    """Test that self.output_dir still works for backward compatibility."""
    print("\n" + "="*80)
    print("TEST 4: Backward Compatibility")
    print("="*80)
    
    config = EnhancedSelectiveInferenceConfig(
        target_genes=['ENSG00000169239'],
        model_path=Path('results/meta_model_1000genes_3mers_fresh/model_multiclass.pkl'),
        inference_mode='base_only'
    )
    
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    
    # Old code may still reference self.output_dir
    assert hasattr(workflow, 'output_dir'), "output_dir not set"
    print(f"📁 Legacy output_dir: {workflow.output_dir}")
    
    # Should match OutputManager's mode directory
    mode_dir = workflow.output_manager.registry.get_mode_dir('base_only', is_test=False)
    assert workflow.output_dir == mode_dir, "output_dir doesn't match OutputManager"
    print("✅ Backward compatibility maintained (self.output_dir available)")
    
    return True


def main():
    """Run all tests."""
    print("\n")
    print("🧪 Testing OutputManager Integration")
    print("="*80)
    
    tests = [
        test_output_manager_initialization,
        test_combined_output_path,
        test_base_model_configuration,
        test_backward_compatibility
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    print(f"✅ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(tests)}")
    else:
        print("🎉 All tests passed!")
    print("="*80)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

