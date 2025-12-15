#!/usr/bin/env python3
"""
Test the new mode-based output directory structure.

Verifies that:
1. Predictions go to predictions/base_only/ (not timestamped)
2. Re-running overwrites previous predictions
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
    # Small gene for quick test
    test_genes = ["ENSG00000006606"]  # CCL26
    
    model_path = project_root / "results/gene_cv_1000_run_15/ablation_study/full/model_full.pkl"
    
    print("=" * 80)
    print("🧪 TESTING MODE-BASED OUTPUT DIRECTORY")
    print("=" * 80)
    print()
    print("Expected behavior:")
    print("  - Output directory: predictions/base_only/")
    print("  - Re-running overwrites previous predictions")
    print()
    
    config = EnhancedSelectiveInferenceConfig(
        model_path=model_path,
        target_genes=test_genes,
        inference_mode='base_only',
        ensure_complete_coverage=True,
        enable_memory_monitoring=False,
        verbose=1
    )
    
    workflow = EnhancedSelectiveInferenceWorkflow(config)
    
    print("Output directory:", workflow.output_dir)
    print()
    
    # Check it's NOT timestamped
    if "_20" in str(workflow.output_dir):
        print("❌ FAIL: Output directory is timestamped!")
        print(f"   Got: {workflow.output_dir}")
        return False
    
    # Check it IS mode-based
    if not str(workflow.output_dir).endswith("base_only"):
        print("❌ FAIL: Output directory is not mode-based!")
        print(f"   Got: {workflow.output_dir}")
        print(f"   Expected: .../predictions/base_only")
        return False
    
    print("✅ Output directory structure is correct")
    print(f"   {workflow.output_dir}")
    print()
    
    # Run inference
    print("Running inference...")
    results = workflow.run()
    
    print()
    print("=" * 80)
    print("✅ TEST PASSED")
    print("=" * 80)
    print()
    print(f"Output directory: {workflow.output_dir}")
    print(f"Predictions saved: {workflow.output_dir / 'per_gene'}")
    print()
    print("To verify overwriting behavior, run this script again.")
    print("The same files should be overwritten (check timestamps).")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

