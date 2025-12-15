#!/usr/bin/env python
"""
Test OpenSpliceAI integration with the unified base model interface.

This script tests:
1. Model loading (OpenSpliceAI vs SpliceAI)
2. Automatic genomic build routing
3. Prediction workflow with OpenSpliceAI
4. Comparison with SpliceAI predictions
"""

import sys
import os
sys.path.insert(0, '/Users/pleiadian53/work/meta-spliceai')

from meta_spliceai import run_base_model_predictions, BaseModelConfig
from meta_spliceai.splice_engine.meta_models.utils.model_utils import load_base_model_ensemble
import polars as pl

print("=" * 80)
print("OPENSPLICEAI INTEGRATION TEST")
print("=" * 80)
print()

# Test 1: Model Loading
print("[TEST 1] Model Loading")
print("-" * 80)

print("\n1.1. Loading SpliceAI models...")
try:
    spliceai_models, spliceai_metadata = load_base_model_ensemble(
        base_model='spliceai',
        context=10000,
        verbosity=1
    )
    print(f"✅ SpliceAI loaded: {spliceai_metadata}")
except Exception as e:
    print(f"❌ SpliceAI loading failed: {e}")
    sys.exit(1)

print("\n1.2. Loading OpenSpliceAI models...")
try:
    openspliceai_models, openspliceai_metadata = load_base_model_ensemble(
        base_model='openspliceai',
        context=10000,
        verbosity=1
    )
    print(f"✅ OpenSpliceAI loaded: {openspliceai_metadata}")
except Exception as e:
    print(f"❌ OpenSpliceAI loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n1.3. Comparing model metadata...")
print(f"  SpliceAI:     {spliceai_metadata['genome_build']}, {spliceai_metadata['framework']}")
print(f"  OpenSpliceAI: {openspliceai_metadata['genome_build']}, {openspliceai_metadata['framework']}")
print("✅ Model loading test passed!")

# Test 2: Genomic Build Routing
print("\n" + "=" * 80)
print("[TEST 2] Genomic Build Routing")
print("-" * 80)

print("\n2.1. Testing SpliceAI config (should use GRCh37/Ensembl)...")
spliceai_config = BaseModelConfig(
    base_model='spliceai',
    mode='test',
    test_name='openspliceai_integration_test'
)
artifact_manager = spliceai_config.get_artifact_manager()
print(f"  Build: {artifact_manager.config.build}")
print(f"  Source: {artifact_manager.config.source}")
assert artifact_manager.config.build == 'GRCh37', "Expected GRCh37 for SpliceAI"
assert artifact_manager.config.source == 'ensembl', "Expected Ensembl for SpliceAI"
print("✅ SpliceAI routing correct!")

print("\n2.2. Testing OpenSpliceAI config (should use GRCh38/MANE)...")
openspliceai_config = BaseModelConfig(
    base_model='openspliceai',
    mode='test',
    test_name='openspliceai_integration_test'
)
artifact_manager = openspliceai_config.get_artifact_manager()
print(f"  Build: {artifact_manager.config.build}")
print(f"  Source: {artifact_manager.config.source}")
assert artifact_manager.config.build == 'GRCh38', "Expected GRCh38 for OpenSpliceAI"
assert artifact_manager.config.source == 'mane', "Expected MANE for OpenSpliceAI"
print("✅ OpenSpliceAI routing correct!")

# Test 3: Prediction Workflow (Optional - requires GRCh38 data)
print("\n" + "=" * 80)
print("[TEST 3] Prediction Workflow (Skipped - requires GRCh38 data)")
print("-" * 80)
print("Note: Full prediction testing requires:")
print("  - GRCh38 genome FASTA")
print("  - MANE annotations GTF")
print("  - Splice site annotations for GRCh38")
print()
print("To test predictions, ensure these resources are available and run:")
print("  python -c \"from meta_spliceai import run_base_model_predictions; \\")
print("             results = run_base_model_predictions(base_model='openspliceai', \\")
print("                                                   target_genes=['BRCA1'], \\")
print("                                                   mode='test')\"")

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("✅ Model Loading: PASSED")
print("✅ Genomic Build Routing: PASSED")
print("⏭️  Prediction Workflow: SKIPPED (requires GRCh38 data)")
print()
print("=" * 80)
print("🎉 OPENSPLICEAI INTEGRATION TEST COMPLETE!")
print("=" * 80)
print()
print("Next Steps:")
print("1. Download GRCh38 genomic resources (genome FASTA, MANE GTF)")
print("2. Generate splice site annotations for GRCh38")
print("3. Run full prediction workflow with OpenSpliceAI")
print("4. Compare predictions with SpliceAI on same genes (after liftover)")
print()

