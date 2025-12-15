#!/usr/bin/env python3
"""
Test Artifact Manager Integration

Tests the artifact management system with the workflow in test mode.
Validates:
- Directory structure creation
- Overwrite policy (test mode should always overwrite)
- Artifact path resolution
- Integration with workflow
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from meta_spliceai.system.artifact_manager import ArtifactManager, ArtifactConfig
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.system.genomic_resources import Registry


def test_artifact_config():
    """Test ArtifactConfig creation and validation."""
    print("\n" + "="*80)
    print("TEST 1: ArtifactConfig Creation")
    print("="*80 + "\n")
    
    # Test 1a: Default config (test mode)
    config = ArtifactConfig()
    print(f"✓ Default config created")
    print(f"  Mode: {config.mode}")
    print(f"  Coverage: {config.coverage}")
    print(f"  Test name: {config.test_name}")
    assert config.mode == "test", "Default mode should be 'test'"
    assert config.coverage == "gene_subset", "Default coverage should be 'gene_subset'"
    assert config.test_name is not None, "Test name should be auto-generated"
    
    # Test 1b: Production config
    config = ArtifactConfig(
        mode="production",
        coverage="full_genome",
        source="ensembl",
        build="GRCh37",
        base_model="spliceai"
    )
    print(f"\n✓ Production config created")
    print(f"  Mode: {config.mode}")
    print(f"  Coverage: {config.coverage}")
    assert config.mode == "production"
    assert config.coverage == "full_genome"
    
    # Test 1c: Auto-detection (full_genome -> production)
    config = ArtifactConfig(
        coverage="full_genome"
    )
    print(f"\n✓ Auto-detection test")
    print(f"  Coverage: {config.coverage}")
    print(f"  Mode (auto-detected): {config.mode}")
    assert config.mode == "production", "full_genome should auto-switch to production"
    
    print("\n✅ TEST 1 PASSED\n")


def test_artifact_manager():
    """Test ArtifactManager functionality."""
    print("\n" + "="*80)
    print("TEST 2: ArtifactManager Functionality")
    print("="*80 + "\n")
    
    # Test 2a: Test mode manager
    config = ArtifactConfig(
        mode="test",
        coverage="gene_subset",
        source="ensembl",
        build="GRCh37",
        base_model="spliceai",
        test_name="artifact_manager_test",
        data_root=project_root / "data"
    )
    manager = ArtifactManager(config)
    
    print("✓ Test mode manager created")
    print(f"  Base dir: {manager.base_dir}")
    print(f"  Artifacts dir: {manager.artifacts_dir}")
    
    # Check directory structure
    expected_base = project_root / "data" / "ensembl" / "GRCh37" / "spliceai_eval"
    expected_artifacts = expected_base / "tests" / "artifact_manager_test" / "meta_models" / "predictions"
    
    assert manager.base_dir == expected_base, f"Base dir mismatch: {manager.base_dir} != {expected_base}"
    assert manager.artifacts_dir == expected_artifacts, f"Artifacts dir mismatch: {manager.artifacts_dir} != {expected_artifacts}"
    print("  ✓ Directory structure correct")
    
    # Test 2b: Get artifact paths
    positions_path = manager.get_artifact_path('full_splice_positions_enhanced.tsv')
    errors_path = manager.get_artifact_path('full_splice_errors.tsv')
    
    print(f"\n✓ Artifact paths resolved")
    print(f"  Positions: {positions_path}")
    print(f"  Errors: {errors_path}")
    
    assert positions_path.parent == expected_artifacts
    assert errors_path.parent == expected_artifacts
    
    # Test 2c: Overwrite policy (test mode)
    print(f"\n✓ Testing overwrite policy (test mode)")
    
    # Create directory and test file
    artifacts_dir = manager.get_artifacts_dir(create=True)
    test_file = artifacts_dir / "test_artifact.tsv"
    test_file.write_text("test data")
    
    should_overwrite = manager.should_overwrite(test_file)
    print(f"  Should overwrite existing file: {should_overwrite}")
    assert should_overwrite, "Test mode should always overwrite"
    
    should_overwrite_new = manager.should_overwrite(artifacts_dir / "new_file.tsv")
    print(f"  Should write new file: {should_overwrite_new}")
    assert should_overwrite_new, "Should always write new files"
    
    # Test 2d: Production mode manager
    config_prod = ArtifactConfig(
        mode="production",
        coverage="full_genome",
        source="ensembl",
        build="GRCh37",
        base_model="spliceai",
        data_root=project_root / "data"
    )
    manager_prod = ArtifactManager(config_prod)
    
    print(f"\n✓ Production mode manager created")
    print(f"  Artifacts dir: {manager_prod.artifacts_dir}")
    
    expected_prod_artifacts = expected_base / "meta_models"
    assert manager_prod.artifacts_dir == expected_prod_artifacts
    
    # Test overwrite policy (production mode)
    print(f"\n✓ Testing overwrite policy (production mode)")
    should_overwrite_prod = manager_prod.should_overwrite(test_file)
    print(f"  Should overwrite existing file: {should_overwrite_prod}")
    assert not should_overwrite_prod, "Production mode should not overwrite by default"
    
    should_overwrite_force = manager_prod.should_overwrite(test_file, force=True)
    print(f"  Should overwrite with force=True: {should_overwrite_force}")
    assert should_overwrite_force, "Production mode should overwrite with force=True"
    
    print("\n✅ TEST 2 PASSED\n")


def test_spliceai_config_integration():
    """Test SpliceAIConfig integration with ArtifactManager."""
    print("\n" + "="*80)
    print("TEST 3: SpliceAIConfig Integration")
    print("="*80 + "\n")
    
    # Get registry for paths
    registry = Registry(build='GRCh37', release='87')
    gtf_file = registry.get_gtf_path(validate=False)
    fasta_file = registry.get_fasta_path(validate=False)
    
    # Test 3a: Test mode config
    config = SpliceAIConfig(
        gtf_file=str(gtf_file) if gtf_file else "data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf",
        genome_fasta=str(fasta_file) if fasta_file else "data/ensembl/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa",
        eval_dir=str(project_root / 'results' / 'artifact_test'),
        local_dir=str(registry.data_dir),
        mode='test',
        coverage='gene_subset',
        test_name='spliceai_config_test'
    )
    
    print("✓ SpliceAIConfig created (test mode)")
    print(f"  Mode: {config.mode}")
    print(f"  Coverage: {config.coverage}")
    print(f"  Test name: {config.test_name}")
    
    # Get artifact manager from config
    manager = config.get_artifact_manager()
    
    print(f"\n✓ ArtifactManager obtained from config")
    print(f"  Mode: {manager.config.mode}")
    print(f"  Build: {manager.config.build}")
    print(f"  Base model: {manager.config.base_model}")
    
    # Print summary
    print(f"\n✓ Artifact manager summary:")
    manager.print_summary()
    
    # Test 3b: Production mode config with auto-detection
    config_prod = SpliceAIConfig(
        gtf_file=str(gtf_file) if gtf_file else "data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf",
        genome_fasta=str(fasta_file) if fasta_file else "data/ensembl/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa",
        eval_dir=str(project_root / 'data' / 'ensembl' / 'GRCh37' / 'spliceai_eval'),
        local_dir=str(registry.data_dir),
        coverage='full_genome'  # Should auto-detect production mode
    )
    
    print(f"\n✓ SpliceAIConfig created (auto-detect production)")
    print(f"  Coverage: {config_prod.coverage}")
    print(f"  Mode (auto-detected): {config_prod.mode}")
    
    assert config_prod.mode == "production", "full_genome should auto-detect production mode"
    
    manager_prod = config_prod.get_artifact_manager()
    print(f"\n✓ Production artifact manager:")
    print(f"  Artifacts dir: {manager_prod.artifacts_dir}")
    
    print("\n✅ TEST 3 PASSED\n")


def test_directory_structure():
    """Test that directory structure matches specification."""
    print("\n" + "="*80)
    print("TEST 4: Directory Structure Validation")
    print("="*80 + "\n")
    
    config = ArtifactConfig(
        mode="test",
        source="ensembl",
        build="GRCh37",
        base_model="spliceai",
        test_name="structure_test",
        data_root=project_root / "data"
    )
    manager = ArtifactManager(config)
    
    # Expected structure:
    # data/ensembl/GRCh37/spliceai_eval/tests/structure_test/meta_models/predictions/
    
    print("✓ Testing directory structure")
    print(f"  Base dir: {manager.base_dir}")
    print(f"  Expected: data/ensembl/GRCh37/spliceai_eval")
    
    assert "ensembl" in str(manager.base_dir)
    assert "GRCh37" in str(manager.base_dir)
    assert "spliceai_eval" in str(manager.base_dir)
    
    print(f"\n  Artifacts dir: {manager.artifacts_dir}")
    print(f"  Expected: .../tests/structure_test/meta_models/predictions")
    
    assert "tests" in str(manager.artifacts_dir)
    assert "structure_test" in str(manager.artifacts_dir)
    assert "meta_models" in str(manager.artifacts_dir)
    assert "predictions" in str(manager.artifacts_dir)
    
    # Test training data directory
    training_dir = manager.get_training_data_dir()
    print(f"\n  Training data dir: {training_dir}")
    print(f"  Expected: .../spliceai_eval/training_data")
    
    assert "training_data" in str(training_dir)
    assert "spliceai_eval" in str(training_dir)
    
    # Test model checkpoint directory
    checkpoint_dir = manager.get_model_checkpoint_dir("v1.0")
    print(f"\n  Checkpoint dir: {checkpoint_dir}")
    print(f"  Expected: .../spliceai_eval/models/v1.0")
    
    assert "models" in str(checkpoint_dir)
    assert "v1.0" in str(checkpoint_dir)
    
    print("\n✅ TEST 4 PASSED\n")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ARTIFACT MANAGER TEST SUITE")
    print("="*80)
    
    try:
        test_artifact_config()
        test_artifact_manager()
        test_spliceai_config_integration()
        test_directory_structure()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80 + "\n")
        
        print("Summary:")
        print("  ✓ ArtifactConfig creation and validation")
        print("  ✓ ArtifactManager functionality")
        print("  ✓ Overwrite policy (test vs production)")
        print("  ✓ SpliceAIConfig integration")
        print("  ✓ Directory structure validation")
        print("  ✓ Auto-detection (full_genome -> production)")
        print()
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

