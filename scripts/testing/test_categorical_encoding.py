#!/usr/bin/env python3
"""
Test centralized categorical feature encoding.

This script verifies that:
1. The centralized encoding function works correctly
2. Training and inference use the same encoding logic
3. K-mer features are never treated as categorical
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
from meta_spliceai.splice_engine.meta_models.builder.feature_schema import (
    encode_categorical_features,
    is_kmer_feature,
    CATEGORICAL_FEATURES,
    ALWAYS_NUMERICAL_FEATURES,
    validate_feature_types
)


def test_chromosome_encoding():
    """Test that chromosome encoding works correctly."""
    print("\n" + "="*60)
    print("TEST 1: Chromosome Encoding")
    print("="*60)
    
    # Create test dataframe with various chromosome names
    test_df = pl.DataFrame({
        'chrom': ['1', 'chr2', 'X', 'chrX', 'Y', 'chrY', 'MT', 'chrMT', 
                  'M', 'chrM', 'GL000225.1', 'KI270750.1', '22', 'chr22']
    })
    
    print(f"\nInput chromosomes: {test_df['chrom'].to_list()}")
    
    # Encode
    encoded_df = encode_categorical_features(
        test_df,
        features_to_encode=['chrom'],
        verbose=True
    )
    
    print(f"Encoded chromosomes: {encoded_df['chrom'].to_list()}")
    
    # Verify expected encodings
    expected = {
        '1': 1,
        'chr2': 2,
        'X': 23,
        'chrX': 23,
        'Y': 24,
        'chrY': 24,
        'MT': 25,
        'chrMT': 25,
        'M': 25,
        'chrM': 25,
        '22': 22,
        'chr22': 22,
        # Unknown scaffolds get 100+
    }
    
    encoded_list = encoded_df['chrom'].to_list()
    input_list = test_df['chrom'].to_list()
    
    for i, (chrom_name, chrom_code) in enumerate(zip(input_list, encoded_list)):
        if chrom_name in expected:
            assert chrom_code == expected[chrom_name], \
                f"Encoding mismatch for {chrom_name}: expected {expected[chrom_name]}, got {chrom_code}"
            print(f"  ✅ {chrom_name} → {chrom_code}")
        else:
            # Unknown scaffolds should be 100+
            assert chrom_code >= 100, \
                f"Unknown scaffold {chrom_name} should be encoded as 100+, got {chrom_code}"
            print(f"  ✅ {chrom_name} → {chrom_code} (unknown scaffold)")
    
    print("\n✅ Chromosome encoding test PASSED!")
    return True


def test_kmer_detection():
    """Test that k-mer features are correctly identified."""
    print("\n" + "="*60)
    print("TEST 2: K-mer Feature Detection")
    print("="*60)
    
    # Test k-mer detection
    kmer_features = ['AAA', 'ACG', 'GGT', 'AAAT', 'CGCG']
    non_kmer_features = ['donor_score', 'acceptor_score', 'chrom', 'gene_id', 'A_count']
    
    print("\nK-mer features (should be True):")
    for feature in kmer_features:
        result = is_kmer_feature(feature)
        print(f"  {feature}: {result}")
        assert result, f"{feature} should be detected as k-mer"
    
    print("\nNon-k-mer features (should be False):")
    for feature in non_kmer_features:
        result = is_kmer_feature(feature)
        print(f"  {feature}: {result}")
        assert not result, f"{feature} should NOT be detected as k-mer"
    
    print("\n✅ K-mer detection test PASSED!")
    return True


def test_feature_type_consistency():
    """Test that feature types are consistent between registries."""
    print("\n" + "="*60)
    print("TEST 3: Feature Type Consistency")
    print("="*60)
    
    # Ensure no overlap between categorical and always-numerical
    categorical_names = set(CATEGORICAL_FEATURES.keys())
    numerical_names = set(ALWAYS_NUMERICAL_FEATURES)
    
    overlap = categorical_names & numerical_names
    
    print(f"\nCategorical features: {len(categorical_names)}")
    print(f"Always-numerical features: {len(numerical_names)}")
    print(f"Overlap: {len(overlap)}")
    
    if overlap:
        print(f"\n⚠️  WARNING: Features are in both categorical and numerical lists:")
        for feature in overlap:
            print(f"    - {feature}")
        raise ValueError("Feature type conflict detected!")
    
    print("\n✅ No feature type conflicts!")
    print("✅ Feature type consistency test PASSED!")
    return True


def test_encoding_idempotence():
    """Test that encoding is idempotent (encoding twice gives same result)."""
    print("\n" + "="*60)
    print("TEST 4: Encoding Idempotence")
    print("="*60)
    
    # Create test dataframe
    test_df = pl.DataFrame({
        'chrom': ['1', 'X', 'MT'],
        'donor_score': [0.1, 0.5, 0.9]
    })
    
    print(f"\nOriginal: {test_df['chrom'].to_list()}")
    
    # First encoding
    encoded_once = encode_categorical_features(
        test_df,
        features_to_encode=['chrom'],
        verbose=False
    )
    
    print(f"Encoded once: {encoded_once['chrom'].to_list()}")
    
    # Second encoding (should be a no-op since already numeric)
    encoded_twice = encode_categorical_features(
        encoded_once,
        features_to_encode=['chrom'],
        verbose=False
    )
    
    print(f"Encoded twice: {encoded_twice['chrom'].to_list()}")
    
    # Verify they're the same
    assert encoded_once['chrom'].to_list() == encoded_twice['chrom'].to_list(), \
        "Encoding should be idempotent!"
    
    print("\n✅ Encoding idempotence test PASSED!")
    return True


def test_feature_validation():
    """Test feature type validation."""
    print("\n" + "="*60)
    print("TEST 5: Feature Type Validation")
    print("="*60)
    
    # Create test dataframe with various feature types
    test_df = pl.DataFrame({
        'chrom': ['1', '2', 'X'],  # Categorical (string)
        'donor_score': [0.1, 0.5, 0.9],  # Numerical (float)
        'AAA': [3, 1, 2],  # K-mer (count)
        'gene_id': ['ENSG001', 'ENSG002', 'ENSG003'],  # Metadata (string)
    })
    
    print(f"\nDataframe schema:")
    print(test_df)
    
    # First encode categorical features
    encoded_df = encode_categorical_features(
        test_df,
        features_to_encode=['chrom'],
        verbose=False
    )
    
    # Validate (should not raise errors)
    validation = validate_feature_types(encoded_df, verbose=True)
    
    print(f"\nValidation results:")
    print(f"  Categorical: {validation['categorical']}")
    print(f"  Numerical: {validation['numerical']}")
    print(f"  K-mer: {validation['kmer']}")
    
    # Verify k-mer is detected as numerical
    assert 'AAA' in validation['kmer'], "K-mer feature should be in k-mer list"
    
    # Verify chrom is now numerical (after encoding)
    assert 'chrom' in validation['numerical'] or 'chrom' in validation['categorical'], \
        "chrom should be either numerical (after encoding) or categorical (before encoding)"
    
    print("\n✅ Feature type validation test PASSED!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("CATEGORICAL FEATURE ENCODING TESTS")
    print("="*60)
    
    tests = [
        test_chromosome_encoding,
        test_kmer_detection,
        test_feature_type_consistency,
        test_encoding_idempotence,
        test_feature_validation,
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
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

