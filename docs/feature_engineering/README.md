# Feature Engineering Documentation

**Created:** 2025-10-28  
**Last Updated:** 2025-10-28

This directory contains all documentation related to feature engineering for the meta-model, including feature alignment, categorical encoding, k-mer generation, and troubleshooting guides.

## Core Concepts

### Feature Alignment
- **[FEATURE_ALIGNMENT_STRATEGY.md](FEATURE_ALIGNMENT_STRATEGY.md)** - Comprehensive strategy for aligning training and inference features
  - Date: 2025-10-24
  - Status: ✅ Implemented
  - Key insight: Handles k-mer exceptions and extra/missing features

### Categorical Encoding
- **[CATEGORICAL_FEATURE_ENCODING.md](CATEGORICAL_FEATURE_ENCODING.md)** - Categorical feature encoding strategy
  - Date: 2025-10-24
  - Status: ✅ Implemented
  
- **[CATEGORICAL_ENCODING_SUMMARY.md](CATEGORICAL_ENCODING_SUMMARY.md)** - Summary of categorical encoding implementation
  - Date: 2025-10-24
  - Status: ✅ Complete

### K-mer Features
- **[KMER_FEATURE_ALIGNMENT_EXPLAINED.md](KMER_FEATURE_ALIGNMENT_EXPLAINED.md)** - Why k-mers are special in feature alignment
  - Date: 2025-10-24
  - Status: ✅ Complete

## Troubleshooting & Verification

### Quick References
- **[FEATURE_MISMATCH_QUICK_REFERENCE.md](FEATURE_MISMATCH_QUICK_REFERENCE.md)** - Quick troubleshooting guide for feature mismatches
  - Date: 2025-10-24
  - Status: ✅ Current
  - Use this first when encountering feature mismatch errors

### Detailed Analysis
- **[FEATURE_MISMATCH_VERIFICATION.md](FEATURE_MISMATCH_VERIFICATION.md)** - Comprehensive feature mismatch analysis
  - Date: 2025-10-24
  - Status: ✅ Complete
  
- **[feature_consistency_analysis.md](feature_consistency_analysis.md)** - Feature consistency analysis across training/inference
  - Date: 2025-10-24
  - Status: ✅ Complete

- **[FEATURE_MANIFEST_COMPARISON.md](FEATURE_MANIFEST_COMPARISON.md)** - Feature manifest comparison tools
  - Date: 2025-10-24
  - Status: ✅ Complete

## Planning & Future Work

- **[FEATURE_WISHLIST.md](FEATURE_WISHLIST.md)** - Future feature improvements and enhancements
  - Date: 2025-10-24
  - Status: 📋 Planning

## Key Principles

1. **Feature Consistency**: Training and inference must use identical feature sets (121 features)
2. **K-mer Exception**: Missing k-mers in test data are filled with 0 (not an error)
3. **Extra Features**: Can be safely dropped if not in training set
4. **Missing Non-K-mer Features**: Critical error - indicates incomplete feature generation
5. **Categorical Encoding**: Must be applied consistently (e.g., `chrom` → numeric)

## Common Issues & Solutions

### Issue: Feature shape mismatch (expected: 121, got: 130)
**Solution**: Extra features need to be dropped. Check `_align_features_with_model()`

### Issue: Feature shape mismatch (expected: 121, got: 115)
**Solution**: Missing features. Verify:
1. K-mers generated? (`_generate_kmer_features`)
2. Genomic features enriched? (`GenomicFeatureEnricher`)
3. Categorical encoding applied? (`_apply_dynamic_chrom_encoding`)

### Issue: Model doesn't have `feature_names_in_`
**Solution**: Load from `train.features.json` (implemented in `_align_features_with_model`)

## Related Documentation

- [../development/](../development/) - General development documentation
- [../analysis/](../analysis/) - Analysis and evaluation documentation
- [../data/](../data/) - Data pipeline documentation

## Testing

Feature alignment is tested in:
- `scripts/testing/test_all_modes_comprehensive.py` - End-to-end testing
- `scripts/testing/diagnose_feature_mismatch.py` - Diagnostic tool
- `scripts/testing/verify_mode_differences.py` - Mode-specific verification

## Change Log

### 2025-10-28
- ✅ Fixed meta-only mode feature mismatch by loading from `train.features.json`
- ✅ Organized all feature engineering docs into dedicated directory
- ✅ Added this README with document dates and organization

### 2025-10-24
- ✅ Implemented feature alignment strategy
- ✅ Added categorical encoding support
- ✅ Created k-mer feature handling logic
- ✅ Documented troubleshooting guides

