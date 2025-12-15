# Feature Consistency Analysis: Training vs Inference

## Overview
This document analyzes the consistency between the position-centric data representation and featurization logic in training data assembly (`incremental_builder.py`) and test data assembly (inference workflow).

## Training Data Assembly Pipeline (`incremental_builder.py`)

### Step 1: Base Feature Extraction (`dataset_builder.py`)
1. **Load analysis sequences TSV files** containing:
   - Position-level predictions from base model
   - Probability scores: `donor_score`, `acceptor_score`, `neither_score`
   - Advanced probability features: `relative_donor_probability`, `splice_probability`, `donor_acceptor_logodds`, `splice_neither_logodds`, `probability_entropy`
   - Context features from enhanced workflow
   - Sequence data

2. **K-mer Feature Extraction** (`make_kmer_features`):
   - Extract k-mers of specified sizes (default: 6-mers)
   - Add GC content, sequence length, sequence complexity
   - Ensure all k-mer columns present (fill missing with 0)
   - Standardize dtypes (float64 for k-mers, int64 for booleans)

### Step 2: Feature Enrichment (`apply_feature_enrichers`)
1. **Gene-level features**:
   - Gene length, transcript count
   - Number of splice sites
   - Gene type (protein_coding, lncRNA, etc.)

2. **Performance features**:
   - Error rates, prediction accuracy metrics

3. **Structural features**:
   - Distance to nearest splice site
   - Position within gene/transcript

4. **Patching**:
   - Fill missing gene_type values
   - Fill missing structural features
   - Update n_splice_sites from authoritative source

### Step 3: Column Exclusion (`preprocessing.py`)
Remove columns that shouldn't be features:
- **Metadata**: `gene_id`, `position`, `chrom`, `strand`, `transcript_id`, `gene_type`
- **Leakage**: `splice_type`, `pred_type`, `true_position`, `predicted_position`
- **Sequence**: Raw `sequence` string
- **Redundant**: Duplicate or derived columns

## Inference Data Assembly Pipeline

### StandardizedFeaturizer Approach
1. **Base Probability Scores**:
   - Merge base model predictions if not already present
   - Ensure `donor_score`, `acceptor_score`, `neither_score` available

2. **K-mer Features** (same `make_kmer_features`):
   - Extract k-mers with same sizes as training
   - Add same additional features (GC content, etc.)
   - Harmonize with training schema (fill missing k-mers with 0)

3. **Feature Enrichment** (same `apply_feature_enrichers`):
   - Apply same enrichers as training
   - Get same gene-level, performance, structural features

4. **Feature Harmonization**:
   - Match training schema exactly
   - Fill missing features with 0
   - Remove extra features not in training
   - Encode categorical features (e.g., `chrom` to numeric)

### Column Exclusion in Inference
Using same exclusion lists from `preprocessing.py`:
- `METADATA_COLUMNS`
- `LEAKAGE_COLUMNS`
- `SEQUENCE_COLUMNS`
- Additional inference-specific: base scores when used as inputs

## Key Consistency Points ✅

### ✅ CONSISTENT: Core Feature Generation
- Both use `make_kmer_features` for k-mer extraction
- Both use `apply_feature_enrichers` for enrichment
- Both start from same analysis_sequences artifacts

### ✅ CONSISTENT: Feature Types
1. **Probability features**: Same advanced features from enhanced workflow
2. **K-mer features**: Same extraction logic and sizes
3. **Genomic features**: Same enrichment pipeline
4. **Context features**: Same neighboring position analysis

### ✅ CONSISTENT: Data Representation
- **Position-centric**: Each row = one nucleotide position
- **Gene context**: Position identified by (gene_id, position)
- **Prediction context**: Contains base model scores and derived features

### ✅ CONSISTENT: Column Handling
- Same metadata exclusion logic
- Same leakage prevention
- Same dtype standardization

## Potential Issues Found 🔍

### Issue 1: Feature Processor Column Exclusion (FIXED)
**Problem**: `feature_processor.py` was only excluding `['gene_id', 'position']`
**Solution**: Now uses full standardized exclusion list from `preprocessing.py`

### Issue 2: Schema Harmonization
**Observation**: Training directly builds features, inference must harmonize
**Status**: Properly handled by `StandardizedFeaturizer` with training schema

### Issue 3: Chrom Encoding
**Observation**: `chrom` can be string ('X', 'Y') or numeric
**Status**: Handled by encoding to numeric in StandardizedFeaturizer

## Verification Results

### Training Pipeline Flow:
```
TSV Files → build_training_dataset() → k-mer extraction → enrichment → patching → exclusion → final features
```

### Inference Pipeline Flow:
```
Analysis DF → StandardizedFeaturizer → k-mer extraction → enrichment → harmonization → exclusion → final features
```

### Key Differences:
1. **Harmonization Step**: Inference adds explicit harmonization with training schema
2. **Chunking**: Inference supports chunked processing for memory efficiency
3. **Selective Processing**: Inference can process subset of positions

## Recommendations

1. **✅ Current Approach is Consistent**: The core featurization logic is shared between training and inference

2. **✅ Proper Abstraction**: `StandardizedFeaturizer` provides good abstraction for inference

3. **✅ Schema Enforcement**: Training schema properly enforced during inference

4. **Minor Enhancement Suggested**:
   - Consider caching enricher results for repeated inference on same genes
   - Add validation to ensure k-mer sizes match between training and inference

## Conclusion

The position-centric data representation and featurization logic is **CONSISTENT** between training and inference pipelines. Both:
1. Start from same analysis_sequences artifacts
2. Use same feature extraction functions
3. Apply same enrichment pipeline
4. Exclude same metadata/leakage columns
5. Produce same feature matrix structure

The inference pipeline correctly adds harmonization to ensure exact feature match with training, which is the appropriate approach for test data assembly.
