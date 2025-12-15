# Replicating Production Dataset Structure

## Overview

This guide explains how to replicate the structure and characteristics of the previous 5000-gene production dataset (`train_pc_5000_3mers_diverse`) using the updated `incremental_builder.py`.

## Reference Dataset

**Original dataset**: `meta_spliceai/splice_engine/case_studies/data_sources/datasets/train_pc_5000_3mers_diverse`

Key characteristics based on the reference README:
- **Gene count**: 5000 protein-coding genes
- **Gene selection**: Random diverse selection
- **K-mer features**: 3-mers
- **Gene types**: Primarily protein_coding, with some lncRNA for diversity
- **Features**: ~50-100 features including probability, context, and k-mer features
- **Structure**: Master directory with partitioned Parquet files
- **Manifest**: Gene manifest with metadata

## Incremental Testing Strategy

Test with progressively larger datasets to validate scalability:

### Phase 1: Quick Validation (100 genes, ~5-10 min)
```bash
./scripts/testing/test_incremental_sizes.sh 100
```

**Purpose**: Verify basic functionality and output structure
**Expected output**: `data/train_pc_100_3mers_diverse/`

### Phase 2: Medium Scale (500 genes, ~20-30 min)
```bash
./scripts/testing/test_incremental_sizes.sh 500
```

**Purpose**: Test scalability and memory management
**Expected output**: `data/train_pc_500_3mers_diverse/`

### Phase 3: Production-Like (1000 genes, ~40-60 min)
```bash
./scripts/testing/test_incremental_sizes.sh 1000
```

**Purpose**: Validate production-ready dataset generation
**Expected output**: `data/train_pc_1000_3mers_diverse/`

### Phase 4: Full Production (5000 genes, ~3-4 hours)
```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 5000 \
    --subset-policy random \
    --gene-types protein_coding lncRNA \
    --batch-size 100 \
    --batch-rows 500000 \
    --run-workflow \
    --kmer-sizes 3 \
    --output-dir data/train_pc_5000_3mers_diverse \
    --overwrite \
    --verbose
```

**Purpose**: Generate full production dataset matching original
**Expected output**: `data/train_pc_5000_3mers_diverse/`

## Expected Dataset Structure

### Directory Layout
```
data/train_pc_<N>_3mers_diverse/
├── master/                      # Master dataset directory
│   ├── batch_00001.parquet      # Partitioned training data
│   ├── batch_00002.parquet
│   └── ...
├── gene_manifest.csv            # Gene metadata and statistics
├── batch_00001_raw.parquet      # Raw batch data (optional, can be cleaned)
├── batch_00001_trim.parquet     # Trimmed batch data (before master assembly)
└── ...
```

### Master Directory Contents

The `master/` directory contains the final training dataset:
- **Format**: Parquet files partitioned by batch
- **Compression**: zstd compression for efficiency
- **Rows**: 50K-200K per 100 genes (varies by gene size and splice site density)
- **Columns**: ~50-100 features

### Key Dataset Features

#### 1. Position Information
- `position` - Genomic position (1-based)
- `chrom` - Chromosome (encoded as integer)

#### 2. Base Model Predictions
- `donor_score` - SpliceAI donor probability
- `acceptor_score` - SpliceAI acceptor probability
- `neither_score` - SpliceAI neither probability

#### 3. Derived Probability Features
- `relative_donor_probability` - Donor relative to acceptor
- `splice_probability` - Combined splice probability
- `donor_acceptor_diff` - Difference between donor and acceptor
- `splice_neither_diff` - Difference between splice and neither
- `probability_entropy` - Uncertainty measure

#### 4. Context Features
- `context_score_m2`, `context_score_m1` - Upstream context
- `context_score_p1`, `context_score_p2` - Downstream context
- `context_neighbor_mean` - Neighborhood average
- `context_asymmetry` - Upstream vs downstream asymmetry
- `context_max` - Maximum context score

#### 5. Donor-Specific Features
- `donor_diff_m1` - Donor change from previous position
- `donor_surge_ratio` - Ratio to neighborhood average
- `donor_is_local_peak` - Binary peak indicator
- `donor_peak_height_ratio` - Height relative to neighbors
- `donor_signal_strength` - Overall signal quality
- `donor_second_derivative` - Curvature measure

#### 6. Acceptor-Specific Features
- Similar to donor-specific features
- `acceptor_diff_m1`, `acceptor_surge_ratio`, etc.

#### 7. Cross-Type Features
- `donor_acceptor_peak_ratio` - Ratio between donor and acceptor peaks
- `type_signal_difference` - Difference in signal quality
- `score_difference_ratio` - Normalized score difference

#### 8. K-mer Features
- `3mer_AAA` through `3mer_TTT` - All 64 3-mer combinations
- One-hot encoded k-mer representation

#### 9. Gene Metadata (if available)
- `gene_id` - Ensembl gene ID
- `gene_name` - Gene symbol
- `gene_type` - Biotype (protein_coding, lncRNA, etc.)

#### 10. Labels
- `splice_type` - Target label (0=neither, 1=donor, 2=acceptor)

## Gene Manifest Structure

The `gene_manifest.csv` contains metadata for all genes in the dataset:

```csv
global_index,gene_id,gene_name,gene_type,chrom,strand,gene_length,start,end,total_splice_sites,donor_sites,acceptor_sites,splice_density_per_kb,file_index,file_name
0,ENSG00000142611,PRDX5,protein_coding,11,+,12345,123000,135345,45,23,22,3.65,1,batch_00001.parquet
1,ENSG00000007314,UNC13A,protein_coding,19,-,345678,890000,1235678,156,78,78,0.45,1,batch_00001.parquet
...
```

Key columns:
- **gene_id**: Ensembl ID (primary identifier)
- **gene_name**: Human-readable gene symbol
- **gene_type**: Biotype classification
- **gene_length**: Gene span in base pairs
- **splice_density_per_kb**: Splice sites per kilobase
- **file_index**: Which batch file contains this gene

## Verification Checklist

After generating a dataset, verify it matches expected structure:

### ✅ Directory Structure
- [ ] `master/` directory exists
- [ ] Multiple Parquet files in `master/`
- [ ] `gene_manifest.csv` generated

### ✅ Dataset Content
- [ ] Total rows match expectations (~50K-200K per 100 genes)
- [ ] All key columns present (position, splice_type, scores)
- [ ] 64 k-mer features (3-mers)
- [ ] ~40-50 probability/context features
- [ ] Label distribution: ~30-40% neither, ~30-35% donor, ~30-35% acceptor

### ✅ Gene Manifest
- [ ] Number of genes matches request
- [ ] Gene types match request (protein_coding, lncRNA)
- [ ] Splice site counts reasonable (>0 for all genes)
- [ ] All genes have metadata

### ✅ Feature Quality
- [ ] No NaN values in critical columns
- [ ] Probability features in [0, 1] range
- [ ] K-mer features are binary (0 or 1)
- [ ] Context features have reasonable values

## Common Differences from Original Dataset

The new `incremental_builder.py` may produce slight differences:

### Expected Differences (Normal)
1. **Gene selection**: Different random seed → different genes
2. **Batch boundaries**: Different batch sizes → different file organization
3. **Feature order**: Column order may differ slightly
4. **File sizes**: Different compression → slightly different sizes

### Structural Similarities (Should Match)
1. **Directory structure**: master/ with Parquet files
2. **Feature set**: Same ~50-100 features
3. **Label distribution**: Similar class balance
4. **Data types**: Same schema and types
5. **Gene manifest format**: Same CSV structure

## Comparing Datasets

To compare a new dataset with the reference structure:

```bash
# Compare directory structures
ls -lh data/train_pc_100_3mers_diverse/master/
ls -lh meta_spliceai/splice_engine/case_studies/data_sources/datasets/train_pc_5000_3mers_diverse/master/

# Compare schemas
python << EOF
import polars as pl
from pathlib import Path

# Load sample from new dataset
new_df = pl.read_parquet("data/train_pc_100_3mers_diverse/master/batch_00001.parquet", n_rows=100)
print("New dataset schema:")
print(f"  Columns: {len(new_df.columns)}")
print(f"  Sample: {', '.join(new_df.columns[:20])}...")

# If reference dataset still exists:
# ref_df = pl.read_parquet("meta_spliceai/.../train_pc_5000_3mers_diverse/master/batch_00001.parquet", n_rows=100)
# print("\nReference dataset schema:")
# print(f"  Columns: {len(ref_df.columns)}")
# print(f"  Sample: {', '.join(ref_df.columns[:20])}...")
EOF
```

## Performance Benchmarks

Expected dataset generation times:

| Genes | Time      | Output Size | Rows       |
|-------|-----------|-------------|------------|
| 100   | 5-10 min  | ~50-100 MB  | ~50K-200K  |
| 500   | 20-30 min | ~250-500 MB | ~250K-1M   |
| 1000  | 40-60 min | ~500MB-1GB  | ~500K-2M   |
| 5000  | 3-4 hours | ~2-5 GB     | ~2.5M-10M  |

*Times assume `--run-workflow` is enabled (includes base model pass)*

## Troubleshooting

### Issue: Different number of features
**Cause**: Feature engineering pipeline may have evolved
**Solution**: This is expected if new features were added. Verify core features exist.

### Issue: Different label distribution
**Cause**: Different gene selection or down-sampling
**Solution**: Check down-sampling parameters match reference.

### Issue: Missing gene manifest
**Cause**: Dataset was down-sampled (removed gene_id column)
**Solution**: This is normal for training datasets after down-sampling.

### Issue: Very different row counts
**Cause**: Different down-sampling ratio or gene selection
**Solution**: Verify `--hard-prob`, `--window`, `--easy-ratio` parameters.

## Production Workflow

For generating a full 5000-gene dataset matching the reference:

```bash
# 1. Generate dataset (3-4 hours)
./scripts/builder/run_builder_resumable.sh tmux

# Or with explicit parameters:
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 5000 \
    --subset-policy random \
    --gene-types protein_coding lncRNA \
    --batch-size 100 \
    --batch-rows 500000 \
    --run-workflow \
    --kmer-sizes 3 \
    --output-dir data/train_pc_5000_3mers_diverse \
    --overwrite \
    --verbose

# 2. Verify dataset structure
ls -lh data/train_pc_5000_3mers_diverse/master/
cat data/train_pc_5000_3mers_diverse/gene_manifest.csv | head -10

# 3. Train meta-model (2-3 hours)
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset data/train_pc_5000_3mers_diverse/master \
    --out-dir data/model_5000genes \
    --n-folds 5 \
    --n-estimators 800 \
    --verbose
```

## Related Documentation

- [Incremental Size Testing Script](../../scripts/testing/test_incremental_sizes.sh) - Automated testing
- [Builder Usage Guide](USAGE_GUIDE.md) - Complete builder options
- [Pipeline Test Guide](PIPELINE_TEST_GUIDE.md) - End-to-end testing
- [Original Dataset README](../../meta_spliceai/splice_engine/case_studies/data_sources/datasets/train_pc_5000_3mers_diverse/README.md) - Reference structure

---

**Last Updated**: October 17, 2025

