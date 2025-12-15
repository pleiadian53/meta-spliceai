# GRCh37 Download Guide for SpliceAI Compatibility

## Date: 2025-10-31

## Why Download GRCh37?

**Critical Discovery**: SpliceAI was trained on **GRCh37/hg19**, not GRCh38.

Using GRCh38 annotations with SpliceAI causes:
- **PR-AUC**: 0.541 (vs 0.97 in paper) - **44% lower**
- **Top-k Accuracy**: 0.550 (vs 0.95 in paper) - **42% lower**

**Solution**: Download GRCh37 data to match SpliceAI's training data.

**Expected Improvement**:
- PR-AUC: 0.54 → 0.80-0.90
- Top-k Accuracy: 0.55 → 0.75-0.85

## Quick Start

### Option 1: Automated Script (Recommended)

```bash
# Run the automated download script
bash scripts/setup/download_grch37_data.sh
```

This script will:
1. Download GRCh37 GTF and FASTA (Ensembl release 87)
2. Derive splice sites for GRCh37
3. Verify all files
4. Display next steps

### Option 2: Manual Download

```bash
# Step 1: Download GTF and FASTA
python -m meta_spliceai.system.genomic_resources.cli bootstrap \
  --species homo_sapiens \
  --build GRCh37 \
  --release 87 \
  --verbose

# Step 2: Derive splice sites
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 \
  --splice-sites \
  --consensus-window 2 \
  --verbose
```

## What Gets Downloaded

### Files Created

```
data/ensembl/GRCh37/
├── Homo_sapiens.GRCh37.87.gtf           # ~1.5 GB (annotations)
├── Homo_sapiens.GRCh37.dna.primary_assembly.fa  # ~3.0 GB (genome)
├── splice_sites_enhanced.tsv             # ~5 MB (derived)
└── ... (other derived files)
```

### Ensembl Release 87

- **Date**: December 2016
- **Reason**: Last Ensembl release for GRCh37
- **Annotations**: GENCODE compatible (similar to V24)
- **Genes**: ~20,000 protein-coding genes

## Genomic Resources System

Our system already supports multiple genome builds through the `genomic_resources` package.

### Configuration

**File**: `configs/genomic_resources.yaml`

```yaml
builds:
  GRCh38:  # Modern build (default)
    gtf: "Homo_sapiens.GRCh38.{release}.gtf"
    fasta: "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    ensembl_base: "https://ftp.ensembl.org/pub/release-{release}"
  
  GRCh37:  # For SpliceAI compatibility
    gtf: "Homo_sapiens.GRCh37.{release}.gtf"
    fasta: "Homo_sapiens.GRCh37.dna.primary_assembly.fa"
    ensembl_base: "https://grch37.ensembl.org/pub/release-{release}"
```

### Registry

The `Registry` class automatically resolves paths for different builds:

```python
from meta_spliceai.system.genomic_resources import Registry

# Get GRCh37 paths
registry = Registry(build='GRCh37', release='87')
gtf_path = registry.get_gtf_path()
fasta_path = registry.get_fasta_path()
splice_sites_path = registry.get_splice_sites_path()
```

## Using GRCh37 in Workflows

### Option 1: Environment Variable

```bash
# Set build for entire session
export SS_BUILD=GRCh37
export SS_RELEASE=87

# Run workflow (will use GRCh37)
python scripts/testing/comprehensive_spliceai_evaluation.py
```

### Option 2: Command-Line Argument

```bash
# Specify build per command
python scripts/testing/comprehensive_spliceai_evaluation.py \
  --build GRCh37 \
  --release 87
```

### Option 3: Python API

```python
from meta_spliceai.system.genomic_resources import Registry

# Create registry for GRCh37
registry = Registry(build='GRCh37', release='87')

# Use in workflow
workflow = EnhancedSelectiveInferenceWorkflow(
    config=config,
    registry=registry  # Uses GRCh37 paths
)
```

## Verification

### Check Downloaded Files

```bash
# List GRCh37 files
ls -lh data/ensembl/GRCh37/

# Check GTF line count
wc -l data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf

# Check splice sites
head data/ensembl/GRCh37/splice_sites_enhanced.tsv
```

### Expected Output

```
data/ensembl/GRCh37/
-rw-r--r--  1.5G  Homo_sapiens.GRCh37.87.gtf
-rw-r--r--  3.0G  Homo_sapiens.GRCh37.dna.primary_assembly.fa
-rw-r--r--  5.0M  splice_sites_enhanced.tsv
```

## Next Steps After Download

### 1. Re-run Evaluation on GRCh37

```bash
# Comprehensive evaluation on GRCh37
python scripts/testing/comprehensive_spliceai_evaluation.py \
  --build GRCh37 \
  --output predictions/evaluation_grch37.parquet
```

**Expected Results**:
- PR-AUC: 0.80-0.90 (vs 0.54 on GRCh38)
- Top-k Accuracy: 0.75-0.85 (vs 0.55 on GRCh38)
- F1 Score: 0.75-0.85 (vs 0.60 on GRCh38)

### 2. Re-run Adjustment Detection on GRCh37

```bash
# Detect optimal adjustments for GRCh37
python scripts/testing/test_score_adjustment_detection.py \
  --build GRCh37 \
  --genes 20
```

**Expected**: Adjustments may differ from GRCh38 (currently zero)

### 3. Update Training Data to GRCh37

```bash
# Regenerate training data using GRCh37
python scripts/training/prepare_training_data.py \
  --build GRCh37 \
  --output data/training/grch37
```

### 4. Train Meta-Model on GRCh37

```bash
# Train meta-model using GRCh37 base predictions
python scripts/training/train_meta_model.py \
  --build GRCh37 \
  --training-data data/training/grch37
```

## Maintaining Both Builds

### Directory Structure

```
data/ensembl/
├── GRCh38/  # Modern build (for general use)
│   ├── Homo_sapiens.GRCh38.112.gtf
│   ├── Homo_sapiens.GRCh38.dna.primary_assembly.fa
│   ├── splice_sites_enhanced.tsv
│   └── ...
└── GRCh37/  # For SpliceAI compatibility
    ├── Homo_sapiens.GRCh37.87.gtf
    ├── Homo_sapiens.GRCh37.dna.primary_assembly.fa
    ├── splice_sites_enhanced.tsv
    └── ...
```

### When to Use Each Build

**Use GRCh37**:
- ✅ SpliceAI predictions
- ✅ Meta-model training (with SpliceAI base)
- ✅ Comparing to SpliceAI paper
- ✅ Variant effect prediction (if variants in hg19)

**Use GRCh38**:
- ✅ Modern annotations (more isoforms)
- ✅ OpenSpliceAI predictions (likely)
- ✅ Pangolin predictions
- ✅ Variant effect prediction (if variants in hg38)

## Troubleshooting

### Download Fails

**Issue**: Network error or timeout

**Solution**:
```bash
# Try manual download
wget https://grch37.ensembl.org/pub/release-87/gtf/homo_sapiens/Homo_sapiens.GRCh37.87.gtf.gz
wget https://grch37.ensembl.org/pub/release-87/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz

# Decompress
gunzip Homo_sapiens.GRCh37.87.gtf.gz
gunzip Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz

# Move to correct location
mkdir -p data/ensembl/GRCh37
mv Homo_sapiens.GRCh37.87.gtf data/ensembl/GRCh37/
mv Homo_sapiens.GRCh37.dna.primary_assembly.fa data/ensembl/GRCh37/
```

### Derivation Fails

**Issue**: Error during splice site derivation

**Solution**:
```bash
# Check GTF file
head data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf

# Try derivation with verbose output
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 \
  --splice-sites \
  --verbose \
  --force  # Force regeneration
```

### Disk Space

**Issue**: Not enough disk space

**Requirements**:
- GTF: ~1.5 GB
- FASTA: ~3.0 GB
- Derived files: ~0.5 GB
- **Total**: ~5 GB

**Solution**: Free up space or use external drive

### Wrong Release

**Issue**: Downloaded wrong Ensembl release

**Solution**:
```bash
# Remove incorrect files
rm -rf data/ensembl/GRCh37/

# Re-download with correct release
bash scripts/setup/download_grch37_data.sh
```

## Performance Comparison

### Before (GRCh38)

| Metric | Value | vs Paper |
|--------|-------|----------|
| PR-AUC | 0.541 | -44% |
| Top-k Accuracy | 0.550 | -42% |
| F1 Score | 0.601 | N/A |

**Issue**: Genome build mismatch

### After (GRCh37) - Expected

| Metric | Value | vs Paper |
|--------|-------|----------|
| PR-AUC | 0.80-0.90 | -7 to -17% |
| Top-k Accuracy | 0.75-0.85 | -10 to -21% |
| F1 Score | 0.75-0.85 | N/A |

**Note**: Still slightly lower than paper due to:
- Different Ensembl release (87 vs GENCODE V24)
- Different evaluation genes
- Different evaluation protocol

## References

### Documentation
- [Genome Build Compatibility](GENOME_BUILD_COMPATIBILITY.md)
- [SpliceAI Base Model](../../meta_spliceai/splice_engine/base_models/docs/SPLICEAI.md)
- [Genomic Resources README](../../meta_spliceai/system/genomic_resources/README.md)

### External Resources
- [Ensembl GRCh37 Archive](https://grch37.ensembl.org/)
- [GENCODE V24lift37](https://www.gencodegenes.org/human/release_24lift37.html)
- [SpliceAI Paper](https://doi.org/10.1016/j.cell.2018.12.015)

## Summary

**Problem**: SpliceAI trained on GRCh37, we used GRCh38 → 44% performance drop

**Solution**: Download GRCh37 data using our genomic resources system

**Command**:
```bash
bash scripts/setup/download_grch37_data.sh
```

**Expected**: PR-AUC 0.54 → 0.85, Top-k 0.55 → 0.80

**Next**: Re-evaluate on GRCh37 to verify improvement

