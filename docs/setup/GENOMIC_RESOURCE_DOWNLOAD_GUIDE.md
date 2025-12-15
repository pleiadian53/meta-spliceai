# Genomic Resource Download Guide

**Date**: 2025-11-06  
**Purpose**: Complete guide for downloading genomic resources for all supported base models

---

## Overview

MetaSpliceAI requires different genomic resources depending on the base model:

| Base Model | Build | Annotation Source | Download Script |
|------------|-------|-------------------|-----------------|
| **SpliceAI** | GRCh37 | Ensembl (release 87) | `download_grch37_data.sh` ✅ |
| **OpenSpliceAI** | GRCh38 | MANE (v1.3) | `download_grch38_mane_data.sh` ✅ |

---

## Quick Start

### For SpliceAI (GRCh37/Ensembl)

```bash
# Download GRCh37 genome and annotations
./scripts/setup/download_grch37_data.sh
```

**What it downloads**:
- GRCh37 GTF (~1.5 GB)
- GRCh37 FASTA (~3.0 GB)
- Derived splice sites (~5 MB)

**Time**: 15-30 minutes

### For OpenSpliceAI (GRCh38/MANE)

**Step 1: Download Models**
```bash
# Download OpenSpliceAI PyTorch models
./scripts/base_model/download_openspliceai_models.sh
```

**Step 2: Download Genomic Data**
```bash
# Download GRCh38 genome and MANE annotations
./scripts/setup/download_grch38_mane_data.sh
```

**What it downloads**:
- MANE GFF (~200 MB)
- GRCh38 FASTA (~3.0 GB)
- Derived splice sites (~5 MB)

**Time**: 20-40 minutes

---

## Detailed Instructions

### 1. SpliceAI Resources (GRCh37)

#### Why GRCh37?

SpliceAI was trained on **GRCh37/hg19** (GENCODE V24lift37). Using GRCh38 causes:
- **44% performance drop** (PR-AUC: 0.97 → 0.54)
- Coordinate mismatches
- Incorrect predictions

#### Download Script

**Location**: `scripts/setup/download_grch37_data.sh`

**What it does**:
1. Downloads Ensembl release 87 GTF and FASTA
2. Indexes FASTA file
3. Derives splice sites with consensus windows
4. Verifies all files

**Usage**:
```bash
cd /Users/pleiadian53/work/meta-spliceai
./scripts/setup/download_grch37_data.sh
```

**Interactive prompts**:
- Asks before re-downloading existing files
- Asks before regenerating derived data

**Output**:
```
data/ensembl/GRCh37/
├── Homo_sapiens.GRCh37.87.gtf           # ~1.5 GB
├── Homo_sapiens.GRCh37.dna.primary_assembly.fa  # ~3.0 GB
├── Homo_sapiens.GRCh37.dna.primary_assembly.fa.fai  # Index
└── splice_sites_enhanced.tsv             # ~5 MB
```

#### Manual Download (Alternative)

```bash
# Activate environment
source ~/.bash_profile && mamba activate surveyor

# Download GTF and FASTA
python -m meta_spliceai.system.genomic_resources.cli bootstrap \
  --build GRCh37 \
  --release 87 \
  --verbose

# Derive splice sites
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 \
  --release 87 \
  --splice-sites \
  --consensus-window 2 \
  --verbose
```

---

### 2. OpenSpliceAI Resources (GRCh38/MANE)

#### Why GRCh38 and MANE?

OpenSpliceAI was trained on:
- **GRCh38**: Modern genome build
- **MANE v1.3**: Matched Annotation from NCBI and EMBL-EBI
  - High-quality, clinically relevant transcripts
  - One representative transcript per gene
  - Better for variant interpretation

#### Step 1: Download Models

**Location**: `scripts/base_model/download_openspliceai_models.sh`

**What it does**:
1. Downloads 5 PyTorch models (10,000nt context)
2. Creates metadata file
3. Verifies downloads

**Usage**:
```bash
cd /Users/pleiadian53/work/meta-spliceai
./scripts/base_model/download_openspliceai_models.sh
```

**Output**:
```
data/models/openspliceai/
├── model_10000nt_rs10.pt  # 2.7 MB
├── model_10000nt_rs11.pt  # 2.7 MB
├── model_10000nt_rs12.pt  # 2.7 MB
├── model_10000nt_rs13.pt  # 2.7 MB
├── model_10000nt_rs14.pt  # 2.7 MB
└── metadata.json
```

**Time**: 2-5 minutes

#### Step 2: Download Genomic Data

**Location**: `scripts/setup/download_grch38_mane_data.sh` (NEW)

**What it does**:
1. Downloads MANE v1.3 GFF from NCBI
2. Downloads GRCh38 FASTA from NCBI
3. Indexes FASTA file
4. Converts GFF to GTF (optional, for compatibility)
5. Derives splice sites

**Usage**:
```bash
cd /Users/pleiadian53/work/meta-spliceai
./scripts/setup/download_grch38_mane_data.sh
```

**Interactive prompts**:
- Asks before re-downloading existing files
- Asks before regenerating derived data

**Output**:
```
data/mane/GRCh38/
├── MANE.GRCh38.v1.3.refseq_genomic.gff  # ~200 MB
├── MANE.GRCh38.v1.3.refseq_genomic.gtf  # ~180 MB (optional)
├── GCF_000001405.40_GRCh38.p14_genomic.fna  # ~3.0 GB
├── GCF_000001405.40_GRCh38.p14_genomic.fna.fai  # Index
└── splice_sites_enhanced.tsv             # ~5 MB
```

**Time**: 20-40 minutes

#### Manual Download (Alternative)

```bash
# Activate environment
source ~/.bash_profile && mamba activate surveyor

# Download MANE GFF
curl -L "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_1.3/MANE.GRCh38.v1.3.refseq_genomic.gff.gz" \
  -o data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gff.gz
gunzip data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gff.gz

# Download GRCh38 FASTA
curl -L "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GCF_000001405.40_GRCh38.p14_genomic.fna.gz" \
  -o data/mane/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna.gz
gunzip data/mane/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna.gz

# Index FASTA
samtools faidx data/mane/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna

# Convert GFF to GTF (optional)
gffread data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gff \
  -T -o data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf

# Derive splice sites
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh38_MANE \
  --release 1.3 \
  --splice-sites \
  --consensus-window 2 \
  --verbose
```

---

## System Architecture

### Unified Genomic Resources Framework

**Configuration**: `configs/genomic_resources.yaml`

```yaml
builds:
  GRCh37:
    annotation_source: ensembl
    gtf: "Homo_sapiens.GRCh37.{release}.gtf"
    fasta: "Homo_sapiens.GRCh37.dna.primary_assembly.fa"
    ensembl_base: "https://ftp.ensembl.org/pub/grch37/release-{release}"
    
  GRCh38_MANE:
    annotation_source: mane
    gtf: "MANE.GRCh38.v{release}.refseq_genomic.gff"
    fasta: "GCF_000001405.40_GRCh38.p14_genomic.fna"
    base_url: "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_{release}"
```

### CLI Tool

**Module**: `meta_spliceai.system.genomic_resources.cli`

**Commands**:
1. `bootstrap`: Download GTF and FASTA
2. `derive`: Generate derived data (splice sites, gene features, etc.)

**Usage**:
```bash
# Bootstrap (download)
python -m meta_spliceai.system.genomic_resources.cli bootstrap \
  --build GRCh37 \
  --release 87 \
  --verbose

# Derive (generate)
python -m meta_spliceai.system.genomic_resources.cli derive \
  --build GRCh37 \
  --splice-sites \
  --gene-features \
  --verbose
```

### Registry System

**Module**: `meta_spliceai.system.genomic_resources.Registry`

**Purpose**: Automatic path resolution for different builds

**Usage**:
```python
from meta_spliceai.system.genomic_resources import Registry

# GRCh37
registry_37 = Registry(build='GRCh37', release='87')
gtf_37 = registry_37.get_gtf_path()
fasta_37 = registry_37.get_fasta_path()
splice_sites_37 = registry_37.get_splice_sites_path()

# GRCh38 MANE
registry_38 = Registry(build='GRCh38_MANE', release='1.3')
gtf_38 = registry_38.get_gtf_path()
fasta_38 = registry_38.get_fasta_path()
splice_sites_38 = registry_38.get_splice_sites_path()
```

---

## Verification

### Check Downloaded Files

```bash
# List GRCh37 files
ls -lh data/ensembl/GRCh37/

# List GRCh38 MANE files
ls -lh data/mane/GRCh38/
```

### Verify File Integrity

```bash
# Check GTF line count (GRCh37)
wc -l data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf
# Expected: ~2.7 million lines

# Check GFF line count (MANE)
wc -l data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gff
# Expected: ~1.5 million lines

# Check splice sites
head data/ensembl/GRCh37/splice_sites_enhanced.tsv
head data/mane/GRCh38/splice_sites_enhanced.tsv
```

### Test with Base Models

**SpliceAI**:
```python
from meta_spliceai import run_base_model_predictions

results = run_base_model_predictions(
    base_model='spliceai',
    target_genes=['BRCA1'],
    mode='test'
)
print(f"✅ SpliceAI: {results['positions'].height} positions")
```

**OpenSpliceAI**:
```python
from meta_spliceai import run_base_model_predictions

results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1'],
    mode='test'
)
print(f"✅ OpenSpliceAI: {results['positions'].height} positions")
```

---

## Troubleshooting

### Issue: Download fails

**Symptoms**: Script exits with error during download

**Solutions**:
1. Check internet connection
2. Verify URLs are accessible:
   ```bash
   curl -I "https://ftp.ensembl.org/pub/grch37/release-87/"
   curl -I "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/"
   ```
3. Try manual download with verbose output

### Issue: Derivation fails

**Symptoms**: Splice sites file not created

**Solutions**:
1. Ensure FASTA is indexed:
   ```bash
   samtools faidx data/ensembl/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa
   ```
2. Check GTF/GFF format:
   ```bash
   head -20 data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf
   ```
3. Run derivation manually with verbose output

### Issue: GFF format not supported

**Symptoms**: Derivation fails with MANE GFF

**Solutions**:
1. Convert GFF to GTF:
   ```bash
   gffread data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gff \
     -T -o data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf
   ```
2. Install gffread if missing:
   ```bash
   conda install -c bioconda gffread
   ```

### Issue: Disk space

**Symptoms**: Download fails due to insufficient space

**Required Space**:
- GRCh37: ~5 GB
- GRCh38 MANE: ~3.5 GB
- Total (both): ~8.5 GB

**Solutions**:
1. Check available space:
   ```bash
   df -h data/
   ```
2. Clean up old files
3. Use external storage

---

## Download Scripts Summary

| Script | Purpose | Size | Time | Status |
|--------|---------|------|------|--------|
| `download_grch37_data.sh` | SpliceAI genomic data | ~5 GB | 15-30 min | ✅ Existing |
| `download_openspliceai_models.sh` | OpenSpliceAI models | ~14 MB | 2-5 min | ✅ Existing |
| `download_grch38_mane_data.sh` | OpenSpliceAI genomic data | ~3.5 GB | 20-40 min | ✅ NEW |

---

## Next Steps

### After Downloading GRCh37

1. **Test SpliceAI**:
   ```bash
   python scripts/testing/test_base_model_comprehensive.py
   ```

2. **Run evaluation**:
   ```bash
   python scripts/testing/comprehensive_spliceai_evaluation.py --build GRCh37
   ```

### After Downloading GRCh38 MANE

1. **Test OpenSpliceAI**:
   ```bash
   python scripts/testing/test_openspliceai_integration.py
   ```

2. **Run predictions**:
   ```python
   from meta_spliceai import run_base_model_predictions
   
   results = run_base_model_predictions(
       base_model='openspliceai',
       target_genes=['BRCA1', 'TP53'],
       mode='test'
   )
   ```

3. **Compare models**:
   ```bash
   python scripts/analysis/compare_base_models.py
   ```

---

## References

### External Resources

- **Ensembl GRCh37**: https://ftp.ensembl.org/pub/grch37/
- **MANE Project**: https://www.ncbi.nlm.nih.gov/refseq/MANE/
- **OpenSpliceAI**: https://github.com/Kuanhao-Chao/OpenSpliceAI
- **GRCh38 Reference**: https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/

### Internal Documentation

- [BASE_MODEL_SELECTION_AND_ROUTING.md](../development/BASE_MODEL_SELECTION_AND_ROUTING.md)
- [OPENSPLICEAI_INTEGRATION_COMPLETE.md](../development/OPENSPLICEAI_INTEGRATION_COMPLETE.md)
- [GRCH37_DOWNLOAD_GUIDE.md](../base_models/GRCH37_DOWNLOAD_GUIDE.md)

---

**Last Updated**: 2025-11-06  
**Status**: ✅ Complete - All download utilities available


