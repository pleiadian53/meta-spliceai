# Splice Sites CLI - User Guide

**Command**: `annotate_splice_sites`  
**Purpose**: Generate and validate splice site annotations from GTF files with enhanced metadata

---

## 🎯 Quick Start

### Generate with validation (recommended)
```bash
annotate_splice_sites --build mane-grch38 --validate
```

### List available builds
```bash
annotate_splice_sites --list-builds
```

### Custom GTF file
```bash
annotate_splice_sites \
    --gtf path/to/annotations.gtf \
    --output path/to/splice_sites.tsv \
    --validate
```

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Command-Line Options](#command-line-options)
3. [Available Builds](#available-builds)
4. [Examples](#examples)
5. [Output Format](#output-format)
6. [Validation](#validation)
7. [Advanced Usage](#advanced-usage)

---

## Overview

The `annotate_splice_sites` command provides a unified interface for:

- **Generating** splice site annotations from GTF files
- **Enhancing** annotations with 14 columns of metadata
- **Validating** consistency with source GTF files
- **Managing** multiple genomic builds easily

### What It Generates

A **14-column TSV file** containing:
- **8 core columns**: Basic splice site information
- **6 enhanced columns**: Gene names, biotypes, exon identifiers

### Key Features

✅ **Predefined builds** for common references (MANE, Ensembl)  
✅ **Automatic validation** with 5 consistency checks  
✅ **Rich metadata** for downstream analysis  
✅ **Fast processing** with progress indicators  

---

## Command-Line Options

### Input Options

| Option | Type | Description |
|--------|------|-------------|
| `--gtf GTF` | Path | Path to input GTF file (for custom builds) |
| `--build BUILD` | Choice | Use predefined build: `mane-grch38`, `ensembl-grch37`, `ensembl-grch38` |

### Output Options

| Option | Type | Description |
|--------|------|-------------|
| `--output PATH`, `-o PATH` | Path | Output TSV file path (auto-determined if using `--build`) |

### Processing Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--consensus-window N`, `-w N` | Integer | 2 | Consensus window size (nucleotides around splice site) |
| `--validate` | Flag | False | Validate consistency after generation |

### Utility Options

| Option | Type | Description |
|--------|------|-------------|
| `--list-builds` | Flag | Show available genomic build configurations |
| `--verbose`, `-v` | Flag | Enable detailed output |
| `--quiet`, `-q` | Flag | Suppress output (errors only) |
| `--help`, `-h` | Flag | Show help message |

---

## Available Builds

### View All Builds
```bash
annotate_splice_sites --list-builds
```

### Predefined Configurations

#### 1. **mane-grch38** (Recommended for OpenSpliceAI)
- **Name**: MANE Select v1.3
- **Genome**: GRCh38/hg38
- **GTF**: `data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf`
- **Output**: `data/mane/GRCh38/splice_sites_enhanced.tsv`
- **Used by**: OpenSpliceAI base model

#### 2. **ensembl-grch37** (Recommended for SpliceAI)
- **Name**: Ensembl Release 87
- **Genome**: GRCh37/hg19
- **GTF**: `data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf`
- **Output**: `data/ensembl/GRCh37/splice_sites_enhanced.tsv`
- **Used by**: SpliceAI base model

#### 3. **ensembl-grch38**
- **Name**: Ensembl Release 112
- **Genome**: GRCh38/hg38
- **GTF**: `data/ensembl/GRCh38/Homo_sapiens.GRCh38.112.gtf`
- **Output**: `data/ensembl/GRCh38/splice_sites_enhanced.tsv`

---

## Examples

### Example 1: Generate with Validation (Recommended)

Generate splice sites for MANE/GRCh38 with full validation:

```bash
annotate_splice_sites --build mane-grch38 --validate
```

**Output**:
```
================================================================================
GENERATING SPLICE SITES WITH ENHANCED METADATA
================================================================================

Input GTF:  data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf
Output TSV: data/mane/GRCh38/splice_sites_enhanced.tsv
Consensus window: ±2 nucleotides
Validation: Enabled

[splice_sites] Extracting splice sites from: MANE.GRCh38.v1.3.refseq_genomic.gtf
[splice_sites] ✓ Processed 18,264 transcripts
[splice_sites] ✓ Extracted 369,918 splice sites
[splice_sites] ✓ Unique genes: 18,200
[splice_sites] ✓ Donor/Acceptor ratio: 1.00 (balanced)

--------------------------------------------------------------------------------
VALIDATING SPLICE SITES CONSISTENCY
...
✅ ALL VALIDATION TESTS PASSED

================================================================================
✅ SPLICE SITES GENERATION COMPLETE
================================================================================
```

### Example 2: Generate Without Validation (Faster)

For quick iteration during development:

```bash
annotate_splice_sites --build ensembl-grch37
```

### Example 3: Custom GTF File

Generate from your own GTF file:

```bash
annotate_splice_sites \
    --gtf /path/to/custom_annotations.gtf \
    --output /path/to/output/splice_sites.tsv \
    --validate
```

### Example 4: Custom Consensus Window

Generate with wider consensus window (±3 nucleotides):

```bash
annotate_splice_sites \
    --build mane-grch38 \
    --consensus-window 3 \
    --validate
```

### Example 5: Verbose Output

See detailed progress and statistics:

```bash
annotate_splice_sites \
    --build ensembl-grch37 \
    --validate \
    --verbose
```

### Example 6: Quiet Mode

Minimal output (errors only):

```bash
annotate_splice_sites \
    --build mane-grch38 \
    --quiet
```

### Example 7: Regenerate Both Base Model Builds

Regenerate splice sites for both common base models:

```bash
# OpenSpliceAI (GRCh38)
annotate_splice_sites --build mane-grch38 --validate

# SpliceAI (GRCh37)
annotate_splice_sites --build ensembl-grch37 --validate
```

---

## Output Format

### File Structure

**Format**: Tab-separated values (TSV)  
**Columns**: 14  
**Header**: Yes (first row)

### Column Descriptions

| # | Column | Type | Description | Example |
|---|--------|------|-------------|---------|
| 1 | `chrom` | String | Chromosome | `chr1`, `chrX` |
| 2 | `start` | Integer | Start coordinate (0-based) | `65432` |
| 3 | `end` | Integer | End coordinate (1-based) | `65436` |
| 4 | `position` | Integer | Exact splice position (1-based) | `65434` |
| 5 | `strand` | String | Strand | `+`, `-` |
| 6 | `site_type` | String | Site type | `donor`, `acceptor` |
| 7 | `gene_id` | String | Gene identifier | `gene-OR4F5` |
| 8 | `transcript_id` | String | Transcript identifier | `rna-NM_001005484.2` |
| 9 | **`gene_name`** | String | Human-readable gene name | `OR4F5`, `BRCA1` |
| 10 | **`gene_biotype`** | String | Gene classification | `protein_coding` |
| 11 | **`transcript_biotype`** | String | Transcript type | `mRNA` |
| 12 | **`exon_id`** | String | Exon identifier | `ENSE00001` |
| 13 | **`exon_number`** | String | Exon number from GTF | `1`, `2` |
| 14 | **`exon_rank`** | Integer | Exon position in transcript | `1`, `2`, `3` |

**Bold** = Enhanced columns (new in v0.2.0)

### Sample Output

```tsv
chrom  start  end    position  strand  site_type  gene_id     transcript_id       gene_name  gene_biotype      transcript_biotype  exon_id  exon_number  exon_rank
chr1   65432  65436  65434     +       donor      gene-OR4F5  rna-NM_001005484.2  OR4F5      protein_coding    mRNA                         1            1
chr1   65517  65521  65519     +       acceptor   gene-OR4F5  rna-NM_001005484.2  OR4F5      protein_coding    mRNA                         2            2
chr1   65572  65576  65574     +       donor      gene-OR4F5  rna-NM_001005484.2  OR4F5      protein_coding    mRNA                         2            2
```

---

## Validation

### Validation Tests

When `--validate` is specified, 5 tests are performed:

#### 1. Transcript Count Match
**Check**: Transcripts in GTF (≥2 exons) == Transcripts in splice sites file  
**Pass**: Exact match (or <1% difference)

#### 2. Gene Count Match
**Check**: Unique genes in GTF == Unique genes in splice sites file  
**Pass**: Exact match (or very close)

#### 3. Donor/Acceptor Balance
**Check**: Ratio of donor to acceptor sites ≈ 1.0  
**Pass**: Ratio between 0.9-1.1 (well-balanced)

#### 4. No Duplicates
**Check**: Each (chrom, position, strand, type, transcript) is unique  
**Pass**: Zero duplicates found

#### 5. Splice Sites Per Transcript
**Check**: Each transcript has ≥2 splice sites  
**Pass**: Min ≥ 2, reasonable average (10-30 typical)

### Example Validation Output

```
================================================================================
VALIDATION RESULTS
================================================================================

1. Transcript Count Validation
--------------------------------------------------------------------------------
  GTF transcripts (≥2 exons):    18,264
  Splice sites transcripts:      18,264
  ✓ PASS: Transcript counts match exactly

2. Gene Count Validation
--------------------------------------------------------------------------------
  GTF genes (with splicing):     18,200
  Splice sites genes:            18,200
  ✓ PASS: Gene counts match exactly

3. Donor/Acceptor Balance
--------------------------------------------------------------------------------
  Donor sites:                   184,959
  Acceptor sites:                184,959
  Ratio (donor/acceptor):        1.000
  ✓ PASS: Well-balanced donor/acceptor sites

4. Duplicate Sites Check
--------------------------------------------------------------------------------
  ✓ PASS: No duplicate splice sites found

5. Splice Sites Per Transcript
--------------------------------------------------------------------------------
  Average sites per transcript:  20.3
  Min sites per transcript:      2
  Max sites per transcript:      724
  ✓ PASS: All transcripts have ≥2 splice sites (expected)

================================================================================
✅ ALL VALIDATION TESTS PASSED

The splice sites file is consistent with the GTF file.
================================================================================
```

---

## Advanced Usage

### Integration with Base Model Pass

The generated splice sites files are automatically used by base model passes:

```bash
# Generate splice sites first (if not present)
annotate_splice_sites --build mane-grch38 --validate

# Run base model pass (uses generated splice sites)
meta-spliceai-run --base-model openspliceai --chromosomes 21
```

### Programmatic Usage

You can also use the underlying Python API:

```python
from meta_spliceai.system.genomic_resources import extract_splice_sites_from_gtf

# Generate splice sites
df = extract_splice_sites_from_gtf(
    gtf_path='data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf',
    consensus_window=2,
    output_file='data/mane/GRCh38/splice_sites_enhanced.tsv',
    verbosity=1
)

# Returns pandas DataFrame with 14 columns
print(f"Generated {len(df):,} splice sites")
print(f"Columns: {list(df.columns)}")
```

### Custom Build Configuration

To add your own build to the predefined list:

1. Edit `meta_spliceai/cli/splice_sites_cli.py`
2. Add to `BUILD_CONFIGS` dictionary:

```python
BUILD_CONFIGS = {
    # ... existing configs ...
    'custom-build': {
        'name': 'Custom Build Name',
        'gtf': 'path/to/custom.gtf',
        'output': 'path/to/output.tsv',
        'description': 'Custom genomic build description'
    }
}
```

3. Reinstall: `pip install -e .`

### Batch Processing

Process multiple builds in a script:

```bash
#!/bin/bash
# regenerate_all_splice_sites.sh

builds=("mane-grch38" "ensembl-grch37" "ensembl-grch38")

for build in "${builds[@]}"; do
    echo "Processing: $build"
    annotate_splice_sites --build "$build" --validate
    echo ""
done

echo "✅ All splice sites regenerated!"
```

---

## Troubleshooting

### Error: GTF file not found

**Problem**: The GTF file path is incorrect or file doesn't exist

**Solution**: 
```bash
# Check if file exists
ls -lh data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf

# Or use absolute path
annotate_splice_sites \
    --gtf /absolute/path/to/file.gtf \
    --output /absolute/path/to/output.tsv
```

### Error: Validation failed

**Problem**: Generated splice sites don't match GTF

**Possible causes**:
- Corrupted GTF file
- Non-standard GTF format
- Database issues

**Solution**:
```bash
# Regenerate with verbose output
annotate_splice_sites --build mane-grch38 --validate --verbose

# Check GTF format
head -n 100 data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf
```

### Issue: Slow performance

**Problem**: Processing large GTF files takes time

**Expected timings**:
- MANE/GRCh38 (~19K transcripts): ~10-30 seconds
- Ensembl/GRCh37 (~196K transcripts): ~2-5 minutes

**Tips**:
- Use `--quiet` for less I/O overhead
- Skip validation during development: `--no-validate`
- Ensure GTF database (`.db`) exists (faster subsequent runs)

---

## See Also

- **Base Model CLI**: `meta-spliceai-run --help`
- **Evaluation CLI**: `meta-spliceai-eval --help`
- **Validation Details**: `dev/VALIDATION_VERIFICATION_ADDED.md`
- **Technical Details**: `meta_spliceai/system/genomic_resources/splice_sites.py`

---

## Summary

The `annotate_splice_sites` command provides a complete solution for splice site annotation management:

✅ **Easy**: Predefined builds for common references  
✅ **Validated**: 5 consistency checks ensure data integrity  
✅ **Enhanced**: 14 columns with rich metadata  
✅ **Fast**: Efficient processing with progress indicators  
✅ **Flexible**: Custom GTF files and options supported  

**Recommended workflow**:
```bash
# 1. List available builds
annotate_splice_sites --list-builds

# 2. Generate with validation
annotate_splice_sites --build mane-grch38 --validate

# 3. Use in downstream analysis
meta-spliceai-run --base-model openspliceai --chromosomes 21
```

---

**Version**: 0.2.0  
**Last Updated**: 2025-11-18

