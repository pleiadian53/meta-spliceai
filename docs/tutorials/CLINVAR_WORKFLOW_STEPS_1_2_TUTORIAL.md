# ClinVar Variant Analysis Workflow: Steps 1-2 Tutorial

## Overview

This tutorial documents the complete process for successfully running Steps 1-2 of the ClinVar variant analysis workflow, including all prerequisite setup, troubleshooting steps, and validation procedures. 

**Complete 5-Step Workflow Documentation:**
- **Steps 1-2**: This tutorial (VCF normalization and variant parsing)
- **Steps 3-5**: See `<META_SPLICEAI_ROOT>/docs/development/VCF_VARIANT_ANALYSIS_WORKFLOW.md` and `<META_SPLICEAI_ROOT>/meta_spliceai/splice_engine/case_studies/docs/OPENSPLICEAI_VARIANT_ANALYSIS_Q8_Q9.md`

## Path Conventions

- **`<META_SPLICEAI_ROOT>`**: Project root directory where MetaSpliceAI is installed (e.g., `/home/user/meta-spliceai/`, `/opt/meta-spliceai/`)
- **`<OUTPUT_DIR>`**: User-defined output directory for workflow results (e.g., `/tmp/clinvar_analysis/`, `/data/output/`)

## Table of Contents

1. [Prerequisites and Dependencies](#prerequisites-and-dependencies)
2. [Data Preparation](#data-preparation)
3. [Step 1: VCF Normalization](#step-1-vcf-normalization)
4. [Step 2: Variant Filtering and Parsing](#step-2-variant-filtering-and-parsing)
5. [Validation and Testing](#validation-and-testing)
6. [Troubleshooting](#troubleshooting)
7. [Expected Results](#expected-results)

## Prerequisites and Dependencies

### Required Tools

```bash
# Install bcftools (required for VCF processing)
mamba activate surveyor
mamba install -c bioconda bcftools

# Install pysam (required for contig compatibility testing)
mamba install -c bioconda pysam
```

### Why pysam is Required

`pysam` is needed for:
- **VCF/FASTA contig compatibility testing**: Reading VCF headers and FASTA indices
- **Genomic coordinate validation**: Ensuring chromosome naming consistency
- **File format validation**: Checking VCF and FASTA file integrity

The contig compatibility test (`test_fasta_vcf_contig_compatibility.py`) uses pysam to:
```python
import pysam

# Read VCF header contigs
vcf_file = pysam.VariantFile(vcf_path)
vcf_contigs = set(vcf_file.header.contigs.keys())

# Read FASTA reference names
fasta_file = pysam.FastaFile(fasta_path)
fasta_refs = set(fasta_file.references)
```

## Data Preparation

### 1. Fix VCF Header Contig Information

**Problem**: ClinVar VCF files often lack proper contig header lines, causing compatibility issues.

**Solution**: Use bcftools reheader to add contig information from FASTA index:

```bash
# Step 1: Create FASTA index if it doesn't exist
samtools faidx <META_SPLICEAI_ROOT>/data/ensembl/reference.fa

# Step 2: Add contig headers to VCF using FASTA index
bcftools reheader --fai <META_SPLICEAI_ROOT>/data/ensembl/reference.fa.fai \
    <META_SPLICEAI_ROOT>/data/ensembl/clinvar/vcf/clinvar.vcf.gz \
    -o <META_SPLICEAI_ROOT>/data/ensembl/clinvar/vcf/clinvar_reheadered.vcf.gz

# Step 3: Index the reheadered VCF
bcftools index <META_SPLICEAI_ROOT>/data/ensembl/clinvar/vcf/clinvar_reheadered.vcf.gz
```

**Validation**: Check that contig headers are present:
```bash
bcftools view -h clinvar_reheadered.vcf.gz | grep "##contig" | head -5
```

### 2. Filter to Main Chromosomes

**Problem**: VCF files may contain contigs not present in the reference FASTA.

**Solution**: Filter to main chromosomes only:

```bash
bcftools view -r 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X,Y,MT \
    clinvar_reheadered.vcf.gz \
    -Oz -o clinvar_main_chroms.vcf.gz

bcftools index clinvar_main_chroms.vcf.gz
```

### 3. Verify Contig Compatibility

Run the contig compatibility test:

```bash
cd <META_SPLICEAI_ROOT>/meta_spliceai/splice_engine/case_studies/tests
python test_fasta_vcf_contig_compatibility.py \
    --vcf <META_SPLICEAI_ROOT>/data/ensembl/clinvar/vcf/clinvar_main_chroms.vcf.gz \
    --fasta Homo_sapiens.GRCh38.dna.primary_assembly.fa
```

**Expected Success Output**:
```
================================================================================
FASTA/VCF CONTIG COMPATIBILITY TEST RESULTS
================================================================================

Files tested:
  VCF:   <META_SPLICEAI_ROOT>/data/ensembl/clinvar/vcf/clinvar_main_chroms.vcf.gz
  FASTA: <META_SPLICEAI_ROOT>/data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa

Contig Statistics:
  VCF contigs:     194
  FASTA references: 194
  Intersection:    194
  VCF coverage:    100.0%
  FASTA coverage:  100.0%

Chromosome Completeness (Intersection):
  Autosomes (1-22): 22/22
  Sex chromosomes:  2/2 (X, Y)
  Mitochondrial:    ✓
  Overall score:    100.0%

Compatibility Status: ✓ COMPATIBLE

================================================================================
✓ Compatibility test PASSED
```

## Step 1: VCF Normalization

### Purpose
- Split multiallelic variants into single-allele records
- Left-align and normalize indels
- Add variant type annotations
- Create properly indexed output

### Command

```bash
cd <META_SPLICEAI_ROOT>
python -m meta_spliceai.splice_engine.case_studies.workflows.clinvar_variant_analysis \
    --step 1 \
    --input-vcf <META_SPLICEAI_ROOT>/data/ensembl/clinvar/vcf/clinvar_main_chroms.vcf.gz \
    --output-dir <OUTPUT_DIR>/step1_output
```

### bcftools Commands Used Internally

The workflow executes these bcftools commands:

1. **Multiallelic splitting and normalization**:
```bash
bcftools norm \
    -f /path/to/reference.fa \
    -m -both \
    -Oz \
    -o step1_temp.vcf.gz \
    input.vcf.gz
```

2. **Add variant type annotations**:
```bash
bcftools +fill-tags \
    -Oz \
    -o step1_normalized.vcf.gz \
    step1_temp.vcf.gz
```

3. **Index the output**:
```bash
tabix -p vcf step1_normalized.vcf.gz
```

### Expected Results

**Success Indicators**:
- Process completes without errors
- Output file `step1_normalized.vcf.gz` created (~171MB for full ClinVar)
- Index file `step1_normalized.vcf.gz.tbi` created
- Log shows: "✓ Multiallelic splitting completed"
- Log shows: "✓ TYPE annotation added"
- Log shows: "✓ VCF indexing completed"

**Validation**:
```bash
# Check variant count
bcftools stats step1_normalized.vcf.gz | grep "number of records"

# Verify no multiallelic sites remain
bcftools stats step1_normalized.vcf.gz | grep "multiallelic sites"
# Should show: "number of multiallelic sites: 0"

# Check file integrity
bcftools view -H step1_normalized.vcf.gz | head -3
```

**Example Output Statistics**:
```
SN    0    number of records:      3678845
SN    0    number of multiallelic sites:   0
SN    0    number of SNPs: 3404132
SN    0    number of indels:       255266
```

## Step 2: Variant Filtering and Parsing

### Purpose
- Extract variant information from normalized VCF
- Parse ClinVar-specific INFO fields
- Create structured TSV output for downstream analysis

### Command

```bash
python -m meta_spliceai.splice_engine.case_studies.workflows.clinvar_variant_analysis \
    --step 2 \
    --input-vcf <OUTPUT_DIR>/step1_output/step1_normalized.vcf.gz \
    --output-dir <OUTPUT_DIR>/step2_output
```

### bcftools Commands Used Internally

The workflow uses:

```bash
bcftools view -H input.vcf.gz
```

Then parses the output to extract:
- Standard VCF fields: CHROM, POS, ID, REF, ALT, QUAL, FILTER
- ClinVar INFO fields: CLNSIG, CLNREVSTAT, MC, CLNDN, TYPE

### Expected Results

**Success Indicators**:
- Process completes in ~1-2 minutes for 3.6M variants
- Output file `step2_filtered_variants.tsv` created (~584MB for full ClinVar)
- Log shows: "Found 3678845 variant records"
- Log shows: "Parsed 3678845 variant records"
- Log shows: "Saved 3678845 filtered variants to ..."

**Output Format**:
The TSV file contains columns:
```
CHROM	POS	ID	REF	ALT	QUAL	FILTER	CLNSIG	CLNREVSTAT	MC	CLNDN	TYPE
1	66926	1525977	AG	A	.	.	Uncertain_significance	no_assertion_criteria_provided	SO:0000159|Deletion	...	snv
```

## Validation and Testing

### Run Systematic Tests

```bash
cd <META_SPLICEAI_ROOT>
python -m pytest tests/case_studies/test_step1_vcf_normalization.py -v
```

**Expected Test Results**:
```
tests/case_studies/test_step1_vcf_normalization.py::TestStep1VCFNormalization::test_step1_multiallelic_splitting_success PASSED
tests/case_studies/test_step1_vcf_normalization.py::TestStep1VCFNormalization::test_step1_fill_tags_optional PASSED
tests/case_studies/test_step1_vcf_normalization.py::TestStep1VCFNormalization::test_step1_bcftools_not_available PASSED
tests/case_studies/test_step1_vcf_normalization.py::TestStep1VCFNormalization::test_step1_validation_metrics PASSED
tests/case_studies/test_step1_vcf_normalization.py::TestStep1VCFNormalization::test_step1_expected_success_criteria PASSED
tests/case_studies/test_step1_vcf_normalization.py::test_step1_success_criteria_checklist PASSED

====================== 6 passed, 2 skipped, 1 warning ======================
```

### Validate File Sizes and Content

```bash
# Check file sizes (approximate for full ClinVar dataset)
ls -lh step1_normalized.vcf.gz        # ~171MB
ls -lh step2_filtered_variants.tsv    # ~584MB

# Verify variant counts match
wc -l step2_filtered_variants.tsv     # Should be 3,678,846 (header + variants)
bcftools view -H step1_normalized.vcf.gz | wc -l  # Should be 3,678,845
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "No such file or directory" for VCF
**Cause**: VCF file path incorrect or file doesn't exist
**Solution**: Verify file path and ensure VCF is properly created

#### 2. "Failed to open file" with bcftools
**Cause**: Missing contig headers in VCF
**Solution**: Run the VCF reheader step as described above

#### 3. "sequence was not found" error
**Cause**: VCF contains contigs not in reference FASTA
**Solution**: Filter VCF to main chromosomes only

#### 4. Import errors with pysam
**Cause**: pysam not installed
**Solution**: `mamba install -c bioconda pysam`

#### 5. Contig compatibility test fails
**Cause**: Chromosome naming mismatch (e.g., "chr1" vs "1")
**Solution**: Ensure consistent naming between VCF and FASTA

### Debug Commands

```bash
# Check VCF header
bcftools view -h input.vcf.gz | head -20

# Check contig headers specifically
bcftools view -h input.vcf.gz | grep "##contig"

# Verify FASTA index exists
ls -la reference.fa.fai

# Test bcftools functionality
bcftools --version
bcftools view input.vcf.gz | head -5
```

## Performance Notes

### Processing Times (3.6M variants)
- **Step 1 VCF Normalization**: ~3-4 minutes
- **Step 2 Variant Parsing**: ~1-2 minutes
- **Contig Compatibility Test**: ~30 seconds

### Memory Usage
- Peak memory usage: ~2-4GB
- Recommended system: 8GB+ RAM
- Storage: ~1GB for intermediate files

### Optimization Tips
- Use SSD storage for faster I/O
- Ensure sufficient /tmp space for intermediate files
- Run on systems with adequate RAM to avoid swapping

## Next Steps

After completing Steps 1-2:
1. **Step 3**: OpenSpliceAI scoring (requires OpenSpliceAI installation)
2. **Step 4**: Delta score parsing and event classification
3. **Step 5**: PR-AUC evaluation and stratification

The normalized VCF and parsed TSV files are now ready for splice site analysis and evaluation.
