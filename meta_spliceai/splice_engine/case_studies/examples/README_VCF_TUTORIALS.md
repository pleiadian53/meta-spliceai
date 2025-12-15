# VCF Parsing Tutorials for Splice Site Analysis

This directory contains tutorials for parsing VCF files and analyzing variants that affect splicing.

## Overview

Three main components are available:

1. **`vcf_parsing_tutorial.py`** - Basic VCF parsing tutorial demonstrating core concepts
2. **`vcf_clinvar_tutorial.py`** - Enhanced tutorial for processing real ClinVar data
3. **`vcf_openspliceai_integration.py`** - Complete pipeline integration with OpenSpliceAI

## Tutorial 1: Basic VCF Parsing (`vcf_parsing_tutorial.py`)

### Purpose
Demonstrates the fundamental workflow for VCF parsing and variant analysis:
- Creating sample VCF files
- Parsing VCF format
- Variant standardization
- WT/ALT sequence construction
- Mock splice site analysis

### Key Features
- Educational walkthrough with inline documentation
- Works with synthetic data (no external dependencies)
- Shows complete pipeline from VCF to splice analysis preparation

### Usage
```bash
python vcf_parsing_tutorial.py
```

## Tutorial 2: ClinVar Integration (`vcf_clinvar_tutorial.py`)

### Purpose
Production-ready tutorial for processing real ClinVar VCF files:
- Parses ClinVar-specific annotations
- Filters for pathogenic splice-affecting variants
- Handles large-scale variant datasets
- Exports structured data for downstream analysis

### Key Features
- ClinVar-specific INFO field parsing (CLNSIG, CLNDN, MC)
- Pathogenicity filtering
- Molecular consequence analysis
- Batch processing capabilities
- Data persistence in TSV/Parquet formats

### Directory Structure
```
data/ensembl/clinvar/
├── vcf/                    # Raw VCF files from ClinVar
├── processed/              # Parsed variant data
├── splice_variants/        # Filtered splice-affecting variants
└── logs/                   # Processing logs
```

### Usage

#### Step 1: Download ClinVar Data
Use the provided download helper script:
```bash
python scripts/data_management/download_clinvar.py --genome GRCh38 --output-dir data/ensembl/clinvar/vcf
```

Or manually download:
```bash
# Download latest ClinVar VCF (GRCh38)
wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz
wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi

# Move to appropriate directory
mv clinvar.vcf.gz data/ensembl/clinvar/vcf/
mv clinvar.vcf.gz.tbi data/ensembl/clinvar/vcf/
```

#### Step 2: Run the Tutorial
```bash
# Activate the surveyor environment first
mamba activate surveyor

# Run with auto-detection (will find clinvar_20250831.vcf.gz automatically)
python vcf_clinvar_tutorial.py

# Process more variants
python vcf_clinvar_tutorial.py --max-variants 1000

# Process all variants (remove limit)
python vcf_clinvar_tutorial.py --max-variants 0

# Specify a particular VCF file
python vcf_clinvar_tutorial.py --vcf-file data/ensembl/clinvar/vcf/clinvar_20250831.vcf.gz
```

### Output Files

| File | Description |
|------|-------------|
| `clinvar_variants_all.tsv` | All parsed ClinVar variants |
| `clinvar_splice_pathogenic.tsv` | Pathogenic splice-affecting variants |
| `clinvar_splice_variants_processed.tsv` | Standardized variants with sequences |
| `processing_stats.txt` | Summary statistics |

## Tutorial 3: OpenSpliceAI Integration (`vcf_openspliceai_integration.py`)

### Purpose
Complete pipeline demonstrating integration with OpenSpliceAI for splice site prediction:
- Loads processed ClinVar variants
- Computes OpenSpliceAI delta scores
- Analyzes splicing patterns
- Generates comprehensive reports

### Key Features
- Real or mock OpenSpliceAI predictions
- Splicing pattern analysis
- Impact classification
- Comprehensive reporting
- Visualization-ready outputs

### Prerequisites
- Completed ClinVar tutorial (generates input data)
- Optional: OpenSpliceAI model weights
- Optional: Reference genome FASTA

### Usage
```bash
# Basic usage with mock predictions
python vcf_openspliceai_integration.py --input-dir data/ensembl/clinvar/splice_variants

# With real OpenSpliceAI model
python vcf_openspliceai_integration.py \
    --input-dir data/ensembl/clinvar/splice_variants \
    --openspliceai-model path/to/model.h5 \
    --fasta-path data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa
```

### Output Files

| File | Description |
|------|-------------|
| `openspliceai_analysis_results.tsv` | Detailed results with scores and patterns |
| `analysis_summary_report.txt` | Summary statistics and top variants |

## Data Model

### StandardizedVariant
```python
@dataclass
class StandardizedVariant:
    chrom: str                  # Chromosome
    start: int                  # 1-based start position
    end: int                    # 1-based end position (inclusive)
    ref: str                    # Reference allele
    alt: str                    # Alternate allele
    variant_type: str           # SNV, insertion, deletion, etc.
    coordinate_system: str      # "1-based" or "0-based"
    reference_genome: str       # GRCh37, GRCh38, etc.
```

### ClinVar Annotations
- **CLNSIG**: Clinical significance (Pathogenic, Likely_pathogenic, etc.)
- **CLNDN**: Disease name
- **CLNREVSTAT**: Review status
- **MC**: Molecular consequence (SO terms)

### OpenSpliceAI Scores
- **delta_score_acceptor**: Acceptor site gain score
- **delta_score_donor**: Donor site gain score
- **delta_score_acceptor_loss**: Acceptor site loss score
- **delta_score_donor_loss**: Donor site loss score
- **max_delta_score**: Maximum absolute delta score

## Helper Scripts

### ClinVar Download Helper (`scripts/data_management/download_clinvar.py`)

Automates downloading ClinVar VCF files from NCBI:

```bash
# List available files
python scripts/data_management/download_clinvar.py --list-available --genome GRCh38

# Download latest file
python scripts/data_management/download_clinvar.py \
    --genome GRCh38 \
    --output-dir data/ensembl/clinvar/vcf

# Download specific file
python scripts/data_management/download_clinvar.py \
    --filename clinvar_20240101.vcf.gz \
    --output-dir data/ensembl/clinvar/vcf
```

## Complete Workflow Example

```bash
# 1. Download ClinVar data
python scripts/data_management/download_clinvar.py \
    --genome GRCh38 \
    --output-dir data/ensembl/clinvar/vcf

# 2. Process ClinVar variants
python meta_spliceai/splice_engine/case_studies/examples/vcf_clinvar_tutorial.py

# 3. Integrate with OpenSpliceAI
python meta_spliceai/splice_engine/case_studies/examples/vcf_openspliceai_integration.py \
    --input-dir data/ensembl/clinvar/splice_variants \
    --output-dir data/ensembl/clinvar/openspliceai_analysis
```

## Next Steps

1. **Scale Analysis**: Process full ClinVar dataset without variant limits
2. **Model Integration**: Connect with real OpenSpliceAI model weights
3. **Visualization**: Add plotting and visualization capabilities
4. **Validation**: Compare predictions with known splice-affecting variants
5. **Database Integration**: Store results in structured database for querying

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the correct directory with proper Python path
2. **Missing Files**: Check that ClinVar data has been downloaded to the expected location
3. **Memory Issues**: For large datasets, consider processing in batches
4. **Model Loading**: Verify OpenSpliceAI model path and compatibility

### Getting Help

- Check log files in `data/ensembl/clinvar/logs/`
- Review processing statistics in output directories
- Examine sample outputs to verify data format expectations
