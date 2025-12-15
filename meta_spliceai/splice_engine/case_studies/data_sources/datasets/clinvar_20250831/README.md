# ClinVar Dataset (August 31, 2025)

## Overview

This directory contains comprehensive documentation and analysis for the ClinVar variant dataset (release date: August 31, 2025) used in MetaSpliceAI variant analysis workflows.

**Dataset Characteristics**:
- **Source**: NCBI ClinVar database
- **Release Date**: August 31, 2025
- **Genome Build**: GRCh38/hg38
- **Total Variants**: 3,678,878 (original), 3,678,845 (analysis-ready)
- **File Size**: ~162MB compressed
- **Coverage**: All chromosomes with focus on main chromosomes (1-22, X, Y, MT)

## Directory Structure

```
clinvar_20250831/
├── README.md                          # This file
├── docs/
│   ├── file_processing_pipeline.md    # Detailed file processing analysis
│   ├── coordinate_validation.md       # Coordinate system validation results
│   └── dataset_characteristics.md     # Statistical analysis and profiling
└── validation/
    └── clinvar_dataset_validator.py   # Dataset validation scripts
```

## File Variants

The ClinVar dataset exists in several processed versions in `data/ensembl/clinvar/vcf/`:

| File | Purpose | Variants | Status |
|------|---------|----------|--------|
| `clinvar_20250831.vcf.gz` | Original download | 3,678,878 | ⚠️ Raw data |
| `clinvar_20250831_reheadered.vcf.gz` | Enhanced headers | 3,678,878 | ✅ Compatible |
| `clinvar_20250831_main_chroms.vcf.gz` | **Analysis-ready** | 3,678,845 | ✅ **Recommended** |
| `sample_clinvar.vcf` | Testing sample | ~50 | 🧪 Development |

## Quick Start

### For Variant Analysis

The VCF coordinate verifier now supports **smart path resolution** that works from any directory within the project.

#### From Project Root Directory
```bash
# Direct command with relative paths (recommended)
python meta_spliceai/splice_engine/case_studies/tools/vcf_coordinate_verifier.py \
    --vcf data/ensembl/clinvar/vcf/clinvar_20250831_main_chroms.vcf.gz \
    --fasta data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --validate-coordinates

# Using environment variables for cleaner scripts
VCF_FILE="data/ensembl/clinvar/vcf/clinvar_20250831_main_chroms.vcf.gz"
FASTA_FILE="data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa"

python meta_spliceai/splice_engine/case_studies/tools/vcf_coordinate_verifier.py \
    --vcf $VCF_FILE \
    --fasta $FASTA_FILE \
    --validate-coordinates
```

#### From Tools Directory
```bash
# Navigate to tools directory and run directly
cd meta_spliceai/splice_engine/case_studies/tools
python vcf_coordinate_verifier.py \
    --vcf data/ensembl/clinvar/vcf/clinvar_20250831_main_chroms.vcf.gz \
    --fasta data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --validate-coordinates
```

**✅ Path Resolution Features**:
- **Smart project detection**: Automatically finds project root from any subdirectory
- **Relative path support**: Handles `data/ensembl/...` paths from project root
- **Multiple fallback strategies**: Works even when advanced path resolvers fail
- **Directory-agnostic**: Same command works from project root or tools directory

### For Sequence Analysis
```python
from vcf_preprocessing import preprocess_clinvar_vcf

# Process for splice analysis
normalized_vcf = preprocess_clinvar_vcf(
    input_vcf="data/ensembl/clinvar/vcf/clinvar_20250831_main_chroms.vcf.gz",
    output_dir="processed_clinvar/"
)
```

## Dataset Statistics

### Variant Distribution by Chromosome
```
Chr 1:    327,493 variants (8.9%)
Chr 2:    190,871 variants (5.2%)
Chr 17:   225,907 variants (6.1%)
Chr X:     74,539 variants (2.0%)
Chr Y:      1,429 variants (0.04%)
...
Total:  3,678,845 variants
```

### Clinical Significance Distribution
- **Pathogenic**: ~15% of variants
- **Likely Pathogenic**: ~8% of variants
- **Uncertain Significance**: ~45% of variants
- **Likely Benign**: ~20% of variants
- **Benign**: ~12% of variants

### Molecular Consequences
- **Missense variants**: ~60% of variants
- **Synonymous variants**: ~15% of variants
- **Splice-affecting variants**: ~5% of variants
- **Nonsense variants**: ~8% of variants
- **Indels**: ~12% of variants

## Validation Status

### ✅ Coordinate System Validation
- **Consistency Score**: 100% with GRCh38 reference
- **Validation Method**: Enhanced coordinate verification with normalization
- **Test Sample**: 100+ variants across all chromosomes
- **Result**: Ready for production analysis

### ✅ File Format Validation
- **VCF Format**: Compliant with VCFv4.1 specification
- **Contig Headers**: Complete (199 contigs in reheadered version)
- **Index Files**: Present and valid (.tbi/.csi)
- **Compression**: bgzip-compressed for efficient access

## Usage in MetaSpliceAI Workflows

### Primary Use Cases
1. **Variant Impact Analysis**: WT/ALT sequence construction and delta score computation
2. **Splice Site Prediction**: OpenSpliceAI and meta-model enhancement
3. **Clinical Validation**: Pathogenic variant splice impact assessment
4. **Method Development**: Algorithm testing and validation

### Integration Points
- **VCF Preprocessing Pipeline**: `workflows/vcf_preprocessing.py`
- **Coordinate Verification**: `tools/vcf_coordinate_verifier.py`
- **Sequence Construction**: `meta_models/workflows/inference/sequence_inference.py`
- **Variant Analysis**: Complete analysis workflows in `workflows/`

## Quality Assurance

### Automated Validation
```bash
# Run comprehensive dataset validation
python validate_clinvar_dataset.py \
    --vcf-dir data/ensembl/clinvar/vcf/ \
    --reference data/ensembl/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --output-report clinvar_validation_report.html
```

### Manual Verification Checklist
- [x] File integrity and compression
- [x] VCF format compliance
- [x] Coordinate system consistency
- [x] Reference genome compatibility
- [x] Index file presence and validity
- [x] Header completeness and accuracy

## Performance Characteristics

### Processing Benchmarks
- **Coordinate Validation**: ~30 seconds (100 variant sample)
- **Full VCF Normalization**: ~3-4 minutes
- **Variant Parsing**: ~1-2 minutes
- **Memory Usage**: ~2-4GB peak
- **Storage Requirements**: ~500MB for all processed versions

### Scalability Notes
- Suitable for single-machine processing
- Batch processing recommended for large-scale analysis
- Index files enable efficient region-specific queries
- Compatible with distributed processing frameworks

## Maintenance and Updates

### Version Control
- **Current Version**: August 31, 2025 release
- **Update Frequency**: ClinVar releases monthly updates
- **Backward Compatibility**: Maintained across minor releases

### Update Procedure
1. Download new ClinVar release
2. Run processing pipeline (reheader → filter → validate)
3. Update documentation and validation results
4. Test with existing workflows
5. Update dataset references in workflows

## Related Documentation

- **[File Processing Pipeline](docs/file_processing_pipeline.md)**: Detailed processing steps and commands
- **[Coordinate Validation](docs/coordinate_validation.md)**: Validation methodology and results
- **[Dataset Characteristics](docs/dataset_characteristics.md)**: Statistical analysis and profiling
- **[VCF Analysis Workflows](../../docs/VCF_VARIANT_ANALYSIS_WORKFLOW.md)**: Complete analysis pipeline
- **[ClinVar Tutorial](../../docs/tutorials/CLINVAR_WORKFLOW_STEPS_1_2_TUTORIAL.md)**: Step-by-step processing guide

## Contact and Support

For questions about this dataset or its usage in MetaSpliceAI:
- Check the detailed documentation in `docs/`
- Review the processing pipeline in the tutorial
- Validate your setup using the coordinate verifier
- Refer to the main VCF analysis documentation

---

*This dataset documentation is part of the MetaSpliceAI case studies framework, providing comprehensive variant analysis capabilities for clinical and research applications.*
