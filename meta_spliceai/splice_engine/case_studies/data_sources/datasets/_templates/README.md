# Dataset Documentation Templates

This directory contains templates for creating consistent documentation for new datasets.

## Template Structure

### `dataset_template/`
Standard template for dataset documentation including:
- `README.md` - Dataset overview and quick start
- Template placeholders for profile and technical specification documents
- Validation script template structure

## Using Templates

When creating a new dataset documentation:

1. **Copy Template Directory**:
   ```bash
   cp -r _templates/dataset_template/ {new_dataset_name}/
   ```

2. **Replace Placeholders**:
   Replace all `{PLACEHOLDER}` values with actual dataset information:
   - `{DATASET_NAME}` - Dataset directory name
   - `{DATASET_PURPOSE}` - Brief purpose description
   - `{DATASET_SIZE}` - Size in MB/GB with record counts
   - `{FEATURE_COUNT}` - Number of features
   - `{KEY_FEATURES}` - Main feature categories
   - `{CREATION_DATE}` - Dataset creation date
   - `{GENE_COUNT}` - Number of genes
   - `{GENE_TYPES}` - Gene biotypes included
   - `{CHROMOSOME_COVERAGE}` - Chromosomes covered
   - `{BATCH_COUNT}` - Number of batch files
   - `{MEMORY_USAGE}` - Estimated memory usage
   - `{USE_CASE_1-4}` - Primary use cases
   - `{DATASET_VERSION}` - Dataset version
   - `{DOC_VERSION}` - Documentation version
   - `{LAST_UPDATED}` - Last update date

3. **Create Full Documentation**:
   - Write comprehensive profile document
   - Create detailed technical specification
   - Develop validation script
   - Update main datasets README

## Template Placeholders Reference

### Common Placeholders
```
{DATASET_NAME}           # e.g., train_regulatory_enhanced_kmers
{DATASET_PURPOSE}        # e.g., Multi-gene-type regulatory analysis
{DATASET_SIZE}           # e.g., 1.2GB (10,000 genes, ~15M records)
{FEATURE_COUNT}          # e.g., 180
{KEY_FEATURES}           # e.g., SpliceAI predictions, 3/5-mer composition, regulatory elements
{CREATION_DATE}          # e.g., 2025-08-25
{GENE_COUNT}             # e.g., 10,000
{GENE_TYPES}             # e.g., protein_coding, lncRNA, miRNA, snoRNA, snRNA
{CHROMOSOME_COVERAGE}    # e.g., All autosomes (1-22) + sex chromosomes (X, Y)
{BATCH_COUNT}            # e.g., 40
{MEMORY_USAGE}           # e.g., ~2.5GB RAM
{USE_CASE_1}             # e.g., Regulatory variant impact assessment
{USE_CASE_2}             # e.g., Non-coding splice pattern analysis
{USE_CASE_3}             # e.g., Multi-gene-type meta-model training
{USE_CASE_4}             # e.g., Comparative splicing mechanism research
{DATASET_VERSION}        # e.g., 1.0
{DOC_VERSION}            # e.g., 1.0
{LAST_UPDATED}           # e.g., 2025-08-25
```

## Quality Standards

All dataset documentation should include:
- ✅ Comprehensive overview with clear purpose
- ✅ Detailed technical specifications
- ✅ Validation and quality assurance tools
- ✅ Usage examples and integration guidelines
- ✅ Performance characteristics and memory requirements
- ✅ Version control and maintenance information

---

**Templates Version**: 1.0  
**Last Updated**: 2025-08-23
