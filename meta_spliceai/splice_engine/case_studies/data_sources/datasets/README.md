# Curated Training Datasets

This directory contains comprehensive documentation for curated training datasets used in meta-spliceai case studies and meta-model development.

## Overview

The datasets documented here are specifically designed for:
- **Alternative Splicing Analysis**: Understanding complex splicing patterns
- **Variant Impact Assessment**: Evaluating genetic variant effects on splicing
- **Disease-Specific Meta-Learning**: Training models for disease-associated splicing changes
- **Meta-Model Development**: Improving splice prediction model capacity

## Available Datasets

### [clinvar_20250831](clinvar_20250831/)
**Purpose**: Clinical variant analysis and splice impact assessment  
**Size**: 162MB (3,678,845 variants from ClinVar August 2025 release)  
**Coverage**: All main chromosomes (1-22, X, Y, MT) with GRCh38 coordinates  
**Status**: ✅ Complete and validated (100% coordinate consistency)  
**Documentation**: 
- [Dataset Overview](clinvar_20250831/README.md) - Quick start and file variants
- [File Processing Pipeline](clinvar_20250831/docs/file_processing_pipeline.md) - Detailed processing analysis with flowcharts
- [Coordinate Validation](clinvar_20250831/docs/coordinate_validation.md) - Comprehensive validation methodology and results
- [Dataset Characteristics](clinvar_20250831/docs/dataset_characteristics.md) - Statistical analysis and profiling
- [Validation Script](clinvar_20250831/validation/clinvar_dataset_validator.py) - Automated quality assurance tool

### [train_pc_7000_3mers_opt](train_pc_7000_3mers_opt/)
**Purpose**: Protein-coding gene training dataset for meta-model development  
**Size**: 595MB (6,708 genes, ~7.7M records)  
**Features**: 143 features including SpliceAI predictions, 3-mer composition, and genomic context  
**Status**: ✅ Complete and validated  
**Documentation**: 
- [Dataset Overview](train_pc_7000_3mers_opt/README.md) - Quick start and summary
- [Dataset Profile](train_pc_7000_3mers_opt/train_pc_7000_3mers_opt_profile.md) - Comprehensive characteristics
- [Technical Specification](train_pc_7000_3mers_opt/train_pc_7000_3mers_opt_technical_spec.md) - Detailed schema
- [Validation Script](train_pc_7000_3mers_opt/validate_train_pc_7000_3mers_opt.py) - Quality assurance tool

### [train_pc_5000_3mers_diverse](train_pc_5000_3mers_diverse/)
**Purpose**: Diverse gene training dataset for comprehensive meta-model development  
**Size**: 213MB (3,111 genes, ~569K records)  
**Features**: 143-148 features including SpliceAI predictions, 3-mer composition, and diverse gene types  
**Status**: ✅ Complete and validated  
**Documentation**: 
- [Dataset Overview](train_pc_5000_3mers_diverse/README.md) - Quick start and summary
- [Dataset Profile](train_pc_5000_3mers_diverse/train_pc_5000_3mers_diverse_profile.md) - Comprehensive characteristics
- [Technical Specification](train_pc_5000_3mers_diverse/train_pc_5000_3mers_diverse_technical_spec.md) - Detailed schema
- [Validation Script](train_pc_5000_3mers_diverse/validate_train_pc_5000_3mers_diverse.py) - Quality assurance tool

### [train_regulatory_10k_kmers](train_regulatory_10k_kmers/)
**Purpose**: Large-scale regulatory dataset for advanced meta-model development  
**Size**: 1.91GB (9,280 genes, ~3.7M records)  
**Features**: 1,167 features including SpliceAI predictions, 3-mer + 5-mer composition, and regulatory gene types  
**Status**: ✅ Complete and validated  
**Documentation**: 
- [Dataset Overview](train_regulatory_10k_kmers/README.md) - Quick start and summary
- [Dataset Profile](train_regulatory_10k_kmers/train_regulatory_10k_kmers_profile.md) - Comprehensive characteristics
- [Technical Specification](train_regulatory_10k_kmers/train_regulatory_10k_kmers_technical_spec.md) - Detailed schema
- [Validation Script](train_regulatory_10k_kmers/validate_train_regulatory_10k_kmers.py) - Quality assurance tool

### train_regulatory_enhanced_kmers *(Planned)*
**Purpose**: Multi-gene-type dataset including regulatory non-coding genes  
**Size**: ~10,000 genes (protein_coding, lncRNA, miRNA, snoRNA, snRNA)  
**Features**: Enhanced k-mer analysis (3-mer + 5-mer) with regulatory elements  
**Status**: 🚧 In planning phase  
**Expected**: Multi-gene-type splice pattern analysis and regulatory variant assessment

## Documentation Standards

Each dataset includes:

### 1. Dataset Profile (`*_profile.md`)
- **Overview**: Purpose, creation date, and key characteristics
- **Dataset Structure**: File organization and statistics
- **Gene Characteristics**: Selection criteria and genomic distribution
- **Feature Schema**: Complete feature descriptions and categories
- **Data Quality**: Distribution analysis and validation results
- **Use Cases**: Recommended applications and analysis approaches
- **Maintenance**: Version information and update history

### 2. Technical Specification (`*_technical_spec.md`)
- **Data Schema**: Complete column definitions and data types
- **File Formats**: Parquet and CSV specifications
- **Validation Rules**: Data integrity requirements
- **Performance Characteristics**: Loading and query performance
- **Integration Guidelines**: Code examples and best practices

## Usage Guidelines

### Loading Datasets
```python
import pandas as pd

# Load gene manifest
manifest = pd.read_csv('path/to/dataset/master/gene_manifest.csv')

# Load training data (batch processing recommended)
batch_files = [f'path/to/dataset/master/batch_{i:05d}.parquet' 
               for i in range(1, num_batches + 1)]
training_data = pd.concat([pd.read_parquet(f) for f in batch_files])
```

### Memory Considerations
- Large datasets require 2-3GB RAM for full loading
- Use batch processing for memory-constrained environments
- Consider column selection for specific analyses

### Cross-Validation Best Practices
- Use gene-level splits to avoid data leakage
- Maintain genomic balance across validation folds
- Consider splice site density when stratifying

## Dataset Naming Convention

Dataset directories follow the pattern: `{purpose}_{gene_count}_{features}_{version}`

Examples:
- `train_pc_7000_3mers_opt`: Training dataset, protein-coding, 7000 genes, 3-mers, optimized
- `train_pc_5000_3mers_diverse`: Training dataset, diverse genes, 5000 genes, 3-mers, diverse
- `train_regulatory_10k_kmers`: Training dataset, regulatory genes, 10k genes, multi-kmers
- `eval_nc_1000_5mers_v1`: Evaluation dataset, non-coding, 1000 genes, 5-mers, version 1

## Quality Assurance

All datasets undergo validation for:
- ✅ **Data Integrity**: Schema compliance and value ranges
- ✅ **Genomic Accuracy**: Coordinate validation and strand consistency  
- ✅ **Feature Completeness**: No missing critical features
- ✅ **Statistical Validity**: Appropriate distributions and correlations
- ✅ **Reproducibility**: Documented generation procedures

## Contributing New Datasets

When adding new datasets:

1. **Create Dataset Directory**: Follow naming convention
2. **Generate Documentation**: Both profile and technical specification
3. **Validate Data Quality**: Run integrity checks
4. **Update This README**: Add dataset to available list
5. **Version Control**: Tag dataset versions appropriately

### Required Documentation Template
```
dataset_name/
├── README.md                        # Quick overview and usage
├── dataset_name_profile.md          # Comprehensive overview
├── dataset_name_technical_spec.md   # Technical details
└── validate_dataset_name.py         # Validation and QA script
```

### Template Usage
Use the provided templates in `_templates/dataset_template/` to ensure consistent documentation across all datasets.

## Integration with Case Studies

These datasets are designed to integrate seamlessly with:
- **Variant Analysis Workflows**: VCF to alternative splice sites
- **Disease-Specific Studies**: Pathogenic variant impact assessment
- **Meta-Model Training**: Cross-validation and performance evaluation
- **Comparative Analysis**: Multi-dataset benchmarking

## Support and Maintenance

### Dataset Versioning
- **Major Version**: Significant schema or content changes
- **Minor Version**: Feature additions or quality improvements
- **Patch Version**: Bug fixes and documentation updates

### Update Schedule
- **Quarterly Reviews**: Data quality and relevance assessment
- **Annual Updates**: Integration of new genomic annotations
- **On-Demand**: Critical bug fixes and urgent improvements

## Contact

For questions about datasets or requests for new curated datasets, please refer to the main meta-spliceai documentation or contact the development team.

---

**Directory**: `meta_spliceai/splice_engine/case_studies/data_sources/datasets/`  
**Last Updated**: 2025-01-27  
**Enhanced Manifests**: Both major datasets now include comprehensive gene characteristics  
**Maintainer**: MetaSpliceAI Development Team
