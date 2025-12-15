# Dataset Profile: train_pc_7000_3mers_opt

## Overview

**Dataset Name**: `train_pc_7000_3mers_opt`  
**Creation Date**: 2025-08-23  
**Purpose**: Protein-coding gene training dataset for meta-model development focused on alternative splicing pattern analysis  
**Dataset Type**: Curated training dataset for splice site prediction and variant impact assessment  
**Location**: `/home/bchiu/work/meta-spliceai/train_pc_7000_3mers_opt/`

## Dataset Description

This dataset represents a comprehensive collection of splice site predictions and genomic features derived from 6,708 protein-coding genes. It was specifically curated to evaluate and improve meta-model capacity for capturing alternative splicing patterns induced by variants or diseases.

### Key Characteristics

- **Gene Selection Strategy**: Random sampling from protein-coding genes for biological diversity
- **Feature Engineering**: 3-mer k-mer analysis with comprehensive splice site characterization
- **Prediction Framework**: SpliceAI-based predictions with enhanced meta-features
- **Target Application**: Disease-specific meta-learning and variant impact assessment

## Dataset Structure

### Directory Organization
```
train_pc_7000_3mers_opt/
├── master/                          # Final training dataset
│   ├── batch_00001.parquet         # Training data batches (28 files)
│   ├── batch_00002.parquet
│   ├── ...
│   ├── batch_00028.parquet
│   └── gene_manifest.csv           # Gene metadata and manifest
├── batch_00001_trim.parquet        # Intermediate files (28 files)
├── batch_00002_trim.parquet
├── ...
└── batch_00028_trim.parquet
```

### File Statistics
- **Total Dataset Size**: 595MB (master directory)
- **Number of Batch Files**: 28 parquet files
- **Gene Manifest**: 6,708 genes with enhanced characteristics (15 columns)
- **Training Records**: ~275,076 records per batch (estimated ~7.7M total records)

## Gene Characteristics

### Gene Selection Criteria
- **Gene Type**: Exclusively protein-coding genes (100%)
- **Selection Method**: Random sampling for biological diversity
- **Total Genes**: 6,708 unique genes
- **Genome Coverage**: All autosomes (1-22) plus sex chromosomes (X, Y)

### Gene Characteristics (Enhanced Manifest)
- **Mean Splice Density**: 6.28 sites/kb
- **Median Splice Density**: 2.16 sites/kb
- **Mean Gene Length**: 70,902 bp
- **Median Gene Length**: 30,476 bp
- **Splice Site Balance**: Perfect donor/acceptor balance (1:1 ratio)

### Genomic Distribution
- **Chromosomes Covered**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, X, Y
- **Gene Length Distribution**:
  - Mean: 70,902 bp
  - Median: 30,476 bp
  - Range: 446 bp - 1,743,270 bp
  - 25th percentile: 12,381 bp
  - 75th percentile: 75,815 bp

### Splice Site Density
- **Splice Density per kb**:
  - Mean: 6.28 sites/kb
  - Median: 2.16 sites/kb
  - Range: 0.012 - 147.2 sites/kb
  - 25th percentile: 0.81 sites/kb
  - 75th percentile: 7.20 sites/kb

## Feature Schema

### Data Dimensions
- **Feature Count**: 143 features per record
- **Record Structure**: Splice site predictions with comprehensive meta-features

### Feature Categories

#### 1. Splice Site Prediction Features (Core SpliceAI)
- `donor_score`, `acceptor_score`, `neither_score`
- `splice_type` (donor/acceptor/neither)
- `pred_type` (TP/TN/FP/FN)
- `score`, `splice_probability`

#### 2. Context and Signal Analysis
- `donor_context_diff_ratio`, `acceptor_context_diff_ratio`
- `donor_signal_strength`, `acceptor_signal_strength`
- `donor_peak_height_ratio`, `acceptor_peak_height_ratio`
- `context_asymmetry`, `context_max`, `context_neighbor_mean`

#### 3. Differential Analysis Features
- `donor_diff_m1`, `donor_diff_m2`, `donor_diff_p1`, `donor_diff_p2`
- `acceptor_diff_m1`, `acceptor_diff_m2`, `acceptor_diff_p1`, `acceptor_diff_p2`
- `donor_acceptor_diff`, `donor_acceptor_logodds`

#### 4. K-mer Features (3-mer Analysis)
- Complete 3-mer composition: `3mer_AAA` through `3mer_TTT` (64 features)
- Sequence complexity metrics: `gc_content`, `sequence_complexity`, `sequence_length`

#### 5. Genomic Context Features
- **Positional**: `chrom`, `position`, `strand`, `absolute_position`
- **Gene Structure**: `gene_start`, `gene_end`, `gene_length`, `gene_type`
- **Transcript Features**: `transcript_id`, `transcript_count`, `transcript_length`
- **Exon Analysis**: `num_exons`, `avg_exon_length`, `median_exon_length`
- **Distance Metrics**: `distance_to_start`, `distance_to_end`

#### 6. Meta-Analysis Features
- **Peak Detection**: `donor_is_local_peak`, `acceptor_is_local_peak`
- **Statistical Measures**: `probability_entropy`, `score_difference_ratio`
- **Quality Indicators**: `has_tx_info`, `has_gene_info`, `missing_transcript_feats`

## Data Quality and Distribution

### Splice Site Distribution (Sample Batch)
- **Neither Sites**: 253,402 records (92.1%)
- **Donor Sites**: 10,860 records (3.9%)
- **Acceptor Sites**: 10,814 records (3.9%)

### Prediction Performance Distribution (Sample Batch)
- **True Negatives (TN)**: 251,632 records (91.5%)
- **True Positives (TP)**: 16,558 records (6.0%)
- **False Negatives (FN)**: 5,116 records (1.9%)
- **False Positives (FP)**: 1,770 records (0.6%)

## Generation Methodology

### Workflow Pipeline
1. **Gene Selection**: Random sampling of 7,000 protein-coding genes
2. **Artifact Generation**: SpliceAI prediction workflow for missing genes
3. **Feature Engineering**: Comprehensive meta-feature extraction
4. **Batch Processing**: Chunked processing with 250 genes per batch, 20,000 rows per batch
5. **Quality Control**: Validation and trimming of intermediate results

### Processing Parameters
- **Batch Size**: 250 genes per batch
- **Batch Rows**: 20,000 rows per batch
- **K-mer Size**: 3-mers
- **Chunked Processing**: Enabled for memory efficiency
- **Chunk Size**: 5,000 records

### SpliceAI Configuration
- **Models**: 5 SpliceAI models loaded
- **Threshold**: 0.5
- **Consensus Window**: 2
- **Error Window**: 500

## Use Cases and Applications

### Primary Applications
1. **Meta-Model Training**: Training splice site prediction meta-models
2. **Variant Impact Assessment**: Evaluating splicing effects of genetic variants
3. **Disease-Specific Analysis**: Understanding disease-associated splicing patterns
4. **Alternative Splicing Research**: Studying complex splicing mechanisms

### Recommended Analysis Approaches
- **Cross-Validation**: 5-fold gene-level cross-validation
- **Feature Selection**: Utilize comprehensive feature set for model training
- **Imbalanced Learning**: Account for splice site class imbalance
- **Multi-Task Learning**: Leverage donor/acceptor prediction tasks

## Data Access and Usage

### File Format
- **Primary Data**: Apache Parquet format for efficient storage and access
- **Metadata**: CSV format for gene manifest
- **Compression**: Optimized for analytical workloads

### Loading Recommendations
```python
import pandas as pd

# Load gene manifest
manifest = pd.read_csv('train_pc_7000_3mers_opt/master/gene_manifest.csv')

# Load training batches
batch_files = [f'train_pc_7000_3mers_opt/master/batch_{i:05d}.parquet' 
               for i in range(1, 29)]
training_data = pd.concat([pd.read_parquet(f) for f in batch_files])
```

## Quality Assurance

### Validation Checks
- ✅ Gene count verification: 6,708 genes (target: ~7,000)
- ✅ Feature completeness: 143 features per record
- ✅ Data type consistency: Appropriate dtypes for all features
- ✅ Batch file integrity: 28 complete batch files
- ✅ Genomic coverage: All target chromosomes represented

### Known Limitations
- **Gene Count**: Final count (6,708) slightly below target (7,000) due to processing constraints
- **Log Truncation**: Generation log was truncated but dataset completed successfully
- **Memory Requirements**: Large dataset requires sufficient RAM for full loading

## Maintenance and Updates

### Version Information
- **Version**: 1.0
- **Last Updated**: 2025-08-23
- **Generation Log**: `logs/train_pc_7000_3mers_opt.log`

### Future Enhancements
- Expansion to include non-coding genes
- Integration of additional variant databases
- Enhanced feature engineering for disease-specific patterns
- Cross-species comparative analysis capabilities

## Contact and Support

For questions about this dataset or requests for additional analysis, please refer to the meta-spliceai documentation or contact the development team.

---

**Generated**: 2025-08-23  
**Dataset Path**: `/home/bchiu/work/meta-spliceai/train_pc_7000_3mers_opt/`  
**Documentation Path**: `meta_spliceai/splice_engine/case_studies/data_sources/datasets/`
