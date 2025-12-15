# Train PC 1000 3-mers Dataset Documentation

## Overview

**Dataset Name**: `train_pc_1000_3mers`  
**Creation Date**: [Date of creation]  
**Purpose**: Meta-model training dataset for splice site prediction  
**Size**: 1,329,518 rows × 148 columns  
**Format**: Parquet files (11 partitions)  

## Dataset Characteristics

### **Gene Selection Strategy**
- **Primary Selection**: Top 1,000 genes by error count (policy: `error_total`)
- **Additional Genes**: 2 custom genes (UNC13A, STMN2)
- **Total Unique Genes**: 1,002
- **Transcript Coverage**: 17,181 unique transcripts

### **Data Distribution**

#### **Splice Site Types**
| Type | Count | Percentage |
|------|-------|------------|
| neither | 1,203,558 | 90.5% |
| donor | 63,242 | 4.8% |
| acceptor | 62,718 | 4.7% |

#### **Prediction Outcomes**
| Outcome | Count | Percentage |
|---------|-------|------------|
| TN (True Negative) | 1,187,386 | 89.3% |
| TP (True Positive) | 67,680 | 5.1% |
| FN (False Negative) | 58,280 | 4.4% |
| FP (False Positive) | 16,172 | 1.2% |

### **Feature Set**

#### **K-mer Features**
- **K-mer Size**: 3-mers (trinucleotides)
- **Feature Count**: 64 standard 3-mers + special cases with N's
- **Purpose**: Capture local sequence motifs around splice sites

#### **Biological Features** (84 features)
- **Gene-level**: gene_type, gene_biotype, n_splice_sites
- **Transcript-level**: tx_start, tx_end, strand
- **Position-level**: position, distance_to_splice_site
- **Structural**: donor_is_local_peak, acceptor_is_local_peak
- **Performance**: Various error metrics and prediction scores

## Dataset Structure

### **File Organization**
```
train_pc_1000_3mers/
├── master/
│   ├── batch_00001.parquet
│   ├── batch_00002.parquet
│   ├── ...
│   ├── batch_00011.parquet
│   └── gene_manifest.csv
└── [temporary batch files]
```

### **Batch Information**
| Batch | Genes | Rows | Purpose |
|-------|-------|------|---------|
| 1-10 | 100 each | ~120K each | Main processing batches |
| 11 | 2 | 382 | Final batch (additional genes) |

### **Gene Manifest**
- **Total Genes**: 1,002 unique genes
- **Named Genes**: 861 (86.0%)
- **Unnamed Genes**: 141 (14.0%)
- **File**: `master/gene_manifest.csv`

## Data Quality Metrics

### **Completeness**
- ✅ All 1,002 target genes present
- ✅ Gene count validation passed
- ✅ Complete feature set (148 columns)
- ✅ Proper class distribution for splice site prediction

### **Balance**
- **Class Imbalance**: Expected for splice site prediction
- **TN Downsampling**: Applied to reduce computational load
- **Hard Negatives**: Preserved during downsampling

### **Coverage**
- **Gene Types**: Primarily protein-coding genes
- **Error Distribution**: Top genes by prediction error
- **Transcript Diversity**: 17.1 transcripts per gene average

## Usage Guidelines

### **Training Applications**
- **Meta-model Training**: Primary use case
- **Cross-validation**: Gene-wise and chromosome-wise CV supported
- **Feature Analysis**: Rich feature set for model interpretation

### **Performance Considerations**
- **Memory Usage**: 11 partitions for efficient loading
- **Streaming**: Supports memory-efficient data loading
- **Scalability**: Designed for incremental processing

### **Validation**
- **Gene Count**: Verify 1,002 unique genes
- **Class Balance**: Check TN/TP/FN/FP distribution
- **Feature Completeness**: Ensure all 148 columns present

## Technical Details

### **Creation Process**
1. **Gene Selection**: Error-based policy for top 1,000 genes
2. **Batch Processing**: 11 batches of 100 genes each (final: 2 genes)
3. **Feature Extraction**: 3-mer k-mers + biological features
4. **Enrichment**: Distance features and structural annotations
5. **Downsampling**: TN reduction for class balance
6. **Assembly**: Master dataset with 11 partitions

### **Data Sources**
- **SpliceAI Predictions**: Base prediction data
- **Ensembl Annotations**: Gene and transcript information
- **Error Analysis**: Performance metrics for gene selection
- **Custom Genes**: UNC13A, STMN2 (specific research targets)

### **Quality Assurance**
- **Schema Validation**: Consistent column structure across batches
- **Gene Coverage**: Complete coverage of selected genes
- **Feature Integrity**: All biological features properly populated
- **Class Distribution**: Appropriate for splice site prediction task

## Related Datasets

- **Gene Manifest**: `master/gene_manifest.csv`
- **Source Artifacts**: SpliceAI prediction outputs
- **Reference Data**: Ensembl gene annotations

## Notes

- Dataset optimized for 3-mer features (vs 6-mer alternatives)
- Designed for protein-coding gene focus (PC in name)
- Supports both gene-wise and chromosome-wise cross-validation
- Includes both error-prone genes and specific research targets 