# Train PC 1000 Dataset Documentation

## Overview

**Dataset Name**: `train_pc_1000`  
**Creation Date**: [Date of creation]  
**Purpose**: Meta-model training dataset for splice site prediction (6-mer features)  
**Size**: 816,900 rows × 4,179 columns  
**Format**: Parquet files (4 partitions)  

## Dataset Characteristics

### **Gene Selection Strategy**
- **Primary Selection**: Top 1,000 genes by error count (policy: `error_total`)
- **Additional Genes**: None (pure error-based selection)
- **Total Unique Genes**: 1,000
- **Gene Coverage**: 967 named genes, 33 unnamed genes

### **Data Distribution**

#### **Splice Site Types**
| Type | Count | Percentage |
|------|-------|------------|
| neither | 748,537 | 91.6% |
| donor | 34,940 | 4.3% |
| acceptor | 33,423 | 4.1% |

#### **Prediction Outcomes**
| Outcome | Count | Percentage |
|---------|-------|------------|
| TN (True Negative) | 741,492 | 90.8% |
| TP (True Positive) | 46,500 | 5.7% |
| FN (False Negative) | 21,863 | 2.7% |
| FP (False Positive) | 7,045 | 0.9% |

### **Feature Set**

#### **K-mer Features**
- **K-mer Size**: 6-mers (hexanucleotides)
- **Feature Count**: 4,100 6-mer features
- **Purpose**: Capture longer sequence motifs around splice sites
- **Coverage**: All possible 6-mers (4^6 = 4,096 + special cases)

#### **Biological Features** (79 features)
- **Gene-level**: gene_type, gene_length, num_exons, n_splice_sites
- **Transcript-level**: transcript_length, tx_start, tx_end, strand
- **Position-level**: position, distance_to_start, distance_to_end
- **Structural**: donor_is_local_peak, acceptor_is_local_peak
- **Performance**: Various error metrics and prediction scores
- **Context**: Context scores, signal strength, peak ratios

## Dataset Structure

### **File Organization**
```
train_pc_1000/
├── master/
│   ├── batch_00001.parquet
│   ├── batch_00002.parquet
│   ├── batch_00003.parquet
│   ├── batch_00004.parquet
│   ├── gene_position_index.csv
│   └── gene_manifest.csv
└── [temporary batch files]
```

### **Batch Information**
| Batch | Genes | Rows | Purpose |
|-------|-------|------|---------|
| 1 | 250 | 193,901 | First quarter |
| 2 | 250 | 204,416 | Second quarter |
| 3 | 250 | 218,748 | Third quarter |
| 4 | 250 | 199,835 | Fourth quarter |

### **Gene Manifest**
- **Total Genes**: 1,000 unique genes
- **Named Genes**: 967 (96.7%)
- **Unnamed Genes**: 33 (3.3%)
- **File**: `gene_manifest.csv`

## Data Quality Metrics

### **Completeness**
- ✅ All 1,000 target genes present
- ✅ Complete feature set (4,179 columns)
- ✅ Proper class distribution for splice site prediction
- ✅ High gene name coverage (96.7%)

### **Balance**
- **Class Imbalance**: Expected for splice site prediction
- **TN Downsampling**: Applied to reduce computational load
- **Hard Negatives**: Preserved during downsampling

### **Coverage**
- **Gene Types**: Primarily protein-coding genes
- **Error Distribution**: Top genes by prediction error
- **Feature Completeness**: All 6-mer features populated

## Usage Guidelines

### **Training Applications**
- **Meta-model Training**: Primary use case
- **Cross-validation**: Gene-wise and chromosome-wise CV supported
- **Feature Analysis**: Rich 6-mer feature set for model interpretation
- **CV Development**: Used for testing gene-aware and chromosome-aware CV

### **Performance Considerations**
- **Memory Usage**: 4 partitions for efficient loading
- **Feature Density**: 4,100 6-mer features require more memory than 3-mers
- **Streaming**: Supports memory-efficient data loading
- **Scalability**: Designed for incremental processing

### **Validation**
- **Gene Count**: Verify 1,000 unique genes
- **Class Balance**: Check TN/TP/FN/FP distribution
- **Feature Completeness**: Ensure all 4,179 columns present
- **6-mer Coverage**: Validate 4,100 6-mer features

## Technical Details

### **Creation Process**
1. **Gene Selection**: Error-based policy for top 1,000 genes
2. **Batch Processing**: 4 batches of 250 genes each
3. **Feature Extraction**: 6-mer k-mers + biological features
4. **Enrichment**: Context features and structural annotations
5. **Downsampling**: TN reduction for class balance
6. **Assembly**: Master dataset with 4 partitions

### **Data Sources**
- **SpliceAI Predictions**: Base prediction data
- **Ensembl Annotations**: Gene and transcript information
- **Error Analysis**: Performance metrics for gene selection
- **6-mer Extraction**: Complete hexanucleotide motif coverage

### **Quality Assurance**
- **Schema Validation**: Consistent column structure across batches
- **Gene Coverage**: Complete coverage of selected genes
- **Feature Integrity**: All 6-mer features properly populated
- **Class Distribution**: Appropriate for splice site prediction task

## Comparison with 3-mer Dataset

### **Feature Comparison**
| Aspect | train_pc_1000 (6-mers) | train_pc_1000_3mers (3-mers) |
|--------|------------------------|-------------------------------|
| K-mer Size | 6-mers | 3-mers |
| K-mer Features | 4,100 | 64 |
| Total Features | 4,179 | 148 |
| Memory Usage | Higher | Lower |
| Sequence Context | Longer motifs | Shorter motifs |

### **Use Case Differences**
- **6-mers**: Better for capturing longer sequence patterns
- **3-mers**: More memory-efficient, faster processing
- **CV Testing**: 6-mer dataset used for CV development
- **Production**: 3-mer dataset optimized for efficiency

## Related Datasets

- **Gene Manifest**: `gene_manifest.csv`
- **Source Artifacts**: SpliceAI prediction outputs
- **Reference Data**: Ensembl gene annotations
- **Comparison Dataset**: `train_pc_1000_3mers` (3-mer version)

## Notes

- Dataset optimized for 6-mer features (vs 3-mer alternatives)
- Designed for protein-coding gene focus (PC in name)
- Used for testing gene-aware and chromosome-aware cross-validation
- Higher memory requirements due to 4,100 6-mer features
- Excellent gene name coverage (96.7% named genes)
- Balanced batch distribution (250 genes per batch) 