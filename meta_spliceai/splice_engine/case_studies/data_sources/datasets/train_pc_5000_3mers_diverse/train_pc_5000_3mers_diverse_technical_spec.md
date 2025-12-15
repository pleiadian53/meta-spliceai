# Technical Specification: train_pc_5000_3mers_diverse

## Dataset Schema Overview

**Total Features**: 143-148 (varies by batch)  
**Data Format**: Apache Parquet with Snappy compression  
**Record Structure**: Position-centered splice site predictions  
**Primary Key**: Combination of `gene_id`, `transcript_id`, `position`  

## Feature Categories and Schema

### 1. SpliceAI Prediction Features (45 features)

#### Donor Site Features (15 features)
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `donor_score` | float64 | [0.0, 1.0] | SpliceAI donor site probability |
| `donor_signal_strength` | float64 | [0.0, 1.0] | Signal strength at donor position |
| `donor_peak_height_ratio` | float64 | [0.0, ∞] | Peak height relative to neighbors |
| `donor_is_local_peak` | int8 | {0, 1} | Boolean: is local maximum |
| `donor_context_diff_ratio` | float64 | [-∞, ∞] | Context difference ratio |
| `donor_weighted_context` | float64 | [0.0, 1.0] | Weighted context score |
| `donor_surge_ratio` | float64 | [0.0, ∞] | Signal surge ratio |
| `donor_second_derivative` | float64 | [-∞, ∞] | Second derivative of signal |
| `donor_diff_m1` | float64 | [-1.0, 1.0] | Difference from position -1 |
| `donor_diff_m2` | float64 | [-1.0, 1.0] | Difference from position -2 |
| `donor_diff_p1` | float64 | [-1.0, 1.0] | Difference from position +1 |
| `donor_diff_p2` | float64 | [-1.0, 1.0] | Difference from position +2 |
| `donor_acceptor_diff` | float64 | [-1.0, 1.0] | Donor minus acceptor score |
| `donor_acceptor_logodds` | float64 | [-∞, ∞] | Log odds ratio donor/acceptor |
| `donor_acceptor_peak_ratio` | float64 | [0.0, ∞] | Peak ratio donor/acceptor |

#### Acceptor Site Features (15 features)
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `acceptor_score` | float64 | [0.0, 1.0] | SpliceAI acceptor site probability |
| `acceptor_signal_strength` | float64 | [0.0, 1.0] | Signal strength at acceptor position |
| `acceptor_peak_height_ratio` | float64 | [0.0, ∞] | Peak height relative to neighbors |
| `acceptor_is_local_peak` | int8 | {0, 1} | Boolean: is local maximum |
| `acceptor_context_diff_ratio` | float64 | [-∞, ∞] | Context difference ratio |
| `acceptor_weighted_context` | float64 | [0.0, 1.0] | Weighted context score |
| `acceptor_surge_ratio` | float64 | [0.0, ∞] | Signal surge ratio |
| `acceptor_second_derivative` | float64 | [-∞, ∞] | Second derivative of signal |
| `acceptor_diff_m1` | float64 | [-1.0, 1.0] | Difference from position -1 |
| `acceptor_diff_m2` | float64 | [-1.0, 1.0] | Difference from position -2 |
| `acceptor_diff_p1` | float64 | [-1.0, 1.0] | Difference from position +1 |
| `acceptor_diff_p2` | float64 | [-1.0, 1.0] | Difference from position +2 |

#### Context and Comparative Features (15 features)
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `context_asymmetry` | float64 | [-1.0, 1.0] | Asymmetry in context scores |
| `context_max` | float64 | [0.0, 1.0] | Maximum context score |
| `context_neighbor_mean` | float64 | [0.0, 1.0] | Mean of neighboring context scores |
| `context_score_m1` | float64 | [0.0, 1.0] | Context score at position -1 |
| `context_score_m2` | float64 | [0.0, 1.0] | Context score at position -2 |
| `context_score_p1` | float64 | [0.0, 1.0] | Context score at position +1 |
| `context_score_p2` | float64 | [0.0, 1.0] | Context score at position +2 |
| `neither_score` | float64 | [0.0, 1.0] | SpliceAI neither (non-splice) score |
| `score` | float64 | [0.0, 1.0] | Combined splice score |
| `score_difference_ratio` | float64 | [-∞, ∞] | Ratio of score differences |
| `signal_strength_ratio` | float64 | [0.0, ∞] | Signal strength ratio |
| `splice_neither_diff` | float64 | [-1.0, 1.0] | Splice minus neither score |
| `splice_neither_logodds` | float64 | [-∞, ∞] | Log odds ratio splice/neither |
| `splice_probability` | float64 | [0.0, 1.0] | Combined splice probability |
| `probability_entropy` | float64 | [0.0, log(3)] | Entropy of probability distribution |

### 2. 3-mer K-mer Features (64 features)

Complete trinucleotide composition analysis covering all possible 3-mers:

| Feature Pattern | Type | Range | Count | Description |
|----------------|------|-------|-------|-------------|
| `3mer_AAA` to `3mer_TTT` | float64 | [0.0, ∞] | 64 | Raw count of each 3-mer in sequence window |

**Complete 3-mer List**:
```
AAA, AAC, AAG, AAT, ACA, ACC, ACG, ACT,
AGA, AGC, AGG, AGT, ATA, ATC, ATG, ATT,
CAA, CAC, CAG, CAT, CCA, CCC, CCG, CCT,
CGA, CGC, CGG, CGT, CTA, CTC, CTG, CTT,
GAA, GAC, GAG, GAT, GCA, GCC, GCG, GCT,
GGA, GGC, GGG, GGT, GTA, GTC, GTG, GTT,
TAA, TAC, TAG, TAT, TCA, TCC, TCG, TCT,
TGA, TGC, TGG, TGT, TTA, TTC, TTG, TTT
```

### 3. Genomic Context Features (22 features)

#### Position and Coordinate Features (8 features)
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `position` | int64 | [1, ∞] | Position within transcript |
| `absolute_position` | int64 | [1, ∞] | Absolute genomic position |
| `window_start` | int64 | [1, ∞] | Analysis window start position |
| `window_end` | int64 | [1, ∞] | Analysis window end position |
| `predicted_position` | int64 | [1, ∞] | SpliceAI predicted position |
| `true_position` | int64 | [1, ∞] | True splice site position |
| `distance_to_start` | int64 | [0, ∞] | Distance to transcript start |
| `distance_to_end` | int64 | [0, ∞] | Distance to transcript end |

#### Gene Structure Features (14 features)
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `gene_start` | int64 | [1, ∞] | Gene start coordinate |
| `gene_end` | int64 | [1, ∞] | Gene end coordinate |
| `gene_length` | int64 | [1, ∞] | Total gene length in bp |
| `tx_start` | int64 | [1, ∞] | Transcript start coordinate |
| `tx_end` | int64 | [1, ∞] | Transcript end coordinate |
| `transcript_length` | int64 | [1, ∞] | Transcript length in bp |
| `num_exons` | int64 | [1, ∞] | Number of exons in transcript |
| `avg_exon_length` | float64 | [1.0, ∞] | Average exon length |
| `median_exon_length` | float64 | [1.0, ∞] | Median exon length |
| `total_exon_length` | int64 | [1, ∞] | Total exonic sequence length |
| `total_intron_length` | int64 | [0, ∞] | Total intronic sequence length |
| `num_overlaps` | int64 | [0, ∞] | Number of overlapping features |
| `transcript_count` | int64 | [1, ∞] | Number of transcripts for gene |
| `n_splice_sites` | int64 | [0, ∞] | Total splice sites in gene |

### 4. Sequence Composition Features (3 features)

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `gc_content` | float64 | [0.0, 1.0] | GC content in analysis window |
| `sequence_length` | int64 | [1, ∞] | Length of analyzed sequence |
| `sequence_complexity` | float64 | [0.0, 1.0] | Sequence complexity measure |

### 5. Annotation and Metadata Features (11 features)

#### Identifiers (4 features)
| Feature | Type | Format | Description |
|---------|------|--------|-------------|
| `gene_id` | object | ENSG\d{11} | Ensembl gene identifier |
| `transcript_id` | object | ENST\d{11} | Ensembl transcript identifier |
| `chrom` | object | [1-22,X,Y] | Chromosome identifier |
| `strand` | object | {+, -} | Genomic strand |

#### Classification Features (3 features)
| Feature | Type | Values | Description |
|---------|------|--------|-------------|
| `gene_type` | object | See Gene Types | Ensembl gene biotype |
| `pred_type` | object | {donor, acceptor, neither} | Predicted splice type |
| `splice_type` | object | {donor, acceptor, neither} | True splice type |

#### Quality Control Features (4 features)
| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `has_tx_info` | int64 | {0, 1} | Boolean: transcript info available |
| `has_gene_info` | int64 | {0, 1} | Boolean: gene info available |
| `missing_transcript_feats` | int64 | [0, ∞] | Count of missing transcript features |
| `relative_donor_probability` | float64 | [0.0, 1.0] | Relative donor probability |

## Gene Type Enumeration

The dataset includes the following gene types with their characteristics:

| Gene Type | Count (Est.) | Description | Splice Characteristics |
|-----------|--------------|-------------|----------------------|
| `protein_coding` | ~2,000 | Standard protein-coding genes | Canonical splice sites |
| `lncRNA` | ~400 | Long non-coding RNAs | Regulatory splice patterns |
| `processed_pseudogene` | ~300 | Processed pseudogenes | Degraded splice signals |
| `transcribed_unprocessed_pseudogene` | ~150 | Active unprocessed pseudogenes | Intermediate splice patterns |
| `unprocessed_pseudogene` | ~100 | Unprocessed pseudogenes | Ancestral splice remnants |
| `transcribed_processed_pseudogene` | ~80 | Active processed pseudogenes | Modified splice patterns |
| `IG_V_pseudogene` | ~50 | Immunoglobulin V pseudogenes | Immune-related variants |
| `IG_V_gene` | ~20 | Immunoglobulin V genes | Specialized immune splicing |
| `transcribed_unitary_pseudogene` | ~10 | Unitary pseudogenes | Unique splice characteristics |
| `artifact` | ~1 | Annotation artifacts | Quality control examples |

## Data Quality Specifications

### Completeness Requirements
- **No Missing Values**: All core features must be present
- **Gene Annotation**: All records must have valid gene_id and gene_type
- **Coordinate Integrity**: All positions must be within valid genomic ranges
- **Feature Consistency**: Core feature set must be consistent across batches

### Value Range Validation
- **Probabilities**: All probability scores in [0.0, 1.0]
- **K-mer Counts**: All 3-mer counts ≥ 0 (raw counts, not normalized)
- **Coordinates**: All genomic coordinates > 0
- **Lengths**: All length measurements > 0
- **Ratios**: Finite values only (no NaN or infinite values)

### Batch Consistency
- **Schema Alignment**: Core features present in all batches
- **Data Types**: Consistent data types across batches
- **Value Ranges**: Consistent value distributions
- **Gene Uniqueness**: No duplicate genes across batches

## Loading and Processing Guidelines

### Memory Management
```python
# Efficient batch loading
def load_batch_efficient(batch_num, columns=None):
    """Load specific batch with optional column selection"""
    file_path = f'train_pc_5000_3mers_diverse/master/batch_{batch_num:05d}.parquet'
    return pd.read_parquet(file_path, columns=columns)

# Memory usage per batch: ~31MB
# Full dataset memory usage: ~620MB
```

### Feature Selection Strategies
```python
# Core features for basic analysis
core_features = [
    'gene_id', 'gene_type', 'chrom', 'position',
    'donor_score', 'acceptor_score', 'neither_score'
]

# K-mer features only
kmer_features = [col for col in df.columns if col.startswith('3mer_')]

# SpliceAI features only
spliceai_features = [col for col in df.columns 
                    if any(x in col.lower() for x in ['donor', 'acceptor', 'score', 'context'])]
```

### Data Validation
```python
def validate_batch(df):
    """Validate batch data quality"""
    checks = {
        'no_missing_core': df[core_features].isnull().sum().sum() == 0,
        'valid_probabilities': ((df['donor_score'] >= 0) & (df['donor_score'] <= 1)).all(),
        'valid_chromosomes': df['chrom'].isin([str(i) for i in range(1, 23)] + ['X', 'Y']).all(),
        'positive_positions': (df['position'] > 0).all()
    }
    return all(checks.values()), checks
```

## File Format Specifications

### Parquet Configuration
- **Compression**: Snappy (optimal balance of speed and size)
- **Row Group Size**: ~50,000 rows per row group
- **Column Encoding**: Dictionary encoding for categorical columns
- **Metadata**: Embedded schema and statistics

### Batch Organization
- **Batch Size**: Variable (21K-34K records per batch)
- **Naming Convention**: `batch_NNNNN.parquet` (zero-padded 5 digits)
- **Gene Distribution**: Approximately balanced across batches
- **Processing Order**: Sequential batch processing recommended

---

**Technical Specification Version**: 1.0  
**Last Updated**: 2025-01-27  
**Schema Validation**: Passed  
**Compatibility**: Pandas ≥1.3.0, PyArrow ≥5.0.0
