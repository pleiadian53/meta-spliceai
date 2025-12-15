# Technical Specification: train_pc_7000_3mers_opt Dataset

## Data Schema Definition

### Gene Manifest Schema (`gene_manifest.csv`)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `global_index` | int | Sequential index across all genes | 0, 1, 2, ... |
| `gene_id` | string | Ensembl gene identifier | ENSG00000139719 |
| `gene_name` | string | HGNC gene symbol | VPS33A |
| `gene_type` | string | Gene biotype (all protein_coding) | protein_coding |
| `chrom` | string | Chromosome identifier | 12, X, Y |
| `strand` | string | Genomic strand | +, - |
| `gene_length` | int | Gene length in base pairs | 36939 |
| `start` | int | Gene start position (0-based) | 122229564 |
| `end` | int | Gene end position (0-based) | 122266502 |
| `total_splice_sites` | int | Total splice sites in gene | 288 |
| `donor_sites` | int | Number of donor sites | 144 |
| `acceptor_sites` | int | Number of acceptor sites | 144 |
| `splice_density_per_kb` | float | Splice sites per kilobase | 7.796637699991878 |
| `file_index` | int | Batch file number | 1, 2, ..., 28 |
| `file_name` | string | Batch file name | batch_00001.parquet |

### Training Data Schema (`batch_*.parquet`)

#### Core Prediction Features
| Feature | Type | Range/Values | Description |
|---------|------|--------------|-------------|
| `splice_type` | string | donor/acceptor/neither | Ground truth splice site type |
| `pred_type` | string | TP/TN/FP/FN | Prediction classification |
| `score` | float64 | [0.0, 1.0] | Combined splice prediction score |
| `donor_score` | float64 | [0.0, 1.0] | SpliceAI donor probability |
| `acceptor_score` | float64 | [0.0, 1.0] | SpliceAI acceptor probability |
| `neither_score` | float64 | [0.0, 1.0] | SpliceAI neither probability |
| `splice_probability` | float64 | [0.0, 1.0] | Normalized splice probability |

#### Positional and Genomic Context
| Feature | Type | Description |
|---------|------|-------------|
| `chrom` | string | Chromosome identifier |
| `position` | int64 | Relative position within gene |
| `strand` | string | Genomic strand (+/-) |
| `absolute_position` | int64 | Absolute genomic position |
| `predicted_position` | int64 | SpliceAI predicted position |
| `true_position` | int64 | Ground truth position |
| `window_start` | int64 | Analysis window start |
| `window_end` | int64 | Analysis window end |

#### Signal Analysis Features
| Feature | Type | Description |
|---------|------|-------------|
| `donor_signal_strength` | float64 | Donor signal intensity |
| `acceptor_signal_strength` | float64 | Acceptor signal intensity |
| `signal_strength_ratio` | float64 | Ratio of donor/acceptor signals |
| `donor_peak_height_ratio` | float64 | Donor peak relative height |
| `acceptor_peak_height_ratio` | float64 | Acceptor peak relative height |
| `donor_is_local_peak` | bool | Whether donor is local maximum |
| `acceptor_is_local_peak` | bool | Whether acceptor is local maximum |

#### Context Differential Features
| Feature | Type | Description |
|---------|------|-------------|
| `donor_context_diff_ratio` | float64 | Donor context differential ratio |
| `acceptor_context_diff_ratio` | float64 | Acceptor context differential ratio |
| `donor_diff_m1` | float64 | Donor difference at position -1 |
| `donor_diff_m2` | float64 | Donor difference at position -2 |
| `donor_diff_p1` | float64 | Donor difference at position +1 |
| `donor_diff_p2` | float64 | Donor difference at position +2 |
| `acceptor_diff_m1` | float64 | Acceptor difference at position -1 |
| `acceptor_diff_m2` | float64 | Acceptor difference at position -2 |
| `acceptor_diff_p1` | float64 | Acceptor difference at position +1 |
| `acceptor_diff_p2` | float64 | Acceptor difference at position +2 |

#### Statistical and Comparative Features
| Feature | Type | Description |
|---------|------|-------------|
| `donor_acceptor_diff` | float64 | Difference between donor and acceptor scores |
| `donor_acceptor_logodds` | float64 | Log odds ratio of donor vs acceptor |
| `donor_acceptor_peak_ratio` | float64 | Ratio of peak heights |
| `splice_neither_diff` | float64 | Difference between splice and neither |
| `splice_neither_logodds` | float64 | Log odds of splice vs neither |
| `probability_entropy` | float64 | Entropy of probability distribution |
| `score_difference_ratio` | float64 | Ratio of score differences |

#### Sequence Composition Features (3-mer K-mers)
All 64 possible 3-mer combinations as float64 features:
- `3mer_AAA`, `3mer_AAC`, `3mer_AAG`, `3mer_AAT`
- `3mer_ACA`, `3mer_ACC`, `3mer_ACG`, `3mer_ACT`
- ... (complete set of 64 3-mers)
- `3mer_TTA`, `3mer_TTC`, `3mer_TTG`, `3mer_TTT`

#### Sequence Properties
| Feature | Type | Description |
|---------|------|-------------|
| `gc_content` | float64 | GC content of sequence window |
| `sequence_length` | int64 | Length of analyzed sequence |
| `sequence_complexity` | float64 | Sequence complexity measure |

#### Gene and Transcript Structure
| Feature | Type | Description |
|---------|------|-------------|
| `gene_id` | string | Ensembl gene identifier |
| `transcript_id` | string | Ensembl transcript identifier |
| `transcript_count` | int64 | Number of transcripts for gene |
| `transcript_length` | int64 | Length of transcript |
| `tx_start` | int64 | Transcript start position |
| `tx_end` | int64 | Transcript end position |
| `gene_start` | int64 | Gene start position |
| `gene_end` | int64 | Gene end position |
| `gene_type` | string | Gene biotype |
| `gene_length` | int64 | Total gene length |

#### Exon and Intron Analysis
| Feature | Type | Description |
|---------|------|-------------|
| `num_exons` | int64 | Number of exons in transcript |
| `avg_exon_length` | float64 | Average exon length |
| `median_exon_length` | float64 | Median exon length |
| `total_exon_length` | int64 | Total exonic sequence length |
| `total_intron_length` | int64 | Total intronic sequence length |
| `n_splice_sites` | int64 | Number of splice sites in gene |

#### Distance and Positional Metrics
| Feature | Type | Description |
|---------|------|-------------|
| `distance_to_start` | int64 | Distance to gene/transcript start |
| `distance_to_end` | int64 | Distance to gene/transcript end |
| `num_overlaps` | int64 | Number of overlapping features |

#### Quality and Completeness Indicators
| Feature | Type | Description |
|---------|------|-------------|
| `has_tx_info` | int32 | Whether transcript info is available (0/1) |
| `has_gene_info` | int32 | Whether gene info is available (0/1) |
| `missing_transcript_feats` | int8 | Count of missing transcript features |

## Data Types and Memory Usage

### Numeric Precision
- **float64**: High-precision features (scores, ratios, statistical measures)
- **int64**: Large integer values (positions, lengths)
- **int32**: Medium integer values (boolean flags)
- **int8**: Small integer values (counts, flags)
- **string**: Categorical and identifier fields

### Memory Optimization
- Categorical strings for repeated values (gene_id, transcript_id)
- Appropriate integer sizes for count data
- Float64 precision for scientific calculations

## File Format Specifications

### Parquet Configuration
- **Compression**: Snappy (default)
- **Row Group Size**: Optimized for analytical queries
- **Column Storage**: Efficient for feature-based access
- **Metadata**: Embedded schema information

### CSV Configuration (Gene Manifest)
- **Encoding**: UTF-8
- **Delimiter**: Comma (,)
- **Header**: First row contains column names
- **Quoting**: Minimal (only when necessary)

## Data Validation Rules

### Gene Manifest Validation
1. All `gene_id` values must be valid Ensembl identifiers
2. `gene_type` must be "protein_coding" for all records
3. `start` < `end` for all genes
4. `donor_sites` + `acceptor_sites` = `total_splice_sites`
5. `file_index` must correspond to existing batch files

### Training Data Validation
1. All probability scores must be in range [0.0, 1.0]
2. `donor_score` + `acceptor_score` + `neither_score` ≈ 1.0
3. Position values must be non-negative integers
4. K-mer features must sum to sequence length
5. Boolean flags must be 0 or 1

## Performance Characteristics

### Loading Performance
- **Single Batch**: ~275K records, ~50MB, loads in <1 second
- **Full Dataset**: ~7.7M records, ~595MB, loads in ~30 seconds
- **Memory Usage**: ~2-3GB RAM for full dataset in pandas DataFrame

### Query Performance
- **Column Access**: Optimized for feature-based filtering
- **Row Filtering**: Efficient for gene-based subsetting
- **Aggregation**: Fast for statistical analysis across batches

## Integration Guidelines

### Recommended Usage Patterns
```python
# Efficient batch processing
for i in range(1, 29):
    batch = pd.read_parquet(f'master/batch_{i:05d}.parquet')
    # Process batch
    
# Memory-efficient loading with specific columns
columns = ['splice_type', 'donor_score', 'acceptor_score', 'gene_id']
batch = pd.read_parquet('master/batch_00001.parquet', columns=columns)

# Gene-based filtering
manifest = pd.read_csv('master/gene_manifest.csv')
target_genes = manifest[manifest['chrom'] == '1']['gene_id'].tolist()
filtered_data = batch[batch['gene_id'].isin(target_genes)]
```

### Cross-Validation Considerations
- Use gene-level splits to avoid data leakage
- Maintain chromosome balance across folds
- Consider splice density when stratifying

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-23  
**Schema Version**: train_pc_7000_3mers_opt_v1.0
