# train_regulatory_10k_kmers - Technical Specifications

## Dataset Schema

### Core Identification Columns
- **gene_id**: STRING - Ensembl gene identifier (e.g., "ENSG00000000003")
- **chr**: STRING - Chromosome identifier (1-22, X, Y)
- **position**: INTEGER - 1-based genomic coordinate
- **strand**: STRING - Gene strand orientation ("+", "-")

### SpliceAI Base Predictions
- **delta_score_donor**: FLOAT64 - Raw donor splice site score (0.0-1.0)
- **delta_score_acceptor**: FLOAT64 - Raw acceptor splice site score (0.0-1.0)
- **delta_score_donor_alt**: FLOAT64 - Alternative donor score
- **delta_score_acceptor_alt**: FLOAT64 - Alternative acceptor score

### Derived Probability Features
- **prob_donor**: FLOAT64 - Sigmoid-transformed donor probability
- **prob_acceptor**: FLOAT64 - Sigmoid-transformed acceptor probability
- **prob_donor_alt**: FLOAT64 - Alternative donor probability
- **prob_acceptor_alt**: FLOAT64 - Alternative acceptor probability

### Contextual Score Features
- **contextual_donor**: FLOAT64 - Context-aware donor score
- **contextual_acceptor**: FLOAT64 - Context-aware acceptor score
- **max_contextual**: FLOAT64 - Maximum contextual score
- **contextual_ratio**: FLOAT64 - Ratio of contextual scores

### Ground Truth Labels
- **is_donor**: BOOLEAN - True donor splice site annotation
- **is_acceptor**: BOOLEAN - True acceptor splice site annotation
- **is_canonical**: BOOLEAN - Canonical splice site indicator
- **is_cryptic**: BOOLEAN - Cryptic splice site indicator

### Sequence Features (3-mers)
64 columns in format `3mer_XXX` where XXX is a 3-nucleotide sequence:
- **3mer_AAA** through **3mer_TTT**: INTEGER - Count of 3-mer occurrences
- Valid nucleotides: A, C, G, T only (no ambiguous bases)
- Range: 0-50 (typical window-based counting)

### Sequence Features (5-mers)
1024 columns in format `5mer_XXXXX` where XXXXX is a 5-nucleotide sequence:
- **5mer_AAAAA** through **5mer_TTTTT**: INTEGER - Count of 5-mer occurrences
- Valid nucleotides: A, C, G, T only (no ambiguous bases)
- Range: 0-20 (typical window-based counting)

### Genomic Features
- **gene_length**: INTEGER - Gene length in base pairs
- **exon_count**: INTEGER - Number of exons in gene
- **splice_site_density**: FLOAT64 - Splice sites per kb
- **gc_content**: FLOAT64 - GC content percentage (0.0-1.0)

### Quality Control Features
- **sequence_quality**: FLOAT64 - Sequence quality score
- **alignment_score**: FLOAT64 - Reference alignment confidence
- **coverage_depth**: INTEGER - Sequencing coverage depth

## Data Types and Constraints

### Column Count
- **Total Features**: 1,167 columns
- **Core Features**: 75 (non-kmer features)
- **3-mer Features**: 64 columns
- **5-mer Features**: 1,024 columns
- **Metadata**: 4 columns

### Data Type Distribution
```
STRING: 3 columns (gene_id, chr, strand)
INTEGER: 1,092 columns (position, counts, genomic features)
FLOAT64: 71 columns (scores, probabilities, ratios)
BOOLEAN: 4 columns (ground truth labels)
```

### Value Constraints
- **Probabilities**: [0.0, 1.0] range
- **Scores**: [-1.0, 1.0] typical range
- **K-mer counts**: [0, 50] for 3-mers, [0, 20] for 5-mers
- **Chromosomes**: {1,2,...,22,X,Y}
- **Strand**: {"+", "-"}

## File Format Specifications

### Storage Format
- **Format**: Apache Parquet
- **Compression**: Snappy
- **Partitioning**: Batch-based (20 files)
- **File Pattern**: `batch_NNNNN.parquet`

### Batch Distribution
```
Total Records: 3,737,000
Records per Batch: ~186,850
Batch Files: 20
Size per Batch: ~95 MB
Total Dataset Size: 1.91 GB
```

### Schema Validation Rules
1. **No NULL values** in core identification columns
2. **Consistent data types** across all batch files
3. **Valid chromosome identifiers** only
4. **K-mer sequences contain only ACGT**
5. **Probability values within [0,1] bounds**

## Performance Characteristics

### Memory Requirements
- **Loading Full Dataset**: ~8 GB RAM (4x file size)
- **Batch Processing**: ~500 MB RAM per batch
- **Feature Matrix**: Dense representation recommended
- **Streaming**: Supported via Polars lazy evaluation

### Processing Performance
- **Full Dataset Load**: ~45 seconds (SSD)
- **Schema Validation**: ~30 seconds
- **Feature Extraction**: ~2 minutes
- **Cross-validation Split**: ~15 seconds

### Recommended Hardware
- **RAM**: Minimum 16 GB, Recommended 32 GB
- **Storage**: SSD recommended for I/O performance
- **CPU**: Multi-core beneficial for parallel processing
- **GPU**: Optional, useful for large-scale training

## Quality Assurance

### Data Quality Metrics
- **Completeness**: 100% (no missing values in core features)
- **Consistency**: Validated across all 20 batch files
- **Accuracy**: Ground truth from Ensembl annotations
- **Freshness**: Based on Ensembl release 108

### Validation Checks
1. **Schema Consistency**: All batches have identical column schemas
2. **K-mer Validity**: No ambiguous nucleotides (N, R, Y, etc.)
3. **Range Validation**: All numerical values within expected bounds
4. **Cross-Reference**: Gene IDs validated against Ensembl database
5. **Duplicate Detection**: No duplicate (gene_id, position) pairs

### Known Limitations
- **Chromosome Coverage**: Autosomes + X/Y only (no mitochondrial)
- **Gene Types**: Limited to protein_coding and lncRNA
- **Assembly Version**: GRCh38/hg38 only
- **Splice Site Types**: Canonical and cryptic only

## Integration Specifications

### Compatibility
- **Polars**: Primary data manipulation library
- **Pandas**: Supported via conversion
- **Scikit-learn**: Direct feature matrix compatibility
- **XGBoost**: Native Parquet support
- **PyTorch**: Via custom DataLoader

### API Integration
```python
# Standard loading pattern
import polars as pl
from meta_spliceai.splice_engine.meta_models.training import datasets

# Load full dataset
df = datasets.load_dataset("train_regulatory_10k_kmers/master")

# Load specific batches
df = datasets.load_dataset_batches(["batch_00001.parquet", "batch_00002.parquet"])

# Schema validation
from meta_spliceai.splice_engine.meta_models.builder.validate_dataset_schema import validate_schema
validate_schema("train_regulatory_10k_kmers/master", fix=True)
```

### Cross-Validation Compatibility
- **Gene-aware CV**: Genes split across folds
- **Chromosome-aware CV**: Chromosomes held out
- **Stratified CV**: Balanced by splice site types
- **Time-series CV**: Not applicable (static annotations)

## Maintenance and Updates

### Version Control
- **Dataset Version**: 1.0.0
- **Schema Version**: 2.1.0 (enhanced manifest format)
- **Last Updated**: 2025-01-26
- **Update Frequency**: Quarterly (following Ensembl releases)

### Backup and Recovery
- **Primary Storage**: Project directory
- **Backup Location**: Not specified
- **Recovery Time**: <1 hour (re-generation from source)
- **Integrity Checks**: SHA-256 checksums available

### Monitoring
- **Schema Drift**: Automated validation on load
- **Performance Degradation**: Tracked via training metrics
- **Data Quality**: Periodic validation reports
- **Usage Analytics**: Training frequency and success rates

## Related Documentation

### Dataset Family
- **train_pc_5000_3mers_diverse**: 5K protein-coding genes, 3-mers only
- **train_pc_7000_3mers_opt**: 7K protein-coding genes, optimized selection
- **train_regulatory_10k_kmers**: 10K mixed genes, 3-mers + 5-mers (this dataset)

### Technical References
- **Incremental Builder**: `meta_models/builder/docs/INCREMENTAL_BUILDER_CORE.md`
- **Training Workflows**: `meta_models/training/docs/COMPLETE_META_MODEL_WORKFLOW.md`
- **Schema Validation**: `meta_models/builder/validate_dataset_schema.py`
- **Utility Scripts**: `meta_models/training/docs/UTILITY_SCRIPTS_REFERENCE.md`

### External Dependencies
- **Ensembl Annotations**: Release 108
- **SpliceAI Model**: v1.3.1
- **Reference Genome**: GRCh38/hg38
- **Gene Features**: Enhanced manifest format

---

**Note**: This dataset represents a significant scale-up from previous training sets, incorporating both protein-coding and long non-coding RNA genes with multi-scale k-mer features. It is specifically designed for advanced meta-learning applications in regulatory variant analysis and alternative splicing pattern detection.


