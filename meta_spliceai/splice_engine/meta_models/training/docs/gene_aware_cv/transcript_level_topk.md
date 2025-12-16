# Transcript-Level Top-K Accuracy

## Overview

Transcript-level top-k accuracy is an extension of gene-level top-k accuracy that provides a more granular evaluation of model performance. While gene-level metrics group predictions by gene, transcript-level metrics group predictions by transcript, offering a more precise assessment of how well a model predicts splice sites within specific transcript isoforms.

## Implementation Details

### Core Components

The implementation consists of two main components:

1. **Position-to-Transcript Mapping**: A multi-step algorithm to map genomic positions to transcript IDs
2. **Transcript-Level Top-K Calculation**: Grouping and evaluating predictions by transcript instead of by gene

### Position-to-Transcript Mapping Algorithm

The algorithm maps genomic positions to transcript IDs through a three-step fallback process:

1. **Exact Position Matching**: First attempts to find exact matches between query positions and annotated splice sites
2. **Boundary-Based Mapping**: For positions without exact matches, checks if they fall within the boundaries (start/end) of any transcript
3. **Proximity-Based Mapping**: For remaining positions, finds the closest annotated splice site within a 50bp window

This approach maximizes the number of positions that can be successfully mapped to transcripts while maintaining biological relevance.

### Data Requirements

The implementation relies on three annotation files:

1. **splice_sites.tsv**: Contains information about splice site positions
   - Required columns: `chrom`, `position`, `site_type` (or `splice_type`), `transcript_id`
   - Each row represents a single splice site with its genomic location and transcript association

2. **transcript_features.tsv**: Contains information about transcript boundaries
   - Required columns: `chrom`, `start`, `end`, `transcript_id`, `gene_id`
   - Each row represents a transcript with its genomic coordinates

3. **gene_features.tsv** (optional): Contains information about gene boundaries
   - Used for optional gene-level aggregation of transcript metrics
   - Required columns: `chrom`, `start`, `end`, `gene_id`

### Metric Calculation

After mapping positions to transcripts, the implementation:

1. Groups predictions by transcript ID
2. For each transcript, calculates:
   - Top-k accuracy for donor sites
   - Top-k accuracy for acceptor sites
   - Combined top-k accuracy
3. Aggregates results across all transcripts
4. Calculates additional statistics:
   - Number of transcripts with donor/acceptor sites
   - Average number of donor/acceptor sites per transcript
   - Percentage of positions successfully mapped
   - Gene-level aggregation (if gene features are provided)

## Usage

### Command Line Arguments

In the gene-wise cross-validation pipeline (`run_gene_cv_sigmoid.py`), the following arguments enable transcript-level top-k accuracy:

```
--transcript-topk            Enable transcript-level top-k accuracy calculation
--splice-sites-path PATH     Path to splice site annotations file
--transcript-features-path PATH  Path to transcript features file
--gene-features-path PATH    Path to gene features file (optional)
--position-col NAME          Name of column with genomic positions (default: position)
--chrom-col NAME             Name of column with chromosome (default: chrom)
```

### Example Usage

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
  --transcript-topk \
  --splice-sites-path data/ensembl/splice_sites.tsv \
  --transcript-features-path data/ensembl/spliceai_analysis/transcript_features.tsv \
  --gene-features-path data/ensembl/spliceai_analysis/gene_features.tsv \
  [other arguments]
```

### Testing the Implementation

A dedicated test script `test_transcript_topk.py` is available to validate the implementation:

```bash
python test_transcript_topk.py \
  --data-path /path/to/your/data.parquet \
  --splice-sites-path data/ensembl/splice_sites.tsv \
  --transcript-features-path data/ensembl/spliceai_analysis/transcript_features.tsv \
  --gene-features-path data/ensembl/spliceai_analysis/gene_features.tsv
```

## Interpretation of Results

The transcript-level top-k accuracy report includes:

1. **Basic Top-K Metrics**:
   - Donor top-k: Accuracy for donor sites across all transcripts
   - Acceptor top-k: Accuracy for acceptor sites across all transcripts
   - Combined top-k: Weighted average of donor and acceptor accuracy

2. **Transcript Statistics**:
   - Number of transcripts with donor/acceptor sites
   - Average number of donor/acceptor sites per transcript
   - Percentage of positions successfully mapped to transcripts

3. **Gene-Level Aggregation** (if gene features are provided):
   - Number of genes with transcripts
   - Average number of transcripts per gene

## Code Structure

The implementation is organized into two main files:

1. `transcript_mapping.py`: Contains the core functionality for mapping positions to transcripts and calculating transcript-level metrics
2. `run_gene_cv_sigmoid.py`: Contains the integration with the gene-wise cross-validation pipeline

## Notes on Data Compatibility

- The implementation handles both `site_type` and `splice_type` column names in annotation files
- The "neither" class can be encoded as either "neither" or "0" in the training data
- If a position cannot be mapped to any transcript, it is labeled as "unknown" and excluded from transcript-level metrics

## Performance Optimization: Caching

The position-to-transcript mapping process can be time-consuming, especially with large datasets and annotation files. To address this, the implementation includes a caching mechanism:

1. **First Run**: When transcript mapping is performed for the first time on a dataset, the results are cached to disk
2. **Subsequent Runs**: For the same dataset and annotation files, the cached mapping is loaded, dramatically improving performance

### Cache Configuration

- **Location**: Cached files are stored in `meta_spliceai/splice_engine/meta_models/evaluation/cache/`
- **Naming**: Cache files follow the pattern `tx_map_[HASH].pkl` where the hash is derived from:
  - Input data fingerprint (sample of positions and chromosomes)
  - Annotation file modification timestamps
  - Dataset size

### Using Caching

Caching is enabled by default and controlled with the `use_cache` parameter:

```python
transcript_top_k_metrics = calculate_transcript_level_top_k(
    df=data,
    splice_sites_path=path_to_splice_sites,
    transcript_features_path=path_to_transcript_features,
    use_cache=True  # Set to False to disable caching
)
```

In the cross-validation pipeline, you can control caching with the `--no-transcript-cache` flag:

```bash
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
  --transcript-topk \
  --no-transcript-cache \
  [other arguments]
```

## Limitations and Future Work

- The current implementation prioritizes the first matching transcript when multiple transcripts contain a position
- Proximity-based mapping is limited to a 50bp window, which may need adjustment based on specific use cases
- Performance may degrade with very large annotation files; optimization may be needed for genome-wide analyses
