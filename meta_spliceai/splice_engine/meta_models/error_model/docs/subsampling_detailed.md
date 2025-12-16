# Detailed Subsampling Logic Documentation

## Overview

The subsampling logic in the error model workflow is designed to maintain genomic integrity while managing dataset size and ensuring balanced class distribution. This document provides an in-depth explanation of the implementation.

## Core Implementation

### Consolidation and Sampling Module

The subsampling logic is implemented in `consolidate_analysis_data.py`, which processes chunked analysis sequence files with early filtering and gene-level sampling. Here's the detailed flow:

```python
def sample_by_genes(df, gene_col, max_samples, seed=42):
    """
    Core subsampling algorithm that maintains gene boundaries.
    
    Key Points:
    1. Groups all rows by gene_name or gene_id (fallback)
    2. Calculates total rows per gene
    3. Randomly selects complete genes
    4. Allows 10% overage for gene boundary accommodation
    5. Preserves genomic structure by keeping all rows of selected genes
    """
```

### Step-by-Step Process

#### 1. Early Chunk-Level Filtering

```python
# Extract chromosome from filename for early filtering
def extract_chromosome_from_filename(filepath):
    import re
    pattern = r'analysis_sequences_([^_]+)_chunk_'
    match = re.search(pattern, filepath.name)
    if match:
        return match.group(1)
    return None

# Filter chunks by chromosome before loading
for chunk_file in chunk_files:
    # Skip irrelevant chromosomes based on filename
    if target_chromosomes:
        chrom = extract_chromosome_from_filename(chunk_file)
        if chrom and chrom not in target_chromosomes:
            continue
    
    # Load and filter by prediction type immediately
    df = pl.read_csv(chunk_file, separator='\t')
    df = df.filter(pl.col('pred_type').is_in(['FP', 'TP']))
    
    # Filter by genes if specified
    if target_genes:
        gene_col = identify_gene_column(df)
        if gene_col:
            df = df.filter(pl.col(gene_col).is_in(target_genes))
```

#### 2. Gene Column Identification with Fallback

```python
def identify_gene_column(df):
    """Identify which gene column to use (gene_name or gene_id as fallback)."""
    if 'gene_name' in df.columns:
        non_null_count = df.filter(pl.col('gene_name').is_not_null()).height
        if non_null_count > 0:
            return 'gene_name'
    
    if 'gene_id' in df.columns:
        non_null_count = df.filter(pl.col('gene_id').is_not_null()).height
        if non_null_count > 0:
            logger.info("Using 'gene_id' column as gene_name is not available")
            return 'gene_id'
    
    logger.warning("No gene column found. Using row-level sampling.")
    return None
```

#### 3. Gene-Level Random Sampling

```python
def sample_by_genes(df, gene_col, max_samples, seed=42):
    """Sample at gene level to preserve genomic structure."""
    np.random.seed(seed)
    
    # Get unique genes and their sample counts
    gene_counts = df.group_by(gene_col).agg(pl.len().alias('count'))
    gene_counts = gene_counts.filter(pl.col(gene_col).is_not_null())
    
    # Shuffle genes
    genes = gene_counts[gene_col].to_list()
    counts = gene_counts['count'].to_list()
    indices = np.arange(len(genes))
    np.random.shuffle(indices)
    
    # Select genes until reaching approximate sample size
    selected_genes = []
    current_samples = 0
    
    for idx in indices:
        gene = genes[idx]
        count = counts[idx]
        
        if current_samples + count <= max_samples * 1.1:  # Allow 10% overshoot
            selected_genes.append(gene)
            current_samples += count
        elif current_samples < max_samples * 0.9:  # If under 90%, include anyway
            selected_genes.append(gene)
            current_samples += count
            break
        else:
            break
    
    # Filter to selected genes
    return df.filter(pl.col(gene_col).is_in(selected_genes))
```

#### 4. Balanced Sampling by Prediction Type

The system performs balanced sampling per prediction type to ensure equal representation:

```python
# Apply gene-level sampling per prediction type
if max_rows_per_type is not None:
    sampled_dfs = []
    
    for pred_type in required_pred_types:  # e.g., ['FP', 'TP']
        type_df = consolidated_df.filter(pl.col('pred_type') == pred_type)
        
        if type_df.height > max_rows_per_type:
            # Gene-level sampling for this prediction type
            type_df = sample_by_genes(type_df, gene_col, max_rows_per_type, seed)
        else:
            logger.info(f"Kept all {pred_type}: {type_df.height:,} rows")
        
        sampled_dfs.append(type_df)
    
    consolidated_df = pl.concat(sampled_dfs, how='vertical')

# Add binary labels based on prediction type
label_map = {'FP': 1, 'TP': 0}  # FP is error (positive), TP is correct (negative)
consolidated_df = consolidated_df.with_columns(
    pl.col('pred_type').replace(label_map).alias('label')
)
```

## Implementation Details

### Efficient Chunk Processing

The workflow processes chunks with early filtering for efficiency:

```python
# Process chunks with early filtering
consolidated_dfs = []
for chunk_file in chunk_files:
    # Read chunk directly (not lazy) for better control
    df = pl.read_csv(chunk_file, separator='\t')
    
    # Filter by prediction type first (most selective filter)
    df = df.filter(pl.col('pred_type').is_in(required_pred_types))
    
    if df.height == 0:
        continue
    
    # Filter by genes if specified
    if target_genes:
        gene_col = identify_gene_column(df)
        if gene_col:
            df = df.filter(pl.col(gene_col).is_in(target_genes))
    
    # Add gene_id as gene_name if using gene_id fallback
    if gene_col == 'gene_id' and 'gene_name' not in df.columns:
        df = df.with_columns(pl.col('gene_id').alias('gene_name'))
    
    consolidated_dfs.append(df)

# Concatenate all filtered chunks
consolidated_df = pl.concat(consolidated_dfs, how='vertical')
```

### Memory Optimization

For large datasets, the system uses lazy evaluation where possible:

```python
# Example of memory-efficient processing
def process_chunks_lazily(chunk_files, filters):
    for chunk_file in chunk_files:
        # Process one chunk at a time
        chunk = pd.read_csv(chunk_file, chunksize=10000)
        for sub_chunk in chunk:
            # Apply filters immediately
            filtered = apply_filters(sub_chunk, filters)
            if not filtered.empty:
                yield filtered
```

## Edge Cases and Handling

### 1. Missing gene_name Column

```python
# Automatic fallback to gene_id
if 'gene_name' not in df.columns:
    if 'gene_id' in df.columns:
        logger.info("Using 'gene_id' column as gene_name is not available")
        # Add gene_id as gene_name for consistency
        df = df.with_columns(pl.col('gene_id').alias('gene_name'))
    else:
        logger.warning("No gene column found, falling back to row-level sampling")
        # Fallback to simple random sampling
        return df.sample(n=min(sample_size, df.height), seed=42)
```

### 2. Insufficient Data

```python
if total_rows < sample_size:
    logger.info(f"Dataset has {total_rows} rows, less than requested {sample_size}")
    # Return all data
    return df
```

### 3. Single Gene Exceeds Sample Size

```python
# Check for genes that alone exceed sample size
large_genes = gene_sizes[gene_sizes['row_count'] > sample_size]
if not large_genes.empty:
    logger.warning(f"Found {len(large_genes)} genes exceeding sample size")
    # Option 1: Skip these genes
    # Option 2: Include anyway with warning
    # Current implementation: Include with warning
```

### 4. Label Mapping for Error Analysis

```python
# Map prediction types to binary labels
def get_prediction_types_for_error_analysis(error_type):
    """Get required prediction types and label mapping."""
    error_type_mapping = {
        'fp_vs_tp': ({'FP', 'TP'}, {'FP': 1, 'TP': 0}),  # FP is error
        'fn_vs_tn': ({'FN', 'TN'}, {'FN': 1, 'TN': 0}),  # FN is error
        'fn_vs_tp': ({'FN', 'TP'}, {'FN': 1, 'TP': 0}),  # FN is error
        'fp_vs_tn': ({'FP', 'TN'}, {'FP': 1, 'TN': 0}),  # FP is error
        'error_vs_correct': ({'FP', 'FN', 'TP'}, {'FP': 1, 'FN': 1, 'TP': 0}),
        'all': ({'FP', 'FN', 'TP', 'TN'}, {'FP': 1, 'FN': 1, 'TP': 0, 'TN': 0})
    }
    return error_type_mapping[error_type]
```

## Validation and Logging

### Sampling Statistics

The workflow logs detailed statistics:

```python
logger.info(f"Subsampling statistics:")
logger.info(f"  Total genes available: {total_genes}")
logger.info(f"  Genes selected: {len(selected_genes)}")
logger.info(f"  Total rows before: {len(df)}")
logger.info(f"  Total rows after: {len(sampled_df)}")
logger.info(f"  Sampling ratio: {len(sampled_df)/len(df):.2%}")

if 'prediction_type' in df.columns:
    # Log class distribution
    before_dist = df['prediction_type'].value_counts()
    after_dist = sampled_df['prediction_type'].value_counts()
    logger.info(f"  Class distribution before: {before_dist.to_dict()}")
    logger.info(f"  Class distribution after: {after_dist.to_dict()}")
```

### Integrity Checks

```python
def validate_gene_integrity(original_df, sampled_df):
    """
    Ensure complete genes are preserved.
    """
    for gene in sampled_df['gene_name'].unique():
        original_rows = set(original_df[original_df['gene_name'] == gene].index)
        sampled_rows = set(sampled_df[sampled_df['gene_name'] == gene].index)
        
        if sampled_rows != original_rows:
            raise ValueError(f"Gene {gene} has partial rows!")
    
    logger.info("✓ Gene integrity validated")
```

## Performance Considerations

### Computational Complexity

- **Gene grouping**: O(n) where n is number of rows
- **Gene selection**: O(g log g) where g is number of genes
- **Data filtering**: O(n)
- **Total complexity**: O(n + g log g)

### Memory Usage

- Peak memory: ~2x the size of largest chunk
- Optimization: Process chunks iteratively when possible
- Trade-off: More I/O operations vs. lower memory footprint

## Example Configurations

### Example 1: Focus on Cancer Genes

```bash
# Create gene list file
cat > cancer_genes.txt << EOF
BRCA1
BRCA2
TP53
KRAS
EGFR
MYC
EOF

# Activate environment
mamba activate surveyor

# Run consolidation with gene filtering
python -m meta_spliceai.splice_engine.meta_models.error_model.dataset.consolidate_analysis_data \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-file consolidated_cancer_genes.tsv \
    --genes-file cancer_genes.txt \
    --error-type fp_vs_tp
```

### Example 2: Chromosome-Specific Consolidation

```bash
# Activate environment
mamba activate surveyor

# Consolidate data from specific chromosomes
python -m meta_spliceai.splice_engine.meta_models.error_model.dataset.consolidate_analysis_data \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-file consolidated_autosomal.tsv \
    --chromosomes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 \
    --max-rows-per-type 50000 \
    --error-type fn_vs_tn
```

### Example 3: Balanced Small Dataset

```bash
# Activate environment
mamba activate surveyor

# Create balanced dataset with 10,000 samples per type
python -m meta_spliceai.splice_engine.meta_models.error_model.dataset.consolidate_analysis_data \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-file consolidated_balanced.tsv \
    --max-rows-per-type 10000 \
    --error-type fp_vs_tp \
    --verbose
```

### Example 4: Large-Scale Consolidation

```bash
# Activate environment
mamba activate surveyor

# Consolidate with large sample size from specific chromosomes
python -m meta_spliceai.splice_engine.meta_models.error_model.dataset.consolidate_analysis_data \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-file consolidated_large.tsv \
    --max-rows-per-type 500000 \
    --chromosomes 1 2 3 4 5 \
    --error-type all \
    --verbose
```

## Debugging Subsampling Issues

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via command line
--log-level DEBUG
```

### Inspect Sampling Results

```python
# After subsampling
sampled_df.to_csv('debug_sampled_data.csv', index=False)

# Check gene completeness
gene_counts_before = df.groupby('gene_name').size()
gene_counts_after = sampled_df.groupby('gene_name').size()

# Genes should have same count or be absent
for gene in gene_counts_after.index:
    assert gene_counts_after[gene] == gene_counts_before[gene]
```

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM during sampling | Loading all chunks at once | Process chunks iteratively |
| Imbalanced output | Skewed input distribution | Use balanced sampling flag |
| Missing genes | Gene filtered out | Check gene names/IDs match |
| Slow sampling | Large number of genes | Use approximate sampling |
| Partial genes | Bug in implementation | Enable integrity checks |

## Integration with Downstream Tasks

### Model Training Integration

The subsampled data maintains all necessary columns for training:
- `sequence`: DNA sequence for transformer input
- `label`: Binary label (0/1) for classification
- `gene_name`: Preserved for analysis
- `chrom`, `start`, `end`: Genomic coordinates
- `prediction_type`: Error type (FP/TP/FN/TN)

### Feature Importance Analysis

Gene-level sampling ensures:
- Complete genomic context for each gene
- Valid attribution analysis
- Meaningful biological interpretation

## Best Practices

1. **Always validate gene integrity** after subsampling
2. **Log sampling statistics** for reproducibility
3. **Use balanced sampling** for classification tasks
4. **Consider chromosome distribution** in sampling
5. **Document random seeds** for reproducibility
6. **Monitor memory usage** during processing
7. **Test with small samples** before large-scale runs

## References

- Gene-level sampling maintains biological validity (Frankish et al., 2019)
- Balanced sampling improves model generalization (He & Garcia, 2009)
- Genomic context critical for splice site prediction (Jaganathan et al., 2019)
