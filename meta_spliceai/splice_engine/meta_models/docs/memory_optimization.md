# Memory Optimization for Genomic Sequence Processing

This document outlines strategies implemented in MetaSpliceAI to optimize memory usage when processing large genomic datasets, particularly when extracting and analyzing sequences around splice sites.

## Background

Genomic sequence analysis in MetaSpliceAI can become memory-intensive due to:

1. The large size of the human genome (~3.2 billion bases)
2. Multiple chromosomes with thousands of genes
3. Sequence extraction around numerous splice sites
4. Storage of extracted sequences (typically 1KB per site × millions of sites)
5. Concurrent storage of feature vectors and probability scores

Without proper memory management, this can lead to:
- Out-of-memory errors
- Poor performance due to excessive garbage collection
- Memory thrashing when RAM is insufficient

## Implemented Optimization Strategies

### 1. Chunk-Level Persistence

**Problem:** Accumulating all `df_seq` dataframes in memory quickly consumes RAM when processing the entire genome.

**Solution:** Write each chromosome chunk's sequence data to disk immediately:

```python
# Extract sequences for this chunk
df_seq = extract_analysis_sequences(
    seq_chunk,
    positions_df_chunk,
    include_empty_entries=True,
    verbose=1
)

# Write this chunk's sequences straight to disk; don't keep in RAM
analysis_seq_path = data_handler.save_analysis_sequences(
    df_seq,
    chr=chr_,
    chunk_start=chunk_start + 1,
    chunk_end=chunk_end,
    aggregated=False  # chunk-level file to minimize memory footprint
)

# Explicitly free memory for this chunk
del df_seq
```

This change eliminates the need for the `full_analysis_seq_df` accumulator, keeping memory usage constant regardless of the number of chromosomes processed.

### 2. Schema Inference in Polars

**Problem:** Type inconsistencies in the first few rows can cause Polars to infer incorrect schema, leading to `ComputeError` exceptions when encountering incompatible values later.

**Solution:** Use explicit schema inference across all rows:

```python
# Force Polars to scan the entire dataset for proper schema inference
output_df = pl.from_dicts(extracted_sequences, infer_schema_length=None)
```

This avoids errors when early rows contain nulls or different types than later rows.

### 3. Lazy Evaluation with Polars

**Problem:** Reading large chromosome files eagerly loads entire datasets into memory.

**Solution:** Use lazy execution with `scan_*` operations:

```python
lazy_seq_df = pl.scan_parquet(chr_files[0]) if format == 'parquet' else pl.scan_csv(chr_files[0], separator=separator)
```

This defers actual data loading until necessary and enables Polars to optimize the query plan.

### 4. Other Available Optimization Strategies

#### Streaming Parquet Writes

For ultra-large datasets, consider using the PyArrow streaming API:

```python
import pyarrow as pa
import pyarrow.parquet as pq

schema = pa.schema([
    ('gene_id', pa.string()),
    ('position', pa.int64()),
    ('sequence', pa.string()),
    # additional fields...
])

with pq.ParquetWriter('all_sequences.parquet', schema) as writer:
    for chromosome in chromosomes:
        # Process chromosome data
        table = pa.Table.from_pandas(df_chunk)
        writer.write_table(table)
```

#### Memory-Managed Dataframe Concatenation

When you must combine dataframes, consider these approaches:

1. **Columnar concatenation**: Join one column at a time instead of whole dataframes.
2. **Checkpoint-based concatenation**: Periodically write to disk and reset the accumulator.
3. **Database-backed storage**: Use SQLite or DuckDB for larger-than-memory datasets.

#### Dynamic Chunk Sizing

The workflow already implements adaptive chunk sizing through `adjust_chunk_size()`. This can be enhanced by monitoring memory usage:

```python
import psutil

def adjust_chunk_size_by_memory(current_chunk_size, target_memory_percent=70):
    """Adjust chunk size based on memory pressure"""
    current_memory_percent = psutil.virtual_memory().percent
    
    if current_memory_percent > target_memory_percent + 10:
        # Reduce chunk size under high memory pressure
        return max(10, current_chunk_size // 2)
    elif current_memory_percent < target_memory_percent - 20:
        # Increase chunk size when memory is available
        return min(1000, current_chunk_size * 2)
    
    # Keep current size
    return current_chunk_size
```

## Performance Considerations

These optimizations have several impacts:

1. **Memory Usage**: Significantly reduced peak memory consumption.
2. **Disk I/O**: Increased disk read/write operations.
3. **Processing Time**: Minor overhead from increased I/O, but prevents catastrophic slow-down from memory thrashing.

For systems with limited RAM, consider:
- Reducing window sizes (e.g., 250nt instead of 500nt)
- Processing fewer chromosomes at once
- Running with a subset of genes (`target_genes` parameter)
- Using memory-efficient string types where possible

## Environment Recommendations

Based on our optimization testing:

| Dataset Size | Recommended RAM | Storage Requirements |
|--------------|----------------|---------------------|
| Single chromosome | 8GB | ~5GB |
| Whole genome | 16GB+ | ~50GB |
| Full analysis with sequence windows | 32GB+ | ~100GB |

GPU memory considerations are separate and primarily apply to the transformer-based error sequence models.
