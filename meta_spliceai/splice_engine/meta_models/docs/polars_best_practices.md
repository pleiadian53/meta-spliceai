# Polars Best Practices in MetaSpliceAI

This document outlines useful Polars patterns and techniques used throughout the MetaSpliceAI codebase. It serves as a reference for developers working with genomic sequence data processing.

## Table of Contents
1. [Working with Columns](#working-with-columns)
2. [Data Type Handling](#data-type-handling)
3. [Filtering and Selection](#filtering-and-selection)
4. [Aggregation Functions](#aggregation-functions)
5. [Performance Optimization](#performance-optimization)
6. [Pandas vs Polars](#pandas-vs-polars)

## Working with Columns

### Applying Functions to Column Elements

#### Using `map_elements()`

The `map_elements()` function applies a Python function to each element in a column. This is particularly useful for operations that don't have direct Polars equivalents.

```python
# Calculate the length of each sequence in a column
df.select(
    pl.col("sequence").map_elements(len, return_dtype=pl.Int64)
)

# Calculate the average sequence length
avg_length = df.select(
    pl.col("sequence").map_elements(len, return_dtype=pl.Int64).mean()
).item()
```

Breakdown of this pattern:
1. `pl.col("sequence")` - Selects the "sequence" column from the dataframe
2. `.map_elements(len, return_dtype=pl.Int64)` - Applies the Python built-in `len()` function to each sequence string in the column
   - The `len()` function calculates the length of each sequence string
   - `return_dtype=pl.Int64` explicitly tells Polars the returned values will be 64-bit integers
3. `.mean()` - Calculates the mean (average) of all the sequence lengths
4. `.item()` - Extracts the single value from the resulting DataFrame

This approach is more efficient than iterating through rows manually because:
- **Vectorization** - Polars optimizes the operation to process elements in parallel
- **Type Safety** - Specifying `return_dtype` ensures consistent type handling
- **Query Optimization** - Polars creates an execution plan that minimizes memory usage

**Important**: Always specify the `return_dtype` parameter to avoid warnings and ensure consistent behavior.

#### Using `with_columns()`

The `with_columns()` method allows you to add or modify columns:

```python
# Convert chromosome names to string type
df = df.with_columns(pl.col("seqname").cast(pl.Utf8))

# Add a new column with sequence lengths
df = df.with_columns(
    pl.col("sequence").map_elements(len, return_dtype=pl.Int64).alias("seq_length")
)
```

## Data Type Handling

### Type Conversion

Consistent type handling is crucial when working with genomic data, especially for chromosome names that may be stored as strings or integers.

```python
# Convert chromosome column to string type
df = df.with_columns(pl.col("seqname").cast(pl.Utf8))

# Ensure numeric types for position information
df = df.with_columns([
    pl.col("start").cast(pl.Int64),
    pl.col("end").cast(pl.Int64)
])
```

### Handling Mixed Types

When combining data from different sources, type inconsistencies can occur:

```python
# Detect column type by name
chrom_col = 'seqname' if 'seqname' in df.columns else 'chrom'

# Ensure consistent type before operations
df = df.with_columns(pl.col(chrom_col).cast(pl.Utf8))
```

### Schema Overrides for Consistent Type Handling

When reading data from files, Polars infers types by default. This can lead to inconsistencies, especially with genomic data where chromosome identifiers might be treated inconsistently (sometimes as integers, sometimes as strings).

#### Using `schema_overrides` for Robust Type Handling

The `schema_overrides` parameter in functions like `read_csv()` allows you to explicitly define data types for specific columns, ensuring consistent behavior:

```python
# Handle chromosome identifiers that mix numeric and non-numeric formats (e.g., "1" vs "X" vs "GL000194.1")
df = pl.read_csv(
    "overlapping_gene_counts.tsv",
    separator="\t",
    schema_overrides={
        # Ensure chromosome names are always treated as strings
        "chrom": pl.Utf8,
        "seqname": pl.Utf8
    },
    ignore_errors=True  # Handle other potential type issues gracefully
)
```

**Key benefits of this approach:**

1. **Defensive Programming**: You can specify schemas for columns that might appear in some datasets but not others. If a column specified in `schema_overrides` doesn't exist in the data, it's simply ignored.

2. **Mixed Data Handling**: When chromosome identifiers contain both numeric values ("1", "2") and non-numeric values ("X", "Y", "GL000194.1"), forcing them to be strings avoids type inference errors.

3. **Consistent Cross-Framework Processing**: Ensures consistent behavior when data moves between Polars and Pandas, where type handling differs.

#### Common Use Cases in MetaSpliceAI

- **Chromosome Fields**: Always define as `pl.Utf8` to handle the full range of chromosome naming conventions
- **Gene and Transcript IDs**: Define as `pl.Utf8` since they often contain alphanumeric patterns
- **Genomic Coordinates**: Define as `pl.Int64` for consistent numeric operations 
- **Floating-Point Scores**: Use `pl.Float64` for values like prediction scores or p-values

```python
# Example of comprehensive type handling for genomic data
schema_overrides = {
    # Identifier fields (always strings)
    "gene_id": pl.Utf8,
    "transcript_id": pl.Utf8,
    "chrom": pl.Utf8,
    "seqname": pl.Utf8,
    "strand": pl.Utf8,
    
    # Coordinate fields (always integers)
    "start": pl.Int64,
    "end": pl.Int64,
    "exon_number": pl.Int64,
    
    # Score fields (always floats)
    "score": pl.Float64,
    "confidence": pl.Float64
}

df = pl.read_csv("annotations.tsv", separator="\t", schema_overrides=schema_overrides)
```

Remember that explicitly defining types not only prevents errors but also improves performance by avoiding unnecessary type conversions during processing.

## Filtering and Selection

### Filtering Rows

```python
# Filter by chromosome
filtered_df = df.filter(pl.col("seqname").is_in(target_chromosomes))

# Filter by gene ID
filtered_df = df.filter(pl.col("gene_id").is_in(target_genes))

# Complex filtering
filtered_df = df.filter(
    (pl.col("seqname") == "chr1") & 
    (pl.col("start") > 1000000) & 
    (pl.col("end") < 2000000)
)
```

### Extracting Values

```python
# Get a single value from a dataframe
first_sequence = df.select(pl.col("sequence")).row(0)[0]

# Get unique values from a column
unique_chromosomes = df["seqname"].unique().to_list()
```

## Aggregation Functions

```python
# Count occurrences
gene_counts = df.group_by("gene_id").count()

# Multiple aggregations
stats = df.group_by("seqname").agg([
    pl.count().alias("count"),
    pl.col("sequence").map_elements(len, return_dtype=pl.Int64).mean().alias("avg_length"),
    pl.col("sequence").map_elements(len, return_dtype=pl.Int64).min().alias("min_length"),
    pl.col("sequence").map_elements(len, return_dtype=pl.Int64).max().alias("max_length")
])
```

## Performance Optimization

### Lazy Evaluation

For large genomic datasets, lazy evaluation can significantly improve performance:

```python
# Create a lazy query
lazy_query = (
    df.lazy()
    .filter(pl.col("seqname").is_in(target_chromosomes))
    .filter(pl.col("gene_id").is_in(target_genes))
    .select([
        pl.col("gene_id"),
        pl.col("gene_name"),
        pl.col("seqname"),
        pl.col("sequence")
    ])
)

# Execute the query only when needed
result = lazy_query.collect()
```

### Reading Large Files

```python
# Read large TSV files efficiently
df = pl.read_csv(
    "large_gene_sequences.tsv",
    separator="\t",
    low_memory=True,
    schema_overrides={"seqname": pl.Utf8}
)
```

## Pandas vs Polars

In our codebase, we support both Pandas and Polars for compatibility. Here's how to detect and handle both types:

```python
def process_dataframe(df):
    """Process either a Pandas or Polars dataframe."""
    import polars as pl
    import pandas as pd
    
    is_polars = isinstance(df, pl.DataFrame)
    
    if is_polars:
        # Polars approach
        result = df.with_columns(pl.col("seqname").cast(pl.Utf8))
    else:
        # Pandas approach
        result = df.copy()
        result["seqname"] = result["seqname"].astype(str)
    
    return result
```

### Auto-detection Pattern

The pattern used in our `combine_sequence_dataframes` function:

```python
# Auto-detect dataframe type if not specified
if use_polars is None:
    import polars as pl
    # Check if the first dataframe is a Polars dataframe
    use_polars = isinstance(dfs[0], pl.DataFrame)
```

---

This documentation will be expanded as more Polars patterns emerge in the codebase.
