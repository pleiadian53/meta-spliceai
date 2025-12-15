# Gene-Position Index for Training Datasets

## Overview

The position-centric nature of our training data means each gene has thousands of positions, each with its own feature vector. This utility creates an index to enable efficient gene-to-feature-vector lookups.

## Problem

**Current Training Data Structure:**
- Each row = one nucleotide position
- Each gene has 1,000-10,000+ positions
- Feature vectors include: k-mers, scores, probabilities, etc.
- No direct way to quickly find all positions for a gene

**Example:**
```
gene_id          position  donor_score  acceptor_score  splice_type  ...
ENSG00000139618  1234      0.85         0.12           neither      ...
ENSG00000139618  1235      0.87         0.11           neither      ...
ENSG00000139618  1236      0.92         0.08           donor        ...
... (thousands more relative positions for this gene)
```

**Note:** The `position` column contains strand-dependent relative coordinates, not absolute genomic coordinates.

## Solution: Gene-Position Index

### 1. Create the Index

```bash
# After your training dataset is complete
python scripts/scaling_solutions/utilities/create_gene_index.py train_pc_1000_3mers/master/
```

This creates `gene_position_index.csv` with:
- `global_index`: Sequential gene index
- `gene_id`: Ensembl gene ID
- `chrom`: Chromosome
- `position_count`: Number of positions for this gene
- `positions_json`: JSON string of all relative positions for this gene
- `splice_types_json`: JSON string of splice types for each position
- `file_index`: Which batch file contains this gene
- `file_name`: Specific parquet file name

**Note:** The `positions_json` column contains strand-dependent relative coordinates within each gene, not absolute genomic coordinates. JSON format is used for CSV compatibility.

**Important:** The script automatically normalizes splice_type encoding for compatibility:
- Preserves `"donor"` and `"acceptor"` (correct format)
- Converts anything else (`"0"`, `None`, `null`, empty strings, etc.) → `"neither"`
- This handles all edge cases in older and newer datasets

### 2. Use the Index for Quick Lookups

```python
from scripts.scaling_solutions.utilities.create_gene_index import load_gene_positions

# Load all positions and features for a specific gene
gene_data = load_gene_positions(
    index_path="train_pc_1000_3mers/master/gene_position_index.csv",
    gene_id="ENSG00000139618"
)

print(f"Gene has {len(gene_data)} positions")
print(f"Feature columns: {gene_data.columns}")
```

### 3. Gene-Level Analysis

```python
import polars as pl

# Get all positions for a gene
gene_data = load_gene_positions("gene_position_index.csv", "ENSG00000139618")

# Analyze splice sites
splice_sites = gene_data.filter(pl.col("splice_type") != "neither")
print(f"Found {len(splice_sites)} splice sites")

# Calculate gene-level statistics
gene_stats = gene_data.group_by("gene_id").agg([
    pl.col("donor_score").mean().alias("avg_donor_score"),
    pl.col("acceptor_score").mean().alias("avg_acceptor_score"),
    pl.col("splice_type").filter(pl.col("splice_type") != "neither").count().alias("splice_site_count")
])
```

## Performance Benefits

### Before (Inefficient):
```python
# Load entire dataset (memory intensive)
df = pl.read_parquet("train_pc_1000_3mers/master/*.parquet")
# Filter by gene (slow for large datasets)
gene_data = df.filter(pl.col("gene_id") == "ENSG00000139618")
```

### After (Efficient):
```python
# Load only the specific file containing the gene
gene_data = load_gene_positions("gene_position_index.csv", "ENSG00000139618")
```

## Integration with Gene Manifest

The gene manifest (`gene_manifest.csv`) and gene-position index work together:

- **Gene Manifest**: Lists all genes in the dataset
- **Gene-Position Index**: Maps genes to their positions and features

```python
# Load both files
manifest = pl.read_csv("train_pc_1000_3mers/master/gene_manifest.csv")
index = pl.read_csv("train_pc_1000_3mers/master/gene_position_index.csv")

# Find genes with many positions
high_position_genes = index.filter(pl.col("position_count") > 5000)
print(f"Genes with >5000 positions: {len(high_position_genes)}")

# Get all feature vectors for a high-position gene
gene_id = high_position_genes.select("gene_id").item(0)
gene_data = load_gene_positions("gene_position_index.csv", gene_id)
```

## Usage Examples

### 1. Find Genes by Position Count
```python
index = pl.read_csv("gene_position_index.csv")
large_genes = index.filter(pl.col("position_count") > 10000)
print(f"Large genes: {large_genes.select('gene_id').to_list()}")
```

### 2. Analyze Splice Site Distribution
```python
gene_data = load_gene_positions("gene_position_index.csv", "ENSG00000139618")
splice_sites = gene_data.filter(pl.col("splice_type") != "neither")
print(f"Splice sites: {splice_sites.select('position').to_list()}")
```

### 3. Compare Multiple Genes
```python
gene_ids = ["ENSG00000139618", "ENSG00000141510", "ENSG00000157764"]
for gene_id in gene_ids:
    gene_data = load_gene_positions("gene_position_index.csv", gene_id)
    splice_count = gene_data.filter(pl.col("splice_type") != "neither").height
    print(f"{gene_id}: {splice_count} splice sites")
```

## File Structure

After running the index creator:
```
train_pc_1000_3mers/
├── master/
│   ├── batch_00001.parquet
│   ├── batch_00002.parquet
│   ├── ...
│   ├── gene_manifest.csv          # List of genes
│   └── gene_position_index.csv    # Gene-to-position mapping
```

## Memory Considerations

- **Index file**: Small (~1MB for 1000 genes)
- **Gene data loading**: Only loads the specific file containing the gene
- **Position lists**: Stored as JSON arrays in the index for quick reference

This approach provides efficient gene-to-feature-vector lookups while maintaining the position-centric data structure. 