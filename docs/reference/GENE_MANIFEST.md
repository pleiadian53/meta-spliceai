# Gene Manifest Feature

The incremental builder now includes an optional gene manifest feature that creates a CSV file tracking all genes included in the training dataset. This makes it easy to:

- **Look up which genes are present** in your training dataset
- **Find the file location** of specific genes (which Parquet file contains them)
- **Get gene names** for easier interpretation (when available)
- **Track gene indices** for efficient data access

## Manifest File Format

The manifest is saved as `gene_manifest.csv` in your training dataset directory and contains the following columns:

| Column | Description |
|--------|-------------|
| `global_index` | Sequential index of the gene in the manifest |
| `gene_id` | Ensembl gene ID (e.g., ENSG00000104435) |
| `gene_name` | Human-readable gene name (e.g., STMN2) |
| `file_index` | Index of the Parquet file containing this gene |
| `file_name` | Name of the Parquet file containing this gene |

## Usage

### Automatic Generation

The manifest is automatically generated when building new training datasets:

```bash
# Build a training dataset (manifest generated automatically)
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 \
    --output-dir train_pc_1000 \
    --verbose

# Skip manifest generation if not needed
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 \
    --output-dir train_pc_1000 \
    --no-manifest
```

### Manual Generation for Existing Datasets

For datasets created before this feature was added, you can generate a manifest manually. **This works perfectly with existing datasets** - the system automatically looks up gene names from `gene_features.tsv`:

```bash
# Generate manifest for existing dataset
python scripts/generate_gene_manifest.py /path/to/train_dataset_trimmed

# Example with an actual dataset
python scripts/generate_gene_manifest.py train_pc_1000/

# Specify custom output location
python scripts/generate_gene_manifest.py /path/to/train_dataset_trimmed --output my_manifest.csv
```

The tool automatically:
- **Detects missing gene names** in older datasets
- **Looks up gene names** from the systematic gene features file
- **Uses the Config system** for robust path resolution
- **Handles different data sources** (ensembl, fabric, lakehouse, etc.)

### Querying the Manifest

Use the query tool to search and explore your manifest:

```bash
# Show statistics about the manifest
python scripts/query_gene_manifest.py gene_manifest.csv --stats

# Search for a specific gene by name
python scripts/query_gene_manifest.py gene_manifest.csv --gene STMN2

# Search for a specific gene by ID
python scripts/query_gene_manifest.py gene_manifest.csv --gene-id ENSG00000104435

# List all genes (first 20)
python scripts/query_gene_manifest.py gene_manifest.csv --list-genes

# List more genes
python scripts/query_gene_manifest.py gene_manifest.csv --list-genes --limit 50
```

## Example Manifest Output

```
global_index,gene_id,gene_name,file_index,file_name
0,ENSG00000104435,STMN2,1,batch_00001.parquet
1,ENSG00000112345,UNC13A,1,batch_00001.parquet
2,ENSG00000167890,TP53,2,batch_00002.parquet
...
```

## Use Cases

### 1. Quick Gene Lookup

```python
import pandas as pd

# Load the manifest
manifest = pd.read_csv("train_pc_1000/gene_manifest.csv")

# Find a specific gene
gene_info = manifest[manifest['gene_name'] == 'STMN2']
if len(gene_info) > 0:
    print(f"STMN2 is in file: {gene_info.iloc[0]['file_name']}")
    print(f"Global index: {gene_info.iloc[0]['global_index']}")
```

### 2. Efficient Data Loading

```python
import polars as pl

# Load manifest
manifest = pl.read_csv("train_pc_1000/gene_manifest.csv")

# Find which file contains your gene of interest
gene_file = manifest.filter(pl.col("gene_name") == "STMN2").select("file_name").item()

# Load only that specific file
df = pl.read_parquet(f"train_pc_1000/master/{gene_file}")
```

### 3. Dataset Statistics

```python
import pandas as pd

manifest = pd.read_csv("train_pc_1000/gene_manifest.csv")

print(f"Total genes: {len(manifest)}")
print(f"Genes with names: {manifest['gene_name'].notna().sum()}")
print(f"Files in dataset: {manifest['file_index'].nunique()}")
```

## Integration with Training

The manifest can be useful during model training and evaluation:

```python
# Load manifest to track which genes are in training vs test sets
train_manifest = pd.read_csv("train_dataset/gene_manifest.csv")
test_manifest = pd.read_csv("test_dataset/gene_manifest.csv")

# Find genes that are in training but not in test
train_genes = set(train_manifest['gene_id'])
test_genes = set(test_manifest['gene_id'])
unseen_genes = train_genes - test_genes

print(f"Genes in training but not in test: {len(unseen_genes)}")
```

## Notes

- The manifest is generated **after** the master dataset is assembled
- Gene names may be missing (showing as `null`) if not available in the source data
- The `global_index` provides a stable ordering of genes across different runs
- File indices correspond to the order in which Parquet files were processed
- The manifest is lightweight and can be easily shared or version-controlled

## Troubleshooting

### Manifest Generation Fails

If manifest generation fails, check:

1. **File permissions**: Ensure you can read the Parquet files
2. **Memory**: Large datasets may require more memory
3. **File structure**: Ensure the `master/` directory contains `.parquet` files

### Missing Gene Names

If many genes show `null` for `gene_name`:

1. Check that gene-level feature enrichment is enabled
2. Verify that `gene_features.tsv` contains gene names
3. Ensure the GTF file used for feature extraction includes gene names

### Performance

For very large datasets (>100k genes), manifest generation may take a few minutes. The process is memory-efficient as it only loads the `gene_id` and `gene_name` columns from each Parquet file. 