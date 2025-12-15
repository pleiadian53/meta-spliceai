# Incremental Builder Command-Line Examples

## Overview

This document provides real-world, copy-paste ready examples for using `incremental_builder.py` in various scenarios.

## Complete Production Example

### Full-Featured Build (1000 genes)

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 \
    --subset-policy random \
    --gene-types protein_coding lncRNA \
    --gene-ids-file additional_genes.tsv \
    --batch-size 100 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3 \
    --output-dir data/train_random_1000_3mers \
    --overwrite \
    --verbose 2>&1 | tee -a logs/train_random_1000_3mers.log
```

**What this does**:
- Selects 1000 genes randomly (diverse sampling)
- Filters to protein-coding and lncRNA genes
- Includes priority genes from `additional_genes.tsv`
- Uses memory-efficient batching (100 genes, 20K rows)
- Runs base model pass (`--run-workflow`)
- Generates 3-mer features
- Logs everything to file while showing progress

**When to use**: Production dataset with diversity

## Quick Test Examples

### Minimal Test (100 genes, 5-15 minutes)

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 100 \
    --subset-policy random \
    --gene-types protein_coding lncRNA \
    --gene-ids-file additional_genes.tsv \
    --batch-size 50 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3 \
    --output-dir data/train_test_100_3mers \
    --overwrite \
    --verbose 2>&1 | tee -a logs/test_100.log
```

**When to use**: Quick validation, testing changes

### Ultra-Fast Test (10 genes, 1-2 minutes)

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 10 \
    --subset-policy random \
    --gene-types protein_coding \
    --batch-size 10 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3 \
    --output-dir data/train_test_10_3mers \
    --overwrite \
    -vv
```

**When to use**: Debugging, development, CI/CD

## Specialized Examples

### Error-Focused Dataset

Select genes with the most SpliceAI errors:

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 5000 \
    --subset-policy error_total \
    --gene-types protein_coding \
    --batch-size 100 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3 5 \
    --output-dir data/train_errors_5000 \
    --overwrite \
    --verbose 2>&1 | tee -a logs/train_errors_5000.log
```

**When to use**: Training meta-model to fix errors

### Disease-Specific Dataset (ALS Example)

Build dataset focused on ALS-related genes:

```bash
# First, create ALS gene list
cat > als_genes.tsv <<EOF
gene_id	gene_name	description
ENSG00000130402	UNC13A	Cryptic exon in ALS
ENSG00000075558	STMN2	TDP-43 target
ENSG00000087086	FUS	ALS/FTD gene
ENSG00000120948	TARDBP	TDP-43
ENSG00000100448	CTSD	Cathepsin D
ENSG00000134243	SORT1	Sortilin 1
ENSG00000171608	PIK3CA	PI3K pathway
EOF

# Build dataset with ALS focus
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 500 \
    --subset-policy random \
    --gene-types protein_coding \
    --gene-ids-file als_genes.tsv \
    --batch-size 100 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3 \
    --output-dir data/train_als_500_3mers \
    --overwrite \
    --verbose 2>&1 | tee -a logs/train_als_500.log
```

**When to use**: Disease-specific model training

### Custom Gene List Only

Build dataset using ONLY specified genes (no additional selection):

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --subset-policy custom \
    --gene-ids-file my_genes.tsv \
    --gene-col gene_id \
    --batch-size 50 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3 \
    --output-dir data/train_custom \
    --overwrite \
    --verbose 2>&1 | tee -a logs/train_custom.log
```

**Note**: Don't specify `--n-genes` when using `--subset-policy custom`

**When to use**: Specific genes for testing or analysis

### Multi-Kmer Dataset

Generate multiple k-mer feature sets:

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 \
    --subset-policy random \
    --gene-types protein_coding lncRNA \
    --batch-size 100 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3 5 7 \
    --output-dir data/train_multi_kmer_1000 \
    --overwrite \
    --verbose 2>&1 | tee -a logs/train_multi_kmer.log
```

**When to use**: Comparing k-mer sizes, feature selection experiments

### Large-Scale Production Dataset

Build massive dataset for production meta-model:

```bash
# WARNING: This will take 10-15 hours!
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 10000 \
    --subset-policy random \
    --gene-types protein_coding lncRNA \
    --batch-size 200 \
    --batch-rows 50000 \
    --run-workflow \
    --kmer-sizes 3 \
    --output-dir data/train_production_10k \
    --overwrite \
    --verbose 2>&1 | tee -a logs/train_production_10k.log
```

**Recommendation**: Use resumable execution:
```bash
./scripts/builder/run_builder_resumable.sh tmux
# Edit the script to set --n-genes 10000
```

**When to use**: Final production model training

## Memory-Constrained Environments

### Laptop-Friendly (4GB RAM)

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 500 \
    --subset-policy random \
    --gene-types protein_coding \
    --batch-size 50 \
    --batch-rows 10000 \
    --run-workflow \
    --kmer-sizes 3 \
    --output-dir data/train_laptop_500 \
    --overwrite \
    --verbose 2>&1 | tee -a logs/train_laptop.log
```

### Server (32GB+ RAM)

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 5000 \
    --subset-policy random \
    --gene-types protein_coding lncRNA \
    --batch-size 500 \
    --batch-rows 200000 \
    --run-workflow \
    --kmer-sizes 3 5 \
    --output-dir data/train_server_5000 \
    --overwrite \
    --verbose 2>&1 | tee -a logs/train_server.log
```

## Without Base Model Pass

### Rebuild Dataset (Artifacts Already Exist)

```bash
# If you already have SpliceAI artifacts and just want to rebuild the dataset
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 \
    --subset-policy random \
    --gene-types protein_coding lncRNA \
    --batch-size 100 \
    --batch-rows 20000 \
    --kmer-sizes 3 \
    --output-dir data/train_rebuild_1000 \
    --overwrite \
    --verbose 2>&1 | tee -a logs/train_rebuild.log
```

**Note**: Omit `--run-workflow` when artifacts exist

**When to use**: Testing different k-mers, batch sizes, or downsampling strategies

## Parameter Guidelines

### Batch Size Selection

| Available RAM | --batch-size | --batch-rows | Genes/Hour* |
|---------------|--------------|--------------|-------------|
| 4GB           | 50           | 10000        | ~150        |
| 8GB           | 100          | 20000        | ~180        |
| 16GB          | 200          | 50000        | ~200        |
| 32GB+         | 500          | 200000       | ~220        |

*Approximate, varies by gene complexity

### Gene Type Combinations

Common combinations:

```bash
# Coding genes only
--gene-types protein_coding

# Coding + long non-coding
--gene-types protein_coding lncRNA

# Include pseudogenes (for error analysis)
--gene-types protein_coding lncRNA pseudogene

# All annotated types
# (omit --gene-types to include all)
```

### K-mer Size Selection

| K-mer Size | Features | Use Case | Speed |
|------------|----------|----------|-------|
| 3          | 64       | Fast, general | Fast |
| 5          | 1024     | More specific | Medium |
| 7          | 16384    | Very specific | Slow |
| 3,5        | 1088     | Comparison | Medium |

**Recommendation**: Start with `--kmer-sizes 3` for development, experiment with multiple sizes for production.

## Common Patterns

### Development Cycle

```bash
# 1. Ultra-fast test (verify setup)
--n-genes 10 --batch-size 10 --kmer-sizes 3

# 2. Quick test (verify logic)
--n-genes 100 --batch-size 50 --kmer-sizes 3

# 3. Medium test (verify performance)
--n-genes 500 --batch-size 100 --kmer-sizes 3

# 4. Production
--n-genes 5000 --batch-size 200 --kmer-sizes 3 5
```

### Logging Pattern

Always use this pattern for production runs:

```bash
mkdir -p logs
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    [your options] \
    --verbose 2>&1 | tee -a logs/$(date +%Y%m%d_%H%M%S)_build.log
```

### Verification Pattern

After each build:

```bash
# Check output
ls -lh data/your_output_dir/

# Verify gene count
wc -l data/your_output_dir/gene_manifest.csv

# Check dataset size
python -c "import polars as pl; df = pl.scan_parquet('data/your_output_dir/master/*.parquet'); print(f'Rows: {df.select(pl.count()).collect().item():,}')"
```

## Troubleshooting Examples

### Debugging Failed Build

Add extra verbosity:

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    [your options] \
    -vvv 2>&1 | tee -a logs/debug.log
```

### Memory Issues

Reduce batch sizes aggressively:

```bash
--batch-size 25 \
--batch-rows 5000
```

### Check What Would Be Selected

Dry-run gene selection (without building):

```bash
python -c "
from meta_spliceai.splice_engine.meta_models.io.handlers import MetaModelDataHandler
from meta_spliceai.splice_engine.meta_models.features.gene_selection import subset_analysis_sequences

dh = MetaModelDataHandler()
_, genes = subset_analysis_sequences(
    data_handler=dh,
    n_genes=100,
    subset_policy='random',
    verbose=2
)
print(f'Would select {len(genes)} genes')
print('First 10:', genes[:10])
"
```

## Summary

**Most Common Command** (recommended starting point):

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 \
    --subset-policy random \
    --gene-types protein_coding lncRNA \
    --gene-ids-file additional_genes.tsv \
    --batch-size 100 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3 \
    --output-dir data/train_random_1000_3mers \
    --overwrite \
    --verbose 2>&1 | tee -a logs/train_random_1000_3mers.log
```

**Key Adjustments**:
- Smaller dataset: Change `--n-genes 100`
- More memory: Increase `--batch-size 200 --batch-rows 50000`
- Less memory: Decrease `--batch-size 50 --batch-rows 10000`
- Different genes: Modify gene list file or `--subset-policy`
- Multiple k-mers: Add sizes `--kmer-sizes 3 5 7`

---

**Pro Tip**: Start with a small test (100 genes), verify the output, then scale up to production size.

