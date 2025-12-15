# Incremental Builder Usage Guide

## Overview

This guide provides comprehensive instructions for using `incremental_builder.py` to generate training datasets for the meta-model, with a focus on the critical `--run-workflow` option.

## Quick Start

### Minimal Example (Test)

```bash
cd /Users/pleiadian53/work/meta-spliceai

# Quick test with 100 genes (5-15 minutes)
./scripts/testing/test_incremental_builder.sh
```

### Production Example (1000 genes)

```bash
# Full build with resumable execution (40-60 minutes)
./scripts/builder/run_builder_resumable.sh tmux
```

## Understanding the Pipeline

### Two-Phase Architecture

The incremental builder operates in **two distinct phases**:

```
Phase 1: BASE MODEL PASS (--run-workflow)
└─ Runs SpliceAI inference to generate prediction artifacts
   └─ Output: data/ensembl/spliceai_analysis/*.tsv

Phase 2: DATASET BUILDING
└─ Reads artifacts and builds training dataset
   └─ Output: data/train_*/master/*.parquet
```

### Phase 1: Base Model Pass (`--run-workflow`)

**Purpose**: Generate SpliceAI predictions with enriched features

**What it does**:
1. Loads SpliceAI ensemble models (5 models from `data/models/spliceai/`)
2. Runs per-nucleotide inference on selected genes
3. Computes donor/acceptor/neither probabilities for every position
4. Evaluates predictions against ground truth (TP/FP/FN/TN classification)
5. Extracts context features (surrounding scores at ±1, ±2 positions)
6. Engineers derived features (peak ratios, signal strength, entropy, etc.)
7. Writes enriched artifacts to `data/ensembl/spliceai_analysis/`

**When to use**:
- ✅ Building dataset for genes without existing predictions
- ✅ Creating a new dataset with different gene sets
- ✅ Updating predictions after annotation changes
- ❌ Rebuilding with same genes (use `--overwrite` instead)

**Example**:
```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 \
    --subset-policy error_total \
    --run-workflow \
    --output-dir data/train_pc_1000_3mers
```

### Phase 2: Dataset Building (Always Runs)

**Purpose**: Transform artifacts into meta-model training dataset

**What it does**:
1. Loads analysis artifacts from `data/ensembl/spliceai_analysis/`
2. Featurizes sequences with k-mers (e.g., 3-mers)
3. Applies additional feature enrichment
4. Downsamples true negatives (TN) to balance dataset
5. Writes training data to Parquet format

**Key Parameters**:
- `--kmer-sizes`: K-mer sizes for sequence featurization (default: 6)
- `--batch-size`: Genes per batch (default: 1000)
- `--batch-rows`: Rows per batch (default: 500000)

## Command-Line Options Reference

### Gene Selection

```bash
--n-genes N                # Number of genes to select (default: 20000)
--subset-policy POLICY     # Selection strategy:
                          #   - error_total: Most errors (FP+FN)
                          #   - error_fp: Most false positives
                          #   - error_fn: Most false negatives
                          #   - random: Random sampling
                          #   - custom: Use provided gene list
                          #   - all: All available genes
--gene-ids-file FILE      # File with custom gene IDs (TSV/CSV/TXT)
--gene-col COLUMN         # Column name for gene IDs (default: gene_id)
--gene-types TYPE [TYPE...] # Filter by gene type (e.g., protein_coding lncRNA)
```

### Workflow Control

```bash
--run-workflow            # Run base model pass (SpliceAI inference)
--workflow-kwargs JSON    # JSON dict with workflow parameters
--overwrite               # Overwrite existing files
```

### Feature Engineering

```bash
--kmer-sizes SIZE [SIZE...] # K-mer sizes (space or comma separated)
                           # Examples:
                           #   --kmer-sizes 3
                           #   --kmer-sizes 3 5
                           #   --kmer-sizes 3,5,7
--batch-size N            # Genes per batch (default: 1000)
                          # Reduce for memory-constrained environments
--batch-rows N            # Rows per batch (default: 500000)
                          # Reduce for lower memory usage
```

**Note on Gene Types**: Multiple gene types can be specified with spaces or commas:
```bash
--gene-types protein_coding lncRNA     # Space-separated
--gene-types protein_coding,lncRNA     # Comma-separated (both work)
```

### TN Downsampling

```bash
--hard-prob THRESHOLD     # Hard negative threshold (default: 0.15)
--window SIZE             # Window around TPs (default: 75)
--easy-ratio RATIO        # Easy negative ratio (default: 0.5)
```

### Output Control

```bash
--output-dir DIR          # Output directory (absolute or relative)
--no-manifest             # Skip gene manifest generation
--patch-dataset           # Run post-build patch scripts
```

### Verbosity

```bash
-v                        # Standard output
-vv                       # Detailed output
-vvv                      # Debug output
```

## Common Workflows

### 1. Build Dataset from Scratch (Complete Example)

When you need to generate predictions for genes that don't have artifacts:

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
- Selects 1000 genes using random sampling (for diversity)
- Filters to protein-coding and lncRNA genes only
- Includes additional genes from `additional_genes.tsv` (e.g., ALS genes)
- Uses smaller batches for memory efficiency (100 genes/batch, 20K rows/batch)
- Runs base model pass to generate SpliceAI predictions
- Generates 3-mer features
- Logs all output to `logs/train_random_1000_3mers.log`

**Memory footprint**: ~2-3 GB (smaller batches reduce peak memory)

**Timeline**:
- Gene selection: < 1 minute
- Base model pass: ~3 hours (1000 genes × ~10 sec/gene)
- Dataset building: ~30 minutes (smaller batches = more I/O)
- **Total**: ~3.5 hours

**For smaller datasets**: Simply adjust `--n-genes`:
```bash
# 100-gene dataset (5-15 minutes total)
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 100 \
    --subset-policy random \
    --gene-types protein_coding lncRNA \
    --gene-ids-file additional_genes.tsv \
    --batch-size 50 \
    --batch-rows 20000 \
    --run-workflow \
    --kmer-sizes 3 \
    --output-dir data/train_random_100_3mers \
    --overwrite \
    --verbose 2>&1 | tee -a logs/train_random_100_3mers.log
```

### 2. Build Dataset with Existing Artifacts

When artifacts already exist and you just want to rebuild the dataset:

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 \
    --subset-policy error_total \
    --kmer-sizes 3 \
    --output-dir data/train_pc_1000_3mers \
    --overwrite \
    -v
```

**Timeline**:
- Gene selection: < 1 minute
- Dataset building: ~20 minutes
- **Total**: ~20 minutes

### 3. Build Dataset with Custom Gene List

Use specific genes of interest (e.g., ALS-related genes):

```bash
# Create gene list file (additional_genes.tsv)
cat > additional_genes.tsv <<EOF
gene_id	gene_name	description
ENSG00000130402	UNC13A	ALS-related gene
ENSG00000075558	STMN2	ALS-related gene
ENSG00000087086	FUS	ALS/FTD-related
ENSG00000120948	TARDBP	Major ALS gene
EOF

# Build dataset with these genes
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 100 \
    --subset-policy error_total \
    --gene-ids-file additional_genes.tsv \
    --gene-col gene_id \
    --run-workflow \
    --kmer-sizes 3 \
    --output-dir data/train_als_100_3mers \
    -vv
```

### 4. Build Dataset with Gene Type Filter

Build dataset using only protein-coding genes:

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 5000 \
    --subset-policy random \
    --gene-types protein_coding \
    --run-workflow \
    --kmer-sizes 3 5 \
    --output-dir data/train_pc_5000_multi_kmers \
    -v
```

### 5. Resumable Build for Long-Running Jobs

For laptop/desktop environments where interruptions are possible:

```bash
# Using tmux (recommended)
./scripts/builder/run_builder_resumable.sh tmux

# Using nohup (background)
./scripts/builder/run_builder_resumable.sh nohup

# Using screen (alternative to tmux)
./scripts/builder/run_builder_resumable.sh screen
```

**Resumability**: 
- Artifacts are written incrementally per chromosome
- Dataset batches are written as separate files
- Can resume by re-running with same parameters (skips existing files)

## Logging Best Practices

### Capture Complete Output

Always log your builder runs for debugging and record-keeping:

```bash
# Using tee (recommended - see output AND save log)
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    [options] \
    --verbose 2>&1 | tee -a logs/my_dataset_build.log

# Using nohup (background job)
nohup python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    [options] \
    --verbose > logs/my_dataset_build.log 2>&1 &
```

**Why use `2>&1 | tee -a`?**:
- `2>&1`: Redirects stderr to stdout (captures errors)
- `| tee -a`: Shows output on screen AND appends to log file
- `-a`: Appends instead of overwriting (preserves previous runs)

### Log Organization

```bash
# Create logs directory
mkdir -p logs

# Use descriptive log names with timestamps
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    [options] \
    2>&1 | tee -a logs/train_${TIMESTAMP}.log
```

## Output Structure

### Training Dataset Directory

```
data/train_random_1000_3mers/
├── batch_00001_raw.parquet      # Temporary batch files
├── batch_00001_trim.parquet     # Downsampled batches
├── batch_00002_trim.parquet
├── ...
├── master/                      # Final training dataset
│   ├── batch_00001.parquet
│   ├── batch_00002.parquet
│   ├── ...
│   └── batch_00010.parquet
└── gene_manifest.csv            # Gene metadata

logs/
└── train_random_1000_3mers.log  # Complete build log
```

### Gene Manifest

The manifest provides metadata about genes in the dataset:

```csv
global_index,gene_id,gene_name,gene_type,chrom,strand,gene_length,start,end,total_splice_sites,donor_sites,acceptor_sites,splice_density_per_kb
0,ENSG00000130402,UNC13A,protein_coding,19,+,119795,20364586,20484381,142,71,71,1.19
1,ENSG00000075558,STMN2,protein_coding,8,+,67234,79582583,79649817,48,24,24,0.71
...
```

**Columns**:
- `global_index`: Sequential index
- `gene_id`: Ensembl gene ID
- `gene_name`: HGNC gene symbol
- `gene_type`: Biotype (protein_coding, lncRNA, etc.)
- `chrom`: Chromosome
- `strand`: + or -
- `gene_length`: Length in base pairs
- `start`, `end`: Genomic coordinates
- `total_splice_sites`: Count of annotated splice sites
- `donor_sites`, `acceptor_sites`: Counts by type
- `splice_density_per_kb`: Sites per kilobase

## Performance Optimization

### Memory Management

**Issue**: Out of memory during building

**Solution**:
```bash
# Reduce batch sizes (both genes and rows)
--batch-size 100 \        # Fewer genes per batch
--batch-rows 20000        # Fewer rows per batch
```

**Memory usage by batch size**:
- `--batch-size 1000, --batch-rows 500000`: ~4-6 GB (default)
- `--batch-size 500, --batch-rows 200000`: ~2-3 GB (moderate)
- `--batch-size 100, --batch-rows 20000`: ~1-2 GB (low memory)

### Speed Optimization

**Issue**: Slow dataset building

**Solutions**:
1. **Use faster k-mer sizes**: Smaller k-mers (3-mers) are faster than larger ones (6-mers)
2. **Increase batch size**: Larger batches reduce I/O overhead (if memory allows)
3. **Skip TN sampling**: Use `--no-tn-sampling` for faster builds (larger datasets)

### Disk Space

**Approximate sizes**:
- Artifacts: ~500 KB/gene → 5000 genes = ~2.5 GB
- Training dataset: ~50 MB/100 genes → 1000 genes = ~500 MB
- Master dataset: ~1-2 GB for 5000 genes (after downsampling)

## Verification

### Check Dataset Validity

```bash
# Count rows in master dataset
python -c "
import polars as pl
df = pl.scan_parquet('data/train_pc_1000_3mers/master/*.parquet')
print(f'Total rows: {df.select(pl.count()).collect().item():,}')
"

# Check schema
python -c "
import polars as pl
df = pl.scan_parquet('data/train_pc_1000_3mers/master/*.parquet')
print('Columns:', df.collect_schema().names())
print('Dtypes:', df.collect_schema().dtypes())
"
```

### Validate Gene Manifest

```bash
# Check gene count
wc -l data/train_pc_1000_3mers/gene_manifest.csv

# Expected: 1001 lines (1000 genes + 1 header)

# Check for ALS genes
grep -E "UNC13A|STMN2|FUS|TARDBP" data/train_pc_1000_3mers/gene_manifest.csv
```

### Verify Base Model Artifacts

```bash
# Check artifact directory
ls -lh data/ensembl/spliceai_analysis/*.tsv

# Count genes in aggregated file
python -c "
import polars as pl
df = pl.read_csv('data/ensembl/spliceai_analysis/full_splice_positions_enhanced.tsv', separator='\t')
print(f'Unique genes: {df[\"gene_id\"].n_unique():,}')
"
```

## Troubleshooting

### Issue: "Missing genes from artifacts"

**Error**:
```
[incremental-builder] Missing 342 requested genes from artifacts
[incremental-builder] CRITICAL: Missing genes must be generated with --run-workflow
```

**Solution**: Add `--run-workflow` flag to generate missing predictions

### Issue: "splice_sites.tsv not found"

**Error**:
```
FileNotFoundError: splice_sites.tsv not found at data/ensembl/splice_sites.tsv
```

**Solution**: Generate splice sites using genomic resources manager:
```bash
python -m meta_spliceai.system.genomic_resources.cli derive --resource splice_sites
```

### Issue: "SpliceAI models not found"

**Error**:
```
FileNotFoundError: data/models/spliceai/spliceai1.h5 not found
```

**Solution**: Verify model installation and symlink:
```bash
# Check if models exist
python -c "from pkg_resources import resource_filename; print(resource_filename('spliceai', 'models/spliceai1.h5'))"

# Create symlink if needed
# (Use output from above command)
ln -s /path/to/spliceai/models data/models/spliceai
```

### Issue: Process killed during workflow

**Symptoms**: Process terminates unexpectedly during base model pass

**Causes**:
1. Out of memory (OOM killer)
2. Laptop went to sleep
3. SSH connection lost

**Solutions**:
1. Use resumable execution: `./scripts/builder/run_builder_resumable.sh tmux`
2. Reduce batch size: `--batch-size 100`
3. Monitor memory: `htop` (ensure >4GB free)
4. Prevent sleep: System settings or `caffeinate` (macOS)

## Next Steps

After successfully building a training dataset:

1. **Verify dataset integrity**:
   ```bash
   python -m meta_spliceai.splice_engine.meta_models.validation.validate_dataset \
       data/train_pc_1000_3mers/master
   ```

2. **Train meta-model**:
   ```bash
   python -m meta_spliceai.splice_engine.meta_models.training.train_meta_model \
       --dataset-dir data/train_pc_1000_3mers/master \
       --output-dir models/meta_model_v1
   ```

3. **Evaluate meta-model**:
   ```bash
   python -m meta_spliceai.splice_engine.meta_models.evaluation.evaluate_meta_model \
       --model-dir models/meta_model_v1 \
       --test-genes additional_genes.tsv
   ```

## References

- **Base Model Pass Workflow**: `docs/base_models/BASE_MODEL_PASS_WORKFLOW.md`
- **Builder Scripts**: `scripts/builder/README.md`
- **Testing Scripts**: `scripts/testing/README.md`
- **Genomic Resources**: `meta_spliceai/system/genomic_resources/README.md`

