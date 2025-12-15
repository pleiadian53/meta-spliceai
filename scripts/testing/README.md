# Testing Scripts

This directory contains scripts for testing various components of the meta-spliceai system.

## Scripts

### `test_incremental_builder.sh`

Quick test of the incremental training dataset builder with a small dataset.

**Purpose**: Verify that all command-line argument logic works correctly before running large-scale dataset builds.

**Features**:
- Tests with only 100 genes (fast execution)
- Includes ALS-related genes from `additional_genes.tsv`
- Tests all major parameters
- Comprehensive logging
- Pre-flight checks

**Usage:**
```bash
cd /Users/pleiadian53/work/meta-spliceai
./scripts/testing/test_incremental_builder.sh
```

**What it tests:**
- `--n-genes` parameter
- `--subset-policy` gene selection
- `--gene-ids-file` custom gene list
- `--batch-size` and `--batch-rows` memory management
- `--kmer-sizes` feature extraction
- `--output-dir` custom output location
- `--overwrite` flag
- Verbose logging with `-vv`

**Output:**
- Dataset: `data/train_test_100genes_3mers/`
- Log file: `logs/incremental_builder_test_YYYYMMDD_HHMMSS.log`

**Expected Duration**: 5-15 minutes (depending on hardware)

## Testing Workflow

### 1. Quick Smoke Test (Recommended First)

```bash
# Test with minimal dataset
./scripts/testing/test_incremental_builder.sh
```

Verifies:
- ✅ Environment setup
- ✅ Import resolution
- ✅ Path resolution (GTF, FASTA, splice sites)
- ✅ Gene selection logic
- ✅ Batch processing
- ✅ K-mer extraction
- ✅ Feature enrichment
- ✅ Output generation

### 2. Medium-Scale Test

```bash
# Edit scripts/builder/run_builder_resumable.sh
# Change N_GENES=1000 to N_GENES=500

./scripts/builder/run_builder_resumable.sh direct
```

Verifies:
- ✅ Performance at moderate scale
- ✅ Memory usage patterns
- ✅ Processing time estimation

### 3. Production Run

```bash
# Use full 1000 genes with resumable execution
./scripts/builder/run_builder_resumable.sh tmux

# Detach: Ctrl+b then d
# Reattach anytime: tmux attach -t builder
```

## Test Files

### `additional_genes.tsv`

Located at project root, contains priority genes to include:

```tsv
gene_id              gene_name  description
ENSG00000130402     UNC13A     ALS-related gene
ENSG00000075558     STMN2      ALS-related gene  
ENSG00000087086     FUS        ALS/FTD-related
ENSG00000120948     TARDBP     TAR DNA binding protein (TDP-43)
ENSG00000127334     PRDM9      Complex splicing test gene
ENSG00000184009     ACTB       Housekeeping gene control
```

## Validation Checks

After running tests, validate output:

```bash
# Check dataset was created
ls -lh data/train_test_100genes_3mers/

# Inspect gene manifest
head data/train_test_100genes_3mers/gene_manifest.csv

# Count rows in dataset
python -c "
import polars as pl
df = pl.scan_parquet('data/train_test_100genes_3mers/master/**/*.parquet')
print(f'Total rows: {df.select(pl.count()).collect().item():,}')
print(f'Columns: {df.columns}')
"

# Check for priority genes
grep -E 'UNC13A|STMN2|TARDBP' data/train_test_100genes_3mers/gene_manifest.csv
```

## Common Test Issues

### Issue: "splice_sites.tsv not found"

**Solution**: Verify genomic resources are set up
```bash
python -m meta_spliceai.system.genomic_resources.cli audit
# Should show 8/8 resources found
```

### Issue: "No genes with splice sites found"

**Solution**: Check gene types
```bash
# Use protein_coding genes only
./test_incremental_builder.sh --gene-types protein_coding
```

### Issue: Memory errors

**Solution**: Reduce batch parameters
```bash
# Edit test_incremental_builder.sh
BATCH_SIZE=25      # Down from 50
BATCH_ROWS=10000   # Down from 20000
```

### Issue: Import errors

**Solution**: Verify environment
```bash
mamba activate surveyor
python -c "import meta_spliceai; print(meta_spliceai.__file__)"
```

## Test Results Interpretation

### Success Indicators

✅ **Test Passed** if you see:
- "✅ SUCCESS: Incremental builder completed"
- gene_manifest.csv created
- master/ directory with parquet files
- Log shows all batches processed
- No error messages in log

⚠️ **Partial Success** if:
- Dataset created but missing some genes
- Warnings about missing features (can be patched later)
- Performance warnings (expected on laptop)

❌ **Test Failed** if:
- Process exits with error code
- No output files created
- "FAILED" message in log
- Critical errors about missing files

## Performance Benchmarks

Expected processing times (M1 MacBook Pro):

| Genes | Batch Size | K-mer Size | Time      | Memory  |
|-------|------------|------------|-----------|---------|
| 100   | 50         | 3          | 5-10 min  | 2-4 GB  |
| 500   | 100        | 3          | 20-30 min | 4-6 GB  |
| 1000  | 100        | 3          | 40-60 min | 6-8 GB  |
| 1000  | 100        | 3,6        | 60-90 min | 8-12 GB |

## Integration Testing

The incremental builder integrates multiple subsystems:

1. **Genomic Resources Manager** → Path resolution
2. **Gene Selection** → Subset analysis sequences  
3. **Splice Prediction Workflow** → Base model predictions
4. **Feature Extraction** → K-mer and structural features
5. **Feature Enrichment** → Gene/transcript metadata
6. **Downsampling** → True negative reduction
7. **Arrow Dataset** → Efficient storage

Test verifies all integrations work correctly.

## Related Documentation

- Builder scripts: `scripts/builder/README.md`
- Incremental builder docs: `meta_spliceai/splice_engine/meta_models/builder/docs/`
- Workflow comparison: `meta_spliceai/splice_engine/meta_models/builder/docs/workflow_comparison.md`

