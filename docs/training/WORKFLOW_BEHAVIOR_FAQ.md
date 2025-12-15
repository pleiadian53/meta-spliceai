# Workflow Behavior FAQ

## Gene Subset Mode

### Q: Can I run the base model pass on a subset of genes?

**Yes!** The workflow supports gene subsetting through the `target_genes` parameter.

```python
from meta_spliceai import run_base_model_predictions

results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1', 'TP53', 'EGFR'],  # Specific genes
    verbosity=1
)
```

### Q: In gene subset mode, does it load the entire chromosome or just those genes?

**It depends on the size of your gene set:**

#### Small Gene Sets (≤1000 genes) - Pre-loaded Mode

**What happens:**
1. Sequences for all target genes are loaded into memory at once
2. Filtered in memory by gene ID/name
3. Processed directly from memory (no disk I/O during processing)

**Memory Usage:** Moderate (~500MB - 2GB depending on gene sizes)

**Speed:** Fast (no repeated disk access)

**Code reference:**
```python:508:516:meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py
use_preloaded_sequences = (target_genes and 
                           len(target_genes) <= MAX_GENES_FOR_PRELOAD and
                           seq_result.get('sequences_df') is not None and 
                           seq_result['sequences_df'] is not None)

if target_genes and len(target_genes) > MAX_GENES_FOR_PRELOAD:
    if verbosity >= 1:
        print_emphasized(f"[info] Large gene set ({len(target_genes)} genes) - using per-chromosome streaming for memory efficiency")
```

#### Large Gene Sets (>1000 genes) - Streaming Mode

**What happens:**
1. Sequences are loaded per-chromosome from disk
2. Each chromosome is filtered by target genes
3. Processed in chunks (default: 500 genes per chunk)
4. Memory is freed after each chunk

**Memory Usage:** Low (~200-500MB regardless of gene set size)

**Speed:** Slower (repeated disk I/O) but memory-efficient

**Code reference:**
```python:574:587:meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py
else:
    # Original path: load per-chromosome from files
    try:
        # Use the new utility function for lazily loading chromosome sequences
        lazy_seq_df = scan_chromosome_sequence(
            seq_result=seq_result,
            chromosome=chr_,
            format=seq_format,
            separator=separator,
            verbosity=verbosity
        )
    except FileNotFoundError as e:
        print(f"[warning] Sequence file for chr{chr_} not found – skipping. ({e})")
        continue
```

### Q: What if I want to process 100 specific genes across many chromosomes?

**Answer:** The workflow is smart about this:

```python
# This will:
# 1. Pre-load sequences for all 100 genes (fits in memory)
# 2. Identify which chromosomes have these genes
# 3. Only iterate through those chromosomes
# 4. Filter each chromosome to only your 100 genes

results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=[...],  # 100 genes
    verbosity=1
)
```

**Workflow behavior:**
```python:540:546:meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py
# Get unique chromosomes from the filtered data
actual_chromosomes = preloaded_df.select(pl.col('chrom').unique()).to_series().to_list()
if verbosity >= 1:
    print_with_indent(f"[info] Target genes found on {len(actual_chromosomes)} chromosomes: {actual_chromosomes}", indent_level=1)

# Override chromosomes list to only process chromosomes with target genes
chromosomes = actual_chromosomes
```

**Result:** Only chromosomes containing your genes are processed!

## Production Mode Behavior

### Q: If I've run tests on gene subsets, will production mode overwrite them?

**No! Production mode never overwrites existing artifacts.**

The workflow uses an **Artifact Manager** with mode-dependent overwrite policies:

### Test Mode (mode='test')

**Overwrite Policy:** ✅ Always overwrites

**Use case:** Iterative development, debugging, experimentation

**Behavior:**
- Each run overwrites previous test artifacts
- Allows rapid iteration without manual cleanup
- Safe for development

```python
config = BaseModelConfig(
    mode='test',  # Overwrites existing files
    test_name='my_test'
)
```

### Production Mode (mode='production')

**Overwrite Policy:** ❌ Never overwrites (skips existing files)

**Use case:** Final runs, dataset generation, reproducible results

**Behavior:**
- Checks if artifacts already exist
- If they exist, skips saving and uses existing files
- Prevents accidental data loss
- Enables incremental processing

```python
config = BaseModelConfig(
    mode='production',  # Preserves existing files
    test_name='final_run'
)
```

**Code reference:**
```python:911:929:meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py
# ====================================================================
# ARTIFACT MANAGER: Check overwrite policy before saving
# ====================================================================
positions_artifact = artifact_manager.get_artifact_path('full_splice_positions_enhanced.tsv')
errors_artifact = artifact_manager.get_artifact_path('full_splice_errors.tsv')

should_save_positions = artifact_manager.should_overwrite(positions_artifact)
should_save_errors = artifact_manager.should_overwrite(errors_artifact)

if verbosity >= 1:
    if not should_save_positions:
        print_with_indent(
            f"[artifact_manager] Skipping positions save (production mode, file exists): {positions_artifact}",
            indent_level=1
        )
    if not should_save_errors:
        print_with_indent(
            f"[artifact_manager] Skipping errors save (production mode, file exists): {errors_artifact}",
            indent_level=1
        )
```

### Q: What happens to chunk-level files (analysis_sequences_*)?

**Chunk files are ALWAYS written**, regardless of mode.

**Reason:** 
- Chunk files are uniquely named by chromosome and gene range
- Different runs process different chromosomes/genes
- No risk of conflicts
- Enables parallel processing

**Example:**
```
analysis_sequences_1_chunk_1_500.tsv       ← Chr 1, genes 1-500
analysis_sequences_1_chunk_501_1000.tsv    ← Chr 1, genes 501-1000
analysis_sequences_2_chunk_1_500.tsv       ← Chr 2, genes 1-500
```

### Q: Can I force overwrite in production mode?

**Yes, but you need to manually delete the artifact directory:**

```bash
# Option 1: Delete entire artifact directory
rm -rf data/mane/GRCh38/openspliceai_eval/meta_models/

# Option 2: Delete specific artifacts
rm data/mane/GRCh38/openspliceai_eval/meta_models/full_splice_positions_enhanced.tsv
rm data/mane/GRCh38/openspliceai_eval/meta_models/full_splice_errors.tsv

# Option 3: Use test mode (always overwrites)
python scripts/training/run_full_genome_base_model_pass.py \
    --mode test  # Will overwrite
```

## Incremental Processing Workflow

### Scenario: Process Genome Incrementally

You can process chromosomes in multiple runs without conflicts:

**Run 1: Chromosomes 1-5**
```bash
python scripts/training/run_full_genome_base_model_pass.py \
    --chromosomes "1,2,3,4,5" \
    --mode production
```

**Output:**
```
data/mane/GRCh38/openspliceai_eval/meta_models/
├── analysis_sequences_1_chunk_*.tsv   ← Created
├── analysis_sequences_2_chunk_*.tsv   ← Created
├── analysis_sequences_3_chunk_*.tsv   ← Created
├── analysis_sequences_4_chunk_*.tsv   ← Created
├── analysis_sequences_5_chunk_*.tsv   ← Created
├── full_splice_positions_enhanced.tsv ← Created (chr 1-5)
└── full_splice_errors.tsv             ← Created (chr 1-5)
```

**Run 2: Chromosomes 6-10** (same directory)
```bash
python scripts/training/run_full_genome_base_model_pass.py \
    --chromosomes "6,7,8,9,10" \
    --mode production
```

**Output:**
```
data/mane/GRCh38/openspliceai_eval/meta_models/
├── analysis_sequences_1_chunk_*.tsv   ← From Run 1
├── analysis_sequences_2_chunk_*.tsv   ← From Run 1
├── ...
├── analysis_sequences_6_chunk_*.tsv   ← NEW from Run 2
├── analysis_sequences_7_chunk_*.tsv   ← NEW from Run 2
├── analysis_sequences_8_chunk_*.tsv   ← NEW from Run 2
├── analysis_sequences_9_chunk_*.tsv   ← NEW from Run 2
├── analysis_sequences_10_chunk_*.tsv  ← NEW from Run 2
├── full_splice_positions_enhanced.tsv ← SKIPPED (exists from Run 1)
└── full_splice_errors.tsv             ← SKIPPED (exists from Run 1)
```

**⚠️ Note:** The aggregated files (`full_splice_positions_enhanced.tsv`) will only contain data from Run 1 because production mode doesn't overwrite. To get complete aggregated data:

**Option A: Use separate test_names**
```bash
# Run 1
python scripts/training/run_full_genome_base_model_pass.py \
    --chromosomes "1,2,3,4,5" \
    --mode production \
    --config.test_name openspliceai_chr1_5

# Run 2
python scripts/training/run_full_genome_base_model_pass.py \
    --chromosomes "6,7,8,9,10" \
    --mode production \
    --config.test_name openspliceai_chr6_10
```

**Option B: Aggregate manually later**
```python
import polars as pl
from pathlib import Path

# Read all chunk files
chunk_files = sorted(Path("data/mane/GRCh38/openspliceai_eval/meta_models/").glob("analysis_sequences_*_chunk_*.tsv"))

all_data = []
for file in chunk_files:
    df = pl.read_csv(file, separator='\t')
    all_data.append(df)

# Combine all chunks
combined = pl.concat(all_data)
print(f"Total positions: {combined.height:,}")
```

## Summary

### Gene Subset Mode
- **≤1000 genes**: Pre-loaded in memory, fast
- **>1000 genes**: Streamed per-chromosome, memory-efficient
- Only chromosomes with target genes are processed
- Automatically optimized based on gene set size

### Production Mode
- **Never overwrites** aggregated artifacts
- **Always writes** chunk-level files (uniquely named)
- Enables safe incremental processing
- Use separate `test_name` for parallel runs

### Best Practices
1. Use **test mode** for development/debugging
2. Use **production mode** for final data generation
3. Use **unique test_names** for different experiments
4. Use **chromosome selection** for memory-constrained systems
5. **Manually aggregate** chunks if doing incremental processing

## See Also

- [Chromosome Selection Guide](CHROMOSOME_SELECTION_GUIDE.md)
- [Full Genome Pass Readiness](FULL_GENOME_PASS_READINESS.md)
- [Artifact Management](../development/ARTIFACT_MANAGEMENT.md)




