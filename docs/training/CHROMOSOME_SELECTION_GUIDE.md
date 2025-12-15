# Chromosome Selection Guide

## Overview

The base model pass now supports chromosome-level selection, allowing you to process specific chromosomes instead of the entire genome. This is particularly useful for:
- **Memory-constrained systems** (process one chromosome at a time)
- **Testing workflows** (validate on chromosome 21 or 22)
- **Parallel processing** (run multiple chromosomes in parallel on different machines)
- **Incremental analysis** (process chromosomes individually and aggregate later)

## Usage

### Run Full Genome Pass on Specific Chromosomes

```bash
# Single chromosome
python scripts/training/run_full_genome_base_model_pass.py \
    --base-model openspliceai \
    --mode production \
    --coverage full_genome \
    --chromosomes 1

# Multiple chromosomes
python scripts/training/run_full_genome_base_model_pass.py \
    --base-model openspliceai \
    --mode production \
    --coverage full_genome \
    --chromosomes "1,2,X"

# All chromosomes (default)
python scripts/training/run_full_genome_base_model_pass.py \
    --base-model openspliceai \
    --mode production \
    --coverage full_genome
```

### Run in Background (Recommended for Long Jobs)

```bash
# Process chromosome 1 in background
nohup python scripts/training/run_full_genome_base_model_pass.py \
    --base-model openspliceai \
    --mode production \
    --coverage full_genome \
    --chromosomes 1 \
    2>&1 | tee logs/openspliceai_chr1_$(date +%Y%m%d_%H%M%S).log &

# Check progress
tail -f logs/openspliceai_chr1_*.log
```

### Process All Chromosomes Sequentially

```bash
#!/bin/bash
# Process chromosomes 1-22, X, Y sequentially
for chr in {1..22} X Y; do
    echo "===================="
    echo "Processing chromosome $chr..."
    echo "===================="
    
    python scripts/training/run_full_genome_base_model_pass.py \
        --base-model openspliceai \
        --mode production \
        --coverage full_genome \
        --chromosomes $chr \
        2>&1 | tee logs/openspliceai_chr${chr}_$(date +%Y%m%d_%H%M%S).log
    
    if [ $? -eq 0 ]; then
        echo "✅ Chromosome $chr completed successfully"
    else
        echo "❌ Chromosome $chr failed"
        exit 1
    fi
done

echo "🎉 All chromosomes processed successfully!"
```

## Memory Efficiency

### Per-Chromosome Processing

Processing chromosomes individually is significantly more memory-efficient:

| Mode | Memory Usage | Time per Chr | Total Time (24 chroms) |
|------|-------------|--------------|------------------------|
| **Full genome** | ~32GB | N/A | ~5 days |
| **Per-chromosome** | ~2-4GB | ~5 hours | ~5 days (sequential) |
| **Per-chromosome (parallel)** | ~2-4GB × N jobs | ~5 hours | ~5 hours (24 parallel jobs) |

**Recommendation for 16GB RAM systems**: Process chromosomes sequentially (1 at a time)

### Estimated Times

Based on MANE/GRCh38 dataset (~19,226 genes):

| Chromosome | Genes | Estimated Time |
|------------|-------|----------------|
| Chr 1 | 1,994 | ~5 hours |
| Chr 2 | 1,241 | ~3 hours |
| Chr X | 803 | ~2 hours |
| Chr 21 | 227 | ~30 minutes |
| Chr 22 | 423 | ~1 hour |

## Python API Usage

### Using run_base_model_predictions

```python
from meta_spliceai import run_base_model_predictions, BaseModelConfig

# Single chromosome
config = BaseModelConfig(
    base_model='openspliceai',
    mode='production',
    coverage='full_genome',
    test_name='openspliceai_chr1'
)

results = run_base_model_predictions(
    base_model='openspliceai',
    target_chromosomes=['1'],
    config=config,
    verbosity=1
)

# Multiple chromosomes
results = run_base_model_predictions(
    base_model='openspliceai',
    target_chromosomes=['1', '2', 'X'],
    config=config,
    verbosity=1
)
```

### Loop Through Chromosomes

```python
from meta_spliceai import run_base_model_predictions, BaseModelConfig
from pathlib import Path
import json

# Process each chromosome and save results
chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y']

for chrom in chromosomes:
    print(f"Processing chromosome {chrom}...")
    
    config = BaseModelConfig(
        base_model='openspliceai',
        mode='production',
        coverage='full_genome',
        test_name=f'openspliceai_chr{chrom}'
    )
    
    results = run_base_model_predictions(
        base_model='openspliceai',
        target_chromosomes=[chrom],
        config=config,
        verbosity=1
    )
    
    # Save summary
    summary = {
        'chromosome': chrom,
        'genes': results['positions']['gene_id'].n_unique(),
        'positions': results['positions'].height,
        'success': results['success']
    }
    
    summary_file = Path(f"results/chr{chrom}_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Chromosome {chrom} complete: {summary['genes']} genes, {summary['positions']} positions")
```

## Outputs

### File Organization

When processing chromosomes individually, outputs are organized by chromosome:

```
data/mane/GRCh38/openspliceai_eval/meta_models/
├── analysis_sequences_1_chunk_1_500.tsv
├── analysis_sequences_1_chunk_501_1000.tsv
├── analysis_sequences_1_chunk_1001_1500.tsv
├── analysis_sequences_1_chunk_1501_1994.tsv
├── full_splice_positions_enhanced.tsv
├── full_splice_errors.tsv
├── gene_manifest.tsv
└── evaluation_metrics.json
```

### Aggregating Multiple Chromosome Runs

If you process chromosomes separately, you can aggregate them later:

```python
import polars as pl
from pathlib import Path

# Aggregate positions from multiple chromosome runs
positions_files = list(Path("data/mane/GRCh38/openspliceai_eval/").glob("*/full_splice_positions_enhanced.tsv"))

all_positions = []
for file in positions_files:
    df = pl.read_csv(file, separator='\t')
    all_positions.append(df)

# Combine all positions
combined_positions = pl.concat(all_positions)
print(f"Total positions: {combined_positions.height:,}")
print(f"Total genes: {combined_positions['gene_id'].n_unique():,}")
```

## Best Practices

### For Memory-Constrained Systems (≤16GB RAM)

1. **Process one chromosome at a time**:
   ```bash
   for chr in {1..22} X Y; do
       python scripts/training/run_full_genome_base_model_pass.py \
           --base-model openspliceai \
           --chromosomes $chr \
           --mode production
   done
   ```

2. **Start with small chromosomes** (21, 22) to validate the workflow

3. **Monitor memory usage** during processing:
   ```bash
   watch -n 5 'ps aux | grep python | grep run_full_genome'
   ```

### For Systems with Adequate RAM (≥32GB)

1. **Process 2-3 chromosomes** in a single run:
   ```bash
   python scripts/training/run_full_genome_base_model_pass.py \
       --chromosomes "1,2,3"
   ```

2. **Or process the full genome** at once (fastest, but requires most memory)

### For Cluster/Cloud Environments

1. **Run chromosomes in parallel** on different nodes/instances
2. **Aggregate results** after all jobs complete
3. Use job schedulers (SLURM, PBS) to manage chromosome-level jobs

## Troubleshooting

### Memory Errors

**Symptom**: Process becomes "stuck" or system swaps heavily

**Solution**: 
- Kill the process: `kill -9 <PID>`
- Switch to per-chromosome mode
- Process smaller chromosomes first to test

### Incomplete Outputs

**Symptom**: Some chromosomes missing from outputs

**Solution**:
- Check logs for each chromosome
- Rerun failed chromosomes individually
- Verify input data availability for each chromosome

## Examples

### Example 1: Test on Chromosome 21 (Fast)

```bash
# ~30 minutes
python scripts/training/run_full_genome_base_model_pass.py \
    --base-model openspliceai \
    --chromosomes 21 \
    --mode test
```

### Example 2: Process Large Chromosomes Individually

```bash
# Chromosome 1 (~5 hours)
nohup python scripts/training/run_full_genome_base_model_pass.py \
    --base-model openspliceai \
    --chromosomes 1 \
    --mode production \
    > logs/chr1.log 2>&1 &

# After chr1 completes, run chr2
nohup python scripts/training/run_full_genome_base_model_pass.py \
    --base-model openspliceai \
    --chromosomes 2 \
    --mode production \
    > logs/chr2.log 2>&1 &
```

### Example 3: Sex Chromosomes Only

```bash
python scripts/training/run_full_genome_base_model_pass.py \
    --base-model openspliceai \
    --chromosomes "X,Y" \
    --mode production
```

## See Also

- [Full Genome Pass Readiness](FULL_GENOME_PASS_READINESS.md)
- [Full Genome Pass Outputs](FULL_GENOME_PASS_OUTPUTS.md)
- [Base Model Comparison Guide](../base_models/BASE_MODEL_COMPARISON_GUIDE.md)




