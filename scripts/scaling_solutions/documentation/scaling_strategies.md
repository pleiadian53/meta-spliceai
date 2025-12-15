# Scaling Strategies for Incremental Builder

This document provides comprehensive guidance on scaling the incremental builder for different dataset sizes and system configurations.

## Overview

The incremental builder can handle datasets from 1K to 20K+ genes, but requires different strategies based on:
- **Dataset size** (number of genes)
- **Available system resources** (RAM, CPU, disk)
- **Performance requirements** (speed vs. reliability)

## Strategy Selection Matrix

| Dataset Size | System RAM | Recommended Strategy | Expected Time | Risk Level |
|--------------|------------|---------------------|---------------|------------|
| 1K-5K genes | 16GB+ | Conservative | 2-4 hours | Low |
| 5K-10K genes | 32GB+ | Moderate + Swap | 4-8 hours | Medium |
| 10K-15K genes | 46GB+ | Split Build | 8-16 hours | Medium |
| 15K+ genes | 64GB+ | Aggressive | 12-24 hours | High |

## Strategy Details

### 1. Conservative Strategy (1K-5K genes)

**Best for:** Small datasets, limited RAM, high reliability requirements

**Configuration:**
```bash
# Use pre-configured conservative settings
./scripts/scaling_solutions/memory_optimization/low_memory_build.sh

# Or manual configuration
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 \
    --batch-size 50 \
    --batch-rows 5000 \
    --kmer-sizes 3 \
    --output-dir train_pc_1000_conservative
```

**Memory Profile:**
- Peak RAM: 8-12GB
- Swap: Optional
- Processing: Slow but reliable

### 2. Moderate Strategy (5K-10K genes)

**Best for:** Medium datasets, adequate RAM, balanced performance

**Prerequisites:**
```bash
# Set up swap space first
./scripts/scaling_solutions/memory_optimization/swap_setup.sh
```

**Configuration:**
```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 5000 \
    --batch-size 200 \
    --batch-rows 15000 \
    --kmer-sizes 3 \
    --output-dir train_pc_5000_moderate
```

**Memory Profile:**
- Peak RAM: 20-30GB
- Swap: 8-16GB recommended
- Processing: Balanced speed/reliability

### 3. Split Build Strategy (10K+ genes)

**Best for:** Large datasets, any RAM configuration, maximum reliability

**Process:**
```bash
# 1. Split gene list into manageable chunks
python scripts/scaling_solutions/utilities/split_gene_build.py genes.tsv 4

# 2. Build each chunk separately
# (Script handles this automatically)

# 3. Combine results
# (Script handles this automatically)
```

**Memory Profile:**
- Peak RAM: 15-25GB per chunk
- Swap: 8GB per chunk
- Processing: Reliable but slower overall

### 4. Aggressive Strategy (15K+ genes, 64GB+ RAM)

**Best for:** Large datasets, high RAM systems, maximum speed

**Configuration:**
```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 20000 \
    --batch-size 500 \
    --batch-rows 25000 \
    --kmer-sizes 3 4 \
    --output-dir train_pc_20000_aggressive
```

**Memory Profile:**
- Peak RAM: 40-50GB
- Swap: 16-32GB recommended
- Processing: Fast but higher risk

## Memory Configuration Files

Pre-configured settings are available in `scripts/scaling_solutions/memory_optimization/memory_configs/`:

- `conservative.json` - For 1K-5K genes
- `moderate.json` - For 5K-10K genes  
- `aggressive.json` - For 10K+ genes

## Monitoring and Troubleshooting

### Real-time Monitoring
```bash
# Monitor during build
python scripts/scaling_solutions/monitoring/memory_profile_builder.py \
    python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 --batch-size 200 --kmer-sizes 3

# System-level monitoring
./scripts/scaling_solutions/monitoring/performance_monitor.sh monitor &
```

### Resuming Failed Builds
```bash
# Analyze failed build
python scripts/scaling_solutions/utilities/resume_failed_build.py \
    --analyze train_pc_1000_3mers

# Resume with original command
python scripts/scaling_solutions/utilities/resume_failed_build.py \
    --resume train_pc_1000_3mers \
    python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 --batch-size 250 --kmer-sizes 3
```

## Common Issues and Solutions

### Issue: OOM Kills
**Symptoms:** Process terminated with "Killed" message
**Solutions:**
1. Add swap space: `./scripts/scaling_solutions/memory_optimization/swap_setup.sh`
2. Reduce batch size: `--batch-size 100` instead of 250
3. Reduce batch rows: `--batch-rows 10000` instead of 20000
4. Use 3-mer features: `--kmer-sizes 3` instead of 6

### Issue: Slow Processing
**Symptoms:** Build takes much longer than expected
**Solutions:**
1. Increase batch size (if RAM allows)
2. Increase batch rows (if RAM allows)
3. Use more CPU threads: `export POLARS_MAX_THREADS=8`
4. Check disk I/O: `iostat -x 1`

### Issue: Disk Space Exhaustion
**Symptoms:** "No space left on device" errors
**Solutions:**
1. Check available space: `df -h`
2. Clean up temporary files: `rm -rf train_pc_*/batch_*_raw.parquet`
3. Use different output directory with more space
4. Monitor disk usage during build

## Performance Benchmarks

### Expected Performance by Strategy

| Strategy | Genes | RAM | Time | Success Rate |
|----------|-------|-----|------|--------------|
| Conservative | 1K | 16GB | 2h | 99% |
| Conservative | 5K | 16GB | 8h | 95% |
| Moderate | 5K | 32GB | 4h | 98% |
| Moderate | 10K | 32GB | 8h | 90% |
| Split Build | 10K | 16GB | 12h | 99% |
| Aggressive | 20K | 64GB | 12h | 85% |

### Memory Usage Patterns

- **K-mer generation**: Highest memory usage (3-5x input size)
- **Feature enrichment**: Moderate memory usage (1.5-2x input size)
- **TN downsampling**: Low memory usage (1.2x input size)
- **File I/O**: Minimal memory usage

## Best Practices

1. **Always start small**: Test with 1K genes before scaling up
2. **Monitor first few batches**: Watch memory usage patterns
3. **Use swap for >5K genes**: Prevents OOM kills
4. **Split large datasets**: More reliable than single large builds
5. **Resume failed builds**: Don't start over from scratch
6. **Clean up after completion**: Remove temporary files to save disk space

## Advanced Techniques

### Parallel Processing
For very large datasets, consider running multiple smaller builds in parallel:

```bash
# Split into 4 chunks and run in parallel
python scripts/scaling_solutions/utilities/split_gene_build.py genes.tsv 4

# Each chunk can run on different machines/containers
# Results can be combined later
```

### Incremental Expansion
Build datasets incrementally to test different gene subsets:

```bash
# Start with 1K genes
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 --output-dir train_pc_1000

# Add 1K more genes
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 2000 --output-dir train_pc_2000

# Continue until desired size
```

### Resource Optimization
- **SSD storage**: 5-10x faster than HDD for swap
- **High RAM systems**: Can use larger batch sizes
- **Multi-core systems**: Increase `POLARS_MAX_THREADS`
- **Network storage**: Consider local storage for temporary files 