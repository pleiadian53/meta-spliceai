# Quick Reference: Scaling Solutions

## 🚀 Immediate Solutions for Your Use Cases

### **Recommended: Flexible Build (All Sizes)**
```bash
# Auto-detect settings and reuse directories
./scripts/scaling_solutions/memory_optimization/flexible_build.sh --n-genes 1000 --auto

# Overwrite existing failed build
./scripts/scaling_solutions/memory_optimization/flexible_build.sh --n-genes 1000 --strategy conservative --overwrite

# Resume interrupted build
./scripts/scaling_solutions/memory_optimization/flexible_build.sh --n-genes 5000 --resume
```

### For 1K Genes (Conservative)
```bash
# Ultra-safe build
./scripts/scaling_solutions/memory_optimization/low_memory_build.sh
```

### For 5K Genes (Moderate + Swap)
```bash
# 1. Set up swap
./scripts/scaling_solutions/memory_optimization/swap_setup.sh

# 2. Run build
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 5000 --batch-size 200 --batch-rows 15000 \
    --kmer-sizes 3 --output-dir train_pc_5000
```

### For 10K+ Genes (Split Build)
```bash
# Automatic split and build
python scripts/scaling_solutions/utilities/split_gene_build.py genes.tsv 4
```

### For 20K+ Genes (Aggressive)
```bash
# High-performance build
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 20000 --batch-size 500 --batch-rows 25000 \
    --kmer-sizes 3 4 --output-dir train_pc_20000
```

## 🔧 Monitoring Tools

### Real-time Memory Monitoring
```bash
python scripts/scaling_solutions/monitoring/memory_profile_builder.py \
    python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 --batch-size 200 --kmer-sizes 3
```

### System Performance Monitoring
```bash
./scripts/scaling_solutions/monitoring/performance_monitor.sh monitor &
```

## 🛠️ Troubleshooting

### Resume Failed Build
```bash
# Analyze what happened
python scripts/scaling_solutions/utilities/resume_failed_build.py \
    --analyze train_pc_1000_3mers

# Resume with original command
python scripts/scaling_solutions/utilities/resume_failed_build.py \
    --resume train_pc_1000_3mers \
    python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 --batch-size 250 --kmer-sizes 3
```

### Quick Swap Setup
```bash
# Temporary swap (no permanent changes)
sudo fallocate -l 8G /tmp/swapfile
sudo chmod 600 /tmp/swapfile
sudo mkswap /tmp/swapfile
sudo swapon /tmp/swapfile

# Verify
swapon --show
```

## 📊 Memory Requirements Quick Guide

| Dataset | RAM Needed | Swap Needed | Strategy |
|---------|------------|-------------|----------|
| 1K genes | 16GB+ | Optional | Conservative |
| 5K genes | 32GB+ | 8GB | Moderate |
| 10K genes | 46GB+ | 16GB | Split Build |
| 20K+ genes | 64GB+ | 32GB | Aggressive |

## ⚡ Performance Tips

1. **Use 3-mer features** instead of 6-mer (64x memory savings)
2. **Add swap space** for datasets >5K genes
3. **Monitor first few batches** to catch issues early
4. **Resume failed builds** instead of starting over
5. **Split large datasets** for maximum reliability

## 🆘 Emergency Solutions

### If OOM Killed:
```bash
# 1. Add swap immediately
sudo fallocate -l 8G /tmp/swapfile && sudo chmod 600 /tmp/swapfile && sudo mkswap /tmp/swapfile && sudo swapon /tmp/swapfile

# 2. Resume with smaller batches
python scripts/scaling_solutions/utilities/resume_failed_build.py --resume train_pc_1000_3mers \
    python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 1000 --batch-size 100 --batch-rows 10000 --kmer-sizes 3
```

### If Very Slow:
```bash
# Check system resources
./scripts/scaling_solutions/monitoring/performance_monitor.sh analyze

# Increase parallelism (if RAM allows)
export POLARS_MAX_THREADS=8
```

### If Disk Full:
```bash
# Clean up temporary files
find . -name "batch_*_raw.parquet" -delete

# Check space
df -h
```

## 📁 Directory Structure

```
scripts/scaling_solutions/
├── README.md                           # Complete documentation
├── QUICK_REFERENCE.md                  # This file
├── memory_optimization/                # Memory management
│   ├── swap_setup.sh                   # Swap configuration
│   ├── low_memory_build.sh            # Conservative builds
│   └── memory_configs/                 # Pre-configured settings
├── monitoring/                         # Performance tools
│   ├── memory_profile_builder.py      # Memory monitoring
│   └── performance_monitor.sh         # System monitoring
├── utilities/                          # Helper tools
│   ├── split_gene_build.py            # Split large datasets
│   └── resume_failed_build.py         # Resume failed builds
└── documentation/                      # Detailed guides
    └── scaling_strategies.md          # Complete strategies
```

## 🎯 Recommended Workflow

1. **Start with conservative settings** for new dataset sizes
2. **Monitor first few batches** to understand memory patterns
3. **Scale up gradually** once you understand the requirements
4. **Use appropriate strategy** based on dataset size and system resources
5. **Keep monitoring tools running** during long builds
6. **Resume rather than restart** if builds fail

## 📞 Need Help?

- Check `scripts/scaling_solutions/documentation/scaling_strategies.md` for detailed guides
- Use monitoring tools to identify bottlenecks
- Start with smaller datasets to test configurations
- Consider split builds for very large datasets 