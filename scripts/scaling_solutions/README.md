# MetaSpliceAI Scaling Solutions

This directory contains tools and solutions for scaling the incremental builder and other memory-intensive workflows in the meta-spliceai pipeline.

## Directory Structure

```
scaling_solutions/
├── README.md                           # This file
├── memory_optimization/                # Memory management tools
│   ├── swap_setup.sh                   # Swap file configuration
│   ├── low_memory_build.sh            # Ultra-conservative build settings
│   └── memory_configs/                 # Pre-configured memory settings
│       ├── conservative.json           # For 1K-5K genes
│       ├── moderate.json              # For 5K-10K genes  
│       └── aggressive.json            # For 10K+ genes
├── monitoring/                         # Performance monitoring tools
│   ├── memory_profile_builder.py      # Real-time memory monitoring
│   ├── performance_monitor.sh         # System resource monitoring
│   └── build_progress_tracker.py      # Progress tracking and resumption
├── utilities/                          # Helper utilities
│   ├── split_gene_build.py            # Split large gene lists
│   ├── resume_failed_build.py         # Resume interrupted builds
│   └── dataset_validator.py           # Validate completed datasets
└── documentation/                      # Guides and best practices
    ├── scaling_strategies.md          # When to use which approach
    ├── memory_troubleshooting.md      # Common issues and solutions
    └── performance_benchmarks.md      # Expected performance metrics
```

## Quick Start

### For Small Datasets (1K-5K genes)
```bash
# Use conservative settings
./scripts/scaling_solutions/memory_optimization/low_memory_build.sh
```

### For Medium Datasets (5K-10K genes)
```bash
# Set up swap first
./scripts/scaling_solutions/memory_optimization/swap_setup.sh

# Then run with moderate settings
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 5000 --batch-size 200 --batch-rows 15000 \
    --kmer-sizes 3 --output-dir train_pc_5000
```

### For Large Datasets (10K+ genes)
```bash
# Split into manageable chunks
python scripts/scaling_solutions/utilities/split_gene_build.py genes.tsv 4

# Monitor progress
python scripts/scaling_solutions/monitoring/memory_profile_builder.py \
    python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
    --n-genes 20000 --batch-size 500 --batch-rows 25000
```

## Problem-Specific Solutions

| Problem | Solution | Tool |
|---------|----------|------|
| OOM kills during k-mer generation | Add swap space | `swap_setup.sh` |
| Large gene lists (>10K genes) | Split and combine | `split_gene_build.py` |
| Need to resume failed builds | Resume functionality | `resume_failed_build.py` |
| Memory monitoring during builds | Real-time tracking | `memory_profile_builder.py` |
| Ultra-conservative memory usage | Pre-configured settings | `low_memory_build.sh` |

## Memory Requirements by Dataset Size

| Dataset Size | Recommended RAM | Swap Needed | Batch Size | Batch Rows |
|--------------|-----------------|-------------|------------|------------|
| 1K genes     | 16GB+          | Optional    | 100        | 10K        |
| 5K genes     | 32GB+          | 8GB         | 200        | 15K        |
| 10K genes    | 46GB+          | 16GB        | 500        | 20K        |
| 20K+ genes   | 64GB+          | 32GB        | 1000       | 25K        |

## Best Practices

1. **Always start with swap setup** for datasets >5K genes
2. **Monitor memory usage** during first few batches
3. **Use 3-mer features** instead of 6-mer when possible
4. **Split large gene lists** for datasets >10K genes
5. **Resume failed builds** rather than starting over

## Contributing

When adding new scaling solutions:

1. Place tools in appropriate subdirectory
2. Update this README with usage examples
3. Add documentation in `documentation/`
4. Test with different dataset sizes
5. Include memory usage benchmarks 