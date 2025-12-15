# Dataset Builder Scripts

This directory contains scripts for building training datasets for the meta-model.

## Scripts

### `run_builder_resumable.sh`

Runs the incremental training dataset builder in a resumable way that survives:
- Laptop sleep/standby mode
- Terminal closure
- SSH disconnections

**Usage:**
```bash
# Run with nohup (simplest, process continues in background)
./run_builder_resumable.sh nohup

# Run in tmux session (recommended, can reattach)
./run_builder_resumable.sh tmux

# Run in screen session (alternative to tmux)
./run_builder_resumable.sh screen

# Run directly (for testing, not resumable)
./run_builder_resumable.sh direct
```

**Monitoring:**
```bash
# Monitor log output
tail -f logs/incremental_builder_*.log

# Check if process is running
ps aux | grep incremental_builder

# Reattach to tmux session
tmux attach -t builder

# Reattach to screen session
screen -r builder
```

**Configuration:**
Edit the script to customize:
- `N_GENES`: Number of genes (default: 1000)
- `SUBSET_POLICY`: Gene selection strategy (default: error_total)
- `BATCH_SIZE`: Genes per batch (default: 100)
- `KMER_SIZES`: K-mer sizes to extract (default: 3)
- `OUTPUT_DIR`: Where to save dataset

## Key Features

### Resumable Execution
- **nohup**: Process continues after terminal closure, uses minimal resources
- **tmux**: Can detach/reattach anytime, view live output
- **screen**: Alternative to tmux with similar features

### Enhanced Splice Sites Integration
The builder automatically uses `splice_sites_enhanced.tsv` when available via the Genomic Resources Manager Registry. This provides:
- Three-class probability scores (donor, acceptor, neither)
- 24 contextual sequence features
- Gene name annotations
- Enhanced position metadata

### Batch Processing
Processes genes in batches to:
- Manage memory usage
- Enable checkpoint/resume capability
- Provide incremental progress updates

### Custom Gene Lists
Use `additional_genes.tsv` to ensure specific genes (e.g., ALS-related genes) are included in the dataset:
```bash
# Format: gene_id, gene_name, description (TSV)
ENSG00000130402    UNC13A    ALS-related gene
ENSG00000075558    STMN2     ALS-related gene
```

## Workflow Integration

The incremental builder integrates with:

1. **Genomic Resources Manager**: Automatically resolves paths to GTF, FASTA, and splice sites
2. **Splice Prediction Workflow**: Can run base model predictions first with `--run-workflow`
3. **Feature Enrichment**: Automatically enriches with gene/transcript/structural features
4. **K-mer Extraction**: Generates k-mer features for sequence analysis

## Output Structure

```
data/train_pc_1000_3mers/
├── master/                      # Master Arrow dataset
│   ├── chr1/                    # Partitioned by chromosome
│   │   └── *.parquet
│   ├── chr2/
│   └── ...
├── gene_manifest.csv            # Gene characteristics and metadata
├── batch_000_genes.txt          # List of genes in batch 0
├── batch_000_raw.parquet        # Raw batch 0 (before enrichment)
├── batch_000_enriched.parquet   # Enriched batch 0
└── batch_000_downsampled.parquet # Final balanced batch 0
```

## Troubleshooting

### Process Died or Interrupted

If the process stops unexpectedly:

1. Check the log file: `tail -f logs/incremental_builder_*.log`
2. Look for error messages
3. Check disk space: `df -h`
4. Check memory usage: `htop` or `top`

### Resume from Checkpoint

The builder supports resuming by not using `--overwrite`:
```bash
# First run (interrupted)
./run_builder_resumable.sh nohup

# Resume (remove --overwrite from script first)
./run_builder_resumable.sh nohup
```

### Memory Issues

If running out of memory:
- Reduce `--batch-size` (fewer genes per batch)
- Reduce `--batch-rows` (smaller memory buffer)
- Close other applications
- Use swap space if available

## Best Practices

1. **Start Small**: Test with `--n-genes 100` before scaling up
2. **Use tmux for Development**: Easy to monitor and debug
3. **Use nohup for Production**: Minimal overhead, runs in background
4. **Monitor Logs**: Keep an eye on progress and errors
5. **Check Output**: Verify dataset integrity before training

## Related Documentation

- Main builder documentation: `meta_spliceai/splice_engine/meta_models/builder/docs/`
- Genomic resources guide: `meta_spliceai/system/docs/rebuild_genomic_resources.md`
- Training workflow: `meta_spliceai/splice_engine/meta_models/builder/docs/training_dataset_workflows.md`

