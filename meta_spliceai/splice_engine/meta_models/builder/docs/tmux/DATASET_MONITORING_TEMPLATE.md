# Tmux Monitoring Usage Template for Training Datasets

**Instructions:** Copy this template to your training dataset directory (e.g., `train_pc_1000/`, `train_nc_5000/`) and customize the values below.

---

# Tmux Monitoring Usage for Training Dataset

**Dataset:** `[DATASET_NAME]` (e.g., `train_pc_1000`, `train_nc_5000`, `train_custom_subset`)  
**Generated:** [DATE]  
**Description:** [DESCRIPTION] (e.g., "1000 protein-coding genes with 3-mer features")

## Current Build Command

This training dataset was generated using:

```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
  --n-genes [N_GENES] \
  --subset-policy [POLICY] \
  --gene-ids-file [GENE_FILE] \
  --gene-col [GENE_COL] \
  --batch-size [BATCH_SIZE] \
  --batch-rows [BATCH_ROWS] \
  --run-workflow \
  --kmer-sizes [KMER_SIZES] \
  --output-dir [DATASET_NAME] \
  --overwrite \
  -v
```

**Example for protein-coding genes:**
```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
  --n-genes 1000 \
  --subset-policy error_total \
  --batch-size 250 \
  --batch-rows 20000 \
  --run-workflow \
  --kmer-sizes 3 \
  --output-dir train_pc_1000 \
  --overwrite \
  -v
```

**Example for non-protein-coding genes:**
```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
  --n-genes 1000 \
  --subset-policy error_total \
  --batch-size 250 \
  --batch-rows 20000 \
  --run-workflow \
  --kmer-sizes 3 \
  --gene-types lncRNA \
  --output-dir train_nc_1000 \
  --overwrite \
  -v
```

Note that `--gene-types` parameter accepts multiple values, so you could also specify multiple gene types:
```bash
python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder \
  --n-genes 1000 \
  --subset-policy error_total \
  --batch-size 250 \
  --batch-rows 20000 \
  --run-workflow \
  --kmer-sizes 3 \
  --gene-types lncRNA pseudogene miRNA \
  --output-dir train_nc_1000 \
  --overwrite \
  -v
```

## Tmux Session Management

### Monitor This Dataset
```bash
# Monitor with auto-detection (from project root)
./meta_spliceai/splice_engine/meta_models/builder/docs/tmux/monitor_builder.sh

# Monitor with specific output directory
./meta_spliceai/splice_engine/meta_models/builder/docs/tmux/monitor_builder.sh incremental_builder [DATASET_NAME]

# Monitor custom session with this output directory  
./meta_spliceai/splice_engine/meta_models/builder/docs/tmux/monitor_builder.sh [CUSTOM_SESSION] [DATASET_NAME]
```

### Real Examples for Different Dataset Types

#### Protein-Coding Genes
```bash
./meta_spliceai/splice_engine/meta_models/builder/docs/tmux/monitor_builder.sh incremental_builder train_pc_1000
./meta_spliceai/splice_engine/meta_models/builder/docs/tmux/monitor_builder.sh incremental_builder train_pc_5000
./meta_spliceai/splice_engine/meta_models/builder/docs/tmux/monitor_builder.sh incremental_builder train_pc_20000
```

#### Non-Protein-Coding Genes
```bash
./meta_spliceai/splice_engine/meta_models/builder/docs/tmux/monitor_builder.sh incremental_builder train_nc_1000
./meta_spliceai/splice_engine/meta_models/builder/docs/tmux/monitor_builder.sh incremental_builder train_nc_5000
```

#### Custom/POC Datasets
```bash
./meta_spliceai/splice_engine/meta_models/builder/docs/tmux/monitor_builder.sh poc_session train_custom_subset
./meta_spliceai/splice_engine/meta_models/builder/docs/tmux/monitor_builder.sh test_session train_validation_genes
```

### Sample Monitor Output
```
🔍 Monitoring Incremental Builder
=================================
Session: incremental_builder
Output Directory: [DATASET_NAME]
Time: [TIMESTAMP]

✅ Session 'incremental_builder' is running
📊 incremental_builder: 1 windows (created [TIME]) [SIZE]

📝 Recent output (last 10 lines):
-----------------------------------
Processing gene sequences: [PROGRESS]% |████████████████| [CURRENT]/[TOTAL] [TIME_REMAINING]
(predict_splice_sites_gene) Processing gene [GENE_ID] (at chr=[CHR]) ...
    Sequence Length: [LENGTH]
[debug] Generated [N] blocks for gene [GENE_ID].
[MODEL_OUTPUT]

📁 Output directory: [DATASET_NAME] (user-specified)
  • Files created: [N_FILES]
  • Disk usage: [SIZE]
  • Latest files:
    [LATEST_FILES]

💻 System resources:
  • Memory: [USED]/[TOTAL] ([PERCENT]% used)
  • Disk: [USED]/[TOTAL] ([PERCENT]% used)
  • Process: ✅ Running
```

## Quick Commands

### Attach to Session
```bash
tmux a -t [SESSION_NAME]
```

### Check Progress Without Monitoring Script
```bash
# View recent output
tmux capture-pane -t [SESSION_NAME] -p | tail -10

# Check file creation
ls -la [DATASET_NAME]/master/ | wc -l

# Check disk usage
du -sh [DATASET_NAME]/
```

### Manual Process Monitoring
```bash
# Check if process is running
ps aux | grep incremental_builder | grep -v grep

# Monitor system resources
htop
# or
top | grep python
```

## Dataset-Specific Information

### Expected Structure
```
[DATASET_NAME]/
├── master/                    # Training data files
│   ├── batch_00001.parquet
│   ├── batch_00002.parquet
│   └── ...
└── TMUX_MONITORING_USAGE.md   # This documentation
```

### Build Parameters (Customize These)
- **Gene Count:** [N_GENES]
- **Gene Type:** [protein-coding | non-protein-coding | custom subset]
- **Selection Policy:** [error_total | random | custom]
- **Batch Size:** [BATCH_SIZE] genes per batch
- **Batch Rows:** [BATCH_ROWS] positions per batch file
- **K-mer Features:** [KMER_SIZES]-mer sequences
- **Custom Genes:** [gene file description if applicable]

### Expected Runtime (Estimates)
- **Small Dataset (50-100 genes):** ~2-4 hours
- **Medium Dataset (1000 genes):** ~30-40 hours  
- **Large Dataset (5000+ genes):** ~150-200 hours
- **Progress Tracking:** Shows X/Y genes processed with ETA
- **Memory Usage:** ~25-30GB during processing
- **Disk Space:** ~2-5GB per 100 genes processed

## Troubleshooting

### Session Lost Connection
```bash
# List all sessions
tmux ls

# Reattach to session
tmux a -t [SESSION_NAME]

# If session died, check logs
tail -f *.log
```

### Process Monitoring Issues
```bash
# Check if process is still running
pgrep -f "incremental_builder"

# View process details
ps aux | grep incremental_builder
```

### Output Directory Issues
```bash
# Check if directory exists
ls -la [DATASET_NAME]/

# Check master directory creation
ls -la [DATASET_NAME]/master/

# Monitor file creation in real-time
watch "ls -la [DATASET_NAME]/master/ | tail -5"
```

## Related Documentation

- **Full Tmux Guide:** `meta_spliceai/splice_engine/meta_models/builder/docs/tmux/README.md`
- **Quick Start:** `meta_spliceai/splice_engine/meta_models/builder/docs/tmux/QUICK_START.md`
- **Run Builder Script:** `meta_spliceai/splice_engine/meta_models/builder/docs/tmux/run_builder.sh`
- **Monitor Script:** `meta_spliceai/splice_engine/meta_models/builder/docs/tmux/monitor_builder.sh`

---
*This documentation is specific to the `[DATASET_NAME]` dataset. For other datasets, copy this template and adjust the dataset name and parameters accordingly.* 