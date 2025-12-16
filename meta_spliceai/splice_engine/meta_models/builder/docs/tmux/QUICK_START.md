# Quick Start: Tmux with Incremental Builder

## TL;DR - Just Run It

```bash
# Navigate to the tmux docs directory
cd meta_spliceai/splice_engine/meta_models/builder/docs/tmux/

# Start the builder (easiest way)
./run_builder.sh

# Monitor progress
./monitor_builder.sh

# Attach to see live output
tmux a -t [session_name]
```

## What You Get

### 1. **Automated Runner** (`run_builder.sh`)
- Creates tmux session with timestamp
- Activates conda environment
- Runs incremental builder with logging
- Provides helpful commands

**Usage:**
```bash
./run_builder.sh                                              # Default settings
./run_builder.sh my_session                                   # Custom session name
./run_builder.sh my_session custom_output_dir                 # Custom output directory
./run_builder.sh my_session custom_output_dir 2000            # Custom gene count
./run_builder.sh my_session custom_output_dir 2000 genes.tsv  # Custom gene file
```

**Parameters:**
- `session_name` - tmux session name (default: `builder_YYYYMMDD_HHMMSS`)
- `output_dir` - output directory (default: `train_pc_1000_plus_custom_3mers`)
- `n_genes` - number of genes (default: `1000`)
- `gene_file` - custom gene list file (default: `additional_genes.tsv`)

**Example: Custom Gene List Workflow**
```bash
# Your use case - equivalent to:
# python -m meta_spliceai...incremental_builder \
#   --gene-ids-file additional_genes.tsv --gene-col gene_id \
#   --n-genes 1000 --output-dir train_pc_1000_plus_custom_3mers

./run_builder.sh \
  "custom_genes_$(date +%H%M%S)" \
  "train_pc_1000_plus_custom_3mers" \
  1000 \
  "additional_genes.tsv"
```

### 2. **Smart Monitor** (`monitor_builder.sh`)
- Shows session status
- Displays recent output
- Counts created files
- Shows system resources
- Provides useful commands

**Usage:**
```bash
./monitor_builder.sh                    # Monitor default session
./monitor_builder.sh my_session         # Monitor specific session
```

## Example Workflow

### Start a Job
```bash
$ ./run_builder.sh custom_genes train_pc_custom 1500

🚀 Starting Incremental Builder
================================
Session name: custom_genes
Output directory: train_pc_custom
Number of genes: 1500
Gene file: additional_genes.tsv
Log file: builder_custom_genes_20250115_143022.log

✅ Session created successfully!

📋 Useful Commands:
  • Attach to session:    tmux a -t custom_genes
  • Check status:         tmux ls
  • Monitor log:          tail -f builder_custom_genes_20250115_143022.log
  • Monitor output:       ls -la train_pc_custom/master/
  • Kill session:         tmux kill-session -t custom_genes
```

### Monitor Progress
```bash
# Monitor with auto-detection
$ ./monitor_builder.sh

# Monitor custom session
$ ./monitor_builder.sh custom_genes

# Monitor specific output directory
$ ./monitor_builder.sh custom_genes train_pc_custom

🔍 Monitoring Incremental Builder
=================================
Session: custom_genes
Output Directory: train_pc_custom
Time: Mon Jan 15 14:35:42 2025

✅ Session 'custom_genes' is running
📊 custom_genes: 1 windows (created Mon Jan 15 14:30:22 2025) [120x40]

📝 Recent output (last 10 lines):
-----------------------------------
Processing gene ENSG00000123456 (batch 15/60)
Generated 25000 positions for current batch
Saving batch_00015.parquet...
Progress: 25% complete (375/1500 genes)

📁 Output directory: train_pc_custom
  • Files created: 15
  • Disk usage: 2.3G
  • Latest files:
    -rw-rw-r-- 1 user user 156M Jan 15 14:35 batch_00015.parquet
    -rw-rw-r-- 1 user user 142M Jan 15 14:33 batch_00014.parquet
    -rw-rw-r-- 1 user user 138M Jan 15 14:31 batch_00013.parquet

💻 System resources:
  • Memory: 12G/32G (37% used)
  • Disk: 45G/100G (45% used)
  • Process: ✅ Running
  • CPU usage: 85%
```

### Attach to Watch Live
```bash
$ tmux a -t custom_genes
# You'll see the live output
# Press Ctrl+B then D to detach
```

## Key Benefits

### ✅ **Persistence**
- Jobs continue after SSH disconnection
- Survive network interruptions
- Can reconnect from different terminals

### ✅ **Monitoring**
- Real-time progress tracking
- Resource usage monitoring
- File creation tracking

### ✅ **Logging**
- Automatic log file creation
- Timestamped sessions
- Complete output capture

### ✅ **Convenience**
- One-command startup
- Smart defaults
- Error checking

## Common Use Cases

### Different Dataset Types
```bash
# Protein-coding genes (standard)
./run_builder.sh pc_1000 train_pc_1000 1000
./monitor_builder.sh pc_1000 train_pc_1000

# Non-protein-coding genes  
./run_builder.sh nc_1000 train_nc_1000 1000
./monitor_builder.sh nc_1000 train_nc_1000

# Custom gene subset for POC/testing
./run_builder.sh poc_test train_custom_subset 50
./monitor_builder.sh poc_test train_custom_subset

# Large production dataset
./run_builder.sh production train_pc_20000 20000  
./monitor_builder.sh production train_pc_20000
```

### Standard 1000-Gene Build
```bash
./run_builder.sh standard_1000 train_pc_1000_3mers 1000
./monitor_builder.sh standard_1000 train_pc_1000_3mers
```

### Custom Gene List
```bash
./run_builder.sh custom_genes train_pc_custom 1000 my_genes.tsv
./monitor_builder.sh custom_genes train_pc_custom
```

### Large Dataset
```bash
./run_builder.sh large_build train_pc_5000_3mers 5000
./monitor_builder.sh large_build train_pc_5000_3mers
```

### Development Testing
```bash
./run_builder.sh test_build test_output 50
./monitor_builder.sh test_build test_output
```

## Troubleshooting

### Script Won't Start
```bash
# Check if you're in the right directory
pwd
# Should show: .../meta_spliceai/splice_engine/meta_models/builder/docs/tmux

# Check if scripts are executable
ls -la *.sh
```

### Session Not Found
```bash
# List all sessions
tmux ls

# Check if process is running
ps aux | grep incremental_builder
```

### Display Issues
```bash
# Attach with force resize
tmux a -t session_name -d

# Or recreate session
tmux kill-session -t session_name
./run_builder.sh session_name
```

## Next Steps

- Read the full documentation: `README.md`
- Customize scripts for your workflow
- Set up automated monitoring with cron jobs
- Explore advanced tmux features

---

**Pro Tip**: Always use `./monitor_builder.sh` to check status before attaching to avoid interrupting the process unnecessarily! 