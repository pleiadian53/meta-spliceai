# Tmux Usage Guide for Incremental Builder

This guide covers using tmux to run long-running incremental builder processes that need to continue even after disconnecting from SSH sessions.

## Table of Contents

1. [Why Use Tmux?](#why-use-tmux)
2. [Basic Tmux Concepts](#basic-tmux-concepts)
3. [Running Incremental Builder with Tmux](#running-incremental-builder-with-tmux)
4. [Managing Sessions](#managing-sessions)
5. [Monitoring Progress](#monitoring-progress)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)
8. [Advanced Usage](#advanced-usage)

## Why Use Tmux?

The incremental builder can run for hours or days when processing large datasets. Tmux provides:

- **Persistence**: Processes continue running after SSH disconnection
- **Resumability**: Reconnect to see progress from any login session
- **Monitoring**: Check status without interrupting the process
- **Logging**: Capture output for later analysis
- **Reliability**: Immune to network interruptions

## Basic Tmux Concepts

### Key Terms
- **Session**: A collection of windows that persists after disconnection
- **Window**: Like a tab in a browser (we typically use one for the builder)
- **Pane**: Split sections within a window (useful for monitoring)

### Essential Key Bindings
- **Prefix Key**: `Ctrl+B` (press this before other commands)
- **Detach**: `Ctrl+B` then `D`
- **Scroll Mode**: `Ctrl+B` then `[` (use arrows to scroll, `q` to exit)
- **Command Mode**: `Ctrl+B` then `:`

## Running Incremental Builder with Tmux

### Method 1: Quick Start (Recommended)

```bash
# Create session and start builder in one command
tmux new-session -d -s "builder_$(date +%Y%m%d_%H%M%S)" \
  'conda activate surveyor && python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder --n-genes 1000 --subset-policy error_total --gene-ids-file additional_genes.tsv --gene-col gene_id --batch-size 250 --batch-rows 20000 --run-workflow --kmer-sizes 3 --output-dir train_pc_1000_plus_custom_3mers --overwrite -v 2>&1 | tee incremental_builder.log'
```

### Method 2: Step-by-Step

```bash
# 1. Create a new session
tmux new-session -d -s incremental_builder

# 2. Activate conda environment
tmux send-keys -t incremental_builder "conda activate surveyor" Enter

# 3. Start the incremental builder
tmux send-keys -t incremental_builder "python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder --n-genes 1000 --subset-policy error_total --gene-ids-file additional_genes.tsv --gene-col gene_id --batch-size 250 --batch-rows 20000 --run-workflow --kmer-sizes 3 --output-dir train_pc_1000_plus_custom_3mers --overwrite -v" Enter

# 4. Attach to monitor
tmux attach-session -t incremental_builder
```

### Method 3: With Logging

```bash
# Create session with output logging
tmux new-session -d -s incremental_builder \
  'conda activate surveyor && python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder --n-genes 1000 --subset-policy error_total --gene-ids-file additional_genes.tsv --gene-col gene_id --batch-size 250 --batch-rows 20000 --run-workflow --kmer-sizes 3 --output-dir train_pc_1000_plus_custom_3mers --overwrite -v 2>&1 | tee incremental_builder_$(date +%Y%m%d_%H%M%S).log'
```

## Managing Sessions

### List All Sessions
```bash
tmux list-sessions
# or shorthand:
tmux ls
```

### Attach to Session
```bash
# Attach to specific session
tmux attach-session -t incremental_builder
# or shorthand:
tmux a -t incremental_builder

# Attach with force resize (fixes display issues)
tmux a -t incremental_builder -d
```

### Detach from Session
```bash
# While attached, press:
Ctrl+B then D

# Or from outside (kills your attachment, not the session):
tmux detach-client -t incremental_builder
```

### Kill Session
```bash
# When the job is completely done
tmux kill-session -t incremental_builder
```

## Monitoring Progress

### Check Session Status
```bash
# Quick check if session exists
tmux has-session -t incremental_builder && echo "Running" || echo "Not running"

# List with details
tmux ls -F "#{session_name}: #{session_windows} windows (created #{session_created_string}) [#{session_width}x#{session_height}]"
```

### Monitor Output Directory
```bash
# Check created files
ls -la train_pc_1000_plus_custom_3mers/master/

# Count batches created
ls train_pc_1000_plus_custom_3mers/master/ | wc -l

# Check latest batch
ls -lt train_pc_1000_plus_custom_3mers/master/ | head -5
```

### Monitor Log Files
```bash
# If you used logging method
tail -f incremental_builder_*.log

# Follow last 50 lines
tail -n 50 -f incremental_builder_*.log
```

### Peek at Session Output
```bash
# Capture current screen content
tmux capture-pane -t incremental_builder -p

# Save screen content to file
tmux capture-pane -t incremental_builder -p > current_output.txt
```

## Troubleshooting

### Session Not Found
```bash
# Check if session exists
tmux ls

# If not found, check if process is still running
ps aux | grep incremental_builder

# Recreate session if needed
tmux new-session -d -s incremental_builder 'conda activate surveyor && python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder --n-genes 1000 --subset-policy error_total --gene-ids-file additional_genes.tsv --gene-col gene_id --batch-size 250 --batch-rows 20000 --run-workflow --kmer-sizes 3 --output-dir train_pc_1000_plus_custom_3mers --overwrite -v'
```

### Display Issues (Dots on Screen)
```bash
# Method 1: Resize window
# While attached, press: Ctrl+B then :
# Type: resize-window -A
# Press Enter

# Method 2: Detach and reattach
tmux detach-client -t incremental_builder
tmux a -t incremental_builder -d

# Method 3: Kill and recreate with proper size
tmux kill-session -t incremental_builder
tmux new-session -d -s incremental_builder -x $(tput cols) -y $(tput lines) 'conda activate surveyor && python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder --n-genes 1000 --subset-policy error_total --gene-ids-file additional_genes.tsv --gene-col gene_id --batch-size 250 --batch-rows 20000 --run-workflow --kmer-sizes 3 --output-dir train_pc_1000_plus_custom_3mers --overwrite -v'
```

### Process Appears Stuck
```bash
# Check if process is actually running
tmux capture-pane -t incremental_builder -p | tail -10

# Check system resources
top -p $(pgrep -f incremental_builder)

# Check disk space
df -h

# Check memory usage
free -h
```

### Permission Issues
```bash
# If you can't attach from different login
# Make sure you're the same user
whoami

# Check session ownership
tmux ls -F "#{session_name}: #{session_user}"
```

## Best Practices

### 1. Use Descriptive Session Names
```bash
# Good: Include date and purpose
tmux new-session -d -s "builder_1000genes_$(date +%Y%m%d_%H%M%S)"

# Bad: Generic names
tmux new-session -d -s "test"
```

### 2. Always Use Logging
```bash
# Capture both stdout and stderr
command 2>&1 | tee logfile.log

# Include timestamp in log name
tee "builder_$(date +%Y%m%d_%H%M%S).log"
```

### 3. Monitor Resource Usage
```bash
# Before starting large jobs
df -h  # Check disk space
free -h  # Check memory
```

### 4. Create Reusable Scripts
```bash
# Create run_builder.sh
cat > run_builder.sh << 'EOF'
#!/bin/bash
SESSION_NAME="builder_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="builder_$(date +%Y%m%d_%H%M%S).log"

tmux new-session -d -s "$SESSION_NAME" \
  "conda activate surveyor && python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder --n-genes 1000 --subset-policy error_total --gene-ids-file additional_genes.tsv --gene-col gene_id --batch-size 250 --batch-rows 20000 --run-workflow --kmer-sizes 3 --output-dir train_pc_1000_plus_custom_3mers --overwrite -v 2>&1 | tee $LOG_FILE"

echo "Started session: $SESSION_NAME"
echo "Log file: $LOG_FILE"
echo "Attach with: tmux a -t $SESSION_NAME"
echo "Monitor with: tail -f $LOG_FILE"
EOF

chmod +x run_builder.sh
```

### 5. Clean Up Completed Sessions
```bash
# List old sessions
tmux ls

# Kill completed sessions
tmux kill-session -t old_session_name

# Kill all sessions (careful!)
tmux kill-server
```

## Advanced Usage

### Multiple Panes for Monitoring
```bash
# Create session with split panes
tmux new-session -d -s builder_monitor

# Split window horizontally
tmux split-window -h -t builder_monitor

# Run builder in left pane
tmux send-keys -t builder_monitor.0 "conda activate surveyor && python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder --n-genes 1000 --subset-policy error_total --gene-ids-file additional_genes.tsv --gene-col gene_id --batch-size 250 --batch-rows 20000 --run-workflow --kmer-sizes 3 --output-dir train_pc_1000_plus_custom_3mers --overwrite -v" Enter

# Monitor output directory in right pane
tmux send-keys -t builder_monitor.1 "watch -n 30 'ls -la train_pc_1000_plus_custom_3mers/master/ | tail -10'" Enter

# Attach to see both
tmux a -t builder_monitor
```

### Automated Monitoring Script
```bash
# Create monitor_builder.sh
cat > monitor_builder.sh << 'EOF'
#!/bin/bash
SESSION_NAME="incremental_builder"

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Session $SESSION_NAME is running"
    
    # Show last few lines of output
    echo "Recent output:"
    tmux capture-pane -t $SESSION_NAME -p | tail -5
    
    # Show file count
    echo "Files created:"
    ls train_pc_1000_plus_custom_3mers/master/ 2>/dev/null | wc -l
    
    # Show disk usage
    echo "Disk usage:"
    du -sh train_pc_1000_plus_custom_3mers/ 2>/dev/null
else
    echo "Session $SESSION_NAME is not running"
fi
EOF

chmod +x monitor_builder.sh
```

### Session Persistence Across Reboots
```bash
# Install tmux-resurrect plugin for session persistence
# (Advanced topic - requires tmux plugin manager)
```

## Common Incremental Builder Commands

### Standard 1000 Gene Build
```bash
tmux new-session -d -s "builder_1000_standard" \
  'conda activate surveyor && python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder --n-genes 1000 --subset-policy error_total --batch-size 250 --batch-rows 20000 --run-workflow --kmer-sizes 3 --output-dir train_pc_1000_3mers --overwrite -v 2>&1 | tee builder_1000_standard.log'
```

### Custom Gene List Build
```bash
tmux new-session -d -s "builder_custom_genes" \
  'conda activate surveyor && python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder --n-genes 1000 --subset-policy error_total --gene-ids-file additional_genes.tsv --gene-col gene_id --batch-size 250 --batch-rows 20000 --run-workflow --kmer-sizes 3 --output-dir train_pc_1000_plus_custom_3mers --overwrite -v 2>&1 | tee builder_custom_genes.log'
```

### Large Dataset Build
```bash
tmux new-session -d -s "builder_large" \
  'conda activate surveyor && python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder --n-genes 5000 --subset-policy error_total --batch-size 500 --batch-rows 50000 --run-workflow --kmer-sizes 3 --output-dir train_pc_5000_3mers --overwrite -v 2>&1 | tee builder_large.log'
```

## Quick Reference

### Essential Commands
```bash
# Start builder
tmux new-session -d -s builder 'conda activate surveyor && python -m meta_spliceai.splice_engine.meta_models.builder.incremental_builder [options]'

# Check status
tmux ls

# Attach
tmux a -t builder

# Detach
Ctrl+B then D

# Kill
tmux kill-session -t builder

# Monitor files
ls -la train_pc_1000_plus_custom_3mers/master/

# Check logs
tail -f builder.log
```

### Keyboard Shortcuts (While Attached)
- `Ctrl+B then D`: Detach
- `Ctrl+B then [`: Scroll mode
- `Ctrl+B then :`: Command mode
- `Ctrl+L`: Clear screen
- `Ctrl+C`: Interrupt current command (use carefully!)

---

**Note**: This documentation is specific to the incremental builder use case. For general tmux usage, consult the tmux manual (`man tmux`). 