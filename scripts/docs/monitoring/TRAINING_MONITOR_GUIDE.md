# Universal Training Monitor Guide

## Overview

The `monitor_training_universal.py` script provides comprehensive monitoring for all splice surveyor training modes:
- **Single-Instance Training** (small/medium datasets)
- **Multi-Instance Ensemble Training** (large datasets)
- **Batch Ensemble Training** (alternative large dataset approach)

## Quick Usage

### Auto-detect Active Runs
```bash
python scripts/monitoring/monitor_training_universal.py --auto-detect
```

### Monitor Specific Run
```bash
python scripts/monitoring/monitor_training_universal.py --run-name gene_cv_reg_10k_kmers_run_8_complete
```

### List All Available Runs
```bash
python scripts/monitoring/monitor_training_universal.py --list-runs
```

### Continuous Monitoring (Updates Every 30 Seconds)
```bash
python scripts/monitoring/monitor_training_universal.py --run-name <run_name> --watch
```

## What It Monitors

### Process Information
- ✅ **Process Status**: Running/stopped with PID
- ✅ **Resource Usage**: Memory, CPU, runtime
- ✅ **Health Indicators**: Process stability

### Training Progress
- ✅ **Training Mode Detection**: Single/Multi-instance/Batch ensemble
- ✅ **Milestone Tracking**: Setup → Training → Consolidation → Analysis
- ✅ **Error Detection**: Exceptions, warnings, critical issues
- ✅ **Recent Activity**: Last 5 meaningful log entries

### Output Analysis
- ✅ **File Counts**: Models, metadata, visualizations
- ✅ **Critical Files**: Main model, metrics, manifests
- ✅ **Directory Structure**: All expected output directories
- ✅ **Multi-Instance Details**: Instance completion tracking

### SHAP Analysis Tracking
- ✅ **Standard SHAP**: Traditional feature importance
- ✅ **Enhanced SHAP**: Comprehensive ensemble analysis
- ✅ **Fallback Detection**: Identifies when fallbacks are used
- ✅ **Success Verification**: Real vs dummy SHAP values

## Example Outputs

### Active Multi-Instance Training
```bash
🔍 Training Monitor Report: gene_cv_reg_10k_kmers_run_8_complete
================================================================================
🔄 Process Status:
  ✅ Running (PID: 1298216)
  ⏰ Runtime: 14:03
  💾 Memory: 22905.7 MB

📊 Training Analysis (Mode: Multi_Instance):
  ✅ Completed milestones:
    - Multi-instance started: 1
    - Instance training: 1
  🔢 Multi-instance details:
    Instances: 0/9 completed
```

### Completed Single-Instance Training
```bash
🔍 Training Monitor Report: gene_cv_pc_5000_3mers_diverse_run3
================================================================================
🔄 Process Status:
  ❌ Not running

📊 Training Analysis (Mode: Single_Instance):
  ✅ Completed milestones:
    - Training completed: 1
  📦 Files: 4 models, 13 metadata, 17 data, 19 plots
  ✅ feature_importance_analysis (20 files)
```

## Monitoring Best Practices

### For Long-Running Training
```bash
# Start monitoring in background
nohup python monitor_training_universal.py --run-name <run_name> --watch > monitoring.log 2>&1 &
```

### For Quick Status Checks
```bash
# Check all active runs
python monitor_training_universal.py --auto-detect

# Check specific run
python monitor_training_universal.py --run-name <run_name>
```

### For Debugging Issues
```bash
# Monitor with verbose output
python monitor_training_universal.py --run-name <run_name> --verbose
```

## Key Features

✅ **Universal**: Works with all training modes  
✅ **Comprehensive**: Process + Log + Output analysis  
✅ **Real-time**: Live progress tracking  
✅ **Error Detection**: Identifies issues early  
✅ **Resource Monitoring**: Memory and CPU tracking  
✅ **SHAP Validation**: Enhanced SHAP analysis verification  
✅ **Output Validation**: Complete file structure checking  

## Replaces Previous Scripts

This universal monitor consolidates the functionality of:
- `monitor_configurable_test.sh`
- `monitor_multi_instance.sh` 
- `monitor_run_8.py`
- `monitor_sanity_check.sh`
- `monitor_training.sh`

All previous monitoring scripts have been removed in favor of this unified solution.
