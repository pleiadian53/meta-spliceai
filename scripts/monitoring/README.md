# Monitoring Scripts

Scripts for monitoring meta-model training progress and diagnostics.

---

## Overview

This directory contains tools for tracking and analyzing ongoing meta-model training runs. All scripts here are focused on **monitoring training processes**, not inference workflows.

---

## Available Tools

### 1. `monitor_meta_training.sh` (Simple Shell Monitor)

**Purpose:** Quick, lightweight monitoring of a single meta-model training run  
**Best for:** Fast status checks during development  
**Language:** Shell script

**Usage:**
```bash
# Monitor the 1000-gene training
./scripts/monitoring/monitor_meta_training.sh

# Or with custom PID
./scripts/monitoring/monitor_meta_training.sh 12345
```

**What it shows:**
- ✅ Process status (PID, runtime, CPU, memory)
- 📈 Current fold being processed
- 📊 Recent fold metrics (F1, AP, top-k)
- 📝 Last 5 log lines
- 💡 Quick reference commands

**Output:**
```
==================================
Meta-Model Training Monitor
==================================

📊 Process Status:
  PID    ... %CPU  %MEM ...

📈 Training Progress:
  Log file: logs/meta_training_1000genes_20251020.log
  
  Current: 🔀 Fold 3/5: 19,995 test positions
  
  Recent Fold Results:
    ✅ Fold 1 results: F1=0.999, AP=1.000, Top-k=1.000 ...
    ✅ Fold 2 results: F1=0.999, AP=1.000, Top-k=1.000 ...
```

---

### 2. `monitor_training_universal.py` (Comprehensive Monitor)

**Purpose:** Full-featured monitoring with detailed analysis and diagnostics  
**Best for:** Production runs, multi-instance training, debugging  
**Language:** Python 3

**Usage:**
```bash
# Monitor specific run
python scripts/monitoring/monitor_training_universal.py --run-name meta_model_1000genes_3mers

# Auto-detect active runs
python scripts/monitoring/monitor_training_universal.py --auto-detect

# List all available runs
python scripts/monitoring/monitor_training_universal.py --list-runs

# Continuous monitoring (updates every 30s)
python scripts/monitoring/monitor_training_universal.py --run-name <name> --watch
```

**What it shows:**
- 🔄 Process status with detailed resource usage
- 📊 Training mode detection (single/multi-instance/batch)
- ✅ Milestone tracking (CV started, folds completed, SHAP analysis, etc.)
- ⚠️  Issue detection (errors, warnings, critical failures)
- 📁 Output artifact analysis
  - Critical files (model, metrics, manifests)
  - Key directories (CV viz, feature analysis, SHAP)
  - File counts (PKL, JSON, CSV, PDF)
- 🔢 Multi-instance specific tracking
  - Instance completion status
  - Per-instance model sizes
  - Consolidation progress

**Output:**
```
🔍 Training Monitor Report: meta_model_1000genes_3mers
================================================================================
Generated: 2025-10-20 16:30:45

🔄 Process Status:
  ✅ Running (PID: 12345)
  ⏰ Runtime: 02:34:15
  💾 Memory: 8542.3 MB
  🔧 CPU: 245.7%

📊 Training Analysis (Mode: Single_Instance):
  📝 Log lines: 15,234
  🕐 Last update: 0.2 minutes ago
  ✅ Completed milestones:
    - Cross-validation started: 1
    - CV fold progress: 5
    - CV completed: 1
    - Final model training: 1
    - Model saved: 1
    - SHAP analysis: 1
  📝 Recent activity:
    🎉 Training pipeline completed successfully!
    ✅ Production model saved: model_multiclass.pkl

📁 Output Analysis:
  📦 Files: 3 models, 12 metadata, 8 data, 15 plots
  🎯 Critical files:
    ✅ main_model (23.4MB)
    ✅ training_results (0.1MB)
    ✅ cv_metrics (0.3MB)
    ✅ feature_manifest (0.02MB)
  📂 Key directories:
    ✅ cv_metrics_visualization (47 files)
    ✅ feature_importance_analysis (23 files)
    ✅ comprehensive_shap_analysis (31 files)
```

---

## Comparison: Which Tool to Use?

| Feature | `monitor_meta_training.sh` | `monitor_training_universal.py` |
|---------|---------------------------|--------------------------------|
| **Speed** | ⚡ Instant | 🐢 2-3 seconds |
| **Setup** | None | Python required |
| **Process detection** | Manual PID | Auto-detect |
| **Training mode** | N/A | ✅ Auto-detects |
| **Milestone tracking** | ❌ | ✅ Comprehensive |
| **Error detection** | ❌ | ✅ Pattern-based |
| **Output validation** | ❌ | ✅ File-by-file |
| **Multi-instance support** | ❌ | ✅ Full support |
| **Watch mode** | ❌ | ✅ Auto-refresh |
| **Best for** | Quick checks | Production/debugging |

---

## Related Scripts

### Training Scripts
- **`scripts/builder/`** - Dataset generation (incremental_builder.py)
- **`scripts/testing/`** - End-to-end pipeline tests
- **`meta_spliceai/.../training/run_gene_cv_sigmoid.py`** - Main training driver

### Inference Scripts
- **`scripts/inference/`** - Inference workflow testing (separate from training)

---

## Workflow Example

```bash
# 1. Start training in background
tmux new -s training
python -m meta_spliceai.splice_engine.meta_models.training.run_gene_cv_sigmoid \
    --dataset data/train_pc_1000_3mers/master \
    --out-dir results/meta_model_test \
    ... 2>&1 | tee logs/training_test.log
# Detach: Ctrl+B, D

# 2. Quick check (simple monitor)
./scripts/monitoring/monitor_meta_training.sh

# 3. Detailed analysis (comprehensive monitor)
python scripts/monitoring/monitor_training_universal.py --auto-detect

# 4. Continuous monitoring
python scripts/monitoring/monitor_training_universal.py \
    --run-name meta_model_test --watch

# 5. Check all runs
python scripts/monitoring/monitor_training_universal.py --list-runs
```

---

## Notes

- ⚠️ These tools are for **meta-model training only** (not inference)
- ✅ Both scripts are read-only and safe to run during training
- 💡 Use simple monitor for quick checks, comprehensive for production
- 🔄 Auto-detect mode finds active processes automatically
- 📝 Log files must follow pattern: `logs/<run_name>.log`

---

## Future Enhancements

- [ ] Add email/Slack notifications for completion
- [ ] Integrate with MLflow for experiment tracking
- [ ] Add performance regression detection
- [ ] Create web dashboard for multi-run comparison

