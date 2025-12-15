# Inference Workflow Output Structure

## Overview

When you run the `main_inference_workflow.py` with the command:

```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
  --model results/gene_cv_pc_1000_3mers_run_4 \
  --training-dataset train_pc_1000_3mers \
  --genes ENSG00000284616,ENSG00000115705,ENSG00000234661 \
  --complete-coverage \
  --output-dir results/test_scenario2b_full_meta_fixed \
  --inference-mode meta_only \
  --enable-chunked-processing \
  --chunk-size 5000 \
  --force-recompute \
  --verbose
```

The workflow creates outputs in **TWO main locations** (not three):

## 1. Primary Output Directory (`--output-dir`)

**Location**: `results/test_scenario2b_full_meta_fixed/` (as specified by `--output-dir`)

**Purpose**: Main inference results and gene-specific predictions

**Structure**:
```
results/test_scenario2b_full_meta_fixed/
├── selective_inference/           # Selective inference outputs
│   ├── predictions/              # Prediction files
│   │   ├── complete_coverage_predictions.parquet  # Hybrid predictions
│   │   ├── meta_model_predictions.parquet        # Meta-only predictions
│   │   └── base_model_predictions.parquet        # Base model predictions
│   ├── cache/                    # Cached data
│   │   └── gene_manifests/       # Gene processing manifests
│   │       └── selective_inference_manifest.csv
│   └── artifacts/                # Intermediate artifacts (if --keep-artifacts-dir)
├── genes/                        # Per-gene results
│   ├── ENSG00000284616/
│   │   └── ENSG00000284616_predictions.parquet
│   ├── ENSG00000115705/
│   │   └── ENSG00000115705_predictions.parquet
│   └── ENSG00000234661/
│       └── ENSG00000234661_predictions.parquet
├── performance_report.txt        # Overall performance metrics
├── inference_summary.json        # Summary statistics
└── inference_workflow.log        # Workflow execution log
```

## 2. Test Data Directory (Automatic)

**Location**: `test_pc_1000_3mers/` (derived from `--training-dataset train_pc_1000_3mers`)

**Purpose**: Organized test data in the same format as training data

**Structure**:
```
test_pc_1000_3mers/
├── master/                       # Master batch files
│   ├── batch_00001.parquet      # Test data batch
│   └── gene_manifest.csv        # Gene manifest
├── predictions/                  # Prediction outputs (if saved here)
│   └── scenario2b/              # Scenario-specific predictions
└── metadata/                     # Metadata files
    └── inference_metadata.json   # Inference run metadata
```

**Note**: This directory is automatically created based on the training dataset name:
- `train_pc_1000_3mers` → `test_pc_1000_3mers`
- `train_pc_500_5mers` → `test_pc_500_5mers`

## 3. MLflow Tracking (Optional)

**Location**: Depends on MLflow configuration:
- **Local file mode**: `mlruns/` in current directory
- **Server mode**: `/home/bchiu/output/mlflow_data/` (as configured in your setup)

**Purpose**: Experiment tracking, metrics, and artifact storage

**Structure** (Server mode):
```
/home/bchiu/output/mlflow_data/
├── mlflow.db                    # SQLite database for metadata
└── artifacts/                    # Artifact storage
    └── <experiment_id>/
        └── <run_id>/
            ├── artifacts/
            │   ├── selective_inference/
            │   │   ├── complete_coverage_predictions.parquet
            │   │   └── meta_model_predictions.parquet
            │   ├── performance/
            │   │   ├── performance_report.txt
            │   │   └── roc_pr_curves.png
            │   └── logs/
            │       └── inference_console.log
            └── metrics/          # Logged metrics
```

## Key Differences Between Output Locations

### 1. Primary Output Directory (`--output-dir`)
- **User-specified location**: You control where this goes
- **Contains final results**: All predictions, performance reports, summaries
- **Persistent**: Not cleaned up automatically
- **Direct access**: Easy to find and share results

### 2. Test Data Directory
- **Automatically determined**: Based on training dataset name
- **Contains formatted test data**: Organized like training data for consistency
- **May contain temporary files**: During processing
- **Follows naming convention**: Ensures consistency across runs

### 3. MLflow Directory (if enabled)
- **Managed by MLflow**: Don't modify directly
- **Contains tracked experiments**: All runs, parameters, metrics
- **Versioned**: Each run gets a unique ID
- **Web UI accessible**: View via `http://localhost:5000`

## How to Access Results

### Primary Results
```bash
# View predictions
ls results/test_scenario2b_full_meta_fixed/selective_inference/predictions/

# Check performance report
cat results/test_scenario2b_full_meta_fixed/performance_report.txt

# View summary
cat results/test_scenario2b_full_meta_fixed/inference_summary.json
```

### Test Data
```bash
# View organized test data
ls test_pc_1000_3mers/master/

# Check gene manifest
cat test_pc_1000_3mers/master/gene_manifest.csv
```

### MLflow Results (if enabled)
```bash
# Start MLflow UI (if not already running)
mlflow ui --backend-store-uri /home/bchiu/output/mlflow_data/db/mlflow.db

# Or if using server mode (already running in your case)
# Browse to http://localhost:5000
```

## Best Practices

1. **Use descriptive output directories**: Include scenario, date, or experiment details
2. **Enable MLflow for tracking**: Helps compare multiple runs
3. **Keep artifacts for debugging**: Use `--keep-artifacts-dir` when troubleshooting
4. **Clean up test directories**: These can accumulate over time

## Environment Variables

You can set `SURVEYOR_CONSOLE_LOG` to specify where console output is saved:
```bash
export SURVEYOR_CONSOLE_LOG=/path/to/console.log
```

This will be automatically uploaded to MLflow if tracking is enabled.