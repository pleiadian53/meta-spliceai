# MLflow Integration Guide for Splice Surveyor Inference

## Overview

MLflow has been integrated into the Splice Surveyor inference workflow to provide comprehensive experiment tracking, metrics logging, and artifact management. This guide covers setup, usage, and interpretation of MLflow tracking data.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration Options](#configuration-options)
4. [MLflow UI Setup](#mlflow-ui-setup)
5. [Tracked Metrics and Artifacts](#tracked-metrics-and-artifacts)
6. [Usage Examples](#usage-examples)
7. [Viewing Results](#viewing-results)
8. [Advanced Setup](#advanced-setup)

## Installation

### Adding MLflow to Your Environment

```bash
# Activate your Splice Surveyor environment
mamba activate surveyor

# Install MLflow (conda-forge or pip)
mamba install -c conda-forge mlflow=3.2.0 || pip install mlflow==3.2.0

# Verify installation
python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
```

## Quick Start

### Basic Usage (Local Tracking)

1. **Run inference with MLflow tracking enabled:**

```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000154358,ENSG00000100490 \
    --output-dir ./inference_results \
    --mlflow-enable
```

2. **Start the MLflow UI to view results:**

```bash
# From your project root directory
mlflow ui -h 127.0.0.1 -p 5000

# Open browser to http://127.0.0.1:5000
```

### Local file mode (centralized store) + workflow usage

If you prefer a centralized file store (no tracking server), point both the UI and the workflow to the same file-backed store. This is a UI-only process; runs are logged directly to the filesystem.

```bash
# UI reads a local FileStore (no REST API)
mlflow ui -h 127.0.0.1 -p 5000 \
  --backend-store-uri "$HOME/output/mlflow_data/mlruns"

# Run the inference workflow and log runs to the same file store
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
  --model "$PWD/results/gene_cv_pc_1000_3mers_run_4" \
  --training-dataset "$PWD/train_pc_1000_3mers" \
  --genes ENSG00000284616,ENSG00000115705 \
  --complete-coverage \
  --output-dir "$PWD/test_pc_1000_3mers/predictions/scenario2b" \
  --inference-mode hybrid \
  --mlflow-enable \
  --mlflow-experiment surveyor-inference \
  --mlflow-tracking-uri "file://$HOME/output/mlflow_data/mlruns" \
  --verbose

# View at http://127.0.0.1:5000
```

Notes:
- This does not start a tracking server; the UI reads the file store directly.
- Keep the file URI (`file://.../mlruns`) identical for both the UI and workflow.

## Configuration Options

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mlflow-enable` | Enable MLflow tracking | False |
| `--mlflow-experiment` | Experiment name | surveyor-inference |
| `--mlflow-tracking-uri` | MLflow server URI | None (local) |
| `--mlflow-tags` | Tags as key=value pairs | None |

### Example with All Options

```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000154358 \
    --output-dir ./inference_results \
    --mlflow-enable \
    --mlflow-experiment "splice-inference-production" \
    --mlflow-tracking-uri "http://localhost:5000" \
    --mlflow-tags scenario=training_gene model_version=v4 run_type=validation
```

## MLflow UI Setup

### Option 1: Local File Store (Simplest)

No setup required! MLflow automatically creates an `mlruns` directory in your project root.

```bash
# Start UI pointing to local mlruns directory
mlflow ui -h 127.0.0.1 -p 5000

# Access at http://127.0.0.1:5000
```

Centralized variant (explicit path):

```bash
mlflow ui -h 127.0.0.1 -p 5000 \
  --backend-store-uri "$HOME/output/mlflow_data/mlruns"
```

### Option 2: Centralized Tracking Server

Set up a persistent MLflow server with SQLite backend:

```bash
# If you used the provided setup script, data lives under $HOME/output/mlflow_data
# Quick start with the generated helper script (recommended):
./start_mlflow.sh

# Or start manually with SQLite backend (bind localhost for security)
mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///$HOME/output/mlflow_data/db/mlflow.db \
  --default-artifact-root file://$HOME/output/mlflow_data/artifacts \
  --serve-artifacts

# Point clients at the server
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### VSCode Remote-SSH Auto Port Forwarding

To automatically forward MLflow and Jupyter ports when working on a remote VM via VSCode, add the following to `.vscode/settings.json` in your project:

```json
{
  "python.defaultInterpreterPath": "/home/bchiu/.local/share/mamba/envs/surveyor/bin/python",
  "remote.portsAttributes": {
    "5000": { "label": "MLflow", "onAutoForward": "openPreview" },
    "8888": { "label": "Jupyter", "onAutoForward": "openPreview" }
  },
  "remote.autoForwardPorts": true
}
```

With this, VSCode detects the MLflow server and Jupyter Lab automatically and opens a preview window when they start.

### Using the inference workflow with the tracking server (recommended)

Run the workflow and log to the server. The server provides both REST API and the UI at the same address.

```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
  --model "$PWD/results/gene_cv_pc_1000_3mers_run_4" \
  --training-dataset "$PWD/train_pc_1000_3mers" \
  --genes ENSG00000284616,ENSG00000115705 \
  --complete-coverage \
  --output-dir "$PWD/test_pc_1000_3mers/predictions/scenario2b" \
  --inference-mode hybrid \
  --mlflow-enable \
  --mlflow-experiment surveyor-inference \
  --mlflow-tracking-uri http://127.0.0.1:5000 \
  --verbose

# Browse the built-in UI at http://127.0.0.1:5000
```

### Consistency checklist

- UI and clients must target the same backend store.
- Local file mode: `mlflow ui --backend-store-uri file:///.../mlruns` and workflow uses `--mlflow-tracking-uri file:///.../mlruns`.
- Server mode: start `mlflow server` (or `./start_mlflow.sh`) and workflow uses `--mlflow-tracking-uri http://127.0.0.1:5000`. Do not start a separate `mlflow ui` in this mode.

### Why prefer server mode (even for a single user)

- Consistent endpoint: One stable `http://127.0.0.1:5000` works from any directory or script.
- Centralization by config: Avoid scattered `./mlruns`; no dependence on where the UI was launched.
- Concurrency safety: The server mediates writes to the DB and artifact root across processes.
- Remote- and multi-user-ready: Other machines or notebooks can log via SSH tunnel/port-forward without shared filesystems.
- Artifact serving: `--serve-artifacts` enables clean access (and future S3/remote backends).
- Operability: Easy to run via systemd user service; survives shell exits.
- Extensibility: Simplifies migration to Postgres/S3, reverse proxy, TLS, auth, and Model Registry.

### Option 3: Production Setup with PostgreSQL

For production environments with multiple users:

```bash
# Install PostgreSQL driver
pip install psycopg2-binary

# Start MLflow server with PostgreSQL
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri postgresql://user:password@postgres:5432/mlflow \
    --default-artifact-root s3://my-mlflow-bucket/artifacts
```

## Tracked Metrics and Artifacts

### Metrics Logged

#### Workflow-Level Metrics
- `total_genes_requested`: Number of genes requested for processing
- `successful_genes`: Number of successfully processed genes
- `failed_genes`: Number of failed genes
- `success_rate`: Percentage of successful gene processing
- `total_positions_analyzed`: Total nucleotide positions analyzed
- `meta_model_positions`: Positions processed by meta-model
- `meta_model_usage_percentage`: Percentage of positions using meta-model
- `total_processing_time`: Total workflow runtime in seconds

#### Per-Gene Metrics
- `gene_{gene_id}_positions`: Total positions for specific gene
- `gene_{gene_id}_meta_positions`: Meta-model positions for gene
- `gene_{gene_id}_processing_time`: Processing time for gene

#### Selective Inference Metrics
- `selective_total_positions`: Total positions in selective inference
- `selective_uncertain_positions`: Positions identified as uncertain
- `selective_reused_positions`: Positions reusing base model predictions
- `selective_efficiency`: Efficiency percentage of selective approach
- `selective_processing_time`: Time for selective inference

#### Class Distribution Metrics
- `class_TP_count`: True positive count
- `class_TN_count`: True negative count
- `class_FP_count`: False positive count
- `class_FN_count`: False negative count

### Artifacts Logged

#### Main Workflow Artifacts
- `gene_manifest.json`: Gene processing manifest
- `inference_summary.json`: Comprehensive results summary
- `performance_report.txt`: Detailed performance analysis
- `inference_workflow.log`: Complete execution log
- `directory_tree.txt`: Output directory structure snapshot
- `directory_tree.json`: Machine-readable tree with absolute paths to large artifacts (tracked by reference only)

#### Per-Gene Artifacts
- `genes/{gene_id}/{gene_id}_predictions.parquet`: Predictions for each gene
- `genes/{gene_id}/{gene_id}_statistics.json`: Gene-specific statistics

#### Selective Inference Artifacts
- `selective_inference/hybrid_predictions.parquet`: Combined predictions
- `selective_inference/meta_predictions.parquet`: Meta-model predictions
- `selective_inference/base_predictions.parquet`: Base model predictions
- `selective_inference/inference_metadata.json`: Inference configuration

## Usage Examples

### Example 1: Basic Inference with Tracking

```bash
# Simple inference with MLflow tracking
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000154358 \
    --output-dir results/mlflow_test \
    --mlflow-enable
```

### Example 2: Batch Processing with Tags

```bash
# Process multiple genes with descriptive tags
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes-file test_genes.txt \
    --output-dir results/batch_inference \
    --mlflow-enable \
    --mlflow-experiment "batch-processing" \
    --mlflow-tags dataset=test_set batch_size=100 purpose=validation
```

### Example 3: Complete Coverage with Tracking

```bash
# Complete coverage analysis with MLflow
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4/model_multiclass.pkl \
    --training-dataset train_pc_1000_3mers \
    --genes ENSG00000154358 \
    --complete-coverage \
    --output-dir results/complete_coverage \
    --mlflow-enable \
    --mlflow-tags coverage=complete analysis_type=comprehensive
```

## Viewing Results

### Using the MLflow UI

1. **Start the UI:**
```bash
mlflow ui -h 127.0.0.1 -p 5000
```

2. **Navigate to http://127.0.0.1:5000**

3. **Explore the interface:**
   - **Experiments**: Select your experiment from the dropdown
   - **Runs**: View all runs with their metrics
   - **Compare**: Select multiple runs to compare metrics
   - **Artifacts**: Download prediction files and logs

### Key UI Features

#### Run Details Page
- **Parameters**: View all configuration parameters
- **Metrics**: See performance metrics with charts
- **Artifacts**: Browse and download all logged files
- **Tags**: View run metadata and tags

#### Comparison View
- Select multiple runs using checkboxes
- Click "Compare" to see side-by-side metrics
- Generate plots comparing key metrics across runs

#### Search and Filter
- Filter runs by parameter values: `params.inference_mode = "hybrid"`
- Filter by metrics: `metrics.success_rate > 95`
- Filter by tags: `tags.scenario = "training_gene"`

### Programmatic Access

```python
import mlflow
import pandas as pd

# Set tracking URI if using server
mlflow.set_tracking_uri("http://localhost:5000")

# Get experiment
experiment = mlflow.get_experiment_by_name("surveyor-inference")

# List all runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Filter successful runs
successful_runs = runs[runs['metrics.success_rate'] == 100]

# Get specific run artifacts
run_id = successful_runs.iloc[0]['run_id']
client = mlflow.tracking.MlflowClient()
client.download_artifacts(run_id, "gene_manifest.json", dst_path="./downloads")
```

## Advanced Setup

### Environment Capture

Automatically log environment details for reproducibility:

```python
# In your inference script
import mlflow

# Log conda environment
mlflow.log_artifact("environment.yml")

# Log pip requirements
mlflow.log_artifact("requirements.txt")
# Large Artifact Tracking by Reference (No Duplication)

For very large outputs (e.g., FASTA/Parquet batches, `analysis_sequences_*` intermediates), log a directory tree manifest instead of uploading files:

```python
import json, subprocess, os
from pathlib import Path
import mlflow

def snapshot_directory_tree(root_dir: str, out_txt: str, out_json: str) -> None:
    # Human-readable tree
    with open(out_txt, "w") as fh:
        subprocess.run(["bash", "-lc", f"cd {root_dir} && tree -sh --du"], stdout=fh, check=False)
    # Machine-readable manifest
    entries = []
    for path in Path(root_dir).rglob("*"):
        try:
            stat = path.stat()
            entries.append({
                "path": str(path.resolve()),
                "size": stat.st_size,
                "is_dir": path.is_dir()
            })
        except FileNotFoundError:
            pass
    with open(out_json, "w") as fh:
        json.dump({"root": str(Path(root_dir).resolve()), "entries": entries}, fh, indent=2)

# Usage inside a run
root_outputs = "results/test_pc_1000_3mers"
snapshot_directory_tree(root_outputs, "directory_tree.txt", "directory_tree.json")
mlflow.log_artifact("directory_tree.txt")
mlflow.log_artifact("directory_tree.json")
```

This preserves discoverability and provenance without copying huge files into MLflow’s artifact store.

# Log git commit
import subprocess
git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
mlflow.set_tag("git_commit", git_commit)
```

### Custom Metrics

Add custom metrics in your workflow:

```python
# Log custom metric
mlflow.log_metric("custom_accuracy", 0.95)

# Log metric over time
for step in range(10):
    mlflow.log_metric("processing_progress", step * 10, step=step)
```

### Model Registry Integration

Register successful models:

```python
# After successful inference
if results['success']:
    mlflow.register_model(
        f"runs:/{mlflow.active_run().info.run_id}/model",
        "splice-meta-model"
    )
```

## Troubleshooting

### Common Issues

1. **MLflow not installed:**
   ```
   ImportError: No module named 'mlflow'
   ```
   Solution: `pip install mlflow`

2. **Cannot connect to tracking server:**
   ```
   ConnectionError: Failed to connect to MLflow server
   ```
   Solution: Ensure server is running and URI is correct

3. **Artifacts not found:**
   ```
   FileNotFoundError: Artifact not found
   ```
   Solution: Check artifact root directory permissions

### Best Practices

1. **Use descriptive experiment names**: Group related runs logically
2. **Tag consistently**: Use standard tag keys across runs
3. **Log incrementally**: For long-running workflows, log progress metrics
4. **Clean up old runs**: Periodically archive or delete old experiments
5. **Backup MLflow data**: Regular backups of mlruns directory or database

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Inference with MLflow Tracking

on:
  push:
    branches: [main]

jobs:
  inference:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install mlflow
      
      - name: Run inference with MLflow
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_SERVER_URI }}
        run: |
          python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
            --model ${{ github.workspace }}/models/latest.pkl \
            --genes ENSG00000154358 \
            --mlflow-enable \
            --mlflow-tags ci_run=true commit=${{ github.sha }}
```

## Summary

MLflow integration provides:
- 📊 **Comprehensive tracking** of all inference runs
- 📈 **Detailed metrics** for performance analysis
- 📁 **Organized artifacts** for result retrieval
- 🔍 **Easy comparison** between different runs
- 🏷️ **Flexible tagging** for experiment organization
- 🔄 **Reproducibility** through parameter logging

For questions or issues, refer to the [MLflow documentation](https://mlflow.org/docs/latest/index.html) or the Splice Surveyor team.


