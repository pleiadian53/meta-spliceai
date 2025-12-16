# 📊 **Model Comparison Workflow**

A comprehensive guide for comparing inference modes (base_only, hybrid, meta_only) across multiple genes using the Splice Surveyor meta-model system.

> **📋 Note**: This document focuses specifically on **Phase 3 (Inference Comparison)** of the complete workflow. For the full end-to-end process including training data assembly and model training, see [COMPLETE_SPLICE_WORKFLOW.md](COMPLETE_SPLICE_WORKFLOW.md).

## 🎯 **Overview**

The model comparison workflow involves running the same set of genes through three different inference modes and then statistically comparing their performance. This allows you to evaluate:

- **Base Model Performance**: SpliceAI predictions alone
- **Hybrid Performance**: SpliceAI + meta-model for uncertain positions  
- **Meta-Only Performance**: Meta-model recalibration for all positions

### Prerequisites
Before running model comparison, ensure you have:
1. **✅ Trained Meta-Model**: From Phase 2 of the complete workflow (e.g., `results/gene_cv_reg_10k_kmers_run1/`)
2. **✅ Training Dataset**: Used for model training (e.g., `train_regulatory_10k_kmers/`)
3. **✅ Environment Setup**: `mamba activate surveyor`

## Complete Streamlined Workflow

### 🚀 Ultra-Quick Start (1 Command!)
**NEW STREAMLINED APPROACH** - No JSON files, no complex scripts:

```bash
# 1. Prepare genes and get ready-to-use commands in one step
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.prepare_gene_lists \
    --unseen 10 --study-name "my_study"

# 2. Copy-paste the generated commands (shown in output)
# 3. Run analysis commands (also generated in output)
```

**Why this is better:**
- ✅ No intermediate JSON files to manage
- ✅ No complex command-line scripts to construct gene lists  
- ✅ Readable gene types: `--training`, `--unseen`, `--mixed`
- ✅ Auto-generated commands with all proper parameters
- ✅ Includes chunked processing, MLflow, proper chunk sizes

### Alternative: Traditional Multi-Step Approach (Old Method)
For users who prefer more control over the process (or need the old method):

```bash
# 1. Find genes with gene discovery
bash meta_spliceai/splice_engine/meta_models/workflows/inference/find_test_genes.sh \
    --scenario2b-count 10 --output my_study_genes.json

# 2. Extract gene lists
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.extract_gene_lists \
    --input my_study_genes.json --scenarios scenario2b --prefix my_study --show-examples

# 3. Copy and run the generated commands
# 4. Analyze results with inference_analyzer.py
```

### Detailed Step-by-Step Workflow

### Step 0: Identify Target Genes (Optional)

If you don't have a specific gene set, use the enhanced gene finder to identify suitable test genes:

#### Option A: Use Default Gene Counts
```bash
# Find default gene sets (6 training, 8 unseen with artifacts, 8 unseen without artifacts)
cd /path/to/meta-spliceai
bash meta_spliceai/splice_engine/meta_models/workflows/inference/find_test_genes.sh
```

#### Option B: Custom Gene Counts for Different Study Sizes

**Small-Scale Study (Development/Testing)**
```bash
bash meta_spliceai/splice_engine/meta_models/workflows/inference/find_test_genes.sh \
    --scenario1-count 5 \
    --scenario2a-count 5 \
    --scenario2b-count 10 \
    --output small_study_genes.json
```

**Medium-Scale Study (Validation)**
```bash
bash meta_spliceai/splice_engine/meta_models/workflows/inference/find_test_genes.sh \
    --scenario1-count 20 \
    --scenario2a-count 30 \
    --scenario2b-count 50 \
    --output medium_study_genes.json \
    --verbose
```

**Large-Scale Study (Comprehensive Evaluation)**
```bash
bash meta_spliceai/splice_engine/meta_models/workflows/inference/find_test_genes.sh \
    --scenario1-count 100 \
    --scenario2a-count 80 \
    --scenario2b-count 200 \
    --output large_study_genes.json \
    --verbose
```

**Generalization-Focused Study (Focus on Unseen Genes)**
```bash
bash meta_spliceai/splice_engine/meta_models/workflows/inference/find_test_genes.sh \
    --scenario1-count 10 \
    --scenario2a-count 20 \
    --scenario2b-count 100 \
    --output generalization_study_genes.json \
    --verbose
```

### Step 1: Extract Gene Lists for Inference Workflow

After running the gene finder, extract the gene lists into individual files ready for the inference workflow:

#### Option A: Extract All Scenarios (Recommended)
```bash
# Extract all scenarios into separate files
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.extract_gene_lists \
    --input test_genes.json \
    --output-dir gene_lists \
    --show-examples

# This creates:
# - gene_lists/scenario1_genes.txt
# - gene_lists/scenario2a_genes.txt  
# - gene_lists/scenario2b_genes.txt
# - Ready-to-use inference commands
```

#### Option B: Extract Specific Scenario Only
```bash
# Extract only scenario2b genes (unseen genes)
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.extract_gene_lists \
    --input test_genes.json \
    --scenarios scenario2b \
    --prefix unseen_study \
    --show-examples

# This creates:
# - unseen_study_scenario2b_genes.txt
```

#### Option C: Extract with Custom Study Name
```bash
# Extract with study-specific naming
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.extract_gene_lists \
    --input large_study_genes.json \
    --prefix generalization_study \
    --create-combined \
    --study-name "generalization_analysis" \
    --show-examples

# This creates:
# - generalization_study_scenario1_genes.txt
# - generalization_study_scenario2a_genes.txt
# - generalization_study_scenario2b_genes.txt
# - generalization_study_all_genes.txt (combined)
# - Ready-to-use commands with study name
```

### Step 2: Run Inference for All Three Modes

**IMPORTANT**: Always activate the surveyor environment first:
```bash
mamba activate surveyor
```

#### Prerequisites
- **Trained Meta-Model**: `results/gene_cv_pc_1000_3mers_run_4/`
- **Training Dataset**: `train_pc_1000_3mers/`
- **Target Genes**: Gene list files from Step 1

#### Create Log Directory
```bash
mkdir -p logs
```

#### Run All Three Modes
Use the gene files created in Step 1. For example, if you extracted scenario2b genes:

**Base-Only Mode (SpliceAI Only)**
```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4 \
    --training-dataset train_pc_1000_3mers \
    --genes-file gene_lists/scenario2b_genes.txt \
    --output-dir results/comparison_study_base \
    --inference-mode base_only \
    --enable-chunked-processing \
    --chunk-size 5000 \
    --verbose \
    --mlflow-enable \
    --mlflow-experiment "model_comparison_study" \
    2>&1 | tee logs/base_only_inference.log
```

**Hybrid Mode (SpliceAI + Meta-Model for Uncertain Positions)**
```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4 \
    --training-dataset train_pc_1000_3mers \
    --genes-file gene_lists/scenario2b_genes.txt \
    --output-dir results/comparison_study_hybrid \
    --inference-mode hybrid \
    --enable-chunked-processing \
    --chunk-size 5000 \
    --uncertainty-low 0.02 \
    --uncertainty-high 0.80 \
    --verbose \
    --mlflow-enable \
    --mlflow-experiment "model_comparison_study" \
    2>&1 | tee logs/hybrid_inference.log
```

**Meta-Only Mode (Meta-Model for All Positions)**
```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
    --model results/gene_cv_pc_1000_3mers_run_4 \
    --training-dataset train_pc_1000_3mers \
    --genes-file gene_lists/scenario2b_genes.txt \
    --output-dir results/comparison_study_meta \
    --inference-mode meta_only \
    --enable-chunked-processing \
    --chunk-size 5000 \
    --complete-coverage \
    --verbose \
    --mlflow-enable \
    --mlflow-experiment "model_comparison_study" \
    2>&1 | tee logs/meta_only_inference.log
```

### Step 3: Analyze Results with Inference Analyzer

Once all three inference runs complete, analyze and compare the results:

```bash
# Activate environment
mamba activate surveyor

# Run comprehensive analysis
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.inference_analyzer \
    --results-dir results \
    --base-suffix comparison_study_base \
    --hybrid-suffix comparison_study_hybrid \
    --meta-suffix comparison_study_meta \
    --output-dir comparison_analysis_results \
    --batch-size 50 \
    --verbose
```

### Step 4: Generate Statistical Comparison Report

Perform advanced statistical comparison with significance testing:

```bash
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.batch_comparator \
    --analysis-results comparison_analysis_results/detailed_report.json \
    --output-dir statistical_comparison_results \
    --reference-mode base_only \
    --significance-level 0.05 \
    --primary-metric ap_score \
    --include-effect-sizes \
    --create-plots \
    --verbose
```

## Parameter Specifications

### Model and Training Dataset
- **`--model`**: Path to trained meta-model directory
  - Example: `results/gene_cv_pc_1000_3mers_run_4`
  - Contains: `model_multiclass.pkl`, `feature_manifest.csv`, `excluded_features.txt`
  
- **`--training-dataset`**: Path to original training dataset
  - Example: `train_pc_1000_3mers`
  - Contains: `master/gene_manifest.csv`, feature schemas, training metadata

### Processing Options

#### Chunked Processing (Recommended)
```bash
--enable-chunked-processing    # Enable memory-efficient processing
--chunk-size 5000             # Process 5000 positions at a time
```

**⚠️ IMPORTANT DEFAULT BEHAVIOR ISSUE**: 
Currently, `--enable-chunked-processing` is **disabled by default**, but this can cause Out-of-Memory (OOM) errors for:
- **Hybrid mode**: When processing large genes (>100kb)
- **Meta-only mode**: When processing ALL positions (essential for this mode)

**RECOMMENDATION**: Always include `--enable-chunked-processing` in your commands until this default is changed.

**Proposed Default Changes** (for future implementation):
- **Base-only mode**: Keep disabled (not needed)
- **Hybrid mode**: Enable by default with chunk_size=5000
- **Meta-only mode**: Enable by default with chunk_size=5000

#### Coverage Options
```bash
--complete-coverage           # Generate predictions for ALL positions (meta_only)
--uncertainty-low 0.02        # Lower threshold for hybrid mode
--uncertainty-high 0.80       # Upper threshold for hybrid mode
```

#### Performance Options
```bash
--parallel-workers 1          # Number of parallel workers (default: 1)
--verbose                     # Enable detailed logging
--mlflow-enable              # Enable MLflow experiment tracking
```

## Output Structure

Each inference mode creates a structured output directory:

```
results/comparison_study_base/
├── genes/                                    # Per-gene results
│   ├── ENSG00000123456/
│   │   ├── ENSG00000123456_predictions.parquet
│   │   └── ENSG00000123456_statistics.json
│   └── ...
├── selective_inference/                      # Consolidated results
│   ├── predictions/
│   │   ├── base_model_predictions.parquet
│   │   ├── complete_coverage_predictions.parquet
│   │   └── meta_model_predictions.parquet
│   └── cache/
├── performance_analysis/                     # Built-in analysis
│   ├── consolidated_performance_report.txt
│   ├── mode_comparison_report.txt
│   ├── roc_pr_curves_inference.pdf
│   └── curve_metrics.json
├── gene_manifest.json                       # Processing metadata
├── inference_summary.json                   # Summary statistics
├── performance_report.txt                   # Performance summary
└── inference_workflow.log                   # Detailed execution log
```

## MLflow Integration

### Storage Locations
1. **Primary Output**: Files saved to `--output-dir` (e.g., `results/comparison_study_base/`)
2. **MLflow Tracking**: Copies of artifacts logged to MLflow tracking system
   - **Default location**: `./mlruns/`
   - **Custom location**: Specified by `--mlflow-tracking-uri`

### MLflow Artifacts Structure
```
mlruns/{experiment_id}/{run_id}/artifacts/
├── genes/                           # Per-gene artifacts
│   ├── ENSG00000123456/
│   │   ├── ENSG00000123456_predictions.parquet
│   │   └── ENSG00000123456_statistics.json
├── performance_analysis/            # Analysis reports
│   ├── consolidated_performance_report.txt
│   ├── roc_pr_curves_inference.pdf
│   └── mode_comparison_report.txt
├── gene_manifest.json
├── inference_summary.json
├── performance_report.txt
└── directory_tree.txt
```

### MLflow Usage
```bash
# Enable MLflow tracking
--mlflow-enable
--mlflow-experiment "model_comparison_study"
--mlflow-tracking-uri "http://localhost:5000"  # Optional: custom tracking server
--mlflow-tags scenario=scenario2b model_version=v1  # Optional: custom tags
```

## Example Complete Workflows

### Example 1: Ultra-Streamlined Development Test (NEW!)
```bash
# Single command to prepare 5 unseen genes
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.prepare_gene_lists \
    --unseen 5 --study-name "dev_test" --verbose

# Copy-paste the generated commands (shown in output above)
# All commands include proper parameters: model path, training dataset, chunked processing, MLflow, etc.
```

### Example 2: Traditional Small Development Test
```bash
# Step 0: Find test genes
bash find_test_genes.sh --scenario2b-count 5 --output dev_genes.json

# Step 1: Extract genes
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.extract_gene_lists \
    --input dev_genes.json \
    --scenarios scenario2b \
    --prefix dev_test \
    --show-examples

# Step 2: Run inference (all modes)
for mode in base_only hybrid meta_only; do
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
        --model results/gene_cv_pc_1000_3mers_run_4 \
        --training-dataset train_pc_1000_3mers \
        --genes-file dev_test_scenario2b_genes.txt \
        --output-dir results/dev_comparison_${mode} \
        --inference-mode ${mode} \
        --enable-chunked-processing \
        --chunk-size 5000 \
        --verbose
done

# Step 3: Analyze results
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.inference_analyzer \
    --results-dir results \
    --base-suffix dev_comparison_base_only \
    --hybrid-suffix dev_comparison_hybrid \
    --meta-suffix dev_comparison_meta_only \
    --output-dir dev_analysis_results

# Step 4: Statistical comparison
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.batch_comparator \
    --analysis-results dev_analysis_results/detailed_report.json \
    --output-dir dev_statistical_comparison \
    --reference-mode base_only
```

### Example 2: Ultra-Streamlined Production Study (NEW!)
```bash
# Single command for 100 unseen genes - no JSON intermediates!
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.prepare_gene_lists \
    --unseen 100 \
    --study-name "production_generalization_study" \
    --prefix "production" \
    --output-dir gene_lists \
    --verbose

# Output: Ready-to-use production_unseen_genes.txt + all inference commands!
# Just copy-paste the generated commands - no manual parameter setup needed!
```

### Example 3: Traditional Production Study (Old Method - AWKWARD!)
```bash
# Step 0: Find large gene set focused on unseen genes
bash find_test_genes.sh \
    --scenario1-count 20 \
    --scenario2a-count 50 \
    --scenario2b-count 100 \
    --output production_genes.json \
    --verbose

# Step 1: Extract scenario 2B genes (unseen genes) - COMPLICATED!
python3 -c "
import json
with open('production_genes.json') as f:
    data = json.load(f)
genes = [g['gene_id'] for g in data['scenario2b']['genes']]
with open('unseen_genes_100.txt', 'w') as f:
    for gene in genes:
        f.write(f'{gene}\n')
print(f'Extracted {len(genes)} unseen genes for generalization study')
"

# Step 2: Run inference with MLflow tracking
mkdir -p logs

for mode in base_only hybrid meta_only; do
    echo "Running ${mode} mode..."
    python -m meta_spliceai.splice_engine.meta_models.workflows.inference.main_inference_workflow \
        --model results/gene_cv_pc_1000_3mers_run_4 \
        --training-dataset train_pc_1000_3mers \
        --genes-file unseen_genes_100.txt \
        --output-dir results/generalization_study_${mode} \
        --inference-mode ${mode} \
        --enable-chunked-processing \
        --chunk-size 5000 \
        --complete-coverage \
        --verbose \
        --mlflow-enable \
        --mlflow-experiment "generalization_study_100_genes" \
        --mlflow-tags study_type=generalization gene_count=100 scenario=2b \
        2>&1 | tee logs/${mode}_inference.log
    
    echo "Completed ${mode} mode"
done

# Step 3: Comprehensive analysis
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.inference_analyzer \
    --results-dir results \
    --base-suffix generalization_study_base_only \
    --hybrid-suffix generalization_study_hybrid \
    --meta-suffix generalization_study_meta_only \
    --output-dir generalization_analysis_results \
    --batch-size 25 \
    --verbose

# Step 4: Statistical comparison with publication-ready outputs
python -m meta_spliceai.splice_engine.meta_models.workflows.inference.batch_comparator \
    --analysis-results generalization_analysis_results/detailed_report.json \
    --output-dir generalization_statistical_comparison \
    --reference-mode base_only \
    --significance-level 0.01 \
    --primary-metric ap_score \
    --include-effect-sizes \
    --create-plots \
    --verbose
```

## Key Parameters Explained

### Model Specification
```bash
--model results/gene_cv_pc_1000_3mers_run_4
```
This points to a directory containing:
- `model_multiclass.pkl`: The trained meta-model
- `feature_manifest.csv`: List of features used during training
- `excluded_features.txt`: Features excluded during training
- `train.features.json`: Training schema information

### Training Dataset Specification
```bash
--training-dataset train_pc_1000_3mers
```
This points to a directory containing:
- `master/gene_manifest.csv`: List of genes used in training
- Training metadata and schemas
- Used for feature harmonization and consistency checking

### Processing Recommendations

#### Chunked Processing (RECOMMENDED)
```bash
--enable-chunked-processing    # Should be DEFAULT for hybrid and meta_only
--chunk-size 5000             # Balance between memory and I/O efficiency
```

**Current Issue**: Chunked processing is disabled by default, but should be enabled by default for:
- **Hybrid mode**: To handle large genes (>10kb) without OOM
- **Meta-only mode**: Essential for processing all positions without memory constraints
- **Base-only mode**: Optional but recommended for consistency

#### Memory Management
- **Small genes (<50kb)**: Default chunk size (5000) works well
- **Large genes (>200kb)**: Consider smaller chunk size (2000-3000)
- **Very large genes (>1MB)**: Use chunk size 1000-2000

#### Inference Mode Selection
- **base_only**: Fastest, uses only SpliceAI predictions
- **hybrid**: Balanced, uses meta-model for uncertain positions (0.02-0.80 range)
- **meta_only**: Most comprehensive, recalibrates all positions

## Performance Monitoring

### Log File Analysis
Monitor progress using the generated log files:
```bash
# Check progress
tail -f logs/meta_only_inference.log

# Check for errors
grep -i "error\|failed\|traceback" logs/*.log

# Check completion status
grep "COMPLETED SUCCESSFULLY" logs/*.log
```

### MLflow Dashboard
If MLflow is enabled, monitor progress via web interface:
```bash
# Start MLflow server (if not already running)
mlflow server --host 0.0.0.0 --port 5000

# View experiments at http://localhost:5000
```

## Expected Outputs

### Analysis Results
```
comparison_analysis_results/
├── summary_report.json          # High-level metrics across all modes
├── detailed_report.json         # Per-gene detailed metrics  
├── per_gene_metrics.csv         # Tabular data for further analysis
└── analysis_log.txt             # Processing log
```

### Statistical Comparison Results
```
statistical_comparison_results/
├── comparison_report.json       # Detailed statistical results
├── comparison_report.txt        # Human-readable summary
├── per_gene_comparisons.csv     # Per-gene improvement data
└── visualizations/              # Interactive plots (if --create-plots)
    ├── f1_score_comparison.html
    ├── ap_score_comparison.html
    └── improvement_scatter.html
```

## Troubleshooting

### Common Issues

#### Memory Issues
- **Symptom**: OOM errors during meta_only mode
- **Solution**: Enable chunked processing with smaller chunk size
```bash
--enable-chunked-processing --chunk-size 2000
```

#### Feature Inconsistency
- **Symptom**: "Feature count mismatch" errors
- **Solution**: Ensure training dataset path is correct and feature consistency checker is working

#### Missing Dependencies
- **Symptom**: "No module named 'polars'" errors
- **Solution**: Activate surveyor environment: `mamba activate surveyor`

#### Partial Results
- **Symptom**: Only some genes processed successfully
- **Solution**: Check individual gene logs in the output directory

### Performance Optimization

#### For Large Gene Sets (100+ genes)
```bash
--parallel-workers 4              # Use multiple cores
--batch-size 25                   # Smaller batches for analysis
--chunk-size 3000                 # Smaller chunks for memory efficiency
```

#### For Very Large Genes (>500kb)
```bash
--chunk-size 1000                 # Very small chunks
--max-positions 50000             # Limit positions per gene if needed
```

## Interpretation Guidelines

### Statistical Significance
- **p < 0.05**: Statistically significant difference
- **p < 0.01**: Highly significant difference
- **p < 0.001**: Very highly significant difference

### Practical Significance for Splice Site Prediction
- **F1 improvement > 0.05**: Meaningful improvement
- **AP improvement > 0.1**: Substantial improvement in ranking
- **AUC improvement > 0.01**: Notable discrimination improvement

### Expected Results by Scenario

#### Scenario 1 (Training Genes)
- **Meta-model should excel**: High AP and F1 improvements expected
- **Hybrid mode optimal**: Good balance of performance and efficiency

#### Scenario 2A (Unseen Genes with Artifacts)  
- **Moderate improvements**: Meta-model should still help
- **Variable performance**: Some genes benefit more than others

#### Scenario 2B (Completely Unseen Genes)
- **Conservative improvements**: Meta-model more cautious
- **Higher thresholds needed**: Meta-model requires different optimization
- **Focus on AP**: Average Precision most informative metric

## File Management

### Recommended Directory Structure
```
meta_spliceai_analysis/
├── gene_discovery/
│   ├── test_genes.json
│   ├── scenario1_genes.txt
│   ├── scenario2a_genes.txt
│   └── scenario2b_genes.txt
├── inference_results/
│   ├── comparison_study_base/
│   ├── comparison_study_hybrid/
│   └── comparison_study_meta/
├── analysis_results/
│   ├── comparison_analysis_results/
│   └── statistical_comparison_results/
├── logs/
│   ├── base_only_inference.log
│   ├── hybrid_inference.log
│   └── meta_only_inference.log
└── mlruns/                              # MLflow tracking data
```

### Cleanup Commands
```bash
# Clean up large intermediate files (optional)
find results/ -name "*.parquet" -size +100M -delete

# Archive completed analysis
tar -czf analysis_$(date +%Y%m%d).tar.gz results/ logs/ *.json
```

## Integration with Downstream Analysis

### Export to R/Python
```python
import pandas as pd

# Load per-gene metrics for custom analysis
metrics = pd.read_csv('comparison_analysis_results/per_gene_metrics.csv')

# Filter by mode
base_metrics = metrics[metrics['mode'] == 'base_only']
meta_metrics = metrics[metrics['mode'] == 'meta_only']

# Custom analysis
improvement = meta_metrics['f1_score'].values - base_metrics['f1_score'].values
```

### Publication-Ready Outputs
- **Tables**: Use `per_gene_comparisons.csv` for LaTeX tables
- **Figures**: Interactive HTML plots can be exported to PNG/PDF
- **Statistics**: JSON reports contain all statistical test results

## Best Practices

### Resource Planning
- **CPU**: 1-4 cores recommended
- **Memory**: 8-32GB depending on gene sizes and chunk size
- **Storage**: ~100MB-1GB per 100 genes
- **Time**: ~1-10 minutes per gene depending on size and mode

### Reproducibility
- **Use consistent gene lists** across all modes
- **Set random seeds** if using sampling
- **Document parameters** in MLflow tags
- **Save complete command lines** in log files

### Quality Control
- **Check log files** for errors and warnings
- **Verify gene counts** match expectations
- **Compare with CV results** for training genes
- **Use statistical significance** for decision making

This workflow provides a comprehensive framework for rigorous evaluation of meta-model performance across different gene sets and scenarios.
