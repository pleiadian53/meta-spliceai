# Error Model Usage Examples

This document provides comprehensive examples for using the MetaSpliceAI error modeling workflow, covering the complete pipeline from data access to model evaluation and experiment tracking.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start Examples](#quick-start-examples)
3. [Data Processing Examples](#data-processing-examples)
4. [Model Training Examples](#model-training-examples)
5. [Evaluation & Analysis](#evaluation--analysis)
6. [MLflow Integration](#mlflow-integration)
7. [Advanced Configurations](#advanced-configurations)
8. [Batch Processing Scripts](#batch-processing-scripts)

## Prerequisites

Ensure you have the required environment and data:

```bash
# Activate the surveyor environment
mamba activate surveyor

# Verify data directory exists with meta-model artifacts
ls data/ensembl/spliceai_eval/meta_models/analysis_sequences_*
ls data/ensembl/spliceai_eval/meta_models/splice_positions_enhanced_*

# Check for chunked analysis sequence files (auto-consolidated if needed)
ls data/ensembl/spliceai_eval/meta_models/analysis_sequences_*_chunk_*.tsv
```

## Quick Start Examples

### Minimal Test Run

```bash
# Quick test with minimal settings
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/test_run \
    --error-type fp \
    --num-epochs 1 \
    --max-ig-samples 50 \
    --batch-size 8
```

### Standard FP vs TP Analysis

```bash
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/fp_analysis \
    --error-type fp \
    --num-epochs 10 \
    --batch-size 16 \
    --max-ig-samples 500
```

### Standard FN vs TN Analysis

```bash
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/fn_analysis \
    --error-type fn \
    --num-epochs 10 \
    --batch-size 16 \
    --max-ig-samples 500
```

## Data Processing Examples

### Manual Data Consolidation

```python
from meta_spliceai.splice_engine.meta_models.error_model import AnalysisDataConsolidator
from pathlib import Path

# Manual consolidation (automatic in workflow)
consolidator = AnalysisDataConsolidator(verbose=True)
result = consolidator.consolidate_and_filter_analysis_sequences(
    data_dir=Path("data/ensembl/spliceai_eval/meta_models"),
    output_file=Path("full_analysis_sequences_simple_fp_tp.tsv"),
    error_type="fp_vs_tp",
    max_rows_per_type=25000
)

print(f"Consolidated {result['chunk_files_processed']} files")
print(f"Total rows: {result['total_rows']:,}")
print(f"Rows by type: {result['rows_by_type']}")
```

### Contextual Sequence Extraction

```python
from meta_spliceai.splice_engine.meta_models.error_model import ErrorDatasetPreparer, ErrorModelConfig

# Configure contextual sequence extraction
config = ErrorModelConfig(
    context_length=200,  # ±100 nucleotides around splice site
    error_label="FP",
    correct_label="TP",
    splice_type="any"
)

# Prepare datasets with contextual sequences
preparer = ErrorDatasetPreparer(config)
dataset_info = preparer.prepare_dataset_from_dataframe(
    df=analysis_df,
    error_label="FP",
    correct_label="TP"
)

print(f"Training samples: {len(dataset_info['datasets']['train'])}")
print(f"Validation samples: {len(dataset_info['datasets']['val'])}")
print(f"Test samples: {len(dataset_info['datasets']['test'])}")
```

### Gene-Level Subsampling

```bash
# Sample specific genes
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/gene_specific \
    --genes BRCA1 TP53 EGFR \
    --error-type fp

# Sample from gene list file
cat > cancer_genes.txt << EOF
BRCA1
BRCA2
TP53
KRAS
EGFR
MYC
EOF

python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/cancer_genes \
    --genes-file cancer_genes.txt \
    --error-type fp
```

### Chromosome-Specific Sampling

```bash
# Focus on specific chromosomes
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/chr1_2_analysis \
    --chromosomes 1 2 \
    --sample-size 50000 \
    --error-type fp

# Autosomal chromosomes only
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/autosomal \
    --chromosomes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 \
    --sample-size 100000 \
    --error-type fn
```

## Model Training Examples

### Donor-Specific Analysis

```bash
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/donor_fp_analysis \
    --error-type fp \
    --splice-type donor \
    --context-length 300 \
    --num-epochs 15
```

### Acceptor-Specific Analysis

```bash
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/acceptor_fp_analysis \
    --error-type fp \
    --splice-type acceptor \
    --context-length 300 \
    --num-epochs 15
```

### Custom Model Architecture

```bash
# Large model with extended context
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/large_model \
    --error-type fp \
    --model-name zhihan1996/DNABERT-2-117M \
    --context-length 400 \
    --batch-size 8 \
    --num-epochs 15 \
    --learning-rate 1e-5
```

### Multi-Modal Feature Configuration

```python
from meta_spliceai.splice_engine.meta_models.error_model import ErrorModelConfig

# Configure multi-modal features
config = ErrorModelConfig(
    # Primary: DNA sequence (always included)
    
    # Additional tabular features:
    include_base_scores=True,        # Raw SpliceAI scores
    include_context_features=True,   # Context-aware features
    include_donor_features=True,     # Donor-specific features  
    include_acceptor_features=True,  # Acceptor-specific features
    include_derived_features=True,   # Statistical features
    include_genomic_features=False   # Gene-level features (optional)
)

# Features automatically extracted:
# - Base Scores: donor_score, acceptor_score, neither_score, score
# - Context Features: context_score_m1, context_score_p1, context_neighbor_mean
# - Donor Features: donor_diff_m1, donor_surge_ratio, donor_peak_height_ratio
# - Acceptor Features: acceptor_diff_m1, acceptor_surge_ratio, acceptor_peak_height_ratio
# - Derived Features: probability_entropy, signal_strength_ratio, type_signal_difference
```

## Evaluation & Analysis

### Model Performance Evaluation

```python
from meta_spliceai.splice_engine.meta_models.error_model import ModelEvaluator

# Evaluate trained model
evaluator = ModelEvaluator(config)
metrics = evaluator.evaluate_model(
    model=trained_model,
    test_dataset=test_dataset,
    output_dir=Path("evaluation_results")
)

# Performance metrics
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
```

### Integrated Gradients Analysis

```bash
# Extended IG analysis
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/extended_ig \
    --error-type fp \
    --max-ig-samples 1000 \
    --num-epochs 10
```

```python
from meta_spliceai.splice_engine.meta_models.error_model import IntegratedGradientsAnalyzer

# Perform interpretability analysis
ig_analyzer = IntegratedGradientsAnalyzer(config)
ig_results = ig_analyzer.analyze_model_predictions(
    model=trained_model,
    dataset=test_dataset,
    num_samples=500,
    steps=50
)

# Key insights from IG analysis
print("Top discriminative sequence motifs:")
for motif, importance in ig_results['top_motifs'].items():
    print(f"  {motif}: {importance:.4f}")

print("Most important tabular features:")
for feature, importance in ig_results['feature_importance'].items():
    print(f"  {feature}: {importance:.4f}")
```

### Skip Training for Analysis Only

```bash
# Use existing model for IG analysis
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/ig_only \
    --error-type fp \
    --skip-training \
    --max-ig-samples 2000
```

## MLflow Integration

### Basic MLflow Tracking

```bash
# Start MLflow server (optional - can use local tracking)
mlflow server --host 0.0.0.0 --port 5000 &

# Run workflow with MLflow tracking
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir results/mlflow_experiment \
    --error-type fp \
    --enable-mlflow \
    --mlflow-tracking-uri "http://localhost:5000" \
    --mlflow-experiment-name "splice_error_fp_analysis"
```

### Multi-Platform Tracking

```bash
# Track with MLflow, Weights & Biases, and TensorBoard
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir results/multi_tracking \
    --error-type fp \
    --enable-mlflow \
    --mlflow-experiment-name "production_error_models" \
    --enable-wandb \
    --wandb-project "meta-spliceai-errors" \
    --enable-tensorboard
```

### MLflow Experiment Analysis

```python
import mlflow
import mlflow.pytorch

# Connect to MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("splice_error_fp_analysis")

# Query experiments
experiments = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.test_f1 > 0.85",
    order_by=["metrics.test_f1 DESC"]
)

print("Top performing models:")
for idx, run in experiments.iterrows():
    print(f"Run {run['run_id'][:8]}: F1={run['metrics.test_f1']:.4f}, "
          f"Context={run['params.context_length']}, "
          f"LR={run['params.learning_rate']}")

# Load best model
best_run_id = experiments.iloc[0]['run_id']
model = mlflow.pytorch.load_model(f"runs:/{best_run_id}/model")
```

## Advanced Configurations

### Memory-Optimized Settings

```bash
# For limited GPU memory
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/memory_optimized \
    --error-type fp \
    --batch-size 4 \
    --context-length 150 \
    --max-ig-samples 100 \
    --use-mixed-precision
```

### High-Performance Settings

```bash
# For powerful hardware
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/high_performance \
    --error-type fp \
    --batch-size 64 \
    --context-length 512 \
    --num-epochs 20 \
    --max-ig-samples 2000 \
    --device cuda \
    --use-mixed-precision
```

### Balanced Dataset Creation

```bash
# Create balanced dataset with equal FP/TP samples
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/balanced \
    --error-type fp \
    --sample-size 20000 \
    --balanced-sampling
```

## Batch Processing Scripts

### Compare Context Lengths

```bash
#!/bin/bash
# compare_contexts.sh

for context in 150 200 300 400; do
    echo "Training with context length: $context"
    python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
        --data-dir data/ensembl/spliceai_eval/meta_models \
        --output-dir results/context_${context} \
        --error-type fp \
        --context-length $context \
        --enable-mlflow \
        --mlflow-experiment-name "context_comparison" \
        --num-epochs 8
done
```

### Compare Learning Rates

```bash
#!/bin/bash
# compare_learning_rates.sh

for lr in 1e-5 2e-5 5e-5; do
    echo "Training with learning rate: $lr"
    python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
        --data-dir data/ensembl/spliceai_eval/meta_models \
        --output-dir results/lr_${lr} \
        --error-type fp \
        --learning-rate $lr \
        --enable-mlflow \
        --mlflow-experiment-name "lr_comparison" \
        --num-epochs 8
done
```

### Process All Error Types

```bash
#!/bin/bash
# process_all_errors.sh

# Process both error types with both splice types
for error_type in fp fn; do
    for splice_type in donor acceptor any; do
        echo "Processing: $error_type - $splice_type"
        python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
            --data-dir data/ensembl/spliceai_eval/meta_models \
            --output-dir results/${error_type}_${splice_type} \
            --error-type $error_type \
            --splice-type $splice_type \
            --num-epochs 10 \
            --enable-mlflow \
            --mlflow-experiment-name "comprehensive_analysis"
    done
done
```

### GPU Monitoring During Training

```bash
#!/bin/bash
# monitored_training.sh

# Monitor GPU usage during training
nvidia-smi -l 1 > gpu_usage.log &
NVIDIA_PID=$!

# Run workflow
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/monitored_run \
    --error-type fp \
    --batch-size 16 \
    --num-epochs 10

# Stop monitoring
kill $NVIDIA_PID
echo "GPU usage log saved to gpu_usage.log"
```

## Expected Outputs

After successful completion:

```
output_directory/
├── error_model/
│   ├── config.json                      # Model configuration
│   ├── pytorch_model.bin                # Model weights
│   └── tokenizer/                       # Tokenizer files
├── training_config.json                 # Training hyperparameters
├── feature_columns.json                 # Feature names used
├── consolidated_data.tsv                # Processed training data
├── logs/
│   ├── workflow.log                     # Complete workflow log
│   ├── training.log                     # Training progress
│   └── evaluation.log                   # Evaluation metrics
├── ig_analysis/
│   ├── attributions.parquet             # Raw IG attributions
│   ├── analysis_results.json            # Pattern analysis
│   ├── token_frequencies.csv            # Token importance
│   └── motif_analysis.json              # Sequence motifs
├── visualizations/
│   ├── token_frequency_comparison.png   # Token usage comparison
│   ├── attribution_distribution.png     # Attribution statistics
│   ├── top_tokens_analysis.png          # Important tokens
│   ├── positional_analysis.png          # Position patterns
│   └── confusion_matrix.png             # Model performance
├── evaluation/
│   ├── test_predictions.csv             # Predictions on test set
│   ├── performance_metrics.json         # Detailed metrics
│   └── roc_curve.png                    # ROC curve
└── analysis_report.md                   # Comprehensive report
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size and context length
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --batch-size 4 \
    --context-length 150 \
    --max-ig-samples 100
```

#### Missing Data Files
```bash
# Verify data directory structure
ls -la data/ensembl/spliceai_eval/meta_models/
ls data/ensembl/spliceai_eval/meta_models/analysis_sequences_*_chunk_*.tsv
```

#### Import Errors
```bash
# Verify environment
mamba activate surveyor
python -c "from meta_spliceai.splice_engine.meta_models.error_model import run_error_model_workflow; print('Success')"
```

#### Slow Training
```bash
# Enable performance optimizations
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --use-mixed-precision \
    --device cuda \
    --batch-size 32
```

## Next Steps

1. **Review results**: Check `analysis_report.md` for key findings
2. **Analyze patterns**: Examine token frequencies and motifs
3. **Compare models**: Use MLflow UI to compare experiments
4. **Deploy models**: Use saved models for production inference
5. **Iterate**: Refine based on insights from IG analysis
