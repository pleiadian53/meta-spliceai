# Error Model Usage Examples

This document provides comprehensive examples for using the MetaSpliceAI error modeling workflow, covering the complete pipeline from data access to model evaluation and experiment tracking.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Complete Workflow Overview](#complete-workflow-overview)
3. [Data Processing Pipeline](#data-processing-pipeline)
4. [Model Architecture & Training](#model-architecture--training)
5. [Evaluation & Analysis](#evaluation--analysis)
6. [MLflow Integration](#mlflow-integration)
7. [Advanced Usage Examples](#advanced-usage-examples)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

Ensure you have the required environment and data:

```bash
# Activate the surveyor environment
mamba activate surveyor

# Verify data directory exists with meta-model artifacts (from project root)
ls data/ensembl/spliceai_eval/meta_models/analysis_sequences_*
ls data/ensembl/spliceai_eval/meta_models/splice_positions_enhanced_*

# Check for chunked analysis sequence files (auto-consolidated if needed)
ls data/ensembl/spliceai_eval/meta_models/analysis_sequences_*_chunk_*.tsv
```

## Complete Workflow Overview

The error model workflow consists of several key stages:

1. **Data Access & Consolidation**: Automatic discovery and consolidation of chunked analysis sequences
2. **Feature Engineering**: Extraction of contextual DNA sequences and additional tabular features
3. **Model Architecture**: Multi-modal transformer combining sequence and numerical features
4. **Training**: Fine-tuning pre-trained DNA language models for error classification
5. **Evaluation**: Performance assessment and interpretability analysis
6. **Experiment Tracking**: MLflow integration for reproducible experiments

### Complete End-to-End Example

```bash
# Full workflow with MLflow tracking
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir results/fp_vs_tp_analysis \
    --error-type fp \
    --device auto \
    --batch-size 16 \
    --num-epochs 10 \
    --max-ig-samples 500 \
    --enable-mlflow \
    --mlflow-experiment-name "splice_error_analysis"
```

## Data Processing Pipeline

### 1. Artifact Access and Automatic Consolidation

The workflow automatically handles data preparation:

```python
from meta_spliceai.splice_engine.meta_models.error_model import AnalysisDataConsolidator

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

### 2. Contextual Sequence Extraction

The workflow extracts contextual DNA sequences around splice sites:

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

### 3. Feature Engineering

The system automatically extracts multiple feature categories:

```python
# Feature configuration in ErrorModelConfig
config = ErrorModelConfig(
    # Primary feature: DNA sequence (always included)
    
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

## Model Architecture & Training

### 1. Multi-Modal Transformer Architecture

The error model uses a hybrid architecture combining DNA sequence and tabular features:

```python
from meta_spliceai.splice_engine.meta_models.error_model import TransformerTrainer, ErrorModelConfig

# Model architecture configuration
config = ErrorModelConfig(
    # Base model: Pre-trained DNA language model
    model_name="zhihan1996/DNABERT-2-117M",  # or "microsoft/DialoGPT-medium"
    
    # Architecture parameters
    context_length=200,           # Sequence context window
    hidden_size=768,             # Transformer hidden dimension
    num_attention_heads=12,      # Multi-head attention
    num_hidden_layers=12,        # Transformer layers
    
    # Multi-modal fusion
    tabular_feature_dim=64,      # Tabular feature embedding dimension
    fusion_method="concatenate", # How to combine sequence + tabular features
    
    # Classification head
    num_labels=2,                # Binary classification (error vs correct)
    dropout_rate=0.1,           # Regularization
    
    # Training parameters
    learning_rate=2e-5,          # Fine-tuning learning rate
    batch_size=16,               # Training batch size
    num_epochs=10,               # Training epochs
    warmup_steps=500,            # Learning rate warmup
    weight_decay=0.01            # L2 regularization
)

# Model architecture overview:
# Input: [DNA Sequence] + [Tabular Features]
#   ↓
# DNA Tokenizer → Sequence Embeddings (768-dim)
# Tabular Features → Dense Embedding (64-dim)
#   ↓
# Multi-Head Attention Layers (12 layers)
#   ↓
# Feature Fusion (concatenate/add/multiply)
#   ↓
# Classification Head → Binary Prediction (error/correct)
```

### 2. Training Process

```python
# Initialize trainer
trainer = TransformerTrainer(config)

# Training with automatic validation
results = trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    output_dir=Path("models/error_classifier")
)

print(f"Training completed in {results['training_time']:.2f}s")
print(f"Best validation F1: {results['best_val_f1']:.4f}")
print(f"Final test accuracy: {results['test_accuracy']:.4f}")
```

### 3. Key Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | DNABERT-2-117M | Pre-trained DNA language model |
| `context_length` | 200 | Sequence window (±100 around splice site) |
| `learning_rate` | 2e-5 | Fine-tuning learning rate |
| `batch_size` | 16 | Training batch size |
| `num_epochs` | 10 | Training epochs |
| `dropout_rate` | 0.1 | Regularization strength |
| `warmup_steps` | 500 | Learning rate warmup |

## Evaluation & Analysis

### 1. Model Performance Evaluation

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

### 2. Interpretability Analysis with Integrated Gradients

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

## MLflow Integration

### 1. Basic MLflow Tracking

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

### 2. Advanced MLflow Configuration

```bash
# Multi-platform experiment tracking
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir results/comprehensive_tracking \
    --error-type fp \
    --enable-mlflow \
    --mlflow-experiment-name "production_error_models" \
    --enable-wandb \
    --wandb-project "meta-spliceai-errors" \
    --enable-tensorboard
```

### 3. MLflow Experiment Analysis

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

### 4. Automated Experiment Comparison

```bash
#!/bin/bash
# Script: compare_experiments.sh

# Compare different context lengths with MLflow
for context in 150 200 300; do
    python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
        --data-dir data/ensembl/spliceai_eval/meta_models \
        --output-dir results/context_${context} \
        --error-type fp \
        --context-length $context \
        --enable-mlflow \
        --mlflow-experiment-name "context_length_comparison" \
        --num-epochs 8
done

# Compare different learning rates
for lr in 1e-5 2e-5 5e-5; do
    python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
        --data-dir data/ensembl/spliceai_eval/meta_models \
        --output-dir results/lr_${lr} \
        --error-type fp \
        --learning-rate $lr \
        --enable-mlflow \
        --mlflow-experiment-name "learning_rate_comparison" \
        --num-epochs 8
done
```

## Advanced Usage Examples

### 1. Quick Test Run

Start with a minimal test to verify everything works:

```bash
# Run from project root (recommended)
python -m meta_spliceai.splice_engine.meta_models.error_model.run_quick_analysis \
    --data_dir data/ensembl/spliceai_eval/meta_models \
    --test

# Alternative: Run from module directory
# cd meta_spliceai/splice_engine/meta_models/error_model
# python run_quick_analysis.py --data_dir data/ensembl/spliceai_eval/meta_models --test
```

This runs a fast test with:
- 1 training epoch
- 50 samples for IG analysis
- Small batch size (8)

### 2. Standard FP vs TP Analysis

Run a complete FP (False Positive) vs TP (True Positive) analysis:

```bash
# Run from project root (recommended)
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data_dir data/ensembl/spliceai_eval/meta_models \
    --output_dir output/fp_analysis \
    --error-type fp \
    --num-epochs 10 \
    --batch-size 16 \
    --max-ig-samples 500
```

### 3. Standard FN vs TP Analysis

Run a complete FN (False Negative) vs TP analysis:

```bash
# Run from project root (recommended)
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/fn_analysis \
    --error-type fn \
    --num-epochs 10 \
    --batch-size 16 \
    --max-ig-samples 500
```

### 3. Donor-Specific Analysis

Analyze only donor splice sites:

```bash
# Run from project root (recommended)
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/donor_fp_analysis \
    --error-type fp \
    --splice-type donor \
    --context-length 300 \
    --num-epochs 15
```

### 4. Acceptor-Specific Analysis

Analyze only acceptor splice sites:

```bash
# Run from project root (recommended)
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/acceptor_fp_analysis \
    --error-type fp \
    --splice-type acceptor \
    --context-length 300 \
    --num-epochs 15
```

### 5. Custom Model Configuration

Use different model architectures and parameters:

```bash
# Large model with extended context
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/large_model_analysis \
    --error-type fp \
    --model-name "zhihan1996/DNABERT-2-117M" \
    --context-length 400 \
    --batch-size 8 \
    --num-epochs 15 \
    --learning-rate 1e-5 \
    --enable-mlflow
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```bash
# Reduce batch size and context length
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir output/memory_optimized \
    --error-type fp \
    --batch-size 4 \
    --context-length 150 \
    --max-ig-samples 100
```

#### 2. Missing Data Files
```bash
# Verify data directory structure
ls -la data/ensembl/spliceai_eval/meta_models/

# Check for chunked files (auto-consolidated)
ls data/ensembl/spliceai_eval/meta_models/analysis_sequences_*_chunk_*.tsv

# Verify consolidated files
ls data/ensembl/spliceai_eval/meta_models/full_analysis_sequences_*.tsv
```

#### 3. Import Errors
```bash
# Verify environment
mamba activate surveyor
which python

# Test imports
python -c "from meta_spliceai.splice_engine.meta_models.error_model import run_error_model_workflow; print('Success')"
```

#### 4. MLflow Connection Issues
```bash
# Use local MLflow tracking (no server required)
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --error-type fp \
    --enable-mlflow \
    --mlflow-experiment-name "local_experiment"
    # No --mlflow-tracking-uri needed for local tracking
```

### Performance Monitoring

```bash
# Monitor GPU usage during training
nvidia-smi -l 1 &
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
```

## Expected Outputs

After successful completion, you should see the following directory structure:

```
output_directory/
├── models/
│   ├── best_model.pt                    # Trained PyTorch model
│   ├── tokenizer/                       # Saved tokenizer
│   ├── config.json                      # Model configuration
│   └── training_args.json               # Training arguments
├── logs/
│   ├── workflow.log                     # Complete workflow log
│   ├── training.log                     # Training progress log
│   └── evaluation.log                   # Evaluation metrics log
├── ig_analysis/
│   ├── attributions.parquet             # Raw integrated gradients
│   ├── analysis_results.json            # IG analysis summary
│   ├── token_frequencies.csv            # Token importance ranking
│   ├── feature_importance.csv           # Tabular feature importance
│   └── motif_analysis.json              # Discovered sequence motifs
├── visualizations/
│   ├── token_frequency_comparison.png   # Error vs correct token usage
│   ├── attribution_distribution.png     # Attribution score distributions
│   ├── top_tokens_analysis.png          # Most discriminative tokens
│   ├── positional_analysis.png          # Position-specific importance
│   ├── feature_importance_plot.png      # Tabular feature importance
│   └── confusion_matrix.png             # Model performance visualization
├── evaluation/
│   ├── test_predictions.csv             # Model predictions on test set
│   ├── performance_metrics.json         # Detailed performance metrics
│   ├── classification_report.txt        # Sklearn classification report
│   └── roc_curve.png                    # ROC curve visualization
├── mlflow_artifacts/                    # MLflow experiment artifacts (if enabled)
│   ├── model/                           # MLflow model format
│   ├── metrics.json                     # Logged metrics
│   └── params.json                      # Logged parameters
└── analysis_report.md                   # Comprehensive summary report
```

### Key Output Files Explained

| File | Description |
|------|-------------|
| `analysis_report.md` | Human-readable summary with key findings and biological insights |
| `models/best_model.pt` | Best performing model checkpoint for inference |
| `ig_analysis/token_frequencies.csv` | Ranked list of most discriminative sequence tokens |
| `evaluation/performance_metrics.json` | Complete performance metrics (accuracy, F1, AUC, etc.) |
| `visualizations/*.png` | Publication-ready plots for presentations and papers |

### Sample Analysis Report Contents

The `analysis_report.md` includes:

1. **Executive Summary**: Key findings and model performance
2. **Data Overview**: Dataset statistics and class distribution
3. **Model Architecture**: Configuration and parameter summary
4. **Performance Metrics**: Detailed evaluation results
5. **Feature Importance**: Most discriminative sequence and tabular features
6. **Biological Insights**: Discovered splice motifs and patterns
7. **Recommendations**: Suggested next steps for model improvement

### MLflow Integration Outputs

When MLflow is enabled, additional artifacts are logged:

- **Metrics**: Training/validation loss, accuracy, F1, AUC-ROC
- **Parameters**: All model and training hyperparameters
- **Artifacts**: Model files, visualizations, analysis results
- **Tags**: Experiment metadata for easy filtering and comparison

## Next Steps

### 1. Analysis and Interpretation
```bash
# Review the comprehensive analysis report
cat output/fp_analysis/analysis_report.md

# Examine top discriminative tokens
head -20 output/fp_analysis/ig_analysis/token_frequencies.csv

# Check model performance
cat output/fp_analysis/evaluation/performance_metrics.json
```

### 2. Model Deployment
```python
import torch
from meta_spliceai.splice_engine.meta_models.error_model import ErrorModelConfig

# Load trained model for inference
config = ErrorModelConfig.from_json("output/fp_analysis/models/config.json")
model = torch.load("output/fp_analysis/models/best_model.pt")

# Use model for prediction on new data
predictions = model.predict(new_sequences)
```

### 3. Experiment Comparison
```python
import mlflow
import pandas as pd

# Compare multiple experiments
experiments = mlflow.search_runs(
    experiment_ids=["1", "2", "3"],
    order_by=["metrics.test_f1 DESC"]
)

# Analyze hyperparameter impact
print(experiments[['params.context_length', 'params.learning_rate', 'metrics.test_f1']])
```

### 4. Biological Discovery
- **Sequence Motifs**: Analyze discovered splice site motifs from IG analysis
- **Feature Patterns**: Examine which SpliceAI features are most predictive of errors
- **Error Characterization**: Understand systematic biases in splice site prediction
- **Model Improvement**: Use insights to enhance SpliceAI architecture or training

## Support and Documentation

For additional help:
- **Architecture Overview**: `README.md` - System design and components
- **Detailed Workflow**: `WORKFLOW_GUIDE.md` - Step-by-step process documentation
- **API Reference**: Package docstrings and type hints
- **Troubleshooting**: Log files in `logs/` directory for debugging
- **Experiment Tracking**: MLflow UI for experiment comparison and analysis
