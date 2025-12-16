# Error Model Workflow Guide

Comprehensive guide for running the MetaSpliceAI error model workflow, from data preparation through model training and interpretability analysis.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Command Line Arguments](#command-line-arguments)
3. [Input Data Requirements](#input-data-requirements)
4. [Workflow Steps](#workflow-steps)
5. [Output Structure](#output-structure)
6. [Integration with Existing Workflows](#integration-with-existing-workflows)
7. [Performance Optimization](#performance-optimization)
8. [Advanced Usage Patterns](#advanced-usage-patterns)

## Quick Start

```bash
# Activate environment
mamba activate surveyor

# Run FP vs TP analysis
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir results/fp_analysis \
    --error-type fp \
    --num-epochs 10

# Run FN vs TN analysis
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir results/fn_analysis \
    --error-type fn \
    --num-epochs 10
```

## Command Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--data-dir` | Directory containing meta-model analysis data | `data/ensembl/spliceai_eval/meta_models` |
| `--output-dir` | Directory for workflow outputs | `results/error_analysis` |
| `--error-type` | Type of error analysis | `fp` (FP vs TP) or `fn` (FN vs TN) |

### Data Processing Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--sample-size` | Number of samples per class | None (use all) | `50000` |
| `--chromosomes` | Chromosomes to include | All | `--chromosomes 1 2 3` |
| `--genes` | Specific genes to analyze | None | `--genes BRCA1 TP53` |
| `--genes-file` | File with gene list | None | `--genes-file cancer_genes.txt` |
| `--balanced-sampling` | Ensure equal class distribution | False | `--balanced-sampling` |
| `--skip-consolidation` | Skip data consolidation step | False | `--skip-consolidation` |

### Model Configuration Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--model-name` | HuggingFace model name | `zhihan1996/DNABERT-2-117M` | `--model-name bert-base-uncased` |
| `--context-length` | Sequence context length | 200 | `--context-length 300` |
| `--splice-type` | Splice site type filter | `any` | `--splice-type donor` |
| `--batch-size` | Training batch size | 16 | `--batch-size 32` |
| `--num-epochs` | Training epochs | 5 | `--num-epochs 10` |
| `--learning-rate` | Learning rate | 2e-5 | `--learning-rate 1e-5` |
| `--warmup-steps` | Warmup steps | 0 | `--warmup-steps 100` |

### Feature Configuration Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--include-base-scores` | Include raw SpliceAI scores | True |
| `--include-context-features` | Include context features | True |
| `--include-donor-features` | Include donor-specific features | True |
| `--include-acceptor-features` | Include acceptor-specific features | True |
| `--include-derived-features` | Include statistical features | True |
| `--include-genomic-features` | Include gene-level features | False |

### Analysis Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--max-ig-samples` | Max samples for IG analysis | 500 | `--max-ig-samples 1000` |
| `--ig-steps` | Integration steps for IG | 50 | `--ig-steps 100` |
| `--skip-training` | Skip training, only analyze | False | `--skip-training` |
| `--skip-ig-analysis` | Skip IG analysis | False | `--skip-ig-analysis` |

### Tracking Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--enable-mlflow` | Enable MLflow tracking | False | `--enable-mlflow` |
| `--mlflow-tracking-uri` | MLflow server URI | None | `--mlflow-tracking-uri http://localhost:5000` |
| `--mlflow-experiment-name` | MLflow experiment name | None | `--mlflow-experiment-name splice_errors` |
| `--enable-wandb` | Enable Weights & Biases | False | `--enable-wandb` |
| `--wandb-project` | W&B project name | None | `--wandb-project meta-spliceai` |
| `--enable-tensorboard` | Enable TensorBoard | False | `--enable-tensorboard` |

### System Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--device` | Compute device | `cuda` if available | `--device cpu` |
| `--use-mixed-precision` | Use mixed precision | False | `--use-mixed-precision` |
| `--num-workers` | Data loading workers | 4 | `--num-workers 8` |
| `--verbose` | Verbose output | False | `--verbose` |
| `--random-seed` | Random seed | 42 | `--random-seed 123` |

## Input Data Requirements

### Required Data Files

The workflow expects these files in the `--data-dir`:

```
data/ensembl/spliceai_eval/meta_models/
├── analysis_sequences_*.tsv           # Chunked sequence data
├── splice_positions_enhanced_*.tsv    # Enhanced splice positions
└── chromosome_mappings.tsv (optional) # Chromosome mappings
```

### Data Format

**Analysis Sequences TSV:**
```
gene_name  chromosome  position  sequence  label  prediction_type  scores...
BRCA1      17         41234420  ATCG...   FP     donor           0.95,0.02,...
```

**Splice Positions Enhanced TSV:**
```
gene_name  chromosome  position  donor_score  acceptor_score  features...
BRCA1      17         41234420  0.95        0.02            ...
```

### Gene List File Format

For `--genes-file`:
```
BRCA1
BRCA2
TP53
KRAS
```

## Workflow Steps

### Step 1: Data Loading and Consolidation

The workflow begins by consolidating chunked analysis data:

```python
# Automatic consolidation
consolidator = AnalysisDataConsolidator(verbose=True)
consolidated_data = consolidator.consolidate_and_filter_analysis_sequences(
    data_dir=Path(args.data_dir),
    output_file=output_dir / "consolidated_data.tsv",
    error_type=error_type_mapping[args.error_type],
    max_rows_per_type=args.sample_size,
    chromosomes=args.chromosomes,
    genes=combined_genes
)
```

**Processing:**
- Merges chunked TSV files
- Filters by error type (FP/TP or FN/TN)
- Applies chromosome/gene filters
- Performs balanced sampling if requested

### Step 2: Dataset Preparation

Prepares training, validation, and test datasets:

```python
# Dataset creation with contextual sequences
preparer = ErrorDatasetPreparer(config)
dataset_info = preparer.prepare_dataset_from_dataframe(
    df=consolidated_df,
    error_label=error_label,
    correct_label=correct_label
)
```

**Processing:**
- Extracts contextual sequences (±context_length/2 nucleotides)
- Adds tabular features based on configuration
- Creates 70/15/15 train/val/test split
- Applies tokenization and encoding

### Step 3: Model Training

Trains DNABERT-2 transformer model:

```python
# Model training
trainer = ErrorModelTrainer(config)
model, training_history = trainer.train_model(
    train_dataset=dataset_info['datasets']['train'],
    val_dataset=dataset_info['datasets']['val'],
    num_epochs=args.num_epochs,
    batch_size=args.batch_size
)
```

**Training Features:**
- Automatic mixed precision if enabled
- Early stopping with patience
- Learning rate scheduling
- Gradient accumulation for large batches
- Checkpoint saving

### Step 4: Model Evaluation

Evaluates model performance:

```python
# Comprehensive evaluation
evaluator = ModelEvaluator(config)
metrics = evaluator.evaluate_model(
    model=model,
    test_dataset=dataset_info['datasets']['test'],
    output_dir=output_dir / "evaluation"
)
```

**Metrics Generated:**
- Accuracy, Precision, Recall, F1
- AUC-ROC, AUC-PR
- Confusion matrix
- Per-class performance
- Calibration metrics

### Step 5: Integrated Gradients Analysis

Performs interpretability analysis:

```python
# IG analysis for interpretability
ig_analyzer = IntegratedGradientsAnalyzer(config)
ig_results = ig_analyzer.analyze_model_predictions(
    model=model,
    dataset=test_dataset,
    num_samples=args.max_ig_samples,
    steps=args.ig_steps
)
```

**Analysis Components:**
- Token-level attributions
- Sequence motif discovery
- Feature importance ranking
- Position-specific patterns
- Error-type specific patterns

### Step 6: Visualization Generation

Creates comprehensive visualizations:

```python
# Generate all visualizations
visualizer = ErrorModelVisualizer(config)
visualizer.create_all_visualizations(
    ig_results=ig_results,
    model_metrics=metrics,
    output_dir=output_dir / "visualizations"
)
```

**Visualizations Created:**
- Token frequency comparisons
- Attribution distributions
- Positional analysis heatmaps
- ROC and PR curves
- Confusion matrices
- Feature importance plots

### Step 7: Report Generation

Generates comprehensive analysis report:

```python
# Create final report
reporter = WorkflowReporter(config)
reporter.generate_comprehensive_report(
    workflow_results=all_results,
    output_file=output_dir / "analysis_report.md"
)
```

## Output Structure

```
output_dir/
├── consolidated_data.tsv               # Processed input data
├── error_model/
│   ├── config.json                    # Model configuration
│   ├── pytorch_model.bin              # Trained model weights
│   └── tokenizer/                     # Tokenizer files
├── training_config.json               # Training hyperparameters
├── feature_columns.json               # Feature definitions
├── logs/
│   ├── workflow.log                   # Complete execution log
│   ├── training.log                   # Training progress
│   └── evaluation.log                 # Evaluation details
├── ig_analysis/
│   ├── attributions.parquet          # Raw IG attributions
│   ├── analysis_results.json         # Processed analysis
│   ├── token_frequencies.csv         # Token importance
│   ├── motif_analysis.json           # Discovered motifs
│   └── pattern_summary.json          # Pattern insights
├── visualizations/
│   ├── token_frequency_comparison.png
│   ├── attribution_distribution.png
│   ├── top_tokens_analysis.png
│   ├── positional_analysis.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── evaluation/
│   ├── test_predictions.csv          # Model predictions
│   ├── performance_metrics.json      # Detailed metrics
│   └── calibration_plot.png          # Calibration analysis
├── mlflow/                            # MLflow artifacts (if enabled)
├── wandb/                             # W&B artifacts (if enabled)
└── analysis_report.md                 # Comprehensive report
```

## Integration with Existing Workflows

### Using Trained Models in Production

```python
from meta_spliceai.splice_engine.meta_models.error_model import ErrorModelPredictor

# Load trained model
predictor = ErrorModelPredictor.from_pretrained("output_dir/error_model")

# Make predictions
sequences = ["ATCGATCG...", "GCTAGCTA..."]
predictions = predictor.predict(sequences)
```

### Incorporating into MetaSpliceAI Pipeline

```python
from meta_spliceai import MetaSpliceAIPipeline

# Add error model as post-processor
pipeline = MetaSpliceAIPipeline()
pipeline.add_error_correction(
    model_path="output_dir/error_model",
    threshold=0.8
)
```

### Batch Processing Integration

```python
# Process large datasets
from meta_spliceai.splice_engine.meta_models.error_model import BatchProcessor

processor = BatchProcessor(
    model_path="output_dir/error_model",
    batch_size=1000
)

results = processor.process_file(
    input_file="new_predictions.tsv",
    output_file="corrected_predictions.tsv"
)
```

## Performance Optimization

### GPU Optimization

```bash
# Single GPU optimization
export CUDA_VISIBLE_DEVICES=0
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --batch-size 32 \
    --use-mixed-precision \
    --num-workers 8
```

```bash
# Multi-GPU training (data parallel)
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --batch-size 128 \
    --use-mixed-precision \
    --distributed
```

### Memory Optimization

For limited memory systems:

```bash
# Reduce memory usage
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --batch-size 4 \
    --context-length 150 \
    --gradient-accumulation-steps 4 \
    --max-ig-samples 100 \
    --clear-cache-frequently
```

### CPU Optimization

For CPU-only systems:

```bash
# Optimize for CPU
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --device cpu \
    --batch-size 8 \
    --num-workers 16 \
    --use-cpu-optimizations
```

### Data Loading Optimization

```python
# Custom data loading configuration
config = ErrorModelConfig(
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
    num_workers=8
)
```

## Advanced Usage Patterns

### Hyperparameter Tuning

```python
from meta_spliceai.splice_engine.meta_models.error_model import HyperparameterTuner

# Grid search
tuner = HyperparameterTuner(
    data_dir="data/ensembl/spliceai_eval/meta_models",
    output_dir="tuning_results"
)

best_params = tuner.grid_search({
    'learning_rate': [1e-5, 2e-5, 5e-5],
    'batch_size': [8, 16, 32],
    'context_length': [150, 200, 300]
})
```

### Cross-Validation

```python
from meta_spliceai.splice_engine.meta_models.error_model import CrossValidator

# 5-fold cross-validation
cv = CrossValidator(config)
cv_results = cv.run_cv(
    data=consolidated_df,
    n_folds=5,
    stratified=True
)

print(f"Mean F1: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
```

### Ensemble Models

```python
from meta_spliceai.splice_engine.meta_models.error_model import EnsemblePredictor

# Create ensemble from multiple models
models = [
    "output/donor_model",
    "output/acceptor_model",
    "output/combined_model"
]

ensemble = EnsemblePredictor(models, voting='soft')
predictions = ensemble.predict(test_sequences)
```

### Transfer Learning

```python
# Fine-tune on specific dataset
from meta_spliceai.splice_engine.meta_models.error_model import TransferLearner

transfer_learner = TransferLearner(
    base_model="output_dir/error_model",
    freeze_layers=10
)

fine_tuned_model = transfer_learner.fine_tune(
    new_data="specific_gene_data.tsv",
    num_epochs=5,
    learning_rate=1e-6
)
```

### Custom Feature Engineering

```python
from meta_spliceai.splice_engine.meta_models.error_model import CustomFeatureExtractor

# Add custom features
class MyFeatureExtractor(CustomFeatureExtractor):
    def extract(self, df):
        # Add GC content
        df['gc_content'] = df['sequence'].apply(
            lambda s: (s.count('G') + s.count('C')) / len(s)
        )
        # Add custom score
        df['custom_score'] = df['donor_score'] * df['acceptor_score']
        return df

# Use in workflow
config.custom_feature_extractor = MyFeatureExtractor()
```

### Model Interpretation Extensions

```python
# Advanced interpretability
from meta_spliceai.splice_engine.meta_models.error_model import AdvancedInterpreter

interpreter = AdvancedInterpreter(model)

# SHAP analysis
shap_values = interpreter.compute_shap_values(test_data)

# Attention visualization
attention_maps = interpreter.visualize_attention(sequences)

# Counterfactual analysis
counterfactuals = interpreter.generate_counterfactuals(
    original_sequences,
    target_class='TP'
)
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size, context length, or enable gradient accumulation |
| Slow data loading | Increase num_workers, enable persistent_workers |
| Poor model performance | Increase training epochs, tune learning rate, check data balance |
| Missing dependencies | Run `pip install -e .` in meta_spliceai directory |
| MLflow connection error | Check MLflow server is running, verify tracking URI |
| IG analysis timeout | Reduce max_ig_samples or ig_steps |
| Imbalanced predictions | Enable balanced_sampling or adjust class weights |

### Debug Mode

Enable detailed debugging:

```bash
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --verbose \
    --debug \
    --log-level DEBUG \
    --save-intermediate-results
```

### Performance Profiling

```bash
# Profile execution
python -m cProfile -o profile.stats \
    meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir data/ensembl/spliceai_eval/meta_models \
    --output-dir profiling_results \
    --error-type fp

# Analyze profile
python -m pstats profile.stats
```

## Support

For issues or questions:
- Check the [Error Model Documentation](./README.md)
- Review [Usage Examples](./usage_examples.md)
- Examine [Subsampling Details](./subsampling_detailed.md)
- Open an issue on the MetaSpliceAI repository
