# Error Model Workflow Guide

This guide explains how to use the complete error modeling workflow for splice site prediction analysis.

## Overview

The `run_error_model_workflow.py` script provides a streamlined interface to the entire error modeling pipeline:

1. **Data Loading**: Loads position-centric data from meta-model artifacts
2. **Dataset Preparation**: Creates binary classification datasets (FP vs TP, FN vs TP)
3. **Model Training**: Fine-tunes transformer models for error classification
4. **IG Analysis**: Performs Integrated Gradients analysis for interpretability
5. **Visualization**: Creates comprehensive plots and visualizations
6. **Reporting**: Generates detailed analysis reports

## Quick Start

### Basic Usage

```bash
# Run FP vs TP analysis
python run_error_model_workflow.py \
    --data_dir data/ensembl/spliceai_eval/meta_models \
    --output_dir output/fp_analysis \
    --error_type FP_vs_TP

# Run FN vs TP analysis
python run_error_model_workflow.py \
    --data_dir data/ensembl/spliceai_eval/meta_models \
    --output_dir output/fn_analysis \
    --error_type FN_vs_TP
```

### Advanced Usage

```bash
# Custom training parameters
python run_error_model_workflow.py \
    --data_dir data/ensembl/spliceai_eval/meta_models \
    --output_dir output/custom_analysis \
    --error_type FP_vs_TP \
    --model_name zhihan1996/DNABERT-2-117M \
    --context_length 300 \
    --batch_size 32 \
    --num_epochs 15 \
    --max_ig_samples 1000

# Skip training and use existing model
python run_error_model_workflow.py \
    --data_dir data/ensembl/spliceai_eval/meta_models \
    --output_dir output/ig_only_analysis \
    --error_type FP_vs_TP \
    --skip_training

# Donor-specific analysis
python run_error_model_workflow.py \
    --data_dir data/ensembl/spliceai_eval/meta_models \
    --output_dir output/donor_analysis \
    --error_type FP_vs_TP \
    --splice_type donor
```

## Command Line Arguments

### Required Arguments

- `--data_dir`: Directory containing meta-model artifacts (`analysis_sequences_*`, `splice_positions_enhanced_*`)
- `--output_dir`: Output directory for all results and reports

### Data Selection Arguments

- `--error_type`: Type of error analysis (`FP_vs_TP`, `FN_vs_TP`)
- `--splice_type`: Splice site type to analyze (`donor`, `acceptor`, `any`)
- `--facet`: Data facet to use (default: `simple`)

### Model Configuration

- `--model_name`: Pre-trained transformer model (default: `zhihan1996/DNABERT-2-117M`)
- `--context_length`: Total sequence context length (default: 200)
- `--batch_size`: Training batch size (default: 16)
- `--num_epochs`: Number of training epochs (default: 10)

### IG Analysis Parameters

- `--max_ig_samples`: Maximum samples for IG analysis (default: 500)

### Workflow Control

- `--skip_training`: Skip training and load existing model
- `--skip_ig`: Skip IG analysis entirely
- `--log_level`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

## Input Data Requirements

The workflow expects meta-model artifacts in the specified data directory:

### Required Files

```
data_dir/
├── analysis_sequences_simple*.parquet    # Genomic sequences with context
├── splice_positions_enhanced_*.parquet   # Position annotations with features
└── [other meta-model artifacts]
```

### Data Structure

**analysis_sequences_***:
- `gene_id`: Gene identifier
- `transcript_id`: Transcript identifier  
- `chrom`: Chromosome
- `position`: Genomic position
- `sequence`: DNA sequence with context
- `pred_type`: Prediction type (FP, FN, TP, TN)
- `splice_type`: Splice site type (donor, acceptor)
- `strand`: Genomic strand

**splice_positions_enhanced_***:
- Position-level features and annotations
- Splice site scores and probabilities
- Additional genomic features

## Output Structure

The workflow creates a comprehensive output directory:

```
output_dir/
├── models/
│   ├── best_model.pt                    # Trained model checkpoint
│   ├── training_args.json               # Training configuration
│   └── trainer_state.json               # Training state
├── logs/
│   ├── workflow.log                     # Complete workflow log
│   └── training_logs/                   # Detailed training logs
├── data/
│   ├── train_dataset.parquet            # Prepared training data
│   ├── val_dataset.parquet              # Validation data
│   └── test_dataset.parquet             # Test data
├── ig_analysis/
│   ├── attributions.parquet             # Raw IG attributions
│   ├── analysis_results.json            # Pattern analysis results
│   └── token_frequencies.csv            # Token frequency analysis
├── visualizations/
│   ├── token_frequency_comparison.png   # Token frequency plots
│   ├── attribution_distribution.png     # Attribution statistics
│   ├── top_tokens_analysis.png          # Important tokens analysis
│   └── positional_analysis.png          # Positional patterns
└── analysis_report.md                   # Comprehensive report
```

## Workflow Steps Explained

### 1. Data Loading

```python
# Uses ModelEvaluationFileHandler for compatibility
analysis_df = load_data(data_dir, facet, error_label, correct_label, splice_type)
```

- Loads `analysis_sequences_*` files using existing handlers
- Filters by error type and splice type
- Validates data structure and completeness

### 2. Dataset Preparation

```python
# Uses ErrorDatasetPreparer from the error model package
dataset_info = prepare_datasets(analysis_df, config, error_label, correct_label)
```

- Creates binary classification labels (error vs correct)
- Extracts sequence context and additional features
- Splits into train/validation/test sets
- Handles class imbalance and data quality

### 3. Model Training

```python
# Uses TransformerTrainer with automatic configuration
trainer, model_path = train_model(dataset_info, config, output_dir)
```

- Fine-tunes DNABERT-like models for binary classification
- Supports multi-modal input (sequence + features)
- Implements early stopping and best model selection
- Evaluates on test set and saves results

### 4. IG Analysis

```python
# Uses IGAnalyzer for comprehensive attribution analysis
attributions, analysis_results = run_ig_analysis(trainer, dataset_info, config, ig_config)
```

- Computes token-level attributions using Integrated Gradients
- Analyzes error patterns and token importance
- Compares attribution distributions between error classes
- Identifies top important tokens and motifs

### 5. Visualization

```python
# Uses visualization package for comprehensive plots
saved_plots = create_visualizations(attributions, analysis_results, output_dir)
```

- Token frequency comparison charts
- Attribution distribution analysis
- Top tokens importance visualization
- Positional pattern analysis

### 6. Report Generation

```python
# Creates comprehensive markdown report
report_path = generate_report(analysis_results, saved_plots, config, output_dir)
```

- Summarizes key findings and statistics
- Documents important tokens and patterns
- Lists all generated files and visualizations
- Provides interpretation guidelines

## Integration with Existing Workflows

### Meta-Model Training Integration

The workflow seamlessly integrates with existing meta-model training:

```bash
# 1. Run meta-model training to generate artifacts
python run_meta_model_training.py --output_dir data/ensembl/spliceai_eval/meta_models

# 2. Run error model analysis on the artifacts
python run_error_model_workflow.py \
    --data_dir data/ensembl/spliceai_eval/meta_models \
    --output_dir output/error_analysis
```

### Batch Processing Multiple Error Types

```bash
# Process both error types
for error_type in FP_vs_TP FN_vs_TP; do
    python run_error_model_workflow.py \
        --data_dir data/ensembl/spliceai_eval/meta_models \
        --output_dir output/${error_type}_analysis \
        --error_type $error_type
done
```

### Splice Type Specific Analysis

```bash
# Analyze donor and acceptor sites separately
for splice_type in donor acceptor; do
    python run_error_model_workflow.py \
        --data_dir data/ensembl/spliceai_eval/meta_models \
        --output_dir output/${splice_type}_fp_analysis \
        --error_type FP_vs_TP \
        --splice_type $splice_type
done
```

## Performance Considerations

### Memory Usage

- **Large datasets**: Use smaller batch sizes (`--batch_size 8`)
- **IG analysis**: Limit samples (`--max_ig_samples 200`)
- **Context length**: Reduce if memory constrained (`--context_length 150`)

### Computational Requirements

- **GPU recommended**: For transformer training and IG computation
- **Training time**: ~1-3 hours depending on data size and epochs
- **IG analysis time**: ~30-60 minutes for 500 samples

### Scalability

```bash
# For large-scale analysis
python run_error_model_workflow.py \
    --data_dir data/ensembl/spliceai_eval/meta_models \
    --output_dir output/large_scale_analysis \
    --error_type FP_vs_TP \
    --batch_size 8 \
    --max_ig_samples 200 \
    --context_length 150
```

## Troubleshooting

### Common Issues

1. **Data not found**: Ensure meta-model artifacts exist in `data_dir`
2. **Memory errors**: Reduce batch size and IG samples
3. **CUDA errors**: Check GPU availability and memory
4. **Import errors**: Verify project root is in Python path

### Debug Mode

```bash
# Enable detailed logging
python run_error_model_workflow.py \
    --data_dir data/ensembl/spliceai_eval/meta_models \
    --output_dir output/debug_analysis \
    --error_type FP_vs_TP \
    --log_level DEBUG
```

### Validation Steps

```bash
# Test with minimal configuration
python run_error_model_workflow.py \
    --data_dir data/ensembl/spliceai_eval/meta_models \
    --output_dir output/test_run \
    --error_type FP_vs_TP \
    --num_epochs 1 \
    --max_ig_samples 50
```

## Advanced Usage Patterns

### Custom Model Fine-tuning

```bash
# Use different pre-trained model
python run_error_model_workflow.py \
    --data_dir data/ensembl/spliceai_eval/meta_models \
    --output_dir output/custom_model_analysis \
    --error_type FP_vs_TP \
    --model_name microsoft/DialoGPT-medium \
    --context_length 400
```

### Iterative Analysis

```bash
# 1. Initial training
python run_error_model_workflow.py \
    --data_dir data/ensembl/spliceai_eval/meta_models \
    --output_dir output/initial_analysis \
    --error_type FP_vs_TP \
    --num_epochs 5

# 2. Extended IG analysis on trained model
python run_error_model_workflow.py \
    --data_dir data/ensembl/spliceai_eval/meta_models \
    --output_dir output/extended_ig_analysis \
    --error_type FP_vs_TP \
    --skip_training \
    --max_ig_samples 2000
```

### Comparative Analysis

```bash
# Compare different context lengths
for context in 150 200 300; do
    python run_error_model_workflow.py \
        --data_dir data/ensembl/spliceai_eval/meta_models \
        --output_dir output/context_${context}_analysis \
        --error_type FP_vs_TP \
        --context_length $context
done
```

## Next Steps

After running the workflow:

1. **Review the analysis report** (`analysis_report.md`)
2. **Examine visualizations** in the `visualizations/` directory
3. **Analyze top important tokens** for biological significance
4. **Compare results** across different error types or splice types
5. **Integrate findings** into model improvement strategies
6. **Validate discoveries** with domain knowledge and literature

## Support

For issues or questions:

1. Check the workflow log (`logs/workflow.log`)
2. Review the comprehensive documentation in `README.md`
3. Examine example usage in `examples/`
4. Consult the error model package documentation
