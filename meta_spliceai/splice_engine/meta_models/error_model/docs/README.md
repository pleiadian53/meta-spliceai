# Error Model Workflow Documentation

This documentation covers the MetaSpliceAI error modeling workflow for analyzing false positives (FP) vs true positives (TP) and false negatives (FN) vs true negatives (TN) in splice site predictions.

## Documentation Index

- **[Usage Examples](./usage_examples.md)** - Comprehensive examples covering all workflow scenarios
- **[Workflow Guide](./workflow_guide.md)** - Detailed guide for running the complete workflow
- **[Subsampling Details](./subsampling_detailed.md)** - In-depth explanation of the subsampling logic

## Overview

The error model workflow uses a transformer-based approach (DNABERT-2) to learn patterns that distinguish between correct and incorrect splice site predictions. This helps understand systematic errors in splice site prediction models and can be used to improve prediction accuracy.

## Table of Contents

1. [Training Data Assembly](#training-data-assembly)
2. [Subsampling Logic](#subsampling-logic)
3. [Model Training](#model-training)
4. [Model Evaluation](#model-evaluation)
5. [Model Persistence](#model-persistence)
6. [Using Trained Models for Prediction](#using-trained-models-for-prediction)

## Training Data Assembly

### Data Sources

The workflow loads data from meta-model artifacts containing splice site predictions with error labels:
- **FP (False Positive)**: Predicted splice sites that are incorrect
- **TP (True Positive)**: Predicted splice sites that are correct
- **FN (False Negative)**: Missed real splice sites
- **TN (True Negative)**: Correctly rejected non-splice sites

### Basic Usage

```bash
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --data-dir /path/to/meta_models/artifacts \
    --output-dir /path/to/results \
    --error-type fp \
    --splice-type donor
```

### Data Loading Process

1. **Chunk Discovery**: Automatically finds all `analysis_sequences_*_chunk_*.tsv` files
2. **Format Support**: Handles TSV, CSV, and Parquet formats
3. **Schema Alignment**: Reconciles schema differences across chunks
4. **Type Casting**: Ensures consistent numeric types for genomic coordinates

## Subsampling Logic

The workflow implements sophisticated subsampling to maintain genomic structure integrity while managing dataset size.

### Core Principle: Gene-Level Sampling

**Critical**: The system samples at the gene level, not row level. This means either ALL rows for a gene are kept or ALL rows are discarded. This maintains the genomic context necessary for accurate splice site prediction.

### Four Filtering Modes

#### 1. No Constraints (Default)
```bash
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --sample-size 50000
```
- Randomly selects complete genes until approaching sample size
- Allows 20% overage to accommodate gene boundaries
- Maintains balance between error types when possible

#### 2. Chromosome Filtering
```bash
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --chromosomes 1 2 X \
    --sample-size 50000
```
- Filters to specified chromosomes first
- Then applies gene-level subsampling within those chromosomes
- Useful for chromosome-specific models or large chromosome focus

#### 3. Gene Filtering
```bash
# Via command line
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --genes BRCA1 TP53 EGFR

# Via file (for many genes)
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --genes-file gene_list.txt
```
- **No subsampling applied** when specific genes requested
- Supports both gene names and IDs (e.g., ENSG00000012048)
- Gene file format: one gene per line

#### 4. Combined Filtering
```bash
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --chromosomes 1 2 \
    --genes BRCA1 TP53 \
    --sample-size 50000
```
- Applies both chromosome and gene filters
- Useful for targeted analysis within specific regions

### Balanced Sampling

When prediction types are available, the system attempts balanced sampling:

```python
# Pseudocode of balanced sampling logic
if 'prediction_type' in data:
    fp_genes = select_random_genes(type='FP', target=sample_size/2)
    tp_genes = select_random_genes(type='TP', target=sample_size/2)
    selected_data = combine(fp_genes, tp_genes)
```

### Context Length Trimming

Optional sequence trimming for memory efficiency:
```bash
--context-length 200  # Trim sequences to 200nt centered on splice site
```

## Model Training

### Model Architecture

- **Base Model**: DNABERT-2-117M transformer
- **Task**: Binary classification (error vs correct)
- **Fine-tuning**: Task-specific layers added on top

### Training Configuration

```bash
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --model-name zhihan1996/DNABERT-2-117M \
    --batch-size 16 \
    --num-epochs 10 \
    --device cuda \
    --use-mixed-precision
```

### Key Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch-size` | 16 | Training batch size |
| `--num-epochs` | 10 | Number of training epochs |
| `--learning-rate` | 2e-5 | Initial learning rate |
| `--warmup-steps` | 500 | Linear warmup steps |
| `--weight-decay` | 0.01 | L2 regularization |
| `--use-mixed-precision` | True | FP16 training for efficiency |

### Performance Optimizations

- **Sequence Bucketing**: Groups sequences by length for efficient batching
- **Gradient Accumulation**: Simulates larger batches on limited GPU memory
- **Mixed Precision**: FP16 computation with FP32 master weights
- **Multi-GPU Support**: Distributed training via NCCL

## Model Evaluation

### Integrated Gradients Analysis

The workflow includes automated feature importance analysis:

```bash
python -m meta_spliceai.splice_engine.meta_models.error_model.run_error_model_workflow \
    --max-ig-samples 500  # Number of samples for IG analysis
```

Generates:
- Position-wise attribution scores
- Nucleotide importance heatmaps
- Aggregate importance patterns

### Evaluation Metrics

The system tracks:
- **Accuracy**: Overall prediction correctness
- **Precision/Recall**: Per-class performance
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Model discrimination ability
- **Confusion Matrix**: Detailed error analysis

### Visualization Outputs

Located in `{output_dir}/ig_analysis/`:
- `attribution_heatmap.png`: Sequence importance visualization
- `position_importance.png`: Aggregate position scores
- `metrics_report.json`: Detailed performance metrics

## Model Persistence

### Saved Artifacts

The workflow saves:

```
output_dir/
├── error_model/
│   ├── config.json           # Model configuration
│   ├── pytorch_model.bin     # Model weights
│   └── tokenizer/             # Tokenizer files
├── training_config.json      # Training hyperparameters
├── feature_columns.json      # Feature names used
└── consolidated_data.tsv     # Processed training data
```

### Loading a Saved Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "path/to/output_dir/error_model"
)
tokenizer = AutoTokenizer.from_pretrained(
    "path/to/output_dir/error_model"
)

# Set to evaluation mode
model.eval()
```

## Using Trained Models for Prediction

### Basic Prediction Script

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

def predict_sequences(model_path, sequences):
    """
    Predict error probability for new sequences.
    
    Args:
        model_path: Path to saved model directory
        sequences: List of DNA sequences or DataFrame with 'sequence' column
    
    Returns:
        DataFrame with predictions
    """
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Prepare sequences
    if isinstance(sequences, pd.DataFrame):
        seq_list = sequences['sequence'].tolist()
    else:
        seq_list = sequences
    
    # Tokenize
    inputs = tokenizer(
        seq_list,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(outputs.logits, dim=-1)
    
    # Format results
    results = pd.DataFrame({
        'sequence': seq_list,
        'predicted_class': predictions.cpu().numpy(),
        'error_probability': probs[:, 1].cpu().numpy(),  # Probability of being an error
        'correct_probability': probs[:, 0].cpu().numpy()
    })
    
    return results

# Example usage
sequences = [
    "ACGTACGTACGTACGT...GTACGT",  # Your sequences here
    "TGCATGCATGCATGCA...CATGCA"
]

predictions = predict_sequences(
    model_path="path/to/output_dir/error_model",
    sequences=sequences
)

print(predictions)
```

### Batch Prediction for Large Datasets

```python
def predict_large_dataset(model_path, input_file, output_file, batch_size=32):
    """
    Predict on large datasets with batching.
    
    Args:
        model_path: Path to saved model
        input_file: TSV/CSV with sequences
        output_file: Output predictions file
        batch_size: Prediction batch size
    """
    import numpy as np
    from tqdm import tqdm
    
    # Load model once
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load data
    df = pd.read_csv(input_file, sep='\t')
    
    # Process in batches
    all_predictions = []
    
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        sequences = batch['sequence'].tolist()
        
        # Tokenize and predict
        inputs = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Store results
        batch_results = {
            'predicted_class': predictions.cpu().numpy(),
            'error_probability': probs[:, 1].cpu().numpy()
        }
        all_predictions.append(batch_results)
    
    # Combine results
    pred_classes = np.concatenate([p['predicted_class'] for p in all_predictions])
    error_probs = np.concatenate([p['error_probability'] for p in all_predictions])
    
    df['predicted_class'] = pred_classes
    df['error_probability'] = error_probs
    
    # Save
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Predictions saved to {output_file}")
```

### Integration with Splice Site Analysis

```python
def filter_splice_predictions(splice_predictions_df, model_path, threshold=0.5):
    """
    Filter splice site predictions using error model.
    
    Args:
        splice_predictions_df: DataFrame with splice site predictions and sequences
        model_path: Path to trained error model
        threshold: Error probability threshold for filtering
    
    Returns:
        Filtered DataFrame with high-confidence predictions
    """
    # Get error probabilities
    predictions = predict_sequences(
        model_path=model_path,
        sequences=splice_predictions_df
    )
    
    # Add to original data
    splice_predictions_df['error_probability'] = predictions['error_probability']
    
    # Filter high-confidence predictions
    high_confidence = splice_predictions_df[
        splice_predictions_df['error_probability'] < threshold
    ]
    
    print(f"Filtered {len(splice_predictions_df)} -> {len(high_confidence)} predictions")
    print(f"Removed {len(splice_predictions_df) - len(high_confidence)} likely errors")
    
    return high_confidence
```

## Advanced Features

### Experiment Tracking

Enable MLflow tracking:
```bash
--enable-mlflow \
--mlflow-tracking-uri http://localhost:5000 \
--mlflow-experiment-name error_model_experiments
```

Enable Weights & Biases:
```bash
--enable-wandb \
--wandb-project splice-error-models
```

### Custom Feature Selection

Use specific feature columns:
```bash
--features sequence score conservation_score
```

### Skip Training for Analysis Only

```bash
--skip-training  # Use existing model
--skip-ig        # Skip IG analysis
```

## Troubleshooting

### Out of Memory Errors

1. Reduce batch size: `--batch-size 8`
2. Enable gradient accumulation
3. Reduce sequence length: `--context-length 150`
4. Use smaller sample size: `--sample-size 10000`

### Slow Training

1. Enable mixed precision: `--use-mixed-precision`
2. Use GPU: `--device cuda`
3. Reduce IG samples: `--max-ig-samples 100`

### Data Loading Issues

1. Check chunk file formats
2. Verify gene_name column exists for subsampling
3. Ensure chromosome values are consistent
4. Check for schema mismatches in logs

## References

- [DNABERT-2 Paper](https://arxiv.org/abs/2306.15006)
- [Integrated Gradients](https://arxiv.org/abs/1703.01365)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
