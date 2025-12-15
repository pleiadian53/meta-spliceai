# Transformer-Based Approach for Error Analysis

## Overview

The transformer-based approach in MetaSpliceAI uses state-of-the-art language models pre-trained on genomic sequences to analyze and classify splice site prediction errors. This approach excels at:

1. Capturing complex sequence patterns and motifs
2. Modeling long-range dependencies in genomic sequences
3. Learning hierarchical feature representations
4. Identifying subtle patterns that classical ML models might miss

## Key Components

### Base Models

MetaSpliceAI supports multiple pre-trained genomic language models:

- **DNABERT-2-117M**: A BERT-based model specifically pre-trained on genomic sequences
- **HyenaDNA**: Specialized for long-range genomic sequence modeling
- **Custom transformers**: Project-specific models with domain adaptations

### Error Sequence Model

The `error_sequence_model.py` module implements specialized transformer architectures for error classification. Key features include:

- Fine-tuning pre-trained models for splice site error detection
- Bucketing sequences by length for memory efficiency
- Mixed precision training for performance optimization
- Multi-GPU support for large models
- Attention weight visualization for interpretability

## Data Preparation

Preparing data for transformer-based error analysis involves:

1. **Sequence Extraction**: Obtaining genomic sequences around splice sites
2. **Tokenization**: Converting sequences to token IDs suitable for the transformer model
3. **Bucketing**: Grouping similar-length sequences to minimize padding and optimize memory usage
4. **Label Preparation**: Creating appropriate labels for error classification

## Training Process

```python
from meta_spliceai.splice_engine.error_sequence_model import (
    train_error_sequence_model,
    prepare_datasets_for_training
)

# Prepare datasets with appropriate tokenization
train_dataset, val_dataset = prepare_datasets_for_training(
    train_data=train_df,
    val_data=val_df,
    tokenizer="DNABERT-2",
    max_length=1024,
    bucket_boundaries=[512, 768, 1024]
)

# Train the model
model, training_args = train_error_sequence_model(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    base_model="DNABERT-2-117M",
    output_dir="/path/to/output",
    learning_rate=2e-5,
    batch_size=32,
    num_epochs=5,
    warmup_steps=500,
    weight_decay=0.01,
    use_fp16=True
)
```

## Attention Analysis

One of the key advantages of transformer models is their self-attention mechanism, which can be analyzed to understand:

1. **Relevant Regions**: Which parts of the sequence most influence the prediction
2. **Motif Detection**: Patterns in the sequence that contribute to errors
3. **Position Impact**: How the relative position of nucleotides affects predictions

The `visualize_attention_regions` function provides tools for visualizing these attention patterns:

```python
from meta_spliceai.splice_engine.error_sequence_model import visualize_attention_regions

# Visualize attention weights for a specific sequence
visualize_attention_regions(
    model=trained_model,
    tokenizer=tokenizer,
    sequence=test_sequence,
    prediction_type="FP",  # Focus on False Positives
    output_path="attention_visualization.pdf",
    highlight_motifs=True
)
```

## Region Importance Analysis

Beyond attention weights, the package includes methods for determining which regions of a sequence most contribute to errors:

1. **Attribution Methods**: Integrated Gradients and other attribution techniques
2. **Occlusion Analysis**: Systematically masking parts of the sequence
3. **Saliency Maps**: Gradient-based visualization of important nucleotides

These techniques help identify:
- Critical motifs that lead to prediction errors
- Long-range dependencies that classical models miss
- Sequence context patterns correlated with different error types

## Hardware Requirements

Transformer models typically require more computational resources than classical ML models:

- **GPU Memory**: 8GB+ VRAM recommended (16GB+ for larger models)
- **Multi-GPU Support**: NVIDIA with NCCL support
- **CPU**: 4+ cores for data loading
- **RAM**: 16GB+ recommended

The module includes several optimizations to improve training efficiency:
- Sequence bucketing to minimize padding
- Mixed precision (FP16) training
- Gradient accumulation
- Dynamic batch sizing

## Integration with Classical Approach

The transformer-based and classical ML approaches are complementary:

1. **Hybrid Models**: Combining transformer features with classical ML for interpretability
2. **Feature Verification**: Using transformers to verify patterns detected by classical models
3. **Ensemble Methods**: Creating ensemble predictions from both approaches
4. **Cross-Validation**: Comparing performance across different modeling paradigms

## Example Workflow

```python
from meta_spliceai.splice_engine.error_sequence_model import (
    prepare_datasets_for_training,
    train_error_sequence_model,
    evaluate_error_sequence_model,
    visualize_attention_regions
)

# Prepare data
train_ds, val_ds, test_ds = prepare_datasets_for_training(
    train_data, val_data, test_data,
    tokenizer="DNABERT-2",
    max_length=1024
)

# Train model
model, training_args = train_error_sequence_model(
    train_dataset=train_ds,
    val_dataset=val_ds,
    base_model="DNABERT-2-117M",
    output_dir="/path/to/output"
)

# Evaluate model
metrics = evaluate_error_sequence_model(
    model=model,
    test_dataset=test_ds,
    output_dir="/path/to/output"
)
print(f"Test accuracy: {metrics['accuracy']:.4f}")
print(f"Test F1 score: {metrics['f1']:.4f}")

# Analyze attention for specific errors
for sequence, label, pred_type in error_examples:
    visualize_attention_regions(
        model=model,
        tokenizer=tokenizer,
        sequence=sequence,
        prediction_type=pred_type,
        output_path=f"attention_{pred_type}.pdf"
    )
```
