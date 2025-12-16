# Training Parameter Summary Feature

## Overview
The error model training workflow now includes a comprehensive parameter summary that provides complete visibility into training configurations, model architecture, and dataset characteristics.

## Features

### 1. Model Architecture Details
- Base model name (e.g., DNABERT-2-117M)
- Hidden size, number of layers, attention heads
- Vocabulary size and maximum position embeddings
- Total and trainable parameters count
- Estimated memory requirements

### 2. Context Length Analysis
The summary includes detailed analysis of sequence lengths in your dataset:

- **Training Context**: The genomic context extracted for training (default: 200 nt, ±100 around splice site)
  - This is the actual sequence length used during model training
  - Extracted from the ~500 bp sequences in the artifact files
- **Max Tokenization**: The maximum token length for DNABERT-2 tokenization (default: 512 tokens)
  - Provides headroom for tokenization and special tokens
- **Extracted Sequences**: Statistical distribution of the extracted context sequences
  - Should typically show consistent 200 nt lengths after extraction
- **Variation Warning**: Alerts if extracted contexts deviate from expected length

Note: The raw artifact files (`analysis_sequences_*`) contain ~500 bp sequences, but the model trains on extracted 200 bp contexts centered on splice sites.

### 3. Training Hyperparameters
- Learning rate and optimizer settings
- Batch size and number of epochs
- Warmup steps and weight decay
- Gradient accumulation steps
- Mixed precision (FP16) training settings
- Maximum gradient norm for clipping

### 4. Optimization Details
- Optimizer type (AdamW)
- Loss function (CrossEntropy or custom)
- Learning rate scheduler configuration
- Evaluation and save strategies
- Early stopping configuration with patience and minimum delta

### 5. Data Configuration
- Number of training and validation samples
- Feature dimensions (input features × embedding dimension)
- Class distribution with percentages
- Primary sequence handling information

### 6. Hardware Configuration
- Device type (CPU/GPU)
- Multi-GPU settings if applicable
- Available memory and compute capabilities

## Usage

The parameter summary is automatically displayed at the beginning of training when using the `TransformerTrainer` class:

```python
from meta_spliceai.splice_engine.meta_models.error_model.modeling.transformer_trainer import TransformerTrainer

# Initialize trainer
trainer = TransformerTrainer(config)

# The summary is displayed when calling train()
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    num_train_samples=len(train_dataset),
    val_dataset=val_dataset  # Optional for class distribution analysis
)
```

## Implementation Details

The feature is implemented through the `_log_training_configuration()` method in `TransformerTrainer`:

```python
def _log_training_configuration(self, train_loader, num_train_samples, val_dataset=None):
    """
    Log comprehensive training configuration including:
    - Model architecture details
    - Context length analysis
    - Training hyperparameters
    - Optimization configuration
    - Data statistics
    - Hardware setup
    """
```

### Key Improvements
1. **Context Length Insights**: Helps identify if sequences are being truncated unexpectedly
2. **Memory Estimates**: Provides rough GPU memory requirements for planning
3. **Class Balance Info**: Shows distribution to identify potential imbalance issues
4. **Safe Attribute Access**: Uses `getattr()` with defaults for optional config parameters

## Benefits

1. **Reproducibility**: Complete record of experimental settings
2. **Debugging**: Easy identification of configuration issues
3. **Optimization**: Helps tune hyperparameters based on actual data characteristics
4. **Documentation**: Automatic logging of all training runs
5. **Transparency**: Clear visibility into model and data properties

## Example Output

```
================================================================================
TRAINING CONFIGURATION SUMMARY
================================================================================

MODEL ARCHITECTURE:
----------------------------------------
  Base Model:          zhihan1996/DNABERT-2-117M
  Hidden Size:         768
  Num Layers:          12
  Attention Heads:     12
  Vocab Size:          4096
  Max Position Emb:    512

CONTEXT LENGTH ANALYSIS:
----------------------------------------
  Training Context:    200 nt (±100 around splice site)
  Max Tokenization:    512 tokens
  Extracted Sequences:
    - Average:         200.0 nt
    - Median:          200.0 nt
    - Min:             200 nt
    - Max:             200 nt
    - Std Dev:         0.0 nt

TRAINING HYPERPARAMETERS:
----------------------------------------
  Learning Rate:       2e-05
  Batch Size:          16
  Num Epochs:          10
  Warmup Steps:        100
  Weight Decay:        0.01
  Gradient Accum:      1
  FP16 Training:       False
  Max Grad Norm:      1.0

[... continues with other sections ...]
================================================================================
```

## Related Features
- Feature summary logging (see `data_utils.py`)
- Architecture visualization
- Training metrics tracking
- Model checkpointing
