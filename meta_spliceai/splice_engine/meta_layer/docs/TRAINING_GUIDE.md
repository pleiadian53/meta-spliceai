# Meta-Layer Training Guide

**Last Updated**: December 2025  
**Status**: Design Document

---

## Overview

This guide provides step-by-step instructions for training the multimodal meta-layer.

---

## Prerequisites

### 1. Environment Setup

```bash
# Activate the environment
mamba activate metaspliceai

# Verify PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check device
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

### 2. Base Layer Artifacts

Ensure you have run the base layer pass:

```bash
# Check artifacts exist
ls data/mane/GRCh38/openspliceai_eval/meta_models/analysis_sequences_*.tsv

# Count positions
wc -l data/mane/GRCh38/openspliceai_eval/meta_models/analysis_sequences_*.tsv
```

### 3. SpliceVarDB (Optional)

```bash
# Check SpliceVarDB data
ls meta_spliceai/splice_engine/case_studies/workflows/splicevardb/splicevardb.download.tsv
```

---

## Quick Start

### Minimal Training (M1 Mac)

```python
from meta_spliceai.splice_engine.meta_layer import MetaLayerConfig
from meta_spliceai.splice_engine.meta_layer.workflows import train_meta_model

# Configure for M1 Mac
config = MetaLayerConfig(
    base_model='openspliceai',
    sequence_encoder='cnn',  # Lightweight, no pretrained LM
    max_samples=50_000,      # Subset for quick training
    batch_size=32,
    max_epochs=10,
    device='mps'  # Apple Silicon
)

# Train
results = train_meta_model(config, output_dir='models/meta_layer_lite')

print(f"Final PR-AUC: {results['metrics']['pr_auc_splice']:.4f}")
```

### Full Training (GPU)

```python
from meta_spliceai.splice_engine.meta_layer import MetaLayerConfig
from meta_spliceai.splice_engine.meta_layer.workflows import train_meta_model

# Configure for GPU
config = MetaLayerConfig(
    base_model='openspliceai',
    sequence_encoder='hyenadna',  # Full DNA LM
    variant_source='splicevardb',
    max_samples=None,  # Use all data
    batch_size=128,
    max_epochs=100,
    patience=10,
    mixed_precision=True,
    device='cuda'
)

# Train
results = train_meta_model(config, output_dir='models/meta_layer_v1')
```

---

## Training Pipeline

### Step 1: Prepare Training Data

```python
from meta_spliceai.splice_engine.meta_layer.core import MetaLayerConfig, ArtifactLoader
from meta_spliceai.splice_engine.meta_layer.data import prepare_training_dataset

# Configure
config = MetaLayerConfig(base_model='openspliceai')

# Load and prepare
loader = ArtifactLoader(config)
stats = loader.get_statistics()
print(f"Available positions: {stats['total_positions']:,}")

# Prepare dataset
dataset = prepare_training_dataset(
    config=config,
    variant_source='splicevardb',
    output_dir='data/meta_training/openspliceai_v1'
)

print(f"Training samples: {len(dataset):,}")
```

### Step 2: Create Data Loaders

```python
from meta_spliceai.splice_engine.meta_layer.data import MetaLayerDataset, create_dataloaders
from torch.utils.data import DataLoader

# Create dataset
dataset = MetaLayerDataset(
    data_path='data/meta_training/openspliceai_v1/training_data.parquet',
    sequence_col='sequence',
    feature_cols=config.get_feature_columns(),
    label_col='label',
    weight_col='sample_weight'
)

# Split and create loaders
train_loader, val_loader, test_loader = create_dataloaders(
    dataset=dataset,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    seed=config.seed
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
```

### Step 3: Initialize Model

```python
from meta_spliceai.splice_engine.meta_layer.models import MetaSpliceModel

# Create model
model = MetaSpliceModel(
    sequence_encoder=config.sequence_encoder,
    num_score_features=50,
    hidden_dim=config.hidden_dim,
    num_classes=3,
    dropout=config.dropout
)

# Move to device
device = config.get_device()
model = model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

### Step 4: Train

```python
from meta_spliceai.splice_engine.meta_layer.training import Trainer

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    output_dir='models/meta_layer_v1'
)

# Train
history = trainer.train()

# Plot training curves
trainer.plot_history()
```

### Step 5: Evaluate

```python
from meta_spliceai.splice_engine.meta_layer.training import MetaLayerEvaluator

# Create evaluator
evaluator = MetaLayerEvaluator(
    top_k_values=[10, 50, 100, 500]
)

# Evaluate on test set
metrics = evaluator.evaluate_dataloader(model, test_loader, device)

# Print results
print("\n📊 Test Set Results:")
print(f"  PR-AUC (splice): {metrics['pr_auc_splice']:.4f}")
print(f"  AP (splice):     {metrics['ap_splice']:.4f}")
print(f"  Top-100 Acc (donor): {metrics['top100_acc_donor']:.4f}")
print(f"  Top-100 Acc (acceptor): {metrics['top100_acc_acceptor']:.4f}")
print(f"  ECE: {metrics['ece']:.4f}")
```

---

## Configuration Options

### Lightweight (M1 Mac)

```yaml
# configs/lightweight.yaml
base_model: openspliceai
sequence_encoder: cnn

sequence_encoder_config:
  output_dim: 128
  kernel_sizes: [3, 5, 7]
  num_filters: 32

hidden_dim: 128
batch_size: 32
max_samples: 50000
max_epochs: 20
mixed_precision: false
device: mps
```

### HyenaDNA (GPU)

```yaml
# configs/hyenadna.yaml
base_model: openspliceai
sequence_encoder: hyenadna

sequence_encoder_config:
  model_size: small
  output_dim: 256
  pretrained: true
  freeze: true

hidden_dim: 256
batch_size: 128
max_samples: null  # All data
max_epochs: 100
patience: 10
mixed_precision: true
device: cuda
```

---

## Hyperparameter Tuning

### Key Hyperparameters

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| `learning_rate` | 1e-5 to 1e-3 | 1e-4 | Lower for pretrained encoders |
| `batch_size` | 32-256 | 64 | Limited by GPU memory |
| `hidden_dim` | 128-512 | 256 | Larger = more capacity |
| `dropout` | 0.0-0.3 | 0.1 | Regularization |
| `weight_decay` | 0.0-0.1 | 0.01 | L2 regularization |

### Recommended Search

```python
from meta_spliceai.splice_engine.meta_layer.training import hyperparameter_search

# Define search space
search_space = {
    'learning_rate': [1e-5, 1e-4, 1e-3],
    'hidden_dim': [128, 256],
    'dropout': [0.0, 0.1, 0.2],
}

# Run search
best_config, results = hyperparameter_search(
    base_config=config,
    search_space=search_space,
    n_trials=10,
    metric='pr_auc_splice',
    direction='maximize'
)

print(f"Best config: {best_config}")
```

---

## Monitoring Training

### Weights & Biases (Optional)

```python
import wandb

# Initialize
wandb.init(project='meta-spliceai', config=config.__dict__)

# In trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    output_dir='models/meta_layer_v1',
    use_wandb=True
)
```

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir models/meta_layer_v1/logs

# Open in browser: http://localhost:6006
```

---

## Checkpointing

### Automatic Checkpointing

The trainer automatically saves:
- Best model (by validation PR-AUC)
- Last model
- Training history

```
models/meta_layer_v1/
├── best_model.pt           # Best by val_pr_auc
├── last_model.pt           # Final model
├── config.yaml             # Training config
├── history.json            # Training curves
└── logs/                   # TensorBoard logs
```

### Resume Training

```python
# Load checkpoint and resume
trainer = Trainer.from_checkpoint(
    checkpoint_path='models/meta_layer_v1/last_model.pt',
    train_loader=train_loader,
    val_loader=val_loader
)

# Continue training
trainer.train(additional_epochs=50)
```

---

## Common Issues

### Out of Memory

```python
# Reduce batch size
config.batch_size = 16

# Use gradient accumulation
config.gradient_accumulation_steps = 4  # Effective batch = 64

# Reduce sequence encoder
config.sequence_encoder = 'cnn'
```

### Slow Training

```python
# Enable mixed precision
config.mixed_precision = True

# Freeze sequence encoder
config.sequence_encoder_config['freeze'] = True

# Use fewer workers on Mac
config.num_workers = 0
```

### Poor Performance

```python
# Check data quality
from meta_spliceai.splice_engine.meta_layer.data import verify_dataset
verify_dataset('data/meta_training/openspliceai_v1/training_data.parquet')

# Use class weighting
config.balance_classes = True

# Add SpliceVarDB
config.variant_source = 'splicevardb'
```

---

## Example: Complete Training Script

```python
#!/usr/bin/env python3
"""
train_meta_layer.py - Complete meta-layer training script.

Usage:
    python train_meta_layer.py --config configs/hyenadna.yaml
"""

import argparse
from pathlib import Path

from meta_spliceai.splice_engine.meta_layer import MetaLayerConfig
from meta_spliceai.splice_engine.meta_layer.core import ArtifactLoader
from meta_spliceai.splice_engine.meta_layer.data import (
    prepare_training_dataset,
    MetaLayerDataset,
    create_dataloaders
)
from meta_spliceai.splice_engine.meta_layer.models import MetaSpliceModel
from meta_spliceai.splice_engine.meta_layer.training import (
    Trainer,
    MetaLayerEvaluator
)


def main(args):
    # Load config
    if args.config:
        config = MetaLayerConfig.from_yaml(args.config)
    else:
        config = MetaLayerConfig(
            base_model=args.base_model,
            sequence_encoder=args.encoder
        )
    
    print(f"Configuration:\n{config}")
    
    # Prepare data
    print("\n📁 Preparing training data...")
    data_dir = Path('data/meta_training') / f'{config.base_model}_v1'
    
    if not (data_dir / 'training_data.parquet').exists():
        prepare_training_dataset(config, output_dir=data_dir)
    
    # Create loaders
    print("\n📦 Creating data loaders...")
    dataset = MetaLayerDataset(data_dir / 'training_data.parquet')
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, 
        batch_size=config.batch_size,
        seed=config.seed
    )
    
    # Create model
    print("\n🧠 Creating model...")
    model = MetaSpliceModel(
        sequence_encoder=config.sequence_encoder,
        hidden_dim=config.hidden_dim
    )
    
    device = config.get_device()
    model = model.to(device)
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\n🚀 Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=args.output_dir
    )
    
    history = trainer.train()
    
    # Evaluate
    print("\n📊 Evaluating on test set...")
    evaluator = MetaLayerEvaluator()
    metrics = evaluator.evaluate_dataloader(model, test_loader, device)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"PR-AUC (splice):     {metrics['pr_auc_splice']:.4f}")
    print(f"AP (splice):         {metrics['ap_splice']:.4f}")
    print(f"Top-100 Acc (donor): {metrics['top100_acc_donor']:.4f}")
    print(f"ECE:                 {metrics['ece']:.4f}")
    
    print(f"\n✅ Model saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train meta-layer')
    parser.add_argument('--config', type=str, help='YAML config file')
    parser.add_argument('--base-model', type=str, default='openspliceai')
    parser.add_argument('--encoder', type=str, default='cnn')
    parser.add_argument('--output-dir', type=str, default='models/meta_layer_v1')
    
    args = parser.parse_args()
    main(args)
```

---

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [LABELING_STRATEGY.md](LABELING_STRATEGY.md) - Label creation
- [ALTERNATIVE_SPLICING_PIPELINE.md](ALTERNATIVE_SPLICING_PIPELINE.md) - Inference

---

*Last Updated: December 2025*






