#!/usr/bin/env python3
"""
train_simple.py - Simple training example for meta-layer.

This example demonstrates:
1. Loading artifacts from base layer
2. Creating dataset and dataloaders
3. Training a meta-layer model
4. Evaluating and saving results

Usage:
    mamba activate metaspliceai
    python meta_spliceai/splice_engine/meta_layer/examples/train_simple.py

For GPU training (RunPods):
    python train_simple.py --device cuda --epochs 50 --batch-size 256
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[5]
sys.path.insert(0, str(project_root))

import torch

from meta_spliceai.splice_engine.meta_layer import (
    MetaLayerConfig,
    ArtifactLoader,
    MetaSpliceModel,
    MetaLayerDataset,
    create_dataloaders
)
from meta_spliceai.splice_engine.meta_layer.training import (
    Trainer,
    TrainingConfig,
    Evaluator
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train meta-layer model')
    
    # Model configuration
    parser.add_argument('--base-model', type=str, default='openspliceai',
                        choices=['openspliceai', 'spliceai'],
                        help='Base model to use')
    parser.add_argument('--sequence-encoder', type=str, default='cnn',
                        choices=['cnn', 'hyenadna', 'none'],
                        help='Sequence encoder type')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use')
    
    # Data configuration
    parser.add_argument('--chromosomes', type=str, nargs='+', 
                        default=['21', '22'],
                        help='Chromosomes to use for training')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples to use (for quick testing)')
    
    # Output
    parser.add_argument('--output-dir', type=str, 
                        default='models/meta_layer_test',
                        help='Output directory for checkpoints')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Meta-Layer Training Example")
    print("=" * 60)
    
    # 1. Configuration
    print("\n[1/5] Configuring...")
    
    config = MetaLayerConfig(
        base_model=args.base_model,
        sequence_encoder=args.sequence_encoder,
        hidden_dim=args.hidden_dim
    )
    
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.output_dir,
        early_stopping_patience=5,
        log_every=50
    )
    
    print(f"  Base model: {config.base_model}")
    print(f"  Sequence encoder: {config.sequence_encoder}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Device: {training_config.get_device()}")
    
    # 2. Load data
    print("\n[2/5] Loading artifacts...")
    
    loader = ArtifactLoader(config)
    df = loader.load_analysis_sequences(
        chromosomes=args.chromosomes,
        verbose=True
    )
    
    print(f"  Loaded {len(df)} positions")
    
    # Optionally limit samples
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, seed=42)
        print(f"  Limited to {len(df)} samples")
    
    # 3. Create dataset and dataloaders
    print("\n[3/5] Creating dataset...")
    
    dataset = MetaLayerDataset(df, max_seq_length=501)
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1
    )
    
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Feature columns: {dataset.num_features}")
    
    # 4. Create model
    print("\n[4/5] Creating model...")
    
    model = MetaSpliceModel(
        sequence_encoder=config.sequence_encoder,
        num_score_features=dataset.num_features,
        hidden_dim=config.hidden_dim,
        dropout=training_config.dropout
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Get class weights
    class_weights = dataset.get_class_weights()
    print(f"  Class weights: {class_weights}")
    
    # 5. Train
    print("\n[5/5] Training...")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        class_weights=class_weights
    )
    
    result = trainer.train()
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest epoch: {result.best_epoch}")
    print(f"Best PR-AUC: {result.best_metrics.get('val_pr_auc', 0):.4f}")
    print(f"Best ROC-AUC: {result.best_metrics.get('val_roc_auc', 0):.4f}")
    print(f"Best Accuracy: {result.best_metrics.get('val_accuracy', 0):.4f}")
    print(f"Training time: {result.total_time_seconds:.1f} seconds")
    
    if result.final_model_path:
        print(f"\nModel saved: {result.final_model_path}")
    
    # Test evaluation
    print("\n" + "-" * 40)
    print("Test Set Evaluation")
    print("-" * 40)
    
    evaluator = Evaluator()
    model.eval()
    device = training_config.get_device()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequence = batch['sequence'].to(device)
            features = batch['features'].to(device)
            labels = batch['label']
            
            logits = model(sequence, features)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_probs.append(probs.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    
    test_result = evaluator.evaluate(all_preds, all_labels, all_probs)
    
    print(f"Test PR-AUC: {test_result.pr_auc_macro:.4f}")
    print(f"Test ROC-AUC: {test_result.roc_auc_macro:.4f}")
    print(f"Test Accuracy: {test_result.accuracy:.4f}")
    print(f"Test AP: {test_result.average_precision_macro:.4f}")
    
    print("\nPer-class PR-AUC:")
    for i, name in enumerate(['donor', 'acceptor', 'neither']):
        print(f"  {name}: {test_result.per_class_pr_auc[i]:.4f}")
    
    print("\n✅ Training complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())






