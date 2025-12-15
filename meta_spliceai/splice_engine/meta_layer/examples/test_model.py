#!/usr/bin/env python3
"""
test_model.py - Test the meta-layer model components.

This script tests:
1. Dataset loading from artifacts
2. Model forward pass
3. Basic training step

Usage:
    mamba activate metaspliceai
    python meta_spliceai/splice_engine/meta_layer/examples/test_model.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parents[5]
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F

from meta_spliceai.splice_engine.meta_layer import (
    MetaLayerConfig,
    ArtifactLoader,
    MetaSpliceModel,
    ScoreOnlyModel,
    MetaLayerDataset,
    create_dataloaders
)


def test_config():
    """Test configuration and path resolution."""
    print("\n" + "=" * 60)
    print("Test 1: Configuration (genomic_resources integration)")
    print("=" * 60)
    
    config = MetaLayerConfig(base_model='openspliceai')
    
    print(f"\nBase model: {config.base_model}")
    print(f"Genome build: {config.genome_build}")
    print(f"Annotation source: {config.annotation_source}")
    print(f"Artifacts dir: {config.artifacts_dir}")
    print(f"Coordinate column: {config.coordinate_column}")
    
    assert config.genome_build == 'GRCh38', "Expected GRCh38 for OpenSpliceAI"
    assert config.annotation_source == 'mane', "Expected mane annotation source"
    
    print("\n✅ Configuration test passed!")


def test_artifact_loader():
    """Test loading artifacts."""
    print("\n" + "=" * 60)
    print("Test 2: Artifact Loader")
    print("=" * 60)
    
    config = MetaLayerConfig(base_model='openspliceai')
    loader = ArtifactLoader(config)
    
    # Get statistics
    stats = loader.get_statistics()
    print(f"\nArtifact statistics:")
    print(f"  Files: {stats['num_files']}")
    print(f"  Positions: {stats['total_positions']:,}")
    print(f"  Size: {stats['total_size_mb']:.1f} MB")
    
    # Get feature columns
    feature_cols = loader.get_feature_columns()
    print(f"\nFeature columns:")
    print(f"  Base scores: {len(feature_cols['base_scores'])}")
    print(f"  Context scores: {len(feature_cols['context_scores'])}")
    print(f"  Derived features: {len(feature_cols['derived_features'])}")
    
    # Load sample data
    print("\nLoading sample data from chr21...")
    df = loader.load_analysis_sequences(chromosomes=['21'], verbose=False)
    print(f"  Loaded {len(df)} positions")
    
    print("\n✅ Artifact loader test passed!")
    return df


def test_dataset(df):
    """Test dataset creation."""
    print("\n" + "=" * 60)
    print("Test 3: Dataset")
    print("=" * 60)
    
    # Create dataset
    dataset = MetaLayerDataset(df, max_seq_length=501)
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Num features: {dataset.num_features}")
    print(f"Num classes: {dataset.num_classes}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  sequence: {sample['sequence'].shape}")
    print(f"  features: {sample['features'].shape}")
    print(f"  label: {sample['label']}")
    print(f"  weight: {sample['weight']}")
    
    # Get class weights
    class_weights = dataset.get_class_weights()
    print(f"\nClass weights: {class_weights}")
    
    print("\n✅ Dataset test passed!")
    return dataset


def test_model(dataset):
    """Test model forward pass."""
    print("\n" + "=" * 60)
    print("Test 4: Model Forward Pass")
    print("=" * 60)
    
    # Create model
    model = MetaSpliceModel(
        sequence_encoder='cnn',
        num_score_features=dataset.num_features,
        hidden_dim=128,
        dropout=0.1
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")
    print(f"Trainable: {trainable:,}")
    
    # Create dataloader
    train_loader, val_loader, _ = create_dataloaders(
        dataset, 
        batch_size=32,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  sequence: {batch['sequence'].shape}")
    print(f"  features: {batch['features'].shape}")
    print(f"  label: {batch['label'].shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(batch['sequence'], batch['features'])
        probs = F.softmax(logits, dim=-1)
    
    print(f"\nOutput shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  probs: {probs.shape}")
    print(f"\nSample predictions (first 5):")
    print(f"  {probs[:5]}")
    
    print("\n✅ Model forward pass test passed!")
    return model, train_loader


def test_training_step(model, train_loader):
    """Test a single training step."""
    print("\n" + "=" * 60)
    print("Test 5: Training Step")
    print("=" * 60)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Get batch
    batch = next(iter(train_loader))
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    logits = model(batch['sequence'], batch['features'])
    loss = F.cross_entropy(logits, batch['label'], weight=None)
    
    print(f"\nLoss: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # Check gradients
    total_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad += p.grad.abs().sum().item()
    
    print(f"Total gradient magnitude: {total_grad:.4f}")
    
    # Optimizer step
    optimizer.step()
    
    print("\n✅ Training step test passed!")


def test_score_only_baseline(dataset):
    """Test score-only baseline model."""
    print("\n" + "=" * 60)
    print("Test 6: Score-Only Baseline")
    print("=" * 60)
    
    model = ScoreOnlyModel(
        num_score_features=dataset.num_features,
        hidden_dim=128
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    
    # Get sample
    sample = dataset[0]
    features = sample['features'].unsqueeze(0)  # Add batch dim
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(features)
    
    print(f"Output shape: {logits.shape}")
    print(f"Predictions: {F.softmax(logits, dim=-1)}")
    
    print("\n✅ Score-only baseline test passed!")


def main():
    print("=" * 60)
    print("Meta-Layer Model Test Suite")
    print("=" * 60)
    print("\nThis tests all components of the meta-layer package.")
    
    try:
        # Run tests
        test_config()
        df = test_artifact_loader()
        dataset = test_dataset(df)
        model, train_loader = test_model(dataset)
        test_training_step(model, train_loader)
        test_score_only_baseline(dataset)
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe meta-layer package is ready for training.")
        print("Next steps:")
        print("  1. Prepare full training dataset")
        print("  2. Train with: python examples/train_simple.py")
        print("  3. Evaluate and tune hyperparameters")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())






