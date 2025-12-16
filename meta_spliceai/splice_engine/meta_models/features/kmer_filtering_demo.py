#!/usr/bin/env python3
"""
Demonstration script for k-mer filtering strategies.

This script shows how to use the k-mer filtering functionality
with different strategies and configurations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_spliceai.splice_engine.meta_models.features import (
    KmerFilterConfig,
    integrate_kmer_filtering_in_cv,
    filter_dataset_kmers,
    get_kmer_filtering_stats
)


def create_demo_dataset(n_samples: int = 1000, n_kmers: int = 100) -> pd.DataFrame:
    """Create a demo dataset with k-mer features."""
    np.random.seed(42)
    
    # Create feature names
    feature_names = []
    
    # Add some non-k-mer features
    feature_names.extend([
        'donor_score', 'acceptor_score', 'neither_score',
        'gc_content', 'sequence_length', 'gene_id'
    ])
    
    # Add k-mer features
    for i in range(n_kmers):
        if i < 50:  # 6-mers
            seq = ''.join(np.random.choice(['A', 'C', 'G', 'T'], 6))
            feature_names.append(f'6mer_{seq}')
        else:  # 4-mers
            seq = ''.join(np.random.choice(['A', 'C', 'G', 'T'], 4))
            feature_names.append(f'4mer_{seq}')
    
    # Create feature matrix
    X = np.random.random((n_samples, len(feature_names)))
    
    # Make k-mers sparse (most are 0, some are 1)
    kmer_indices = [i for i, name in enumerate(feature_names) if 'mer_' in name]
    for i in kmer_indices:
        # Make 90% of k-mers zero
        mask = np.random.random(n_samples) > 0.1
        X[mask, i] = 0
    
    # Create target labels (multiclass: 0=neither, 1=donor, 2=acceptor)
    y = np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.15, 0.15])
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['splice_type'] = y
    
    return df


def demo_basic_filtering():
    """Demonstrate basic k-mer filtering."""
    print("🔬 Basic K-mer Filtering Demo")
    print("=" * 50)
    
    # Create demo dataset
    df = create_demo_dataset(n_samples=500, n_kmers=50)
    print(f"Created demo dataset: {df.shape}")
    
    # Show initial statistics
    stats = get_kmer_filtering_stats(df)
    print(f"\nInitial k-mer statistics:")
    print(f"  Total features: {stats['total_features']}")
    print(f"  K-mer features: {stats['kmer_features']}")
    print(f"  K-mer ratio: {stats['kmer_ratio']:.1%}")
    
    # Apply different filtering strategies
    strategies = ['motif', 'mi', 'sparsity', 'ensemble']
    
    for strategy in strategies:
        print(f"\n--- Testing {strategy.upper()} filtering ---")
        
        try:
            filtered_df = filter_dataset_kmers(
                df, 
                target_col='splice_type',
                strategy_name=strategy
            )
            
            print(f"  Original features: {df.shape[1] - 1}")  # -1 for target
            print(f"  Filtered features: {filtered_df.shape[1] - 1}")
            print(f"  Reduction: {(1 - (filtered_df.shape[1] - 1)/(df.shape[1] - 1))*100:.1f}%")
            
        except Exception as e:
            print(f"  Error with {strategy}: {e}")


def demo_preset_configs():
    """Demonstrate preset configurations."""
    print("\n🎛️  Preset Configurations Demo")
    print("=" * 50)
    
    # Create demo dataset
    df = create_demo_dataset(n_samples=300, n_kmers=30)
    X = df.drop('splice_type', axis=1).values
    y = df['splice_type'].values
    feature_names = df.drop('splice_type', axis=1).columns.tolist()
    
    # Test different presets
    presets = ['conservative', 'balanced', 'aggressive', 'motif_only', 'mi_only']
    
    for preset_name in presets:
        print(f"\n--- Testing {preset_name.upper()} preset ---")
        
        try:
            from meta_spliceai.splice_engine.meta_models.features import get_preset_config
            config = get_preset_config(preset_name)
            X_filtered, filtered_features = integrate_kmer_filtering_in_cv(
                X, y, feature_names, config
            )
            
            print(f"  Original features: {len(feature_names)}")
            print(f"  Filtered features: {len(filtered_features)}")
            print(f"  Reduction: {(1 - len(filtered_features)/len(feature_names))*100:.1f}%")
            
        except Exception as e:
            print(f"  Error with {preset_name}: {e}")


def demo_custom_config():
    """Demonstrate custom configuration."""
    print("\n⚙️  Custom Configuration Demo")
    print("=" * 50)
    
    # Create demo dataset
    df = create_demo_dataset(n_samples=400, n_kmers=40)
    X = df.drop('splice_type', axis=1).values
    y = df['splice_type'].values
    feature_names = df.drop('splice_type', axis=1).columns.tolist()
    
    # Create custom configurations
    custom_configs = [
        ("High MI threshold", KmerFilterConfig(
            enabled=True, strategy='mi', threshold=0.05
        )),
        ("Low sparsity range", KmerFilterConfig(
            enabled=True, strategy='sparsity', 
            min_occurrence_rate=0.05, max_occurrence_rate=0.8
        )),
        ("Donor-only motifs", KmerFilterConfig(
            enabled=True, strategy='motif', splice_type='donor'
        )),
        ("Conservative ensemble", KmerFilterConfig(
            enabled=True, strategy='ensemble',
            threshold=0.002,  # Very low MI threshold
            min_occurrence_rate=0.0001,  # Allow very rare features
            max_occurrence_rate=0.99  # Allow very common features
        ))
    ]
    
    for name, config in custom_configs:
        print(f"\n--- Testing {name} ---")
        
        try:
            X_filtered, filtered_features = integrate_kmer_filtering_in_cv(
                X, y, feature_names, config
            )
            
            print(f"  Original features: {len(feature_names)}")
            print(f"  Filtered features: {len(filtered_features)}")
            print(f"  Reduction: {(1 - len(filtered_features)/len(feature_names))*100:.1f}%")
            
        except Exception as e:
            print(f"  Error: {e}")


def demo_integration_with_cv():
    """Demonstrate integration with CV pipeline."""
    print("\n🔄 CV Integration Demo")
    print("=" * 50)
    
    # Create demo dataset
    df = create_demo_dataset(n_samples=600, n_kmers=60)
    X = df.drop('splice_type', axis=1).values
    y = df['splice_type'].values
    feature_names = df.drop('splice_type', axis=1).columns.tolist()
    
    # Simulate CV fold - split only X and y, keep feature_names intact
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Apply filtering to training set
    from meta_spliceai.splice_engine.meta_models.features import get_preset_config
    config = get_preset_config('balanced')
    X_train_filtered, filtered_features = integrate_kmer_filtering_in_cv(
        X_train, y_train, feature_names, config
    )
    
    # Apply same filtering to test set
    # (In real CV, you'd use the same feature indices)
    feature_indices = [feature_names.index(f) for f in filtered_features]
    X_test_filtered = X_test[:, feature_indices]
    
    print(f"\nAfter filtering:")
    print(f"  Training features: {X_train_filtered.shape[1]}")
    print(f"  Test features: {X_test_filtered.shape[1]}")
    print(f"  Feature reduction: {(1 - X_train_filtered.shape[1]/X_train.shape[1])*100:.1f}%")


def main():
    """Run all demos."""
    print("🧬 K-mer Filtering Demonstration")
    print("=" * 60)
    
    try:
        demo_basic_filtering()
        demo_preset_configs()
        demo_custom_config()
        demo_integration_with_cv()
        
        print("\n✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 