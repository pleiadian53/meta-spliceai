"""
Configuration and integration utilities for k-mer filtering in CV scripts.

This module provides easy integration of k-mer filtering strategies
with the gene-aware and chromosome-aware CV scripts.
"""

import argparse
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import numpy as np

from .kmer_filtering import (
    KmerFilterManager, MotifBasedFilter, MutualInformationFilter,
    SparsityAwareFilter, VarianceFilter, EnsembleFilter
)

logger = logging.getLogger(__name__)


class KmerFilterConfig:
    """Configuration class for k-mer filtering in CV scripts."""
    
    def __init__(self, enabled: bool = False, strategy: str = 'ensemble', **kwargs):
        self.enabled = enabled
        self.strategy = strategy
        self.kwargs = kwargs
        self.manager = KmerFilterManager() if enabled else None
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'KmerFilterConfig':
        """Create configuration from command-line arguments."""
        enabled = getattr(args, 'filter_kmers', False)
        strategy = getattr(args, 'kmer_filter_strategy', 'ensemble')
        
        # Extract strategy-specific arguments
        kwargs = {}
        if hasattr(args, 'kmer_mi_threshold'):
            kwargs['threshold'] = args.kmer_mi_threshold
        if hasattr(args, 'kmer_sparsity_min'):
            kwargs['min_occurrence_rate'] = args.kmer_sparsity_min
        if hasattr(args, 'kmer_sparsity_max'):
            kwargs['max_occurrence_rate'] = args.kmer_sparsity_max
        if hasattr(args, 'kmer_variance_threshold'):
            kwargs['threshold'] = args.kmer_variance_threshold
        
        return cls(enabled=enabled, strategy=strategy, **kwargs)
    
    def apply_filtering(self, X: np.ndarray, y: np.ndarray, 
                       feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Apply k-mer filtering if enabled."""
        if not self.enabled:
            return X, feature_names
        
        try:
            return self.manager.filter_kmers(X, y, feature_names, self.strategy, **self.kwargs)
        except Exception as e:
            logger.warning(f"K-mer filtering failed: {e}. Proceeding without filtering.")
            return X, feature_names


def add_kmer_filtering_args(parser: argparse.ArgumentParser) -> None:
    """
    Add k-mer filtering arguments to an argument parser.
    
    Parameters:
    -----------
    parser : argparse.ArgumentParser
        The argument parser to add arguments to
    """
    # Main filtering options
    parser.add_argument(
        '--filter-kmers',
        action='store_true',
        help='Enable k-mer feature filtering'
    )
    
    parser.add_argument(
        '--kmer-filter-strategy',
        choices=['motif', 'mi', 'sparsity', 'variance', 'ensemble'],
        default='ensemble',
        help='K-mer filtering strategy to use (default: ensemble)'
    )
    
    # Strategy-specific arguments
    parser.add_argument(
        '--kmer-mi-threshold',
        type=float,
        default=0.01,
        help='Mutual information threshold for MI-based filtering (default: 0.01)'
    )
    
    parser.add_argument(
        '--kmer-sparsity-min',
        type=float,
        default=0.001,
        help='Minimum occurrence rate for sparsity-based filtering (default: 0.001)'
    )
    
    parser.add_argument(
        '--kmer-sparsity-max',
        type=float,
        default=0.95,
        help='Maximum occurrence rate for sparsity-based filtering (default: 0.95)'
    )
    
    parser.add_argument(
        '--kmer-variance-threshold',
        type=float,
        default=0.001,
        help='Variance threshold for variance-based filtering (default: 0.001)'
    )
    
    # Advanced options
    parser.add_argument(
        '--kmer-splice-type',
        choices=['donor', 'acceptor', 'both'],
        default='both',
        help='Splice site type for motif-based filtering (default: both)'
    )
    
    parser.add_argument(
        '--kmer-combination-method',
        choices=['union', 'intersection'],
        default='union',
        help='Method for combining ensemble strategies (default: union)'
    )


def create_preset_configs() -> Dict[str, KmerFilterConfig]:
    """Create preset configurations for common use cases."""
    configs = {}
    
    # Conservative filtering - preserves most biologically relevant features
    configs['conservative'] = KmerFilterConfig(
        enabled=True,
        strategy='motif',
        splice_type='both'
    )
    
    # Balanced filtering - good balance between feature reduction and preservation
    configs['balanced'] = KmerFilterConfig(
        enabled=True,
        strategy='ensemble',
        threshold=0.005,  # Lower MI threshold
        min_occurrence_rate=0.0005,  # Allow rarer features
        max_occurrence_rate=0.98  # Allow more common features
    )
    
    # Aggressive filtering - maximum feature reduction
    configs['aggressive'] = KmerFilterConfig(
        enabled=True,
        strategy='ensemble',
        threshold=0.02,  # Higher MI threshold
        min_occurrence_rate=0.01,  # Higher minimum occurrence
        max_occurrence_rate=0.9  # Lower maximum occurrence
    )
    
    # Motif-only filtering - preserves only biologically known motifs
    configs['motif_only'] = KmerFilterConfig(
        enabled=True,
        strategy='motif',
        splice_type='both'
    )
    
    # MI-only filtering - based purely on information content
    configs['mi_only'] = KmerFilterConfig(
        enabled=True,
        strategy='mi',
        threshold=0.01
    )
    
    return configs


def get_preset_config(preset_name: str) -> KmerFilterConfig:
    """Get a preset configuration by name."""
    configs = create_preset_configs()
    if preset_name not in configs:
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(configs.keys())}")
    return configs[preset_name]


def print_filtering_stats(X: np.ndarray, feature_names: List[str], 
                         config: KmerFilterConfig) -> None:
    """Print statistics about k-mer filtering."""
    if not config.enabled:
        return
    
    manager = config.manager
    stats = manager.get_filtering_stats(X, feature_names)
    
    print(f"\n📊 K-mer Filtering Statistics:")
    print(f"   Total features: {stats['total_features']}")
    print(f"   K-mer features: {stats['kmer_features']} ({stats['kmer_ratio']:.1%})")
    print(f"   Non-k-mer features: {stats['non_kmer_features']}")
    
    if 'kmer_size_distribution' in stats:
        print(f"   K-mer size distribution:")
        for size, count in stats['kmer_size_distribution'].items():
            print(f"     {size}: {count} features")
    
    print(f"   Filtering strategy: {config.strategy}")
    if config.kwargs:
        print(f"   Strategy parameters: {config.kwargs}")


def validate_filtering_config(config: KmerFilterConfig) -> bool:
    """Validate a k-mer filtering configuration."""
    if not config.enabled:
        return True
    
    # Validate strategy
    valid_strategies = ['motif', 'mi', 'sparsity', 'variance', 'ensemble']
    if config.strategy not in valid_strategies:
        logger.error(f"Invalid k-mer filtering strategy: {config.strategy}")
        return False
    
    # Validate parameters based on strategy
    if config.strategy == 'mi':
        if 'threshold' in config.kwargs and config.kwargs['threshold'] <= 0:
            logger.error("MI threshold must be positive")
            return False
    
    elif config.strategy == 'sparsity':
        min_rate = config.kwargs.get('min_occurrence_rate', 0.001)
        max_rate = config.kwargs.get('max_occurrence_rate', 0.95)
        if min_rate >= max_rate:
            logger.error("Sparsity min rate must be less than max rate")
            return False
    
    elif config.strategy == 'variance':
        if 'threshold' in config.kwargs and config.kwargs['threshold'] < 0:
            logger.error("Variance threshold must be non-negative")
            return False
    
    return True


# Integration helper for CV scripts
def integrate_kmer_filtering_in_cv(X: np.ndarray, y: np.ndarray, 
                                 feature_names: List[str],
                                 config: KmerFilterConfig) -> Tuple[np.ndarray, List[str]]:
    """
    Integrate k-mer filtering into CV training pipeline.
    
    This function can be called during the CV training process to apply
    k-mer filtering to the feature matrix.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    feature_names : List[str]
        List of feature names
    config : KmerFilterConfig
        Filtering configuration
        
    Returns:
    --------
    Tuple[np.ndarray, List[str]]
        Filtered feature matrix and feature names
    """
    if not config.enabled:
        return X, feature_names
    
    print(f"\n🔍 Applying k-mer filtering with strategy: {config.strategy}")
    
    # Print initial statistics
    print_filtering_stats(X, feature_names, config)
    
    # Apply filtering
    X_filtered, filtered_features = config.apply_filtering(X, y, feature_names)
    
    # Print final statistics
    print(f"   Features after filtering: {len(filtered_features)}/{len(feature_names)}")
    print(f"   Feature reduction: {(1 - len(filtered_features)/len(feature_names))*100:.1f}%")
    
    return X_filtered, filtered_features 